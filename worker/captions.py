"""Per-clip caption slicing and validation for the clip-level SRT edit gate.

The master SRT (``subtitles.corrected.srt``) is the single source of caption
timing. ``build_clip_srt`` slices one clip's block range out of it, re-times
it to clip-relative milliseconds, and renumbers it from 1. An operator may
edit that slice before creative runs, under one contract:

  * Edited slices may change TEXT ONLY (including blanking a block).
  * Block count, block indexes, and start/end timings are owned by the
    pipeline; ``validate_edited_slice`` rejects any edit that changes them.

Consumers burn the validated blocks re-serialised by ``format_clip_srt`` —
never the raw edited bytes — so there is exactly one SRT serialiser.

Pure functions only: no S3, DB, or ffmpeg access, and no imports from
``worker.stages``, so build, creative, and assemble can all depend on it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

SRT_TIMESTAMP_RE = re.compile(
    r"(?P<hours>\d{2}):(?P<minutes>\d{2}):"
    r"(?P<seconds>\d{2}),(?P<milliseconds>\d{3})"
)
SRT_BLOCK_RE = re.compile(
    r"(?P<number>\d+)(?:\r\n|\n)"
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})[^\r\n]*"
    r"(?:\r\n|\n)(?P<text>.*)\Z",
    re.DOTALL,
)
# Clip-SRT variant: identical except the text line is optional, because the
# edit gate allows an operator to blank a block's text entirely.
CLIP_SRT_BLOCK_RE = re.compile(
    r"(?P<number>\d+)(?:\r\n|\n)"
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})[^\r\n]*"
    r"(?:(?:\r\n|\n)(?P<text>.*))?\Z",
    re.DOTALL,
)


class CaptionError(RuntimeError):
    """Raised when an SRT cannot be parsed or sliced safely.

    A RuntimeError like assemble's ``AssembleError``, raised with the same
    messages, so assemble's behaviour is unchanged when it switches over.
    """


@dataclass(frozen=True)
class Block:
    """One parsed SRT block with verbatim subtitle text."""

    index: int
    start_ms: int
    end_ms: int
    text: str


def parse_srt_timestamp(timestamp: str) -> int:
    """Parse an SRT timestamp into integer milliseconds."""

    match = SRT_TIMESTAMP_RE.fullmatch(timestamp)
    if match is None:
        raise CaptionError(f"assemble: invalid SRT timestamp {timestamp!r}")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    milliseconds = int(match.group("milliseconds"))
    if minutes >= 60 or seconds >= 60:
        raise CaptionError(f"assemble: invalid SRT timestamp {timestamp!r}")
    return (((hours * 60) + minutes) * 60 + seconds) * 1000 + milliseconds


def format_srt_timestamp(milliseconds: int) -> str:
    """Format non-negative milliseconds as an SRT timestamp."""

    if milliseconds < 0:
        raise ValueError("SRT milliseconds cannot be negative")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_master_srt(data: bytes) -> list[Block]:
    """Parse master SRT structure while retaining each subtitle text verbatim."""

    return _parse_srt(data, label="master SRT", block_re=SRT_BLOCK_RE)


def parse_clip_srt(data: bytes) -> list[Block]:
    """Parse a clip-relative SRT (build_clip_srt output or an operator edit).

    Same structural strictness as :func:`parse_master_srt`, with two
    edit-gate allowances: a block's text may be empty, and line endings
    inside text are normalised to LF (a browser textarea hands back CRLF).
    """

    return [
        Block(
            index=block.index,
            start_ms=block.start_ms,
            end_ms=block.end_ms,
            text=_normalise_newlines(block.text),
        )
        for block in _parse_srt(data, label="clip SRT", block_re=CLIP_SRT_BLOCK_RE)
    ]


def format_clip_srt(blocks: Sequence[Block]) -> bytes:
    """Serialise blocks as an LF-separated UTF-8 SRT, the one serialiser.

    Writes each ``block.index`` as given (no renumbering) and each text
    verbatim. An empty-text block serialises as just its index and timing
    lines, which :func:`parse_clip_srt` reads back as empty text.
    """

    output: list[str] = []
    for block in blocks:
        entry = (
            f"{block.index}\n"
            f"{format_srt_timestamp(block.start_ms)} --> "
            f"{format_srt_timestamp(block.end_ms)}"
        )
        if block.text:
            entry += f"\n{block.text}"
        output.append(entry)
    return ("\n\n".join(output) + "\n").encode("utf-8")


def build_clip_srt(
    blocks: Sequence[Block],
    *,
    start_block: int,
    end_block: int,
    clip_start_ms: int,
    clip_id: str,
) -> bytes:
    """Slice, re-time, and renumber master blocks without changing their text."""

    by_index = {block.index: block for block in blocks}
    sliced: list[Block] = []
    for output_number, block_number in enumerate(
        range(start_block, end_block + 1), start=1
    ):
        block = by_index.get(block_number)
        if block is None:
            raise CaptionError(
                f"assemble: clip {clip_id} references missing master SRT "
                f"block {block_number}"
            )
        shifted_start = block.start_ms - clip_start_ms
        shifted_end = block.end_ms - clip_start_ms
        if shifted_start < 0 or shifted_end < 0:
            raise CaptionError(
                f"assemble: clip {clip_id} block {block_number} shifts to "
                "a negative subtitle timestamp"
            )
        sliced.append(
            Block(
                index=output_number,
                start_ms=shifted_start,
                end_ms=shifted_end,
                text=block.text,
            )
        )
    return format_clip_srt(sliced)


def validate_edited_slice(
    generated: bytes, edited: bytes, clip_id: str
) -> list[Block]:
    """Return the edited blocks if only their text differs from the slice.

    Raises ValueError naming the clip and the FIRST difference: block count,
    then per block (in order) index, start, end. A malformed edited file is
    also reported as ValueError so callers handle one error type for every
    rejected edit; a malformed GENERATED slice is a pipeline bug and
    propagates as :class:`CaptionError`.
    """

    expected = parse_clip_srt(generated)
    try:
        actual = parse_clip_srt(edited)
    except CaptionError as exc:
        raise ValueError(
            f"captions: clip {clip_id} edited SRT is malformed: {exc}"
        ) from exc

    if len(actual) != len(expected):
        raise ValueError(
            f"captions: clip {clip_id} edited SRT has {len(actual)} block(s), "
            f"expected {len(expected)}"
        )
    for position, (want, got) in enumerate(
        zip(expected, actual, strict=True), start=1
    ):
        if got.index != want.index:
            raise ValueError(
                f"captions: clip {clip_id} block at position {position} has "
                f"index {got.index}, expected {want.index}"
            )
        if got.start_ms != want.start_ms:
            raise ValueError(
                f"captions: clip {clip_id} block {want.index} start "
                f"{format_srt_timestamp(got.start_ms)}, expected "
                f"{format_srt_timestamp(want.start_ms)}"
            )
        if got.end_ms != want.end_ms:
            raise ValueError(
                f"captions: clip {clip_id} block {want.index} end "
                f"{format_srt_timestamp(got.end_ms)}, expected "
                f"{format_srt_timestamp(want.end_ms)}"
            )
    return actual


def transcript_text(blocks: Sequence[Block]) -> str:
    """Flatten block text into the transcript shape build's resolve_blocks emits.

    Build parses with ``splitlines()`` and drops blank lines, so CRLF inside a
    block becomes LF and a blanked block contributes nothing. Matching that
    keeps creative's prompt input byte-identical for an unedited slice.
    """

    return "\n".join(
        line
        for block in blocks
        for line in block.text.splitlines()
        if line.strip() != ""
    )


# --------------------------------------------------------------------------- #
# Internal
# --------------------------------------------------------------------------- #


def _parse_srt(
    data: bytes, *, label: str, block_re: re.Pattern[str]
) -> list[Block]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CaptionError(f"assemble: {label} must be valid UTF-8") from exc

    payload = text.strip("\r\n")
    if not payload:
        raise CaptionError(f"assemble: {label} has no subtitle blocks")

    blocks: list[Block] = []
    seen: set[int] = set()
    for position, chunk in enumerate(re.split(r"(?:\r\n|\n){2,}", payload), start=1):
        match = block_re.fullmatch(chunk)
        if match is None:
            raise CaptionError(
                f"assemble: {label} block at position {position} is malformed"
            )
        number = int(match.group("number"))
        if number in seen:
            raise CaptionError(
                f"assemble: {label} contains duplicate block {number}"
            )
        seen.add(number)
        blocks.append(
            Block(
                index=number,
                start_ms=parse_srt_timestamp(match.group("start")),
                end_ms=parse_srt_timestamp(match.group("end")),
                text=match.group("text") or "",
            )
        )
    return blocks


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")
