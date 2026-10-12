from __future__ import annotations

import re

import pytest

from worker.captions import (
    Block,
    CaptionError,
    build_clip_srt,
    format_clip_srt,
    parse_clip_srt,
    parse_master_srt,
    transcript_text,
    validate_edited_slice,
)

CLIP_ID = "clip_04"
# Copied from tests/test_assemble.py (test modules do not import each other).
MASTER_SRT = (
    b"1\r\n"
    b"00:00:10,000 --> 00:00:11,250\r\n"
    b"Primera l\xc3\xadnea\r\nsegunda l\xc3\xadnea\r\n"
    b"\r\n"
    b"2\r\n"
    b"00:00:11,500 --> 00:00:12,500\r\n"
    b"Texto final\r\n"
)
# build_clip_srt(MASTER_SRT blocks 1-2, clip_start_ms=9_500): LF separators,
# block text verbatim from the master (inner CRLF kept).
CLIP_SLICE = (
    b"1\n"
    b"00:00:00,500 --> 00:00:01,750\n"
    b"Primera l\xc3\xadnea\r\nsegunda l\xc3\xadnea\n"
    b"\n"
    b"2\n"
    b"00:00:02,000 --> 00:00:03,000\n"
    b"Texto final\n"
)


# --------------------------------------------------------------------------- #
# parse_master_srt / parse_clip_srt
# --------------------------------------------------------------------------- #


def test_parse_master_srt_reads_crlf_and_keeps_text_verbatim() -> None:
    assert parse_master_srt(MASTER_SRT) == [
        Block(
            index=1,
            start_ms=10_000,
            end_ms=11_250,
            text="Primera línea\r\nsegunda línea",
        ),
        Block(index=2, start_ms=11_500, end_ms=12_500, text="Texto final"),
    ]


def test_parse_master_srt_strips_utf8_bom() -> None:
    assert parse_master_srt(b"\xef\xbb\xbf" + MASTER_SRT) == parse_master_srt(
        MASTER_SRT
    )


def test_parse_master_srt_rejects_duplicate_index() -> None:
    duplicate = (
        b"1\n00:00:00,000 --> 00:00:01,000\nUno\n"
        b"\n"
        b"1\n00:00:01,000 --> 00:00:02,000\nOtra vez uno\n"
    )

    with pytest.raises(CaptionError, match="master SRT contains duplicate block 1"):
        parse_master_srt(duplicate)


def test_parse_master_srt_rejects_malformed_block() -> None:
    malformed = (
        b"1\n00:00:00,000 --> 00:00:01,000\nUno\n"
        b"\n"
        b"2\nnot a timing line\nDos\n"
    )

    with pytest.raises(
        CaptionError, match="master SRT block at position 2 is malformed"
    ):
        parse_master_srt(malformed)


def test_parse_master_srt_rejects_non_utf8() -> None:
    latin1 = b"1\n00:00:00,000 --> 00:00:01,000\nl\xednea\n"

    with pytest.raises(CaptionError, match="master SRT must be valid UTF-8"):
        parse_master_srt(latin1)


def test_parse_clip_srt_accepts_empty_text_block() -> None:
    data = (
        b"1\n00:00:00,000 --> 00:00:01,000\n"
        b"\n"
        b"2\n00:00:01,000 --> 00:00:02,000\nHola\n"
    )

    assert parse_clip_srt(data) == [
        Block(index=1, start_ms=0, end_ms=1_000, text=""),
        Block(index=2, start_ms=1_000, end_ms=2_000, text="Hola"),
    ]
    # The master parser stays strict: a text line is required there.
    with pytest.raises(CaptionError, match="block at position 1 is malformed"):
        parse_master_srt(data)


# --------------------------------------------------------------------------- #
# build_clip_srt
# --------------------------------------------------------------------------- #


def test_build_clip_srt_slices_shifts_and_keeps_text() -> None:
    result = build_clip_srt(
        parse_master_srt(MASTER_SRT),
        start_block=1,
        end_block=2,
        clip_start_ms=9_500,
        clip_id=CLIP_ID,
    )

    assert result == CLIP_SLICE


def test_build_clip_srt_renumbers_from_one() -> None:
    result = build_clip_srt(
        parse_master_srt(MASTER_SRT),
        start_block=2,
        end_block=2,
        clip_start_ms=11_000,
        clip_id=CLIP_ID,
    )

    assert result == b"1\n00:00:00,500 --> 00:00:01,500\nTexto final\n"


def test_build_clip_srt_rejects_missing_block() -> None:
    with pytest.raises(
        CaptionError,
        match=f"clip {CLIP_ID} references missing master SRT block 3",
    ):
        build_clip_srt(
            parse_master_srt(MASTER_SRT),
            start_block=2,
            end_block=3,
            clip_start_ms=10_000,
            clip_id=CLIP_ID,
        )


def test_build_clip_srt_rejects_negative_shift() -> None:
    with pytest.raises(
        CaptionError,
        match=f"clip {CLIP_ID} block 1 shifts to a negative subtitle timestamp",
    ):
        build_clip_srt(
            parse_master_srt(MASTER_SRT),
            start_block=1,
            end_block=1,
            clip_start_ms=10_001,
            clip_id=CLIP_ID,
        )


# --------------------------------------------------------------------------- #
# format_clip_srt
# --------------------------------------------------------------------------- #


def test_format_clip_srt_round_trips_with_blank_middle_block() -> None:
    blocks = [
        Block(index=1, start_ms=0, end_ms=1_000, text="Uno"),
        Block(index=2, start_ms=1_000, end_ms=2_000, text=""),
        Block(index=3, start_ms=2_000, end_ms=3_000, text="Tres\nlíneas"),
    ]

    data = format_clip_srt(blocks)

    assert data == (
        b"1\n00:00:00,000 --> 00:00:01,000\nUno\n"
        b"\n"
        b"2\n00:00:01,000 --> 00:00:02,000\n"
        b"\n"
        b"3\n00:00:02,000 --> 00:00:03,000\nTres\nl\xc3\xadneas\n"
    )
    assert parse_clip_srt(data) == blocks


# --------------------------------------------------------------------------- #
# validate_edited_slice
# --------------------------------------------------------------------------- #


def test_validate_edited_slice_accepts_text_only_edit() -> None:
    edited = CLIP_SLICE.replace(b"Texto final", b"Texto corregido")

    assert validate_edited_slice(CLIP_SLICE, edited, CLIP_ID) == [
        Block(
            index=1,
            start_ms=500,
            end_ms=1_750,
            text="Primera línea\nsegunda línea",
        ),
        Block(index=2, start_ms=2_000, end_ms=3_000, text="Texto corregido"),
    ]


def test_validate_edited_slice_accepts_blanked_block() -> None:
    edited = CLIP_SLICE.replace(b"Texto final\n", b"")

    blocks = validate_edited_slice(CLIP_SLICE, edited, CLIP_ID)

    assert blocks[1] == Block(index=2, start_ms=2_000, end_ms=3_000, text="")


def test_validate_edited_slice_normalises_crlf_to_lf() -> None:
    # A browser textarea hands the whole file back with CRLF line endings.
    edited = CLIP_SLICE.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")

    blocks = validate_edited_slice(CLIP_SLICE, edited, CLIP_ID)

    assert [block.text for block in blocks] == [
        "Primera línea\nsegunda línea",
        "Texto final",
    ]
    assert b"\r" not in format_clip_srt(blocks)


@pytest.mark.parametrize(
    "edited, message",
    [
        pytest.param(
            CLIP_SLICE.split(b"\n\n2\n")[0] + b"\n",
            f"clip {CLIP_ID} edited SRT has 1 block(s), expected 2",
            id="count",
        ),
        pytest.param(
            CLIP_SLICE.replace(b"\n\n2\n", b"\n\n3\n"),
            f"clip {CLIP_ID} block at position 2 has index 3, expected 2",
            id="index",
        ),
        pytest.param(
            CLIP_SLICE.replace(b"00:00:02,000 -->", b"00:00:02,001 -->"),
            f"clip {CLIP_ID} block 2 start 00:00:02,001, expected 00:00:02,000",
            id="start",
        ),
        pytest.param(
            CLIP_SLICE.replace(b"--> 00:00:01,750", b"--> 00:00:01,751"),
            f"clip {CLIP_ID} block 1 end 00:00:01,751, expected 00:00:01,750",
            id="end",
        ),
    ],
)
def test_validate_edited_slice_rejects_structural_change(
    edited: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_edited_slice(CLIP_SLICE, edited, CLIP_ID)


def test_validate_edited_slice_malformed_edit_is_value_error() -> None:
    with pytest.raises(
        ValueError, match=f"clip {CLIP_ID} edited SRT is malformed"
    ) as raised:
        validate_edited_slice(CLIP_SLICE, b"this is not an srt\n", CLIP_ID)

    assert isinstance(raised.value.__cause__, CaptionError)


def test_validate_edited_slice_malformed_generated_is_caption_error() -> None:
    with pytest.raises(CaptionError, match="clip SRT block at position 1"):
        validate_edited_slice(b"this is not an srt\n", CLIP_SLICE, CLIP_ID)


# --------------------------------------------------------------------------- #
# transcript_text
# --------------------------------------------------------------------------- #


def test_transcript_text_joins_blocks_with_lf() -> None:
    blocks = [
        Block(index=1, start_ms=0, end_ms=1_000, text="Uno"),
        Block(index=2, start_ms=1_000, end_ms=2_000, text="Dos"),
    ]

    assert transcript_text(blocks) == "Uno\nDos"


def test_transcript_text_drops_blank_lines() -> None:
    blocks = [Block(index=1, start_ms=0, end_ms=1_000, text="Uno\n\n   \nDos")]

    assert transcript_text(blocks) == "Uno\nDos"


def test_transcript_text_flattens_crlf_inside_text() -> None:
    blocks = parse_master_srt(MASTER_SRT)

    assert transcript_text(blocks) == (
        "Primera línea\nsegunda línea\nTexto final"
    )


def test_transcript_text_blank_block_contributes_nothing() -> None:
    blocks = [
        Block(index=1, start_ms=0, end_ms=1_000, text="Uno"),
        Block(index=2, start_ms=1_000, end_ms=2_000, text=""),
        Block(index=3, start_ms=2_000, end_ms=3_000, text="Tres"),
    ]

    assert transcript_text(blocks) == "Uno\nTres"
