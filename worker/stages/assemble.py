"""Assemble generated assets and approved clips into finished vertical shorts."""

from __future__ import annotations

import copy
import json
import logging
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, TypedDict, cast

from worker import jobs
from worker.artifacts import main_clip_key

if TYPE_CHECKING:
    from psycopg import Connection
    from psycopg.rows import DictRow

    from worker.jobs import Job
    from worker.workspace import Workspace

logger = logging.getLogger(__name__)

AssetName = Literal["hook", "outro", "bridge_in", "bridge_out"]
MainLayout = Literal["blur", "bars", "crop"]
SegmentName = Literal["hook", "bridge_in", "main", "bridge_out", "outro"]

# Bundled brand fonts live in worker/fonts/ (Iron Rule 5). Resolved relative to
# the package so they load identically from source and inside the container (the
# wheel does not ship fonts as package-data, and the worker runs from source):
#   Anton-Regular.ttf         -> hook_text drawtext overlay
#   Montserrat-ExtraBold.ttf  -> captions (libass) + close_text drawtext overlay
FONTS_DIR = Path(__file__).resolve().parents[1] / "fonts"
ASSET_ORDER: tuple[AssetName, ...] = (
    "hook",
    "outro",
    "bridge_in",
    "bridge_out",
)
# (fade_in, fade_out) per concat segment: both sides of all four boundaries
# dip to black; the video opens hard on the hook and the bed music's fade-out
# closes the outro.
BOUNDARY_FADE_SIDES: dict[SegmentName, tuple[bool, bool]] = {
    "hook": (False, True),
    "bridge_in": (True, True),
    "main": (True, True),
    "bridge_out": (True, True),
    "outro": (True, False),
}
T = TypeVar("T")

# Overlay text must never exceed the frame. The rendered width (glyph advance
# sum + 2*borderw, since drawtext's border sits OUTSIDE text_w) is held within
# 90% of the output width -- a 5% safe margin per side. When the base fontsize
# overflows, the assembler shrinks to the largest integer size that fits,
# floored at OVERLAY_MIN_FONTSIZE; below the floor it logs and proceeds rather
# than clip silently.
OVERLAY_FRAME_MARGIN = 0.05
OVERLAY_MIN_FONTSIZE = 40
# unitsPerEm + {codepoint: advanceWidth} per TTF path, loaded once (fontTools).
_FONT_METRICS_CACHE: dict[str, tuple[int, dict[int, int]]] = {}


class OverlayStyle(TypedDict):
    """Deterministic drawtext styling for one on-screen text overlay."""

    fontfile: str
    fontsize: int
    fontcolor: str
    borderw: int
    bordercolor: str
    y_fraction: float
    appear_time_s: float
    anim_duration_s: float
    rise_px: int


class AssembleConfig(TypedDict):
    """Tunable assembly policy and encoding settings."""

    ffmpeg_binary: str
    ffprobe_binary: str
    bed_music_key: str
    logo_key: str
    output_width: int
    output_height: int
    output_fps: int
    video_codec: str
    audio_codec: str
    pixel_format: str
    sample_aspect_ratio: int
    audio_sample_rate: int
    audio_layout: str
    hook_speed: float
    bridge_speed: float
    outro_speed: float
    main_speed: float
    main_layout: MainLayout
    blur_radius: int
    blur_power: int
    boundary_fade_seconds: float
    bridge_out_tail_trim_seconds: float
    silence_threshold_db: float
    silence_min_duration_s: float
    silence_padding_s: float
    voice_volume: float
    bed_volume: float
    bed_fade_out_s: float
    logo_width_fraction: float
    logo_y_offset: int
    caption_font: str
    caption_bold: int
    caption_font_size: int
    caption_primary_colour: str
    caption_outline_colour: str
    caption_border_style: int
    caption_outline: int
    caption_shadow: int
    caption_alignment: int
    caption_margin_v: int
    hook_overlay: OverlayStyle
    close_overlay: OverlayStyle
    ffmpeg_error_tail: int


CONFIG: AssembleConfig = {
    "ffmpeg_binary": "ffmpeg",
    "ffprobe_binary": "ffprobe",
    "bed_music_key": "pipeline-assets/bed_music.mp3",
    "logo_key": "pipeline-assets/stc_logo.png",
    "output_width": 1080,
    "output_height": 1920,
    "output_fps": 30,
    "video_codec": "libx264",
    "audio_codec": "aac",
    "pixel_format": "yuv420p",
    "sample_aspect_ratio": 1,
    "audio_sample_rate": 48000,
    "audio_layout": "stereo",
    "hook_speed": 1.5,
    "bridge_speed": 2.0,
    "outro_speed": 1.5,
    "main_speed": 1.0,
    "main_layout": "blur",
    "blur_radius": 30,
    "blur_power": 1,
    "boundary_fade_seconds": 0.25,
    # The generated bridge_out's final frames stutter; drop this much off its
    # tail before the boundary fade is applied (see _process_clip). 0 disables.
    "bridge_out_tail_trim_seconds": 0.4,
    "silence_threshold_db": -35.0,
    "silence_min_duration_s": 0.8,
    "silence_padding_s": 0.15,
    "voice_volume": 0.68,
    "bed_volume": 0.02,
    "bed_fade_out_s": 1.5,
    "logo_width_fraction": 0.35,
    "logo_y_offset": 120,
    # Caption force_style values are in libass PlayResY=288 units (ffmpeg's
    # SRT->ASS conversion default), NOT output pixels. FontSize=16 is the
    # libass default made explicit; MarginV=50 is ~17% of frame height up
    # from the bottom, centering captions in the blurred band below the
    # 16:9 main video instead of on the footage.
    #
    # FontName is the libass family name — verified with fc-match/fc-scan to
    # resolve to worker/fonts/Montserrat-ExtraBold.ttf via the subtitles filter's
    # fontsdir=. Bold=0 because the ExtraBold weight is baked into the face;
    # Bold=1 makes libass request a 700-weight variant that isn't in the dir and
    # fall back to a system font.
    "caption_font": "Montserrat ExtraBold",
    "caption_bold": 0,
    "caption_font_size": 16,
    "caption_primary_colour": "&H00FFFFFF",
    "caption_outline_colour": "&H00000000",
    "caption_border_style": 1,
    "caption_outline": 3,
    "caption_shadow": 0,
    "caption_alignment": 2,
    "caption_margin_v": 50,
    # drawtext takes a direct fontfile path (freetype, no fontconfig). Hook text
    # uses Anton; close text uses Montserrat ExtraBold — both from worker/fonts/.
    "hook_overlay": {
        "fontfile": str(FONTS_DIR / "Anton-Regular.ttf"),
        "fontsize": 72,
        "fontcolor": "white",
        "borderw": 4,
        "bordercolor": "black",
        "y_fraction": 0.72,
        "appear_time_s": 0.5,
        "anim_duration_s": 0.4,
        "rise_px": 30,
    },
    "close_overlay": {
        "fontfile": str(FONTS_DIR / "Montserrat-ExtraBold.ttf"),
        "fontsize": 72,
        "fontcolor": "white",
        "borderw": 4,
        "bordercolor": "black",
        "y_fraction": 0.72,
        "appear_time_s": 2.5,
        "anim_duration_s": 0.4,
        "rise_px": 30,
    },
    "ffmpeg_error_tail": 1600,
}

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
SILENCE_EVENT_RE = re.compile(
    r"silence_(?P<event>start|end):\s*"
    r"(?P<seconds>[-+]?\d+(?:\.\d+)?)"
)


class AssembleStageContext(Protocol):
    """Context fields consumed by the assemble stage."""

    job: Job
    workspace: Workspace
    conn: Connection[DictRow]
    checkpoint_heartbeat: Callable[[], None]


class AssembleError(RuntimeError):
    """Raised when a final short cannot be assembled safely."""


@dataclass(frozen=True)
class SrtBlock:
    """One parsed master-SRT block with verbatim subtitle text."""

    number: int
    start_ms: int
    end_ms: int
    text: str


@dataclass
class AssembleClip:
    """Validated clip fields and mutable manifest entry."""

    raw: dict[str, object]
    clip_id: str
    start_ms: int
    end_ms: int
    start_block: int
    end_block: int
    generated_keys: dict[AssetName, str]
    assembled: dict[str, str] | None
    hook_text: str | None
    close_text: str | None


@dataclass(frozen=True)
class TextOverlay:
    """One textfile-backed drawtext overlay burned onto a segment."""

    textfile: Path
    style: OverlayStyle


@dataclass(frozen=True)
class SilenceInterval:
    """One silence range reported by ffmpeg."""

    start_s: float
    end_s: float


def run_assemble(ctx: AssembleStageContext) -> None:
    """Assemble and checkpoint every approved clip missing a final short."""

    loaded = _retry_once(
        lambda: jobs.load_manifest(ctx.conn, ctx.job.id),
        description="authoritative DB manifest load",
    )
    manifest, srt_key, clips = validate_assemble_manifest(loaded)
    _run_preflight(ctx, clips)

    master_path = _retry_once(
        lambda: ctx.workspace.download(srt_key, "assemble/master.srt"),
        description=f"master SRT download ({srt_key})",
    )
    master_blocks = parse_master_srt(master_path.read_bytes())
    clip_srts = {
        clip.clip_id: build_clip_srt(
            master_blocks,
            start_block=clip.start_block,
            end_block=clip.end_block,
            clip_start_ms=clip.start_ms,
            clip_id=clip.clip_id,
        )
        for clip in clips
    }
    bed_music = _retry_once(
        lambda: ctx.workspace.download(
            CONFIG["bed_music_key"], "assemble/bed_music.mp3"
        ),
        description=f"static asset download ({CONFIG['bed_music_key']})",
    )
    logo = _retry_once(
        lambda: ctx.workspace.download(CONFIG["logo_key"], "assemble/stc_logo.png"),
        description=f"static asset download ({CONFIG['logo_key']})",
    )

    for clip in clips:
        if clip.assembled is not None:
            logger.info(
                "assemble[%s]: clip=%s checkpoint exists; skipping",
                ctx.job.id,
                clip.clip_id,
            )
            continue
        _process_clip(
            ctx,
            manifest,
            clip,
            clip_srts[clip.clip_id],
            bed_music,
            logo,
        )

    authoritative = _retry_once(
        lambda: jobs.load_manifest(ctx.conn, ctx.job.id),
        description="final authoritative DB manifest load",
    )
    if not isinstance(authoritative, dict):
        raise AssembleError(
            "assemble: authoritative DB manifest missing after checkpoints"
        )
    final_manifest = cast(dict[str, object], authoritative)
    manifest_path = ctx.workspace.path("assemble/manifest.json")
    manifest_path.write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _retry_once(
        lambda: ctx.workspace.upload(
            manifest_path, f"pipeline/{ctx.job.id}/manifest.json"
        ),
        description="final manifest S3 upload",
    )
    logger.info("assemble[%s]: all approved clips complete", ctx.job.id)


def validate_assemble_manifest(
    manifest: object | None,
) -> tuple[dict[str, object], str, list[AssembleClip]]:
    """Copy and validate all manifest data required before clip processing."""

    if manifest is None:
        raise AssembleError("assemble: job manifest is missing")
    if not isinstance(manifest, dict):
        raise AssembleError("assemble: job manifest must be a JSON object")

    data = cast(dict[str, object], copy.deepcopy(manifest))
    srt_key = data.get("srt_file")
    if not isinstance(srt_key, str) or not srt_key:
        raise AssembleError(
            "assemble: manifest srt_file must be a non-empty S3 key"
        )

    raw_clips = data.get("clips")
    if not isinstance(raw_clips, list):
        raise AssembleError("assemble: manifest clips must be a list")

    approved: list[AssembleClip] = []
    seen_ids: set[str] = set()
    for index, raw_clip in enumerate(raw_clips):
        if not isinstance(raw_clip, dict):
            raise AssembleError(
                f"assemble: manifest clips[{index}] must be an object"
            )
        clip = cast(dict[str, object], raw_clip)
        if clip.get("approved") is not True:
            continue

        clip_id = _required_clip_string(clip, "id", index)
        if (
            clip_id in {".", ".."}
            or Path(clip_id).name != clip_id
            or "\x00" in clip_id
        ):
            raise AssembleError(
                f"assemble: manifest clips[{index}].id is not a safe filename"
            )
        if clip_id in seen_ids:
            raise AssembleError(f"assemble: duplicate approved clip id {clip_id!r}")
        seen_ids.add(clip_id)

        start_ms = parse_srt_timestamp(
            _required_clip_string(clip, "start", index)
        )
        end_ms = parse_srt_timestamp(_required_clip_string(clip, "end", index))
        if end_ms <= start_ms:
            raise AssembleError(
                f"assemble: clip {clip_id} end must be after start"
            )
        start_block = _required_clip_int(clip, "start_block", index)
        end_block = _required_clip_int(clip, "end_block", index)
        if start_block < 1 or end_block < start_block:
            raise AssembleError(
                f"assemble: clip {clip_id} has invalid block range "
                f"{start_block}-{end_block}"
            )

        generated = clip.get("generated")
        if not isinstance(generated, dict):
            raise AssembleError(
                f"assemble: clip {clip_id} missing generated asset package"
            )
        generated_map = cast(dict[str, object], generated)
        generated_keys: dict[AssetName, str] = {}
        for asset in ASSET_ORDER:
            generated_keys[asset] = _generated_s3_key(
                generated_map, clip_id, asset
            )

        approved.append(
            AssembleClip(
                raw=clip,
                clip_id=clip_id,
                start_ms=start_ms,
                end_ms=end_ms,
                start_block=start_block,
                end_block=end_block,
                generated_keys=generated_keys,
                assembled=_assembled_checkpoint(clip, clip_id),
                hook_text=_optional_clip_text(clip, "hook_text", index),
                close_text=_optional_clip_text(clip, "close_text", index),
            )
        )

    if not approved:
        raise AssembleError("assemble: manifest has no approved clips")
    return data, srt_key, approved


def parse_srt_timestamp(timestamp: str) -> int:
    """Parse an SRT timestamp into integer milliseconds."""

    match = SRT_TIMESTAMP_RE.fullmatch(timestamp)
    if match is None:
        raise AssembleError(f"assemble: invalid SRT timestamp {timestamp!r}")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    milliseconds = int(match.group("milliseconds"))
    if minutes >= 60 or seconds >= 60:
        raise AssembleError(f"assemble: invalid SRT timestamp {timestamp!r}")
    return (((hours * 60) + minutes) * 60 + seconds) * 1000 + milliseconds


def parse_master_srt(content: bytes) -> dict[int, SrtBlock]:
    """Parse master SRT structure while retaining each subtitle text verbatim."""

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AssembleError("assemble: master SRT must be valid UTF-8") from exc

    payload = text.strip("\r\n")
    if not payload:
        raise AssembleError("assemble: master SRT has no subtitle blocks")

    blocks: dict[int, SrtBlock] = {}
    for position, chunk in enumerate(re.split(r"(?:\r\n|\n){2,}", payload), start=1):
        match = SRT_BLOCK_RE.fullmatch(chunk)
        if match is None:
            raise AssembleError(
                f"assemble: master SRT block at position {position} is malformed"
            )
        number = int(match.group("number"))
        if number in blocks:
            raise AssembleError(
                f"assemble: master SRT contains duplicate block {number}"
            )
        blocks[number] = SrtBlock(
            number=number,
            start_ms=parse_srt_timestamp(match.group("start")),
            end_ms=parse_srt_timestamp(match.group("end")),
            text=match.group("text"),
        )
    return blocks


def build_clip_srt(
    blocks: Mapping[int, SrtBlock],
    *,
    start_block: int,
    end_block: int,
    clip_start_ms: int,
    clip_id: str,
) -> bytes:
    """Slice, re-time, and renumber master blocks without changing their text."""

    output: list[str] = []
    for output_number, block_number in enumerate(
        range(start_block, end_block + 1), start=1
    ):
        block = blocks.get(block_number)
        if block is None:
            raise AssembleError(
                f"assemble: clip {clip_id} references missing master SRT "
                f"block {block_number}"
            )
        shifted_start = block.start_ms - clip_start_ms
        shifted_end = block.end_ms - clip_start_ms
        if shifted_start < 0 or shifted_end < 0:
            raise AssembleError(
                f"assemble: clip {clip_id} block {block_number} shifts to "
                "a negative subtitle timestamp"
            )
        output.append(
            f"{output_number}\n"
            f"{format_srt_timestamp(shifted_start)} --> "
            f"{format_srt_timestamp(shifted_end)}\n"
            f"{block.text}"
        )
    return ("\n\n".join(output) + "\n").encode("utf-8")


def format_srt_timestamp(milliseconds: int) -> str:
    """Format non-negative milliseconds as an SRT timestamp."""

    if milliseconds < 0:
        raise ValueError("SRT milliseconds cannot be negative")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_generated_segment_command(
    source: Path,
    output: Path,
    *,
    speed: float,
    logo: Path | None = None,
    text: TextOverlay | None = None,
) -> list[str]:
    """Build a normalized, sped-up generated-segment command with silent audio."""

    width = CONFIG["output_width"]
    height = CONFIG["output_height"]
    fps = CONFIG["output_fps"]
    video_filters = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setpts=PTS/{speed},fps={fps}"
    )
    drawtext = "" if text is None else f",{_drawtext_filter(text)}"
    command = [CONFIG["ffmpeg_binary"], "-y", "-i", str(source)]
    if logo is None:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                _silent_audio_source(),
                "-filter_complex",
                f"[0:v]{video_filters}{drawtext},"
                f"{_normalized_video_output()}[v]",
                "-map",
                "[v]",
                "-map",
                "1:a",
            ]
        )
    else:
        logo_width = round(width * CONFIG["logo_width_fraction"])
        command.extend(
            [
                "-i",
                str(logo),
                "-f",
                "lavfi",
                "-i",
                _silent_audio_source(),
                "-filter_complex",
                f"[0:v]{video_filters}[base];"
                f"[1:v]scale={logo_width}:-1[logo];"
                f"[base][logo]overlay=(W-w)/2:{CONFIG['logo_y_offset']}:"
                f"eof_action=repeat{drawtext},{_normalized_video_output()}[v]",
                "-map",
                "[v]",
                "-map",
                "2:a",
            ]
        )
    command.extend(_encoding_args(output, shortest=True))
    return command


def build_main_caption_command(
    source: Path,
    srt_path: Path,
    output: Path,
) -> list[str]:
    """Build the main layout/caption command; voice speed remains unchanged."""

    width = CONFIG["output_width"]
    height = CONFIG["output_height"]
    layout = CONFIG["main_layout"]
    if layout == "blur":
        layout_filter = (
            f"[0:v]split=2[background][foreground];"
            f"[background]scale={width}:{height}:"
            "force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"boxblur=luma_radius={CONFIG['blur_radius']}:"
            f"luma_power={CONFIG['blur_power']}[blurred];"
            f"[foreground]scale={width}:{height}:"
            "force_original_aspect_ratio=decrease[fit];"
            "[blurred][fit]overlay=(W-w)/2:(H-h)/2[laidout]"
        )
    elif layout == "bars":
        layout_filter = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black[laidout]"
        )
    elif layout == "crop":
        layout_filter = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}[laidout]"
        )
    else:
        raise AssembleError(f"assemble: unsupported main layout {layout!r}")

    style = ",".join(
        (
            f"FontName={CONFIG['caption_font']}",
            f"Bold={CONFIG['caption_bold']}",
            f"FontSize={CONFIG['caption_font_size']}",
            f"PrimaryColour={CONFIG['caption_primary_colour']}",
            f"OutlineColour={CONFIG['caption_outline_colour']}",
            f"BorderStyle={CONFIG['caption_border_style']}",
            f"Outline={CONFIG['caption_outline']}",
            f"Shadow={CONFIG['caption_shadow']}",
            f"Alignment={CONFIG['caption_alignment']}",
            f"MarginV={CONFIG['caption_margin_v']}",
        )
    )
    subtitle_path = _escape_filter_value(str(srt_path))
    fonts_dir = _escape_filter_value(str(FONTS_DIR))
    filter_complex = (
        f"{layout_filter};"
        f"[laidout]subtitles=filename='{subtitle_path}':"
        f"fontsdir='{fonts_dir}':"
        f"force_style='{style}',"
        f"setpts=PTS/{CONFIG['main_speed']},"
        f"fps={CONFIG['output_fps']},"
        f"{_normalized_video_output()}[v];"
        f"[0:a]aformat=sample_rates={CONFIG['audio_sample_rate']}:"
        f"channel_layouts={CONFIG['audio_layout']},"
        f"volume={CONFIG['voice_volume']}[a]"
    )
    command = [
        CONFIG["ffmpeg_binary"],
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
    ]
    command.extend(_encoding_args(output, shortest=False))
    return command


def build_silence_detect_command(source: Path) -> list[str]:
    """Build the analysis command for silence ranges in the captioned main clip."""

    return [
        CONFIG["ffmpeg_binary"],
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-af",
        f"silencedetect=noise={CONFIG['silence_threshold_db']}dB:"
        f"d={CONFIG['silence_min_duration_s']}",
        "-f",
        "null",
        "-",
    ]


def parse_silence_intervals(stderr: str) -> list[SilenceInterval]:
    """Parse complete silence intervals from ffmpeg's silencedetect output."""

    intervals: list[SilenceInterval] = []
    pending_start: float | None = None
    for match in SILENCE_EVENT_RE.finditer(stderr):
        event = match.group("event")
        seconds = float(match.group("seconds"))
        if event == "start":
            pending_start = seconds
        elif pending_start is not None:
            if seconds - pending_start > CONFIG["silence_min_duration_s"]:
                intervals.append(
                    SilenceInterval(start_s=pending_start, end_s=seconds)
                )
            pending_start = None
    return intervals


def build_silence_cut_command(
    source: Path,
    output: Path,
    intervals: Sequence[SilenceInterval],
) -> list[str]:
    """Build synchronized video/audio removal for detected silence interiors."""

    removals: list[SilenceInterval] = []
    for interval in intervals:
        start = max(0.0, interval.start_s + CONFIG["silence_padding_s"])
        end = interval.end_s - CONFIG["silence_padding_s"]
        if end > start:
            removals.append(SilenceInterval(start_s=start, end_s=end))

    if removals:
        terms = [
            f"between(t\\,{interval.start_s:.6f}\\,{interval.end_s:.6f})"
            for interval in removals
        ]
        keep_expression = f"not({'+'.join(terms)})"
        video_filter = (
            f"[0:v]select='{keep_expression}',"
            f"setpts=N/({CONFIG['output_fps']}*TB),"
            f"{_normalized_video_output()}[v]"
        )
        audio_filter = (
            f"[0:a]aselect='{keep_expression}',"
            "asetpts=N/SR/TB[a]"
        )
    else:
        video_filter = f"[0:v]{_normalized_video_output()}[v]"
        audio_filter = "[0:a]anull[a]"

    command = [
        CONFIG["ffmpeg_binary"],
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        f"{video_filter};{audio_filter}",
        "-map",
        "[v]",
        "-map",
        "[a]",
    ]
    command.extend(_encoding_args(output, shortest=False))
    return command


def build_duration_probe_command(source: Path) -> list[str]:
    """Build the ffprobe command for a processed segment's final duration."""

    return [
        CONFIG["ffprobe_binary"],
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]


def build_boundary_fade_command(
    source: Path,
    output: Path,
    *,
    duration_s: float,
    fade_in: bool,
    fade_out: bool,
) -> list[str]:
    """Build the dip-to-black fade pass; audio passes through untouched."""

    fade = CONFIG["boundary_fade_seconds"]
    filters: list[str] = []
    if fade_in:
        filters.append(f"fade=t=in:st=0:d={fade}")
    if fade_out:
        start = max(0.0, duration_s - fade)
        filters.append(f"fade=t=out:st={start:.6f}:d={fade}")
    filters.append(_normalized_video_output())
    return [
        CONFIG["ffmpeg_binary"],
        "-y",
        "-i",
        str(source),
        "-vf",
        ",".join(filters),
        "-c:v",
        CONFIG["video_codec"],
        "-pix_fmt",
        CONFIG["pixel_format"],
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
    ]


def build_tail_trim_command(
    source: Path,
    output: Path,
    *,
    keep_duration_s: float,
) -> list[str]:
    """Build a command that keeps only the first ``keep_duration_s`` of a segment.

    Used to drop the stuttering tail of the generated bridge_out before its
    boundary fade. Video is re-encoded so the cut is frame-accurate; the silent
    audio track is copied.
    """

    return [
        CONFIG["ffmpeg_binary"],
        "-y",
        "-i",
        str(source),
        "-t",
        f"{keep_duration_s:.6f}",
        "-c:v",
        CONFIG["video_codec"],
        "-pix_fmt",
        CONFIG["pixel_format"],
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
    ]


def build_final_command(
    segments: Sequence[Path],
    bed_music: Path,
    output: Path,
) -> list[str]:
    """Build the fixed-order concat plus full-duration bed-music mix."""

    command = [CONFIG["ffmpeg_binary"], "-y"]
    for segment in segments:
        command.extend(["-i", str(segment)])
    command.extend(["-stream_loop", "-1", "-i", str(bed_music)])

    filters: list[str] = []
    concat_inputs: list[str] = []
    for index in range(len(segments)):
        filters.append(f"[{index}:v]setpts=PTS-STARTPTS[v{index}]")
        filters.append(f"[{index}:a]asetpts=PTS-STARTPTS[a{index}]")
        concat_inputs.extend((f"[v{index}]", f"[a{index}]"))
    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(segments)}:v=1:a=1[video][voice]"
    )
    bed_index = len(segments)
    filters.append(
        f"[{bed_index}:a]aformat=sample_rates={CONFIG['audio_sample_rate']}:"
        f"channel_layouts={CONFIG['audio_layout']},"
        f"volume={CONFIG['bed_volume']}[bed]"
    )
    filters.append(
        "[voice][bed]amix=inputs=2:duration=first:normalize=0:"
        "dropout_transition=0,"
        "areverse,"
        f"afade=t=in:d={CONFIG['bed_fade_out_s']},"
        "areverse,asetpts=N/SR/TB[audio]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[video]",
            "-map",
            "[audio]",
            "-c:v",
            CONFIG["video_codec"],
            "-pix_fmt",
            CONFIG["pixel_format"],
            "-r",
            str(CONFIG["output_fps"]),
            "-c:a",
            CONFIG["audio_codec"],
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return command


def _run_preflight(
    ctx: AssembleStageContext,
    clips: Sequence[AssembleClip],
) -> None:
    executable = CONFIG["ffmpeg_binary"]
    if shutil.which(executable) is None:
        raise AssembleError(
            f"assemble: ffmpeg executable {executable!r} not found on PATH"
        )
    probe_executable = CONFIG["ffprobe_binary"]
    if shutil.which(probe_executable) is None:
        raise AssembleError(
            f"assemble: ffprobe executable {probe_executable!r} not found "
            "on PATH"
        )

    process = subprocess.run(
        [executable, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        raise AssembleError(
            "assemble: ffmpeg filter probe failed: "
            f"{process.stderr[-CONFIG['ffmpeg_error_tail']:]}"
        )
    if not any(
        len(parts := line.split()) >= 2 and parts[1] == "subtitles"
        for line in process.stdout.splitlines()
    ):
        raise AssembleError(
            "assemble: ffmpeg subtitles filter is unavailable; "
            "install an ffmpeg build with libass"
        )

    _head_required_object(
        ctx,
        CONFIG["bed_music_key"],
        description="static bed music",
    )
    _head_required_object(
        ctx,
        CONFIG["logo_key"],
        description="static STC logo",
    )
    for clip in clips:
        _head_required_object(
            ctx,
            main_clip_key(ctx.job.id, clip.clip_id),
            description=f"main clip {clip.clip_id}",
        )


def _process_clip(
    ctx: AssembleStageContext,
    manifest: dict[str, object],
    clip: AssembleClip,
    clip_srt_content: bytes,
    bed_music: Path,
    logo: Path,
) -> None:
    logger.info("assemble[%s]: processing clip=%s", ctx.job.id, clip.clip_id)
    with tempfile.TemporaryDirectory(
        prefix=f"{clip.clip_id}-", dir=ctx.workspace.dir
    ) as temp_name:
        directory = Path(temp_name)
        clip_srt = directory / f"{clip.clip_id}.srt"
        clip_srt.write_bytes(clip_srt_content)

        main_source = directory / "main.mp4"
        _download_to(
            ctx,
            main_clip_key(ctx.job.id, clip.clip_id),
            main_source,
            description=f"clip {clip.clip_id} main clip download",
        )
        sources: dict[AssetName, Path] = {}
        for asset in ASSET_ORDER:
            source = directory / f"{asset}.mp4"
            _download_to(
                ctx,
                clip.generated_keys[asset],
                source,
                description=f"clip {clip.clip_id} asset {asset} download",
            )
            sources[asset] = source

        # Measure each overlay against its real TTF and shrink the fontsize so
        # the rendered text can never exceed the frame (a None overlay -- empty
        # or absent manifest field -- is logged and skipped inside the helper).
        overlays: dict[AssetName, TextOverlay | None] = {
            "hook": _prepare_overlay(
                directory,
                job_id=ctx.job.id,
                clip_id=clip.clip_id,
                asset="hook",
                field="hook_text",
                text=clip.hook_text,
                base_style=CONFIG["hook_overlay"],
            ),
            "outro": _prepare_overlay(
                directory,
                job_id=ctx.job.id,
                clip_id=clip.clip_id,
                asset="outro",
                field="close_text",
                text=clip.close_text,
                base_style=CONFIG["close_overlay"],
            ),
            "bridge_in": None,
            "bridge_out": None,
        }
        normalized: dict[AssetName, Path] = {}
        speeds: dict[AssetName, float] = {
            "hook": CONFIG["hook_speed"],
            "outro": CONFIG["outro_speed"],
            "bridge_in": CONFIG["bridge_speed"],
            "bridge_out": CONFIG["bridge_speed"],
        }
        for asset in ASSET_ORDER:
            output = directory / f"{asset}_normalized.mp4"
            _run_ffmpeg(
                build_generated_segment_command(
                    sources[asset],
                    output,
                    speed=speeds[asset],
                    logo=logo if asset == "outro" else None,
                    text=overlays[asset],
                ),
                description=f"clip {clip.clip_id} asset {asset} normalization",
            )
            normalized[asset] = output

        captioned_main = directory / "main_captioned.mp4"
        _run_ffmpeg(
            build_main_caption_command(main_source, clip_srt, captioned_main),
            description=f"clip {clip.clip_id} main caption burn",
        )
        silence_output = _run_ffmpeg(
            build_silence_detect_command(captioned_main),
            description=f"clip {clip.clip_id} silence detection",
        )
        trimmed_main = directory / "main_trimmed.mp4"
        _run_ffmpeg(
            build_silence_cut_command(
                captioned_main,
                trimmed_main,
                parse_silence_intervals(silence_output),
            ),
            description=f"clip {clip.clip_id} silence removal",
        )

        # bridge_out tail trim: the generated bridge_out's final frames stutter,
        # so drop its last BRIDGE_OUT_TAIL_TRIM seconds BEFORE the boundary fade,
        # so the dip-to-black lands on the trimmed endpoint. Guard: if the
        # segment is shorter than trim + fade, skip it (trimming would leave no
        # room for the fade, or produce a zero-length clip).
        bridge_out_processed = normalized["bridge_out"]
        tail_trim = CONFIG["bridge_out_tail_trim_seconds"]
        if tail_trim > 0:
            fade = CONFIG["boundary_fade_seconds"]
            bridge_out_duration = _probe_duration(
                normalized["bridge_out"],
                description=f"clip {clip.clip_id} segment bridge_out (pre-trim)",
            )
            if bridge_out_duration < tail_trim + fade:
                logger.warning(
                    "assemble[%s]: clip %s bridge_out is %.3fs, shorter than "
                    "trim (%.3fs) + fade (%.3fs); skipping tail trim",
                    ctx.job.id,
                    clip.clip_id,
                    bridge_out_duration,
                    tail_trim,
                    fade,
                )
            else:
                trimmed_bridge_out = directory / "bridge_out_trimmed.mp4"
                _run_ffmpeg(
                    build_tail_trim_command(
                        normalized["bridge_out"],
                        trimmed_bridge_out,
                        keep_duration_s=bridge_out_duration - tail_trim,
                    ),
                    description=(
                        f"clip {clip.clip_id} segment bridge_out tail trim"
                    ),
                )
                bridge_out_processed = trimmed_bridge_out

        processed: dict[SegmentName, Path] = {
            "hook": normalized["hook"],
            "bridge_in": normalized["bridge_in"],
            "main": trimmed_main,
            "bridge_out": bridge_out_processed,
            "outro": normalized["outro"],
        }
        faded: dict[SegmentName, Path] = {}
        for segment, (fade_in, fade_out) in BOUNDARY_FADE_SIDES.items():
            faded_output = directory / f"{segment}_faded.mp4"
            duration_s = _probe_duration(
                processed[segment],
                description=f"clip {clip.clip_id} segment {segment}",
            )
            _run_ffmpeg(
                build_boundary_fade_command(
                    processed[segment],
                    faded_output,
                    duration_s=duration_s,
                    fade_in=fade_in,
                    fade_out=fade_out,
                ),
                description=f"clip {clip.clip_id} segment {segment} "
                "boundary fades",
            )
            faded[segment] = faded_output

        final_path = directory / f"final_{clip.clip_id}_9x16.mp4"
        _run_ffmpeg(
            build_final_command(
                (
                    faded["hook"],
                    faded["bridge_in"],
                    faded["main"],
                    faded["bridge_out"],
                    faded["outro"],
                ),
                bed_music,
                final_path,
            ),
            description=f"clip {clip.clip_id} final concat and audio mix",
        )

        output_key = (
            f"pipeline/{ctx.job.id}/final/final_{clip.clip_id}_9x16.mp4"
        )
        _retry_once(
            lambda: ctx.workspace.upload(final_path, output_key),
            description=f"clip {clip.clip_id} final S3 upload",
        )
        checkpoint = {
            "s3Key": output_key,
            "completedAt": datetime.now(timezone.utc).isoformat(),
        }
        clip.raw["assembled"] = checkpoint
        _retry_once(
            lambda: jobs.save_manifest_checkpoint(ctx.conn, ctx.job.id, manifest),
            description=f"clip {clip.clip_id} assembled DB checkpoint",
        )
        ctx.checkpoint_heartbeat()
        logger.info(
            "assemble[%s]: clip=%s checkpointed key=%s",
            ctx.job.id,
            clip.clip_id,
            output_key,
        )


def _write_overlay_textfile(
    directory: Path,
    name: str,
    text: str | None,
    style: OverlayStyle,
) -> TextOverlay | None:
    """Write the exact manifest string as UTF-8 bytes for drawtext textfile=."""

    if text is None:
        return None
    textfile = directory / f"{name}.txt"
    textfile.write_bytes(text.encode("utf-8"))
    return TextOverlay(textfile=textfile, style=style)


def _overlay_budget_px() -> float:
    """Usable overlay width: the output frame minus a 5% safe margin each side."""

    return CONFIG["output_width"] * (1 - 2 * OVERLAY_FRAME_MARGIN)


def _font_metrics(fontfile: str) -> tuple[int, dict[int, int]]:
    """Return (unitsPerEm, {codepoint: advanceWidth}) for a TTF, cached per file."""

    cached = _FONT_METRICS_CACHE.get(fontfile)
    if cached is not None:
        return cached
    from fontTools.ttLib import TTFont

    font = TTFont(fontfile)
    try:
        units_per_em = int(font["head"].unitsPerEm)
        hmtx = font["hmtx"]
        advances: dict[int, int] = {
            int(codepoint): int(hmtx[glyph_name][0])
            for codepoint, glyph_name in font.getBestCmap().items()
        }
    finally:
        font.close()
    metrics = (units_per_em, advances)
    _FONT_METRICS_CACHE[fontfile] = metrics
    return metrics


def _measure_overlay_width_px(text: str, fontfile: str, fontsize: int) -> float:
    """Sum of glyph advances for ``text`` at ``fontsize`` in px (kerning ignored).

    advance_px = advanceWidth * fontsize / unitsPerEm. Missing glyphs fall back
    to the space advance so an exotic character never measures as zero width.
    """

    units_per_em, advances = _font_metrics(fontfile)
    if units_per_em <= 0:
        return 0.0
    fallback = advances.get(ord(" "), units_per_em // 2)
    total_units = sum(advances.get(ord(char), fallback) for char in text)
    return total_units * fontsize / units_per_em


def _fit_overlay_fontsize(text: str, style: OverlayStyle) -> int:
    """Largest integer fontsize <= base whose rendered width fits the frame.

    Rendered width is the glyph advance sum plus 2*borderw (the border sits
    outside drawtext's text_w). Width scales linearly with fontsize, so the
    ideal size is base * glyph_budget / width_at_base. Result is floored at
    OVERLAY_MIN_FONTSIZE; the caller warns when even the floor overflows.
    """

    base_fontsize = style["fontsize"]
    glyph_budget = _overlay_budget_px() - 2 * style["borderw"]
    width_at_base = _measure_overlay_width_px(text, style["fontfile"], base_fontsize)
    if width_at_base <= glyph_budget:
        return base_fontsize
    ideal = base_fontsize * glyph_budget / width_at_base
    return max(OVERLAY_MIN_FONTSIZE, min(base_fontsize, int(ideal)))


def _prepare_overlay(
    directory: Path,
    *,
    job_id: str,
    clip_id: str,
    asset: AssetName,
    field: str,
    text: str | None,
    base_style: OverlayStyle,
) -> TextOverlay | None:
    """Measure, shrink-to-fit, log, and materialize one overlay (or None)."""

    if text is None:
        logger.info(
            "assemble[%s]: clip=%s asset=%s no overlay text; skipping drawtext",
            job_id,
            clip_id,
            asset,
        )
        return None

    base_fontsize = base_style["fontsize"]
    fontfile = base_style["fontfile"]
    budget = _overlay_budget_px()
    border = 2 * base_style["borderw"]
    width_at_base = _measure_overlay_width_px(text, fontfile, base_fontsize)
    fontsize = _fit_overlay_fontsize(text, base_style)

    if fontsize < base_fontsize:
        logger.info(
            "assemble[%s]: clip=%s asset=%s overlay shrink: len=%d "
            "width@%d=%.1fpx (+%dpx border) budget=%.0fpx -> fontsize=%d",
            job_id,
            clip_id,
            asset,
            len(text),
            base_fontsize,
            width_at_base,
            border,
            budget,
            fontsize,
        )
        rendered = _measure_overlay_width_px(text, fontfile, fontsize) + border
        if rendered > budget:
            logger.warning(
                "assemble[%s]: clip=%s asset=%s overlay STILL overflows at "
                "floor fontsize=%d: rendered=%.1fpx > budget=%.0fpx; "
                "proceeding (text may be clipped)",
                job_id,
                clip_id,
                asset,
                fontsize,
                rendered,
                budget,
            )

    style = cast(OverlayStyle, {**base_style, "fontsize": fontsize})
    return _write_overlay_textfile(directory, field, text, style)


def _generated_s3_key(
    generated: Mapping[str, object],
    clip_id: str,
    asset: AssetName,
) -> str:
    raw_checkpoint = generated.get(asset)
    if not isinstance(raw_checkpoint, dict):
        raise AssembleError(
            f"assemble: clip {clip_id} missing generated asset {asset}"
        )
    checkpoint = cast(dict[str, object], raw_checkpoint)
    s3_key = checkpoint.get("s3Key")
    if not isinstance(s3_key, str) or not s3_key:
        raise AssembleError(
            f"assemble: clip {clip_id} missing generated asset {asset}.s3Key"
        )
    for field in ("generationId", "completedAt"):
        value = checkpoint.get(field)
        if not isinstance(value, str) or not value:
            raise AssembleError(
                f"assemble: clip {clip_id} generated asset "
                f"{asset}.{field} must be a non-empty string"
            )
    credits = checkpoint.get("estimatedCredits")
    if isinstance(credits, bool) or not isinstance(credits, int):
        raise AssembleError(
            f"assemble: clip {clip_id} generated asset "
            f"{asset}.estimatedCredits must be an integer"
        )
    return s3_key


def _assembled_checkpoint(
    clip: Mapping[str, object],
    clip_id: str,
) -> dict[str, str] | None:
    raw_checkpoint = clip.get("assembled")
    if raw_checkpoint is None:
        return None
    if not isinstance(raw_checkpoint, dict):
        raise AssembleError(
            f"assemble: clip {clip_id} assembled checkpoint must be an object"
        )
    checkpoint = cast(dict[str, object], raw_checkpoint)
    result: dict[str, str] = {}
    for field in ("s3Key", "completedAt"):
        value = checkpoint.get(field)
        if not isinstance(value, str) or not value:
            raise AssembleError(
                f"assemble: clip {clip_id} assembled.{field} "
                "must be a non-empty string"
            )
        result[field] = value
    return result


def _required_clip_string(
    clip: Mapping[str, object],
    field: str,
    index: int,
) -> str:
    value = clip.get(field)
    if not isinstance(value, str) or not value:
        raise AssembleError(
            f"assemble: manifest clips[{index}].{field} "
            "must be a non-empty string"
        )
    return value


def _optional_clip_text(
    clip: Mapping[str, object],
    field: str,
    index: int,
) -> str | None:
    """Return overlay text, treating an absent or empty field as no overlay."""

    value = clip.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AssembleError(
            f"assemble: manifest clips[{index}].{field} must be a string"
        )
    return value or None


def _required_clip_int(
    clip: Mapping[str, object],
    field: str,
    index: int,
) -> int:
    value = clip.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssembleError(
            f"assemble: manifest clips[{index}].{field} must be an integer"
        )
    return value


def _head_required_object(
    ctx: AssembleStageContext,
    s3_key: str,
    *,
    description: str,
) -> None:
    try:
        ctx.workspace.s3.head_object(Bucket=ctx.workspace.bucket, Key=s3_key)
    except Exception as exc:
        raise AssembleError(
            f"assemble: required {description} missing or inaccessible: {s3_key}"
        ) from exc


def _download_to(
    ctx: AssembleStageContext,
    s3_key: str,
    local_path: Path,
    *,
    description: str,
) -> None:
    relative_name = local_path.relative_to(ctx.workspace.dir)
    _retry_once(
        lambda: ctx.workspace.download(s3_key, str(relative_name)),
        description=description,
    )


def _drawtext_filter(text: TextOverlay) -> str:
    style = text.style
    appear = style["appear_time_s"]
    duration = style["anim_duration_s"]
    # Appearance animation: fade-in (alpha 0->1) plus a gentle rise (text
    # starts rise_px below its resting line and settles over anim_duration_s).
    # progress = clamped (t-appear)/duration; enable stays as a coarse gate so
    # no draw cost is paid before the appear time (alpha alone would draw
    # invisible text every frame from t=0).
    progress = f"(t-{appear})/{duration}"
    return (
        f"drawtext=fontfile='{_escape_filter_value(style['fontfile'])}':"
        f"textfile='{_escape_filter_value(str(text.textfile))}':"
        # Render the textfile bytes literally. drawtext's default
        # expansion=normal treats a bare '%' (e.g. hook_text 'PUEDES PERDER
        # 80%') as a malformed %{...} token and silently renders an EMPTY
        # frame -- ffmpeg exits 0, encodes, and writes nothing to stderr. We
        # emit no %{...} features, so literal rendering is the correct semantic.
        "expansion=none:"
        f"fontsize={style['fontsize']}:"
        f"fontcolor={style['fontcolor']}:"
        f"borderw={style['borderw']}:"
        f"bordercolor={style['bordercolor']}:"
        "x=(w-text_w)/2:"
        f"y='h*{style['y_fraction']}+"
        f"{style['rise_px']}*(1-min(1\\,{progress}))':"
        f"alpha='min(1\\,max(0\\,{progress}))':"
        f"enable='gte(t\\,{appear})'"
    )


def _silent_audio_source() -> str:
    return (
        f"anullsrc=r={CONFIG['audio_sample_rate']}:"
        f"cl={CONFIG['audio_layout']}"
    )


def _normalized_video_output() -> str:
    return (
        f"format={CONFIG['pixel_format']},"
        f"setsar={CONFIG['sample_aspect_ratio']}"
    )


def _encoding_args(output: Path, *, shortest: bool) -> list[str]:
    args = [
        "-c:v",
        CONFIG["video_codec"],
        "-pix_fmt",
        CONFIG["pixel_format"],
        "-c:a",
        CONFIG["audio_codec"],
        "-movflags",
        "+faststart",
    ]
    if shortest:
        args.append("-shortest")
    args.append(str(output))
    return args


def _escape_filter_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for character in ("'", ":", ",", "[", "]", ";"):
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _probe_duration(source: Path, *, description: str) -> float:
    """Return the container duration of a processed intermediate in seconds."""

    process = subprocess.run(
        build_duration_probe_command(source),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        stderr_tail = process.stderr[-CONFIG["ffmpeg_error_tail"] :]
        raise AssembleError(
            f"assemble: {description} ffprobe exited "
            f"{process.returncode}: {stderr_tail}"
        )
    try:
        duration = float(process.stdout.strip())
    except ValueError as exc:
        raise AssembleError(
            f"assemble: {description} ffprobe returned an unparsable "
            f"duration {process.stdout.strip()!r}"
        ) from exc
    if duration <= 0:
        raise AssembleError(
            f"assemble: {description} ffprobe returned a non-positive "
            f"duration {duration}"
        )
    return duration


def _run_ffmpeg(command: list[str], *, description: str) -> str:
    logger.info("assemble: %s ffmpeg argv: %s", description, shlex.join(command))
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        stderr_tail = process.stderr[-CONFIG["ffmpeg_error_tail"] :]
        raise AssembleError(
            f"assemble: {description} ffmpeg exited "
            f"{process.returncode}: {stderr_tail}"
        )
    return process.stderr


def _retry_once(operation: Callable[[], T], *, description: str) -> T:
    try:
        return operation()
    except Exception as first_error:
        logger.warning("%s failed (%s); retrying once", description, first_error)
    try:
        return operation()
    except Exception as second_error:
        raise AssembleError(
            f"assemble: {description} failed after retry: {second_error}"
        ) from second_error
