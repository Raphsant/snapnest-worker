"""Cut stage: render approved manifest clips and publish them to S3."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from worker.stages import StageContext

logger = logging.getLogger(__name__)

FFMPEG_ERROR_TAIL = 4000
SRT_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}")


class CutError(RuntimeError):
    """Raised when approved clip artifacts cannot be produced."""


@dataclass(frozen=True)
class ApprovedClip:
    """Validated fields needed to render one approved clip."""

    clip_id: str
    start: str
    end: str
    duration_seconds: int | float


@dataclass(frozen=True)
class RenderedClip:
    """Local files produced for one approved clip."""

    clip: ApprovedClip
    main: Path
    first_five: Path
    last_five: Path


def run_cut(ctx: StageContext) -> None:
    """Render every approved clip from the DB manifest and upload the results."""

    manifest, clips = validate_manifest(ctx.job.manifest)
    prefix = f"pipeline/{ctx.job.id}/"
    source_key = f"{prefix}session_bleeped.mp4"

    logger.info("cut[%s]: downloading bleeped video %s", ctx.job.id, source_key)
    source = ctx.workspace.download(source_key, "session_bleeped.mp4")

    rendered: list[RenderedClip] = []
    for clip in clips:
        main = ctx.workspace.path(f"{clip.clip_id}.mp4")
        first_five = ctx.workspace.path(f"{clip.clip_id}_first5.mp4")
        last_five = ctx.workspace.path(f"{clip.clip_id}_last5.mp4")

        _run_ffmpeg(build_main_clip_command(source, main, clip.start, clip.end))
        _run_ffmpeg(build_first_five_command(main, first_five))
        _run_ffmpeg(build_last_five_command(main, last_five))

        logger.info(
            "cut[%s]: clip=%s start=%s end=%s duration=%ss "
            "sizes(main=%d, first5=%d, last5=%d)",
            ctx.job.id,
            clip.clip_id,
            clip.start,
            clip.end,
            clip.duration_seconds,
            main.stat().st_size,
            first_five.stat().st_size,
            last_five.stat().st_size,
        )
        rendered.append(
            RenderedClip(
                clip=clip,
                main=main,
                first_five=first_five,
                last_five=last_five,
            )
        )

    for result in rendered:
        clip_id = result.clip.clip_id
        ctx.workspace.upload(result.main, f"{prefix}clips/{clip_id}.mp4")
        ctx.workspace.upload(
            result.first_five,
            f"{prefix}clips/subclips/{clip_id}_first5.mp4",
        )
        ctx.workspace.upload(
            result.last_five,
            f"{prefix}clips/subclips/{clip_id}_last5.mp4",
        )

    manifest_path = ctx.workspace.path("manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ctx.workspace.upload(manifest_path, f"{prefix}manifest.json")
    logger.info("cut[%s]: complete; uploaded %d clip(s)", ctx.job.id, len(clips))


# --------------------------------------------------------------------------- #
# Pure helpers (no DB, S3, or ffmpeg side effects)
# --------------------------------------------------------------------------- #


def srt_to_ffmpeg_timestamp(timestamp: str) -> str:
    """Convert SRT comma milliseconds to ffmpeg dot milliseconds verbatim."""

    if SRT_TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise ValueError(f"invalid SRT timestamp: {timestamp!r}")
    return timestamp.replace(",", ".", 1)


def validate_manifest(
    manifest: object | None,
) -> tuple[dict[str, object], list[ApprovedClip]]:
    """Validate an approved DB manifest and return its approved clips."""

    if manifest is None:
        raise CutError("cut: job manifest is missing")
    if not isinstance(manifest, dict):
        raise CutError("cut: job manifest must be a JSON object")

    data = cast(dict[str, object], manifest)
    if data.get("status") != "approved":
        raise CutError("cut: manifest status must be 'approved'")

    raw_clips = data.get("clips")
    if not isinstance(raw_clips, list):
        raise CutError("cut: manifest clips must be a list")

    approved: list[ApprovedClip] = []
    seen_ids: set[str] = set()
    for index, raw_clip in enumerate(raw_clips):
        if not isinstance(raw_clip, dict):
            raise CutError(f"cut: manifest clips[{index}] must be an object")
        clip = cast(dict[str, object], raw_clip)
        if clip.get("approved") is not True:
            continue

        approved_clip = _validate_approved_clip(clip, index)
        if approved_clip.clip_id in seen_ids:
            raise CutError(
                f"cut: duplicate approved clip id {approved_clip.clip_id!r}"
            )
        seen_ids.add(approved_clip.clip_id)
        approved.append(approved_clip)

    if not approved:
        raise CutError("cut: manifest has no approved clips")
    return data, approved


def build_main_clip_command(
    source: Path, output: Path, start: str, end: str
) -> list[str]:
    """Build the ffmpeg command for one manifest clip."""

    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ss",
        srt_to_ffmpeg_timestamp(start),
        "-to",
        srt_to_ffmpeg_timestamp(end),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(output),
    ]


def build_first_five_command(source: Path, output: Path) -> list[str]:
    """Build the ffmpeg command for a clip's first five seconds."""

    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-t",
        "5",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(output),
    ]


def build_last_five_command(source: Path, output: Path) -> list[str]:
    """Build the ffmpeg command for a clip's last five seconds."""

    return [
        "ffmpeg",
        "-y",
        "-sseof",
        "-5",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(output),
    ]


def _validate_approved_clip(clip: dict[str, object], index: int) -> ApprovedClip:
    clip_id = _required_string(clip, "id", index)
    if (
        clip_id in {".", ".."}
        or Path(clip_id).name != clip_id
        or "\x00" in clip_id
    ):
        raise CutError(f"cut: manifest clips[{index}].id is not a safe filename")

    start = _required_string(clip, "start", index)
    end = _required_string(clip, "end", index)
    try:
        srt_to_ffmpeg_timestamp(start)
        srt_to_ffmpeg_timestamp(end)
    except ValueError as exc:
        raise CutError(f"cut: manifest clips[{index}] has {exc}") from exc

    duration = clip.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or duration <= 0
    ):
        raise CutError(
            f"cut: manifest clips[{index}].duration_seconds must be positive"
        )

    return ApprovedClip(
        clip_id=clip_id,
        start=start,
        end=end,
        duration_seconds=duration,
    )


def _required_string(clip: dict[str, object], field: str, index: int) -> str:
    value = clip.get(field)
    if not isinstance(value, str) or not value:
        raise CutError(f"cut: manifest clips[{index}].{field} must be a string")
    return value


def _run_ffmpeg(command: list[str]) -> None:
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        stderr_tail = process.stderr[-FFMPEG_ERROR_TAIL:]
        raise RuntimeError(
            f"ffmpeg exited {process.returncode}: {stderr_tail}"
        )
