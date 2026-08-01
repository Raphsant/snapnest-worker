"""Ingest stage.

Turns a job's raw source video into the artifacts later stages build on:

  * ``session_bleeped.mp4``       - source with masked words muted + beeped
  * ``transcript.corrected.json`` - Transcribe JSON with domain corrections
  * ``subtitles.corrected.srt``   - Transcribe SRT with domain corrections
  * ``bleep_ranges.json``         - list of [start, end] float pairs

Flow: download source from S3 -> run AWS Transcribe (es-US, mask filter,
subtitles) -> apply corrections -> extract bleep ranges -> ffmpeg mute+beep ->
upload everything under ``pipeline/<jobId>/`` and record the keys in the
workspace state for the next stage.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import boto3

from worker.corrections import apply_corrections

if TYPE_CHECKING:
    from worker.config import Config
    from worker.stages import StageContext

logger = logging.getLogger(__name__)

# Transcribe emits this as the content of a filtered (masked) word.
MASK_TOKEN = "***"

# ffmpeg tail length kept in a raised error (the DB column is truncated anyway,
# but this keeps the log line and exception readable).
FFMPEG_ERROR_TAIL = 4000


def run_ingest(ctx: StageContext) -> None:
    """Run the full ingest pipeline for one job. Raises on any failure."""

    cfg = ctx.config
    ws = ctx.workspace
    job = ctx.job

    # source_s3_key is already resolved by load_job's COALESCE: the MediaFile
    # key for a file job, or the job's own sourceS3Key (written by the download
    # stage) for a YouTube job. Either way ingest just reads it. It is only None
    # when neither exists — a file job with no MediaFile *and* no key, or a
    # YouTube job that reached ingest without a download. That's a backend/route
    # bug, so fail loudly rather than call S3 with a None key.
    source_key = job.source_s3_key
    if source_key is None:
        raise RuntimeError(
            f"ingest: job {job.id} has no source to ingest — sourceFileId="
            f"{job.source_file_id!r} and sourceS3Key are both NULL "
            "(backend bug, or the download stage did not run)"
        )
    ext = Path(source_key).suffix or ".mp4"
    logger.info("ingest[%s]: downloading source %s", job.id, source_key)
    local_source = ws.download(source_key, f"source{ext}")

    prefix = f"pipeline/{job.id}/"
    job_name = f"snapnest-{job.id}"
    media_uri = f"s3://{ws.bucket}/{source_key}"

    transcribe = _transcribe_client(cfg)
    logger.info(
        "ingest[%s]: starting Transcribe job %s (vocabulary=%s, filter=%s)",
        job.id,
        job_name,
        cfg.transcribe_vocabulary or "NONE",
        cfg.transcribe_filter or "NONE — unfiltered",
    )
    _start_transcription(
        transcribe,
        cfg=cfg,
        job_name=job_name,
        media_uri=media_uri,
        bucket=ws.bucket,
        output_prefix=prefix,
    )
    transcript_uri, subtitle_uri = _await_transcription(
        transcribe,
        job_name=job_name,
        poll_seconds=cfg.transcribe_poll_seconds,
        job_id=job.id,
    )

    local_json = ws.download(_s3_key_from_uri(transcript_uri, ws.bucket), "transcript.json")
    local_srt = ws.download(_s3_key_from_uri(subtitle_uri, ws.bucket), "subtitles.srt")

    # Corrections: write corrected copies, keep the originals.
    corrected = correct_transcript_json(json.loads(local_json.read_text()))
    corrected_json = ws.path("transcript.corrected.json")
    corrected_json.write_text(json.dumps(corrected, ensure_ascii=False, indent=2))

    corrected_srt = ws.path("subtitles.corrected.srt")
    corrected_srt.write_text(correct_srt(local_srt.read_text()))

    # Bleep ranges + ffmpeg mute/beep.
    items = corrected.get("results", {}).get("items", [])
    ranges = extract_bleep_ranges(
        items, pad=cfg.bleep_pad_seconds, merge_gap=cfg.bleep_merge_gap_seconds
    )
    logger.info("ingest[%s]: %d bleep range(s)", job.id, len(ranges))

    bleeped = ws.path("session_bleeped.mp4")
    _run_ffmpeg(
        build_ffmpeg_command(
            local_source,
            bleeped,
            ranges,
            beep_hz=cfg.bleep_hz,
            beep_volume=cfg.bleep_volume,
        )
    )

    ranges_path = ws.path("bleep_ranges.json")
    ranges_path.write_text(json.dumps(ranges))

    # Upload artifacts and record their keys for later stages.
    keys = {
        "source_key": source_key,
        "bleeped_video_key": f"{prefix}session_bleeped.mp4",
        "corrected_transcript_key": f"{prefix}transcript.corrected.json",
        "corrected_srt_key": f"{prefix}subtitles.corrected.srt",
        "bleep_ranges_key": f"{prefix}bleep_ranges.json",
    }
    ws.upload(bleeped, keys["bleeped_video_key"])
    ws.upload(corrected_json, keys["corrected_transcript_key"])
    ws.upload(corrected_srt, keys["corrected_srt_key"])
    ws.upload(ranges_path, keys["bleep_ranges_key"])

    state = ws.read_state()
    state["ingest"] = keys
    ws.write_state(state)
    logger.info("ingest[%s]: complete", job.id)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested; no AWS/ffmpeg side effects)
# --------------------------------------------------------------------------- #


def _is_masked(item: dict[str, Any]) -> bool:
    alternatives = item.get("alternatives") or [{}]
    return bool(alternatives[0].get("content") == MASK_TOKEN)


def merge_ranges(ranges: list[list[float]], merge_gap: float) -> list[list[float]]:
    """Merge overlapping/near-adjacent [start, end] pairs (input need not be sorted)."""

    merged: list[list[float]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def extract_bleep_ranges(
    items: list[dict[str, Any]], *, pad: float, merge_gap: float
) -> list[list[float]]:
    """Build merged bleep ranges from masked ("***") Transcribe items."""

    ranges = [
        [max(0.0, float(item["start_time"]) - pad), float(item["end_time"]) + pad]
        for item in items
        if "start_time" in item and _is_masked(item)
    ]
    return merge_ranges(ranges, merge_gap)


def build_ffmpeg_command(
    source: Path,
    output: Path,
    ranges: list[list[float]],
    *,
    beep_hz: int,
    beep_volume: float,
) -> list[str]:
    """Build the ffmpeg command for the mute+beep pass.

    With no ranges (a clean session) an empty filter chain would be invalid, so
    we fall back to a remux copy of the source.
    """

    if not ranges:
        return ["ffmpeg", "-y", "-i", str(source), "-c", "copy", str(output)]

    mute = ",".join(
        f"volume=enable='between(t,{a:.3f},{b:.3f})':volume=0" for a, b in ranges
    )
    parts = [f"[0:a]{mute}[muted]"]
    for idx, (a, b) in enumerate(ranges):
        ms = int(a * 1000)
        parts.append(
            f"sine=frequency={beep_hz}:duration={b - a:.3f},"
            f"adelay={ms}|{ms},volume={beep_volume}[b{idx}]"
        )
    labels = "".join(f"[b{i}]" for i in range(len(ranges)))
    parts.append(
        f"[muted]{labels}amix=inputs={len(ranges) + 1}:duration=first:normalize=0[out]"
    )
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        ";".join(parts),
        "-map",
        "0:v",
        "-map",
        "[out]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output),
    ]


def correct_srt(srt_text: str) -> str:
    """Apply corrections to SRT subtitle text only (never indexes or timestamps)."""

    corrected: list[str] = []
    for line in srt_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or "-->" in line:
            corrected.append(line)
        else:
            corrected.append(apply_corrections(line))
    return "\n".join(corrected)


def correct_transcript_json(data: dict[str, Any]) -> dict[str, Any]:
    """Apply corrections in place to a Transcribe JSON doc's items + transcript."""

    results = data.get("results", {})
    for item in results.get("items", []):
        for alt in item.get("alternatives", []):
            content = alt.get("content")
            if isinstance(content, str):
                alt["content"] = apply_corrections(content)
    for transcript in results.get("transcripts", []):
        text = transcript.get("transcript")
        if isinstance(text, str):
            transcript["transcript"] = apply_corrections(text)
    return data


def _s3_key_from_uri(uri: str, bucket: str) -> str:
    """Extract the S3 object key from a Transcribe output URI.

    Handles ``s3://bucket/key``, path-style ``https://s3.../bucket/key`` and
    virtual-hosted ``https://bucket.s3.../key`` forms.
    """

    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return parsed.path.lstrip("/")
    path = parsed.path.lstrip("/")
    prefix = f"{bucket}/"
    return path[len(prefix):] if path.startswith(prefix) else path


# --------------------------------------------------------------------------- #
# AWS Transcribe (thin, side-effecting wrappers)
# --------------------------------------------------------------------------- #

# The Transcribe client is intentionally typed as Any: boto3-stubs isn't
# installed for this service, and the alternative (casting every str to the
# generated Literal/TypedDict types) buys little for a couple of calls.


def _transcribe_client(cfg: Config) -> Any:
    session = boto3.session.Session(
        region_name=cfg.aws_region,
        aws_access_key_id=cfg.aws_access_key_id,
        aws_secret_access_key=cfg.aws_secret_access_key,
    )
    return session.client("transcribe")


def _start_transcription(
    client: Any,
    *,
    cfg: Config,
    job_name: str,
    media_uri: str,
    bucket: str,
    output_prefix: str,
) -> None:
    settings: dict[str, Any] = {}
    if cfg.transcribe_vocabulary:
        settings["VocabularyName"] = cfg.transcribe_vocabulary
    if cfg.transcribe_filter:
        settings["VocabularyFilterName"] = cfg.transcribe_filter
        settings["VocabularyFilterMethod"] = "mask"

    kwargs: dict[str, Any] = {
        "TranscriptionJobName": job_name,
        "LanguageCode": cfg.transcribe_language,
        "Media": {"MediaFileUri": media_uri},
        "OutputBucketName": bucket,
        "OutputKey": output_prefix,
        "Subtitles": {"Formats": ["srt"]},
    }
    if settings:
        kwargs["Settings"] = settings
    client.start_transcription_job(**kwargs)


def _await_transcription(
    client: Any, *, job_name: str, poll_seconds: int, job_id: str
) -> tuple[str, str]:
    """Poll until the job finishes; return (transcript_uri, subtitle_uri).

    Raises RuntimeError if Transcribe reports FAILED (with the failure reason)
    or completes without a subtitle file.
    """

    while True:
        response = client.get_transcription_job(TranscriptionJobName=job_name)
        transcription = response["TranscriptionJob"]
        status = str(transcription["TranscriptionJobStatus"])
        if status == "COMPLETED":
            transcript_uri = str(transcription["Transcript"]["TranscriptFileUri"])
            subtitle_uris = transcription.get("Subtitles", {}).get(
                "SubtitleFileUris", []
            )
            if not subtitle_uris:
                raise RuntimeError(
                    f"Transcribe job {job_name} completed without a subtitle file"
                )
            return transcript_uri, str(subtitle_uris[0])
        if status == "FAILED":
            reason = str(transcription.get("FailureReason", "unknown"))
            raise RuntimeError(f"Transcribe job {job_name} FAILED: {reason}")
        logger.info(
            "ingest[%s]: transcribe status %s; waiting %ss",
            job_id,
            status,
            poll_seconds,
        )
        time.sleep(poll_seconds)


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr[-FFMPEG_ERROR_TAIL:]
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {tail}")
