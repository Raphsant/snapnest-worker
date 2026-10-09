"""Transcribe stage — a standalone, transcribe-only route.

Used for agency intake answers: one short source video per job, transcribed to
SRT + JSON and nothing else. No corrections table, no bleep pass, no curation.

Flow: download source from S3 -> remux to mp4 if Transcribe can't read the
container -> run AWS Transcribe (per-job language/vocabulary/filter from
``manifest["transcribe"]``) -> server-side copy the SRT and JSON to fixed keys
the backend presigns:

  * ``pipeline/<jobId>/transcript.srt``
  * ``pipeline/<jobId>/transcript.json``
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from worker.stages.ingest import (
    _await_transcription,
    _s3_key_from_uri,
    _transcribe_client,
)

if TYPE_CHECKING:
    from worker.jobs import Job
    from worker.stages import StageContext

logger = logging.getLogger(__name__)

# Containers AWS Transcribe accepts as batch input. Anything else is remuxed.
TRANSCRIBE_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".mp4", ".wav", ".flac", ".ogg", ".amr", ".webm", ".m4a"}
)

FFMPEG_ERROR_TAIL = 4000


def run_transcribe(ctx: StageContext) -> None:
    """Transcribe the job's single source file to SRT + JSON. Raises on failure."""

    cfg = ctx.config
    ws = ctx.workspace
    job = ctx.job

    source_key = job.source_s3_key
    if source_key is None:
        raise RuntimeError(
            f"transcribe: job {job.id} has no source — sourceFileId="
            f"{job.source_file_id!r} and sourceS3Key are both NULL"
        )

    prefix = f"pipeline/{job.id}/"
    media_key = source_key
    if needs_remux(source_key):
        ext = Path(source_key).suffix.lower()
        logger.info(
            "transcribe[%s]: %s is not a Transcribe container; remuxing to mp4",
            job.id,
            ext or "(no extension)",
        )
        local_source = ws.download(source_key, f"source{ext or '.bin'}")
        remuxed = ws.path("source.mp4")
        _run_ffmpeg(build_remux_command(local_source, remuxed))
        media_key = f"{prefix}source/source.mp4"
        ws.upload(remuxed, media_key)

    options = transcribe_options(job, default_language=cfg.transcribe_language)
    job_name = f"snapnest-tr-{job.id}-{int(time.time())}"

    transcribe = _transcribe_client(cfg)
    logger.info(
        "transcribe[%s]: starting Transcribe job %s (language=%s, vocabulary=%s, "
        "filter=%s)",
        job.id,
        job_name,
        options["language"],
        options["vocabulary"] or "NONE",
        options["filter"] or "NONE",
    )
    transcribe.start_transcription_job(
        **build_start_kwargs(
            job_name=job_name,
            media_uri=f"s3://{ws.bucket}/{media_key}",
            bucket=ws.bucket,
            output_key=f"{prefix}raw/",
            language=options["language"],
            vocabulary=options["vocabulary"],
            vocabulary_filter=options["filter"],
        )
    )
    transcript_uri, subtitle_uri = _await_transcription(
        transcribe,
        job_name=job_name,
        poll_seconds=cfg.transcribe_poll_seconds,
        job_id=job.id,
    )

    keys = {
        "source_key": source_key,
        "media_key": media_key,
        "transcribe_job_name": job_name,
        "srt_key": f"{prefix}transcript.srt",
        "json_key": f"{prefix}transcript.json",
    }
    _copy(ws.s3, ws.bucket, _s3_key_from_uri(subtitle_uri, ws.bucket), keys["srt_key"])
    _copy(
        ws.s3, ws.bucket, _s3_key_from_uri(transcript_uri, ws.bucket), keys["json_key"]
    )

    state = ws.read_state()
    state["transcribe"] = keys
    ws.write_state(state)
    logger.info(
        "transcribe[%s]: complete (srt=%s, json=%s)",
        job.id,
        keys["srt_key"],
        keys["json_key"],
    )


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested; no AWS/ffmpeg side effects)
# --------------------------------------------------------------------------- #


def needs_remux(s3_key: str) -> bool:
    """True when the key's extension is not a container Transcribe accepts."""

    return Path(s3_key).suffix.lower() not in TRANSCRIBE_EXTENSIONS


def build_remux_command(source: Path, output: Path) -> list[str]:
    """Stream-copy remux into mp4 (no re-encode)."""

    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _option(raw: Mapping[str, Any], name: str) -> str:
    value = raw.get(name)
    return value.strip() if isinstance(value, str) else ""


def transcribe_options(job: Job, *, default_language: str) -> dict[str, str]:
    """Read ``manifest["transcribe"]``; missing/blank values fall back.

    Language defaults to the configured TRANSCRIBE_LANGUAGE; vocabulary and
    filter default to "" (omitted from the request). The global STC vocabulary
    and profanity filter are deliberately NOT inherited.
    """

    manifest = job.manifest if isinstance(job.manifest, Mapping) else {}
    raw = manifest.get("transcribe")
    if not isinstance(raw, Mapping):
        raw = {}
    return {
        "language": _option(raw, "language") or default_language,
        "vocabulary": _option(raw, "vocabulary"),
        "filter": _option(raw, "filter"),
    }


def build_start_kwargs(
    *,
    job_name: str,
    media_uri: str,
    bucket: str,
    output_key: str,
    language: str,
    vocabulary: str,
    vocabulary_filter: str,
) -> dict[str, Any]:
    """Build StartTranscriptionJob kwargs; optional settings only when set."""

    settings: dict[str, Any] = {}
    if vocabulary:
        settings["VocabularyName"] = vocabulary
    if vocabulary_filter:
        settings["VocabularyFilterName"] = vocabulary_filter
        settings["VocabularyFilterMethod"] = "mask"

    kwargs: dict[str, Any] = {
        "TranscriptionJobName": job_name,
        "LanguageCode": language,
        "Media": {"MediaFileUri": media_uri},
        "OutputBucketName": bucket,
        "OutputKey": output_key,
        "Subtitles": {"Formats": ["srt"]},
    }
    if settings:
        kwargs["Settings"] = settings
    return kwargs


# --------------------------------------------------------------------------- #
# Side-effecting wrappers
# --------------------------------------------------------------------------- #


def _copy(s3: Any, bucket: str, src_key: str, dest_key: str) -> None:
    s3.copy_object(
        Bucket=bucket, Key=dest_key, CopySource={"Bucket": bucket, "Key": src_key}
    )


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr[-FFMPEG_ERROR_TAIL:]
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {tail}")
