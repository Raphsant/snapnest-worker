"""Download stage — the first stage for YouTube-sourced jobs.

A YouTube job arrives with no source in S3: ``sourceFileId`` is NULL and the
canonical URL lives in ``sourceUrl``. This stage fetches that URL with yt-dlp
(best video+audio up to 1080p, merged to a single mp4), uploads it to
``pipeline/<jobId>/source/source.mp4``, and records the key on the job row so
the *unchanged* ingest stage resolves it exactly like a file-triggered job.

Checkpoint discipline (same rule as generation): once ``sourceS3Key`` is set and
the object really exists in S3, a redelivered message must NOT re-download. The
multi-GB temp file is always removed in a ``finally`` block.

On failure this raises :class:`DownloadError`; the poll loop records the job
FAILED (with the yt-dlp stderr tail) and deletes the message — a bad URL is a
human decision to retry, not a redelivery loop.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from worker import jobs

if TYPE_CHECKING:
    from worker.stages import StageContext

logger = logging.getLogger(__name__)

# yt-dlp format selection: prefer a clean H.264/AAC mp4 up to 1080p, then fall
# back through any codecs at that cap, and finally to whatever exists so we
# never fail to produce a file. ``--merge-output-format``/``--remux-video`` (see
# below) guarantee the final container is mp4 regardless of which branch wins.
_FORMAT = (
    "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/"
    "b[height<=1080][ext=mp4]/"
    "bv*[height<=1080]+ba/"
    "b[height<=1080]/"
    "bv*+ba/b"
)

# Tail of yt-dlp output kept in a raised error (the DB column is truncated
# anyway, but this keeps the log line and exception readable).
YTDLP_ERROR_TAIL = 4000


class DownloadError(RuntimeError):
    """Raised when the source video cannot be fetched or staged to S3."""


def run_download(ctx: StageContext) -> None:
    """Fetch the job's YouTube source to S3 and record its key. Raises on failure."""

    job = ctx.job
    dest_key = f"pipeline/{job.id}/source/source.mp4"

    # Checkpoint: a completed download is durably recorded as sourceS3Key on the
    # row. If it's set AND the object is actually present, skip and advance — a
    # redelivered message must not re-download a multi-GB file.
    if job.source_s3_key and _s3_object_exists(ctx, job.source_s3_key):
        logger.info(
            "download[%s]: source already staged (%s); skipping download",
            job.id,
            job.source_s3_key,
        )
        return

    url = job.source_url
    if not url:
        raise DownloadError(
            f"download: job {job.id} has no sourceUrl to download "
            "(backend should set it for YOUTUBE jobs)"
        )

    work_dir = ctx.workspace.path("download")
    local_source = work_dir / "source.mp4"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("download[%s]: fetching %s", job.id, url)
        _run_yt_dlp(url, local_source)

        size_mb = local_source.stat().st_size / (1024 * 1024)
        logger.info(
            "download[%s]: fetched %.1f MB; uploading to %s",
            job.id,
            size_mb,
            dest_key,
        )
        ctx.workspace.upload(local_source, dest_key)
        jobs.set_source_s3_key(ctx.conn, job.id, dest_key)
        # Make the just-set key visible to the ingest stage in this same run.
        # Job is frozen; StageContext holds the live reference we can rebind.
        ctx.job = replace(job, source_s3_key=dest_key)
        ctx.checkpoint_heartbeat()
        logger.info("download[%s]: complete; sourceS3Key=%s", job.id, dest_key)
    finally:
        # Multi-GB temp: always remove the whole download dir (merged output
        # plus any yt-dlp fragments), on success or failure.
        shutil.rmtree(work_dir, ignore_errors=True)


def _s3_object_exists(ctx: StageContext, s3_key: str) -> bool:
    """True if the key resolves to an object in the job's bucket.

    Any error (404, and conservatively anything else) reads as "not present";
    the worst case is a safe re-download, never a false skip.
    """

    try:
        ctx.workspace.s3.head_object(Bucket=ctx.workspace.bucket, Key=s3_key)
    except Exception:
        logger.debug("download: head_object miss for %s", s3_key, exc_info=True)
        return False
    return True


def _run_yt_dlp(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` as a merged mp4. Raises DownloadError on failure.

    Invoked as ``python -m yt_dlp`` (not a bare ``yt-dlp`` on PATH) so it always
    resolves to the interpreter's installed copy — the same in the local venv
    and the Docker system Python.
    """

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        _FORMAT,
        "--merge-output-format",
        "mp4",
        "--remux-video",
        "mp4",
        "--no-playlist",
        "--no-progress",
        "-o",
        str(dest),
        "--",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-YTDLP_ERROR_TAIL:]
        raise DownloadError(f"yt-dlp exited {proc.returncode} for {url}: {tail}")
    if not dest.exists():
        tail = (proc.stderr or proc.stdout or "")[-YTDLP_ERROR_TAIL:]
        raise DownloadError(
            f"yt-dlp reported success but produced no file at {dest} "
            f"for {url}: {tail}"
        )
