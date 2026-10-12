"""Canonical S3 keys for pipeline artifacts shared across stages."""

from __future__ import annotations


def main_clip_key(job_id: str, clip_id: str) -> str:
    """Return the S3 key for a rendered main clip."""

    return f"pipeline/{job_id}/clips/{clip_id}.mp4"


def clip_caption_key(job_id: str, clip_id: str) -> str:
    """Return the S3 key for a clip's generated caption slice (written by the worker)."""

    return f"pipeline/{job_id}/clips/captions/{clip_id}.srt"


def edited_clip_caption_key(job_id: str, clip_id: str) -> str:
    """Return the S3 key for a clip's text-only caption edit (written by the dashboard)."""

    return f"pipeline/{job_id}/clips/captions/{clip_id}.edited.srt"
