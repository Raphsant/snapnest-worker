"""Canonical S3 keys for pipeline artifacts shared across stages."""

from __future__ import annotations


def main_clip_key(job_id: str, clip_id: str) -> str:
    """Return the S3 key for a rendered main clip."""

    return f"pipeline/{job_id}/clips/{clip_id}.mp4"
