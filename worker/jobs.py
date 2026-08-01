"""PipelineJob lifecycle: load a row and update status / stage / error.

This repo NEVER migrates the schema — the ``"PipelineJob"`` table is owned by
the NestJS backend's Prisma migrations. Column names are Prisma camelCase and
must be double-quoted in SQL. Every write bumps ``"updatedAt"`` to now().
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from psycopg import Connection
from psycopg.rows import DictRow
from psycopg.types.json import Json

from worker import db

# PipelineJob.error is TEXT; keep it bounded so a giant stack trace or ffmpeg
# dump can't bloat the row.
ERROR_MAX_LEN = 2000


class PipelineJobStatus(StrEnum):
    """Mirror of the backend's Prisma ``PipelineJobStatus`` enum."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    AWAITING_MANIFEST_APPROVAL = "AWAITING_MANIFEST_APPROVAL"
    APPROVED = "APPROVED"
    AWAITING_CREATIVE_APPROVAL = "AWAITING_CREATIVE_APPROVAL"
    CREATIVE_APPROVED = "CREATIVE_APPROVED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Job:
    """A single PipelineJob row (the columns the worker cares about).

    ``source_s3_key`` is resolved with ``COALESCE(j."sourceS3Key", f."s3Key")``:
    a file-triggered job gets its key from the related ``MediaFile`` row, while
    a YouTube job carries its own ``sourceS3Key`` (written by the download stage)
    and has no ``MediaFile``. It is therefore ``None`` for a YouTube job that has
    not been downloaded yet.

    ``source_file_id``/``agency_id`` are ``None`` for YouTube jobs; ``source_type``
    and ``source_url`` distinguish and drive them (``source_type == "YOUTUBE"``,
    ``source_url`` = the canonical URL to fetch).
    """

    id: str
    source_file_id: str | None
    source_s3_key: str | None
    agency_id: str | None
    requested_by_id: str
    status: str
    current_stage: str | None
    error: str | None
    source_type: str | None = None
    source_url: str | None = None
    manifest: object | None = None

    @classmethod
    def from_row(cls, row: DictRow) -> Job:
        return cls(
            id=row["id"],
            source_file_id=row["sourceFileId"],
            source_s3_key=row["sourceS3Key"],
            agency_id=row["agencyId"],
            requested_by_id=row["requestedById"],
            status=row["status"],
            current_stage=row["currentStage"],
            error=row["error"],
            source_type=row.get("sourceType"),
            source_url=row.get("sourceUrl"),
            manifest=row.get("manifest"),
        )


def load_job(conn: Connection[DictRow], job_id: str) -> Job | None:
    """Fetch a job by id, or None if missing.

    A LEFT JOIN keeps YouTube jobs (``sourceFileId IS NULL``, no ``MediaFile``)
    visible; without it the inner join dropped them and the worker saw
    "not found". ``sourceS3Key`` coalesces the job's own column (YouTube jobs,
    set by the download stage) over the joined ``MediaFile.s3Key`` (file jobs).
    """

    row = db.fetch_one(
        conn,
        'SELECT j."id", j."sourceFileId", j."agencyId", j."requestedById", '
        'j."status", j."currentStage", j."error", j."manifest", '
        'j."sourceType", j."sourceUrl", '
        'COALESCE(j."sourceS3Key", f."s3Key") AS "sourceS3Key" '
        'FROM "PipelineJob" j '
        'LEFT JOIN "MediaFile" f ON f."id" = j."sourceFileId" '
        'WHERE j."id" = %s',
        (job_id,),
    )
    return Job.from_row(row) if row is not None else None


def mark_running(conn: Connection[DictRow], job_id: str) -> None:
    db.execute(
        conn,
        'UPDATE "PipelineJob" SET "status" = %s, "updatedAt" = now() WHERE "id" = %s',
        (PipelineJobStatus.RUNNING.value, job_id),
    )


def set_current_stage(conn: Connection[DictRow], job_id: str, stage: str) -> None:
    db.execute(
        conn,
        'UPDATE "PipelineJob" SET "currentStage" = %s, "updatedAt" = now() WHERE "id" = %s',
        (stage, job_id),
    )


def set_source_s3_key(conn: Connection[DictRow], job_id: str, s3_key: str) -> None:
    """Record the downloaded source's S3 key on the job row (download stage).

    This is the durable checkpoint for a YouTube download: once set (and the
    object exists in S3) a redelivered message must not re-download.
    """

    db.execute(
        conn,
        'UPDATE "PipelineJob" SET "sourceS3Key" = %s, "updatedAt" = now() '
        'WHERE "id" = %s',
        (s3_key, job_id),
    )


def current_status(conn: Connection[DictRow], job_id: str) -> str | None:
    """Return the job's current status string, or None if the row is gone."""

    row = db.fetch_one(
        conn, 'SELECT "status" FROM "PipelineJob" WHERE "id" = %s', (job_id,)
    )
    return str(row["status"]) if row is not None else None


def load_manifest(conn: Connection[DictRow], job_id: str) -> object | None:
    """Reload the authoritative manifest from the job row."""

    row = db.fetch_one(
        conn, 'SELECT "manifest" FROM "PipelineJob" WHERE "id" = %s', (job_id,)
    )
    return row["manifest"] if row is not None else None


def save_manifest_checkpoint(
    conn: Connection[DictRow], job_id: str, manifest: Mapping[str, Any]
) -> None:
    """Persist generated-asset progress without changing job lifecycle state."""

    db.execute(
        conn,
        'UPDATE "PipelineJob" '
        'SET "manifest" = %s, "updatedAt" = now() '
        'WHERE "id" = %s',
        (Json(manifest), job_id),
    )


def save_manifest_awaiting_approval(
    conn: Connection[DictRow], job_id: str, manifest: Mapping[str, Any]
) -> None:
    """Persist the manifest (jsonb) and pause the job at the human gate.

    Sets status AWAITING_MANIFEST_APPROVAL and clears currentStage — the build
    stage is the last one for now, so the worker leaves the job in this
    non-terminal state (it is NOT marked COMPLETED).
    """

    db.execute(
        conn,
        'UPDATE "PipelineJob" '
        'SET "manifest" = %s, "status" = %s, "currentStage" = NULL, '
        '"updatedAt" = now() '
        'WHERE "id" = %s',
        (Json(manifest), PipelineJobStatus.AWAITING_MANIFEST_APPROVAL.value, job_id),
    )


def save_manifest_awaiting_creative_approval(
    conn: Connection[DictRow], job_id: str, manifest: Mapping[str, Any]
) -> None:
    """Persist creative fields and pause at the creative approval gate."""

    db.execute(
        conn,
        'UPDATE "PipelineJob" '
        'SET "manifest" = %s, "status" = %s, "currentStage" = %s, '
        '"updatedAt" = now() '
        'WHERE "id" = %s',
        (
            Json(manifest),
            PipelineJobStatus.AWAITING_CREATIVE_APPROVAL.value,
            "creative",
            job_id,
        ),
    )


def mark_completed(
    conn: Connection[DictRow], job_id: str, *, current_stage: str | None = None
) -> None:
    """Mark the job COMPLETED and set its final current-stage value."""

    db.execute(
        conn,
        'UPDATE "PipelineJob" '
        'SET "status" = %s, "currentStage" = %s, "updatedAt" = now() '
        'WHERE "id" = %s',
        (PipelineJobStatus.COMPLETED.value, current_stage, job_id),
    )


def mark_failed(conn: Connection[DictRow], job_id: str, error: str) -> None:
    """Mark the job FAILED with a (truncated) error message."""

    db.execute(
        conn,
        'UPDATE "PipelineJob" '
        'SET "status" = %s, "error" = %s, "updatedAt" = now() '
        'WHERE "id" = %s',
        (PipelineJobStatus.FAILED.value, truncate_error(error), job_id),
    )


def truncate_error(error: str) -> str:
    """Clamp an error message to the column's practical limit."""

    return error[:ERROR_MAX_LEN]
