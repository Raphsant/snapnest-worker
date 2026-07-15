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

    ``source_s3_key`` is joined in from the related ``MediaFile`` row so stages
    know which object to download without a second query.
    """

    id: str
    source_file_id: str
    source_s3_key: str
    agency_id: str
    requested_by_id: str
    status: str
    current_stage: str | None
    error: str | None
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
            manifest=row.get("manifest"),
        )


def load_job(conn: Connection[DictRow], job_id: str) -> Job | None:
    """Fetch a job by id (with its source file's S3 key), or None if missing."""

    row = db.fetch_one(
        conn,
        'SELECT j."id", j."sourceFileId", j."agencyId", j."requestedById", '
        'j."status", j."currentStage", j."error", j."manifest", '
        'f."s3Key" AS "sourceS3Key" '
        'FROM "PipelineJob" j '
        'JOIN "MediaFile" f ON f."id" = j."sourceFileId" '
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


def current_status(conn: Connection[DictRow], job_id: str) -> str | None:
    """Return the job's current status string, or None if the row is gone."""

    row = db.fetch_one(
        conn, 'SELECT "status" FROM "PipelineJob" WHERE "id" = %s', (job_id,)
    )
    return str(row["status"]) if row is not None else None


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
