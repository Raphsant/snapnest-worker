"""PipelineJob lifecycle: load a row and update status / stage / error.

This repo NEVER migrates the schema — the ``"PipelineJob"`` table is owned by
the NestJS backend's Prisma migrations. Column names are Prisma camelCase and
must be double-quoted in SQL. Every write bumps ``"updatedAt"`` to now().
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from psycopg import Connection
from psycopg.rows import DictRow

from worker import db

# PipelineJob.error is TEXT; keep it bounded so a giant stack trace or ffmpeg
# dump can't bloat the row.
ERROR_MAX_LEN = 2000

_COLUMNS = (
    '"id", "sourceFileId", "agencyId", "requestedById", '
    '"status", "currentStage", "error"'
)


class PipelineJobStatus(StrEnum):
    """Mirror of the backend's Prisma ``PipelineJobStatus`` enum."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    AWAITING_MANIFEST_APPROVAL = "AWAITING_MANIFEST_APPROVAL"
    APPROVED = "APPROVED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Job:
    """A single PipelineJob row (the columns the worker cares about)."""

    id: str
    source_file_id: str
    agency_id: str
    requested_by_id: str
    status: str
    current_stage: str | None
    error: str | None

    @classmethod
    def from_row(cls, row: DictRow) -> Job:
        return cls(
            id=row["id"],
            source_file_id=row["sourceFileId"],
            agency_id=row["agencyId"],
            requested_by_id=row["requestedById"],
            status=row["status"],
            current_stage=row["currentStage"],
            error=row["error"],
        )


def load_job(conn: Connection[DictRow], job_id: str) -> Job | None:
    """Fetch a job by id, or None if it doesn't exist."""

    row = db.fetch_one(
        conn,
        f'SELECT {_COLUMNS} FROM "PipelineJob" WHERE "id" = %s',
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


def mark_completed(conn: Connection[DictRow], job_id: str) -> None:
    """Mark the job COMPLETED (terminal for now — no manifest gate stage yet)."""

    db.execute(
        conn,
        'UPDATE "PipelineJob" '
        'SET "status" = %s, "currentStage" = NULL, "updatedAt" = now() '
        'WHERE "id" = %s',
        (PipelineJobStatus.COMPLETED.value, job_id),
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
