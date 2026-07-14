from __future__ import annotations

from typing import Any, cast

import pytest

from worker import jobs
from worker.jobs import Job


def test_from_row_maps_prisma_camelcase() -> None:
    row: dict[str, Any] = {
        "id": "job-1",
        "sourceFileId": "file-1",
        "agencyId": "agency-1",
        "requestedById": "user-1",
        "status": "QUEUED",
        "currentStage": None,
        "error": None,
    }

    job = Job.from_row(row)

    assert job.id == "job-1"
    assert job.source_file_id == "file-1"
    assert job.agency_id == "agency-1"
    assert job.requested_by_id == "user-1"
    assert job.status == "QUEUED"
    assert job.current_stage is None
    assert job.error is None


def test_truncate_error_clamps_to_max() -> None:
    clamped = jobs.truncate_error("x" * 5000)
    assert len(clamped) == jobs.ERROR_MAX_LEN


def test_truncate_error_leaves_short_messages() -> None:
    assert jobs.truncate_error("boom") == "boom"


def test_every_write_bumps_updated_at(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_execute(conn: Any, sql: str, params: Any = None) -> int:
        calls.append(sql)
        return 1

    monkeypatch.setattr("worker.db.execute", fake_execute)
    conn = cast(Any, object())

    jobs.mark_running(conn, "job-1")
    jobs.set_current_stage(conn, "job-1", "download")
    jobs.mark_completed(conn, "job-1")
    jobs.mark_failed(conn, "job-1", "boom")

    assert len(calls) == 4
    for sql in calls:
        assert '"updatedAt" = now()' in sql
