from __future__ import annotations

from typing import Any, cast

import pytest

from worker import jobs
from worker.jobs import Job


def test_from_row_maps_prisma_camelcase() -> None:
    row: dict[str, Any] = {
        "id": "job-1",
        "sourceFileId": "file-1",
        "sourceS3Key": "agency/raw/file-1.mp4",
        "agencyId": "agency-1",
        "requestedById": "user-1",
        "status": "QUEUED",
        "currentStage": None,
        "error": None,
    }

    job = Job.from_row(row)

    assert job.id == "job-1"
    assert job.source_file_id == "file-1"
    assert job.source_s3_key == "agency/raw/file-1.mp4"
    assert job.agency_id == "agency-1"
    assert job.requested_by_id == "user-1"
    assert job.status == "QUEUED"
    assert job.current_stage is None
    assert job.error is None


def test_from_row_youtube_job_has_null_source_file() -> None:
    row: dict[str, Any] = {
        "id": "yt-1",
        "sourceFileId": None,
        "sourceS3Key": None,
        "agencyId": None,
        "requestedById": "user-1",
        "status": "QUEUED",
        "currentStage": "download",
        "error": None,
        "sourceType": "YOUTUBE",
        "sourceUrl": "https://www.youtube.com/watch?v=abc",
    }

    job = Job.from_row(row)

    assert job.source_file_id is None
    assert job.source_s3_key is None
    assert job.agency_id is None
    assert job.source_type == "YOUTUBE"
    assert job.source_url == "https://www.youtube.com/watch?v=abc"
    assert job.current_stage == "download"


def test_load_job_left_joins_and_coalesces_source_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_fetch_one(conn: Any, sql: str, params: Any = None) -> dict[str, Any]:
        seen["sql"] = sql
        seen["params"] = params
        return {
            "id": "yt-1",
            "sourceFileId": None,
            "sourceS3Key": "pipeline/yt-1/source/source.mp4",
            "agencyId": None,
            "requestedById": "user-1",
            "status": "QUEUED",
            "currentStage": "download",
            "error": None,
            "sourceType": "YOUTUBE",
            "sourceUrl": "https://youtu.be/abc",
        }

    monkeypatch.setattr("worker.db.fetch_one", fake_fetch_one)

    job = jobs.load_job(cast(Any, object()), "yt-1")

    assert job is not None
    # A YouTube job (no MediaFile) must survive the join, and its own sourceS3Key
    # must win the COALESCE.
    assert "LEFT JOIN" in seen["sql"]
    assert 'COALESCE(j."sourceS3Key", f."s3Key")' in seen["sql"]
    assert '"sourceType"' in seen["sql"] and '"sourceUrl"' in seen["sql"]
    assert seen["params"] == ("yt-1",)
    assert job.source_s3_key == "pipeline/yt-1/source/source.mp4"


def test_set_source_s3_key_writes_key_and_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    def fake_execute(conn: Any, sql: str, params: Any = None) -> int:
        calls.append((sql, params))
        return 1

    monkeypatch.setattr("worker.db.execute", fake_execute)

    jobs.set_source_s3_key(cast(Any, object()), "yt-1", "pipeline/yt-1/source/source.mp4")

    assert len(calls) == 1
    sql, params = calls[0]
    assert '"sourceS3Key" = %s' in sql
    assert '"updatedAt" = now()' in sql
    assert params == ("pipeline/yt-1/source/source.mp4", "yt-1")


def test_truncate_error_clamps_to_max() -> None:
    clamped = jobs.truncate_error("x" * 5000)
    assert len(clamped) == jobs.ERROR_MAX_LEN


def test_truncate_error_leaves_short_messages() -> None:
    assert jobs.truncate_error("boom") == "boom"


def test_load_manifest_returns_authoritative_db_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {"status": "approved", "clips": []}

    def fake_fetch_one(conn: Any, sql: str, params: Any = None) -> dict[str, Any]:
        assert '"manifest"' in sql
        assert params == ("job-1",)
        return {"manifest": manifest}

    monkeypatch.setattr("worker.db.fetch_one", fake_fetch_one)

    assert jobs.load_manifest(cast(Any, object()), "job-1") is manifest


def test_checkpoint_updates_only_manifest_and_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_execute(conn: Any, sql: str, params: Any = None) -> int:
        calls.append(sql)
        return 1

    monkeypatch.setattr("worker.db.execute", fake_execute)

    jobs.save_manifest_checkpoint(
        cast(Any, object()),
        "job-1",
        {"status": "approved", "clips": []},
    )

    assert len(calls) == 1
    assert '"manifest" = %s' in calls[0]
    assert '"updatedAt" = now()' in calls[0]
    assert '"status"' not in calls[0]
    assert '"currentStage"' not in calls[0]


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
    jobs.save_manifest_awaiting_approval(conn, "job-1", {"status": "pending_approval"})

    assert len(calls) == 5
    for sql in calls:
        assert '"updatedAt" = now()' in sql
