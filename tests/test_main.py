from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import Mock

import pytest

from worker.jobs import Job
from worker.main import (
    CHECKPOINT_VISIBILITY_TIMEOUT_S,
    Worker,
    _needs_download,
    resolve_entry_stage,
)
from worker.queue import Message
from worker.stages import ENTRY_POINTS


class FakeQueue:
    def __init__(self, *, fail_visibility: bool = False) -> None:
        self.fail_visibility = fail_visibility
        self.deleted: list[str] = []
        self.visibility_changes: list[tuple[str, int]] = []

    def delete(self, receipt_handle: str) -> None:
        self.deleted.append(receipt_handle)

    def change_visibility(self, receipt_handle: str, timeout_seconds: int) -> None:
        if self.fail_visibility:
            raise RuntimeError("SQS unavailable")
        self.visibility_changes.append((receipt_handle, timeout_seconds))


def _worker(queue: FakeQueue) -> Worker:
    worker = Worker.__new__(Worker)
    untyped = cast(Any, worker)
    untyped._queue = queue
    untyped._conn = object()
    return worker


def _job(status: str) -> Job:
    return Job(
        id="job-1",
        source_file_id="file-1",
        source_s3_key="source.mp4",
        agency_id="agency-1",
        requested_by_id="user-1",
        status=status,
        current_stage=None,
        error=None,
        manifest={"status": "approved", "clips": []},
    )


def _youtube_job(status: str) -> Job:
    return Job(
        id="job-1",
        source_file_id=None,
        source_s3_key=None,
        agency_id=None,
        requested_by_id="user-1",
        status=status,
        current_stage="download",
        error=None,
        source_type="YOUTUBE",
        source_url="https://youtu.be/abc",
    )


def test_checkpoint_heartbeat_resets_visibility_to_one_hour() -> None:
    queue = FakeQueue()
    worker = _worker(queue)

    worker._checkpoint_heartbeat("receipt-1")

    assert CHECKPOINT_VISIBILITY_TIMEOUT_S == 3600
    assert queue.visibility_changes == [("receipt-1", 3600)]


def test_checkpoint_heartbeat_failure_warns_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = _worker(FakeQueue(fail_visibility=True))

    with caplog.at_level(logging.WARNING):
        worker._checkpoint_heartbeat("receipt-1")

    assert "Checkpoint heartbeat failed" in caplog.text


def test_running_generate_redelivery_is_deleted_without_job_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue()
    worker = _worker(queue)
    process = Mock()
    mark_running = Mock()
    set_current_stage = Mock()
    mark_failed = Mock()
    mark_completed = Mock()
    monkeypatch.setattr(
        "worker.main.jobs.load_job",
        lambda conn, job_id: _job("RUNNING"),
    )
    monkeypatch.setattr(worker, "_process", process)
    monkeypatch.setattr("worker.main.jobs.mark_running", mark_running)
    monkeypatch.setattr("worker.main.jobs.set_current_stage", set_current_stage)
    monkeypatch.setattr("worker.main.jobs.mark_failed", mark_failed)
    monkeypatch.setattr("worker.main.jobs.mark_completed", mark_completed)
    message = Message(
        message_id="message-1",
        receipt_handle="receipt-1",
        body={"jobId": "job-1", "stage": "generate"},
    )

    worker._handle_message(message)

    assert queue.deleted == ["receipt-1"]
    process.assert_not_called()
    mark_running.assert_not_called()
    set_current_stage.assert_not_called()
    mark_failed.assert_not_called()
    mark_completed.assert_not_called()


# --- not-found hardening --------------------------------------------------- #


def test_job_not_found_leaves_message_for_redelivery(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    queue = FakeQueue()
    worker = _worker(queue)
    process = Mock()
    monkeypatch.setattr("worker.main.jobs.load_job", lambda conn, job_id: None)
    monkeypatch.setattr(worker, "_process", process)
    message = Message(
        message_id="message-1", receipt_handle="receipt-1", body={"jobId": "job-1"}
    )

    with caplog.at_level(logging.ERROR):
        worker._handle_message(message)

    # A missing job must NOT destroy the message — it redelivers on timeout.
    assert queue.deleted == []
    process.assert_not_called()
    assert "NOT FOUND" in caplog.text


# --- row-driven download routing ------------------------------------------- #


def test_plain_message_youtube_job_enters_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue()
    worker = _worker(queue)
    process = Mock()
    monkeypatch.setattr(
        "worker.main.jobs.load_job", lambda conn, job_id: _youtube_job("QUEUED")
    )
    monkeypatch.setattr(worker, "_process", process)
    # Plain {jobId} — the backend sends no "stage" for YouTube jobs.
    message = Message(
        message_id="message-1", receipt_handle="receipt-1", body={"jobId": "job-1"}
    )

    worker._handle_message(message)

    process.assert_called_once()
    _job_arg, _msg, entry_stage, stages = process.call_args.args
    assert entry_stage == "download"
    assert stages == ENTRY_POINTS["download"].stages
    assert queue.deleted == []  # _process owns the message lifecycle


def test_plain_message_file_job_enters_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = FakeQueue()
    worker = _worker(queue)
    process = Mock()
    monkeypatch.setattr(
        "worker.main.jobs.load_job", lambda conn, job_id: _job("QUEUED")
    )
    monkeypatch.setattr(worker, "_process", process)
    message = Message(
        message_id="message-1", receipt_handle="receipt-1", body={"jobId": "job-1"}
    )

    worker._handle_message(message)

    process.assert_called_once()
    _job_arg, _msg, entry_stage, stages = process.call_args.args
    assert entry_stage == "ingest"
    assert stages == ENTRY_POINTS["ingest"].stages


def test_resolve_entry_stage_redirects_youtube_default_entry() -> None:
    assert resolve_entry_stage("ingest", _youtube_job("QUEUED")) == "download"


def test_resolve_entry_stage_leaves_file_job_default_entry() -> None:
    assert resolve_entry_stage("ingest", _job("QUEUED")) == "ingest"


def test_resolve_entry_stage_never_reroutes_explicit_stage() -> None:
    # An explicit re-drive (a gate resumption like stage:"cut") is left alone,
    # even for a YouTube job — only the default "ingest" entry is redirected.
    assert resolve_entry_stage("cut", _youtube_job("APPROVED")) == "cut"


def test_needs_download_predicate() -> None:
    assert _needs_download(_youtube_job("QUEUED")) is True
    assert _needs_download(_job("QUEUED")) is False
