from __future__ import annotations

import logging
from typing import Any, cast
from unittest.mock import Mock

import pytest

from worker.jobs import Job
from worker.main import CHECKPOINT_VISIBILITY_TIMEOUT_S, Worker
from worker.queue import Message


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
