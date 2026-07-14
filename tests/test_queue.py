from __future__ import annotations

import json
import time
from typing import Any, cast

from worker.queue import Heartbeat, Message, Queue


class FakeSQS:
    """Minimal stand-in for the boto3 SQS client used in unit tests."""

    def __init__(self, receive_response: dict[str, Any]) -> None:
        self._receive_response = receive_response
        self.deleted: list[str] = []
        self.visibility_changes: list[tuple[str, int]] = []

    def receive_message(self, **_kwargs: Any) -> dict[str, Any]:
        return self._receive_response

    def delete_message(self, **kwargs: Any) -> dict[str, Any]:
        self.deleted.append(kwargs["ReceiptHandle"])
        return {}

    def change_message_visibility(self, **kwargs: Any) -> dict[str, Any]:
        self.visibility_changes.append(
            (kwargs["ReceiptHandle"], kwargs["VisibilityTimeout"])
        )
        return {}


def _queue(fake: FakeSQS) -> Queue:
    return Queue(
        cast(Any, fake),
        "https://sqs.example/queue",
        wait_time_seconds=0,
        visibility_timeout_seconds=30,
    )


def test_message_from_sqs_parses_json_body() -> None:
    raw: dict[str, Any] = {
        "MessageId": "m-1",
        "ReceiptHandle": "rh-1",
        "Body": json.dumps({"jobId": "job-1"}),
    }

    message = Message.from_sqs(raw)

    assert message.message_id == "m-1"
    assert message.receipt_handle == "rh-1"
    assert message.body == {"jobId": "job-1"}


def test_receive_returns_none_when_empty() -> None:
    assert _queue(FakeSQS({})).receive() is None


def test_receive_returns_parsed_message() -> None:
    fake = FakeSQS(
        {
            "Messages": [
                {
                    "MessageId": "m-1",
                    "ReceiptHandle": "rh-1",
                    "Body": json.dumps({"jobId": "job-1"}),
                }
            ]
        }
    )

    message = _queue(fake).receive()

    assert message is not None
    assert message.body["jobId"] == "job-1"


def test_delete_calls_client() -> None:
    fake = FakeSQS({})
    _queue(fake).delete("rh-1")
    assert fake.deleted == ["rh-1"]


def test_heartbeat_extends_visibility() -> None:
    fake = FakeSQS({})
    queue = _queue(fake)

    with Heartbeat(queue, "rh-1", interval_seconds=0, extension_seconds=30):
        time.sleep(0.05)

    assert ("rh-1", 30) in fake.visibility_changes
