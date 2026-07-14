"""SQS access: long-poll receive, delete, and a visibility heartbeat.

The worker processes one message at a time. While a job runs, a background
:class:`Heartbeat` thread periodically calls ChangeMessageVisibility so the
message stays invisible to other consumers until we're done (or we crash and
let it time back out).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mypy_boto3_sqs.client import SQSClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Message:
    """A received SQS message with its JSON body already parsed."""

    message_id: str
    receipt_handle: str
    body: dict[str, Any]

    @classmethod
    def from_sqs(cls, raw: dict[str, Any]) -> Message:
        return cls(
            message_id=raw["MessageId"],
            receipt_handle=raw["ReceiptHandle"],
            body=json.loads(raw["Body"]),
        )


class Queue:
    """Wraps a single SQS queue URL and its client."""

    def __init__(
        self,
        client: SQSClient,
        queue_url: str,
        *,
        wait_time_seconds: int,
        visibility_timeout_seconds: int,
    ) -> None:
        self._client = client
        self._queue_url = queue_url
        self._wait_time_seconds = wait_time_seconds
        self._visibility_timeout_seconds = visibility_timeout_seconds

    def receive(self) -> Message | None:
        """Long-poll for a single message. Returns None if none arrived."""

        response = self._client.receive_message(
            QueueUrl=self._queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=self._wait_time_seconds,
            VisibilityTimeout=self._visibility_timeout_seconds,
        )
        messages = response.get("Messages", [])
        if not messages:
            return None
        return Message.from_sqs(dict(messages[0]))

    def delete(self, receipt_handle: str) -> None:
        self._client.delete_message(
            QueueUrl=self._queue_url, ReceiptHandle=receipt_handle
        )

    def change_visibility(self, receipt_handle: str, timeout_seconds: int) -> None:
        self._client.change_message_visibility(
            QueueUrl=self._queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=timeout_seconds,
        )


class Heartbeat:
    """Background thread that extends a message's visibility on an interval.

    Used as a context manager around job processing::

        with Heartbeat(queue, handle, interval_seconds=600, extension_seconds=900):
            run_the_job()
    """

    def __init__(
        self,
        queue: Queue,
        receipt_handle: str,
        *,
        interval_seconds: int,
        extension_seconds: int,
    ) -> None:
        self._queue = queue
        self._receipt_handle = receipt_handle
        self._interval_seconds = interval_seconds
        self._extension_seconds = extension_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="sqs-heartbeat", daemon=True
        )

    def _run(self) -> None:
        # Event.wait returns True when set (stop requested) — so the loop exits
        # promptly on __exit__ instead of sleeping out the full interval.
        while not self._stop.wait(self._interval_seconds):
            try:
                self._queue.change_visibility(
                    self._receipt_handle, self._extension_seconds
                )
                logger.debug(
                    "Extended visibility by %ss", self._extension_seconds
                )
            except Exception:
                logger.exception("Heartbeat failed to extend message visibility")

    def __enter__(self) -> Heartbeat:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
