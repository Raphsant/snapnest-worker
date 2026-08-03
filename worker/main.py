"""Entrypoint: the single-threaded SQS poll loop.

Run with ``python -m worker.main``.

Loop semantics:
  * Long-poll SQS, one message at a time.
  * Parse {jobId, stage?}; a missing stage defaults to ingest.
  * Select the configured route and enforce its required job status.
  * Set RUNNING, execute the route's stages in order, then finalize or pause.
  * On any failure, record FAILED + error and STILL delete the message — a
    failed job must not redeliver and retry-loop; re-running is a human decision.
  * On SIGTERM/SIGINT, stop accepting new work, finish the current job, exit.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TypeVar

import boto3
import psycopg

from worker import jobs
from worker.config import Config, ConfigError, load_config
from worker.db import connect
from worker.jobs import Job, PipelineJobStatus
from worker.queue import Heartbeat, Message, Queue
from worker.stages import ENTRY_POINTS, EntryPoint, StageContext, StageSequence
from worker.workspace import Workspace

logger = logging.getLogger(__name__)

CHECKPOINT_VISIBILITY_TIMEOUT_S = 3600

T = TypeVar("T")


@dataclass(frozen=True)
class ParsedMessage:
    """Validated routing fields from an SQS message body."""

    job_id: str
    entry_stage: str


def parse_message_body(body: Mapping[str, object]) -> ParsedMessage:
    """Parse a worker message, defaulting an absent stage to ingest."""

    job_id = body.get("jobId")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("message has no usable jobId")

    entry_stage = body["stage"] if "stage" in body else "ingest"
    if not isinstance(entry_stage, str) or entry_stage not in ENTRY_POINTS:
        raise ValueError(f"message has unknown stage {entry_stage!r}")
    return ParsedMessage(job_id=job_id, entry_stage=entry_stage)


def resolve_entry_point(entry_stage: str, job_status: str) -> EntryPoint | None:
    """Return a route only when its required DB status matches."""

    entry_point = ENTRY_POINTS.get(entry_stage)
    if (
        entry_point is None
        or job_status != entry_point.required_status.value
    ):
        return None
    return entry_point


def _needs_download(job: Job) -> bool:
    """A YouTube job has no source in S3 yet; it must start at the download stage.

    Routed off the job ROW (its ``sourceType``/``sourceFileId``), not a message
    field — the backend enqueues a plain ``{jobId}`` for YouTube jobs.
    """

    return job.source_type == "YOUTUBE" or job.source_file_id is None


def resolve_entry_stage(entry_stage: str, job: Job) -> str:
    """Redirect the default pipeline entry ("ingest") to "download" for YouTube.

    A plain ``{jobId}`` message defaults to the ``ingest`` entry stage
    (:func:`parse_message_body`). A YouTube job can't ingest until its source
    has been fetched, so it enters at ``download`` instead. File jobs, and
    explicit non-ingest entry stages (manual re-drives, gate resumptions), are
    left exactly as they were.
    """

    if entry_stage == "ingest" and _needs_download(job):
        return "download"
    return entry_stage


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


class Worker:
    """Owns the AWS clients, DB connection, and the poll loop."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._shutdown = threading.Event()

        session = boto3.session.Session(
            region_name=config.aws_region,
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
        )
        self._sqs = session.client("sqs")
        self._s3 = session.client("s3")
        self._queue = Queue(
            self._sqs,
            config.pipeline_queue_url,
            wait_time_seconds=config.wait_time_seconds,
            visibility_timeout_seconds=config.visibility_timeout_seconds,
        )
        self._conn = connect(config.database_url)

    def request_shutdown(self) -> None:
        self._shutdown.set()

    def _reconnect(self) -> None:
        """Discard the current DB connection and open a fresh one."""

        try:
            self._conn.close()
        except Exception:
            logger.debug(
                "Ignoring error while closing a stale DB connection", exc_info=True
            )
        self._conn = connect(self._config.database_url)

    def _db_operation(self, operation: Callable[[], T]) -> T:
        """Run a DB operation, reconnecting and retrying it ONCE on a dead connection.

        A connection left idle for days can be dead by the first query after the
        gap (production hit ``SSL error: unexpected eof while reading``). TCP
        keepalives prevent most of these; this is the backstop for the rest. A
        second failure is allowed to propagate — the container restart policy is
        the final backstop, so we never loop.
        """

        try:
            return operation()
        except psycopg.OperationalError:
            logger.warning(
                "DB operation failed on a stale connection; "
                "reconnecting and retrying once",
                exc_info=True,
            )
            self._reconnect()
            return operation()

    def run(self) -> None:
        logger.info("Worker started; polling %s", self._config.pipeline_queue_url)
        try:
            while not self._shutdown.is_set():
                try:
                    message = self._queue.receive()
                except Exception:
                    logger.exception("Error receiving from SQS; backing off 5s")
                    self._shutdown.wait(5)
                    continue

                if message is None:
                    continue

                self._handle_message(message)
        finally:
            logger.info("Shutting down; closing DB connection")
            self._conn.close()

    def _handle_message(self, message: Message) -> None:
        try:
            parsed = parse_message_body(message.body)
        except ValueError as exc:
            logger.error("Message %s invalid: %s; deleting", message.message_id, exc)
            self._queue.delete(message.receipt_handle)
            return

        job_id = parsed.job_id
        # First DB touch per message. A connection idle since the last message
        # may be dead; reconnect-and-retry once so we don't crash and wait out
        # the SQS visibility timeout.
        job = self._db_operation(lambda: jobs.load_job(self._conn, job_id))
        if job is None:
            # A missing job is now always a bug or a transient (e.g. the row
            # isn't committed/visible yet), never a reason to destroy the
            # message. Log loudly and leave the message; the visibility timeout
            # redelivers it.
            logger.error(
                "Job %s NOT FOUND in DB; leaving message for redelivery "
                "(NOT deleting)",
                job_id,
            )
            return

        entry_stage = resolve_entry_stage(parsed.entry_stage, job)
        entry_point = resolve_entry_point(entry_stage, job.status)
        if entry_point is None:
            required = ENTRY_POINTS[entry_stage].required_status.value
            logger.info(
                "Job %s is %s, not %s required for entry stage %s; "
                "skipping and deleting (idempotency)",
                job_id,
                job.status,
                required,
                entry_stage,
            )
            self._queue.delete(message.receipt_handle)
            return

        self._process(job, message, entry_stage, entry_point.stages)

    def _process(
        self,
        job: Job,
        message: Message,
        entry_stage: str,
        stages: StageSequence,
    ) -> None:
        job_id = job.id
        logger.info("Processing job %s from entry stage %s", job_id, entry_stage)
        try:
            jobs.mark_running(self._conn, job_id)
            with Heartbeat(
                self._queue,
                message.receipt_handle,
                interval_seconds=self._config.heartbeat_interval_seconds,
                extension_seconds=self._config.heartbeat_extension_seconds,
            ):
                with Workspace(
                    job_id,
                    Path(self._config.job_workspace_root),
                    self._s3,
                    self._config.s3_bucket,
                ) as workspace:
                    self._run_stages(
                        job,
                        workspace,
                        stages,
                        checkpoint_heartbeat=lambda: self._checkpoint_heartbeat(
                            message.receipt_handle
                        ),
                    )
            self._finalize(job_id, stages[-1][0])
        except Exception as exc:
            logger.exception("Job %s FAILED", job_id)
            self._record_failure(job_id, exc)
        finally:
            # Success or failure, the message is done. Failures do not redeliver.
            self._queue.delete(message.receipt_handle)

    def _finalize(self, job_id: str, final_stage: str) -> None:
        """Mark COMPLETED unless a stage already moved the job to another status.

        A stage may pause the job at a non-terminal gate (e.g. build sets
        AWAITING_MANIFEST_APPROVAL). In that case we respect the status the
        stage set and leave the job there — the message is still deleted so it
        doesn't redeliver.
        """

        status = jobs.current_status(self._conn, job_id)
        if status == PipelineJobStatus.RUNNING.value:
            jobs.mark_completed(self._conn, job_id, current_stage=final_stage)
            logger.info("Job %s COMPLETED", job_id)
        else:
            logger.info(
                "Job %s finished stages in status %s; leaving as-is", job_id, status
            )

    def _run_stages(
        self,
        job: Job,
        workspace: Workspace,
        stages: StageSequence,
        *,
        checkpoint_heartbeat: Callable[[], None],
    ) -> None:
        context = StageContext(
            job=job,
            workspace=workspace,
            conn=self._conn,
            config=self._config,
            checkpoint_heartbeat=checkpoint_heartbeat,
        )
        logger.info("Job %s: running %d stage(s)", job.id, len(stages))
        for name, stage in stages:
            logger.info("Job %s -> stage %s", job.id, name)
            jobs.set_current_stage(self._conn, job.id, name)
            stage(context)

    def _checkpoint_heartbeat(self, receipt_handle: str) -> None:
        try:
            self._queue.change_visibility(
                receipt_handle, CHECKPOINT_VISIBILITY_TIMEOUT_S
            )
        except Exception:
            logger.warning(
                "Checkpoint heartbeat failed to reset SQS visibility",
                exc_info=True,
            )

    def _record_failure(self, job_id: str, exc: Exception) -> None:
        detail = f"{type(exc).__name__}: {exc}"
        try:
            jobs.mark_failed(self._conn, job_id, detail)
        except Exception:
            logger.exception("Could not record FAILED status for job %s", job_id)


def _install_signal_handlers(worker: Worker) -> None:
    def handler(signum: int, _frame: FrameType | None) -> None:
        logger.info(
            "Received %s; finishing current work then exiting",
            signal.Signals(signum).name,
        )
        worker.request_shutdown()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def main() -> int:
    _configure_logging()
    try:
        config = load_config()
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    worker = Worker(config)
    _install_signal_handlers(worker)
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
