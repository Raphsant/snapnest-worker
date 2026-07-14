"""Entrypoint: the single-threaded SQS poll loop.

Run with ``python -m worker.main``.

Loop semantics:
  * Long-poll SQS, one message at a time.
  * Parse {jobId} -> load the PipelineJob row.
  * If the job isn't QUEUED, log + delete the message (idempotency guard).
  * Otherwise set RUNNING, run stages in order (updating currentStage before
    each), mark COMPLETED, and delete the message.
  * On any failure, record FAILED + error and STILL delete the message — a
    failed job must not redeliver and retry-loop; re-running is a human decision.
  * On SIGTERM/SIGINT, stop accepting new work, finish the current job, exit.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from pathlib import Path
from types import FrameType

import boto3

from worker import jobs
from worker.config import Config, ConfigError, load_config
from worker.db import connect
from worker.jobs import Job, PipelineJobStatus
from worker.queue import Heartbeat, Message, Queue
from worker.stages import STAGES, StageContext
from worker.workspace import Workspace

logger = logging.getLogger(__name__)


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
        job_id = message.body.get("jobId")
        if not isinstance(job_id, str) or not job_id:
            logger.error(
                "Message %s has no usable jobId; deleting", message.message_id
            )
            self._queue.delete(message.receipt_handle)
            return

        job = jobs.load_job(self._conn, job_id)
        if job is None:
            logger.warning("Job %s not found in DB; deleting message", job_id)
            self._queue.delete(message.receipt_handle)
            return

        if job.status != PipelineJobStatus.QUEUED.value:
            logger.info(
                "Job %s is %s, not QUEUED; skipping and deleting (idempotency)",
                job_id,
                job.status,
            )
            self._queue.delete(message.receipt_handle)
            return

        self._process(job, message)

    def _process(self, job: Job, message: Message) -> None:
        job_id = job.id
        logger.info("Processing job %s", job_id)
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
                    self._run_stages(job, workspace)
            jobs.mark_completed(self._conn, job_id)
            logger.info("Job %s COMPLETED", job_id)
        except Exception as exc:
            logger.exception("Job %s FAILED", job_id)
            self._record_failure(job_id, exc)
        finally:
            # Success or failure, the message is done. Failures do not redeliver.
            self._queue.delete(message.receipt_handle)

    def _run_stages(self, job: Job, workspace: Workspace) -> None:
        context = StageContext(
            job=job,
            workspace=workspace,
            conn=self._conn,
            config=self._config,
        )
        logger.info("Job %s: running %d stage(s)", job.id, len(STAGES))
        for name, stage in STAGES:
            logger.info("Job %s -> stage %s", job.id, name)
            jobs.set_current_stage(self._conn, job.id, name)
            stage(context)

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
