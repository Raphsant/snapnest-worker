"""Stage framework.

A stage is a plain callable that receives a :class:`StageContext` and performs
one step of the pipeline (download, transcode, upload, etc.). Stages are run in
the order they appear in :data:`STAGES`; the worker writes each stage's name to
``PipelineJob."currentStage"`` before invoking it.

There are intentionally NO stages yet — they arrive as separate tasks. With an
empty list the worker loads a job, runs zero stages, and marks it done.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from psycopg import Connection
from psycopg.rows import DictRow

from worker.config import Config
from worker.jobs import Job
from worker.workspace import Workspace


@dataclass
class StageContext:
    """Everything a stage needs to do its work."""

    job: Job
    workspace: Workspace
    conn: Connection[DictRow]
    config: Config


# A stage takes the context and runs for its side effects. Raise to fail the job.
Stage = Callable[[StageContext], None]

# Ordered pipeline stages: (name, callable). Empty for now.
STAGES: list[tuple[str, Stage]] = []
