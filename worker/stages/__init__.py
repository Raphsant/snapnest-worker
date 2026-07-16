"""Stage framework and entry-stage dispatch table."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from psycopg import Connection
from psycopg.rows import DictRow

from worker.config import Config
from worker.jobs import Job, PipelineJobStatus
from worker.workspace import Workspace


def _noop_checkpoint_heartbeat() -> None:
    """Default for contexts created outside the worker poll loop."""


@dataclass
class StageContext:
    """Everything a stage needs to do its work."""

    job: Job
    workspace: Workspace
    conn: Connection[DictRow]
    config: Config
    checkpoint_heartbeat: Callable[[], None] = _noop_checkpoint_heartbeat


# A stage takes the context and runs for its side effects. Raise to fail the job.
Stage = Callable[[StageContext], None]
StageSequence = tuple[tuple[str, Stage], ...]


@dataclass(frozen=True)
class EntryPoint:
    """Required job status and ordered stages for one message entry stage."""

    required_status: PipelineJobStatus
    stages: StageSequence

# Imported after StageContext/Stage are defined so the stages can reference them
# (they only import them under TYPE_CHECKING, so there's no import cycle).
from worker.stages.build import run_build  # noqa: E402
from worker.stages.creative import run_creative  # noqa: E402
from worker.stages.cut import run_cut  # noqa: E402
from worker.stages.curate import run_curate  # noqa: E402
from worker.stages.generate import run_generate  # noqa: E402
from worker.stages.ingest import run_ingest  # noqa: E402

# Message entry stage -> required DB status + ordered stages to execute.
ENTRY_POINTS: dict[str, EntryPoint] = {
    "ingest": EntryPoint(
        required_status=PipelineJobStatus.QUEUED,
        stages=(
            ("ingest", run_ingest),
            ("curate", run_curate),
            ("build", run_build),
        ),
    ),
    "cut": EntryPoint(
        required_status=PipelineJobStatus.APPROVED,
        stages=(
            ("cut", run_cut),
            ("creative", run_creative),
        ),
    ),
    "generate": EntryPoint(
        required_status=PipelineJobStatus.CREATIVE_APPROVED,
        stages=(("generate", cast(Stage, run_generate)),),
    ),
}

# Backward-compatible name for tests and callers that mean the initial pipeline.
STAGES: list[tuple[str, Stage]] = list(ENTRY_POINTS["ingest"].stages)
