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
from worker.stages.assemble import run_assemble  # noqa: E402
from worker.stages.build import run_build  # noqa: E402
from worker.stages.creative import run_creative  # noqa: E402
from worker.stages.cut import run_cut  # noqa: E402
from worker.stages.curate import run_curate  # noqa: E402
from worker.stages.download import run_download  # noqa: E402
from worker.stages.generate import run_generate  # noqa: E402
from worker.stages.ingest import run_ingest  # noqa: E402

# Message entry stage -> required DB status + ordered stages to execute.
ENTRY_POINTS: dict[str, EntryPoint] = {
    # YouTube jobs enter here (routed off the row by resolve_entry_stage, or via
    # an explicit stage:"download" message for a manual re-drive). After the
    # source is staged to S3, the run flows straight into the same ingest ->
    # curate -> build sequence a file job runs.
    "download": EntryPoint(
        required_status=PipelineJobStatus.QUEUED,
        stages=(
            ("download", run_download),
            ("ingest", run_ingest),
            ("curate", run_curate),
            ("build", run_build),
        ),
    ),
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
        stages=(
            ("generate", cast(Stage, run_generate)),
            ("assemble", cast(Stage, run_assemble)),
        ),
    ),
    "assemble": EntryPoint(
        required_status=PipelineJobStatus.CREATIVE_APPROVED,
        stages=(("assemble", cast(Stage, run_assemble)),),
    ),
}

# Backward-compatible name for tests and callers that mean the initial pipeline.
STAGES: list[tuple[str, Stage]] = list(ENTRY_POINTS["ingest"].stages)
