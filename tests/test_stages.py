from __future__ import annotations

import dataclasses

from worker.jobs import PipelineJobStatus
from worker.stages import ENTRY_POINTS, STAGES, StageContext


def test_stages_registered_in_order() -> None:
    assert [name for name, _ in STAGES] == ["ingest", "curate", "build"]


def test_cut_entry_runs_cut_then_creative() -> None:
    entry = ENTRY_POINTS["cut"]

    assert entry.required_status is PipelineJobStatus.APPROVED
    assert [name for name, _ in entry.stages] == ["cut", "creative"]


def test_generate_entry_is_not_registered() -> None:
    assert "generate" not in ENTRY_POINTS


def test_stage_context_has_expected_fields() -> None:
    field_names = {f.name for f in dataclasses.fields(StageContext)}
    assert field_names == {"job", "workspace", "conn", "config"}
