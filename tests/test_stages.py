from __future__ import annotations

import dataclasses

from worker.stages import STAGES, StageContext


def test_no_stages_registered_yet() -> None:
    assert STAGES == []


def test_stage_context_has_expected_fields() -> None:
    field_names = {f.name for f in dataclasses.fields(StageContext)}
    assert field_names == {"job", "workspace", "conn", "config"}
