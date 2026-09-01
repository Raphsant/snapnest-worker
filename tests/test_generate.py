"""Generate-stage tests for the v2 validation-only contract.

No Higgsfield mock appears anywhere in this file — its absence is the test:
the stage must pass or fail without touching worker.higgsfield at all.
"""

from __future__ import annotations

import copy
import io
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from mypy_boto3_s3.client import S3Client
from psycopg import Connection
from psycopg.rows import DictRow

from worker.jobs import Job
from worker.stages.generate import (
    GenerateError,
    GenerateStageContext,
    run_generate,
)
from worker.workspace import Workspace

JOB_ID = "job-1"
CLIP_ID = "clip_01"
SECOND_CLIP_ID = "clip_02"
LIBRARY_PREFIX = "library/"
CATALOG_KEY = f"{LIBRARY_PREFIX}catalog.json"


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.uploads: list[str] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise RuntimeError(f"NoSuchKey: {Key}")
        return {"Body": io.BytesIO(self.objects[Key])}

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.uploads.append(key)
        self.objects[key] = Path(filename).read_bytes()


class FakeManifestDB:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = copy.deepcopy(manifest)
        self.checkpoints: list[dict[str, Any]] = []

    def save(
        self,
        conn: Connection[DictRow],
        job_id: str,
        manifest: Mapping[str, Any],
    ) -> None:
        assert job_id == JOB_ID
        self.manifest = copy.deepcopy(dict(manifest))
        self.checkpoints.append(copy.deepcopy(self.manifest))

    def load(self, conn: Connection[DictRow], job_id: str) -> object:
        assert job_id == JOB_ID
        return copy.deepcopy(self.manifest)


@dataclass
class _FakeConfig:
    library_prefix: str = LIBRARY_PREFIX


@dataclass
class FakeContext:
    job: Job
    workspace: Workspace
    conn: Connection[DictRow]
    config: _FakeConfig


@dataclass
class Harness:
    context: FakeContext
    s3: FakeS3
    db: FakeManifestDB

    def run(self) -> None:
        with self.context.workspace:
            run_generate(cast(GenerateStageContext, self.context))


def _asset(asset_id: str, asset_type: str) -> dict[str, Any]:
    return {
        "id": asset_id,
        "type": asset_type,
        "s3_key": f"library/{asset_type}s/{asset_id}.mp4",
        "duration_s": 4.0,
        "category": ["mindset"],
        "tags": ["edu"],
        "character": None,
        "description": f"{asset_id} description",
        "times_used": 0,
    }


def _catalog_json() -> bytes:
    return json.dumps(
        {
            "version": 1,
            "updated_at": "2026-08-01T00:00:00+00:00",
            "notes": "operator selection rules",
            "assets": [
                _asset("H01", "hook"),
                _asset("H02", "hook"),
                _asset("O01", "outro"),
                _asset("O02", "outro"),
            ],
        }
    ).encode("utf-8")


def _clip(
    clip_id: str = CLIP_ID, *, hook: str = "H01", outro: str = "O01"
) -> dict[str, Any]:
    return {
        "id": clip_id,
        "approved": True,
        "category": "mindset",
        "hook_asset_id": hook,
        "outro_asset_id": outro,
    }


def _manifest(clips: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "status": "approved",
        "clips": [_clip()] if clips is None else clips,
    }


def _harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest: dict[str, Any],
    *,
    with_catalog: bool = True,
) -> Harness:
    fake_s3 = FakeS3()
    if with_catalog:
        fake_s3.objects[CATALOG_KEY] = _catalog_json()
    workspace = Workspace(JOB_ID, tmp_path, cast(S3Client, fake_s3), "bucket")
    context = FakeContext(
        job=Job(
            id=JOB_ID,
            source_file_id="file-1",
            source_s3_key="source.mp4",
            agency_id="agency-1",
            requested_by_id="user-1",
            status="CREATIVE_APPROVED",
            current_stage=None,
            error=None,
            manifest=manifest,
        ),
        workspace=workspace,
        conn=cast(Connection[DictRow], object()),
        config=_FakeConfig(),
    )
    fake_db = FakeManifestDB(manifest)
    monkeypatch.setattr(
        "worker.stages.generate.jobs.save_manifest_checkpoint",
        fake_db.save,
    )
    monkeypatch.setattr(
        "worker.stages.generate.jobs.load_manifest",
        fake_db.load,
    )
    return Harness(context, fake_s3, fake_db)


def test_valid_selections_pass_log_resolved_keys_and_advance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())

    with caplog.at_level(logging.INFO):
        harness.run()

    # Success path: only the final manifest sync touches S3; no checkpoints.
    assert harness.s3.uploads == [f"pipeline/{JOB_ID}/manifest.json"]
    assert harness.db.checkpoints == []
    assert f"clip={CLIP_ID}" in caplog.text
    assert "hook_asset=H01" in caplog.text
    assert "library/hooks/H01.mp4" in caplog.text
    assert "outro_asset=O01" in caplog.text
    assert "library/outros/O01.mp4" in caplog.text


def test_rerun_on_already_valid_manifest_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())

    harness.run()
    harness.run()

    assert harness.s3.uploads == [f"pipeline/{JOB_ID}/manifest.json"] * 2
    assert harness.db.checkpoints == []


def test_missing_asset_id_field_fails_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    del manifest["clips"][0]["outro_asset_id"]
    harness = _harness(monkeypatch, tmp_path, manifest)

    with pytest.raises(GenerateError) as raised:
        harness.run()

    assert "clips[0].outro_asset_id" in str(raised.value)
    assert harness.s3.uploads == []


def test_unknown_asset_id_hard_fails_naming_the_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest([_clip(hook="H99")]))

    with pytest.raises(GenerateError) as raised:
        harness.run()

    message = str(raised.value)
    assert "library-selection lint violation" in message
    assert f"{CLIP_ID}.hook_asset_id" in message
    assert "'H99'" in message
    assert harness.s3.uploads == []


def test_duplicate_selection_across_clips_passes_as_a_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Reuse within a job is allowed policy: the stage records the repeat and
    # advances instead of blocking the run.
    manifest = _manifest(
        [
            _clip(),
            _clip(SECOND_CLIP_ID, hook="H01", outro="O02"),
        ]
    )
    harness = _harness(monkeypatch, tmp_path, manifest)

    with caplog.at_level(logging.INFO):
        harness.run()

    assert harness.s3.uploads == [f"pipeline/{JOB_ID}/manifest.json"]
    message = caplog.text
    assert "selection lint warning" in message
    assert "'H01'" in message
    assert CLIP_ID in message
    assert SECOND_CLIP_ID in message


def test_missing_catalog_fails_the_stage_clearly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest(), with_catalog=False)

    with pytest.raises(GenerateError) as raised:
        harness.run()

    assert "library catalog unavailable or invalid" in str(raised.value)
    assert harness.s3.uploads == []


def test_stale_prompt_field_with_branding_hard_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Pre-v2 manifests carried generation prompts; a surviving one that embeds
    # branding must still hard-fail via the legacy prompt lint.
    manifest = _manifest()
    manifest["clips"][0]["hook_prompt"] = "Push in on the STC logo, bottom-right."
    harness = _harness(monkeypatch, tmp_path, manifest)

    with pytest.raises(GenerateError) as raised:
        harness.run()

    message = str(raised.value)
    assert "generation-prompt lint" in message
    assert f"{CLIP_ID}.hook_prompt" in message
    assert harness.s3.uploads == []
