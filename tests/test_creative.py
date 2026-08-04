from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from mypy_boto3_s3.client import S3Client
from psycopg import Connection
from psycopg.rows import DictRow

from worker.jobs import Job
from worker.stages import StageContext
from worker.stages.creative import (
    CreativeError,
    compose_manifest_fields,
    extract_json,
    run_creative,
    validate_creative_json,
    validate_creative_manifest,
)
from worker.workspace import Workspace


def _package() -> dict[str, str]:
    return {
        "hook_angle": "discipline",
        "hook_text": "TRATA EL TRADING EN SERIO",
        "hook_prompt": "Cinematic trading floor, fast push-in, high contrast.",
        "close_text": "EL PROCESO ES LA VENTAJA",
        "close_prompt": "Calm, premium resolve on an uncluttered desk.",
        "caption_youtube": "Lección de proceso. Suscríbete. #trading",
        "caption_tiktok": "La disciplina se practica. #trading",
        "caption_instagram": "¿Tienes un proceso? Zombie Hour LIVE. #trading",
        "compliance_check": "PASS",
    }


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_extract_json_uses_first_opening_and_last_closing_brace() -> None:
    parsed = extract_json('preamble {"hook_angle": "discipline"} trailing')

    assert parsed == {"hook_angle": "discipline"}


def test_validate_creative_json_accepts_required_non_empty_strings() -> None:
    package = _package()

    assert validate_creative_json(package) == package


@pytest.mark.parametrize("bad_value", [None, 3, "", "   "])
def test_validate_creative_json_rejects_missing_or_empty_fields(
    bad_value: object,
) -> None:
    package: dict[str, object] = {}
    package.update(_package())
    package["hook_prompt"] = bad_value

    with pytest.raises(ValueError, match="hook_prompt"):
        validate_creative_json(package)


def test_compose_manifest_fields_emits_native_text_and_clean_prompts() -> None:
    fields = compose_manifest_fields(_package())

    assert fields == {
        "hook_text": "TRATA EL TRADING EN SERIO",
        "hook_prompt": "Cinematic trading floor, fast push-in, high contrast.",
        "close_text": "EL PROCESO ES LA VENTAJA",
        "close_prompt": "Calm, premium resolve on an uncluttered desk.",
        "post_copy": (
            "### YouTube Shorts\n"
            "Lección de proceso. Suscríbete. #trading\n\n"
            "### TikTok\n"
            "La disciplina se practica. #trading\n\n"
            "### Instagram\n"
            "¿Tienes un proceso? Zombie Hour LIVE. #trading\n\n"
            "Compliance check: PASS"
        ),
    }


def test_validate_creative_manifest_copies_and_selects_only_approved() -> None:
    transcript = "  Texto verbatim.\nSegunda línea.  "
    manifest: dict[str, Any] = {
        "status": "approved",
        "clips": [
            {
                "id": "clip_01",
                "approved": True,
                "category": "mindset",
                "transcript": transcript,
                "hook_prompt": None,
                "close_prompt": None,
                "post_copy": None,
            },
            {
                "id": "clip_02",
                "approved": False,
                "category": "technical",
                "transcript": "Rejected",
                "hook_prompt": None,
                "close_prompt": None,
                "post_copy": None,
            },
        ],
    }

    copied, approved = validate_creative_manifest(manifest)
    approved[0]["hook_prompt"] = "generated"

    assert approved[0]["transcript"] == transcript
    assert len(approved) == 1
    assert manifest["clips"][0]["hook_prompt"] is None
    assert copied["clips"][1]["hook_prompt"] is None


@pytest.mark.parametrize(
    "manifest, message",
    [
        (None, "manifest is missing"),
        ({"status": "pending_approval", "clips": []}, "status must be 'approved'"),
        ({"status": "approved", "clips": []}, "no approved clips"),
    ],
)
def test_validate_creative_manifest_rejects_invalid_inputs(
    manifest: object | None,
    message: str,
) -> None:
    with pytest.raises(CreativeError, match=message):
        validate_creative_manifest(manifest)


# --------------------------------------------------------------------------- #
# run_creative: choke point #1 (native overlay fields + lint wiring at the gate)
# --------------------------------------------------------------------------- #


class _FakeS3:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.uploads[key] = Path(filename).read_bytes()


@dataclass
class _FakeConfig:
    creative_model: str = "claude-test"
    anthropic_api_key: str = "test-key"


def _noop_heartbeat() -> None:
    pass


@dataclass
class _FakeCreativeContext:
    job: Job
    workspace: Workspace
    conn: Connection[DictRow]
    config: Any
    checkpoint_heartbeat: Callable[[], None] = _noop_heartbeat


def _approved_manifest() -> dict[str, Any]:
    return {
        "status": "approved",
        "clips": [
            {
                "id": "clip_01",
                "approved": True,
                "category": "mindset",
                "transcript": "Algo importante dijo Eduardo.",
                "hook_prompt": None,
                "close_prompt": None,
                "post_copy": None,
            }
        ],
    }


def _run_creative_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    manifest: dict[str, Any],
    package: dict[str, str],
) -> tuple[dict[str, Any], _FakeS3]:
    saved: dict[str, Any] = {}

    def fake_save(
        conn: Connection[DictRow], job_id: str, stored: dict[str, Any]
    ) -> None:
        saved["manifest"] = stored

    monkeypatch.setattr(
        "worker.stages.creative.jobs.save_manifest_awaiting_creative_approval",
        fake_save,
    )
    monkeypatch.setattr(
        "worker.stages.creative._anthropic_client",
        lambda cfg: object(),
    )
    monkeypatch.setattr(
        "worker.stages.creative._generate_clip_package",
        lambda *args, **kwargs: (dict(package), 123),
    )

    fake_s3 = _FakeS3()
    job = Job(
        id="job-1",
        source_file_id="file-1",
        source_s3_key="source.mp4",
        agency_id="agency-1",
        requested_by_id="user-1",
        status="APPROVED",
        current_stage=None,
        error=None,
        manifest=manifest,
    )
    workspace = Workspace("job-1", tmp_path, cast(S3Client, fake_s3), "bucket")
    ctx = _FakeCreativeContext(
        job=job,
        workspace=workspace,
        conn=cast(Connection[DictRow], object()),
        config=_FakeConfig(),
    )
    with workspace:
        run_creative(cast(StageContext, ctx))
    return saved["manifest"], fake_s3


def test_run_creative_emits_native_overlay_text_and_clean_prompts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    saved, fake_s3 = _run_creative_capture(
        monkeypatch,
        tmp_path,
        manifest=_approved_manifest(),
        package=_package(),
    )

    clip = saved["clips"][0]
    assert clip["hook_text"] == "TRATA EL TRADING EN SERIO"
    assert clip["close_text"] == "EL PROCESO ES LA VENTAJA"
    assert clip["hook_prompt"] == (
        "Cinematic trading floor, fast push-in, high contrast."
    )
    assert clip["close_prompt"] == "Calm, premium resolve on an uncluttered desk."
    assert saved["lint_violations"] == []
    assert "pipeline/job-1/manifest.json" in fake_s3.uploads


def test_run_creative_records_lint_violations_at_the_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dirty = _package()
    dirty["hook_prompt"] = "Push in as the STC logo animates in."

    saved, _ = _run_creative_capture(
        monkeypatch,
        tmp_path,
        manifest=_approved_manifest(),
        package=dirty,
    )

    violations = saved["lint_violations"]
    assert {v["field"] for v in violations} == {"hook_prompt"}
    assert {v["matched_word"].lower() for v in violations} == {"stc", "logo"}
    # Recorded for operator review, NOT raised: the job still parks at the
    # creative gate with the native overlay text in place.
    assert saved["clips"][0]["hook_text"] == "TRATA EL TRADING EN SERIO"


def test_run_creative_records_negated_lint_warnings_at_the_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A negated branding word is recorded as a non-blocking warning, not a
    # blocking violation — so creative v2's "no logos" phrasings don't wedge.
    negated = _package()
    negated["hook_prompt"] = "Clean trading floor, no logos, plain background."

    saved, _ = _run_creative_capture(
        monkeypatch,
        tmp_path,
        manifest=_approved_manifest(),
        package=negated,
    )

    assert saved["lint_violations"] == []
    warnings = saved["lint_warnings"]
    assert {w["field"] for w in warnings} == {"hook_prompt"}
    assert {w["matched_word"].lower() for w in warnings} == {"logos"}
    assert all(w["negated"] is True for w in warnings)
