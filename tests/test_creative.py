from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mypy_boto3_s3.client import S3Client
from psycopg import Connection
from psycopg.rows import DictRow

from worker.jobs import Job
from worker.library import LibraryCatalog
from worker.stages import StageContext
from worker.stages.creative import (
    DONE_FIELDS,
    CreativeError,
    compose_manifest_fields,
    extract_json,
    load_system_prompt,
    run_creative,
    validate_asset_selection,
    validate_creative_json,
    validate_creative_manifest,
)
from worker.workspace import Workspace


def _catalog() -> LibraryCatalog:
    def hook(asset_id: str) -> dict[str, Any]:
        return {
            "id": asset_id,
            "type": "hook",
            "s3_key": f"library/hooks/{asset_id}.mp4",
            "duration_s": 4.0,
            "category": ["mindset"],
            "tags": ["psychology", "intensity"],
            "character": "zombie_trader",
            "description": "Zombie trader slams desk as charts crash",
            "times_used": 0,
        }

    def outro(asset_id: str) -> dict[str, Any]:
        return {
            "id": asset_id,
            "type": "outro",
            "s3_key": f"library/outros/{asset_id}.mp4",
            "duration_s": 5.0,
            "category": ["mindset"],
            "tags": ["calm", "premium"],
            "character": None,
            "description": "Calm dark studio close with baked logo",
            "times_used": 0,
            "logo_baked": True,
        }

    return LibraryCatalog.from_dict(
        {
            "version": 1,
            "updated_at": "2026-08-25T00:00:00Z",
            "notes": "Prefer unused assets; H10/O04 are universal fallbacks.",
            "assets": [hook("H01"), hook("H02"), outro("O01"), outro("O02")],
        }
    )


def _package() -> dict[str, str]:
    return {
        "hook_angle": "discipline",
        "hook_text": "TRATA EL TRADING EN SERIO",
        "hook_asset_id": "H01",
        "close_text": "EL PROCESO ES LA VENTAJA",
        "outro_asset_id": "O01",
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
    package["hook_asset_id"] = bad_value

    with pytest.raises(ValueError, match="hook_asset_id"):
        validate_creative_json(package)


def test_validate_asset_selection_unknown_id_raises() -> None:
    package = _package()
    package["hook_asset_id"] = "H99"

    with pytest.raises(ValueError, match="hook_asset_id 'H99' does not exist"):
        validate_asset_selection(
            package, _catalog(), category="mindset", clip_id="clip_01"
        )


def test_validate_asset_selection_wrong_type_raises() -> None:
    package = _package()
    package["hook_asset_id"] = "O01"

    with pytest.raises(ValueError, match="has type 'outro', expected 'hook'"):
        validate_asset_selection(
            package, _catalog(), category="mindset", clip_id="clip_01"
        )


def test_validate_asset_selection_allows_reuse_quietly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Reuse is allowed policy — one prior use is not even worth a warning.
    with caplog.at_level("WARNING", logger="worker.stages.creative"):
        validate_asset_selection(
            _package(),
            _catalog(),
            category="mindset",
            clip_id="clip_02",
            usage_counts={"H01": 1, "O01": 1},
        )

    assert caplog.text == ""


def test_validate_asset_selection_warns_on_repeated_reuse(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="worker.stages.creative"):
        validate_asset_selection(
            _package(),
            _catalog(),
            category="mindset",
            clip_id="clip_03",
            usage_counts={"H01": 2, "O01": 1},
        )

    assert "hook_asset_id=H01 already used 2 time(s)" in caplog.text
    assert "reuse allowed" in caplog.text
    # O01 sits below the threshold, so it stays quiet.
    assert "outro_asset_id=O01" not in caplog.text


def test_validate_asset_selection_category_mismatch_warns_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="worker.stages.creative"):
        validate_asset_selection(
            _package(), _catalog(), category="technical", clip_id="clip_01"
        )

    assert "cross-category selection allowed" in caplog.text


def test_load_system_prompt_substitutes_catalog() -> None:
    rendered = load_system_prompt(_catalog())

    assert "{{ASSET_LIBRARY}}" not in rendered
    assert "H01 [hook] (mindset)" in rendered
    assert "H10/O04 are universal fallbacks." in rendered


def test_load_system_prompt_errors_without_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stub = tmp_path / "creative_system.md"
    stub.write_text("a prompt with no placeholder", encoding="utf-8")
    monkeypatch.setattr("worker.stages.creative.PROMPT_PATH", stub)

    with pytest.raises(CreativeError, match="ASSET_LIBRARY"):
        load_system_prompt(_catalog())


def test_compose_manifest_fields_emits_overlay_text_and_asset_ids() -> None:
    fields = compose_manifest_fields(_package())

    assert fields == {
        "hook_text": "TRATA EL TRADING EN SERIO",
        "hook_asset_id": "H01",
        "close_text": "EL PROCESO ES LA VENTAJA",
        "outro_asset_id": "O01",
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
    assert "hook_prompt" not in fields
    assert "close_prompt" not in fields


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
    approved[0]["hook_asset_id"] = "H01"

    assert approved[0]["transcript"] == transcript
    assert len(approved) == 1
    assert "hook_asset_id" not in manifest["clips"][0]
    assert "hook_asset_id" not in copied["clips"][1]


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
# run_creative: selection wiring, dedup, and the creative gate
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


class _FakeMessages:
    """Anthropic messages stub: records create() kwargs, replays canned texts."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._texts.pop(0))]
        )


def _approved_manifest(clip_ids: tuple[str, ...] = ("clip_01",)) -> dict[str, Any]:
    return {
        "status": "approved",
        "clips": [
            {
                "id": clip_id,
                "approved": True,
                "category": "mindset",
                "transcript": "Algo importante dijo Eduardo.",
                "hook_prompt": None,
                "close_prompt": None,
                "post_copy": None,
            }
            for clip_id in clip_ids
        ],
    }


def _done_clip(
    clip_id: str = "clip_01",
    *,
    hook_asset_id: str = "H01",
    outro_asset_id: str = "O01",
) -> dict[str, Any]:
    """An approved clip a previous pass already finished creative for.

    Carries an ``assembled`` checkpoint too, so the tests prove the skip keys
    off the creative fields rather than off assembly, and that the checkpoint
    survives this stage's full-manifest overwrite.
    """

    return {
        "id": clip_id,
        "approved": True,
        "category": "mindset",
        "transcript": "Algo importante dijo Eduardo.",
        "hook_prompt": None,
        "close_prompt": None,
        "hook_text": "YA TIENE GANCHO",
        "hook_asset_id": hook_asset_id,
        "close_text": "YA TIENE CIERRE",
        "outro_asset_id": outro_asset_id,
        "post_copy": "### YouTube Shorts\nya publicado",
        "assembled": {
            "s3Key": f"pipeline/job-1/final/final_{clip_id}_9x16.mp4",
            "completedAt": "2026-08-01T00:00:00+00:00",
        },
    }


def _new_clip(clip_id: str = "clip_02") -> dict[str, Any]:
    """A freshly approved clip with no creative fields yet."""

    return {
        "id": clip_id,
        "approved": True,
        "category": "mindset",
        "transcript": "Un segundo momento importante.",
        "hook_prompt": None,
        "close_prompt": None,
        "post_copy": None,
    }


def _patch_common(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
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
        "worker.stages.creative._load_catalog",
        lambda ctx: _catalog(),
    )
    return saved


def _make_ctx(
    tmp_path: Path, manifest: dict[str, Any]
) -> tuple[_FakeCreativeContext, _FakeS3, Workspace]:
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
    return ctx, fake_s3, workspace


def _run_creative_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    manifest: dict[str, Any],
    package: dict[str, str],
) -> tuple[dict[str, Any], _FakeS3]:
    saved = _patch_common(monkeypatch)
    monkeypatch.setattr(
        "worker.stages.creative._anthropic_client",
        lambda cfg: object(),
    )
    monkeypatch.setattr(
        "worker.stages.creative._generate_clip_package",
        lambda *args, **kwargs: (dict(package), 123),
    )

    ctx, fake_s3, workspace = _make_ctx(tmp_path, manifest)
    with workspace:
        run_creative(cast(StageContext, ctx))
    return saved["manifest"], fake_s3


def _run_creative_with_messages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    manifest: dict[str, Any],
    texts: list[str],
) -> tuple[dict[str, Any], _FakeMessages, _FakeS3]:
    """Run the stage against a canned Anthropic transcript; capture the save.

    Unlike _run_creative_capture this keeps the real _generate_clip_package, so
    the user message (and therefore the availability block) is observable.
    An empty ``texts`` list means "no model call is expected" — an unexpected
    create() pops from an empty list and fails the test loudly.
    """

    saved = _patch_common(monkeypatch)
    fake_messages = _FakeMessages(texts)
    monkeypatch.setattr(
        "worker.stages.creative._anthropic_client",
        lambda cfg: SimpleNamespace(messages=fake_messages),
    )

    ctx, fake_s3, workspace = _make_ctx(tmp_path, manifest)
    with workspace:
        run_creative(cast(StageContext, ctx))
    # KeyError here would itself mean the gate save never ran.
    return saved["manifest"], fake_messages, fake_s3


def test_run_creative_emits_overlay_text_and_asset_selections(
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
    assert clip["hook_asset_id"] == "H01"
    assert clip["outro_asset_id"] == "O01"
    # v2 writes no generation prompts; the legacy keys stay untouched (None).
    assert clip["hook_prompt"] is None
    assert clip["close_prompt"] is None
    assert saved["lint_violations"] == []
    assert saved["lint_warnings"] == []
    assert "pipeline/job-1/manifest.json" in fake_s3.uploads


def test_run_creative_still_lints_stale_prompt_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A pre-v2 manifest can still carry a dirty legacy hook_prompt; the gate
    # lint keeps recording it for operator review even though v2 writes none.
    manifest = _approved_manifest()
    manifest["clips"][0]["hook_prompt"] = "Push in as the STC logo animates in."

    saved, _ = _run_creative_capture(
        monkeypatch,
        tmp_path,
        manifest=manifest,
        package=_package(),
    )

    violations = saved["lint_violations"]
    assert {v["field"] for v in violations} == {"hook_prompt"}
    assert {v["matched_word"].lower() for v in violations} == {"stc", "logo"}


def test_run_creative_two_clips_report_usage_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    saved = _patch_common(monkeypatch)
    packages = [
        _package(),
        {**_package(), "hook_asset_id": "H02", "outro_asset_id": "O02"},
    ]
    fake_messages = _FakeMessages([json.dumps(p) for p in packages])
    monkeypatch.setattr(
        "worker.stages.creative._anthropic_client",
        lambda cfg: SimpleNamespace(messages=fake_messages),
    )

    ctx, _, workspace = _make_ctx(
        tmp_path, _approved_manifest(("clip_01", "clip_02"))
    )
    with workspace:
        run_creative(cast(StageContext, ctx))

    first_user = fake_messages.calls[0]["messages"][0]["content"]
    second_user = fake_messages.calls[1]["messages"][0]["content"]
    # Every asset is listed on every request, untouched ones included.
    assert "HOOKS (id — times used this job): H01 — 0, H02 — 0" in first_user
    assert "OUTROS (id — times used this job): O01 — 0, O02 — 0" in first_user
    assert "Reusing an asset is allowed." in first_user
    # The first clip's picks come back at one use for the second clip.
    assert "HOOKS (id — times used this job): H01 — 1, H02 — 0" in second_user
    assert "OUTROS (id — times used this job): O01 — 1, O02 — 0" in second_user

    clips = saved["manifest"]["clips"]
    assert (clips[0]["hook_asset_id"], clips[0]["outro_asset_id"]) == (
        "H01",
        "O01",
    )
    assert (clips[1]["hook_asset_id"], clips[1]["outro_asset_id"]) == (
        "H02",
        "O02",
    )


def test_run_creative_skips_a_clip_that_already_has_creative_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    done = _done_clip()
    before = copy.deepcopy(done)

    saved, fake_messages, _ = _run_creative_with_messages(
        monkeypatch,
        tmp_path,
        manifest={"status": "approved", "clips": [done]},
        texts=[],
    )

    # The whole point: no model call, so no Anthropic spend on a finished clip.
    assert fake_messages.calls == []

    stored = saved["clips"][0]
    for field in DONE_FIELDS:
        assert stored[field] == before[field]
    # The assembled checkpoint survives the stage's full-manifest overwrite.
    assert stored["assembled"] == before["assembled"]


def test_run_creative_all_skipped_still_reaches_the_creative_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A re-opened job whose approved clips are all already done must still park
    # at AWAITING_CREATIVE_APPROVAL, not fall through with nothing persisted.
    manifest = {
        "status": "approved",
        "clips": [
            _done_clip("clip_01"),
            _done_clip("clip_02", hook_asset_id="H02", outro_asset_id="O02"),
        ],
    }

    saved, fake_messages, fake_s3 = _run_creative_with_messages(
        monkeypatch,
        tmp_path,
        manifest=manifest,
        texts=[],
    )

    assert fake_messages.calls == []
    # The gate save ran (the helper would KeyError otherwise) with a complete,
    # linted manifest, and the S3 copy was refreshed alongside it.
    assert [clip["id"] for clip in saved["clips"]] == ["clip_01", "clip_02"]
    assert saved["lint_violations"] == []
    assert saved["lint_warnings"] == []
    assert "pipeline/job-1/manifest.json" in fake_s3.uploads


def test_run_creative_processes_a_clip_missing_one_creative_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Selections but no post_copy: skipping on the asset id alone would strand
    # the job, because the backend creative gate requires all five fields.
    partial = _done_clip()
    del partial["post_copy"]

    fresh = {**_package(), "hook_asset_id": "H02", "outro_asset_id": "O02"}
    saved, fake_messages, _ = _run_creative_with_messages(
        monkeypatch,
        tmp_path,
        manifest={"status": "approved", "clips": [partial]},
        texts=[json.dumps(fresh)],
    )

    assert len(fake_messages.calls) == 1
    # Seeding scans every clip, so the clip's OWN stale ids come back at one
    # use in its own availability block. Documented consequence of the
    # whole-manifest seed: it nudges, it no longer forbids.
    user = fake_messages.calls[0]["messages"][0]["content"]
    assert "HOOKS (id — times used this job): H01 — 1, H02 — 0" in user
    assert "OUTROS (id — times used this job): O01 — 1, O02 — 0" in user

    stored = saved["clips"][0]
    assert stored["post_copy"].startswith("### YouTube Shorts")
    assert (stored["hook_asset_id"], stored["outro_asset_id"]) == ("H02", "O02")


def test_run_creative_extension_run_processes_only_the_new_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    done = _done_clip("clip_01")
    before = copy.deepcopy(done)

    fresh = {**_package(), "hook_asset_id": "H02", "outro_asset_id": "O02"}
    saved, fake_messages, _ = _run_creative_with_messages(
        monkeypatch,
        tmp_path,
        manifest={"status": "approved", "clips": [done, _new_clip("clip_02")]},
        texts=[json.dumps(fresh)],
    )

    # Exactly one model call, and it is the new clip.
    assert len(fake_messages.calls) == 1
    user = fake_messages.calls[0]["messages"][0]["content"]
    assert "CLIP ID: clip_02" in user
    # The done clip's ids were seeded, so the new clip sees what an earlier
    # pass shipped at one use and leans toward the untouched assets.
    assert "HOOKS (id — times used this job): H01 — 1, H02 — 0" in user
    assert "OUTROS (id — times used this job): O01 — 1, O02 — 0" in user

    stored_done, stored_new = saved["clips"]
    for field in DONE_FIELDS:
        assert stored_done[field] == before[field]
    assert stored_done["assembled"] == before["assembled"]
    assert (stored_new["hook_asset_id"], stored_new["outro_asset_id"]) == (
        "H02",
        "O02",
    )
