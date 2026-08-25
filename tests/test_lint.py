from __future__ import annotations

from typing import Any

import pytest

from worker.library import LibraryCatalog
from worker.lint import (
    GENERATION_PROMPT_FIELDS,
    SELECTION_FIELDS,
    lint_prompts,
    lint_selections,
)


def _manifest(**clip_fields: object) -> dict[str, object]:
    clip: dict[str, object] = {"id": "clip_01", "approved": True}
    clip.update(clip_fields)
    return {"status": "approved", "clips": [clip]}


def _hook_violations(prompt: str) -> list[dict[str, str]]:
    return lint_prompts(_manifest(hook_prompt=prompt)).violations


def _hook_warnings(prompt: str) -> list[dict[str, Any]]:
    return lint_prompts(_manifest(hook_prompt=prompt)).warnings


@pytest.mark.parametrize(
    "prompt",
    [
        "Cinematic trading floor, camera pushes in, charcoal and yellow accents.",
        "Hold on the market context as the candles settle.",
        "Extreme close-up of the brushed-metal texture under key light.",
        "",
    ],
)
def test_clean_generation_prompt_has_no_violations(prompt: str) -> None:
    assert _hook_violations(prompt) == []


@pytest.mark.parametrize(
    "prompt, expected_words",
    [
        ("Text overlay animates in from the left.", ["text"]),
        ("Place the STC logo in the corner.", ["stc", "logo"]),
        ("a lowercase stc watermark", ["stc", "watermark"]),
        ("Bold typography over a custom typeface.", ["typography", "typeface"]),
        (
            "Add branding, a wordmark, and a badge.",
            ["branding", "wordmark", "badge"],
        ),
        ("Subtitles and captions burned in.", ["subtitles", "captions"]),
        ("The company logos flash on screen.", ["logos"]),
    ],
)
def test_blocklisted_words_are_flagged(
    prompt: str, expected_words: list[str]
) -> None:
    words = sorted(v["matched_word"].lower() for v in _hook_violations(prompt))
    assert words == sorted(expected_words)


@pytest.mark.parametrize("token", ["STC", "stc", "Stc", "sTC"])
def test_stc_flagged_in_any_casing(token: str) -> None:
    violations = _hook_violations(f"Show the {token} mark on the wall.")

    assert [v["matched_word"] for v in violations] == [token]


def test_word_boundary_isolates_real_hits() -> None:
    # "context" and "texture" contain "text" only mid-word, so they stay clean;
    # the standalone "text" and "caption" are the only real hits.
    violations = _hook_violations(
        "Within this context and texture, add a text caption."
    )

    words = sorted(v["matched_word"].lower() for v in violations)
    assert words == ["caption", "text"]


def test_matched_word_preserves_original_casing() -> None:
    [violation] = _hook_violations("A big TEXT block appears.")

    assert violation["matched_word"] == "TEXT"


def test_violation_carries_clip_field_and_excerpt() -> None:
    [violation] = _hook_violations("Overlay a logo here.")

    assert violation["clipId"] == "clip_01"
    assert violation["field"] == "hook_prompt"
    assert violation["matched_word"] == "logo"
    assert "logo" in violation["prompt_excerpt"]


def test_repeated_word_reports_single_violation() -> None:
    violations = _hook_violations("logo and another logo and a third logo")

    assert len(violations) == 1
    assert violations[0]["matched_word"] == "logo"


def test_both_generation_prompt_fields_are_linted() -> None:
    manifest = _manifest(
        hook_prompt="Add the logo.",
        close_prompt="Burn a caption.",
    )

    fields = sorted(v["field"] for v in lint_prompts(manifest).violations)
    assert fields == ["close_prompt", "hook_prompt"]


def test_overlay_text_fields_are_exempt() -> None:
    # hook_text / close_text legitimately carry the overlay copy — the very
    # words the blocklist bans in prompts — and must never be linted.
    manifest = _manifest(
        hook_prompt="Clean cinematic push-in on the trading floor.",
        close_prompt="Calm resolve on an uncluttered desk.",
        hook_text="TEXT THAT IS FINE",
        close_text="STC LOGO WORDS",
    )

    result = lint_prompts(manifest)
    assert result.violations == []
    assert result.warnings == []


def test_unapproved_clips_are_not_scanned() -> None:
    manifest = {
        "clips": [
            {"id": "clip_01", "approved": False, "hook_prompt": "Add a logo."},
        ]
    }

    assert lint_prompts(manifest).violations == []


def test_only_approved_clip_violations_are_returned() -> None:
    manifest = {
        "clips": [
            {"id": "clip_01", "approved": False, "hook_prompt": "logo here"},
            {"id": "clip_02", "approved": True, "hook_prompt": "logo here"},
        ]
    }

    ids = [v["clipId"] for v in lint_prompts(manifest).violations]
    assert ids == ["clip_02"]


@pytest.mark.parametrize("manifest", [{}, {"clips": None}, {"clips": "nope"}])
def test_structurally_unusable_manifest_yields_no_violations(
    manifest: dict[str, object],
) -> None:
    assert lint_prompts(manifest).violations == []


def test_non_string_prompt_fields_are_skipped() -> None:
    assert lint_prompts(_manifest(hook_prompt=None, close_prompt=None)).violations == []


def test_overlay_fields_are_not_generation_prompt_fields() -> None:
    assert "hook_text" not in GENERATION_PROMPT_FIELDS
    assert "close_text" not in GENERATION_PROMPT_FIELDS


# --- negation awareness (v1.1) --------------------------------------------- #


@pytest.mark.parametrize(
    "prompt, word",
    [
        ("A clean lower third, no rendered text anywhere.", "text"),
        ("Plain background, no logos.", "logos"),
        ("A calm desk free of branding.", "branding"),  # two-word cue
        ("Uncluttered space, without any captions.", "captions"),
        ("Clean footage instead of typography.", "typography"),  # two-word cue
    ],
)
def test_negated_word_is_a_warning_not_a_violation(prompt: str, word: str) -> None:
    assert _hook_violations(prompt) == []
    warnings = _hook_warnings(prompt)
    assert [w["matched_word"].lower() for w in warnings] == [word]
    assert all(w["negated"] is True for w in warnings)


@pytest.mark.parametrize(
    "prompt, word",
    [
        ("Add text overlay from the left.", "text"),
        ("A logo in the corner.", "logo"),
    ],
)
def test_naked_word_still_blocks(prompt: str, word: str) -> None:
    assert [v["matched_word"].lower() for v in _hook_violations(prompt)] == [word]
    assert _hook_warnings(prompt) == []


def test_stc_blocks_even_when_negated_while_other_word_is_warned() -> None:
    # "no STC branding": STC still blocks (the letters ARE the violation), but
    # "branding" is negated and downgraded to a warning.
    prompt = "A clean plate, no STC branding."

    assert [v["matched_word"].lower() for v in _hook_violations(prompt)] == ["stc"]
    warnings = _hook_warnings(prompt)
    assert [w["matched_word"].lower() for w in warnings] == ["branding"]
    assert warnings[0]["negated"] is True


def test_naked_hit_is_not_masked_by_an_earlier_negated_mention() -> None:
    # Same word negated first, then naked: the naked request must still block.
    prompt = "No text here, but add text overlay later."

    assert [v["matched_word"].lower() for v in _hook_violations(prompt)] == ["text"]
    assert _hook_warnings(prompt) == []


def test_negation_cue_does_not_leak_across_a_sentence_boundary() -> None:
    # The "no" belongs to the previous sentence; "logo" here is a naked request.
    prompt = "Keep it clean, no branding. Show a logo in the corner."

    assert [v["matched_word"].lower() for v in _hook_violations(prompt)] == ["logo"]
    assert [w["matched_word"].lower() for w in _hook_warnings(prompt)] == ["branding"]


# --- library selection lint (v2) ------------------------------------------- #


def _selection_catalog() -> LibraryCatalog:
    def asset(asset_id: str, asset_type: str) -> dict[str, Any]:
        return {
            "id": asset_id,
            "type": asset_type,
            "s3_key": f"library/{asset_type}s/{asset_id}.mp4",
            "duration_s": 4.0,
            "category": ["mindset"],
            "tags": ["psychology"],
            "character": None,
            "description": f"{asset_id} test asset",
            "times_used": 0,
        }

    return LibraryCatalog.from_dict(
        {
            "version": 1,
            "updated_at": "2026-08-25T00:00:00Z",
            "notes": "test notes",
            "assets": [
                asset("H01", "hook"),
                asset("H02", "hook"),
                asset("O01", "outro"),
                asset("O02", "outro"),
            ],
        }
    )


def _selection_clip(clip_id: str, **fields: object) -> dict[str, object]:
    clip: dict[str, object] = {
        "id": clip_id,
        "approved": True,
        "category": "mindset",
        "hook_asset_id": "H01",
        "outro_asset_id": "O01",
    }
    clip.update(fields)
    return clip


def test_valid_selections_yield_empty_result() -> None:
    manifest = {
        "clips": [
            _selection_clip("clip_01"),
            _selection_clip(
                "clip_02", hook_asset_id="H02", outro_asset_id="O02"
            ),
        ]
    }

    result = lint_selections(manifest, _selection_catalog())
    assert result.violations == []
    assert result.warnings == []


@pytest.mark.parametrize("bad_value", [None, "", "   ", 7])
def test_missing_or_empty_selection_is_a_violation(bad_value: object) -> None:
    manifest = {"clips": [_selection_clip("clip_01", hook_asset_id=bad_value)]}

    [violation] = lint_selections(manifest, _selection_catalog()).violations
    assert violation["clipId"] == "clip_01"
    assert violation["field"] == "hook_asset_id"
    assert "non-empty asset id" in violation["reason"]


def test_unknown_asset_id_is_a_violation() -> None:
    manifest = {"clips": [_selection_clip("clip_01", outro_asset_id="O99")]}

    [violation] = lint_selections(manifest, _selection_catalog()).violations
    assert violation["field"] == "outro_asset_id"
    assert "'O99' does not exist" in violation["reason"]


def test_wrong_asset_type_is_a_violation() -> None:
    manifest = {"clips": [_selection_clip("clip_01", hook_asset_id="O01")]}

    [violation] = lint_selections(manifest, _selection_catalog()).violations
    assert violation["field"] == "hook_asset_id"
    assert "has type 'outro', expected 'hook'" in violation["reason"]


def test_cross_clip_duplicate_reports_every_offender() -> None:
    manifest = {
        "clips": [
            _selection_clip("clip_01"),
            _selection_clip("clip_02", outro_asset_id="O02"),  # H01 reused
            _selection_clip("clip_03", hook_asset_id="H02"),  # O01 reused
        ]
    }

    violations = lint_selections(manifest, _selection_catalog()).violations
    offenders = {(v["clipId"], v["field"]) for v in violations}
    assert offenders == {
        ("clip_01", "hook_asset_id"),
        ("clip_02", "hook_asset_id"),
        ("clip_01", "outro_asset_id"),
        ("clip_03", "outro_asset_id"),
    }
    # Every duplicate violation names all clips involved for that asset.
    h01 = [v for v in violations if "'H01'" in v["reason"]]
    assert all(
        "clip_01.hook_asset_id" in v["reason"]
        and "clip_02.hook_asset_id" in v["reason"]
        for v in h01
    )


def test_category_mismatch_is_a_warning_not_a_violation() -> None:
    manifest = {"clips": [_selection_clip("clip_01", category="technical")]}

    result = lint_selections(manifest, _selection_catalog())
    assert result.violations == []
    assert len(result.warnings) == 2  # both hook and outro are mindset-only
    assert all("'technical'" in w["reason"] for w in result.warnings)
    assert {w["field"] for w in result.warnings} == {
        "hook_asset_id",
        "outro_asset_id",
    }


def test_unapproved_clips_produce_no_selection_findings() -> None:
    manifest = {
        "clips": [
            _selection_clip(
                "clip_01",
                approved=False,
                hook_asset_id="GARBAGE",
                outro_asset_id=None,
            ),
        ]
    }

    result = lint_selections(manifest, _selection_catalog())
    assert result.violations == []
    assert result.warnings == []


@pytest.mark.parametrize("manifest", [{}, {"clips": None}, {"clips": "nope"}])
def test_structurally_unusable_manifest_yields_no_selection_findings(
    manifest: dict[str, object],
) -> None:
    assert lint_selections(manifest, _selection_catalog()).violations == []


def test_selection_fields_constant_shape() -> None:
    assert SELECTION_FIELDS == (
        ("hook_asset_id", "hook"),
        ("outro_asset_id", "outro"),
    )
