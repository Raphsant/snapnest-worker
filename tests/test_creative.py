from __future__ import annotations

from typing import Any

import pytest

from worker.stages.creative import (
    CreativeError,
    compose_manifest_fields,
    extract_json,
    find_on_screen_text_mismatches,
    validate_creative_json,
    validate_creative_manifest,
)


def _package() -> dict[str, str]:
    return {
        "hook_angle": "discipline",
        "hook_on_screen_text": "TRATA EL TRADING EN SERIO",
        "hook_prompt": "Cinematic trading desk hook.",
        "close_on_screen_text": "EL PROCESO ES LA VENTAJA",
        "close_prompt": "Calm branded close.",
        "caption_youtube": "Lección de proceso. Suscríbete. #trading",
        "caption_tiktok": "La disciplina se practica. #trading",
        "caption_instagram": "¿Tienes un proceso? Zombie Hour LIVE. #trading",
        "compliance_check": "PASS",
    }


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


def test_compose_manifest_fields_matches_backend_plain_string_format() -> None:
    fields = compose_manifest_fields(_package())

    assert fields == {
        "hook_prompt": (
            'ON-SCREEN TEXT: "TRATA EL TRADING EN SERIO" | ANGLE: discipline'
            "\n\nCinematic trading desk hook."
        ),
        "close_prompt": (
            'ON-SCREEN TEXT: "EL PROCESO ES LA VENTAJA"'
            "\n\nCalm branded close."
        ),
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


def test_on_screen_text_exact_verbatim_containment_passes() -> None:
    package = _package()
    package["hook_prompt"] = (
        'Animate "TRATA EL TRADING EN SERIO" over the market screens.'
    )
    package["close_prompt"] = (
        'Resolve on "EL PROCESO ES LA VENTAJA" beside the STC logo.'
    )

    assert find_on_screen_text_mismatches(package) == []


def test_on_screen_text_mojibake_trips_hook_warning_check() -> None:
    package = _package()
    package["hook_on_screen_text"] = "¿POR QUÉ FALLAN?"
    package["hook_prompt"] = 'Animate "ÃPOR QUÉ FALLAN?" over the chart.'
    package["close_prompt"] = (
        'Resolve on "EL PROCESO ES LA VENTAJA" beside the STC logo.'
    )

    assert find_on_screen_text_mismatches(package) == [
        ("hook_on_screen_text", "hook_prompt")
    ]


def test_on_screen_text_accent_difference_trips_close_warning_check() -> None:
    package = _package()
    package["hook_prompt"] = (
        'Animate "TRATA EL TRADING EN SERIO" over the market screens.'
    )
    package["close_on_screen_text"] = "TÚ CONTROLAS EL RIESGO"
    package["close_prompt"] = 'Resolve on "TU CONTROLAS EL RIESGO".'

    assert find_on_screen_text_mismatches(package) == [
        ("close_on_screen_text", "close_prompt")
    ]


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
