"""Unit tests for the pre-generated asset library catalog."""

from __future__ import annotations

from typing import Any

import pytest

from worker.library import LibraryCatalog, format_for_prompt


def _asset(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "H01",
        "type": "hook",
        "s3_key": "library/hooks/H01.mp4",
        "duration_s": 4.0,
        "category": ["mindset"],
        "tags": ["psychology", "intensity", "edu"],
        "character": "zombie_trader",
        "description": "Zombie trader slams desk as charts crash",
        "times_used": 0,
    }
    base.update(overrides)
    return base


def _catalog_dict(assets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": "2026-08-25T00:00:00Z",
        "s3_bucket": "snapnest-uploads-dev-rs",
        "notes": "Prefer unused assets; never repeat a hook within a batch.",
        "assets": assets,
    }


def test_from_dict_happy_path() -> None:
    catalog = LibraryCatalog.from_dict(
        _catalog_dict(
            [
                _asset(),
                _asset(
                    id="O01",
                    type="outro",
                    s3_key="library/outros/O01.mp4",
                    duration_s=5.0,
                    logo_baked=True,
                ),
            ]
        )
    )

    assert catalog.version == 1
    assert catalog.notes.startswith("Prefer unused assets")

    hook = catalog.get("H01")
    assert hook is not None
    assert hook.type == "hook"
    assert hook.s3_key == "library/hooks/H01.mp4"
    assert hook.duration_s == 4.0
    assert hook.category == ("mindset",)
    assert hook.tags == ("psychology", "intensity", "edu")
    assert hook.character == "zombie_trader"
    assert hook.logo_baked is False  # default for hooks

    outro = catalog.get("O01")
    assert outro is not None
    assert outro.logo_baked is True


def test_get_hooks_outros() -> None:
    catalog = LibraryCatalog.from_dict(
        _catalog_dict(
            [
                _asset(id="H01"),
                _asset(id="H02"),
                _asset(
                    id="O01",
                    type="outro",
                    s3_key="library/outros/O01.mp4",
                    duration_s=5.0,
                ),
            ]
        )
    )

    assert [a.id for a in catalog.hooks()] == ["H01", "H02"]
    assert [a.id for a in catalog.outros()] == ["O01"]
    assert catalog.get("H99") is None


def test_duplicate_id_raises() -> None:
    with pytest.raises(ValueError, match="duplicates id 'H01'"):
        LibraryCatalog.from_dict(_catalog_dict([_asset(), _asset()]))


def test_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown type 'bridge'"):
        LibraryCatalog.from_dict(_catalog_dict([_asset(type="bridge")]))


def test_missing_s3_key_raises() -> None:
    with pytest.raises(ValueError, match="s3_key must be a non-empty string"):
        LibraryCatalog.from_dict(_catalog_dict([_asset(s3_key="")]))


def test_missing_duration_raises() -> None:
    with pytest.raises(ValueError, match="duration_s must be a number"):
        LibraryCatalog.from_dict(_catalog_dict([_asset(duration_s=None)]))


def test_format_for_prompt() -> None:
    catalog = LibraryCatalog.from_dict(
        _catalog_dict(
            [
                _asset(id="H03", category=["mindset"]),
                _asset(
                    id="O02",
                    type="outro",
                    s3_key="library/outros/O02.mp4",
                    duration_s=5.0,
                    category=["discipline"],
                ),
            ]
        )
    )

    rendered = format_for_prompt(catalog)

    assert (
        "H03 [hook] (mindset) tags: psychology,intensity,edu — "
        "Zombie trader slams desk as charts crash" in rendered
    )
    assert "O02 [outro] (discipline)" in rendered
    assert rendered.endswith(
        "Prefer unused assets; never repeat a hook within a batch."
    )
