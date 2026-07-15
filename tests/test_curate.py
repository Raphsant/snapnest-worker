from __future__ import annotations

from typing import Any

import pytest

from worker.stages.curate import (
    CurationError,
    extract_json,
    parse_srt_indexes,
    validate_curation,
)

SRT = (
    "1\n00:00:00,000 --> 00:00:02,000\nhola\n\n"
    "2\n00:00:02,000 --> 00:00:04,000\nque tal\n\n"
    "3\n00:00:04,000 --> 00:00:06,000\nadios\n"
)
INDEXES = {1, 2, 3}


# --- extract_json ---------------------------------------------------------- #


def test_extract_json_ignores_prose_before_and_after() -> None:
    text = '<transcript_review>\n- notes\n</transcript_review>\n{"category": "mindset"} trailing'
    assert extract_json(text) == {"category": "mindset"}


def test_extract_json_handles_nested_objects() -> None:
    text = 'junk {"a": {"b": 1}} more junk'
    assert extract_json(text) == {"a": {"b": 1}}


def test_extract_json_raises_without_braces() -> None:
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_extract_json_raises_on_non_object() -> None:
    with pytest.raises(ValueError):
        extract_json("[1, 2, 3]")


# --- parse_srt_indexes ----------------------------------------------------- #


def test_parse_srt_indexes() -> None:
    assert parse_srt_indexes(SRT) == {1, 2, 3}


def test_parse_srt_indexes_ignores_numeric_text_lines() -> None:
    srt = "1\n00:00:00,000 --> 00:00:02,000\n2024 was tough\n"
    # "2024" is subtitle text, not a cue index (no timestamp line follows it).
    assert parse_srt_indexes(srt) == {1}


# --- validate_curation ----------------------------------------------------- #


def _clip(clip_id: str, start: int, end: int) -> dict[str, Any]:
    return {"id": clip_id, "start_block": start, "end_block": end}


def test_validate_passes_clean_result() -> None:
    data = {
        "category": "mindset",
        "selected_clips": [_clip("c1", 1, 2), _clip("c2", 2, 3)],
    }
    valid = validate_curation(data, category="mindset", srt_indexes=INDEXES)
    assert len(valid) == 2


def test_validate_category_mismatch_raises() -> None:
    data = {"category": "technical", "selected_clips": [_clip("c1", 1, 2)]}
    with pytest.raises(CurationError):
        validate_curation(data, category="mindset", srt_indexes=INDEXES)


def test_validate_empty_clips_raises() -> None:
    data = {"category": "mindset", "selected_clips": []}
    with pytest.raises(CurationError):
        validate_curation(data, category="mindset", srt_indexes=INDEXES)


def test_validate_drops_nonexistent_index() -> None:
    data = {
        "category": "mindset",
        "selected_clips": [
            _clip("c1", 1, 2),
            _clip("c2", 2, 3),
            _clip("c3", 1, 2),
            _clip("bad", 2, 99),
        ],
    }
    valid = validate_curation(data, category="mindset", srt_indexes=INDEXES)
    assert {c["id"] for c in valid} == {"c1", "c2", "c3"}


def test_validate_drops_reversed_range() -> None:
    # 1 of 4 dropped (25%, under threshold) so the drop is isolated cleanly.
    data = {
        "category": "mindset",
        "selected_clips": [
            _clip("c1", 1, 3),
            _clip("c2", 1, 2),
            _clip("c3", 2, 3),
            _clip("bad", 3, 1),
        ],
    }
    valid = validate_curation(data, category="mindset", srt_indexes=INDEXES)
    assert {c["id"] for c in valid} == {"c1", "c2", "c3"}


def test_validate_drops_duplicate_ids() -> None:
    data = {
        "category": "mindset",
        "selected_clips": [
            _clip("dup", 1, 2),
            _clip("c2", 1, 3),
            _clip("c3", 2, 3),
            _clip("dup", 2, 3),
        ],
    }
    valid = validate_curation(data, category="mindset", srt_indexes=INDEXES)
    # First "dup" kept; the later duplicate is dropped.
    assert [c["id"] for c in valid] == ["dup", "c2", "c3"]


def test_validate_fails_when_over_threshold_dropped() -> None:
    # 2 of 3 invalid (66% > 30%) -> hard failure.
    data = {
        "category": "mindset",
        "selected_clips": [_clip("ok", 1, 2), _clip("b1", 1, 99), _clip("b2", 5, 6)],
    }
    with pytest.raises(CurationError):
        validate_curation(data, category="mindset", srt_indexes=INDEXES)
