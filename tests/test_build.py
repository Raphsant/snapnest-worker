from __future__ import annotations

from typing import Any

import pytest

from worker.stages.build import (
    BuildError,
    InvalidClip,
    build_manifest,
    intersect_beeps,
    parse_srt,
    resolve_blocks,
    ts_to_seconds,
)

SRT = (
    "1\n00:00:00,000 --> 00:00:05,000\nBloque uno.\n\n"
    "2\n00:00:05,000 --> 00:00:12,000\nBloque dos\ncon dos lineas.\n\n"
    "3\n00:00:12,000 --> 00:00:30,000\nBloque tres.\n"
)


# --- ts_to_seconds --------------------------------------------------------- #


def test_ts_to_seconds_comma_and_dot() -> None:
    assert ts_to_seconds("00:01:57,320") == pytest.approx(117.32)
    assert ts_to_seconds("00:00:12.000") == pytest.approx(12.0)


# --- parse_srt / resolve_blocks (index -> timestamps + verbatim text) ------ #


def test_parse_srt_indexes_and_fields() -> None:
    blocks = parse_srt(SRT)
    assert set(blocks) == {1, 2, 3}
    assert blocks[2]["start"] == "00:00:05,000"
    assert blocks[2]["end"] == "00:00:12,000"
    assert blocks[2]["text"] == "Bloque dos\ncon dos lineas."


def test_resolve_single_block() -> None:
    blocks = parse_srt(SRT)
    start, end, transcript = resolve_blocks(blocks, 1, 1)
    assert (start, end) == ("00:00:00,000", "00:00:05,000")
    assert transcript == "Bloque uno."


def test_resolve_multi_block_transcript_is_verbatim_join() -> None:
    blocks = parse_srt(SRT)
    start, end, transcript = resolve_blocks(blocks, 1, 3)
    assert start == "00:00:00,000"
    assert end == "00:00:30,000"
    assert transcript == "Bloque uno.\nBloque dos\ncon dos lineas.\nBloque tres."


def test_resolve_missing_index_raises() -> None:
    blocks = parse_srt(SRT)
    with pytest.raises(InvalidClip):
        resolve_blocks(blocks, 2, 9)


def test_resolve_reversed_range_raises() -> None:
    blocks = parse_srt(SRT)
    with pytest.raises(InvalidClip):
        resolve_blocks(blocks, 3, 1)


def test_resolve_non_int_raises() -> None:
    blocks = parse_srt(SRT)
    with pytest.raises(InvalidClip):
        resolve_blocks(blocks, "2", 3)


# --- intersect_beeps (overlap-left/right, contained, disjoint) ------------- #

# Clip window is [10.0, 20.0] absolute seconds.


def test_beep_contained() -> None:
    assert intersect_beeps([[12.0, 14.0]], 10.0, 20.0) == [[2.0, 4.0]]


def test_beep_overlap_left_clamps_to_zero() -> None:
    assert intersect_beeps([[8.0, 12.0]], 10.0, 20.0) == [[0.0, 2.0]]


def test_beep_overlap_right_clamps_to_duration() -> None:
    assert intersect_beeps([[18.0, 25.0]], 10.0, 20.0) == [[8.0, 10.0]]


def test_beep_disjoint_dropped() -> None:
    assert intersect_beeps([[0.0, 5.0], [21.0, 25.0]], 10.0, 20.0) == []


def test_beep_boundary_touch_is_not_overlap() -> None:
    assert intersect_beeps([[20.0, 22.0]], 10.0, 20.0) == []


def test_beep_multiple_sorted_passthrough() -> None:
    beeps = intersect_beeps([[8.0, 12.0], [15.0, 16.0]], 10.0, 20.0)
    assert beeps == [[0.0, 2.0], [5.0, 6.0]]


# --- build_manifest -------------------------------------------------------- #


def _curation(category: str, clips: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "category": category,
        "selected_clips": clips,
        "rejected_segments": [{"start_block": 9, "end_block": 9, "topic": "x"}],
    }


def test_build_manifest_merges_categories_and_computes_fields() -> None:
    mindset = _curation(
        "mindset",
        [{"id": "a", "start_block": 1, "end_block": 3, "title": "T", "confidence": "borderline"}],
    )
    technical = _curation(
        "technical",
        [{"id": "b", "start_block": 2, "end_block": 3}],
    )

    manifest = build_manifest(
        srt_text=SRT,
        curations=[mindset, technical],
        bleep_ranges=[[6.0, 8.0]],
        source_video="pipeline/job1/session_bleeped.mp4",
        srt_file="pipeline/job1/subtitles.corrected.srt",
        generated="2026-07-14",
    )

    assert manifest["status"] == "pending_approval"
    assert manifest["source_video"] == "pipeline/job1/session_bleeped.mp4"
    assert manifest["generated"] == "2026-07-14"
    assert [c["category"] for c in manifest["clips"]] == ["mindset", "technical"]
    assert [c["id"] for c in manifest["clips"]] == ["clip_01", "clip_02"]

    first = manifest["clips"][0]
    assert first["approved"] is None
    assert first["hook_prompt"] is None and first["post_copy"] is None
    assert first["transcript"].startswith("Bloque uno.")
    assert first["duration_seconds"] == 30.0
    # bleep [6,8] within clip [0,30] -> clip-relative [6,8]
    assert first["beep_timestamps"] == [[6.0, 8.0]]

    # rejected_segments merged from both files, each tagged with its category.
    assert [r["category"] for r in manifest["rejected_segments"]] == [
        "mindset",
        "technical",
    ]


def test_build_manifest_skips_invalid_and_keeps_counter() -> None:
    data = _curation(
        "mindset",
        [
            {"id": "bad", "start_block": 2, "end_block": 99},
            {"id": "good", "start_block": 1, "end_block": 2},
        ],
    )
    manifest = build_manifest(
        srt_text=SRT,
        curations=[data],
        bleep_ranges=[],
        source_video="v",
        srt_file="s",
    )
    # First clip skipped (invalid), but the counter advanced, so the survivor
    # keeps id clip_02 (faithful to the reference script's numbering).
    assert [c["id"] for c in manifest["clips"]] == ["clip_02"]


def test_build_manifest_raises_on_empty_srt() -> None:
    with pytest.raises(BuildError):
        build_manifest(
            srt_text="not an srt",
            curations=[],
            bleep_ranges=[],
            source_video="v",
            srt_file="s",
        )
