from __future__ import annotations

from pathlib import Path
from typing import Any

from worker.stages.ingest import (
    _s3_key_from_uri,
    build_ffmpeg_command,
    correct_srt,
    correct_transcript_json,
    extract_bleep_ranges,
    merge_ranges,
)


# --- merge_ranges ---------------------------------------------------------- #


def test_merge_overlapping_ranges() -> None:
    assert merge_ranges([[0.0, 1.0], [0.5, 1.5]], 0.05) == [[0.0, 1.5]]


def test_merge_adjacent_within_gap() -> None:
    assert merge_ranges([[0.0, 1.0], [1.02, 2.0]], 0.05) == [[0.0, 2.0]]


def test_keeps_ranges_separated_beyond_gap() -> None:
    assert merge_ranges([[0.0, 1.0], [1.5, 2.0]], 0.05) == [[0.0, 1.0], [1.5, 2.0]]


def test_merge_sorts_unsorted_input() -> None:
    assert merge_ranges([[2.0, 3.0], [0.0, 1.0]], 0.05) == [[0.0, 1.0], [2.0, 3.0]]


def test_merge_empty() -> None:
    assert merge_ranges([], 0.05) == []


# --- extract_bleep_ranges -------------------------------------------------- #


def _item(start: str, end: str, content: str) -> dict[str, Any]:
    return {
        "start_time": start,
        "end_time": end,
        "alternatives": [{"content": content}],
    }


def test_extract_masks_only() -> None:
    items = [
        _item("1.0", "1.5", "***"),
        _item("2.0", "2.5", "hola"),
        {"alternatives": [{"content": "."}], "type": "punctuation"},
    ]
    assert extract_bleep_ranges(items, pad=0.0, merge_gap=0.05) == [[1.0, 1.5]]


def test_extract_no_masks_is_empty() -> None:
    items = [_item("1.0", "1.5", "hola")]
    assert extract_bleep_ranges(items, pad=0.15, merge_gap=0.05) == []


def test_extract_applies_pad_and_clamps_to_zero() -> None:
    ranges = extract_bleep_ranges([_item("0.1", "0.5", "***")], pad=0.15, merge_gap=0.05)
    assert ranges[0][0] == 0.0
    assert abs(ranges[0][1] - 0.65) < 1e-9


# --- build_ffmpeg_command -------------------------------------------------- #


def test_ffmpeg_empty_ranges_is_remux_copy() -> None:
    cmd = build_ffmpeg_command(
        Path("in.mp4"), Path("out.mp4"), [], beep_hz=1000, beep_volume=0.35
    )
    assert cmd == ["ffmpeg", "-y", "-i", "in.mp4", "-c", "copy", "out.mp4"]


def test_ffmpeg_builds_filter_for_ranges() -> None:
    cmd = build_ffmpeg_command(
        Path("in.mp4"), Path("out.mp4"), [[0.0, 1.0]], beep_hz=1000, beep_volume=0.35
    )
    assert "-filter_complex" in cmd
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert "sine=frequency=1000" in filter_complex
    assert "amix=inputs=2" in filter_complex
    assert cmd[-1] == "out.mp4"


# --- correct_srt ----------------------------------------------------------- #


def test_correct_srt_only_touches_text_lines() -> None:
    srt = (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "primarket\n"
        "\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "hola\n"
    )
    out = correct_srt(srt)
    lines = out.splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:01,000"
    assert lines[2] == "premarket"
    assert "00:00:01,000 --> 00:00:02,000" in out


# --- correct_transcript_json ---------------------------------------------- #


def test_correct_transcript_json_updates_items_and_transcript() -> None:
    data: dict[str, Any] = {
        "results": {
            "transcripts": [{"transcript": "the primarket"}],
            "items": [
                {"alternatives": [{"content": "primarket"}], "type": "pronunciation"},
                {"alternatives": [{"content": "***"}], "type": "pronunciation"},
            ],
        }
    }
    out = correct_transcript_json(data)
    assert out["results"]["items"][0]["alternatives"][0]["content"] == "premarket"
    assert out["results"]["items"][1]["alternatives"][0]["content"] == "***"
    assert out["results"]["transcripts"][0]["transcript"] == "the premarket"


# --- _s3_key_from_uri ------------------------------------------------------ #


def test_key_from_path_style_uri() -> None:
    uri = "https://s3.us-east-1.amazonaws.com/bucket/pipeline/job1/snapnest-job1.json"
    assert _s3_key_from_uri(uri, "bucket") == "pipeline/job1/snapnest-job1.json"


def test_key_from_virtual_hosted_uri() -> None:
    uri = "https://bucket.s3.us-east-1.amazonaws.com/pipeline/job1/x.srt"
    assert _s3_key_from_uri(uri, "bucket") == "pipeline/job1/x.srt"


def test_key_from_s3_scheme_uri() -> None:
    assert _s3_key_from_uri("s3://bucket/pipeline/job1/x.srt", "bucket") == (
        "pipeline/job1/x.srt"
    )
