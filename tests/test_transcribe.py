from __future__ import annotations

from pathlib import Path

import pytest

from worker.jobs import Job
from worker.stages.transcribe import (
    build_remux_command,
    build_start_kwargs,
    needs_remux,
    transcribe_options,
)


def _job(manifest: object | None) -> Job:
    return Job(
        id="job-1",
        source_file_id="file-1",
        source_s3_key="uploads/answer.mov",
        agency_id="agency-1",
        requested_by_id="user-1",
        status="QUEUED",
        current_stage=None,
        error=None,
        manifest=manifest,
    )


@pytest.mark.parametrize(
    "key",
    [
        "a/b.mp3",
        "a/b.mp4",
        "a/b.wav",
        "a/b.flac",
        "a/b.ogg",
        "a/b.amr",
        "a/b.webm",
        "a/b.m4a",
        "a/b.MP4",
        "a/b.WebM",
    ],
)
def test_accepted_extensions_skip_remux(key: str) -> None:
    assert needs_remux(key) is False


@pytest.mark.parametrize("key", ["a/b.mov", "a/b.MOV", "a/b.mkv", "a/b", "a/b.3gp"])
def test_other_extensions_need_remux(key: str) -> None:
    assert needs_remux(key) is True


def test_remux_command_is_stream_copy_with_faststart() -> None:
    cmd = build_remux_command(Path("in.mov"), Path("out.mp4"))

    assert cmd == [
        "ffmpeg", "-y", "-i", "in.mov", "-c", "copy",
        "-movflags", "+faststart", "out.mp4",
    ]


@pytest.mark.parametrize(
    "manifest", [None, {}, {"transcribe": None}, {"transcribe": "x"}, ["x"]]
)
def test_options_default_when_manifest_missing(manifest: object | None) -> None:
    assert transcribe_options(_job(manifest), default_language="es-US") == {
        "language": "es-US",
        "vocabulary": "",
        "filter": "",
    }


def test_options_read_from_manifest() -> None:
    manifest = {
        "transcribe": {"language": "en-US", "vocabulary": "v1", "filter": "f1"}
    }

    assert transcribe_options(_job(manifest), default_language="es-US") == {
        "language": "en-US",
        "vocabulary": "v1",
        "filter": "f1",
    }


def test_start_kwargs_omit_settings_when_unset() -> None:
    kwargs = build_start_kwargs(
        job_name="snapnest-tr-job-1-1",
        media_uri="s3://bkt/k.mp4",
        bucket="bkt",
        output_key="pipeline/job-1/raw/",
        language="es-US",
        vocabulary="",
        vocabulary_filter="",
    )

    assert "Settings" not in kwargs
    assert kwargs["OutputKey"] == "pipeline/job-1/raw/"
    assert kwargs["Subtitles"] == {"Formats": ["srt"]}


def test_start_kwargs_include_only_set_settings() -> None:
    kwargs = build_start_kwargs(
        job_name="n",
        media_uri="s3://bkt/k.mp4",
        bucket="bkt",
        output_key="pipeline/job-1/raw/",
        language="es-US",
        vocabulary="v1",
        vocabulary_filter="",
    )

    assert kwargs["Settings"] == {"VocabularyName": "v1"}
