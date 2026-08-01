from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from worker.jobs import Job
from worker.stages import download
from worker.stages.download import DownloadError, run_download


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeS3:
    """Minimal S3 stand-in: head_object 404s unless the key was pre-seeded."""

    def __init__(self, existing: tuple[str, ...] = ()) -> None:
        self.existing = set(existing)
        self.head_calls: list[tuple[str, str]] = []

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        self.head_calls.append((Bucket, Key))
        if Key not in self.existing:
            raise RuntimeError("404 Not Found")
        return {"ContentLength": 1}


class FakeWorkspace:
    def __init__(self, root: Path, s3: FakeS3, bucket: str = "bucket") -> None:
        self.dir = root
        self.s3 = s3
        self.bucket = bucket
        self.uploads: list[tuple[Path, str]] = []

    def path(self, name: str) -> Path:
        return self.dir / name

    def upload(self, local_path: Path, s3_key: str) -> None:
        self.uploads.append((local_path, s3_key))


def _job(
    *, source_s3_key: str | None = None, source_url: str | None = "https://youtu.be/x"
) -> Job:
    return Job(
        id="job-1",
        source_file_id=None,
        source_s3_key=source_s3_key,
        agency_id=None,
        requested_by_id="user-1",
        status="QUEUED",
        current_stage="download",
        error=None,
        source_type="YOUTUBE",
        source_url=source_url,
    )


def _ctx(job: Job, ws: FakeWorkspace) -> SimpleNamespace:
    return SimpleNamespace(
        job=job,
        workspace=ws,
        conn=object(),
        checkpoint_heartbeat=Mock(),
    )


# --------------------------------------------------------------------------- #
# run_download — checkpoint / routing behavior
# --------------------------------------------------------------------------- #


def test_skips_when_source_already_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = "pipeline/job-1/source/source.mp4"
    ws = FakeWorkspace(tmp_path, FakeS3(existing=(staged,)))
    job = _job(source_s3_key=staged)
    ctx = _ctx(job, ws)

    set_key = Mock()
    monkeypatch.setattr("worker.stages.download.jobs.set_source_s3_key",set_key)
    monkeypatch.setattr(
        download, "_run_yt_dlp", Mock(side_effect=AssertionError("must not download"))
    )

    run_download(ctx)  # type: ignore[arg-type]

    assert ws.uploads == []
    set_key.assert_not_called()
    assert ws.s3.head_calls == [("bucket", staged)]


def test_redownloads_when_key_set_but_object_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # sourceS3Key set on the row, but the object isn't actually in S3 -> re-fetch.
    ws = FakeWorkspace(tmp_path, FakeS3(existing=()))
    job = _job(source_s3_key="pipeline/job-1/source/source.mp4")
    ctx = _ctx(job, ws)

    monkeypatch.setattr("worker.stages.download.jobs.set_source_s3_key",Mock())
    monkeypatch.setattr(download, "_run_yt_dlp", _fake_download)

    run_download(ctx)  # type: ignore[arg-type]

    assert [key for _, key in ws.uploads] == ["pipeline/job-1/source/source.mp4"]


def test_missing_url_raises_without_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = FakeWorkspace(tmp_path, FakeS3())
    ctx = _ctx(_job(source_url=None), ws)
    monkeypatch.setattr(
        download, "_run_yt_dlp", Mock(side_effect=AssertionError("must not download"))
    )

    with pytest.raises(DownloadError, match="no sourceUrl"):
        run_download(ctx)  # type: ignore[arg-type]

    assert ws.uploads == []


def test_happy_path_uploads_records_key_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = FakeWorkspace(tmp_path, FakeS3())
    job = _job(source_s3_key=None)
    ctx = _ctx(job, ws)

    set_key = Mock()
    monkeypatch.setattr("worker.stages.download.jobs.set_source_s3_key",set_key)
    monkeypatch.setattr(download, "_run_yt_dlp", _fake_download)

    run_download(ctx)  # type: ignore[arg-type]

    dest_key = "pipeline/job-1/source/source.mp4"
    assert [key for _, key in ws.uploads] == [dest_key]
    set_key.assert_called_once_with(ctx.conn, "job-1", dest_key)
    # The just-set key is rebound onto the context so ingest sees it this run.
    assert ctx.job.source_s3_key == dest_key
    ctx.checkpoint_heartbeat.assert_called_once()
    # Multi-GB temp dir removed on success.
    assert not (tmp_path / "download").exists()


def test_cleans_up_temp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = FakeWorkspace(tmp_path, FakeS3())
    ctx = _ctx(_job(source_s3_key=None), ws)

    set_key = Mock()
    monkeypatch.setattr("worker.stages.download.jobs.set_source_s3_key",set_key)

    def boom(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"partial")  # a partial file that must still be swept
        raise DownloadError("yt-dlp exited 1")

    monkeypatch.setattr(download, "_run_yt_dlp", boom)

    with pytest.raises(DownloadError):
        run_download(ctx)  # type: ignore[arg-type]

    assert ws.uploads == []
    set_key.assert_not_called()
    assert not (tmp_path / "download").exists()


def _fake_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"\x00" * 16)


# --------------------------------------------------------------------------- #
# _run_yt_dlp — command shape and error handling
# --------------------------------------------------------------------------- #


def test_yt_dlp_command_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        captured["cmd"] = cmd
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"x")  # satisfy the no-file guard
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("worker.stages.download.subprocess.run", fake_run)
    dest = tmp_path / "source.mp4"

    download._run_yt_dlp("https://youtu.be/abc", dest)

    cmd = captured["cmd"]
    assert cmd[:3] == [sys.executable, "-m", "yt_dlp"]
    assert cmd[cmd.index("-f") + 1] == download._FORMAT
    assert cmd[cmd.index("--merge-output-format") + 1] == "mp4"
    assert cmd[cmd.index("--remux-video") + 1] == "mp4"
    assert "--no-playlist" in cmd
    assert cmd[cmd.index("-o") + 1] == str(dest)
    # `--` guards a URL that could otherwise look like a flag.
    assert cmd[-2:] == ["--", "https://youtu.be/abc"]


def test_yt_dlp_nonzero_raises_with_stderr_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=1, stdout="", stderr="ERROR: Video unavailable"
        )

    monkeypatch.setattr("worker.stages.download.subprocess.run", fake_run)

    with pytest.raises(DownloadError, match="Video unavailable"):
        download._run_yt_dlp("https://youtu.be/abc", tmp_path / "source.mp4")


def test_yt_dlp_success_but_no_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="", stderr="")  # no file written

    monkeypatch.setattr("worker.stages.download.subprocess.run", fake_run)

    with pytest.raises(DownloadError, match="produced no file"):
        download._run_yt_dlp("https://youtu.be/abc", tmp_path / "source.mp4")
