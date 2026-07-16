from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
from mypy_boto3_s3.client import S3Client
from psycopg import Connection
from psycopg.rows import DictRow

from worker import higgsfield
from worker.higgsfield import GenerationParams, GenerationResult
from worker.jobs import Job
from worker.stages.generate import (
    BRIDGE_IN_PROMPT,
    BRIDGE_OUT_PROMPT,
    GenerateError,
    GenerateStageContext,
    build_frame_command,
    run_generate,
)
from worker.workspace import Workspace

JOB_ID = "job-1"
CLIP_ID = "clip_01"
RESULT_URL = "https://cdn.example.test/result.mp4"


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.uploads: list[str] = []
        self.downloads: list[str] = []

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloads.append(key)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(self.objects[key])

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.uploads.append(key)
        self.objects[key] = Path(filename).read_bytes()


class FakeManifestDB:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = copy.deepcopy(manifest)
        self.checkpoints: list[dict[str, Any]] = []

    def save(
        self,
        conn: Connection[DictRow],
        job_id: str,
        manifest: Mapping[str, Any],
    ) -> None:
        assert job_id == JOB_ID
        self.manifest = copy.deepcopy(dict(manifest))
        self.checkpoints.append(copy.deepcopy(self.manifest))

    def load(self, conn: Connection[DictRow], job_id: str) -> object:
        assert job_id == JOB_ID
        return copy.deepcopy(self.manifest)


@dataclass
class FakeContext:
    job: Job
    workspace: Workspace
    conn: Connection[DictRow]
    checkpoint_heartbeat: Callable[[], None]


@dataclass
class Harness:
    context: FakeContext
    s3: FakeS3
    db: FakeManifestDB
    heartbeats: list[int]

    def run(self) -> None:
        with self.context.workspace:
            run_generate(cast(GenerateStageContext, self.context))


def _checkpoint(asset: str) -> dict[str, object]:
    return {
        "s3Key": f"pipeline/{JOB_ID}/generated/{CLIP_ID}/{asset}.mp4",
        "generationId": f"existing-{asset}",
        "estimatedCredits": 36,
        "completedAt": "2026-07-15T12:00:00+00:00",
    }


def _manifest(*, checkpointed: tuple[str, ...] = ()) -> dict[str, Any]:
    generated = {asset: _checkpoint(asset) for asset in checkpointed}
    return {
        "status": "approved",
        "clips": [
            {
                "id": CLIP_ID,
                "approved": True,
                "hook_prompt": "hook prompt",
                "close_prompt": "close prompt",
                "generated": generated,
            }
        ],
    }


def _harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest: dict[str, Any],
) -> Harness:
    fake_s3 = FakeS3()
    fake_s3.objects.update(
        {
            f"pipeline/{JOB_ID}/clips/subclips/{CLIP_ID}_first5.mp4": b"first5",
            f"pipeline/{JOB_ID}/clips/subclips/{CLIP_ID}_last5.mp4": b"last5",
        }
    )
    for asset, checkpoint in manifest["clips"][0]["generated"].items():
        fake_s3.objects[str(checkpoint["s3Key"])] = f"{asset}-video".encode()

    workspace = Workspace(
        JOB_ID,
        tmp_path,
        cast(S3Client, fake_s3),
        "bucket",
    )
    heartbeats: list[int] = []
    context = FakeContext(
        job=Job(
            id=JOB_ID,
            source_file_id="file-1",
            source_s3_key="source.mp4",
            agency_id="agency-1",
            requested_by_id="user-1",
            status="CREATIVE_APPROVED",
            current_stage=None,
            error=None,
            manifest=manifest,
        ),
        workspace=workspace,
        conn=cast(Connection[DictRow], object()),
        checkpoint_heartbeat=lambda: heartbeats.append(1),
    )
    fake_db = FakeManifestDB(manifest)
    monkeypatch.setattr(
        "worker.stages.generate.jobs.save_manifest_checkpoint",
        fake_db.save,
    )
    monkeypatch.setattr(
        "worker.stages.generate.jobs.load_manifest",
        fake_db.load,
    )

    def fake_ffmpeg(command: list[str]) -> None:
        Path(command[-1]).write_bytes(b"jpg")

    monkeypatch.setattr("worker.stages.generate._run_ffmpeg", fake_ffmpeg)
    return Harness(context, fake_s3, fake_db, heartbeats)


def _install_costs_and_balance(
    monkeypatch: pytest.MonkeyPatch,
    *,
    balance: int,
) -> list[GenerationParams]:
    cost_params: list[GenerationParams] = []

    def fake_get_cost(params: GenerationParams) -> int:
        cost_params.append(params)
        return {4: 36, 5: 45}[params.duration_s]

    monkeypatch.setattr(
        "worker.stages.generate.higgsfield.get_cost",
        fake_get_cost,
    )
    monkeypatch.setattr(
        "worker.stages.generate.higgsfield.balance",
        lambda: balance,
    )
    return cost_params


def _install_successful_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> list[GenerationParams]:
    calls: list[GenerationParams] = []

    def fake_generate(
        params: GenerationParams, output_path: Path
    ) -> GenerationResult:
        calls.append(params)
        output_path.write_bytes(b"generated mp4")
        return GenerationResult(
            id=f"generation-{len(calls)}",
            result_url=RESULT_URL,
            credits_charged=None,
            elapsed_s=1.0,
        )

    monkeypatch.setattr(
        "worker.stages.generate.higgsfield.generate",
        fake_generate,
    )
    return calls


def test_preflight_insufficient_balance_never_generates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())
    cost_params = _install_costs_and_balance(monkeypatch, balance=100)
    generate = Mock()
    monkeypatch.setattr(
        "worker.stages.generate.higgsfield.generate",
        generate,
    )

    with pytest.raises(GenerateError) as raised:
        harness.run()

    message = str(raised.value)
    assert "balance=100" in message
    assert "required=153" in message
    assert "hook=36" in message
    assert "outro=45" in message
    assert "bridge_each=36" in message
    assert [params.duration_s for params in cost_params] == [4, 5, 4]
    assert cost_params[2].start_image is None
    assert cost_params[2].end_image is None
    generate.assert_not_called()
    assert harness.db.checkpoints == []
    assert harness.heartbeats == []


def test_resume_skips_checkpointed_hook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(
        monkeypatch,
        tmp_path,
        _manifest(checkpointed=("hook",)),
    )
    _install_costs_and_balance(monkeypatch, balance=1000)
    generation_calls = _install_successful_generation(monkeypatch)

    harness.run()

    assert [params.prompt for params in generation_calls] == [
        "close prompt",
        BRIDGE_IN_PROMPT,
        BRIDGE_OUT_PROMPT,
    ]
    assert len(harness.db.checkpoints) == 3
    assert len(harness.heartbeats) == 3
    assert (
        f"pipeline/{JOB_ID}/generated/{CLIP_ID}/hook.mp4"
        in harness.s3.downloads
    )


def test_download_error_retries_download_without_regenerating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(
        monkeypatch,
        tmp_path,
        _manifest(checkpointed=("outro", "bridge_in", "bridge_out")),
    )
    _install_costs_and_balance(monkeypatch, balance=1000)
    generation_calls: list[GenerationParams] = []
    download_calls: list[tuple[str, Path]] = []

    def failed_download_generation(
        params: GenerationParams, output_path: Path
    ) -> GenerationResult:
        generation_calls.append(params)
        raise higgsfield.HiggsfieldDownloadError(
            "remote generation completed",
            generation_id="spent-generation",
            result_url=RESULT_URL,
        )

    def fake_download(result_url: str, output_path: Path) -> None:
        download_calls.append((result_url, output_path))
        output_path.write_bytes(b"recovered mp4")

    monkeypatch.setattr(
        "worker.stages.generate.higgsfield.generate",
        failed_download_generation,
    )
    monkeypatch.setattr(
        "worker.stages.generate.higgsfield.download",
        fake_download,
    )

    harness.run()

    assert len(generation_calls) == 1
    assert len(download_calls) == 1
    assert download_calls[0][0] == RESULT_URL
    hook = harness.db.manifest["clips"][0]["generated"]["hook"]
    assert hook["generationId"] == "spent-generation"


def test_mid_package_failure_preserves_prior_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())
    _install_costs_and_balance(monkeypatch, balance=1000)
    calls: list[str] = []

    def fail_outro(
        params: GenerationParams, output_path: Path
    ) -> GenerationResult:
        calls.append(output_path.name)
        if output_path.name == "hook.mp4":
            output_path.write_bytes(b"hook")
            return GenerationResult(
                id="hook-generation",
                result_url=RESULT_URL,
                credits_charged=None,
                elapsed_s=1.0,
            )
        raise higgsfield.HiggsfieldError(
            "generation rejected",
            command=("higgsfield",),
            exit_code=3,
        )

    monkeypatch.setattr(
        "worker.stages.generate.higgsfield.generate",
        fail_outro,
    )

    with pytest.raises(GenerateError) as raised:
        harness.run()

    assert CLIP_ID in str(raised.value)
    assert "outro" in str(raised.value)
    assert calls == ["hook.mp4", "outro.mp4", "outro.mp4"]
    generated = harness.db.manifest["clips"][0]["generated"]
    assert set(generated) == {"hook"}
    assert generated["hook"]["generationId"] == "hook-generation"
    assert len(harness.db.checkpoints) == 1
    assert harness.heartbeats == [1]


def test_ambiguous_submit_is_not_retried_and_preserves_raw_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(
        monkeypatch,
        tmp_path,
        _manifest(checkpointed=("outro", "bridge_in", "bridge_out")),
    )
    _install_costs_and_balance(monkeypatch, balance=1000)
    raw_stdout = '["possible-orphan-id", "unexpected-second-value"]'
    generate_calls: list[GenerationParams] = []

    def ambiguous_submit(
        params: GenerationParams, output_path: Path
    ) -> GenerationResult:
        generate_calls.append(params)
        raise higgsfield.HiggsfieldAmbiguousSubmitError(
            command=("higgsfield", "generate", "create"),
            raw_stdout=raw_stdout,
        )

    monkeypatch.setattr(
        "worker.stages.generate.higgsfield.generate",
        ambiguous_submit,
    )

    with pytest.raises(GenerateError) as raised:
        harness.run()

    assert len(generate_calls) == 1
    assert raw_stdout in str(raised.value)
    assert "NOT retrying" in str(raised.value)
    assert harness.db.checkpoints == []
    assert harness.heartbeats == []


def test_success_checkpoints_each_asset_and_heartbeats_each_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())
    cost_params = _install_costs_and_balance(monkeypatch, balance=1000)
    generation_calls = _install_successful_generation(monkeypatch)

    harness.run()

    assert [path for path in harness.s3.uploads if path.endswith(".mp4")] == [
        f"pipeline/{JOB_ID}/generated/{CLIP_ID}/hook.mp4",
        f"pipeline/{JOB_ID}/generated/{CLIP_ID}/outro.mp4",
        f"pipeline/{JOB_ID}/generated/{CLIP_ID}/bridge_in.mp4",
        f"pipeline/{JOB_ID}/generated/{CLIP_ID}/bridge_out.mp4",
    ]
    assert harness.s3.uploads[-1] == f"pipeline/{JOB_ID}/manifest.json"
    assert len(harness.db.checkpoints) == 4
    assert harness.heartbeats == [1, 1, 1, 1]
    assert [params.duration_s for params in generation_calls] == [4, 5, 4, 4]
    assert all(params.model == "seedance_2_0" for params in generation_calls)
    assert all(params.aspect_ratio == "9:16" for params in generation_calls)
    assert all(params.resolution == "1080p" for params in generation_calls)
    assert all(params.generate_audio is False for params in generation_calls)
    assert generation_calls[1].start_image is None
    assert generation_calls[1].end_image is None
    assert generation_calls[2].start_image is not None
    assert generation_calls[2].end_image is not None
    assert generation_calls[3].start_image is not None
    assert generation_calls[3].end_image is not None
    assert [params.duration_s for params in cost_params] == [4, 5, 4]


def test_frame_commands_match_first_and_last_frame_recipes(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "frame.jpg"

    assert build_frame_command(source, output, last=False) == [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
    assert build_frame_command(source, output, last=True) == [
        "ffmpeg",
        "-y",
        "-sseof",
        "-0.1",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
