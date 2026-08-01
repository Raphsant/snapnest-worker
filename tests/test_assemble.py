from __future__ import annotations

import copy
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from mypy_boto3_s3.client import S3Client
from psycopg import Connection
from psycopg.rows import DictRow

from worker import jobs
from worker.artifacts import main_clip_key
from worker.jobs import Job
from worker.stages import assemble
from worker.stages.assemble import (
    AssembleError,
    AssembleStageContext,
    CONFIG,
    SilenceInterval,
    TextOverlay,
    build_boundary_fade_command,
    build_clip_srt,
    build_duration_probe_command,
    build_final_command,
    build_generated_segment_command,
    build_main_caption_command,
    build_silence_cut_command,
    build_silence_detect_command,
    parse_master_srt,
    run_assemble,
)
from worker.workspace import Workspace

JOB_ID = "job-assemble"
CLIP_ID = "clip_04"
BUCKET = "snapnest-uploads-dev-rs"
MASTER_SRT_KEY = f"pipeline/{JOB_ID}/subtitles.corrected.srt"
MAIN_CLIP_KEY = f"pipeline/{JOB_ID}/clips/{CLIP_ID}.mp4"
MASTER_SRT = (
    b"1\r\n"
    b"00:00:10,000 --> 00:00:11,250\r\n"
    b"Primera l\xc3\xadnea\r\nsegunda l\xc3\xadnea\r\n"
    b"\r\n"
    b"2\r\n"
    b"00:00:11,500 --> 00:00:12,500\r\n"
    b"Texto final\r\n"
)
PROBED_DURATION_S = 7.5
TEXTFILE_RE = re.compile(r"textfile='([^']+)'")


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.heads: list[str] = []
        self.downloads: list[str] = []
        self.uploads: list[str] = []

    def head_object(self, *, Bucket: str, Key: str) -> None:
        assert Bucket == BUCKET
        self.heads.append(Key)
        if Key not in self.objects:
            raise FileNotFoundError(Key)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        assert bucket == BUCKET
        self.downloads.append(key)
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[key])

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        assert bucket == BUCKET
        self.uploads.append(key)
        self.objects[key] = Path(filename).read_bytes()


class FakeManifestDB:
    def __init__(self, manifest: dict[str, object]) -> None:
        self.manifest = copy.deepcopy(manifest)
        self.checkpoints: list[dict[str, object]] = []

    def load(self, conn: Connection[DictRow], job_id: str) -> object:
        assert job_id == JOB_ID
        return copy.deepcopy(self.manifest)

    def save(
        self,
        conn: Connection[DictRow],
        job_id: str,
        manifest: Mapping[str, object],
    ) -> None:
        assert job_id == JOB_ID
        self.manifest = copy.deepcopy(dict(manifest))
        self.checkpoints.append(copy.deepcopy(self.manifest))


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
    ffmpeg_commands: list[list[str]]
    ffmpeg_descriptions: list[str]
    probed_segments: list[str]
    overlay_textfile_bytes: dict[str, bytes]

    def run(self) -> None:
        with self.context.workspace:
            run_assemble(cast(AssembleStageContext, self.context))

    def command_for(self, description_suffix: str) -> list[str]:
        matches = [
            command
            for command, description in zip(
                self.ffmpeg_commands, self.ffmpeg_descriptions, strict=True
            )
            if description.endswith(description_suffix)
        ]
        assert len(matches) == 1, description_suffix
        return matches[0]


def _generated_checkpoint(asset: str) -> dict[str, object]:
    return {
        "s3Key": f"pipeline/{JOB_ID}/generated/{CLIP_ID}/{asset}.mp4",
        "generationId": f"generation-{asset}",
        "completedAt": "hand-written opaque value",
        "estimatedCredits": 36,
    }


def _manifest(
    *,
    assembled: bool = False,
    missing_asset: str | None = None,
    hook_text: str | None = None,
    close_text: str | None = None,
) -> dict[str, object]:
    assets = ("hook", "outro", "bridge_in", "bridge_out")
    generated = {
        asset: _generated_checkpoint(asset)
        for asset in assets
        if asset != missing_asset
    }
    clip: dict[str, object] = {
        "id": CLIP_ID,
        "approved": True,
        "start": "00:00:10,000",
        "end": "00:00:12,500",
        "start_block": 1,
        "end_block": 2,
        "duration_seconds": 2.5,
        "transcript": "flat text is deliberately unused",
        "generated": generated,
    }
    if hook_text is not None:
        clip["hook_text"] = hook_text
    if close_text is not None:
        clip["close_text"] = close_text
    if assembled:
        clip["assembled"] = {
            "s3Key": f"pipeline/{JOB_ID}/final/final_{CLIP_ID}_9x16.mp4",
            "completedAt": "2026-07-16T12:00:00+00:00",
        }
    return {
        "srt_file": MASTER_SRT_KEY,
        "source_video": f"pipeline/{JOB_ID}/session_bleeped.mp4",
        "status": "approved",
        "clips": [clip],
        "generated": "2026-07-16",
        "rejected_segments": [],
    }


def _approved_clip(manifest: Mapping[str, object]) -> dict[str, object]:
    clips = manifest["clips"]
    assert isinstance(clips, list)
    clip = clips[0]
    assert isinstance(clip, dict)
    return cast(dict[str, object], clip)


def _successful_probe(
    command: list[str],
    *,
    capture_output: bool,
    text: bool,
    encoding: str,
    errors: str,
) -> subprocess.CompletedProcess[str]:
    assert command == ["ffmpeg", "-hide_banner", "-filters"]
    assert capture_output is True
    assert text is True
    assert encoding == "utf-8"
    assert errors == "replace"
    return subprocess.CompletedProcess(
        command,
        0,
        stdout=" ... subtitles         V->V       Render text subtitles\n",
        stderr="",
    )


def _harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest: dict[str, object],
) -> Harness:
    fake_s3 = FakeS3()
    fake_s3.objects.update(
        {
            CONFIG["bed_music_key"]: b"bed",
            CONFIG["logo_key"]: b"logo",
            MASTER_SRT_KEY: MASTER_SRT,
            MAIN_CLIP_KEY: b"main",
        }
    )
    clip = _approved_clip(manifest)
    generated = clip.get("generated")
    if isinstance(generated, dict):
        for raw_checkpoint in generated.values():
            if not isinstance(raw_checkpoint, dict):
                continue
            key = raw_checkpoint.get("s3Key")
            if isinstance(key, str):
                fake_s3.objects[key] = b"generated"

    workspace = Workspace(
        JOB_ID,
        tmp_path,
        cast(S3Client, fake_s3),
        BUCKET,
    )
    heartbeats: list[int] = []
    context = FakeContext(
        job=Job(
            id=JOB_ID,
            source_file_id="source-file",
            source_s3_key="source.mp4",
            agency_id="agency",
            requested_by_id="user",
            status="CREATIVE_APPROVED",
            current_stage="generate",
            error=None,
            # Proves assemble reloads the authoritative DB manifest.
            manifest=None,
        ),
        workspace=workspace,
        conn=cast(Connection[DictRow], object()),
        checkpoint_heartbeat=lambda: heartbeats.append(1),
    )
    fake_db = FakeManifestDB(manifest)
    monkeypatch.setattr(jobs, "load_manifest", fake_db.load)
    monkeypatch.setattr(jobs, "save_manifest_checkpoint", fake_db.save)
    monkeypatch.setattr(shutil, "which", lambda executable: "/usr/bin/ffmpeg")
    monkeypatch.setattr(subprocess, "run", _successful_probe)

    ffmpeg_commands: list[list[str]] = []
    ffmpeg_descriptions: list[str] = []
    probed_segments: list[str] = []
    overlay_textfile_bytes: dict[str, bytes] = {}

    def fake_ffmpeg(command: list[str], *, description: str) -> str:
        ffmpeg_commands.append(command)
        ffmpeg_descriptions.append(description)
        # Capture temp textfile bytes while they still exist on disk.
        for argument in command:
            for textfile in TEXTFILE_RE.findall(argument):
                path = Path(textfile)
                overlay_textfile_bytes[path.name] = path.read_bytes()
        if command[-1] != "-":
            Path(command[-1]).write_bytes(b"encoded mp4")
        return ""

    def fake_probe_duration(source: Path, *, description: str) -> float:
        probed_segments.append(source.name)
        return PROBED_DURATION_S

    monkeypatch.setattr(assemble, "_run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(assemble, "_probe_duration", fake_probe_duration)
    return Harness(
        context=context,
        s3=fake_s3,
        db=fake_db,
        heartbeats=heartbeats,
        ffmpeg_commands=ffmpeg_commands,
        ffmpeg_descriptions=ffmpeg_descriptions,
        probed_segments=probed_segments,
        overlay_textfile_bytes=overlay_textfile_bytes,
    )


def test_master_srt_slice_preserves_text_and_retimes_blocks() -> None:
    blocks = parse_master_srt(MASTER_SRT)

    result = build_clip_srt(
        blocks,
        start_block=1,
        end_block=2,
        clip_start_ms=9_500,
        clip_id=CLIP_ID,
    )

    assert blocks[1].text.encode("utf-8") == (
        b"Primera l\xc3\xadnea\r\nsegunda l\xc3\xadnea"
    )
    assert result == (
        b"1\n"
        b"00:00:00,500 --> 00:00:01,750\n"
        b"Primera l\xc3\xadnea\r\nsegunda l\xc3\xadnea\n"
        b"\n"
        b"2\n"
        b"00:00:02,000 --> 00:00:03,000\n"
        b"Texto final\n"
    )


def test_master_srt_slice_rejects_negative_shift() -> None:
    blocks = parse_master_srt(MASTER_SRT)

    with pytest.raises(AssembleError, match="negative subtitle timestamp"):
        build_clip_srt(
            blocks,
            start_block=1,
            end_block=1,
            clip_start_ms=10_001,
            clip_id=CLIP_ID,
        )


def test_master_srt_slice_rejects_missing_block() -> None:
    blocks = parse_master_srt(MASTER_SRT)

    with pytest.raises(AssembleError, match="missing master SRT block 3"):
        build_clip_srt(
            blocks,
            start_block=2,
            end_block=3,
            clip_start_ms=10_000,
            clip_id=CLIP_ID,
        )


def test_preflight_rejects_missing_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())
    monkeypatch.setattr(shutil, "which", lambda executable: None)

    with pytest.raises(AssembleError, match="not found on PATH"):
        harness.run()

    assert harness.db.checkpoints == []
    assert harness.s3.uploads == []


def test_preflight_rejects_failed_filter_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())

    def failed_probe(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="filter listing failed",
        )

    monkeypatch.setattr(subprocess, "run", failed_probe)

    with pytest.raises(AssembleError, match="filter probe failed"):
        harness.run()

    assert harness.db.checkpoints == []


def test_preflight_rejects_missing_subtitles_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())

    def no_subtitles(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=" ... scale             V->V       Scale video\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", no_subtitles)

    with pytest.raises(AssembleError, match="subtitles filter is unavailable"):
        harness.run()

    assert harness.db.checkpoints == []


def test_preflight_names_missing_bed_music(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())
    del harness.s3.objects[CONFIG["bed_music_key"]]

    with pytest.raises(AssembleError, match=CONFIG["bed_music_key"]):
        harness.run()

    assert harness.db.checkpoints == []


def test_preflight_names_missing_logo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())
    del harness.s3.objects[CONFIG["logo_key"]]

    with pytest.raises(AssembleError, match=CONFIG["logo_key"]):
        harness.run()

    assert harness.db.checkpoints == []


def test_preflight_names_missing_generated_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(
        monkeypatch,
        tmp_path,
        _manifest(missing_asset="bridge_out"),
    )

    with pytest.raises(
        AssembleError,
        match=f"clip {CLIP_ID} missing generated asset bridge_out",
    ):
        harness.run()

    assert harness.s3.heads == []
    assert harness.db.checkpoints == []


def test_preflight_names_missing_main_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())
    del harness.s3.objects[MAIN_CLIP_KEY]

    with pytest.raises(AssembleError, match=MAIN_CLIP_KEY):
        harness.run()

    assert harness.db.checkpoints == []


def test_resume_skips_assembled_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest(assembled=True))

    harness.run()

    assert harness.ffmpeg_commands == []
    assert harness.probed_segments == []
    assert harness.db.checkpoints == []
    assert harness.heartbeats == []
    assert harness.s3.uploads == [f"pipeline/{JOB_ID}/manifest.json"]
    assert not harness.context.workspace.dir.exists()


def test_success_uploads_and_checkpoints_once_per_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())

    harness.run()

    final_key = f"pipeline/{JOB_ID}/final/final_{CLIP_ID}_9x16.mp4"
    assert harness.s3.uploads == [
        final_key,
        f"pipeline/{JOB_ID}/manifest.json",
    ]
    assert len(harness.db.checkpoints) == 1
    checkpoint = _approved_clip(harness.db.manifest)["assembled"]
    assert isinstance(checkpoint, dict)
    assert set(checkpoint) == {"s3Key", "completedAt"}
    assert checkpoint["s3Key"] == final_key
    assert isinstance(checkpoint["completedAt"], str)
    assert harness.heartbeats == [1]
    assert len(harness.ffmpeg_commands) == 14
    assert harness.ffmpeg_descriptions == [
        f"clip {CLIP_ID} asset hook normalization",
        f"clip {CLIP_ID} asset outro normalization",
        f"clip {CLIP_ID} asset bridge_in normalization",
        f"clip {CLIP_ID} asset bridge_out normalization",
        f"clip {CLIP_ID} main caption burn",
        f"clip {CLIP_ID} silence detection",
        f"clip {CLIP_ID} silence removal",
        f"clip {CLIP_ID} segment bridge_out tail trim",
        f"clip {CLIP_ID} segment hook boundary fades",
        f"clip {CLIP_ID} segment bridge_in boundary fades",
        f"clip {CLIP_ID} segment main boundary fades",
        f"clip {CLIP_ID} segment bridge_out boundary fades",
        f"clip {CLIP_ID} segment outro boundary fades",
        f"clip {CLIP_ID} final concat and audio mix",
    ]
    assert not harness.context.workspace.dir.exists()


def test_boundary_fades_probe_processed_intermediates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())

    harness.run()

    # Durations come from the processed intermediates, post speed change and
    # post silence removal, never from arithmetic on the sources. The first
    # probe is the bridge_out tail-trim precheck; the boundary-fade loop then
    # probes each segment it fades (bridge_out post-trim).
    assert harness.probed_segments == [
        "bridge_out_normalized.mp4",
        "hook_normalized.mp4",
        "bridge_in_normalized.mp4",
        "main_trimmed.mp4",
        "bridge_out_trimmed.mp4",
        "outro_normalized.mp4",
    ]

    fade_in = "fade=t=in:st=0:d=0.25"
    fade_out = f"fade=t=out:st={PROBED_DURATION_S - 0.25:.6f}:d=0.25"
    expected_filters = {
        "hook": f"{fade_out},format=yuv420p,setsar=1",
        "bridge_in": f"{fade_in},{fade_out},format=yuv420p,setsar=1",
        "main": f"{fade_in},{fade_out},format=yuv420p,setsar=1",
        "bridge_out": f"{fade_in},{fade_out},format=yuv420p,setsar=1",
        "outro": f"{fade_in},format=yuv420p,setsar=1",
    }
    for segment, expected_filter in expected_filters.items():
        command = harness.command_for(f"segment {segment} boundary fades")
        assert command[command.index("-vf") + 1] == expected_filter
        assert command[command.index("-c:a") + 1] == "copy"

    concat_command = harness.command_for("final concat and audio mix")
    inputs = [
        Path(concat_command[index + 1]).name
        for index, argument in enumerate(concat_command)
        if argument == "-i"
    ]
    assert inputs == [
        "hook_faded.mp4",
        "bridge_in_faded.mp4",
        "main_faded.mp4",
        "bridge_out_faded.mp4",
        "outro_faded.mp4",
        "bed_music.mp3",
    ]


def test_success_burns_overlays_only_where_text_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hook_text = "¿Sobrevivirías la hora zombi?"
    close_text = "SÍGUENOS — Época de terror É"
    harness = _harness(
        monkeypatch,
        tmp_path,
        _manifest(hook_text=hook_text, close_text=close_text),
    )

    harness.run()

    hook_command = harness.command_for("asset hook normalization")
    hook_filter = hook_command[hook_command.index("-filter_complex") + 1]
    assert "drawtext=" in hook_filter
    assert "textfile='" in hook_filter
    assert "enable='gte(t\\,0.5)'" in hook_filter

    outro_command = harness.command_for("asset outro normalization")
    outro_filter = outro_command[outro_command.index("-filter_complex") + 1]
    assert "eof_action=repeat,drawtext=" in outro_filter
    assert "enable='gte(t\\,2.5)'" in outro_filter

    for description in (
        "asset bridge_in normalization",
        "asset bridge_out normalization",
        "main caption burn",
        "silence detection",
        "silence removal",
        "segment hook boundary fades",
        "segment outro boundary fades",
        "final concat and audio mix",
    ):
        command = harness.command_for(description)
        assert not any("drawtext" in argument for argument in command)

    # Bytes written for drawtext are exactly the manifest field bytes.
    assert harness.overlay_textfile_bytes == {
        "hook_text.txt": hook_text.encode("utf-8"),
        "close_text.txt": close_text.encode("utf-8"),
    }


def test_success_without_overlay_fields_produces_no_drawtext(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())

    harness.run()

    assert not any(
        "drawtext" in argument
        for command in harness.ffmpeg_commands
        for argument in command
    )
    assert harness.overlay_textfile_bytes == {}


def test_success_with_empty_overlay_fields_produces_no_drawtext(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(
        monkeypatch,
        tmp_path,
        _manifest(hook_text="", close_text=""),
    )

    harness.run()

    assert not any(
        "drawtext" in argument
        for command in harness.ffmpeg_commands
        for argument in command
    )
    assert harness.overlay_textfile_bytes == {}


def test_manifest_rejects_non_string_overlay_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    _approved_clip(manifest)["hook_text"] = 123
    harness = _harness(monkeypatch, tmp_path, manifest)

    with pytest.raises(
        AssembleError, match=r"clips\[0\].hook_text must be a string"
    ):
        harness.run()

    assert harness.db.checkpoints == []


def test_generated_segment_command_argv(tmp_path: Path) -> None:
    source = tmp_path / "hook.mp4"
    output = tmp_path / "hook_normalized.mp4"

    assert build_generated_segment_command(
        source,
        output,
        speed=CONFIG["hook_speed"],
    ) == [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo",
        "-filter_complex",
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setpts=PTS/1.5,fps=30,format=yuv420p,setsar=1[v]",
        "-map",
        "[v]",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output),
    ]


def test_outro_segment_command_overlays_logo_argv(tmp_path: Path) -> None:
    source = tmp_path / "outro.mp4"
    logo = tmp_path / "logo.png"
    output = tmp_path / "outro_normalized.mp4"

    assert build_generated_segment_command(
        source,
        output,
        speed=CONFIG["outro_speed"],
        logo=logo,
    ) == [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-i",
        str(logo),
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo",
        "-filter_complex",
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setpts=PTS/1.5,fps=30[base];"
        "[1:v]scale=378:-1[logo];"
        "[base][logo]overlay=(W-w)/2:120:eof_action=repeat,"
        "format=yuv420p,setsar=1[v]",
        "-map",
        "[v]",
        "-map",
        "2:a",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output),
    ]


def test_generated_segment_command_burns_hook_text_argv(tmp_path: Path) -> None:
    source = tmp_path / "hook.mp4"
    output = tmp_path / "hook_normalized.mp4"
    textfile = tmp_path / "hook_text.txt"

    command = build_generated_segment_command(
        source,
        output,
        speed=CONFIG["hook_speed"],
        text=TextOverlay(textfile=textfile, style=CONFIG["hook_overlay"]),
    )

    filter_index = command.index("-filter_complex") + 1
    assert command[filter_index] == (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setpts=PTS/1.5,fps=30,"
        f"drawtext=fontfile='{CONFIG['hook_overlay']['fontfile']}':"
        f"textfile='{textfile}':"
        "fontsize=72:fontcolor=white:borderw=4:bordercolor=black:"
        "x=(w-text_w)/2:y=h*0.72:enable='gte(t\\,0.5)',"
        "format=yuv420p,setsar=1[v]"
    )
    # Everything around the filter graph is byte-identical to the v1 command.
    v1_command = build_generated_segment_command(
        source, output, speed=CONFIG["hook_speed"]
    )
    assert command[:filter_index] == v1_command[:filter_index]
    assert command[filter_index + 1 :] == v1_command[filter_index + 1 :]


def test_outro_segment_command_burns_close_text_above_logo_argv(
    tmp_path: Path,
) -> None:
    source = tmp_path / "outro.mp4"
    logo = tmp_path / "logo.png"
    output = tmp_path / "outro_normalized.mp4"
    textfile = tmp_path / "close_text.txt"

    command = build_generated_segment_command(
        source,
        output,
        speed=CONFIG["outro_speed"],
        logo=logo,
        text=TextOverlay(textfile=textfile, style=CONFIG["close_overlay"]),
    )

    filter_index = command.index("-filter_complex") + 1
    assert command[filter_index] == (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setpts=PTS/1.5,fps=30[base];"
        "[1:v]scale=378:-1[logo];"
        "[base][logo]overlay=(W-w)/2:120:eof_action=repeat,"
        f"drawtext=fontfile='{CONFIG['close_overlay']['fontfile']}':"
        f"textfile='{textfile}':"
        "fontsize=72:fontcolor=white:borderw=4:bordercolor=black:"
        "x=(w-text_w)/2:y=h*0.72:enable='gte(t\\,2.5)',"
        "format=yuv420p,setsar=1[v]"
    )


def test_overlay_textfile_preserves_manifest_bytes(tmp_path: Path) -> None:
    text = "¿Éxito con Ñ y acentos? ¡Sí!"

    overlay = assemble._write_overlay_textfile(
        tmp_path, "hook_text", text, CONFIG["hook_overlay"]
    )

    assert overlay is not None
    assert overlay.textfile == tmp_path / "hook_text.txt"
    assert overlay.textfile.read_bytes() == text.encode("utf-8")
    assert overlay.style == CONFIG["hook_overlay"]
    assert (
        assemble._write_overlay_textfile(
            tmp_path, "close_text", None, CONFIG["close_overlay"]
        )
        is None
    )


def test_boundary_fade_command_argv(tmp_path: Path) -> None:
    source = tmp_path / "segment.mp4"
    output = tmp_path / "segment_faded.mp4"

    assert build_boundary_fade_command(
        source,
        output,
        duration_s=12.5,
        fade_in=True,
        fade_out=True,
    ) == [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        "fade=t=in:st=0:d=0.25,"
        "fade=t=out:st=12.250000:d=0.25,"
        "format=yuv420p,setsar=1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
    ]

    hook_command = build_boundary_fade_command(
        source, output, duration_s=12.5, fade_in=False, fade_out=True
    )
    assert hook_command[hook_command.index("-vf") + 1] == (
        "fade=t=out:st=12.250000:d=0.25,format=yuv420p,setsar=1"
    )
    outro_command = build_boundary_fade_command(
        source, output, duration_s=12.5, fade_in=True, fade_out=False
    )
    assert outro_command[outro_command.index("-vf") + 1] == (
        "fade=t=in:st=0:d=0.25,format=yuv420p,setsar=1"
    )


def test_duration_probe_command_argv(tmp_path: Path) -> None:
    source = tmp_path / "segment.mp4"

    assert build_duration_probe_command(source) == [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]


def test_main_caption_command_argv(tmp_path: Path) -> None:
    source = tmp_path / "main.mp4"
    srt_path = tmp_path / "clip.srt"
    output = tmp_path / "main_captioned.mp4"

    expected_filter = (
        "[0:v]split=2[background][foreground];"
        "[background]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=luma_radius=30:luma_power=1[blurred];"
        "[foreground]scale=1080:1920:"
        "force_original_aspect_ratio=decrease[fit];"
        "[blurred][fit]overlay=(W-w)/2:(H-h)/2[laidout];"
        f"[laidout]subtitles=filename='{srt_path}':"
        f"fontsdir='{assemble._escape_filter_value(str(assemble.FONTS_DIR))}':"
        "force_style='FontName=Montserrat ExtraBold,Bold=0,FontSize=16,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,"
        "Alignment=2,MarginV=50',setpts=PTS/1.0,fps=30,"
        "format=yuv420p,setsar=1[v];"
        "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,"
        "volume=0.68[a]"
    )
    assert build_main_caption_command(source, srt_path, output) == [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        expected_filter,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output),
    ]


def test_captions_burn_onto_composited_frame_before_silence_removal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())

    harness.run()

    # Subtitles are burned onto the composited 1080x1920 canvas, after the
    # blur overlay, so force_style coordinates are full-frame.
    caption_command = harness.command_for("main caption burn")
    caption_filter = caption_command[
        caption_command.index("-filter_complex") + 1
    ]
    composite_index = caption_filter.index(
        "overlay=(W-w)/2:(H-h)/2[laidout]"
    )
    subtitles_index = caption_filter.index("[laidout]subtitles=")
    assert composite_index < subtitles_index

    # The burn stays in the pass before silence removal: SRT sync depends on
    # captioning an untrimmed timeline.
    descriptions = harness.ffmpeg_descriptions
    assert descriptions.index(
        f"clip {CLIP_ID} main caption burn"
    ) < descriptions.index(f"clip {CLIP_ID} silence detection")
    for description in (
        f"clip {CLIP_ID} silence detection",
        f"clip {CLIP_ID} silence removal",
    ):
        command = harness.command_for(description.removeprefix(f"clip {CLIP_ID} "))
        assert not any("subtitles" in argument for argument in command)


def test_caption_style_values_flow_from_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(CONFIG, "caption_alignment", 8)
    monkeypatch.setitem(CONFIG, "caption_margin_v", 77)
    monkeypatch.setitem(CONFIG, "caption_font_size", 21)

    command = build_main_caption_command(
        tmp_path / "main.mp4",
        tmp_path / "clip.srt",
        tmp_path / "main_captioned.mp4",
    )

    style_filter = command[command.index("-filter_complex") + 1]
    assert "Alignment=8" in style_filter
    assert "MarginV=77" in style_filter
    assert "FontSize=21" in style_filter


def test_silence_commands_argv(tmp_path: Path) -> None:
    source = tmp_path / "main_captioned.mp4"
    output = tmp_path / "main_trimmed.mp4"

    assert build_silence_detect_command(source) == [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-af",
        "silencedetect=noise=-35.0dB:d=0.8",
        "-f",
        "null",
        "-",
    ]
    assert build_silence_cut_command(
        source,
        output,
        [SilenceInterval(start_s=1.0, end_s=3.0)],
    ) == [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        "[0:v]select='not(between(t\\,1.150000\\,2.850000))',"
        "setpts=N/(30*TB),format=yuv420p,setsar=1[v];"
        "[0:a]aselect='not(between(t\\,1.150000\\,2.850000))',"
        "asetpts=N/SR/TB[a]",
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output),
    ]


def test_all_five_concat_segments_are_normalized_to_square_pixels(
    tmp_path: Path,
) -> None:
    logo = tmp_path / "logo.png"
    generated_commands = [
        build_generated_segment_command(
            tmp_path / "hook.mp4",
            tmp_path / "hook_normalized.mp4",
            speed=CONFIG["hook_speed"],
        ),
        build_generated_segment_command(
            tmp_path / "bridge_in.mp4",
            tmp_path / "bridge_in_normalized.mp4",
            speed=CONFIG["bridge_speed"],
        ),
        build_generated_segment_command(
            tmp_path / "bridge_out.mp4",
            tmp_path / "bridge_out_normalized.mp4",
            speed=CONFIG["bridge_speed"],
        ),
        build_generated_segment_command(
            tmp_path / "outro.mp4",
            tmp_path / "outro_normalized.mp4",
            speed=CONFIG["outro_speed"],
            logo=logo,
        ),
    ]
    main_caption_command = build_main_caption_command(
        tmp_path / "main.mp4",
        tmp_path / "clip.srt",
        tmp_path / "main_captioned.mp4",
    )
    main_trim_command = build_silence_cut_command(
        tmp_path / "main_captioned.mp4",
        tmp_path / "main_trimmed.mp4",
        (),
    )

    for command in (*generated_commands, main_caption_command, main_trim_command):
        filter_index = command.index("-filter_complex") + 1
        assert "setsar=1" in command[filter_index]


def test_final_concat_and_bed_mix_command_argv(tmp_path: Path) -> None:
    segments = tuple(tmp_path / f"segment-{index}.mp4" for index in range(5))
    bed = tmp_path / "bed.mp3"
    output = tmp_path / "final.mp4"

    expected_filter = (
        "[0:v]setpts=PTS-STARTPTS[v0];"
        "[0:a]asetpts=PTS-STARTPTS[a0];"
        "[1:v]setpts=PTS-STARTPTS[v1];"
        "[1:a]asetpts=PTS-STARTPTS[a1];"
        "[2:v]setpts=PTS-STARTPTS[v2];"
        "[2:a]asetpts=PTS-STARTPTS[a2];"
        "[3:v]setpts=PTS-STARTPTS[v3];"
        "[3:a]asetpts=PTS-STARTPTS[a3];"
        "[4:v]setpts=PTS-STARTPTS[v4];"
        "[4:a]asetpts=PTS-STARTPTS[a4];"
        "[v0][a0][v1][a1][v2][a2][v3][a3][v4][a4]"
        "concat=n=5:v=1:a=1[video][voice];"
        "[5:a]aformat=sample_rates=48000:channel_layouts=stereo,"
        "volume=0.02[bed];"
        "[voice][bed]amix=inputs=2:duration=first:normalize=0:"
        "dropout_transition=0,areverse,afade=t=in:d=1.5,"
        "areverse,asetpts=N/SR/TB[audio]"
    )
    expected = ["ffmpeg", "-y"]
    for segment in segments:
        expected.extend(["-i", str(segment)])
    expected.extend(
        [
            "-stream_loop",
            "-1",
            "-i",
            str(bed),
            "-filter_complex",
            expected_filter,
            "-map",
            "[video]",
            "-map",
            "[audio]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )

    assert build_final_command(segments, bed, output) == expected


def test_ffmpeg_failure_preserves_stderr_tail_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    stderr = ("x" * 2_000) + "distinct stderr tail"

    def failed_ffmpeg(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 9, stdout="", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", failed_ffmpeg)

    with pytest.raises(AssembleError) as raised:
        assemble._run_ffmpeg(
            ["ffmpeg", "-i", "broken.mp4"],
            description="clip clip_04 final concat",
        )

    assert calls == [["ffmpeg", "-i", "broken.mp4"]]
    assert "ffmpeg exited 9" in str(raised.value)
    assert "distinct stderr tail" in str(raised.value)
