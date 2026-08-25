from __future__ import annotations

import copy
import logging
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from mypy_boto3_s3.client import S3Client
from psycopg import Connection
from psycopg.rows import DictRow

from worker import jobs
from worker.jobs import Job
from worker.library import LibraryCatalog
from worker.stages import assemble
from worker.stages.assemble import (
    AssembleError,
    AssembleStageContext,
    CONFIG,
    SilenceInterval,
    TextOverlay,
    build_clip_srt,
    build_duration_probe_command,
    build_final_command,
    build_library_segment_command,
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
# Probed durations of the normalized segment FILES (never catalog nominals):
# hook 4.04, main 135.2, outro 5.04 -> offsets 3.79 / 138.49, total 143.53.
HOOK_DUR = 4.04
MAIN_DUR = 135.2
OUTRO_DUR = 5.04
SEGMENT_DURATIONS = {
    "hook_normalized.mp4": HOOK_DUR,
    "main_trimmed.mp4": MAIN_DUR,
    "outro_normalized.mp4": OUTRO_DUR,
}
LIBRARY_KEYS = {
    "H01": "library/hooks/H01.mp4",
    "H02": "library/hooks/H02.mp4",
    "O01": "library/outros/O01.mp4",  # logo_baked True
    "O02": "library/outros/O02.mp4",  # logo_baked False
}
TEXTFILE_RE = re.compile(r"textfile='([^']+)'")
# Any video dip-to-black: fade=t=... NOT preceded by 'a' (the audio end fade
# afade=t=in must survive; xfade=transition=fade never matches this pattern).
VIDEO_FADE_RE = re.compile(r"(?<!a)fade=t=")


def _test_catalog() -> LibraryCatalog:
    def asset(
        asset_id: str, asset_type: str, *, logo_baked: bool = False
    ) -> dict[str, Any]:
        return {
            "id": asset_id,
            "type": asset_type,
            "s3_key": LIBRARY_KEYS[asset_id],
            "duration_s": 4.0 if asset_type == "hook" else 5.0,
            "category": ["mindset"],
            "tags": ["psychology"],
            "character": None,
            "description": f"{asset_id} test asset",
            "times_used": 0,
            "logo_baked": logo_baked,
        }

    return LibraryCatalog.from_dict(
        {
            "version": 1,
            "updated_at": "2026-08-25T00:00:00Z",
            "notes": "test notes",
            "assets": [
                asset("H01", "hook"),
                asset("H02", "hook"),
                asset("O01", "outro", logo_baked=True),
                asset("O02", "outro", logo_baked=False),
            ],
        }
    )


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


def _clip(
    clip_id: str = CLIP_ID,
    *,
    assembled: bool = False,
    hook_text: str | None = None,
    close_text: str | None = None,
    hook_asset_id: str = "H01",
    outro_asset_id: str = "O01",
) -> dict[str, object]:
    clip: dict[str, object] = {
        "id": clip_id,
        "approved": True,
        "start": "00:00:10,000",
        "end": "00:00:12,500",
        "start_block": 1,
        "end_block": 2,
        "duration_seconds": 2.5,
        "transcript": "flat text is deliberately unused",
        "hook_asset_id": hook_asset_id,
        "outro_asset_id": outro_asset_id,
    }
    if hook_text is not None:
        clip["hook_text"] = hook_text
    if close_text is not None:
        clip["close_text"] = close_text
    if assembled:
        clip["assembled"] = {
            "s3Key": f"pipeline/{JOB_ID}/final/final_{clip_id}_9x16.mp4",
            "completedAt": "2026-07-16T12:00:00+00:00",
        }
    return clip


def _manifest_for(clips: list[dict[str, object]]) -> dict[str, object]:
    return {
        "srt_file": MASTER_SRT_KEY,
        "source_video": f"pipeline/{JOB_ID}/session_bleeped.mp4",
        "status": "approved",
        "clips": clips,
        "generated": "2026-07-16",
        "rejected_segments": [],
    }


def _manifest(**clip_kwargs: Any) -> dict[str, object]:
    return _manifest_for([_clip(**clip_kwargs)])


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
        }
    )
    for key in LIBRARY_KEYS.values():
        fake_s3.objects[key] = b"library"
    raw_clips = manifest["clips"]
    assert isinstance(raw_clips, list)
    for raw_clip in raw_clips:
        assert isinstance(raw_clip, dict)
        fake_s3.objects[f"pipeline/{JOB_ID}/clips/{raw_clip['id']}.mp4"] = b"main"

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
    monkeypatch.setattr(assemble, "_load_catalog", lambda ctx: _test_catalog())

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
        return SEGMENT_DURATIONS[source.name]

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


# --------------------------------------------------------------------------- #
# SRT slicing
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Preflight, catalog resolution, and validation
# --------------------------------------------------------------------------- #


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


def test_preflight_names_missing_logo_only_when_needed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # O02 is the only library outro without a baked logo -> the logo becomes a
    # hard requirement again.
    harness = _harness(monkeypatch, tmp_path, _manifest(outro_asset_id="O02"))
    del harness.s3.objects[CONFIG["logo_key"]]

    with pytest.raises(AssembleError, match=CONFIG["logo_key"]):
        harness.run()

    assert harness.db.checkpoints == []


def test_logo_skipped_when_outro_logo_baked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # O01 has logo_baked=True: the logo object may not even exist and the run
    # must neither HEAD nor download it, and the outro gets no overlay input.
    harness = _harness(monkeypatch, tmp_path, _manifest())
    del harness.s3.objects[CONFIG["logo_key"]]

    harness.run()

    assert CONFIG["logo_key"] not in harness.s3.heads
    assert CONFIG["logo_key"] not in harness.s3.downloads
    outro_command = harness.command_for("asset outro normalization")
    assert outro_command.count("-i") == 1
    outro_filter = outro_command[outro_command.index("-filter_complex") + 1]
    assert "overlay=" not in outro_filter


def test_logo_overlaid_when_outro_not_logo_baked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest(outro_asset_id="O02"))

    harness.run()

    assert CONFIG["logo_key"] in harness.s3.heads
    assert CONFIG["logo_key"] in harness.s3.downloads
    outro_command = harness.command_for("asset outro normalization")
    assert outro_command.count("-i") == 2
    outro_filter = outro_command[outro_command.index("-filter_complex") + 1]
    assert "[base][logo]overlay=(W-w)/2:120:eof_action=repeat" in outro_filter


def test_preflight_names_missing_main_clip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())
    del harness.s3.objects[MAIN_CLIP_KEY]

    with pytest.raises(AssembleError, match=MAIN_CLIP_KEY):
        harness.run()

    assert harness.db.checkpoints == []


def test_unknown_asset_id_fails_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest(hook_asset_id="H99"))

    with pytest.raises(
        AssembleError,
        match="asset id 'H99' is not in the library catalog",
    ):
        harness.run()

    # Resolution happens before preflight/downloads: nothing was touched.
    assert harness.s3.heads == []
    assert harness.s3.downloads == []
    assert harness.ffmpeg_commands == []
    assert harness.db.checkpoints == []


def test_wrong_type_asset_id_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest(hook_asset_id="O01"))

    with pytest.raises(
        AssembleError, match="has type 'outro', expected 'hook'"
    ):
        harness.run()

    assert harness.ffmpeg_commands == []


def test_manifest_requires_selection_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    del _approved_clip(manifest)["hook_asset_id"]
    harness = _harness(monkeypatch, tmp_path, manifest)

    with pytest.raises(
        AssembleError,
        match=r"clips\[0\].hook_asset_id must be a non-empty string",
    ):
        harness.run()

    assert harness.db.checkpoints == []


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


# --------------------------------------------------------------------------- #
# run_assemble end-to-end (mocked ffmpeg/probe)
# --------------------------------------------------------------------------- #


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
    # No library asset is fetched for a clip that is already assembled.
    assert not set(LIBRARY_KEYS.values()) & set(harness.s3.downloads)
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
    assert len(harness.ffmpeg_commands) == 6
    assert harness.ffmpeg_descriptions == [
        f"clip {CLIP_ID} asset hook normalization",
        f"clip {CLIP_ID} asset outro normalization",
        f"clip {CLIP_ID} main caption burn",
        f"clip {CLIP_ID} silence detection",
        f"clip {CLIP_ID} silence removal",
        f"clip {CLIP_ID} final xfade and audio mix",
    ]
    # Durations for offsets come from the normalized files, in order.
    assert harness.probed_segments == [
        "hook_normalized.mp4",
        "main_trimmed.mp4",
        "outro_normalized.mp4",
    ]
    assert not harness.context.workspace.dir.exists()


def test_final_command_xfade_math_from_probed_durations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())

    harness.run()

    final = harness.command_for("final xfade and audio mix")
    graph = final[final.index("-filter_complex") + 1]
    # offset1 = 4.04 - 0.25; offset2 = 4.04 + 135.2 - 0.75
    assert (
        "[v0][v1]xfade=transition=fade:duration=0.25:offset=3.790000[vx]"
        in graph
    )
    assert (
        "[vx][v2]xfade=transition=fade:duration=0.5:offset=138.490000[video]"
        in graph
    )
    # Voice starts at the hook->main crossfade: (4.04 - 0.25) * 1000 ms.
    assert "[1:a]adelay=3790|3790,apad[voice]" in graph
    # Bed and end fade are unchanged from the concat era.
    assert "volume=0.02[bed]" in graph
    assert "areverse,afade=t=in:d=1.5,areverse" in graph
    # -t clamps to hook + main + outro - 0.75.
    assert final[final.index("-t") + 1] == "143.530000"
    inputs = [
        Path(final[index + 1]).name
        for index, argument in enumerate(final)
        if argument == "-i"
    ]
    assert inputs == [
        "hook_normalized.mp4",
        "main_trimmed.mp4",
        "outro_normalized.mp4",
        "bed_music.mp3",
    ]


def test_settb_on_every_xfade_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())

    harness.run()

    # In the final graph, each xfade input is timebase-normalized explicitly.
    final = harness.command_for("final xfade and audio mix")
    graph = final[final.index("-filter_complex") + 1]
    for chain in ("[0:v]settb=AVTB[v0]", "[1:v]settb=AVTB[v1]", "[2:v]settb=AVTB[v2]"):
        assert chain in graph

    # And every segment-producing pass already normalizes its output timebase.
    for suffix in (
        "asset hook normalization",
        "asset outro normalization",
        "main caption burn",
        "silence removal",
    ):
        command = harness.command_for(suffix)
        filter_string = command[command.index("-filter_complex") + 1]
        assert "setsar=1,settb=AVTB[v]" in filter_string


def test_no_video_fades_anywhere(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _harness(monkeypatch, tmp_path, _manifest())

    harness.run()

    for command in harness.ffmpeg_commands:
        for argument in command:
            assert not VIDEO_FADE_RE.search(argument), argument
    # Sanity: the joins really are crossfades.
    final = harness.command_for("final xfade and audio mix")
    graph = final[final.index("-filter_complex") + 1]
    assert graph.count("xfade=transition=fade") == 2


def test_library_assets_download_once_per_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Two clips selecting the SAME hook and outro: each library asset is
    # downloaded exactly once for the whole job.
    manifest = _manifest_for([_clip("clip_04"), _clip("clip_05")])
    harness = _harness(monkeypatch, tmp_path, manifest)

    harness.run()

    assert harness.s3.downloads.count(LIBRARY_KEYS["H01"]) == 1
    assert harness.s3.downloads.count(LIBRARY_KEYS["O01"]) == 1
    assert f"pipeline/{JOB_ID}/clips/clip_04.mp4" in harness.s3.downloads
    assert f"pipeline/{JOB_ID}/clips/clip_05.mp4" in harness.s3.downloads
    assert len(harness.ffmpeg_commands) == 12
    assert len(harness.db.checkpoints) == 2
    assert harness.s3.uploads == [
        f"pipeline/{JOB_ID}/final/final_clip_04_9x16.mp4",
        f"pipeline/{JOB_ID}/final/final_clip_05_9x16.mp4",
        f"pipeline/{JOB_ID}/manifest.json",
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
    assert "drawtext=" in outro_filter
    assert "enable='gte(t\\,2.5)'" in outro_filter

    for description in (
        "main caption burn",
        "silence detection",
        "silence removal",
        "final xfade and audio mix",
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


# --------------------------------------------------------------------------- #
# Command builders (pure argv tests)
# --------------------------------------------------------------------------- #


def test_library_segment_command_argv(tmp_path: Path) -> None:
    source = tmp_path / "H01.mp4"
    output = tmp_path / "hook_normalized.mp4"

    assert build_library_segment_command(source, output) == [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
        "fps=30,format=yuv420p,setsar=1,settb=AVTB[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def test_library_segment_command_with_logo_argv(tmp_path: Path) -> None:
    source = tmp_path / "O02.mp4"
    logo = tmp_path / "logo.png"
    output = tmp_path / "outro_normalized.mp4"

    assert build_library_segment_command(source, output, logo=logo) == [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-i",
        str(logo),
        "-filter_complex",
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
        "fps=30[base];"
        "[1:v]scale=378:-1[logo];"
        "[base][logo]overlay=(W-w)/2:120:eof_action=repeat,"
        "format=yuv420p,setsar=1,settb=AVTB[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def test_library_segment_command_burns_hook_text_argv(tmp_path: Path) -> None:
    source = tmp_path / "H01.mp4"
    output = tmp_path / "hook_normalized.mp4"
    textfile = tmp_path / "hook_text.txt"

    command = build_library_segment_command(
        source,
        output,
        text=TextOverlay(textfile=textfile, style=CONFIG["hook_overlay"]),
    )

    filter_index = command.index("-filter_complex") + 1
    assert command[filter_index] == (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
        "fps=30,"
        f"drawtext=fontfile='{CONFIG['hook_overlay']['fontfile']}':"
        f"textfile='{textfile}':"
        "expansion=none:"
        "fontsize=72:fontcolor=white:borderw=4:bordercolor=black:"
        "x=(w-text_w)/2:"
        "y='h*0.72+30*(1-min(1\\,(t-0.5)/0.4))':"
        "alpha='min(1\\,max(0\\,(t-0.5)/0.4))':"
        "enable='gte(t\\,0.5)',"
        "format=yuv420p,setsar=1,settb=AVTB[v]"
    )
    # No speed step: library assets are authored at final pacing.
    assert "setpts=PTS/" not in command[filter_index]
    # Everything around the filter graph is byte-identical to the plain command.
    plain = build_library_segment_command(source, output)
    assert command[:filter_index] == plain[:filter_index]
    assert command[filter_index + 1 :] == plain[filter_index + 1 :]


def test_library_segment_command_burns_close_text_above_logo_argv(
    tmp_path: Path,
) -> None:
    source = tmp_path / "O02.mp4"
    logo = tmp_path / "logo.png"
    output = tmp_path / "outro_normalized.mp4"
    textfile = tmp_path / "close_text.txt"

    command = build_library_segment_command(
        source,
        output,
        logo=logo,
        text=TextOverlay(textfile=textfile, style=CONFIG["close_overlay"]),
    )

    filter_index = command.index("-filter_complex") + 1
    assert command[filter_index] == (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
        "fps=30[base];"
        "[1:v]scale=378:-1[logo];"
        "[base][logo]overlay=(W-w)/2:120:eof_action=repeat,"
        f"drawtext=fontfile='{CONFIG['close_overlay']['fontfile']}':"
        f"textfile='{textfile}':"
        "expansion=none:"
        "fontsize=72:fontcolor=white:borderw=4:bordercolor=black:"
        "x=(w-text_w)/2:"
        "y='h*0.72+30*(1-min(1\\,(t-2.5)/0.4))':"
        "alpha='min(1\\,max(0\\,(t-2.5)/0.4))':"
        "enable='gte(t\\,2.5)',"
        "format=yuv420p,setsar=1,settb=AVTB[v]"
    )
    assert "expansion=none" in command[filter_index]


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


def test_hook_text_with_percent_renders_literally(tmp_path: Path) -> None:
    # Regression for the production hook_text 'PUEDES PERDER 80%': drawtext's
    # default expansion=normal reads a bare '%' as a malformed %{...} token and
    # silently renders an EMPTY frame (ffmpeg exits 0, no stderr). The builder
    # now emits expansion=none, and the text still comes from textfile= so the
    # '%' never reaches the filter-graph parser.
    text = "PUEDES PERDER 80%"

    overlay = assemble._write_overlay_textfile(
        tmp_path, "hook_text", text, CONFIG["hook_overlay"]
    )
    assert overlay is not None
    # The literal text -- including the bare '%' -- is written unmodified.
    assert overlay.textfile.read_bytes() == text.encode("utf-8")
    assert overlay.textfile.read_bytes() == b"PUEDES PERDER 80%"

    command = build_library_segment_command(
        tmp_path / "H01.mp4",
        tmp_path / "hook_normalized.mp4",
        text=overlay,
    )
    filter_string = command[command.index("-filter_complex") + 1]
    assert "expansion=none" in filter_string
    # Text stays in the file, not inlined into the filter graph.
    assert f"textfile='{overlay.textfile}'" in filter_string
    assert "80%" not in filter_string


def test_measure_overlay_width_matches_known_glyph() -> None:
    # Montserrat ExtraBold 'W' advance at fontsize 72 is ~85.25px (fontTools).
    style = CONFIG["close_overlay"]
    width = assemble._measure_overlay_width_px("W", style["fontfile"], 72)
    assert abs(width - 85.25) <= 0.5


def test_close_overlay_shrinks_to_fit_frame() -> None:
    # The confirmed overflow specimen in Montserrat ExtraBold: must shrink below
    # 72 and the rendered width (incl. border) must fit the 972px usable frame.
    text = "MOLDEA TU ZONA DE CONFORT"
    style = CONFIG["close_overlay"]

    fontsize = assemble._fit_overlay_fontsize(text, style)

    assert 40 <= fontsize < style["fontsize"]
    rendered = (
        assemble._measure_overlay_width_px(text, style["fontfile"], fontsize)
        + 2 * style["borderw"]
    )
    assert rendered <= 972


def test_hook_overlay_specimen_fits_without_shrink() -> None:
    # The same string in Anton (condensed) fits at the base size -- no shrink.
    text = "MOLDEA TU ZONA DE CONFORT"
    style = CONFIG["hook_overlay"]

    assert assemble._fit_overlay_fontsize(text, style) == style["fontsize"]


def test_short_overlay_keeps_base_fontsize() -> None:
    for style in (CONFIG["hook_overlay"], CONFIG["close_overlay"]):
        assert assemble._fit_overlay_fontsize("GG", style) == style["fontsize"]


def test_pathological_overlay_floors_at_min_fontsize(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    # A 60-char all-'W' string can't fit even at the floor; the assembler must
    # clamp to fontsize 40 and WARN rather than clip silently.
    text = "W" * 60
    style = CONFIG["close_overlay"]

    assert assemble._fit_overlay_fontsize(text, style) == 40

    with caplog.at_level(logging.WARNING):
        overlay = assemble._prepare_overlay(
            tmp_path,
            job_id="job-x",
            clip_id="clip_03",
            asset="outro",
            field="close_text",
            text=text,
            base_style=style,
        )

    assert overlay is not None
    assert overlay.style["fontsize"] == 40
    assert overlay.textfile.read_bytes() == text.encode("utf-8")
    assert any(
        r.levelno == logging.WARNING and "STILL overflows" in r.getMessage()
        for r in caplog.records
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
        "format=yuv420p,setsar=1,settb=AVTB[v];"
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
        "setpts=N/(30*TB),format=yuv420p,setsar=1,settb=AVTB[v];"
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


def test_all_three_segments_are_normalized_for_xfade(tmp_path: Path) -> None:
    logo = tmp_path / "logo.png"
    commands = [
        build_library_segment_command(
            tmp_path / "H01.mp4",
            tmp_path / "hook_normalized.mp4",
        ),
        build_library_segment_command(
            tmp_path / "O02.mp4",
            tmp_path / "outro_normalized.mp4",
            logo=logo,
        ),
        build_main_caption_command(
            tmp_path / "main.mp4",
            tmp_path / "clip.srt",
            tmp_path / "main_captioned.mp4",
        ),
        build_silence_cut_command(
            tmp_path / "main_captioned.mp4",
            tmp_path / "main_trimmed.mp4",
            (),
        ),
    ]

    for command in commands:
        filter_index = command.index("-filter_complex") + 1
        assert "setsar=1" in command[filter_index]
        assert "settb=AVTB" in command[filter_index]


def test_final_command_argv(tmp_path: Path) -> None:
    hook = tmp_path / "hook_normalized.mp4"
    main = tmp_path / "main_trimmed.mp4"
    outro = tmp_path / "outro_normalized.mp4"
    bed = tmp_path / "bed.mp3"
    output = tmp_path / "final.mp4"

    expected_filter = (
        "[0:v]settb=AVTB[v0];"
        "[1:v]settb=AVTB[v1];"
        "[2:v]settb=AVTB[v2];"
        "[v0][v1]xfade=transition=fade:duration=0.25:offset=3.790000[vx];"
        "[vx][v2]xfade=transition=fade:duration=0.5:offset=138.490000[video];"
        "[1:a]adelay=3790|3790,apad[voice];"
        "[3:a]aformat=sample_rates=48000:channel_layouts=stereo,"
        "volume=0.02[bed];"
        "[voice][bed]amix=inputs=2:duration=first:normalize=0:"
        "dropout_transition=0,areverse,afade=t=in:d=1.5,"
        "areverse,asetpts=N/SR/TB[audio]"
    )
    assert build_final_command(
        hook,
        main,
        outro,
        bed,
        output,
        hook_duration_s=HOOK_DUR,
        main_duration_s=MAIN_DUR,
        outro_duration_s=OUTRO_DUR,
    ) == [
        "ffmpeg",
        "-y",
        "-i",
        str(hook),
        "-i",
        str(main),
        "-i",
        str(outro),
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
        "-t",
        "143.530000",
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
