from __future__ import annotations

import io
import json
import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest

from worker import higgsfield
from worker.higgsfield import (
    GenerationParams,
    HiggsfieldDownloadError,
    HiggsfieldError,
)

GENERATION_ID = "generation-123"
RESULT_URL = "https://cdn.example.test/result.mp4"


def _completed(
    stdout: str,
    *,
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["higgsfield"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _install_subprocess_responses(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[subprocess.CompletedProcess[str]],
) -> list[list[str]]:
    queued: Iterator[subprocess.CompletedProcess[str]] = iter(responses)
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return next(queued)

    monkeypatch.setattr("worker.higgsfield.subprocess.run", fake_run)
    return calls


def _install_download(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    def fake_urlopen(_url: str, *, timeout: float) -> io.BytesIO:
        assert timeout == higgsfield.DOWNLOAD_TIMEOUT_S
        return io.BytesIO(payload)

    monkeypatch.setattr("worker.higgsfield.urllib.request.urlopen", fake_urlopen)


def _no_sleep(_seconds: float) -> None:
    return None


def _params() -> GenerationParams:
    return GenerationParams(
        prompt="cinematic market chart",
        duration_s=4,
    )


def test_balance_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_subprocess_responses(
        monkeypatch,
        [_completed('{"credits": 2048}')],
    )

    assert higgsfield.balance() == 2048
    assert calls == [["higgsfield", "account", "status", "--json"]]


def test_get_cost_happy_path_with_full_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_subprocess_responses(
        monkeypatch,
        [_completed('{"credits": 36}')],
    )
    start = tmp_path / "start.png"
    end = tmp_path / "end.png"

    cost = higgsfield.get_cost(
        GenerationParams(
            prompt="cinematic market chart",
            duration_s=4,
            generate_audio=True,
            start_image=start,
            end_image=end,
        )
    )

    assert cost == 36
    command = calls[0]
    assert command[:4] == ["higgsfield", "generate", "cost", "seedance_2_0"]
    assert command[command.index("--duration") + 1] == "4"
    assert command[command.index("--aspect-ratio") + 1] == "9:16"
    assert command[command.index("--resolution") + 1] == "1080p"
    assert command[command.index("--generate-audio") + 1] == "true"
    assert command[command.index("--start-image") + 1] == str(start)
    assert command[command.index("--end-image") + 1] == str(end)
    assert command[-1] == "--json"


def test_generate_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_subprocess_responses(
        monkeypatch,
        [
            _completed(json.dumps({"id": GENERATION_ID})),
            _completed(
                json.dumps(
                    {
                        "id": GENERATION_ID,
                        "status": "completed",
                        "result_url": RESULT_URL,
                    }
                )
            ),
        ],
    )
    _install_download(monkeypatch, b"mp4 bytes")
    output = tmp_path / "video.mp4"

    result = higgsfield.generate(_params(), output)

    assert output.read_bytes() == b"mp4 bytes"
    assert result.id == GENERATION_ID
    assert result.result_url == RESULT_URL
    assert result.credits_charged is None
    assert result.elapsed_s >= 0


def test_public_download_writes_non_empty_file_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_download(monkeypatch, b"remote mp4")
    output = tmp_path / "nested" / "video.mp4"

    higgsfield.download(RESULT_URL, output)

    assert output.read_bytes() == b"remote mp4"
    assert list(output.parent.iterdir()) == [output]


def test_public_download_failure_carries_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def failed_urlopen(_url: str, *, timeout: float) -> io.BytesIO:
        raise OSError(f"download failed after {timeout}s")

    monkeypatch.setattr(
        "worker.higgsfield.urllib.request.urlopen",
        failed_urlopen,
    )
    output = tmp_path / "video.mp4"

    with pytest.raises(HiggsfieldDownloadError) as raised:
        higgsfield.download(RESULT_URL, output)

    assert raised.value.generation_id is None
    assert raised.value.result_url == RESULT_URL
    assert not output.exists()


def test_cost_exit_zero_with_non_json_stdout_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_subprocess_responses(monkeypatch, [_completed("Usage: help text")])

    with pytest.raises(HiggsfieldError) as raised:
        higgsfield.get_cost(_params())

    assert raised.value.exit_code == 0
    assert "Usage: help text" in raised.value.stdout_tail


def test_submit_exit_zero_with_non_json_stdout_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_subprocess_responses(monkeypatch, [_completed("Usage: help text")])

    with pytest.raises(HiggsfieldError) as raised:
        higgsfield.generate(_params(), tmp_path / "video.mp4")

    assert raised.value.exit_code == 0
    assert "Usage: help text" in raised.value.stdout_tail


def test_poll_failure_status_raises_immediately_with_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_subprocess_responses(
        monkeypatch,
        [
            _completed(json.dumps({"id": GENERATION_ID})),
            _completed(
                json.dumps({"id": GENERATION_ID, "status": "failed"})
            ),
        ],
    )

    with pytest.raises(HiggsfieldError, match=GENERATION_ID):
        higgsfield.generate(_params(), tmp_path / "video.mp4")

    assert len(calls) == 2


def test_unknown_status_warns_and_polling_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = _install_subprocess_responses(
        monkeypatch,
        [
            _completed(json.dumps({"id": GENERATION_ID})),
            _completed(
                json.dumps({"id": GENERATION_ID, "status": "transcoding"})
            ),
            _completed(
                json.dumps(
                    {
                        "id": GENERATION_ID,
                        "status": "completed",
                        "result_url": RESULT_URL,
                    }
                )
            ),
        ],
    )
    _install_download(monkeypatch, b"video")
    monkeypatch.setattr("worker.higgsfield.time.sleep", _no_sleep)

    with caplog.at_level(logging.WARNING):
        result = higgsfield.generate(_params(), tmp_path / "video.mp4")

    assert result.id == GENERATION_ID
    assert len(calls) == 3
    assert "unrecognized status" in caplog.text
    assert "transcoding" in caplog.text


def test_generation_timeout_raises_with_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_subprocess_responses(
        monkeypatch,
        [
            _completed(json.dumps({"id": GENERATION_ID})),
            _completed(json.dumps({"id": GENERATION_ID, "status": "running"})),
        ],
    )
    times = iter([0.0, 0.0, 1.1])
    monkeypatch.setattr(higgsfield, "GENERATION_TIMEOUT_S", 1.0)
    monkeypatch.setattr(
        "worker.higgsfield.time.monotonic",
        lambda: next(times),
    )
    monkeypatch.setattr("worker.higgsfield.time.sleep", _no_sleep)

    with pytest.raises(HiggsfieldError, match=GENERATION_ID):
        higgsfield.generate(_params(), tmp_path / "video.mp4")


def test_zero_byte_download_raises_distinct_error_and_leaves_no_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_subprocess_responses(
        monkeypatch,
        [
            _completed(json.dumps({"id": GENERATION_ID})),
            _completed(
                json.dumps(
                    {
                        "id": GENERATION_ID,
                        "status": "completed",
                        "result_url": RESULT_URL,
                    }
                )
            ),
        ],
    )
    _install_download(monkeypatch, b"")
    output = tmp_path / "video.mp4"

    with pytest.raises(HiggsfieldDownloadError):
        higgsfield.generate(_params(), output)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_download_failure_carries_generation_id_and_result_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_subprocess_responses(
        monkeypatch,
        [
            _completed(json.dumps({"id": GENERATION_ID})),
            _completed(
                json.dumps(
                    {
                        "id": GENERATION_ID,
                        "status": "completed",
                        "result_url": RESULT_URL,
                    }
                )
            ),
        ],
    )

    def failed_urlopen(_url: str, *, timeout: float) -> io.BytesIO:
        raise OSError(f"download failed after {timeout}s")

    monkeypatch.setattr(
        "worker.higgsfield.urllib.request.urlopen",
        failed_urlopen,
    )

    with pytest.raises(HiggsfieldDownloadError) as raised:
        higgsfield.generate(_params(), tmp_path / "video.mp4")

    assert raised.value.generation_id == GENERATION_ID
    assert raised.value.result_url == RESULT_URL


def test_duration_validation_precedes_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock()
    monkeypatch.setattr("worker.higgsfield.subprocess.run", run)

    with pytest.raises(ValueError, match="at least 4"):
        GenerationParams(prompt="test", duration_s=3)

    run.assert_not_called()


def test_image_pair_validation_precedes_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = Mock()
    monkeypatch.setattr("worker.higgsfield.subprocess.run", run)

    with pytest.raises(ValueError, match="provided together"):
        GenerationParams(
            prompt="test",
            duration_s=4,
            start_image=tmp_path / "start.png",
        )

    run.assert_not_called()


def test_cli_nonzero_exit_carries_code_and_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_subprocess_responses(
        monkeypatch,
        [_completed("", returncode=3, stderr="API rejected request")],
    )

    with pytest.raises(HiggsfieldError) as raised:
        higgsfield.balance()

    assert raised.value.exit_code == 3
    assert raised.value.stderr_tail == "API rejected request"


def test_subprocess_timeout_raises_higgsfield_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timed_out(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(
            command,
            timeout=1,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr("worker.higgsfield.subprocess.run", timed_out)

    with pytest.raises(HiggsfieldError) as raised:
        higgsfield.balance()

    assert raised.value.exit_code is None
    assert raised.value.stdout_tail == "partial stdout"
    assert raised.value.stderr_tail == "partial stderr"


def test_malformed_structured_cost_output_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_subprocess_responses(
        monkeypatch,
        [_completed('{"credits": "thirty-six"}')],
    )

    with pytest.raises(HiggsfieldError, match="must be an integer"):
        higgsfield.get_cost(_params())
