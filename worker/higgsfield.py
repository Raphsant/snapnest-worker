"""Typed subprocess wrapper around the Higgsfield CLI."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)

CLI_TIMEOUT_S = 120.0
GENERATION_TIMEOUT_S = 20 * 60.0
POLL_INTERVAL_S = 5.0
DOWNLOAD_TIMEOUT_S = 120.0
OUTPUT_TAIL_CHARS = 2000
PROMPT_LOG_CHARS = 120

_COMPLETED_STATUSES = frozenset({"completed"})
_FAILED_STATUSES = frozenset({"failed", "error", "cancelled", "canceled"})
_PENDING_STATUSES = frozenset(
    {"created", "pending", "queued", "running", "processing", "in_progress"}
)


@dataclass(frozen=True, kw_only=True)
class GenerationParams:
    """Parameters accepted by a raw Higgsfield video-model call."""

    model: str = "seedance_2_0"
    prompt: str
    duration_s: int
    aspect_ratio: str = "9:16"
    resolution: str = "1080p"
    generate_audio: bool = False
    start_image: Path | None = None
    end_image: Path | None = None

    def __post_init__(self) -> None:
        if self.duration_s < 4:
            raise ValueError("duration_s must be at least 4 seconds")
        if (self.start_image is None) != (self.end_image is None):
            raise ValueError("start_image and end_image must be provided together")


@dataclass(frozen=True)
class GenerationResult:
    """Completed remote generation and local download metadata."""

    id: str
    result_url: str
    credits_charged: int | None
    elapsed_s: float


class HiggsfieldError(RuntimeError):
    """A CLI, protocol, validation, or generation failure."""

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str],
        exit_code: int | None,
        stderr_tail: str = "",
        stdout_tail: str = "",
    ) -> None:
        self.command = tuple(_redact_command(command))
        self.exit_code = exit_code
        self.stderr_tail = _redact_text(stderr_tail[-OUTPUT_TAIL_CHARS:])
        self.stdout_tail = _redact_text(stdout_tail[-OUTPUT_TAIL_CHARS:])

        details = [message, f"command={shlex.join(self.command)}"]
        if exit_code is not None:
            details.append(f"exit_code={exit_code}")
        if self.stderr_tail:
            details.append(f"stderr_tail={self.stderr_tail}")
        if self.stdout_tail:
            details.append(f"stdout_tail={self.stdout_tail}")
        super().__init__("; ".join(details))


class HiggsfieldDownloadError(HiggsfieldError):
    """A completed generation whose remote MP4 could not be saved locally."""

    def __init__(
        self,
        message: str,
        *,
        generation_id: str | None,
        result_url: str,
    ) -> None:
        self.generation_id = generation_id
        self.result_url = result_url
        super().__init__(
            message,
            command=("download", result_url),
            exit_code=None,
        )


def balance() -> int:
    """Return the signed-in account's available credit balance."""

    command = ["higgsfield", "account", "status", "--json"]
    data = _run_json(command, required_fields=("credits",))
    return _required_int(data, "credits", command)


def get_cost(params: GenerationParams) -> int:
    """Return the raw-model credit estimate without submitting a generation."""

    command = [
        "higgsfield",
        "generate",
        "cost",
        *_generation_arguments(params),
        "--json",
    ]
    data = _run_json(command, required_fields=("credits",))
    return _required_int(data, "credits", command)


def generate(params: GenerationParams, output_path: Path) -> GenerationResult:
    """Submit, poll, and atomically download one video generation."""

    started = time.monotonic()
    deadline = started + GENERATION_TIMEOUT_S
    create_command = [
        "higgsfield",
        "generate",
        "create",
        *_generation_arguments(params),
        "--json",
    ]
    created = _run_json(create_command, required_fields=("id",))
    generation_id = _required_string(created, "id", create_command)

    result_url = _poll_until_complete(generation_id, deadline)
    logger.info(
        "Higgsfield generation completed id=%s credits_charged=%s",
        generation_id,
        "unknown",
    )
    try:
        download(result_url, output_path)
    except HiggsfieldDownloadError as exc:
        raise HiggsfieldDownloadError(
            f"generation {generation_id} completed but download failed",
            generation_id=generation_id,
            result_url=result_url,
        ) from exc

    return GenerationResult(
        id=generation_id,
        result_url=result_url,
        credits_charged=None,
        elapsed_s=time.monotonic() - started,
    )


def _generation_arguments(params: GenerationParams) -> list[str]:
    arguments = [
        params.model,
        "--prompt",
        params.prompt,
        "--duration",
        str(params.duration_s),
        "--aspect-ratio",
        params.aspect_ratio,
        "--resolution",
        params.resolution,
        "--generate-audio",
        str(params.generate_audio).lower(),
    ]
    if params.start_image is not None and params.end_image is not None:
        arguments.extend(
            [
                "--start-image",
                str(params.start_image),
                "--end-image",
                str(params.end_image),
            ]
        )
    return arguments


def _poll_until_complete(generation_id: str, deadline: float) -> str:
    command = [
        "higgsfield",
        "generate",
        "get",
        generation_id,
        "--json",
    ]

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _generation_timeout(generation_id, command)

        try:
            data = _run_json(
                command,
                required_fields=("id", "status"),
                timeout_s=min(CLI_TIMEOUT_S, remaining),
            )
        except HiggsfieldError as exc:
            if time.monotonic() >= deadline:
                raise _generation_timeout(generation_id, command) from exc
            raise

        response_id = _required_string(data, "id", command)
        if response_id != generation_id:
            raise HiggsfieldError(
                f"generation id mismatch: expected {generation_id}, got {response_id}",
                command=command,
                exit_code=0,
                stdout_tail=json.dumps(data),
            )

        status = _required_string(data, "status", command)
        normalized_status = status.lower()
        if normalized_status in _COMPLETED_STATUSES:
            return _required_string(data, "result_url", command)
        if normalized_status in _FAILED_STATUSES:
            raise HiggsfieldError(
                f"generation {generation_id} ended with status {status}",
                command=command,
                exit_code=0,
                stdout_tail=json.dumps(data),
            )
        if normalized_status not in _PENDING_STATUSES:
            logger.warning(
                "Higgsfield generation %s returned unrecognized status %r; "
                "continuing to poll",
                generation_id,
                status,
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _generation_timeout(generation_id, command)
        time.sleep(min(POLL_INTERVAL_S, remaining))


def _generation_timeout(
    generation_id: str, command: Sequence[str]
) -> HiggsfieldError:
    return HiggsfieldError(
        f"generation {generation_id} exceeded {GENERATION_TIMEOUT_S:g}s timeout; "
        "the remote job may still complete and spend credits",
        command=command,
        exit_code=None,
    )


def download(result_url: str, output_path: Path) -> None:
    """Atomically download a non-empty remote generation result."""

    temp_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            urllib.request.urlopen(result_url, timeout=DOWNLOAD_TIMEOUT_S) as response,
            tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_path.parent,
                delete=False,
            ) as destination,
        ):
            temp_path = Path(destination.name)
            shutil.copyfileobj(response, destination)

        if temp_path.stat().st_size <= 0:
            raise OSError("downloaded file is empty")
        temp_path.replace(output_path)
    except Exception as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise HiggsfieldDownloadError(
            f"download failed: {exc}",
            generation_id=None,
            result_url=result_url,
        ) from exc


def _run_json(
    command: Sequence[str],
    *,
    required_fields: Sequence[str],
    timeout_s: float = CLI_TIMEOUT_S,
) -> dict[str, object]:
    safe_command = _redact_command(command)
    logger.info("Running Higgsfield command: %s", _command_for_log(safe_command))

    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HiggsfieldError(
            f"Higgsfield CLI timed out after {timeout_s:g}s",
            command=command,
            exit_code=None,
            stderr_tail=_output_text(exc.stderr),
            stdout_tail=_output_text(exc.stdout),
        ) from exc
    except OSError as exc:
        raise HiggsfieldError(
            f"could not execute Higgsfield CLI: {exc}",
            command=command,
            exit_code=None,
        ) from exc

    if completed.returncode != 0:
        raise HiggsfieldError(
            "Higgsfield CLI failed",
            command=command,
            exit_code=completed.returncode,
            stderr_tail=completed.stderr,
            stdout_tail=completed.stdout,
        )

    try:
        parsed: object = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HiggsfieldError(
            "Higgsfield CLI exited 0 but stdout was not valid JSON",
            command=command,
            exit_code=0,
            stderr_tail=completed.stderr,
            stdout_tail=completed.stdout,
        ) from exc

    if not isinstance(parsed, dict):
        raise HiggsfieldError(
            "Higgsfield CLI exited 0 but JSON stdout was not an object",
            command=command,
            exit_code=0,
            stderr_tail=completed.stderr,
            stdout_tail=completed.stdout,
        )
    data = cast(dict[str, object], parsed)

    missing = [field for field in required_fields if field not in data]
    if missing:
        raise HiggsfieldError(
            f"Higgsfield JSON response missing fields: {', '.join(missing)}",
            command=command,
            exit_code=0,
            stderr_tail=completed.stderr,
            stdout_tail=completed.stdout,
        )
    return data


def _required_string(
    data: dict[str, object], field: str, command: Sequence[str]
) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise HiggsfieldError(
            f"Higgsfield JSON field {field!r} must be a non-empty string",
            command=command,
            exit_code=0,
            stdout_tail=json.dumps(data),
        )
    return value


def _required_int(
    data: dict[str, object], field: str, command: Sequence[str]
) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HiggsfieldError(
            f"Higgsfield JSON field {field!r} must be an integer",
            command=command,
            exit_code=0,
            stdout_tail=json.dumps(data),
        )
    return value


def _command_for_log(command: Sequence[str]) -> str:
    display = list(command)
    for index, argument in enumerate(display[:-1]):
        if argument == "--prompt":
            display[index + 1] = _truncate(display[index + 1], PROMPT_LOG_CHARS)
    return shlex.join(display)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _redact_command(command: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            redacted.append("[REDACTED]")
            redact_next = False
            continue
        redacted.append(_redact_text(argument))
        if argument in {"--api-key", "--token"}:
            redact_next = True
    return redacted


def _redact_text(value: str) -> str:
    api_key = os.environ.get("HIGGSFIELD_API_KEY", "")
    return value.replace(api_key, "[REDACTED]") if api_key else value


def _output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
