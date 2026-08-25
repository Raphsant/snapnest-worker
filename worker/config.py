"""Startup configuration.

All required environment variables are validated once, at startup. If any are
missing we raise a single error that lists *every* missing var (fail loud), so
an operator can fix the deployment in one pass instead of playing whack-a-mole.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

REQUIRED_VARS: tuple[str, ...] = (
    "DATABASE_URL",
    "PIPELINE_QUEUE_URL",
    "AWS_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "S3_BUCKET",
    "ANTHROPIC_API_KEY",
)


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    """Validated, immutable runtime configuration."""

    database_url: str
    pipeline_queue_url: str
    aws_region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    s3_bucket: str
    anthropic_api_key: str

    # Optional tunables (env overridable; defaults chosen for a single-threaded worker).
    wait_time_seconds: int = 20
    visibility_timeout_seconds: int = 900
    heartbeat_interval_seconds: int = 600
    heartbeat_extension_seconds: int = 900
    job_workspace_root: str = "/tmp/jobs"

    # Pre-generated asset library (hooks/outros). Key prefix inside S3_BUCKET;
    # the catalog lives at "{library_prefix}catalog.json".
    library_prefix: str = "library/"

    # AWS Transcribe (ingest stage). Vocabulary/filter are optional: an empty
    # string means "omit that setting" so we don't send a name Transcribe would
    # reject. The filter method is always "mask" (masked words become "***").
    transcribe_language: str = "es-US"
    transcribe_vocabulary: str = ""
    transcribe_filter: str = ""
    transcribe_poll_seconds: int = 30

    # Bleep/beep tuning for the ffmpeg mute+beep pass (ingest stage).
    #   pad       - seconds of padding added around each masked word
    #   merge_gap - ranges closer than this are merged into one beep
    #   volume    - beep loudness in the amix (0-1)
    #   hz        - beep tone frequency
    bleep_pad_seconds: float = 0.15
    bleep_merge_gap_seconds: float = 0.05
    bleep_volume: float = 0.35
    bleep_hz: int = 1000

    # Anthropic stages. The API key is required (see REQUIRED_VARS); models are
    # overridable. Creative inherits the resolved curator model when unset.
    curator_model: str = "claude-sonnet-4-6"
    creative_model: str = "claude-sonnet-4-6"


def _require(env: dict[str, str], missing: list[str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        missing.append(name)
        return ""
    return value


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(env: dict[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def load_config() -> Config:
    """Load and validate configuration from the environment (and .env if present).

    Raises:
        ConfigError: if any required variable is missing, or an optional
            integer variable is set to a non-integer value.
    """

    load_dotenv()
    env = dict(os.environ)

    missing: list[str] = []
    values = {name: _require(env, missing, name) for name in REQUIRED_VARS}

    if missing:
        raise ConfigError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    curator_model = env.get("CURATOR_MODEL", "").strip() or "claude-sonnet-4-6"

    return Config(
        database_url=values["DATABASE_URL"],
        pipeline_queue_url=values["PIPELINE_QUEUE_URL"],
        aws_region=values["AWS_REGION"],
        aws_access_key_id=values["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=values["AWS_SECRET_ACCESS_KEY"],
        s3_bucket=values["S3_BUCKET"],
        anthropic_api_key=values["ANTHROPIC_API_KEY"],
        wait_time_seconds=_env_int(env, "WAIT_TIME_SECONDS", 20),
        visibility_timeout_seconds=_env_int(env, "VISIBILITY_TIMEOUT_SECONDS", 900),
        heartbeat_interval_seconds=_env_int(env, "HEARTBEAT_INTERVAL_SECONDS", 600),
        heartbeat_extension_seconds=_env_int(env, "HEARTBEAT_EXTENSION_SECONDS", 900),
        job_workspace_root=env.get("JOB_WORKSPACE_ROOT", "").strip() or "/tmp/jobs",
        library_prefix=env.get("LIBRARY_PREFIX", "").strip() or "library/",
        transcribe_language=env.get("TRANSCRIBE_LANGUAGE", "").strip() or "es-US",
        transcribe_vocabulary=env.get("TRANSCRIBE_VOCABULARY", "").strip(),
        transcribe_filter=env.get("TRANSCRIBE_FILTER", "").strip(),
        transcribe_poll_seconds=_env_int(env, "TRANSCRIBE_POLL_SECONDS", 30),
        bleep_pad_seconds=_env_float(env, "BLEEP_PAD", 0.15),
        bleep_merge_gap_seconds=_env_float(env, "BLEEP_MERGE_GAP", 0.05),
        bleep_volume=_env_float(env, "BLEEP_VOLUME", 0.35),
        bleep_hz=_env_int(env, "BLEEP_HZ", 1000),
        curator_model=curator_model,
        creative_model=env.get("CREATIVE_MODEL", "").strip() or curator_model,
    )
