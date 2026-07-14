from __future__ import annotations

import pytest

from worker.config import REQUIRED_VARS, ConfigError, load_config


def _no_dotenv(*_args: object, **_kwargs: object) -> bool:
    return False


def _set_all_required(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in REQUIRED_VARS:
        monkeypatch.setenv(name, f"value-{name}")


def test_missing_required_lists_every_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("worker.config.load_dotenv", _no_dotenv)
    for name in REQUIRED_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    message = str(exc_info.value)
    for name in REQUIRED_VARS:
        assert name in message


def test_success_populates_config_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("worker.config.load_dotenv", _no_dotenv)
    _set_all_required(monkeypatch)

    config = load_config()

    assert config.database_url == "value-DATABASE_URL"
    assert config.s3_bucket == "value-S3_BUCKET"
    assert config.wait_time_seconds == 20
    assert config.heartbeat_interval_seconds == 600
    assert config.job_workspace_root == "/tmp/jobs"


def test_optional_int_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("worker.config.load_dotenv", _no_dotenv)
    _set_all_required(monkeypatch)
    monkeypatch.setenv("WAIT_TIME_SECONDS", "7")

    config = load_config()

    assert config.wait_time_seconds == 7


def test_optional_int_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("worker.config.load_dotenv", _no_dotenv)
    _set_all_required(monkeypatch)
    monkeypatch.setenv("WAIT_TIME_SECONDS", "not-a-number")

    with pytest.raises(ConfigError):
        load_config()
