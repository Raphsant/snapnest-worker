"""Creative stage: generate prompts and captions for approved clips."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from worker import jobs

if TYPE_CHECKING:
    from worker.config import Config
    from worker.stages import StageContext

logger = logging.getLogger(__name__)

MAX_TOKENS = 16000
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "creative_system.md"
REQUIRED_RESPONSE_FIELDS: tuple[str, ...] = (
    "hook_angle",
    "hook_on_screen_text",
    "hook_prompt",
    "close_on_screen_text",
    "close_prompt",
    "caption_youtube",
    "caption_tiktok",
    "caption_instagram",
    "compliance_check",
)
ON_SCREEN_TEXT_PROMPT_PAIRS: tuple[tuple[str, str], ...] = (
    ("hook_on_screen_text", "hook_prompt"),
    ("close_on_screen_text", "close_prompt"),
)


class CreativeError(RuntimeError):
    """Raised when a creative package cannot be produced safely."""


def run_creative(ctx: StageContext) -> None:
    """Generate creative fields for every approved clip, then pause for review."""

    manifest, approved_clips = validate_creative_manifest(ctx.job.manifest)
    system_prompt = load_system_prompt()
    client = _anthropic_client(ctx.config)

    generated: list[tuple[dict[str, Any], dict[str, str]]] = []
    for clip in approved_clips:
        clip_id = _required_clip_string(clip, "id")
        category = _required_clip_string(clip, "category")
        transcript = _required_clip_string(clip, "transcript")

        try:
            package, response_size = _generate_clip_package(
                client,
                model=ctx.config.creative_model,
                system_prompt=system_prompt,
                clip_id=clip_id,
                category=category,
                transcript=transcript,
            )
        except Exception as exc:
            raise CreativeError(f"creative: clip {clip_id} failed: {exc}") from exc

        _log_package_warnings(ctx.job.id, clip_id, package)
        logger.info(
            "creative[%s]: clip=%s hook_angle=%s compliance=%s response_bytes=%d",
            ctx.job.id,
            clip_id,
            package["hook_angle"],
            package["compliance_check"],
            response_size,
        )
        generated.append((clip, package))

    # Do not mutate even the in-memory manifest until every approved clip has a
    # valid package. This keeps persistence all-or-nothing on model failures.
    for clip, package in generated:
        clip.update(compose_manifest_fields(package))

    manifest_path = ctx.workspace.path("manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest_key = f"pipeline/{ctx.job.id}/manifest.json"
    ctx.workspace.upload(manifest_path, manifest_key)
    jobs.save_manifest_awaiting_creative_approval(ctx.conn, ctx.job.id, manifest)
    logger.info(
        "creative[%s]: manifest stored; awaiting creative approval", ctx.job.id
    )


def load_system_prompt() -> str:
    """Load the creative system prompt byte-for-byte."""

    return PROMPT_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pure helpers (no Anthropic, DB, or S3 side effects)
# --------------------------------------------------------------------------- #


def extract_json(text: str) -> dict[str, Any]:
    """Parse the JSON object between the first opening and last closing brace."""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    parsed: Any = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def validate_creative_json(data: object) -> dict[str, str]:
    """Require every creative response field to be a non-empty string."""

    if not isinstance(data, dict):
        raise ValueError("creative response must be a JSON object")
    raw = cast(dict[str, object], data)

    validated: dict[str, str] = {}
    for field in REQUIRED_RESPONSE_FIELDS:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        validated[field] = value
    return validated


def compose_manifest_fields(package: Mapping[str, str]) -> dict[str, str]:
    """Compose the three plain-string fields consumed by the backend gate."""

    return {
        "hook_prompt": (
            f'ON-SCREEN TEXT: "{package["hook_on_screen_text"]}" | '
            f'ANGLE: {package["hook_angle"]}\n\n{package["hook_prompt"]}'
        ),
        "close_prompt": (
            f'ON-SCREEN TEXT: "{package["close_on_screen_text"]}"\n\n'
            f'{package["close_prompt"]}'
        ),
        "post_copy": (
            f'### YouTube Shorts\n{package["caption_youtube"]}\n\n'
            f'### TikTok\n{package["caption_tiktok"]}\n\n'
            f'### Instagram\n{package["caption_instagram"]}\n\n'
            f'Compliance check: {package["compliance_check"]}'
        ),
    }


def find_on_screen_text_mismatches(
    package: Mapping[str, str],
) -> list[tuple[str, str]]:
    """Return field pairs whose exact on-screen text is absent from the prompt."""

    return [
        (text_field, prompt_field)
        for text_field, prompt_field in ON_SCREEN_TEXT_PROMPT_PAIRS
        if package[text_field] not in package[prompt_field]
    ]


def validate_creative_manifest(
    manifest: object | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy and validate the approved DB manifest used by this stage."""

    if manifest is None:
        raise CreativeError("creative: job manifest is missing")
    if not isinstance(manifest, dict):
        raise CreativeError("creative: job manifest must be a JSON object")

    data = cast(dict[str, Any], copy.deepcopy(manifest))
    if data.get("status") != "approved":
        raise CreativeError("creative: manifest status must be 'approved'")

    raw_clips = data.get("clips")
    if not isinstance(raw_clips, list):
        raise CreativeError("creative: manifest clips must be a list")

    approved: list[dict[str, Any]] = []
    for index, raw_clip in enumerate(raw_clips):
        if not isinstance(raw_clip, dict):
            raise CreativeError(
                f"creative: manifest clips[{index}] must be an object"
            )
        clip = cast(dict[str, Any], raw_clip)
        if clip.get("approved") is True:
            approved.append(clip)

    if not approved:
        raise CreativeError("creative: manifest has no approved clips")
    return data, approved


def _required_clip_string(clip: Mapping[str, object], field: str) -> str:
    value = clip.get(field)
    if not isinstance(value, str) or not value.strip():
        clip_id = clip.get("id", "<unknown>")
        raise CreativeError(
            f"creative: clip {clip_id} field {field} must be a non-empty string"
        )
    return value


# --------------------------------------------------------------------------- #
# Anthropic (thin, side-effecting wrappers)
# --------------------------------------------------------------------------- #


def _anthropic_client(cfg: Config) -> Any:
    import anthropic

    return anthropic.Anthropic(api_key=cfg.anthropic_api_key)


def _generate_clip_package(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    clip_id: str,
    category: str,
    transcript: str,
) -> tuple[dict[str, str], int]:
    """Call Creative for one clip, with one parse/validation repair retry."""

    user = (
        f"CLIP ID: {clip_id}\n"
        f"CATEGORY: {category}\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )
    messages = [{"role": "user", "content": user}]

    first = _create_message(
        client,
        model=model,
        system=system_prompt,
        messages=messages,
    )
    first_text = _message_text(first)
    try:
        package = validate_creative_json(extract_json(first_text))
        return package, len(first_text.encode("utf-8"))
    except ValueError as first_error:
        error_description = str(first_error)
        logger.warning(
            "creative: clip=%s response invalid (%s); retrying once",
            clip_id,
            error_description,
        )

    correction = (
        f"Your previous output failed validation: {error_description}. "
        "Return one corrected JSON object only."
    )
    second = _create_message(
        client,
        model=model,
        system=system_prompt,
        messages=[
            *messages,
            {"role": "assistant", "content": first_text},
            {"role": "user", "content": correction},
        ],
    )
    second_text = _message_text(second)
    try:
        package = validate_creative_json(extract_json(second_text))
    except ValueError as second_error:
        raise ValueError(
            f"validation failed after corrective retry: {second_error}"
        ) from second_error
    return package, len(second_text.encode("utf-8"))


def _create_message(
    client: Any, *, model: str, system: str, messages: list[dict[str, str]]
) -> Any:
    return client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
    )


def _message_text(message: Any) -> str:
    parts = [block.text for block in message.content if block.type == "text"]
    return "".join(parts)


def _log_package_warnings(
    job_id: str, clip_id: str, package: Mapping[str, str]
) -> None:
    for text_field, prompt_field in find_on_screen_text_mismatches(package):
        logger.warning(
            "creative[%s]: clip=%s on-screen text mismatch: %s=%r | %s=%r",
            job_id,
            clip_id,
            text_field,
            package[text_field],
            prompt_field,
            package[prompt_field],
        )

    hook_words = len(package["hook_on_screen_text"].split())
    if hook_words > 6:
        logger.warning(
            "creative[%s]: clip=%s hook on-screen text has %d words (max 6)",
            job_id,
            clip_id,
            hook_words,
        )

    close_words = len(package["close_on_screen_text"].split())
    if close_words > 7:
        logger.warning(
            "creative[%s]: clip=%s close on-screen text has %d words (max 7)",
            job_id,
            clip_id,
            close_words,
        )

    compliance = package["compliance_check"]
    if compliance != "PASS":
        logger.warning(
            "creative[%s]: clip=%s COMPLIANCE CHECK REQUIRES REVIEW: %s",
            job_id,
            clip_id,
            compliance,
        )
