"""Creative stage: select library assets and write captions for approved clips."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping, Set as AbstractSet
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from worker import jobs
from worker.library import LibraryCatalog, format_for_prompt
from worker.lint import lint_prompts

if TYPE_CHECKING:
    from worker.config import Config
    from worker.stages import StageContext

logger = logging.getLogger(__name__)

MAX_TOKENS = 16000
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "creative_system.md"
# Literal token in creative_system.md replaced with the formatted asset library.
ASSET_LIBRARY_TOKEN = "{{ASSET_LIBRARY}}"
REQUIRED_RESPONSE_FIELDS: tuple[str, ...] = (
    "hook_angle",
    "hook_text",
    "hook_asset_id",
    "close_text",
    "outro_asset_id",
    "caption_youtube",
    "caption_tiktok",
    "caption_instagram",
    "compliance_check",
)

# Overlay copy should fit ONE line of the 9:16 drawtext overlay at full size.
# These caps are WARNING thresholds only — never a hard failure. They are now
# measured against the real v3.1 fonts at fontsize 72 vs 972px usable width
# (1080px frame minus a 5% margin per side): Montserrat ExtraBold (close_text)
# averages ~53.5px/char, Anton (hook_text) ~34.1px/char. Clipping itself is no
# longer at stake — the assembler guarantees fit via dynamic fontsize shrink —
# so these caps exist to keep overlay text rendering at FULL size, not to
# prevent clipping.
HOOK_TEXT_MAX_CHARS = 22
CLOSE_TEXT_MAX_CHARS = 16


class CreativeError(RuntimeError):
    """Raised when a creative package cannot be produced safely."""


def run_creative(ctx: StageContext) -> None:
    """Select assets and write creative fields per approved clip, then pause."""

    manifest, approved_clips = validate_creative_manifest(ctx.job.manifest)
    catalog = _load_catalog(ctx)
    system_prompt = load_system_prompt(catalog)
    client = _anthropic_client(ctx.config)

    generated: list[tuple[dict[str, Any], dict[str, str]]] = []
    used_ids: set[str] = set()
    for clip in approved_clips:
        clip_id = _required_clip_string(clip, "id")
        category = _required_clip_string(clip, "category")
        transcript = _required_clip_string(clip, "transcript")

        try:
            package, response_size = _generate_clip_package(
                client,
                model=ctx.config.creative_model,
                system_prompt=system_prompt,
                catalog=catalog,
                clip_id=clip_id,
                category=category,
                transcript=transcript,
                used_ids=used_ids,
            )
        except Exception as exc:
            raise CreativeError(f"creative: clip {clip_id} failed: {exc}") from exc

        _log_package_warnings(ctx.job.id, clip_id, package)
        logger.info(
            "creative[%s]: clip=%s hook_angle=%s hook_asset=%s outro_asset=%s "
            "compliance=%s response_bytes=%d",
            ctx.job.id,
            clip_id,
            package["hook_angle"],
            package["hook_asset_id"],
            package["outro_asset_id"],
            package["compliance_check"],
            response_size,
        )
        used_ids.add(package["hook_asset_id"])
        used_ids.add(package["outro_asset_id"])
        generated.append((clip, package))

    # Do not mutate even the in-memory manifest until every approved clip has a
    # valid package. This keeps persistence all-or-nothing on model failures.
    for clip, package in generated:
        clip.update(compose_manifest_fields(package))

    # Enforcement choke point #1, retained for stale manifests: v2 composes no
    # generation prompts, but a pre-v2 manifest may still carry dirty
    # hook_prompt/close_prompt fields — lint records them for operator review.
    lint = lint_prompts(manifest)
    manifest["lint_violations"] = lint.violations
    manifest["lint_warnings"] = lint.warnings
    if lint.violations:
        logger.warning(
            "creative[%s]: %d generation-prompt lint violation(s) recorded "
            "for operator review: %s",
            ctx.job.id,
            len(lint.violations),
            "; ".join(
                f"{v['clipId']}.{v['field']} matched {v['matched_word']!r}"
                for v in lint.violations
            ),
        )
    if lint.warnings:
        logger.info(
            "creative[%s]: %d negated lint warning(s) recorded (non-blocking): %s",
            ctx.job.id,
            len(lint.warnings),
            "; ".join(
                f"{w['clipId']}.{w['field']} matched {w['matched_word']!r}"
                for w in lint.warnings
            ),
        )

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


def load_system_prompt(catalog: LibraryCatalog) -> str:
    """Load the creative system prompt and inject the asset library catalog."""

    template = PROMPT_PATH.read_text(encoding="utf-8")
    if ASSET_LIBRARY_TOKEN not in template:
        raise CreativeError(
            f"creative: system prompt is missing the {ASSET_LIBRARY_TOKEN} "
            "placeholder"
        )
    return template.replace(ASSET_LIBRARY_TOKEN, format_for_prompt(catalog))


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


def validate_asset_selection(
    package: Mapping[str, str],
    catalog: LibraryCatalog,
    *,
    category: str,
    clip_id: str,
    used_ids: AbstractSet[str] = frozenset(),
) -> None:
    """Require selected ids to exist, match type, and be unused in this batch.

    A missing id, a type mismatch, or an id already selected for another clip
    raises ValueError (feeding the repair retry). A clip category absent from
    the asset's categories is only a warning: universal assets legitimately
    cross categories.
    """

    for field, expected_type in (
        ("hook_asset_id", "hook"),
        ("outro_asset_id", "outro"),
    ):
        asset_id = package[field]
        asset = catalog.get(asset_id)
        if asset is None:
            raise ValueError(
                f"{field} {asset_id!r} does not exist in the asset library"
            )
        if asset.type != expected_type:
            raise ValueError(
                f"{field} {asset_id!r} has type {asset.type!r}, "
                f"expected {expected_type!r}"
            )
        if asset_id in used_ids:
            raise ValueError(
                f"{field} {asset_id!r} was already selected in this batch "
                f"(used: {', '.join(sorted(used_ids))})"
            )
        if category not in asset.category:
            logger.warning(
                "creative: clip=%s %s=%s clip category %r not in asset "
                "categories %s (cross-category selection allowed)",
                clip_id,
                field,
                asset_id,
                category,
                list(asset.category),
            )


def compose_manifest_fields(package: Mapping[str, str]) -> dict[str, str]:
    """Compose the manifest fields written to each approved clip.

    Emits native overlay copy (hook_text/close_text) consumed by the
    assembler's drawtext, the selected library asset ids (hooks/outros are
    pre-generated — no generation prompts are written), and the platform
    post copy.
    """

    return {
        "hook_text": package["hook_text"],
        "hook_asset_id": package["hook_asset_id"],
        "close_text": package["close_text"],
        "outro_asset_id": package["outro_asset_id"],
        "post_copy": (
            f'### YouTube Shorts\n{package["caption_youtube"]}\n\n'
            f'### TikTok\n{package["caption_tiktok"]}\n\n'
            f'### Instagram\n{package["caption_instagram"]}\n\n'
            f'Compliance check: {package["compliance_check"]}'
        ),
    }


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
# S3 (catalog load)
# --------------------------------------------------------------------------- #


def _load_catalog(ctx: StageContext) -> LibraryCatalog:
    """Load the asset library catalog; the stage cannot run without it."""

    try:
        return LibraryCatalog.from_s3(
            ctx.workspace.s3, ctx.workspace.bucket, ctx.config.library_prefix
        )
    except Exception as exc:
        raise CreativeError(
            f"creative: asset library catalog unavailable or invalid: {exc}"
        ) from exc


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
    catalog: LibraryCatalog,
    clip_id: str,
    category: str,
    transcript: str,
    used_ids: AbstractSet[str] = frozenset(),
) -> tuple[dict[str, str], int]:
    """Call Creative for one clip, with one parse/validation/selection repair retry."""

    user = (
        f"CLIP ID: {clip_id}\n"
        f"CATEGORY: {category}\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )
    if used_ids:
        user += (
            "\n\nALREADY SELECTED IN THIS BATCH (do not reuse): "
            + ", ".join(sorted(used_ids))
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
        package = _validated_package(
            first_text,
            catalog,
            category=category,
            clip_id=clip_id,
            used_ids=used_ids,
        )
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
        package = _validated_package(
            second_text,
            catalog,
            category=category,
            clip_id=clip_id,
            used_ids=used_ids,
        )
    except ValueError as second_error:
        raise ValueError(
            f"validation failed after corrective retry: {second_error}"
        ) from second_error
    return package, len(second_text.encode("utf-8"))


def _validated_package(
    text: str,
    catalog: LibraryCatalog,
    *,
    category: str,
    clip_id: str,
    used_ids: AbstractSet[str],
) -> dict[str, str]:
    """Parse, shape-validate, and selection-validate one model response."""

    package = validate_creative_json(extract_json(text))
    validate_asset_selection(
        package, catalog, category=category, clip_id=clip_id, used_ids=used_ids
    )
    return package


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
    hook_chars = len(package["hook_text"])
    if hook_chars > HOOK_TEXT_MAX_CHARS:
        logger.warning(
            "creative[%s]: clip=%s hook_text is %d chars (max %d)",
            job_id,
            clip_id,
            hook_chars,
            HOOK_TEXT_MAX_CHARS,
        )

    close_chars = len(package["close_text"])
    if close_chars > CLOSE_TEXT_MAX_CHARS:
        logger.warning(
            "creative[%s]: clip=%s close_text is %d chars (max %d)",
            job_id,
            clip_id,
            close_chars,
            CLOSE_TEXT_MAX_CHARS,
        )

    compliance = package["compliance_check"]
    if compliance != "PASS":
        logger.warning(
            "creative[%s]: clip=%s COMPLIANCE CHECK REQUIRES REVIEW: %s",
            job_id,
            clip_id,
            compliance,
        )
