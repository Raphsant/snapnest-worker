"""Creative stage: select library assets and write captions for approved clips."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from worker import jobs
from worker.library import LibraryAsset, LibraryCatalog, format_for_prompt
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

# The per-clip manifest fields this stage writes (see compose_manifest_fields).
# A clip carrying all five is DONE and must not be re-processed: an extension
# run re-enters this stage with earlier passes' clips still approved. The list
# deliberately mirrors the backend's REQUIRED_CREATIVE_FIELDS gate
# (approveCreative), so a clip this stage skips is exactly a clip the creative
# gate accepts — a partially written clip is re-processed rather than left to
# deadlock the gate at 409.
DONE_FIELDS: tuple[str, ...] = (
    "hook_asset_id",
    "outro_asset_id",
    "hook_text",
    "close_text",
    "post_copy",
)

# Library selection fields harvested from the manifest to seed usage_counts.
SELECTION_ID_FIELDS: tuple[str, ...] = ("hook_asset_id", "outro_asset_id")

# Reuse within a job is ALLOWED policy: selection prefers the least-used asset
# among comparable fits, and repetition is never an error. This threshold only
# decides when a repeat is loud enough to log for the operator — an asset the
# model has already leaned on twice is worth a look, a single repeat is not.
REUSE_WARN_THRESHOLD = 2

# Immutable empty default for the usage_counts keyword.
_NO_USAGE: Mapping[str, int] = MappingProxyType({})

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
    # Seeded from EVERY clip in the manifest, not just the ones processed
    # below: on an extension run the counts an earlier pass shipped carry into
    # this run's availability block, so "least-used" means least-used across
    # the whole job rather than only within this pass.
    usage_counts: dict[str, int] = _seed_usage_counts(manifest)
    for clip in approved_clips:
        clip_id = _required_clip_string(clip, "id")
        if _clip_is_done(clip):
            logger.info(
                "creative[%s]: clip=%s creative fields exist; skipping",
                ctx.job.id,
                clip_id,
            )
            continue
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
                usage_counts=usage_counts,
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
        for field in SELECTION_ID_FIELDS:
            selected = package[field]
            usage_counts[selected] = usage_counts.get(selected, 0) + 1
        generated.append((clip, package))

    logger.info(
        "creative[%s]: %d clip(s) processed, %d already complete",
        ctx.job.id,
        len(generated),
        len(approved_clips) - len(generated),
    )

    # Do not mutate even the in-memory manifest until every approved clip has a
    # valid package. This keeps persistence all-or-nothing on model failures.
    # An all-skipped run leaves this empty and still falls through to the gate
    # save below — that is what parks a re-opened job back at the creative gate.
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
    usage_counts: Mapping[str, int] = _NO_USAGE,
) -> None:
    """Require selected ids to exist and match type; warn on heavy reuse.

    A missing id or a type mismatch raises ValueError (feeding the repair
    retry). Reuse is NOT an error — an id another clip in this job already
    carries is a legitimate selection — it is only logged once the asset's
    in-job usage has reached REUSE_WARN_THRESHOLD. A clip category absent from
    the asset's categories is likewise only a warning: universal assets
    legitimately cross categories.
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
        used = usage_counts.get(asset_id, 0)
        if used >= REUSE_WARN_THRESHOLD:
            logger.warning(
                "creative: clip=%s %s=%s already used %d time(s) in this job "
                "(reuse allowed)",
                clip_id,
                field,
                asset_id,
                used,
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


def compose_availability_block(
    catalog: LibraryCatalog, usage_counts: Mapping[str, int]
) -> str:
    """Render every asset id with its in-job usage, plus the selection rule.

    EVERY id is listed on EVERY request — unused ones at 0 included — so the
    model always picks from the whole library and can see which assets this job
    has already leaned on. The counts are advisory: they steer the tiebreak
    between comparable fits, they do not fence anything off.
    """

    def line(label: str, assets: list[LibraryAsset]) -> str:
        pairs = ", ".join(
            f"{asset.id} — {usage_counts.get(asset.id, 0)}" for asset in assets
        )
        return f"{label} (id — times used this job): {pairs}"

    return (
        f"{line('HOOKS', catalog.hooks())}\n"
        f"{line('OUTROS', catalog.outros())}\n"
        "Pick the best fit for this clip; among comparable fits prefer the "
        "least-used. Reusing an asset is allowed."
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


def _clip_is_done(clip: Mapping[str, object]) -> bool:
    """True when a clip already carries every creative field this stage writes.

    All-or-nothing by design: a clip missing even one DONE_FIELDS entry (a
    hand-edited manifest, or a run that died mid-write) is re-processed, because
    the backend creative gate refuses to advance on a partially written clip.
    Deliberately does NOT look at the ``assembled`` checkpoint — a clip whose
    creative succeeded but whose assembly never ran is done for THIS stage.
    """

    for field in DONE_FIELDS:
        value = clip.get(field)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def _seed_usage_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Count how often each library asset id is selected across the manifest.

    Scans ALL clips — approved or not, assembled or not — so the availability
    block reports usage for the whole job, including what earlier passes
    shipped. Unapproved clips carry no selections in practice; including them
    costs nothing and keeps the count correct for a clip that was de-approved
    between passes.
    """

    counts: dict[str, int] = {}
    clips = manifest.get("clips")
    if not isinstance(clips, list):
        return counts
    for raw_clip in clips:
        if not isinstance(raw_clip, Mapping):
            continue
        for field in SELECTION_ID_FIELDS:
            value = raw_clip.get(field)
            if isinstance(value, str) and value.strip():
                counts[value] = counts.get(value, 0) + 1
    return counts


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
    usage_counts: Mapping[str, int] = _NO_USAGE,
) -> tuple[dict[str, str], int]:
    """Call Creative for one clip, with one parse/validation/selection repair retry."""

    user = (
        f"CLIP ID: {clip_id}\n"
        f"CATEGORY: {category}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"{compose_availability_block(catalog, usage_counts)}"
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
            usage_counts=usage_counts,
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
            usage_counts=usage_counts,
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
    usage_counts: Mapping[str, int],
) -> dict[str, str]:
    """Parse, shape-validate, and selection-validate one model response."""

    package = validate_creative_json(extract_json(text))
    validate_asset_selection(
        package,
        catalog,
        category=category,
        clip_id=clip_id,
        usage_counts=usage_counts,
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
