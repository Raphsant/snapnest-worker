"""Generate stage: validation-only gate for library asset selections.

Creative v2 selects pre-generated library hooks/outros per approved clip
(hook_asset_id / outro_asset_id); bridges no longer exist. Operators can
edit the manifest between the creative gate and this stage, so selections
are re-validated here against the asset library catalog. Nothing is
generated and no Higgsfield call of any kind is made — zero credit spend.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from worker import jobs
from worker.library import LibraryCatalog
from worker.lint import lint_prompts, lint_selections

if TYPE_CHECKING:
    from psycopg import Connection
    from psycopg.rows import DictRow

    from worker.config import Config
    from worker.jobs import Job
    from worker.workspace import Workspace

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GenerateStageContext(Protocol):
    """Context fields consumed by the generate stage."""

    job: Job
    workspace: Workspace
    conn: Connection[DictRow]
    config: Config


class GenerateError(RuntimeError):
    """Raised when the manifest's library selections cannot be validated."""


def run_generate(ctx: GenerateStageContext) -> None:
    """Validate approved clips' library selections and advance; spend nothing."""

    manifest, clips = validate_generate_manifest(ctx.job.manifest)

    # Legacy enforcement, kept for stale pre-v2 manifests: any surviving
    # hook_prompt/close_prompt whose text embeds branding hard-fails so the
    # operator regenerates creative rather than advancing a poisoned manifest.
    prompt_lint = lint_prompts(manifest)
    if prompt_lint.violations:
        detail = "; ".join(
            f"{v['clipId']}.{v['field']} matched {v['matched_word']!r} "
            f"in {v['prompt_excerpt']!r}"
            for v in prompt_lint.violations
        )
        raise GenerateError(
            f"generate: {len(prompt_lint.violations)} generation-prompt lint "
            f"violation(s) block the pipeline; regenerate creative to clear "
            f"them: {detail}"
        )
    if prompt_lint.warnings:
        logger.info(
            "generate: %d negated lint warning(s), non-blocking: %s",
            len(prompt_lint.warnings),
            "; ".join(
                f"{w['clipId']}.{w['field']} matched {w['matched_word']!r}"
                for w in prompt_lint.warnings
            ),
        )

    catalog = _load_catalog(ctx)

    selection_lint = lint_selections(manifest, catalog)
    if selection_lint.violations:
        detail = "; ".join(
            f"{v['clipId']}.{v['field']}: {v['reason']}"
            for v in selection_lint.violations
        )
        raise GenerateError(
            f"generate: {len(selection_lint.violations)} library-selection "
            f"lint violation(s) block the pipeline; fix the manifest "
            f"selections: {detail}"
        )
    if selection_lint.warnings:
        logger.info(
            "generate: %d selection lint warning(s), non-blocking: %s",
            len(selection_lint.warnings),
            "; ".join(
                f"{w['clipId']}.{w['field']}: {w['reason']}"
                for w in selection_lint.warnings
            ),
        )

    for clip in clips:
        hook_id = str(clip["hook_asset_id"])
        outro_id = str(clip["outro_asset_id"])
        hook = catalog.get(hook_id)
        outro = catalog.get(outro_id)
        # lint_selections guarantees both ids resolve; guard for the checker.
        logger.info(
            "generate[%s]: clip=%s hook_asset=%s (s3=%s) outro_asset=%s (s3=%s)",
            ctx.job.id,
            clip["id"],
            hook_id,
            hook.s3_key if hook else "<unresolved>",
            outro_id,
            outro.s3_key if outro else "<unresolved>",
        )

    authoritative = jobs.load_manifest(ctx.conn, ctx.job.id)
    if not isinstance(authoritative, dict):
        raise GenerateError("generate: authoritative DB manifest missing")
    final_manifest = cast(dict[str, Any], authoritative)
    manifest_path = ctx.workspace.path("manifest.json")
    manifest_path.write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _retry_once(
        lambda: ctx.workspace.upload(
            manifest_path, f"pipeline/{ctx.job.id}/manifest.json"
        ),
        description="final manifest S3 upload",
    )
    logger.info("generate[%s]: all approved selections validated", ctx.job.id)


def validate_generate_manifest(
    manifest: object | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy and validate the authoritative manifest used for validation."""

    if manifest is None:
        raise GenerateError("generate: job manifest is missing")
    if not isinstance(manifest, dict):
        raise GenerateError("generate: job manifest must be a JSON object")

    data = cast(dict[str, Any], copy.deepcopy(manifest))
    raw_clips = data.get("clips")
    if not isinstance(raw_clips, list):
        raise GenerateError("generate: manifest clips must be a list")

    approved: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_clip in enumerate(raw_clips):
        if not isinstance(raw_clip, dict):
            raise GenerateError(
                f"generate: manifest clips[{index}] must be an object"
            )
        clip = cast(dict[str, Any], raw_clip)
        if clip.get("approved") is not True:
            continue

        clip_id = _required_clip_string(clip, "id", index)
        if (
            clip_id in {".", ".."}
            or Path(clip_id).name != clip_id
            or "\x00" in clip_id
        ):
            raise GenerateError(
                f"generate: manifest clips[{index}].id is not a safe filename"
            )
        if clip_id in seen_ids:
            raise GenerateError(f"generate: duplicate approved clip id {clip_id!r}")
        seen_ids.add(clip_id)

        _required_clip_string(clip, "hook_asset_id", index)
        _required_clip_string(clip, "outro_asset_id", index)
        approved.append(clip)

    if not approved:
        raise GenerateError("generate: manifest has no approved clips")
    return data, approved


def _load_catalog(ctx: GenerateStageContext) -> LibraryCatalog:
    """Load the asset library catalog; the stage cannot run without it."""

    try:
        return LibraryCatalog.from_s3(
            ctx.workspace.s3, ctx.workspace.bucket, ctx.config.library_prefix
        )
    except Exception as exc:
        raise GenerateError(
            f"generate: asset library catalog unavailable or invalid: {exc}"
        ) from exc


def _required_clip_string(
    clip: Mapping[str, object], field: str, index: int
) -> str:
    value = clip.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GenerateError(
            f"generate: manifest clips[{index}].{field} must be a non-empty string"
        )
    return value


def _retry_once(operation: Callable[[], T], *, description: str) -> T:
    try:
        return operation()
    except Exception as first_error:
        logger.warning("%s failed (%s); retrying once", description, first_error)
    try:
        return operation()
    except Exception as second_error:
        raise GenerateError(
            f"generate: {description} failed after retry: {second_error}"
        ) from second_error
