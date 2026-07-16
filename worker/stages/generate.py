"""Generate stage: create and checkpoint Higgsfield assets per approved clip."""

from __future__ import annotations

import copy
import json
import logging
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from worker import higgsfield, jobs
from worker.higgsfield import GenerationParams, GenerationResult

if TYPE_CHECKING:
    from psycopg import Connection
    from psycopg.rows import DictRow

    from worker.jobs import Job
    from worker.workspace import Workspace

logger = logging.getLogger(__name__)

BRIDGE_IN_PROMPT = (
    "Pure visual connective tissue between the provided start and end frames. "
    "Fast, accelerating push forward — energy building into the content. "
    "No text, no logos, silent. Red and green allowed only as stock-chart "
    "semantics."
)
BRIDGE_OUT_PROMPT = (
    "Pure visual connective tissue between the provided start and end frames. "
    "Calm deceleration — settling into the close. "
    "No text, no logos, silent. Red and green allowed only as stock-chart "
    "semantics."
)

MODEL = "seedance_2_0"
ASPECT_RATIO = "9:16"
RESOLUTION = "1080p"
HOOK_DURATION_S = 4
OUTRO_DURATION_S = 5
BRIDGE_DURATION_S = 4
FFMPEG_ERROR_TAIL = 4000

AssetName = Literal["hook", "outro", "bridge_in", "bridge_out"]
ASSET_ORDER: tuple[AssetName, ...] = (
    "hook",
    "outro",
    "bridge_in",
    "bridge_out",
)
T = TypeVar("T")


class GenerateStageContext(Protocol):
    """Context fields consumed by the generate stage."""

    job: Job
    workspace: Workspace
    conn: Connection[DictRow]
    checkpoint_heartbeat: Callable[[], None]


class GenerateError(RuntimeError):
    """Raised when generated assets cannot be completed safely."""


@dataclass(frozen=True)
class ShapeCosts:
    """Preflight credit estimates for the three distinct asset shapes."""

    hook: int
    outro: int
    bridge: int

    def for_asset(self, asset: AssetName) -> int:
        if asset == "hook":
            return self.hook
        if asset == "outro":
            return self.outro
        return self.bridge


@dataclass(frozen=True)
class Preflight:
    """Outstanding credit requirement, including per-clip packages."""

    costs: ShapeCosts
    total: int
    per_clip: dict[str, int]


@dataclass(frozen=True)
class AssetGeneration:
    """Remote identity for a locally available generated asset."""

    generation_id: str
    result_url: str


def run_generate(ctx: GenerateStageContext) -> None:
    """Generate all missing approved assets and synchronize the final manifest."""

    manifest, clips = validate_generate_manifest(ctx.job.manifest)
    preflight = _preflight(clips)
    available = higgsfield.balance()
    if available < preflight.total:
        packages = ", ".join(
            f"{clip_id}={credits}cr"
            for clip_id, credits in preflight.per_clip.items()
        )
        raise GenerateError(
            "generate: insufficient Higgsfield credits: "
            f"balance={available}, required={preflight.total}; "
            f"shape costs hook={preflight.costs.hook}, "
            f"outro={preflight.costs.outro}, "
            f"bridge_each={preflight.costs.bridge}; "
            f"outstanding packages: {packages or 'none'}"
        )

    logger.info(
        "generate[%s]: preflight balance=%d required=%d "
        "(hook=%d outro=%d bridge_each=%d)",
        ctx.job.id,
        available,
        preflight.total,
        preflight.costs.hook,
        preflight.costs.outro,
        preflight.costs.bridge,
    )

    for clip in clips:
        _process_clip(ctx, manifest, clip, preflight.costs)

    authoritative = jobs.load_manifest(ctx.conn, ctx.job.id)
    if not isinstance(authoritative, dict):
        raise GenerateError(
            "generate: authoritative DB manifest missing after checkpoints"
        )
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
    logger.info("generate[%s]: all approved assets complete", ctx.job.id)


def validate_generate_manifest(
    manifest: object | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy and validate the authoritative manifest used for generation."""

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

        _required_clip_string(clip, "hook_prompt", index)
        _required_clip_string(clip, "close_prompt", index)
        for asset in ASSET_ORDER:
            _checkpoint(clip, asset)
        approved.append(clip)

    if not approved:
        raise GenerateError("generate: manifest has no approved clips")
    return data, approved


def calculate_preflight(clips: list[dict[str, Any]], costs: ShapeCosts) -> Preflight:
    """Calculate outstanding credits while skipping valid checkpoints."""

    per_clip: dict[str, int] = {}
    total = 0
    for clip in clips:
        clip_id = str(clip["id"])
        package = sum(
            costs.for_asset(asset)
            for asset in ASSET_ORDER
            if _checkpoint(clip, asset) is None
        )
        per_clip[clip_id] = package
        total += package
    return Preflight(costs=costs, total=total, per_clip=per_clip)


def build_frame_command(source: Path, output: Path, *, last: bool) -> list[str]:
    """Build one ffmpeg first/last-frame extraction command."""

    command = ["ffmpeg", "-y"]
    if last:
        command.extend(["-sseof", "-0.1"])
    command.extend(
        [
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output),
        ]
    )
    return command


def _preflight(clips: list[dict[str, Any]]) -> Preflight:
    first = clips[0]
    hook_cost = higgsfield.get_cost(
        _generation_params(
            prompt=str(first["hook_prompt"]),
            duration_s=HOOK_DURATION_S,
        )
    )
    outro_cost = higgsfield.get_cost(
        _generation_params(
            prompt=str(first["close_prompt"]),
            duration_s=OUTRO_DURATION_S,
        )
    )
    # Cost is determined by model/duration/resolution. The measured 4-second
    # bridge cost equals the 4-second hook cost, so no media is uploaded here.
    bridge_cost = higgsfield.get_cost(
        _generation_params(
            prompt=BRIDGE_IN_PROMPT,
            duration_s=BRIDGE_DURATION_S,
        )
    )
    return calculate_preflight(
        clips,
        ShapeCosts(hook=hook_cost, outro=outro_cost, bridge=bridge_cost),
    )


def _process_clip(
    ctx: GenerateStageContext,
    manifest: dict[str, Any],
    clip: dict[str, Any],
    costs: ShapeCosts,
) -> None:
    clip_id = str(clip["id"])
    logger.info("generate[%s]: processing clip=%s", ctx.job.id, clip_id)

    with tempfile.TemporaryDirectory(
        prefix=f"{clip_id}-", dir=ctx.workspace.dir
    ) as temp_name:
        directory = Path(temp_name)
        local_assets: dict[AssetName, Path] = {
            "hook": directory / "hook.mp4",
            "outro": directory / "outro.mp4",
            "bridge_in": directory / "bridge_in.mp4",
            "bridge_out": directory / "bridge_out.mp4",
        }

        if _checkpoint(clip, "hook") is None:
            _generate_and_checkpoint(
                ctx,
                manifest,
                clip,
                asset="hook",
                params=_generation_params(
                    prompt=str(clip["hook_prompt"]),
                    duration_s=HOOK_DURATION_S,
                ),
                local_path=local_assets["hook"],
                estimated_credits=costs.hook,
            )
        else:
            logger.info(
                "generate[%s]: clip=%s asset=hook checkpoint exists; skipping",
                ctx.job.id,
                clip_id,
            )

        if _checkpoint(clip, "outro") is None:
            _generate_and_checkpoint(
                ctx,
                manifest,
                clip,
                asset="outro",
                params=_generation_params(
                    prompt=str(clip["close_prompt"]),
                    duration_s=OUTRO_DURATION_S,
                ),
                local_path=local_assets["outro"],
                estimated_credits=costs.outro,
            )
        else:
            logger.info(
                "generate[%s]: clip=%s asset=outro checkpoint exists; skipping",
                ctx.job.id,
                clip_id,
            )

        needs_bridge_in = _checkpoint(clip, "bridge_in") is None
        needs_bridge_out = _checkpoint(clip, "bridge_out") is None
        frames: dict[str, Path] = {}
        if needs_bridge_in or needs_bridge_out:
            frames = _prepare_frames(
                ctx,
                clip,
                directory,
                local_assets,
                needs_bridge_in=needs_bridge_in,
                needs_bridge_out=needs_bridge_out,
            )

        if needs_bridge_in:
            _generate_and_checkpoint(
                ctx,
                manifest,
                clip,
                asset="bridge_in",
                params=_generation_params(
                    prompt=BRIDGE_IN_PROMPT,
                    duration_s=BRIDGE_DURATION_S,
                    start_image=frames["hook_last"],
                    end_image=frames["main_first"],
                ),
                local_path=local_assets["bridge_in"],
                estimated_credits=costs.bridge,
            )
        else:
            logger.info(
                "generate[%s]: clip=%s asset=bridge_in checkpoint exists; skipping",
                ctx.job.id,
                clip_id,
            )

        if needs_bridge_out:
            _generate_and_checkpoint(
                ctx,
                manifest,
                clip,
                asset="bridge_out",
                params=_generation_params(
                    prompt=BRIDGE_OUT_PROMPT,
                    duration_s=BRIDGE_DURATION_S,
                    start_image=frames["main_last"],
                    end_image=frames["outro_first"],
                ),
                local_path=local_assets["bridge_out"],
                estimated_credits=costs.bridge,
            )
        else:
            logger.info(
                "generate[%s]: clip=%s asset=bridge_out checkpoint exists; skipping",
                ctx.job.id,
                clip_id,
            )


def _prepare_frames(
    ctx: GenerateStageContext,
    clip: dict[str, Any],
    directory: Path,
    local_assets: Mapping[AssetName, Path],
    *,
    needs_bridge_in: bool,
    needs_bridge_out: bool,
) -> dict[str, Path]:
    clip_id = str(clip["id"])
    prefix = f"pipeline/{ctx.job.id}/clips/subclips/"
    frames: dict[str, Path] = {}

    if needs_bridge_in:
        hook = _ensure_generated_asset_local(
            ctx, clip, "hook", local_assets["hook"]
        )
        first_five = _retry_once(
            lambda: ctx.workspace.download(
                f"{prefix}{clip_id}_first5.mp4",
                f"{directory.name}/{clip_id}_first5.mp4",
            ),
            description=f"clip {clip_id} main first-five S3 download",
        )
        frames["hook_last"] = directory / "hook_last.jpg"
        frames["main_first"] = directory / "main_first.jpg"
        _extract_frame_with_retry(
            hook,
            frames["hook_last"],
            last=True,
            description=f"clip {clip_id} hook last-frame extraction",
        )
        _extract_frame_with_retry(
            first_five,
            frames["main_first"],
            last=False,
            description=f"clip {clip_id} main first-frame extraction",
        )

    if needs_bridge_out:
        outro = _ensure_generated_asset_local(
            ctx, clip, "outro", local_assets["outro"]
        )
        last_five = _retry_once(
            lambda: ctx.workspace.download(
                f"{prefix}{clip_id}_last5.mp4",
                f"{directory.name}/{clip_id}_last5.mp4",
            ),
            description=f"clip {clip_id} main last-five S3 download",
        )
        frames["main_last"] = directory / "main_last.jpg"
        frames["outro_first"] = directory / "outro_first.jpg"
        _extract_frame_with_retry(
            last_five,
            frames["main_last"],
            last=True,
            description=f"clip {clip_id} main last-frame extraction",
        )
        _extract_frame_with_retry(
            outro,
            frames["outro_first"],
            last=False,
            description=f"clip {clip_id} outro first-frame extraction",
        )

    return frames


def _ensure_generated_asset_local(
    ctx: GenerateStageContext,
    clip: dict[str, Any],
    asset: AssetName,
    local_path: Path,
) -> Path:
    if local_path.exists():
        return local_path

    checkpoint = _checkpoint(clip, asset)
    if checkpoint is None:
        raise GenerateError(
            f"generate: clip {clip['id']} asset {asset} has no checkpoint"
        )
    relative_name = local_path.relative_to(ctx.workspace.dir)
    return _retry_once(
        lambda: ctx.workspace.download(
            str(checkpoint["s3Key"]), str(relative_name)
        ),
        description=f"clip {clip['id']} asset {asset} S3 download",
    )


def _generate_and_checkpoint(
    ctx: GenerateStageContext,
    manifest: dict[str, Any],
    clip: dict[str, Any],
    *,
    asset: AssetName,
    params: GenerationParams,
    local_path: Path,
    estimated_credits: int,
) -> None:
    clip_id = str(clip["id"])
    generated = _generate_with_retry(
        params,
        local_path,
        clip_id=clip_id,
        asset=asset,
    )
    s3_key = f"pipeline/{ctx.job.id}/generated/{clip_id}/{asset}.mp4"
    _retry_once(
        lambda: ctx.workspace.upload(local_path, s3_key),
        description=f"clip {clip_id} asset {asset} S3 upload",
    )

    checkpoint = {
        "s3Key": s3_key,
        "generationId": generated.generation_id,
        "estimatedCredits": estimated_credits,
        "completedAt": datetime.now(timezone.utc).isoformat(),
    }
    generated_map = clip.get("generated")
    if generated_map is None:
        generated_map = {}
        clip["generated"] = generated_map
    if not isinstance(generated_map, dict):
        raise GenerateError(f"generate: clip {clip_id} generated must be an object")
    generated_map[asset] = checkpoint

    _retry_once(
        lambda: jobs.save_manifest_checkpoint(ctx.conn, ctx.job.id, manifest),
        description=f"clip {clip_id} asset {asset} DB checkpoint",
    )
    ctx.checkpoint_heartbeat()
    logger.info(
        "generate[%s]: clip=%s asset=%s checkpointed generation=%s "
        "estimated_credits=%d",
        ctx.job.id,
        clip_id,
        asset,
        generated.generation_id,
        estimated_credits,
    )


def _generate_with_retry(
    params: GenerationParams,
    output_path: Path,
    *,
    clip_id: str,
    asset: AssetName,
) -> AssetGeneration:
    for attempt in (1, 2):
        try:
            result = higgsfield.generate(params, output_path)
            return _asset_generation(result)
        except higgsfield.HiggsfieldAmbiguousSubmitError as exc:
            raise GenerateError(
                f"generate: clip {clip_id} asset {asset} submit response was "
                "ambiguous; generation may be paid, NOT retrying; "
                f"raw_stdout={exc.raw_stdout}"
            ) from exc
        except higgsfield.HiggsfieldDownloadError as exc:
            generation_id = exc.generation_id
            if generation_id is None:
                raise GenerateError(
                    f"generate: clip {clip_id} asset {asset} completed remotely "
                    f"but generation id is missing; recover at {exc.result_url}"
                ) from exc
            try:
                higgsfield.download(exc.result_url, output_path)
            except higgsfield.HiggsfieldDownloadError as retry_exc:
                raise GenerateError(
                    f"generate: clip {clip_id} asset {asset} download retry failed; "
                    f"generation={generation_id}, result_url={exc.result_url}"
                ) from retry_exc
            return AssetGeneration(
                generation_id=generation_id,
                result_url=exc.result_url,
            )
        except Exception as exc:
            if attempt == 2:
                raise GenerateError(
                    f"generate: clip {clip_id} asset {asset} generation failed "
                    f"after retry: {exc}"
                ) from exc
            logger.warning(
                "generate: clip=%s asset=%s generation attempt 1 failed (%s); "
                "retrying once",
                clip_id,
                asset,
                exc,
            )

    raise AssertionError("unreachable")


def _asset_generation(result: GenerationResult) -> AssetGeneration:
    return AssetGeneration(
        generation_id=result.id,
        result_url=result.result_url,
    )


def _generation_params(
    *,
    prompt: str,
    duration_s: int,
    start_image: Path | None = None,
    end_image: Path | None = None,
) -> GenerationParams:
    return GenerationParams(
        model=MODEL,
        prompt=prompt,
        duration_s=duration_s,
        aspect_ratio=ASPECT_RATIO,
        resolution=RESOLUTION,
        generate_audio=False,
        start_image=start_image,
        end_image=end_image,
    )


def _checkpoint(
    clip: Mapping[str, Any], asset: AssetName
) -> dict[str, Any] | None:
    generated = clip.get("generated")
    if generated is None:
        return None
    if not isinstance(generated, dict):
        raise GenerateError(
            f"generate: clip {clip.get('id')} generated must be an object"
        )
    if asset not in generated:
        return None

    raw = generated[asset]
    if not isinstance(raw, dict):
        raise GenerateError(
            f"generate: clip {clip.get('id')} checkpoint {asset} must be an object"
        )
    checkpoint = cast(dict[str, Any], raw)
    for field in ("s3Key", "generationId", "completedAt"):
        value = checkpoint.get(field)
        if not isinstance(value, str) or not value:
            raise GenerateError(
                f"generate: clip {clip.get('id')} checkpoint {asset}.{field} "
                "must be a non-empty string"
            )
    credits = checkpoint.get("estimatedCredits")
    if isinstance(credits, bool) or not isinstance(credits, int):
        raise GenerateError(
            f"generate: clip {clip.get('id')} checkpoint "
            f"{asset}.estimatedCredits must be an integer"
        )
    return checkpoint


def _required_clip_string(
    clip: Mapping[str, object], field: str, index: int
) -> str:
    value = clip.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GenerateError(
            f"generate: manifest clips[{index}].{field} must be a non-empty string"
        )
    return value


def _extract_frame_with_retry(
    source: Path,
    output: Path,
    *,
    last: bool,
    description: str,
) -> None:
    _retry_once(
        lambda: _run_ffmpeg(build_frame_command(source, output, last=last)),
        description=description,
    )


def _run_ffmpeg(command: list[str]) -> None:
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited {process.returncode}: "
            f"{process.stderr[-FFMPEG_ERROR_TAIL:]}"
        )


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
