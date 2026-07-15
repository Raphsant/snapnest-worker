"""Curate stage.

Runs the STC Curator prompt twice (mindset, technical) via the Anthropic API
against the corrected SRT produced by ingest, validates the model's selections
against the SRT's real block indexes (the anti-hallucination fence), and stores
one curation JSON per category in the workspace and S3.

The system prompt lives at ``worker/prompts/curator_system.md`` and is loaded
byte-for-byte at runtime. It is the source of truth for the curator's rules and
must never be edited or "improved" from here.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worker.config import Config
    from worker.stages import StageContext

logger = logging.getLogger(__name__)

CATEGORIES = ("mindset", "technical")

# Anthropic call parameters.
MAX_TOKENS = 16000

# If more than this fraction of a category's clips reference indexes that don't
# exist (or are otherwise invalid), we assume the model misread the SRT and fail
# the whole job rather than store a mangled selection.
DROP_THRESHOLD = 0.30

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "curator_system.md"

_SRT_INDEX_RE = re.compile(r"-->")


class CurationError(RuntimeError):
    """Raised when a curation result fails validation and the job must fail."""


def run_curate(ctx: StageContext) -> None:
    """Curate mindset + technical clips for one job. Raises on any failure."""

    cfg = ctx.config
    ws = ctx.workspace
    job = ctx.job

    srt_key = _corrected_srt_key(ws.read_state())
    logger.info("curate[%s]: reading corrected SRT %s", job.id, srt_key)
    srt_text = ws.download(srt_key, "curate_input.srt").read_text()
    srt_indexes = parse_srt_indexes(srt_text)
    if not srt_indexes:
        raise CurationError("curate: corrected SRT contains no cue indexes")

    system_prompt = load_system_prompt()
    client = _anthropic_client(cfg)
    prefix = f"pipeline/{job.id}/"
    curation_keys: dict[str, str] = {}

    for category in CATEGORIES:
        logger.info("curate[%s]: curating %s", job.id, category)
        result = _curate_category(
            client,
            model=cfg.curator_model,
            system_prompt=system_prompt,
            srt_text=srt_text,
            category=category,
            job_id=job.id,
        )
        result["selected_clips"] = validate_curation(
            result, category=category, srt_indexes=srt_indexes, job_id=job.id
        )

        out_path = ws.path(f"curation_{category}.json")
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        key = f"{prefix}curation_{category}.json"
        ws.upload(out_path, key)
        curation_keys[category] = key
        logger.info(
            "curate[%s]: %s stored with %d clip(s) -> %s",
            job.id,
            category,
            len(result["selected_clips"]),
            key,
        )

    state = ws.read_state()
    state["curate"] = curation_keys
    ws.write_state(state)
    logger.info("curate[%s]: complete", job.id)


def load_system_prompt() -> str:
    """Load the curator system prompt byte-for-byte (never modified)."""

    return PROMPT_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested; no Anthropic/network side effects)
# --------------------------------------------------------------------------- #


def extract_json(text: str) -> dict[str, Any]:
    """Parse the JSON object embedded in model output.

    The curator emits a prose ``<transcript_review>`` section before the JSON,
    so we take the substring from the first ``{`` to the last ``}``.
    """

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model output")
    parsed: Any = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def parse_srt_indexes(srt_text: str) -> set[int]:
    """Return the set of cue index numbers present in an SRT document.

    A cue index is a line of digits immediately followed by a timestamp line
    (the one containing ``-->``).
    """

    indexes: set[int] = set()
    lines = srt_text.splitlines()
    for i, line in enumerate(lines[:-1]):
        stripped = line.strip()
        if stripped.isdigit() and _SRT_INDEX_RE.search(lines[i + 1]):
            indexes.add(int(stripped))
    return indexes


def validate_curation(
    data: dict[str, Any],
    *,
    category: str,
    srt_indexes: set[int],
    job_id: str = "",
) -> list[dict[str, Any]]:
    """Validate a curation result, dropping bad clips and failing hard when needed.

    Hard failures (raise :class:`CurationError`):
      * ``category`` doesn't match the requested one
      * ``selected_clips`` is missing/empty/not a list
      * more than ``DROP_THRESHOLD`` of the clips were invalid (model misread)
      * nothing valid survives

    Soft failures (drop the clip, log a warning): a clip whose blocks don't
    exist in the SRT, whose range is reversed, or whose id duplicates another.
    """

    if data.get("category") != category:
        raise CurationError(
            f"{category}: category mismatch (got {data.get('category')!r})"
        )

    clips = data.get("selected_clips")
    if not isinstance(clips, list) or not clips:
        raise CurationError(f"{category}: selected_clips is empty or not a list")

    valid: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    dropped = 0
    for clip in clips:
        problem = _clip_problem(clip, srt_indexes, seen_ids)
        if problem is not None:
            dropped += 1
            clip_id = clip.get("id") if isinstance(clip, dict) else clip
            logger.warning(
                "curate[%s]: skipping %s clip %r: %s",
                job_id,
                category,
                clip_id,
                problem,
            )
            continue
        seen_ids.add(str(clip["id"]))
        valid.append(clip)

    total = len(clips)
    if dropped / total > DROP_THRESHOLD:
        raise CurationError(
            f"{category}: dropped {dropped}/{total} clips "
            f"(> {DROP_THRESHOLD:.0%}); model likely misread the SRT"
        )
    if not valid:
        raise CurationError(f"{category}: no valid clips after validation")
    return valid


def _clip_problem(
    clip: Any, srt_indexes: set[int], seen_ids: set[str]
) -> str | None:
    if not isinstance(clip, dict):
        return "not an object"
    clip_id = clip.get("id")
    if clip_id is None:
        return "missing id"
    start = clip.get("start_block")
    end = clip.get("end_block")
    # bool is a subclass of int; exclude it so True/False aren't treated as blocks.
    if not _is_int(start) or not _is_int(end):
        return "start_block/end_block are not integers"
    if start not in srt_indexes:
        return f"start_block {start} not in SRT"
    if end not in srt_indexes:
        return f"end_block {end} not in SRT"
    if start > end:
        return f"reversed range {start} > {end}"
    if str(clip_id) in seen_ids:
        return f"duplicate id {clip_id!r}"
    return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _corrected_srt_key(state: dict[str, Any]) -> str:
    ingest_state = state.get("ingest", {})
    key = ingest_state.get("corrected_srt_key") if isinstance(ingest_state, dict) else None
    if not isinstance(key, str) or not key:
        raise CurationError(
            "curate: corrected SRT key missing from workspace state (run ingest first)"
        )
    return key


# --------------------------------------------------------------------------- #
# Anthropic (thin, side-effecting wrappers)
# --------------------------------------------------------------------------- #

# The Anthropic client is typed as Any for the same reason as the Transcribe
# client in ingest: it keeps strict mypy clean without casting request payloads
# to the SDK's generated TypedDicts. The official SDK is still what runs.


def _anthropic_client(cfg: Config) -> Any:
    import anthropic

    return anthropic.Anthropic(api_key=cfg.anthropic_api_key)


def _curate_category(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    srt_text: str,
    category: str,
    job_id: str,
) -> dict[str, Any]:
    """Call the curator for one category, with a single JSON-repair retry."""

    user = f"CATEGORY: {category}\n\n{srt_text}"

    first = _create_message(client, model=model, system=system_prompt, messages=[
        {"role": "user", "content": user},
    ])
    _log_usage(first, category=category, job_id=job_id, attempt=1)
    first_text = _message_text(first)
    try:
        return extract_json(first_text)
    except ValueError:
        logger.warning(
            "curate[%s]: %s output was not valid JSON; retrying once",
            job_id,
            category,
        )

    second = _create_message(client, model=model, system=system_prompt, messages=[
        {"role": "user", "content": user},
        {"role": "assistant", "content": first_text},
        {
            "role": "user",
            "content": "Your previous output was not valid JSON. Output the JSON only.",
        },
    ])
    _log_usage(second, category=category, job_id=job_id, attempt=2)
    # A second failure raises ValueError -> the job fails (better than garbage).
    return extract_json(_message_text(second))


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


def _log_usage(message: Any, *, category: str, job_id: str, attempt: int) -> None:
    usage = getattr(message, "usage", None)
    if usage is None:
        return
    logger.info(
        "curate[%s]: %s attempt %d tokens in=%s out=%s",
        job_id,
        category,
        attempt,
        getattr(usage, "input_tokens", "?"),
        getattr(usage, "output_tokens", "?"),
    )
