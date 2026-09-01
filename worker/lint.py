"""Manifest lint: prompt purity (legacy) and library selection validity.

CLAUDE.md iron rule #1: AI stages generate CLEAN footage; the deterministic
assembler applies ALL branding, text overlays, and captions. Generation prompts
must therefore never request logos, brand marks, or rendered text — doing so is
what makes generated outros hallucinate fake logos. This module is the
enforcement layer, wired at two choke points: the creative stage surfaces
violations in the manifest for operator review, and the generate stage
hard-fails the job before any credit-bearing Higgsfield call.

Negation-aware (v1.1): a blocklisted word used inside a prohibition ("no logos",
"free of text") is a non-blocking WARNING, not a violation — so prompts that
describe cleanliness by naming what to avoid don't hard-fail. "STC" is the
exception: the letters appearing at all is the violation, even negated.

Selection lint (v2): creative selects pre-generated library assets
(hook_asset_id/outro_asset_id) instead of writing prompts. Operators can edit
the manifest between the creative gate and generation, so ``lint_selections``
re-validates every approved clip's selections against the catalog at both
choke points.

Pure functions only — no DB, S3, Anthropic, or Higgsfield side effects.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from worker.library import LibraryCatalog

# Only generation-prompt fields are linted. hook_text / close_text are overlay
# copy consumed by the assembler's drawtext — they are EXEMPT (the words belong
# there, never inside a generation prompt).
GENERATION_PROMPT_FIELDS: tuple[str, ...] = ("hook_prompt", "close_prompt")

# Library selection fields written by creative v2, with the asset type each
# must resolve to in the catalog.
SELECTION_FIELDS: tuple[tuple[str, str], ...] = (
    ("hook_asset_id", "hook"),
    ("outro_asset_id", "outro"),
)

# Words that request rendered branding or on-screen text. This is the one
# obvious place to extend the policy — add a token and it is enforced at both
# choke points. Plurals are listed explicitly because word-boundary matching
# treats "logo" and "logos" as distinct tokens. Matched case-insensitively on
# word boundaries, so "text" fires on "Text overlay" but never inside
# "context" or "texture", and "STC" fires in any casing but not inside "STCorp".
BLOCKLIST: tuple[str, ...] = (
    "logo",
    "logos",
    "logotype",
    "brand",
    "branding",
    "brandmark",
    "wordmark",
    "watermark",
    "emblem",
    "badge",
    "insignia",
    "lettering",
    "typography",
    "typeface",
    "font",
    "caption",
    "captions",
    "subtitle",
    "subtitles",
    "title",
    "titles",
    "text",
    "STC",
)

# Characters of surrounding context to include on each side of a match so the
# operator can see where the offending word landed in the prompt.
EXCERPT_RADIUS = 40

_BLOCKLIST_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in BLOCKLIST) + r")\b",
    re.IGNORECASE,
)

# Negation cues that flip a blocklist hit from a request into a prohibition. A
# hit is NEGATED when one of these appears within NEGATION_LOOKBACK_WORDS words
# before it, in the same sentence: "no logos", "without any text", "free of
# branding". Negated hits are warnings, never blocks — EXCEPT "STC", whose mere
# appearance is the violation in any polarity.
_NEGATION_WORDS: frozenset[str] = frozenset(
    {"no", "not", "without", "never", "avoid"}
)
_NEGATION_PHRASES: frozenset[tuple[str, str]] = frozenset(
    {("free", "of"), ("instead", "of")}
)
NEGATION_LOOKBACK_WORDS = 3

# Sentence boundaries so a cue can't leak across "." / "!" / "?" / newline, and
# word tokens for the small look-back window before a match.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]")
_WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class LintResult:
    """Split lint outcome for approved clips' generation prompts.

    ``violations`` block: naked blocklist hits, plus any "STC" in any polarity.
    ``warnings`` are negated hits — a blocklisted word inside a prohibition ("no
    logos", "free of text") — surfaced for review but never blocking; each has
    the same shape as a violation plus ``negated: True``.
    """

    violations: list[dict[str, str]]
    warnings: list[dict[str, Any]]


def lint_prompts(manifest: Mapping[str, Any]) -> LintResult:
    """Return blocking violations and negated warnings for approved clips' prompts.

    A blocking ``violation`` is ``{clipId, field, matched_word, prompt_excerpt}``
    (naked blocklist hits, plus any "STC"); the generate stage hard-fails on
    these. A ``warning`` has the same shape plus ``negated: True`` (a blocklisted
    word used inside a prohibition); it is surfaced but never blocks. Only
    approved clips and only ``GENERATION_PROMPT_FIELDS`` are scanned — the
    hook_text / close_text overlay copy is intentionally exempt. A structurally
    unusable manifest (no clips list) yields an empty result; structural
    validation is the calling stage's responsibility, not the lint's.
    """

    clips = manifest.get("clips")
    if not isinstance(clips, list):
        return LintResult(violations=[], warnings=[])

    violations: list[dict[str, str]] = []
    warnings: list[dict[str, Any]] = []
    for clip in clips:
        if not isinstance(clip, Mapping):
            continue
        if clip.get("approved") is not True:
            continue
        clip_id = str(clip.get("id", "<unknown>"))
        for field in GENERATION_PROMPT_FIELDS:
            value = clip.get(field)
            if not isinstance(value, str):
                continue
            blocking, negated = _scan(value)
            for matched_word, excerpt in blocking:
                violations.append(
                    {
                        "clipId": clip_id,
                        "field": field,
                        "matched_word": matched_word,
                        "prompt_excerpt": excerpt,
                    }
                )
            for matched_word, excerpt in negated:
                warnings.append(
                    {
                        "clipId": clip_id,
                        "field": field,
                        "matched_word": matched_word,
                        "prompt_excerpt": excerpt,
                        "negated": True,
                    }
                )
    return LintResult(violations=violations, warnings=warnings)


def lint_selections(
    manifest: Mapping[str, Any], catalog: LibraryCatalog
) -> LintResult:
    """Validate approved clips' library selections against the catalog.

    Same result shape as ``lint_prompts``, with entries of
    ``{clipId, field, reason}``. Violations (the generate stage hard-fails on
    these): a missing/empty selection field, an id absent from the catalog, or
    an id of the wrong type. Warnings (recorded only): the clip's category not
    being in the selected asset's category list — universal assets
    legitimately cross categories — and the same id selected by more than one
    approved clip, which is allowed reuse rather than a defect (every clip
    involved is still reported, hook and outro fields checked jointly).
    Unapproved clips are ignored entirely; a structurally unusable manifest
    yields an empty result, as in ``lint_prompts``.
    """

    clips = manifest.get("clips")
    if not isinstance(clips, list):
        return LintResult(violations=[], warnings=[])

    violations: list[dict[str, str]] = []
    warnings: list[dict[str, Any]] = []
    selected_by: dict[str, list[tuple[str, str]]] = {}
    for clip in clips:
        if not isinstance(clip, Mapping):
            continue
        if clip.get("approved") is not True:
            continue
        clip_id = str(clip.get("id", "<unknown>"))
        category = clip.get("category")
        for field, expected_type in SELECTION_FIELDS:
            value = clip.get(field)
            if not isinstance(value, str) or not value.strip():
                violations.append(
                    {
                        "clipId": clip_id,
                        "field": field,
                        "reason": f"{field} must be a non-empty asset id",
                    }
                )
                continue
            asset = catalog.get(value)
            if asset is None:
                violations.append(
                    {
                        "clipId": clip_id,
                        "field": field,
                        "reason": (
                            f"asset id {value!r} does not exist in the "
                            "library catalog"
                        ),
                    }
                )
                continue
            if asset.type != expected_type:
                violations.append(
                    {
                        "clipId": clip_id,
                        "field": field,
                        "reason": (
                            f"asset {value!r} has type {asset.type!r}, "
                            f"expected {expected_type!r}"
                        ),
                    }
                )
                continue
            selected_by.setdefault(value, []).append((clip_id, field))
            if isinstance(category, str) and category not in asset.category:
                warnings.append(
                    {
                        "clipId": clip_id,
                        "field": field,
                        "reason": (
                            f"clip category {category!r} not in asset "
                            f"{value!r} categories {list(asset.category)}"
                        ),
                    }
                )

    # Reuse within a job is allowed policy: selection prefers the least-used
    # asset among comparable fits, so repetition is worth recording but never
    # blocks. Every clip sharing the asset is still named.
    for asset_id, users in selected_by.items():
        if len(users) < 2:
            continue
        usage = ", ".join(f"{c}.{f}" for c, f in users)
        for clip_id, field in users:
            warnings.append(
                {
                    "clipId": clip_id,
                    "field": field,
                    "reason": (
                        f"asset {asset_id!r} selected by multiple approved "
                        f"clips ({usage})"
                    ),
                }
            )
    return LintResult(violations=violations, warnings=warnings)


def _scan(prompt: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split a prompt's distinct blocklist hits into (blocking, negated) matches.

    One entry per distinct blocklisted word (case-insensitively). A word blocks
    if ANY of its occurrences is naked — so a naked request is never masked by an
    earlier negated mention of the same word — and is a negated warning only if
    EVERY occurrence is negated. "STC" always blocks, in any polarity: the
    letters appearing at all is the violation, even in "no STC".
    """

    blocking: dict[str, tuple[str, str]] = {}
    negated: dict[str, tuple[str, str]] = {}
    for match in _BLOCKLIST_RE.finditer(prompt):
        key = match.group(0).lower()
        entry = (match.group(0), _excerpt(prompt, match.start(), match.end()))
        if key != "stc" and _is_negated(prompt, match.start()):
            # Only record as negated if no naked hit for this word blocks already.
            if key not in blocking:
                negated.setdefault(key, entry)
        else:
            # Naked (or STC): this word blocks; drop any earlier negated entry.
            negated.pop(key, None)
            blocking.setdefault(key, entry)
    return list(blocking.values()), list(negated.values())


def _is_negated(prompt: str, start: int) -> bool:
    """True if a negation cue sits within NEGATION_LOOKBACK_WORDS words before the
    match, without crossing a sentence boundary."""

    sentence = _SENTENCE_SPLIT_RE.split(prompt[:start])[-1]
    window = [word.lower() for word in _WORD_RE.findall(sentence)]
    window = window[-NEGATION_LOOKBACK_WORDS:]
    if any(word in _NEGATION_WORDS for word in window):
        return True
    return any(pair in _NEGATION_PHRASES for pair in zip(window, window[1:]))


def _excerpt(prompt: str, start: int, end: int) -> str:
    """Slice a single-line window around a match, marking any truncation."""

    left = max(0, start - EXCERPT_RADIUS)
    right = min(len(prompt), end + EXCERPT_RADIUS)
    snippet = " ".join(prompt[left:right].split())
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(prompt) else ""
    return f"{prefix}{snippet}{suffix}"
