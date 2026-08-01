"""Generation-prompt lint: reject logo/brand/text language before Higgsfield.

CLAUDE.md iron rule #1: AI stages generate CLEAN footage; the deterministic
assembler applies ALL branding, text overlays, and captions. Generation prompts
must therefore never request logos, brand marks, or rendered text — doing so is
what makes generated outros hallucinate fake logos. This module is the
enforcement layer, wired at two choke points: the creative stage surfaces
violations in the manifest for operator review, and the generate stage
hard-fails the job before any credit-bearing Higgsfield call.

Pure functions only — no DB, S3, Anthropic, or Higgsfield side effects.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# Only generation-prompt fields are linted. hook_text / close_text are overlay
# copy consumed by the assembler's drawtext — they are EXEMPT (the words belong
# there, never inside a generation prompt).
GENERATION_PROMPT_FIELDS: tuple[str, ...] = ("hook_prompt", "close_prompt")

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


def lint_prompts(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return blocklist violations in approved clips' generation-prompt fields.

    Each violation is ``{clipId, field, matched_word, prompt_excerpt}``. Only
    approved clips are scanned, and only ``GENERATION_PROMPT_FIELDS`` — the
    hook_text / close_text overlay copy is intentionally exempt. A structurally
    unusable manifest (no clips list) yields no violations; structural
    validation is the calling stage's responsibility, not the lint's.
    """

    clips = manifest.get("clips")
    if not isinstance(clips, list):
        return []

    violations: list[dict[str, str]] = []
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
            for matched_word, excerpt in _scan(value):
                violations.append(
                    {
                        "clipId": clip_id,
                        "field": field,
                        "matched_word": matched_word,
                        "prompt_excerpt": excerpt,
                    }
                )
    return violations


def _scan(prompt: str) -> list[tuple[str, str]]:
    """Return (matched_word, excerpt) once per distinct blocklisted word.

    Repeated hits on the same word (case-insensitively) collapse to their first
    occurrence, so a prompt that says "logo" three times reports one violation.
    """

    found: dict[str, tuple[str, str]] = {}
    for match in _BLOCKLIST_RE.finditer(prompt):
        key = match.group(0).lower()
        if key not in found:
            found[key] = (
                match.group(0),
                _excerpt(prompt, match.start(), match.end()),
            )
    return list(found.values())


def _excerpt(prompt: str, start: int, end: int) -> str:
    """Slice a single-line window around a match, marking any truncation."""

    left = max(0, start - EXCERPT_RADIUS)
    right = min(len(prompt), end + EXCERPT_RADIUS)
    snippet = " ".join(prompt[left:right].split())
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(prompt) else ""
    return f"{prefix}{snippet}{suffix}"
