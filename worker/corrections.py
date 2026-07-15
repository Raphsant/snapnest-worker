"""Transcript text corrections.

AWS Transcribe reliably mishears a handful of domain terms. Rather than a full
NLP pass we keep a small, explicit substitution table and apply it with a
whole-word, case-insensitive regex so that:

  * only complete words are replaced ("primarket" -> "premarket", but
    "primarketing" is left alone),
  * the first letter's case is preserved ("Primarket" -> "Premarket").

Keys MUST be lowercase; matching is case-insensitive and the lookup lowercases
the matched word.
"""

from __future__ import annotations

import re

# Seed table. Add domain corrections here as they're discovered.
CORRECTIONS: dict[str, str] = {
    "primarket": "premarket",
}


def _build_pattern(corrections: dict[str, str]) -> re.Pattern[str]:
    if not corrections:
        # A pattern that can never match, so apply_corrections is a no-op.
        return re.compile(r"(?!)")
    # Longest keys first so multi-word/overlapping terms win before shorter ones.
    alternation = "|".join(
        sorted((re.escape(key) for key in corrections), key=len, reverse=True)
    )
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


_PATTERN = _build_pattern(CORRECTIONS)


def apply_corrections(text: str) -> str:
    """Return ``text`` with every known term replaced (whole-word, case-aware)."""

    def _replace(match: re.Match[str]) -> str:
        original = match.group(0)
        replacement = CORRECTIONS[original.lower()]
        if original[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    return _PATTERN.sub(_replace, text)
