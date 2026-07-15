from __future__ import annotations

from worker.corrections import CORRECTIONS, apply_corrections


def test_seed_correction_present() -> None:
    assert CORRECTIONS["primarket"] == "premarket"


def test_replaces_whole_lowercase_word() -> None:
    assert apply_corrections("the primarket is open") == "the premarket is open"


def test_preserves_leading_capital() -> None:
    assert apply_corrections("Primarket rally today") == "Premarket rally today"


def test_leaves_substring_untouched() -> None:
    # "primarketing" contains "primarket" but is a different word.
    assert apply_corrections("primarketing strategy") == "primarketing strategy"


def test_handles_multiple_occurrences_with_mixed_case() -> None:
    assert (
        apply_corrections("primarket and Primarket")
        == "premarket and Premarket"
    )


def test_respects_word_boundary_with_punctuation() -> None:
    assert apply_corrections("up in primarket, then...") == "up in premarket, then..."


def test_unrelated_text_unchanged() -> None:
    assert apply_corrections("no corrections here") == "no corrections here"


def test_does_not_touch_mask_token() -> None:
    assert apply_corrections("say *** loudly") == "say *** loudly"
