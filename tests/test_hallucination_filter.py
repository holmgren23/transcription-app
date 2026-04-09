"""Unit tests for hallucination_filter module."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hallucination_filter import (
    KNOWN_HALLUCINATIONS,
    clean_transcript,
    filter_known_hallucinations,
    remove_repeated_phrases,
)


# ---------------------------------------------------------------------------
# filter_known_hallucinations
# ---------------------------------------------------------------------------

def test_removes_known_swedish_hallucination():
    text = "Tack till elever och personer som hjälpte med videon!"
    assert filter_known_hallucinations(text) == ""


def test_removes_hallucination_embedded_in_real_text():
    text = "Här är en föreläsning. Tack till elever och personer som hjälpte med videon! Slut."
    result = filter_known_hallucinations(text)
    assert "Tack till elever" not in result
    assert "Här är en föreläsning" in result
    assert "Slut" in result


def test_removes_blank_audio_marker():
    text = "Studenten sa: [BLANK_AUDIO] och sedan fortsatte."
    result = filter_known_hallucinations(text)
    assert "[BLANK_AUDIO]" not in result


def test_removes_multiple_hallucinations_in_one_pass():
    text = (
        "Föreläsning börjar. "
        "Tack för att du tittade på videon! "
        "[BLANK_AUDIO] "
        "Prenumerera på kanalen! "
        "Text från videon."
    )
    result = filter_known_hallucinations(text)
    assert "Tack för att du tittade på videon!" not in result
    assert "[BLANK_AUDIO]" not in result
    assert "Prenumerera på kanalen!" not in result
    assert "Text från videon." in result


def test_real_text_unchanged():
    text = "Professor Andersson förklarade kvantmekanik idag."
    assert filter_known_hallucinations(text) == text


def test_known_hallucinations_list_is_nonempty():
    assert len(KNOWN_HALLUCINATIONS) > 0


# ---------------------------------------------------------------------------
# remove_repeated_phrases
# ---------------------------------------------------------------------------

def test_phrase_repeated_over_limit_collapsed_to_one():
    """4 identical sentences → collapsed to 1 (4 > 3)."""
    text = "Hej. Hej. Hej. Hej."
    result = remove_repeated_phrases(text, max_repeats=3)
    assert result == "Hej."


def test_phrase_repeated_exactly_at_limit_kept():
    """3 identical sentences → kept as-is (3 == max_repeats, not exceeded)."""
    text = "Hej. Hej. Hej."
    result = remove_repeated_phrases(text, max_repeats=3)
    assert result.count("Hej.") == 3


def test_phrase_repeated_under_limit_kept():
    """2 identical sentences → kept as-is (2 < 3)."""
    text = "Hej. Hej."
    result = remove_repeated_phrases(text, max_repeats=3)
    assert result.count("Hej.") == 2


def test_mixed_sentences_only_long_run_collapsed():
    """Only runs exceeding max_repeats are collapsed; others kept intact."""
    text = "A. A. A. A. B. B. C."
    result = remove_repeated_phrases(text, max_repeats=3)
    assert result.count("A.") == 1  # 4 → 1
    assert result.count("B.") == 2  # 2 stays
    assert result.count("C.") == 1  # 1 stays


def test_no_repetition_returns_original_structure():
    text = "Idag pratar vi om fysik. Sedan matematik. Avslutningsvis kemi."
    result = remove_repeated_phrases(text)
    assert "fysik" in result
    assert "matematik" in result
    assert "kemi" in result


def test_empty_string_returns_empty():
    assert remove_repeated_phrases("") == ""


def test_single_sentence_unchanged():
    text = "En mening."
    result = remove_repeated_phrases(text)
    assert result == "En mening."


def test_five_repeats_with_custom_max():
    """5 repeats with max_repeats=2 → collapsed to 1."""
    text = "OK! OK! OK! OK! OK!"
    result = remove_repeated_phrases(text, max_repeats=2)
    assert result.count("OK!") == 1


def test_whisper_hallucination_repetition_pattern():
    """Typical Whisper looping hallucination is collapsed."""
    phrase = "Tack för att ni tittade."
    text = " ".join([phrase] * 6)
    result = remove_repeated_phrases(text, max_repeats=3)
    assert result.count(phrase) == 1


# ---------------------------------------------------------------------------
# clean_transcript (combined pipeline)
# ---------------------------------------------------------------------------

def test_clean_transcript_removes_known_phrase_and_collapses_repeats():
    text = (
        "Föreläsning om kemi. "
        "Tack till elever och personer som hjälpte med videon! "
        "Kemi är kul. Kemi är kul. Kemi är kul. Kemi är kul."
    )
    result = clean_transcript(text)
    assert "Tack till elever" not in result
    assert "Föreläsning om kemi." in result
    assert result.count("Kemi är kul.") == 1


def test_clean_transcript_passthrough_for_normal_text():
    text = "En normal föreläsning utan repetitioner eller hallucinationer."
    result = clean_transcript(text)
    assert result == text
