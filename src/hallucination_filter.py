"""Post-processing to remove Whisper hallucinations from transcribed text."""

import re
from typing import List

# ---------------------------------------------------------------------------
# Known hallucination phrases
# ---------------------------------------------------------------------------
# Add new entries here to expand the filter.  The comparison is exact-string,
# so include the exact punctuation / capitalisation that Whisper produces.

KNOWN_HALLUCINATIONS: List[str] = [
    "Tack till elever och personer som hjälpte med videon!",
    "Tack för att du tittade på videon!",
    "Tack för att ni tittade på videon!",
    "Tack för att ni tittade.",
    "Tack för att du lyssnade.",
    "Tack för att du tittade!",
    "Prenumerera på kanalen!",
    "Glöm inte att prenumerera!",
    "Gilla och prenumerera!",
    "Subtitles by the Amara.org community",
    "Amara.org",
    "Translated by",
    "Transcribed by",
    "[ Silence ]",
    "[BLANK_AUDIO]",
    "[ Music ]",
    "[Music]",
    "(Music)",
    "Thank you for watching!",
    "Thanks for watching!",
    "Please like and subscribe.",
    "Don't forget to subscribe!",
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def filter_known_hallucinations(text: str) -> str:
    """Remove every known hallucination phrase from *text*."""
    for phrase in KNOWN_HALLUCINATIONS:
        text = text.replace(phrase, "")
    # Collapse runs of 3+ newlines left behind by removed phrases
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_repeated_phrases(text: str, max_repeats: int = 3) -> str:
    """Collapse runs of identical consecutive sentences that exceed *max_repeats*.

    When a sentence is repeated more than *max_repeats* times in a row the
    entire run is reduced to a single occurrence.  Runs of *max_repeats* or
    fewer are left untouched (genuine repetition in speech).
    """
    # Split on sentence endings while keeping trailing whitespace context, and
    # also on bare newlines so that multi-line output is handled correctly.
    raw_parts = re.split(r"(?<=[.!?])\s+|\n", text.strip())
    parts = [p.strip() for p in raw_parts if p.strip()]

    if not parts:
        return text

    result: List[str] = []
    i = 0
    while i < len(parts):
        current = parts[i]
        run_length = 1
        while i + run_length < len(parts) and parts[i + run_length] == current:
            run_length += 1

        if run_length > max_repeats:
            result.append(current)          # collapse to one occurrence
        else:
            result.extend([current] * run_length)  # keep as-is
        i += run_length

    return " ".join(result)


def clean_transcript(text: str) -> str:
    """Apply all hallucination filters and return cleaned text."""
    text = filter_known_hallucinations(text)
    text = remove_repeated_phrases(text)
    return text
