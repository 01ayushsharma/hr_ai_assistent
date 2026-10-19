"""Deterministic detection of a clear confirmation ("yes", "confirm", "go ahead").

Only a short message made entirely of confirmation words counts. Anything
else ("yes but make it Friday", "no", a new question) is not a confirmation,
so the pending proposal is discarded and the message is handled normally.
"""
from __future__ import annotations

import re

_ALLOWED = {
    "yes", "yeah", "yep", "yup", "y", "sure", "ok", "okay", "k", "confirm", "confirmed", "go", "ahead",
    "please", "do", "it", "proceed", "correct", "thats", "right", "sounds", "good", "absolutely",
    "thanks", "thank", "you", "submit", "book", "fine", "perfect", "great", "affirmative", "i", "that",
    "is", "please", "definitely", "alright", "all", "cancel",
}
_TRIGGERS = {"yes", "yeah", "yep", "yup", "y", "sure", "ok", "okay", "confirm", "confirmed", "proceed",
             "submit", "absolutely", "affirmative", "definitely", "alright"}
_TRIGGER_PHRASES = ("go ahead", "do it", "book it", "sounds good", "that's right", "thats right")
_NEGATIONS = {"no", "not", "dont", "don't", "nope", "wait", "stop", "but", "instead", "change", "actually"}


def is_confirmation(message: str) -> bool:
    text = message.lower().replace("’", "'").strip()
    words = re.findall(r"[a-z']+", text)
    if not words or len(words) > 8:
        return False
    if any(w in _NEGATIONS for w in words) or re.search(r"\d", text):
        return False
    normalised = [w.replace("'", "") for w in words]
    if not all(w in _ALLOWED for w in normalised):
        return False
    joined = " ".join(words)
    return any(w in _TRIGGERS for w in normalised) or any(p in joined for p in _TRIGGER_PHRASES)
