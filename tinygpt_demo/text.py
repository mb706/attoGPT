from __future__ import annotations

import re


WORD_RE = re.compile(r"[a-z]+|[.,!?]")


def normalize_text(text: str, mode: str = "simple") -> str:
    """Normalize TinyStories-style prose into a smaller English alphabet."""
    text = text.lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if mode == "simple":
        text = re.sub(r"[^a-z.,!?\n ]+", " ", text)
    elif mode == "period_only":
        text = re.sub(r"[!?]+", ".", text)
        text = re.sub(r"[,]+", " ", text)
        text = re.sub(r"[^a-z.\n ]+", " ", text)
    elif mode == "apostrophe":
        text = re.sub(r"[^a-z.,!?\n' ]+", " ", text)
        text = re.sub(r"\s+'\s+|\s+'|'\s+", " ", text)
    else:
        raise ValueError(f"unknown text normalization mode: {mode}")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)
