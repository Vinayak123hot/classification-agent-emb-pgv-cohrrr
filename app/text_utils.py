"""
text_utils.py — the shared text-normalization pipeline, as a class.

`TextNormalizer.normalize()` is one canonical path used wherever we compare the
user's words to a KB phrase (the follow-up relevance scoring): lower-case -> fold
known multi-word phrases -> tokenize -> drop stopwords/short tokens -> Porter-stem.

A module-level `NORMALIZER` singleton is provided for convenience.
"""
from __future__ import annotations

import re


class TextNormalizer:
    """Turns free text into a list of comparable stemmed tokens."""

    # Common English + Outlook-support filler that carries no matching signal.
    STOPWORDS = {
        "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or", "for",
        "with", "this", "that", "was", "are", "be", "at", "by", "has", "have",
        "had", "not", "but", "from", "as", "do", "did", "will", "would", "can",
        "could", "my", "your", "their", "its", "i", "you", "he", "she", "they",
        "we", "our", "no", "so", "if", "when", "how", "what", "which", "where",
        "there", "been", "more", "also", "than", "then", "all", "any", "some",
        "were", "should", "may", "might", "over", "after", "before", "get", "gets",
        "getting", "im", "ive", "please", "help", "need", "want", "trying", "tried",
        "issue", "problem", "outlook",   # 'outlook' is in every doc -> zero signal
    }

    # Multi-word phrases folded to a single token BEFORE tokenizing (longest first).
    PHRASES = [
        (r"send\s*/?\s*receive", " sendreceive "),
        (r"working\s+offline", " offline "),
        (r"work\s+offline", " offline "),
        (r"go(ing)?\s+offline", " offline "),
        (r"add[\s\-]?in(s)?", " addin "),
        (r"report\s+spam", " phishing "),
        (r"task\s+manager", " taskmanager "),
        (r"(can\s*'?t|cannot|can\s*not|wo\s*n\s*'?t|will\s+not|does\s*n\s*'?t|do\s+not|fail(s|ed)?\s+to)\s+(open|launch|start)", " wontopen "),
        (r"not\s+respond(ing)?", " unresponsive "),
        (r"loading\s+screen", " loadingscreen "),
    ]

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self):
        # Prefer nltk's PorterStemmer (needs no downloaded corpus). If nltk is
        # unavailable, fall back to a tiny suffix stemmer so the service still starts.
        try:
            from nltk.stem import PorterStemmer
            self._porter = PorterStemmer()
            self._use_porter = True
        except Exception:                       # pragma: no cover - fallback only
            self._porter = None
            self._use_porter = False

    def _stem(self, tok: str) -> str:
        """Reduce a token to its stem ('crashes' -> 'crash')."""
        if self._use_porter:
            return self._porter.stem(tok)
        for suf in ("ing", "ed", "es", "s"):    # crude fallback stemmer
            if len(tok) > len(suf) + 2 and tok.endswith(suf):
                return tok[: -len(suf)]
        return tok

    def normalize(self, text: str) -> list[str]:
        """Full pipeline -> list of stemmed tokens (stopwords/short removed)."""
        if not text:
            return []
        s = text.lower()                        # 1) case-fold
        for pat, repl in self.PHRASES:          # 2) fold known multi-word phrases
            s = re.sub(pat, repl, s)
        out: list[str] = []
        for tok in self._TOKEN_RE.findall(s):   # 3) tokenize
            if len(tok) <= 2 or tok in self.STOPWORDS:   # 4) drop short/stopwords
                continue
            out.append(self._stem(tok))         # 5) stem
        return out


# Default singleton (stateless apart from the stemmer handle).
NORMALIZER = TextNormalizer()
