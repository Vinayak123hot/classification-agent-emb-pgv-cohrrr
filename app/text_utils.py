"""
text_utils.py — the shared text-normalization pipeline.

One canonical path used at BOTH index time and query time (symmetry matters
for BM25): lower-case → fold known multi-word phrases → tokenize → drop
stopwords/short tokens → Porter-stem.

`nltk`'s PorterStemmer is used for stemming (so "crashes"→"crash",
"loading"→"load", "notifications"→"notif") — it needs no downloaded corpus,
so it is safe to run on a bare App Service. If nltk is somehow unavailable we
fall back to a tiny suffix stemmer so the service still starts.
"""
from __future__ import annotations

import re

try:
    from nltk.stem import PorterStemmer
    _porter = PorterStemmer()

    def _stem(tok: str) -> str:
        return _porter.stem(tok)
except Exception:  # pragma: no cover - fallback only
    def _stem(tok: str) -> str:
        for suf in ("ing", "ed", "es", "s"):
            if len(tok) > len(suf) + 2 and tok.endswith(suf):
                return tok[: -len(suf)]
        return tok


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
    "issue", "problem", "outlook",   # 'outlook' is in every doc → zero signal
}

# Multi-word phrases folded to a single token BEFORE tokenizing, so the
# synonym/keyword layer can treat them atomically. Order matters (longest
# first). Keep this list small and high-value.
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


def normalize(text: str) -> list[str]:
    """Full pipeline → list of stemmed tokens (stopwords/short removed)."""
    if not text:
        return []
    s = text.lower()
    for pat, repl in PHRASES:
        s = re.sub(pat, repl, s)
    out: list[str] = []
    for tok in _TOKEN_RE.findall(s):
        if len(tok) <= 2 or tok in STOPWORDS:
            continue
        out.append(_stem(tok))
    return out
