"""
followup.py — grounded follow-up selection (ported from the legacy main.py
`select_discriminating_symptoms`, now operating over heading/cause/question).

Core concept preserved from the AI-Search main.py: when the retrieval spread is
'high' (no clear winner), we pick the field phrases that BEST (a) match the
user's description and (b) discriminate between the top candidates, so the agent
can ask a targeted question instead of guessing. The only change is the *source*
of phrases: the production docs have no "Symptoms" field, so the phrases come
from each candidate's user-experience/question and cause text (falling back to
the title when a doc is title-only).

Scoring per phrase (identical formula to the legacy version):
    rel   = cosine(description, phrase)      over stopword-filtered, stemmed TF vectors
    mass  = share of candidate score-mass carrying the phrase
    dist  = 1 - |2*mass - 1|                 (peaks when ~half the mass carries it)
    final = 0.5*rel + 0.5*dist

If the best phrase's `final` is below MIN_FIELD_SCORE (i.e. nothing is relevant
enough to the user's words), we return [] — and the endpoint hands the agent the
freedom to ask its own focused Outlook follow-up. This is exactly the
"relevant -> grounded question, else freeform" behaviour requested.
"""
from __future__ import annotations

import math

try:
    from text_utils import normalize
except ImportError:  # uvicorn app.main:app from repo root
    from app.text_utils import normalize

import config


def _tf_vector(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for t in normalize(text):
        counts[t] = counts.get(t, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(a.get(k, 0.0) * v for k, v in b.items())


def _candidate_phrases(cand: dict) -> list[str]:
    """The phrases a follow-up question can be grounded in for this candidate:
    its question + cause; or just the title if the doc is title-only."""
    phrases = [p.strip() for p in (cand.get("question"), cand.get("cause")) if p and p.strip()]
    if not phrases and cand.get("title"):
        phrases = [cand["title"].strip()]
    return phrases


def select_discriminating_fields(
    description: str,
    candidates: list[dict],
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[str]:
    """Return up to `top_k` grounded follow-up phrases (best-first); [] if none
    clear the relevance floor (-> agent asks its own follow-up)."""
    top_k = top_k or config.TOP_FIELDS_K
    min_score = config.MIN_FIELD_SCORE if min_score is None else min_score
    if not candidates:
        return []

    # phrase -> list of carrying candidate scores (for the distinctiveness term)
    sources: dict[str, list[float]] = {}
    for c in candidates:
        score = float(c.get("score", c.get("cos", 0.0)))
        for phrase in _candidate_phrases(c):
            sources.setdefault(phrase, []).append(score)
    if not sources:
        return []

    total = sum(float(c.get("score", c.get("cos", 0.0))) for c in candidates) or 1.0
    dvec = _tf_vector(description)

    scored: list[tuple[float, str]] = []
    for phrase, carriers in sources.items():
        rel = _cosine(dvec, _tf_vector(phrase))
        mass = sum(carriers) / total
        dist = 1.0 - abs(2.0 * mass - 1.0)
        scored.append((rel * 0.5 + dist * 0.5, phrase))
    scored.sort(key=lambda x: -x[0])

    selected: list[str] = []
    vecs: list[dict[str, float]] = []
    for final, phrase in scored:
        if final < min_score:          # sorted desc — first miss ends it
            break
        v = _tf_vector(phrase)
        if any(_cosine(v, sv) > 0.85 for sv in vecs):   # drop near-duplicates
            continue
        selected.append(phrase)
        vecs.append(v)
        if len(selected) >= top_k:
            break
    return selected
