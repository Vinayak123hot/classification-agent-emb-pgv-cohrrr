"""
followup.py — grounded follow-up selection, as a class.

`FollowupSelector.select()` picks the heading/cause/question phrases that best
(a) match the user's description and (b) discriminate between the top candidates,
so the agent can ask a targeted question instead of guessing. If nothing clears
the relevance floor it returns [] and the agent asks its own freeform follow-up.

Scoring per phrase (ported from the legacy AI-Search main.py):
    rel   = cosine(description, phrase)   over stemmed TF vectors
    mass  = share of candidate score carrying the phrase
    dist  = 1 - |2*mass - 1|              (peaks when ~half the mass carries it)
    final = 0.5*rel + 0.5*dist
"""
from __future__ import annotations

import math


class FollowupSelector:
    def __init__(self, config, normalizer):
        self._config = config          # TOP_FIELDS_K / MIN_FIELD_SCORE
        self._normalizer = normalizer  # TextNormalizer

    # ── tiny TF-cosine helpers (stemmed bag-of-words) ─────────────────────────
    def _tf_vector(self, text: str) -> dict[str, float]:
        """L2-normalized term-frequency vector over normalized tokens."""
        counts: dict[str, float] = {}
        for t in self._normalizer.normalize(text):
            counts[t] = counts.get(t, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return {k: v / norm for k, v in counts.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        """Cosine similarity of two TF vectors (dot product of unit vectors)."""
        return sum(a.get(k, 0.0) * v for k, v in b.items())

    @staticmethod
    def _candidate_phrases(cand: dict) -> list[str]:
        """The phrases a follow-up can be grounded in for this candidate:
        its question + cause; or just the title if the doc is title-only."""
        phrases = [p.strip() for p in (cand.get("question"), cand.get("cause"))
                   if p and p.strip()]
        if not phrases and cand.get("title"):
            phrases = [cand["title"].strip()]
        return phrases

    # ── public API ────────────────────────────────────────────────────────────
    def select(self, description: str, candidates: list[dict],
               top_k: "int | None" = None, min_score: "float | None" = None) -> list[str]:
        """Return up to `top_k` grounded follow-up phrases (best-first); [] if none
        clear the relevance floor (-> agent asks its own follow-up)."""
        top_k = top_k or self._config.TOP_FIELDS_K
        min_score = self._config.MIN_FIELD_SCORE if min_score is None else min_score
        if not candidates:
            return []

        # phrase -> list of carrying-candidate scores (for the distinctiveness term)
        sources: dict[str, list[float]] = {}
        for c in candidates:
            score = float(c.get("score", c.get("cos", 0.0)))
            for phrase in self._candidate_phrases(c):
                sources.setdefault(phrase, []).append(score)
        if not sources:
            return []

        total = sum(float(c.get("score", c.get("cos", 0.0))) for c in candidates) or 1.0
        dvec = self._tf_vector(description)

        scored: list[tuple[float, str]] = []
        for phrase, carriers in sources.items():
            rel = self._cosine(dvec, self._tf_vector(phrase))   # relevance to the user's words
            mass = sum(carriers) / total                        # share of score carrying it
            dist = 1.0 - abs(2.0 * mass - 1.0)                  # distinctiveness (peaks at 50%)
            scored.append((rel * 0.5 + dist * 0.5, phrase))
        scored.sort(key=lambda x: -x[0])                        # best combined score first

        selected: list[str] = []
        vecs: list[dict[str, float]] = []
        for final, phrase in scored:
            if final < min_score:                               # sorted desc -> first miss ends it
                break
            v = self._tf_vector(phrase)
            if any(self._cosine(v, sv) > 0.85 for sv in vecs):  # drop near-duplicate phrases
                continue
            selected.append(phrase)
            vecs.append(v)
            if len(selected) >= top_k:
                break
        return selected
