"""
common.py — shared infrastructure for the 4-layer evaluation harness.

Everything the layer scripts need lives here as small, documented classes:

    Settings      - gate targets (eval/config.json) + the LIVE thresholds from
                    the production app/config.py (so the eval always measures the
                    exact settings the service runs with).
    GoldCase      - one labelled test case.
    GoldSet       - loads and filters the gold set.
    Pipeline      - runs the REAL production pipeline in-process (zero drift):
                      .rank()     -> ordered candidates (retrieval + scoring)
                      .classify() -> the tool's full routing decision (calls the
                                     actual main.get_kb_candidates endpoint fn)
    UserSimulator - deterministic follow-up answers (uses the gold 'followup'
                    field) so the end-to-end layers need no live LLM.
    Metrics       - tiny metric helpers (rank, MRR term, percentage).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

# ── Make the production app importable, then import the REAL modules ───────────
# eval/harness/common.py -> repo root -> app/
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_APP = os.path.join(_REPO, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from config import CONFIG          # app/config.py - the live Config instance
from main import SERVICE           # app/main.py   - the composed KBService

# ── Eval speed-up (correctness-preserving) ────────────────────────────────────
# Each pipeline call makes network round-trips (Azure embedding + pgvector in
# Central India ~ a few seconds). The layers re-run the same queries, and the
# end-to-end loop re-sends the same description across rounds. In cosine-fallback
# mode the pipeline is fully deterministic, so we memoize the retriever's search
# (which internally does the embedding) on the live service's retriever instance.
# This changes NOTHING about the results — only the runtime.
_search_cache: dict = {}
_orig_search = SERVICE.retriever.search
def _cached_search(query, return_k=None):
    key = (query, return_k)
    if key not in _search_cache:
        _search_cache[key] = _orig_search(query, return_k=return_k)
    return [dict(c) for c in _search_cache[key]]   # fresh copies -> safe to mutate/score
SERVICE.retriever.search = _cached_search          # patch the instance method

_EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GOLD = os.path.join(_EVAL_DIR, "gold_set.json")
_CONFIG = os.path.join(_EVAL_DIR, "config.json")


# ══════════════════════════════════════════════════════════════════════════════
class Settings:
    """Gate targets from eval/config.json + the live 0..1 thresholds the service
    actually uses (read straight from app/config.py, so the report can never lie
    about which settings were measured)."""

    def __init__(self):
        with open(_CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
        self.k: int = int(cfg.get("k", CONFIG.RETURN_K))     # recall@K / top-K
        self.gates: dict = cfg.get("gates", {})
        # snapshot the live thresholds for the report header
        self.thresholds = {
            "CONFIDENT_SCORE": CONFIG.CONFIDENT_SCORE,
            "MIN_DISPLAY_SCORE": CONFIG.MIN_DISPLAY_SCORE,
            "FOLLOWUP_FLOOR": CONFIG.FOLLOWUP_FLOOR,
            "SPREAD_THRESHOLD": CONFIG.SPREAD_THRESHOLD,
            "MIN_FIELD_SCORE": CONFIG.MIN_FIELD_SCORE,
            "MAX_ROUNDS": CONFIG.MAX_ROUNDS,
            "RERANK_K": CONFIG.RERANK_K,
        }


# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class GoldCase:
    id: str
    query: str
    expected_kb: str | None      # None for out_of_kb cases
    tier: str                    # strong | weak | ambiguous | out_of_kb
    intent: str
    followup: str | None = None  # extra detail the user gives when asked (rounds >1)

    @property
    def in_kb(self) -> bool:
        return self.tier != "out_of_kb" and bool(self.expected_kb)


class GoldSet:
    """Loads gold_set.json into GoldCase objects and offers tier filters."""

    def __init__(self, path: str = _GOLD):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.version = data.get("version", "?")
        self.cases: list[GoldCase] = [
            GoldCase(id=c["id"], query=c["query"], expected_kb=c.get("expected_kb"),
                     tier=c.get("tier", "strong"), intent=c.get("intent", ""),
                     followup=c.get("followup"))
            for c in data["cases"]
        ]

    def by_tier(self, *tiers: str) -> list[GoldCase]:
        return [c for c in self.cases if c.tier in tiers]

    def in_kb(self) -> list[GoldCase]:
        return [c for c in self.cases if c.in_kb]

    def out_of_kb(self) -> list[GoldCase]:
        return [c for c in self.cases if c.tier == "out_of_kb"]


# ══════════════════════════════════════════════════════════════════════════════
class Pipeline:
    """Thin wrapper over the REAL production KBService (no HTTP, no drift).

    - rank()     -> KBService.rank() (retrieval + scoring): the full ordered list.
    - classify() -> KBService.classify(): the exact routing decision (spread +
                    score bands + round cap) the deployed service returns."""

    def __init__(self):
        self._service = SERVICE           # the composed, live service instance

    def rank(self, description: str) -> list[dict]:
        """Ordered candidates, best first. Each dict has kb_id, score, cos, ..."""
        return self._service.rank(description)

    def classify(self, description: str, session_id: str) -> dict:
        """The tool's full response for one call (turn), using the same in-memory
        round counter, keyed by session_id."""
        return self._service.classify(description, session_id)

    @staticmethod
    def reset_rounds() -> None:
        """Clear the per-session round counter between independent test cases."""
        SERVICE.rounds.reset()


# ══════════════════════════════════════════════════════════════════════════════
class UserSimulator:
    """Deterministic stand-in for a human answering the agent's follow-up. On the
    first follow-up it appends the case's 'followup' detail to the running
    description; after that it repeats the same description (so the tool proceeds
    to its round cap rather than looping on new information)."""

    def __init__(self):
        self._used: set[str] = set()

    def answer(self, case: GoldCase, current_desc: str) -> str:
        if case.followup and case.id not in self._used:
            self._used.add(case.id)
            return f"{current_desc}, {case.followup}"
        return current_desc


# ══════════════════════════════════════════════════════════════════════════════
class Metrics:
    """Tiny, dependency-free metric helpers."""

    @staticmethod
    def rank_of(expected: str, ordered_ids: list[str]) -> int:
        """1-based rank of expected in the list, or 0 if absent."""
        return ordered_ids.index(expected) + 1 if expected in ordered_ids else 0

    @staticmethod
    def pct(numerator: int, denominator: int) -> float:
        """Percentage rounded to 1 dp; 0.0 when denominator is 0."""
        return round(100.0 * numerator / denominator, 1) if denominator else 0.0
