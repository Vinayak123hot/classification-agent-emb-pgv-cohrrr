"""
Layer 2 — THRESHOLD / ROUTING CALIBRATION (deterministic, single turn).

Question answered: "Given the 0..1 score bands, does the tool make the right
turn-1 decision — resolve the clear cases, and refuse to resolve off-topic ones?"

    display_precision    : of the cases it RESOLVES on turn 1, how many are correct
                           (a wrong resolve is the worst failure -> protect this)
    display_recall       : of STRONG cases, how many resolve correctly on turn 1
    out_of_kb_rejection  : of OUT-OF-KB cases, how many it correctly does NOT resolve
    threshold_sweep      : how those numbers move as CONFIDENT_SCORE varies

Instead of re-calling the endpoint, we rank() each case once and replay the exact
turn-1 RESOLVE rule (top_score > CONFIDENT_SCORE AND spread == 'low'), which lets
us sweep CONFIDENT_SCORE cheaply over the same scores.
"""
from __future__ import annotations

from common import GoldSet, Metrics, Pipeline, Settings
from config import CONFIG          # live routing thresholds


class Layer2Threshold:
    def __init__(self, settings: Settings, gold: GoldSet, pipeline: Pipeline):
        self.s = settings
        self.gold = gold
        self.pipe = pipeline

    @staticmethod
    def _resolves(top: float, gap: float, confident: float) -> bool:
        """Replay main.py's turn-1 RESOLVE gate for a given CONFIDENT_SCORE:
        resolve only when the top score clears the threshold AND the top-two gap
        is not a near-tie (spread 'low')."""
        weak_absolute = top < confident
        weak_dominance = gap < CONFIG.SPREAD_THRESHOLD
        spread_low = not (weak_absolute or weak_dominance)
        return top > confident and spread_low

    def run(self) -> dict:
        # rank every case once; capture top score, top-2 gap, top kb_id, expected.
        rows = []
        for c in self.gold.cases:
            ordered = self.pipe.rank(c.query)
            top = ordered[0]["score"] if ordered else 0.0
            second = ordered[1]["score"] if len(ordered) > 1 else 0.0
            rows.append({"case": c, "top": top, "gap": round(top - second, 4),
                         "top_kb": ordered[0]["kb_id"] if ordered else None})

        C = CONFIG.CONFIDENT_SCORE
        resolved = correct_resolved = false_resolved = 0
        strong_total = strong_resolved_correct = 0
        ook_total = ook_rejected = 0

        for r in rows:
            c = r["case"]
            does_resolve = self._resolves(r["top"], r["gap"], C)
            if c.tier == "strong":
                strong_total += 1
                if does_resolve and r["top_kb"] == c.expected_kb:
                    strong_resolved_correct += 1
            if c.tier == "out_of_kb":
                ook_total += 1
                if not does_resolve:
                    ook_rejected += 1
            if does_resolve:
                resolved += 1
                if c.in_kb and r["top_kb"] == c.expected_kb:
                    correct_resolved += 1
                else:
                    false_resolved += 1   # resolved to wrong KB, or resolved an out-of-KB

        # threshold sweep over CONFIDENT_SCORE
        sweep = []
        for cval in [round(x / 100, 2) for x in range(40, 86, 5)]:
            cr = fr = 0
            for r in rows:
                c = r["case"]
                if self._resolves(r["top"], r["gap"], cval):
                    if c.in_kb and r["top_kb"] == c.expected_kb:
                        cr += 1
                    else:
                        fr += 1
            sweep.append({"confident_score": cval, "correct_resolves": cr, "false_resolves": fr})

        return {
            "metrics": {
                "display_precision": Metrics.pct(correct_resolved, resolved),
                "display_recall": Metrics.pct(strong_resolved_correct, strong_total),
                "out_of_kb_rejection": Metrics.pct(ook_rejected, ook_total),
                "confident_score_in_use": C,
                "resolved_turn1": resolved,
                "false_resolves": false_resolved,
            },
            "detail": {"threshold_sweep": sweep},
        }
