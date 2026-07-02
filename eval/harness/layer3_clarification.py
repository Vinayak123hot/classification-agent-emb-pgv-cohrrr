"""
Layer 3 — CLARIFICATION QUALITY (weak + ambiguous cases, 2 turns).

Question answered: "When the tool is unsure, does it ASK instead of guessing —
and does one good follow-up actually resolve the case?"

    clarification_trigger  : of weak/ambiguous cases, how many the tool does NOT
                             resolve on turn 1 (i.e. it asks a follow-up)
    grounded_rate          : of those follow-ups, how many return grounded phrases
                             (discriminating_symptoms) vs. handing the agent freeform
    resolved_after_followup: how many resolve to the CORRECT KB on turn 2 once the
                             user's follow-up detail is added

Turn 2 detail comes from the gold 'followup' field via UserSimulator (no LLM).
"""
from __future__ import annotations

from common import GoldSet, Metrics, Pipeline, Settings, UserSimulator


class Layer3Clarification:
    def __init__(self, settings: Settings, gold: GoldSet, pipeline: Pipeline):
        self.s = settings
        self.gold = gold
        self.pipe = pipeline

    def run(self) -> dict:
        cases = self.gold.by_tier("weak", "ambiguous")
        triggered = grounded = resolved_after = 0
        detail = []

        for c in cases:
            sim = UserSimulator()
            self.pipe.reset_rounds()
            sid = f"L3-{c.id}"

            # ── turn 1: expect a follow-up (not an immediate resolve) ──
            r1 = self.pipe.classify(c.query, sid)
            asked = bool(r1.get("followup_required"))
            if asked:
                triggered += 1
                if r1.get("discriminating_symptoms"):
                    grounded += 1

            # ── turn 2: add the user's follow-up detail, expect correct resolve ──
            desc2 = sim.answer(c, c.query)
            r2 = self.pipe.classify(desc2, sid)
            ok2 = (r2.get("followup_required") is False and r2.get("kb_id") == c.expected_kb)
            if ok2:
                resolved_after += 1

            detail.append({"id": c.id, "tier": c.tier,
                           "turn1_asked": asked,
                           "turn1_grounded": bool(r1.get("discriminating_symptoms")),
                           "turn1_phrases": r1.get("discriminating_symptoms") or [],
                           "turn2_kb": r2.get("kb_id"), "expected": c.expected_kb,
                           "turn2_resolved_correct": ok2})

        n = len(cases) or 1
        n_asked = triggered or 1
        return {
            "metrics": {
                "clarification_trigger": Metrics.pct(triggered, n),
                "grounded_rate": Metrics.pct(grounded, n_asked),
                "resolved_after_followup": Metrics.pct(resolved_after, n),
                "cases": len(cases),
            },
            "detail": {"cases": detail},
        }
