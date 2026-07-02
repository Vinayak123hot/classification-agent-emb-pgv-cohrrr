"""
Layer 4 — END-TO-END OUTCOME (full multi-round conversation, every case).

Question answered: "Playing the whole conversation the way the agent would —
call the tool, answer its follow-up, call again — does the user end up at the
right article (or a clean no-match for off-topic queries)?"

    end_to_end_top1   : of in-KB cases, final resolved kb_id == expected
    out_of_kb_correct : of out-of-KB cases, conversation ends with kb_id == null
                        (i.e. it NEVER resolved to a wrong article)
    avg_rounds        : average number of tool calls per conversation

The loop mirrors the agent protocol exactly: keep calling with the SAME
session_id (so the real round cap applies); when the tool asks a follow-up, the
UserSimulator supplies the gold 'followup' detail; stop when the tool concludes
(followup_required == False) or the round cap is hit.
"""
from __future__ import annotations

from common import GoldSet, Metrics, Pipeline, Settings, UserSimulator

from config import CONFIG


class Layer4EndToEnd:
    def __init__(self, settings: Settings, gold: GoldSet, pipeline: Pipeline):
        self.s = settings
        self.gold = gold
        self.pipe = pipeline

    def _run_conversation(self, case) -> dict:
        """Drive one full conversation; return the final kb_id and #rounds."""
        sim = UserSimulator()
        self.pipe.reset_rounds()
        sid = f"L4-{case.id}"
        desc = case.query
        rounds = 0
        # hard stop a little beyond MAX_ROUNDS as a safety net
        for _ in range(CONFIG.MAX_ROUNDS + 2):
            rounds += 1
            resp = self.pipe.classify(desc, sid)
            if resp.get("followup_required") is False:
                return {"final_kb": resp.get("kb_id"), "rounds": rounds}
            desc = sim.answer(case, desc)      # user answers the follow-up
        # tool never concluded (shouldn't happen once the cap fires)
        return {"final_kb": resp.get("kb_id"), "rounds": rounds}

    def run(self) -> dict:
        in_kb = self.gold.in_kb()
        ook = self.gold.out_of_kb()
        e2e_correct = ook_correct = 0
        total_rounds = 0
        detail = []

        for c in in_kb:
            out = self._run_conversation(c)
            ok = out["final_kb"] == c.expected_kb
            if ok:
                e2e_correct += 1
            total_rounds += out["rounds"]
            detail.append({"id": c.id, "tier": c.tier, "expected": c.expected_kb,
                           "final_kb": out["final_kb"], "rounds": out["rounds"], "ok": ok})

        for c in ook:
            out = self._run_conversation(c)
            ok = out["final_kb"] is None      # correct = concluded with NO match
            if ok:
                ook_correct += 1
            total_rounds += out["rounds"]
            detail.append({"id": c.id, "tier": c.tier, "expected": None,
                           "final_kb": out["final_kb"], "rounds": out["rounds"], "ok": ok})

        n_all = (len(in_kb) + len(ook)) or 1
        return {
            "metrics": {
                "end_to_end_top1": Metrics.pct(e2e_correct, len(in_kb)),
                "out_of_kb_correct": Metrics.pct(ook_correct, len(ook)),
                "avg_rounds": round(total_rounds / n_all, 2),
                "in_kb_cases": len(in_kb),
                "out_of_kb_cases": len(ook),
            },
            "detail": {"cases": detail},
        }
