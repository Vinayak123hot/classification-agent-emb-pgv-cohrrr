"""
tune.py — grid-search the cutoff-threshold hyperparameters against the gold set.

Efficiency trick: the expensive work (embedding + pgvector retrieval + scoring)
does NOT depend on the thresholds — only the resulting ranked scores do, and the
thresholds only drive pure-arithmetic routing on top. So we:
  1. PRECOMPUTE, once, each case's round-1 ranking and (for weak/ambiguous) its
     round-2 ranking (query + the user's follow-up detail).  [network, ~1x]
  2. Replay the exact main.py routing (spread + score bands + round cap) for every
     threshold combination purely in memory.                 [instant, ~1000s/sec]

Tuned (cutoff thresholds):
  - CONFIDENT_SCORE, MIN_DISPLAY_SCORE, SPREAD_THRESHOLD  -> joint GRID
    (objective = end-to-end accuracy: correct in-KB resolves + correct out-of-KB
    no-matches, over all cases; tie-broken by display precision, then more
    turn-1 resolves, then fewer rounds)
  - FOLLOWUP_FLOOR, MIN_FIELD_SCORE                       -> SWEEP
    (these change grounded-vs-freeform follow-ups, not end-to-end accuracy;
    optimized for grounded-rate on weak/ambiguous without surfacing weak phrases)

Not cutoffs (accounted for, not grid-searched): TOP_FIELDS_K (verbosity),
MAX_ROUNDS (latency), VECTOR_K/KEYWORD_K/RRF_K/RETURN_K/RERANK_K (retrieval
breadth — on an 8-doc/17-chunk KB everything is retrieved, so they don't move the
metrics; they matter only at larger scale).

    python eval/harness/tune.py          # precompute + grid + sweeps + report
"""
from __future__ import annotations

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import GoldSet, Pipeline, Settings
import config
import followup

_EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_ROUNDS = config.MAX_ROUNDS            # not a cutoff; kept fixed during tuning

# ── grids for the cutoff thresholds ───────────────────────────────────────────
GRID_CONFIDENT = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
GRID_MINDISP   = [0.40, 0.45, 0.50, 0.55, 0.60]
GRID_SPREAD    = [0.03, 0.05, 0.08, 0.10, 0.15]
SWEEP_FLOOR    = [0.10, 0.15, 0.20, 0.25]
SWEEP_FIELD    = [0.30, 0.35, 0.40, 0.45, 0.50]


# ══════════════════════════════════════════════════════════════════════════════
def precompute(pipe: Pipeline, gold: GoldSet) -> list[dict]:
    """For every case, get the scored round-1 candidates, and round-2 candidates
    (query + follow-up) where a follow-up exists. This is the only network work."""
    data = []
    for c in gold.cases:
        r1 = [{"kb_id": x["kb_id"], "score": x["score"], "title": x.get("title"),
               "question": x.get("question"), "cause": x.get("cause")}
              for x in pipe.rank(c.query)]
        if c.followup:
            desc2 = f"{c.query}, {c.followup}"
            r2 = [{"kb_id": x["kb_id"], "score": x["score"], "title": x.get("title"),
                   "question": x.get("question"), "cause": x.get("cause")}
                  for x in pipe.rank(desc2)]
        else:
            r2 = r1                      # no new info -> same ranking on later rounds
        data.append({"case": c, "r1": r1, "r2": r2})
    return data


def decide(cands: list[dict], round_no: int, C: float, MD: float, ST: float):
    """Pure replay of main.get_kb_candidates routing for one call.
    Returns ('resolve', kb) | ('conclude', None) | ('followup', None)."""
    if not cands:
        return ("conclude", None) if round_no >= MAX_ROUNDS else ("followup", None)
    scores = [c["score"] for c in cands]
    top = scores[0]
    weak_absolute = top < C
    weak_dominance = len(scores) > 1 and (scores[0] - scores[1]) < ST
    spread_high = weak_absolute or weak_dominance
    if top > C and not spread_high:                 # RESOLVE (confident + clear)
        return ("resolve", cands[0]["kb_id"])
    if round_no >= MAX_ROUNDS:                        # ROUND CAP
        if top >= MD:
            return ("resolve", cands[0]["kb_id"])     # present-best-at-cap
        return ("conclude", None)                     # no confident match
    return ("followup", None)


def simulate(cd: dict, C: float, MD: float, ST: float):
    """Run the full conversation; return (final_kb, rounds, turn1_action, turn1_kb)."""
    turn1 = ("followup", None)
    for r in range(1, MAX_ROUNDS + 1):
        cands = cd["r1"] if r == 1 else cd["r2"]
        action, kb = decide(cands, r, C, MD, ST)
        if r == 1:
            turn1 = (action, kb)
        if action == "resolve":
            return kb, r, turn1[0], turn1[1]
        if action == "conclude":
            return None, r, turn1[0], turn1[1]
    return None, MAX_ROUNDS, turn1[0], turn1[1]


def evaluate(data: list[dict], C: float, MD: float, ST: float) -> dict:
    """Compute end-to-end + turn-1 metrics for one (C, MD, ST) combination."""
    n_inkb = n_ook = 0
    e2e_correct = ook_correct = 0
    turn1_resolves = turn1_correct = 0
    ook_reject_t1 = 0
    total_rounds = 0
    for cd in data:
        case = cd["case"]
        final_kb, rounds, t1_action, t1_kb = simulate(cd, C, MD, ST)
        total_rounds += rounds
        if case.in_kb:
            n_inkb += 1
            if final_kb == case.expected_kb:
                e2e_correct += 1
        else:
            n_ook += 1
            if final_kb is None:
                ook_correct += 1
            if t1_action != "resolve":
                ook_reject_t1 += 1
        if t1_action == "resolve":
            turn1_resolves += 1
            if case.in_kb and t1_kb == case.expected_kb:
                turn1_correct += 1
    n = n_inkb + n_ook
    return {
        "C": C, "MD": MD, "ST": ST,
        "total_correct": e2e_correct + ook_correct,          # PRIMARY objective (of n)
        "total_cases": n,
        "e2e_in_kb_pct": round(100 * e2e_correct / (n_inkb or 1), 1),
        "out_of_kb_correct_pct": round(100 * ook_correct / (n_ook or 1), 1),
        "display_precision_pct": round(100 * turn1_correct / (turn1_resolves or 1), 1),
        "out_of_kb_rejection_t1_pct": round(100 * ook_reject_t1 / (n_ook or 1), 1),
        "turn1_resolves": turn1_resolves,
        "turn1_correct": turn1_correct,
        "avg_rounds": round(total_rounds / (n or 1), 2),
    }


def objective_key(m: dict):
    """Rank combos: most total end-to-end correct, then never mis-resolve on turn
    1 (precision), then resolve more clear cases up front, then fewer rounds."""
    return (m["total_correct"], m["display_precision_pct"], m["turn1_correct"], -m["avg_rounds"])


# ── secondary sweep: grounded follow-up quality (FOLLOWUP_FLOOR, MIN_FIELD_SCORE) ─
def grounded_rate(data: list[dict], floor: float, field_min: float) -> dict:
    """On weak/ambiguous turn-1 follow-ups: how often do we return a GROUNDED
    phrase (vs freeform)? Grounded requires top_score >= FOLLOWUP_FLOOR and a
    phrase clearing MIN_FIELD_SCORE."""
    cases = [cd for cd in data if cd["case"].tier in ("weak", "ambiguous")]
    grounded = 0
    for cd in cases:
        top = cd["r1"][0]["score"] if cd["r1"] else 0.0
        phrases = []
        if top >= floor:
            phrases = followup.select_discriminating_fields(
                cd["case"].query, cd["r1"][:config.RETURN_K], min_score=field_min)
        if phrases:
            grounded += 1
    return {"floor": floor, "field_min": field_min,
            "grounded_pct": round(100 * grounded / (len(cases) or 1), 1),
            "n": len(cases)}


# ══════════════════════════════════════════════════════════════════════════════
def main():
    settings = Settings()
    gold = GoldSet()
    pipe = Pipeline()
    print(f"precomputing rankings for {len(gold.cases)} cases (network, one pass)...")
    data = precompute(pipe, gold)
    print("done. running grid...")

    # baseline (current live thresholds) for comparison
    base = evaluate(data, config.CONFIDENT_SCORE, config.MIN_DISPLAY_SCORE, config.SPREAD_THRESHOLD)

    # joint grid (respect ordering FOLLOWUP_FLOOR <= MD <= C)
    combos = []
    for C, MD, ST in itertools.product(GRID_CONFIDENT, GRID_MINDISP, GRID_SPREAD):
        if MD > C:
            continue
        combos.append(evaluate(data, C, MD, ST))
    combos.sort(key=objective_key, reverse=True)
    best = combos[0]

    # secondary sweep at the best core settings
    sweeps = [grounded_rate(data, f, fm)
              for f in SWEEP_FLOOR for fm in SWEEP_FIELD]
    best_grounded = max(sweeps, key=lambda s: s["grounded_pct"])

    # ── console summary ─────────────────────────────────────────────────────────
    def line(m):
        return (f"C={m['C']:.2f} MD={m['MD']:.2f} ST={m['ST']:.2f} | "
                f"correct={m['total_correct']}/{m['total_cases']} "
                f"inKB={m['e2e_in_kb_pct']}% ook={m['out_of_kb_correct_pct']}% "
                f"disp_prec={m['display_precision_pct']}% t1={m['turn1_resolves']} "
                f"rounds={m['avg_rounds']}")
    print("\nBASELINE  " + line(base))
    print("BEST      " + line(best))
    print("\nTop 8 combinations:")
    for m in combos[:8]:
        print("  " + line(m))
    print(f"\nGrounded follow-up sweep best: floor={best_grounded['floor']} "
          f"field_min={best_grounded['field_min']} -> grounded {best_grounded['grounded_pct']}% "
          f"(vs current floor={config.FOLLOWUP_FLOOR}/field={config.MIN_FIELD_SCORE})")

    # ── write reports ───────────────────────────────────────────────────────────
    bundle = {"baseline": base, "best": best, "top": combos[:12],
              "grounded_sweep": sweeps, "best_grounded": best_grounded,
              "grids": {"CONFIDENT_SCORE": GRID_CONFIDENT, "MIN_DISPLAY_SCORE": GRID_MINDISP,
                        "SPREAD_THRESHOLD": GRID_SPREAD, "FOLLOWUP_FLOOR": SWEEP_FLOOR,
                        "MIN_FIELD_SCORE": SWEEP_FIELD}}
    with open(os.path.join(_EVAL_DIR, "tuning_results.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    lines = ["# Hyperparameter Tuning — cutoff thresholds",
             f"\nGold `{gold.version}` · {len(gold.cases)} cases · in-memory grid over precomputed rankings.",
             "\n## Recommended settings",
             f"- **CONFIDENT_SCORE = {best['C']}**  (was {config.CONFIDENT_SCORE})",
             f"- **MIN_DISPLAY_SCORE = {best['MD']}**  (was {config.MIN_DISPLAY_SCORE})",
             f"- **SPREAD_THRESHOLD = {best['ST']}**  (was {config.SPREAD_THRESHOLD})",
             f"- **FOLLOWUP_FLOOR = {best_grounded['floor']}**  (was {config.FOLLOWUP_FLOOR}) · "
             f"**MIN_FIELD_SCORE = {best_grounded['field_min']}**  (was {config.MIN_FIELD_SCORE})",
             "\n## Baseline vs tuned (end-to-end over all cases)",
             "\n| | in-KB top-1 | out-of-KB correct | display precision | turn-1 resolves | avg rounds |",
             "|---|---|---|---|---|---|",
             f"| baseline | {base['e2e_in_kb_pct']}% | {base['out_of_kb_correct_pct']}% | "
             f"{base['display_precision_pct']}% | {base['turn1_resolves']} | {base['avg_rounds']} |",
             f"| **tuned** | {best['e2e_in_kb_pct']}% | {best['out_of_kb_correct_pct']}% | "
             f"{best['display_precision_pct']}% | {best['turn1_resolves']} | {best['avg_rounds']} |",
             "\n## Top combinations (C = CONFIDENT_SCORE, MD = MIN_DISPLAY_SCORE, ST = SPREAD_THRESHOLD)",
             "\n| C | MD | ST | correct | in-KB | out-of-KB | disp.prec | t1 | rounds |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in combos[:12]:
        lines.append(f"| {m['C']} | {m['MD']} | {m['ST']} | {m['total_correct']}/{m['total_cases']} "
                     f"| {m['e2e_in_kb_pct']}% | {m['out_of_kb_correct_pct']}% | "
                     f"{m['display_precision_pct']}% | {m['turn1_resolves']} | {m['avg_rounds']} |")
    lines += ["\n## Grounded follow-up sweep (FOLLOWUP_FLOOR x MIN_FIELD_SCORE -> grounded % on weak+ambiguous)",
              "\n| FOLLOWUP_FLOOR | MIN_FIELD_SCORE | grounded % |", "|---|---|---|"]
    for s in sweeps:
        lines.append(f"| {s['floor']} | {s['field_min']} | {s['grounded_pct']}% |")
    lines += ["\n## Parameters NOT grid-searched (accounted for)",
              "- `TOP_FIELDS_K` (3): max grounded phrases returned — verbosity, not accuracy.",
              "- `MAX_ROUNDS` (5): rounds before the cap — affects latency/avg-rounds, not final accuracy in this sim.",
              "- `VECTOR_K`/`KEYWORD_K`/`RRF_K`/`RETURN_K`/`RERANK_K`: retrieval breadth — on an 8-doc/17-chunk KB "
              "everything is already retrieved, so they don't move the metrics; revisit at larger corpus scale."]
    with open(os.path.join(_EVAL_DIR, "TUNING_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nWrote eval/TUNING_REPORT.md + eval/tuning_results.json")


if __name__ == "__main__":
    main()
