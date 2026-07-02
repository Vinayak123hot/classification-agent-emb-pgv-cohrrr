"""
run_all.py — THE single entry point. Runs every layer, applies the CI gate, and
writes the report.

    python eval/harness/run_all.py                # all layers + gate + report
    python eval/harness/run_all.py --layers 1,2   # a subset
    python eval/harness/run_all.py --report-only   # never fail the process

Needs the same environment as the service (PG_CONN + TEVA_AI_KEY, or Key Vault
access) because it runs the REAL pipeline in-process (embeddings + pgvector).

Outputs:
    eval/eval_results.json       machine-readable (all layers + gate)
    eval/EVAL_REPORT.md          human-readable report
Exit code is non-zero if any gate target is missed (wire straight into CI).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# make sibling modules importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import GoldSet, Pipeline, Settings
from layer1_retrieval import Layer1Retrieval
from layer2_threshold import Layer2Threshold
from layer3_clarification import Layer3Clarification
from layer4_end_to_end import Layer4EndToEnd

_EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# gate key -> (layer number, metric name inside that layer's 'metrics')
GATE_SOURCE = {
    "layer1_recall_at_k": (1, "recall_at_k"),
    "layer1_top1": (1, "top1"),
    "layer2_display_precision": (2, "display_precision"),
    "layer2_out_of_kb_rejection": (2, "out_of_kb_rejection"),
    "layer3_clarification_trigger": (3, "clarification_trigger"),
    "layer3_resolved_after_followup": (3, "resolved_after_followup"),
    "layer4_end_to_end_top1": (4, "end_to_end_top1"),
    "layer4_out_of_kb_correct": (4, "out_of_kb_correct"),
}

LAYER_CLASSES = {1: Layer1Retrieval, 2: Layer2Threshold,
                 3: Layer3Clarification, 4: Layer4EndToEnd}
LAYER_NAMES = {1: "Retrieval", 2: "Threshold", 3: "Clarification", 4: "End-to-end"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,2,3,4")
    ap.add_argument("--report-only", action="store_true", help="always exit 0")
    args = ap.parse_args()
    want = {int(x) for x in args.layers.split(",") if x.strip()}

    settings = Settings()
    gold = GoldSet()
    pipe = Pipeline()
    print(f"gold set {gold.version} — {len(gold.cases)} cases | thresholds: {settings.thresholds}")

    results: dict[int, dict] = {}
    for n in (1, 2, 3, 4):
        if n in want:
            print(f"\n=== LAYER {n}: {LAYER_NAMES[n]} ===")
            results[n] = LAYER_CLASSES[n](settings, gold, pipe).run()
            print(f"  {results[n]['metrics']}")

    # ── gate ──────────────────────────────────────────────────────────────────
    gate_rows, failed = [], 0
    for key, target in settings.gates.items():
        if not isinstance(target, (int, float)):
            continue
        layer, metric = GATE_SOURCE[key]
        if layer not in results:
            gate_rows.append((key, target, None, "SKIP"))
            continue
        actual = results[layer]["metrics"].get(metric)     # percentage (0..100)
        frac = (actual / 100.0) if isinstance(actual, (int, float)) else None
        if frac is None:
            gate_rows.append((key, target, actual, "SKIP"))
        elif frac >= target:
            gate_rows.append((key, target, actual, "PASS"))
        else:
            gate_rows.append((key, target, actual, "FAIL"))
            failed += 1

    bundle = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "gold_version": gold.version,
        "total_cases": len(gold.cases),
        "thresholds": settings.thresholds,
        "metrics": {n: results[n]["metrics"] for n in results},
        "gate": [{"check": k, "target": t, "actual": a, "status": s} for k, t, a, s in gate_rows],
        "details": {n: results[n]["detail"] for n in results},
    }
    with open(os.path.join(_EVAL_DIR, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    _write_report(bundle)

    # ── console summary + exit code ─────────────────────────────────────────────
    print("\n" + "=" * 62 + "\nGATE\n" + "=" * 62)
    for k, t, a, s in gate_rows:
        print(f"  [{s:<4}] {k:<32} target>={t:<5} actual={a}")
    print(f"\nResults: eval/eval_results.json | Report: eval/EVAL_REPORT.md")
    sys.exit(0 if (args.report_only or not failed) else 1)


def _write_report(b: dict):
    L = b["metrics"]
    lines = ["# KB Classification — pgvector 4-Layer Evaluation",
             f"\nGenerated {b['generated']} · gold `{b['gold_version']}` · {b['total_cases']} cases",
             f"\nLive thresholds: `{b['thresholds']}`",
             "\n## Gate", "\n| Check | Target | Actual | Status |", "|---|---|---|---|"]
    for g in b["gate"]:
        lines.append(f"| {g['check']} | >={g['target']} | {g['actual']} | {g['status']} |")
    if 1 in L:
        m = L[1]
        lines += ["\n## Layer 1 — Retrieval",
                  f"- recall@{m['k']}: **{m['recall_at_k']}%** · top-1: **{m['top1']}%** · MRR: {m['mrr']} "
                  f"({m['cases']} in-KB cases)",
                  f"- per-doc top-1: {b['details'][1]['per_doc_top1']}"]
        if b["details"][1]["confusions"]:
            lines.append(f"- not-#1 cases: {len(b['details'][1]['confusions'])} "
                         f"(see eval_results.json for the list)")
    if 2 in L:
        m = L[2]
        lines += ["\n## Layer 2 — Threshold / routing",
                  f"- display precision: **{m['display_precision']}%** · display recall (strong): "
                  f"**{m['display_recall']}%** · out-of-KB rejection: **{m['out_of_kb_rejection']}%**",
                  f"- CONFIDENT_SCORE in use: {m['confident_score_in_use']} · turn-1 resolves: "
                  f"{m['resolved_turn1']} (false: {m['false_resolves']})",
                  "\n  CONFIDENT_SCORE sweep (correct vs false turn-1 resolves):",
                  "\n  | CONFIDENT_SCORE | correct | false |", "  |---|---|---|"]
        for s in b["details"][2]["threshold_sweep"]:
            lines.append(f"  | {s['confident_score']} | {s['correct_resolves']} | {s['false_resolves']} |")
    if 3 in L:
        m = L[3]
        lines += ["\n## Layer 3 — Clarification (weak + ambiguous)",
                  f"- clarification trigger: **{m['clarification_trigger']}%** · grounded phrases: "
                  f"{m['grounded_rate']}% · resolved after 1 follow-up: **{m['resolved_after_followup']}%** "
                  f"({m['cases']} cases)"]
    if 4 in L:
        m = L[4]
        lines += ["\n## Layer 4 — End-to-end",
                  f"- end-to-end top-1: **{m['end_to_end_top1']}%** ({m['in_kb_cases']} in-KB) · "
                  f"out-of-KB correct: **{m['out_of_kb_correct']}%** ({m['out_of_kb_cases']}) · "
                  f"avg rounds: {m['avg_rounds']}"]
    with open(os.path.join(_EVAL_DIR, "EVAL_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
