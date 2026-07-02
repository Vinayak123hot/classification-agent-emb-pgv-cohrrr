"""
calibrate.py — measure retrieval quality + recommend the 0..1 score bands, as a class.

`Calibrator.run()` executes the FULL live pipeline over the gold set and reports
top-1, recall@K, MRR, plus the score separation between in-KB-correct top hits and
out-of-KB top hits — which is what CONFIDENT_SCORE / MIN_DISPLAY_SCORE should sit
between.

    python app/calibrate.py                 # uses vinny-ai-search/eval/gold_set.json
    GOLD_SET=/path/to/gold.json python app/calibrate.py

(For a full 4-layer gate + hyperparameter tuning use eval/harness/run_all.py and
eval/harness/tune.py. This script is the quick single-number calibration probe.)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from main import SERVICE   # reuse the composed service (retriever + scorer)


class Calibrator:
    _DEFAULT_GOLD = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "vinny-ai-search", "eval", "gold_set.json",
    )

    def __init__(self, service, config, gold_path: "str | None" = None):
        self._service = service
        self._config = config
        self._gold_path = gold_path or os.environ.get("GOLD_SET", self._DEFAULT_GOLD)

    def _load_gold(self) -> list[dict]:
        """Load the gold cases from a list- or dict-shaped gold file."""
        d = json.load(open(self._gold_path, encoding="utf-8"))
        if isinstance(d, list):
            return d
        for k in ("cases", "items", "data", "gold"):
            if isinstance(d.get(k), list):
                return d[k]
        for v in d.values():
            if isinstance(v, list):
                return v
        raise ValueError("could not find a case list in gold set")

    def _rank(self, query: str) -> list[dict]:
        """Ordered candidates (retrieval + scoring) for one query."""
        return self._service.rank(query)

    def run(self) -> None:
        cases = self._load_gold()
        print(f"gold set: {self._gold_path}  ({len(cases)} cases)")
        k = self._config.RETURN_K
        in_kb = [c for c in cases if c.get("tier") != "out_of_kb" and c.get("expected_kb")]
        out_kb = [c for c in cases if c.get("tier") == "out_of_kb"]

        top1 = recall_k = 0
        mrr = 0.0
        scorer_seen = set()
        correct_scores, outkb_scores = [], []

        for c in in_kb:
            q, exp = c["query"], c["expected_kb"]
            cands = self._rank(q)
            if cands:
                scorer_seen.add(cands[0]["scorer"])
            ids = [x["kb_id"] for x in cands]
            rank = ids.index(exp) + 1 if exp in ids else 0
            if rank == 1:
                top1 += 1
                correct_scores.append(cands[0]["score"])
            if 1 <= rank <= k:
                recall_k += 1
            if rank:
                mrr += 1.0 / rank

        for c in out_kb:
            cands = self._rank(c["query"])
            if cands:
                outkb_scores.append(cands[0]["score"])

        n = len(in_kb) or 1
        print(f"\nscorer in use: {', '.join(scorer_seen) or '-'}")
        print(f"top-1     : {top1}/{len(in_kb)} = {100*top1/n:.1f}%")
        print(f"recall@{k}  : {recall_k}/{len(in_kb)} = {100*recall_k/n:.1f}%")
        print(f"MRR       : {mrr/n:.3f}")

        print("\nscore separation (drives the thresholds):")
        if correct_scores:
            print(f"  in-KB correct  top scores: min={min(correct_scores):.3f} "
                  f"mean={sum(correct_scores)/len(correct_scores):.3f} max={max(correct_scores):.3f}")
        if outkb_scores:
            print(f"  out-of-KB      top scores: min={min(outkb_scores):.3f} "
                  f"mean={sum(outkb_scores)/len(outkb_scores):.3f} max={max(outkb_scores):.3f}")
        if correct_scores and outkb_scores:
            lo, hi = max(outkb_scores), min(correct_scores)
            mid = round((lo + hi) / 2, 2)
            print(f"\n  -> out-of-KB ceiling = {lo:.3f} ; in-KB correct floor = {hi:.3f}")
            if hi > lo:
                print(f"  -> RECOMMEND: CONFIDENT_SCORE ~ {mid}, MIN_DISPLAY_SCORE just above {lo:.2f}")
            else:
                print(f"  -> overlap: keep CONFIDENT_SCORE high (>= {hi:.2f}) to protect precision")
        print(f"\ncurrent config: CONFIDENT_SCORE={self._config.CONFIDENT_SCORE} "
              f"MIN_DISPLAY_SCORE={self._config.MIN_DISPLAY_SCORE} "
              f"FOLLOWUP_FLOOR={self._config.FOLLOWUP_FLOOR} "
              f"SPREAD_THRESHOLD={self._config.SPREAD_THRESHOLD}")


if __name__ == "__main__":
    Calibrator(SERVICE, CONFIG).run()
