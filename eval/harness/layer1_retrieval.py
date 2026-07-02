"""
Layer 1 — RETRIEVAL QUALITY (deterministic, tool-only).

Question answered: "For an in-KB query, does the pipeline put the correct article
near the top?" This measures the pgvector + scoring stage in isolation — before
any routing/thresholds — using the classic search metrics.

    recall@K : correct article appears anywhere in the top K   (was it FOUND?)
    top-1    : correct article is ranked #1                     (was it FIRST?)
    MRR      : mean of 1/rank-of-correct (rank1=1.0, rank2=0.5) (how HIGH on avg?)
"""
from __future__ import annotations

from common import GoldSet, Metrics, Pipeline, Settings


class Layer1Retrieval:
    def __init__(self, settings: Settings, gold: GoldSet, pipeline: Pipeline):
        self.s = settings
        self.gold = gold
        self.pipe = pipeline

    def run(self) -> dict:
        cases = self.gold.in_kb()          # strong + weak + ambiguous (have an expected_kb)
        k = self.s.k
        top1 = recall = 0
        mrr = 0.0
        per_doc: dict[str, list[int]] = {}   # kb_id -> [hits, total] for per-doc top-1
        confusions = []                      # cases where the correct KB was NOT #1

        for c in cases:
            ordered = [x["kb_id"] for x in self.pipe.rank(c.query)]
            rank = Metrics.rank_of(c.expected_kb, ordered)      # 1-based, 0 if missing
            if rank == 1:
                top1 += 1
            if 1 <= rank <= k:
                recall += 1
            if rank:
                mrr += 1.0 / rank
            # per-doc top-1 bookkeeping
            d = per_doc.setdefault(c.expected_kb, [0, 0])
            d[1] += 1
            if rank == 1:
                d[0] += 1
            if rank != 1:
                confusions.append({"id": c.id, "query": c.query,
                                   "expected": c.expected_kb, "got_top3": ordered[:3]})

        n = len(cases) or 1
        per_doc_top1 = {kb: Metrics.pct(h, t) for kb, (h, t) in sorted(per_doc.items())}
        return {
            "metrics": {
                "k": k,
                "recall_at_k": Metrics.pct(recall, n),
                "top1": Metrics.pct(top1, n),
                "mrr": round(mrr / n, 3),
                "cases": len(cases),
            },
            "detail": {"per_doc_top1": per_doc_top1, "confusions": confusions},
        }
