"""
rerank.py — Cohere Rerank v4 client (Azure AI Foundry), as a class.

`RerankClient.rerank()` returns [(original_index, score 0..1), ...] best-first, or
None when reranking is disabled or the call fails — so the caller can fall back to
the embedding cosine and the service keeps working with slightly less-precise
ordering. The exact REST route is config-driven (RERANK_PATH / RERANK_API_VERSION
/ RERANK_AUTH_STYLE) because that Foundry surface is new.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger("kb-tool.rerank")


class RerankClient:
    def __init__(self, config):
        self._config = config
        self._warned = False      # so the fallback warning is logged only once

    def _url(self) -> str:
        """Build the rerank endpoint URL from config (+ optional api-version)."""
        c = self._config
        url = c.RERANK_ENDPOINT.rstrip("/") + "/" + c.RERANK_PATH.lstrip("/")
        if c.RERANK_API_VERSION:
            url += ("&" if "?" in url else "?") + "api-version=" + c.RERANK_API_VERSION
        return url

    def rerank(self, query: str, documents: list[str], top_n: int):
        """Return [(index, score), ...] best-first, or None on disable/failure."""
        c = self._config
        if not c.RERANK_ENABLED or not documents:
            return None
        body = {
            "model": c.RERANK_DEPLOYMENT,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        req = urllib.request.Request(self._url(), data=json.dumps(body).encode("utf-8"),
                                    method="POST")
        req.add_header("Content-Type", "application/json")
        key = c.get_ai_key()
        if c.RERANK_AUTH_STYLE == "api-key":                  # Foundry auth style is env-driven
            req.add_header("api-key", key)
        else:
            req.add_header("Authorization", "Bearer " + key)
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                out = json.loads(r.read())
        except Exception as exc:                              # HTTPError or transport error
            if not self._warned:
                logger.warning("Cohere rerank unavailable (%s) — falling back to cosine "
                               "ranking. Set RERANK_PATH/RERANK_API_VERSION/RERANK_AUTH_STYLE "
                               "to enable.", getattr(exc, "code", type(exc).__name__))
                self._warned = True
            return None
        # Cohere v2 shape: {"results": [{"index": i, "relevance_score": s}, ...]}
        results = out.get("results")
        if results is None:
            return None
        pairs = [(int(r["index"]),
                  float(r.get("relevance_score", r.get("score", 0.0)))) for r in results]
        pairs.sort(key=lambda p: -p[1])                       # best relevance first
        return pairs
