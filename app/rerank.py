"""
rerank.py — Cohere Rerank v4 client (Azure AI Foundry), stdlib only.

`rerank(query, documents, top_n)` returns a list of (original_index, score)
pairs sorted best-first, where score is Cohere's 0..1 relevance. It returns
**None** when reranking is disabled or the call fails — the caller
(retrieval/main) then falls back to the embedding cosine score, so the service
keeps working with degraded ranking instead of erroring.

The exact REST route for Cohere-rerank-v4 on the Foundry endpoint
(teva.services.ai.azure.com) is config-driven (RERANK_PATH / RERANK_API_VERSION
/ RERANK_AUTH_STYLE) because that surface is new and not yet pinned down. Once
the working route is known, set it via env/Key Vault — no code change needed.
"""
from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error

import config

logger = logging.getLogger("kb-tool.rerank")

_warned = False


def _url() -> str:
    base = config.RERANK_ENDPOINT.rstrip("/")
    path = "/" + config.RERANK_PATH.lstrip("/")
    url = base + path
    if config.RERANK_API_VERSION:
        url += ("&" if "?" in url else "?") + "api-version=" + config.RERANK_API_VERSION
    return url


def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]] | None:
    """Return [(index, score), ...] best-first, or None on disable/failure."""
    global _warned
    if not config.RERANK_ENABLED or not documents:
        return None
    body = {
        "model": config.RERANK_DEPLOYMENT,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(_url(), data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    key = config.get_ai_key()
    if config.RERANK_AUTH_STYLE == "api-key":
        req.add_header("api-key", key)
    else:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            out = json.loads(r.read())
    except Exception as e:  # HTTPError or transport error
        if not _warned:
            logger.warning("Cohere rerank unavailable (%s) — falling back to cosine "
                           "ranking. Set RERANK_PATH/RERANK_API_VERSION/RERANK_AUTH_STYLE "
                           "to enable.", getattr(e, "code", type(e).__name__))
            _warned = True
        return None

    # Cohere v2 shape: {"results": [{"index": i, "relevance_score": s}, ...]}
    results = out.get("results")
    if results is None:
        return None
    pairs = [(int(r["index"]),
              float(r.get("relevance_score", r.get("score", 0.0)))) for r in results]
    pairs.sort(key=lambda p: -p[1])
    return pairs
