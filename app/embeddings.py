"""
embeddings.py — Azure OpenAI text-embedding-3-small client (stdlib only).

Returns L2-normalized vectors so that the pgvector cosine operator (<=>) and a
plain dot-product agree, and so the cosine-similarity fallback score in
retrieval.py is a clean 0..1-ish relevance signal.

Endpoint/key/deployment come from config (Key Vault key `teva-ai-services-key`).
Uses urllib so the runtime needs no extra HTTP dependency.
"""
from __future__ import annotations

import json
import logging
import math
import urllib.request
import urllib.error

import config

logger = logging.getLogger("kb-tool.embeddings")


def _l2_normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings. Returns one normalized vector per input."""
    if not texts:
        return []
    url = (f"{config.EMBED_ENDPOINT.rstrip('/')}/openai/deployments/"
           f"{config.EMBED_DEPLOYMENT}/embeddings?api-version={config.EMBED_API_VERSION}")
    body = {"input": texts}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("api-key", config.get_ai_key())
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "ignore")
        raise RuntimeError(f"Embedding call failed: HTTP {e.code} {detail}") from e
    # API returns items possibly out of order — sort by index to be safe.
    items = sorted(out["data"], key=lambda d: d["index"])
    return [_l2_normalize(it["embedding"]) for it in items]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def to_pgvector(vec: list[float]) -> str:
    """Format a vector as a pgvector literal: '[v1,v2,...]'."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"
