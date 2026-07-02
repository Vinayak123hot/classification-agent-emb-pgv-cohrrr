"""
embeddings.py — Azure OpenAI text-embedding-3-small client, as a class.

`EmbeddingClient.embed_texts()` returns L2-normalized vectors so that pgvector's
cosine operator and a plain dot-product agree (and the cosine fallback score is a
clean 0..1-ish relevance signal). Uses stdlib urllib — no extra HTTP dependency.
"""
from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request

logger = logging.getLogger("kb-tool.embeddings")


class EmbeddingClient:
    def __init__(self, config):
        self._config = config      # endpoint / deployment / api-version / get_ai_key()

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        """Scale a vector to unit length (so cosine == dot product)."""
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Returns one normalized vector per input."""
        if not texts:
            return []
        c = self._config
        url = (f"{c.EMBED_ENDPOINT.rstrip('/')}/openai/deployments/"
               f"{c.EMBED_DEPLOYMENT}/embeddings?api-version={c.EMBED_API_VERSION}")
        data = json.dumps({"input": texts}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("api-key", c.get_ai_key())              # Azure OpenAI key header
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "ignore")
            raise RuntimeError(f"Embedding call failed: HTTP {exc.code} {detail}") from exc
        # API may return items out of order — sort by index to be safe.
        items = sorted(out["data"], key=lambda d: d["index"])
        return [self._l2_normalize(it["embedding"]) for it in items]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([text])[0]

    @staticmethod
    def to_pgvector(vec: list[float]) -> str:
        """Format a vector as the pgvector literal '[v1,v2,...]'."""
        return "[" + ",".join(repr(float(x)) for x in vec) + "]"
