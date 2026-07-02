"""
retrieval.py — hybrid (vector + keyword) retrieval with RRF fusion.

Per query:
  1. embed the query (text-embedding-3-small).
  2. VECTOR leg  : pgvector cosine KNN over kb_chunks  -> top VECTOR_K chunks.
  3. KEYWORD leg : Postgres full-text (tsv) ranked by ts_rank_cd -> top KEYWORD_K.
  4. RRF fuse the two chunk lists: score = sum 1/(RRF_K + rank).
  5. aggregate chunks -> distinct ARTICLES (best fused score + best cosine per
     kb_id), attach the article's title/question/cause for downstream rerank +
     follow-up.

Returns a list of Candidate dicts sorted by fused score (desc). Final ordering
and the 0..1 `score` are decided later by the Cohere reranker (main.py); the
cosine here is the graceful-fallback score if rerank is unavailable.
"""
from __future__ import annotations

import logging

import config
import db
import embeddings

logger = logging.getLogger("kb-tool.retrieval")


def _vector_leg(qvec_literal: str) -> list[dict]:
    rows = db.fetch_all(
        """SELECT id, kb_id, field_type, content,
                  1 - (embedding <=> %s::vector) AS cos
           FROM kb_chunks
           ORDER BY embedding <=> %s::vector
           LIMIT %s;""",
        (qvec_literal, qvec_literal, config.VECTOR_K),
    )
    return [{"chunk_id": r[0], "kb_id": r[1], "field_type": r[2],
             "content": r[3], "cos": float(r[4])} for r in rows]


def _keyword_leg(query: str) -> list[dict]:
    rows = db.fetch_all(
        """SELECT id, kb_id, field_type, content,
                  ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) AS rank
           FROM kb_chunks
           WHERE tsv @@ websearch_to_tsquery('english', %s)
           ORDER BY rank DESC
           LIMIT %s;""",
        (query, query, config.KEYWORD_K),
    )
    return [{"chunk_id": r[0], "kb_id": r[1], "field_type": r[2],
             "content": r[3], "rank": float(r[4])} for r in rows]


def hybrid_search(query: str, return_k: int | None = None) -> list[dict]:
    """Return distinct-article candidates, best fused first."""
    return_k = return_k or config.RERANK_K
    qvec = embeddings.to_pgvector(embeddings.embed_query(query))

    vec_rows = _vector_leg(qvec)
    kw_rows = _keyword_leg(query)

    # RRF over chunk ids.
    rrf: dict[int, float] = {}
    for rank, row in enumerate(vec_rows, start=1):
        rrf[row["chunk_id"]] = rrf.get(row["chunk_id"], 0.0) + 1.0 / (config.RRF_K + rank)
    for rank, row in enumerate(kw_rows, start=1):
        rrf[row["chunk_id"]] = rrf.get(row["chunk_id"], 0.0) + 1.0 / (config.RRF_K + rank)

    # Best cosine per kb_id (from the vector leg).
    best_cos: dict[str, float] = {}
    for row in vec_rows:
        best_cos[row["kb_id"]] = max(best_cos.get(row["kb_id"], 0.0), row["cos"])

    # Aggregate chunk RRF -> article RRF (max chunk score per kb_id).
    art_rrf: dict[str, float] = {}
    chunk_to_kb = {r["chunk_id"]: r["kb_id"] for r in (vec_rows + kw_rows)}
    for chunk_id, score in rrf.items():
        kb = chunk_to_kb[chunk_id]
        art_rrf[kb] = max(art_rrf.get(kb, 0.0), score)

    if not art_rrf:
        return []

    # Pull article fields for the surviving kb_ids.
    kb_ids = sorted(art_rrf, key=lambda k: -art_rrf[k])[:return_k]
    placeholders = ",".join(["%s"] * len(kb_ids))
    arows = db.fetch_all(
        f"""SELECT kb_id, title, question, cause, environment,
                   guidance_troubleshoot, embed_text
            FROM kb_articles WHERE kb_id IN ({placeholders});""",
        tuple(kb_ids),
    )
    by_id = {r[0]: r for r in arows}

    candidates = []
    for kb in kb_ids:
        r = by_id.get(kb)
        if not r:
            continue
        candidates.append({
            "kb_id": kb,
            "title": r[1], "question": r[2], "cause": r[3],
            "environment": r[4], "guidance_troubleshoot": r[5],
            "embed_text": r[6],
            "cos": round(best_cos.get(kb, 0.0), 4),
            "fused": round(art_rrf[kb], 6),
        })
    candidates.sort(key=lambda c: -c["fused"])
    logger.info("hybrid_search %r -> %d articles (top=%s cos=%.3f)",
                query[:80], len(candidates),
                candidates[0]["kb_id"] if candidates else None,
                candidates[0]["cos"] if candidates else 0.0)
    return candidates
