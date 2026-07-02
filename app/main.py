"""
main.py — KB Candidates tool, PGVECTOR + EMBEDDING + COHERE-RERANK edition.

Same /get_kb_candidates contract and the SAME routing engine as the legacy
AI-Search main.py (compute_spread → resolve-vs-follow-up, per-session round cap,
RESOLVE / ASK-SYMPTOMS / ASK-FREEFORM bands). Only the retrieval underneath is
swapped:

    description ─▶ embed (text-embedding-3-small)
                ─▶ pgvector cosine KNN  +  Postgres full-text  ─▶ RRF fuse
                ─▶ Cohere rerank v4 (0..1 relevance; cosine fallback)
                ─▶ score bands + spread + round cap
                ─▶ {followup_required, kb_id, top_score, discriminating_symptoms, message}

Follow-up grounding uses heading/cause/question (no Symptoms field in prod):
the best phrase that matches the user's description AND splits the candidates is
returned; if none is relevant enough, the agent is given freedom to ask its own
focused Outlook follow-up. (see followup.py)

`discriminating_symptoms` is kept as the response key for contract compatibility;
it now carries heading/cause/question-derived phrases.
"""
from __future__ import annotations

import os
import sys
import time
import logging
from threading import Lock

# Make app/ importable whether launched as `app.main:app` or `main:app`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
import db
import rerank
import retrieval
from followup import select_discriminating_fields

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("kb-tool.pgvector")

RESOLVE_MESSAGE = os.environ.get(
    "RESOLVE_MESSAGE",
    "Confident match. Present this article to the user — no follow-up needed.",
)
NO_SYMPTOMS_MESSAGE = os.environ.get(
    "NO_SYMPTOMS_MESSAGE",
    "No discriminating details are available. Ask one focused follow-up question "
    "based on the user's own description, staying strictly on Outlook issues.",
)
DISPLAY_AT_CAP_MESSAGE = os.environ.get(
    "DISPLAY_AT_CAP_MESSAGE",
    "Reached the follow-up limit. Present this best-available match — no further follow-up.",
)
CONCLUDED_NO_MATCH_MESSAGE = os.environ.get(
    "CONCLUDED_NO_MATCH_MESSAGE",
    "Reached the follow-up limit with no sufficiently confident match. Stop asking "
    "and close or hand off per policy.",
)

app = FastAPI(title="KB Candidates Tool (pgvector + Cohere rerank)")


class CandidateRequest(BaseModel):
    description: str
    session_id: str | None = None


# ── Round tracking (in-memory, per-process — single worker) ────────────────────
_rounds_lock = Lock()
_round_state: dict[str, dict] = {}


def _bump_round(session_id: str | None) -> int:
    if not session_id:
        logger.warning("No session_id — round cap cannot be enforced this call.")
        return 1
    now = time.monotonic()
    with _rounds_lock:
        for s in [s for s, st in _round_state.items()
                  if now - st["ts"] > config.SESSION_TTL_SECONDS]:
            _round_state.pop(s, None)
        st = _round_state.setdefault(session_id, {"rounds": 0, "ts": now})
        st["rounds"] += 1
        st["ts"] = now
        return st["rounds"]


def _clear_round(session_id: str | None) -> None:
    if session_id:
        with _rounds_lock:
            _round_state.pop(session_id, None)


# ── Spread gate — identical logic to the legacy AI-Search main.py (0..1 scale) ──
def compute_spread(scores: list[float]) -> str:
    if not scores:
        return "low"
    weak_absolute = scores[0] < config.CONFIDENT_SCORE
    weak_dominance = len(scores) > 1 and (scores[0] - scores[1]) < config.SPREAD_THRESHOLD
    return "high" if (weak_absolute or weak_dominance) else "low"


# ── Scoring: Cohere rerank (0..1), graceful cosine fallback ────────────────────
def _score_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """Attach a 0..1 `score` to each candidate and return them best-first.
    Uses Cohere rerank when available; otherwise the embedding cosine."""
    if not candidates:
        return candidates
    docs = [c.get("embed_text") or c.get("title") or "" for c in candidates]
    pairs = rerank.rerank(query, docs, top_n=len(docs))
    if pairs is not None:
        ordered = []
        seen = set()
        for idx, score in pairs:
            c = candidates[idx]
            c["score"] = round(float(score), 4)
            c["scorer"] = "cohere"
            ordered.append(c)
            seen.add(idx)
        for idx, c in enumerate(candidates):     # any not returned -> cosine tail
            if idx not in seen:
                c["score"] = c.get("cos", 0.0)
                c["scorer"] = "cohere-missing"
                ordered.append(c)
        return ordered
    # Fallback: cosine similarity as the 0..1 score.
    for c in candidates:
        c["score"] = c.get("cos", 0.0)
        c["scorer"] = "cosine"
    candidates.sort(key=lambda c: -c["score"])
    return candidates


def _check_key(x_api_key: str):
    if config.REQUIRE_API_KEY and config.TOOL_API_KEY and x_api_key != config.TOOL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.exception_handler(Exception)
async def _global(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.get("/healthz")
def health():
    try:
        n = db.fetch_all("SELECT count(*) FROM kb_articles;")[0][0]
    except Exception as e:
        return JSONResponse(status_code=503,
                            content={"status": "degraded", "error": f"{type(e).__name__}: {e}"})
    return {"status": "ok", "engine": "pgvector+cohere-rerank",
            "articles": n, "rerank_enabled": config.RERANK_ENABLED,
            "tools": ["get_kb_candidates"]}


@app.post("/get_kb_candidates")
def get_kb_candidates(body: CandidateRequest, x_api_key: str = Header(default="")):
    _check_key(x_api_key)
    description = (body.description or "").strip()
    if not description:
        raise HTTPException(status_code=400, detail="'description' is required")

    session_id = (body.session_id or "").strip() or None
    round_no = _bump_round(session_id)

    candidates = retrieval.hybrid_search(description, return_k=config.RERANK_K)
    candidates = _score_candidates(description, candidates)
    top_score = candidates[0]["score"] if candidates else 0.0
    spread = compute_spread([c["score"] for c in candidates])
    logger.info("get_kb_candidates | round=%d | session=%s | top=%.3f | spread=%s | "
                "scorer=%s | q=%r", round_no, session_id, top_score, spread,
                candidates[0]["scorer"] if candidates else "-", description[:160])

    # RESOLVE — confident AND a clear winner.
    if candidates and top_score > config.CONFIDENT_SCORE and spread == "low":
        _clear_round(session_id)
        return {"followup_required": False, "kb_id": candidates[0]["kb_id"],
                "top_score": top_score, "message": RESOLVE_MESSAGE}

    # ROUND CAP — stop looping, conclude.
    if round_no >= config.MAX_ROUNDS:
        _clear_round(session_id)
        if candidates and top_score >= config.MIN_DISPLAY_SCORE:
            return {"followup_required": False, "kb_id": candidates[0]["kb_id"],
                    "top_score": top_score, "message": DISPLAY_AT_CAP_MESSAGE}
        return {"followup_required": False, "kb_id": None,
                "top_score": top_score, "message": CONCLUDED_NO_MATCH_MESSAGE}

    # Sub-confident → follow-up. Ground it in heading/cause/question when a phrase
    # is relevant to the description; otherwise let the agent ask its own.
    phrases: list[str] = []
    if candidates and top_score >= config.FOLLOWUP_FLOOR:
        phrases = select_discriminating_fields(description, candidates[:config.RETURN_K])
    if phrases:
        return {"followup_required": True, "top_score": top_score,
                "discriminating_symptoms": phrases}
    return {"followup_required": True, "top_score": top_score,
            "discriminating_symptoms": [], "message": NO_SYMPTOMS_MESSAGE}


@app.post("/debug_search")
def debug_search(body: CandidateRequest, x_api_key: str = Header(default="")):
    _check_key(x_api_key)
    description = (body.description or "").strip()
    if not description:
        raise HTTPException(status_code=400, detail="'description' is required")
    candidates = retrieval.hybrid_search(description, return_k=config.RERANK_K)
    candidates = _score_candidates(description, candidates)
    returned = candidates[:config.RETURN_K]
    return {
        "query": description,
        "scorer": candidates[0]["scorer"] if candidates else "-",
        "results": [{"rank": i + 1, "kb_id": c["kb_id"], "score": c["score"],
                     "cos": c["cos"], "fused": c["fused"], "title": c["title"],
                     "question": c["question"], "cause": c["cause"]}
                    for i, c in enumerate(candidates)],
        "spread_on_returned": compute_spread([c["score"] for c in returned]),
        "discriminating_fields": select_discriminating_fields(description, returned),
    }
