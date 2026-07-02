"""
main.py — the KB Candidates service (pgvector + embeddings + Cohere rerank).

Object model:
    SessionRounds   - in-memory per-session round counter (the round cap).
    KBService       - the decision engine: retrieve -> score -> spread -> route.
    create_app()    - builds the FastAPI app around a service (reused by the
                      traced entry point so tracing adds no duplicate endpoints).

Composition root at the bottom wires the concrete components together into the
default SERVICE and exposes `app`. Same /get_kb_candidates contract and routing
as the legacy AI-Search version — only the retrieval technology differs.
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

from config import CONFIG
from db import Database
from embeddings import EmbeddingClient
from rerank import RerankClient
from retrieval import HybridRetriever
from text_utils import NORMALIZER
from followup import FollowupSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("kb-tool.pgvector")


# ══════════════════════════════════════════════════════════════════════════════
class SessionRounds:
    """Counts follow-up calls per session_id so a conversation can't loop forever.
    In-memory / per-process -> the service runs a single worker."""

    def __init__(self, ttl_seconds: int):
        self._ttl = ttl_seconds        # idle sessions are pruned after this
        self._lock = Lock()
        self._state: dict[str, dict] = {}   # session_id -> {"rounds": int, "ts": float}

    def bump(self, session_id: "str | None") -> int:
        """Increment and return this session's round count (1 if no session_id)."""
        if not session_id:
            logger.warning("No session_id — round cap cannot be enforced this call.")
            return 1
        now = time.monotonic()
        with self._lock:
            for s in [s for s, st in self._state.items() if now - st["ts"] > self._ttl]:
                self._state.pop(s, None)       # prune idle sessions
            st = self._state.setdefault(session_id, {"rounds": 0, "ts": now})
            st["rounds"] += 1
            st["ts"] = now
            return st["rounds"]

    def clear(self, session_id: "str | None") -> None:
        """Drop a session's counter once the conversation concludes."""
        if session_id:
            with self._lock:
                self._state.pop(session_id, None)

    def reset(self) -> None:
        """Wipe all counters (used by the eval harness between test cases)."""
        with self._lock:
            self._state.clear()


# ══════════════════════════════════════════════════════════════════════════════
class KBService:
    """The decision engine. Depends on injected components so it can be traced or
    tested by swapping any of them."""

    def __init__(self, config, db: Database, retriever: HybridRetriever,
                 reranker: RerankClient, followup: FollowupSelector, rounds: SessionRounds):
        self._config = config
        self._db = db
        self._retriever = retriever
        self._reranker = reranker
        self._followup = followup
        self._rounds = rounds
        # Agent-facing messages (env-overridable).
        self.RESOLVE_MESSAGE = os.environ.get(
            "RESOLVE_MESSAGE",
            "Confident match. Present this article to the user — no follow-up needed.")
        self.NO_SYMPTOMS_MESSAGE = os.environ.get(
            "NO_SYMPTOMS_MESSAGE",
            "No discriminating details are available. Ask one focused follow-up question "
            "based on the user's own description, staying strictly on Outlook issues.")
        self.DISPLAY_AT_CAP_MESSAGE = os.environ.get(
            "DISPLAY_AT_CAP_MESSAGE",
            "Reached the follow-up limit. Present this best-available match — no further follow-up.")
        self.CONCLUDED_NO_MATCH_MESSAGE = os.environ.get(
            "CONCLUDED_NO_MATCH_MESSAGE",
            "Reached the follow-up limit with no sufficiently confident match. Stop asking "
            "and close or hand off per policy.")

    @property
    def rounds(self) -> SessionRounds:
        return self._rounds

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    def rank(self, query: str) -> list[dict]:
        """Ordered, scored candidates for one query (retrieval + scoring). Used by
        the eval harness / calibration where the full ranked list is needed."""
        return self.score_candidates(query, self._retriever.search(query, self._config.RERANK_K))

    # ── scoring: Cohere rerank (0..1) with graceful cosine fallback ───────────
    def score_candidates(self, query: str, candidates: list[dict]) -> list[dict]:
        """Attach a 0..1 `score` to each candidate and return them best-first."""
        if not candidates:
            return candidates
        docs = [c.get("embed_text") or c.get("title") or "" for c in candidates]
        pairs = self._reranker.rerank(query, docs, top_n=len(docs))
        if pairs is not None:                              # reranker available
            ordered, seen = [], set()
            for idx, score in pairs:
                c = candidates[idx]
                c["score"] = round(float(score), 4)
                c["scorer"] = "cohere"
                ordered.append(c)
                seen.add(idx)
            for idx, c in enumerate(candidates):           # any not returned -> cosine tail
                if idx not in seen:
                    c["score"] = c.get("cos", 0.0)
                    c["scorer"] = "cohere-missing"
                    ordered.append(c)
            return ordered
        for c in candidates:                               # fallback: cosine as the score
            c["score"] = c.get("cos", 0.0)
            c["scorer"] = "cosine"
        candidates.sort(key=lambda c: -c["score"])
        return candidates

    # ── spread gate: is there a clear, strong winner? ─────────────────────────
    def compute_spread(self, scores: list[float]) -> str:
        """'high' = no clear winner (weak top OR near-tie with #2); else 'low'."""
        c = self._config
        if not scores:
            return "low"
        weak_absolute = scores[0] < c.CONFIDENT_SCORE
        weak_dominance = len(scores) > 1 and (scores[0] - scores[1]) < c.SPREAD_THRESHOLD
        return "high" if (weak_absolute or weak_dominance) else "low"

    # ── the routing decision (one tool call / one turn) ───────────────────────
    def classify(self, description: str, session_id: "str | None") -> dict:
        """Run the full pipeline and return the tool's contract response."""
        c = self._config
        round_no = self._rounds.bump(session_id)
        candidates = self._retriever.search(description, return_k=c.RERANK_K)
        candidates = self.score_candidates(description, candidates)
        top_score = candidates[0]["score"] if candidates else 0.0
        spread = self.compute_spread([x["score"] for x in candidates])
        logger.info("classify | round=%d | session=%s | top=%.3f | spread=%s | scorer=%s | q=%r",
                    round_no, session_id, top_score, spread,
                    candidates[0]["scorer"] if candidates else "-", description[:160])

        # RESOLVE — confident AND a clear winner.
        if candidates and top_score > c.CONFIDENT_SCORE and spread == "low":
            self._rounds.clear(session_id)
            return {"followup_required": False, "kb_id": candidates[0]["kb_id"],
                    "top_score": top_score, "message": self.RESOLVE_MESSAGE}

        # ROUND CAP — stop looping, conclude.
        if round_no >= c.MAX_ROUNDS:
            self._rounds.clear(session_id)
            if candidates and top_score >= c.MIN_DISPLAY_SCORE:
                return {"followup_required": False, "kb_id": candidates[0]["kb_id"],
                        "top_score": top_score, "message": self.DISPLAY_AT_CAP_MESSAGE}
            return {"followup_required": False, "kb_id": None,
                    "top_score": top_score, "message": self.CONCLUDED_NO_MATCH_MESSAGE}

        # Sub-confident -> follow-up (grounded when a relevant phrase exists).
        phrases: list[str] = []
        if candidates and top_score >= c.FOLLOWUP_FLOOR:
            phrases = self._followup.select(description, candidates[:c.RETURN_K])
        if phrases:
            return {"followup_required": True, "top_score": top_score,
                    "discriminating_symptoms": phrases}
        return {"followup_required": True, "top_score": top_score,
                "discriminating_symptoms": [], "message": self.NO_SYMPTOMS_MESSAGE}

    # ── diagnostics + health ──────────────────────────────────────────────────
    def debug(self, description: str) -> dict:
        """Full ranked view for /debug_search (the agent never calls this)."""
        c = self._config
        candidates = self.score_candidates(description, self._retriever.search(description, c.RERANK_K))
        returned = candidates[:c.RETURN_K]
        return {
            "query": description,
            "scorer": candidates[0]["scorer"] if candidates else "-",
            "results": [{"rank": i + 1, "kb_id": x["kb_id"], "score": x["score"],
                         "cos": x["cos"], "fused": x["fused"], "title": x["title"],
                         "question": x["question"], "cause": x["cause"]}
                        for i, x in enumerate(candidates)],
            "spread_on_returned": self.compute_spread([x["score"] for x in returned]),
            "discriminating_fields": self._followup.select(description, returned),
        }

    def health(self) -> dict:
        """Liveness + article count (raises if the DB is unreachable)."""
        n = self._db.fetch_all("SELECT count(*) FROM kb_articles;")[0][0]
        return {"status": "ok", "engine": "pgvector+cohere-rerank", "articles": n,
                "rerank_enabled": self._config.RERANK_ENABLED, "tools": ["get_kb_candidates"]}


# ══════════════════════════════════════════════════════════════════════════════
def build_default_service(config=CONFIG) -> KBService:
    """Compose the concrete components into a ready-to-use service."""
    db = Database(config)
    embedder = EmbeddingClient(config)
    retriever = HybridRetriever(config, db, embedder)
    reranker = RerankClient(config)
    followup = FollowupSelector(config, NORMALIZER)
    rounds = SessionRounds(config.SESSION_TTL_SECONDS)
    return KBService(config, db, retriever, reranker, followup, rounds)


class CandidateRequest(BaseModel):
    description: str
    session_id: "str | None" = None


def create_app(service: KBService, config=CONFIG) -> FastAPI:
    """Build the FastAPI app around a service. Shared by main.py and the traced
    entry point so the endpoints are defined exactly once."""
    app = FastAPI(title="KB Candidates Tool (pgvector + Cohere rerank)")

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
            return service.health()
        except Exception as exc:
            return JSONResponse(status_code=503,
                                content={"status": "degraded", "error": f"{type(exc).__name__}: {exc}"})

    @app.post("/get_kb_candidates")
    def get_kb_candidates(body: CandidateRequest, x_api_key: str = Header(default="")):
        _check_key(x_api_key)
        description = (body.description or "").strip()
        if not description:
            raise HTTPException(status_code=400, detail="'description' is required")
        session_id = (body.session_id or "").strip() or None
        return service.classify(description, session_id)

    @app.post("/debug_search")
    def debug_search(body: CandidateRequest, x_api_key: str = Header(default="")):
        _check_key(x_api_key)
        description = (body.description or "").strip()
        if not description:
            raise HTTPException(status_code=400, detail="'description' is required")
        return service.debug(description)

    return app


# ── composition root (what uvicorn app.main:app imports) ──────────────────────
SERVICE = build_default_service()
app = create_app(SERVICE, CONFIG)
