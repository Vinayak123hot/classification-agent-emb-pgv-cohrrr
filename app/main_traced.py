"""
main_traced.py — traced entry point (deployed as app.main_traced:app).

Instead of monkey-patching, each pipeline component is subclassed to a Traced*
version that logs its step and then calls super(). A traced KBService is composed
from them, wrapped by the SAME create_app() factory as the plain service, plus an
HTTP middleware (opens/closes the trace) and a browser viewer at /trace.

Captured per /get_kb_candidates call, in order: request -> embedding -> vector leg
-> keyword leg -> RRF candidates -> scoring -> spread -> follow-up -> response ->
what-happens-next. Turns group by session_id.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.responses import HTMLResponse, PlainTextResponse
from starlette.responses import Response

from config import CONFIG
from db import Database
from embeddings import EmbeddingClient
from rerank import RerankClient
from retrieval import HybridRetriever
from text_utils import NORMALIZER
from followup import FollowupSelector
from trace_kit import Tracer, short
import main as core                     # SessionRounds, KBService, create_app


# ══════════════════════════════════════════════════════════════════════════════
# Traced component subclasses — each logs its step, then delegates to super().
# ══════════════════════════════════════════════════════════════════════════════
class TracedEmbeddingClient(EmbeddingClient):
    def __init__(self, config, tracer: Tracer):
        super().__init__(config)
        self._tracer = tracer

    def embed_query(self, text: str):
        vec = super().embed_query(text)
        t = self._tracer.current()
        if t is not None:
            norm = sum(x * x for x in vec) ** 0.5
            t.step("EMBEDDING — query -> text-embedding-3-small")
            t.kv("model", self._config.EMBED_DEPLOYMENT)
            t.kv("query", repr(short(text, 200)))
            t.kv("dimensions", len(vec))
            t.kv("L2 norm (should be ~1.0)", round(norm, 4))
            t.kv("first 5 values", [round(x, 4) for x in vec[:5]])
            t.note("This normalized vector is compared to every KB chunk's stored "
                   "embedding using pgvector cosine distance (<=>) in the vector leg.")
        return vec


class TracedHybridRetriever(HybridRetriever):
    def __init__(self, config, db, embedder, tracer: Tracer):
        super().__init__(config, db, embedder)
        self._tracer = tracer

    def _vector_leg(self, qvec_literal: str):
        rows = super()._vector_leg(qvec_literal)
        t = self._tracer.current()
        if t is not None:
            t.step("VECTOR LEG — pgvector cosine KNN (kb_chunks.embedding <=> query)")
            t.kv("chunks returned", f"{len(rows)} (VECTOR_K={self._config.VECTOR_K})")
            t.log("")
            for i, r in enumerate(rows, 1):
                t.log(f"#{i:>2}  {r['kb_id']:<12} [{r['field_type']:<8}] cos={r['cos']:.4f}  "
                      f"\"{short(r['content'], 70)}\"")
        return rows

    def _keyword_leg(self, query: str):
        rows = super()._keyword_leg(query)
        t = self._tracer.current()
        if t is not None:
            t.step("KEYWORD LEG — Postgres full-text (tsv @@ websearch_to_tsquery)")
            t.kv("matches", f"{len(rows)} (KEYWORD_K={self._config.KEYWORD_K})")
            t.note("Lexical signal (ts_rank_cd). Not true BM25; the vector leg + rerank "
                   "carry ranking quality, this catches exact-word overlaps.")
            t.log("")
            for i, r in enumerate(rows, 1):
                t.log(f"#{i:>2}  {r['kb_id']:<12} [{r['field_type']:<8}] rank={r['rank']:.4f}")
            if not rows:
                t.log("(no lexical matches)")
        return rows

    def search(self, query: str, return_k=None):
        cands = super().search(query, return_k=return_k)   # runs the (traced) legs first
        t = self._tracer.current()
        if t is not None:
            t.step("RRF FUSION -> DISTINCT CANDIDATES")
            t.note(f"Vector + keyword chunk rankings fused by RRF (score = sum 1/(RRF_K+rank), "
                   f"RRF_K={self._config.RRF_K}), then collapsed to distinct articles (best chunk "
                   f"per kb_id). Top {self._config.RERANK_K} go to the reranker.")
            t.log("")
            for i, c in enumerate(cands, 1):
                t.log(f"#{i:>2}  {c['kb_id']:<12} fused={c['fused']:.5f}  cos={c['cos']:.4f}")
                t.log(f"      title   : {short(c.get('title'), 90)}", indent=1)
                if c.get("question"):
                    t.log(f"      question: {short(c.get('question'), 90)}", indent=1)
                if c.get("cause"):
                    t.log(f"      cause   : {short(c.get('cause'), 90)}", indent=1)
        return cands


class TracedFollowupSelector(FollowupSelector):
    def __init__(self, config, normalizer, tracer: Tracer):
        super().__init__(config, normalizer)
        self._tracer = tracer

    def select(self, description, candidates, top_k=None, min_score=None):
        t = self._tracer.current()
        if t is not None:
            t.step("FOLLOW-UP SELECTION — grounded phrase from heading/cause/question")
            t.note("Each candidate phrase is scored: rel = cosine(description, phrase); "
                   "mass = share of candidate score carrying it; dist = 1-|2*mass-1|; "
                   f"final = 0.5*rel+0.5*dist. Kept only if final >= MIN_FIELD_SCORE "
                   f"({self._config.MIN_FIELD_SCORE}); else the agent asks freeform.")
            try:
                total = sum(float(c.get("score", c.get("cos", 0.0))) for c in candidates) or 1.0
                dvec = self._tf_vector(description)
                t.log("")
                t.kv("description tokens", sorted(dvec.keys()))
                for c in candidates:
                    sc = float(c.get("score", c.get("cos", 0.0)))
                    for ph in self._candidate_phrases(c):
                        rel = self._cosine(dvec, self._tf_vector(ph))
                        mass = sc / total
                        dist = 1.0 - abs(2.0 * mass - 1.0)
                        final = rel * 0.5 + dist * 0.5
                        keep = "KEEP" if final >= self._config.MIN_FIELD_SCORE else "drop"
                        t.log(f"[{keep}] final={final:.3f} (rel={rel:.3f} mass={mass:.3f} "
                              f"dist={dist:.3f}) {c['kb_id']}: \"{short(ph, 70)}\"")
            except Exception as exc:
                t.log(f"(scoring preview unavailable: {type(exc).__name__}: {exc})")
        result = super().select(description, candidates, top_k=top_k, min_score=min_score)
        if t is not None:
            t.log("")
            t.log(f"SELECTED ({len(result)}): {result}")
            t.note("Grounded phrases -> the agent's follow-up question." if result
                   else "Empty -> the agent asks its OWN focused Outlook follow-up.")
        return result


class TracedKBService(core.KBService):
    def __init__(self, config, db, retriever, reranker, followup, rounds, tracer: Tracer):
        super().__init__(config, db, retriever, reranker, followup, rounds)
        self._tracer = tracer

    def score_candidates(self, query, candidates):
        out = super().score_candidates(query, candidates)
        t = self._tracer.current()
        if t is not None and out:
            scorer = out[0].get("scorer", "?")
            t.step("SCORING — assign 0..1 relevance and re-order")
            if scorer.startswith("cohere"):
                t.kv("scorer", f"Cohere rerank ({self._config.RERANK_DEPLOYMENT})")
            else:
                t.kv("scorer", "COSINE FALLBACK (Cohere rerank unavailable)")
            t.log("")
            for i, c in enumerate(out, 1):
                t.log(f"#{i:>2}  {c['kb_id']:<12} score={c['score']:.4f}  (cos={c.get('cos'):.4f})")
        return out

    def compute_spread(self, scores):
        result = super().compute_spread(scores)
        t = self._tracer.current()
        if t is not None:
            c = self._config
            t.step("SPREAD — resolve-vs-follow-up decision (0..1 scale)")
            t.kv("candidate scores", [round(s, 4) for s in scores])
            if scores:
                t.kv("check 1 - weak absolute",
                     f"top {scores[0]:.4f} < CONFIDENT_SCORE {c.CONFIDENT_SCORE} ? -> "
                     f"{scores[0] < c.CONFIDENT_SCORE}")
                if len(scores) > 1:
                    gap = round(scores[0] - scores[1], 4)
                    t.kv("check 2 - weak dominance",
                         f"gap #1-#2 = {gap} < SPREAD_THRESHOLD {c.SPREAD_THRESHOLD} ? -> "
                         f"{gap < c.SPREAD_THRESHOLD}")
            t.kv("VERDICT", result.upper())
            t.note("HIGH = no clear winner -> follow-up." if result == "high"
                   else "LOW = confident clear winner -> may RESOLVE.")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Composition root for the traced service + app.
# ══════════════════════════════════════════════════════════════════════════════
TRACER = Tracer()
_db = Database(CONFIG)
_embedder = TracedEmbeddingClient(CONFIG, TRACER)
_retriever = TracedHybridRetriever(CONFIG, _db, _embedder, TRACER)
_reranker = RerankClient(CONFIG)
_followup = TracedFollowupSelector(CONFIG, NORMALIZER, TRACER)
_rounds = core.SessionRounds(CONFIG.SESSION_TTL_SECONDS)
SERVICE = TracedKBService(CONFIG, _db, _retriever, _reranker, _followup, _rounds, TRACER)

app = core.create_app(SERVICE, CONFIG)
_TRACED_PATHS = ("/get_kb_candidates", "/debug_search")


@app.middleware("http")
async def _trace_middleware(request, call_next):
    if request.url.path not in _TRACED_PATHS:
        return await call_next(request)

    body_bytes = await request.body()
    t = TRACER.start(f"POST {request.url.path}")
    t.step(f"AGENT -> TOOL CALL RECEIVED: {request.url.path.lstrip('/')}")
    key = request.headers.get("x-api-key", "")
    t.kv("x-api-key", f"present ({key[:4]}...)" if key else "none (REQUIRE_API_KEY may be false)")
    try:
        payload = json.loads(body_bytes.decode("utf-8") or "{}")
    except Exception:
        payload = {"_raw": body_bytes.decode("utf-8", "replace")}
    t.log("request body:")
    t.log(json.dumps(payload, indent=2, ensure_ascii=False, default=str), indent=2)

    # Group all rounds of one conversation by session_id (or an explicit header).
    turn_header = os.environ.get("TRACE_TURN_HEADER", "").strip().lower()
    header_key = request.headers.get(turn_header) if turn_header else None
    sid = payload.get("session_id") if isinstance(payload, dict) else None
    t.meta["turn_key"] = header_key or sid or None
    if isinstance(payload, dict):
        t.meta["description"] = payload.get("description") or ""

    status, error = 500, None
    try:
        response = await call_next(request)
        status = response.status_code
        chunks = [section async for section in response.body_iterator]
        resp_bytes = b"".join(chunks)
        try:
            resp_json = json.loads(resp_bytes.decode("utf-8"))
        except Exception:
            resp_json = None
        t.step(f"TOOL -> AGENT RESPONSE (HTTP {status})")
        t.log(json.dumps(resp_json if resp_json is not None else resp_bytes.decode("utf-8", "replace"),
                         indent=2, ensure_ascii=False, default=str), indent=2)
        if status == 200 and isinstance(resp_json, dict) and request.url.path == "/get_kb_candidates":
            _explain_next(t, resp_json)
        return Response(content=resp_bytes, status_code=status,
                        headers=dict(response.headers), media_type=response.media_type)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        TRACER.finish(t, status=status, error=error)


def _explain_next(t, resp: dict):
    """Append a 'what happens next' interpretation of the response."""
    t.step("WHAT HAPPENS NEXT (per classification-agent protocol)")
    followup_req = resp.get("followup_required")
    top, kb_id = resp.get("top_score"), resp.get("kb_id")
    symptoms, msg = resp.get("discriminating_symptoms") or [], resp.get("message")
    if followup_req is False:
        if kb_id:
            t.log(f"followup_required=FALSE with kb_id={kb_id} (top_score={top}).")
            t.log("-> The agent PRESENTS this article; no follow-up.")
        else:
            t.log(f"followup_required=FALSE with kb_id=None (top_score={top}).")
            t.log("-> No confident match. The agent closes / hands off.")
        if msg:
            t.log(f"   tool message: {msg}", indent=2)
        return
    if symptoms:
        t.log(f"followup_required=TRUE (top_score={top}) — ASK, grounded in these phrases:")
        for s in symptoms:
            t.log(f"   - {s}", indent=2)
    else:
        t.log(f"followup_required=TRUE (top_score={top}) — ASK FREEFORM (no relevant phrase).")
        if msg:
            t.log(f"   tool message: {msg}", indent=2)
    t.note(f"Round cap: after MAX_ROUNDS ({CONFIG.MAX_ROUNDS}) follow-ups the tool force-concludes.")


# ══════════════════════════════════════════════════════════════════════════════
# Browser viewer — /trace (index of turns), /trace?turn=..., /trace?day=...
# ══════════════════════════════════════════════════════════════════════════════
_CSS = ("body{background:#0d1117;color:#c9d1d9;font-family:Consolas,monospace;margin:16px}"
        "a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}"
        ".blk{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;"
        "margin:14px 0;white-space:pre-wrap;font-size:12.5px;line-height:1.45}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "td,th{border-bottom:1px solid #21262d;padding:7px 10px;text-align:left}")
_BLOCK_SPLIT = re.compile(r"(?=^={20,}\nTRACE )", re.M)


def _check_view_key(key: str):
    from fastapi import HTTPException
    expected = os.environ.get("TRACE_VIEW_KEY", "").strip() or CONFIG.TOOL_API_KEY
    if expected and key != expected:
        raise HTTPException(status_code=401, detail="missing or invalid ?key=")


def _page(title: str, top: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"<!doctype html><html><head><title>{title}</title>"
                        f"<style>{_CSS}</style></head><body>{top}{body}</body></html>")


@app.get("/trace/raw", response_class=PlainTextResponse)
def trace_raw(key: str = "", day: str = "", turn: str = ""):
    _check_view_key(key)
    if turn:
        return TRACER.read_turn(turn)
    days = TRACER.trace_days()
    return TRACER.read_day(day if day in days else days[-1]) if days else "No traces yet."


@app.get("/trace", response_class=HTMLResponse)
def trace_view(key: str = "", turn: str = "", day: str = ""):
    _check_view_key(key)
    k = html.escape(key)
    if turn:                                                # one conversation, full story
        blocks = [b for b in _BLOCK_SPLIT.split(TRACER.read_turn(turn)) if b.strip()]
        top = f"<a href='/trace?key={k}'>&larr; all turns</a> &nbsp; <b>{html.escape(turn)}</b>"
        body = "\n".join(f"<pre class='blk'>{html.escape(b.rstrip())}</pre>" for b in blocks)
        return _page(f"turn {turn}", top, body)
    if day:                                                 # flat day view
        days = TRACER.trace_days()
        sel = day if day in days else (days[-1] if days else "")
        blocks = list(reversed([b for b in _BLOCK_SPLIT.split(TRACER.read_day(sel)) if b.strip()])) if sel else []
        top = f"<a href='/trace?key={k}'>&larr; turns</a> &nbsp; <b>day {sel}</b>"
        body = "\n".join(f"<pre class='blk'>{html.escape(b.rstrip())}</pre>" for b in blocks[:80])
        return _page(f"day {sel}", top, body)
    turns = TRACER.turn_files()                             # index of turns
    days = " · ".join(f"<a href='/trace?key={k}&day={d}'>{d}</a>" for d in TRACER.trace_days()[-14:]) or "—"
    top = f"<b>KB trace (pgvector) — turns</b> ({len(turns)}) &nbsp; days: {days}"
    if not turns:
        return _page("turns", top, "<p>No turns yet — call the tool once and reload.</p>")
    rows = ["<table><tr><th>when</th><th>initial user description</th><th>calls</th><th></th></tr>"]
    for tt in turns:
        rows.append(f"<tr><td>{html.escape(tt['opened'].replace('TURN ','').split('|')[-1].strip())}</td>"
                    f"<td><a href='/trace?key={k}&turn={html.escape(tt['name'])}'>"
                    f"{html.escape(tt['desc'] or tt['name'])}</a></td><td>{tt['calls']}</td>"
                    f"<td><a href='/trace/raw?key={k}&turn={html.escape(tt['name'])}'>raw</a></td></tr>")
    rows.append("</table>")
    return _page("turns", top, "\n".join(rows))
