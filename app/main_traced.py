"""
main_traced.py — drop-in traced version of the pgvector KB tool service.

ZERO changes to main.py: it imports the existing FastAPI app and wraps every
pipeline stage plus an HTTP middleware, so each agent tool call produces ONE
human-readable, step-by-step trace block under TRACE_LOG_DIR (default
/home/LogFiles/kbtrace on Azure), grouped per conversation turn — exactly like
teva-kb-trace.

Captured for every /get_kb_candidates call, in order:
    1  AGENT -> TOOL CALL RECEIVED (request body, session_id, description)
    2  EMBEDDING (query -> text-embedding-3-small vector)
    3  VECTOR LEG (pgvector cosine KNN over kb_chunks)
    4  KEYWORD LEG (Postgres full-text / tsv)
    5  RRF FUSION -> DISTINCT CANDIDATES
    6  SCORING (Cohere rerank v4, or cosine fallback)
    7  SPREAD (resolve-vs-follow-up decision math)
    8  FOLLOW-UP SELECTION (heading/cause/question rel/mass/dist) [when needed]
    9  TOOL -> AGENT RESPONSE + WHAT HAPPENS NEXT

Run it instead of app.main:app — behaviour is identical:
    uvicorn app.main_traced:app --host 0.0.0.0 --port 8000 --workers 1
Browser viewer at /trace?key=<TRACE_VIEW_KEY>.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from starlette.responses import Response

import main
import trace_kit as tk
import config
import retrieval
import embeddings
import followup

app = main.app
TRACED_PATHS = ("/get_kb_candidates", "/debug_search")


# ════════════════════════════════════════════════════════════════════
# 2. EMBEDDING — query -> text-embedding-3-small vector
# ════════════════════════════════════════════════════════════════════
_orig_embed = embeddings.embed_query

def _traced_embed(text: str):
    vec = _orig_embed(text)
    t = tk.current()
    if t is not None:
        norm = sum(x * x for x in vec) ** 0.5
        t.step("EMBEDDING — query -> text-embedding-3-small")
        t.kv("model", config.EMBED_DEPLOYMENT)
        t.kv("query", repr(tk.short(text, 200)))
        t.kv("dimensions", len(vec))
        t.kv("L2 norm (should be ~1.0)", round(norm, 4))
        t.kv("first 5 values", [round(x, 4) for x in vec[:5]])
        t.note("This normalized vector is compared to every KB chunk's stored "
               "embedding using pgvector cosine distance (<=>) in the vector leg.")
        t.data(dims=len(vec), norm=round(norm, 4))
    return vec

embeddings.embed_query = _traced_embed


# ════════════════════════════════════════════════════════════════════
# 3. VECTOR LEG — pgvector cosine KNN over kb_chunks
# ════════════════════════════════════════════════════════════════════
_orig_vec = retrieval._vector_leg

def _traced_vector_leg(qvec_literal: str):
    t0 = time.perf_counter()
    rows = _orig_vec(qvec_literal)
    ms = (time.perf_counter() - t0) * 1000.0
    t = tk.current()
    if t is not None:
        t.step("VECTOR LEG — pgvector cosine KNN (kb_chunks.embedding <=> query)",
               took_ms=round(ms, 1))
        t.kv("chunks returned", f"{len(rows)} (VECTOR_K={config.VECTOR_K})")
        t.kv("took", f"{ms:.0f} ms")
        t.log("")
        for i, r in enumerate(rows, 1):
            t.log(f"#{i:>2}  {r['kb_id']:<12} [{r['field_type']:<8}] cos={r['cos']:.4f}  "
                  f"\"{tk.short(r['content'], 70)}\"")
        t.data(vector_rows=[{"kb_id": r["kb_id"], "field": r["field_type"],
                             "cos": round(r["cos"], 4)} for r in rows])
    return rows

retrieval._vector_leg = _traced_vector_leg


# ════════════════════════════════════════════════════════════════════
# 4. KEYWORD LEG — Postgres full-text (tsv)
# ════════════════════════════════════════════════════════════════════
_orig_kw = retrieval._keyword_leg

def _traced_keyword_leg(query: str):
    t0 = time.perf_counter()
    rows = _orig_kw(query)
    ms = (time.perf_counter() - t0) * 1000.0
    t = tk.current()
    if t is not None:
        t.step("KEYWORD LEG — Postgres full-text (tsv @@ websearch_to_tsquery)",
               took_ms=round(ms, 1))
        t.kv("matches", f"{len(rows)} (KEYWORD_K={config.KEYWORD_K})")
        t.kv("took", f"{ms:.0f} ms")
        t.note("Lexical signal (ts_rank_cd). Not true BM25; the vector leg + rerank "
               "carry ranking quality, this catches exact-word overlaps.")
        t.log("")
        for i, r in enumerate(rows, 1):
            t.log(f"#{i:>2}  {r['kb_id']:<12} [{r['field_type']:<8}] rank={r['rank']:.4f}  "
                  f"\"{tk.short(r['content'], 60)}\"")
        if not rows:
            t.log("(no lexical matches)")
        t.data(keyword_rows=[{"kb_id": r["kb_id"], "field": r["field_type"],
                              "rank": round(r["rank"], 4)} for r in rows])
    return rows

retrieval._keyword_leg = _traced_keyword_leg


# ════════════════════════════════════════════════════════════════════
# 5. RRF FUSION -> DISTINCT CANDIDATES
# ════════════════════════════════════════════════════════════════════
_orig_hybrid = retrieval.hybrid_search

def _traced_hybrid(query: str, return_k=None):
    cands = _orig_hybrid(query, return_k=return_k)
    t = tk.current()
    if t is not None:
        t.step("RRF FUSION -> DISTINCT CANDIDATES")
        t.note(f"Vector + keyword chunk rankings are fused by Reciprocal Rank Fusion "
               f"(score = sum 1/(RRF_K+rank), RRF_K={config.RRF_K}), then collapsed to "
               f"distinct articles (best chunk per kb_id). 'cos' = best cosine for that "
               f"article; 'fused' = its RRF score. Top {config.RERANK_K} go to the reranker.")
        t.log("")
        for i, c in enumerate(cands, 1):
            t.log(f"#{i:>2}  {c['kb_id']:<12} fused={c['fused']:.5f}  cos={c['cos']:.4f}")
            t.log(f"      title   : {tk.short(c.get('title'), 90)}", indent=1)
            if c.get("question"):
                t.log(f"      question: {tk.short(c.get('question'), 90)}", indent=1)
            if c.get("cause"):
                t.log(f"      cause   : {tk.short(c.get('cause'), 90)}", indent=1)
        t.data(candidates=[{"kb_id": c["kb_id"], "fused": c["fused"], "cos": c["cos"]}
                           for c in cands])
    return cands

retrieval.hybrid_search = _traced_hybrid


# ════════════════════════════════════════════════════════════════════
# 6. SCORING — Cohere rerank v4 (or cosine fallback)
# ════════════════════════════════════════════════════════════════════
_orig_score = main._score_candidates

def _traced_score(query: str, candidates):
    out = _orig_score(query, candidates)
    t = tk.current()
    if t is not None and out:
        scorer = out[0].get("scorer", "?")
        t.step("SCORING — assign 0..1 relevance and re-order")
        if scorer.startswith("cohere"):
            t.kv("scorer", f"Cohere rerank ({config.RERANK_DEPLOYMENT})")
            t.note("Cross-encoder read the query together with each candidate's "
                   "title+question+cause and produced a 0..1 relevance_score; "
                   "candidates are re-ordered by it.")
        else:
            t.kv("scorer", "COSINE FALLBACK (Cohere rerank unavailable)")
            t.note("Reranker route not reachable, so the embedding cosine is used as "
                   "the 0..1 score. Ranking is already strong; enable rerank via "
                   "RERANK_PATH/RERANK_API_VERSION for tighter top-of-list ordering.")
        t.log("")
        for i, c in enumerate(out, 1):
            t.log(f"#{i:>2}  {c['kb_id']:<12} score={c['score']:.4f}  (cos={c.get('cos'):.4f})")
        t.data(scorer=scorer, scored=[{"kb_id": c["kb_id"], "score": c["score"]} for c in out])
    return out

main._score_candidates = _traced_score


# ════════════════════════════════════════════════════════════════════
# 7. SPREAD — resolve-vs-follow-up decision math
# ════════════════════════════════════════════════════════════════════
_orig_spread = main.compute_spread

def _traced_spread(scores):
    result = _orig_spread(scores)
    t = tk.current()
    if t is not None:
        t.step("SPREAD — retrieval-confidence decision (0..1 scale)")
        t.kv("candidate scores", [round(s, 4) for s in scores])
        if scores:
            weak_abs = scores[0] < config.CONFIDENT_SCORE
            t.kv("check 1 - weak absolute",
                 f"top {scores[0]:.4f} < CONFIDENT_SCORE {config.CONFIDENT_SCORE} ? -> {weak_abs}")
            if len(scores) > 1:
                gap = round(scores[0] - scores[1], 4)
                weak_dom = gap < config.SPREAD_THRESHOLD
                t.kv("check 2 - weak dominance",
                     f"gap #1-#2 = {gap} < SPREAD_THRESHOLD {config.SPREAD_THRESHOLD} ? -> {weak_dom}")
            else:
                t.kv("check 2 - weak dominance", "only one candidate -> n/a (False)")
        t.kv("VERDICT", result.upper())
        if result == "high":
            t.note("HIGH = no clear strong winner -> the endpoint will NOT resolve; it "
                   "returns followup_required=true and (if a field is relevant) grounds a "
                   "follow-up question in it.")
        else:
            t.note("LOW = confident, clear single winner -> combined with top_score > "
                   "CONFIDENT_SCORE this lets the endpoint RESOLVE.")
        t.data(scores=[round(s, 4) for s in scores], verdict=result)
    return result

main.compute_spread = _traced_spread


# ════════════════════════════════════════════════════════════════════
# 8. FOLLOW-UP SELECTION — heading/cause/question rel/mass/dist
# ════════════════════════════════════════════════════════════════════
_orig_fields = main.select_discriminating_fields

def _traced_fields(description, candidates, top_k=None, min_score=None):
    t = tk.current()
    if t is not None:
        t.step("FOLLOW-UP SELECTION — grounded phrase from heading/cause/question")
        t.note("Each candidate phrase (question/cause; title if title-only) is scored: "
               "rel = cosine(description, phrase) on stemmed TF vectors; mass = share of "
               "candidate score carrying it; dist = 1-|2*mass-1|; final = 0.5*rel+0.5*dist. "
               f"A phrase is kept only if final >= MIN_FIELD_SCORE ({config.MIN_FIELD_SCORE}); "
               "if none clears it, the agent is told to ask its own follow-up (freeform).")
        try:
            total = sum(float(c.get("score", c.get("cos", 0.0))) for c in candidates) or 1.0
            dvec = followup._tf_vector(description)
            t.log("")
            t.kv("description", repr(tk.short(description, 160)))
            t.kv("description tokens", sorted(dvec.keys()))
            t.log("")
            scored = []
            for c in candidates:
                sc = float(c.get("score", c.get("cos", 0.0)))
                for ph in followup._candidate_phrases(c):
                    rel = followup._cosine(dvec, followup._tf_vector(ph))
                    mass = sc / total
                    dist = 1.0 - abs(2.0 * mass - 1.0)
                    scored.append((rel * 0.5 + dist * 0.5, rel, mass, dist, c["kb_id"], ph))
            scored.sort(key=lambda x: -x[0])
            for final, rel, mass, dist, kb, ph in scored:
                keep = "KEEP" if final >= config.MIN_FIELD_SCORE else "drop"
                t.log(f"[{keep}] final={final:.3f} (rel={rel:.3f} mass={mass:.3f} dist={dist:.3f}) "
                      f"{kb}: \"{tk.short(ph, 80)}\"")
        except Exception as exc:
            t.log(f"(scoring preview unavailable: {type(exc).__name__}: {exc})")

    result = _orig_fields(description, candidates, top_k=top_k, min_score=min_score)

    if t is not None:
        t.log("")
        t.log(f"SELECTED ({len(result)}): {result}")
        if result:
            t.note("Returned as discriminating_symptoms -> the agent grounds its follow-up "
                   "question ONLY in these phrases.")
        else:
            t.note("Empty -> no phrase was relevant enough; the agent asks its OWN focused "
                   "Outlook follow-up.")
        t.data(selected=result)
    return result

main.select_discriminating_fields = _traced_fields


# ════════════════════════════════════════════════════════════════════
# 1 & 9. HTTP middleware — open/close trace, request, response, next
# ════════════════════════════════════════════════════════════════════
@app.middleware("http")
async def _trace_middleware(request, call_next):
    if request.url.path not in TRACED_PATHS:
        return await call_next(request)

    body_bytes = await request.body()
    tool_name = request.url.path.lstrip("/")
    t = tk.start(f"POST {request.url.path}")
    t.step(f"AGENT -> TOOL CALL RECEIVED: {tool_name}")
    key = request.headers.get("x-api-key", "")
    t.kv("x-api-key", f"present ({key[:4]}...)" if key else "none (REQUIRE_API_KEY may be false)")
    try:
        payload = json.loads(body_bytes.decode("utf-8") or "{}")
    except Exception:
        payload = {"_raw": body_bytes.decode("utf-8", "replace")}
    t.log("request body:")
    t.log(tk.pretty_json(payload), indent=2)
    t.data(request=payload)

    # Group all rounds of one conversation into a single turn file. Prefer an
    # explicit correlation header, else the session_id the agent passes (the
    # round-cap key), else fall back to description-overlap (in trace_kit).
    turn_header = os.environ.get("TRACE_TURN_HEADER", "").strip().lower()
    header_key = request.headers.get(turn_header) if turn_header else None
    sid = payload.get("session_id") if isinstance(payload, dict) else None
    t.meta["turn_key"] = header_key or sid or None
    if isinstance(payload, dict):
        t.meta["endpoint_kind"] = "candidates"
        t.meta["description"] = payload.get("description") or ""
        t.meta["session_id"] = sid or ""

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
        t.log(tk.pretty_json(resp_json if resp_json is not None
                             else resp_bytes.decode("utf-8", "replace")), indent=2)
        t.data(status=status, response=resp_json)
        if status == 200 and isinstance(resp_json, dict) and request.url.path == "/get_kb_candidates":
            _explain_next(t, resp_json)

        return Response(content=resp_bytes, status_code=status,
                        headers=dict(response.headers), media_type=response.media_type)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        tk.finish(t, status=status, error=error)


def _explain_next(t, resp: dict):
    t.step("WHAT HAPPENS NEXT (per classification-agent protocol)")
    followup_req = resp.get("followup_required")
    top = resp.get("top_score")
    kb_id = resp.get("kb_id")
    symptoms = resp.get("discriminating_symptoms") or []
    msg = resp.get("message")
    if followup_req is False:
        if kb_id:
            t.log(f"followup_required=FALSE with kb_id={kb_id} (top_score={top}).")
            t.log("-> Confident match (or best-at-cap). The agent PRESENTS this article; no follow-up.")
        else:
            t.log(f"followup_required=FALSE with kb_id=None (top_score={top}).")
            t.log("-> No confident match. The agent stops asking and closes / hands off.")
        if msg:
            t.log(f"   tool message: {msg}", indent=2)
        return
    if symptoms:
        t.log(f"followup_required=TRUE (top_score={top}) — ASK, grounded in these phrases:")
        for s in symptoms:
            t.log(f"   - {s}", indent=2)
    else:
        t.log(f"followup_required=TRUE (top_score={top}) — ASK FREEFORM (no relevant phrase).")
        t.log("-> Agent asks its own focused, Outlook-scoped follow-up, then calls again (same session_id).")
        if msg:
            t.log(f"   tool message: {msg}", indent=2)
    t.note(f"Round cap: after MAX_ROUNDS ({config.MAX_ROUNDS}) follow-ups per session_id the tool "
           f"force-concludes (present best if top_score >= MIN_DISPLAY_SCORE, else kb_id=null).")


# ════════════════════════════════════════════════════════════════════
# Browser viewer — /trace (index of turns), /trace?turn=..., /trace?day=...
#   Auth: TRACE_VIEW_KEY app setting, else the tool API key.
# ════════════════════════════════════════════════════════════════════
_BLOCK_SPLIT = re.compile(r"(?=^={20,}\nTRACE )", re.M)
_CSS = """
  body {background:#0d1117;color:#c9d1d9;font-family:Consolas,monospace;margin:16px;}
  a {color:#58a6ff;text-decoration:none;} a:hover{text-decoration:underline;}
  .top{position:sticky;top:0;background:#0d1117;padding:8px 0;border-bottom:1px solid #30363d;margin-bottom:8px;}
  .blk{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;margin:14px 0;white-space:pre-wrap;font-size:12.5px;line-height:1.45;}
  table{border-collapse:collapse;width:100%;font-size:13px;} td,th{border-bottom:1px solid #21262d;padding:7px 10px;text-align:left;vertical-align:top;}
  th{color:#8b949e;} tr:hover td{background:#161b22;} .desc{color:#e6edf3;} .meta{color:#8b949e;}
"""


def _check_view_key(key: str):
    expected = os.environ.get("TRACE_VIEW_KEY", "").strip() or config.TOOL_API_KEY
    if expected and key != expected:
        raise HTTPException(status_code=401, detail="missing or invalid ?key=")


def _trace_days() -> list:
    try:
        return sorted(f[len("trace_"):-len(".log")] for f in os.listdir(tk.LOG_DIR)
                      if f.startswith("trace_") and f.endswith(".log"))
    except FileNotFoundError:
        return []


def _read_day(day: str) -> str:
    with open(os.path.join(tk.LOG_DIR, f"trace_{day}.log"), encoding="utf-8") as f:
        return f.read()


def _turn_files() -> list:
    out = []
    try:
        names = os.listdir(tk.TURN_DIR)
    except FileNotFoundError:
        return out
    for name in names:
        if not (name.startswith("turn_") and name.endswith(".log")):
            continue
        path = os.path.join(tk.TURN_DIR, name)
        try:
            st = os.stat(path)
            desc, opened = "", ""
            with open(path, encoding="utf-8") as f:
                head = [next(f, "") for _ in range(4)]
                body = f.read()
            for ln in head:
                if ln.startswith("initial user description:"):
                    desc = ln.split(":", 1)[1].strip()
                elif ln.startswith("TURN "):
                    opened = ln.strip()
            calls = body.count("\nTRACE ") + (1 if body.startswith("TRACE ") else 0)
        except Exception:
            continue
        out.append({"name": name, "desc": desc, "opened": opened, "calls": calls, "mtime": st.st_mtime})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def _read_turn(name: str) -> str:
    if "/" in name or "\\" in name or not name.endswith(".log"):
        raise HTTPException(status_code=400, detail="bad turn name")
    path = os.path.join(tk.TURN_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="turn not found")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _page(title: str, top_html: str, body_html: str, refresh: int = 0) -> HTMLResponse:
    meta = f"<meta http-equiv='refresh' content='{int(refresh)}'>" if refresh > 0 else ""
    return HTMLResponse(f"<!doctype html><html><head><title>{title}</title>{meta}"
                        f"<style>{_CSS}</style></head><body>"
                        f"<div class='top'>{top_html}</div>{body_html}</body></html>")


@app.get("/trace/raw", response_class=PlainTextResponse)
def trace_raw(key: str = "", day: str = "", turn: str = ""):
    _check_view_key(key)
    if turn:
        return _read_turn(turn)
    days = _trace_days()
    if not days:
        return "No traces recorded yet."
    return _read_day(day if day in days else days[-1])


@app.get("/trace", response_class=HTMLResponse)
def trace_view(key: str = "", turn: str = "", day: str = "", refresh: int = 0, limit: int = 80):
    _check_view_key(key)
    k = html.escape(key)
    if turn:
        text = _read_turn(turn)
        blocks = [b for b in _BLOCK_SPLIT.split(text) if b.strip()]
        top = (f"<a href='/trace?key={k}'>&larr; all turns</a> &nbsp;|&nbsp; <b>{html.escape(turn)}</b> "
               f"&nbsp;|&nbsp; <a href='/trace?key={k}&turn={html.escape(turn)}&refresh=10'>auto-refresh 10s</a> "
               f"&nbsp;|&nbsp; <a href='/trace/raw?key={k}&turn={html.escape(turn)}'>raw</a>")
        body = "\n".join(f"<pre class='blk'>{html.escape(b.rstrip())}</pre>" for b in blocks)
        return _page(f"turn {turn}", top, body, refresh)
    if day:
        days = _trace_days()
        sel = day if day in days else (days[-1] if days else "")
        blocks = [b for b in _BLOCK_SPLIT.split(_read_day(sel)) if b.strip()] if sel else []
        total = len(blocks)
        blocks = list(reversed(blocks))[:max(1, limit)]
        top = (f"<a href='/trace?key={k}'>&larr; turns</a> &nbsp;|&nbsp; <b>day {sel}</b> — "
               f"{len(blocks)} of {total} calls &nbsp;|&nbsp; <a href='/trace/raw?key={k}&day={sel}'>raw</a>")
        body = "\n".join(f"<pre class='blk'>{html.escape(b.rstrip())}</pre>" for b in blocks)
        return _page(f"day {sel}", top, body, refresh)
    turns = _turn_files()
    days = _trace_days()
    daylinks = " · ".join(f"<a href='/trace?key={k}&day={d}'>{d}</a>" for d in days[-14:]) or "—"
    top = (f"<b>KB trace (pgvector) — turns</b> ({len(turns)}) &nbsp;|&nbsp; each row = one user "
           f"question and its tool call(s) &nbsp;|&nbsp; <a href='/trace?key={k}&refresh=10'>auto-refresh 10s</a> "
           f"&nbsp;|&nbsp; days: {daylinks}")
    if not turns:
        return _page("turns", top, "<p style='margin-top:20px'>No turns yet — call the tool once and reload.</p>", refresh)
    rows = ["<table><tr><th>when</th><th>initial user description</th><th>calls</th><th></th></tr>"]
    for tt in turns:
        rows.append(f"<tr><td class='meta'>{html.escape(tt['opened'].replace('TURN ','').split('|')[-1].strip())}</td>"
                    f"<td class='desc'><a href='/trace?key={k}&turn={html.escape(tt['name'])}'>"
                    f"{html.escape(tt['desc'] or tt['name'])}</a></td><td>{tt['calls']}</td>"
                    f"<td class='meta'><a href='/trace/raw?key={k}&turn={html.escape(tt['name'])}'>raw</a></td></tr>")
    rows.append("</table>")
    return _page("turns", top, "\n".join(rows), refresh)
