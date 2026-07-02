"""
trace_kit.py — human-readable, end-to-end tracing core (pgvector edition).

Every HTTP request to the tool becomes ONE ordered trace block (all its pipeline
steps grouped together, never interleaved) written under TRACE_LOG_DIR:

    trace_YYYY-MM-DD.log     human-readable, step-by-step execution story
    trace_YYYY-MM-DD.jsonl   the same events as one JSON object per request
    turns/turn_*.log         one file per conversation turn (candidates call +
                             its follow-up rounds), named after the user's
                             initial description — exactly like teva-kb-trace.

Consumed only by main_traced.py; main.py is never modified.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime

_DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
LOG_DIR = os.environ.get("TRACE_LOG_DIR", _DEFAULT_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
TURN_DIR = os.path.join(LOG_DIR, "turns")
os.makedirs(TURN_DIR, exist_ok=True)

# A tool call continues an open turn when it arrives within this many seconds of
# that turn's last activity and overlaps it by description tokens.
TURN_IDLE_SECONDS = int(os.environ.get("TRACE_TURN_IDLE_SECONDS", "900"))

_write_lock = threading.Lock()
_current: ContextVar = ContextVar("kb_trace_current", default=None)
_sessions: list = []

RULE_HEAVY = "=" * 100
RULE_LIGHT = "-" * 100


class Trace:
    """Collects all steps of one request, flushed as a single block at the end."""

    def __init__(self, endpoint: str):
        self.id = uuid.uuid4().hex[:8]
        self.endpoint = endpoint
        self.started = datetime.now()
        self.step_no = 0
        self.lines: list[str] = []
        self.events: list[dict] = []
        self.meta: dict = {}

    def step(self, title: str, **data):
        self.step_no += 1
        self.lines.append("")
        self.lines.append(f"STEP {self.step_no} -- {title}")
        self.lines.append(RULE_LIGHT)
        self.events.append({"step": self.step_no, "title": title, **data})

    def log(self, text: str = "", indent: int = 1):
        pad = "    " * indent
        for ln in str(text).splitlines() or [""]:
            self.lines.append(pad + ln)

    def kv(self, key: str, value, indent: int = 1):
        self.log(f"{key:<40}: {value}", indent=indent)

    def note(self, text: str, indent: int = 1):
        self.log(f"i {text}", indent=indent)

    def data(self, **kwargs):
        if self.events:
            self.events[-1].update(kwargs)
        else:
            self.events.append(kwargs)


def start(endpoint: str) -> Trace:
    t = Trace(endpoint)
    t._token = _current.set(t)
    return t


def current() -> "Trace | None":
    return _current.get()


def finish(t: Trace, status: int = 200, error: "str | None" = None):
    duration_ms = (datetime.now() - t.started).total_seconds() * 1000.0
    ts = t.started.strftime("%Y-%m-%d %H:%M:%S")
    status_txt = f"{status}" + ("  x ERROR" if (error or status >= 400) else "  OK")
    header = [RULE_HEAVY,
              f"TRACE {t.id} | {ts} | {t.endpoint} | status={status_txt} | {duration_ms:.0f} ms",
              RULE_HEAVY]
    footer = ["", f"    x UNHANDLED ERROR: {error}", ""] if error else [""]
    block = "\n".join(header + t.lines + footer) + "\n"

    day = t.started.strftime("%Y-%m-%d")
    log_path = os.path.join(LOG_DIR, f"trace_{day}.log")
    jsonl_path = os.path.join(LOG_DIR, f"trace_{day}.jsonl")
    record = {"trace_id": t.id, "time": t.started.isoformat(), "endpoint": t.endpoint,
              "status": status, "duration_ms": round(duration_ms, 1), "error": error,
              "events": t.events}

    with _write_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(block)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        try:
            _route_to_turn(t, block)
        except Exception as exc:
            try:
                with open(os.path.join(LOG_DIR, "turn_routing_errors.log"), "a",
                          encoding="utf-8") as f:
                    f.write(f"{ts} {type(exc).__name__}: {exc}\n")
            except Exception:
                pass
    try:
        _current.reset(t._token)
    except Exception:
        _current.set(None)


# ── Per-turn grouping (best-effort; Foundry OpenAPI tools pass no thread id) ────
def _slug(text: str, max_words: int = 9, max_len: int = 64) -> str:
    words = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    s = "-".join(words[:max_words]) or "no-description"
    return s[:max_len].rstrip("-")


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _new_turn(desc: str, now: datetime, turn_key=None) -> dict:
    sid = uuid.uuid4().hex[:6]
    fname = f"turn_{now.strftime('%Y-%m-%d_%H%M%S')}_{sid}_{_slug(desc)}.log"
    s = {"id": sid, "turn_key": turn_key, "tokens": _tokens(desc),
         "file": os.path.join(TURN_DIR, fname), "name": fname,
         "last_activity": now, "calls": 0}
    with open(s["file"], "a", encoding="utf-8") as f:
        f.write(RULE_HEAVY + "\n")
        f.write(f"TURN {sid} | opened {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"initial user description: {short(desc, 240)}\n")
        f.write(RULE_HEAVY + "\n")
    _sessions.append(s)
    return s


def _route_to_turn(t: "Trace", block: str):
    """Append the rendered block to the right per-turn file. Caller holds the lock."""
    now = t.started
    turn_key = t.meta.get("turn_key")
    global _sessions
    _sessions = [s for s in _sessions
                 if (now - s["last_activity"]).total_seconds() <= TURN_IDLE_SECONDS]

    desc = t.meta.get("description", "") or t.endpoint
    session = None
    if turn_key:
        # Strict grouping by the correlation key (session_id): all rounds of one
        # conversation land in the same turn file; no token-overlap guessing.
        session = next((s for s in _sessions if s.get("turn_key") == turn_key), None)
        if session is None:
            session = _new_turn(desc, now, turn_key)
    else:
        toks = _tokens(desc)
        best, best_score = None, 0.0
        for s in _sessions:
            sc = _jaccard(toks, s["tokens"])
            if sc > best_score:
                best, best_score = s, sc
        if best is not None and best_score >= 0.5:
            session = best
            session["tokens"] |= toks
        else:
            session = _new_turn(desc, now, turn_key)

    session["last_activity"] = now
    session["calls"] += 1
    t.meta["turn_file"] = session["name"]
    with open(session["file"], "a", encoding="utf-8") as f:
        f.write(block)


# ── small helpers used by main_traced.py ───────────────────────────────────
def short(text, limit: int = 160) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "..."


def pretty_json(obj, limit: int = 6000) -> str:
    try:
        s = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if len(s) > limit:
        s = s[:limit] + f"\n... (truncated, {len(s)} chars total)"
    return s
