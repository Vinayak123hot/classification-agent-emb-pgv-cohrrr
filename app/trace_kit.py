"""
trace_kit.py — human-readable, end-to-end tracing engine, as classes.

`Trace`  collects the ordered steps of ONE request.
`Tracer` owns the output location, the current-trace context, and per-turn
         grouping. Every request becomes one block written to (under TRACE_LOG_DIR,
         default /home/LogFiles/kbtrace on Azure):
             trace_YYYY-MM-DD.log     human-readable
             trace_YYYY-MM-DD.jsonl   machine-readable
             turns/turn_*.log         one file per conversation (by session_id)
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime

RULE_HEAVY = "=" * 100
RULE_LIGHT = "-" * 100


def short(text, limit: int = 160) -> str:
    """One-line preview of arbitrary text."""
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "..."


def pretty_json(obj, limit: int = 6000) -> str:
    """Indented JSON, truncated for the log."""
    try:
        s = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + f"\n... (truncated, {len(s)} chars total)"


class Trace:
    """Collects all steps of a single request, flushed as one block at the end."""

    def __init__(self, endpoint: str):
        self.id = uuid.uuid4().hex[:8]
        self.endpoint = endpoint
        self.started = datetime.now()
        self.step_no = 0
        self.lines: list[str] = []      # human-readable body
        self.events: list[dict] = []    # structured mirror for the .jsonl
        self.meta: dict = {}            # scratchpad shared between steps

    def step(self, title: str, **data):
        """Open a new numbered step."""
        self.step_no += 1
        self.lines += ["", f"STEP {self.step_no} -- {title}", RULE_LIGHT]
        self.events.append({"step": self.step_no, "title": title, **data})

    def log(self, text: str = "", indent: int = 1):
        """Add free-form line(s) under the current step."""
        pad = "    " * indent
        for ln in str(text).splitlines() or [""]:
            self.lines.append(pad + ln)

    def kv(self, key: str, value, indent: int = 1):
        """Add an aligned 'key : value' line."""
        self.log(f"{key:<40}: {value}", indent=indent)

    def note(self, text: str, indent: int = 1):
        """Add an explanatory note."""
        self.log(f"i {text}", indent=indent)

    def data(self, **kwargs):
        """Attach structured data to the most recent step (jsonl only)."""
        if self.events:
            self.events[-1].update(kwargs)
        else:
            self.events.append(kwargs)


class Tracer:
    """Writes traces and groups them into per-conversation turn files."""

    def __init__(self, log_dir: "str | None" = None):
        self.LOG_DIR = log_dir or os.environ.get(
            "TRACE_LOG_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"))
        self.TURN_DIR = os.path.join(self.LOG_DIR, "turns")
        os.makedirs(self.TURN_DIR, exist_ok=True)          # also creates LOG_DIR
        # a new call continues an open turn if within this idle window
        self.TURN_IDLE_SECONDS = int(os.environ.get("TRACE_TURN_IDLE_SECONDS", "900"))
        self._lock = threading.Lock()
        self._sessions: list[dict] = []                    # open turns
        self._current: ContextVar = ContextVar(f"kb_trace_{id(self)}", default=None)

    # ── current-trace handling (survives the threadpool hop) ──────────────────
    def start(self, endpoint: str) -> Trace:
        t = Trace(endpoint)
        t._token = self._current.set(t)
        return t

    def current(self) -> "Trace | None":
        return self._current.get()

    def finish(self, t: Trace, status: int = 200, error: "str | None" = None):
        """Render the whole block and append it to today's logs + its turn file."""
        duration_ms = (datetime.now() - t.started).total_seconds() * 1000.0
        ts = t.started.strftime("%Y-%m-%d %H:%M:%S")
        status_txt = f"{status}" + ("  x ERROR" if (error or status >= 400) else "  OK")
        header = [RULE_HEAVY,
                  f"TRACE {t.id} | {ts} | {t.endpoint} | status={status_txt} | {duration_ms:.0f} ms",
                  RULE_HEAVY]
        footer = ["", f"    x UNHANDLED ERROR: {error}", ""] if error else [""]
        block = "\n".join(header + t.lines + footer) + "\n"

        day = t.started.strftime("%Y-%m-%d")
        record = {"trace_id": t.id, "time": t.started.isoformat(), "endpoint": t.endpoint,
                  "status": status, "duration_ms": round(duration_ms, 1), "error": error,
                  "events": t.events}
        with self._lock:
            with open(os.path.join(self.LOG_DIR, f"trace_{day}.log"), "a", encoding="utf-8") as f:
                f.write(block)
            with open(os.path.join(self.LOG_DIR, f"trace_{day}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            try:
                self._route_to_turn(t, block)
            except Exception as exc:                        # tracing must never break the request
                try:
                    with open(os.path.join(self.LOG_DIR, "turn_routing_errors.log"), "a",
                              encoding="utf-8") as f:
                        f.write(f"{ts} {type(exc).__name__}: {exc}\n")
                except Exception:
                    pass
        try:
            self._current.reset(t._token)
        except Exception:
            self._current.set(None)

    # ── per-turn grouping ─────────────────────────────────────────────────────
    @staticmethod
    def _slug(text: str, max_words: int = 9, max_len: int = 64) -> str:
        words = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        return ("-".join(words[:max_words]) or "no-description")[:max_len].rstrip("-")

    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        return len(a & b) / len(a | b) if a and b else 0.0

    def _new_turn(self, desc: str, now: datetime, turn_key=None) -> dict:
        sid = uuid.uuid4().hex[:6]
        fname = f"turn_{now.strftime('%Y-%m-%d_%H%M%S')}_{sid}_{self._slug(desc)}.log"
        s = {"id": sid, "turn_key": turn_key, "tokens": self._tokens(desc),
             "file": os.path.join(self.TURN_DIR, fname), "name": fname,
             "last_activity": now, "calls": 0}
        with open(s["file"], "a", encoding="utf-8") as f:
            f.write(RULE_HEAVY + "\n")
            f.write(f"TURN {sid} | opened {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"initial user description: {short(desc, 240)}\n")
            f.write(RULE_HEAVY + "\n")
        self._sessions.append(s)
        return s

    def _route_to_turn(self, t: Trace, block: str):
        """Append the block to the right per-turn file. Caller holds the lock."""
        now = t.started
        turn_key = t.meta.get("turn_key")
        # drop idle turns
        self._sessions = [s for s in self._sessions
                          if (now - s["last_activity"]).total_seconds() <= self.TURN_IDLE_SECONDS]
        desc = t.meta.get("description", "") or t.endpoint
        session = None
        if turn_key:                                        # strict grouping by session_id
            session = next((s for s in self._sessions if s.get("turn_key") == turn_key), None)
            if session is None:
                session = self._new_turn(desc, now, turn_key)
        else:                                               # fallback: description overlap
            toks = self._tokens(desc)
            best, best_score = None, 0.0
            for s in self._sessions:
                sc = self._jaccard(toks, s["tokens"])
                if sc > best_score:
                    best, best_score = s, sc
            if best is not None and best_score >= 0.5:
                session = best
                session["tokens"] |= toks
            else:
                session = self._new_turn(desc, now, turn_key)
        session["last_activity"] = now
        session["calls"] += 1
        t.meta["turn_file"] = session["name"]
        with open(session["file"], "a", encoding="utf-8") as f:
            f.write(block)

    # ── viewer helpers (read the files back) ──────────────────────────────────
    def trace_days(self) -> list[str]:
        try:
            return sorted(f[len("trace_"):-len(".log")] for f in os.listdir(self.LOG_DIR)
                          if f.startswith("trace_") and f.endswith(".log"))
        except FileNotFoundError:
            return []

    def read_day(self, day: str) -> str:
        with open(os.path.join(self.LOG_DIR, f"trace_{day}.log"), encoding="utf-8") as f:
            return f.read()

    def turn_files(self) -> list[dict]:
        out = []
        try:
            names = os.listdir(self.TURN_DIR)
        except FileNotFoundError:
            return out
        for name in names:
            if not (name.startswith("turn_") and name.endswith(".log")):
                continue
            path = os.path.join(self.TURN_DIR, name)
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
            out.append({"name": name, "desc": desc, "opened": opened,
                        "calls": calls, "mtime": st.st_mtime})
        out.sort(key=lambda r: r["mtime"], reverse=True)
        return out

    def read_turn(self, name: str) -> str:
        if "/" in name or "\\" in name or not name.endswith(".log"):
            raise ValueError("bad turn name")
        path = os.path.join(self.TURN_DIR, name)
        if not os.path.isfile(path):
            raise FileNotFoundError("turn not found")
        with open(path, encoding="utf-8") as f:
            return f.read()
