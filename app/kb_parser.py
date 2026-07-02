"""
kb_parser.py — turn the KB .docx files into embeddable field records.

What a user actually describes when they ask for help is the **heading**, the
**user experience / question**, and (sometimes) the **cause** — never the fix.
So we keep ONLY those three fields and DROP the Resolution / Option steps,
Answers, Prerequisites, Estimated time, and boilerplate. This matches the real
production docs, which have NO uniform "Symptoms" section — the matchable signal
lives in title + user-experience/questions (+ cause when present).

Heterogeneity handled (verified against the 8 sample docs):
  - the user-facing field may be labelled 'User Experience' OR 'Questions'
    OR 'Question' OR 'Description'  -> unified into `question`
  - 'Cause' is optional
  - some docs are title-only (e.g. KB0013608) -> question/cause empty
  - a 'Questions' block can bleed into 'Answers:/Estimated time/Prerequisite'
    -> those are treated as stop headers

Pure standard library: a .docx is a zip of XML, so we read word/document.xml
directly — no python-docx / Office dependency.
"""
from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass

# Section headers we KEEP, mapped to the canonical field name.
_QUESTION_HEADERS = (
    "user experience / symptoms", "questions / symptoms",
    "user experience", "questions", "question", "description", "symptoms",
)
_CAUSE_HEADERS = ("cause",)
# Anything from here on is dropped (fix steps, answers, metadata, boilerplate).
_STOP_HEADERS = (
    "resolution", "steps", "option", "note", "workaround", "solution",
    "answers", "answer", "estimated time", "prerequisite", "prerequisites",
)
_META_HEADERS = ("environment", "kb id", "guidance troubleshoot",
                 "priority", "category", "impact")

_ALL_HEADERS = _QUESTION_HEADERS + _CAUSE_HEADERS + _STOP_HEADERS + _META_HEADERS


@dataclass
class KBDoc:
    kb_id: str
    title: str          # cleaned heading (no leading "(GLOBAL)" tag)
    raw_title: str       # original first line
    question: str        # user-experience / questions text (may be "")
    cause: str           # cause text (may be "")
    environment: str
    guidance_troubleshoot: bool | None
    source_file: str

    def fields(self) -> list[tuple[str, str]]:
        """The non-empty embeddable fields as (field_type, content) — one per
        chunk row. Order = title, question, cause. Empty fields are skipped, so a
        title-only doc yields exactly one chunk."""
        out: list[tuple[str, str]] = []
        if self.title:
            out.append(("title", self.title))
        if self.question:
            out.append(("question", self.question))
        if self.cause:
            out.append(("cause", self.cause))
        return out

    def embed_text(self) -> str:
        """The article-level text used for reranking (title + question + cause)."""
        return ". ".join(p for p in (self.title, self.question, self.cause) if p)


def _docx_paragraphs(path: str) -> list[str]:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    paras: list[str] = []
    for chunk in re.split(r"</w:p>", xml):
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", chunk, re.S)
        line = "".join(texts)
        line = (line.replace("&amp;", "&").replace("&lt;", "<")
                    .replace("&gt;", ">").replace("&quot;", '"')
                    .replace("&apos;", "'").replace("�", "-")).strip()
        if line:
            paras.append(line)
    return paras


def _match_header(line: str) -> str | None:
    """Return the matched header key if `line` starts a known section, else None."""
    low = line.lower().strip()
    bare = low.rstrip(":").strip()
    for h in _ALL_HEADERS:
        if bare == h or low.startswith(h + ":") or low.startswith(h + " :"):
            return h
        # header on its own line with no colon (e.g. "Cause", "User Experience")
        if bare == h:
            return h
    # also catch a header word that begins the line followed by non-alpha
    for h in _ALL_HEADERS:
        if low.startswith(h) and (len(low) == len(h) or not low[len(h)].isalnum()):
            return h
    return None


def _field_of(header: str) -> str:
    if header in _QUESTION_HEADERS:
        return "question"
    if header in _CAUSE_HEADERS:
        return "cause"
    return "_drop"   # stop/meta


def parse_doc(path: str) -> KBDoc:
    paras = _docx_paragraphs(path)
    full = "\n".join(paras)

    # kb_id: prefer an explicit "KB ID: KBxxxx", else the filename.
    m = re.search(r"KB\s*ID\s*:?\s*(KB\d{4,})", full, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(KB\d{5,})\b", full)
    if m:
        kb_id = m.group(1).upper()
    else:
        fm = re.search(r"(KB\d{5,})", os.path.basename(path), re.IGNORECASE)
        kb_id = (fm.group(1).upper() if fm
                 else os.path.splitext(os.path.basename(path))[0].upper())

    raw_title = paras[0] if paras else ""
    title = re.sub(r"^\(.*?\)\s*", "", raw_title).strip()   # drop "(GLOBAL)" tag

    env = ""
    me = re.search(r"Environment\s*:?\s*(.+)", full)
    if me:
        env = me.group(1).strip()

    gt = None
    mg = re.search(r"Guidance\s*Troubleshoot\s*:?\s*(true|false)", full, re.IGNORECASE)
    if mg:
        gt = mg.group(1).lower() == "true"

    # Walk the body; collect lines into the current KEEP section, stopping the
    # moment a stop/meta header appears.
    buckets: dict[str, list[str]] = {"question": [], "cause": []}
    current: str | None = None
    for line in paras[1:]:
        h = _match_header(line)
        if h is not None:
            field = _field_of(h)
            if field in ("question", "cause"):
                current = field
                inline = line.split(":", 1)[1].strip() if ":" in line else ""
                if inline:
                    buckets[field].append(inline)
            else:               # stop/meta header → leave KEEP mode
                current = None
            continue
        if current in ("question", "cause"):
            buckets[current].append(line)

    def _clean(parts: list[str]) -> str:
        seen, kept = set(), []
        for p in parts:
            p = p.strip(" -•\t")
            if p and p.lower() not in seen and len(p) > 3:
                seen.add(p.lower())
                kept.append(p)
        return " ".join(kept).strip()

    return KBDoc(
        kb_id=kb_id,
        title=title,
        raw_title=raw_title,
        question=_clean(buckets["question"]),
        cause=_clean(buckets["cause"]),
        environment=env,
        guidance_troubleshoot=gt,
        source_file=os.path.basename(path),
    )


def load_docs(data_dir: str) -> list[KBDoc]:
    docs: list[KBDoc] = []
    for fn in sorted(os.listdir(data_dir)):
        if fn.lower().endswith(".docx") and not fn.startswith("~"):
            docs.append(parse_doc(os.path.join(data_dir, fn)))
    return docs


if __name__ == "__main__":   # quick manual check
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(os.path.dirname(here), "data")
    for d in load_docs(data):
        print(f"{d.kb_id}  gt={d.guidance_troubleshoot}  env={d.environment!r}")
        print(f"   title   : {d.title}")
        print(f"   question: {d.question}")
        print(f"   cause   : {d.cause}")
        print(f"   chunks  : {[ft for ft, _ in d.fields()]}")
        print()
