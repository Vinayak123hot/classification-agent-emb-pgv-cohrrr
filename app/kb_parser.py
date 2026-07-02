"""
kb_parser.py — parse the KB .docx files into embeddable field records.

`KBDoc` is the value object (one parsed document); `KBParser` does the parsing.
We keep ONLY the parts a user describes when asking for help — heading (title),
user-experience/question, and (when present) cause — and DROP the resolution
steps / answers / boilerplate. Pure standard library (a .docx is a zip of XML).
"""
from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass


@dataclass
class KBDoc:
    """One parsed KB article (value object)."""
    kb_id: str
    title: str            # cleaned heading (no leading "(GLOBAL)" tag)
    raw_title: str        # original first line
    question: str         # user-experience / questions text (may be "")
    cause: str            # cause text (may be "")
    environment: str
    guidance_troubleshoot: "bool | None"
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


class KBParser:
    """Reads .docx files and produces KBDoc objects."""

    # Section headers we KEEP, mapped to the canonical field name.
    QUESTION_HEADERS = (
        "user experience / symptoms", "questions / symptoms",
        "user experience", "questions", "question", "description", "symptoms",
    )
    CAUSE_HEADERS = ("cause",)
    # Anything from here on is dropped (fix steps, answers, metadata, boilerplate).
    STOP_HEADERS = (
        "resolution", "steps", "option", "note", "workaround", "solution",
        "answers", "answer", "estimated time", "prerequisite", "prerequisites",
    )
    META_HEADERS = ("environment", "kb id", "guidance troubleshoot",
                    "priority", "category", "impact")

    def __init__(self):
        # one flat tuple of every header we recognize (used by _match_header)
        self._all_headers = (self.QUESTION_HEADERS + self.CAUSE_HEADERS
                             + self.STOP_HEADERS + self.META_HEADERS)

    # ── low-level docx reading ────────────────────────────────────────────────
    def _docx_paragraphs(self, path: str) -> list[str]:
        """Return non-empty paragraph strings from a .docx using only stdlib."""
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
        paras: list[str] = []
        for chunk in re.split(r"</w:p>", xml):                 # split on paragraph end
            texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", chunk, re.S)  # pull text runs
            line = "".join(texts)
            line = (line.replace("&amp;", "&").replace("&lt;", "<")   # unescape entities
                        .replace("&gt;", ">").replace("&quot;", '"')
                        .replace("&apos;", "'").replace("�", "-")).strip()
            if line:
                paras.append(line)
        return paras

    # ── header detection ──────────────────────────────────────────────────────
    def _match_header(self, line: str) -> "str | None":
        """Return the matched header key if `line` starts a known section, else None."""
        low = line.lower().strip()
        bare = low.rstrip(":").strip()
        for h in self._all_headers:
            if bare == h or low.startswith(h + ":") or low.startswith(h + " :"):
                return h
            if bare == h:                                       # header on its own line
                return h
        for h in self._all_headers:                             # header word then non-alpha
            if low.startswith(h) and (len(low) == len(h) or not low[len(h)].isalnum()):
                return h
        return None

    def _field_of(self, header: str) -> str:
        """Map a matched header to a bucket name."""
        if header in self.QUESTION_HEADERS:
            return "question"
        if header in self.CAUSE_HEADERS:
            return "cause"
        return "_drop"                                          # stop/meta -> not kept

    @staticmethod
    def _clean(parts: list[str]) -> str:
        """De-dup + trim collected lines into one clean string."""
        seen, kept = set(), []
        for p in parts:
            p = p.strip(" -•\t")
            if p and p.lower() not in seen and len(p) > 3:
                seen.add(p.lower())
                kept.append(p)
        return " ".join(kept).strip()

    # ── public API ────────────────────────────────────────────────────────────
    def parse_doc(self, path: str) -> KBDoc:
        """Parse a single .docx into a KBDoc."""
        paras = self._docx_paragraphs(path)
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
        current: "str | None" = None
        for line in paras[1:]:
            h = self._match_header(line)
            if h is not None:                                   # this line is a header
                field = self._field_of(h)
                if field in ("question", "cause"):              # a KEEP section
                    current = field
                    inline = line.split(":", 1)[1].strip() if ":" in line else ""
                    if inline:
                        buckets[field].append(inline)
                else:                                           # stop/meta -> leave KEEP mode
                    current = None
                continue
            if current in ("question", "cause"):                # body line of a KEEP section
                buckets[current].append(line)

        return KBDoc(
            kb_id=kb_id, title=title, raw_title=raw_title,
            question=self._clean(buckets["question"]),
            cause=self._clean(buckets["cause"]),
            environment=env, guidance_troubleshoot=gt,
            source_file=os.path.basename(path),
        )

    def load_docs(self, data_dir: str) -> list[KBDoc]:
        """Parse every .docx in a directory (skips Word lock files)."""
        docs: list[KBDoc] = []
        for fn in sorted(os.listdir(data_dir)):
            if fn.lower().endswith(".docx") and not fn.startswith("~"):
                docs.append(self.parse_doc(os.path.join(data_dir, fn)))
        return docs


if __name__ == "__main__":                                      # quick manual check
    from config import CONFIG
    for d in KBParser().load_docs(CONFIG.DATA_DIR):
        print(f"{d.kb_id}  title={d.title!r} chunks={[ft for ft, _ in d.fields()]}")
