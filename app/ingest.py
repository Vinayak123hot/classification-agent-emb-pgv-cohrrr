"""
ingest.py — offline build job: parse docs -> field chunks -> embed -> upsert.

Idempotent: re-running rebuilds every article's chunks from scratch (delete +
insert), so editing a .docx and re-running is safe. Run it whenever the KB
content changes — the live service reads the DB, so no app restart is needed.

    python app/ingest.py            # ingest all docs in DATA_DIR
    python app/ingest.py --recreate # drop & recreate the tables first

Phases: 1 parse (kb_parser) · 2 field chunks · 3 embed (text-embedding-3-small)
· 4 upsert into pgvector (kb_articles + kb_chunks).
"""
from __future__ import annotations

import argparse
import logging
import os

import config
import db
import embeddings
from kb_parser import load_docs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("kb-tool.ingest")

_SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


def ensure_schema(recreate: bool = False) -> None:
    conn = db.get_conn()
    with conn.cursor() as cur:
        if recreate:
            logger.info("Dropping existing tables (--recreate).")
            cur.execute("DROP TABLE IF EXISTS kb_chunks CASCADE;")
            cur.execute("DROP TABLE IF EXISTS kb_articles CASCADE;")
        with open(_SCHEMA, encoding="utf-8") as f:
            cur.execute(f.read())
    logger.info("Schema ensured (pgvector + tables + indexes).")


def ingest(recreate: bool = False) -> None:
    docs = load_docs(config.DATA_DIR)
    logger.info("Parsed %d docs from %s", len(docs), config.DATA_DIR)
    ensure_schema(recreate)

    conn = db.get_conn()
    n_chunks = 0
    for d in docs:
        fields = d.fields()                      # [(field_type, content), ...]
        if not fields:
            logger.warning("%s has no embeddable fields — skipped.", d.kb_id)
            continue
        # Phase 3: embed every field of this doc in one batch.
        vectors = embeddings.embed_texts([content for _, content in fields])

        with conn.cursor() as cur:
            # Phase 4: upsert article, then rebuild its chunks.
            cur.execute(
                """INSERT INTO kb_articles
                       (kb_id, title, question, cause, environment,
                        guidance_troubleshoot, embed_text)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (kb_id) DO UPDATE SET
                       title=EXCLUDED.title, question=EXCLUDED.question,
                       cause=EXCLUDED.cause, environment=EXCLUDED.environment,
                       guidance_troubleshoot=EXCLUDED.guidance_troubleshoot,
                       embed_text=EXCLUDED.embed_text;""",
                (d.kb_id, d.title, d.question, d.cause, d.environment,
                 d.guidance_troubleshoot, d.embed_text()),
            )
            cur.execute("DELETE FROM kb_chunks WHERE kb_id=%s;", (d.kb_id,))
            for (field_type, content), vec in zip(fields, vectors):
                cur.execute(
                    """INSERT INTO kb_chunks (kb_id, field_type, content, embedding)
                       VALUES (%s,%s,%s,%s::vector);""",
                    (d.kb_id, field_type, content, embeddings.to_pgvector(vec)),
                )
                n_chunks += 1
        logger.info("  %s -> %d chunk(s): %s", d.kb_id, len(fields),
                    [ft for ft, _ in fields])

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM kb_articles;")
        n_articles = cur.fetchone()[0]
    logger.info("DONE: %d articles, %d chunks ingested.", n_articles, n_chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true",
                    help="drop and recreate the tables before ingesting")
    args = ap.parse_args()
    ingest(recreate=args.recreate)
