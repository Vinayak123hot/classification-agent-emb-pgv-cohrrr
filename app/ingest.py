"""
ingest.py — offline build job, as a class.

`Ingestor.ingest()`: parse docs -> field chunks -> embed -> upsert into pgvector
(kb_articles + kb_chunks). Idempotent (delete + insert per article), so editing a
.docx and re-running is safe; the live service reads the DB, so no restart needed.

    python app/ingest.py            # ingest all docs in DATA_DIR
    python app/ingest.py --recreate # drop & recreate the tables first
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from db import Database
from embeddings import EmbeddingClient
from kb_parser import KBParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("kb-tool.ingest")


class Ingestor:
    _SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

    def __init__(self, config, db: Database, embedder: EmbeddingClient, parser: KBParser):
        self._config = config
        self._db = db
        self._embedder = embedder
        self._parser = parser

    def ensure_schema(self, recreate: bool = False) -> None:
        """Create the pgvector extension + tables + indexes (optionally dropping first)."""
        conn = self._db.get_conn()
        with conn.cursor() as cur:
            if recreate:
                logger.info("Dropping existing tables (--recreate).")
                cur.execute("DROP TABLE IF EXISTS kb_chunks CASCADE;")
                cur.execute("DROP TABLE IF EXISTS kb_articles CASCADE;")
            with open(self._SCHEMA, encoding="utf-8") as f:
                cur.execute(f.read())
        logger.info("Schema ensured (pgvector + tables + indexes).")

    def ingest(self, recreate: bool = False) -> None:
        """Full build: parse -> field chunks -> embed -> upsert."""
        docs = self._parser.load_docs(self._config.DATA_DIR)
        logger.info("Parsed %d docs from %s", len(docs), self._config.DATA_DIR)
        self.ensure_schema(recreate)

        conn = self._db.get_conn()
        n_chunks = 0
        for d in docs:
            fields = d.fields()                            # [(field_type, content), ...]
            if not fields:
                logger.warning("%s has no embeddable fields — skipped.", d.kb_id)
                continue
            vectors = self._embedder.embed_texts([content for _, content in fields])  # batch embed

            with conn.cursor() as cur:
                # upsert the article row (text + combined embed_text for reranking)
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
                cur.execute("DELETE FROM kb_chunks WHERE kb_id=%s;", (d.kb_id,))  # rebuild chunks
                for (field_type, content), vec in zip(fields, vectors):
                    cur.execute(
                        """INSERT INTO kb_chunks (kb_id, field_type, content, embedding)
                           VALUES (%s,%s,%s,%s::vector);""",
                        (d.kb_id, field_type, content, self._embedder.to_pgvector(vec)),
                    )
                    n_chunks += 1
            logger.info("  %s -> %d chunk(s): %s", d.kb_id, len(fields), [ft for ft, _ in fields])

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM kb_articles;")
            n_articles = cur.fetchone()[0]
        logger.info("DONE: %d articles, %d chunks ingested.", n_articles, n_chunks)


def build_ingestor(config=CONFIG) -> Ingestor:
    """Compose an Ingestor with the default components."""
    return Ingestor(config, Database(config), EmbeddingClient(config), KBParser())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true",
                    help="drop and recreate the tables before ingesting")
    args = ap.parse_args()
    build_ingestor().ingest(recreate=args.recreate)
