-- schema.sql — pgvector schema for the KB Candidates tool.
-- Two tables: doc-level metadata (kb_articles) + field-level chunks (kb_chunks).
-- A "chunk" here = one embeddable FIELD (title | question | cause), NOT a
-- sliding token window — the docs are short and the matchable signal is
-- field-shaped. Resolution text is never stored.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS kb_articles (
    kb_id        text PRIMARY KEY,
    title        text NOT NULL,
    question     text DEFAULT '',
    cause        text DEFAULT '',
    environment  text DEFAULT '',
    guidance_troubleshoot boolean,
    embed_text   text DEFAULT ''          -- title + question + cause (for reranking)
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id          bigserial PRIMARY KEY,
    kb_id       text NOT NULL REFERENCES kb_articles(kb_id) ON DELETE CASCADE,
    field_type  text NOT NULL,            -- 'title' | 'question' | 'cause'
    content     text NOT NULL,
    embedding   vector(1536) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Vector (semantic) leg of the hybrid search.
CREATE INDEX IF NOT EXISTS kb_chunks_embedding_hnsw
    ON kb_chunks USING hnsw (embedding vector_cosine_ops);

-- Keyword (lexical) leg of the hybrid search.
CREATE INDEX IF NOT EXISTS kb_chunks_tsv_gin
    ON kb_chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS kb_chunks_kb_id ON kb_chunks (kb_id);
