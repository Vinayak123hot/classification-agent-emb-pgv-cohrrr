-- inspect.sql — quick look at the pgvector KB database (kbtool).
--
-- HOW TO USE in VS Code:
--   1. Install the "PostgreSQL" extension by Microsoft (ms-ossdata.vscode-pgsql).
--   2. PostgreSQL panel (elephant icon) -> Add New Connection:
--        host   teva-kb-vectordb.postgres.database.azure.com
--        port   5432
--        db     kbtool
--        user   kbadmin
--        pass   (Key Vault vinny-kb-tool-vault1 / secret pg-vector-admin-password)
--        SSL    Require
--   3. Open this file, make sure the connection is selected (top-right), then run
--      the whole file or highlight one statement and press the "Run" button.

-- ── overview ──────────────────────────────────────────────────────────────
-- extensions (expect 'vector' = pgvector, and pg_trgm)
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'pg_trgm');

-- tables + row counts
SELECT 'kb_articles' AS table, count(*) FROM kb_articles
UNION ALL
SELECT 'kb_chunks', count(*) FROM kb_chunks;

-- ── the 8 KB articles (title / user-experience / cause; NO resolution stored) ─
SELECT kb_id, title, question, cause, environment, guidance_troubleshoot
FROM kb_articles
ORDER BY kb_id;

-- ── the 17 field-level chunks (one row per embeddable field) ─────────────────
-- embedding is a vector(1536); shown here as a short preview so the grid stays readable
SELECT kb_id,
       field_type,
       content,
       left(embedding::text, 40) || ' ...]' AS embedding_preview
FROM kb_chunks
ORDER BY kb_id, field_type;

-- ── the two search "legs" (confirm the indexes) ──────────────────────────────
-- kb_chunks_embedding_hnsw = HNSW cosine (semantic/vector search)
-- kb_chunks_tsv_gin        = GIN (keyword / full-text search)
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'kb_chunks';

-- ── pgvector in action: nearest chunks to a chosen article's title ───────────
-- change the WHERE kb_id / field_type to try other seeds; smaller distance = closer meaning
SELECT a.kb_id,
       a.field_type,
       a.content,
       round((a.embedding <=> seed.embedding)::numeric, 4) AS cosine_distance
FROM kb_chunks a,
     (SELECT embedding FROM kb_chunks
      WHERE kb_id = 'KB0015622' AND field_type = 'title') AS seed
ORDER BY a.embedding <=> seed.embedding
LIMIT 6;

-- ── keyword (full-text) leg on its own ───────────────────────────────────────
-- returns chunks whose text lexically matches the query terms
SELECT kb_id, field_type, content,
       round(ts_rank_cd(tsv, websearch_to_tsquery('english', 'calendar crash'))::numeric, 4) AS ts_rank
FROM kb_chunks
WHERE tsv @@ websearch_to_tsquery('english', 'calendar crash')
ORDER BY ts_rank DESC;


select * from kb_chunks