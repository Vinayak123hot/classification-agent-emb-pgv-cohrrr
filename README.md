# KB Candidates Tool — pgvector + embeddings + Cohere rerank (Azure-native)

An Azure-native replacement for the Azure AI Search retrieval service. It answers
the **exact same `get_kb_candidates` contract** as the AI-Search `main.py`, and
keeps that service's **routing engine** (spread gate → resolve vs. follow-up,
per-session round cap) — only the retrieval underneath is swapped:

| | AI Search (`vinny-ai-search`) | This service |
|---|---|---|
| Vectors | AI Search vector index | **pgvector** (Postgres, HNSW cosine) |
| Keyword | AI Search BM25 | **Postgres full-text** (`tsvector`) |
| Fusion | built-in hybrid | **RRF** (in `retrieval.py`) |
| Rerank | semantic reranker (0–4) | **Cohere Rerank v4** (0–1) · cosine fallback |
| Embeddings | integrated vectorizer | **Azure OpenAI `text-embedding-3-small`** |
| Score scale | 0–4 | **0–1** |
| Gold-set ranking | top-1 84.9% / recall@3 98.6% | **top-1 95.9% / recall@3 100% / MRR 0.975** |

> All components live in Azure: Postgres Flexible Server (`teva-kb-vectordb`,
> Central India), embeddings + Cohere on the `Teva` AI Services account, secrets
> in Key Vault `vinny-kb-tool-vault1`.

---

## 1. What gets embedded (and what does NOT)

Production docs have **no uniform "Symptoms" section** — the matchable signal is
the **heading**, the **user-experience / question**, and (when present) the
**cause**. Resolution/Option steps are the *fix*, which a user never describes,
so they are **dropped**. `kb_parser.py` extracts exactly:

```
title      (heading, "(GLOBAL)" tag stripped)   — always present
question   (User Experience / Questions / …)    — often present
cause      (Cause)                              — sometimes present
```

Docs are heterogeneous and handled gracefully: KB0013608 is **title-only**;
KB0010863's Questions block stops before "Answers/Estimated time".

## 2. Chunking = field decomposition

The docs are short, so a "chunk" is one **field** (`title` | `question` |
`cause`), not a sliding token window. Each non-empty field becomes one row in
`kb_chunks` (title-only docs → one row). A query can then match the strongest
single field; results are de-duplicated back to distinct articles at retrieval.

## 3. Data flow

```
data/*.docx ─▶ kb_parser ─▶ (title/question/cause)  ─▶ embed (3-small) ─▶ pgvector
                             drop resolution                                (ingest.py)

query ─▶ embed ─▶ pgvector cosine KNN  ┐
      └▶ Postgres full-text (tsv)      ┘─▶ RRF fuse ─▶ dedupe to articles
      ─▶ Cohere rerank v4 (0–1; cosine fallback)
      ─▶ compute_spread + score bands + round cap
      ─▶ {followup_required, kb_id, top_score, discriminating_symptoms, message}
```

## 4. Files

| File | Role |
|---|---|
| `app/kb_parser.py` | docx → title/question/cause records (stdlib; drops resolution). |
| `app/config.py` | central config; secrets from Key Vault (env override). |
| `app/db.py` | psycopg2 connection to `kbtool`. |
| `app/embeddings.py` | Azure OpenAI `text-embedding-3-small`, L2-normalized. |
| `app/rerank.py` | Cohere Rerank v4 client; **config-driven route + graceful fallback**. |
| `app/schema.sql` | pgvector DDL (`kb_articles` + `kb_chunks`, HNSW + GIN indexes). |
| `app/ingest.py` | offline build: parse → field chunks → embed → upsert. |
| `app/retrieval.py` | vector + keyword legs → RRF fusion → distinct articles. |
| `app/followup.py` | **ported legacy logic**: pick the heading/cause/question phrase most relevant to the description AND most discriminating; else `[]` → agent freeform. |
| `app/main.py` | FastAPI service; **legacy routing engine** + retrieval swap. |
| `app/calibrate.py` | gold-set metrics (top-1 / recall@K / MRR) + threshold recommendations. |
| `app/text_utils.py` | shared normalize (stopwords + Porter stem) for follow-up relevance. |

## 5. Routing (unchanged from the AI-Search `main.py`)

- **RESOLVE** — `top_score > CONFIDENT_SCORE` **and** spread `low` (clear winner) → return `kb_id`.
- **Round cap** — at `MAX_ROUNDS`, present best if `≥ MIN_DISPLAY_SCORE`, else `kb_id=null`.
- **Follow-up** — sub-confident: if a heading/cause/question phrase is relevant to
  the description, return it in `discriminating_symptoms` for a grounded question;
  otherwise return `[]` and let the agent ask its own focused Outlook follow-up.

`compute_spread` = weak-absolute (`top < CONFIDENT_SCORE`) OR weak-dominance
(`top − #2 < SPREAD_THRESHOLD`) — identical to the legacy version, on the 0–1 scale.
`discriminating_symptoms` is kept as the response key for contract compatibility;
it now carries heading/cause/question phrases.

## 6. Score bands (0–1)

Defaults are **validated on the gold set in cosine mode** (see `calibrate.py`):
`CONFIDENT_SCORE=0.60` sits above the out-of-KB ceiling (0.536), so off-topic
queries can't auto-resolve. `MIN_DISPLAY_SCORE=0.40`, `FOLLOWUP_FLOOR=0.15`,
`SPREAD_THRESHOLD=0.10`. **Re-run `calibrate.py` if you enable Cohere rerank** —
its scores are more polarized than cosine.

## 7. Run it

```bash
pip install -r requirements.txt

# secrets (or rely on Key Vault via managed identity)
export PG_CONN='postgresql://kbadmin:***@teva-kb-vectordb.postgres.database.azure.com:5432/kbtool?sslmode=require'
export TEVA_AI_KEY='***'

python app/ingest.py --recreate        # build the index in pgvector
python app/calibrate.py                # metrics + threshold check
python -m uvicorn app.main:app --port 8000 --workers 1
curl -s localhost:8000/get_kb_candidates -H 'content-type: application/json' \
     -d '{"description":"outlook is stuck on the loading screen","session_id":"t1"}'
```

## 8. Deployment

Azure App Service (Linux, Python 3.12), single worker (in-memory round counter).
Startup: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1`.
App settings mirror `.env.example`; secrets come from Key Vault via managed
identity (grant the app **Key Vault Secrets User** on `vinny-kb-tool-vault1`, and
add its outbound IP to the Postgres firewall — or use VNet/private endpoint).

## 9. Known open item — Cohere rerank route

The Cohere-rerank-v4 deployment exists and is healthy, but the exact REST route
on `teva.services.ai.azure.com` is **not yet pinned down** (every standard
`/v2/rerank`, `/models/rerank`, … returned 404 — v4 is new). Until it's set, the
service runs in **cosine-fallback mode**, which already achieves **top-1 95.9% /
recall@3 100%** on the gold set. To enable rerank: copy the deployment's **Target
URI** from Azure AI Foundry and set `RERANK_PATH` (+ `RERANK_API_VERSION` /
`RERANK_AUTH_STYLE`) — no code change needed — then re-run `calibrate.py`.

## 10. Limitations

- **Postgres full-text ≠ true BM25** (it's TF-IDF-style); vector + rerank carry
  the quality, so this is a minor keyword-leg caveat, not a ranking problem.
- **Title-only docs** (e.g. KB0013608) have less signal to embed; improve by
  adding a User-Experience/Cause line, not by changing the algorithm.
- **Single worker / in-memory round counter** — scale-out needs the counter in
  Redis/Table storage (the vectors are already shared in Postgres).
