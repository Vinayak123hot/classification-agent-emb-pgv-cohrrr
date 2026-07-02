"""
config.py — central configuration for the pgvector + embedding + Cohere-rerank
edition of the KB Candidates tool.

Everything is env-overridable. Secrets (Postgres connection string, the shared
Azure AI Services key used by BOTH the embedding model and the Cohere reranker)
are read from Azure Key Vault when available, with a plain-env fallback so the
ingest job / calibration can run locally with exported variables.

Resources provisioned 2026-06-30 (see repo README):
  - Postgres:  teva-kb-vectordb.postgres.database.azure.com  (db 'kbtool', pgvector 0.8.2)
  - Embedding: text-embedding-3-small on the 'Teva' Azure AI Services account
  - Rerank:    cohere-rerank-v4-fast on the same account
  - Key Vault: vinny-kb-tool-vault1  (secrets: pg-vector-conn, teva-ai-services-key)
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger("kb-tool.config")

# ── Key Vault ──────────────────────────────────────────────────────────────
KEY_VAULT_URL = os.environ.get("AZURE_KEY_VAULT_URL",
                               "https://vinny-kb-tool-vault1.vault.azure.net")

_secret_cache: dict[str, str] = {}


def _kv_secret(name: str) -> str | None:
    """Fetch a secret from Key Vault via DefaultAzureCredential. Returns None on
    any failure (so the env fallback can take over)."""
    if name in _secret_cache:
        return _secret_cache[name]
    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient
        client = SecretClient(vault_url=KEY_VAULT_URL, credential=DefaultAzureCredential())
        val = client.get_secret(name).value
        _secret_cache[name] = val
        return val
    except Exception as e:  # pragma: no cover
        logger.warning("Key Vault secret '%s' unavailable (%s: %s)", name, type(e).__name__, e)
        return None


def _secret(env_var: str, kv_name: str) -> str:
    """Env var wins (handy for local runs / CI); else Key Vault; else ''."""
    v = os.environ.get(env_var)
    if v:
        return v
    return _kv_secret(kv_name) or ""


def get_pg_conn() -> str:
    """libpq URI to the kbtool database (sslmode=require)."""
    return _secret("PG_CONN", "pg-vector-conn")


def get_ai_key() -> str:
    """The single Azure AI Services key shared by embeddings + Cohere rerank."""
    return _secret("TEVA_AI_KEY", "teva-ai-services-key")


# ── Embeddings (Azure OpenAI, text-embedding-3-small) ──────────────────────
# The Azure OpenAI host serves the embedding model most reliably.
EMBED_ENDPOINT    = os.environ.get("EMBED_ENDPOINT", "https://teva.openai.azure.com")
EMBED_DEPLOYMENT  = os.environ.get("EMBED_DEPLOYMENT", "text-embedding-3-small")
EMBED_API_VERSION = os.environ.get("EMBED_API_VERSION", "2023-05-15")
EMBED_DIMS        = int(os.environ.get("EMBED_DIMS", "1536"))   # text-embedding-3-small native

# ── Cohere rerank (Azure AI Foundry) ───────────────────────────────────────
# NOTE: the exact REST route for Cohere-rerank-v4 on the Foundry endpoint is
# config-driven on purpose — see rerank.py. If the call fails or RERANK_ENABLED
# is false, retrieval falls back to the embedding cosine score (degraded but
# functional), so the service always works.
RERANK_ENABLED     = os.environ.get("RERANK_ENABLED", "true").lower() == "true"
RERANK_ENDPOINT    = os.environ.get("RERANK_ENDPOINT", "https://teva.services.ai.azure.com")
RERANK_DEPLOYMENT  = os.environ.get("RERANK_DEPLOYMENT", "cohere-rerank-v4-fast")
RERANK_PATH        = os.environ.get("RERANK_PATH", "/v2/rerank")
RERANK_API_VERSION = os.environ.get("RERANK_API_VERSION", "")   # appended as ?api-version= if set
RERANK_AUTH_STYLE  = os.environ.get("RERANK_AUTH_STYLE", "bearer")  # 'bearer' | 'api-key'

# ── Database / data ─────────────────────────────────────────────────────────
DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)

# ── Retrieval knobs ─────────────────────────────────────────────────────────
VECTOR_K   = int(os.environ.get("VECTOR_K", "30"))    # chunk rows from the vector leg
KEYWORD_K  = int(os.environ.get("KEYWORD_K", "30"))   # chunk rows from the FTS leg
RRF_K      = int(os.environ.get("RRF_K", "60"))       # RRF constant
RETURN_K   = int(os.environ.get("RETURN_K", "3"))     # distinct articles fed to follow-up
RERANK_K   = int(os.environ.get("RERANK_K", "10"))    # distinct articles sent to the reranker

# ── Score bands (0..1) — routing identical to the legacy AI-Search main.py ──
# Defaults are tuned for the Cohere rerank 0..1 relevance scale. RE-CALIBRATE
# with calibrate.py once the corpus is ingested (and especially if you run in
# cosine-fallback mode, whose score distribution is lower/tighter).
# Defaults below are TUNED on the gold set (eval/harness/tune.py, 2026-07-01,
# cosine-fallback mode): raising MIN_DISPLAY_SCORE above the out-of-KB score
# ceiling (~0.54) lifted out-of-KB rejection from 33% -> 92% at the round cap
# while keeping in-KB end-to-end at ~98% and display precision at 100%.
CONFIDENT_SCORE   = float(os.environ.get("CONFIDENT_SCORE", "0.60"))   # auto-resolve threshold
FOLLOWUP_FLOOR    = float(os.environ.get("FOLLOWUP_FLOOR", "0.15"))    # below = no usable signal
MIN_DISPLAY_SCORE = float(os.environ.get("MIN_DISPLAY_SCORE", "0.55")) # present-best-at-cap floor (tuned 0.40->0.55)
SPREAD_THRESHOLD  = float(os.environ.get("SPREAD_THRESHOLD", "0.08"))  # near-tie gap (tuned 0.10->0.08)

# ── Follow-up (description ↔ heading/cause/question relevance) ──────────────
TOP_FIELDS_K   = int(os.environ.get("TOP_FIELDS_K", "3"))     # max grounded follow-up phrases
MIN_FIELD_SCORE = float(os.environ.get("MIN_FIELD_SCORE", "0.35"))  # relevance floor (tuned 0.40->0.35: grounded follow-ups 64%->86%)

# ── Round cap (per session_id) ──────────────────────────────────────────────
MAX_ROUNDS          = int(os.environ.get("MAX_ROUNDS", "5"))
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))

# ── Optional API-key auth on the tool itself ────────────────────────────────
TOOL_API_KEY    = os.environ.get("TOOL_API_KEY", "")
REQUIRE_API_KEY = os.environ.get("REQUIRE_API_KEY", "false").lower() == "true"
