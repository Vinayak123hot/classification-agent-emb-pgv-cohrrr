"""
config.py — all configuration in ONE object.

`Config` reads every setting from environment variables (which on Azure come from
the App Service "Application settings") and resolves the two secrets (Postgres
connection string + the shared Azure AI key) from Key Vault when they are not
supplied directly. A module-level singleton `CONFIG` is created for convenience,
but any component can be handed a different Config instance (useful for tests).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("kb-tool.config")


class Config:
    """Typed configuration holder. All attributes are plain values so the rest of
    the app reads e.g. `config.CONFIDENT_SCORE` without any lookups."""

    def __init__(self, env: "dict[str, str] | None" = None):
        # `env` defaults to the real process environment; injectable for tests.
        e = env if env is not None else os.environ

        # ── Key Vault (secret source of last resort) ──────────────────────────
        self.KEY_VAULT_URL = e.get("AZURE_KEY_VAULT_URL",
                                   "https://vinny-kb-tool-vault1.vault.azure.net")
        self._secret_cache: dict[str, str] = {}   # memoize Key Vault reads
        self._env = e                              # kept so secret getters can read it

        # ── Embeddings (Azure OpenAI, text-embedding-3-small) ─────────────────
        self.EMBED_ENDPOINT = e.get("EMBED_ENDPOINT", "https://teva.openai.azure.com")
        self.EMBED_DEPLOYMENT = e.get("EMBED_DEPLOYMENT", "text-embedding-3-small")
        self.EMBED_API_VERSION = e.get("EMBED_API_VERSION", "2023-05-15")
        self.EMBED_DIMS = int(e.get("EMBED_DIMS", "1536"))   # native size of 3-small

        # ── Cohere rerank (Azure AI Foundry) — route is env-driven on purpose ─
        self.RERANK_ENABLED = e.get("RERANK_ENABLED", "true").lower() == "true"
        self.RERANK_ENDPOINT = e.get("RERANK_ENDPOINT", "https://teva.services.ai.azure.com")
        self.RERANK_DEPLOYMENT = e.get("RERANK_DEPLOYMENT", "cohere-rerank-v4-fast")
        self.RERANK_PATH = e.get("RERANK_PATH", "/v2/rerank")
        self.RERANK_API_VERSION = e.get("RERANK_API_VERSION", "")   # appended if set
        self.RERANK_AUTH_STYLE = e.get("RERANK_AUTH_STYLE", "bearer")  # 'bearer'|'api-key'

        # ── Data location (source .docx for the offline ingest) ───────────────
        self.DATA_DIR = e.get(
            "DATA_DIR",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        )

        # ── Retrieval breadth ─────────────────────────────────────────────────
        self.VECTOR_K = int(e.get("VECTOR_K", "30"))    # chunk rows from the vector leg
        self.KEYWORD_K = int(e.get("KEYWORD_K", "30"))  # chunk rows from the full-text leg
        self.RRF_K = int(e.get("RRF_K", "60"))          # Reciprocal-Rank-Fusion constant
        self.RETURN_K = int(e.get("RETURN_K", "3"))     # distinct articles fed to follow-up
        self.RERANK_K = int(e.get("RERANK_K", "10"))    # distinct articles sent to the reranker

        # ── Score bands (0..1) — TUNED on the gold set (eval/harness/tune.py) ──
        # MIN_DISPLAY_SCORE sits ABOVE the out-of-KB score ceiling (~0.54) so the
        # round-cap fallback never presents an off-topic query.
        self.CONFIDENT_SCORE = float(e.get("CONFIDENT_SCORE", "0.60"))    # auto-resolve gate
        self.FOLLOWUP_FLOOR = float(e.get("FOLLOWUP_FLOOR", "0.15"))      # below = no signal
        self.MIN_DISPLAY_SCORE = float(e.get("MIN_DISPLAY_SCORE", "0.55"))  # present-at-cap floor
        self.SPREAD_THRESHOLD = float(e.get("SPREAD_THRESHOLD", "0.08"))  # near-tie gap

        # ── Follow-up phrase selection ────────────────────────────────────────
        self.TOP_FIELDS_K = int(e.get("TOP_FIELDS_K", "3"))               # max grounded phrases
        self.MIN_FIELD_SCORE = float(e.get("MIN_FIELD_SCORE", "0.35"))    # phrase relevance floor

        # ── Round cap (per session_id) ────────────────────────────────────────
        self.MAX_ROUNDS = int(e.get("MAX_ROUNDS", "5"))
        self.SESSION_TTL_SECONDS = int(e.get("SESSION_TTL_SECONDS", "3600"))

        # ── Optional API-key auth on the tool ─────────────────────────────────
        self.TOOL_API_KEY = e.get("TOOL_API_KEY", "")
        self.REQUIRE_API_KEY = e.get("REQUIRE_API_KEY", "false").lower() == "true"

    # ── secret resolution: env var wins, else Key Vault, else "" ──────────────
    def _kv_secret(self, name: str) -> "str | None":
        """Read a secret from Key Vault via managed identity / az login. Returns
        None on any failure so the caller can fall back to an env var."""
        if name in self._secret_cache:                      # already fetched once
            return self._secret_cache[name]
        try:
            # imported lazily: these packages are optional (only needed for the KV path)
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            client = SecretClient(vault_url=self.KEY_VAULT_URL,
                                  credential=DefaultAzureCredential())
            value = client.get_secret(name).value
            self._secret_cache[name] = value                # cache for next time
            return value
        except Exception as exc:                            # missing pkg / no access / etc.
            logger.warning("Key Vault secret '%s' unavailable (%s: %s)",
                           name, type(exc).__name__, exc)
            return None

    def _secret(self, env_var: str, kv_name: str) -> str:
        """Resolve a secret: explicit env var first (handy for local/CI), else KV."""
        val = self._env.get(env_var)
        if val:
            return val
        return self._kv_secret(kv_name) or ""

    def get_pg_conn(self) -> str:
        """libpq URI to the kbtool database (sslmode=require)."""
        return self._secret("PG_CONN", "pg-vector-conn")

    def get_ai_key(self) -> str:
        """The single Azure AI Services key shared by embeddings + Cohere rerank."""
        return self._secret("TEVA_AI_KEY", "teva-ai-services-key")


# Default singleton used by the composition root and scripts.
CONFIG = Config()
