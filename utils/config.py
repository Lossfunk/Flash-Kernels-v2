"""
Minimal configuration module for the Kernel Development System.

Only the options that are still referenced elsewhere in the codebase are kept:
  • API keys for Google Gemini and Anthropic Claude (via OpenRouter).
  • LLM provider / model / temperature selection.
  • Global log-level used by utils.logging_utils.

If additional settings are needed in the future, add them here where they are
actually consumed to avoid configuration bloat.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load configuration from YAML and .env
# ---------------------------------------------------------------------------

_CONFIG_PATH = (
    Path("config.yaml") if Path("config.yaml").exists() else Path("config/config.yaml")
)
_YAML_CONFIG: dict = {}

if _CONFIG_PATH.exists():
    try:
        _YAML_CONFIG = yaml.safe_load(_CONFIG_PATH.read_text()) or {}
    except Exception as e:  # pragma: no cover – config errors should not crash
        print(f"⚠️  Failed to parse {_CONFIG_PATH}: {e}")

# Ensure environment variables from .env are loaded *after* YAML so they win
load_dotenv()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _yaml(key: str, default: str | float | None = None):
    """Read a key from YAML with a default."""
    return _YAML_CONFIG.get(key, default)


def _yaml_float(key: str, default: float) -> float:
    try:
        return float(_yaml(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Public configuration object
# ---------------------------------------------------------------------------

class KernelDevConfig:
    """Light-weight configuration accessor holding only live options."""

    # === API keys ===
    @property
    def google_api_key(self) -> str:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set")
        return key

    @property
    def anthropic_api_key(self) -> str:
        """Return API key for Anthropic Claude.

        Precedence:
          1. `ANTHROPIC_API_KEY`  – recommended for direct Anthropic access.
          2. `OPENROUTER_API_KEY` – fallback for routed access via OpenRouter.
        """
        key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "Neither ANTHROPIC_API_KEY nor OPENROUTER_API_KEY is set in the environment"
            )
        return key

    @property
    def openrouter_api_key(self) -> str:
        """Return API key for OpenRouter (always OPENROUTER_API_KEY)."""
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
        return key

    @property
    def openai_api_key(self) -> str:
        """Return API key for OpenAI (official OpenAI endpoints)."""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        return key

    # === LLM selection ===
    @property
    def llm_provider(self) -> str:
        """Return the provider name (`gemini`, `anthropic`, or `openrouter`)."""
        return str(_yaml("LLM_PROVIDER", "anthropic"))

    @property
    def llm_model(self) -> str:
        """Model identifier, e.g. `gemini-1.5-flash` or `anthropic:claude-sonnet-4`."""
        return str(_yaml("LLM_MODEL", "claude-sonnet-4"))

    @property
    def llm_temperature(self) -> float:
        """Sampling temperature (0.0 – 2.0)."""
        return _yaml_float("LLM_TEMPERATURE", 0.2)

    @property
    def llm_max_tokens(self) -> int:
        """Maximum tokens for generation (provider-specific)."""
        try:
            return int(_yaml("LLM_MAX_TOKENS", 4096))
        except (TypeError, ValueError):
            return 4096

    # === Logging ===
    @property
    def log_level(self) -> str:
        """Return the configured log-level name (upper-case)."""
        return str(_yaml("LOG_LEVEL", "INFO")).upper()

    @property
    def synthesis_max_attempts(self) -> int:
        """Maximum number of synth/compile attempts before hard failure."""
        try:
            return int(_yaml("SYNTHESIS_MAX_ATTEMPTS", 15))
        except (TypeError, ValueError):
            return 15

    @property
    def benchmark_warmup_runs(self) -> int:
        """Number of warm-up iterations before timing."""
        try:
            return int(_yaml("BENCHMARK_WARMUP_RUNS", 10))
        except (TypeError, ValueError):
            return 10

    @property
    def benchmark_timing_runs(self) -> int:
        """Number of measured iterations for timing statistics."""
        try:
            return int(_yaml("BENCHMARK_TIMING_RUNS", 100))
        except (TypeError, ValueError):
            return 100


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

config = KernelDevConfig()