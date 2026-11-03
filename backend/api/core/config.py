"""Settings from environment variables and backend/.env (see backend/.env.example)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # LLM: Anthropic Messages API, called directly (LLM_API_KEY) or through Vercel AI Gateway (AI_GATEWAY_API_KEY)
    llm_api_key: str = ""
    ai_gateway_api_key: str = ""
    ai_gateway_base_url: str = "https://ai-gateway.vercel.sh"
    llm_model: str = "claude-sonnet-5-5"  # with the gateway use the provider-prefixed id, e.g. anthropic/claude-sonnet-5
    llm_max_tokens: int = 700

    # Speech-to-text
    stt_provider: str = "openai"  # openai | local
    openai_api_key: str = ""
    stt_model: str = "whisper-1"

    # Storage / infra
    uploads_dir: Path = BACKEND_DIR / "uploads"
    registry_path: Path = BACKEND_DIR / "registry" / "model_registry.json"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = ""

    # Model workers
    worker_tabular_url: str = "http://localhost:8101"
    worker_torch_url: str = "http://localhost:8102"
    worker_tf_url: str = "http://localhost:8103"

    # Model search
    retrieval_threshold: float = 0.35
    retrieval_top_k: int = 4

    # App
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"
    log_json: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
