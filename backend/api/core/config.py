"""Settings from environment variables and backend/.env (see backend/.env.example)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # LLM: any OpenAI-compatible chat-completions API with tool calling.
    # Free options: Groq (default), Google Gemini, OpenRouter ":free" models, local Ollama. See .env.example.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_max_tokens: int = 700
    llm_timeout_s: float = 30
    llm_reasoning_effort: str = "low"  # sent only to reasoning models that accept it (gpt-oss)
    # Vercel AI Gateway: when set it overrides the key and base URL above (use a provider-prefixed model id)
    ai_gateway_api_key: str = ""
    ai_gateway_base_url: str = "https://ai-gateway.vercel.sh/v1"

    # Speech-to-text (OpenAI-compatible /audio/transcriptions). Empty key/URL reuse the LLM ones (Groq serves Whisper free).
    stt_api_key: str = ""
    stt_base_url: str = ""
    stt_model: str = "whisper-large-v3-turbo"

    # Storage / infra
    uploads_dir: Path = BACKEND_DIR / "uploads"
    upload_max_bytes: int = 10 * 1024 * 1024
    registry_path: Path = BACKEND_DIR / "registry" / "model_registry.json"
    embeddings_cache: Path = BACKEND_DIR / "registry" / "embeddings.npz"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_dummy_model: bool = False  # adds the dummy_echo test model to model search
    preview_models: str = ""  # comma-separated model ids searchable before they pass the switch-on bar (local testing only)
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = ""

    # Model workers
    worker_tabular_url: str = "http://localhost:8101"
    worker_torch_url: str = "http://localhost:8102"
    worker_tf_url: str = "http://localhost:8103"
    worker_timeout_s: float = 40

    # Model search
    retrieval_threshold: float = 0.35
    retrieval_top_k: int = 4
    max_tool_calls: int = 3

    # App
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def llm_endpoint(self) -> tuple[str, str]:
        if self.ai_gateway_api_key:
            return self.ai_gateway_base_url, self.ai_gateway_api_key
        return self.llm_base_url, self.llm_api_key

    @property
    def stt_endpoint(self) -> tuple[str, str]:
        return self.stt_base_url or self.llm_base_url, self.stt_api_key or self.llm_api_key

    def worker_url(self, worker: str) -> str:
        return {"tabular": self.worker_tabular_url, "torch": self.worker_torch_url, "tf": self.worker_tf_url}[worker]


@lru_cache
def get_settings() -> Settings:
    return Settings()
