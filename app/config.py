from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "AI_Student_Assistant"

    gemini_api_key: str = ""
    generation_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    # Give up on a 429 if the server asks us to wait longer than this (daily
    # quota exhaustion comes back as a 429 too - no point sleeping a minute).
    rate_limit_max_wait_s: float = 20.0

    notes_dir: Path = PROJECT_ROOT / "data" / "notes"
    store_path: Path = PROJECT_ROOT / "data" / "knowledge.sqlite"

    retrieval_top_k: int = 4
    chunk_max_chars: int = 800
    chunk_overlap_chars: int = 100

    max_agent_iterations: int = 6
    memory_window_messages: int = 20

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
