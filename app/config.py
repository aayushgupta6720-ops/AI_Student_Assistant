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
    # Per-request limit on Gemini calls, including each wait between streamed
    # chunks. A slow-but-working call has taken up to ~47s during a demand
    # spike; past this, a stalled call is abandoned.
    gemini_timeout_s: float = 60.0

    notes_dir: Path = PROJECT_ROOT / "data" / "notes"
    store_path: Path = PROJECT_ROOT / "data" / "knowledge.sqlite"

    retrieval_top_k: int = 4
    # Drop passages scoring this far below the best match. For a question about
    # the sample notes, the right note scored 0.64-0.78 and unrelated ones
    # 0.50-0.58, and without this every search returned (and cited) the top 4.
    retrieval_score_margin: float = 0.1
    chunk_max_chars: int = 800
    chunk_overlap_chars: int = 100

    max_agent_iterations: int = 6
    memory_window_messages: int = 20
    max_sessions: int = 1000  # conversations kept in memory; least recently used go first

    # Per-visitor limits (see app/api/ratelimit.py) so one visitor can't use
    # up the shared daily Gemini quota. 0 turns a limit off.
    chat_limit_per_minute: int = 6
    chat_limit_per_day: int = 30
    upload_limit_per_hour: int = 10
    # Uploads also count their chunks, since a big file is hundreds of chunks
    # to embed and keep in memory (a 200,000-character file is ~500).
    upload_chunk_limit_per_day: int = 1000
    ingest_limit_per_hour: int = 3
    # Private chunks the whole app keeps, across every session. Search holds
    # them all in memory: 10,000 took the process from 86 MB to a 205 MB peak,
    # and a free instance has 512 MB.
    max_private_chunks: int = 10_000
    # A header holding the visitor's IP, set by a proxy in front of the app
    # that overwrites any client-sent copy (behind Cloudflare, as on Render:
    # CF-Connecting-IP). Unset means the connecting address, right when
    # nothing sits in front. Never X-Forwarded-For: visitors can forge it.
    client_ip_header: str | None = None

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
