"""Composition root: builds each layer once and wires them together.
This is the only place that knows about all five layers at once."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.body_limit import BodySizeLimit
from app.api.ratelimit import build_rate_limiters
from app.api.routes import router
from app.api.session import SessionCookie
from app.config import PROJECT_ROOT, get_settings
from app.inference.gemini import get_provider          # inference layer
from app.intelligence.agent import Agent                # intelligence layer
from app.intelligence.memory import SessionStore
from app.knowledge.ingest import embedded_with, ingest_dir
from app.knowledge.retrieval import get_store           # knowledge layer
from app.knowledge.uploads import MAX_UPLOAD_BYTES, UPLOAD_TTL_S
from app.observability import configure_logging, log_event
from app.tools.builtin import build_registry            # tools layer

CLIENT_DIR = PROJECT_ROOT / "client"                    # client layer


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    provider = get_provider()
    store = get_store()
    registry = build_registry(provider, store)
    # Chat history quotes private notes, so it expires with them.
    memory = SessionStore(settings.memory_window_messages, settings.max_sessions, max_age_s=UPLOAD_TTL_S)

    store.purge_uploads(UPLOAD_TTL_S)
    stale = store.purge_stale_private(embedded_with())
    if stale:
        log_event(event="stale_private_notes_dropped", chunks=stale)

    # Sync the index with data/notes on every start. On an ephemeral disk
    # (Render, Cloud Run) it's empty after each deploy; elsewhere unchanged
    # notes cost nothing, and changed ones, or all of them after an embedding
    # model change, are embedded again. Only syncing an empty index left a
    # model change breaking search until someone re-ingested by hand.
    if settings.notes_dir.exists():
        try:
            counts = await ingest_dir(settings.notes_dir, store, provider)
            log_event(event="startup_ingest", docs=len(counts), chunks=store.count(),
                      keyword_search=store.keyword_search_enabled)
        except Exception as exc:
            # A quota 429, a bad key or a network blip shouldn't keep the whole
            # app down: chat still works, and search_notes finds whatever was
            # already indexed (nothing, on a fresh disk) until POST /ingest succeeds.
            log_event(
                event="startup_ingest_failed",
                error=f"{type(exc).__name__}: {exc}",
                chunks=store.count(),
            )

    app.state.provider = provider
    app.state.store = store
    app.state.registry = registry
    app.state.memory = memory
    app.state.agent = Agent(provider, registry, memory)
    app.state.rate_limiters = build_rate_limiters(settings)
    yield


app = FastAPI(title=get_settings().app_name, lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory=CLIENT_DIR), name="static")
# A chat message is at most 8,000 chars, well under 128 KB even fully escaped;
# an upload is up to MAX_UPLOAD_BYTES plus the multipart form around it.
app.add_middleware(SessionCookie)  # every request gets a session the server issued
app.add_middleware(
    BodySizeLimit,
    max_bytes=128 * 1024,
    max_bytes_by_path={"/notes/upload": MAX_UPLOAD_BYTES + 64 * 1024},
)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(CLIENT_DIR / "index.html")
