"""Composition root: builds each layer once and wires them together.
This is the only place that knows about all five layers at once."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import PROJECT_ROOT, get_settings
from app.inference.gemini import get_provider          # inference layer
from app.intelligence.agent import Agent                # intelligence layer
from app.intelligence.memory import SessionStore
from app.knowledge.retrieval import get_store           # knowledge layer
from app.observability import configure_logging
from app.tools.builtin import build_registry            # tools layer

CLIENT_DIR = PROJECT_ROOT / "client"                    # client layer


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    provider = get_provider()
    store = get_store()
    registry = build_registry(provider, store)
    memory = SessionStore(settings.memory_window_messages)

    app.state.provider = provider
    app.state.store = store
    app.state.registry = registry
    app.state.memory = memory
    app.state.agent = Agent(provider, registry, memory)
    yield


app = FastAPI(title=get_settings().app_name, lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory=CLIENT_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(CLIENT_DIR / "index.html")
