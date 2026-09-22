"""HTTP edge between the client layer and the intelligence layer. Translates
AgentEvents into Server-Sent Events; owns no business logic."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.intelligence.agent import (
    Agent,
    AgentDone,
    AgentStatus,
    AgentToken,
    AgentToolCall,
    AgentToolResult,
)
from app.knowledge.ingest import ingest_dir
from app.observability import log_event

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=8000)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _preview(result: dict, limit: int = 300) -> dict:
    """Trim tool results for the wire; the model still gets the full thing."""
    text = json.dumps(result, default=str)
    return {"preview": text[:limit] + ("…" if len(text) > limit else "")}


@router.get("/health")
async def health(request: Request) -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.generation_model,
        "chunks_indexed": request.app.state.store.count(),
        "tools": request.app.state.registry.names(),
    }


@router.get("/notes")
async def notes(request: Request) -> dict:
    return {"docs": request.app.state.store.list_docs()}


@router.post("/ingest")
async def ingest(request: Request) -> dict:
    settings = get_settings()
    counts = await ingest_dir(settings.notes_dir, request.app.state.store, request.app.state.provider)
    return {"ingested": counts, "chunks_indexed": request.app.state.store.count()}


@router.post("/reset/{session_id}")
async def reset(session_id: str, request: Request) -> dict:
    request.app.state.memory.reset(session_id)
    return {"reset": session_id}


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    agent: Agent = request.app.state.agent

    async def stream():
        started = time.perf_counter()
        try:
            async for ev in agent.run_turn(body.session_id, body.message):
                if isinstance(ev, AgentStatus):
                    yield _sse("status", asdict(ev))
                elif isinstance(ev, AgentToolCall):
                    yield _sse("tool_call", asdict(ev))
                elif isinstance(ev, AgentToolResult):
                    yield _sse("tool_result", {"id": ev.id, "name": ev.name, "is_error": ev.is_error, **_preview(ev.result)})
                elif isinstance(ev, AgentToken):
                    yield _sse("token", {"text": ev.text})
                elif isinstance(ev, AgentDone):
                    payload = {
                        "answer": ev.answer,
                        "sources": ev.sources,
                        "iterations": ev.iterations,
                        "tools_used": ev.tools_used,
                        "prompt_version": ev.prompt_version,
                        "per_layer_ms": ev.trace.per_layer_ms(),
                        "total_tokens": ev.trace.total_tokens,
                        "steps": ev.trace.as_dicts(),
                    }
                    log_event(
                        event="chat_call",
                        session_id=body.session_id,
                        query=body.message,
                        latency_ms=round((time.perf_counter() - started) * 1000, 2),
                        **{k: v for k, v in payload.items() if k != "answer"},
                    )
                    yield _sse("done", payload)
        except Exception as exc:  # noqa: BLE001 - report to the client instead of a dead stream
            log_event(event="chat_error", session_id=body.session_id, error=repr(exc))
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
