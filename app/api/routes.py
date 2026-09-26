"""HTTP edge between the client layer and the intelligence layer. Translates
AgentEvents into Server-Sent Events; owns no business logic."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.ratelimit import charge, rate_limit
from app.config import get_settings
from app.intelligence.agent import (
    Agent,
    AgentDone,
    AgentStatus,
    AgentToken,
    AgentToolCall,
    AgentToolResult,
)
from app.inference.provider import ModelOverloadedError, ModelTimeoutError, QuotaExceededError
from app.knowledge.ingest import ingest_dir
from app.knowledge.uploads import (
    MAX_UPLOAD_BYTES,
    UPLOAD_TTL_S,
    PrivateNotesFullError,
    UploadError,
    ingest_upload,
)
from app.observability import log_event

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=8000)
    # IANA name, e.g. "Asia/Kolkata", so "today" means the user's today.
    timezone: str | None = Field(default=None, max_length=64)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _quota_error(exc: QuotaExceededError) -> dict:
    """What the chat shows instead of the raw 429. The client adds resets_at
    in the viewer's own timezone."""
    if exc.daily:
        message = (
            "The assistant has used up today's free Gemini quota, so it can't "
            "answer right now. The quota resets at midnight Pacific time."
        )
    else:
        message = (
            "The assistant is getting more requests than its Gemini quota "
            "allows. Wait a minute and try again."
        )
    resets_at = exc.resets_at.isoformat() if exc.resets_at else None
    return {"kind": "quota", "message": message, "resets_at": resets_at}


def _loggable(value):
    """`value` as the log may keep it: text becomes its length unless
    LOG_CHAT_TEXT is on. A message or a tool's arguments can hold a private
    note ("Save a note titled…", save_note's content), and logs outlive the
    note's 24 hours and are readable by whoever runs the app."""
    if get_settings().log_chat_text:
        return value
    if isinstance(value, str):
        return f"<{len(value)} chars>"
    if isinstance(value, dict):
        return {k: _loggable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_loggable(v) for v in value]
    return value  # numbers, bools, None: no text in them


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
async def notes(request: Request, session_id: str | None = Query(default=None, max_length=64)) -> dict:
    """The shared notes, plus this session's private uploads (flagged)."""
    store = request.app.state.store
    store.purge_uploads(UPLOAD_TTL_S)
    return {"docs": store.list_docs(owner=session_id)}


@router.post("/notes/upload", dependencies=[rate_limit("upload")])
async def upload_note(
    request: Request,
    session_id: str = Form(min_length=1, max_length=64),
    file: UploadFile = File(),
) -> dict:
    """Add a .md/.txt/.pdf note that only this session's searches can see."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)  # one byte over is enough to refuse it
    try:
        result = await ingest_upload(
            session_id,
            file.filename or "upload.txt",
            data,
            request.app.state.store,
            request.app.state.provider,
            charge=lambda chunks: charge(request, "upload_chunks", chunks),
        )
    except PrivateNotesFullError as exc:
        log_event(event="private_notes_full", session_id=session_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (QuotaExceededError, ModelOverloadedError, ModelTimeoutError) as exc:
        # Uploading embeds the note, so it fails when the model provider does.
        log_event(event="upload_failed", session_id=session_id, error=str(exc))
        raise HTTPException(
            status_code=503, detail="The embedding model is unavailable right now, so the note couldn't be indexed. Try again later."
        ) from exc
    log_event(event="note_uploaded", session_id=session_id, **result)
    return result


@router.delete("/notes/{doc_id}")
async def delete_note(doc_id: str, request: Request, session_id: str = Query(min_length=1, max_length=64)) -> dict:
    """Remove one of this session's uploads. Shared notes can't be deleted here."""
    return {"deleted": request.app.state.store.delete_doc(doc_id, owner=session_id)}


@router.post("/ingest", dependencies=[rate_limit("ingest")])
async def ingest(request: Request) -> dict:
    settings = get_settings()
    try:
        counts = await ingest_dir(settings.notes_dir, request.app.state.store, request.app.state.provider)
    except (QuotaExceededError, ModelOverloadedError, ModelTimeoutError) as exc:
        # Notes embedded before the failure stay updated; the rest keep their old chunks.
        log_event(event="ingest_failed", error=str(exc))
        raise HTTPException(
            status_code=503, detail="The embedding model is unavailable right now, so the notes couldn't be re-indexed. Try again later."
        ) from exc
    return {"ingested": counts, "chunks_indexed": request.app.state.store.count()}


@router.post("/reset/{session_id}")
async def reset(session_id: str, request: Request, keep_uploads: bool = False) -> dict:
    """Forget the conversation, and (unless keep_uploads) the session's uploads.
    The page calls it with keep_uploads on reload, which keeps the uploads
    but clears the chat it no longer shows."""
    request.app.state.memory.reset(session_id)
    if not keep_uploads:
        request.app.state.store.delete_owner(session_id)
    return {"reset": session_id}


@router.post("/chat", dependencies=[rate_limit("chat")])
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    agent: Agent = request.app.state.agent

    async def stream():
        started = time.perf_counter()
        try:
            async for ev in agent.run_turn(body.session_id, body.message, timezone=body.timezone):
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
                        "finish_reason": ev.finish_reason,
                        "notice": ev.notice,
                        "steps": ev.trace.as_dicts(),
                    }
                    log_event(
                        event="chat_call",
                        session_id=body.session_id,
                        query=_loggable(body.message),
                        latency_ms=round((time.perf_counter() - started) * 1000, 2),
                        **{k: v for k, v in payload.items() if k not in ("answer", "notice", "steps")},
                        steps=[{**s, "meta": _loggable(s["meta"])} for s in payload["steps"]],
                    )
                    yield _sse("done", payload)
        except QuotaExceededError as exc:
            log_event(event="chat_quota_exceeded", session_id=body.session_id, daily=exc.daily, error=str(exc))
            yield _sse("error", _quota_error(exc))
        except ModelOverloadedError as exc:
            log_event(event="chat_model_overloaded", session_id=body.session_id, error=str(exc))
            yield _sse("error", {
                "kind": "overloaded",
                "message": (
                    "Gemini is overloaded right now (a temporary Google-side issue, "
                    "not a problem with your message). Try again in a minute."
                ),
                "resets_at": None,
            })
        except ModelTimeoutError as exc:
            log_event(event="chat_model_timeout", session_id=body.session_id, error=str(exc))
            yield _sse("error", {
                "kind": "timeout",
                "message": (
                    f"Gemini didn't respond within {get_settings().gemini_timeout_s:g} seconds, so the "
                    "request was stopped. It's usually a temporary slowdown on Google's side; "
                    "try again in a minute."
                ),
                "resets_at": None,
            })
        except Exception as exc:  # noqa: BLE001 - report to the client instead of a dead stream
            # The details stay in the log: an exception's text can be a whole
            # provider error body, which means nothing to the person chatting.
            log_event(event="chat_error", session_id=body.session_id, error=repr(exc))
            yield _sse("error", {
                "kind": "internal",
                "message": "Something went wrong on the server while answering. Try again in a moment.",
                "resets_at": None,
            })

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
