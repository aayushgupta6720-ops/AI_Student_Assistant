"""HTTP edge between the client layer and the intelligence layer. Translates
AgentEvents into Server-Sent Events; owns no business logic."""

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.ratelimit import ALL, charge, rate_limit
from app.api.session import session_id as visitor_session
from app.api.session import session_tag
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
    note_charge,
)
from app.observability import CallTrace, log_event, start_trace

router = APIRouter()


class ChatRequest(BaseModel):
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


def _loggable_sources(request: Request, sources: list[str]) -> list[str]:
    """Source doc_ids as the log may keep them: a private note's is its title
    or file name ("bank-pin-4321"), so it's "<private>" unless LOG_CHAT_TEXT
    is on. Private ids never equal shared ones (uploads.clear_of_shared)."""
    if get_settings().log_chat_text:
        return sources
    shared = {d["doc_id"] for d in request.app.state.store.list_docs()}
    return [s if s in shared else "<private>" for s in sources]


def _timings(trace: CallTrace, started: float) -> dict:
    """How long a request took, in all and step by step, for its log line.
    Steps' details go through _loggable: an upload's doc_id is its file name."""
    return {
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "per_layer_ms": trace.per_layer_ms(),
        "steps": [{**s, "meta": _loggable(s["meta"])} for s in trace.as_dicts()],
    }


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
async def notes(request: Request, session: str = Depends(visitor_session)) -> dict:
    """The shared notes, plus this session's private uploads (flagged)."""
    store = request.app.state.store
    store.purge_uploads(UPLOAD_TTL_S)
    return {"docs": store.list_docs(owner=session)}


@router.post("/notes/upload", dependencies=[rate_limit("upload")])
async def upload_note(
    request: Request,
    file: UploadFile = File(),
    session: str = Depends(visitor_session),
) -> dict:
    """Add a .md/.txt/.pdf note that only this session's searches can see."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)  # one byte over is enough to refuse it
    # Traced like a chat turn, so a slow upload's log says which step was slow.
    trace, started = start_trace(), time.perf_counter()
    try:
        result = await ingest_upload(
            session,
            file.filename or "upload.txt",
            data,
            request.app.state.store,
            request.app.state.provider,
            charge=lambda chunks: charge(request, "upload_chunks", chunks),
            charge_transcription=lambda: charge(request, "transcribe"),
        )
    except PrivateNotesFullError as exc:
        log_event(event="private_notes_full", session=session_tag(session))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (QuotaExceededError, ModelOverloadedError, ModelTimeoutError) as exc:
        # Uploading embeds the note (and reads a scanned PDF), so it fails when the model provider does.
        log_event(event="upload_failed", session=session_tag(session), error=str(exc), **_timings(trace, started))
        raise HTTPException(
            status_code=503, detail="The model is unavailable right now, so the note couldn't be read and indexed. Try again later."
        ) from exc
    log_event(event="note_uploaded", session=session_tag(session), doc_id=_loggable(result["doc_id"]), chunks=result["chunks"],
              transcribed=result.get("transcribed", False), **_timings(trace, started))
    return result


@router.delete("/notes/{doc_id}")
async def delete_note(doc_id: str, request: Request, session: str = Depends(visitor_session)) -> dict:
    """Remove one of this session's uploads. Shared notes can't be deleted here."""
    return {"deleted": request.app.state.store.delete_doc(doc_id, owner=session)}


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


@router.post("/reset")
async def reset(request: Request, keep_uploads: bool = False, session: str = Depends(visitor_session)) -> dict:
    """Forget the conversation, and (unless keep_uploads) the session's uploads.
    The page calls it with keep_uploads on reload, which keeps the uploads
    but clears the chat it no longer shows."""
    request.app.state.memory.reset(session)
    if not keep_uploads:
        request.app.state.store.delete_owner(session)
    return {"reset": True}


@router.post("/chat")
async def chat(body: ChatRequest, request: Request, session: str = Depends(visitor_session)) -> StreamingResponse:
    # Counted here, once the body is valid, not as a route dependency: those
    # run before validation, so an over-long message used up an allowance.
    charge(request, "chat")
    agent: Agent = request.app.state.agent

    def charge_saved_notes(chunks: int) -> None:
        # Notes saved from chat fill the same memory as uploads, so they count
        # against the same daily budget. Tools report errors to the model, so
        # it gets the reason to relay rather than an HTTP 429.
        try:
            charge(request, "upload_chunks", chunks)
        except HTTPException as exc:
            raise UploadError(exc.detail) from exc

    def charge_model_call() -> None:
        # Each model call counts against the visitor's daily budget and the
        # one every visitor shares: a message can take up to six calls.
        charge(request, "model_calls")
        charge(request, "model_calls_all", key=ALL)

    async def stream():
        started = time.perf_counter()
        first_token_ms = None  # how long the visitor waited for the first word
        note_charge.set(charge_saved_notes)  # for save_note, like the agent's current_session
        try:
            async for ev in agent.run_turn(session, body.message, timezone=body.timezone,
                                           before_model_call=charge_model_call):
                if isinstance(ev, AgentStatus):
                    yield _sse("status", asdict(ev))
                elif isinstance(ev, AgentToolCall):
                    yield _sse("tool_call", asdict(ev))
                elif isinstance(ev, AgentToolResult):
                    yield _sse("tool_result", {"id": ev.id, "name": ev.name, "is_error": ev.is_error, **_preview(ev.result)})
                elif isinstance(ev, AgentToken):
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - started) * 1000, 2)
                    yield _sse("token", {"text": ev.text})
                elif isinstance(ev, AgentDone):
                    payload = {
                        "answer": ev.answer,
                        "sources": ev.sources,
                        "passages": ev.passages,
                        "iterations": ev.iterations,
                        "tools_used": ev.tools_used,
                        "prompt_version": ev.prompt_version,
                        "per_layer_ms": ev.trace.per_layer_ms(),
                        "total_ms": round((time.perf_counter() - started) * 1000, 2),
                        "first_token_ms": first_token_ms,
                        "total_tokens": ev.trace.total_tokens,
                        "finish_reason": ev.finish_reason,
                        "notice": ev.notice,
                        "steps": ev.trace.as_dicts(),
                    }
                    log_event(
                        event="chat_call",
                        session=session_tag(session),
                        query=_loggable(body.message),
                        latency_ms=round((time.perf_counter() - started) * 1000, 2),
                        # Not the passages either: they're the notes' own text.
                        **{k: v for k, v in payload.items() if k not in ("answer", "notice", "steps", "passages", "sources")},
                        sources=_loggable_sources(request, ev.sources),
                        steps=[{**s, "meta": _loggable(s["meta"])} for s in payload["steps"]],
                    )
                    yield _sse("done", payload)
        except (asyncio.CancelledError, GeneratorExit):
            # The client went away (Stop, a closed tab, a dropped connection):
            # the agent has already tidied the turn out of memory.
            log_event(
                event="chat_stopped",
                session=session_tag(session),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise
        except HTTPException as exc:  # a model-call budget ran out partway through the turn
            yield _sse("error", {"kind": "rate_limited", "message": exc.detail, "resets_at": None})
        except QuotaExceededError as exc:
            log_event(event="chat_quota_exceeded", session=session_tag(session), daily=exc.daily, error=str(exc))
            yield _sse("error", _quota_error(exc))
        except ModelOverloadedError as exc:
            log_event(event="chat_model_overloaded", session=session_tag(session), error=str(exc))
            yield _sse("error", {
                "kind": "overloaded",
                "message": (
                    "Gemini is overloaded right now (a temporary Google-side issue, "
                    "not a problem with your message). Try again in a minute."
                ),
                "resets_at": None,
            })
        except ModelTimeoutError as exc:
            log_event(event="chat_model_timeout", session=session_tag(session), error=str(exc))
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
            log_event(event="chat_error", session=session_tag(session), error=repr(exc))
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
