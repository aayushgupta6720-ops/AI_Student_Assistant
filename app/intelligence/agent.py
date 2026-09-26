"""The agent loop. This *is* the intelligence layer: it decides what the model
sees, when tools run, and when the turn is finished.

    loop:
        stream model (inference layer)
        forward text to the client as it arrives
        if the model requested tools:
            run them (tools layer) -> append results -> continue
        else:
            done
"""

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Union

from app.config import get_settings
from app.inference.provider import LLMProvider
from app.inference.types import (
    Message,
    StreamEnd,
    TextDelta,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolResultPart,
    Usage,
)
from app.intelligence.memory import SessionStore
from app.knowledge.retrieval import current_session
from app.intelligence.prompts import FINAL_CALL_NOTE, SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION
from app.observability import CallTrace, start_trace, time_step
from app.tools.builtin import user_timezone
from app.tools.registry import ToolRegistry

# ---- events the intelligence layer emits to whoever is driving it -----------


@dataclass
class AgentStatus:
    text: str
    iteration: int


@dataclass
class AgentToolCall:
    id: str
    name: str
    args: dict


@dataclass
class AgentToolResult:
    id: str
    name: str
    result: dict
    is_error: bool


@dataclass
class AgentToken:
    text: str


@dataclass
class AgentDone:
    answer: str
    sources: list[str]
    iterations: int
    trace: CallTrace
    prompt_version: str = SYSTEM_PROMPT_VERSION
    tools_used: list[str] = field(default_factory=list)
    finish_reason: str = "stop"  # why the turn's last model call stopped
    notice: str | None = None  # what to tell the user when that wasn't a normal finish


AgentEvent = Union[AgentStatus, AgentToolCall, AgentToolResult, AgentToken, AgentDone]

# Shown under the answer (or instead of it), so a cut-off or blocked answer
# says why rather than ending mid-sentence or showing nothing.
_NOTICES = {
    "max_tokens": "The answer was cut off because it reached the model's length limit.",
    "safety": "Gemini's safety filters blocked this answer.",
    "recitation": "Gemini stopped this answer because it would have repeated copyrighted text.",
    "tool_call_error": "The model tried to use a tool but couldn't form the request. Try rephrasing.",
    "other": "The model stopped before finishing its answer.",
}


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        memory: SessionStore,
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.memory = memory
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations or get_settings().max_agent_iterations

    async def run_turn(
        self, session_id: str, user_text: str, timezone: str | None = None
    ) -> AsyncIterator[AgentEvent]:
        trace = start_trace()
        current_session.set(session_id)  # tools' searches see this session's uploads
        user_timezone.set(timezone)  # current_datetime answers in the user's zone
        self.memory.append(session_id, Message("user", [TextPart(user_text)]))

        answer_parts: list[str] = []
        sources: list[str] = []
        tools_used: list[str] = []
        iterations = 0
        finish_reason = "stop"

        while iterations < self.max_iterations:
            iterations += 1
            # The last call gets no tools, so the model has to answer with what
            # it has instead of asking for results the turn would never read.
            final = iterations == self.max_iterations
            yield AgentStatus("thinking", iterations)

            # -- 1. ask the model (inference layer) ------------------------------
            text_parts: list[TextPart] = []
            tool_calls: list[ToolCall] = []
            with time_step("inference", "generate", iteration=iterations) as usage:
                async for event in self.provider.stream_generate(
                    system=self.system_prompt + (FINAL_CALL_NOTE if final else ""),
                    messages=self.memory.history(session_id),
                    tools=[] if final else self.registry.specs(),
                ):
                    if isinstance(event, TextDelta):
                        text_parts.append(TextPart(event.text, event.provider_state))
                        yield AgentToken(event.text)
                    elif isinstance(event, ToolCall) and not final:  # none offered on the last call
                        tool_calls.append(event)
                    elif isinstance(event, Usage):
                        usage["input_tokens"] = event.input_tokens
                        usage["output_tokens"] = event.output_tokens
                    elif isinstance(event, StreamEnd):
                        finish_reason = event.finish_reason

            # Record exactly what the model said so the next request replays it.
            assistant_parts = [*_merge_text(text_parts), *[
                ToolCallPart(c.id, c.name, c.args, c.provider_state) for c in tool_calls
            ]]
            if assistant_parts:
                self.memory.append(session_id, Message("assistant", assistant_parts))
            answer_parts.extend(p.text for p in text_parts)

            if not tool_calls:
                break  # plain answer -> turn is over

            # -- 2. run the requested tools (tools layer), in parallel ---------
            yield AgentStatus(f"running {len(tool_calls)} tool(s)", iterations)
            for call in tool_calls:
                yield AgentToolCall(call.id, call.name, call.args)

            with time_step("intelligence", "execute_tools", count=len(tool_calls)):
                outcomes = await asyncio.gather(
                    *(self.registry.execute(c.name, c.args) for c in tool_calls)
                )

            result_parts: list[ToolResultPart] = []
            for call, outcome in zip(tool_calls, outcomes):
                tools_used.append(call.name)
                for doc_id in _extract_sources(outcome.result):
                    if doc_id not in sources:
                        sources.append(doc_id)
                result_parts.append(
                    ToolResultPart(call.id, call.name, outcome.result, outcome.is_error)
                )
                yield AgentToolResult(call.id, call.name, outcome.result, outcome.is_error)

            # -- 3. feed results back and loop --------------------------------
            self.memory.append(session_id, Message("tool", result_parts))

        yield AgentDone(
            answer="".join(answer_parts),
            sources=sources,
            iterations=iterations,
            trace=trace,
            tools_used=tools_used,
            finish_reason=finish_reason,
            notice=_NOTICES.get(finish_reason),
        )


def _merge_text(parts: list[TextPart]) -> list[TextPart]:
    """Collapse streamed deltas into one TextPart, keeping the last provider
    state (Gemini attaches its signature to the final part)."""
    if not parts:
        return []
    state = next((p.provider_state for p in reversed(parts) if p.provider_state), {})
    return [TextPart("".join(p.text for p in parts), state)]


def _extract_sources(result: dict) -> list[str]:
    """search_notes returns {"results": [{"doc_id": ...}]}; save_note returns {"doc_id"}."""
    if "results" in result:
        return [r["doc_id"] for r in result["results"] if "doc_id" in r]
    if "doc_id" in result:
        return [result["doc_id"]]
    return []
