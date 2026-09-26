import pytest

from app.inference.types import Message, StreamEnd, TextDelta, ToolCall, ToolCallPart, ToolResultPart
from app.intelligence.agent import Agent, AgentDone, AgentToken, AgentToolCall, AgentToolResult
from app.intelligence.memory import SessionStore
from app.intelligence.prompts import FINAL_CALL_NOTE
from app.knowledge.store import VectorStore
from app.tools.builtin import build_registry
from app.tools.registry import Tool, ToolRegistry
from tests.fake_provider import FakeProvider, text_turn, tool_turn


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool("calculator", "calc", {"type": "object", "properties": {"expression": {"type": "string"}}},
             lambda expression: {"result": 42}),
    )
    return reg


async def _collect(agent, session, text):
    return [e async for e in agent.run_turn(session, text)]


async def test_tool_loop_feeds_result_back():
    provider = FakeProvider([
        tool_turn("calculator", {"expression": "6*7"}),
        [TextDelta("It is "), TextDelta("42.")],
    ])
    memory = SessionStore()
    agent = Agent(provider, _registry(), memory)

    events = await _collect(agent, "s1", "what is 6*7?")

    kinds = [type(e).__name__ for e in events]
    assert kinds.index("AgentToolCall") < kinds.index("AgentToolResult") < kinds.index("AgentToken")
    done = events[-1]
    assert isinstance(done, AgentDone)
    assert done.answer == "It is 42." and done.iterations == 2 and done.tools_used == ["calculator"]

    # second model call must have seen: user, assistant(tool call), tool(result)
    second = provider.seen[1]
    assert [m.role for m in second] == ["user", "assistant", "tool"]
    assert isinstance(second[1].parts[0], ToolCallPart)
    assert isinstance(second[2].parts[0], ToolResultPart) and second[2].parts[0].result == {"result": 42}

    # trace has one step per layer touched
    layers = {s.layer for s in done.trace.steps}
    assert layers == {"inference", "intelligence", "tools"}


async def test_plain_answer_needs_one_iteration_and_memory_persists():
    provider = FakeProvider([text_turn("hi!"), text_turn("you said hi")])
    agent = Agent(provider, _registry(), SessionStore())
    d1 = (await _collect(agent, "s", "hi"))[-1]
    d2 = (await _collect(agent, "s", "what did I say?"))[-1]
    assert d1.iterations == 1 and d2.answer == "you said hi"
    assert [m.role for m in provider.seen[1]] == ["user", "assistant", "user"]


async def test_max_iterations_guard():
    provider = FakeProvider([tool_turn("calculator", {"expression": "1"})] * 3)
    agent = Agent(provider, _registry(), SessionStore(), max_iterations=3)
    done = (await _collect(agent, "s", "loop"))[-1]
    assert done.iterations == 3 and len(provider.turns) == 0
    assert done.tools_used == ["calculator"] * 2  # a call on the last turn, offered no tools, isn't run


async def test_the_last_allowed_call_gets_no_tools_so_the_turn_ends_with_an_answer():
    # It used to offer tools too, so a model still searching ended the turn
    # with results it never read and "(no text answer)" on screen.
    provider = FakeProvider([
        tool_turn("calculator", {"expression": "1"}, "c1"),
        tool_turn("calculator", {"expression": "2"}, "c2"),
        text_turn("From what I found: 42."),
    ])
    memory = SessionStore()
    agent = Agent(provider, _registry(), memory, max_iterations=3)

    done = (await _collect(agent, "s", "loop"))[-1]

    assert done.answer == "From what I found: 42." and done.iterations == 3
    assert provider.tools_seen == [["calculator"], ["calculator"], []]
    assert [FINAL_CALL_NOTE in s for s in provider.systems] == [False, False, True]
    assert memory.history("s")[-1].role == "assistant"  # no unread tool results left behind


@pytest.mark.parametrize(
    ("events", "answer", "notice"),
    [
        ([TextDelta("The first three books are")], "The first three books are", None),  # scripted "stop"
        ([TextDelta("The first three books are"), StreamEnd("max_tokens")], "The first three books are", "cut off"),
        ([StreamEnd("safety")], "", "safety filters"),
    ],
)
async def test_an_answer_that_stops_early_says_why(events, answer, notice):
    # A cut-off answer just ended mid-sentence, and a blocked one showed
    # "(no text answer)".
    agent = Agent(FakeProvider([events]), _registry(), SessionStore())

    done = (await _collect(agent, "s", "what's on my reading list?"))[-1]

    assert done.answer == answer
    if notice is None:
        assert done.notice is None and done.finish_reason == "stop"
    else:
        assert notice in done.notice


def test_memory_trim_never_orphans_tool_results():
    mem = SessionStore(window_messages=3)
    mem.append("s", Message("user"), Message("assistant", [ToolCallPart("1", "t", {})]),
               Message("tool", [ToolResultPart("1", "t", {})]), Message("assistant"), Message("user"))
    assert [m.role for m in mem.history("s")] == ["user"]


def test_memory_forgets_the_least_recently_used_session_past_the_cap():
    mem = SessionStore(max_sessions=2)
    mem.append("a", Message("user"))
    mem.append("b", Message("user"))
    mem.append("a", Message("user"))  # a is now the most recent
    mem.append("c", Message("user"))

    assert [len(mem.history(s)) for s in ("a", "b", "c")] == [2, 0, 1]
    assert mem.history("never-seen") == [] and len(mem._sessions) == 2  # reading doesn't add a session


async def test_the_turns_time_zone_reaches_current_datetime():
    provider = FakeProvider([tool_turn("current_datetime", {}), text_turn("ok")])
    agent = Agent(provider, build_registry(provider, VectorStore(":memory:")), SessionStore())

    events = [e async for e in agent.run_turn("s", "what time is it?", timezone="Asia/Kolkata")]

    [result] = [e for e in events if isinstance(e, AgentToolResult)]
    assert result.result["timezone"] == "Asia/Kolkata" and result.result["iso"].endswith("+05:30")
