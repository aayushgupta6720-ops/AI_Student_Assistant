from app.inference.types import Message, TextDelta, ToolCall, ToolCallPart, ToolResultPart
from app.intelligence.agent import Agent, AgentDone, AgentToken, AgentToolCall, AgentToolResult
from app.intelligence.memory import SessionStore
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


def test_memory_trim_never_orphans_tool_results():
    mem = SessionStore(window_messages=3)
    mem.append("s", Message("user"), Message("assistant", [ToolCallPart("1", "t", {})]),
               Message("tool", [ToolResultPart("1", "t", {})]), Message("assistant"), Message("user"))
    assert [m.role for m in mem.history("s")] == ["user"]
