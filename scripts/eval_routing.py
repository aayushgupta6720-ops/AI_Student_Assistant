"""Check which tools the model picks for typical questions, against the real
Gemini API. Each question is a fresh chat of 1-3 model calls, so a run spends
~20 of the day's free-tier requests (shared with anything else on the key).

    python -m scripts.eval_routing            # every case once
    python -m scripts.eval_routing --repeat 3 --only "assistant built"
"""

import argparse
import asyncio
import sys
import uuid

from app.config import get_settings
from app.inference.gemini import get_provider
from app.inference.provider import QuotaExceededError
from app.intelligence.agent import Agent, AgentDone
from app.intelligence.memory import SessionStore
from app.intelligence.prompts import SYSTEM_PROMPT_VERSION
from app.knowledge.ingest import ingest_dir
from app.knowledge.retrieval import get_store
from app.tools.builtin import build_registry

# (question, the tools it should use): the page's suggestions, plus questions
# that are easy to route wrong in either direction.
CASES: list[tuple[str, set[str]]] = [
    ("What's on my reading list?", {"search_notes"}),
    ("How is this assistant built?", {"search_notes"}),  # answered from memory once in production
    ("Which layer of this app talks to Gemini?", {"search_notes"}),
    ("What do I need to buy for the pasta?", {"search_notes"}),
    ("What is 17% of 2,340?", {"calculator"}),
    ("What day is it today?", {"current_datetime"}),
    ("Save a note titled Groceries: milk, eggs, coffee", {"save_note"}),
    ("Hi", set()),
    ("What is the capital of France?", set()),
]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1, help="run each case this many times")
    parser.add_argument("--only", default="", help="only cases whose question contains this text")
    args = parser.parse_args()

    settings = get_settings()
    provider, store = get_provider(), get_store()
    if store.count() == 0:  # as the app does on startup
        await ingest_dir(settings.notes_dir, store, provider)
    agent = Agent(provider, build_registry(provider, store), SessionStore())

    cases = [c for c in CASES if args.only.lower() in c[0].lower()]
    print(f"prompt {SYSTEM_PROMPT_VERSION}, model {settings.generation_model}")
    passed = total = 0
    sessions = []
    for question, expected in cases:
        for _ in range(args.repeat):
            for attempt in (1, 2):
                session = f"eval-{uuid.uuid4().hex[:8]}"  # a fresh chat per question
                sessions.append(session)
                try:
                    [done] = [e async for e in agent.run_turn(session, question) if isinstance(e, AgentDone)]
                    used = set(done.tools_used)
                    ok = used == expected
                    got = ", ".join(sorted(used)) or "no tools"
                    break
                except QuotaExceededError as exc:
                    ok, got = False, f"QuotaExceededError (daily={exc.daily})"
                    if exc.daily or attempt == 2:
                        break
                    await asyncio.sleep(60)  # a per-minute limit: wait it out, then retry once
                except Exception as exc:  # noqa: BLE001 - one error shouldn't hide the other results
                    ok, got = False, f"{type(exc).__name__}: {str(exc)[:80]}"
                    break
            passed += ok
            total += 1
            want = ", ".join(sorted(expected)) or "no tools"
            print(f"{'PASS' if ok else 'FAIL'}  {question!r}: {got}" + ("" if ok else f" (expected {want})"))
            await asyncio.sleep(2)  # stay under the per-minute quota

    for session in sessions:
        store.delete_owner(session)  # notes the save_note case stored
    print(f"{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
