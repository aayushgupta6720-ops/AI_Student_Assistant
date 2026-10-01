"""Measure how well the assistant answers from uploaded study notes, against
the real Gemini API. The notes in data/eval/notes are uploaded the way a
visitor uploads them (as one session's private notes, alongside the shared
notes), and each case in data/eval/cases.json is asked in a fresh chat.

    python -m scripts.eval_answers                     # every case
    python -m scripts.eval_answers --only whole_doc    # a category, or case ids

A run spends about 60 chat and 60 embedding requests, so it uses its own key:
set EVAL_GEMINI_API_KEY (in .env or the environment) to a key from another
Google project, and the live demo keeps its 500 requests a day.

Scores, per case:
  facts    how many of the case's facts (regexes) the answer contains; a
           case passes when it has all of them, or for a not_in_notes case,
           when the answer says the notes don't cover it
  search   whether one search_notes-style retrieval of the question returns
           the expected note, and how many of the facts its passages hold:
           what a single search can give the model to work with
  sourced  whether the turn's searches returned the expected note at all
  searches how many times the model called search_notes (0 means it
           answered from its own knowledge, not the notes)
"""

import argparse
import asyncio
import json
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.config import PROJECT_ROOT, get_settings
from app.inference.gemini import get_provider
from app.inference.provider import LLMProvider, QuotaExceededError
from app.intelligence.agent import Agent, AgentDone
from app.intelligence.memory import SessionStore
from app.intelligence.prompts import SYSTEM_PROMPT_VERSION
from app.knowledge.ingest import ingest_dir
from app.knowledge.retrieval import current_session, retrieve
from app.knowledge.store import VectorStore
from app.knowledge.uploads import ingest_upload
from app.tools.builtin import build_registry

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
CATEGORIES = ("single_fact", "exact_term", "whole_doc", "not_in_notes")
# How an answer says the notes don't cover something: "Your notes don't
# mention...", "I couldn't find anything about...", "There's no information on...".
REFUSAL = re.compile(
    r"(\bnot\b|\bno\b|n't\b|\bnothing\b|\bnone\b)[^.\n]{0,80}?"
    r"\b(mention\w*|cover\w*|contain\w*|includ\w*|find|found|information|anything|say|says|discuss\w*|about)\b",
    re.IGNORECASE,
)


@dataclass
class Case:
    id: str
    category: str
    question: str
    docs: list[str]
    facts: list[str]


@dataclass
class Result:
    case: Case
    answer: str = ""
    sources: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()  # every tool call the turn made
    answer_hits: tuple[bool, ...] = ()
    search_hit: bool | None = None  # None for not_in_notes: there's no right note
    context_hits: tuple[bool, ...] = ()
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if self.case.category == "not_in_notes":
            return bool(REFUSAL.search(normalise(self.answer)))
        return all(self.answer_hits)

    @property
    def sourced(self) -> bool | None:
        return None if not self.case.docs else set(self.case.docs) <= set(self.sources)


def load_cases(path: Path = EVAL_DIR / "cases.json") -> list[Case]:
    return [Case(**c) for c in json.loads(path.read_text())]


def normalise(text: str) -> str:
    """Drop Markdown emphasis and collapse whitespace, so "**Stroma**" or a
    fact split across lines still matches."""
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", text))


def hits(facts: list[str], text: str) -> tuple[bool, ...]:
    text = normalise(text)
    return tuple(re.search(f, text, re.IGNORECASE) is not None for f in facts)


async def _run_case(agent: Agent, provider: LLMProvider, store: VectorStore, session: str, case: Case) -> Result:
    result = Result(case)
    if case.docs:
        current_session.set(session)  # the probe sees this session's uploads, as the tool does
        chunks = await retrieve(case.question, provider, store)
        result.search_hit = any(c.doc_id in case.docs for c in chunks)
        result.context_hits = hits(case.facts, "\n".join(c.text for c in chunks))
    agent.memory.reset(session)  # a fresh chat; the uploads stay
    [done] = [e async for e in agent.run_turn(session, case.question) if isinstance(e, AgentDone)]
    result.answer = done.answer
    result.sources = tuple(done.sources)
    result.tools = tuple(done.tools_used)
    result.answer_hits = hits(case.facts, done.answer)
    return result


async def evaluate(
    provider: LLMProvider,
    cases: list[Case],
    pause_s: float = 4.0,
    on_result: Callable[[Result], None] | None = None,
) -> list[Result]:
    """Upload the eval notes into a fresh in-memory index (with the shared
    notes, which compete in every search as they do live) and run each case."""
    store = VectorStore(":memory:")
    await ingest_dir(get_settings().notes_dir, store, provider)
    session = f"eval-{uuid.uuid4().hex[:8]}"
    for path in sorted((EVAL_DIR / "notes").glob("*.md")):
        await ingest_upload(session, path.name, path.read_bytes(), store, provider)
    agent = Agent(provider, build_registry(provider, store), SessionStore())

    results = []
    for case in cases:
        for attempt in (1, 2):
            try:
                result = await _run_case(agent, provider, store, session, case)
                break
            except QuotaExceededError as exc:
                result = Result(case, error=f"QuotaExceededError (daily={exc.daily})")
                if exc.daily or attempt == 2:
                    break
                await asyncio.sleep(60)  # a per-minute limit: wait it out, then retry once
            except Exception as exc:  # noqa: BLE001 - one error shouldn't hide the other results
                result = Result(case, error=f"{type(exc).__name__}: {str(exc)[:120]}")
                break
        results.append(result)
        if on_result:
            on_result(result)
        await asyncio.sleep(pause_s)  # stay under the per-minute quota
    return results


def _ratio(flags: tuple[bool, ...]) -> str:
    return f"{sum(flags)}/{len(flags)}"


def case_report(r: Result) -> str:
    c = r.case
    line = f"{'PASS' if r.passed else 'FAIL'}  {c.id:<24} {c.category:<12}"
    if r.error:
        return f"{line} ERROR {r.error}"
    if c.docs:
        search = "hit" if r.search_hit else "miss"
        line += (f" facts {_ratio(r.answer_hits):<5} search {search}, {_ratio(r.context_hits)} facts"
                 f"  sourced {'yes' if r.sourced else 'no'}")
    line += f"  searches {r.tools.count('search_notes')}"
    lines = [line]
    if not r.passed:
        missing = [f for f, ok in zip(c.facts, r.answer_hits) if not ok]
        if missing:
            lines.append(f"      missing: {', '.join(missing)}")
        lines.append(f"      answer: {normalise(r.answer)[:300]}")
    return "\n".join(lines)


def summary(results: list[Result]) -> str:
    lines = [f"{'category':<14}{'passed':>8}{'facts':>8}{'search hit':>12}{'search facts':>14}"]
    for category in (*CATEGORIES, "all"):
        group = [r for r in results if category in (r.case.category, "all")]
        if not group:
            continue
        answered = [r for r in group if r.case.facts and not r.error]
        searched = [r for r in answered if r.search_hit is not None]
        facts = [h for r in answered for h in r.answer_hits]
        context = [h for r in searched for h in r.context_hits]
        lines.append(
            f"{category:<14}{sum(r.passed for r in group):>4}/{len(group):<3}"
            f"{_pct(facts):>8}{_ratio(tuple(r.search_hit for r in searched)) if searched else '-':>12}"
            f"{_pct(context):>14}"
        )
    return "\n".join(lines)


def _pct(flags: list[bool]) -> str:
    return f"{sum(flags) / len(flags):.0%}" if flags else "-"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=[], help="case ids or categories to run")
    parser.add_argument("--allow-demo-key", action="store_true",
                        help="run on GEMINI_API_KEY, spending the live demo's quota")
    args = parser.parse_args()

    settings = get_settings()
    if settings.eval_gemini_api_key:
        settings.gemini_api_key = settings.eval_gemini_api_key  # before the client is first built
    elif not args.allow_demo_key:
        print("Set EVAL_GEMINI_API_KEY to a key from another Google project (or pass --allow-demo-key).")
        return 2

    cases = [c for c in load_cases() if not args.only or {c.id, c.category} & set(args.only)]
    print(f"prompt {SYSTEM_PROMPT_VERSION}, model {settings.generation_model}, {len(cases)} cases\n")
    results = asyncio.run(evaluate(get_provider(), cases, on_result=lambda r: print(case_report(r), flush=True)))
    print("\n" + summary(results))
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
