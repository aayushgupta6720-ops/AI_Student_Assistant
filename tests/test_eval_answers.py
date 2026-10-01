import re
import sys

import pytest

from app.config import get_settings
from app.knowledge.ingest import slugify
from scripts import eval_answers
from scripts.eval_answers import CATEGORIES, EVAL_DIR, REFUSAL, case_report, evaluate, load_cases, summary
from tests.fake_provider import FakeProvider, text_turn, tool_turn

CASES = load_cases()
NOTES = {slugify(p.stem): p.read_text() for p in (EVAL_DIR / "notes").glob("*.md")}


def test_cases_are_well_formed():
    assert len({c.id for c in CASES}) == len(CASES)
    for case in CASES:
        assert case.category in CATEGORIES, case.id
        # A not_in_notes case is scored on the refusal alone; every other one needs facts to check.
        assert (case.category == "not_in_notes") == (not case.docs and not case.facts), case.id
        assert set(case.docs) <= set(NOTES), case.id


@pytest.mark.parametrize("case", [c for c in CASES if c.facts], ids=lambda c: c.id)
def test_every_fact_is_in_the_notes_the_case_names(case):
    # Otherwise the case could never pass, and the eval would blame the model.
    text = eval_answers.normalise("\n".join(NOTES[d] for d in case.docs))
    missing = [f for f in case.facts if not re.search(f, text, re.IGNORECASE)]
    assert not missing


@pytest.mark.parametrize("answer", [
    "Your notes don't mention the French Revolution.",
    "I couldn't find anything about the Doppler effect in your formula sheet.",
    "There's no information on the nervous system in your biology notes.",
    "Your formula sheet doesn't cover the Doppler effect.",
])
def test_refusals_are_recognised(answer):
    assert REFUSAL.search(answer)


@pytest.mark.parametrize("answer", [
    "The Calvin cycle happens in the stroma of the chloroplast.",
    "Mitosis has four phases: prophase, metaphase, anaphase and telophase.",
])
def test_answers_are_not_taken_for_refusals(answer):
    assert not REFUSAL.search(answer)


async def test_a_run_scores_answers_and_refusals_offline():
    cases = {c.id: c for c in CASES}
    provider = FakeProvider([
        tool_turn("search_notes", {"query": "Calvin cycle"}), text_turn("It happens in the **stroma**."),
        tool_turn("search_notes", {"query": "Doppler"}), text_turn("Your formula sheet doesn't mention the Doppler effect."),
    ])

    results = await evaluate(provider, [cases["bio-calvin-location"], cases["none-doppler"]], pause_s=0)

    assert [(r.passed, r.error) for r in results] == [(True, None), (True, None)]
    assert isinstance(results[0].search_hit, bool) and results[1].search_hit is None
    assert results[0].answer_hits == (True,)
    assert results[0].tools == ("search_notes",)
    assert "PASS" in case_report(results[0])
    assert re.search(r"^all\s+2/2", summary(results), re.MULTILINE)


def test_it_refuses_to_spend_the_demo_key_unless_told_to(monkeypatch, capsys):
    monkeypatch.setattr(get_settings(), "eval_gemini_api_key", "")
    monkeypatch.setattr(sys, "argv", ["eval_answers"])

    assert eval_answers.main() == 2
    assert "EVAL_GEMINI_API_KEY" in capsys.readouterr().out
