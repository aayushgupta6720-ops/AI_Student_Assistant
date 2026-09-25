import time

import pytest

from app.knowledge.retrieval import current_session
from app.knowledge.store import VectorStore
from app.tools.builtin import build_registry, calculator, current_datetime, user_timezone
from app.tools.registry import Tool, ToolRegistry
from tests.fake_provider import FakeProvider


def test_calculator():
    assert calculator("17% * 2,340")["result"] == pytest.approx(397.8)
    assert calculator("2^10")["result"] == 1024
    with pytest.raises(ValueError):
        calculator("__import__('os').system('ls')")


def test_calculator_tells_percent_from_modulo():
    assert calculator("50%")["result"] == 0.5
    assert calculator("10 % 3")["result"] == 1
    assert calculator("10 % -3")["result"] == -2


@pytest.mark.parametrize("expression", ["9**9**9", "2**(10**400)", "(10**1000) * (10**1000)"])
def test_calculator_refuses_huge_results_instead_of_stalling(expression):
    # 9**9**9 used to run for minutes on the event loop, stalling every chat
    started = time.perf_counter()
    with pytest.raises(ValueError, match="too large"):
        calculator(expression)
    assert time.perf_counter() - started < 0.1


async def test_registry_specs_and_errors():
    reg = ToolRegistry()
    reg.register(Tool("boom", "explodes", {"type": "object", "properties": {}}, lambda: 1 / 0))
    assert [s.name for s in reg.specs()] == ["boom"]
    out = await reg.execute("boom", {})
    assert out.is_error and "ZeroDivisionError" in out.result["error"]
    out = await reg.execute("nope", {})
    assert out.is_error and "unknown tool" in out.result["error"]


async def test_builtin_registry_search_and_save(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "notes_dir", tmp_path)
    provider = FakeProvider(turns=[])
    store = VectorStore(tmp_path / "s.sqlite")
    reg = build_registry(provider, store)
    assert set(reg.names()) == {"search_notes", "save_note", "calculator", "current_datetime", "fetch_url"}

    current_session.set("alice")
    saved = await reg.execute("save_note", {"title": "Groceries", "content": "milk, eggs"})
    assert not saved.is_error and saved.result["doc_id"] == "groceries"

    found = await reg.execute("search_notes", {"query": "milk eggs"})
    assert found.result["results"][0]["doc_id"] == "groceries"


async def test_saved_notes_are_private_to_the_session_and_never_touch_shared_notes(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "notes_dir", tmp_path)
    (tmp_path / "reading-list.md").write_text("# Reading list\n\nDune")
    store = VectorStore(tmp_path / "s.sqlite")
    store.upsert_doc("reading-list", ["# Reading list\n\nDune"], [[1.0, 0, 0, 0]])
    reg = build_registry(FakeProvider(turns=[]), store)

    current_session.set("alice")
    await reg.execute("save_note", {"title": "My secret", "content": "PIN 4321"})
    await reg.execute("save_note", {"title": "Reading list", "content": "vandalised"})

    assert (tmp_path / "reading-list.md").read_text() == "# Reading list\n\nDune"  # no files written
    assert sorted(p.name for p in tmp_path.glob("*.md")) == ["reading-list.md"]
    current_session.set("bob")
    assert [d["doc_id"] for d in store.list_docs(owner="bob")] == ["reading-list"]
    hits = (await reg.execute("search_notes", {"query": "PIN reading", "top_k": 10})).result["results"]
    assert [(h["doc_id"], h["text"]) for h in hits] == [("reading-list", "# Reading list\n\nDune")]


async def test_save_note_needs_a_session(tmp_path):
    reg = build_registry(FakeProvider(turns=[]), VectorStore(tmp_path / "s.sqlite"))
    current_session.set(None)
    out = await reg.execute("save_note", {"title": "x", "content": "y"})
    assert out.is_error and "need a session" in out.result["error"]


async def test_save_note_titles_that_slugify_alike_get_separate_notes(tmp_path):
    store = VectorStore(tmp_path / "s.sqlite")
    reg = build_registry(FakeProvider(turns=[]), store)
    current_session.set("alice")

    first = await reg.execute("save_note", {"title": "C notes", "content": "pointers"})
    second = await reg.execute("save_note", {"title": "C++ notes", "content": "templates"})
    again = await reg.execute("save_note", {"title": "c notes", "content": "pointers v2"})

    assert [r.result["doc_id"] for r in (first, second, again)] == ["c-notes", "c-notes-2", "c-notes"]
    texts = {h.doc_id: h.text for h in store.search([1, 1, 1, 1], k=10, owner="alice")}
    assert "templates" in texts["c-notes-2"]  # not overwritten
    assert "pointers v2" in texts["c-notes"]  # same title: updated


@pytest.mark.parametrize(("zone", "offset"), [("Asia/Kolkata", "+05:30"), ("America/New_York", None)])
async def test_current_datetime_answers_in_the_users_time_zone(zone, offset):
    user_timezone.set(zone)
    now = current_datetime()
    assert now["timezone"] == zone
    if offset:
        assert now["iso"].endswith(offset)


@pytest.mark.parametrize("zone", [None, "Not/AZone", "../../etc/passwd"])
async def test_current_datetime_falls_back_to_the_server_zone(zone):
    user_timezone.set(zone)
    assert current_datetime()["iso"]  # no error; answers in the server's zone
