import pytest

from app.knowledge.store import VectorStore
from app.tools.builtin import build_registry, calculator
from app.tools.registry import Tool, ToolRegistry
from tests.fake_provider import FakeProvider


def test_calculator():
    assert calculator("17% * 2,340")["result"] == pytest.approx(397.8)
    assert calculator("2^10")["result"] == 1024
    with pytest.raises(ValueError):
        calculator("__import__('os').system('ls')")


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

    settings = get_settings()
    monkeypatch.setattr(settings, "notes_dir", tmp_path)
    provider = FakeProvider(turns=[])
    store = VectorStore(tmp_path / "s.sqlite")
    reg = build_registry(provider, store)
    assert set(reg.names()) == {"search_notes", "save_note", "calculator", "current_datetime", "fetch_url"}

    saved = await reg.execute("save_note", {"title": "Groceries", "content": "milk, eggs"})
    assert not saved.is_error and saved.result["doc_id"] == "groceries"
    assert (tmp_path / "groceries.md").read_text().startswith("# Groceries")

    found = await reg.execute("search_notes", {"query": "milk eggs"})
    assert found.result["results"][0]["doc_id"] == "groceries"


async def test_save_note_titles_that_slugify_alike_get_separate_files(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "notes_dir", tmp_path)
    reg = build_registry(FakeProvider(turns=[]), VectorStore(tmp_path / "s.sqlite"))

    first = await reg.execute("save_note", {"title": "C notes", "content": "pointers"})
    second = await reg.execute("save_note", {"title": "C++ notes", "content": "templates"})
    again = await reg.execute("save_note", {"title": "c notes", "content": "pointers v2"})

    assert [r.result["doc_id"] for r in (first, second, again)] == ["c-notes", "c-notes-2", "c-notes"]
    assert "templates" in (tmp_path / "c-notes-2.md").read_text()  # not overwritten
    assert "pointers v2" in (tmp_path / "c-notes.md").read_text()  # same title: updated
