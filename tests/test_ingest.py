import pytest

import app.main as main
from app.knowledge.ingest import ingest_dir, ingest_file
from app.knowledge.store import VectorStore
from tests.fake_provider import FakeProvider


class FailingProvider(FakeProvider):
    """Embeds normally until `fail_on` shows up in a batch, then raises."""

    def __init__(self, fail_on: str) -> None:
        super().__init__(turns=[])
        self.fail_on = fail_on

    async def embed(self, texts, kind):
        if any(self.fail_on in t for t in texts):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return await super().embed(texts, kind)


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "s.sqlite")


def _write(notes_dir, name, text):
    notes_dir.mkdir(exist_ok=True)
    (notes_dir / f"{name}.md").write_text(text)


async def test_chunks_are_stored_under_their_note_and_section_headings(tmp_path, store, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "chunk_max_chars", 40)
    _write(tmp_path, "trip", "# Trip\n\n## Packing\n\nsocks and shirts\n\nmore socks and a hat")

    await ingest_file(tmp_path / "trip.md", store, FakeProvider(turns=[]))

    texts = [h.text for h in store.search([1, 1, 1, 1], k=5)]
    assert "# Trip\n## Packing\n\nsocks and shirts\n\nmore socks and a hat" in texts


async def test_ingest_dir_drops_notes_deleted_from_the_folder(tmp_path, store):
    notes = tmp_path / "notes"
    _write(notes, "keep", "kept note")
    _write(notes, "gone", "deleted note")
    await ingest_dir(notes, store, FakeProvider(turns=[]))

    (notes / "gone.md").unlink()
    await ingest_dir(notes, store, FakeProvider(turns=[]))

    assert [d["doc_id"] for d in store.list_docs()] == ["keep"]


async def test_failure_partway_through_prunes_nothing(tmp_path, store):
    notes = tmp_path / "notes"
    _write(notes, "a", "note a")
    _write(notes, "b", "note b")
    await ingest_dir(notes, store, FakeProvider(turns=[]))

    (notes / "a.md").unlink()
    with pytest.raises(RuntimeError):
        await ingest_dir(notes, store, FailingProvider(fail_on="note b"))

    # the run died before pruning: nothing deleted, b's old chunks still there
    assert [d["doc_id"] for d in store.list_docs()] == ["a", "b"]


async def test_missing_notes_dir_does_not_wipe_the_store(tmp_path, store):
    notes = tmp_path / "notes"
    _write(notes, "a", "note a")
    await ingest_dir(notes, store, FakeProvider(turns=[]))

    assert await ingest_dir(tmp_path / "typo", store, FakeProvider(turns=[])) == {}
    assert store.count() == 1


async def test_app_still_boots_when_the_startup_ingest_fails(tmp_path, store, monkeypatch):
    from app.config import get_settings

    notes = tmp_path / "notes"
    _write(notes, "a", "note a")
    monkeypatch.setattr(get_settings(), "notes_dir", notes)
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr(main, "get_provider", lambda: FailingProvider(fail_on="note"))

    async with main.lifespan(main.app):
        assert main.app.state.store.count() == 0
        assert main.app.state.agent is not None
