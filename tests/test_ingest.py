import httpx
import pytest

import app.main as main
from app.api.ratelimit import build_rate_limiters
from app.inference.provider import QuotaExceededError
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
    _write(notes, "b", "note b, edited")  # so b is embedded again, and that fails
    with pytest.raises(RuntimeError):
        await ingest_dir(notes, store, FailingProvider(fail_on="note b"))

    # the run died before pruning: nothing deleted, b's old chunks still there
    assert [d["doc_id"] for d in store.list_docs()] == ["a", "b"]


class CountingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(turns=[])
        self.embedded: list[str] = []

    async def embed(self, texts, kind):
        self.embedded.extend(texts)
        return await super().embed(texts, kind)


async def test_reingesting_only_embeds_new_and_changed_notes(tmp_path, store, monkeypatch):
    # POST /ingest is open to every visitor and used to embed every note on
    # every call, spending the shared quota for nothing.
    from app.config import get_settings

    notes = tmp_path / "notes"
    _write(notes, "same", "unchanged note")
    _write(notes, "edited", "first version")
    await ingest_dir(notes, store, FakeProvider(turns=[]))

    _write(notes, "edited", "second version")
    _write(notes, "new", "brand new note")
    provider = CountingProvider()
    counts = await ingest_dir(notes, store, provider)

    assert counts == {"edited": 1, "new": 1, "same": 1}
    assert provider.embedded == ["second version", "brand new note"]  # "same" skipped

    # A different embedding model makes every stored vector stale.
    monkeypatch.setattr(get_settings(), "embedding_model", "another-model")
    provider = CountingProvider()
    await ingest_dir(notes, store, provider)
    assert sorted(provider.embedded) == ["brand new note", "second version", "unchanged note"]


async def test_missing_notes_dir_does_not_wipe_the_store(tmp_path, store):
    notes = tmp_path / "notes"
    _write(notes, "a", "note a")
    await ingest_dir(notes, store, FakeProvider(turns=[]))

    assert await ingest_dir(tmp_path / "typo", store, FakeProvider(turns=[])) == {}
    assert store.count() == 1


async def test_reingest_during_a_quota_outage_is_a_503_with_a_reason(tmp_path, store, monkeypatch):
    # It used to be a bare 500, which the page showed as "HTTP 500".
    from app.config import get_settings

    class QuotaProvider(FakeProvider):
        async def embed(self, texts, kind):
            raise QuotaExceededError("429 RESOURCE_EXHAUSTED", daily=True)

    notes = tmp_path / "notes"
    _write(notes, "a", "note a")
    monkeypatch.setattr(get_settings(), "notes_dir", notes)
    for name, value in {"store": store, "provider": QuotaProvider(turns=[]),
                        "rate_limiters": build_rate_limiters(get_settings())}.items():
        monkeypatch.setattr(main.app.state, name, value, raising=False)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        r = await client.post("/ingest")

    assert r.status_code == 503 and "couldn't be re-indexed" in r.json()["detail"]


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
