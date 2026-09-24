import sqlite3
import time

import pytest

from app.knowledge.store import VectorStore


def test_upsert_search_delete(tmp_path):
    store = VectorStore(tmp_path / "s.sqlite")
    store.upsert_doc("a", ["a0", "a1"], [[1, 0, 0], [0, 1, 0]])
    store.upsert_doc("b", ["b0"], [[0, 0, 1]])
    assert store.count() == 3

    hits = store.search([0.9, 0.1, 0], k=2)
    assert [h.text for h in hits] == ["a0", "a1"]
    assert hits[0].score > hits[1].score

    store.upsert_doc("a", ["a-new"], [[0, 0, 1]])  # replaces both old chunks
    assert store.count() == 2
    assert store.list_docs() == [
        {"doc_id": "a", "chunks": 1, "uploaded": False},
        {"doc_id": "b", "chunks": 1, "uploaded": False},
    ]

    store.delete_doc("b")
    assert [h.text for h in store.search([0, 0, 1], k=5)] == ["a-new"]


def test_empty_store(tmp_path):
    assert VectorStore(tmp_path / "e.sqlite").search([1, 2, 3], k=3) == []


@pytest.fixture
def store(tmp_path):
    """A shared note plus one upload each for sessions alice and bob."""
    s = VectorStore(tmp_path / "s.sqlite")
    s.upsert_doc("shared", ["shared note"], [[1, 0, 0]])
    s.upsert_doc("lecture", ["alice's lecture"], [[1, 0.1, 0]], owner="alice")
    s.upsert_doc("lecture", ["bob's lecture"], [[1, 0.2, 0]], owner="bob")
    return s


def test_a_search_sees_shared_notes_and_only_its_own_uploads(store):
    assert {h.text for h in store.search([1, 0, 0], k=10, owner="alice")} == {"shared note", "alice's lecture"}
    assert {h.text for h in store.search([1, 0, 0], k=10, owner="bob")} == {"shared note", "bob's lecture"}
    assert {h.text for h in store.search([1, 0, 0], k=10)} == {"shared note"}


def test_list_and_count_are_scoped_the_same_way(store):
    assert store.list_docs(owner="alice") == [
        {"doc_id": "shared", "chunks": 1, "uploaded": False},
        {"doc_id": "lecture", "chunks": 1, "uploaded": True},
    ]
    assert store.list_docs() == [{"doc_id": "shared", "chunks": 1, "uploaded": False}]
    assert store.count() == 1 and store.count(owner="alice") == 2


def test_a_session_can_only_delete_its_own_uploads(store):
    assert not store.delete_doc("shared", owner="alice")  # not alice's
    assert store.delete_doc("lecture", owner="alice")
    assert {h.text for h in store.search([1, 0, 0], k=10, owner="bob")} == {"shared note", "bob's lecture"}


def test_delete_owner_removes_one_sessions_uploads_and_refuses_the_shared_notes(store):
    assert store.delete_owner("alice") == 1
    assert store.list_docs(owner="alice") == [{"doc_id": "shared", "chunks": 1, "uploaded": False}]
    with pytest.raises(ValueError):
        store.delete_owner("")


def test_purge_removes_old_uploads_but_never_shared_notes(store, monkeypatch):
    later = time.time() + 25 * 3600
    monkeypatch.setattr(time, "time", lambda: later)

    assert store.purge_uploads(older_than_s=24 * 3600) == 2
    assert store.list_docs(owner="alice") == [{"doc_id": "shared", "chunks": 1, "uploaded": False}]


def test_a_store_in_the_old_layout_is_rebuilt_empty(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE chunks (doc_id TEXT, chunk_index INTEGER, text TEXT, embedding BLOB)")
    conn.execute("INSERT INTO chunks VALUES ('a', 0, 'old', x'00')")
    conn.commit()
    conn.close()

    store = VectorStore(path)

    # empty, so startup re-ingests data/notes into the new layout
    assert store.count() == 0
    store.upsert_doc("a", ["new"], [[1.0]], owner="s")
    assert store.list_docs(owner="s") == [{"doc_id": "a", "chunks": 1, "uploaded": True}]
