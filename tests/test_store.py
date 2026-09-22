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
    assert store.list_docs() == [{"doc_id": "a", "chunks": 1}, {"doc_id": "b", "chunks": 1}]

    store.delete_doc("b")
    assert [h.text for h in store.search([0, 0, 1], k=5)] == ["a-new"]


def test_empty_store(tmp_path):
    assert VectorStore(tmp_path / "e.sqlite").search([1, 2, 3], k=3) == []
