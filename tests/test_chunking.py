from app.knowledge.chunking import chunk_text


def test_packs_paragraphs():
    text = "a" * 300 + "\n\n" + "b" * 300 + "\n\n" + "c" * 300
    chunks = chunk_text(text, max_chars=700, overlap=50)
    assert chunks == ["a" * 300 + "\n\n" + "b" * 300, "c" * 300]


def test_hard_splits_long_paragraph_with_overlap():
    chunks = chunk_text("x" * 1000, max_chars=400, overlap=100)
    assert [len(c) for c in chunks] == [400, 400, 400]
    assert chunks[0][-100:] == chunks[1][:100]


def test_empty():
    assert chunk_text("\n\n  \n") == []
