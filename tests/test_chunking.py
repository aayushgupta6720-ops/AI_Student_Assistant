from app.knowledge.chunking import chunk_markdown, chunk_text


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


# ---- overlap between chunks --------------------------------------------------

TEN_WORDS = "one two three four five six seven eight nine ten"  # 48 chars


def test_new_chunk_opens_with_the_previous_chunks_tail_cut_at_a_word_boundary():
    chunks = chunk_text(f"{TEN_WORDS}\n\neleven twelve", max_chars=50, overlap=10)

    # the last 10 chars are "t nine ten"; the partial "t" is dropped
    assert chunks == [TEN_WORDS, "nine ten\n\neleven twelve"]


def test_overlap_shrinks_rather_than_hard_splitting_a_paragraph_that_fits_alone():
    roomy, tight = "z" * 44, "z" * 47

    # 44 + separator leaves room for 4 chars of overlap ("ten"), 47 for none
    assert chunk_text(f"{TEN_WORDS}\n\n{roomy}", 50, 10) == [TEN_WORDS, f"ten\n\n{roomy}"]
    assert chunk_text(f"{TEN_WORDS}\n\n{tight}", 50, 10) == [TEN_WORDS, tight]


def test_paragraph_after_a_hard_split_overlaps_only_the_last_window():
    chunks = chunk_text(f"{'word ' * 14}end\n\ntail", max_chars=50, overlap=10)

    assert chunks[-1] == "word end\n\ntail"


# ---- markdown headings ---------------------------------------------------------

NOTE = (
    "# Trip\n\nIntro line.\n\n"
    "## Packing\n\n" + " ".join(["sock"] * 12) + "\n\n" + " ".join(["shirt"] * 8) + "\n\n"
    "## Day of travel\n\nLeave early."
)


def test_chunks_carry_the_headings_they_sit_under():
    chunks = chunk_markdown(NOTE, max_chars=80, overlap=15)

    assert [(c.headings, c.text.split("\n", 1)[0]) for c in chunks] == [
        ([], "# Trip"),  # opens with its own heading: nothing to add
        (["# Trip"], "## Packing"),
        (["# Trip", "## Packing"], "sock sock sock"),  # starts mid-section
        (["# Trip"], "## Day of travel"),
    ]


def test_a_chunk_never_ends_on_a_heading_whose_body_went_to_the_next_chunk():
    for chunk in chunk_markdown(NOTE, max_chars=80, overlap=15):
        assert not chunk.text.rsplit("\n\n", 1)[-1].startswith("#")


def test_chunks_that_open_a_new_section_get_no_overlap():
    chunks = chunk_markdown(NOTE, max_chars=80, overlap=15)

    # mid-section chunk: overlaps; new-section chunks: start at their heading
    assert chunks[2].text.startswith("sock sock sock\n\nshirt")
    assert chunks[3].text == "## Day of travel\n\nLeave early."


def test_a_heading_closes_deeper_sections_but_keeps_its_parents():
    x, more = " ".join(["x"] * 30), " ".join(["more"] * 14)  # 59 and 69 chars
    note = f"# T\n\n## A\n\n### A1\n\n{x}\n\n{more}\n\n## B\n\nlast"

    chunks = chunk_markdown(note, max_chars=80, overlap=0)

    assert [c.headings for c in chunks] == [[], ["# T", "## A", "### A1"], ["# T"]]


def test_hard_split_windows_after_the_first_carry_the_paragraphs_own_heading():
    chunks = chunk_markdown("# T\n\n## Long\n\n" + "w " * 60, max_chars=50, overlap=10)

    assert chunks[0].headings == []
    assert all(c.headings == ["# T", "## Long"] for c in chunks[1:])
