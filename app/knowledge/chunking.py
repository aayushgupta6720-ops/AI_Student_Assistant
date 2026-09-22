def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Pack paragraphs into chunks up to max_chars, hard-splitting any single
    paragraph that alone exceeds the limit."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            step = max_chars - overlap
            for i in range(0, len(paragraph), step):
                chunks.append(paragraph[i : i + max_chars])
                if i + max_chars >= len(paragraph):
                    break  # the tail is already covered; don't emit a redundant sliver

    if current:
        chunks.append(current)

    return chunks
