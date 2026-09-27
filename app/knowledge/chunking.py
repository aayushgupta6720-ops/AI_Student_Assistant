import re
from dataclasses import dataclass, field

_SEPARATOR = "\n\n"
_HEADING_RE = re.compile(r"(#{1,6})\s")
_FENCE_RE = re.compile(r" {0,3}(```|~~~)")


@dataclass
class Chunk:
    text: str
    # Markdown headings in effect where the chunk opens, outermost first,
    # minus any the chunk itself opens with. Lets a chunk that starts
    # mid-section still say which note and section it belongs to.
    headings: list[str] = field(default_factory=list)


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    return [chunk.text for chunk in chunk_markdown(text, max_chars, overlap)]


def chunk_markdown(text: str, max_chars: int = 800, overlap: int = 100) -> list[Chunk]:
    """Pack paragraphs into chunks up to max_chars, hard-splitting any single
    paragraph that alone exceeds the limit. A chunk that starts mid-section
    opens with up to `overlap` chars from the end of the one before it, cut at
    a word boundary, so text on either side of the cut keeps some context.
    Chunks starting a new section don't: the old section's tail is noise."""
    paragraphs = _attach_headings(_blocks(text))
    chunks: list[Chunk] = []
    current: Chunk | None = None
    path: list[str] = []  # headings in effect, outermost first

    for paragraph in paragraphs:
        if current:
            candidate = f"{current.text}{_SEPARATOR}{paragraph}"
            if len(candidate) <= max_chars:
                current.text = candidate
                path = _enter(path, paragraph)
                continue
            chunks.append(current)
            current = None

        # paragraph opens a new chunk, whether after a full one or a hard split
        opening_context = _context(path, paragraph)
        path = _enter(path, paragraph)
        if chunks and not _heading_level(paragraph):
            paragraph = _with_overlap(chunks[-1].text, paragraph, max_chars, overlap)

        if len(paragraph) <= max_chars:
            current = Chunk(paragraph, opening_context)
        else:
            step = max_chars - overlap
            for i in range(0, len(paragraph), step):
                # later windows no longer contain the paragraph's own heading
                context = opening_context if i == 0 else list(path)
                chunks.append(Chunk(paragraph[i : i + max_chars], context))
                if i + max_chars >= len(paragraph):
                    break  # the tail is already covered; don't emit a redundant sliver

    if current:
        chunks.append(current)

    return chunks


def _blocks(text: str) -> list[str]:
    """The note's blocks: runs of lines between blank lines, except that a
    heading line is a block of its own even with no blank line before it (as
    markdown allows after a list), and a fenced code block stays whole, so
    its blank lines don't split it and a "# comment" in it isn't a heading.
    Splitting on blank lines alone labelled every chunk after a commented
    code block with that comment instead of the note's own headings."""
    blocks: list[list[str]] = []
    current: list[str] = []
    fence: str | None = None  # the marker of the code block we're in, if any
    for line in text.split("\n"):
        marker = _FENCE_RE.match(line)
        if fence:
            current.append(line)
            if marker and marker.group(1) == fence:
                fence = None
        elif marker:
            fence = marker.group(1)
            current.append(line)
        elif not line.strip():
            blocks.append(current)
            current = []
        elif _heading_level(line):
            blocks.extend([current, [line]])
            current = []
        else:
            current.append(line)
    blocks.append(current)
    return [block for block in ("\n".join(lines).strip() for lines in blocks) if block]


def _heading_level(paragraph: str) -> int:
    """1 for a paragraph opening with "# ", 2 for "## " and so on; 0 if it
    doesn't open with a heading."""
    match = _HEADING_RE.match(paragraph)
    return len(match.group(1)) if match else 0


def _attach_headings(paragraphs: list[str]) -> list[str]:
    """Glue each heading that stands alone as a paragraph to the paragraph
    after it, so a chunk can't end on a heading whose section starts in the
    next chunk."""
    units: list[str] = []
    pending: list[str] = []
    for paragraph in paragraphs:
        pending.append(paragraph)
        if not (_heading_level(paragraph) and "\n" not in paragraph):
            units.append(_SEPARATOR.join(pending))
            pending = []
    if pending:  # headings at the very end, with no body after them
        units.append(_SEPARATOR.join(pending))
    return units


def _context(path: list[str], paragraph: str) -> list[str]:
    level = _heading_level(paragraph)
    if not level:
        return list(path)
    # the paragraph's own heading replaces any at its level or deeper
    return [h for h in path if _heading_level(h) < level]


def _enter(path: list[str], paragraph: str) -> list[str]:
    """The heading path after reading paragraph, which may open with several
    headings once _attach_headings has glued them to their body. Only those
    leading ones count: a "#" line further in is body text, e.g. code."""
    for part in paragraph.split(_SEPARATOR):
        if not _heading_level(part):
            break
        path = _context(path, part) + [part.split("\n", 1)[0]]
    return path


def _with_overlap(previous: str, paragraph: str, max_chars: int, overlap: int) -> str:
    """paragraph, opened with the tail of the previous chunk. A paragraph that
    fits in a chunk on its own only gets as much overlap as still fits, rather
    than being pushed into a hard split."""
    limit = overlap
    if len(paragraph) <= max_chars:
        limit = min(overlap, max_chars - len(paragraph) - len(_SEPARATOR))
    tail = _tail_at_word_boundary(previous, limit)
    return f"{tail}{_SEPARATOR}{paragraph}" if tail else paragraph


def _tail_at_word_boundary(text: str, limit: int) -> str:
    """The longest suffix of text, at most limit chars, that starts at a word
    boundary. Empty if there's none, e.g. text with no whitespace near its end."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    start = len(text) - limit
    if not text[start - 1].isspace():
        boundary = re.search(r"\s", text[start:])
        if not boundary:
            return ""
        start += boundary.end()
    return text[start:].strip()
