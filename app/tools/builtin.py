"""The five built-in tools. Note which layers each one touches:

  search_notes     -> knowledge layer (which itself calls inference for embeddings)
  save_note        -> filesystem + knowledge layer (re-ingests the new note)
  calculator       -> pure computation, no other layer
  current_datetime -> pure computation
  fetch_url        -> external I/O
"""

import ast
import operator
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.config import get_settings
from app.inference.provider import LLMProvider
from app.knowledge.ingest import ingest_file, slugify
from app.knowledge.retrieval import get_store, retrieve
from app.knowledge.store import VectorStore
from app.tools.registry import Tool, ToolRegistry

# ---- calculator: a safe arithmetic evaluator (no eval()) --------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> dict:
    cleaned = expression.replace(",", "").replace("^", "**").replace("%", "/100")
    tree = ast.parse(cleaned, mode="eval")
    return {"expression": expression, "result": _eval_node(tree)}


def current_datetime() -> dict:
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "weekday": now.strftime("%A"),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)


async def fetch_url(url: str, max_chars: int = 4000) -> dict:
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        response = await client.get(url, headers={"User-Agent": "AI_Student_Assistant/0.1"})
    response.raise_for_status()
    text = _TAG_RE.sub(" ", response.text)
    text = re.sub(r"\s+", " ", text).strip()
    return {"url": str(response.url), "status": response.status_code, "text": text[:max_chars]}


def _saved_title(path: Path) -> str:
    first_line = path.read_text(encoding="utf-8").split("\n", 1)[0]
    return first_line.lstrip("#").strip()


def _note_path(notes_dir: Path, title: str) -> Path:
    """The file for a note titled `title`. Saving under an existing title
    overwrites that note; a different title that slugifies the same ("C notes"
    vs "C++ notes") gets the next free "-2", "-3"... suffix instead."""
    slug = slugify(title)
    n = 1
    while True:
        path = notes_dir / (f"{slug}.md" if n == 1 else f"{slug}-{n}.md")
        if not path.exists() or _saved_title(path).casefold() == title.strip().casefold():
            return path
        n += 1


def build_registry(provider: LLMProvider, store: VectorStore | None = None) -> ToolRegistry:
    """Wire tools to the provider/store they need. Keeping this a factory (rather
    than module-level globals) is what lets tests inject a FakeProvider."""
    settings = get_settings()
    store = store or get_store()
    registry = ToolRegistry()

    async def search_notes(query: str, top_k: int = settings.retrieval_top_k) -> dict:
        chunks = await retrieve(query, provider, store, k=top_k)
        return {
            "query": query,
            "results": [
                {"doc_id": c.doc_id, "score": c.score, "text": c.text} for c in chunks
            ],
        }

    async def save_note(title: str, content: str) -> dict:
        settings.notes_dir.mkdir(parents=True, exist_ok=True)
        path = _note_path(settings.notes_dir, title)
        path.write_text(f"# {title}\n\n{content.strip()}\n", encoding="utf-8")
        chunks = await ingest_file(path, store, provider)
        return {"doc_id": path.stem, "path": str(path), "chunks_indexed": chunks}

    registry.register(
        Tool(
            "search_notes",
            "Semantic search over the user's personal notes. Use this whenever the "
            "user asks about their notes, plans, lists, recipes, or anything they "
            "may have written down. Returns the most relevant passages with doc_ids.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "top_k": {"type": "integer", "description": "Number of passages (default 4)."},
                },
                "required": ["query"],
            },
            search_notes,
        )
    )
    registry.register(
        Tool(
            "save_note",
            "Create a note in the user's notes folder, or overwrite the one with the "
            "same title, and index it so it becomes searchable. Use when the user "
            "asks to save, remember, or write down something.",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title; becomes the filename."},
                    "content": {"type": "string", "description": "Markdown body of the note."},
                },
                "required": ["title", "content"],
            },
            save_note,
        )
    )
    registry.register(
        Tool(
            "calculator",
            "Evaluate an arithmetic expression exactly (+ - * / ** % and parentheses). "
            "Use for any non-trivial arithmetic instead of computing in your head.",
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '0.17 * 2340' or '(3+4)**2'"}
                },
                "required": ["expression"],
            },
            calculator,
        )
    )
    registry.register(
        Tool(
            "current_datetime",
            "Get the current local date, time, and weekday. Use for anything involving "
            "'today', 'now', deadlines, or elapsed time.",
            {"type": "object", "properties": {}},
            current_datetime,
        )
    )
    registry.register(
        Tool(
            "fetch_url",
            "Fetch a web page and return its visible text (truncated). Use when the "
            "user gives a URL or asks about the contents of a specific page.",
            {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Absolute http(s) URL."}},
                "required": ["url"],
            },
            fetch_url,
        )
    )
    return registry
