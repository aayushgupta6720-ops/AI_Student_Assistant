"""A deliberately small vector store: SQLite for durability, numpy for
cosine search. Fine for thousands of chunks. Swap this file for a Qdrant /
pgvector client and nothing else in the app changes.

Every chunk has an owner: "" for the shared notes in data/notes, or the
session id that uploaded it. A search sees the shared notes plus the
searching session's own uploads, never another session's."""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SHARED = ""
# Bumped when the table layout changes. The store only holds derived data
# (re-ingestable from data/notes), so an older layout is dropped rather than
# migrated, and startup re-ingests the now-empty store.
SCHEMA_VERSION = 2


@dataclass
class StoredChunk:
    doc_id: str
    chunk_index: int
    text: str
    score: float = 0.0


class VectorStore:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        if self._conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
            with self._conn:
                self._conn.execute("DROP TABLE IF EXISTS chunks")
                self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                   owner TEXT NOT NULL,
                   doc_id TEXT NOT NULL,
                   chunk_index INTEGER NOT NULL,
                   text TEXT NOT NULL,
                   embedding BLOB NOT NULL,
                   created_at REAL NOT NULL,
                   PRIMARY KEY (owner, doc_id, chunk_index)
               )"""
        )
        self._conn.commit()
        self._cache: tuple[np.ndarray, list[StoredChunk], list[str]] | None = None

    def upsert_doc(
        self, doc_id: str, texts: list[str], embeddings: list[list[float]], owner: str | None = None
    ) -> None:
        assert len(texts) == len(embeddings)
        owner = owner or SHARED
        now = time.time()
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE owner = ? AND doc_id = ?", (owner, doc_id))
            self._conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (owner, doc_id, i, text, np.asarray(vec, dtype=np.float32).tobytes(), now)
                    for i, (text, vec) in enumerate(zip(texts, embeddings))
                ],
            )
        self._cache = None

    def delete_doc(self, doc_id: str, owner: str | None = None) -> bool:
        """Delete one doc of `owner` (the shared notes when None). Returns
        whether anything was deleted."""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM chunks WHERE owner = ? AND doc_id = ?", (owner or SHARED, doc_id)
            )
        self._cache = None
        return cur.rowcount > 0

    def delete_owner(self, owner: str) -> int:
        """Delete every upload of one session. Returns how many chunks went."""
        if not owner:
            raise ValueError("refusing to delete the shared notes")
        with self._conn:
            cur = self._conn.execute("DELETE FROM chunks WHERE owner = ?", (owner,))
        self._cache = None
        return cur.rowcount

    def purge_uploads(self, older_than_s: float) -> int:
        """Delete uploads (never shared notes) stored more than
        `older_than_s` seconds ago. Returns how many chunks went."""
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM chunks WHERE owner != ? AND created_at < ?",
                (SHARED, time.time() - older_than_s),
            )
        if cur.rowcount:  # runs before every search, so keep the cache when nothing expired
            self._cache = None
        return cur.rowcount

    def first_line(self, doc_id: str, owner: str | None = None) -> str | None:
        """The first line of a doc's first chunk (a note's "# Title"), or
        None if `owner` has no such doc."""
        row = self._conn.execute(
            "SELECT text FROM chunks WHERE owner = ? AND doc_id = ? AND chunk_index = 0",
            (owner or SHARED, doc_id),
        ).fetchone()
        return row[0].split("\n", 1)[0] if row else None

    def list_docs(self, owner: str | None = None) -> list[dict]:
        """The shared notes, plus `owner`'s uploads when an owner is given."""
        rows = self._conn.execute(
            """SELECT doc_id, COUNT(*), owner != ? FROM chunks
               WHERE owner = ? OR owner = ?
               GROUP BY owner, doc_id ORDER BY owner != ?, doc_id""",
            (SHARED, SHARED, owner or SHARED, SHARED),
        ).fetchall()
        return [{"doc_id": d, "chunks": n, "uploaded": bool(u)} for d, n, u in rows]

    def count(self, owner: str | None = None) -> int:
        """Chunks visible to `owner`: the shared notes, plus its uploads."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE owner = ? OR owner = ?", (SHARED, owner or SHARED)
        ).fetchone()[0]

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks")
        self._cache = None

    def _matrix(self) -> tuple[np.ndarray, list[StoredChunk], list[str]]:
        if self._cache is None:
            rows = self._conn.execute(
                "SELECT owner, doc_id, chunk_index, text, embedding FROM chunks"
            ).fetchall()
            owners = [o for o, *_ in rows]
            chunks = [StoredChunk(d, i, t) for _, d, i, t, _ in rows]
            if rows:
                matrix = np.stack([np.frombuffer(e, dtype=np.float32) for *_, e in rows])
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                matrix = matrix / np.where(norms == 0, 1, norms)
            else:
                matrix = np.zeros((0, 0), dtype=np.float32)
            self._cache = (matrix, chunks, owners)
        return self._cache

    def search(self, query_vec: list[float], k: int, owner: str | None = None) -> list[StoredChunk]:
        """Top-k chunks among the shared notes and `owner`'s uploads."""
        matrix, chunks, owners = self._matrix()
        visible = np.array([o == SHARED or o == owner for o in owners], dtype=bool)
        if not visible.any():
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1)
        scores = np.where(visible, matrix @ q, -np.inf)
        top = np.argsort(-scores)[: min(k, int(visible.sum()))]
        return [
            StoredChunk(chunks[i].doc_id, chunks[i].chunk_index, chunks[i].text, float(scores[i]))
            for i in top
        ]
