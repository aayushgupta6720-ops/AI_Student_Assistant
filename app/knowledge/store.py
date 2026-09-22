"""A deliberately small vector store: SQLite for durability, numpy for
cosine search. Fine for thousands of chunks. Swap this file for a Qdrant /
pgvector client and nothing else in the app changes."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                   doc_id TEXT NOT NULL,
                   chunk_index INTEGER NOT NULL,
                   text TEXT NOT NULL,
                   embedding BLOB NOT NULL,
                   PRIMARY KEY (doc_id, chunk_index)
               )"""
        )
        self._conn.commit()
        self._cache: tuple[np.ndarray, list[StoredChunk]] | None = None

    def upsert_doc(self, doc_id: str, texts: list[str], embeddings: list[list[float]]) -> None:
        assert len(texts) == len(embeddings)
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                [
                    (doc_id, i, text, np.asarray(vec, dtype=np.float32).tobytes())
                    for i, (text, vec) in enumerate(zip(texts, embeddings))
                ],
            )
        self._cache = None

    def delete_doc(self, doc_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._cache = None

    def list_docs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT doc_id, COUNT(*) FROM chunks GROUP BY doc_id ORDER BY doc_id"
        ).fetchall()
        return [{"doc_id": d, "chunks": n} for d, n in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks")
        self._cache = None

    def _matrix(self) -> tuple[np.ndarray, list[StoredChunk]]:
        if self._cache is None:
            rows = self._conn.execute(
                "SELECT doc_id, chunk_index, text, embedding FROM chunks"
            ).fetchall()
            chunks = [StoredChunk(d, i, t) for d, i, t, _ in rows]
            if rows:
                matrix = np.stack([np.frombuffer(e, dtype=np.float32) for *_, e in rows])
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                matrix = matrix / np.where(norms == 0, 1, norms)
            else:
                matrix = np.zeros((0, 0), dtype=np.float32)
            self._cache = (matrix, chunks)
        return self._cache

    def search(self, query_vec: list[float], k: int) -> list[StoredChunk]:
        matrix, chunks = self._matrix()
        if not chunks:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1)
        scores = matrix @ q
        top = np.argsort(-scores)[:k]
        return [
            StoredChunk(chunks[i].doc_id, chunks[i].chunk_index, chunks[i].text, float(scores[i]))
            for i in top
        ]
