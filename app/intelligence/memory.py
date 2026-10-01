"""Session memory: the conversation history the intelligence layer feeds
back to the model each turn. In-memory here; the shape (a list of neutral
Messages keyed by session) is what a Redis/Postgres version would keep."""

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from app.inference.types import Message


@dataclass
class _Entry:
    message: Message
    added_at: float


class SessionStore:
    def __init__(
        self,
        window_messages: int = 20,
        max_sessions: int = 1000,
        max_age_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # Least recently used first. Past max_sessions the oldest conversation
        # is forgotten, so memory stays bounded however many visitors come by.
        self._sessions: OrderedDict[str, list[_Entry]] = OrderedDict()
        self.window_messages = window_messages
        self.max_sessions = max_sessions
        # Messages older than this are forgotten, in every session. Tool
        # results quote private notes, so without it a passage outlived its
        # note's 24 hours in any chat that stayed in memory.
        self.max_age_s = max_age_s
        self._clock = clock

    def history(self, session_id: str) -> list[Message]:
        self._expire()
        return [e.message for e in self._sessions.get(session_id, [])]

    def append(self, session_id: str, *messages: Message) -> None:
        self._expire()
        now = self._clock()
        self._sessions.setdefault(session_id, []).extend(_Entry(m, now) for m in messages)
        self._sessions.move_to_end(session_id)
        self._trim(session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def remove(self, session_id: str, messages: list[Message]) -> bool:
        """Remove these exact messages (by identity) from the session, and
        say whether the first was still there: it isn't after a reset, or
        once trimmed out of the window."""
        entries = self._sessions.get(session_id)
        if not entries or not messages:
            return False
        found = any(e.message is messages[0] for e in entries)
        drop = {id(m) for m in messages}
        self._sessions[session_id] = [e for e in entries if id(e.message) not in drop]
        return found

    def _trim(self, session_id: str) -> None:
        """Keep the last N messages."""
        entries = self._sessions[session_id]
        if len(entries) > self.window_messages:
            self._keep_from(session_id, len(entries) - self.window_messages)

    def _expire(self) -> None:
        """Drop every message older than max_age_s, from every session, so a
        chat nobody comes back to (a closed tab) expires too."""
        if self.max_age_s is None:
            return
        cutoff = self._clock() - self.max_age_s
        for session_id in list(self._sessions):
            entries = self._sessions[session_id]
            # Oldest first, so only these need work. remove() can leave a session empty.
            if entries and entries[0].added_at < cutoff:
                expired = next((i for i, e in enumerate(entries) if e.added_at >= cutoff), len(entries))
                self._keep_from(session_id, expired)

    def _keep_from(self, session_id: str, start: int) -> None:
        """Keep the session's messages from index `start` on, but never cut
        between a tool-call assistant message and its tool results - the
        model rejects orphans. Forgets the session if nothing is left."""
        entries = self._sessions[session_id]
        while start < len(entries) and entries[start].message.role != "user":
            start += 1
        if start < len(entries):
            self._sessions[session_id] = entries[start:]
        else:
            del self._sessions[session_id]
