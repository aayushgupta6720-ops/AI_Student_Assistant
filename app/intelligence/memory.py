"""Session memory: the conversation history the intelligence layer feeds
back to the model each turn. In-memory here; the shape (a list of neutral
Messages keyed by session) is what a Redis/Postgres version would keep."""

from collections import OrderedDict

from app.inference.types import Message


class SessionStore:
    def __init__(self, window_messages: int = 20, max_sessions: int = 1000) -> None:
        # Least recently used first. Past max_sessions the oldest conversation
        # is forgotten, so memory stays bounded however many visitors come by.
        self._sessions: OrderedDict[str, list[Message]] = OrderedDict()
        self.window_messages = window_messages
        self.max_sessions = max_sessions

    def history(self, session_id: str) -> list[Message]:
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, *messages: Message) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)
        self._sessions.move_to_end(session_id)
        self._trim(session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _trim(self, session_id: str) -> None:
        """Keep the last N messages, but never cut between a tool-call
        assistant message and its tool results - the model rejects orphans."""
        msgs = self._sessions[session_id]
        if len(msgs) <= self.window_messages:
            return
        start = len(msgs) - self.window_messages
        while start < len(msgs) and msgs[start].role != "user":
            start += 1
        self._sessions[session_id] = msgs[start:]
