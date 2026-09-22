"""Session memory: the conversation history the intelligence layer feeds
back to the model each turn. In-memory here; the shape (a list of neutral
Messages keyed by session) is what a Redis/Postgres version would keep."""

from collections import defaultdict

from app.inference.types import Message


class SessionStore:
    def __init__(self, window_messages: int = 20) -> None:
        self._sessions: dict[str, list[Message]] = defaultdict(list)
        self.window_messages = window_messages

    def history(self, session_id: str) -> list[Message]:
        return list(self._sessions[session_id])

    def append(self, session_id: str, *messages: Message) -> None:
        self._sessions[session_id].extend(messages)
        self._trim(session_id)

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
