import httpx

import app.main as main
from app.api.ratelimit import build_rate_limiters
from app.config import get_settings
from app.intelligence.memory import SessionStore
from app.knowledge.store import VectorStore
from app.knowledge.uploads import MAX_UPLOAD_BYTES
from tests.fake_provider import FakeProvider


class NeverCalledAgent:
    async def run_turn(self, *args, **kwargs):
        raise AssertionError("an oversized request must not reach the agent")
        yield


async def _post(monkeypatch, tmp_path, path, **kwargs):
    state = {"agent": NeverCalledAgent(), "store": VectorStore(tmp_path / "s.sqlite"), "provider": FakeProvider([]),
             "memory": SessionStore(), "rate_limiters": build_rate_limiters(get_settings())}
    for name, value in state.items():
        monkeypatch.setattr(main.app.state, name, value, raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        return await client.post(path, **kwargs)


async def test_an_oversized_chat_body_is_refused_before_it_is_read(monkeypatch, tmp_path):
    # FastAPI reads and parses the whole body before checking max_length: a
    # 52 MB message used to cost ~250 MB of memory before its 422.
    r = await _post(monkeypatch, tmp_path, "/chat", json={"session_id": "s", "message": "x" * 200_000})
    assert r.status_code == 413 and r.json() == {"detail": "The request is over the 128 KB limit."}


async def test_uploads_get_room_for_a_full_size_file(monkeypatch, tmp_path):
    # A 2 MB file is still refused by the upload's own check, with its own message
    near_limit = await _post(monkeypatch, tmp_path, "/notes/upload", data={"session_id": "a"},
                             files={"file": ("big.md", b"x" * (MAX_UPLOAD_BYTES + 1))})
    assert near_limit.status_code == 400 and "over the 2 MB limit" in near_limit.json()["detail"]
    way_over = await _post(monkeypatch, tmp_path, "/notes/upload", data={"session_id": "a"},
                           files={"file": ("huge.md", b"x" * (3 * MAX_UPLOAD_BYTES))})
    assert way_over.status_code == 413
