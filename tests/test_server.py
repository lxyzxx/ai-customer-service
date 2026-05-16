import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.rag import RAGService
from app.server import create_app
from app.storage import Storage


async def request(
    app,
    method: str,
    path: str,
    payload: dict | None = None,
) -> tuple[int, dict[str, str], bytes]:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    headers = [(b"host", b"testserver")]
    if payload is not None:
        headers.append((b"content-type", b"application/json"))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    received = False
    messages = []

    async def receive() -> dict:
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    await app(scope, receive, send)

    start = next(message for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    response_headers = {
        key.decode("latin-1"): value.decode("latin-1")
        for key, value in start.get("headers", [])
    }
    return int(start["status"]), response_headers, response_body


class ServerTest(unittest.TestCase):
    def make_app(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        storage = Storage(Path(temp_dir.name) / "app.db")
        return create_app(storage, RAGService(storage))

    def test_health_endpoint(self) -> None:
        status, _, body = asyncio.run(request(self.make_app(), "GET", "/api/health"))

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "ok"})

    def test_document_endpoints(self) -> None:
        app = self.make_app()
        create_status, _, create_body = asyncio.run(
            request(
                app,
                "POST",
                "/api/documents",
                {"title": "会议室预约制度", "content": "会议室预约需要提前 1 个工作日。"},
            )
        )
        list_status, _, list_body = asyncio.run(request(app, "GET", "/api/documents"))

        self.assertEqual(create_status, 201)
        self.assertEqual(json.loads(create_body)["title"], "会议室预约制度")
        self.assertEqual(list_status, 200)
        self.assertTrue(
            any(doc["title"] == "会议室预约制度" for doc in json.loads(list_body)["documents"])
        )

    def test_chat_endpoint(self) -> None:
        with patch("app.rag.generate_chat_answer", return_value="你好"):
            status, _, body = asyncio.run(
                request(self.make_app(), "POST", "/api/chat", {"question": "你好"})
            )

        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["answer"], "你好")
        self.assertEqual(data["route"]["handler"], "llm_chat")


if __name__ == "__main__":
    unittest.main()
