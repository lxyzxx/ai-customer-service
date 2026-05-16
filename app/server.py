from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.config import STATIC_DIR, settings
from app.rag import RAGService
from app.storage import Storage


storage = Storage(settings.database_path)
rag_service = RAGService(storage)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "InternalQABot/0.1"

    def do_OPTIONS(self) -> None:
        self._send_response(204, None)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_response(200, {"status": "ok"})
            return
        if path == "/api/documents":
            self._send_response(200, {"documents": storage.list_documents()})
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/documents":
                result = storage.add_document(
                    title=str(payload.get("title", "")),
                    content=str(payload.get("content", "")),
                )
                self._send_response(201, result)
                return
            if path == "/api/chat":
                result = rag_service.answer(
                    question=str(payload.get("question", "")),
                    session_id=payload.get("session_id"),
                )
                self._send_response(200, result)
                return
            self._send_response(404, {"error": "not found"})
        except ValueError as exc:
            self._send_response(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - last-resort API guard
            self._send_response(500, {"error": f"internal server error: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _serve_static(self, path: str) -> None:
        file_path = STATIC_DIR / "index.html" if path == "/" else STATIC_DIR / path.lstrip("/")
        resolved = file_path.resolve()
        if not str(resolved).startswith(str(STATIC_DIR.resolve())) or not resolved.exists() or resolved.is_dir():
            self._send_response(404, {"error": "not found"})
            return

        content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        body = resolved.read_bytes()
        self.send_response(200)
        self._send_common_headers(content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_response(self, status: int, payload: dict[str, Any] | None) -> None:
        body = b"" if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_common_headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_common_headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")


def seed_sample_data() -> None:
    sample_path = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "sample_faq.md"
    if not sample_path.exists():
        return

    documents = storage.list_documents()
    if storage.is_empty():
        storage.add_document("示例内部问答 FAQ", sample_path.read_text(encoding="utf-8"))
        return

    for document in documents:
        if document["title"] == "示例客服 FAQ":
            storage.delete_document(int(document["id"]))
            storage.add_document("示例内部问答 FAQ", sample_path.read_text(encoding="utf-8"))
            return


def main() -> None:
    seed_sample_data()
    server = ThreadingHTTPServer((settings.host, settings.port), RequestHandler)
    print(f"Server running at http://{settings.host}:{settings.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
