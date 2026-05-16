from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import STATIC_DIR, settings
from app.embedding import EmbeddingConfig
from app.qdrant_index import NullVectorIndex, QdrantConfig, QdrantVectorIndex
from app.rag import RAGService
from app.storage import Storage


storage = Storage(settings.database_path)
vector_index = QdrantVectorIndex(
    QdrantConfig(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        dimensions=settings.embedding_dimensions,
    ),
    EmbeddingConfig(
        provider=settings.embedding_provider,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    ),
)
rag_service = RAGService(storage, vector_index)


class DocumentRequest(BaseModel):
    title: str
    content: str


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


def seed_sample_data(
    storage_instance: Storage = storage,
    vector_index_instance: NullVectorIndex | QdrantVectorIndex = vector_index,
) -> None:
    sample_path = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "sample_faq.md"
    if not sample_path.exists():
        return

    documents = storage_instance.list_documents()
    if storage_instance.is_empty():
        result = storage_instance.add_document(
            "示例内部问答 FAQ",
            sample_path.read_text(encoding="utf-8"),
        )
        sync_vector_index(vector_index_instance, storage_instance, int(result["id"]))
        return

    for document in documents:
        if document["title"] == "示例客服 FAQ":
            try:
                vector_index_instance.delete_document(int(document["id"]))
            except Exception:
                pass
            storage_instance.delete_document(int(document["id"]))
            result = storage_instance.add_document(
                "示例内部问答 FAQ",
                sample_path.read_text(encoding="utf-8"),
            )
            sync_vector_index(vector_index_instance, storage_instance, int(result["id"]))
            return


def sync_vector_index(
    vector_index_instance: NullVectorIndex | QdrantVectorIndex,
    storage_instance: Storage,
    document_id: int,
) -> dict[str, Any]:
    if not vector_index_instance.enabled:
        return {"vector_indexed": False, "vector_index_status": "disabled"}

    try:
        vector_index_instance.upsert_chunks(storage_instance.list_chunks(document_id))
    except Exception as exc:
        return {
            "vector_indexed": False,
            "vector_index_status": "failed",
            "vector_index_error": str(exc),
        }
    return {"vector_indexed": True, "vector_index_status": "ok"}


def create_app(
    storage_instance: Storage = storage,
    rag_service_instance: RAGService = rag_service,
    vector_index_instance: NullVectorIndex | QdrantVectorIndex = vector_index,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        seed_sample_data(storage_instance, vector_index_instance)
        yield

    api = FastAPI(
        title="Internal QA Bot",
        version="0.1.0",
        lifespan=lifespan,
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @api.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/api/documents")
    async def list_documents() -> dict[str, list[dict[str, Any]]]:
        return {"documents": storage_instance.list_documents()}

    @api.post("/api/documents", status_code=201)
    async def add_document(payload: DocumentRequest) -> dict[str, Any]:
        try:
            result = storage_instance.add_document(payload.title, payload.content)
            return {
                **result,
                **sync_vector_index(vector_index_instance, storage_instance, int(result["id"])),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/chat")
    async def chat(payload: ChatRequest) -> dict[str, Any]:
        try:
            return rag_service_instance.answer(payload.question, payload.session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - last-resort API guard
            raise HTTPException(status_code=500, detail=f"internal server error: {exc}") from exc

    api.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return api


app = create_app()


def main() -> None:
    uvicorn.run(
        "app.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
