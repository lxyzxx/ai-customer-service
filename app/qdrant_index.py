from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.embedding import EmbeddingConfig, chunk_embedding_text, embed_texts
from app.retriever import Chunk, ExternalRetrievalScore


@dataclass(frozen=True)
class QdrantConfig:
    url: str
    api_key: str
    collection: str
    dimensions: int


class QdrantVectorIndex:
    def __init__(
        self,
        qdrant_config: QdrantConfig,
        embedding_config: EmbeddingConfig,
    ) -> None:
        self.qdrant_config = qdrant_config
        self.embedding_config = embedding_config
        self._client: Any | None = None

    @property
    def enabled(self) -> bool:
        if not self.qdrant_config.url:
            return False
        if self.embedding_config.provider == "local":
            return bool(self.embedding_config.model)
        return bool(self.embedding_config.api_key)

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not self.enabled or not chunks:
            return

        vectors = embed_texts(
            self.embedding_config,
            [chunk_embedding_text(chunk.title, chunk.content) for chunk in chunks],
        )
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            points.append(
                self._models().PointStruct(
                    id=chunk.id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "position": chunk.position,
                    },
                )
            )

        self._ensure_collection()
        self._client_instance().upsert(
            collection_name=self.qdrant_config.collection,
            points=points,
        )

    def delete_document(self, document_id: int) -> None:
        if not self.enabled:
            return

        models = self._models()
        self._client_instance().delete(
            collection_name=self.qdrant_config.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def search(self, query: str, limit: int = 8) -> dict[int, ExternalRetrievalScore]:
        if not self.enabled or not query.strip():
            return {}

        vector = embed_texts(self.embedding_config, [query])[0]
        self._ensure_collection()
        client = self._client_instance()
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=self.qdrant_config.collection,
                query=vector,
                limit=limit,
                with_payload=True,
            )
            points = response.points
        else:
            points = client.search(
                collection_name=self.qdrant_config.collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )

        scores: dict[int, ExternalRetrievalScore] = {}
        for point in points:
            payload = point.payload or {}
            chunk_id = int(payload.get("chunk_id", point.id))
            scores[chunk_id] = ExternalRetrievalScore(
                score=float(point.score) * 1.4,
                evidence=("Qdrant 向量召回",),
            )
        return scores

    def _client_instance(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self.qdrant_config.url,
                api_key=self.qdrant_config.api_key or None,
                timeout=10,
            )
        return self._client

    def _models(self):
        from qdrant_client import models

        return models

    def _ensure_collection(self) -> None:
        client = self._client_instance()
        if client.collection_exists(self.qdrant_config.collection):
            return

        models = self._models()
        client.create_collection(
            collection_name=self.qdrant_config.collection,
            vectors_config=models.VectorParams(
                size=self.qdrant_config.dimensions,
                distance=models.Distance.COSINE,
            ),
        )


class NullVectorIndex:
    enabled = False

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        return

    def delete_document(self, document_id: int) -> None:
        return

    def search(self, query: str, limit: int = 8) -> dict[int, ExternalRetrievalScore]:
        return {}
