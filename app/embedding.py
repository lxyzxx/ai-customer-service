from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str
    base_url: str
    model: str
    dimensions: int


def chunk_embedding_text(title: str, content: str) -> str:
    return f"{title}\n{content}".strip()


def embed_texts(config: EmbeddingConfig, texts: list[str]) -> list[list[float]]:
    if not config.api_key:
        raise ValueError("embedding api key is required")
    if not texts:
        return []

    from openai import OpenAI

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url.rstrip("/"),
        timeout=30,
    )
    payload = {"model": config.model, "input": texts}
    if config.dimensions > 0:
        payload["dimensions"] = config.dimensions

    response = client.embeddings.create(**payload)
    return [list(item.embedding) for item in response.data]
