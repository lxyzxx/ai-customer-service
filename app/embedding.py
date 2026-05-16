from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    dimensions: int


def chunk_embedding_text(title: str, content: str) -> str:
    return f"{title}\n{content}".strip()


def embed_texts(config: EmbeddingConfig, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if config.provider == "local":
        return embed_texts_local(config.model, texts)
    return embed_texts_openai(config, texts)


def embed_texts_openai(config: EmbeddingConfig, texts: list[str]) -> list[list[float]]:
    if not config.api_key:
        raise ValueError("embedding api key is required")
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


def embed_texts_local(model_name: str, texts: list[str]) -> list[list[float]]:
    model = get_sentence_transformer(model_name)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [
        vector.tolist() if hasattr(vector, "tolist") else list(vector)
        for vector in vectors
    ]


@lru_cache(maxsize=2)
def get_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "local embedding requires installing the optional dependency: "
            'python3 -m pip install -e ".[local-embedding]"'
        ) from exc

    return SentenceTransformer(model_name)
