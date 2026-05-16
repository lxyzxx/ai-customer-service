from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
ENV_PATH = BASE_DIR / ".env"


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


class Settings:
    def __init__(self) -> None:
        self.host = os.getenv("APP_HOST", "127.0.0.1")
        self.port = int(os.getenv("APP_PORT", "8000"))
        self.database_path = Path(os.getenv("DATABASE_PATH", DATA_DIR / "app.db"))
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", "")
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        self.qdrant_url = os.getenv("QDRANT_URL", "")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "internal_qa_chunks")
        self.top_k = int(os.getenv("RAG_TOP_K", "4"))


settings = Settings()
