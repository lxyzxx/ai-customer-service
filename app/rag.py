from __future__ import annotations

from uuid import uuid4

from app.config import settings
from app.llm import LLMConfig, generate_answer
from app.retriever import retrieve
from app.storage import Storage


class RAGService:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def answer(self, question: str, session_id: str | None = None) -> dict:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question is required")

        active_session_id = session_id or str(uuid4())
        history = self.storage.get_recent_messages(active_session_id)
        hits = retrieve(cleaned_question, self.storage.list_chunks(), top_k=settings.top_k)

        answer = generate_answer(
            LLMConfig(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                model=settings.openai_model,
            ),
            cleaned_question,
            hits,
            history,
        )

        self.storage.add_message(active_session_id, "user", cleaned_question)
        self.storage.add_message(active_session_id, "assistant", answer)

        return {
            "session_id": active_session_id,
            "answer": answer,
            "sources": [
                {
                    "chunk_id": hit.chunk.id,
                    "document_id": hit.chunk.document_id,
                    "title": hit.chunk.title,
                    "score": round(hit.score, 4),
                    "content": hit.chunk.content,
                }
                for hit in hits
            ],
        }

