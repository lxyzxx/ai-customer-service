from __future__ import annotations

from uuid import uuid4

from app.chatbot import answer_general_chat
from app.config import settings
from app.llm import LLMConfig, generate_answer
from app.problem_layers import (
    BUSINESS_TOOL,
    DETERMINISTIC_RULE,
    GENERAL_CHAT,
    classify_problem,
    route_to_dict,
)
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
        route = classify_problem(cleaned_question)
        memories: list[dict[str, str | float]] = []

        if route.layer == GENERAL_CHAT:
            answer, memories = answer_general_chat(cleaned_question)
            hits = []
        elif route.layer == DETERMINISTIC_RULE:
            answer = (
                "这个问题涉及投诉、赔偿、账号安全、金额异常或明确的人工转接诉求，"
                "应按确定性规则转接人工客服，并保留当前对话记录。"
            )
            hits = []
        elif route.layer == BUSINESS_TOOL:
            answer = (
                "这个问题需要查询实时业务系统，例如订单、物流、退款或售后进度。"
                "当前 MVP 尚未接入这些工具；生产环境应在这一层调用对应 API，而不是只查知识库。"
            )
            hits = []
        else:
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
            "route": route_to_dict(route),
            "memories": memories,
            "sources": [
                {
                    "chunk_id": hit.chunk.id,
                    "document_id": hit.chunk.document_id,
                    "title": hit.chunk.title,
                    "score": round(hit.score, 4),
                    "content": hit.chunk.content,
                    "context": hit.context,
                    "evidence": list(hit.evidence),
                }
                for hit in hits
            ],
        }
