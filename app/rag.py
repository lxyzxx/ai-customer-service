from __future__ import annotations

from uuid import uuid4

from app.chatbot import answer_general_chat
from app.config import settings
from app.llm import LLMConfig, generate_answer, generate_chat_answer
from app.problem_layers import (
    BUSINESS_TOOL,
    DETERMINISTIC_RULE,
    GENERAL_CHAT,
    classify_problem,
    route_to_dict,
)
from app.retriever import RetrievalHit, retrieve
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
        chatbot_knowledge: list[dict[str, str | float]] = []
        llm_config = LLMConfig(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
        )

        if route.layer == GENERAL_CHAT:
            answer = generate_chat_answer(llm_config, cleaned_question, history)
            if answer is None:
                answer, chatbot_knowledge = answer_general_chat(cleaned_question)
            hits = []
        elif route.layer == DETERMINISTIC_RULE:
            answer = (
                "这个问题涉及合规风险、账号安全、敏感信息、金额异常或明确的负责人介入诉求，"
                "应按确定性规则转交对应负责人，并保留当前对话记录。"
            )
            hits = []
        elif route.layer == BUSINESS_TOOL:
            answer = (
                "这个问题需要查询实时内部系统，例如审批、工单、报销、考勤或权限状态。"
                "当前 MVP 尚未接入这些工具；生产环境应在这一层调用对应 API，而不是只查知识库。"
            )
            hits = []
        else:
            chunks = self.storage.list_chunks()
            fts_scores = self.storage.search_chunks_fts(cleaned_question, limit=settings.top_k * 3)
            hits = retrieve(
                cleaned_question,
                chunks,
                top_k=settings.top_k,
                external_scores=fts_scores,
            )

            answer = generate_answer(
                llm_config,
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
            "retrieval_trace": build_retrieval_trace(hits),
            "chatbot_knowledge": chatbot_knowledge,
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


def build_retrieval_trace(hits: list[RetrievalHit]) -> list[dict[str, str | int]]:
    if not hits:
        return []

    channels = [
        ("keyword", "原文关键词命中", "原文命中"),
        ("bm25", "SQLite FTS5/BM25 全文召回", "SQLite FTS5/BM25 召回"),
        ("tfidf", "TF-IDF 弱线索召回", "TF-IDF 弱线索召回"),
        ("vector", "向量语义召回", "向量语义召回"),
    ]
    trace: list[dict[str, str | int]] = []
    for name, label, marker in channels:
        matched = sum(
            1
            for hit in hits
            if any(marker in evidence for evidence in hit.evidence)
        )
        if matched > 0:
            trace.append({"name": name, "label": label, "matched_sources": matched})

    if any(hit.context for hit in hits):
        trace.append(
            {
                "name": "context",
                "label": "相邻 chunk 上下文核验",
                "matched_sources": len(hits),
            }
        )
    return trace
