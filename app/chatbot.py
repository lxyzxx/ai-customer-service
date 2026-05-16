from __future__ import annotations

from dataclasses import dataclass

from app.vector_retriever import vector_scores


@dataclass(frozen=True)
class ChatbotKnowledge:
    id: int
    title: str
    content: str


CHATBOT_KNOWLEDGE = [
    ChatbotKnowledge(
        id=1,
        title="能力说明",
        content="我可以回答公司制度、流程说明、系统使用、权限申请和常见内部问题。",
    ),
    ChatbotKnowledge(
        id=2,
        title="使用方式",
        content="你可以直接描述问题。如果涉及审批、工单、报销状态或假期余额，需要提供对应编号并查询业务系统。",
    ),
    ChatbotKnowledge(
        id=3,
        title="证据优先",
        content="制度、流程、FAQ 和操作手册类问题会先检索知识库原文，再基于证据和上下文回答。",
    ),
]


def answer_general_chat(question: str) -> tuple[str, list[dict[str, str | float]]]:
    if any(term in question for term in ("你能做什么", "功能", "介绍一下", "你是谁")):
        best = CHATBOT_KNOWLEDGE[0]
        score = 1.0
        return _format_knowledge_answer(best, score)

    documents = [(item.id, f"{item.title}\n{item.content}") for item in CHATBOT_KNOWLEDGE]
    scores = vector_scores(question, documents)
    ranked = sorted(CHATBOT_KNOWLEDGE, key=lambda item: scores.get(item.id, 0.0), reverse=True)
    best = ranked[0] if ranked else CHATBOT_KNOWLEDGE[0]
    score = scores.get(best.id, 0.0)

    if score == 0.0:
        best = CHATBOT_KNOWLEDGE[0]

    return _format_knowledge_answer(best, score)


def _format_knowledge_answer(
    best: ChatbotKnowledge,
    score: float,
) -> tuple[str, list[dict[str, str | float]]]:
    answer = best.content
    knowledge = [
        {
            "title": best.title,
            "content": best.content,
            "score": round(score, 4),
            "evidence": "chatbot 内置知识语义召回",
        }
    ]
    return answer, knowledge
