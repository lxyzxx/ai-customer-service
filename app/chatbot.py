from __future__ import annotations

from dataclasses import dataclass

from app.vector_retriever import vector_scores


@dataclass(frozen=True)
class ChatbotMemory:
    id: int
    title: str
    content: str


CHATBOT_MEMORY = [
    ChatbotMemory(
        id=1,
        title="能力说明",
        content="我可以回答售后政策、退款规则、发票、物流异常、人工转接等客服问题。",
    ),
    ChatbotMemory(
        id=2,
        title="使用方式",
        content="你可以直接描述客户问题。如果涉及订单、物流、退款进度，需要提供对应单号并查询业务系统。",
    ),
    ChatbotMemory(
        id=3,
        title="证据优先",
        content="政策、FAQ 和售后条款类问题会先检索知识库原文，再基于证据和上下文回答。",
    ),
]


def answer_general_chat(question: str) -> tuple[str, list[dict[str, str | float]]]:
    if any(term in question for term in ("你能做什么", "功能", "介绍一下", "你是谁")):
        best = CHATBOT_MEMORY[0]
        score = 1.0
        return _format_memory_answer(best, score)

    documents = [(item.id, f"{item.title}\n{item.content}") for item in CHATBOT_MEMORY]
    scores = vector_scores(question, documents)
    ranked = sorted(CHATBOT_MEMORY, key=lambda item: scores.get(item.id, 0.0), reverse=True)
    best = ranked[0] if ranked else CHATBOT_MEMORY[0]
    score = scores.get(best.id, 0.0)

    if score == 0.0:
        best = CHATBOT_MEMORY[0]

    return _format_memory_answer(best, score)


def _format_memory_answer(
    best: ChatbotMemory,
    score: float,
) -> tuple[str, list[dict[str, str | float]]]:
    answer = best.content
    memory = [
        {
            "title": best.title,
            "content": best.content,
            "score": round(score, 4),
            "evidence": "chatbot 向量语义记忆",
        }
    ]
    return answer, memory
