from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.retriever import RetrievalHit


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str


def build_prompt(
    question: str,
    hits: list[RetrievalHit],
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[来源 {index}: {hit.chunk.title}]\n"
        f"检索证据: {', '.join(hit.evidence) or '相关片段召回'}\n"
        f"{hit.context or hit.chunk.content}"
        for index, hit in enumerate(hits, start=1)
    )
    recent_history = "\n".join(f"{item['role']}: {item['content']}" for item in history[-6:])

    system = (
        "你是企业智能客服助手。只根据给定知识库上下文回答。"
        "如果上下文没有答案，明确说明知识库中没有找到相关信息，并给出需要补充的资料。"
        "优先使用被精确命中的条款和原文上下文，回答要简洁、准确，并在相关结论后标注来源编号。"
    )
    user = f"历史对话:\n{recent_history or '无'}\n\n知识库上下文:\n{context or '无'}\n\n用户问题:\n{question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_answer(
    config: LLMConfig,
    question: str,
    hits: list[RetrievalHit],
    history: list[dict[str, str]],
) -> str:
    if not config.api_key:
        return _fallback_answer(question, hits)

    payload = {
        "model": config.model,
        "messages": build_prompt(question, hits, history),
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        url=f"{config.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        return f"模型服务暂时不可用，已返回检索结果摘要。\n\n{_fallback_answer(question, hits)}\n\n错误信息: {exc}"

    return result["choices"][0]["message"]["content"].strip()


def _fallback_answer(question: str, hits: list[RetrievalHit]) -> str:
    if not hits:
        return "知识库中没有找到相关信息。建议补充与该问题相关的产品说明、售后政策或常见问题文档。"

    lines = ["当前未配置模型 API Key，先返回基于检索结果的摘要："]
    for index, hit in enumerate(hits, start=1):
        snippet = (hit.context or hit.chunk.content).replace("\n", " ")
        if len(snippet) > 180:
            snippet = f"{snippet[:180]}..."
        evidence = "；".join(hit.evidence) or "相关片段召回"
        lines.append(f"{index}. {snippet}（来源：{hit.chunk.title}，证据：{evidence}，相关度 {hit.score:.2f}）")
    return "\n".join(lines)
