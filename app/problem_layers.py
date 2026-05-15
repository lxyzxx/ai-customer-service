from __future__ import annotations

import re
from dataclasses import dataclass


GENERAL_CHAT = "general_chat"
DETERMINISTIC_RULE = "deterministic_rule"
BUSINESS_TOOL = "business_tool"
KNOWLEDGE_EVIDENCE = "knowledge_evidence"


@dataclass(frozen=True)
class ProblemRoute:
    layer: str
    handler: str
    reason: str
    should_retrieve: bool


GENERAL_PATTERNS = [
    re.compile(r"^(你好|您好|hello|hi|谢谢|感谢|再见|拜拜)[！!。.\s]*$", re.IGNORECASE),
]

ESCALATION_TERMS = [
    "人工",
    "转人工",
    "投诉",
    "赔偿",
    "账号安全",
    "盗号",
    "支付异常",
    "订单金额异常",
    "金额异常",
]

BUSINESS_TOOL_PATTERNS = [
    re.compile(r"(查|查询|看一下|帮我看).*(订单|物流|快递|退款进度|售后进度)"),
    re.compile(r"(我的|我这个).*(订单|物流|快递|退款|售后)"),
    re.compile(r"(订单号|物流单号|运单号|售后单号)[:：]?\s*[a-zA-Z0-9-]{6,}"),
]


def classify_problem(question: str) -> ProblemRoute:
    text = question.strip()
    lowered = text.lower()

    if any(pattern.search(text) for pattern in GENERAL_PATTERNS):
        return ProblemRoute(
            layer=GENERAL_CHAT,
            handler="chatbot",
            reason="普通寒暄，不需要查知识库或业务系统",
            should_retrieve=False,
        )

    if any(term in text for term in ESCALATION_TERMS):
        return ProblemRoute(
            layer=DETERMINISTIC_RULE,
            handler="rule_engine",
            reason="命中高风险或人工转接规则",
            should_retrieve=False,
        )

    if any(pattern.search(text) for pattern in BUSINESS_TOOL_PATTERNS) or re.search(
        r"\b(order|refund|ticket|tracking)[-_]?[0-9a-z]{6,}\b",
        lowered,
    ):
        return ProblemRoute(
            layer=BUSINESS_TOOL,
            handler="tool_call",
            reason="需要查询实时订单、物流、退款或售后系统",
            should_retrieve=False,
        )

    return ProblemRoute(
        layer=KNOWLEDGE_EVIDENCE,
        handler="dci_retrieval",
        reason="需要从知识库原文中定位政策、流程或 FAQ 证据",
        should_retrieve=True,
    )


def route_to_dict(route: ProblemRoute) -> dict[str, str | bool]:
    return {
        "layer": route.layer,
        "handler": route.handler,
        "reason": route.reason,
        "should_retrieve": route.should_retrieve,
    }

