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
    re.compile(r"(你是谁|你能做什么|怎么用|如何使用|帮助|功能|介绍一下)"),
]

ESCALATION_TERMS = [
    "负责人",
    "转负责人",
    "联系管理员",
    "投诉",
    "举报",
    "合规",
    "数据泄露",
    "敏感信息",
    "账号安全",
    "盗号",
    "金额异常",
    "人事争议",
]

BUSINESS_TOOL_PATTERNS = [
    re.compile(r"(查|查询|看一下|帮我看).*(审批|工单|报销|假期余额|考勤|项目进度|系统状态)"),
    re.compile(r"(我的|我这个).*(审批|工单|报销|假期|考勤|权限申请)"),
    re.compile(r"(审批单号|工单号|报销单号|申请单号|ticket)[:：]?\s*[a-zA-Z0-9-]{6,}"),
]


def classify_problem(question: str) -> ProblemRoute:
    text = question.strip()
    lowered = text.lower()

    if any(pattern.search(text) for pattern in GENERAL_PATTERNS):
        return ProblemRoute(
            layer=GENERAL_CHAT,
            handler="chatbot",
            reason="普通寒暄或帮助类问题，使用 chatbot 内置知识",
            should_retrieve=False,
        )

    if any(term in text for term in ESCALATION_TERMS):
        return ProblemRoute(
            layer=DETERMINISTIC_RULE,
            handler="rule_engine",
            reason="命中高风险、合规或负责人介入规则",
            should_retrieve=False,
        )

    if any(pattern.search(text) for pattern in BUSINESS_TOOL_PATTERNS) or re.search(
        r"\b(ticket|approval|expense|leave)[-_]?[0-9a-z]{6,}\b",
        lowered,
    ):
        return ProblemRoute(
            layer=BUSINESS_TOOL,
            handler="tool_call",
            reason="需要查询实时审批、工单、报销、考勤或权限系统",
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
