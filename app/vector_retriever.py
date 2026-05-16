from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


WORD_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")

SEMANTIC_ALIASES = {
    "报销": ["费用", "差旅", "发票", "打款", "多久到账"],
    "请假": ["年假", "调休", "病假", "假期余额"],
    "权限": ["账号", "开通", "访问", "申请权限"],
    "工单": ["ticket", "问题单", "服务台", "处理进度"],
    "入职": ["新人", "试用期", "offer", "新员工"],
    "VPN": ["远程", "内网", "访问公司系统"],
    "负责人": ["管理员", "转交", "联系谁", "升级处理"],
    "使用": ["怎么用", "如何使用", "帮助"],
    "能力": ["你能做什么", "功能", "介绍一下"],
}


def text_features(text: str) -> list[str]:
    normalized = text.lower()
    features: list[str] = []
    for token in WORD_RE.findall(normalized):
        features.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            for size in (2, 3):
                features.extend(
                    token[index : index + size]
                    for index in range(0, max(0, len(token) - size + 1))
                )

    for concept, aliases in SEMANTIC_ALIASES.items():
        if concept in normalized or any(alias in normalized for alias in aliases):
            features.append(f"concept:{concept}")
    return features


def embed_text(text: str, dimensions: int = 256) -> Counter[int]:
    vector: Counter[int] = Counter()
    for feature in text_features(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=4).digest()
        index = int.from_bytes(digest, "big") % dimensions
        vector[index] += 1
    return vector


def cosine(left: Counter[int], right: Counter[int]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(value * right.get(index, 0) for index, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def vector_scores(
    query: str,
    documents: list[tuple[int, str]],
    dimensions: int = 256,
) -> dict[int, float]:
    query_vector = embed_text(query, dimensions=dimensions)
    scores: dict[int, float] = {}
    for document_id, text in documents:
        score = cosine(query_vector, embed_text(text, dimensions=dimensions))
        if score > 0:
            scores[document_id] = score
    return scores
