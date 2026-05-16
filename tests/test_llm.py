import sys
import types
import unittest
from typing import Any

from app.llm import LLMConfig, generate_answer
from app.retriever import Chunk, RetrievalHit


class LLMTest(unittest.TestCase):
    def test_generate_answer_falls_back_without_api_key(self) -> None:
        answer = generate_answer(
            LLMConfig(api_key="", base_url="https://api.openai.com/v1", model="gpt-4o-mini"),
            "差旅报销需要什么材料？",
            [
                RetrievalHit(
                    chunk=Chunk(
                        id=1,
                        document_id=1,
                        title="报销",
                        content="差旅报销需要审批单、发票和付款凭证。",
                    ),
                    score=1.0,
                    evidence=("原文命中 `报销`",),
                )
            ],
            [],
        )

        self.assertIn("当前未配置模型 API Key", answer)
        self.assertIn("差旅报销需要审批单", answer)

    def test_generate_answer_uses_openai_sdk(self) -> None:
        calls: dict[str, Any] = {}

        class FakeCompletion:
            def model_dump(self) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "差旅报销需要审批单、发票和付款凭证。[来源 1]"
                            }
                        }
                    ]
                }

        class FakeCompletions:
            def create(self, **payload: object) -> FakeCompletion:
                calls["payload"] = payload
                return FakeCompletion()

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, **kwargs: object) -> None:
                calls["client"] = kwargs
                self.chat = FakeChat()

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = FakeOpenAI
        previous_openai = sys.modules.get("openai")
        sys.modules["openai"] = fake_openai

        try:
            answer = generate_answer(
                LLMConfig(
                    api_key="test-key",
                    base_url="https://example.com/v1/",
                    model="gpt-test",
                ),
                "差旅报销需要什么材料？",
                [
                    RetrievalHit(
                        chunk=Chunk(
                            id=1,
                            document_id=1,
                            title="报销",
                            content="差旅报销需要审批单、发票和付款凭证。",
                        ),
                        score=1.0,
                        evidence=("原文命中 `报销`",),
                    )
                ],
                [],
            )
        finally:
            if previous_openai is None:
                sys.modules.pop("openai", None)
            else:
                sys.modules["openai"] = previous_openai

        self.assertEqual(answer, "差旅报销需要审批单、发票和付款凭证。[来源 1]")
        self.assertEqual(calls["client"]["api_key"], "test-key")
        self.assertEqual(calls["client"]["base_url"], "https://example.com/v1")
        self.assertEqual(calls["client"]["timeout"], 30)
        self.assertEqual(calls["payload"]["model"], "gpt-test")
        self.assertEqual(calls["payload"]["temperature"], 0.2)
        self.assertEqual(calls["payload"]["messages"][0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
