import unittest

from app.problem_layers import (
    BUSINESS_TOOL,
    DETERMINISTIC_RULE,
    GENERAL_CHAT,
    KNOWLEDGE_EVIDENCE,
    classify_problem,
)


class ProblemLayerTest(unittest.TestCase):
    def test_general_chat_skips_retrieval(self) -> None:
        route = classify_problem("你好")
        self.assertEqual(route.layer, GENERAL_CHAT)
        self.assertFalse(route.should_retrieve)

    def test_help_question_uses_chatbot_layer(self) -> None:
        route = classify_problem("你能做什么")
        self.assertEqual(route.layer, GENERAL_CHAT)
        self.assertEqual(route.handler, "llm_chat")

    def test_weather_question_uses_general_chat_layer(self) -> None:
        route = classify_problem("今天天气多少")
        self.assertEqual(route.layer, GENERAL_CHAT)
        self.assertFalse(route.should_retrieve)

    def test_model_identity_question_uses_general_chat_layer(self) -> None:
        route = classify_problem("你的模型是什么？")
        self.assertEqual(route.layer, GENERAL_CHAT)
        self.assertEqual(route.handler, "llm_chat")
        self.assertFalse(route.should_retrieve)

    def test_unknown_question_defaults_to_general_chat(self) -> None:
        route = classify_problem("帮我想一个项目名字")
        self.assertEqual(route.layer, GENERAL_CHAT)
        self.assertFalse(route.should_retrieve)

    def test_escalation_uses_deterministic_rule(self) -> None:
        route = classify_problem("我发现敏感信息可能泄露了，需要联系管理员")
        self.assertEqual(route.layer, DETERMINISTIC_RULE)
        self.assertEqual(route.handler, "rule_engine")

    def test_internal_status_uses_business_tool(self) -> None:
        route = classify_problem("帮我查一下工单号 TICKET123456 的处理进度")
        self.assertEqual(route.layer, BUSINESS_TOOL)
        self.assertEqual(route.handler, "tool_call")

    def test_policy_question_uses_knowledge_evidence(self) -> None:
        route = classify_problem("差旅报销需要哪些材料？")
        self.assertEqual(route.layer, KNOWLEDGE_EVIDENCE)
        self.assertTrue(route.should_retrieve)

    def test_internal_policy_keywords_use_knowledge_evidence(self) -> None:
        route = classify_problem("生产系统权限申请需要填写有效期吗？")
        self.assertEqual(route.layer, KNOWLEDGE_EVIDENCE)
        self.assertTrue(route.should_retrieve)


if __name__ == "__main__":
    unittest.main()
