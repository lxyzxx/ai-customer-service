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

    def test_escalation_uses_deterministic_rule(self) -> None:
        route = classify_problem("我要投诉并要求赔偿")
        self.assertEqual(route.layer, DETERMINISTIC_RULE)
        self.assertEqual(route.handler, "rule_engine")

    def test_order_status_uses_business_tool(self) -> None:
        route = classify_problem("帮我查一下订单号 ABC123456 的物流")
        self.assertEqual(route.layer, BUSINESS_TOOL)
        self.assertEqual(route.handler, "tool_call")

    def test_policy_question_uses_knowledge_evidence(self) -> None:
        route = classify_problem("退款多久到账？")
        self.assertEqual(route.layer, KNOWLEDGE_EVIDENCE)
        self.assertTrue(route.should_retrieve)


if __name__ == "__main__":
    unittest.main()

