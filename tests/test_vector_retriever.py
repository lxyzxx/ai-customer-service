import unittest

from app.chatbot import answer_general_chat
from app.vector_retriever import vector_scores


class VectorRetrieverTest(unittest.TestCase):
    def test_vector_scores_match_semantic_aliases(self) -> None:
        scores = vector_scores(
            "差旅费用怎么报",
            [
                (1, "员工完成出差后需要提交差旅报销申请。"),
                (2, "员工请年假需要提前在 OA 系统提交申请。"),
            ],
        )
        self.assertGreater(scores[1], scores.get(2, 0.0))

    def test_chatbot_uses_vector_knowledge(self) -> None:
        answer, knowledge = answer_general_chat("你能做什么")
        self.assertIn("内部问题", answer)
        self.assertEqual(knowledge[0]["evidence"], "chatbot 内置知识语义召回")


if __name__ == "__main__":
    unittest.main()
