import unittest

from app.chatbot import answer_general_chat
from app.vector_retriever import vector_scores


class VectorRetrieverTest(unittest.TestCase):
    def test_vector_scores_match_semantic_aliases(self) -> None:
        scores = vector_scores(
            "东西坏了可以换吗",
            [
                (1, "商品存在质量问题可以申请售后换新。"),
                (2, "企业用户可以申请增值税专用发票。"),
            ],
        )
        self.assertGreater(scores[1], scores.get(2, 0.0))

    def test_chatbot_uses_vector_memory(self) -> None:
        answer, memories = answer_general_chat("你能做什么")
        self.assertIn("客服问题", answer)
        self.assertEqual(memories[0]["evidence"], "chatbot 向量语义记忆")


if __name__ == "__main__":
    unittest.main()

