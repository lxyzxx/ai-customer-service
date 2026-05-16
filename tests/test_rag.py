import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.rag import RAGService
from app.storage import Storage


class RAGServiceTest(unittest.TestCase):
    def test_general_chat_uses_llm_chat_layer_without_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir) / "app.db")
            service = RAGService(storage)

            with patch("app.rag.generate_chat_answer", return_value="普通聊天回答") as chat:
                result = service.answer("今天天气多少")

        chat.assert_called_once()
        self.assertEqual(result["answer"], "普通聊天回答")
        self.assertEqual(result["route"]["layer"], "general_chat")
        self.assertEqual(result["route"]["handler"], "llm_chat")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["chatbot_knowledge"], [])

    def test_knowledge_route_includes_bm25_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir) / "app.db")
            storage.add_document(
                "会议室预约制度",
                "会议室预约需要提前 1 个工作日提交申请。取消预约应提前 2 小时操作。",
            )
            service = RAGService(storage)

            result = service.answer("会议室预约需要提前多久？")

        self.assertEqual(result["route"]["layer"], "knowledge_evidence")
        self.assertEqual(result["sources"][0]["title"], "会议室预约制度")
        self.assertIn("SQLite FTS5/BM25 召回", result["sources"][0]["evidence"])


if __name__ == "__main__":
    unittest.main()
