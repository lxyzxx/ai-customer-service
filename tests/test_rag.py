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


if __name__ == "__main__":
    unittest.main()
