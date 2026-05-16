import unittest

from app.embedding import chunk_embedding_text


class EmbeddingTest(unittest.TestCase):
    def test_chunk_embedding_text_includes_title_and_content(self) -> None:
        text = chunk_embedding_text("会议室预约制度", "会议室预约需要提前 1 个工作日。")

        self.assertIn("会议室预约制度", text)
        self.assertIn("会议室预约需要提前", text)


if __name__ == "__main__":
    unittest.main()
