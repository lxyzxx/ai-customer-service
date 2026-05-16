import unittest
from unittest.mock import Mock, patch

from app.embedding import EmbeddingConfig, chunk_embedding_text, embed_texts


class EmbeddingTest(unittest.TestCase):
    def test_chunk_embedding_text_includes_title_and_content(self) -> None:
        text = chunk_embedding_text("会议室预约制度", "会议室预约需要提前 1 个工作日。")

        self.assertIn("会议室预约制度", text)
        self.assertIn("会议室预约需要提前", text)

    def test_local_provider_uses_sentence_transformer(self) -> None:
        fake_model = Mock()
        fake_model.encode.return_value = [[0.1, 0.2, 0.3]]

        with patch("app.embedding.get_sentence_transformer", return_value=fake_model) as get_model:
            vectors = embed_texts(
                EmbeddingConfig(
                    provider="local",
                    api_key="",
                    base_url="",
                    model="BAAI/bge-m3",
                    dimensions=1024,
                ),
                ["会议室预约制度"],
            )

        get_model.assert_called_once_with("BAAI/bge-m3")
        fake_model.encode.assert_called_once()
        self.assertEqual(vectors, [[0.1, 0.2, 0.3]])


if __name__ == "__main__":
    unittest.main()
