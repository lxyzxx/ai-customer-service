import unittest

from app.embedding import EmbeddingConfig
from app.qdrant_index import NullVectorIndex, QdrantConfig, QdrantVectorIndex


class QdrantIndexTest(unittest.TestCase):
    def test_qdrant_index_disabled_without_url_or_embedding_key(self) -> None:
        index = QdrantVectorIndex(
            QdrantConfig(url="", api_key="", collection="chunks", dimensions=1536),
            EmbeddingConfig(
                provider="openai",
                api_key="",
                base_url="https://api.openai.com/v1",
                model="m",
                dimensions=1536,
            ),
        )

        self.assertFalse(index.enabled)
        self.assertEqual(index.search("会议室预约"), {})

    def test_null_vector_index_is_noop(self) -> None:
        index = NullVectorIndex()

        self.assertFalse(index.enabled)
        self.assertEqual(index.search("anything"), {})


if __name__ == "__main__":
    unittest.main()
