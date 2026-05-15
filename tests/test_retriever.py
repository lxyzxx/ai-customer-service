import unittest

from app.chunker import chunk_text
from app.retriever import Chunk, retrieve, tokenize


class RetrieverTest(unittest.TestCase):
    def test_tokenize_supports_chinese_and_english(self) -> None:
        self.assertIn("退款", "".join(tokenize("退款 Refund 需要多久？")))
        self.assertIn("refund", tokenize("退款 Refund 需要多久？"))

    def test_chunk_text_keeps_content(self) -> None:
        chunks = chunk_text("第一段内容。\n\n第二段内容。", chunk_size=20)
        self.assertEqual(chunks, ["第一段内容。\n\n第二段内容。"])

    def test_retrieve_returns_relevant_chunk(self) -> None:
        chunks = [
            Chunk(id=1, document_id=1, title="发票", content="企业可以申请增值税专用发票。"),
            Chunk(id=2, document_id=2, title="物流", content="物流超过 48 小时没有更新需要提交工单。"),
        ]
        hits = retrieve("物流一直没有更新怎么办", chunks, top_k=1)
        self.assertEqual(hits[0].chunk.title, "物流")


if __name__ == "__main__":
    unittest.main()

