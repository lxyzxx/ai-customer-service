import unittest

from app.chunker import chunk_text
from app.retriever import Chunk, extract_clues, read_context, retrieve, search_corpus, tokenize


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
        self.assertIn("原文命中 `物流`", hits[0].evidence)

    def test_extract_clues_keeps_exact_chinese_phrase(self) -> None:
        clues = extract_clues("定制商品超过 48 小时还能退款吗")
        self.assertIn("定制商品超过", clues)
        self.assertIn("48", clues)

    def test_search_corpus_combines_sparse_clues(self) -> None:
        chunks = [
            Chunk(id=1, document_id=1, title="普通退款", content="普通商品支持 7 天无理由退款。"),
            Chunk(id=2, document_id=2, title="定制商品", content="定制商品不支持 7 天无理由退款。"),
        ]
        matches = search_corpus(["定制商品", "退款"], chunks)
        self.assertGreater(matches[2][0], matches[1][0])

    def test_read_context_returns_neighboring_chunks(self) -> None:
        chunks = [
            Chunk(id=1, document_id=1, title="售后", content="第一段", position=0),
            Chunk(id=2, document_id=1, title="售后", content="第二段", position=1),
            Chunk(id=3, document_id=2, title="物流", content="其他文档", position=0),
        ]
        context = read_context(chunks[1], chunks)
        self.assertIn("第一段", context)
        self.assertIn("第二段", context)
        self.assertNotIn("其他文档", context)

    def test_retrieve_uses_vector_recall_as_auxiliary_channel(self) -> None:
        chunks = [
            Chunk(id=1, document_id=1, title="售后", content="商品存在质量问题可以申请售后换新。"),
            Chunk(id=2, document_id=2, title="发票", content="企业可以申请增值税专用发票。"),
        ]
        hits = retrieve("东西坏了可以换吗", chunks, top_k=1)
        self.assertEqual(hits[0].chunk.title, "售后")
        self.assertIn("向量语义召回", hits[0].evidence)
        self.assertNotIn("原文命中 `可以`", hits[0].evidence)


if __name__ == "__main__":
    unittest.main()
