import unittest

from app.chunker import chunk_text
from app.retriever import Chunk, extract_clues, read_context, retrieve, search_corpus, tokenize


class RetrieverTest(unittest.TestCase):
    def test_tokenize_supports_chinese_and_english(self) -> None:
        self.assertIn("报销", "".join(tokenize("报销 Expense 需要多久？")))
        self.assertIn("expense", tokenize("报销 Expense 需要多久？"))

    def test_chunk_text_keeps_content(self) -> None:
        chunks = chunk_text("第一段内容。\n\n第二段内容。", chunk_size=20)
        self.assertEqual(chunks, ["第一段内容。\n\n第二段内容。"])

    def test_retrieve_returns_relevant_chunk(self) -> None:
        chunks = [
            Chunk(id=1, document_id=1, title="请假", content="请年假需要提前在 OA 系统提交申请。"),
            Chunk(id=2, document_id=2, title="报销", content="差旅报销需要上传审批单、发票和付款凭证。"),
        ]
        hits = retrieve("差旅报销需要什么材料", chunks, top_k=1)
        self.assertEqual(hits[0].chunk.title, "报销")
        self.assertIn("原文命中 `报销`", hits[0].evidence)

    def test_extract_clues_keeps_exact_chinese_phrase(self) -> None:
        clues = extract_clues("生产系统权限超过 48 小时还没开通吗")
        self.assertIn("生产系统权限超过", clues)
        self.assertIn("48", clues)

    def test_search_corpus_combines_sparse_clues(self) -> None:
        chunks = [
            Chunk(id=1, document_id=1, title="普通权限", content="普通系统权限由部门管理员审批。"),
            Chunk(id=2, document_id=2, title="生产系统", content="生产系统权限需要说明用途、范围和有效期。"),
        ]
        matches = search_corpus(["生产系统", "权限"], chunks)
        self.assertGreater(matches[2][0], matches[1][0])

    def test_read_context_returns_neighboring_chunks(self) -> None:
        chunks = [
            Chunk(id=1, document_id=1, title="制度", content="第一段", position=0),
            Chunk(id=2, document_id=1, title="制度", content="第二段", position=1),
            Chunk(id=3, document_id=2, title="工单", content="其他文档", position=0),
        ]
        context = read_context(chunks[1], chunks)
        self.assertIn("第一段", context)
        self.assertIn("第二段", context)
        self.assertNotIn("其他文档", context)

    def test_retrieve_uses_vector_recall_as_auxiliary_channel(self) -> None:
        chunks = [
            Chunk(id=1, document_id=1, title="请假", content="员工请年假需要提前在 OA 系统提交申请。"),
            Chunk(id=2, document_id=2, title="权限", content="访问公司内网系统需要先开通 VPN。"),
        ]
        hits = retrieve("远程访问公司系统怎么开", chunks, top_k=1)
        self.assertEqual(hits[0].chunk.title, "权限")
        self.assertIn("向量语义召回", hits[0].evidence)
        self.assertNotIn("原文命中 `可以`", hits[0].evidence)


if __name__ == "__main__":
    unittest.main()
