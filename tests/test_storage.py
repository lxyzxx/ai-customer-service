import tempfile
import unittest
from pathlib import Path

from app.storage import Storage


class StorageTest(unittest.TestCase):
    def test_search_chunks_fts_returns_bm25_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir) / "app.db")
            storage.add_document(
                "会议室预约制度",
                "会议室预约需要提前 1 个工作日提交申请。取消预约应提前 2 小时操作。",
            )
            storage.add_document("请假制度", "员工请年假需要提前在 OA 系统提交申请。")

            scores = storage.search_chunks_fts("会议室预约需要提前多久")

        self.assertTrue(scores)
        self.assertIn("SQLite FTS5/BM25 召回", next(iter(scores.values())).evidence)

    def test_delete_document_removes_fts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = Storage(Path(temp_dir) / "app.db")
            result = storage.add_document("会议室预约制度", "会议室预约需要提前 1 个工作日。")

            self.assertTrue(storage.search_chunks_fts("会议室预约"))
            storage.delete_document(int(result["id"]))

            self.assertEqual(storage.search_chunks_fts("会议室预约"), {})


if __name__ == "__main__":
    unittest.main()
