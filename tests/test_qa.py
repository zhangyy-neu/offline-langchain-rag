"""基础单元测试（不依赖 Ollama 模型服务，可在离线环境直接运行）。

运行方式：
    python -m unittest tests.test_qa -v

说明：
- 测试覆盖文档加载、文本切分、来源格式化与参数校验等纯逻辑部分；
- 涉及真实模型与向量库的端到端问答请使用 app.py 或 web.py 手动验证。
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document  # noqa: E402

from modules.loader import load_documents  # noqa: E402
from modules.qa_chain import _format_docs  # noqa: E402
from modules.splitter import split_documents  # noqa: E402


def make_doc(content: str, source: str = "test.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


class TestSplitter(unittest.TestCase):
    """文本切分逻辑测试。"""

    def test_long_text_split_into_multiple_chunks(self):
        long_text = "计算机网络课程笔记内容。" * 300
        chunks = split_documents([make_doc(long_text)], chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 220)

    def test_short_text_kept_as_one_chunk(self):
        chunks = split_documents([make_doc("这是一段很短的文本。")], chunk_size=500, chunk_overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, "这是一段很短的文本。")

    def test_chunk_metadata_keeps_source(self):
        chunks = split_documents(
            [make_doc("内容。" * 500, source="manual.md")], chunk_size=200, chunk_overlap=20
        )
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "manual.md")


class TestLoader(unittest.TestCase):
    """文档加载逻辑测试。"""

    def test_load_txt_with_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "note.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("TCP 三次握手：SYN、SYN+ACK、ACK。")
            docs = load_documents(tmp)
            self.assertEqual(len(docs), 1)
            self.assertIn("TCP 三次握手", docs[0].page_content)
            self.assertEqual(docs[0].metadata["source"], path)

    def test_load_txt_with_gbk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gbk.txt")
            with open(path, "w", encoding="gbk") as f:
                f.write("GBK 编码的中文内容。")
            docs = load_documents(tmp)
            self.assertEqual(len(docs), 1)
            self.assertIn("GBK 编码", docs[0].page_content)

    def test_unsupported_files_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "image.png"), "wb") as f:
                f.write(b"fake-png")
            with open(os.path.join(tmp, "notes.txt"), "w", encoding="utf-8") as f:
                f.write("hello")
            docs = load_documents(tmp)
            self.assertEqual(len(docs), 1)

    def test_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_documents(os.path.join(tempfile.gettempdir(), "not_exist_dir_xyz"))


class TestQAChat(unittest.TestCase):
    """问答链纯逻辑测试（不触碰模型）。"""

    def test_empty_question_raises(self):
        from modules.qa_chain import QAChat

        qa = QAChat.__new__(QAChat)  # 跳过 __init__，避免依赖 Ollama 服务
        with self.assertRaises(ValueError):
            qa.ask("   ")

    def test_format_docs_joins_source_and_content(self):
        docs = [
            make_doc("第一段内容", source="a.txt"),
            make_doc("第二段内容", source="b.md"),
        ]
        text = _format_docs(docs)
        self.assertIn("[来源：a.txt]", text)
        self.assertIn("第一段内容", text)
        self.assertIn("[来源：b.md]", text)
        self.assertIn("第二段内容", text)


if __name__ == "__main__":
    unittest.main()
