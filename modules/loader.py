"""文档加载模块。

支持从知识库目录递归加载以下格式：
- TXT（自动兼容 UTF-8 / GBK 编码）
- Markdown（.md / .markdown）
- PDF（.pdf）
"""
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# 支持的文件扩展名
SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".pdf"}

# 常见中文文本编码，按优先级依次尝试
_TEXT_ENCODINGS = ("utf-8", "gbk", "gb18030")


def _load_text_file(path: str) -> list:
    """加载文本类文件（TXT / Markdown），自动探测编码。

    说明：此处自行读取文件并构建 Document，避免部分版本
    TextLoader 将解码失败包装为 RuntimeError 导致无法回退编码。
    """
    text = None
    for encoding in _TEXT_ENCODINGS:
        try:
            with open(path, encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        # 兜底：忽略无法解码的字符
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    return [Document(page_content=text, metadata={"source": path})]


def _load_pdf_file(path: str) -> list:
    """加载 PDF 文件。"""
    return PyPDFLoader(path).load()


def load_documents(knowledge_dir: str) -> list:
    """递归遍历知识库目录，加载所有支持的文档。

    Args:
        knowledge_dir: 知识库目录路径。

    Returns:
        文档对象列表（langchain_core.documents.Document）。
    """
    if not os.path.isdir(knowledge_dir):
        raise FileNotFoundError(f"知识库目录不存在：{knowledge_dir}")

    documents = []
    for root, _dirs, files in os.walk(knowledge_dir):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            path = os.path.join(root, name)
            if ext == ".pdf":
                documents.extend(_load_pdf_file(path))
            else:
                documents.extend(_load_text_file(path))
    return documents
