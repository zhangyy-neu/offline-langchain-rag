"""文本切分模块。

使用 RecursiveCharacterTextSplitter 将长文档切分为适合向量化与检索的文本块。
切分参数可通过 .env 配置（CHUNK_SIZE / CHUNK_OVERLAP）。
"""
import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 中文优先的切分分隔符（按优先级从大到小）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def split_documents(
    documents: list,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list:
    """将文档切分为文本块。

    Args:
        documents: 文档对象列表。
        chunk_size: 每个文本块的最大字符数，默认 500。
        chunk_overlap: 相邻文本块之间的重叠字符数，默认 50。

    Returns:
        切分后的文本块列表。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or DEFAULT_CHUNK_SIZE,
        chunk_overlap=chunk_overlap or DEFAULT_CHUNK_OVERLAP,
        separators=_SEPARATORS,
    )
    return splitter.split_documents(documents)
