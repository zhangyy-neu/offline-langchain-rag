"""文档检索模块。

基于 Chroma 向量库构建相似度检索器，
按语义相似度返回与问题最相关的文本块。
"""
import os

from dotenv import load_dotenv

from modules.vector_store import get_vector_store

load_dotenv()

DEFAULT_K = int(os.getenv("RETRIEVER_K", "4"))


def get_retriever(embeddings, k: int | None = None, persist_dir: str | None = None):
    """构建文档检索器。

    Args:
        embeddings: 向量化模型实例。
        k: 每次检索返回的文本块数量，默认取配置 RETRIEVER_K。
        persist_dir: 向量库存储目录，默认取配置 CHROMA_DIR。

    Returns:
        支持 .invoke() 的检索器对象。
    """
    k = k or DEFAULT_K
    vector_store = get_vector_store(embeddings, persist_dir)
    return vector_store.as_retriever(search_kwargs={"k": k})
