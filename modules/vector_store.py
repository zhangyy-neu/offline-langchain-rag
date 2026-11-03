"""Chroma 向量数据库模块。

负责将文本块向量化后持久化到本地 Chroma，
并提供查询入口（构建检索器时使用）。
"""
import os
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")


def build_vector_store(documents: list, embeddings, persist_dir: str | None = None) -> Chroma:
    """将文档向量化并写入 Chroma（重建索引时调用）。

    语义为“重建索引”：会先清空旧的向量库目录，再写入新数据，
    避免重复执行 build 时向旧库中追加相同文档造成重复向量。

    Args:
        documents: 切分后的文本块列表。
        embeddings: 向量化模型实例。
        persist_dir: 向量库存储目录，默认取配置 CHROMA_DIR。

    Returns:
        Chroma 实例。
    """
    persist_dir = persist_dir or DEFAULT_CHROMA_DIR
    # 重建前清空旧向量库，防止重复索引
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
    )


def get_vector_store(embeddings, persist_dir: str | None = None) -> Chroma:
    """连接已存在的 Chroma 向量库（问答时调用）。

    Args:
        embeddings: 向量化模型实例。
        persist_dir: 向量库存储目录，默认取配置 CHROMA_DIR。

    Returns:
        Chroma 实例。
    """
    persist_dir = persist_dir or DEFAULT_CHROMA_DIR
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
