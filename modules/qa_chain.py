"""问答链模块。

组合 检索器 + Prompt 模板 + 离线大模型，形成完整问答链路：
1. 根据用户问题从 Chroma 检索相关文本块
2. 将问题与上下文拼入 Prompt 模板
3. 由 Ollama 离线大模型生成回答
4. 返回回答与参考来源
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from modules.retriever import get_retriever

PROMPT_TEMPLATE = """你是一个知识库问答助手。请仅依据下面提供的知识库内容回答用户问题。

知识库内容：
{context}

用户问题：{question}

回答要求：
1. 若知识库中存在相关信息，请直接给出准确、简洁的回答；
2. 若知识库中没有相关信息，请明确说明“知识库中未找到相关内容”，不要编造答案；
3. 回答使用中文。
"""


def _format_docs(documents: list) -> str:
    """将检索到的文本块拼接为上下文文本。"""
    return "\n\n".join(f"[来源：{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
                       for doc in documents)


class QAChat:
    """离线知识库问答链。

    Attributes:
        retriever: Chroma 相似度检索器。
        chain: LCEL 问答链（检索 -> 提示词 -> 大模型 -> 文本输出）。
    """

    def __init__(self, llm, embeddings, k: int | None = None):
        """初始化问答链。

        Args:
            llm: 对话大模型实例。
            embeddings: 向量化模型实例。
            k: 检索返回的文本块数量。
        """
        self.retriever = get_retriever(embeddings, k=k)
        self._prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = (
            {
                "context": self.retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | self._prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> tuple[str, list[dict]]:
        """回答用户问题。

        Args:
            question: 用户问题。

        Returns:
            (回答文本, 参考来源列表)。
            参考来源为 [{source: 文件名, content: 文本片段}, ...]。
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        # 检索相关文本块（用于展示来源）
        docs = self.retriever.invoke(question)
        # 生成回答
        answer = self.chain.invoke(question)

        sources = [
            {
                "source": doc.metadata.get("source", "未知"),
                "content": doc.page_content,
            }
            for doc in docs
        ]
        return answer, sources
