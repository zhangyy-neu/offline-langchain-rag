"""LangChain 核心程序（命令行入口）。

提供三种用法：
  python app.py build              # 构建 / 重建知识库向量索引
  python app.py ask "问题"          # 单次问答
  python app.py interactive         # 交互式问答

完整流程：
  加载本地文档 -> 文本切分 -> 向量化 -> 写入 Chroma -> 检索问答
"""
import argparse
import os
import sys

from dotenv import load_dotenv

# 确保可从项目根目录导入 modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from modules.loader import load_documents          # noqa: E402
from modules.model import get_embeddings, get_llm  # noqa: E402
from modules.qa_chain import QAChat                # noqa: E402
from modules.splitter import split_documents       # noqa: E402
from modules.vector_store import build_vector_store  # noqa: E402

KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "data/knowledge")


def build_index() -> None:
    """构建 / 重建知识库向量索引。"""
    print(f"[1/3] 加载知识库文档：{KNOWLEDGE_DIR}")
    documents = load_documents(KNOWLEDGE_DIR)
    if not documents:
        print("错误：知识库目录中未找到支持的文档（TXT / PDF / Markdown）。")
        sys.exit(1)
    print(f"      共加载 {len(documents)} 个文档。")

    print("[2/3] 切分文本……")
    chunks = split_documents(documents)
    print(f"      共切分为 {len(chunks)} 个文本块。")

    print("[3/3] 生成向量并写入 Chroma……")
    build_vector_store(chunks, get_embeddings())
    print("完成：知识库索引已构建，可开始问答。")


def ask(question: str) -> None:
    """单次问答。"""
    qa = QAChat(get_llm(), get_embeddings())
    answer, sources = qa.ask(question)
    print(f"问题：{question}\n")
    print(f"回答：{answer}\n")
    print("参考来源：")
    for index, item in enumerate(sources, start=1):
        print(f"  {index}. {item['source']}")


def interactive() -> None:
    """交互式问答。"""
    qa = QAChat(get_llm(), get_embeddings())
    print("离线智能知识库问答已启动（输入 exit / quit 退出）。\n")
    while True:
        try:
            question = input("请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if question.lower() in ("exit", "quit", "q"):
            print("再见。")
            break
        if not question:
            continue
        answer, sources = qa.ask(question)
        print(f"\n回答：{answer}\n")
        print("参考来源：" + "、".join(item["source"] for item in sources) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基于 LangChain 与 Ollama 的离线智能知识库问答系统"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("build", help="构建 / 重建知识库向量索引")
    subparsers.add_parser("interactive", help="进入交互式问答")

    ask_parser = subparsers.add_parser("ask", help="单次问答")
    ask_parser.add_argument("question", help="要提问的问题")

    args = parser.parse_args()

    if args.command == "build":
        build_index()
    elif args.command == "ask":
        ask(args.question)
    elif args.command == "interactive":
        interactive()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
