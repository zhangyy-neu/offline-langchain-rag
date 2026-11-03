"""模型初始化模块。
负责初始化：
- 对话大模型：离线模式使用 Ollama（默认 qwen2.5:7b），在线模式使用 DeepSeek
- 向量化模型（默认 qllama/bge-small-zh-v1.5），用于文本向量
"""
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 加载 .env 配置（不存在时使用默认值）
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qllama/bge-small-zh-v1.5")

# 在线 DeepSeek 配置（OpenAI 兼容接口）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def get_llm(mode: str = "offline", temperature: float = 0.1):
    """初始化对话大模型。
    Args:
        mode: 推理模式。
            - offline：使用本地 Ollama 大模型（默认，全程离线）
            - online：使用在线 DeepSeek 大模型（需要配置 DEEPSEEK_API_KEY + langchain-openai）
        temperature: 生成温度，越低回答越稳定，默认 0.1。
    Returns:
        ChatOllama（离线）或 ChatOpenAI（对接DeepSeek）实例。
    """
    if mode == "offline":
        return ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )

    if mode == "online":
        api_key = DEEPSEEK_API_KEY
        if not api_key:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY：请在 .env 文件中填写你的 DeepSeek API Key。"
            )
        # 【延迟导入】只有进入在线模式才加载ChatOpenAI
        try:
            from langchain_openai import ChatOpenAI
        except ModuleNotFoundError:
            raise RuntimeError(
                "缺少依赖包 langchain-openai！请执行命令安装：pip install langchain-openai"
            )
        return ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            temperature=temperature,
        )

    raise ValueError(f"未知的推理模式：{mode}（可选：offline / online）")


def get_embeddings() -> OllamaEmbeddings:
    """初始化文本向量化模型。
    Returns:
        OllamaEmbeddings 实例。
    """
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
