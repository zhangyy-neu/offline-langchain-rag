"""Streamlit 网页界面。

运行方式：
  streamlit run web.py

功能：
- 查看当前模型与知识库配置
- 上传 TXT / PDF / Markdown 文档到知识库
- 一键构建 / 重建向量索引
- 输入问题，基于知识库生成回答并展示参考来源
"""
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# 确保可从项目根目录导入 modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from modules.loader import load_documents          # noqa: E402
from modules.model import get_embeddings, get_llm  # noqa: E402
from modules.qa_chain import QAChat                # noqa: E402
from modules.splitter import split_documents       # noqa: E402
from modules.vector_store import build_vector_store  # noqa: E402

KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", "data/knowledge"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "data/chroma"))
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qllama/bge-small-zh-v1.5")

SUPPORTED_UPLOAD_TYPES = ["txt", "md", "markdown", "pdf"]

st.set_page_config(
    page_title="离线智能知识库问答",
    page_icon="📚",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": "基于 LangChain 与 Ollama 的离线智能知识库问答系统",
    },
)

# 将 Streamlit 原生英文界面覆盖为中文
st.markdown(
    """
    <style>
    /* 隐藏右上角 Deploy 按钮 */
    [data-testid="stAppDeployButton"] { display: none !important; }

    /* 文件上传框中文化：按钮 */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        font-size: 0px !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"]::after {
        content: "浏览文件" !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #1668dc !important;
    }

    /* 文件上传框中文化：拖拽提示（隐藏原生英文提示文字） */
    [data-testid="stFileUploaderDropzone"] div {
        font-size: 0px !important;
        line-height: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"]::after {
        content: "将文件拖拽到此处，或点击上方「浏览文件」选择（单个文件 ≤ 200MB）" !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        color: #555555 !important;
        display: block !important;
        text-align: center !important;
        padding: 10px 0 !important;
    }

    /* 隐藏上传框内剩余英文提示（200MB per file 等） */
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        font-size: 0px !important;
    }

    /* 上传按钮中文化 */
    [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] p {
        font-size: 0px !important;
        line-height: 0 !important;
    }
    [data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"]::after {
        content: "上传" !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    /* 隐藏右上角菜单中的英文帮助项 */
    [data-testid="stMainMenu"] [role="menuitem"]:nth-child(-n+1) { display: none !important; }

    /* ===== Streamlit 内置菜单（右上角三点）中文化：保持原生布局 ===== */
    /* 菜单整体宽度与换行保护，避免中文文字被挤压换行 */
    [data-testid="stMainMenuPopover"] { min-width: 300px !important; }
    [data-testid="stMainMenuPopover"] [data-testid^="stMainMenuItem-"] { white-space: nowrap !important; }

    /* 主题按钮（跟随系统/浅色/深色）：隐藏英文直接文本，保留图标，中文追加在图标后 */
    [data-testid="stMainMenuItem-theme-System"],
    [data-testid="stMainMenuItem-theme-Light"],
    [data-testid="stMainMenuItem-theme-Dark"] { font-size: 0px !important; }
    [data-testid^="stMainMenuItem-theme-"] [data-testid="stIconMaterial"] { font-size: 24px !important; }
    [data-testid="stMainMenuItem-theme-System"]::after { content: "跟随系统" !important; font-size: 14px !important; }
    [data-testid="stMainMenuItem-theme-Light"]::after { content: "浅色" !important; font-size: 14px !important; }
    [data-testid="stMainMenuItem-theme-Dark"]::after { content: "深色" !important; font-size: 14px !important; }

    /* 列表项：隐藏英文文字，保留快捷键（R / C），中文显示在项左侧 */
    [data-testid^="stMainMenuItem-"] span[data-testid="stMainMenuItemLabel"] { font-size: 0px !important; }
    [data-testid="stMainMenuItem-rerun"]::before { content: "重新运行" !important; font-size: 14px !important; padding-right: 6px !important; }
    [data-testid="stMainMenuItem-clearCache"]::before { content: "清除缓存" !important; font-size: 14px !important; padding-right: 6px !important; }
    [data-testid="stMainMenuItem-print"]::before { content: "打印" !important; font-size: 14px !important; padding-right: 6px !important; }
    [data-testid="stMainMenuItem-recordScreencast"]::before { content: "录制屏幕" !important; font-size: 14px !important; padding-right: 6px !important; }
    [data-testid="stMainMenuItem-about"]::before { content: "关于" !important; font-size: 14px !important; padding-right: 6px !important; }

    /* 自动重新运行：隐藏按钮直接英文文本，中文在前，开关保持在右侧 */
    [data-testid="stMainMenuItem-autoRerun"] { font-size: 0px !important; }
    [data-testid="stMainMenuItem-autoRerun"]::before { content: "自动重新运行" !important; font-size: 14px !important; padding-right: 6px !important; }
    [data-testid="stMainMenuItem-autoRerun"] > div { font-size: 14px !important; }

    /* 页脚：英文版权信息替换为中文，保留原生底部布局 */
    [data-testid="stMainMenuPopover"] .st-emotion-cache-59i5nt span { font-size: 0px !important; }
    [data-testid="stMainMenuPopover"] .st-emotion-cache-59i5nt span::after { content: "由 Streamlit 驱动" !important; font-size: 12px !important; }

    /* ===== 录屏弹窗中文化（Streamlit 内置，限定在含录屏选项的对话框内） ===== */
    [data-testid="stDialog"]:has([data-testid="stScreencastAudioCheckbox"]) h2 { font-size: 0px !important; }
    [data-testid="stDialog"]:has([data-testid="stScreencastAudioCheckbox"]) h2::after { content: "录制屏幕" !important; font-size: 22px !important; font-weight: 700 !important; }
    [data-testid="stDialog"]:has([data-testid="stScreencastAudioCheckbox"]) > div > section > div > p:not(:has(label)) { font-size: 0px !important; line-height: 0 !important; }
    [data-testid="stDialog"]:has([data-testid="stScreencastAudioCheckbox"]) > div > section > div > p:not(:has(label))::after { content: "将录制您的屏幕画面，方便分享您所看到的内容。" !important; font-size: 14px !important; line-height: 1.5 !important; }
    [data-testid="stScreencastInstruction"] p { font-size: 0px !important; line-height: 0 !important; }
    [data-testid="stScreencastInstruction"] p::after { content: "按 Esc 键可随时停止录制。" !important; font-size: 14px !important; line-height: 1.5 !important; }
    [data-testid="stScreencastAudioCheckbox"] { font-size: 0px !important; }
    [data-testid="stScreencastAudioCheckbox"]::after { content: "同时录制音频" !important; font-size: 14px !important; }
    [data-testid="stDialog"]:has([data-testid="stScreencastAudioCheckbox"]) button[data-testid="stBaseButton-secondary"] { font-size: 0px !important; }
    [data-testid="stDialog"]:has([data-testid="stScreencastAudioCheckbox"]) button[data-testid="stBaseButton-secondary"]::after { content: "开始录制" !important; font-size: 14px !important; font-weight: 600 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("基于 LangChain 的智能知识库问答系统")
st.caption("本地 RAG 应用：离线 Ollama 与在线 DeepSeek 双模式，知识库检索全程在本地完成")


def get_qa(mode: str):
    """获取（或创建）问答链实例，保存在会话状态中。

    按推理模式分别缓存，切换模式后会自动重建对应问答链。
    构建失败（如在线模式未配置 API Key）时返回 None 并在界面提示。
    """
    key = f"qa_chat_{mode}"
    if key not in st.session_state:
        try:
            st.session_state[key] = QAChat(get_llm(mode=mode), get_embeddings())
        except RuntimeError as exc:
            st.error(f"模型初始化失败：{exc}")
            return None
        except Exception as exc:  # noqa: BLE001
            st.error(f"模型初始化异常：{exc}")
            return None
    return st.session_state[key]


def rebuild_index(mode: str) -> int:
    """重建向量索引，返回切分后的文本块数量。"""
    documents = load_documents(str(KNOWLEDGE_DIR))
    if not documents:
        st.error("知识库中没有文档，请先上传文档。")
        return 0
    chunks = split_documents(documents)
    with st.spinner(f"正在生成向量并写入 Chroma（共 {len(chunks)} 个文本块）……"):
        build_vector_store(chunks, get_embeddings())
    # 索引已变化，重建当前模式问答链（其他模式会在切换时按需创建）；
    # 若问答链初始化失败（如在线模式未配置 Key），不影响索引本身，仅提示。
    try:
        st.session_state[f"qa_chat_{mode}"] = QAChat(get_llm(mode=mode), get_embeddings())
    except RuntimeError as exc:
        st.warning(f"索引已构建，但问答链初始化失败：{exc}")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"索引已构建，但问答链初始化异常：{exc}")
    return len(chunks)


def save_uploaded_files(uploaded_files) -> int:
    """保存上传的文档到知识库目录，返回保存数量。"""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for uploaded in uploaded_files:
        ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
        if ext not in SUPPORTED_UPLOAD_TYPES:
            st.warning(f"跳过不支持的文件：{uploaded.name}（仅支持 TXT / PDF / Markdown）")
            continue
        # 仅取文件名部分，防止路径穿越
        filename = Path(uploaded.name).name
        target = KNOWLEDGE_DIR / filename
        target.write_bytes(uploaded.getbuffer())
        saved += 1
    return saved


# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("模型与配置")
    llm_mode = st.selectbox(
        "推理模式",
        options=["离线模式（本地 Ollama）", "在线模式（DeepSeek）"],
        index=0,
        help="离线模式使用本地 qwen2.5:7b，完全离线；在线模式调用 DeepSeek 大模型，需在 .env 配置 DEEPSEEK_API_KEY。",
    )
    run_mode = "offline" if "离线" in llm_mode else "online"
    st.write(f"- 当前模式：`{llm_mode}`")
    st.write(f"- 对话模型：`{LLM_MODEL}`" if run_mode == "offline" else "- 对话模型：`deepseek-chat（在线）`")
    st.write(f"- 向量模型：`{EMBEDDING_MODEL}`")
    st.write(f"- 知识库目录：`{KNOWLEDGE_DIR}`")
    st.write(f"- 向量库目录：`{CHROMA_DIR}`")

    st.divider()
    st.header("知识库管理")

    uploaded_files = st.file_uploader(
        "上传文档（TXT / PDF / Markdown，可多选）",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("保存上传的文档"):
        count = save_uploaded_files(uploaded_files)
        if count > 0:
            st.success(f"已保存 {count} 个文档到知识库，点击下方按钮重建索引。")

    if st.button("重建知识库索引", use_container_width=True):
        chunk_count = rebuild_index(run_mode)
        if chunk_count > 0:
            st.success(f"索引构建完成，共 {chunk_count} 个文本块。")

    existing_files = (
        [p.name for p in KNOWLEDGE_DIR.iterdir() if p.is_file()]
        if KNOWLEDGE_DIR.exists()
        else []
    )
    st.divider()
    st.header("知识库文件")
    if existing_files:
        st.write(f"共 {len(existing_files)} 个文件：")
        for name in existing_files:
            st.write(f"- {name}")
    else:
        st.info("知识库目录为空，请上传文档后重建索引。")

# ---------- 主区域：问答 ----------
st.header("开始问答")

question = st.text_input("请输入你的问题：", placeholder="例如：本系统的运行环境要求是什么？")
ask_clicked = st.button("提问", type="primary")

if ask_clicked:
    question = question.strip()
    if not question:
        st.warning("请输入问题。")
    else:
        try:
            qa = get_qa(run_mode)
            if qa is None:
                st.info("请先在侧边栏切换其他推理模式，或根据上方错误提示处理后重试。")
            else:
                with st.spinner("正在检索知识库并生成回答……"):
                    answer, sources = qa.ask(question)
                st.markdown("#### 回答")
                st.markdown(answer)
                if sources:
                    with st.expander(f"参考来源（{len(sources)} 条）"):
                        for index, item in enumerate(sources, start=1):
                            st.markdown(f"**{index}. {item['source']}**")
                            st.markdown(f"> {item['content']}")
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)
            if "401" in err_msg or "Authorization" in err_msg or "authentication" in err_msg.lower():
                st.error("DeepSeek API 鉴权失败（401）：请检查 .env 中的 DEEPSEEK_API_KEY 是否正确、是否已过期，修改后需重启应用。")
            elif run_mode == "online":
                st.error(f"在线模式问答失败：{err_msg}")
                st.info("请确认：1) .env 中已配置 DEEPSEEK_API_KEY；2) 网络可访问 api.deepseek.com；3) 账户余额或免费额度充足。")
            else:
                st.error(f"问答失败：{exc}")
                st.info("请确认：1) Ollama 服务已启动；2) 已拉取所需模型；3) 已重建知识库索引。")
