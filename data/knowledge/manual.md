# Ollama 离线部署与使用手册（示例知识库）

## 一、Ollama 简介

Ollama 是一个本地大语言模型运行工具，支持在个人电脑上直接运行 Llama、Qwen、Gemma 等开源模型，无需联网调用云端 API，数据完全保存在本地。本项目通过 Ollama 提供离线对话大模型与文本向量模型。

## 二、安装与启动

1. 从官网下载对应操作系统的安装包（Windows / macOS / Linux）。
2. 安装完成后，在终端执行 `ollama --version` 验证安装成功。
3. 启动服务：Windows 与 macOS 安装后服务默认后台运行，Linux 可执行 `ollama serve` 手动启动。
4. 服务默认监听 `http://localhost:11434`，可通过环境变量 `OLLAMA_HOST` 修改监听地址。

## 三、拉取模型

本项目需要两个模型：

- 对话大模型：`ollama pull qwen2.5:7b`（约 4.7 GB，用于生成回答）
- 向量模型：`ollama pull nomic-embed-text`（约 274 MB，用于文本向量化）

查看已安装模型：`ollama list`。删除模型：`ollama rm <模型名>`。

## 四、常用命令

| 命令 | 说明 |
| --- | --- |
| ollama list | 列出本地已安装的模型 |
| ollama pull <模型名> | 下载模型到本地 |
| ollama run <模型名> | 在终端直接与模型对话 |
| ollama rm <模型名> | 删除本地模型 |
| ollama serve | 启动服务（Linux） |

## 五、模型与硬件要求

- qwen2.5:7b 建议至少 8 GB 内存，4 GB 以上显存可获得较好速度；纯 CPU 也可运行，速度较慢。
- nomic-embed-text 体积小，对硬件要求低。
- 完全离线运行的前提：首次拉取模型时需要联网，之后所有推理与向量化均在本地完成。

## 六、常见问题

问：提示连接 localhost:11434 失败怎么办？
答：确认 Ollama 服务已启动，执行 `ollama list` 验证；检查 11434 端口是否被占用。

问：模型回答速度很慢怎么办？
答：可改用更小的模型（如 qwen2.5:3b），或关闭其他占用内存的程序。

问：如何更换其他模型？
答：修改项目 `.env` 文件中的 `LLM_MODEL` 和 `EMBEDDING_MODEL`，并重新构建索引。
