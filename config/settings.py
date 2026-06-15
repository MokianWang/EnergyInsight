"""
EnergyInsight 配置管理
支持 Qwen (通义千问/DashScope) 和 DeepSeek 两种 LLM 提供商
两者均兼容 OpenAI 接口，通过 ChatOpenAI 统一调用

MCP 工具采用混合方案：
- 搜索：阿里云百炼 WebSearch MCP（Streamable HTTP 云端，中文搜索质量高，与 Qwen 共用 API Key）
- 抓取：Playwright MCP（本地 stdio，免费，JS 渲染能力强）
- PDF：PyMuPDF（本地 Python 库，零成本）
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


# ========== LLM 配置 ==========
# 提供商：qwen | deepseek
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Qwen (阿里云 DashScope)
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


# ========== 提供商配置表 ==========
_PROVIDERS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "QWEN_API_KEY",
        "default_model": "qwen-plus",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
}


# ========== MCP Server 配置 ==========
# 是否启用 MCP（关闭则降级到自定义 httpx 工具）
MCP_ENABLED = os.getenv("MCP_ENABLED", "true").lower() == "true"


def get_mcp_servers() -> dict:
    """
    动态构建 MCP Server 配置

    搜索工具优先级：
    1. 阿里云百炼 WebSearch MCP（有 QWEN_API_KEY 时使用，中文搜索质量最佳）
    2. DuckDuckGo MCP（本地 stdio，需安装 uv，降级方案）

    抓取工具：
    - Playwright MCP（本地 stdio，需安装 Node.js/npx）

    Returns:
        MCP Server 配置字典
    """
    servers = {}

    # ---- 搜索工具 ----
    if QWEN_API_KEY:
        # 优先使用阿里云百炼 WebSearch MCP（Streamable HTTP 云端，中文效果好）
        servers["websearch"] = {
            "transport": "http",
            "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
            "headers": {
                "Authorization": f"Bearer {QWEN_API_KEY}",
            },
            "description": "阿里云百炼联网搜索（夸克引擎，中文优化）",
        }
    else:
        # 降级到本地 DuckDuckGo MCP（需安装 uv）
        servers["duckduckgo"] = {
            "transport": "stdio",
            "command": "uvx",
            "args": ["duckduckgo-mcp-server"],
            "description": "DuckDuckGo 本地搜索（免费，中文效果一般）",
        }

    # ---- 网页抓取工具 ----
    servers["playwright"] = {
        "transport": "stdio",
        "command": "npx",
        "args": ["@playwright/mcp@latest", "--headless"],
        "description": "Playwright 浏览器自动化（无头模式，后台静默运行）",
    }

    return servers


# 模块加载时获取配置（供其他模块直接 import）
MCP_SERVERS = get_mcp_servers()


# ========== Agent 配置 ==========
MAX_SEARCHES_PER_QUESTION = 5
MAX_SEARCH_RESULTS = 5

def get_max_review_rounds():
    """动态读取审查轮次（支持运行时通过环境变量修改）"""
    return int(os.getenv("MAX_REVIEW_ROUNDS", "3"))

def get_max_replan_rounds():
    """动态读取重规划轮次"""
    return int(os.getenv("MAX_REPLAN_ROUNDS", "2"))


# ========== Step 2: Knowledge Base 配置 ==========
# Milvus 向量库
MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", "data/milvus/milvus_lite.db")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "energy_docs")
MILVUS_VECTOR_DIM = int(os.getenv("MILVUS_VECTOR_DIM", "1536"))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
CHUNK_MERGE_THRESHOLD = float(os.getenv("CHUNK_MERGE_THRESHOLD", "0.75"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "2000"))
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "100"))

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "25"))


# ========== Step 3: RAG Pipeline 配置 ==========
# 主开关
RAG_PIPELINE_ENABLED = os.getenv("RAG_PIPELINE_ENABLED", "true").lower() == "true"

# 双路召回
RAG_VECTOR_TOP_K = int(os.getenv("RAG_VECTOR_TOP_K", "50"))       # 向量召回候选数
RAG_BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", "50"))           # BM25 召回候选数

# RRF 融合
RAG_RRF_CONSTANT = int(os.getenv("RAG_RRF_CONSTANT", "60"))       # RRF k 常数
RAG_RERANK_TOP_K = int(os.getenv("RAG_RERANK_TOP_K", "50"))       # 送重排序的候选数

# 重排序
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))

# 时间加权
RAG_TIME_DECAY_FACTOR = float(os.getenv("RAG_TIME_DECAY_FACTOR", "0.1"))

# 术语词典
TERMINOLOGY_DICT_PATH = os.getenv(
    "TERMINOLOGY_DICT_PATH",
    "knowledge/terminology_dict.json",
)

# BM25 索引
BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", "data/milvus/bm25_index/")

# ========== Step 5: 能源计算工具 ==========
PYPSA_ENABLED = os.getenv("PYPSA_ENABLED", "true").lower() == "true"
CARBON_PRICE_CACHE_TTL = int(os.getenv("CARBON_PRICE_CACHE_TTL", "3600"))


def get_embedding(texts: list[str]) -> list[list[float]]:
    """
    调用千问 DashScope Embedding API，获取文本向量

    Args:
        texts: 文本列表，每批最多 64 条

    Returns:
        向量列表，每条 1024 维
    """
    import httpx
    import time

    if not QWEN_API_KEY:
        raise ValueError("QWEN_API_KEY 未设置")

    url = f"{_PROVIDERS['qwen']['base_url']}/embeddings"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }

    all_vectors = []
    batch_size = min(EMBEDDING_BATCH_SIZE, 64)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # 截断过长文本（API token 限制）
        batch = [t[:6000] for t in batch]

        payload = {
            "model": EMBEDDING_MODEL,
            "input": batch,
        }
        # text-embedding-v3 supports dimensions param, v2/v1 don't need it
        if "v3" in EMBEDDING_MODEL:
            payload["dimensions"] = EMBEDDING_DIM

        for attempt in range(3):
            try:
                with httpx.Client(timeout=60) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    all_vectors.extend(
                        [item["embedding"] for item in data["data"]]
                    )
                break
            except Exception as e:
                err_msg = f"Embedding API 调用失败 (attempt {attempt+1}/3): {e}"
                if len(batch) > 1:
                    err_msg += f" | batch[{i}:{i+len(batch)}] len={[len(t) for t in batch[:3]]}..."
                print(err_msg)
                if attempt == 2:
                    raise RuntimeError(err_msg)
                time.sleep(2 * (attempt + 1))

    return all_vectors


def get_llm(temperature: float = None):
    """
    获取 LLM 实例，根据 LLM_PROVIDER 自动选择提供商

    Args:
        temperature: 覆盖默认温度参数

    Returns:
        ChatModel 实例
    """
    from langchain_openai import ChatOpenAI

    temp = temperature if temperature is not None else LLM_TEMPERATURE

    provider = LLM_PROVIDER.lower()
    if provider not in _PROVIDERS:
        supported = ", ".join(_PROVIDERS.keys())
        raise ValueError(f"不支持的 LLM_PROVIDER: {provider}，可选: {supported}")

    cfg = _PROVIDERS[provider]
    api_key = os.getenv(cfg["key_env"], "")

    if not api_key:
        raise ValueError(
            f"{cfg['key_env']} 未设置，请在 .env 文件中配置"
        )

    model = LLM_MODEL if LLM_MODEL else cfg["default_model"]

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=cfg["base_url"],
        temperature=temp,
    )
