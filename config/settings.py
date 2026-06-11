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
MAX_REVIEW_ROUNDS = 3
MAX_SEARCH_RESULTS = 5


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
