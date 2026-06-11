"""
MCP (Model Context Protocol) 工具管理

混合方案：
- 搜索：阿里云百炼 WebSearch MCP（Streamable HTTP 云端，夸克引擎，中文搜索质量高）
- 抓取：Playwright MCP（本地 stdio，JS 渲染能力强）
- PDF：PyMuPDF（本地 Python 库）

降级策略：
- 无 QWEN_API_KEY → DuckDuckGo MCP（本地 stdio）
- MCP 全部不可用 → 自定义 httpx + BeautifulSoup 工具
"""

import asyncio
import shutil

from config.settings import MCP_SERVERS, MCP_ENABLED, QWEN_API_KEY


async def get_mcp_client():
    """
    创建 MCP 多服务客户端

    自动根据 MCP_SERVERS 配置连接 SSE 和 stdio 两种类型的 Server。

    用法:
        client = await get_mcp_client()
        tools = await client.get_tools()
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    if not MCP_ENABLED:
        raise RuntimeError("MCP 已禁用（MCP_ENABLED=false）")

    clean_servers = {
        name: {k: v for k, v in cfg.items() if k != "description"}
        for name, cfg in MCP_SERVERS.items()
    }
    client = MultiServerMCPClient(clean_servers)
    return client


def get_mcp_tools_sync():
    """
    同步方式获取 MCP 工具列表

    内部启动事件循环，阻塞直到所有 MCP Server 启动完成。
    返回的 tools 在整个研究流程中保持有效。

    Returns:
        (tools, cleanup) 元组：工具列表 + 清理函数
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # 过滤掉非 MCP 标准字段（如 description），避免参数错误
    clean_servers = {
        name: {k: v for k, v in cfg.items() if k != "description"}
        for name, cfg in MCP_SERVERS.items()
    }

    async def _get_tools():
        client = MultiServerMCPClient(clean_servers)
        tools = await client.get_tools()
        return tools, client

    loop = asyncio.new_event_loop()
    tools, client = loop.run_until_complete(_get_tools())

    def cleanup():
        loop.close()

    return tools, cleanup


def list_mcp_servers() -> list[dict]:
    """
    列出当前配置的 MCP Server 信息

    Returns:
        Server 信息列表
    """
    servers = []
    for name, cfg in MCP_SERVERS.items():
        is_sse = "url" in cfg
        server_info = {
            "name": name,
            "transport": "sse (云端)" if is_sse else "stdio (本地)",
            "description": cfg.get("description", ""),
        }
        if is_sse:
            server_info["url"] = cfg["url"]
        else:
            server_info["command"] = f"{cfg['command']} {' '.join(cfg['args'])}"
        servers.append(server_info)
    return servers


def check_dependencies() -> dict:
    """
    检查所有 MCP Server 所需的运行时依赖

    Returns:
        依赖检查结果字典
    """
    results = {}

    # 阿里云百炼 WebSearch MCP（Streamable HTTP）
    results["websearch"] = {
        "required": bool(QWEN_API_KEY),
        "status": "已配置 (QWEN_API_KEY)" if QWEN_API_KEY else "未配置 (降级到 DuckDuckGo)",
        "action_needed": "" if QWEN_API_KEY else "设置 QWEN_API_KEY 以启用阿里云搜索",
    }

    # Node.js / npx（Playwright MCP 需要）
    npx_path = shutil.which("npx")
    results["npx"] = {
        "installed": npx_path is not None,
        "path": npx_path or "",
        "required_by": "playwright",
        "action_needed": "" if npx_path else "安装 Node.js: https://nodejs.org/",
    }

    # uv / uvx（DuckDuckGo MCP 降级方案需要）
    uv_path = shutil.which("uvx") or shutil.which("uv")
    results["uv"] = {
        "installed": uv_path is not None,
        "path": uv_path or "",
        "required_by": "duckduckgo (降级方案)",
        "action_needed": "" if uv_path or QWEN_API_KEY else "安装 uv: https://docs.astral.sh/uv/",
    }

    return results
