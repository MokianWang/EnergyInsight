"""
MCP (Model Context Protocol) 工具管理

混合方案：
- 搜索：阿里云百炼 WebSearch MCP（Streamable HTTP 云端，夸克引擎）
- 抓取：Playwright MCP（本地 stdio，JS 渲染）
- PDF：PyMuPDF（本地 Python 库）

降级策略：无 QWEN_API_KEY → DuckDuckGo MCP → httpx + BeautifulSoup
"""

import asyncio

from config.settings import MCP_SERVERS, MCP_ENABLED


def get_mcp_tools_sync():
    """
    同步获取 MCP 工具列表。

    内部启动事件循环，阻塞直到所有 MCP Server 启动完成。
    返回的 tools 在 Researcher 中保持有效。

    Returns:
        (tools, cleanup) 元组：工具列表 + 清理函数
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    if not MCP_ENABLED:
        raise RuntimeError("MCP 已禁用（MCP_ENABLED=false）")

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
