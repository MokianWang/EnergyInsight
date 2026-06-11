"""
搜索工具
使用 DuckDuckGo 作为降级搜索方案（MCP 不可用时）
"""


def duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """
    DuckDuckGo 搜索

    Args:
        query: 搜索查询
        max_results: 最大结果数

    Returns:
        搜索结果列表
    """
    from ddgs import DDGS

    results = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("href", ""),
                "content": item.get("body", ""),
                "score": 0,
            })

    return results


def search(query: str, max_results: int = 5) -> list[dict]:
    """
    统一搜索接口：DuckDuckGo

    Args:
        query: 搜索查询
        max_results: 最大结果数

    Returns:
        搜索结果列表
    """
    try:
        return duckduckgo_search(query=query, max_results=max_results)
    except Exception as e:
        print(f"[搜索] DuckDuckGo 搜索失败: {e}")
        return []


def energy_search(query: str, max_results: int = 5) -> list[dict]:
    """
    能源行业搜索：自动添加能源相关域名限定，提高结果相关性

    Args:
        query: 搜索查询
        max_results: 最大结果数

    Returns:
        搜索结果列表
    """
    # 能源行业高质量域名
    energy_domains = [
        "nea.gov.cn",        # 国家能源局
        "ndrc.gov.cn",       # 国家发改委
        "iea.org",           # 国际能源署
        "irena.org",         # 国际可再生能源机构
        "bp.com",            # BP 能源统计
        "energy.gov",        # 美国能源部
        "eia.gov",           # 美国能源信息管理局
    ]

    # 先用通用搜索获取结果
    results = search(query=query, max_results=max_results * 2)

    # 优先返回能源域名匹配的结果
    energy_results = []
    general_results = []
    for r in results:
        url = r.get("url", "")
        if any(d in url for d in energy_domains):
            energy_results.append(r)
        else:
            general_results.append(r)

    # 能源结果优先，不足时补充通用结果
    combined = energy_results + general_results
    return combined[:max_results]
