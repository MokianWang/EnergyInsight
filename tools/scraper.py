"""
网页抓取工具
使用 httpx 获取网页内容，BeautifulSoup 解析提取正文
"""

import re
import httpx
from bs4 import BeautifulSoup


def scrape_webpage(url: str, timeout: int = 15) -> dict:
    """
    抓取网页并提取正文内容

    Args:
        url: 目标 URL
        timeout: 请求超时（秒）

    Returns:
        包含 title, url, content 的字典
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

            # 尝试多种编码
            html = response.text
            if not html:
                html = response.content.decode("utf-8", errors="ignore")

    except Exception as e:
        print(f"[爬虫] 抓取失败 {url}: {e}")
        return {"title": "", "url": url, "content": "", "error": str(e)}

    # 解析 HTML
    soup = BeautifulSoup(html, "html.parser")

    # 提取标题
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # 移除无用标签
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # 提取正文（优先找 article 或 main 标签）
    content_el = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"content|article|post|entry", re.I))
        or soup.body
    )

    if content_el:
        # 提取段落文本
        paragraphs = content_el.find_all(["p", "li", "td", "h2", "h3", "h4"])
        text_parts = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text) > 10:  # 过滤太短的文本
                text_parts.append(text)
        content = "\n\n".join(text_parts)
    else:
        content = soup.get_text(separator="\n", strip=True)

    # 清理多余空白
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r" {2,}", " ", content)

    # 截断过长内容
    max_length = 8000
    if len(content) > max_length:
        content = content[:max_length] + "\n\n[内容已截断]"

    return {
        "title": title,
        "url": url,
        "content": content,
    }


def scrape_multiple(urls: list[str], max_count: int = 3) -> list[dict]:
    """
    批量抓取网页

    Args:
        urls: URL 列表
        max_count: 最大抓取数量

    Returns:
        抓取结果列表
    """
    results = []
    for url in urls[:max_count]:
        result = scrape_webpage(url)
        if result.get("content"):
            results.append(result)
    return results
