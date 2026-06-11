"""
PDF 解析工具
使用 PyMuPDF (fitz) 提取 PDF 文本内容
"""

import os
import httpx
import fitz  # PyMuPDF


def parse_pdf_from_path(file_path: str, max_pages: int = 50) -> dict:
    """
    从本地路径解析 PDF

    Args:
        file_path: PDF 文件路径
        max_pages: 最大解析页数

    Returns:
        包含 filename, pages, content 的字典
    """
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"[PDF] 打开文件失败 {file_path}: {e}")
        return {"filename": file_path, "pages": 0, "content": "", "error": str(e)}

    pages_text = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text = page.get_text()
        if text.strip():
            pages_text.append(f"--- 第 {i + 1} 页 ---\n{text.strip()}")

    doc.close()

    content = "\n\n".join(pages_text)

    # 截断过长内容
    max_length = 20000
    if len(content) > max_length:
        content = content[:max_length] + "\n\n[内容已截断]"

    return {
        "filename": os.path.basename(file_path),
        "pages": len(pages_text),
        "content": content,
    }


def parse_pdf_from_url(url: str, max_pages: int = 50, timeout: int = 30) -> dict:
    """
    从 URL 下载并解析 PDF

    Args:
        url: PDF 文件 URL
        max_pages: 最大解析页数
        timeout: 下载超时（秒）

    Returns:
        包含 url, pages, content 的字典
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            pdf_bytes = response.content
    except Exception as e:
        print(f"[PDF] 下载失败 {url}: {e}")
        return {"url": url, "pages": 0, "content": "", "error": str(e)}

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        print(f"[PDF] 解析失败 {url}: {e}")
        return {"url": url, "pages": 0, "content": "", "error": str(e)}

    pages_text = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text = page.get_text()
        if text.strip():
            pages_text.append(f"--- 第 {i + 1} 页 ---\n{text.strip()}")

    doc.close()

    content = "\n\n".join(pages_text)

    max_length = 20000
    if len(content) > max_length:
        content = content[:max_length] + "\n\n[内容已截断]"

    return {
        "url": url,
        "pages": len(pages_text),
        "content": content,
    }
