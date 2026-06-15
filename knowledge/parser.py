"""
文档解析器
批量解析PDF/HTML/TXT，复用 tools/pdf_parser.py
"""

import re
from bs4 import BeautifulSoup
from tools.pdf_parser import parse_pdf_from_path


def parse_documents_batch(documents: list[dict]) -> list[dict]:
    """
    批量解析文档

    Args:
        documents: 文档元数据列表（来自 collector）

    Returns:
        已解析的文档列表，含 content 字段
    """
    parsed = []

    for doc in documents:
        path = doc["local_path"]
        file_type = doc["file_type"]

        try:
            if file_type == "pdf":
                result = parse_pdf_from_path(path, max_pages=100)
                content = result.get("content", "")
                page_count = result.get("pages", 0)
                error = result.get("error", "")

                if error or (page_count > 0 and len(content) < 100):
                    print(f"[Parser] 扫描件或无文本: {doc['title']}")
                    doc["error"] = error or "无文本内容（可能是扫描件）"

            elif file_type in ("html", "htm"):
                # 本地HTML直接读取，用BeautifulSoup提取正文
                content = _parse_html_file(path)
                page_count = 1

            elif file_type in ("txt", "md"):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                page_count = 1

            else:
                content = ""
                page_count = 0
                doc["error"] = f"不支持的文件类型: {file_type}"

            doc["content"] = content
            doc["char_count"] = len(content)
            doc["page_count"] = page_count

            if content:
                parsed.append(doc)

        except Exception as e:
            print(f"[Parser] 解析失败 {doc['title']}: {e}")
            doc["error"] = str(e)
            doc["content"] = ""

    print(f"[Parser] 已解析: {len(parsed)}/{len(documents)} 个文档")
    return parsed


def _parse_html_file(filepath: str) -> str:
    """解析本地HTML文件，提取正文文本"""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # 移除无用标签
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # 提取正文
    content_el = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"content|article|main|TRS_Editor", re.I))
        or soup.body
    )

    if content_el:
        text = content_el.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # 清理多余空白
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text
