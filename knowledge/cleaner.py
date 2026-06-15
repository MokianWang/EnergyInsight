"""
文本清洗器
规范化、去重、去除PDF解析噪声
"""

import hashlib
import re


def clean_texts(documents: list[dict]) -> list[dict]:
    """
    清洗文档文本

    操作：控制字符移除 → 空白规范化 → 页眉页脚去除 → 标点统一

    Args:
        documents: 已解析的文档列表

    Returns:
        清洗后的文档列表
    """
    cleaned = []
    for doc in documents:
        content = doc.get("content", "")
        if not content or doc.get("error"):
            continue

        # 1. 移除控制字符
        content = _strip_control_chars(content)

        # 2. 规范化空白
        content = _normalize_whitespace(content)

        # 3. 移除常见PDF页眉页脚噪声
        content = _remove_headers_footers(content)

        # 4. 统一标点符号
        content = _normalize_punctuation(content)

        if len(content) < 50:
            continue

        doc["content"] = content
        doc["char_count"] = len(content)
        cleaned.append(doc)

    print(f"[Cleaner] 清洗完成: {len(cleaned)}/{len(documents)} 个文档")
    return cleaned


def deduplicate_by_hash(documents: list[dict]) -> list[dict]:
    """
    SHA256 + 模糊哈希去重

    Args:
        documents: 清洗后的文档列表

    Returns:
        去重后的文档列表
    """
    seen = {}
    deduped = []

    for doc in documents:
        content = doc.get("content", "")
        # 精确哈希
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # 模糊哈希（首500 + 尾500 + 字符数）
        head = content[:500]
        tail = content[-500:]
        fuzzy = hashlib.sha256(f"{head}|{tail}|{len(content)}".encode("utf-8")).hexdigest()

        if sha in seen:
            # 保留更长的版本
            if len(content) > len(seen[sha]["content"]):
                seen[sha] = {"doc": doc, "content": content}
            continue

        if fuzzy in seen.get("_fuzzy", {}):
            continue

        seen[sha] = {"doc": doc, "content": content}
        seen.setdefault("_fuzzy", {})[fuzzy] = True
        deduped.append(doc)

    removed = len(documents) - len(deduped)
    if removed:
        print(f"[Cleaner] 去重: 移除 {removed} 个重复文档")
    return deduped


def _strip_control_chars(text: str) -> str:
    """移除控制字符，保留换行和制表符"""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)


def _normalize_whitespace(text: str) -> str:
    """规范化空白"""
    # 3个以上换行 → 2个换行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 3个以上空格 → 1个空格
    text = re.sub(r" {2,}", " ", text)
    # 移除行首行尾空白
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)


def _remove_headers_footers(text: str) -> str:
    """移除常见的PDF页眉页脚"""
    patterns = [
        r"\bPage\s+\d+\s+of\s+\d+\b",      # Page 1 of 50
        r"\b第\s*\d+\s*页\s*共\s*\d+\s*页\b",  # 第1页共50页
        r"^\d{1,3}\s*$",                    # 单独的页码
        r"IEA\s*\d{4}\b",                   # IEA 2025
        r"All rights reserved\b",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text


def _normalize_punctuation(text: str) -> str:
    """统一中英文标点符号"""
    # 英文引号 → 中文引号（中文上下文中）
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    # em-dash → 标准横线
    text = text.replace("—", "-").replace("–", "-")
    return text
