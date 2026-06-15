"""
文本分块器
RecursiveCharacterTextSplitter + 语义合并两阶段策略
能源领域专属分隔符优先级 + 表格检测
"""

import re
import math
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_MERGE_THRESHOLD,
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
)


# 能源报告特化分隔符（按语义重要性排序）
ENERGY_SEPARATORS = [
    "\n## ",       # Markdown H2
    "\n### ",      # Markdown H3
    "\n# ",        # Markdown H1
    "\n---\n",     # 分隔线
    "\n\n\n",      # 大段落间距
    "\n\n",        # 段落间距
    "\n",          # 行间隔
    "。\n",        # 中文句号+换行
    "。",          # 中文句号
    ". ",          # 英文句点
    "；",          # 中文分号
    "！",          # 中文感叹号
    "？",          # 中文问号
    "，",          # 中文逗号
    ", ",          # 英文逗号
    " ",           # 空格（最后手段）
]


def chunk_documents(documents: list[dict], embedder=None) -> list[dict]:
    """
    两阶段分块：递归拆分 → 语义合并

    Args:
        documents: 清洗后的文档列表
        embedder: EmbeddingEngine实例（语义合并需要）

    Returns:
        chunk列表，每项含 doc_id/chunk_index/content/metadata
    """
    print(f"[Chunker] 开始分块 {len(documents)} 个文档...")

    # 阶段1：递归拆分
    splitter = RecursiveCharacterTextSplitter(
        separators=ENERGY_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        keep_separator=True,
    )

    all_chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        section = _find_nearest_section(doc["content"], 0)

        for idx, chunk_text in enumerate(splits):
            # 查找最近的章节标题
            pos = doc["content"].find(chunk_text[:100])
            if pos >= 0:
                section = _find_nearest_section(doc["content"], pos)

            has_table = _detect_table(chunk_text)
            char_count = len(chunk_text)

            all_chunks.append({
                "doc_id": _make_doc_id(doc),
                "chunk_index": idx,
                "content": chunk_text.strip(),
                "metadata": {
                    "source": doc.get("source", ""),
                    "source_url": doc.get("source_url", ""),
                    "section_title": section,
                    "section_level": 2 if section else 0,
                    "has_table": has_table,
                    "char_count": char_count,
                },
            })

    stage1_count = len(all_chunks)
    print(f"[Chunker] 阶段1-递归拆分: {stage1_count} 个chunk")

    # 阶段2：语义合并（需要 embedder）
    if embedder and len(all_chunks) > 1:
        all_chunks = _semantic_merge(all_chunks, embedder)
        stage2_count = len(all_chunks)
        print(f"[Chunker] 阶段2-语义合并: {stage1_count} -> {stage2_count} 个chunk")

    return all_chunks


def _semantic_merge(chunks: list[dict], embedder) -> list[dict]:
    """
    相邻chunk语义合并：相似度 > 阈值则合并
    表格chunk不参与合并
    """
    if len(chunks) <= 1:
        return chunks

    # 批量提取相邻对，一次性计算相似度
    pairs = []
    pair_indices = []
    for i in range(len(chunks) - 1):
        a, b = chunks[i], chunks[i + 1]
        # 表格不合并，不同文档不合并
        if a["metadata"]["has_table"] or b["metadata"]["has_table"]:
            continue
        if a["doc_id"] != b["doc_id"]:
            continue
        if len(a["content"]) + len(b["content"]) > CHUNK_MAX_CHARS:
            continue
        # 太短的chunk直接尝试合并
        if len(a["content"]) < CHUNK_MIN_CHARS or len(b["content"]) < CHUNK_MIN_CHARS:
            pair_indices.append(i)
            pairs.append(True)  # 强制合并
        else:
            pair_indices.append(i)
            pairs.append((a["content"], b["content"]))

    # 批量计算语义相似度
    sims = {}
    texts = [p for p in pairs if isinstance(p, tuple)]
    valid_indices = [pair_indices[j] for j, p in enumerate(pairs) if isinstance(p, tuple)]

    if texts:
        a_texts = [t[0] for t in texts]
        b_texts = [t[1] for t in texts]
        vecs_a = embedder.embed(a_texts)
        vecs_b = embedder.embed(b_texts)

        for j, idx in enumerate(valid_indices):
            sims[idx] = _cosine_sim(vecs_a[j], vecs_b[j])

    # 强制合并太短的
    for j, p in enumerate(pairs):
        if p is True:  # 太短的强制合并
            sims[pair_indices[j]] = 1.0

    # 自底向上合并
    merged = []
    skip = set()
    for i in range(len(chunks)):
        if i in skip:
            continue
        current = chunks[i]
        j = i
        while j < len(chunks) - 1 and j not in skip:
            if sims.get(j, 0) >= CHUNK_MERGE_THRESHOLD:
                nxt = chunks[j + 1]
                if current["doc_id"] == nxt["doc_id"]:
                    merged_len = len(current["content"]) + len(nxt["content"])
                    if merged_len <= CHUNK_MAX_CHARS:
                        current = _merge_two(current, nxt)
                        skip.add(j + 1)
                        j += 1
                        continue
            break
        merged.append(current)

    return merged


def _merge_two(a: dict, b: dict) -> dict:
    """合并两个相邻chunk"""
    return {
        "doc_id": a["doc_id"],
        "chunk_index": a["chunk_index"],
        "content": a["content"] + "\n" + b["content"],
        "metadata": {
            **a["metadata"],
            "char_count": len(a["content"]) + len(b["content"]),
            "has_table": a["metadata"]["has_table"] or b["metadata"]["has_table"],
        },
    }


def _detect_table(text: str) -> bool:
    """检测文本是否包含表格"""
    lines = text.split("\n")
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    tab_lines = sum(1 for ln in lines if ln.count("\t") >= 2)
    num_lines = sum(1 for ln in lines if re.search(r"\d+[\s\|]+\d+", ln))
    return pipe_lines >= 3 or tab_lines >= 3 or num_lines >= 3


def _find_nearest_section(text: str, pos: int) -> str:
    """向前查找最近的章节标题"""
    # 在pos之前查找最近的 ## 或 ### 标题
    prefix = text[:max(pos, 0)]
    matches = list(re.finditer(r"^#{1,3}\s+(.+)$", prefix, re.MULTILINE))
    if matches:
        return matches[-1].group(1).strip()
    return ""


def _make_doc_id(doc: dict) -> str:
    """生成文档短ID"""
    import hashlib
    key = doc.get("source_url", "") or doc.get("local_path", "") or doc.get("title", "")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
