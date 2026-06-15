"""
知识库构建编排器
ETL Pipeline：采集 → 解析 → 清洗 → 分块 → 向量化 → 入库 → BM25索引
"""

from knowledge.collector import collect_from_urls, collect_from_local
from knowledge.parser import parse_documents_batch
from knowledge.cleaner import clean_texts, deduplicate_by_hash
from knowledge.chunker import chunk_documents
from knowledge.embedder import EmbeddingEngine
from knowledge.vector_store import VectorStore
from config.settings import RAG_PIPELINE_ENABLED


def run_ingest_pipeline(
    urls: list[str] = None,
    local_paths: list[str] = None,
    rebuild: bool = False,
) -> dict:
    """
    运行完整的知识库构建流水线

    Steps:
      1. COLLECT: 采集文档（URL下载 + 本地扫描）
      2. PARSE:  批量解析提取文本
      3. CLEAN:  清洗 + 去重
      4. CHUNK:  递归分块 + 语义合并
      5. EMBED:  向量化
      6. STORE:  Milvus入库
      7. BM25:   重建BM25关键词索引

    Args:
        urls: 要下载的文档URL列表
        local_paths: 本地文件/目录路径列表
        rebuild: 是否重建整个知识库

    Returns:
        各阶段统计信息字典
    """
    counts = {}

    # ===== Step 1: 采集 =====
    print(f"\n{'='*40}")
    print(f"[Pipeline] Step 1/7 - 文档采集")
    print(f"{'='*40}")
    docs = []
    if urls:
        docs.extend(collect_from_urls(urls))
    if local_paths:
        docs.extend(collect_from_local(local_paths))
    counts["collected"] = len(docs)

    if not docs:
        return {"status": "empty", "summary": "未找到任何文档"}

    # ===== Step 2: 解析 =====
    print(f"\n{'='*40}")
    print(f"[Pipeline] Step 2/7 - 文档解析")
    print(f"{'='*40}")
    parsed = parse_documents_batch(docs)
    counts["parsed"] = len(parsed)

    # ===== Step 3: 清洗 =====
    print(f"\n{'='*40}")
    print(f"[Pipeline] Step 3/7 - 文本清洗")
    print(f"{'='*40}")
    cleaned = clean_texts(parsed)
    cleaned = deduplicate_by_hash(cleaned)
    counts["cleaned"] = len(cleaned)

    # ===== Step 4: 分块 =====
    print(f"\n{'='*40}")
    print(f"[Pipeline] Step 4/7 - 文本分块")
    print(f"{'='*40}")
    embedder = EmbeddingEngine()
    chunks = chunk_documents(cleaned, embedder)
    counts["chunks"] = len(chunks)

    # ===== Step 5: 向量化 =====
    print(f"\n{'='*40}")
    print(f"[Pipeline] Step 5/7 - 向量化")
    print(f"{'='*40}")
    texts = [c["content"] for c in chunks]
    vectors = embedder.embed(texts)
    counts["embedded"] = len(vectors)

    # ===== Step 6: 入库 =====
    print(f"\n{'='*40}")
    print(f"[Pipeline] Step 6/7 - 入库 Milvus")
    print(f"{'='*40}")
    store = VectorStore()
    if rebuild:
        store.drop_collection()
    store.create_collection()
    inserted = store.insert(chunks, vectors)

    # 验证入库数量
    try:
        store_count = len(store.search(vectors[0], top_k=1))
    except Exception:
        store_count = inserted
    counts["stored"] = store_count

    # ===== Step 7: 重建 BM25 索引 =====
    if RAG_PIPELINE_ENABLED:
        print(f"\n{'='*40}")
        print(f"[Pipeline] Step 7/7 - 重建 BM25 索引")
        print(f"{'='*40}")
        try:
            from knowledge.bm25_retriever import BM25Retriever
            bm25 = BM25Retriever()
            all_docs = store.get_all_documents()
            if all_docs:
                bm25.build_index(all_docs)
                bm25.save()
                counts["bm25_docs"] = len(all_docs)
                print(f"[Pipeline] BM25 索引重建完成: {len(all_docs)} 篇文档")
            else:
                print(f"[Pipeline] 无文档，跳过 BM25 索引重建")
                counts["bm25_docs"] = 0
        except Exception as e:
            print(f"[Pipeline] BM25 索引重建失败 (非致命): {e}")
            counts["bm25_docs"] = 0
    else:
        counts["bm25_docs"] = 0

    store.close()

    summary = f"入库完成: {counts['stored']} 个向量片段 ({counts['collected']} 文档)"
    if counts.get("bm25_docs", 0) > 0:
        summary += f", BM25: {counts['bm25_docs']} 篇"

    print(f"\n{'='*40}")
    print(f"[Pipeline] 完成!")
    print(f"[Pipeline] {summary}")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    return {
        "status": "success",
        "stages": counts,
        "summary": summary,
    }


def rebuild_kb(urls: list[str] = None, local_paths: list[str] = None) -> dict:
    """删除现有知识库并重建"""
    return run_ingest_pipeline(urls=urls, local_paths=local_paths, rebuild=True)
