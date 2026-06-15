"""
知识库检索器
为 Researcher Agent 提供 RAG 检索能力

当 RAG_PIPELINE_ENABLED=true 时使用多阶段混合检索流水线（BM25 + 向量 + RRF + Reranker）。
当 RAG_PIPELINE_ENABLED=false 时使用纯向量 ANN 检索（向后兼容）。
"""

from knowledge.embedder import EmbeddingEngine
from knowledge.vector_store import VectorStore
from config.settings import RAG_PIPELINE_ENABLED


# 全局单例（避免重复初始化）
_embedder = None
_store = None
_pipeline = None


def _get_engine():
    global _embedder, _store
    if _embedder is None:
        _embedder = EmbeddingEngine()
    if _store is None:
        _store = VectorStore()
        _store.create_collection()
    return _embedder, _store


def _get_pipeline():
    """获取检索流水线单例（惰性初始化）"""
    global _pipeline
    if _pipeline is None:
        from knowledge.retrieval_pipeline import RetrievalPipeline
        _pipeline = RetrievalPipeline()
    return _pipeline


def retrieve(query: str, top_k: int = 5, **kwargs) -> dict:
    """
    从知识库检索与查询最相关的文档片段

    Args:
        query: 检索查询
        top_k: 返回数量
        **kwargs: 透传给流水线的参数（enable_bm25, enable_reranker 等）

    Returns:
        {"results": [...], "total": N}
    """
    # 环境变量覆盖默认值（支持API运行时动态关闭组件）
    import os as _os
    kwargs.setdefault("enable_ner", _os.getenv("RAG_NER_ENABLED", "true").lower() == "true")
    kwargs.setdefault("enable_query_expansion", _os.getenv("RAG_QUERY_EXPANSION_ENABLED", "true").lower() == "true")
    kwargs.setdefault("enable_bm25", _os.getenv("BM25_ENABLED", "true").lower() == "true")
    kwargs.setdefault("enable_reranker", _os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true")

    # 新流水线模式
    if RAG_PIPELINE_ENABLED:
        try:
            pipeline = _get_pipeline()
            return pipeline.retrieve(query, top_k=top_k, **kwargs)
        except Exception as e:
            print(f"[Retriever] 流水线检索失败，降级为纯向量检索: {e}")
            # 降级到原始逻辑

    # 原始纯向量检索逻辑（向后兼容）
    embedder, store = _get_engine()

    try:
        query_vec = embedder.embed_query(query)
        hits = store.search(query_vec, top_k=top_k)

        results = []
        for h in hits:
            meta = h.get("metadata", {})
            results.append({
                "content": h["content"],
                "score": round(h.get("score", 0), 4),
                "source": meta.get("source", ""),
                "source_url": meta.get("source_url", ""),
                "section_title": meta.get("section_title", ""),
                "has_table": meta.get("has_table", False),
            })

        return {"results": results, "total": len(results)}

    except Exception as e:
        print(f"[Retriever] 检索失败: {e}")
        return {"results": [], "total": 0}


def close():
    """释放资源"""
    global _store, _pipeline
    if _pipeline:
        try:
            _pipeline.close()
        except Exception:
            pass
        _pipeline = None
    if _store:
        _store.close()
        _store = None
