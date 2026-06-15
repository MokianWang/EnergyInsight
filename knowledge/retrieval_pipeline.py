"""
多阶段检索流水线编排器

Pipeline Flow:
  Query → NER + Query Expansion
        → [Vector ANN search top-K] + [BM25 keyword search top-K]
        → RRF Fusion → top-K candidates
        → Cross-Encoder Reranking → final top-K
        → Time Weighting (optional)
        → Format Output

每阶段可通过参数独立开关，方便调试和消融实验。
"""

import math


class RetrievalPipeline:
    """
    多阶段检索流水线。

    整合 TerminologyDict, EnergyNER, EmbeddingEngine, VectorStore,
    BM25Retriever, BGEReranker，提供统一检索入口。

    输出格式与 retriever.retrieve() 完全兼容。
    """

    def __init__(
        self,
        embedder=None,    # EmbeddingEngine
        store=None,       # VectorStore
        bm25=None,        # BM25Retriever
        terminology=None, # TerminologyDict
        ner=None,         # EnergyNER
        reranker=None,    # BGEReranker
    ):
        # 惰性导入，避免循环引用
        from knowledge.embedder import EmbeddingEngine
        from knowledge.vector_store import VectorStore

        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.terminology = terminology
        self.ner = ner
        self.reranker = reranker

        # 标记是否需要自行管理组件生命周期
        self._own_embedder = embedder is None
        self._own_store = store is None
        self._own_bm25 = bm25 is None

        # 初始化默认组件
        if self.terminology is None:
            from knowledge.terminology import TerminologyDict
            self.terminology = TerminologyDict()

        if self.ner is None:
            from knowledge.terminology import EnergyNER
            self.ner = EnergyNER(self.terminology)

        # embedder / store 延迟初始化（复用 retriever 的单例模式）

    def _ensure_embedder_store(self):
        """确保 embedder 和 store 已初始化（延迟加载）"""
        if self.embedder is None:
            from knowledge.embedder import EmbeddingEngine
            self.embedder = EmbeddingEngine()
        if self.store is None:
            from knowledge.vector_store import VectorStore
            self.store = VectorStore()
            self.store.create_collection()

    def _ensure_bm25(self):
        """确保 BM25 索引已加载或重建"""
        if self.bm25 is not None and self.bm25.is_ready:
            return

        from knowledge.bm25_retriever import BM25Retriever

        if self.bm25 is None:
            self.bm25 = BM25Retriever(terminology=self.terminology)

        # 尝试从磁盘加载
        if self.bm25.load():
            return

        # 需要重建：从 VectorStore 读取文档
        self._ensure_embedder_store()
        print("[Pipeline] BM25 索引未就绪，从 VectorStore 重建...")
        try:
            self.bm25.rebuild_from_store(self.store)
        except Exception as e:
            print(f"[Pipeline] BM25 索引重建失败: {e}，BM25 检索将不可用")

    def _ensure_reranker(self):
        """确保 reranker 已初始化"""
        if self.reranker is None:
            from knowledge.reranker import BGEReranker
            self.reranker = BGEReranker()

    # ========== 核心入口 ==========

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        enable_ner: bool = True,
        enable_query_expansion: bool = True,
        enable_bm25: bool = True,
        enable_reranker: bool = True,
        enable_time_weight: bool = False,
    ) -> dict:
        """
        全流水线检索。

        Args:
            query: 查询字符串
            top_k: 最终返回条数
            enable_ner: 启用实体识别
            enable_query_expansion: 启用查询扩展
            enable_bm25: 启用 BM25 关键词检索
            enable_reranker: 启用 Cross-Encoder 重排序
            enable_time_weight: 启用时间加权（默认关闭，需要文档含 pub_date）

        Returns:
            {"results": [...], "total": N}
            每条 result:
            {
                "content": str,
                "score": float,
                "source": str,
                "source_url": str,
                "section_title": str,
                "has_table": bool,
            }
        """
        from config.settings import (
            RAG_VECTOR_TOP_K, RAG_BM25_TOP_K, RAG_RERANK_TOP_K,
        )

        self._ensure_embedder_store()

        # ===== Stage 1: NER + Query Expansion =====
        ne_result = {}
        search_query = query

        if enable_ner and self.ner and self.terminology.is_loaded:
            ne_result = self.ner.extract_for_retrieval(query)
            if enable_query_expansion:
                search_query = ne_result.get("expanded_query", query)

        # ===== Stage 2A: 向量检索 =====
        vector_results = self._vector_search(search_query, top_k=RAG_VECTOR_TOP_K)

        # ===== Stage 2B: BM25 检索 =====
        bm25_results = []
        if enable_bm25:
            try:
                self._ensure_bm25()
                if self.bm25 and self.bm25.is_ready:
                    bm25_results = self.bm25.search(search_query, top_k=RAG_BM25_TOP_K)
            except Exception as e:
                print(f"[Pipeline] BM25 检索失败: {e}，仅使用向量检索")

        # ===== Stage 3: RRF 融合 =====
        if bm25_results:
            fused = self._rrf_fuse(vector_results, bm25_results, top_k=RAG_RERANK_TOP_K)
        else:
            fused = vector_results[:RAG_RERANK_TOP_K]

        # ===== Stage 4: Cross-Encoder 重排序 =====
        if enable_reranker:
            try:
                self._ensure_reranker()
                if self.reranker and self.reranker.is_ready:
                    reranked = self.reranker.rerank(search_query, fused, top_k=top_k)
                else:
                    reranked = fused[:top_k]
            except Exception as e:
                print(f"[Pipeline] Reranker 失败: {e}，使用融合结果")
                reranked = fused[:top_k]
        else:
            reranked = fused[:top_k]

        # ===== Stage 5: 时间加权 (bonus) =====
        if enable_time_weight and reranked:
            reranked = self._apply_time_weight(reranked)

        # ===== Stage 6: 格式化输出 =====
        return self._format_results(reranked)

    # ========== 内部方法 ==========

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """
        Embed query → VectorStore.search().

        Returns:
            list[dict]，每条含 chunk_id, doc_id, content, score, metadata
        """
        try:
            query_vec = self.embedder.embed_query(query)
            return self.store.search(query_vec, top_k=top_k)
        except Exception as e:
            print(f"[Pipeline] 向量检索失败: {e}")
            return []

    def _rrf_fuse(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        top_k: int = 50,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF).

        RRF(d) = 1 / (k + rank_v(d)) + 1 / (k + rank_b(d))
        其中 k = RRF constant (默认 60)，rank 从 1 开始。

        Dedup key: doc_id + chunk_index
        """
        from config.settings import RAG_RRF_CONSTANT

        k = RAG_RRF_CONSTANT
        scores: dict[str, dict] = {}

        # 处理向量结果
        for rank, doc in enumerate(vector_results):
            key = f"{doc.get('doc_id', '')}_{doc.get('chunk_index', 0)}"
            rrf = 1.0 / (k + rank + 1)
            scores[key] = {
                "rrf_score": rrf,
                "data": doc,
                "vector_rank": rank + 1,
                "bm25_rank": None,
            }

        # 处理 BM25 结果
        for rank, doc in enumerate(bm25_results):
            key = f"{doc.get('doc_id', '')}_{doc.get('chunk_index', 0)}"
            rrf = 1.0 / (k + rank + 1)
            if key in scores:
                scores[key]["rrf_score"] += rrf
                scores[key]["bm25_rank"] = rank + 1
            else:
                scores[key] = {
                    "rrf_score": rrf,
                    "data": doc,
                    "vector_rank": None,
                    "bm25_rank": rank + 1,
                }

        # 按 RRF 分数降序排列
        sorted_items = sorted(
            scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # 构建结果
        results = []
        for item in sorted_items[:top_k]:
            doc = item["data"].copy()
            doc["rrf_score"] = round(item["rrf_score"], 6)
            doc["vector_rank"] = item["vector_rank"]
            doc["bm25_rank"] = item["bm25_rank"]

            # 将 RRF 分数作为 score（后续 reranker 会覆盖）
            if "score" not in doc or doc.get("rerank_score") is None:
                doc["score"] = doc["rrf_score"]

            results.append(doc)

        return results

    def _apply_time_weight(self, results: list[dict]) -> list[dict]:
        """
        时间加权（bonus 功能）。

        对 pub_date 较新的文档给予更高权重。
        权重公式: time_weight = exp(-decay * years_diff)
        新分数 = 原分数 * time_weight

        无 pub_date 的文档保持原分（time_weight=1.0）。
        """
        from config.settings import RAG_TIME_DECAY_FACTOR
        import time
        import re

        decay = RAG_TIME_DECAY_FACTOR
        current_year = time.localtime().tm_year

        for doc in results:
            meta = doc.get("metadata", {})
            pub_date = meta.get("pub_date", "") or meta.get("source", "")

            # 尝试从文本中提取年份
            year = None
            # 尝试 pub_date 字段（格式: "2025" 或 "2025-06"）
            if isinstance(pub_date, str) and pub_date:
                m = re.search(r'(\d{4})', pub_date)
                if m:
                    year = int(m.group(1))

            if year and 1990 <= year <= current_year + 1:
                years_diff = max(0, current_year - year)
                time_weight = math.exp(-decay * years_diff)
            else:
                time_weight = 1.0

            # 更新分数
            current_score = doc.get("score", 0) or doc.get("rrf_score", 0) or 0
            doc["score"] = round(current_score * time_weight, 4)
            doc["time_weight"] = round(time_weight, 4)

        # 按新分数重排
        results.sort(key=lambda d: d.get("score", 0), reverse=True)
        return results

    def _format_results(self, results: list[dict]) -> dict:
        """
        格式化为 retriever.retrieve() 兼容格式。

        Input: pipeline 内部文档格式 (chunk_id, doc_id, content, score, metadata)
        Output: {"results": [{content, score, source, source_url, section_title, has_table}], "total": N}
        """
        formatted = []
        for doc in results:
            # 兼容两种输入格式
            meta = doc.get("metadata", {})
            formatted.append({
                "content": doc.get("content", ""),
                "score": round(doc.get("score", 0) or 0, 4),
                "source": meta.get("source", ""),
                "source_url": meta.get("source_url", ""),
                "section_title": meta.get("section_title", ""),
                "has_table": meta.get("has_table", False),
            })

        return {"results": formatted, "total": len(formatted)}

    # ========== 维护方法 ==========

    def rebuild_bm25_index(self) -> None:
        """从 VectorStore 重建 BM25 索引并保存到磁盘"""
        self._ensure_embedder_store()
        self._ensure_bm25()  # 会触发重建

    def close(self):
        """释放资源"""
        if self._own_store and self.store:
            self.store.close()
