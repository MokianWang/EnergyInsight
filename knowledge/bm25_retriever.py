"""
BM25 关键词检索器

使用 rank_bm25 + jieba 中文分词，支持术语加权和索引持久化。
"""

import json
import os
import time
from pathlib import Path
from typing import Optional


class BM25Retriever:
    """
    BM25 关键词检索器。

    数据流:
      build_index(docs) → jieba.lcut() 分词 → BM25Okapi 构建 → joblib.dump()
      search(query)     → jieba.lcut() 分词 → 术语加权 → BM25.get_scores() → top_k
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        terminology=None,  # Optional[TerminologyDict]
    ):
        from config.settings import BM25_INDEX_DIR

        # 处理相对路径
        _idx_dir = index_dir or BM25_INDEX_DIR
        if not os.path.isabs(_idx_dir):
            project_root = Path(__file__).parent.parent
            _idx_dir = str(project_root / _idx_dir)

        self.index_dir = Path(_idx_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.terminology = terminology  # TerminologyDict or None
        self._bm25 = None           # BM25Okapi instance
        self._documents: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []
        self._doc_count: int = 0

    # ========== 索引构建 ==========

    def build_index(self, documents: list[dict]) -> None:
        """
        从文档列表全量构建 BM25 索引。

        Args:
            documents: 必须含 "content" 字段，可选 "doc_id", "chunk_index",
                       "metadata": {source, source_url, section_title, has_table, ...}
        """
        import jieba
        from rank_bm25 import BM25Okapi

        # 过滤空内容
        valid_docs = [d for d in documents if d.get("content", "").strip()]
        if not valid_docs:
            print("[BM25] 无有效文档，跳过索引构建")
            return

        # 分词
        tokenized = []
        for doc in valid_docs:
            tokens = jieba.lcut(doc["content"])
            tokenized.append(tokens)

        # 构建 BM25
        self._bm25 = BM25Okapi(tokenized)
        self._documents = valid_docs
        self._tokenized_corpus = tokenized
        self._doc_count = len(valid_docs)

        print(f"[BM25] 索引构建完成: {self._doc_count} 篇文档, "
              f"词汇量: {len(self._bm25.idf)}")

    def rebuild_from_store(self, store) -> None:
        """
        从 VectorStore 读取全部文档并重建 BM25 索引。
        store: VectorStore 实例，必须支持 get_all_documents()
        """
        docs = store.get_all_documents()
        if docs:
            self.build_index(docs)
            self.save()
        else:
            print("[BM25] VectorStore 中无文档，跳过重建")

    # ========== 索引持久化 ==========

    def save(self) -> None:
        """持久化 BM25 索引到磁盘"""
        import joblib

        files = {
            "bm25_model": self._bm25,
            "documents": self._documents,
            "tokenized": self._tokenized_corpus,
        }

        for name, obj in files.items():
            if obj is not None:
                path = self.index_dir / f"{name}.joblib"
                joblib.dump(obj, str(path))

        # 元信息
        meta = {
            "doc_count": self._doc_count,
            "version": "1.0",
            "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path = self.index_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[BM25] 索引已保存至: {self.index_dir}")

    def load(self) -> bool:
        """从磁盘加载 BM25 索引，返回是否成功"""
        import joblib

        model_path = self.index_dir / "bm25_model.joblib"
        docs_path = self.index_dir / "documents.joblib"
        tok_path = self.index_dir / "tokenized.joblib"

        if not model_path.exists():
            print("[BM25] 索引文件不存在，需要重建")
            return False

        try:
            self._bm25 = joblib.load(str(model_path))
            self._documents = joblib.load(str(docs_path))
            if tok_path.exists():
                self._tokenized_corpus = joblib.load(str(tok_path))
            self._doc_count = len(self._documents)
            print(f"[BM25] 索引已加载: {self._doc_count} 篇文档")
            return True
        except Exception as e:
            print(f"[BM25] 索引加载失败: {e}")
            return False

    def needs_rebuild(self, store_doc_count: int) -> bool:
        """
        判断索引是否需要重建。

        Args:
            store_doc_count: VectorStore 中当前文档数量

        Returns:
            True 如果索引不存在或文档数量不匹配
        """
        # 索引文件不存在
        if not (self.index_dir / "bm25_model.joblib").exists():
            return True

        # 读 meta
        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            return True

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("doc_count", 0) != store_doc_count:
                return True
        except Exception:
            return True

        return False

    # ========== 检索 ==========

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        """
        BM25 关键词检索。

        Args:
            query: 查询字符串
            top_k: 返回候选数量

        Returns:
            [{
                "doc_id": str,
                "chunk_index": int,
                "content": str,
                "score": float,              # BM25 原始分数
                "normalized_score": float,   # 归一化到 [0, 1]
                "metadata": {...},
            }, ...]
        """
        if not self.is_ready:
            print("[BM25] 索引未就绪，返回空结果")
            return []

        import jieba

        # 1. 分词
        tokens = jieba.lcut(query)

        # 2. 术语加权：对高权重术语 token 重复 int(weight) 次
        if self.terminology and self.terminology.is_loaded:
            weighted_tokens = []
            for t in tokens:
                w = self.terminology.get_weight(t)
                if w > 1.0:
                    # 重复 token 来模拟加权
                    repeat = int(w)
                    weighted_tokens.extend([t] * repeat)
                else:
                    weighted_tokens.append(t)
            tokens = weighted_tokens

        # 3. BM25 打分
        scores = self._bm25.get_scores(tokens)

        # 4. 归一化（min-max）
        if len(scores) > 0 and max(scores) > min(scores):
            s_min, s_max = min(scores), max(scores)
            norm_scores = [(s - s_min) / (s_max - s_min) for s in scores]
        else:
            norm_scores = [1.0 if s > 0 else 0.0 for s in scores]

        # 5. 排序取 top_k
        indexed = list(enumerate(zip(scores, norm_scores)))
        indexed.sort(key=lambda x: x[1][0], reverse=True)
        top = indexed[:top_k]

        # 6. 构建结果
        results = []
        for idx, (raw_score, norm_score) in top:
            doc = self._documents[idx]
            meta = doc.get("metadata", {})
            results.append({
                "doc_id": doc.get("doc_id", ""),
                "chunk_index": doc.get("chunk_index", 0),
                "content": doc.get("content", ""),
                "score": round(float(raw_score), 4),
                "normalized_score": round(float(norm_score), 4),
                "metadata": {
                    "source": meta.get("source", ""),
                    "source_url": meta.get("source_url", ""),
                    "section_title": meta.get("section_title", ""),
                    "has_table": meta.get("has_table", False),
                    "char_count": meta.get("char_count", 0),
                },
            })

        return results

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and self._doc_count > 0
