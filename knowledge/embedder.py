"""
向量化引擎
调用千问 DashScope text-embedding-v3 API
"""

import math
from config.settings import get_embedding, EMBEDDING_MODEL, EMBEDDING_DIM


class EmbeddingEngine:
    """文本向量化引擎（千问 Embedding API）"""

    def __init__(self, model: str = None, dim: int = None):
        self.model = model or EMBEDDING_MODEL
        self.dim = dim or EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量生成文本向量（文档侧）

        Args:
            texts: 文本列表

        Returns:
            向量列表（已 L2 归一化）
        """
        print(f"[Embedder] 向量化 {len(texts)} 条文本...")
        all_vectors = []

        # 分批调用API（text-embedding-v2 单次最多25条）
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vectors = get_embedding(batch)
            normalized = [self._normalize(v) for v in vectors]
            all_vectors.extend(normalized)

            if (i + batch_size) % 128 == 0:
                print(f"[Embedder]  进度: {min(i + batch_size, len(texts))}/{len(texts)}")

        print(f"[Embedder] 完成: {len(all_vectors)} 条向量")
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        """
        生成查询向量

        Args:
            text: 查询文本

        Returns:
            归一化向量
        """
        vectors = get_embedding([text])
        return self._normalize(vectors[0])

    @property
    def vector_dim(self) -> int:
        return self.dim

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        """L2 归一化（IP = Cosine）"""
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]
