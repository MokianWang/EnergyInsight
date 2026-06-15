"""
Milvus Lite 向量库管理
本地文件存储，零依赖，无需Docker
"""

from pymilvus import (
    MilvusClient,
    DataType,
    CollectionSchema,
    FieldSchema,
)
from config.settings import MILVUS_DB_PATH, MILVUS_COLLECTION, MILVUS_VECTOR_DIM


class VectorStore:
    """Milvus Lite 向量库（本地文件模式）"""

    def __init__(
        self,
        db_path: str = None,
        collection_name: str = None,
        dim: int = None,
    ):
        import os
        self.db_path = db_path or MILVUS_DB_PATH
        self.collection_name = collection_name or MILVUS_COLLECTION
        self.dim = dim or MILVUS_VECTOR_DIM

        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._client = MilvusClient(self.db_path)
        self._ensure_loaded()

    def _ensure_loaded(self):
        """确保集合已加载到内存"""
        if self._client.has_collection(self.collection_name):
            self._client.load_collection(self.collection_name)

    def create_collection(self) -> bool:
        """创建集合（如果不存在）"""
        if self._client.has_collection(self.collection_name):
            return False

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="has_table", dtype=DataType.BOOL),
            FieldSchema(name="char_count", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]

        schema = CollectionSchema(fields, description="EnergyInsight 能源知识库")
        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )

        # 创建索引
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        self._client.create_index(self.collection_name, index_params)

        # 加载集合到内存
        self._client.load_collection(self.collection_name)
        print(f"[VectorStore] 集合已创建并加载: {self.collection_name}")
        return True

    def insert(self, chunks: list[dict], vectors: list[list[float]]) -> int:
        """
        批量插入chunk和向量

        Args:
            chunks: chunk字典列表
            vectors: 对应的向量列表

        Returns:
            插入数量
        """
        data = []
        for chunk, vector in zip(chunks, vectors):
            meta = chunk.get("metadata", {})
            data.append({
                "doc_id": chunk["doc_id"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"][:8192],
                "source": meta.get("source", ""),
                "source_url": meta.get("source_url", ""),
                "section_title": meta.get("section_title", ""),
                "has_table": meta.get("has_table", False),
                "char_count": meta.get("char_count", 0),
                "embedding": vector,
            })

        result = self._client.insert(self.collection_name, data)
        count = result.get("insert_count", 0)
        # 插入后重新加载
        self._client.load_collection(self.collection_name)
        print(f"[VectorStore] 已插入: {count} 条")
        return count

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        expr: str = None,
    ) -> list[dict]:
        """
        ANN检索

        Args:
            query_vector: 查询向量（已归一化）
            top_k: 返回数量
            expr: 过滤表达式（如 'source == "iea"'）

        Returns:
            检索结果列表
        """
        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            filter=expr,
            output_fields=[
                "doc_id", "chunk_index", "content",
                "source", "source_url", "section_title",
                "has_table", "char_count",
            ],
        )

        items = []
        for hits in results:
            for hit in hits:
                items.append({
                    "chunk_id": hit["id"],
                    "doc_id": hit["entity"].get("doc_id", ""),
                    "content": hit["entity"].get("content", ""),
                    "score": hit["distance"],
                    "metadata": {
                        "source": hit["entity"].get("source", ""),
                        "source_url": hit["entity"].get("source_url", ""),
                        "section_title": hit["entity"].get("section_title", ""),
                        "has_table": hit["entity"].get("has_table", False),
                        "char_count": hit["entity"].get("char_count", 0),
                    },
                })

        return items

    def count(self) -> int:
        """返回集合中的chunk数量"""
        result = self._client.query(
            collection_name=self.collection_name,
            filter="id >= 0",
            output_fields=["count(*)"],
        )
        # query returns list[dict], e.g. [{"count(*)": 159}]
        if result and isinstance(result, list) and len(result) > 0:
            return result[0].get("count(*)", 0)
        return 0

    def get_all_documents(self, batch_size: int = 1000) -> list[dict]:
        """
        分页读取集合中所有文档元数据，用于 BM25 索引构建。

        Args:
            batch_size: 每批读取数量

        Returns:
            [{
                "doc_id": str,
                "chunk_index": int,
                "content": str,
                "metadata": {
                    "source": str,
                    "source_url": str,
                    "section_title": str,
                    "has_table": bool,
                    "char_count": int,
                },
            }, ...]
        """
        all_docs = []
        offset = 0
        total = self.count()
        if total == 0:
            return all_docs

        while offset < total:
            results = self._client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=[
                    "id", "doc_id", "chunk_index", "content",
                    "source", "source_url", "section_title",
                    "has_table", "char_count",
                ],
                limit=batch_size,
                offset=offset,
            )
            if not results:
                break

            for r in results:
                all_docs.append({
                    "doc_id": r.get("doc_id", ""),
                    "chunk_index": r.get("chunk_index", 0),
                    "content": r.get("content", ""),
                    "metadata": {
                        "source": r.get("source", ""),
                        "source_url": r.get("source_url", ""),
                        "section_title": r.get("section_title", ""),
                        "has_table": r.get("has_table", False),
                        "char_count": r.get("char_count", 0),
                    },
                })

            offset += len(results)
            if len(results) < batch_size:
                break

        print(f"[VectorStore] 读取全部文档: {len(all_docs)} 条")
        return all_docs

    def drop_collection(self):
        """删除集合（重建场景）"""
        if self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)
            print(f"[VectorStore] 已删除集合: {self.collection_name}")

    def close(self):
        """释放连接"""
        self._client.close()
