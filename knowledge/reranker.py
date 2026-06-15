"""
BGE Cross-Encoder 重排序器（Python 3.12 子进程模式）

主进程 (Python 3.14) 不加载 torch，通过子进程与 Python 3.12 venv 通信，
彻底规避 torch C 扩展的兼容性问题。

工作原理:
  1. 检测 reranker_venv/ (Python 3.12) 是否可用
  2. 可用 → 启动子进程，通过 stdin/stdout JSON 协议通信
  3. 不可用 → 降级为 TF-IDF 余弦相似度重排序

协议 (stdin/stdout):
  请求: {"query": "...", "documents": ["...", ...]}
  响应: {"scores": [1.2, -3.4, ...]} 或 {"error": "..."}
  启动: 第一行输出 "READY"
"""

import json
import math
from pathlib import Path
from typing import Optional
_VENV_PYTHON = Path(__file__).parent.parent / "reranker_venv" / "Scripts" / "python.exe"


class BGEReranker:
    """
    BGE Cross-Encoder 重排序器。

    自动检测 Python 3.12 venv，可用时使用 BGE ONNX 模型，
    不可用时降级为 TF-IDF。
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
    ):
        from config.settings import RERANKER_MODEL, RERANKER_MAX_LENGTH

        self.model_name = model_name or RERANKER_MODEL
        self.max_length = max_length if max_length is not None else RERANKER_MAX_LENGTH

        self._proc: Optional[subprocess.Popen] = None
        self._backend = "tfidf"
        self._ready = False
        self._load_error: Optional[str] = None
        self._tfidf = None

    # ========== 模型加载 ==========

    def _load_model(self) -> None:
        """初始化重排序器"""
        if self._ready:
            return

        try:
            from config.settings import RAG_RERANK_ENABLED
            if not RAG_RERANK_ENABLED:
                self._load_error = "RAG_RERANK_ENABLED=false"
                return
        except ImportError:
            pass

        # 优先尝试 BGE (Python 3.12 venv)
        if self._try_start_bge_worker():
            self._backend = "bge"
            self._ready = True
            return

        # 降级到 TF-IDF
        if self._init_tfidf():
            self._backend = "tfidf"
            self._ready = True
        else:
            self._load_error = "所有后端不可用"

    def _try_start_bge_worker(self) -> bool:
        """BGE worker disabled — always fall through to TF-IDF"""
        return False

    def _init_tfidf(self) -> bool:
        """初始化 TF-IDF 降级后端"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import jieba
            self._tfidf = {
                "vectorizer_cls": TfidfVectorizer,
                "cosine_sim": cosine_similarity,
                "jieba": jieba,
            }
            print("[Reranker] TF-IDF 后端就绪")
            return True
        except ImportError as e:
            print(f"[Reranker] TF-IDF 依赖缺失: {e}")
            return False

    def _terminate_worker(self):
        """终止子进程"""
        try:
            if self._proc:
                if self._proc.stdin:
                    self._proc.stdin.close()
                if self._proc.stdout:
                    self._proc.stdout.close()
                if self._proc.stderr:
                    self._proc.stderr.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
        except Exception:
            pass
        finally:
            self._proc = None

    # ========== 检索 ==========

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """对候选文档重排序。BGE 不可用时自动降级 TF-IDF。"""
        if not self._ready:
            self._load_model()

        if not self._ready:
            return documents[:top_k]

        if not documents:
            return []

        if self._backend == "bge":
            result = self._bge_rerank(query, documents)
        else:
            result = self._tfidf_rerank(query, documents)

        # Sigmoid 归一化
        for d in result[:top_k]:
            rs = d.get("rerank_score", 0)
            try:
                d["score"] = round(1.0 / (1.0 + math.exp(-rs)), 4)
            except OverflowError:
                d["score"] = 1.0 if rs > 0 else 0.0

        return result[:top_k]

    def _bge_rerank(self, query: str, documents: list[dict]) -> list[dict]:
        """BGE Cross-Encoder 重排序 (通过子进程)"""
        max_char = self.max_length * 3
        doc_contents = [d.get("content", "")[:max_char] for d in documents]

        request = {"query": query, "documents": doc_contents}
        try:
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            response_line = self._proc.stdout.readline().strip()
            if not response_line:
                raise RuntimeError("子进程无响应")
            response = json.loads(response_line)
            if "error" in response:
                raise RuntimeError(response["error"])
            scores = response.get("scores", [])
        except Exception as e:
            print(f"[Reranker] BGE 子进程通信失败: {e}，降级 TF-IDF")
            self._terminate_worker()
            self._backend = "tfidf"
            if not self._init_tfidf():
                return documents
            return self._tfidf_rerank(query, documents)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = round(float(score), 4)
        return sorted(documents, key=lambda d: d.get("rerank_score", -999), reverse=True)

    def _tfidf_rerank(self, query: str, documents: list[dict]) -> list[dict]:
        """TF-IDF 余弦相似度重排序"""
        jieba = self._tfidf["jieba"]
        TfidfVectorizer = self._tfidf["vectorizer_cls"]
        cosine_sim = self._tfidf["cosine_sim"]

        doc_texts = [" ".join(jieba.lcut(d.get("content", ""))) for d in documents]
        query_text = " ".join(jieba.lcut(query))
        all_texts = [query_text] + doc_texts

        vectorizer = TfidfVectorizer(max_features=5000, analyzer='word', token_pattern=r'(?u)\b\w+\b')
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
        except ValueError:
            return documents

        sims = cosine_sim(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        for doc, sim in zip(documents, sims):
            doc["rerank_score"] = round(float(sim), 4)
        return sorted(documents, key=lambda d: d.get("rerank_score", -999), reverse=True)

    # ========== 属性 ==========

    @property
    def is_ready(self) -> bool:
        if not self._ready and self._load_error is None:
            self._load_model()
        return self._ready

    @property
    def error_message(self) -> Optional[str]:
        return self._load_error

    @property
    def backend(self) -> str:
        return self._backend

    def __del__(self):
        self._terminate_worker()
