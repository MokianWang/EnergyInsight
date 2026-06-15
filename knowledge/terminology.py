"""
能源术语词典 + 轻量级 NER 实体识别

TerminologyDict: 加载 terminology_dict.json，提供术语查找、jieba分词注册、查询扩展
EnergyNER: 基于正则 + 词典匹配的轻量 NER（不依赖深度学习模型）
"""

import json
import re
import os
from pathlib import Path
from typing import Optional


class TerminologyDict:
    """能源术语词典加载器"""

    def __init__(self, dict_path: Optional[str] = None):
        from config.settings import TERMINOLOGY_DICT_PATH as _default_path

        self.dict_path = dict_path or _default_path
        # 相对路径 → 项目根目录
        if not os.path.isabs(self.dict_path):
            project_root = Path(__file__).parent.parent
            self.dict_path = str(project_root / self.dict_path)

        self._terms: list[dict] = []
        self._by_zh: dict[str, dict] = {}
        self._by_en: dict[str, dict] = {}
        self._by_alias: dict[str, dict] = {}
        self._categories: dict[str, dict] = {}
        self._meta: dict = {}
        self._loaded = False

        self._load()

    # ========== 加载 ==========

    def _load(self) -> None:
        """加载 JSON 词典并构建索引"""
        try:
            with open(self.dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[TerminologyDict] 词典文件未找到: {self.dict_path}，术语功能不可用")
            self._loaded = False
            return
        except json.JSONDecodeError as e:
            print(f"[TerminologyDict] 词典 JSON 解析失败: {e}")
            self._loaded = False
            return

        self._meta = data.get("meta", {})
        self._categories = data.get("categories", {})
        self._terms = data.get("terms", [])

        # 构建索引
        self._by_zh.clear()
        self._by_en.clear()
        self._by_alias.clear()

        for term in self._terms:
            zh = term.get("zh", "")
            en = term.get("en", "")
            if zh:
                self._by_zh[zh] = term
            if en:
                self._by_en[en.lower()] = term
            # 别名索引
            for alias in term.get("aliases", []):
                self._by_alias[alias] = term

        self._loaded = True

        # 向 jieba 注册自定义分词
        self._register_jieba()
        print(f"[TerminologyDict] 已加载 {self.total_count} 条能源术语 "
              f"(zh: {len(self._by_zh)}, en: {len(self._by_en)}, alias: {len(self._by_alias)})")

    def _register_jieba(self) -> None:
        """将所有中文术语注册到 jieba 分词器，防止专有名词被错误切分"""
        try:
            import jieba
        except ImportError:
            print("[TerminologyDict] jieba 未安装，跳过自定义分词注册")
            return

        count = 0
        for term in self._terms:
            zh = term.get("zh", "")
            if zh and len(zh) >= 2:
                jieba.add_word(zh, freq=100, tag="n")
                count += 1
            # 同时注册别名（长度 ≥ 2）
            for alias in term.get("aliases", []):
                if len(alias) >= 2:
                    jieba.add_word(alias, freq=80, tag="n")
                    count += 1
        print(f"[TerminologyDict] 已向 jieba 注册 {count} 个自定义词")

    # ========== 查找 ==========

    def get_term(self, text: str) -> Optional[dict]:
        """精确匹配中文或英文术语，返回完整 term record"""
        if text in self._by_zh:
            return self._by_zh[text]
        if text.lower() in self._by_en:
            return self._by_en[text.lower()]
        if text in self._by_alias:
            return self._by_alias[text]
        return None

    def get_weight(self, term_zh: str) -> float:
        """返回术语权重（默认 1.0），用于 BM25 查询端加权"""
        term = self.get_term(term_zh)
        if term:
            return term.get("weight", 1.0)
        return 1.0

    def find_terms_in_text(self, text: str) -> list[dict]:
        """
        在文本中查找已知术语，长词优先匹配。

        Returns:
            [{"term": {...}, "start": int, "end": int}, ...]
        """
        if not self._loaded or not text:
            return []

        # 收集所有候选词（zh + en + aliases），去重
        candidates: set[str] = set()
        candidates.update(self._by_zh.keys())
        candidates.update(self._by_alias.keys())
        # 英文词只加入长度 ≥2 的（避免单字母匹配）
        candidates.update(k for k in self._by_en.keys() if len(k) >= 2)

        # 按长度降序排列（长词优先）
        sorted_candidates = sorted(candidates, key=lambda x: -len(x))

        found = []
        matched_positions: set[int] = set()

        for candidate in sorted_candidates:
            # 跳过已匹配区域内的候选词
            start = 0
            while True:
                idx = text.find(candidate, start)
                if idx == -1:
                    break

                # 检查是否与已匹配区域重叠
                positions = set(range(idx, idx + len(candidate)))
                if positions & matched_positions:
                    # 有重叠，跳到下一个位置继续搜索
                    start = idx + 1
                    continue

                term = self.get_term(candidate)
                if term:
                    found.append({
                        "term": term,
                        "start": idx,
                        "end": idx + len(candidate),
                    })
                    matched_positions.update(positions)
                break  # 每个候选词只取首次匹配

        return found

    def expand_query(self, query: str) -> str:
        """
        查询扩展：对检测到的中文术语附加英文翻译。
        例如 "钠离子电池储能" → "钠离子电池 sodium-ion battery 储能 energy storage"

        Returns:
            扩展后的查询字符串
        """
        if not self._loaded or not query:
            return query

        found = self.find_terms_in_text(query)
        if not found:
            return query

        # 按起始位置排序
        found.sort(key=lambda x: x["start"])

        # 构建扩展查询
        parts = []
        last_end = 0
        for f in found:
            # 加入匹配段之前的原文
            if f["start"] > last_end:
                parts.append(query[last_end:f["start"]])
            # 加入中文术语
            zh = f["term"].get("zh", "")
            en = f["term"].get("en", "")
            parts.append(zh)
            # 附加英文翻译（如果不为空且与中文不同）
            if en and en != zh:
                parts.append(f" {en} ")
            last_end = f["end"]

        # 加入剩余部分
        if last_end < len(query):
            parts.append(query[last_end:])

        return "".join(parts)

    # ========== 属性 / 统计 ==========

    @property
    def all_terms(self) -> list[dict]:
        return self._terms

    @property
    def total_count(self) -> int:
        return len(self._terms)

    @property
    def categories(self) -> dict:
        return self._categories

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ============================================================================
#  EnergyNER: 轻量级能源领域命名实体识别
# ============================================================================

class EnergyNER:
    """
    轻量级能源领域 NER。

    策略：正则 + 术语词典匹配（不引入模型），提取：
    - 技术名称（词典匹配，category == "technology"）
    - 机构/企业（词典匹配，category in ["organization", "company"]）
    - 政策文号（正则）
    - 容量/能量单位（正则）
    - 年份/日期（正则）
    - 百分比（正则）
    - 五年规划（正则）
    """

    def __init__(self, terminology: Optional[TerminologyDict] = None):
        self.terminology = terminology or TerminologyDict()
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict[str, re.Pattern]:
        """预编译所有提取正则"""
        return {
            # 容量/能量单位: "100MW", "200GWh", "150万千瓦", "3.5亿千瓦时", "500万吨"
            "CAPACITY": re.compile(
                r'(\d+(?:\.\d+)?)\s*(MW|GW|kW|MWh|GWh|kWh|'
                r'万千瓦|亿千瓦时|亿千瓦|万千瓦时|万吨|万kW|万kWh|'
                r'吨标准煤|万tce|亿吨CO2|吨CO2)',
                re.IGNORECASE
            ),
            # 政策文号: "国能发科技规〔2024〕26号", "发改价格〔2023〕1500号"
            "POLICY_ID": re.compile(
                r'[国能发国发改环资工信财政科][一-龥]*〔\d{4}〕\d+号'
            ),
            # 年份
            "YEAR": re.compile(r'(\d{4})\s*年'),
            # 五年规划: "十四五", "十五五", "十一五"
            "FIVE_YEAR": re.compile(r'第[十百千万]+[三四五六七八九]个?五年规划|'
                                     r'[十百千万]+[三四五六七八九]五'),
            # 百分比: "85%", "12.5%"
            "PERCENT": re.compile(r'(\d+(?:\.\d+)?)\s*%'),
            # 日期范围: "2024-2025年", "2024至2025年"
            "DATE_RANGE": re.compile(r'(\d{4})\s*[-至到~]\s*(\d{4})\s*年?'),
        }

    def extract(self, text: str) -> dict:
        """
        对输入文本进行实体提取。

        Returns:
            {
                "technologies": [{"term": "钠离子电池", "category": "technology", "weight": 1.5, ...}],
                "organizations": [{"term": "国家能源局", "category": "organization", ...}],
                "companies": [{"term": "宁德时代", "category": "company", ...}],
                "policy_ids": ["国能发科技规〔2024〕26号"],
                "capacities": [{"value": 100.0, "unit": "MW", "raw": "100MW"}],
                "years": [2024, 2025],
                "date_ranges": [{"start": 2024, "end": 2025}],
                "percents": [85.0],
                "five_year_plans": ["十四五"],
            }
        """
        if not text:
            return {
                "technologies": [], "organizations": [], "companies": [],
                "policy_ids": [], "capacities": [], "years": [],
                "date_ranges": [], "percents": [], "five_year_plans": [],
            }

        result = {
            "technologies": [],
            "organizations": [],
            "companies": [],
            "policy_ids": [],
            "capacities": [],
            "years": [],
            "date_ranges": [],
            "percents": [],
            "five_year_plans": [],
        }

        # 1. 词典匹配分类
        if self.terminology.is_loaded:
            found_terms = self.terminology.find_terms_in_text(text)
            for ft in found_terms:
                term = ft["term"]
                cat = term.get("category", "")
                record = {
                    "term": term["zh"],
                    "en": term.get("en", ""),
                    "category": cat,
                    "weight": term.get("weight", 1.0),
                }
                if cat == "technology":
                    result["technologies"].append(record)
                elif cat == "organization":
                    result["organizations"].append(record)
                elif cat == "company":
                    result["companies"].append(record)

        # 2. 正则提取
        # 容量/能量单位
        for m in self._patterns["CAPACITY"].finditer(text):
            result["capacities"].append({
                "value": float(m.group(1)),
                "unit": m.group(2),
                "raw": m.group(0),
            })

        # 政策文号
        for m in self._patterns["POLICY_ID"].finditer(text):
            result["policy_ids"].append(m.group(0))

        # 年份
        seen_years = set()
        for m in self._patterns["YEAR"].finditer(text):
            yr = int(m.group(1))
            if yr not in seen_years and 1990 <= yr <= 2100:
                result["years"].append(yr)
                seen_years.add(yr)

        # 日期范围
        for m in self._patterns["DATE_RANGE"].finditer(text):
            s, e = int(m.group(1)), int(m.group(2))
            if 1990 <= s <= 2100 and 1990 <= e <= 2100:
                result["date_ranges"].append({"start": s, "end": e})

        # 百分比
        for m in self._patterns["PERCENT"].finditer(text):
            result["percents"].append(float(m.group(1)))

        # 五年规划
        for m in self._patterns["FIVE_YEAR"].finditer(text):
            plan = m.group(0)
            if plan not in result["five_year_plans"]:
                result["five_year_plans"].append(plan)

        return result

    def extract_for_retrieval(self, query: str) -> dict:
        """
        为检索优化的实体提取。

        Returns:
            {
                "entities": {...},          # extract() 的完整结果
                "expanded_query": str,      # 术语扩展后的查询
                "filters": dict,            # 可用于 Milvus 过滤表达式
                "keywords": list[str],      # 提取的关键词（用于 BM25 加权）
            }
        """
        entities = self.extract(query)
        expanded_query = self.terminology.expand_query(query) if self.terminology.is_loaded else query

        # 构建 Milvus 过滤表达式（暂用简单模式）
        filters = {}
        # 如果识别出具体年份，可过滤
        if entities["years"]:
            # 暂不应用于表达式（Milvus 无 pub_date 字段），留给时间加权阶段使用
            pass

        # 提取关键词
        keywords: list[str] = []
        for t in entities["technologies"]:
            keywords.append(t["term"])
            if t.get("en"):
                keywords.append(t["en"])
        for o in entities["organizations"]:
            keywords.append(o["term"])
        for c in entities["companies"]:
            keywords.append(c["term"])

        return {
            "entities": entities,
            "expanded_query": expanded_query,
            "filters": filters,
            "keywords": keywords,
        }
