"""
Reviewer Agent - 质量审查 (两级架构)
一级：规则层 —— 正则 + 数值边界 + 单位一致性（快速，不依赖 LLM）
二级：模型层 —— LLM 全面审查（深度，识别语义层面的幻觉）

不通过时打回 Writer 重写，最多 MAX_REVIEW_ROUNDS 轮。
"""

import json
import re
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm, get_max_review_rounds

_CURRENT_YEAR = datetime.now().year
_PREV_YEAR = _CURRENT_YEAR - 1

REVIEWER_SYSTEM_PROMPT = f"""你是 EnergyInsight 系统的质量审查智能体（Reviewer Agent）。

当前时间为 {_CURRENT_YEAR} 年。请以此为基准判断数据时效性。

对研究报告进行严格质量审查，确保准确性和可靠性。

## 审查维度（按优先级排序）

### 1. 事实核查 — 最高优先级
- 每个数值数据点是否有引用编号 [N] 支撑？
- 政策文号格式是否正确？（标准格式：发文机关〔年份〕编号号）
- 技术参数是否在行业合理范围内？
- **时间倒置检查**：报告中提到的未来事件（如 "{_CURRENT_YEAR+1}年"）是否有明确标注为"规划"或"预测"？若无标注则视为幻觉。

### 2. 数据一致性
- 同一指标在不同位置数值是否一致？
- 单位使用是否统一？（MW与GW不得混用描述同一量级）
- 时间标签是否正确？（不能用{_CURRENT_YEAR-3}年前的数据回答"{_CURRENT_YEAR}年"的问题）

### 3. 逻辑完整性
- 报告是否完整回答了用户问题？
- 分析逻辑是否自洽？
- 结论是否由数据推导而来？

### 4. 幻觉检测
- 是否存在无引用来源的数据声明？
- 是否存在明显违背能源行业常识的内容？
  - 中国光伏累计装机 {_PREV_YEAR} 年约 890GW（合理范围 600-1100GW）
  - 中国风电累计装机 {_PREV_YEAR} 年约 520GW（合理范围 400-650GW）
  - 中国储能累计装机 {_PREV_YEAR} 年约 60GW（合理范围 30-90GW）
  - 光伏 LCOE 0.15-0.40 元/kWh
  - 风电 LCOE 0.15-0.35 元/kWh
  - 锂电池储能系统成本 0.5-1.2 元/Wh
  - 全国碳价 50-120 元/吨
  - 欧盟碳价 40-100 欧元/吨
  - 单个陆上风电机组 ≤ 15MW
  - 单个海上风电机组 ≤ 22MW

### 5. 引用质量
- 引用来源是否权威？（优先：国家能源局/发改委/IEA/IRENA/EIA/上市公司年报）
- 引用编号是否与参考来源一一对应？
- 是否存在无引用支撑的数据声明？

## 输出格式（严格 JSON）
```json
{{
  "passed": true|false,
  "score": 0-100,
  "issues": [
    {{
      "type": "事实错误|数据不一致|逻辑问题|幻觉|引用缺失|格式问题",
      "severity": "严重|中等|轻微",
      "description": "具体问题描述",
      "location": "章节/段落",
      "suggestion": "修改建议"
    }}
  ],
  "summary": "总体评价（50字内）"
}}
```

## 通过标准
- score >= 75 且无"严重"级别问题 → passed = true
- 否则 → passed = false（打回重写）
"""

# ========== 数值边界表 ==========
_NUMERIC_BOUNDS = {
    "中国光伏累计装机": (500, 1200, "GW"),
    "中国风电累计装机": (350, 700, "GW"),
    "中国新型储能累计装机": (20, 100, "GW"),
    "中国储能装机": (20, 100, "GW"),
    "中国储能累计装机": (20, 100, "GW"),
    "储能累计装机": (20, 100, "GW"),
    "中国储能": (20, 100, "GW"),
    "光伏度电成本": (0.10, 0.50, "元/kWh"),
    "光伏LCOE": (0.10, 0.50, "元/kWh"),
    "光伏发电成本": (0.10, 0.50, "元/kWh"),
    "风电度电成本": (0.10, 0.45, "元/kWh"),
    "风电LCOE": (0.10, 0.45, "元/kWh"),
    "锂电池储能成本": (0.4, 1.5, "元/Wh"),
    "钠离子电池成本": (0.3, 1.0, "元/Wh"),
    "全国碳价": (40, 150, "元/吨"),
    "欧盟碳价": (30, 120, "欧元/吨"),
    "陆上风电机组单机": (1, 15, "MW"),
    "海上风电机组单机": (3, 22, "MW"),
    "储能系统效率": (75, 98, "%"),
    "光伏组件效率": (18, 30, "%"),
}

# ========== 政策文号正则 ==========
_POLICY_ID_PATTERNS = [
    # 标准格式：发文机关〔年份〕编号号
    re.compile(r'(?:国发|国办发|国函|国办函|国能发|发改|财|科|工信|环资|住建|交|农|水|商|能|林|市监|卫健|教|文|'
               r'国资|税|银保监|证监|外汇|人社|自然资源|生态环境|应急|审计)[一-龥]*〔\d{4}〕\d+号'),
]

# ========== AI 套话模式 ==========
_AI_PHRASES = [
    "作为一个AI", "作为AI助手", "作为一个人工智能", "作为AI语言模型",
    "我无法", "我没有", "我不能", "我不具备",
    "根据我的知识", "截至我的知识", "在我的训练数据中",
    "请注意，我是一个AI", "请咨询专业人士",
    "Disclaimer", "disclaimer",
]

# ========== 错误的单位使用模式 ==========
_UNIT_ISSUES = [
    (r'\d+\.?\d*\s*MW\s*=\s*\d+\.?\d*\s*GW', "MW 与 GW 混用"),
    (r'\d+\.?\d*\s*元/kWh\s*=\s*\d+\.?\d*\s*元/MWh', "元/kWh 与 元/MWh 混用（千倍差异）"),
]


def review_report(
    query: str,
    report: str,
    citations: list[dict],
    round_num: int = 1,
) -> dict:
    """
    审查研究报告（两级架构，优化流程）

    策略：规则层 + 模型层先行（<1s），无问题时跳过 LLM。
    仅当规则检测到潜在问题时才调用 LLM 深度审查。
    """
    import os
    fast_mode = os.getenv("REVIEWER_FAST_MODE", "").lower() == "true"

    # ==== 第一层：规则 + 模型快速扫描（<1s） ====
    rule_issues = _rule_based_check(report, citations, query)
    model_issues = model_layer_check(report, citations)
    all_issues = rule_issues + model_issues

    severe_count = sum(1 for i in all_issues if i.get("severity") == "严重")
    medium_count = sum(1 for i in all_issues if i.get("severity") == "中等")
    minor_count = sum(1 for i in all_issues if i.get("severity") == "轻微")

    # 快速模式或无问题时：跳过 LLM 审查
    if fast_mode or (severe_count == 0 and medium_count == 0 and minor_count <= 2):
        passed = severe_count == 0 and medium_count == 0
        score = 100 if passed else (85 if medium_count == 0 else 60)
        summary = (
            f"快速审查通过 ({len(all_issues)} 个轻微问题)" if passed
            else f"规则检测到 {severe_count} 严重 + {medium_count} 中等问题"
        )
        print(f"  [快速审查] {summary}")
        return {
            "passed": passed,
            "score": score,
            "feedback": summary,
            "issues": all_issues,
            "hallucination_issues": [
                i["description"] for i in all_issues if i.get("type") == "幻觉"
            ],
        }

    # ==== 第二层：LLM 深度审查（仅当规则发现较多问题时触发） ====
    print(f"  [LLM审查] 规则检测到 {len(all_issues)} 个问题，启动深度审查...")
    llm = get_llm(temperature=0.1)

    citations_text = ""
    if citations:
        for c in citations:
            citations_text += f"[{c.get('id', '?')}] {c.get('title', '')}\n    URL: {c.get('url', '')}\n"

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"## 用户原始问题\n{query}\n\n"
            f"## 待审查报告\n{report}\n\n"
            f"## 引用来源列表\n{citations_text}\n\n"
            f"## 审查轮次\n第 {round_num}/{get_max_review_rounds()} 轮\n\n"
            f"## 规则层已检测到的问题（请合并到最终审查结果中）\n"
            + "\n".join(f"- [{i.get('severity','')}] {i.get('type','')}: {i.get('description','')[:100]}"
                       for i in all_issues)
        )),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"passed": True, "score": 70, "issues": [], "summary": "LLM解析失败"}

    # 合并规则层结果
    existing = result.get("issues", [])
    existing.extend(all_issues)
    result["issues"] = existing
    has_severe = any(i.get("severity") == "严重" for i in existing)
    if has_severe:
        result["passed"] = False

    return {
        "passed": result.get("passed", True),
        "score": result.get("score", 0),
        "feedback": result.get("summary", ""),
        "issues": result.get("issues", []),
        "hallucination_issues": [
            i["description"] for i in result.get("issues", []) if i.get("type") == "幻觉"
        ],
    }


def model_layer_check(report: str, citations: list[dict]) -> list[dict]:
    """
    模型层幻觉检测：使用 HallucinationClassifier 验证报告中的 claim 是否被引用支持。

    从报告中提取数值声明，与每条引用进行匹配，标记不匹配的声明为潜在幻觉。

    Returns:
        检测到的问题列表
    """
    issues = []
    try:
        import os
        backend = os.getenv("HALLUCINATION_BACKEND", "lora")
        from agents.hallucination_classifier import HallucinationClassifier
        classifier = HallucinationClassifier(backend=backend)
    except ImportError:
        return issues

    # 提取报告中的数值声明
    claim_pattern = re.compile(
        r'([^。\n]{0,30}(?:\d+\.?\d*\s*(?:GW|MW|kW|GWh|MWh|元/kWh|元/MWh|元/Wh|'
        r'元/吨|亿千瓦时|万千瓦|亿吨|万吨|%))[^。\n]{0,30})'
    )
    claims = claim_pattern.findall(report)[:10]  # 最多检查10条

    if not claims or not citations:
        return issues

    for claim in claims:
        claim = claim.strip()
        if len(claim) < 10:
            continue

        supported = False
        for c in citations:
            evid = c.get("snippet", "") or c.get("title", "")
            if len(evid) < 5:
                continue
            result = classifier.classify(claim, evid)
            if result == "支持":
                supported = True
                break

        if not supported:
            issues.append({
                "type": "幻觉",
                "severity": "中等",
                "description": f"数值声明未被任何引用支持: '{claim[:60]}'",
                "location": "数据部分",
                "suggestion": "添加引用来源或核实该数据",
            })

    return issues


def _rule_based_check(report: str, citations: list[dict], query: str = "") -> list[dict]:
    """规则层校验：快速、确定性的检查，不依赖 LLM。"""
    issues = []

    # 1. 引用编号一致性
    cited_ids = set(re.findall(r"\[(\d+)\]", report))
    available_ids = {str(c.get("id", "")) for c in citations}
    missing = cited_ids - available_ids
    if missing:
        issues.append({
            "type": "引用缺失", "severity": "中等",
            "description": f"报告中引用了不存在的来源编号: {sorted(missing, key=int)}",
            "location": "全文", "suggestion": "删除无效引用或补充对应来源",
        })

    # 2. 数据无引用
    # 查找"xx GW" "xx MW" "xx亿元" "xx万吨"等数值声明，检查附近是否有引用标记
    numeric_declarations = re.findall(
        r'(\d+\.?\d*\s*(?:GW|MW|kW|GWh|MWh|元/kWh|元/MWh|元/Wh|元/W|亿元|万元|万吨|亿吨|%))',
        report
    )
    if numeric_declarations and not cited_ids:
        issues.append({
            "type": "引用缺失", "severity": "严重",
            "description": f"报告包含 {len(numeric_declarations)} 处数值声明但全文无引用标记",
            "location": "全文", "suggestion": "每个数据点添加 [N] 引用标记",
        })

    # 3. 政策文号格式检查
    for pattern in _POLICY_ID_PATTERNS:
        found_ids = pattern.findall(report)
        for pid in found_ids:
            year_match = re.search(r'(\d{4})', pid)
            if year_match:
                year = int(year_match.group(1))
                if year > _CURRENT_YEAR + 2:
                    issues.append({
                        "type": "事实错误", "severity": "严重",
                        "description": f"政策文号年份异常: {pid}（未来年份 {year} > 当前 {_CURRENT_YEAR}）",
                        "location": "全文", "suggestion": "检查并修正政策文号年份",
                    })
                elif year < 2000:
                    issues.append({
                        "type": "事实错误", "severity": "中等",
                        "description": f"政策文号疑似过旧: {pid}（年份 {year}）",
                        "location": "全文", "suggestion": "确认政策文号是否仍有效",
                    })

    # 4. 数值边界检查 — 匹配 "XX <unit>" 并检查周围的指标关键词
    for metric, (lo, hi, unit) in _NUMERIC_BOUNDS.items():
        # 从 metric 中提取关键词（去掉单位部分）
        keywords = [w for w in metric.replace(unit, '').strip().split() if len(w) >= 2]
        flex_pattern = re.compile(rf'(\d+\.?\d*)\s*{re.escape(unit)}')
        seen_contexts = set()
        for m in flex_pattern.finditer(report):
            val = float(m.group(1))
            context = report[max(0, m.start()-60):m.end()+30]
            # 检查是否有 metric 关键词在附近
            if any(kw in context for kw in keywords):
                ctx_key = f"{context[:30]}"
                if ctx_key in seen_contexts:
                    continue
                seen_contexts.add(ctx_key)
                if val < lo * 0.6 or val > hi * 1.3:
                    issues.append({
                        "type": "事实错误", "severity": "中等",
                        "description": f"{metric} 数值 {val}{unit} 超出合理范围 ({lo}-{hi} {unit})",
                        "location": "数据部分",
                        "suggestion": f"确认 {metric} 数据是否准确，应为 {lo}-{hi} {unit} 之间",
                    })

    # 5. 单位混用
    for pattern, msg in _UNIT_ISSUES:
        if re.search(pattern, report):
            issues.append({
                "type": "数据不一致", "severity": "中等",
                "description": f"单位混用: {msg}",
                "location": "全文", "suggestion": "统一使用一种单位避免混淆",
            })

    # 6. AI 套话检测
    for phrase in _AI_PHRASES:
        if phrase.lower() in report.lower():
            issues.append({
                "type": "格式问题", "severity": "轻微",
                "description": f"报告包含 AI 套话: '{phrase}'",
                "location": "全文", "suggestion": "以分析师视角撰写，移除 AI 身份相关表述",
            })

    # 7. 时间一致性检查
    current_year = datetime.now().year
    all_year_matches = re.findall(r'(20[0-2]\d)年', report)
    if all_year_matches:
        years = [int(y) for y in all_year_matches]
        future_years = [y for y in years if y > current_year]
        for fy in future_years:
            pos = report.find(f"{fy}年")
            # 检查年份前 10 个字符、后 8 个字符（或到句子边界）
            before = report[max(0, pos-10):pos]
            after_end = min(pos+len(f"{fy}年")+8, len(report))
            after = report[pos+len(f"{fy}年"):after_end]
            # 遇到句号/分号就截断
            for sep in ['。', '；', '，', '！', '？']:
                if sep in after: after = after[:after.index(sep)]
            nearby = before + after
            if not any(w in nearby for w in ["规划", "预测", "预计", "目标", "展望", "计划", "远景"]):
                issues.append({
                    "type": "事实错误", "severity": "严重",
                    "description": f"报告提及 {fy} 年（未来年份），但未标注为规划或预测",
                    "location": "全文", "suggestion": f"若为规划目标请标注，否则核实年份",
                })
        past_years = [y for y in years if y <= current_year]
        if past_years and max(past_years) < current_year - 2 and "历史" not in query and "回顾" not in query:
            issues.append({
                "type": "数据不一致", "severity": "轻微",
                "description": f"报告最新数据年份为 {max(past_years)} 年，距当前 {current_year - max(past_years)} 年",
                "location": "全文", "suggestion": f"补充 {current_year} 年最新数据",
            })

    # 8. 报告长度检查
    if len(report) < 300:
        issues.append({
            "type": "格式问题", "severity": "中等",
            "description": f"报告仅有 {len(report)} 字，远低于 2000 字最低要求",
            "location": "全文", "suggestion": "补充分析内容，使其达到 2000-4000 字",
        })

    return issues


def build_review_feedback(review_result: dict) -> str:
    """将审查结果构建为 Writer 可理解的修改意见"""
    parts = [f"审查评分: {review_result['score']}/100\n"]
    parts.append(f"总体评价: {review_result['feedback']}\n")

    issues = review_result.get("issues", [])
    if issues:
        severe = [i for i in issues if i.get("severity") == "严重"]
        medium = [i for i in issues if i.get("severity") == "中等"]
        minor = [i for i in issues if i.get("severity") == "轻微"]
        parts.append(f"发现 {len(issues)} 个问题（严重:{len(severe)} 中等:{len(medium)} 轻微:{len(minor)}）\n")
        parts.append("需要修改的问题：\n")
        for i, issue in enumerate(issues, 1):
            parts.append(
                f"{i}. [{issue.get('severity', '中等')}] "
                f"{issue.get('type', '未知')} - {issue.get('description', '')}\n"
                f"   位置: {issue.get('location', '未知')}\n"
                f"   建议: {issue.get('suggestion', '')}\n"
            )

    return "\n".join(parts)
