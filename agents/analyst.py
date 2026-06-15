"""
Analyst Agent - 深度分析
对 Researcher 搜集的信息进行综合分析、交叉验证，输出结构化结论
"""

from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm

ANALYST_SYSTEM_PROMPT = """你是 EnergyInsight 系统的分析智能体（Analyst Agent）。

你的职责是对研究团队搜集的信息进行深度分析，提取核心结论，并进行交叉验证。

## 分析框架

根据问题类型，选择合适的分析框架：

### 技术问题 → 技术经济性对比框架
- 技术参数对比（使用 Markdown pipe 表格：`| 参数 | 数值 |`，禁止 HTML table）
- 成本结构分析（CAPEX + OPEX）
- 技术成熟度评估
- 未来发展趋势判断

### 政策问题 → 政策影响分析框架
- 政策核心要点提取
- 利好因素分析
- 利空/风险因素分析
- 分区域/分时段影响评估

### 市场问题 → SWOT / PEST 框架
- 市场规模与增速
- 竞争格局分析
- 驱动因素与制约因素
- 趋势研判

### 计算问题 → 定量分析框架
- 计算参数与假设
- 计算过程与结果
- 敏感性分析
- 与行业经验数据对比

## 输出要求

1. 输出 800-1500 字的分析结论
2. 每个结论必须引用具体的数据来源
3. 如果多个来源的数据不一致，必须指出并分析原因
4. 明确标注你的判断依据和置信度（高/中/低）
5. 如果某些方面信息不足，明确指出缺失项

## 输出格式

```json
{
  "analysis": "分析正文（800-1500字，结构化）",
  "key_conclusions": [
    "核心结论1",
    "核心结论2",
    "核心结论3"
  ],
  "confidence": "高|中|低",
  "missing_info": ["缺失的信息项1", "缺失的信息项2"]
}
```
"""


def analyze(
    query: str,
    question_type: str,
    sub_questions: list[dict],
    research_results: dict[str, str],
) -> str:
    """
    对研究结果进行综合分析

    Args:
        query: 用户原始问题
        question_type: 问题类型
        sub_questions: 子问题列表
        research_results: 各子问题的研究结果

    Returns:
        分析结论文本
    """
    llm = get_llm(temperature=0.3)

    # 构建研究结果上下文
    research_context = []
    for sq in sub_questions:
        sq_id = str(sq["id"])
        answer = research_results.get(sq_id, "（未获取到结果）")
        research_context.append(
            f"### 子问题 {sq_id}: {sq['question']}\n{answer}\n"
        )

    context = "\n".join(research_context)

    import sys

    messages = [
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## 用户原始问题\n{query}\n\n"
                f"## 问题类型\n{question_type}\n\n"
                f"## 各子问题研究结果\n{context}\n"
                f"请根据以上研究结果，使用合适的分析框架进行综合分析。"
            )
        ),
    ]

    # 流式生成分析结论（逐 token 输出）
    full = []
    for chunk in llm.stream(messages):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full.append(token)
            try: sys.stdout.write(token)
            except UnicodeEncodeError: sys.stdout.write(token.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            sys.stdout.flush()
    content = "".join(full).strip()

    # 提取 JSON
    if "```json" in content:
        try:
            import json
            json_str = content.split("```json")[1].split("```")[0].strip()
            result = json.loads(json_str)
            # 构建完整的分析文本
            analysis_text = result.get("analysis", content)
            conclusions = result.get("key_conclusions", [])
            confidence = result.get("confidence", "中")
            missing = result.get("missing_info", [])

            full_analysis = f"{analysis_text}\n\n"
            if conclusions:
                full_analysis += "**核心结论：**\n"
                for i, c in enumerate(conclusions, 1):
                    full_analysis += f"{i}. {c}\n"
            full_analysis += f"\n**整体置信度：** {confidence}\n"
            if missing:
                full_analysis += "\n**信息缺失项：**\n"
                for m in missing:
                    full_analysis += f"- {m}\n"

            return full_analysis.strip()
        except Exception:
            pass

    return content


def check_sufficiency(research_results: dict[str, str]) -> dict:
    """
    检查各子问题的信息充分性。

    由 Researcher 执行完毕后调用，判断是否需要触发动态重规划。
    基于启发式规则快速评估，不依赖 LLM 调用。

    Args:
        research_results: 子问题 ID → 研究结果文本的映射

    Returns:
        {
            "overall_sufficient": bool,
            "sufficiency": {id: "sufficient"|"insufficient"},
            "insufficient_ids": [id, ...],
        }
    """
    sufficiency = {}
    insufficient_ids = []

    for sq_id, answer in research_results.items():
        if not answer or len(answer.strip()) < 50:
            # 无结果或结果过短 → 信息不足
            sufficiency[sq_id] = "insufficient"
            insufficient_ids.append(sq_id)
            continue

        # 检查常见的信息不足标记
        low_info_markers = [
            "未找到相关信息", "无法获取数据", "搜索失败", "无结果",
            "no results found", "no data available", "information not found",
            "未获取到结果", "研究失败",
        ]
        is_insufficient = any(m in answer.lower() for m in low_info_markers)

        if is_insufficient:
            sufficiency[sq_id] = "insufficient"
            insufficient_ids.append(sq_id)
        else:
            sufficiency[sq_id] = "sufficient"

    overall = len(insufficient_ids) == 0

    return {
        "overall_sufficient": overall,
        "sufficiency": sufficiency,
        "insufficient_ids": insufficient_ids,
    }

