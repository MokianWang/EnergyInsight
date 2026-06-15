"""
Writer Agent - 报告生成
基于分析结论和引用来源，生成结构化的能源行业研究报告。
支持四种问题类型的专属模板，以及 PyPSA 计算结果的数据表格嵌入。
"""

from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm

WRITER_SYSTEM_PROMPT = """你是 EnergyInsight 系统的报告撰写智能体（Writer Agent）。

你的职责是基于分析结论和引用来源，生成专业、结构化、数据驱动的能源行业研究报告。

## 报告结构（按问题类型）

### 技术问题模板
```
# [技术名称] 技术经济性分析报告

## 摘要
[100字，核心结论 + 关键数据]

## 1. 技术背景与定义
[技术原理、关键参数定义、发展历程]

## 2. 技术参数分析
| 参数 | 数值 | 对比基准 | 来源 |
[表格形式对比]

## 3. 成本结构分析
### 初始投资 (CAPEX)
### 运维成本 (OPEX)
### 度电成本 (LCOE)

## 4. 市场竞争格局
[主要玩家、市场份额、技术路线对比]

## 5. 政策环境
[相关政策、补贴、标准]

## 6. 结论与展望
[核心结论 + 发展趋势 + 投资建议]

## 参考来源
[1] 来源标题 - URL
```

### 政策问题模板
```
# [政策名称/主题] 政策影响分析

## 摘要
[政策核心 + 影响判断]

## 1. 政策背景
[出台背景、政策目标]

## 2. 核心条款解读
[逐条分析关键条款]

## 3. 利好因素分析
[对各利益相关方的正面影响]

## 4. 利空/风险因素
[执行难度、成本增加、竞争加剧等]

## 5. 区域/时段影响评估
[分区域、分时间段的影响差异]

## 6. 结论与建议

## 参考来源
```

### 市场问题模板
```
# [市场/行业] 市场分析报告

## 摘要

## 1. 市场规模与增速
[历史数据 + 预测]

## 2. 竞争格局
### 主要玩家
### 市场集中度
### 进入壁垒

## 3. SWOT 分析
| 优势 | 劣势 | 机会 | 威胁 |
[表格]

## 4. 驱动因素与制约因素

## 5. 趋势研判

## 6. 结论与建议

## 参考来源
```

### 计算问题模板
```
# [计算主题] 定量分析报告

## 摘要
[计算结果 + 关键假设]

## 1. 计算模型与假设
[模型说明、输入参数、数据来源]

## 2. 计算结果
| 指标 | 数值 | 单位 |
[PyPSA/工具输出的关键指标表格]

## 3. 敏感性分析
[关键参数变化对结果的影响]

## 4. 与行业基准对比
[LCOE、容量等与行业报告对比]

## 5. 结论与建议

## 参考来源
```

## 表格格式（强制要求）

**所有数据对比必须使用 Markdown pipe 表格，严禁使用 HTML <table> 标签。**
格式示例：

| 参数 | 数值 | 单位 | 来源 |
|------|------|------|------|
| 装机容量 | 890 | GW | [1] |
| LCOE | 0.35 | 元/kWh | [2] |

多列表格同样用 pipe 语法，列数不限。SWOT 分析也用 pipe 表格呈现。

## 写作规范（强制执行）

1. **数据驱动**：每个数据点必须有引用编号 [N]
2. **专业术语**：使用能源行业标准术语，首次出现时中英文对照（如 LCOE（平准化度电成本））
3. **表格必须用 pipe 语法**：`| col1 | col2 |`，禁止 `<table>` HTML 标签
4. **单位规范**：
   - 容量：MW（<1GW时）、GW（≥1GW时）
   - 电量：MWh（<1GWh时）、GWh（≥1GWh时）
   - 电价：元/kWh（用户侧）、元/MWh（批发侧）
   - 排放：吨CO₂（<1万吨时）、万吨CO₂（≥1万吨时）
   - 货币：元（国内）、美元（国际对比）
4. **时效标注**：所有数据标注年份，如"2024年中国光伏累计装机达890GW"
5. **客观中立**：不确定的结论标注置信度（高/中/低）
6. **引用锚定**：每个 [N] 必须对应参考来源列表中的一条

## 字数要求
- 摘要：80-120字
- 正文：2000-4000字
- 参考来源：至少列出 5 条

## 禁止行为
- 编造数据或来源
- 使用"作为AI助手""根据我的知识"等套话
- 信息不足时强行编造——应标注"信息有限，建议补充研究"
"""


def write_report(
    query: str,
    question_type: str,
    analysis: str,
    citations: list[dict],
    review_feedback: str = "",
) -> str:
    """
    生成研究报告

    Args:
        query: 用户原始问题
        question_type: 问题类型 (技术/政策/市场/计算)
        analysis: Analyst 的分析结论
        citations: 引用来源列表
        review_feedback: Reviewer 的修改意见（重写时使用）

    Returns:
        Markdown 格式的报告文本
    """
    llm = get_llm(temperature=0.4)

    # 模板提示
    template_hint = _get_template_hint(question_type)

    # 引用来源（格式化，带编号 + URL + 摘要）
    citations_text = ""
    if citations:
        citations_text = "## 可用引用来源（必须使用这些来源，不得编造）\n"
        for c in citations:
            ctype = c.get("source_type", "web")
            type_tag = {"pypsa": "[PyPSA计算]", "rag": "[知识库]",
                        "static_data": "[参考数据]", "scraped": "[实时爬取]"}.get(ctype, "")
            citations_text += (
                f"[{c.get('id', '?')}] {c.get('title', '未知来源')} {type_tag}\n"
                f"    URL: {c.get('url', '')}\n"
                f"    摘要: {c.get('snippet', '')}\n\n"
            )

    # 修改意见
    feedback_text = ""
    if review_feedback:
        feedback_text = (
            f"\n## ⚠️ 审查修改意见（上一轮报告未通过质量审查）\n"
            f"{review_feedback}\n"
            f"**请逐条修正上述问题后重新提交。不得忽略任何一条修改意见。**\n"
        )

    import sys
    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## 用户原始问题\n{query}\n\n"
                f"## 问题类型\n{question_type}\n"
                f"{template_hint}\n"
                f"## 分析结论（基于研究数据）\n{analysis}\n\n"
                f"{citations_text}\n"
                f"{feedback_text}\n"
                f"请根据以上信息，按照 {question_type}类问题模板 撰写完整的能源行业研究报告。"
            )
        ),
    ]

    full = []
    for chunk in llm.stream(messages):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full.append(token)
            try: sys.stdout.write(token)
            except UnicodeEncodeError: sys.stdout.write(token.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            sys.stdout.flush()
    return "".join(full).strip()


def _get_template_hint(question_type: str) -> str:
    """根据问题类型返回模板提示"""
    hints = {
        "技术": "请使用「技术经济性分析报告」模板：技术参数表 → 成本结构 → 竞争格局 → 政策环境 → 结论。",
        "政策": "请使用「政策影响分析」模板：政策背景 → 核心条款 → 利好/利空因素 → 区域评估 → 建议。",
        "市场": "请使用「市场分析报告」模板：市场规模 → 竞争格局 → SWOT → 驱动因素 → 趋势研判。",
        "计算": "请使用「定量分析报告」模板：模型假设 → 计算结果表格 → 敏感性分析 → 行业基准对比 → 结论。"
               "特别注意：PyPSA 计算结果的 key_metrics 数据必须嵌入表格。",
    }
    return hints.get(question_type, "请根据内容选择合适的报告结构。")
