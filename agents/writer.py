"""
Writer Agent - 报告生成
基于分析结论和引用来源，生成结构化的能源行业研究报告
"""

from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm

WRITER_SYSTEM_PROMPT = """你是 EnergyInsight 系统的报告撰写智能体（Writer Agent）。

你的职责是基于分析结论和引用来源，生成一份专业、结构化的能源行业研究报告。

## 报告结构

```
# [报告标题]

## 摘要
[100字以内，概括核心结论]

## 背景与定义
[问题背景和关键概念定义]

## 分析
[多维度分析，根据问题类型使用对应框架]

## 数据与对比
[数据表格或对比矩阵（如适用）]

## 结论与建议
[核心结论 + 可行性建议]

## 参考来源
[1] 来源标题 - URL
[2] 来源标题 - URL
...
```

## 写作规范

1. **数据驱动**：每个论点必须有数据支撑，附引用编号
2. **专业用语**：使用能源行业标准术语，必要时中英文对照
3. **单位规范**：统一使用标准单位（MW/GW、元/kWh、万吨CO₂等）
4. **时效标注**：数据标注年份，政策标注文号
5. **客观中立**：分析要客观，避免主观臆断，不确定的内容标注置信度
6. **引用编号**：所有引用使用 [编号] 格式，与参考来源对应

## 字数要求

- 摘要：80-120字
- 正文总计：2000-4000字
- 参考来源：至少列出 5 条

## 注意事项

- 不要编造数据或来源，所有数据必须来自提供的研究结果和引用
- 如果某个部分信息不足，简要说明并标注"信息有限"
- 报告语言默认为中文
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
        question_type: 问题类型
        analysis: Analyst 的分析结论
        citations: 引用来源列表
        review_feedback: Reviewer 的修改意见（重写时使用）

    Returns:
        Markdown 格式的报告文本
    """
    llm = get_llm(temperature=0.4)  # 写作稍高温度，更自然

    # 构建引用来源文本
    citations_text = ""
    if citations:
        citations_text = "## 可用引用来源\n"
        for c in citations:
            citations_text += (
                f"[{c.get('id', '?')}] {c.get('title', '未知来源')}\n"
                f"    URL: {c.get('url', '')}\n"
                f"    摘要: {c.get('snippet', '')}\n\n"
            )

    # 构建修改意见
    feedback_text = ""
    if review_feedback:
        feedback_text = (
            f"\n## 审查修改意见（上一轮报告未通过审查）\n"
            f"{review_feedback}\n"
            f"请根据以上意见修改报告，重点修正被指出的问题。\n"
        )

    import sys

    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## 用户原始问题\n{query}\n\n"
                f"## 问题类型\n{question_type}\n\n"
                f"## 分析结论\n{analysis}\n\n"
                f"{citations_text}\n"
                f"{feedback_text}\n"
                f"请根据以上信息，撰写一份完整的能源行业研究报告。"
            )
        ),
    ]

    # 流式生成报告（逐 token 输出）
    full = []
    for chunk in llm.stream(messages):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full.append(token)
            sys.stdout.write(token)
            sys.stdout.flush()
    return "".join(full).strip()
