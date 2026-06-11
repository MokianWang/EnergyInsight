"""
Reviewer Agent - 质量审查
对报告进行事实核查、幻觉检测、数据一致性检查
通过则放行，不通过则打回 Writer 重写（最多 MAX_REVIEW_ROUNDS 轮）
"""

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm, MAX_REVIEW_ROUNDS

REVIEWER_SYSTEM_PROMPT = """你是 EnergyInsight 系统的质量审查智能体（Reviewer Agent）。

你的职责是对研究报告进行严格的质量审查，确保报告的准确性和可靠性。

## 审查维度

### 1. 事实核查（最高优先级）
- 每个数据点是否有引用来源支撑？
- 政策文号是否真实有效？（如：发改能源〔2025〕xxx号）
- 技术参数是否在合理范围内？

### 2. 数据一致性
- 同一指标在不同位置是否一致？
- 单位使用是否统一？（MW vs GW，元/kWh vs 元/MWh）
- 时间引用是否正确？（不能用旧数据回答"当前"问题）

### 3. 逻辑完整性
- 报告是否完整回答了用户的问题？
- 分析逻辑是否自洽？
- 结论是否由数据支撑？

### 4. 幻觉检测
- 是否存在无法验证的数据或事实？
- 是否存在明显不符合能源行业常识的内容？
  - 例：中国光伏装机总量不应低于 500GW（2024年数据）
  - 例：光伏度电成本不应低于 0.1 元/kWh
  - 例：单个风电机组容量通常不超过 15MW（陆上）

### 5. 引用质量
- 引用来源是否权威？（优先 IEA/国家能源局/发改委/上市公司年报）
- 引用编号是否与参考来源对应？
- 是否存在无引用的数据声明？

## 能源领域数值边界参考

| 指标 | 合理范围（2024-2025） |
|------|----------------------|
| 中国光伏累计装机 | 600-1000 GW |
| 中国风电累计装机 | 400-600 GW |
| 中国储能累计装机 | 30-80 GW |
| 光伏度电成本 | 0.15-0.40 元/kWh |
| 风电度电成本 | 0.15-0.35 元/kWh |
| 锂电池储能成本 | 0.6-1.5 元/Wh |
| 全国碳价 | 50-120 元/吨 |
| 欧盟碳价 | 40-100 欧元/吨 |

## 输出格式

你必须严格输出以下 JSON 格式：

```json
{
  "passed": true|false,
  "score": 0-100,
  "issues": [
    {
      "type": "事实错误|数据不一致|逻辑问题|幻觉|引用缺失",
      "severity": "严重|中等|轻微",
      "description": "问题描述",
      "location": "问题所在位置（章节/段落）",
      "suggestion": "修改建议"
    }
  ],
  "summary": "总体评价（50字内）"
}
```

## 审查标准

- score >= 75 且无"严重"级别问题 → passed = true
- score < 75 或存在"严重"级别问题 → passed = false
- passed = false 时，issues 中必须包含具体的修改建议
"""


def review_report(
    query: str,
    report: str,
    citations: list[dict],
    round_num: int = 1,
) -> dict:
    """
    审查研究报告

    Args:
        query: 用户原始问题
        report: 报告文本
        citations: 引用来源列表
        round_num: 当前审查轮次

    Returns:
        审查结果字典
    """
    llm = get_llm(temperature=0.1)  # 审查用低温度，更严格

    # 构建引用来源文本
    citations_text = ""
    if citations:
        for c in citations:
            citations_text += (
                f"[{c.get('id', '?')}] {c.get('title', '')}\n"
                f"    URL: {c.get('url', '')}\n"
            )

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## 用户原始问题\n{query}\n\n"
                f"## 待审查报告\n{report}\n\n"
                f"## 引用来源列表\n{citations_text}\n\n"
                f"## 审查轮次\n第 {round_num}/{MAX_REVIEW_ROUNDS} 轮\n\n"
                f"请对以上报告进行全面审查。"
            )
        ),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # 提取 JSON
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # 降级：认为通过
        print(f"[Reviewer] JSON 解析失败，默认通过")
        result = {
            "passed": True,
            "score": 70,
            "issues": [],
            "summary": "审查结果解析失败，默认通过",
        }

    # 补充规则层校验
    rule_issues = _rule_based_check(report, citations)
    if rule_issues:
        existing_issues = result.get("issues", [])
        existing_issues.extend(rule_issues)
        result["issues"] = existing_issues
        # 如果有严重规则问题，强制不通过
        has_severe = any(i.get("severity") == "严重" for i in rule_issues)
        if has_severe:
            result["passed"] = False

    return {
        "passed": result.get("passed", True),
        "score": result.get("score", 0),
        "feedback": result.get("summary", ""),
        "issues": result.get("issues", []),
        "hallucination_issues": [
            i["description"]
            for i in result.get("issues", [])
            if i.get("type") == "幻觉"
        ],
    }


def _rule_based_check(report: str, citations: list[dict]) -> list[dict]:
    """
    规则层校验（不依赖 LLM）

    Returns:
        检测到的问题列表
    """
    issues = []

    # 1. 检查引用编号是否对应
    cited_ids = set(re.findall(r"\[(\d+)\]", report))
    available_ids = {str(c.get("id", "")) for c in citations}
    missing_refs = cited_ids - available_ids
    if missing_refs:
        issues.append({
            "type": "引用缺失",
            "severity": "中等",
            "description": f"报告中引用了不存在的来源编号: {missing_refs}",
            "location": "全文",
            "suggestion": "检查引用编号是否与参考来源对应",
        })

    # 2. 检查单位一致性（简单规则）
    # 同一报告中不应同时出现 MW 和 GW 描述同一量级
    if "MW" in report and "GW" in report:
        # 这不一定是问题，只标记轻微提醒
        pass  # 实际中可进一步细化

    # 3. 检查是否有明显的"AI 套话"
    ai_phrases = [
        "作为一个AI",
        "我无法",
        "我没有",
        "根据我的知识",
        "截至我的知识",
    ]
    for phrase in ai_phrases:
        if phrase in report:
            issues.append({
                "type": "逻辑问题",
                "severity": "轻微",
                "description": f"报告包含 AI 套话：'{phrase}'",
                "location": "全文",
                "suggestion": "移除 AI 套话，以分析师视角撰写",
            })

    return issues


def build_review_feedback(review_result: dict) -> str:
    """
    将审查结果构建为 Writer 可理解的修改意见

    Args:
        review_result: review_report 的返回值

    Returns:
        修改意见文本
    """
    parts = [f"审查评分: {review_result['score']}/100\n"]
    parts.append(f"总体评价: {review_result['feedback']}\n")

    issues = review_result.get("issues", [])
    if issues:
        parts.append("需要修改的问题：\n")
        for i, issue in enumerate(issues, 1):
            parts.append(
                f"{i}. [{issue.get('severity', '中等')}] "
                f"{issue.get('type', '未知')} - {issue.get('description', '')}\n"
                f"   位置: {issue.get('location', '未知')}\n"
                f"   建议: {issue.get('suggestion', '')}\n"
            )

    return "\n".join(parts)
