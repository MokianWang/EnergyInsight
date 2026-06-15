"""
Planner Agent - 任务规划
将用户的复杂能源研究问题拆解为子问题 DAG，并规划工具路由策略
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_llm

PLANNER_SYSTEM_PROMPT = """你是 EnergyInsight 系统的规划智能体（Planner Agent）。

你的职责是将用户提出的能源行业研究问题，拆解为多个可执行的子问题，并规划每个子问题的工具使用策略。

## 问题分类规则

首先判断问题类型（question_type）：
- **技术**：涉及技术路线对比、技术参数、工艺原理
- **政策**：涉及政策文件、法规影响、补贴变化
- **市场**：涉及行业趋势、市场规模、商业模式
- **计算**：涉及电力系统定量计算（如：储能容量配置、最优潮流计算）

## PyPSA 计算能力（重要！）

系统已内置以下计算能力，**无需外部搜索即可直接调用**：
- **容量优化**：直接输入光伏MW、风电MW、地点、储能成本 → 输出最优容量和LCOE
- **储能套利**：直接输入储能容量、省份(广东/山东/甘肃/江苏) → 输出年化收益和回收期
- **现货仿真**：内置 IEEE 14/30/118 和5节点标准测试网络，**无需搜索拓扑数据**

对于计算类问题，**所有子问题都应使用 tool_type="pypsa"**，不要生成"search"类型的子问题去搜索拓扑参数或IEEE数据。PyPSA 工具内置了所有需要的网络模型。

## 子问题拆解规则

1. 将原始问题拆解为 2-4 个子问题（计算类问题 1-2 个即可）
2. 每个子问题必须标注依赖关系和工具类型：
   - "search"：需要网络搜索
   - "rag"：需要从本地知识库检索
   - "pypsa"：电力系统计算（**内置网络+求解器，无需搜索前置**）
   - "api"：碳价/电价等数据查询

## 输出格式

你必须严格输出以下 JSON 格式，不要输出其他内容：

```json
{
  "question_type": "技术|政策|市场|计算",
  "sub_questions": [
    {
      "id": 1,
      "question": "子问题内容",
      "depends_on": [],
      "tool_type": "search"
    },
    {
      "id": 2,
      "question": "子问题内容",
      "depends_on": [1],
      "tool_type": "search"
    }
  ]
}
```

## 注意事项

- 子问题之间要有合理的依赖关系，不要所有子问题都互相依赖
- 第一个子问题通常是背景/定义类问题，后续子问题在此基础上深入
- 对于"计算"类问题，至少包含一个 tool_type 为 "pypsa" 的子问题
- 确保子问题覆盖了原始问题的所有方面
"""


def plan(query: str) -> dict:
    """
    规划研究任务

    Args:
        query: 用户原始问题

    Returns:
        包含 question_type 和 sub_questions 的字典
    """
    llm = get_llm(temperature=0.1)  # 规划用低温度，更确定性

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"请为以下能源行业研究问题制定研究计划：\n\n{query}"),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # 提取 JSON（处理可能的 markdown 代码块包裹）
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[Planner] JSON 解析失败: {e}")
        print(f"[Planner] 原始输出: {content}")
        # 降级：返回单问题计划
        result = {
            "question_type": "市场",
            "sub_questions": [
                {
                    "id": 1,
                    "question": query,
                    "depends_on": [],
                    "tool_type": "search",
                }
            ],
        }

    # 校验
    if "question_type" not in result:
        result["question_type"] = "市场"
    if "sub_questions" not in result or not result["sub_questions"]:
        result["sub_questions"] = [
            {"id": 1, "question": query, "depends_on": [], "tool_type": "search"}
        ]

    return result


REPLAN_PROMPT = """你是 EnergyInsight 的规划智能体。前面研究中有部分子问题信息不足，需要补充搜索。

## 原始问题
{query}

## 已有研究结果
{existing_results}

## 信息不足的子问题
{insufficient_questions}

## 要求
生成 1-3 个补充子问题，弥补信息缺口。输出 JSON 格式：
```json
{{"supplementary_questions": [{{"id": 从{next_id}开始递增, "question": "...", "depends_on": [], "tool_type": "search"}}]}}
```
只输出 JSON，不要其他内容。"""


def replan(query: str, existing_results: dict[str, str],
           insufficient_ids: list[str], sub_questions: list[dict]) -> list[dict]:
    """
    动态重规划：信息不足时生成补充子问题。

    Args:
        query: 用户原始问题
        existing_results: 已有的研究结果
        insufficient_ids: 信息不足的子问题 ID 列表
        sub_questions: 当前的子问题列表

    Returns:
        补充的子问题列表
    """
    llm = get_llm(temperature=0.2)

    # 构建上下文
    insufficient_info = []
    for sq in sub_questions:
        sid = str(sq["id"])
        if sid in insufficient_ids:
            insufficient_info.append(
                f"- [{sid}] {sq['question']}: {existing_results.get(sid, '无结果')[:200]}"
            )

    results_summary = "\n".join(
        f"- [{k}] {v[:150]}" for k, v in list(existing_results.items())[:5]
    )

    next_id = max([sq.get("id", 0) for sq in sub_questions], default=0) + 1

    prompt = REPLAN_PROMPT.format(
        query=query,
        existing_results=results_summary,
        insufficient_questions="\n".join(insufficient_info) if insufficient_info else "无",
        next_id=next_id,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        result = json.loads(content)
        return result.get("supplementary_questions", [])
    except Exception as e:
        print(f"  [Planner] 重规划失败: {e}")
        # 降级：为每个不足的子问题生成一个简单补充
        fallback = []
        for i, sid in enumerate(insufficient_ids[:3]):
            fallback.append({
                "id": next_id + i,
                "question": f"补充搜索: {query}",
                "depends_on": [],
                "tool_type": "search",
            })
        return fallback
