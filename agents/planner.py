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
- **技术**：涉及技术路线对比、技术参数、工艺原理（如：磷酸铁锂 vs 钠离子电池）
- **政策**：涉及政策文件、法规影响、补贴变化（如：电力现货市场改革影响）
- **市场**：涉及行业趋势、市场规模、商业模式（如：虚拟电厂发展前景）
- **计算**：涉及电力系统定量计算（如：储能容量配置、最优潮流计算）

一个问题可以同时涉及多个类型，以主要类型为准。

## 子问题拆解规则

1. 将原始问题拆解为 3-6 个子问题
2. 每个子问题必须标注依赖关系（depends_on）：
   - 如果子问题 B 需要子问题 A 的结果才能执行，则 B.depends_on = [A.id]
   - 没有依赖的子问题可以并行执行
3. 每个子问题必须标注工具类型（tool_type）：
   - "search"：需要网络搜索（大多数情况）
   - "rag"：需要从本地知识库检索（步骤2启用）
   - "pypsa"：需要电力系统计算（步骤5启用）
   - "api"：需要调用外部数据 API

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
