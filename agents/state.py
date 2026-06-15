"""
EnergyInsight 全局状态定义
所有 Agent 共享此状态，通过 LangGraph StateGraph 传递
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class Citation(TypedDict):
    """引用来源"""
    id: int                # 引用编号
    title: str             # 来源标题
    url: str               # 来源 URL
    snippet: str           # 来源摘要
    source_type: str       # 来源类型：web_search / pdf / api


class SubQuestion(TypedDict):
    """子问题"""
    id: int                # 子问题编号
    question: str          # 子问题内容
    depends_on: list[int]  # 依赖的子问题编号列表
    tool_type: str         # 建议使用的工具类型：search / rag / pypsa / api
    answer: str            # 子问题的答案（Researcher 填充）


class ResearchState(TypedDict):
    """
    全局研究状态，在 Agent 之间流转

    数据流：
    Planner  → sub_questions, question_type
    Researcher → research_results, citations
    Analyst  → analysis_conclusions
    Writer   → report_draft
    Reviewer → review_feedback, review_passed, review_round
    """

    # ===== 输入 =====
    query: str                        # 用户原始问题

    # ===== Planner 输出 =====
    question_type: str                # 问题分类：技术/政策/市场/计算
    sub_questions: list[SubQuestion]  # 子问题列表（含依赖关系）

    # ===== Researcher 输出 =====
    research_results: dict[str, str]  # key=子问题ID, value=检索结果摘要
    citations: list[Citation]         # 所有引用来源

    # ===== Analyst 输出 =====
    analysis_conclusions: str         # 分析结论 + 证据链

    # ===== Writer 输出 =====
    report_draft: str                 # 报告草稿（Markdown 格式）

    # ===== Reviewer 输出 =====
    review_feedback: str              # 审查意见
    review_passed: bool               # 是否通过审查
    review_round: int                 # 当前审查轮次（最多 MAX_REVIEW_ROUNDS）
    hallucination_issues: list[str]   # 检测到的幻觉问题列表

    # ===== 动态重规划 (Step 4) =====
    replan_count: int                 # 当前重规划轮次
    information_sufficiency: dict[str, str]  # key=子问题ID, value=sufficient/insufficient

    # ===== 系统 =====
    messages: Annotated[list, add_messages]  # Agent 交互消息记录
    current_step: str                 # 当前执行步骤（用于追踪进度）
