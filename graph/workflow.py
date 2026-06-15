"""
EnergyInsight LangGraph 工作流
构建 StateGraph，编排 5 个 Agent 的执行顺序
支持：
  - Planner → Researcher(并行DAG) → Analyst → Writer ↔ Reviewer（审查回环）
  - Analyst → 信息不足 → Planner(replan) → Researcher（动态重规划）
使用流式输出，实时展示每一步的迭代过程
"""

import sys
from langgraph.graph import StateGraph, END
from agents.state import ResearchState
from agents.planner import plan, replan
from agents.researcher import research_all
from agents.analyst import analyze, check_sufficiency
from agents.writer import write_report
from agents.reviewer import review_report, build_review_feedback
from config.settings import get_max_review_rounds, get_max_replan_rounds


# ========== 节点函数 ==========


def planner_node(state: ResearchState) -> dict:
    """Planner：任务拆解"""
    query = state["query"]
    _print_header(f"[步骤 1/5] Planner Agent - 任务规划")
    print(f"  {query}\n")

    result = plan(query)

    question_type = result.get("question_type", "市场")
    sub_questions = result.get("sub_questions", [])

    print(f"  问题类型: {question_type}")
    print(f"  子问题数量: {len(sub_questions)}")
    for sq in sub_questions:
        deps = sq.get("depends_on", [])
        dep_str = f" (依赖: {deps})" if deps else ""
        tool = sq.get("tool_type", "search")
        print(f"    [{sq['id']}] {sq['question'][:60]}... [{tool}]{dep_str}")
    print()

    return {
        "question_type": question_type,
        "sub_questions": sub_questions,
        "current_step": "planner",
    }


def researcher_node(state: ResearchState) -> dict:
    """Researcher：信息搜集"""
    _print_header("[步骤 2/5] Researcher Agent - 信息搜集")

    sub_questions = state["sub_questions"]
    question_type = state["question_type"]

    result = research_all(sub_questions, question_type, progress_cb=None)

    research_results = result["research_results"]
    citations = result["citations"]

    print(f"\n  >>> 已完成 {len(research_results)} 个子问题的研究, 收集 {len(citations)} 条引用")

    return {
        "research_results": research_results,
        "citations": citations,
        "current_step": "researcher",
    }


def analyst_node(state: ResearchState) -> dict:
    """Analyst：深度分析 + 信息充分性判断"""
    _print_header("[步骤 3/5] Analyst Agent - 深度分析")
    print("  ", end="", flush=True)

    query = state["query"]
    question_type = state["question_type"]
    sub_questions = state["sub_questions"]
    research_results = state["research_results"]

    analysis = analyze(query, question_type, sub_questions, research_results)

    # 检查信息充分性
    sufficiency = check_sufficiency(research_results)
    info_suff = sufficiency["sufficiency"]
    overall_suff = sufficiency["overall_sufficient"]
    insufficient_ids = sufficiency["insufficient_ids"]

    if not overall_suff:
        print(f"\n  >>> 信息不足: {len(insufficient_ids)} 个子问题需要补充研究")
    else:
        print(f"\n  >>> 信息充足, 进入撰写阶段")

    print(f"  >>> 分析结论: {len(analysis)} 字")
    return {
        "analysis_conclusions": analysis,
        "information_sufficiency": info_suff,
        "current_step": "analyst",
    }


def replanner_node(state: ResearchState) -> dict:
    """动态重规划：信息不足时生成补充子问题"""
    replan_count = state.get("replan_count", 0)
    _print_header(f"[重规划] Planner Agent - 补充研究 (第 {replan_count + 1} 轮)")

    query = state["query"]
    sub_questions = state["sub_questions"]
    research_results = state["research_results"]
    info_sufficiency = state.get("information_sufficiency", {})

    insufficient_ids = [
        sid for sid, status in info_sufficiency.items()
        if status == "insufficient"
    ]

    supplementary = replan(query, research_results, insufficient_ids, sub_questions)

    if supplementary:
        print(f"  补充 {len(supplementary)} 个子问题:")
        for sq in supplementary:
            print(f"    [{sq['id']}] {sq['question'][:60]}...")

        all_sub_questions = list(sub_questions) + supplementary
    else:
        all_sub_questions = list(sub_questions)

    return {
        "sub_questions": all_sub_questions,
        "replan_count": replan_count + 1,
        "current_step": "replanner",
    }


def writer_node(state: ResearchState) -> dict:
    """Writer：报告生成（流式）"""
    round_num = state.get("review_round", 0)
    if round_num > 0:
        _print_header(f"[步骤 4/5] Writer Agent - 报告重写 (第 {round_num + 1} 轮)")
    else:
        _print_header("[步骤 4/5] Writer Agent - 报告生成")

    query = state["query"]
    question_type = state["question_type"]
    analysis = state["analysis_conclusions"]
    citations = state.get("citations", [])
    review_feedback = state.get("review_feedback", "")

    print("  ", end="", flush=True)
    report = write_report(query, question_type, analysis, citations, review_feedback)

    print(f"\n  >>> 报告: {len(report)} 字")
    return {
        "report_draft": report,
        "current_step": "writer",
    }


def reviewer_node(state: ResearchState) -> dict:
    """Reviewer：质量审查"""
    round_num = state.get("review_round", 0) + 1
    _print_header(f"[步骤 5/5] Reviewer Agent - 质量审查 (第 {round_num} 轮)")

    query = state["query"]
    report = state["report_draft"]
    citations = state.get("citations", [])

    result = review_report(query, report, citations, round_num)

    passed = result["passed"]
    score = result["score"]
    issues = result["issues"]
    hallucination_issues = result.get("hallucination_issues", [])

    status = "PASS" if passed else "FAIL"
    print(f"  评分: {score}/100  [{status}]  问题: {len(issues)} 个")
    for issue in issues:
        sev = issue.get("severity", "?")
        desc = issue.get("description", "")
        print(f"    [{sev}] {desc[:80]}...")

    if not passed:
        feedback_text = build_review_feedback(result)
    else:
        feedback_text = ""

    return {
        "review_passed": passed,
        "review_feedback": feedback_text,
        "review_round": round_num,
        "hallucination_issues": hallucination_issues,
        "current_step": "reviewer",
    }


# ========== 条件路由 ==========


def should_replan(state: ResearchState) -> str:
    """分析完成后判断是否需要动态重规划"""
    max_replan = get_max_replan_rounds()
    if max_replan == 0:
        return "writer"  # 开关关闭：直接跳过重规划

    info_suff = state.get("information_sufficiency", {})
    replan_count = state.get("replan_count", 0)
    insufficient = [k for k, v in info_suff.items() if v == "insufficient"]

    if insufficient and replan_count < max_replan:
        print(f"\n  >>> 触发重规划 (第 {replan_count + 1}/{max_replan} 轮)")
        return "replanner"

    if insufficient:
        print(f"\n  >>> 已达最大重规划轮次 ({max_replan}), 继续执行")
    return "writer"


def should_review(state: ResearchState) -> str:
    """Writer 完成后判断是否需要质量审查"""
    if get_max_review_rounds() == 0:
        print(f"\n  >>> 质量审查已关闭, 跳过")
        return "end"
    return "reviewer"


def should_continue_review(state: ResearchState) -> str:
    """审查结果决定下一步"""
    passed = state.get("review_passed", True)
    round_num = state.get("review_round", 0)

    if passed:
        print(f"\n  >>> 审查通过, 输出最终报告\n")
        return "end"

    if round_num >= get_max_review_rounds():
        print(f"\n  >>> 已达最大审查轮次 ({get_max_review_rounds()}), 强制输出\n")
        return "end"

    print(f"\n  >>> 审查未通过, 打回 Writer 重写 (第 {round_num + 1} 轮)\n")
    return "writer"


# ========== 构建 StateGraph ==========


def build_graph():
    """构建 EnergyInsight 工作流图（含动态重规划）"""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("replanner", replanner_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")

    # 分析后：信息充足→撰写, 信息不足→重规划
    graph.add_conditional_edges(
        "analyst",
        should_replan,
        {"writer": "writer", "replanner": "replanner"},
    )

    # 重规划后：回到 Researcher 执行补充研究
    graph.add_edge("replanner", "researcher")

    # Writer → 质量审查(可关闭) → 条件回环
    graph.add_conditional_edges(
        "writer",
        should_review,
        {"reviewer": "reviewer", "end": END},
    )
    graph.add_conditional_edges(
        "reviewer",
        should_continue_review,
        {"writer": "writer", "end": END},
    )

    return graph.compile()


# ========== 流式运行入口 ==========


def run(query: str, progress_callback=None) -> dict:
    """
    流式运行完整的 EnergyInsight 研究流程

    Args:
        query: 研究问题
        progress_callback: 可选，每完成一个节点时调用 callback(node_name, state_update)
    """
    app = build_graph()

    initial_state = {
        "query": query,
        "question_type": "",
        "sub_questions": [],
        "research_results": {},
        "citations": [],
        "analysis_conclusions": "",
        "report_draft": "",
        "review_feedback": "",
        "review_passed": False,
        "review_round": 0,
        "hallucination_issues": [],
        "replan_count": 0,
        "information_sufficiency": {},
        "messages": [],
        "current_step": "init",
    }

    print(f"\n{'#'*60}")
    print(f"# EnergyInsight - 能源行业深度研究 Agent")
    print(f"{'#'*60}")
    print(f"\nResearch Question: {query}\n")

    # 流式执行：用 values 模式获取全量累积状态（而非 updates 的增量）
    final_state = None
    for event in app.stream(initial_state, stream_mode="values"):
        final_state = event
        node_name = event.get("current_step", "")
        if progress_callback and node_name:
            progress_callback(node_name, event)
        sys.stdout.flush()

    if final_state is None or not isinstance(final_state, dict):
        final_state = initial_state

    # 输出最终报告
    print(f"{'='*60}")
    print(f"RESEARCH COMPLETE")
    print(f"{'='*60}\n")

    report = final_state.get("report_draft", "(no report)")
    print(report)
    print(f"\n{'='*60}")
    print("References")
    print(f"{'='*60}")
    for c in final_state.get("citations", []):
        print(f"  [{c.get('id', '?')}] {c.get('title', '')}")
        url = c.get('url', '')
        if url:
            print(f"      {url}")
    print(f"\nReview rounds: {final_state.get('review_round', 0)}")

    return final_state


def _print_header(title: str):
    print(f"\n{'─'*60}")
    print(f"{title}")
    print(f"{'─'*60}")
