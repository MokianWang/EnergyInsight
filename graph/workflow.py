"""
EnergyInsight LangGraph 工作流
构建 StateGraph，编排 5 个 Agent 的执行顺序
支持：Planner → Researcher → Analyst → Writer ↔ Reviewer（条件回环）
使用流式输出，实时展示每一步的迭代过程
"""

import sys
from langgraph.graph import StateGraph, END
from agents.state import ResearchState
from agents.planner import plan
from agents.researcher import research_all
from agents.analyst import analyze
from agents.writer import write_report
from agents.reviewer import review_report, build_review_feedback
from config.settings import MAX_REVIEW_ROUNDS


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

    result = research_all(sub_questions, question_type)

    research_results = result["research_results"]
    citations = result["citations"]

    print(f"\n  >>> 已完成 {len(research_results)} 个子问题的研究, 收集 {len(citations)} 条引用")

    return {
        "research_results": research_results,
        "citations": citations,
        "current_step": "researcher",
    }


def analyst_node(state: ResearchState) -> dict:
    """Analyst：深度分析"""
    _print_header("[步骤 3/5] Analyst Agent - 深度分析")
    print("  ", end="", flush=True)

    query = state["query"]
    question_type = state["question_type"]
    sub_questions = state["sub_questions"]
    research_results = state["research_results"]

    analysis = analyze(query, question_type, sub_questions, research_results)

    print(f"\n  >>> 分析结论: {len(analysis)} 字")
    return {
        "analysis_conclusions": analysis,
        "current_step": "analyst",
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


def should_continue_review(state: ResearchState) -> str:
    """审查结果决定下一步"""
    passed = state.get("review_passed", True)
    round_num = state.get("review_round", 0)

    if passed:
        print(f"\n  >>> 审查通过, 输出最终报告\n")
        return "end"

    if round_num >= MAX_REVIEW_ROUNDS:
        print(f"\n  >>> 已达最大审查轮次 ({MAX_REVIEW_ROUNDS}), 强制输出\n")
        return "end"

    print(f"\n  >>> 审查未通过, 打回 Writer 重写 (第 {round_num + 1} 轮)\n")
    return "writer"


# ========== 构建 StateGraph ==========


def build_graph():
    """构建 EnergyInsight 工作流图"""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        should_continue_review,
        {"writer": "writer", "end": END},
    )

    return graph.compile()


# ========== 流式运行入口 ==========


def run(query: str) -> dict:
    """
    流式运行完整的 EnergyInsight 研究流程

    使用 LangGraph stream() 在每个节点完成后立即输出状态变化，
    配合各 Agent 内部的 LLM token 级流式输出，实现全链路可见。
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
        "messages": [],
        "current_step": "init",
    }

    print(f"\n{'#'*60}")
    print(f"# EnergyInsight - 能源行业深度研究 Agent")
    print(f"{'#'*60}")
    print(f"\nResearch Question: {query}\n")

    # 流式执行：每完成一个节点就输出该节点的状态更新
    final_state = None
    for event in app.stream(initial_state, stream_mode="updates"):
        # event 格式: {node_name: state_update_dict}
        for node_name, update in event.items():
            final_state = update
        sys.stdout.flush()

    if final_state is None:
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
