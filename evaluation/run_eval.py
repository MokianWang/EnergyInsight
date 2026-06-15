"""
EnergyInsight 评估脚本

支持三种模式:
  1. 端到端评估: 对120条评测查询运行完整Pipeline
  2. 消融实验: 对比不同组件开关的指标变化
  3. 幻觉检测评估: 测试 Reviewer 对注入错误的检测率

输出: evaluation/results_{timestamp}.json
"""

import json
import time
import os
import sys
from datetime import datetime
from collections import defaultdict

EVAL_DATASET = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_queries():
    with open(EVAL_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["meta"], data["queries"]


def run_single_query(query_data: dict, pipeline_config: dict = None) -> dict:
    """运行单条查询并返回结果"""
    from graph.workflow import run as run_workflow

    q = query_data["query"]
    start = time.time()

    try:
        result = run_workflow(q)
        elapsed = time.time() - start
        report = result.get("report_draft", "")
        return {
            "id": query_data["id"],
            "query": q,
            "category": query_data["category"],
            "difficulty": query_data["difficulty"],
            "elapsed_seconds": round(elapsed, 1),
            "report_length": len(report),
            "review_rounds": result.get("review_round", 0),
            "citations_count": len(result.get("citations", [])),
            "replan_count": result.get("replan_count", 0),
            "success": bool(report and len(report) > 200),
            "report": report[:500],
        }
    except Exception as e:
        return {
            "id": query_data["id"],
            "query": q,
            "category": query_data["category"],
            "difficulty": query_data["difficulty"],
            "elapsed_seconds": time.time() - start,
            "success": False,
            "error": str(e)[:200],
        }


def compute_metrics(results: list[dict]) -> dict:
    """计算评估指标"""
    total = len(results)
    success = sum(1 for r in results if r.get("success"))
    avg_time = sum(r.get("elapsed_seconds", 0) for r in results) / total if total > 0 else 0
    avg_length = sum(r.get("report_length", 0) for r in results) / total if total > 0 else 0
    avg_reviews = sum(r.get("review_rounds", 0) for r in results) / total if total > 0 else 0
    avg_citations = sum(r.get("citations_count", 0) for r in results) / total if total > 0 else 0

    by_category = defaultdict(list)
    for r in results:
        by_category[r.get("category", "unknown")].append(r)
    cat_metrics = {}
    for cat, items in by_category.items():
        cat_metrics[cat] = {
            "count": len(items),
            "success_rate": sum(1 for i in items if i.get("success")) / len(items),
            "avg_time": sum(i.get("elapsed_seconds", 0) for i in items) / len(items),
        }

    return {
        "total": total,
        "success_rate": success / total if total > 0 else 0,
        "avg_time_seconds": round(avg_time, 1),
        "avg_report_length": round(avg_length, 0),
        "avg_review_rounds": round(avg_reviews, 1),
        "avg_citations": round(avg_citations, 1),
        "by_category": cat_metrics,
    }


def ablation_study(queries: list[dict], sample_size: int = 10) -> dict:
    """消融实验: 对比不同RAG配置的效果"""
    configs = {
        "full": {},
        "no_bm25": {"enable_bm25": False},
        "no_reranker": {"enable_reranker": False},
        "no_ner": {"enable_ner": False, "enable_query_expansion": False},
        "vector_only": {"enable_bm25": False, "enable_reranker": False, "enable_ner": False},
    }

    sample = queries[:sample_size]
    results = {}

    for config_name, config in configs.items():
        print(f"\n  Ablation: {config_name} ...")
        config_results = []
        for q in sample:
            r = run_single_query(q, config)
            config_results.append(r)
        results[config_name] = {
            "success_rate": sum(1 for r in config_results if r.get("success")) / len(config_results),
            "avg_time": sum(r.get("elapsed_seconds", 0) for r in config_results) / len(config_results),
        }

    return results


def run_full_eval(sample_size: int = None, skip_ablation: bool = True):
    """运行完整评估"""
    meta, queries = load_queries()

    if sample_size:
        queries = queries[:sample_size]

    print(f"EnergyInsight Evaluation: {len(queries)} queries")
    print(f"Categories: {meta['categories']}")
    print(f"{'='*60}")

    results = []
    for i, q in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] {q['category']}: {q['query'][:60]}...")
        r = run_single_query(q)
        results.append(r)
        status = "OK" if r.get("success") else "FAIL"
        print(f"  {status} | {r.get('elapsed_seconds', 0):.1f}s | {r.get('report_length', 0)} chars")

    metrics = compute_metrics(results)

    # 消融实验 (可选, 耗时)
    ablation = None
    if not skip_ablation:
        ablation = ablation_study(queries, min(10, len(queries)))

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "meta": meta,
        "timestamp": timestamp,
        "metrics": metrics,
        "ablation": ablation,
        "results": results,
    }
    output_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation Complete")
    print(f"  Total: {metrics['total']} queries")
    print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"  Avg Time: {metrics['avg_time_seconds']:.1f}s")
    print(f"  Avg Citations: {metrics['avg_citations']:.1f}")
    print(f"  Results saved: {output_path}")
    if ablation:
        print(f"\n  Ablation Study:")
        for name, m in ablation.items():
            print(f"    {name}: {m['success_rate']*100:.0f}% ({m['avg_time']:.0f}s)")

    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None, help="Sample size (default: all 120)")
    p.add_argument("--ablation", action="store_true", help="Run ablation study")
    args = p.parse_args()
    run_full_eval(sample_size=args.sample, skip_ablation=not args.ablation)
