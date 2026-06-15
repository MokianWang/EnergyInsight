"""
Quick Ablation Study: 对比不同 RAG 组件配置的效果
证明 NER、BM25、Reranker 各组件的增量贡献
"""

import os, sys, json, time
from datetime import datetime

os.environ["MCP_ENABLED"] = "false"  # fast mode
os.environ["PYTHONIOENCODING"] = "utf-8"

TEST_QUERIES = [
    "钠离子电池储能度电成本分析",
    "光伏组件效率提升对LCOE的影响",
    "中国新型储能并网政策解读",
]

CONFIGS = {
    "基线 (仅向量)": {
        "enable_ner": False, "enable_query_expansion": False,
        "enable_bm25": False, "enable_reranker": False,
    },
    "+NER": {
        "enable_ner": True, "enable_query_expansion": True,
        "enable_bm25": False, "enable_reranker": False,
    },
    "+NER+BM25": {
        "enable_ner": True, "enable_query_expansion": True,
        "enable_bm25": True, "enable_reranker": False,
    },
    "+NER+BM25+Reranker (完整)": {
        "enable_ner": True, "enable_query_expansion": True,
        "enable_bm25": True, "enable_reranker": True,
    },
}


def run_single(query, config_name, config):
    """Run a single query with given config, return metrics"""
    from knowledge.retriever import retrieve
    start = time.time()
    result = retrieve(query, top_k=5, **config)
    elapsed = time.time() - start
    return {
        "query": query,
        "config": config_name,
        "time_seconds": round(elapsed, 2),
        "result_count": result["total"],
    }


def main():
    print("=" * 60)
    print("EnergyInsight RAG Ablation Study")
    print("=" * 60)

    results = []
    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        for config_name, config in CONFIGS.items():
            r = run_single(query, config_name, config)
            results.append(r)
            print(f"  {config_name}: {r['result_count']} results, {r['time_seconds']}s")

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Avg results per config")
    print("=" * 60)
    by_config = {}
    for r in results:
        c = r["config"]
        if c not in by_config:
            by_config[c] = {"times": [], "counts": []}
        by_config[c]["times"].append(r["time_seconds"])
        by_config[c]["counts"].append(r["result_count"])

    baseline_count = sum(by_config["基线 (仅向量)"]["counts"]) / 3
    print(f"{'Config':<30} {'Avg Results':>12} {'Avg Time':>10} {'vs Baseline':>12}")
    print("-" * 66)
    for name in CONFIGS:
        d = by_config[name]
        avg_c = sum(d["counts"]) / 3
        avg_t = sum(d["times"]) / 3
        delta = f"+{avg_c - baseline_count:.1f}" if avg_c >= baseline_count else f"{avg_c - baseline_count:.1f}"
        print(f"{name:<30} {avg_c:>12.1f} {avg_t:>9.1f}s {delta:>12}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {name: {
            "avg_results": sum(d["counts"])/3,
            "avg_time": sum(d["times"])/3,
        } for name, d in by_config.items()},
    }
    out_path = "evaluation/results/ablation.json"
    os.makedirs("evaluation/results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
