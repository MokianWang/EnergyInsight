"""
官方文档批量采集脚本
运行： python -m knowledge.collect_official
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge.collector import collect_from_urls
from knowledge.collection_sources import OFFICIAL_SOURCES_EN, OFFICIAL_SOURCES_CN


def main():
    print("=" * 50)
    print("EnergyInsight 官方能源文档采集")
    print("=" * 50)

    all_urls = OFFICIAL_SOURCES_EN + OFFICIAL_SOURCES_CN
    print(f"\n共 {len(all_urls)} 个来源:")
    for url in all_urls:
        print(f"  {url[:80]}...")

    print(f"\n开始下载...\n")
    docs = collect_from_urls(all_urls)

    print(f"\n{'='*50}")
    print(f"采集完成: {len(docs)}/{len(all_urls)} 个文档下载成功")
    print(f"保存位置: {os.path.abspath('data/raw/')}")
    for doc in docs:
        print(f"  [{doc['file_type']}] {doc['title'][:50]}")

    failed = len(all_urls) - len(docs)
    if failed:
        print(f"\n{failed} 个下载失败（网络不通或URL失效）")


if __name__ == "__main__":
    main()
