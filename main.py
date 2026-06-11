"""
EnergyInsight - 能源行业多智能体深度研究 Agent 系统

主入口：接收用户问题，执行完整研究流程，输出研究报告

用法：
    python main.py                          # 交互式输入
    python main.py --query "你的研究问题"     # 命令行传入问题
    python main.py --demo                   # 运行演示问题
"""

import sys
import os
import io
import argparse
from datetime import datetime

# 修复 Windows 终端中文/Emoji 编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.workflow import run


DEMO_QUERY = (
    "比亚迪刀片电池在储能场景中的技术经济性分析，"
    "从技术参数、成本结构、市场竞争格局、政策支持四个维度评估其前景。"
)


def main():
    parser = argparse.ArgumentParser(
        description="EnergyInsight - 能源行业深度研究 Agent"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="",
        help="研究问题",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行演示问题",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="",
        help="报告输出文件路径（Markdown）",
    )
    args = parser.parse_args()

    # 确定研究问题
    if args.demo:
        query = DEMO_QUERY
        print(f"[演示模式] 使用预设问题")
    elif args.query:
        query = args.query
    else:
        print("=" * 60)
        print("EnergyInsight - 能源行业深度研究 Agent")
        print("=" * 60)
        print("\n请输入你的能源行业研究问题：")
        print("（示例：2025年虚拟电厂在中国的发展现状和商业模式是什么？）")
        print()
        query = input("> ").strip()
        if not query:
            print("未输入问题，退出。")
            sys.exit(0)

    # 检查环境变量
    _check_env()

    # 运行研究流程
    result = run(query)

    # 保存报告
    report = result.get("report_draft", "")
    output_path = args.output

    if not output_path:
        # 自动生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/report_{timestamp}.md"

    # 确保目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n报告已保存至: {output_path}")


def _check_env():
    """检查必要的环境变量"""
    from config.settings import LLM_PROVIDER, QWEN_API_KEY, DEEPSEEK_API_KEY

    if LLM_PROVIDER == "qwen" and not QWEN_API_KEY:
        print("\n[错误] QWEN_API_KEY 未设置！")
        print("请复制 .env.example 为 .env 并填入通义千问 API Key")
        print("获取方式: https://dashscope.console.aliyun.com/")
        sys.exit(1)

    if LLM_PROVIDER == "deepseek" and not DEEPSEEK_API_KEY:
        print("\n[错误] DEEPSEEK_API_KEY 未设置！")
        print("请复制 .env.example 为 .env 并填入 DeepSeek API Key")
        print("获取方式: https://platform.deepseek.com/")
        sys.exit(1)


if __name__ == "__main__":
    main()
