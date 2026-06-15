# EnergyInsight

能源行业多智能体深度研究系统。输入一个能源研究问题，自动完成：任务规划 → 信息搜集 → 定量计算 → 深度分析 → 报告撰写 → 质量审查，端到端生成带引用溯源的行业研究报告。

## 系统架构

```
用户问题 → Planner(DAG拆解) → Researcher(MCP搜索+RAG+PyPSA计算)
  → Analyst(深度分析) → Writer(研报生成) ↔ Reviewer(质量审查, ≤3轮)
  → Markdown 研报(含引用溯源)
```

## 快速开始

### 环境要求

- Python 3.12+（推荐 venv312）
- Node.js（Playwright MCP 浏览器抓取）
- QWEN_API_KEY（阿里云百炼，用于搜索和 LLM）

### 安装

```bash
cd EnergyInsight
python -m venv venv312
venv312\Scripts\pip install -r requirements.txt
venv312\Scripts\pip install pypsa            # 电力系统计算
venv312\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124  # GPU (可选)
```

### 配置

```bash
# 创建 .env 文件
echo QWEN_API_KEY=sk-your-key-here > .env
```

### 构建知识库

```bash
venv312\Scripts\python -m knowledge.collect_official
```

### 启动

```bash
# Web 界面 (推荐)
PYTHONIOENCODING=utf-8 venv312\Scripts\python api_server.py
# 打开 http://localhost:8000

## 核心能力

| 模块 | 功能 |
|------|------|
| Planner | 问题分类 + 子问题 DAG 拆解 + 工具路由 |
| Researcher | MCP 联网搜索 + 知识库 RAG + PyPSA 计算 |
| Analyst | CoT 推理 + 信息充分性判断 + 动态重规划 |
| Writer | 4 种研报模板 + Markdown pipe 表格 |
| Reviewer | 规则层(8项) + 模型层(LoRA) + LLM 层 三级审查 |
| PyPSA | 容量优化 / 储能套利 / 现货市场仿真 |
| RAG Pipeline | NER + BM25 + 向量检索 + RRF 融合 + Reranker |

## 知识库

- 8 份权威能源文档（国家能源局/EIA/IRENA/UN/国务院）
- 159 个文本块，Milvus Lite 向量存储
- 250 条能源术语词典 + jieba 分词 + NER
- BM25 关键词索引 + TF-IDF/BGE 重排序

## API

```bash
POST /research  {"query": "...", "settings": {...}}   # 发起研究
GET  /status/{id}                                      # 查询进度
GET  /report/{id}                                      # 获取报告
```

## 项目结构

```
EnergyInsight/
├── api_server.py          # FastAPI Web 服务 (主入口)
├── Dockerfile             # Docker 部署
├── config/settings.py     # 全局配置
├── agents/                # 6 个 Agent (planner/researcher/analyst/writer/reviewer/classifier)
├── graph/workflow.py      # LangGraph 编排
├── knowledge/             # 知识库 + RAG 管线 + 术语词典
├── tools/                 # MCP/搜索/PyPSA/碳价/数据源
├── training/              # LoRA 微调脚本
├── evaluation/            # 评估数据集 + 消融实验
├── static/                # Web 前端
├── data/                  # 文档/向量库/电价数据 (gitignored)
└── models/                # 本地模型 (gitignored)
```

## 许可证

仅供学习和研究使用。
