"""
EnergyInsight FastAPI Server

提供 REST API + Web 前端界面。
"""

import uuid
import os
import threading
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="EnergyInsight", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 静态文件 (前端页面)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# 任务存储 (生产环境应换 Redis/DB)
_tasks: dict[str, dict] = {}
_stop_flags: dict[str, bool] = {}  # 中断信号


class ResearchSettings(BaseModel):
    web_search: Optional[bool] = True
    enable_ner: Optional[bool] = True
    enable_qe: Optional[bool] = True
    enable_bm25: Optional[bool] = True
    enable_reranker: Optional[bool] = True
    enable_pypsa: Optional[bool] = True
    enable_review: Optional[bool] = True
    fast_review: Optional[bool] = False
    enable_lora: Optional[bool] = False
    enable_replan: Optional[bool] = True

class ResearchRequest(BaseModel):
    query: str
    provider: Optional[str] = "qwen"
    web_search: Optional[bool] = True
    settings: Optional[ResearchSettings] = None


class TaskStatus(BaseModel):
    task_id: str
    status: str          # pending | running | completed | failed
    query: str
    created_at: str
    completed_at: Optional[str] = None
    progress: str = ""   # 当前步骤
    report_preview: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "timestamp": datetime.now().isoformat()}


@app.post("/research", response_model=TaskStatus)
def create_research(req: ResearchRequest):
    """发起新的能源研究任务"""
    task_id = str(uuid.uuid4())[:8]
    settings = req.settings or ResearchSettings()
    task = {
        "task_id": task_id,
        "status": "pending",
        "query": req.query,
        "settings": settings.model_dump(),
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "progress": "",
        "result": None,
    }
    _tasks[task_id] = task

    thread = threading.Thread(target=_run_research, args=(task_id, req.query, settings))
    thread.start()

    return TaskStatus(**task)


@app.post("/stop/{task_id}")
def stop_research(task_id: str):
    """中断研究任务"""
    _stop_flags[task_id] = True
    if task_id in _tasks:
        _tasks[task_id]["status"] = "stopped"
    return {"status": "stopped", "task_id": task_id}


@app.get("/status/{task_id}", response_model=TaskStatus)
def get_status(task_id: str):
    """查询任务状态"""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatus(**task)


@app.get("/report/{task_id}")
def get_report(task_id: str):
    """获取完整研究报告"""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] == "running":
        return {"status": "running", "progress": task.get("progress", "")}
    if task["status"] == "failed":
        return {"task_id": task_id, "status": "failed", "report": "", "citations": [], "review_rounds": 0}
    result = task.get("result") or {}
    return {
        "task_id": task_id,
        "query": task["query"],
        "report": result.get("report_draft", "") or result.get("report", "") or str(result),
        "citations": result.get("citations", []),
        "review_rounds": result.get("review_round", 0),
    }


def _run_research(task_id: str, query: str, settings=None):
    """后台执行研究任务"""
    try:
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = "正在规划研究方案..."

        import os
        s = settings or ResearchSettings()

        # 应用设置
        if not s.web_search:
            os.environ["MCP_ENABLED"] = "false"
        if not s.enable_pypsa:
            os.environ["PYPSA_ENABLED"] = "false"
        if not s.enable_review:
            os.environ["MAX_REVIEW_ROUNDS"] = "0"
        if s.fast_review:
            os.environ["REVIEWER_FAST_MODE"] = "true"
        if not s.enable_lora:
            os.environ["HALLUCINATION_BACKEND"] = "rule"  # 跳过 LoRA，用快速规则
        if not s.enable_replan:
            os.environ["MAX_REPLAN_ROUNDS"] = "0"

        # RAG 管线开关传递给 retriever
        os.environ["RAG_NER_ENABLED"] = str(s.enable_ner).lower()
        os.environ["RAG_QUERY_EXPANSION_ENABLED"] = str(s.enable_qe).lower()
        os.environ["BM25_ENABLED"] = str(s.enable_bm25).lower()
        os.environ["RAG_RERANK_ENABLED"] = str(s.enable_reranker).lower()

        stage_names = {
            "planner": "正在拆解问题...",
            "researcher": "正在搜集资料...",
            "pypsa": "⚡ PyPSA 电力系统计算中...",
            "analyst": "正在深度分析...",
            "replanner": "补充研究中...",
            "writer": "正在撰写报告...",
            "reviewer": "正在质量审查...",
        }

        def update_progress(node_name, _state):
            msg = stage_names.get(node_name, node_name)
            _tasks[task_id]["progress"] = msg

        from graph.workflow import run as run_workflow
        result = run_workflow(query, progress_callback=update_progress)

        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["completed_at"] = datetime.now().isoformat()
        _tasks[task_id]["result"] = result
        _tasks[task_id]["progress"] = "Done"
    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["progress"] = str(e)[:200]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
