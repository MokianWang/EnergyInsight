"""
Researcher Agent - 信息搜集（MCP 工具版）
通过 MCP Server 执行搜索和网页抓取，LLM 自主决定工具调用策略

工具架构：
- Playwright MCP → 浏览器自动化（JS渲染、完整网页抓取）
- DuckDuckGo MCP → 免费网络搜索（无限速）
- PyMuPDF        → PDF 解析（本地 Python 库）

降级策略：MCP 不可用时自动切换到自定义工具 (tools/search.py, tools/scraper.py)
"""

import concurrent.futures
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from config.settings import get_llm, MAX_SEARCH_RESULTS, MCP_ENABLED

RESEARCHER_SYSTEM_PROMPT = """你是 EnergyInsight 系统的研究智能体（Researcher Agent）。

你的职责是根据给定的子问题，使用可用工具搜集能源行业信息，并整理出结构化的研究结果。

## 工具使用策略

你有以下工具可用：
- **搜索工具**：搜索关键词获取相关网页（优先使用）
- **浏览器工具**：访问特定 URL 获取详细内容
- **PDF工具**：解析 PDF 文档

工作流程：
1. 先搜索关键词（中英文各一次），获取相关结果列表
2. 对最相关的 2-3 个结果，使用浏览器工具访问获取详细内容
3. 综合所有信息，形成结构化答案

## 能源领域搜索技巧

- 优先搜索权威来源：IEA、国家能源局(nea.gov.cn)、发改委(ndrc.gov.cn)
- 中英文双搜：如"储能市场规模" + "energy storage market size China"
- 注意时效性：优先引用最近 1-2 年的数据

## 最终输出格式

完成所有工具调用后，你必须输出以下 JSON 格式的最终结果：

```json
{
  "answer": "针对子问题的结构化答案摘要（500-1000字），包含关键数据和事实",
  "key_facts": [
    "关键事实1（附数据来源）",
    "关键事实2（附数据来源）"
  ],
  "sources": [
    {
      "title": "来源标题",
      "url": "来源URL",
      "snippet": "来源关键内容摘要（50字内）"
    }
  ]
}
```

## 注意事项
- 答案必须基于工具获取的真实信息，不要编造数据
- 每个关键数据点都必须标注来源
- 如果信息不足，在 answer 中明确说明缺失项
"""


def research_single_mcp(sub_question: dict, question_type: str, tools: list) -> dict:
    """
    使用 MCP 工具对单个子问题执行研究
    LLM 自主决定调用哪些工具、调用几次

    Args:
        sub_question: 子问题字典
        question_type: 问题类型
        tools: MCP 工具列表

    Returns:
        研究结果字典
    """
    q_id = sub_question["id"]
    q_text = sub_question["question"]
    tool_type = sub_question.get("tool_type", "search")

    print(f"  [Researcher] 研究子问题 {q_id}: {q_text[:60]}...")

    llm = get_llm(temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)

    # 构建初始消息
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## 子问题\n{q_text}\n\n"
                f"## 问题类型\n{question_type}\n\n"
                f"请使用工具搜集信息，完成后输出 JSON 格式的研究结果。"
                f"最多进行 {MAX_SEARCH_RESULTS} 次搜索。"
            )
        ),
    ]

    # Tool calling 循环
    tool_map = {tool.name: tool for tool in tools}
    max_iterations = MAX_SEARCH_RESULTS + 3  # 搜索次数 + 页面抓取 + 总结
    search_count = 0

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 如果 LLM 不再调用工具，说明研究完成
        if not response.tool_calls:
            break

        # 执行工具调用
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # 追踪搜索次数
            if "search" in tool_name.lower():
                search_count += 1
                if search_count > MAX_SEARCH_RESULTS:
                    messages.append(ToolMessage(
                        content="已达到最大搜索次数限制，请直接基于已搜集的信息输出结果。",
                        tool_call_id=tool_call["id"],
                    ))
                    continue

            print(f"    [Tool] {tool_name}({str(tool_args)[:80]}...)")

            try:
                tool = tool_map[tool_name]
                # MCP 工具为异步，优先用 ainvoke；降级到 invoke
                try:
                    import asyncio
                    result = asyncio.run(tool.ainvoke(tool_args))
                except (RuntimeError, NotImplementedError):
                    result = tool.invoke(tool_args)
                result_str = str(result)
                # 截断过长的工具返回结果
                if len(result_str) > 5000:
                    result_str = result_str[:5000] + "\n\n[内容已截断]"
            except Exception as e:
                result_str = f"工具调用失败: {e}"
                print(f"    [Tool] 调用失败: {e}")

            messages.append(ToolMessage(
                content=result_str,
                tool_call_id=tool_call["id"],
            ))

    # 提取最终结果
    final_content = response.content.strip() if response.content else ""

    # 尝试解析 JSON
    if "```json" in final_content:
        final_content = final_content.split("```json")[1].split("```")[0].strip()
    elif "```" in final_content:
        final_content = final_content.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(final_content)
    except Exception:
        # 降级：用 LLM 原始输出
        result = {
            "answer": final_content or "未能获取到有效信息",
            "key_facts": [],
            "sources": [],
        }

    # 构建引用列表
    citations = []
    for s in result.get("sources", []):
        citations.append({
            "title": s.get("title", ""),
            "url": s.get("url", ""),
            "snippet": s.get("snippet", ""),
            "source_type": "web_search",
        })

    print(f"    [Result] 答案长度: {len(result.get('answer', ''))} 字, "
          f"引用数: {len(citations)}")

    return {
        "answer": result.get("answer", ""),
        "key_facts": result.get("key_facts", []),
        "citations": citations,
    }


def research_single_fallback(sub_question: dict, question_type: str) -> dict:
    """
    降级方案：MCP 不可用时使用自定义工具

    Args:
        sub_question: 子问题字典
        question_type: 问题类型

    Returns:
        研究结果字典
    """
    from tools.search import energy_search, search
    from tools.scraper import scrape_webpage

    q_id = sub_question["id"]
    q_text = sub_question["question"]

    print(f"  [Researcher] 研究子问题 {q_id}: {q_text[:60]}... (降级模式)")

    # 搜索
    search_results = energy_search(q_text, max_results=MAX_SEARCH_RESULTS)
    if len(search_results) < 2:
        general = search(q_text, max_results=MAX_SEARCH_RESULTS)
        seen = {r["url"] for r in search_results}
        for r in general:
            if r["url"] not in seen:
                search_results.append(r)
                seen.add(r["url"])

    # 抓取 Top 页面
    detailed = []
    for r in search_results[:2]:
        if r.get("url"):
            page = scrape_webpage(r["url"])
            if page.get("content"):
                detailed.append(page)

    # 构建上下文
    context_parts = []
    if search_results:
        context_parts.append("## 搜索结果\n")
        for i, r in enumerate(search_results, 1):
            context_parts.append(
                f"[{i}] {r['title']}\n"
                f"    URL: {r['url']}\n"
                f"    内容: {r['content'][:500]}\n"
            )
    if detailed:
        context_parts.append("\n## 详细页面内容\n")
        for i, p in enumerate(detailed, 1):
            context_parts.append(
                f"[详细{i}] {p['title']}\n"
                f"    URL: {p['url']}\n"
                f"    内容: {p['content'][:2000]}\n"
            )

    context = "\n".join(context_parts) or "未找到相关搜索结果。"

    # LLM 整理
    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## 子问题\n{q_text}\n\n"
                f"## 问题类型\n{question_type}\n\n"
                f"{context}\n\n"
                f"请根据以上搜索结果，整理出结构化研究结果。"
            )
        ),
    ]
    response = llm.invoke(messages)
    content = response.content.strip()

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(content)
    except Exception:
        result = {
            "answer": content,
            "key_facts": [],
            "sources": [
                {"title": r["title"], "url": r["url"], "snippet": r["content"][:100]}
                for r in search_results[:3]
            ],
        }

    citations = [
        {"title": s.get("title", ""), "url": s.get("url", ""),
         "snippet": s.get("snippet", ""), "source_type": "web_search"}
        for s in result.get("sources", [])
    ]

    print(f"    [Result] 答案长度: {len(result.get('answer', ''))} 字, "
          f"引用数: {len(citations)}")

    return {
        "answer": result.get("answer", ""),
        "key_facts": result.get("key_facts", []),
        "citations": citations,
    }


def _research_via_rag(sub_question: dict, question_type: str) -> dict:
    """
    RAG 路径：从本地知识库检索回答子问题

    Args:
        sub_question: 子问题字典
        question_type: 问题类型

    Returns:
        研究结果字典
    """
    from knowledge.retriever import retrieve

    q_id = sub_question["id"]
    q_text = sub_question["question"]

    print(f"  [Researcher] RAG检索子问题 {q_id}: {q_text[:60]}...")

    # 中英文双语检索
    results_cn = retrieve(q_text, top_k=3)
    results_en = retrieve(f"energy {q_text}", top_k=3)

    # 合并去重
    seen = set()
    combined = []
    for r in results_cn.get("results", []) + results_en.get("results", []):
        key = r["content"][:100]
        if key not in seen:
            seen.add(key)
            combined.append(r)

    # 构建上下文
    context_parts = []
    for i, r in enumerate(combined[:5], 1):
        source_tag = f"[{r['source']}]" if r["source"] else ""
        section = f" ({r['section_title']})" if r.get("section_title") else ""
        context_parts.append(
            f"[来源{i}] {source_tag}{section}\n{r['content'][:1500]}"
        )
    context = "\n\n".join(context_parts) if context_parts else "（知识库中未找到相关内容）"

    # LLM 整理答案
    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## 子问题\n{q_text}\n\n"
                f"## 问题类型\n{question_type}\n\n"
                f"## 知识库检索结果\n{context}\n\n"
                f"请基于以上知识库内容整理答案。如果知识库信息不足，请明确说明。"
            )
        ),
    ]
    response = llm.invoke(messages)
    content = response.content.strip()

    # 解析 JSON
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(content)
    except Exception:
        result = {
            "answer": content or "无法从知识库获取有效信息",
            "key_facts": [],
            "sources": [{"title": r.get("section_title", ""), "url": r.get("source_url", ""),
                         "snippet": r["content"][:100]} for r in combined[:3]],
        }

    citations = []
    for r in combined[:5]:
        citations.append({
            "title": r.get("section_title", "知识库片段"),
            "url": r.get("source_url", ""),
            "snippet": r["content"][:100],
            "source_type": "rag",
        })

    print(f"    [Result] 答案长度: {len(result.get('answer', ''))} 字, "
          f"引用数: {len(citations)}")

    return {
        "answer": result.get("answer", ""),
        "key_facts": result.get("key_facts", []),
        "citations": citations,
    }


def research_all(sub_questions: list[dict], question_type: str, progress_cb=None) -> dict:
    """
    对所有子问题执行研究（按依赖关系排序）
    自动检测 MCP 可用性，不可用时降级到自定义工具
    progress_cb: 可选, 子问题级别的进度回调 callback(node_name, state_dict)

    Args:
        sub_questions: 子问题列表
        question_type: 问题类型

    Returns:
        包含 research_results 和 citations 的字典
    """
    # 拓扑排序
    resolved = set()
    ordered = []
    remaining = list(sub_questions)
    max_iter = len(remaining) * 2
    iteration = 0

    while remaining and iteration < max_iter:
        iteration += 1
        for sq in remaining[:]:
            deps = set(sq.get("depends_on", []))
            if deps.issubset(resolved):
                ordered.append(sq)
                resolved.add(sq["id"])
                remaining.remove(sq)

    if remaining:
        print(f"  [Researcher] 警告：存在循环依赖，强制处理剩余子问题")
        ordered.extend(remaining)

    # 尝试初始化 MCP
    mcp_tools = []
    mcp_cleanup = None
    use_mcp = False

    if MCP_ENABLED:
        try:
            from tools.mcp_tools import get_mcp_tools_sync
            mcp_tools, mcp_cleanup = get_mcp_tools_sync()
            use_mcp = True
            tool_names = [t.name for t in mcp_tools]
            print(f"  [Researcher] MCP 工具已加载: {tool_names}")
        except Exception as e:
            print(f"  [Researcher] MCP 初始化失败，降级到自定义工具: {e}")

    # 按 DAG 拓扑序分组，每波并行执行独立子问题
    research_results = {}
    all_citations = []
    citation_id = 1

    # 分组：每波包含所有依赖已解决的子问题
    waves = _group_into_waves(ordered)

    try:
        for wave_idx, wave in enumerate(waves, 1):
            if len(wave) == 1:
                # 单子问题直接执行
                sq = wave[0]
                result = _execute_single(sq, question_type, use_mcp, mcp_tools, progress_cb=progress_cb)
                research_results[str(sq["id"])] = result["answer"]
                for c in result["citations"]:
                    c["id"] = citation_id
                    all_citations.append(c)
                    citation_id += 1
            else:
                # 并行执行独立子问题
                print(f"  [Researcher] 并行批次 {wave_idx}: {len(wave)} 个子问题")
                futures_map = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(wave), 4)) as executor:
                    for sq in wave:
                        future = executor.submit(
                            _execute_single, sq, question_type, use_mcp, mcp_tools
                        )
                        futures_map[future] = sq

                    for future in concurrent.futures.as_completed(futures_map):
                        sq = futures_map[future]
                        try:
                            result = future.result()
                            research_results[str(sq["id"])] = result["answer"]
                            for c in result["citations"]:
                                c["id"] = citation_id
                                all_citations.append(c)
                                citation_id += 1
                        except Exception as e:
                            print(f"  [Researcher] 子问题 {sq['id']} 执行失败: {e}")
                            research_results[str(sq["id"])] = f"研究失败: {e}"
    finally:
        if mcp_cleanup:
            mcp_cleanup()

    return {
        "research_results": research_results,
        "citations": all_citations,
    }


def _group_into_waves(ordered: list[dict]) -> list[list[dict]]:
    """
    将拓扑排序后的子问题按依赖关系分组为并行批次。

    每波包含所有依赖已在前波中解决的子问题。
    无依赖的子问题全都在第一波。
    """
    waves = []
    resolved = set()
    remaining = list(ordered)

    while remaining:
        wave = []
        next_remaining = []
        for sq in remaining:
            deps = set(sq.get("depends_on", []))
            if deps.issubset(resolved):
                wave.append(sq)
            else:
                next_remaining.append(sq)

        if not wave:
            # 防止死循环：将剩余的全部放入最后一波
            wave = next_remaining
            next_remaining = []

        waves.append(wave)
        for sq in wave:
            resolved.add(sq["id"])
        remaining = next_remaining

    return waves


def _execute_single(sq: dict, question_type: str, use_mcp: bool, mcp_tools: list, progress_cb=None) -> dict:
    """执行单个子问题研究，根据 tool_type 路由到正确的处理函数"""
    tool_type = sq.get("tool_type", "search")

    if tool_type == "rag":
        return _research_via_rag(sq, question_type)
    elif tool_type == "pypsa":
        return _execute_pypsa(sq, question_type, progress_cb=progress_cb)
    elif tool_type == "api":
        return _execute_api_tool(sq, question_type)
    elif use_mcp and mcp_tools:
        return research_single_mcp(sq, question_type, mcp_tools)
    else:
        # MCP不可用 → 优先RAG（稳定快速），DuckDuckGo降级仅作最后手段
        try:
            return _research_via_rag(sq, question_type)
        except Exception:
            return research_single_fallback(sq, question_type)


def _execute_pypsa(sq: dict, question_type: str, progress_cb=None) -> dict:
    """
    执行 PyPSA 电力系统计算子问题。
    """
    from tools.pypsa_tool import (
        run_capacity_optimization,
        run_storage_arbitrage,
        run_market_simulation,
    )

    q_text = sq.get("question", "")
    q_id = sq.get("id", "?")
    print(f"  ⚡ [PyPSA] 计算子问题 {q_id}: {q_text[:60]}...")
    if progress_cb:
        progress_cb("pypsa", {"sq_id": q_id, "sq_text": q_text[:40]})

    # 用 LLM 解析参数
    params = _extract_pypsa_params(q_text)
    calc_type = params.get("type", "capacity")

    try:
        if calc_type == "arbitrage":
            result = run_storage_arbitrage(params)
        elif calc_type == "market":
            result = run_market_simulation(params)
        else:
            result = run_capacity_optimization(params)

        result["citations"] = result.get("citations", [])
        result.setdefault("key_facts", [result.get("answer", "")])
        return result

    except Exception as e:
        print(f"  [Researcher] PyPSA 计算失败: {e}")
        return {
            "answer": f"PyPSA 计算失败: {e}",
            "key_facts": [],
            "citations": [],
        }


def _execute_api_tool(sq: dict, question_type: str) -> dict:
    """
    执行数据查询子问题（碳价、电价等）。

    根据子问题内容判断所需数据类型，调用对应工具。
    """
    from tools.carbon_price import fetch_carbon_price
    from tools.data_sources import load_china_price_curves

    q_text = sq.get("question", "")
    q_id = sq.get("id", "?")
    print(f"  [Researcher] API 数据查询 {q_id}: {q_text[:60]}...")

    q_lower = q_text.lower()

    # 判断查询类型
    if any(w in q_lower for w in ["碳价", "碳市场", "碳排放", "carbon price", "carbon market"]):
        market = "eu" if any(w in q_lower for w in ["eu", "欧盟", "欧洲", "europe"]) else "china"
        result = fetch_carbon_price(market)
    elif any(w in q_lower for w in ["电价", "峰谷", "electricity price", "tariff"]):
        # 尝试从问题中提取省份
        provinces = ["广东", "山东", "甘肃", "江苏"]
        province = "广东"
        for p in provinces:
            if p in q_text:
                province = p
                break
        price_data = load_china_price_curves(province)
        if "error" in price_data:
            result = {"answer": price_data["error"], "key_facts": [], "citations": []}
        else:
            peak_avg = sum(price_data["peak"]) / 24
            valley_avg = sum(price_data["valley"]) / 24
            spread = peak_avg - valley_avg
            result = {
                "answer": (
                    f"{province}省峰谷电价 ({price_data['meta']['voltage_level']}):\n"
                    f"- 峰时段均价: {peak_avg:.2f} 元/kWh ({price_data['meta']['peak_hours']})\n"
                    f"- 谷时段均价: {valley_avg:.2f} 元/kWh ({price_data['meta']['valley_hours']})\n"
                    f"- 峰谷价差: {spread:.2f} 元/kWh\n"
                    f"- 数据来源: {price_data['meta']['source']}"
                ),
                "key_facts": [f"{province}峰谷价差 {spread:.2f} 元/kWh"],
                "citations": [{
                    "title": f"{province}省发改委一般工商业电价",
                    "url": "", "snippet": f"峰{peak_avg:.2f} 谷{valley_avg:.2f} 元/kWh",
                    "source_type": "static_data",
                }],
            }
    else:
        result = {
            "answer": f"数据查询类型未识别: {q_text}。支持: 碳价查询、电价查询。",
            "key_facts": [],
            "citations": [],
        }

    return result


def _extract_pypsa_params(query: str) -> dict:
    """
    从自然语言查询中提取 PyPSA 计算参数。

    优先使用 LLM 解析，失败时用正则降级。
    """
    import re

    params = {"type": "capacity"}

    # 判断计算类型
    if any(w in query for w in ["套利", "峰谷", "电价差", "arbitrage"]):
        params["type"] = "arbitrage"
    elif any(w in query for w in ["现货", "出清", "节点电价", "边际电价", "LMP", "market", "潮流", "边际", "节点系统", "IEEE"]):
        params["type"] = "market"

    # 提取数值参数 (正则降级，可靠且快速)
    solar_match = re.search(r'(\d+\.?\d*)\s*(MW|万千瓦|兆瓦).*?(光|光伏|PV|solar)', query)
    if solar_match:
        val = float(solar_match.group(1))
        params["solar_mw"] = val if solar_match.group(2) != "万千瓦" else val * 10

    wind_match = re.search(r'(\d+\.?\d*)\s*(MW|万千瓦|兆瓦).*?(风|wind)', query)
    if wind_match:
        val = float(wind_match.group(1))
        params["wind_mw"] = val if wind_match.group(2) != "万千瓦" else val * 10

    storage_match = re.search(r'(\d+\.?\d*)\s*(MW|MWh|兆瓦时|MWh).*?(储能|电池|storage|battery)', query)
    if storage_match:
        params["storage_mw"] = float(storage_match.group(1))

    # 地理坐标 (中国主要城市)
    city_coords = {
        "北京": (39.9, 116.4), "上海": (31.2, 121.5), "广州": (23.1, 113.3),
        "深圳": (22.5, 114.1), "兰州": (36.1, 103.8), "西宁": (36.6, 101.8),
        "乌鲁木齐": (43.8, 87.6), "呼和浩特": (40.8, 111.7), "西安": (34.3, 108.9),
        "成都": (30.6, 104.1), "武汉": (30.6, 114.3), "南京": (32.1, 118.8),
        "杭州": (30.3, 120.2), "济南": (36.7, 117.0), "沈阳": (41.8, 123.4),
    }
    for city, (lat, lon) in city_coords.items():
        if city in query:
            params["latitude"] = lat
            params["longitude"] = lon
            break

    # 省份 → 电价省份映射
    province_map = {"广东": "广东", "山东": "山东", "甘肃": "甘肃", "江苏": "江苏"}
    params.setdefault("latitude", 38.0)
    params.setdefault("longitude", 102.0)
    params.setdefault("solar_mw", 100)

    for prov_name, prov_key in province_map.items():
        if prov_name in query:
            params["province"] = prov_key
            break

    return params
