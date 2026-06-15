"""
碳价查询工具

爬取公开碳排放权交易行情页面，无需 API Key。
支持中国全国碳市场和欧盟 ETS。
"""

import json
import time
from pathlib import Path

# 缓存文件，避免频繁爬取
_CACHE_PATH = Path(__file__).parent.parent / "data" / "carbon_price_cache.json"
_CACHE_TTL = 3600  # 1 小时


def fetch_carbon_price(market: str = "china") -> dict:
    """
    查询碳市场价格。

    Args:
        market: "china" (全国碳排放权交易市场) 或 "eu" (欧盟 ETS)

    Returns:
        {"answer": str, "data": {...}, "key_metrics": {...}, "citations": [...]}
    """
    # 先检查缓存
    cached = _read_cache(market)
    if cached:
        return cached

    if market == "china":
        result = _fetch_china_carbon()
    elif market == "eu":
        result = _fetch_eu_carbon()
    else:
        return _error_result(f"不支持的市场: {market}。可选: china, eu")

    # 写入缓存
    _write_cache(market, result)
    return result


def _fetch_china_carbon() -> dict:
    """爬取中国全国碳排放权交易市场行情"""
    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError:
        return _static_china_data()

    try:
        # 尝试从公开信息页面获取
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(
                "https://www.cneeex.com/qgtpfqjy/mrgk/",
                headers={"User-Agent": "EnergyInsight/1.0"}
            )
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        # 尝试提取价格数据
        price_elements = soup.select(".price, .last-price, .trade-price, .sj")
        volume_elements = soup.select(".volume, .trade-num, .cj")

        price = None
        for el in price_elements:
            text = el.get_text(strip=True)
            try:
                price = float(text.replace(",", "").replace("元", "").strip())
                break
            except ValueError:
                continue

        volume = None
        for el in volume_elements:
            text = el.get_text(strip=True)
            try:
                volume = float(text.replace(",", "").replace("吨", "").strip())
                break
            except ValueError:
                continue

    except Exception as e:
        print(f"[CarbonPrice] 爬取失败: {e}，使用静态参考数据")
        return _static_china_data()

    if price is None:
        return _static_china_data()

    return _format_result("china", price, volume)


def _fetch_eu_carbon() -> dict:
    """获取欧盟碳价 (EU ETS)"""
    try:
        import httpx
    except ImportError:
        return _static_eu_data()

    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(
                "https://www.eex.com/en/market-data/environmental-markets/spot-market",
                headers={"User-Agent": "EnergyInsight/1.0"}
            )
            resp.raise_for_status()
        # 欧盟碳价页面需要 JS 渲染，简单爬取可能失败
        return _static_eu_data()
    except Exception:
        return _static_eu_data()


def _static_china_data() -> dict:
    """中国碳市场静态参考数据 (2025年Q1平均水平)"""
    price = 85.0   # 元/吨
    volume = 120000  # 吨/日
    return _format_result("china", price, volume, is_static=True)


def _static_eu_data() -> dict:
    """欧盟碳市场静态参考数据"""
    price = 72.0   # 欧元/吨 (~580 元/吨)
    volume = 25000  # 吨/日
    return _format_result("eu", price, volume, is_static=True)


def _format_result(market: str, price: float, volume=None, is_static=False) -> dict:
    """统一格式化碳价查询结果"""
    unit = "欧元/吨" if market == "eu" else "元/吨"
    market_name = "欧盟碳排放交易体系 (EU ETS)" if market == "eu" else "中国全国碳排放权交易市场"
    static_note = " (静态参考数据，非实时行情)" if is_static else ""

    answer = (
        f"{market_name}{static_note}:\n"
        f"- 最新成交价: {price:.2f} {unit}\n"
        + (f"- 日成交量: {volume:,.0f} 吨\n" if volume else "")
        + f"- 更新时间: {time.strftime('%Y-%m-%d %H:%M')}"
    )

    return {
        "answer": answer,
        "data": {
            "market": market,
            "price": round(price, 2),
            "unit": unit,
            "volume_tons": volume,
            "is_static": is_static,
        },
        "key_metrics": {
            "carbon_price": f"{price:.2f}{unit}",
        },
        "citations": [
            {
                "title": market_name,
                "url": "https://www.cneeex.com/" if market == "china" else "https://www.eex.com/",
                "snippet": f"碳价 {price:.2f} {unit}" + (" (静态参考)" if is_static else ""),
                "source_type": "scraped" if not is_static else "static_reference",
            }
        ],
    }


def _error_result(msg: str) -> dict:
    return {
        "answer": f"碳价查询失败: {msg}",
        "data": {},
        "key_metrics": {},
        "citations": [],
    }


def _read_cache(market: str) -> dict | None:
    """读取缓存"""
    try:
        if _CACHE_PATH.exists():
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
            entry = cache.get(market, {})
            if time.time() - entry.get("timestamp", 0) < _CACHE_TTL:
                return entry.get("result")
    except Exception:
        pass
    return None


def _write_cache(market: str, result: dict):
    """写入缓存"""
    try:
        cache = {}
        if _CACHE_PATH.exists():
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
        cache[market] = {"timestamp": time.time(), "result": result}
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
