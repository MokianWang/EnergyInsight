"""
开源数据源加载工具

零 API 依赖。数据来源:
  - renewables.ninja: 开源气象数据 (光伏/风电出力曲线)
  - IEEE 标准网络: PyPSA 内置
  - 中国电价曲线: 本地静态 JSON (各省发改委公开文件)
  - 用户自定义拓扑: MATPOWER .m 或 CSV 格式
"""

import json
from pathlib import Path

_PRICE_CURVES_PATH = Path(__file__).parent.parent / "data" / "price_curves.json"


def load_solar_profile(lat: float, lon: float, year: int = 2024) -> list:
    """
    获取指定地点的 8760 小时光伏出力系数。

    优先使用 renewables.ninja (需安装)，否则用简化正弦模型。

    Args:
        lat: 纬度
        lon: 经度
        year: 年份

    Returns:
        8760 个浮点数的出力系数列表 (0.0-1.0 p.u.)
    """
    try:
        from renewables_ninja import get_pv_profile
        return get_pv_profile(lat, lon, year)
    except ImportError:
        pass

    # 简化模型: 正弦曲线 + 季节变化
    import math
    hours = 8760
    profile = []
    for h in range(hours):
        hour_of_day = h % 24
        day_of_year = h // 24
        # 日照时长随季节变化
        season_factor = 1.0 - 0.4 * math.cos(2 * math.pi * (day_of_year - 172) / 365)
        solar = max(0, math.sin(math.pi * (hour_of_day - 6) / 12))
        solar *= season_factor
        solar *= (0.8 + 0.2 * (hash(f"{h}solar") % 1000) / 1000)
        profile.append(round(solar, 4))
    return profile


def load_wind_profile(lat: float, lon: float, year: int = 2024) -> list:
    """
    获取指定地点的 8760 小时风电出力系数。

    Args:
        lat: 纬度
        lon: 经度
        year: 年份

    Returns:
        8760 个浮点数的出力系数列表 (0.0-1.0 p.u.)
    """
    try:
        from renewables_ninja import get_wind_profile
        return get_wind_profile(lat, lon, year)
    except ImportError:
        pass

    # 简化模型: 随机 + 季节变化
    import math
    hours = 8760
    profile = []
    for h in range(hours):
        day_of_year = h // 24
        season_factor = 1.0 + 0.3 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        wind = 0.3 + 0.15 * season_factor * ((hash(f"{h}wind") % 1000) / 1000 - 0.5) * 2
        wind = max(0, min(1, wind))
        profile.append(round(wind, 4))
    return profile


def load_china_price_curves(province: str = "广东") -> dict:
    """
    加载中国典型省份峰谷电价参考数据。

    Args:
        province: 省份名 ("广东"|"山东"|"甘肃"|"江苏")

    Returns:
        {"province": str, "peak": [...], "flat": [...], "valley": [...], "meta": {...}}
    """
    try:
        with open(_PRICE_CURVES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"error": f"电价数据文件未找到: {_PRICE_CURVES_PATH}"}

    provinces = data.get("provinces", {})
    if province not in provinces:
        available = list(provinces.keys())
        return {"error": f"未找到省份 '{province}'，支持: {available}"}

    return {
        "province": province,
        "peak": provinces[province]["peak"],
        "flat": provinces[province]["flat"],
        "valley": provinces[province]["valley"],
        "meta": {
            "peak_hours": provinces[province]["peak_hours"],
            "valley_hours": provinces[province]["valley_hours"],
            "voltage_level": provinces[province]["voltage_level"],
            "source": data["meta"]["source"],
        },
    }


def load_ieee_network(name: str = "ieee14"):
    """
    加载 IEEE 标准测试网络。

    Args:
        name: "ieee14" | "ieee30" | "ieee118"

    Returns:
        PyPSA Network 对象，或 None (PyPSA 未安装时)
    """
    try:
        import pypsa
    except ImportError:
        return None

    if name == "ieee14":
        return pypsa.examples.ieee14()
    elif name == "ieee30":
        return pypsa.examples.ieee30()
    elif name == "ieee118":
        return pypsa.examples.ieee118()
    else:
        return None


def load_custom_grid(file_path: str) -> dict:
    """
    加载用户自定义电网拓扑。

    支持格式:
      - MATPOWER .m 文件 (行业标准)
      - CSV 目录 (含 buses.csv, lines.csv, generators.csv)

    Args:
        file_path: 拓扑文件路径

    Returns:
        {"network": pypsa.Network, "buses": int, "lines": int, "generators": int,
         "answer": str} 或错误信息
    """
    try:
        import pypsa
    except ImportError:
        return {"error": "PyPSA 未安装。请运行: pip install pypsa"}

    path = Path(file_path)
    if not path.exists():
        return {"error": f"文件未找到: {file_path}"}

    n = pypsa.Network()
    ext = path.suffix.lower()
    parent = path.parent

    try:
        if ext == '.m':
            # MATPOWER 格式
            n.import_from_pypower_network(str(path))
        elif ext == '.csv':
            # CSV 目录批量导入
            n.import_from_csv(str(parent))
        elif ext in ('.h5', '.hdf5', '.nc'):
            # PyPSA 原生格式
            n.import_from_hdf5(str(path))
        else:
            return {
                "error": f"不支持的拓扑格式: {ext}。支持: .m (MATPOWER), .csv (CSV目录), .h5 (PyPSA HDF5)",
            }
    except Exception as e:
        return {"error": f"拓扑解析失败: {e}"}

    return {
        "network": n,
        "buses": len(n.buses),
        "lines": len(n.lines),
        "generators": len(n.generators),
        "loads": len(n.loads),
        "answer": (
            f"成功加载自定义电网拓扑:\n"
            f"- 文件: {path.name}\n"
            f"- 节点: {len(n.buses)}\n"
            f"- 线路: {len(n.lines)}\n"
            f"- 发电机: {len(n.generators)}\n"
            f"- 负荷: {len(n.loads)}"
        ),
    }
