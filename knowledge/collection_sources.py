"""
权威官方能源文档采集列表
仅收录政府机构、国际组织的公开报告和 PDF 文件
"""

# ============================================================
# 国际能源机构（英文 PDF）
# ============================================================
OFFICIAL_SOURCES_EN = [
    # --- EIA 美国能源信息署 ---
    # Annual Energy Outlook 2025
    "https://www.eia.gov/outlooks/aeo/pdf/2025/AEO2025-narrative.pdf",

    # --- IRENA 国际可再生能源署 ---
    # Renewable Energy Statistics 2025
    "https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2025/Jul/IRENA_DAT_RE_Statistics_2025.pdf",
    # Renewable Capacity Statistics 2025
    "https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2025/Mar/IRENA_RE_Capacity_Statistics_2025.pdf",
    # Renewable Power Generation Costs in 2024
    "https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2025/Jun/IRENA_Renewable_Power_Generation_Costs_in_2024.pdf",

    # --- UN 联合国 ---
    "https://sdgs.un.org/sites/default/files/2025-11/PV%20plus_UN%20report_2025_EN.pdf",
]

# ============================================================
# 中国官方机构（中文 PDF，已验证可访问）
# ============================================================
OFFICIAL_SOURCES_CN = [
    # --- 国家能源局 ---
    # 中国新型储能发展报告2025（54页PDF，含2024年数据汇总和2025年展望）
    "https://www.nea.gov.cn/20250731/1d40d09f75714280a9218d5bea178fbd/228ab8631dcc4b2293c84dc1f07e6e42.pdf",

    # --- 国家发改委 ---
    # 关于促进新型储能并网和调度运用的通知（国能发科技规〔2024〕26号）
    # HTML格式，采集器会自动保存
    "http://zfxxgk.nea.gov.cn/2024-04/02/c_1310771072.htm",

    # --- 国家能源局 2024年新型储能新闻发布会全文 ---
    "https://www.nea.gov.cn/2024-04/29/c_1212357869.htm",

    # --- 国务院 新型储能制造业高质量发展行动方案 ---
    "https://www.gov.cn/zhengce/zhengceku/202404/content_6945448.htm",

    # --- 国家能源局 2025新型储能答复 ---
    "https://www.nea.gov.cn/20250826/fc4c01be3b4a4c2c9286e570a19e40c8/c.html",
]
