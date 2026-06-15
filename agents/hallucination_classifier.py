"""
Claim-Evidence 幻觉检测分类器

提供两种后端:
  1. 规则分类器 (默认) — 基于关键词 + 数值比较 + NER，快速且零依赖
  2. LoRA 微调模型 (可选) — 基于 Qwen2.5-7B LoRA，需 GPU

分类器接收 (claim, evidence) 对，输出三分类: 支持 | 反驳 | 无关
"""

import os
import re
from typing import Optional

# 全局单例：避免每次都重新加载 7B 模型
_lora_model = None
_lora_tokenizer = None


class HallucinationClassifier:
    """
    Claim-Evidence 三分类器。

    规则模式: 提取 claim 和 evidence 中的数值 + 实体，比较匹配度。
    LoRA 模式: 使用微调的 Qwen2.5-7B LoRA 模型推理。
    """

    def __init__(self, backend: str = "rule", model_path: Optional[str] = None):
        self.backend = backend
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._ready = False

    def classify(self, claim: str, evidence: str) -> str:
        """
        Returns: "support" | "rebut" | "irrelevant"
        """
        if self.backend == "lora":
            return self._classify_lora(claim, evidence)
        return self._classify_rule(claim, evidence)

    def _classify_rule(self, claim: str, evidence: str) -> str:
        """
        基于规则的三分类。

        策略:
          1. 提取双方数值和单位 → 比较匹配度
          2. 提取双方关键实体 → 比较重叠度
          3. 综合判断
        """
        # 提取数值
        claim_nums = _extract_numbers(claim)
        evid_nums = _extract_numbers(evidence)

        # 提取实体
        claim_ents = _extract_entities(claim)
        evid_ents = _extract_entities(evidence)

        # 无重叠实体 → 无关
        if not claim_ents or not evid_ents:
            if not claim_nums and not evid_nums:
                return "irrelevant"
        if claim_ents and evid_ents:
            overlap = len(claim_ents & evid_ents)
            if overlap == 0:
                return "irrelevant"

        # 有数值 → 比较范围
        if claim_nums and evid_nums:
            for c_val, c_unit in claim_nums:
                for e_val, e_unit in evid_nums:
                    if _normalize_unit(c_unit) == _normalize_unit(e_unit):
                        # 同单位数值比较
                        ratio = c_val / e_val if e_val > 0 else float("inf")
                        if 0.6 <= ratio <= 1.7:
                            return "support"
                        else:
                            return "rebut"

        # 有实体重叠但无数值 → 弱支持
        if claim_ents and evid_ents:
            overlap = len(claim_ents & evid_ents)
            total = len(claim_ents | evid_ents)
            if overlap / total > 0.3:
                return "support"

        return "irrelevant"

    def _classify_lora(self, claim: str, evidence: str) -> str:
        """LoRA 微调模型推理"""
        if not self._ready:
            self._load_lora_model()

        if not self._ready:
            return self._classify_rule(claim, evidence)  # 降级规则

        import torch

        prompt = (
            f"<|im_start|>system\n"
            f"You are an energy fact-checker. Given a claim and evidence, classify the relationship. "
            f"Answer with exactly one word: support, rebut, or irrelevant.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"声明: {claim}\n\n证据: {evidence}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=5, do_sample=False)

        response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        response = response.strip()

        valid_labels = {"support", "rebut", "irrelevant"}
        for label in valid_labels:
            if label in response:
                return label
        return "irrelevant"

    def _load_lora_model(self):
        """加载 LoRA 微调模型 (4-bit, 全局单例避免重复加载)"""
        global _lora_model, _lora_tokenizer

        if _lora_model is not None:
            self._model = _lora_model
            self._tokenizer = _lora_tokenizer
            self._ready = True
            return

        try:
            from peft import PeftModel
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch

            if not self.model_path:
                for p in ["models/hallucination-lora", "models/Qwen/Qwen2___5-7B-Instruct"]:
                    if os.path.exists(os.path.join(p, "adapter_config.json")):
                        self.model_path = p
                        break
                else:
                    raise FileNotFoundError("LoRA adapter not found")

            base_model = "models/Qwen/Qwen2___5-7B-Instruct"
            print(f"[HallucinationClassifier] 加载 4-bit LoRA 模型: {self.model_path}")

            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            _lora_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            _lora_tokenizer.pad_token = _lora_tokenizer.eos_token
            base = AutoModelForCausalLM.from_pretrained(
                base_model, quantization_config=bnb, device_map="auto",
                trust_remote_code=True, torch_dtype=torch.bfloat16,
            )
            _lora_model = PeftModel.from_pretrained(base, self.model_path)
            self._model = _lora_model
            self._tokenizer = _lora_tokenizer
            self._ready = True
            print("[HallucinationClassifier] 4-bit LoRA 模型就绪")
        except Exception as e:
            print(f"[HallucinationClassifier] LoRA 加载失败: {e}，使用规则降级")
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready


# ========== 数值提取 ==========

_UNIT_NORMALIZE = {
    "gw": "gw", "mw": "mw", "kw": "kw",
    "gwh": "gwh", "mwh": "mwh", "kwh": "kwh",
    "元/kwh": "元/kwh", "元/mwh": "元/mwh", "元/wh": "元/wh",
    "元/吨": "元/吨", "欧元/吨": "欧元/吨",
    "%": "%", "次": "次", "万辆": "万辆", "gw以上": "gw", "mw以上": "mw",
}


def _normalize_unit(unit: str) -> str:
    return _UNIT_NORMALIZE.get(unit.lower().strip(), unit.lower().strip())


def _extract_numbers(text: str) -> list[tuple[float, str]]:
    """提取文本中的 (数值, 单位) 对"""
    pattern = re.compile(
        r'(\d+\.?\d*)\s*(GW|MW|kW|GWh|MWh|kWh|元/kWh|元/MWh|元/Wh|元/W|'
        r'元/吨|欧元/吨|亿吨|万吨|%|次|万辆|GW以上|MW以上)'
    )
    return [(float(m.group(1)), m.group(2)) for m in pattern.finditer(text)]


def _extract_entities(text: str) -> set[str]:
    """提取能源领域关键实体"""
    entities = set()

    # 技术实体
    tech_keywords = [
        "光伏", "太阳能", "风电", "风力", "储能", "锂电池", "锂离子", "钠离子",
        "液流电池", "固态电池", "压缩空气", "飞轮", "抽水蓄能", "氢能",
        "钙钛矿", "TOPCon", "异质结", "单晶硅", "多晶硅", "刀片电池",
        "电动汽车", "V2G", "充电桩", "换电",
    ]
    for kw in tech_keywords:
        if kw in text:
            entities.add(kw)

    # 机构实体
    org_keywords = [
        "国家能源局", "发改委", "IEA", "IRENA", "EIA", "BNEF",
        "宁德时代", "比亚迪", "阳光电源", "华为", "特斯拉",
    ]
    for kw in org_keywords:
        if kw in text:
            entities.add(kw)

    # 政策实体
    policy_keywords = [
        "碳达峰", "碳中和", "双碳", "十四五", "十五五",
        "电力现货市场", "绿证", "碳交易", "新能源配储",
    ]
    for kw in policy_keywords:
        if kw in text:
            entities.add(kw)

    return entities
