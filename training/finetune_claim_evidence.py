"""
QLoRA 微调脚本：Claim-Evidence 三分类幻觉检测

使用 4-bit 量化 + LoRA 在 RTX 3070 Ti / RTX 4090 上微调 Qwen2.5-7B。
显存需求: ~6GB (4-bit), 训练时间: ~1 小时 (500 样本, 3 epochs)

用法:
  python training/finetune_claim_evidence.py \
    --data data/hallucination_dataset/claim_evidence.jsonl \
    --model models/Qwen2.5-7B-Instruct \
    --output models/hallucination-lora \
    --epochs 3
"""

import argparse
import json
import os
import random as _random

_R = _random.Random(42)


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tune claim-evidence classifier")
    p.add_argument("--data", default="data/hallucination_dataset/claim_evidence.jsonl")
    p.add_argument("--model", default="models/Qwen/Qwen2___5-7B-Instruct")
    p.add_argument("--output", default="models/hallucination-lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)       # 8GB VRAM: batch=2
    p.add_argument("--gradient-accumulation", type=int, default=4)  # effective batch = 2*4 = 8
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)     # 三分类任务无需长文本
    p.add_argument("--val-split", type=float, default=0.15)
    return p.parse_args()


def load_dataset(data_path: str) -> list[dict]:
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"加载 {len(samples)} 条标注数据")
    return samples


# 三分类 prompt 模板
PROMPT_TEMPLATE = """<|im_start|>system
You are an energy fact-checker. Given a claim and evidence, classify: support, rebut, or irrelevant. Answer with exactly one word.<|im_end|>
<|im_start|>user
Claim: {claim}
Evidence: {evidence}<|im_end|>
<|im_start|>assistant
{label}"""

LABEL_MAP = {"support": "support", "rebut": "rebut", "irrelevant": "irrelevant"}


def format_sample(sample: dict) -> str:
    return PROMPT_TEMPLATE.format(
        claim=sample["claim"],
        evidence=sample["evidence"],
        label=LABEL_MAP[sample["label"]],
    )


def main():
    args = parse_args()

    # 检查依赖
    try:
        import torch
        assert torch.cuda.is_available(), "CUDA not available"
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            BitsAndBytesConfig, TrainingArguments, Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("安装: pip install torch transformers peft accelerate bitsandbytes")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    # 加载数据
    samples = load_dataset(args.data)
    _R.shuffle(samples)
    split = int(len(samples) * (1 - args.val_split))
    train_samples, val_samples = samples[:split], samples[split:]
    print(f"训练: {len(train_samples)}, 验证: {len(val_samples)}")

    # 格式化
    train_texts = [format_sample(s) for s in train_samples]
    val_texts = [format_sample(s) for s in val_samples]

    # 4-bit 量化配置 (适配 8GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"加载模型: {args.model} (4-bit QLoRA)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # 梯度检查点需要

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize
    def tokenize(texts):
        return tokenizer(
            texts, truncation=True, max_length=args.max_length,
            padding="max_length", return_tensors="pt",
        )

    train_enc = tokenize(train_texts)
    val_enc = tokenize(val_texts)

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]
            self.labels = encodings["input_ids"].clone()

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    train_dataset = SimpleDataset(train_enc)
    val_dataset = SimpleDataset(val_enc)

    # 训练参数 (RTX 3070 Ti 8GB 优化)
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print(f"\n开始 QLoRA 训练 ({args.epochs} epochs, effective batch={args.batch_size * args.gradient_accumulation})...\n")
    trainer.train()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nQLoRA 模型已保存到: {args.output}")
    print(f"推理使用: HallucinationClassifier(backend='lora', model_path='{args.output}')")


if __name__ == "__main__":
    main()
