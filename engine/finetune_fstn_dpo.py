# -*- coding: utf-8 -*-
"""
finetune_fstn_dpo.py — FSTN-4D DPO 偏好微调（文档 §7 Step 9）

在 SFT 基础上用 dpo_preferences.jsonl 做 DPO，强化协议遵守：
  chosen（高奖励回复）vs rejected（低奖励坏回复）

用法：
  python finetune_fstn_dpo.py --base Qwen/Qwen2.5-1.5B-Instruct \
      --data training_data/dpo_preferences.jsonl \
      --output ./fstn_dpo_out --epochs 2
"""

import argparse
import json
import os
import sys

# 国内网络：默认走 hf-mirror.com 镜像
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "training_data", "dpo_preferences.jsonl"))
    ap.add_argument("--output", default="./fstn_dpo_out")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # 1. 加载偏好数据
    pairs = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"加载 DPO 偏好对: {len(pairs)}")

    if args.dry_run:
        for p in pairs[:2]:
            print("─" * 50)
            print(f"PROMPT:   {p['prompt'][:60]}")
            print(f"CHOSEN:   {p['chosen'][:60]}")
            print(f"REJECTED: {p['rejected'][:60]}")
        print("\n[Dry-run] DPO 数据就绪。")
        return

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"缺少训练依赖: {e}")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("警告：无 CUDA GPU，DPO 训练会非常慢。")
        device_map = "cpu"
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device_map = "auto"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"加载基座模型: {args.base} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb_config,
        device_map=device_map, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    def format_pair(p):
        return {
            "prompt": [{"role": "user", "content": p["prompt"]}],
            "chosen": [{"role": "assistant", "content": p["chosen"]}],
            "rejected": [{"role": "assistant", "content": p["rejected"]}],
        }

    ds = Dataset.from_list([format_pair(p) for p in pairs])
    print(f"DPO 数据集: {len(ds)} 条")

    training_args = DPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(args.output)
    print(f"\n✅ DPO 微调完成: {args.output}")


if __name__ == "__main__":
    main()
