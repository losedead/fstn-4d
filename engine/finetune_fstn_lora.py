# -*- coding: utf-8 -*-
"""
finetune_fstn_lora.py — FSTN-4D QLoRA 微调脚本（文档 §7 Step 8/9）

用 train_fstn_4d.jsonl（400 条，含 agent_thought/tool_calls/final_response）
对中文基座模型做 QLoRA SFT，让模型内化 FSTN-4D 的行为协议。

关键设计：
- QLoRA 4bit：RTX 5060 8GB 可跑 7B 模型
- 目标格式：把 agent_thought + tool_calls 编译为训练目标，
  让模型学会"看到用户话语 → 先走情绪/感知/记忆推理链 → 再回复"
- 输出：adapter 权重（peft），可 merge 回基座

用法：
  python finetune_fstn_lora.py --base Qwen/Qwen2.5-7B-Instruct \
      --data training_data/train_fstn_4d.jsonl \
      --output ./fstn_lora_out --epochs 3 --lora_r 16

注意：首次运行会下载基座模型（需网络），或改用本地已有模型
（如 qwen3.5:9b 的 GGUF 需先转换，或直接用 HF 的 Qwen2.5-1.5B-Instruct 快速验证）。
"""

import argparse
import json
import os
import sys

# 国内网络：默认走 hf-mirror.com 镜像（可被环境变量覆盖）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ── 训练格式：把样本编译为指令微调格式 ─────────────────────

SYSTEM_SHORT = (
    "你是搭载 FSTN-4D 情感记忆引擎的陪伴型 Agent。"
    "处理每句话时必须先做情绪检测（艾克曼六维+复杂情绪）、"
    "感知追踪（七维）、记忆检索（含虫洞/通感/潜意识），"
    "再给出体贴的回复。"
)


def build_prompt(sample: dict) -> str:
    """把 FSTN-4D 样本编译为可训练 prompt。"""
    user_text = sample["messages"][1]["content"]
    thought = sample.get("agent_thought", "")
    tool_calls = sample.get("tool_calls", [])
    calls_repr = json.dumps(tool_calls, ensure_ascii=False)
    lines = [
        f"用户：{user_text}",
        f"[思维链] {thought}",
        f"[引擎调用] {calls_repr}",
    ]
    return "\n".join(lines)


def build_target(sample: dict) -> str:
    return sample.get("final_response", "")


def format_chat(sample: dict) -> dict:
    """转换为 ChatML 格式（system + user + assistant）。"""
    prompt = build_prompt(sample)
    target = build_target(sample)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_SHORT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
    }


def load_data(path: str):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ── 主训练流程 ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="基座模型（HF 名称或本地路径）")
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "training_data", "train_fstn_4d.jsonl"))
    ap.add_argument("--output", default="./fstn_lora_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true",
                    help="只做数据编译检查，不实际训练")
    args = ap.parse_args()

    # 1. 加载并编译数据
    raw = load_data(args.data)
    print(f"加载样本: {len(raw)} 条")
    dataset = [format_chat(s) for s in raw]
    if args.dry_run:
        print("\n[Dry-run] 数据编译检查（前 2 条）:")
        for d in dataset[:2]:
            print("─" * 50)
            for m in d["messages"]:
                print(f"  [{m['role']}] {m['content'][:120]}")
        print(f"\n[Dry-run] 数据就绪，共 {len(dataset)} 条。")
        return

    # 2. 导入训练库（延迟导入，dry_run 不需要）
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
            BitsAndBytesConfig, DataCollatorForSeq2Seq,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
    except ImportError as e:
        print(f"缺少训练依赖: {e}")
        print("请先安装: pip install torch transformers peft trl datasets bitsandbytes accelerate")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("警告：未检测到 CUDA GPU，训练会很慢。推荐 RTX 30/40/50 系列。")
        device_map = "cpu"
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)} 显存 {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        device_map = "auto"

    # 3. 4bit 量化加载（QLoRA）——仅 GPU 可用时启用；CPU 用 fp32 兼容模式
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"加载基座模型（QLoRA 4bit）: {args.base} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
    else:
        print(f"加载基座模型（CPU fp32，无量化）: {args.base} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base, device_map="cpu", trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    # 4. LoRA 配置
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 数据集 tokenize
    def tokenize_fn(examples):
        texts = []
        for msgs in examples["messages"]:
            # 简单拼接（生产可换 apply_chat_template）
            parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                     for m in msgs]
            texts.append("".join(parts) + "<|im_start|>assistant\n")
        tok = tokenizer(texts, truncation=True, max_length=args.max_len,
                        padding=False, return_tensors=None)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    ds = Dataset.from_list(dataset)
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["messages"])
    print(f"Tokenized 数据集: {len(ds)} 条")

    # 6. 训练参数
    # 注意：Blackwell (RTX 50系) + torch 2.9.x + bitsandbytes 4bit 时 AMP 有
    # NotImplementedError（_amp_foreach_non_finite_check_and_unscale_cuda for BFloat16）。
    # 全 fp32 训练（禁用 AMP）：4bit 权重仅 ~1GB + fp32 LoRA 参数很小，8GB 显存足够。
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
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

    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, return_tensors="pt"),
    )

    # 7. 训练
    trainer.train()
    trainer.save_model(args.output)
    print(f"\n✅ 微调完成，adapter 已保存: {args.output}")


if __name__ == "__main__":
    main()
