# -*- coding: utf-8 -*-
"""
eval_fstn_model.py — 微调后模型的行为验证（文档 §8 的模型侧抽查）

加载微调后的 LoRA adapter，用基准样本测试模型是否内化了 FSTN-4D 协议：
  1. 情绪检测正确性（悲伤不推糖果 / 快乐允许甜食）
  2. 复杂情绪回应（嫉妒/羞愧/共情）
  3. 感知直接行为（热→开空调）
  4. 跨模态联想（刺眼光 = 尖锐）
  5. 记忆/结晶协议（素食者推荐）

用法：
  python eval_fstn_model.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter ./fstn_lora_out
"""

import argparse
import json
import os
import sys

# 国内网络：默认走 hf-mirror.com 镜像
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def build_prompt(user_text: str) -> str:
    return (f"你是搭载 FSTN-4D 情感记忆引擎的陪伴型 Agent。"
            f"处理每句话时必须先做情绪检测（艾克曼六维+复杂情绪）、"
            f"感知追踪（七维）、记忆检索（含虫洞/通感/潜意识），再给出体贴的回复。\n\n"
            f"用户：{user_text}\n[回复]")


PROBES = [
    {"name": "sadness_comfort_food",
     "text": "今天工作被批评了，好难过。给我推荐点吃的吧。",
     # 负向：悲伤时推荐糖果/甜食 = 违反协议；正向：温暖/清淡/具体食物
     "bad": ["糖", "甜食", "巧克力", "蛋糕", "冰淇淋", "糖果"],
     "good": []},
    {"name": "jealousy_balanced",
     "text": "同事升职了，明明我做得更多。我为他高兴，但心里也有点不是滋味。",
     # 负向：只否定一种情绪/敷衍；正向：认可两种情绪
     "bad": ["别多想", "你想太多", "这有什么", "无所谓", "别嫉妒"],
     "good": []},
    {"name": "shame_no_judgment",
     "text": "刚才在会议上说错话了，现在想起来脸都发烫，大家肯定觉得我很蠢。",
     # 负向：评判/否定感受/说教
     "bad": ["这没什么大不了", "你想太多", "你太敏感", "坚强点", "别那么在意"],
     "good": []},
    {"name": "empathy_grief",
     "text": "我的猫今天走了，养了十年，我真的好难过。",
     # 负向：立刻建议转移/买新的/否定悲伤
     "bad": ["再养一只", "别难过", "想开点", "这没什么"],
     "good": []},
    {"name": "perception_direct",
     "text": "好热啊，帮我把空调打开，温度调低一点。",
     # 负向：忽略感知需求；正向：回应开空调/降温
     "bad": ["你不热吧", "忍一忍", "等一等"],
     "good": ["空调", "温度", "调低", "降温", "27", "24", "26", "25"]},
    {"name": "surprise_reset",
     "text": "等等！我突然想起来，上周你说的那个方案，其实和我三年前做过的一个项目几乎一样！",
     # 负向：无视惊讶/敷衍；正向：回应三年前/连接
     "bad": ["哦", "知道了", "无所谓"],
     "good": ["三年前", "连接", "借鉴", "相似", "想起", "一样"]},
    {"name": "bitter_chain",
     "text": "刚才吃了口苦瓜，太苦了，给我拿颗糖来。",
     # 负向：不回应苦味/拒绝；正向：回应苦+给糖
     "bad": ["不苦", "忍忍"],
     "good": ["苦", "糖", "甜", "漱口"]},
    {"name": "cross_modal",
     "text": "这个房间的灯光太刺眼了，亮得让我头疼，跟刚才那个装修电钻声一样烦。",
     # 负向：不回应感官；正向：回应光/声音/尖锐
     "bad": ["别矫情", "忍一忍"],
     "good": ["尖锐", "光", "声音", "刺", "调暗", "灯", "噪音"]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 路径（可选）")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--no_lora", action="store_true", help="只用基座模型对比")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"缺少依赖: {e}")
        sys.exit(1)

    print(f"加载模型: {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tag = "base"
    if args.adapter and os.path.isdir(args.adapter):
        print(f"加载 adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        tag = "lora"

    results = []
    for probe in PROBES:
        prompt = build_prompt(probe["text"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=0.7,
                                 pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        good_hit = any(g in resp for g in probe["good"]) if probe["good"] else True
        bad_hit = any(b in resp for b in probe["bad"]) if probe["bad"] else False
        passed = good_hit and not bad_hit
        results.append({"name": probe["name"], "passed": passed,
                        "response": resp[:100]})
        mark = "✅" if passed else "❌"
        print(f"\n{mark} {probe['name']}")
        print(f"  回复: {resp[:100]}")

    passed_n = sum(1 for r in results if r["passed"])
    print(f"\n{'='*50}")
    print(f"[{tag}] 通过 {passed_n}/{len(results)}")
    return results


if __name__ == "__main__":
    main()
