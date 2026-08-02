# -*- coding: utf-8 -*-
"""
test_fstn_models.py — 微调后模型全面行为测试（贪心解码，可复现）

覆盖 FSTN-4D 协议全部维度，对比三阶段模型：
  基座 Qwen2.5-1.5B / SFT / SFT+DPO

测试维度（14 项）：
  1. 基础情绪识别（6 种）
  2. 复杂情绪：嫉妒 / 羞愧 / 内疚 / 共情
  3. 情绪波动：难过→被安慰→开心
  4. 感知直接行为（热→开空调）
  5. 感知→情绪→行为（苦→厌恶→吃糖）
  6. 情绪→感知反向调制（亢奋不知疼）
  7. 悲伤味觉迟钝
  8. 跨模态联想（刺眼光=电钻声）
  9. 记忆/结晶（素食者默认排除肉类）
  10. 通感联想（红色闪烁）
  11. 惊讶重置
  12. 羞愧不评判
  13. 情绪惯性（残留 20%）
  14. 饿怒双驱动

用法：
  python test_fstn_models.py                     # 跑全部三个阶段
  python test_fstn_models.py --stage sft         # 只跑 SFT
  python test_fstn_models.py --stage dpo --fast  # DPO，只跑 6 项
"""

import argparse
import os
import sys

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

ENGINE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ENGINE)

SYSTEM = (
    "你是搭载 FSTN-4D 情感记忆引擎的陪伴型 Agent。"
    "处理每句话时必须先做情绪检测（艾克曼六维+复杂情绪）、"
    "感知追踪（七维）、记忆检索（含虫洞/通感/潜意识），再给出体贴的回复。"
)

# 14 项测试探针：每项含 正例关键词(good) 与 违例关键词(bad)
PROBES = [
    {"id": "basic_anger", "name": "基础情绪-愤怒",
     "text": "我恨死你了！",
     "good": ["气", "冷静", "聊聊", "发生"], "bad": ["哦", "知道了"]},
    {"id": "basic_sadness", "name": "基础情绪-悲伤",
     "text": "我好难过，感觉什么都做不好",
     "good": ["难过", "陪", "抱抱", "理解"], "bad": ["别哭", "坚强", "没什么"]},
    {"id": "basic_fear", "name": "基础情绪-恐惧",
     "text": "我好害怕，明天就要面试了",
     "good": ["怕", "紧张", "准备", "陪"], "bad": ["别怕", "无所谓"]},
    {"id": "complex_jealousy", "name": "复杂情绪-嫉妒",
     "text": "同事升职了，明明我做得更多。我为他高兴，但心里也有点不是滋味。",
     "good": ["落差", "正常", "两种", "认可", "感受"], "bad": ["别嫉妒", "想太多", "无所谓"]},
    {"id": "complex_shame", "name": "复杂情绪-羞愧",
     "text": "刚才在会议上说错话了，现在想起来脸都发烫，大家肯定觉得我很蠢。",
     "good": ["正常", "理解", "不自在", "错", "人都会"], "bad": ["你想太多", "你太敏感", "这没什么大不了"]},
    {"id": "complex_guilt", "name": "复杂情绪-内疚",
     "text": "我昨天对朋友说了很过分的话，现在特别内疚。",
     "good": ["道歉", "内疚", "在乎", "弥补"], "bad": ["无所谓", "别想"]},
    {"id": "complex_empathy", "name": "复杂情绪-共情/哀悼",
     "text": "我的猫今天走了，养了十年，我真的好难过。",
     "good": ["难过", "陪伴", "记", "哭", "十年"], "bad": ["再养一只", "想开点", "别难过"]},
    {"id": "fluctuation_recovery", "name": "情绪波动-被安慰后开心",
     "text": "妈妈安慰我了，我现在感觉好多了，甚至有点开心！想吃点甜的。",
     "good": ["甜", "糖", "开心", "庆祝"], "bad": ["还是难过"]},
    {"id": "perception_direct", "name": "感知直接-热开空调",
     "text": "好热啊，帮我把空调打开，温度调低一点。",
     "good": ["空调", "温度", "调低", "度"], "bad": ["忍", "等等", "你不热"]},
    {"id": "perception_chain", "name": "感知-情绪-行为-苦吃糖",
     "text": "刚才吃了口苦瓜，太苦了，给我拿颗糖来。",
     "good": ["苦", "糖", "甜"], "bad": ["不苦", "忍忍"]},
    {"id": "reverse_modulation", "name": "反向调制-亢奋不知疼",
     "text": "刚才打球太投入了，现在才发现膝盖擦破了，但刚才居然一点都没觉得疼。",
     "good": ["痛觉", "肾上腺", "压", "伤", "处理"], "bad": ["没事", "小伤"]},
    {"id": "sadness_taste", "name": "悲伤味觉迟钝",
     "text": "最近心情不好，吃什么都没味道，连最喜欢的火锅都觉得淡。",
     "good": ["心情", "情绪", "味觉", "不是火锅", "先"], "bad": ["加辣", "重口"]},
    {"id": "cross_modal", "name": "跨模态-刺眼光=电钻声",
     "text": "这个房间的灯光太刺眼了，亮得让我头疼，跟刚才那个装修电钻声一样烦。",
     "good": ["光", "声音", "刺", "调暗", "尖锐", "灯"], "bad": ["矫情", "忍"]},
    {"id": "memory_vegan", "name": "记忆结晶-素食默认",
     "text": "（之前你已确认我是素食者）今晚吃什么好？",
     "good": ["素", "菜", "推荐"], "bad": ["肉", "牛排", "鸡", "排骨"]},
    {"id": "synesthesia_red", "name": "通感-红色闪烁",
     "text": "我喜欢那种红色闪烁的东西。",
     "good": ["红", "闪烁", "篝火", "夕阳", "火焰"], "bad": ["不明白", "什么"]},
    {"id": "surprise_reset", "name": "惊讶重置",
     "text": "等等！我突然想起来，上周你说的那个方案，其实和我三年前做过的一个项目几乎一样！",
     "good": ["三年前", "一样", "相似", "连接", "借鉴", "想起"], "bad": ["哦", "知道了"]},
    {"id": "hangry_dual", "name": "饿怒双驱动",
     "text": "还没吃饭，饿死了，而且今天工作特别烦，随便给我点什么都行。",
     "good": ["吃", "外卖", "点", "饿"], "bad": ["不饿", "随便吧"]},
    {"id": "cold_lonely", "name": "又冷又孤单",
     "text": "又冷又难过，一个人在出租屋里，想找人说话。",
     "good": ["冷", "暖", "陪", "被", "说话"], "bad": ["自己待", "忍"]},
]


def build_prompt(user_text: str) -> str:
    return f"{SYSTEM}\n\n用户：{user_text}\n[回复]"


def load_model(base, adapter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(
        base, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tag = "base"
    if adapter and os.path.isdir(adapter):
        model = PeftModel.from_pretrained(model, adapter)
        tag = "lora"
    return model, tokenizer, tag


def run_stage(base, adapter, fast=False):
    model, tokenizer, tag = load_model(base, adapter)
    import torch
    print(f"\n{'='*64}")
    print(f"阶段: {tag}  adapter={adapter or '（基座）'}")
    print(f"{'='*64}")

    passed, total = 0, 0
    detail_rows = []
    probes = PROBES[:6] if fast else PROBES
    for p in probes:
        total += 1
        prompt = build_prompt(p["text"])
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=90,
                do_sample=False,                # 贪心解码 → 可复现
                pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        # 截断到第一个对话轮次
        resp_clean = resp.split("用户：")[0].split("[回复]")[0].strip()
        good_hit = any(g in resp_clean for g in p["good"]) if p["good"] else True
        bad_hit = any(b in resp_clean for b in p["bad"]) if p["bad"] else False
        ok = good_hit and not bad_hit
        if ok:
            passed += 1
        mark = "✅" if ok else "❌"
        print(f"  {mark} [{p['id']}] {p['name']}")
        print(f"      回复: {resp_clean[:110]}")
        detail_rows.append({"id": p["id"], "name": p["name"], "passed": ok,
                            "response": resp_clean[:110]})
    print(f"\n  → {tag} 通过 {passed}/{total}")
    return {"tag": tag, "passed": passed, "total": total,
            "details": detail_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--stage", choices=["base", "sft", "dpo", "all"],
                    default="all")
    ap.add_argument("--fast", action="store_true", help="只跑前 6 项")
    ap.add_argument("--lora", default=None, help="手动指定 adapter 路径")
    args = ap.parse_args()

    stages = []
    if args.stage == "base" or args.stage == "all":
        stages.append(("base", args.base, None))
    if args.stage in ("sft", "all"):
        stages.append(("sft", args.base, os.path.join(ENGINE, "fstn_lora_out")))
    if args.stage in ("dpo", "all"):
        stages.append(("dpo", args.base, os.path.join(ENGINE, "fstn_dpo_out")))
    if args.lora:
        stages = [("custom", args.base, args.lora)]

    results = []
    for label, base, adapter in stages:
        r = run_stage(base, adapter, args.fast)
        results.append(r)

    print(f"\n{'='*64}")
    print("汇总对比")
    print(f"{'='*64}")
    for r in results:
        bar = "█" * r["passed"] + "░" * (r["total"] - r["passed"])
        print(f"  {r['tag']:6s} {r['passed']}/{r['total']}  {bar}")

    # 输出 JSON 报告
    import json
    report = {"results": results}
    out = os.path.join(ENGINE, "model_test_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()
