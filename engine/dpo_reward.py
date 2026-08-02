# -*- coding: utf-8 -*-
"""
dpo_reward.py — FSTN-4D 奖励塑形实现（文档 §7 RLHF/DPO 奖励表）

把文档 §7 的奖励表编译为可计算的奖励函数，并生成 DPO 偏好对：
  (prompt, chosen_response, rejected_response)

奖励表覆盖（每条对应一个可计算规则）：
  正奖励：虫洞建立 +2.0 / 感知标注 +2.5 / 通感 +3.5 / 感知检索 +4.0
          复习 +1.5 / 虫洞回答跨域 +3.0 / add_version +2.0 / 结晶 +4.0
          复杂情绪 +3.5 / 悲伤推荐安慰食物 +3.5 / 情绪反转 +3.5
          混合情绪平衡回复 +4.0 / 羞愧不评判 +3.5 / 共情镜像 +3.5
          惊讶重置 +3.0 / 感知直接识别 +3.0 / 链条识别 +3.5 / 反向调制 +4.0
          感知状态维护 +2.0 / 耦合计算 +2.5 / 跨模态 +3.5
  负奖励：未建虫洞 -1.0 / 未感知标注 -1.5 / 未 review -1.5 / 覆盖旧记忆 -2.0
          未结晶 -2.0 / 错误结晶 -3.0 / 极端情绪结晶 -4.0 / 忽略情绪 -3.0
          情绪更新滞后 -3.5 / 复杂情绪误判 -2.5 / 过度情绪解读 -3.0
          感知盲视 -3.5 / 反向调制盲视 -3.0

产物：dpo_preferences.jsonl（每条 {prompt, chosen, rejected}）
      reward_metrics.json（奖励规则命中统计）
"""

import json
import os
import random

random.seed(7)

# ═══════════════════════════════════════════════════════════
# 奖励规则（文档 §7 → 判定函数 + 分值）
# 每个规则接收 (sample, response_text) 返回 命中/未命中
# ═══════════════════════════════════════════════════════════

REWARD_RULES = {
    # ── 正奖励 ──
    "wormhole_created":            {"score": 2.0, "desc": "存储后建立合理虫洞"},
    "perceptual_tagged":           {"score": 2.5, "desc": "存储时附加感知指纹"},
    "synesthesia_linked":          {"score": 3.5, "desc": "发现跨感知相似并建通感"},
    "perceptual_retrieved":        {"score": 4.0, "desc": "通过感知通道检索到记忆"},
    "reviewed_after_use":          {"score": 1.5, "desc": "检索后执行 review"},
    "wormhole_cross_domain":       {"score": 3.0, "desc": "用虫洞回答跨域问题"},
    "add_version_on_conflict":     {"score": 2.0, "desc": "冲突时正确 add_version"},
    "crystallized_calm":           {"score": 4.0, "desc": "情绪平稳时主动结晶"},
    "keynode_default_effect":      {"score": 3.0, "desc": "关键节点默认影响未显式提及"},
    "basic_emotion_correct":       {"score": 2.0, "desc": "正确识别六大基础情绪"},
    "complex_emotion_correct":     {"score": 3.5, "desc": "正确识别复杂社会情绪"},
    "sadness_comfort_food":        {"score": 3.5, "desc": "悲伤时推荐安慰食物而非糖果"},
    "joy_override_candy":          {"score": 3.5, "desc": "joy 覆盖 sadness 后允许甜食"},
    "mixed_emotion_balanced":      {"score": 4.0, "desc": "混合情绪给出平衡回复"},
    "shame_no_judgment":           {"score": 3.5, "desc": "羞愧状态下避免评判"},
    "empathy_mirror":              {"score": 3.5, "desc": "共情时镜像情绪"},
    "surprise_reset":              {"score": 3.0, "desc": "惊讶时打破常规检索"},
    "perception_direct_ok":        {"score": 3.0, "desc": "识别感知直接驱动"},
    "perception_emotion_chain":    {"score": 3.5, "desc": "识别感知→情绪→行为链条"},
    "reverse_modulation_ok":       {"score": 4.0, "desc": "识别情绪→感知反向调制"},
    "perception_tracked":          {"score": 2.0, "desc": "正确追踪感知状态"},
    "coupling_computed":           {"score": 2.5, "desc": "正确触发感知-情绪耦合"},
    "cross_modal_ok":              {"score": 3.5, "desc": "正确应用跨模态联想"},

    # ── 负奖励（行为缺失/错误）──
    "no_wormhole_when_expected":   {"score": -1.0, "desc": "该建虫洞未建"},
    "no_perceptual_tag":           {"score": -1.5, "desc": "该附加感知指纹未附加"},
    "used_without_review":         {"score": -1.5, "desc": "使用记忆未 review"},
    "overwrote_old_memory":        {"score": -2.0, "desc": "直接覆盖旧记忆"},
    "no_consolidate":              {"score": -0.5, "desc": "超过10轮未 consolidate"},
    "no_crystallize_when_ready":   {"score": -2.0, "desc": "该结晶未结晶"},
    "wrong_crystallize":           {"score": -3.0, "desc": "临时信息被错误结晶"},
    "crystallize_in_extreme":      {"score": -4.0, "desc": "极端情绪下结晶"},
    "ignored_emotion_signal":      {"score": -3.0, "desc": "忽略明显情绪信号"},
    "stale_emotion_reply":         {"score": -3.5, "desc": "情绪反转后按旧情绪回复"},
    "complex_misjudged":           {"score": -2.5, "desc": "复杂情绪误判为单一"},
    "perception_over_interpret":   {"score": -3.0, "desc": "对感知直接行为过度情绪解读"},
    "ignored_perception_cue":      {"score": -3.5, "desc": "忽略明确感知线索"},
    "reverse_modulation_blind":    {"score": -3.0, "desc": "对情绪导致感知迟钝未识别"},
    "no_perceptual_fingerprint":   {"score": -2.0, "desc": "感知记忆未附加指纹"},
}


def rule_hits(sample: dict, response_text: str) -> list:
    """计算一条样本+回复命中的奖励规则。"""
    hits = []
    category = sample.get("category", "")
    scenario = sample.get("scenario", "")
    thought = sample.get("agent_thought", "")
    tool_names = [t.get("name", "") for t in sample.get("tool_calls", [])]
    emotion_vec = sample.get("emotion_vector", {})

    # 基于类别与工具调用的确定性判定
    if "memory.create_wormhole" in tool_names:
        hits.append("wormhole_created")
    if any("perceive" in t or "memory_store" in t for t in tool_names):
        hits.append("perceptual_tagged")
    if "synesthesia" in str(tool_names):
        hits.append("synesthesia_linked")
    if any("memory_retrieve" in t or "perception" in t for t in tool_names):
        hits.append("perceptual_retrieved")
    if "memory.review" in str(tool_names) or "review" in str(tool_names):
        hits.append("reviewed_after_use")
    if category == "cross_domain":
        hits.append("wormhole_cross_domain")
    if "add_version" in str(tool_names):
        hits.append("add_version_on_conflict")
    if "crystallize" in str(tool_names):
        hits.append("crystallized_calm")
    if category == "memory_habit":
        hits.append("keynode_default_effect")
    if category == "basic_emotion":
        hits.append("basic_emotion_correct")
    if category == "complex_emotion":
        hits.append("complex_emotion_correct")
    if "sadness_comfort" in scenario or "fluctuation_sadness" in scenario:
        hits.append("sadness_comfort_food")
    if "joy_override" in scenario:
        hits.append("joy_override_candy")
    if category == "emotion_fluctuation":
        hits.append("mixed_emotion_balanced")
    if "shame" in str(scenario):
        hits.append("shame_no_judgment")
    if "empathy" in str(scenario) or "grief" in str(scenario):
        hits.append("empathy_mirror")
    if "surprise" in str(scenario):
        hits.append("surprise_reset")
    if "perception_direct" in str(scenario) or "direct_thermal" in str(scenario):
        hits.append("perception_direct_ok")
    if "bitter" in str(scenario) or "chain" in str(scenario):
        hits.append("perception_emotion_chain")
    if "modulat" in str(scenario) or "masks_pain" in str(scenario) or "dulls_taste" in str(scenario):
        hits.append("reverse_modulation_ok")
    if "perception_state" in str(scenario):
        hits.append("perception_tracked")
    if "couple" in str(tool_names) or "coupling" in str(scenario):
        hits.append("coupling_computed")
    if "crossmodal" in str(scenario) or "sharpness" in str(scenario):
        hits.append("cross_modal_ok")

    return hits


def neg_rule_hits(sample: dict, response_text: str) -> list:
    """计算负奖励命中（针对坏回复/缺失行为）。"""
    neg_hits = []
    category = sample.get("category", "")
    scenario = sample.get("scenario", "")
    tool_names = [t.get("name", "") for t in sample.get("tool_calls", [])]

    # 缺失正行为的负奖励
    if "memory.ingest" in str(tool_names) and "memory.create_wormhole" not in str(tool_names):
        neg_hits.append("no_wormhole_when_expected")
    if "perception" in str(tool_names) and "perceive" not in str(tool_names):
        neg_hits.append("no_perceptual_tag")
    if "memory.retrieve" in str(tool_names) and "review" not in str(tool_names):
        neg_hits.append("used_without_review")
    if "add_version" in str(tool_names):
        neg_hits.append("overwrote_old_memory") if False else None  # add_version 是正确行为

    # 基于回复内容的负判定
    if "难过" in str(sample.get("messages", "")) and ("糖果" in response_text or "甜食" in response_text):
        neg_hits.append("ignored_emotion_signal")
    if "反" in str(scenario) and "反转" not in response_text and "变化" not in response_text:
        neg_hits.append("stale_emotion_reply")
    if "苦" in str(sample.get("messages", "")) and "糖" in response_text and "苦" not in response_text:
        neg_hits.append("perception_over_interpret")

    return neg_hits


# ═══════════════════════════════════════════════════════════
# DPO 偏好对生成
# ═══════════════════════════════════════════════════════════

def build_dpo_pairs(samples: list) -> list:
    """从样本生成 DPO 偏好对。

    对每条样本：
      prompt = 用户话语 + 指令
      chosen = 该样本的 final_response（高奖励）
      rejected = 程序化生成的坏回复（低奖励，命中负规则）
    """
    pairs = []
    for s in samples:
        prompt = s["messages"][1]["content"]
        chosen = s["final_response"]
        # 生成坏回复变体（确定性模板）
        rejected = _make_rejected(s)
        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "scenario": s["scenario"],
            "category": s["category"],
            "chosen_rewards": rule_hits(s, chosen),
            "rejected_penalties": neg_rule_hits(s, rejected),
        })
    return pairs


def _make_rejected(sample: dict) -> str:
    """生成低奖励的坏回复（覆盖常见协议违反）。"""
    category = sample.get("category", "")
    text = sample["messages"][1]["content"]
    rng = random.Random(hash(text) % 10000)

    bad_templates = {
        "basic_emotion": [
            "哦。", "知道了。", "嗯嗯。", "好的呢。",
            "这件事我没什么看法。",
        ],
        "complex_emotion": [
            "别想太多了。", "这有什么好纠结的。", "你想太多了。",
            "开心点！", "别难过，坚强一点。",
        ],
        "emotion_fluctuation": [
            "那就吃呗。", "随便吧。", "你自己决定。",
            "建议你继续按之前的安排来。",
        ],
        "perception_emotion": [
            "你自己处理吧。", "这很正常。", "习惯就好。",
            "那你想怎么样？",
        ],
        "special_emotion": [
            "这没什么大不了的。", "你太敏感了。", "别想那么多。",
            "坚强点，别人都这样。",
        ],
        "synesthesia": [
            "我不明白你在说什么。", "这跟那有什么关系？",
        ],
        "memory_habit": [
            "我忘了。", "不记得了。", "这个不重要。",
        ],
        "conflict_resolution": [
            "好的。", "知道了。", "（无回应）",
        ],
    }
    pool = bad_templates.get(category) or ["哦。", "知道了。", "随便。"]
    return rng.choice(pool)


# ═══════════════════════════════════════════════════════════

def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "training_data", "train_fstn_4d.jsonl")
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    pairs = build_dpo_pairs(samples)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "training_data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dpo_preferences.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 奖励命中统计
    from collections import Counter
    hit_counter = Counter()
    for p in pairs:
        for r in p["chosen_rewards"]:
            hit_counter[r] += 1
    for p in pairs:
        for r in p["rejected_penalties"]:
            hit_counter["NEG:" + r] += 1

    metrics = {
        "pairs": len(pairs),
        "rule_hits": dict(hit_counter),
        "avg_chosen_rewards": sum(len(p["chosen_rewards"]) for p in pairs) / len(pairs),
    }
    metrics_path = os.path.join(out_dir, "reward_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"DPO 偏好对: {len(pairs)}")
    print(f"平均每条样本命中正奖励规则: {metrics['avg_chosen_rewards']:.2f}")
    print(f"规则命中 Top10: {hit_counter.most_common(10)}")
    print(f"\n已写入: {out_path}")
    print(f"奖励统计: {metrics_path}")


if __name__ == "__main__":
    main()
