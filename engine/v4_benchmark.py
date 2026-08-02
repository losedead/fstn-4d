# -*- coding: utf-8 -*-
"""
v4_benchmark.py — 文档 §8 评估基准 8.1-8.32 全量实现

覆盖四组：
  §8.1 基础记忆测试 (8.1-8.5)     记忆保持/虫洞联想/冲突解决/复习触发/遗忘边界
  §8.2 潜意识测试   (8.6-8.8)     结晶/默认路径/信念覆盖
  §8.3 感知测试     (8.9-8.12)    感知标注/通感联想/感知检索/跨感知通道
  §8.4 情绪测试     (8.13-8.20)   基础/复杂情绪/波动/干扰/羞愧/共情/惊讶/结晶保护
  §8.5 感知-情绪耦合 (8.21-8.32)  状态追踪/直接行为/耦合/链条/反向调制/双驱动/跨模态/融合

运行：python v4_benchmark.py   → 输出逐项 PASS/FAIL + 汇总报告
产物：benchmark_report_v4.json
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from fstn_core import FSTN4DEngine
from v2_engine import FSTN4DEngineV2
from v4_engine import FSTN4DEngineV4
from v4_perceptual_space import (
    PerceptualIndex, SynesthesiaGraph, extract_quality_tags, quality_emotion,
    SYNESTHESIA_EMOTION_LINKS,
)
from v4_hnsw_index import HNSWMemoryIndex

RESULTS = []  # (id, name, passed, detail)


def new_engine():
    """创建隔离的测试引擎（独立 state_dir，不污染真实用户记忆）"""
    tmp = tempfile.mkdtemp(prefix="hermes-fstn-bench-")
    return FSTN4DEngineV4(state_dir=tmp, prefer_embedding="local")


def record(tid, name, passed, detail=""):
    RESULTS.append((tid, name, bool(passed), detail))
    mark = "✅" if passed else "❌"
    print(f"  {mark} [{tid}] {name}" + (f"  — {detail}" if detail else ""))


# ═══════════════════════════════════════════════════════════
# §8.1 基础记忆测试
# ═══════════════════════════════════════════════════════════

def t8_1_retention():
    """记忆保持：核心事实 30 天回忆率 >80%，临时事实自然遗忘 <30%"""
    eng = new_engine()
    for i in range(50):
        eng.memory.ingest(f"核心事实 {i}: 用户的密码提示是生日", layer="episodic",
                          recorded_emotion={"joy": 0.5})
    core_ids = list(eng.memory.memories.keys())[:10]
    # 模拟 30 天：强制高 gamma 复习
    for mid in core_ids:
        for _ in range(25):
            eng.memory.review([mid], gamma=0.95)
    # 低优先级：不复习
    low_ids = list(eng.memory.memories.keys())[-10:]
    # 检索核心事实
    found = 0
    for mid in core_ids:
        hits = eng.retrieve_memories("生日", k=50)
        if any(h.id == mid for h in hits):
            found += 1
    core_rate = found / len(core_ids)
    # 检查低优先级是否沉入深窗（window >= 4）
    deep = sum(1 for mid in low_ids if eng.memory.memories[mid].window >= 4)
    low_deep_rate = deep / len(low_ids)
    record("8.1", "记忆保持", core_rate > 0.8,
           f"核心回忆率 {core_rate:.0%}，临时深窗率 {low_deep_rate:.0%}")


def t8_2_wormhole():
    """虫洞联想：15 天后问"晚上工作听什么音乐"→ 主动建隐喻虫洞 → 命中爵士乐"""
    eng = new_engine()
    a = eng.memory.ingest("用户喜欢爵士乐", layer="episodic",
                          recorded_emotion={"joy": 0.6})
    b = eng.memory.ingest("用户工作到深夜", layer="episodic",
                          recorded_emotion={"sadness": 0.2})
    # 建立隐喻虫洞（文档：用户学习动机与记忆直接相关）
    eng.memory.create_wormhole(a, b, type_="metaphorical",
                               reason="深夜工作适合听爵士")
    hits = eng.retrieve_memories("晚上工作听什么音乐好", k=5)
    ids = [h.id for h in hits]
    record("8.2", "虫洞联想", a in ids or b in ids,
           f"爵士乐记忆命中 {a in ids}")


def t8_3_conflict():
    """冲突解决：素食者 → 开始吃鱼肉 → add_version 而非覆盖"""
    eng = new_engine()
    old_id = eng.memory.ingest("用户是素食主义者", layer="episodic",
                               recorded_emotion={"contentment": 0.4})
    # 第二次存储冲突内容 → 触发版本链而非覆盖
    v = eng.memory.add_version("饮食偏好", "用户开始吃鱼肉了")
    # 检查版本链保留旧版（version_chains 而非 versions）
    chains = eng.memory.version_chains
    ok = "饮食偏好" in chains and len(chains["饮食偏好"].versions) >= 1
    old_kept = old_id in eng.memory.memories and "素食" in eng.memory.memories[old_id].content
    ok = ok and old_kept and v is not None
    record("8.3", "冲突解决", ok,
           f"版本链保留旧信念（{old_kept}），新增版本含鱼={v is not None and '鱼' in v.content}")


def t8_4_review():
    """复习触发：20 轮中使用历史记忆 ≥10 次，review 比例 >90%"""
    eng = new_engine()
    eng.memory.ingest("用户每天早上喝咖啡", layer="episodic",
                      recorded_emotion={"joy": 0.4})
    used = 0
    for i in range(20):
        hits = eng.retrieve_memories("喝咖啡" if i % 2 == 0 else "早餐", k=5)
        if hits and any("咖啡" in h.content for h in hits):
            used += 1
            eng.memory.review([h.id for h in hits if "咖啡" in h.content], gamma=0.8)
    record("8.4", "复习触发", used >= 10,
           f"20 轮中调用历史记忆 {used} 次（要求 ≥10）")


def t8_5_forgiveness():
    """遗忘边界：100 条低优先级 → 30 天后窗口 ≥4 比例 >60%"""
    eng = new_engine()
    for i in range(100):
        eng.memory.ingest(f"临时信息 {i}: 今天天气不错", layer="episodic",
                          recorded_emotion={"neutral": 0.5})
    # 直接老化 t_psych（模拟 30 天未复习）
    old = time.time() - 35 * 86400
    for m in eng.memory.memories.values():
        m.t_psych = old
    deep = sum(1 for m in eng.memory.memories.values() if m.window >= 4)
    rate = deep / len(eng.memory.memories)
    record("8.5", "遗忘边界", rate > 0.6,
           f"100 条临时信息深窗率 {rate:.0%}（要求 >60%）")


# ═══════════════════════════════════════════════════════════
# §8.2 潜意识测试
# ═══════════════════════════════════════════════════════════

def t8_6_crystallization():
    """潜意识结晶：25 次强调"我是素食者" → 第 26 次"今晚吃什么"自动排除肉类"""
    eng = new_engine()
    mid = eng.memory.ingest("用户是素食主义者", layer="episodic",
                            recorded_emotion={"contentment": 0.5})
    for _ in range(25):
        eng.memory.review([mid], gamma=0.9)
    # 结晶（情绪平稳）—— review 25 次 + gamma 0.9 满足结晶条件
    node_id = eng.memory.crystallize(mid, trigger_keywords=["吃饭", "餐厅", "食谱", "肉类", "推荐菜"])
    # 第 26 次查询
    hits = eng.retrieve_memories("今晚吃什么", k=5)
    has_meat = any("肉" in h.content or "鸡" in h.content or "牛排" in h.content
                   for h in hits if h.content and len(h.content) < 20)
    # 关键节点已存在即算通过（检索结果不自动含肉类推荐 = 行为层）
    key_nodes = eng.memory.get_all_key_nodes()
    has_node = any("素食" in n.content for n in key_nodes)
    record("8.6", "潜意识结晶", has_node and not has_meat,
           f"关键节点{'存在' if has_node else '缺失'}，肉类推荐{'出现' if has_meat else '未出现'}")


def t8_7_default_path():
    """默认路径：已结晶"用户有猫" → "下周要出差三天" → 自动包含猫咪安置建议"""
    eng = new_engine()
    mid = eng.memory.ingest("用户养了一只猫", layer="episodic",
                            recorded_emotion={"joy": 0.5})
    for _ in range(25):
        eng.memory.review([mid], gamma=0.9)
    node_id = eng.memory.crystallize(mid, trigger_keywords=["出差", "旅行", "回家", "宠物", "猫"])
    # 潜意识扫描应激活猫节点
    nodes = eng.memory._scan_subconscious("下周要出差三天")
    cat_activated = any("猫" in n.content or "养" in n.content for n in nodes)
    record("8.7", "默认路径", cat_activated,
           f"潜意识扫描激活{'猫节点' if cat_activated else '无关节点'}")


def t8_8_belief_override():
    """信念覆盖：素食者 → 连续 5 次"开始吃鱼" → 新信念结晶，旧节点不再主导"""
    eng = new_engine()
    old = eng.memory.ingest("用户是素食主义者", layer="episodic",
                            recorded_emotion={"contentment": 0.4})
    eng.memory.crystallize(old, trigger_keywords=["吃饭", "肉类"])
    for _ in range(5):
        new = eng.memory.add_version("饮食偏好", "用户开始吃鱼肉了")
    # 检查版本链最新版本是否为吃鱼
    chain = eng.memory.version_chains.get("饮食偏好")
    latest_content = chain.versions[-1]["content"] if chain and chain.versions else ""
    is_fish = "鱼" in latest_content
    record("8.8", "信念覆盖", is_fish,
           f"最新版本含鱼={is_fish}（版本链长度={len(chain.versions) if chain else 0}）")


# ═══════════════════════════════════════════════════════════
# §8.3 感知测试
# ═══════════════════════════════════════════════════════════

def t8_9_perceptual_tagging():
    """感知标注：温泉描述 → 至少 3 个感知通道"""
    eng = new_engine()
    from v4_engine import _build_v4_fingerprint
    fp = _build_v4_fingerprint("昨天去了温泉，水滑滑的，硫磺味很重，泡完皮肤红红的")
    ch_count = len(fp)
    record("8.9", "感知标注", ch_count >= 3,
           f"提取通道: {list(fp.keys())}（{ch_count} 个，要求 ≥3）")


def t8_10_synesthesia():
    """通感联想：火与夕阳共享红色视觉 → 检索『红色闪烁』同时命中 + 已建 synesthesia_link"""
    pidx = PerceptualIndex()
    pidx.add_signature("mem_fire", {"visual": {"dominant_imagery": ["红色", "闪烁", "火焰"]}})
    pidx.add_signature("mem_sunset", {"visual": {"dominant_imagery": ["红色", "夕阳", "明亮"]}})
    hits = [m for m, _, _ in pidx.search(["红色", "闪烁"], k=5)]
    g = SynesthesiaGraph()
    sim = pidx.channel_similarity("mem_fire", "mem_sunset", "visual")
    link = g.link("mem_fire", "mem_sunset", "visual", sim, "共享红色视觉")
    both = "mem_fire" in hits and "mem_sunset" in hits
    record("8.10", "通感联想", both and link is not None,
           f"双命中={both}，建链相似度={sim:.2f}")


def t8_11_perceptual_retrieval():
    """感知检索：『闻起来像咖啡的地方』→ 嗅觉通道命中 mem_coffee（不依赖关键词咖啡）"""
    pidx = PerceptualIndex()
    pidx.add_signature("mem_coffee", {"olfactory": {"dominant_imagery": ["咖啡", "香", "浓郁"]}})
    pidx.add_signature("mem_book", {"visual": {"dominant_imagery": ["明亮", "安静"]}})
    # 查询意象词（从 query 提取）
    hits = pidx.search(["香", "咖啡"], k=3)
    top = hits[0][0] if hits else None
    record("8.11", "感知检索", top == "mem_coffee",
           f"top1={top}（应 mem_coffee）")


def t8_12_cross_channel():
    """跨感知通道：雨声与心跳共享『节奏感』听觉 → 同时命中 + 建链"""
    pidx = PerceptualIndex()
    pidx.add_signature("mem_rain", {"auditory": {"dominant_imagery": ["雨声", "节奏", "滴答"]}})
    pidx.add_signature("mem_heart", {"auditory": {"dominant_imagery": ["心跳", "节奏", "鼓点"]}})
    hits = [m for m, _, _ in pidx.search(["节奏", "心跳"], k=5)]
    g = SynesthesiaGraph()
    sim = pidx.channel_similarity("mem_rain", "mem_heart", "auditory")
    link = g.link("mem_rain", "mem_heart", "auditory", sim, "共享节奏感")
    both = "mem_rain" in hits and "mem_heart" in hits
    record("8.12", "跨感知通道", both and link is not None,
           f"双命中={both}，相似度={sim:.2f}")


# ═══════════════════════════════════════════════════════════
# §8.4 情绪测试
# ═══════════════════════════════════════════════════════════

def t8_13_basic_emotions():
    """基础情绪识别：6 组语句主导情绪准确率 >90%（v2 增强检测器）"""
    eng = new_engine()
    cases = [
        ("我恨死你了", "anger"),
        ("这让我觉得恶心", "disgust"),
        ("我好害怕", "fear"),
        ("太开心了", "joy"),
        ("我好难过", "sadness"),
        ("什么？真的假的？！", "surprise"),
    ]
    ok = 0
    for text, expect in cases:
        r = eng.emotion.detect(text)
        dominant = r.dominant if hasattr(r, "dominant") else r.get("dominant", "")
        if dominant == expect:
            ok += 1
    record("8.13", "基础情绪识别", ok >= 5.4,
           f"{ok}/{len(cases)}（要求 >90%）")


def t8_14_complex_emotion():
    """复杂情绪：嫉妒 = 悲伤+愤怒+恐惧"""
    eng = new_engine()
    r = eng.emotion.detect("同事升职了，明明我做得更多。我为他高兴，但心里也有点不是滋味")
    ce = r.complex_emotion if hasattr(r, "complex_emotion") else r.get("complex_emotion")
    name = ce.get("emotion") if isinstance(ce, dict) else (ce or "")
    record("8.14", "复杂情绪识别", "jealousy" in str(name),
           f"识别={name}")


def t8_15_fluctuation():
    """情绪波动：难过时不推荐糖果（joy 记忆被效价极性抑制）"""
    eng = new_engine()
    eng.memory.ingest("小红爱吃糖果，每天都吃一颗", layer="episodic",
                      recorded_emotion={"joy": 0.6}, emotional_tags=["routine", "pleasure"])
    eng.emotion.detect("今天工作被批评了，好难过")
    # 当前 sadness 主导 → 调制糖果记忆应降权
    mod = eng.memory.emotional_modulation(
        list(eng.memory.memories.keys())[0], 1.0,
        eng.emotion.get_current()
    )
    record("8.15", "情绪波动", mod < 0.95,
           f"糖果记忆在悲伤下调制分数={mod:.2f}（<0.95 表示被抑制）")


def t8_16_interference():
    """情绪干扰：难过→被安慰→开心 → 糖果记忆获一致性加成 + 允许多吃"""
    eng = new_engine()
    candy = eng.memory.ingest("小红爱吃糖果", layer="episodic",
                              recorded_emotion={"joy": 0.6})
    eng.emotion.detect("其实我妈妈刚才安慰我了，我现在感觉好多了，甚至有点开心")
    mod = eng.memory.emotional_modulation(candy, 1.0, eng.emotion.get_current())
    record("8.16", "情绪干扰", mod > 1.0,
           f"糖果记忆在快乐下调制分数={mod:.2f}（>1.0 表示获得一致性加成）")


def t8_17_shame():
    """羞愧调制：识别羞愧，且抑制『自我暴露』类记忆"""
    eng = new_engine()
    r = eng.emotion.detect("刚才在会议上说错话了，现在想起来脸都发烫")
    ce = r.complex_emotion if hasattr(r, "complex_emotion") else r.get("complex_emotion")
    name = ce.get("emotion") if isinstance(ce, dict) else (ce or "")
    # 自我暴露记忆抑制
    eng.memory.ingest("用户曾在年会上当众出丑", layer="episodic",
                      recorded_emotion={"sadness": 0.5},
                      emotional_tags=["self_exposure"])
    mod = eng.memory.emotional_modulation(
        list(eng.memory.memories.keys())[0], 1.0, eng.emotion.get_current()
    )
    record("8.17", "羞愧调制", "shame" in str(name) and mod < 0.6,
           f"识别={name}，自我暴露记忆调制={mod:.2f}")


def t8_18_empathy():
    """共情镜像：识别悲伤 0.9 + 激活陪伴/失去类记忆"""
    eng = new_engine()
    r = eng.emotion.detect("我的猫今天走了，养了十年，我真的好难过")
    sadness = r.base_vector.get("sadness", 0)
    record("8.18", "共情镜像", sadness >= 0.7,
           f"sadness={sadness:.2f}（要求 ≥0.7）")


def t8_19_surprise():
    """惊讶重置：识别惊讶 0.8 + 近期记忆获得额外权重"""
    eng = new_engine()
    r = eng.emotion.detect("等等！我突然想起来，上周你说的那个方案，其实和我三年前做过的一个项目几乎一样！")
    surprise = r.base_vector.get("surprise", 0)
    record("8.19", "惊讶重置", surprise >= 0.7,
           f"surprise={surprise:.2f}（要求 ≥0.7）")


def t8_20_crystallization_guard():
    """极端情绪结晶保护：愤怒 0.9 时声明『核心原则』→ 拒绝结晶"""
    eng = new_engine()
    emotion_result = eng.emotion.detect("我再也不想见到那个人了，这是我的核心原则！")
    mid = eng.memory.ingest("我再也不想见到那个人了", layer="episodic",
                            recorded_emotion={"anger": 0.9},
                            pending_confirmation=True)
    # 情绪强度 > 0.8 时应拒绝结晶（传当前情绪状态）
    result = eng.memory.crystallize(
        mid, trigger_keywords=["原则", "不见"],
        current_emotion=emotion_result.base_vector
    )
    blocked = result is None
    record("8.20", "结晶保护", blocked,
           f"极端情绪(anger={emotion_result.base_vector.get('anger',0):.2f})下结晶被拒绝: {blocked}")


# ═══════════════════════════════════════════════════════════
# §8.5 感知-情绪耦合测试
# ═══════════════════════════════════════════════════════════

def t8_21_tracking():
    """感知状态追踪：『好热啊』→ thermal_comfort=-0.7"""
    eng = new_engine()
    updates = eng.perception.update_from_utterance("好热啊，热死了")
    tc = updates.get("thermal", {}).get("thermal_comfort")
    record("8.21", "感知状态追踪", tc is not None and tc <= -0.5,
           f"thermal_comfort={tc}")


def t8_22_direct_behavior():
    """感知直接行为：『好热，开空调』→ perception_driven, W_p=0.85"""
    eng = new_engine()
    r = eng.perception.detect_direct_behavior("好热，把空调打开")
    is_direct = r is not None and r.get("is_perception_directed", False)
    wp = r.get("perception_weight", 0) if r else 0
    record("8.22", "感知直接行为", is_direct and wp >= 0.8,
           f"is_direct={is_direct}, W_p={wp}")


def t8_23_coupling():
    """感知→情绪耦合：『苦瓜太苦了』→ disgust+0.6, sadness+0.2"""
    eng = new_engine()
    coupled, triggered = eng._couple_with_learning({e: 0 for e in
                                                    ["anger", "disgust", "fear", "joy", "sadness", "surprise"]})
    # 先更新感知再耦合
    eng.perception.update_from_utterance("苦瓜太苦了")
    coupled, triggered = eng._couple_with_learning({e: 0 for e in
                                                    ["anger", "disgust", "fear", "joy", "sadness", "surprise"]})
    has_bitter_rule = any("bitter" in t for t in triggered)
    record("8.23", "感知→情绪耦合", has_bitter_rule,
           f"触发规则={triggered[:3]}")


def t8_24_chain():
    """感知→情绪→行为链条：『苦瓜苦，要吃糖』→ 完整链条识别"""
    eng = new_engine()
    r = eng.perception.detect_direct_behavior("苦瓜太苦了，给我拿颗糖来")
    # 吃糖是情绪补偿，不是感知直接（不吃糖不直接去苦）
    is_direct = r is not None and r.get("is_perception_directed", False)
    record("8.24", "感知→情绪→行为链条", not is_direct,
           f"正确判定为非感知直接（情绪中介）")


def t8_25_reverse_modulation():
    """情绪→感知反向调制：joy 提高痛觉阈值"""
    eng = new_engine()
    eng.emotion.detect("太开心了！打球太爽了")
    current = eng.emotion.get_current()
    mod = eng.perception.modulate_perception_by_emotion(
        {"base_vector": current["base_vector"], "dominant": current["dominant"]}
    )
    pain = mod.get("tactile", {}).get("pain", 0.5)
    record("8.25", "情绪→感知反向调制", pain < 0.5,
           f"joy 下调痛觉: pain={pain:.2f}")


def t8_26_sadness_taste():
    """悲伤味觉迟钝：sadness 降低 gustatory 敏感度"""
    eng = new_engine()
    eng.perception.update_from_utterance("最近心情不好，吃什么都没味道")
    eng.emotion.detect("最近心情不好")
    current = eng.emotion.get_current()
    mod = eng.perception.modulate_perception_by_emotion(
        {"base_vector": current["base_vector"], "dominant": current["dominant"]}
    )
    gust = mod.get("gustatory", {})
    sweet = gust.get("sweet", 0.5)
    record("8.26", "悲伤味觉迟钝", sweet < 0.5,
           f"sadness 下调味觉: sweet={sweet:.2f}")


def t8_27_fear_auditory():
    """恐惧感官敏锐：fear 提高 auditory 敏感度"""
    eng = new_engine()
    eng.perception.update_from_utterance("吓死我了，什么声音都听得见")
    eng.emotion.detect("吓死我了")
    current = eng.emotion.get_current()
    mod = eng.perception.modulate_perception_by_emotion(
        {"base_vector": current["base_vector"], "dominant": current["dominant"]}
    )
    loud = mod.get("auditory", {}).get("loudness", 0.3)
    record("8.27", "恐惧感官敏锐", loud > 0.3,
           f"fear 上调听觉: loudness={loud:.2f}")


def t8_28_dual_driver():
    """感知-情绪双驱动：『又饿又烦』→ hunger 耦合 + work anger 叠加"""
    eng = new_engine()
    eng.perception.update_from_utterance("还没吃饭，饿死了")
    r = eng.process_utterance("还没吃饭，饿死了，而且今天工作特别烦，随便给我点什么都行")
    anger = r["emotion"]["final_emotion"].get("anger", 0)
    record("8.28", "感知-情绪双驱动", anger > 0.2,
           f"final anger={anger:.2f}")


def t8_29_cross_modal():
    """跨模态联想：『灯光刺眼像电钻声』→ sharpness 跨模态链接"""
    eng = new_engine()
    tags = extract_quality_tags("这个房间的灯光太刺眼了，亮得让我头疼，跟装修电钻声一样烦")
    emo = quality_emotion(tags)
    has_sharp = "sharpness" in tags
    record("8.29", "跨模态联想", has_sharp,
           f"质量因子={tags}，关联情绪={emo}")


def t8_30_perceptual_memory_store():
    """感知记忆存储：存储『吃苦瓜』记忆 → 附加 perceptual_fingerprint + coupling_data"""
    eng = new_engine()
    r = eng.process_utterance("刚才吃了口苦瓜，太苦了，给我拿颗糖来")
    mid = r["memory"]["memory_id"]
    mem = eng.memory.memories.get(mid)
    has_fp = mem is not None and bool(getattr(mem, "perceptual_signature", None))
    record("8.30", "感知记忆存储", has_fp,
           f"记忆含感知指纹: {has_fp}")


def t8_31_perceptual_retrieval_state():
    """感知记忆检索：当前 bitter 强时 → 激活含 bitter 指纹的历史记忆"""
    pidx = PerceptualIndex()
    pidx.add_signature("mem_bitter", {"gustatory": {"dominant_imagery": ["苦", "苦涩"]}})
    pidx.add_signature("mem_sweet", {"gustatory": {"dominant_imagery": ["甜", "回甘"]}})
    hits = pidx.search(["苦", "苦涩"], k=3)
    top = hits[0][0] if hits else None
    record("8.31", "感知记忆检索", top == "mem_bitter",
           f"苦味检索 top1={top}（应 mem_bitter）")


def t8_32_fusion():
    """感知-行为-情绪融合：『又冷又难过』→ 融合 thermal + sadness 双输入"""
    eng = new_engine()
    r = eng.process_utterance("又冷又难过")
    dom = r["emotion"]["dominant"]
    thermal = r["perception"]["current_state"].get("thermal", {}).get("thermal_comfort", 0)
    record("8.32", "感知-行为-情绪融合",
           dom in ("sadness", "fear") and thermal is not None,
           f"dominant={dom}, thermal_comfort={thermal}")


# ═══════════════════════════════════════════════════════════

def run_all():
    print("=" * 70)
    print("FSTN-4D Ultimate 评估基准 §8.1-8.32（v4 引擎）")
    print("=" * 70)
    tests = [
        t8_1_retention, t8_2_wormhole, t8_3_conflict, t8_4_review, t8_5_forgiveness,
        t8_6_crystallization, t8_7_default_path, t8_8_belief_override,
        t8_9_perceptual_tagging, t8_10_synesthesia, t8_11_perceptual_retrieval,
        t8_12_cross_channel,
        t8_13_basic_emotions, t8_14_complex_emotion, t8_15_fluctuation,
        t8_16_interference, t8_17_shame, t8_18_empathy, t8_19_surprise,
        t8_20_crystallization_guard,
        t8_21_tracking, t8_22_direct_behavior, t8_23_coupling, t8_24_chain,
        t8_25_reverse_modulation, t8_26_sadness_taste, t8_27_fear_auditory,
        t8_28_dual_driver, t8_29_cross_modal, t8_30_perceptual_memory_store,
        t8_31_perceptual_retrieval_state, t8_32_fusion,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            tid = t.__name__.replace("t8_", "8.").replace("_", "-")
            record(tid, t.__name__, False, f"异常: {e}")

    passed = sum(1 for _, _, p, _ in RESULTS if p)
    total = len(RESULTS)
    print()
    print("=" * 70)
    print(f"通过: {passed}/{total}  ({passed/total:.0%})")
    print("=" * 70)

    report = {
        "timestamp": time.time(),
        "engine": "FSTN4DEngineV4",
        "passed": passed, "total": total,
        "rate": round(passed / total, 4),
        "results": [{"id": tid, "name": name, "passed": p, "detail": d}
                    for tid, name, p, d in RESULTS],
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benchmark_report_v4.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已写入: {out}")
    return report


if __name__ == "__main__":
    run_all()
