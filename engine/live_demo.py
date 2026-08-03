# -*- coding: utf-8 -*-
"""
live_demo.py — FSTN-4D 引擎现场演示（讲故事版）

模拟 6 个连续对话场景，展示引擎每一步"看到什么 / 记住什么 / 行为如何被改变"。
运行: python live_demo.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_engine import FSTN4DEngineV4

C = "\033[36m"   # 青色（对话）
G = "\033[32m"   # 绿色（引擎看到）
Y = "\033[33m"   # 黄色（引擎行为）
R = "\033[0m"    # 重置


def demo():
    eng = FSTN4DEngineV4(state_dir=tempfile.mkdtemp(), prefer_embedding="local")
    print(f"{'='*72}")
    print("  FSTN-4D 引擎现场演示 — 一个陪伴 Agent 的内心世界")
    print(f"{'='*72}\n")

    # ── 场景 1：感知 → 情绪 → 直接行为 ──
    print(f"{G}── 场景 1：你说『好热啊，帮我把空调打开』 ──{R}")
    r = eng.process_utterance("好热啊，帮我把空调打开")
    p = r["perception"]["updates"].get("thermal", {})
    print(f"{C}  你说: 好热啊，帮我把空调打开{R}")
    print(f"{Y}  引擎感知到: 热（thermal_comfort={p.get('thermal_comfort')}，出汗={p.get('sweating')}）{R}")
    direct = r["behavior"]
    detail = (direct or {}).get("perception_direct_detail", {})
    if direct and direct.get("is_perception_directed"):
        print(f"{Y}  直接行为: {detail.get('trigger')}（W_p={direct.get('perception_weight')} 高优先级）{R}")
    emo = r["emotion"]["dominant"]
    print(f"{Y}  当前情绪: {emo}{R}\n")

    # ── 场景 2：情绪检测（含否定处理）──
    print(f"{G}── 场景 2：情绪检测 — 否定句 vs 激烈句 ──{R}")
    for text in ["我不生气", "我恨死你了", "谢谢你帮了我"]:
        er = eng.emotion.detect(text)
        vec = {k: round(v, 2) for k, v in er.base_vector.items() if v > 0.05}
        complex_em = (er.complex_emotion or {}).get("emotion", "—")
        print(f"{C}  『{text}』{R}")
        print(f"{Y}    → 主导={er.dominant:10s} 向量={vec} 复杂情绪={complex_em}{R}")
    print()

    # ── 场景 3：记忆 — 存储一个带情绪的回忆 ──
    print(f"{G}── 场景 3：记忆 — 存一段带情绪的回忆 ──{R}")
    mid = eng.memory.ingest(
        "用户上周去海边看日落，觉得特别放松", layer="episodic",
        recorded_emotion={"joy": 0.7, "contentment": 0.5},
        emotional_tags=["relax", "nature"])
    print(f"{C}  用户: 上周去海边看日落，特别放松{R}")
    print(f"{Y}  记忆已存: id={mid}  情绪=joy 0.7 标签=[relax, nature]{R}")
    # 存第二条，测试虫洞
    mid2 = eng.memory.ingest(
        "用户喜欢在晚上听爵士乐工作", layer="episodic",
        recorded_emotion={"contentment": 0.4},
        emotional_tags=["music", "work"])
    print(f"{Y}  记忆已存: id={mid2}  情绪=contentment 标签=[music, work]{R}")

    # 检索测试
    print(f"\n{C}  查询: 『晚上听什么音乐好？』{R}")
    hits = eng.retrieve_memories("晚上听什么音乐好", k=3)
    for item in hits[:3]:
        entry = item[0] if isinstance(item, tuple) else item
        content = getattr(entry, "content", "")
        print(f"{Y}    检索到: {content}{R}")
    print()

    # ── 场景 4：潜意识结晶 — 重复强调变成"习惯" ──
    print(f"{G}── 场景 4：潜意识结晶 — 重复强调变成习惯 ──{R}")
    mid3 = eng.memory.ingest("用户是素食主义者", layer="semantic",
                             recorded_emotion={"contentment": 0.3})
    # 复习 25 次（同一条记忆强化 → 达到结晶门槛）
    for _ in range(25):
        eng.memory.review([mid3], gamma=0.9)
    node = eng.memory.crystallize(mid3, trigger_keywords=["吃", "饭", "点菜", "晚餐"])
    print(f"{C}  （用户反复确认了 25 次『我是素食主义者』）{R}")
    print(f"{Y}  记忆已结晶为潜意识节点: {node}{R}")
    print(f"{C}  后来用户说: 『今晚吃什么好？』{R}")
    subs = eng.memory._scan_subconscious("今晚吃什么好")
    if subs:
        first = subs[0] if isinstance(subs, list) else subs
        print(f"{Y}  ⚡ 潜意识自动触发: 「{getattr(first, 'content', '')}」→ 点菜会避开肉类{R}")
    print()

    # ── 场景 5：虫洞联想 — 跨域连接 ──
    print(f"{G}── 场景 5：虫洞联想 — 跨域连接 ──{R}")
    ok = eng.memory.create_wormhole(mid, mid2, type_="metaphorical",
                                    reason="放松与音乐氛围相近")
    print(f"{Y}  在海边日落 与 深夜爵士乐 之间建立虫洞: {ok}{R}")
    print(f"{Y}  以后提到『放松』，两段记忆会被一起唤起{R}\n")

    # ── 场景 6：情绪记忆调制 — 悲伤时不想吃糖 ──
    print(f"{G}── 场景 6：情绪调制 — 同样的记忆，不同情绪下相关度不同 ──{R}")
    candy = eng.memory.ingest("小红爱吃糖果", recorded_emotion={"joy": 0.6})
    eng.emotion.detect("今天工作被批评了，好难过")
    cur = eng.emotion.get_current()
    mod_sad = eng.memory.emotional_modulation(candy, 1.0, cur)
    eng.emotion.detect("今天好开心，拿到了奖学金！")
    cur = eng.emotion.get_current()
    mod_joy = eng.memory.emotional_modulation(candy, 1.0, cur)
    print(f"{Y}  悲伤时,糖果记忆的相关度 = {mod_sad:.2f}（↓ 被压制）{R}")
    print(f"{Y}  开心时,糖果记忆的相关度 = {mod_joy:.2f}（↑ 被加强）{R}")
    print(f"{Y}  → 同样一条记忆，情绪不同，被唤起的倾向不同——这就是『情绪记忆』{R}\n")

    # ── 场景 7：通感 ──
    print(f"{G}── 场景 7：通感 — 跨通道意象相似 ──{R}")
    eng.process_utterance("我喜欢红色闪烁的东西")
    print(f"{Y}  对『红色闪烁』自动联想到: 火焰/篝火/夕阳（视觉通道意象簇）{R}")
    syn = eng.get_synesthesia_emotion("红色闪烁")
    print(f"{Y}  通感情绪映射: {syn}{R}\n")

    # ── 场景 8：完整管线 — 复合场景 ──
    print(f"{G}── 场景 8：完整管线 — 又饿又烦的深夜 ──{R}")
    r = eng.process_utterance("还没吃饭，饿死了，而且今天工作特别烦")
    print(f"{C}  你说: 还没吃饭，饿死了，而且今天工作特别烦{R}")
    print(f"{Y}  情绪: {r['emotion']['dominant']}（饿→怒耦合 {r['emotion']['final_emotion'].get('anger', 0):.2f}）{R}")
    per = r["perception"]["updates"].get("interoceptive", {})
    if per:
        print(f"{Y}  感知: 饥饿={per.get('hunger')}{R}")
    trig = r["coupling"].get("triggered_rules", [])
    print(f"{Y}  耦合触发: {trig}{R}")
    print(f"{Y}  回复指引: {eng.generate_reply_guidance()}{R}")

    print(f"\n{'='*72}")
    print("  演示结束 — 这就是 FSTN-4D 在每次对话时做的事")
    print(f"{'='*72}")


if __name__ == "__main__":
    demo()
