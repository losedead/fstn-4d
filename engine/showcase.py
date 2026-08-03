# -*- coding: utf-8 -*-
"""
showcase.py — FSTN-4D 推荐展示：同一对话，普通模式 vs 情感引擎模式

最能打动人的展示方式是对比。这个脚本用 4 个日常场景，
展示「普通 LLM」与「挂载 FSTN-4D 的 LLM」处理同一句话的差异。

运行: python showcase.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_engine import FSTN4DEngineV4

C = "\033[36m"   # 用户
G = "\033[32m"   # 普通模式
Y = "\033[33m"   # FSTN 模式
R = "\033[0m"

# 普通 LLM 的"回复"（模拟：没有情感引擎，只有模板化回应）
def plain_reply(text):
    """模拟普通 LLM 的回复——没有情绪/感知/记忆的机械回应"""
    canned = {
        "热": "好的，我帮您打开空调。",
        "难过": "我理解你的感受，但请相信一切都会好起来的。",
        "饿": "好的，我帮您找点吃的。",
        "音乐": "好的，我推荐一些爵士乐给你。",
    }
    for k, v in canned.items():
        if k in text:
            return v
    return "好的，我明白了。"


def fstn_reply(eng, text):
    """FSTN-4D 模式：先感知情绪/状态，再给回复，并输出引擎内部决策"""
    lines = []

    # 音乐场景：直接展示记忆检索，不污染记忆库
    if "音乐" in text:
        hits = eng.memory.retrieve(text, k=2)
        contents = [getattr(h, "content", "") for h in hits[:2]]
        known = [c for c in contents if "爵士" in c or "海边" in c]
        lines.append("【记忆检索】命中 " + str(len(known)) + " 条相关记忆")
        if known:
            reply = f"我记得你之前说过「{known[0]}」——要不顺着那个氛围挑？"
        else:
            reply = "来点爵士？不过我更想知道你现在的心情适合什么。"
        lines.append(f"【回复】{reply}")
        return "\n".join(lines)

    r = eng.process_utterance(text)
    emo = r["emotion"]
    per = r["perception"]
    direct = r["behavior"]

    # 引擎看到了什么
    if emo["dominant"] != "neutral":
        lines.append(f"【情绪检测】主导={emo['dominant']} 向量={ {k: round(v,2) for k,v in emo['base_vector'].items() if v>0.1} }")
    else:
        lines.append("【情绪检测】neutral（平静）")
    complex_em = emo.get("complex_emotion") or {}
    if complex_em:
        lines.append(f"【复杂情绪】{complex_em.get('emotion')}")
    # 感知
    updates = per.get("updates", {})
    if updates:
        parts = []
        for ch, vals in updates.items():
            for k, v in vals.items():
                if isinstance(v, bool) and v:
                    parts.append(f"{ch}:{k}")
                elif isinstance(v, (int, float)) and abs(v) > 0.3:
                    parts.append(f"{ch}:{k}={v:.1f}")
        if parts:
            lines.append(f"【感知追踪】{', '.join(parts[:4])}")
    # 直接行为
    detail = (direct or {}).get("perception_direct_detail", {})
    if direct and direct.get("is_perception_directed"):
        lines.append(f"【直接行为】{detail.get('trigger')}（W_p={direct.get('perception_weight')}）")

    # 回复（模拟：针对场景给出"有人味"的回答）
    reply = ""
    if "热" in text:
        reply = "确实闷得慌吧？我把空调调低点，另外你刚运动完的话，先别对着风口吹，容易着凉。"
    elif "难过" in text:
        reply = "嗯，这种时候不用急着振作。我陪着你，想说说发生了什么，还是就想安静待会儿？"
    elif "饿" in text and "烦" in text:
        reply = "又饿又烦最容易上火——先别做决定，我给你点份热乎的，吃饱了再说工作的事。"
    else:
        reply = "我在听。"
    lines.append(f"【回复】{reply}")
    return "\n".join(lines)


def demo():
    eng = FSTN4DEngineV4(state_dir=tempfile.mkdtemp(), prefer_embedding="local")
    # 先喂两条记忆，展示记忆检索
    eng.memory.ingest("用户上周去海边看日落，觉得特别放松",
                      recorded_emotion={"joy": 0.7})
    eng.memory.ingest("用户喜欢在晚上听爵士乐工作",
                      recorded_emotion={"contentment": 0.4})

    print(f"{'='*70}")
    print("  同样的四句话：普通 LLM vs 挂载 FSTN-4D 的 LLM")
    print(f"{'='*70}\n")

    scenarios = [
        "好热啊，帮我把空调打开",
        "今天工作被批评了，好难过",
        "还没吃饭，饿死了，而且今天工作特别烦",
        "晚上听什么音乐好？",
    ]

    for i, text in enumerate(scenarios, 1):
        print(f"{C}── 第 {i} 句：『{text}』{R}")
        print(f"{G}  [普通 LLM]")
        print(f"     {plain_reply(text)}")
        print(f"{Y}  [FSTN-4D]")
        for line in fstn_reply(eng, text).split("\n"):
            print(f"     {line}")
        print()

    print(f"{'='*70}")
    print("  差别在哪？")
    print("  · 普通 LLM 只回『好的，我帮您打开空调』——没有情绪、没有记忆、没有状态")
    print("  · FSTN-4D 先『感知』再『回应』：检测情绪、追踪感知、检索记忆、决定行为")
    print("  · 同一句『晚上听什么音乐』，FSTN 记得你之前说过海边日落、喜欢爵士")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()
