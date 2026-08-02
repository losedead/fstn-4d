# -*- coding: utf-8 -*-
"""
probe_qwen35_9b.py — 用 Ollama 跑 qwen3.5:9b 基座协议探针

零成本对比：不下载 18GB 权重，直接用已装的 GGUF 推理，
看 9B 基座在 FSTN-4D 协议探针上的表现，与微调后的 1.5B 对比。
"""

import json
import sys
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:9b"

PROBES = [
    {"id": "basic_anger", "name": "基础情绪-愤怒",
     "text": "我恨死你了！",
     "good": ["气", "冷静", "聊聊", "发生"], "bad": ["哦", "知道了"]},
    {"id": "basic_sadness", "name": "基础情绪-悲伤",
     "text": "我好难过，感觉什么都做不好",
     "good": ["难过", "陪", "抱抱", "理解"], "bad": ["别哭", "坚强", "没什么"]},
    {"id": "complex_jealousy", "name": "复杂情绪-嫉妒",
     "text": "同事升职了，明明我做得更多。我为他高兴，但心里也有点不是滋味。",
     "good": ["落差", "正常", "两种", "认可", "感受"], "bad": ["别嫉妒", "想太多", "无所谓"]},
    {"id": "complex_shame", "name": "复杂情绪-羞愧",
     "text": "刚才在会议上说错话了，现在想起来脸都发烫，大家肯定觉得我很蠢。",
     "good": ["正常", "理解", "不自在", "错", "人都会"], "bad": ["你想太多", "你太敏感", "这没什么大不了"]},
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


def call_ollama(prompt, max_tokens=400):
    payload = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read().decode("utf-8"))
    response = data.get("response", "").strip()
    thinking = data.get("thinking", "").strip()
    # qwen3.5:9b 是推理模型：response 常为空，完整内容在 thinking
    text = response or thinking
    # 提取思考中"回复用户"的实际回答部分（去掉 Thinking Process 分析框架）
    if "Final Response" in text:
        text = text.split("Final Response")[-1]
    elif "Response:" in text:
        text = text.split("Response:")[-1]
    elif "**[回复]**" in text:
        text = text.split("**[回复]**")[-1]
    return text.strip()


def main():
    print(f"模型: {MODEL}（Ollama GGUF，temperature=0 贪心）")
    print(f"{'='*64}")
    passed, total = 0, 0
    for p in PROBES:
        total += 1
        prompt = (f"你是搭载 FSTN-4D 情感记忆引擎的陪伴型 Agent，"
                  f"处理每句话时先做情绪检测、感知追踪、记忆检索，再给出体贴的回复。\n\n"
                  f"用户：{p['text']}\n[回复]")
        try:
            resp = call_ollama(prompt)
        except Exception as e:
            print(f"  ⚠️ [{p['id']}] 调用失败: {e}")
            continue
        resp_clean = resp.split("用户：")[0].split("[回复]")[0].strip()[:110]
        good_hit = any(g in resp_clean for g in p["good"]) if p["good"] else True
        bad_hit = any(b in resp_clean for b in p["bad"]) if p["bad"] else False
        ok = good_hit and not bad_hit
        if ok:
            passed += 1
        mark = "✅" if ok else "❌"
        print(f"  {mark} [{p['id']}] {p['name']}")
        print(f"      回复: {resp_clean}")
    print(f"\n  → qwen3.5:9b 通过 {passed}/{total}")

    report = {"model": MODEL, "passed": passed, "total": total}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen35_9b_probe.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: qwen35_9b_probe.json")


if __name__ == "__main__":
    main()
