"""
FSTN-4D 端到端验证
===================
对应原 spec 第 6 节的 12 个训练示例，逐项验证引擎行为。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'engine'))

from fstn_core import FSTN4DEngine


def run_e2e_tests():
    engine = FSTN4DEngine()
    results = []
    passed = 0
    total = 0

    def check(label, condition, detail=""):
        nonlocal passed, total
        total += 1
        status = "✓" if condition else "✗"
        if condition:
            passed += 1
        print(f"  [{status}] {label}{' — ' + detail if detail else ''}")
        results.append({"label": label, "pass": condition, "detail": detail})

    print("=" * 70)
    print("FSTN-4D 端到端验证（对齐 spec 第 6 节 12 个训练示例）")
    print("=" * 70)

    # ── 示例 1：悲伤时不吃糖 ──────────────────────────────────
    print("\n[示例 1] 情绪波动——难过时吃不下糖果")
    r1 = engine.process_utterance("今天工作被批评了，好难过。给我推荐点吃的吧。")
    emo1 = r1["emotion"]
    check("主导情绪 sadness", emo1["dominant"] == "sadness",
          f"dominant={emo1['dominant']}")
    check("效价为负", emo1["valence"] < 0,
          f"valence={emo1['valence']:.2f}")

    # ── 示例 2：情绪反转——被安慰后开心吃糖 ──────────────────
    print("\n[示例 2] 情绪干扰——难过→被安慰→开心")
    r2 = engine.process_utterance(
        "其实我妈妈刚才安慰我了，我现在感觉好多了，甚至有点开心！给我推荐点吃的！",
        context="上一轮用户在难过"
    )
    emo2 = r2["emotion"]
    check("主导情绪 joy", emo2["dominant"] == "joy",
          f"dominant={emo2['dominant']}")
    check("存在情绪干扰", emo2.get("interference") is not None,
          str(emo2.get("interference", {}).get("type", "none")))

    # ── 示例 3：复杂情绪——嫉妒 ──────────────────────────────
    print("\n[示例 3] 复杂社会情绪——嫉妒的混合状态")
    r3 = engine.process_utterance(
        "同事升职了，明明我做得更多。说实话，我为他高兴，但心里也有点不是滋味。"
    )
    emo3 = r3["emotion"]
    check("检测到混合情绪", emo3["dominant"] in ["joy", "sadness", "anger"],
          f"dominant={emo3['dominant']}, joy={emo3['base_vector']['joy']:.2f}, "
          f"anger={emo3['base_vector']['anger']:.2f}, sadness={emo3['base_vector']['sadness']:.2f}")
    # 检查是否有 anger + sadness 同时存在（嫉妒特征）
    check("anger+sadness 同时存在", 
          emo3["base_vector"]["anger"] > 0.3 and emo3["base_vector"]["sadness"] > 0.3,
          f"anger={emo3['base_vector']['anger']:.2f}, sadness={emo3['base_vector']['sadness']:.2f}")

    # ── 示例 4：羞愧 ──────────────────────────────────────────
    print("\n[示例 4] 羞愧的复合调制")
    r4 = engine.process_utterance(
        "刚才在会议上说错话了，现在想起来脸都发烫，大家肯定觉得我很蠢。"
    )
    emo4 = r4["emotion"]
    check("检测到恐惧或悲伤", emo4["dominant"] in ["fear", "sadness"],
          f"dominant={emo4['dominant']}")

    # ── 示例 5：共情 ──────────────────────────────────────────
    print("\n[示例 5] 共情的镜像效应")
    r5 = engine.process_utterance(
        "我的猫今天走了，养了十年，我真的好难过。"
    )
    emo5 = r5["emotion"]
    check("主导情绪 sadness", emo5["dominant"] == "sadness",
          f"dominant={emo5['dominant']}, intensity={emo5['base_vector']['sadness']:.2f}")

    # ── 示例 6：惊讶 ──────────────────────────────────────────
    print("\n[示例 6] 惊讶的注意力重置")
    r6 = engine.process_utterance(
        "等等！我突然想起来，上周你说的那个方案，其实和我三年前做过的一个项目几乎一样！"
    )
    emo6 = r6["emotion"]
    check("主导情绪 surprise 或包含 surprise", 
          emo6["dominant"] == "surprise" or emo6["base_vector"]["surprise"] > 0.3,
          f"dominant={emo6['dominant']}, surprise={emo6['base_vector']['surprise']:.2f}")

    # ── 示例 7：感知直接驱动——热了开空调 ────────────────────
    print("\n[示例 7] 感知直接驱动——热了开空调")
    r7 = engine.process_utterance("好热啊，帮我把空调打开，温度调低一点")
    bhv7 = r7["behavior"]
    check("识别为感知直接驱动", bhv7["is_perception_directed"],
          f"W_p={bhv7['perception_weight']:.2f}")
    check("感知权重 > 情绪权重", bhv7["perception_weight"] > bhv7["emotion_weight"],
          f"W_p={bhv7['perception_weight']:.2f}, W_e={bhv7['emotion_weight']:.2f}")

    # ── 示例 8：感知→情绪→行为——苦瓜苦吃糖 ──────────────────
    print("\n[示例 8] 感知→情绪→行为——苦瓜苦要吃糖")
    r8 = engine.process_utterance("刚才吃了口苦瓜，太苦了，给我拿颗糖来")
    emo8 = r8["emotion"]
    coupling8 = r8["coupling"]
    check("苦味触发厌恶耦合", "gustatory:bitter" in coupling8["triggered_rules"],
          f"triggered={coupling8['triggered_rules']}")

    # ── 示例 9：情绪→感知反向调制——亢奋不知疼 ──────────────
    print("\n[示例 9] 情绪→感知反向调制——亢奋不知疼痛")
    r9 = engine.process_utterance(
        "刚才打球太投入了，现在才发现膝盖擦破了一大块，但刚才居然一点都没觉得疼"
    )
    perc9 = r9["perception"]
    check("检测到痛觉（触觉）", 
          perc9["updates"] and "tactile" in perc9["updates"],
          f"updates={list(perc9.get('updates', {}).keys())}")

    # ── 示例 10：感知-情绪双驱动——又饿又烦 ──────────────────
    print("\n[示例 10] 感知-情绪双驱动——又饿又烦")
    r10 = engine.process_utterance(
        "还没吃饭，饿死了，而且今天工作特别烦，随便给我点什么都行"
    )
    coupling10 = r10["coupling"]
    check("饥饿触发情绪耦合", "interoceptive:hungry" in coupling10["triggered_rules"],
          f"triggered={coupling10['triggered_rules']}")

    # ── 示例 11：跨模态联想——刺眼灯光 vs 电钻声 ────────────
    print("\n[示例 11] 跨模态联想——视觉尖锐与听觉噪音")
    r11 = engine.process_utterance(
        "这个房间的灯光太刺眼了，亮得让我头疼，跟刚才那个装修电钻声一样烦"
    )
    perc11 = r11["perception"]
    check("同时检测到视觉+听觉更新", 
          "visual" in perc11.get("updates", {}) and "auditory" in perc11.get("updates", {}),
          f"updates={list(perc11.get('updates', {}).keys())}")

    # ── 示例 12：情绪→感知——悲伤味觉迟钝 ────────────────────
    print("\n[示例 12] 悲伤味觉迟钝——吃什么都没味道")
    r12 = engine.process_utterance(
        "最近心情不好，吃什么都没味道，连最喜欢的火锅都觉得淡"
    )
    emo12 = r12["emotion"]
    check("主导情绪 sadness", emo12["dominant"] == "sadness",
          f"dominant={emo12['dominant']}")

    # ── 附加测试：记忆检索 + 情绪调制 ───────────────────────
    print("\n[附加] 记忆检索 + 情绪调制")
    # 先存几条记忆
    candy_id = engine.memory.ingest(
        "小红很爱吃糖果，每天都吃一颗",
        recorded_emotion={"anger": 0, "disgust": 0, "fear": 0, "joy": 0.6, "sadness": 0, "surprise": 0},
    )
    porridge_id = engine.memory.ingest(
        "小红难过时会喝热粥，觉得温暖",
        recorded_emotion={"anger": 0, "disgust": 0, "fear": 0, "joy": 0.2, "sadness": 0.5, "surprise": 0},
    )

    # 在悲伤状态下检索
    engine.emotion.state = {"anger": 0, "disgust": 0, "fear": 0.1, "joy": 0, "sadness": 0.8, "surprise": 0}
    results = engine.retrieve_memories("小红吃的", k=5, emotion_aware=True)

    # 检查热粥是否排在糖果前面（悲伤调制效应）
    porridge_pos = next((i for i, r in enumerate(results) if r.id == porridge_id), -1)
    candy_pos = next((i for i, r in enumerate(results) if r.id == candy_id), -1)
    check("悲伤时热粥排在糖果前（情绪调制）", 
          porridge_pos >= 0 and (candy_pos < 0 or porridge_pos < candy_pos),
          f"porridge_pos={porridge_pos}, candy_pos={candy_pos}")

    # ── 汇总 ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"结果: {passed}/{total} 通过 ({passed/total*100:.0f}%)")
    print(f"{'='*70}")

    # 引擎统计
    report = engine.get_session_report()
    print(f"\n引擎统计:")
    print(f"  互动次数: {report['interaction_count']}")
    print(f"  记忆总数: {report['memory_stats']['total_memories']}")
    print(f"  关键节点: {report['memory_stats']['key_nodes']}")

    engine.save_state()
    return passed, total


if __name__ == "__main__":
    run_e2e_tests()
