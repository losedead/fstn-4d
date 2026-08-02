"""
FSTN-4D 消融评测主脚本 (Ablation Study Runner)
===============================================
对比 V1（符号化）与 V3（神经-符号混合）在同一测试集上的表现，
输出量化报告 bench/report.md。

评测维度：
  1. 情绪检测准确率（19 条带标准答案）
  2. 感知-情绪耦合触发正确率（7 条）
  3. 记忆检索 top-1 命中率（5 查询 × 5 记忆）
  4. 耦合学习能力（热→怒 权重下降演示）
  5. 检测路径统计（V3 双层检测器的 keyword/llm/heuristic 分布）

运行：python bench/run_ablation.py
"""

import sys
import os
import time
import tempfile
from typing import List, Dict, Tuple

# 目录路径
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V1_DIR = os.path.join(_BASE, "engine")
_V3_DIR = os.path.join(_BASE, "v3")
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
for d in (_V1_DIR, _V3_DIR, _BENCH_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from test_cases import (EMOTION_CASES, COUPLING_CASES, MEMORY_SEED,
                        MEMORY_QUERIES, LEARN_SCENARIO)

# 模块级维度名映射（评测报告渲染用）
DIM_NAMES = {
    "emotion": "情绪检测", "coupling": "感知耦合", "memory": "记忆检索",
}

# V1 引擎
from fstn_core import FSTN4DEngine as EngineV1
# V3 引擎
from hybrid_core import FSTN4DEngineV3 as EngineV3
from adaptive_coupling import AdaptiveCouplingMatrix


# ═══════════════════════════════════════════════════════════════
# 情绪检测评测
# ═══════════════════════════════════════════════════════════════

def run_emotion_eval(engine, engine_name: str) -> Tuple[float, List[Dict]]:
    """对每条情绪用例独立评测（重置状态，避免跨轮污染）"""
    correct = 0
    details = []
    for text, expect_dom, expect_complex, note in EMOTION_CASES:
        # 独立引擎实例（每轮从零开始，隔离干扰）
        fresh = type(engine)(state_dir=tempfile.mkdtemp()) if engine_name != "V3学习" else engine
        try:
            if engine_name.startswith("V1"):
                r = fresh.process_utterance(text, "")
                got_dom = r["emotion"]["dominant"]
                got_complex = (r["emotion"].get("complex_emotion") or {}).get("emotion")
            else:
                r = fresh.process_utterance(text, "")
                got_dom = r["emotion"]["dominant"]
                got_complex = (r["emotion"].get("complex") or {}).get("emotion")
        except Exception as e:
            got_dom, got_complex = f"ERROR:{e}", None

        dom_ok = got_dom == expect_dom
        complex_ok = True
        if expect_complex:
            complex_ok = got_complex == expect_complex
            # 混合情绪用例：dominant 允许是 joy 或 sadness（都是合理主情绪）
            if text.startswith("同事升职") and got_dom in ("joy", "sadness"):
                dom_ok = True
        elif expect_complex is None and expect_dom == "neutral":
            pass  # 中性不需要复杂情绪

        ok = dom_ok and complex_ok
        if ok:
            correct += 1
        details.append({
            "text": text, "expect": expect_dom,
            "expect_complex": expect_complex,
            "got_dom": got_dom, "got_complex": got_complex,
            "ok": ok, "note": note,
        })
    return correct / len(EMOTION_CASES), details


# ═══════════════════════════════════════════════════════════════
# 感知-情绪耦合评测
# ═══════════════════════════════════════════════════════════════

def run_coupling_eval(engine, engine_name: str) -> Tuple[float, List[Dict]]:
    correct = 0
    details = []
    for text, expect_sense, expect_emo in COUPLING_CASES:
        fresh = type(engine)(state_dir=tempfile.mkdtemp()) if engine_name != "V3学习" else engine
        r = fresh.process_utterance(text, "")
        got_sense = r["perception"]["dominant_sense"]
        # 兼容 V1/V3 字段差异
        if "coupled_delta" in r["emotion"]:
            got_coupled = r["emotion"]["coupled_delta"]      # V3
        else:
            got_coupled = r["coupling"]["coupled_emotion"]   # V1
        emo_hit = any(v > 0.05 for k, v in got_coupled.items() if k == expect_emo)
        sense_ok = got_sense == expect_sense
        ok = sense_ok and emo_hit
        if ok:
            correct += 1
        details.append({
            "text": text, "expect_sense": expect_sense, "expect_emo": expect_emo,
            "got_sense": got_sense, "got_emo_hit": emo_hit, "ok": ok,
        })
    return correct / len(COUPLING_CASES), details


# ═══════════════════════════════════════════════════════════════
# 记忆检索评测（V1 关键词 vs V3 向量）
# ═══════════════════════════════════════════════════════════════

def run_memory_eval(engine, engine_name: str) -> Tuple[float, List[Dict]]:
    fresh = type(engine)(state_dir=tempfile.mkdtemp())
    # 注入种子记忆
    for content, importance in MEMORY_SEED:
        fresh.memory.ingest(content, importance=importance,
                            recorded_emotion={})
    correct = 0
    details = []
    for query, expect_sub in MEMORY_QUERIES:
        hits = fresh.retrieve_memories(query, k=3, emotion_aware=False)
        top1 = hits[0].content if hits else ""
        ok = expect_sub in top1
        if ok:
            correct += 1
        details.append({
            "query": query, "expect_sub": expect_sub,
            "top1": top1[:30] if top1 else "(无结果)", "ok": ok,
            "top1_score": getattr(hits[0], "score", 0) if hits else 0,
        })
    return correct / len(MEMORY_QUERIES), details


# ═══════════════════════════════════════════════════════════════
# 耦合学习能力评测（仅 V3）
# ═══════════════════════════════════════════════════════════════

def run_learning_eval() -> Dict:
    m = AdaptiveCouplingMatrix()
    s, st = LEARN_SCENARIO["sense"], LEARN_SCENARIO["state"]
    w_anger_0 = m.get_weight(s, st, "anger")
    w_joy_0 = m.get_weight(s, st, "joy")

    for _ in range(LEARN_SCENARIO["rounds"]):
        m.learn([(s, st)], LEARN_SCENARIO["feedback_target"])

    w_anger_1 = m.get_weight(s, st, "anger")
    w_joy_1 = m.get_weight(s, st, "joy")
    return {
        "rule": f"{s}:{st}",
        "anger_before": round(w_anger_0, 3),
        "anger_after": round(w_anger_1, 3),
        "anger_delta": round(w_anger_1 - w_anger_0, 3),
        "joy_before": round(w_joy_0, 3),
        "joy_after": round(w_joy_1, 3),
        "joy_delta": round(w_joy_1 - w_joy_0, 3),
        "anger_decreased": w_anger_1 < w_anger_0,
        "joy_emerged": w_joy_1 > w_joy_0 + 0.05,
    }


# ═══════════════════════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════════════════════

def render_report(results: Dict) -> str:
    L = []
    L.append("# FSTN-4D 消融评测报告（V1 vs V3）\n")
    L.append(f"> 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    L.append("> 方法: 带标准答案的测试集，机器判定，无 LLM 裁判偏差。\n")

    L.append("\n## 1. 总分对比\n")
    L.append("| 维度 | V1 符号化 | V3 神经-符号混合 | 差距 |")
    L.append("|------|----------|----------------|------|")
    for dim in ["emotion", "coupling", "memory"]:
        a = results[f"v1_{dim}"]["acc"]
        b = results[f"v3_{dim}"]["acc"]
        delta = b - a
        arrow = "✅ 提升" if delta > 0 else ("⚠️ 持平" if delta == 0 else "❌ 下降")
        L.append(f"| {DIM_NAMES[dim]} | {a:.0%} | {b:.0%} | "
                 f"{delta:+.0%} {arrow} |")
    L.append("")

    L.append("\n## 2. 情绪检测明细\n")
    L.append("| 用例 | 期望 | V1 结果 | V3 结果 | V1 | V3 |")
    L.append("|------|------|---------|---------|----|----|")
    for d1, d3 in zip(results["v1_emotion"]["details"],
                      results["v3_emotion"]["details"]):
        v1c = d1.get("got_complex") or "-"
        v3c = d3.get("got_complex") or "-"
        v1s = "✅" if d1["ok"] else "❌"
        v3s = "✅" if d3["ok"] else "❌"
        L.append(f"| {d1['note']} | {d1['expect']}"
                 f"{('+' + d1['expect_complex']) if d1['expect_complex'] else ''} "
                 f"| {d1['got_dom']}"
                 f"{('+' + v1c) if v1c != '-' else ''} "
                 f"| {d3['got_dom']}"
                 f"{('+' + v3c) if v3c != '-' else ''} "
                 f"| {v1s} | {v3s} |")
    L.append("")

    L.append("\n## 3. 感知-情绪耦合明细\n")
    L.append("| 用例 | 期望感知 | 期望情绪 | V1 | V3 |")
    L.append("|------|---------|---------|----|----|")
    for d1, d3 in zip(results["v1_coupling"]["details"],
                      results["v3_coupling"]["details"]):
        L.append(f"| {d1['text'][:18]}... | {d1['expect_sense']} | "
                 f"{d1['expect_emo']} | {'✅' if d1['ok'] else '❌'} | "
                 f"{'✅' if d3['ok'] else '❌'} |")
    L.append("")

    L.append("\n## 4. 记忆检索明细\n")
    L.append("| 查询 | 期望命中 | V1 top-1 | V3 top-1 | V1 | V3 |")
    L.append("|------|---------|----------|----------|----|----|")
    for d1, d3 in zip(results["v1_memory"]["details"],
                      results["v3_memory"]["details"]):
        L.append(f"| {d1['query'][:14]}... | {d1['expect_sub']} | "
                 f"{d1['top1']} | {d3['top1']} | "
                 f"{'✅' if d1['ok'] else '❌'} | {'✅' if d3['ok'] else '❌'} |")
    L.append("")

    L.append("\n## 5. 耦合学习能力（V3 独有）\n")
    lr = results["learning"]
    L.append(f"规则 `{lr['rule']}→anger`: 专家初始 **{lr['anger_before']}** "
             f"→ 学习后 **{lr['anger_after']}** "
             f"({lr['anger_delta']:+.3f}，"
             f"{'✅ 成功削弱' if lr['anger_decreased'] else '❌ 未削弱'})\n")
    L.append(f"规则 `{lr['rule']}→joy`: 专家初始 **{lr['joy_before']}** "
             f"→ 学习后 **{lr['joy_after']}** "
             f"({lr['joy_delta']:+.3f}，"
             f"{'✅ 从无到有' if lr['joy_emerged'] else '⚠️ 未学出'})\n")
    L.append("> 意义：同一用户反复在「热」后表达快乐（如空调房），"
             "V3 会把这个个体模式学进耦合矩阵，"
             "而 V1 的静态矩阵永远按「热→怒」的专家经验响应。\n")

    L.append("\n## 6. V3 检测路径统计\n")
    ds = results["v3_detector_stats"]
    total = sum(ds.values())
    L.append(f"| 路径 | 次数 | 占比 |")
    L.append(f"|------|------|------|")
    for k, v in ds.items():
        pct = v / total * 100 if total else 0
        L.append(f"| {k} | {v} | {pct:.0f}% |")
    L.append("")
    L.append("> 说明: 情绪评测每轮新建引擎实例，"
             "故检测器统计仅来自 coupling/memory 评测的实际运行。")

    L.append("\n## 结论\n")
    v1_total = (results["v1_emotion"]["acc"] + results["v1_coupling"]["acc"]
                + results["v1_memory"]["acc"]) / 3
    v3_total = (results["v3_emotion"]["acc"] + results["v3_coupling"]["acc"]
                + results["v3_memory"]["acc"]) / 3
    L.append(f"综合正确率: V1 = **{v1_total:.1%}** → V3 = **{v3_total:.1%}** "
             f"({v3_total - v1_total:+.1%})\n")
    L.append(f"V3 额外获得 V1 不具备的能力: 耦合在线学习、矛盾情感识别、"
             f"可解释的向量检索。\n")
    return "\n".join(L)


def main():
    print("=" * 64)
    print("FSTN-4D 消融评测（V1 符号化 vs V3 神经-符号混合）")
    print("=" * 64)

    results = {}
    dim_names = DIM_NAMES  # 模块级维度名映射

    for ver, EngineCls in (("v1", EngineV1), ("v3", EngineV3)):
        engine = EngineCls(state_dir=tempfile.mkdtemp())
        name = "V1 引擎" if ver == "v1" else "V3 引擎"
        print(f"\n▶ 运行 {name} ...")
        if ver == "v3":
            # 评测场景：关闭 LLM 节流，逐条真调用语义路径
            engine.detector.llm.min_interval = 0.0

        acc, details = run_emotion_eval(engine, name)
        results[f"{ver}_emotion"] = {"acc": acc, "details": details}
        print(f"  情绪检测: {acc:.0%}")

        acc, details = run_coupling_eval(engine, name)
        results[f"{ver}_coupling"] = {"acc": acc, "details": details}
        print(f"  感知耦合: {acc:.0%}")

        acc, details = run_memory_eval(engine, name)
        results[f"{ver}_memory"] = {"acc": acc, "details": details}
        print(f"  记忆检索: {acc:.0%}")

    results["v3_detector_stats"] = {}
    print("\n▶ 运行耦合学习能力演示 (V3) ...")
    results["learning"] = run_learning_eval()
    lr = results["learning"]
    print(f"  热→怒: {lr['anger_before']} → {lr['anger_after']} "
          f"({lr['anger_delta']:+.3f})")
    print(f"  热→乐: {lr['joy_before']} → {lr['joy_after']} "
          f"({lr['joy_delta']:+.3f})")

    # V3 检测器路径分布采样：用全部情绪用例真实跑一遍检测器本体
    from neural_detector import DualLayerEmotionDetector
    det_sampler = DualLayerEmotionDetector()
    det_sampler.llm.min_interval = 0.0
    for text, _, _, _ in EMOTION_CASES:
        det_sampler.detect(text, "")
    results["v3_detector_stats"] = det_sampler.get_stats()

    report = render_report(results)
    report_path = os.path.join(_BENCH_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ 评测完成，报告已写入: {report_path}")
    print(report[:1200])


if __name__ == "__main__":
    main()
