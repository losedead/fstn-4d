"""
FSTN-4D 记忆架构集成演示
==========================
演示完整的"协议内化→引擎持久化→自动加载"链路。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'engine'))

from fstn_core import FSTN4DEngine
from memory_bridge import MemoryBridge
from auto_loader import analyze, save_and_summarize, auto_init


def demo():
    print("=" * 70)
    print("FSTN-4D 记忆架构集成演示")
    print("=" * 70)

    # ── Phase 1: 模拟多轮对话，建立记忆 ──
    print("\n[Phase 1] 模拟多轮对话，建立记忆和关键节点")

    engine = auto_init()

    conversations = [
        ("小红很爱吃糖果，每天都吃一颗", ""),
        ("用户是素食主义者，不吃任何肉类", "之前小红爱吃糖果"),
        ("用户有猫，养了三年叫咪咪", ""),
        ("用户讨厌电话沟通，偏好文字或邮件", ""),
    ]

    for utterance, ctx in conversations:
        engine.process_utterance(utterance, ctx)

    # 模拟多次复习 -> 结晶
    print("  模拟素食者记忆复习 25 次...")
    results = engine.memory.retrieve("素食", k=1)
    if results:
        veg_id = results[0].id
        for _ in range(25):
            engine.memory.review([veg_id], gamma=0.9)
        node_id = engine.crystallize_if_ready(veg_id, ["吃饭", "餐厅", "菜单", "推荐", "食谱"])
        if node_id:
            print(f"  ✓ 素食者已结晶为关键节点: {node_id}")

    # 模拟猫的记忆结晶
    results = engine.memory.retrieve("猫", k=1)
    if results:
        cat_id = results[0].id
        for _ in range(25):
            engine.memory.review([cat_id], gamma=0.9)
        node_id = engine.crystallize_if_ready(cat_id, ["出差", "旅行", "宠物", "寄养"])
        if node_id:
            print(f"  ✓ 有猫已结晶为关键节点: {node_id}")

    # ── Phase 2: 记忆桥接 → Hermes memory ──
    print("\n[Phase 2] 记忆桥接 → 生成 Hermes memory 条目")

    bridge = MemoryBridge(engine)
    snapshot = bridge.get_sync_snapshot()
    print(f"  关键节点数: {len(snapshot.get('key_nodes', []))}")
    print(f"  记忆总数: {snapshot['stats']['total_memories']}")
    print(f"  引擎状态概要: {snapshot['stats']}")

    payloads = bridge.build_memory_payloads()
    print(f"\n  生成 {len(payloads)} 条 Hermes memory 条目:")
    for i, p in enumerate(payloads):
        print(f"  [{i+1}] {p[:80]}...")

    # ── Phase 3: 模拟「新会话」── 从 payload 恢复 ──
    print("\n[Phase 3] 模拟新会话：从 Hermes memory 恢复")

    # 创建新引擎实例（模拟新会话）
    fresh_engine = FSTN4DEngine()
    fresh_bridge = MemoryBridge(fresh_engine)

    # 从 payload 恢复关键节点
    restored_count = fresh_bridge.restore_from_payloads(payloads)
    print(f"  恢复了 {restored_count} 个关键节点")

    # 验证潜意识扫描
    activated = fresh_engine.memory._scan_subconscious("今晚吃什么好呢，推荐个餐厅")
    print(f"  查询'今晚吃什么' → 激活 {len(activated)} 个关键节点:")
    for node in activated:
        print(f"    ↳ {node.content} (触发关键词: {node.auto_trigger_keywords[:3]})")

    activated2 = fresh_engine.memory._scan_subconscious("下周出差三天")
    print(f"  查询'下周出差' → 激活 {len(activated2)} 个关键节点:")
    for node in activated2:
        print(f"    ↳ {node.content} (触发关键词: {node.auto_trigger_keywords[:3]})")

    # ── Phase 4: 实时情绪分析 ──
    print("\n[Phase 4] 实时情绪分析（模拟实际对话）")

    test_utterances = [
        "今天工作被批评了，好难过。推荐点吃的吧。",
        "好热啊，帮我把空调打开",
        "同事升职了，明明我做得更多。说实话我为他高兴，但心里不是滋味",
    ]

    for utterance in test_utterances:
        guidance = analyze(utterance)
        print(f"\n  输入: {utterance[:40]}...")
        for line in guidance.split("\n"):
            if line.strip():
                print(f"    {line.strip()}")

    # ── Phase 5: 保存并总结 ──
    print(f"\n[Phase 5] 保存引擎状态")
    summary = save_and_summarize()
    print(f"  状态: {summary['status']}")
    print(f"  关键节点: {len(summary.get('key_nodes', []))}")
    print(f"  统计: {json.dumps(summary.get('stats', {}), ensure_ascii=False)}")

    print(f"\n{'='*70}")
    print("演示完成！FSTN-4D 已融入 Hermes 记忆架构。")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()
