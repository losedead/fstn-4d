"""
FSTN-4D Hermes 自动加载器
===========================
将此脚本路径加入 Skill 或 cronjob，实现每轮对话自动情绪分析。
也可以作为独立的 execute_code 调用。
"""

import sys
import os

ENGINE_PATH = os.path.expanduser(r'~\Desktop\fstn_enhancement\engine')
sys.path.insert(0, ENGINE_PATH)


def auto_init():
    """初始化或获取全局引擎实例"""
    import builtins
    if not hasattr(builtins, '_fstn_engine'):
        from fstn_core import FSTN4DEngine
        builtins._fstn_engine = FSTN4DEngine()
        builtins._fstn_engine.load_state()
    return builtins._fstn_engine


def analyze(utterance: str, context: str = "") -> str:
    """
    分析用户话语，返回指导文本。
    将此文本沉默地融入 Agent 思考过程。
    """
    engine = auto_init()
    result = engine.process_utterance(utterance, context)
    guidance = engine.generate_reply_guidance()

    # 额外输出情绪速查
    emo = result["emotion"]
    bhv = result["behavior"]

    lines = [guidance]

    # 情绪状态速查
    emo_line = f"[情绪] {emo['dominant']} V={emo['valence']:+.2f} A={emo['arousal']:.2f}"
    if emo.get("complex_emotion"):
        emo_line += f" 复杂:{emo['complex_emotion']['emotion']}"
    lines.append(emo_line)

    # 行为驱动
    if bhv["is_perception_directed"]:
        lines.append(f"[行为] 感知直接驱动 W_p={bhv['perception_weight']:.2f}")
    else:
        lines.append(f"[行为] 情绪驱动")

    # 回复策略
    strat = result.get("reply_strategy", {}) if "reply_strategy" in result else _get_strategy(emo, bhv)

    return "\n".join(lines)


def _get_strategy(emo: dict, bhv: dict) -> dict:
    """回复策略推断"""
    dom = emo["dominant"]
    strategy = {"tone": "neutral", "focus": "content"}

    if bhv.get("is_perception_directed"):
        strategy["focus"] = "solve_physical_need"

    if dom == "sadness":
        strategy = {"tone": "warm_not_sugary", "focus": "validate_then_support"}
    elif dom == "anger":
        strategy = {"tone": "firm_understanding", "focus": "acknowledge_then_analyze"}
    elif dom == "fear":
        strategy = {"tone": "calm_reassuring", "focus": "safety_first"}
    elif dom == "joy":
        strategy = {"tone": "playful", "focus": "explore_expand"}
    elif dom == "surprise":
        strategy = {"tone": "curious", "focus": "pivot_attention"}
    elif dom == "disgust":
        strategy = {"tone": "respectful", "focus": "acknowledge_boundary"}

    complex_e = emo.get("complex_emotion", {}) or {}
    if complex_e.get("emotion") == "shame":
        strategy = {"tone": "normalizing", "focus": "acceptance_no_judgment"}
    elif complex_e.get("emotion") == "jealousy":
        strategy = {"tone": "balanced", "focus": "acknowledge_both_sides"}

    return strategy


def save_and_summarize():
    """保存引擎状态并返回摘要"""
    engine = auto_init()
    engine.save_state()
    from memory_bridge import MemoryBridge
    bridge = MemoryBridge(engine)
    snapshot = bridge.get_sync_snapshot()
    return snapshot


def restore_from_memory_entries(memory_entries: list):
    """从 Hermes memory 条目恢复关键节点"""
    engine = auto_init()
    from memory_bridge import MemoryBridge
    bridge = MemoryBridge(engine)
    count = bridge.restore_from_payloads(memory_entries)
    return count


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FSTN-4D Auto-Loader")
    parser.add_argument("--analyze", type=str, help="Analyze utterance")
    parser.add_argument("--context", type=str, default="", help="Previous context")
    parser.add_argument("--save", action="store_true", help="Save state and print summary")
    args = parser.parse_args()

    if args.analyze:
        guidance = analyze(args.analyze, args.context)
        print(guidance)

    if args.save:
        summary = save_and_summarize()
        import json
        print(json.dumps(summary, ensure_ascii=False, indent=2))
