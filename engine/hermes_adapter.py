"""
FSTN-4D Hermes 集成适配器
===========================
桥接 FSTN-4D 引擎与 Hermes Agent 的工具集（memory、session_search、skills）。

用法：
    from hermes_adapter import HermesFSTNAdapter
    adapter = HermesFSTNAdapter()
    guidance = adapter.analyze(user_message)
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'engine'))

from fstn_core import FSTN4DEngine


class HermesFSTNAdapter:
    """
    将 FSTN-4D 引擎适配为 Hermes Agent 可调用的分析接口。
    Agent 在每轮对话时调用 adapter.analyze(utterance) 获取情绪指导。
    """

    def __init__(self, engine_dir: str = None):
        if engine_dir is None:
            engine_dir = os.path.expanduser("~/.fstn_engine")
        self.engine = FSTN4DEngine(state_dir=engine_dir)
        self.engine.load_state()
        self.conversation_context = ""

    def analyze(self, utterance: str, context: str = "") -> dict:
        """
        主分析入口。返回 Agent 可直接使用的指导信息。
        """
        # 完整推理链
        result = self.engine.process_utterance(utterance, context or self.conversation_context)
        self.conversation_context = utterance

        # 生成回复指导
        guidance = self.engine.generate_reply_guidance()

        # 提取关键信息
        emo = result["emotion"]
        perc = result["perception"]
        bhv = result["behavior"]

        return {
            "guidance_text": guidance,
            "emotion": {
                "dominant": emo["dominant"],
                "valence": emo["valence"],
                "arousal": emo["arousal"],
                "complex": emo.get("complex_emotion"),
                "action_bias": self._action_bias(emo),
                "should_avoid": self._avoidance(emo),
            },
            "perception": {
                "dominant_sense": perc.get("dominant_sense"),
                "is_direct_behavior": bhv["is_perception_directed"],
                "perception_weight": bhv["perception_weight"],
                "synesthesia_active": len(perc.get("synesthesia", [])) > 0,
            },
            "reply_strategy": self._reply_strategy(emo, bhv),
            "memory_result": result["memory"],
        }

    def retrieve_with_emotion(self, query: str, k: int = 5) -> list:
        """情绪感知记忆检索"""
        results = self.engine.retrieve_memories(query, k=k, emotion_aware=True)
        return [
            {"id": r.id, "content": r.content, "window": r.window}
            for r in results
        ]

    def save(self):
        self.engine.save_state()

    # ── 辅助 ──────────────────────────────────────────────────

    @staticmethod
    def _action_bias(emo: dict) -> str:
        dom = emo["dominant"]
        if dom == "neutral":
            return "neutral"
        if dom == "joy":
            return "explore"
        if dom == "anger":
            return "confront"
        if dom == "fear":
            return "avoid"
        if dom == "sadness":
            return "seek_support"
        if dom == "disgust":
            return "avoid"
        if dom == "surprise":
            return "reset_attention"
        return "neutral"

    @staticmethod
    def _avoidance(emo: dict) -> list:
        avoid = []
        if emo["dominant"] == "sadness" or emo.get("complex_emotion", {}).get("emotion") == "shame":
            avoid.append("dismissing_feelings")
            avoid.append("overly_cheerful")
        if emo["dominant"] == "anger":
            avoid.append("compromise")
            avoid.append("deflection")
        if emo["dominant"] == "fear":
            avoid.append("risky_suggestions")
        return avoid

    @staticmethod
    def _reply_strategy(emo: dict, bhv: dict) -> dict:
        """生成回复策略"""
        strategy = {"tone": "neutral", "pace": "normal", "focus": "content"}

        if bhv["is_perception_directed"]:
            strategy["focus"] = "perception_first"
            strategy["tone"] = "practical"

        dom = emo["dominant"]
        if dom == "sadness":
            strategy["tone"] = "warm_but_not_sugary"
            strategy["pace"] = "slow"
            strategy["focus"] = "validate_then_support"
        elif dom == "anger":
            strategy["tone"] = "firm_understanding"
            strategy["pace"] = "direct"
            strategy["focus"] = "acknowledge_then_analyze"
        elif dom == "fear":
            strategy["tone"] = "calm_reassuring"
            strategy["pace"] = "steady"
            strategy["focus"] = "safety_first"
        elif dom == "joy":
            strategy["tone"] = "playful"
            strategy["pace"] = "lively"
            strategy["focus"] = "explore_expand"
        elif dom == "disgust":
            strategy["tone"] = "respectful"
            strategy["focus"] = "acknowledge_boundary"
        elif dom == "surprise":
            strategy["tone"] = "curious"
            strategy["focus"] = "pivot_attention"
        elif dom == "neutral" and bhv["is_perception_directed"]:
            strategy["focus"] = "solve_perceptual_need"

        complex_e = emo.get("complex_emotion", {})
        if complex_e:
            if complex_e.get("emotion") == "shame":
                strategy["tone"] = "normalizing"
                strategy["focus"] = "acceptance"
            elif complex_e.get("emotion") == "empathy":
                strategy["tone"] = "mirroring"
                strategy["focus"] = "emotional_resonance"
            elif complex_e.get("emotion") == "jealousy":
                strategy["tone"] = "balanced"
                strategy["focus"] = "acknowledge_both_sides"

        return strategy


# ═══════════════════════════════════════════════════════════════
# 作为脚本运行时 → Hermes Agent 可通过 execute_code 调用
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FSTN-4D Hermes Adapter")
    parser.add_argument("--analyze", type=str, help="Analyze utterance and return JSON guidance")
    parser.add_argument("--retrieve", type=str, help="Retrieve memories with emotional modulation")
    parser.add_argument("--save", action="store_true", help="Save engine state")
    args = parser.parse_args()

    adapter = HermesFSTNAdapter()

    if args.analyze:
        result = adapter.analyze(args.analyze)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.retrieve:
        results = adapter.retrieve_with_emotion(args.retrieve)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.save:
        adapter.save()
        print('{"status": "saved"}')
