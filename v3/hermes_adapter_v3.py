"""
FSTN-4D V3 Hermes 集成适配器
=============================
接口对齐 V1 hermes_adapter.HermesFSTNAdapter，
底层换成 V3 神经-符号混合引擎。

用法（与 V1 相同）：
    from hermes_adapter_v3 import HermesFSTNAdapterV3
    adapter = HermesFSTNAdapterV3()
    guidance = adapter.analyze(user_message)
"""

import sys
import os
import json

_V3_DIR = os.path.dirname(os.path.abspath(__file__))
if _V3_DIR not in sys.path:
    sys.path.insert(0, _V3_DIR)
from hybrid_core import FSTN4DEngineV3


class HermesFSTNAdapterV3:
    """FSTN-4D V3 Hermes 适配器（接口与 V1 一致）"""

    def __init__(self, engine_dir: str = None, use_llm: bool = True):
        if engine_dir is None:
            engine_dir = os.path.expanduser("~/.fstn_engine_v3")
        self.engine = FSTN4DEngineV3(state_dir=engine_dir, use_llm=use_llm)
        self.engine.load_state()
        self.conversation_context = ""

    def analyze(self, utterance: str, context: str = "") -> dict:
        """主分析入口。返回 Agent 可直接使用的指导信息。"""
        # 先处理上一轮排队的延迟反馈（感知→情绪配对学习）
        self.engine.drain_coupling_feedback()

        result = self.engine.process_utterance(utterance,
                                               context or self.conversation_context)
        self.conversation_context = utterance
        guidance = self.engine.generate_reply_guidance()

        emo = result["emotion"]
        perc = result["perception"]
        bhv = result["behavior"]

        return {
            "guidance_text": guidance,
            "emotion": {
                "dominant": emo["dominant"],
                "valence": emo["valence"],
                "arousal": emo["arousal"],
                "complex": emo.get("complex"),
                "detector_source": emo.get("detector_source"),
                "detector_confidence": emo.get("detector_confidence"),
                "action_bias": self._action_bias(emo),
                "should_avoid": self._avoidance(emo),
            },
            "perception": {
                "dominant_sense": perc.get("dominant_sense"),
                "is_direct_behavior": bhv["is_perception_directed"],
                "perception_weight": bhv["perception_weight"],
                "synesthesia_active": len(perc.get("synesthesia", [])) > 0,
                "active_coupling": perc.get("active_coupling", []),
            },
            "reply_strategy": self._reply_strategy(emo, bhv),
            "memory_result": result["memory"],
            "learning": result.get("learning", {}),
        }

    def retrieve_with_emotion(self, query: str, k: int = 5) -> list:
        """情绪感知向量检索"""
        results = self.engine.retrieve_memories(query, k=k, emotion_aware=True)
        return [
            {"id": r.id, "content": r.content, "window": r.window,
             "score": r.score, "top_terms": r.top_terms}
            for r in results
        ]

    def save(self):
        self.engine.save_state()

    # ── 辅助（对齐 V1） ───────────────────────────────────────

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
        dom = emo["dominant"]
        complex_e = emo.get("complex") or {}
        if dom == "sadness" or complex_e.get("emotion") in ("shame",):
            avoid += ["dismissing_feelings", "overly_cheerful"]
        if dom == "anger":
            avoid += ["compromise", "deflection"]
        if dom == "fear":
            avoid += ["risky_suggestions"]
        if complex_e.get("emotion") == "ambivalent":
            avoid += ["oversimplifying", "forcing_a_choice"]
        return avoid

    @staticmethod
    def _reply_strategy(emo: dict, bhv: dict) -> dict:
        strategy = {"tone": "neutral", "pace": "normal", "focus": "content"}

        if bhv["is_perception_directed"]:
            strategy["focus"] = "perception_first"
            strategy["tone"] = "practical"

        dom = emo["dominant"]
        if dom == "sadness":
            strategy.update({"tone": "warm_but_not_sugary", "pace": "slow",
                             "focus": "validate_then_support"})
        elif dom == "anger":
            strategy.update({"tone": "firm_understanding", "pace": "direct",
                             "focus": "acknowledge_then_analyze"})
        elif dom == "fear":
            strategy.update({"tone": "calm_reassuring", "pace": "steady",
                             "focus": "safety_first"})
        elif dom == "joy":
            strategy.update({"tone": "playful", "pace": "lively",
                             "focus": "explore_expand"})
        elif dom == "disgust":
            strategy.update({"tone": "respectful", "focus": "acknowledge_boundary"})
        elif dom == "surprise":
            strategy.update({"tone": "curious", "focus": "pivot_attention"})
        elif dom == "neutral" and bhv["is_perception_directed"]:
            strategy["focus"] = "solve_perceptual_need"

        complex_e = emo.get("complex") or {}
        if complex_e:
            name = complex_e.get("emotion")
            if name == "shame":
                strategy.update({"tone": "normalizing", "focus": "acceptance"})
            elif name == "empathy":
                strategy.update({"tone": "mirroring", "focus": "emotional_resonance"})
            elif name == "jealousy":
                strategy.update({"tone": "balanced",
                                 "focus": "acknowledge_both_sides"})
            elif name == "ambivalent":
                strategy.update({"tone": "gentle_validating",
                                 "focus": "hold_both_feelings"})

        return strategy


# ═══════════════════════════════════════════════════════════════
# 作为脚本运行（与 V1 相同接口）
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FSTN-4D V3 Hermes Adapter")
    parser.add_argument("--analyze", type=str, help="Analyze utterance")
    parser.add_argument("--retrieve", type=str, help="Retrieve memories")
    parser.add_argument("--save", action="store_true", help="Save state")
    args = parser.parse_args()

    adapter = HermesFSTNAdapterV3()

    if args.analyze:
        print(json.dumps(adapter.analyze(args.analyze),
                         ensure_ascii=False, indent=2))
    if args.retrieve:
        print(json.dumps(adapter.retrieve_with_emotion(args.retrieve),
                         ensure_ascii=False, indent=2))
    if args.save:
        adapter.save()
        print('{"status": "saved"}')
