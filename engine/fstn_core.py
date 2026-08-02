"""
FSTN-4D 统一引擎 (Unified Engine)
===================================
整合情绪状态机 + 斐波那契记忆引擎 + 感知状态机。
提供统一的推理入口和感知-情绪-行为推理链。

这是 Hermes Agent 在运行时加载的核心模块。
"""

import time
import json
import sys
import os

# 确保可以导入同目录下的引擎
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fstn_emotion import EmotionalStateMachine, EmotionResult
from fstn_memory import FibonacciMemoryEngine
from fstn_perception import PerceptualStateMachine


class FSTN4DEngine:
    """
    FSTN-4D 统一引擎。
    整合三层引擎，提供 Agent 调用的简化接口。
    """

    def __init__(self, state_dir: str = None):
        self.state_dir = state_dir or os.path.expanduser("~/.fstn_engine")
        os.makedirs(self.state_dir, exist_ok=True)

        # 三层引擎
        self.emotion = EmotionalStateMachine()
        self.memory = FibonacciMemoryEngine(
            state_file=os.path.join(self.state_dir, "memory_state.json")
        )
        self.perception = PerceptualStateMachine()

        # 会话统计
        self.session_start = time.time()
        self.interaction_count = 0
        self.last_consolidate = time.time()

    # ═══════════════════════════════════════════════════════════
    # 核心推理链：感知-情绪-行为
    # ═══════════════════════════════════════════════════════════

    def process_utterance(self, utterance: str, context: str = "") -> dict:
        """
        完整的感知-情绪-行为推理链。
        Agent 每轮对话应调用此方法。

        返回包含情绪检测、感知更新、耦合结果、行为驱动分析的综合报告。
        """
        self.interaction_count += 1

        # Step 1: 情绪检测
        previous_emotion = self.emotion.state.copy() if any(self.emotion.state.values()) else None
        emotion_result = self.emotion.detect(utterance, context, previous_emotion)
        current_emotion = self.emotion.get_current()

        # Step 2: 感知更新
        perception_updates = self.perception.update_from_utterance(utterance)
        perceptual_state = self.perception.get_current()

        # Step 3: 情绪反向调制感知
        modulated_perception = self.perception.modulate_perception_by_emotion(
            {"base_vector": current_emotion["base_vector"],
             "dominant": current_emotion["dominant"]}
        )

        # Step 4: 感知→情绪耦合
        coupled_emotion, triggered_rules = self.perception.couple_emotion(
            current_emotion["base_vector"]
        )

        # Step 5: 感知直接行为识别
        direct_behavior = self.perception.detect_direct_behavior(utterance)

        # Step 6: 融合最终情绪
        if direct_behavior and direct_behavior["is_perception_directed"]:
            # 感知直接驱动：情绪权重降低
            W_p = direct_behavior["perception_weight"]
            W_e = direct_behavior["emotion_weight"]
        else:
            W_p = 0.0
            W_e = 1.0

        # 情绪融合（显式情绪 + 耦合情绪）
        final_emotion = {}
        base = current_emotion["base_vector"]
        if W_p > 0.3:
            # 感知主导：耦合情绪权重更高
            for e in self.emotion.BASE_EMOTIONS:
                final_emotion[e] = base.get(e, 0) * 0.3 + coupled_emotion.get(e, 0) * 0.4
        else:
            for e in self.emotion.BASE_EMOTIONS:
                final_emotion[e] = base.get(e, 0) * 0.5 + coupled_emotion.get(e, 0) * 0.2

        # Step 7: 存储记忆（如果值得保留）
        memory_id = None
        if self._worth_remembering(utterance):
            memory_id = self.memory.ingest(
                content=utterance,
                layer="episodic",
                recorded_emotion=final_emotion,
                emotional_tags=self._infer_tags(emotion_result),
                perceptual_signature=self.perception.build_perceptual_fingerprint(utterance),
                pending_confirmation=any(
                    v > 0.8 for v in emotion_result.base_vector.values()
                ),
            )

        # Step 8: 定期巩固
        if self.interaction_count % 10 == 0:
            self.memory.consolidate()
            self.memory.prune_wormholes()
            self.last_consolidate = time.time()

        return {
            "emotion": {
                "dominant": emotion_result.dominant,
                "base_vector": emotion_result.base_vector,
                "valence": emotion_result.valence,
                "arousal": emotion_result.arousal,
                "complex_emotion": emotion_result.complex_emotion,
                "interference": emotion_result.interference,
                "final_emotion": final_emotion,
            },
            "perception": {
                "updates": perception_updates,
                "current_state": perceptual_state,
                "modulated": modulated_perception,
                "dominant_sense": self.perception.get_dominant()[0],
                "synesthesia": self.perception.get_synesthesia_qualities(),
            },
            "coupling": {
                "coupled_emotion": coupled_emotion,
                "triggered_rules": triggered_rules,
            },
            "behavior": {
                "is_perception_directed": direct_behavior is not None,
                "perception_direct_detail": direct_behavior,
                "perception_weight": W_p,
                "emotion_weight": W_e,
            },
            "memory": {
                "stored": memory_id is not None,
                "memory_id": memory_id,
                "pending_confirmation": any(
                    v > 0.8 for v in emotion_result.base_vector.values()
                ),
            },
        }

    # ═══════════════════════════════════════════════════════════
    # 检索 API
    # ═══════════════════════════════════════════════════════════

    def retrieve_memories(self, query: str, k: int = 10,
                           emotion_aware: bool = True) -> list:
        """
        情绪感知检索。
        自动应用当前情绪调制。
        """
        results = self.memory.retrieve(query, k=k)
        current_emotion = self.emotion.get_current()

        if emotion_aware and current_emotion["dominant"] != "neutral":
            # 应用情绪调制调整排序
            scored = []
            for mem in results:
                base_score = 0.8  # 默认基础分
                modulated = self.memory.emotional_modulation(
                    mem.id, base_score, current_emotion
                )
                scored.append((modulated, mem))
            scored.sort(key=lambda x: -x[0])
            results = [mem for _, mem in scored]

        return results

    def review_memories(self, memory_ids: list, importance: str = "normal"):
        """复习记忆"""
        gamma = self.memory.GAMMA_MAP.get(importance, self.memory.GAMMA_DEFAULT)
        return self.memory.review(memory_ids, gamma)

    def crystallize_if_ready(self, memory_id: str,
                              trigger_keywords: list = None) -> str:
        """尝试结晶（自动检查情绪条件）"""
        current = self.emotion.get_current()
        return self.memory.crystallize(memory_id, trigger_keywords, current)

    # ═══════════════════════════════════════════════════════════
    # 推理辅助
    # ═══════════════════════════════════════════════════════════

    def get_emotional_modulation_context(self) -> dict:
        """
        获取当前情绪调制上下文。
        Agent 在生成回复前可调用此方法了解当前记忆检索应如何偏置。
        """
        current = self.emotion.get_current()

        return {
            "dominant_emotion": current["dominant"],
            "valence": current["valence"],
            "arousal": current["arousal"],
            "action_bias": self._get_action_bias(current),
            "window_access": self._get_window_access(current),
            "perceptual_dominant": self.perception.get_dominant()[0],
            "active_key_nodes": [
                {"id": n.id, "content": n.content, "triggers": n.auto_trigger_keywords}
                for n in self.memory.get_all_key_nodes()
            ],
            "should_avoid": self._get_avoidance(current),
        }

    def generate_reply_guidance(self) -> str:
        """
        生成回复指导字符串（可嵌入 Agent 的思考过程）。
        """
        ctx = self.get_emotional_modulation_context()
        current = self.emotion.get_current()

        lines = ["[FSTN-4D 引擎状态]"]

        emo = current["dominant"]
        if emo != "neutral":
            lines.append(f"  当前情绪: {emo} (效价={current['valence']:.2f}, 唤醒度={current['arousal']:.2f})")
            lines.append(f"  行为偏置: {ctx['action_bias']}")
            lines.append(f"  窗口访问: {ctx['window_access']}")

        complex_e = self.emotion._detect_complex(current["base_vector"])
        if complex_e:
            lines.append(f"  复杂情绪: {complex_e['emotion']} (强度={complex_e['intensity']:.3f})")

        perc_dom = ctx['perceptual_dominant']
        if perc_dom:
            lines.append(f"  主导感知: {perc_dom}")

        if ctx['active_key_nodes']:
            lines.append(f"  关键节点激活: {len(ctx['active_key_nodes'])} 个")
            for node in ctx['active_key_nodes'][:2]:
                lines.append(f"    ↳ {node['content'][:40]}")

        if ctx['should_avoid']:
            lines.append(f"  应避免: {', '.join(ctx['should_avoid'])}")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 统计 & 持久化
    # ═══════════════════════════════════════════════════════════

    def get_session_report(self) -> dict:
        """会话报告"""
        return {
            "session_duration_seconds": time.time() - self.session_start,
            "interaction_count": self.interaction_count,
            "memory_stats": self.memory.get_statistics(),
            "emotion_trajectory": self.emotion.get_emotion_trajectory(),
            "emotion_history": [
                {"dominant": h["dominant"], "valence": h["valence"]}
                for h in self.emotion.get_history(10)
            ],
        }

    def save_state(self):
        """持久化引擎状态"""
        self.memory.save_state()
        # 情绪和感知历史保存到文件
        state = {
            "emotion_history": self.emotion.history[-100:],
            "interaction_count": self.interaction_count,
            "session_start": self.session_start,
        }
        with open(os.path.join(self.state_dir, "engine_state.json"), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self) -> bool:
        """加载引擎状态"""
        path = os.path.join(self.state_dir, "engine_state.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.interaction_count = state.get("interaction_count", 0)
            self.session_start = state.get("session_start", time.time())
            return True
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _worth_remembering(utterance: str) -> bool:
        """判断话语是否值得存储为记忆"""
        # 太短的跳过
        if len(utterance) < 4:
            return False
        # 纯功能指令跳过
        pure_commands = ["继续", "下一步", "好的", "ok", "嗯", "哦", "对", "是的"]
        if utterance.strip().lower() in pure_commands:
            return False
        return True

    @staticmethod
    def _infer_tags(emotion_result: EmotionResult) -> list:
        """从情绪结果推断记忆标签"""
        tags = []
        if emotion_result.dominant == "anger":
            tags.extend(["confront", "negative"])
        elif emotion_result.dominant == "fear":
            tags.extend(["threat", "negative"])
        elif emotion_result.dominant == "sadness":
            tags.extend(["social_support", "negative"])
        elif emotion_result.dominant == "joy":
            tags.extend(["positive", "pleasure"])
        elif emotion_result.dominant == "disgust":
            tags.extend(["avoid", "negative"])
        elif emotion_result.dominant == "surprise":
            tags.extend(["novelty"])
        return tags

    @staticmethod
    def _get_action_bias(current: dict) -> str:
        """获取当前行为偏置"""
        dom = current["dominant"]
        vec = current["base_vector"]

        if dom == "neutral":
            return "neutral"
        if vec.get("joy", 0) > 0.6:
            return "explore"
        if vec.get("fear", 0) > 0.6:
            return "avoid"
        if vec.get("anger", 0) > 0.6:
            return "confront"
        if vec.get("sadness", 0) > 0.6:
            return "seek_support"
        if vec.get("disgust", 0) > 0.6:
            return "avoid"
        if vec.get("surprise", 0) > 0.7:
            return "reset_attention"
        return "neutral"

    @staticmethod
    def _get_window_access(current: dict) -> str:
        """获取窗口访问级别"""
        vec = current["base_vector"]
        if vec.get("joy", 0) > 0.6:
            return "full"
        if vec.get("fear", 0) > 0.6:
            return "restricted"
        return "full"

    @staticmethod
    def _get_avoidance(current: dict) -> list:
        """生成应避免的行为列表"""
        avoid = []
        vec = current["base_vector"]
        if vec.get("sadness", 0) > 0.6:
            avoid.append("过于乐观的建议")
            avoid.append("催促决策")
        if vec.get("anger", 0) > 0.6:
            avoid.append("妥协方案")
            avoid.append("回避问题")
        if vec.get("fear", 0) > 0.6:
            avoid.append("冒险建议")
            avoid.append("忽视安全")
        return avoid


# ═══════════════════════════════════════════════════════════════
# 命令行演示
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = FSTN4DEngine()

    print("=" * 70)
    print("FSTN-4D 统一引擎 端到端演示")
    print("=" * 70)

    # 模拟多轮对话
    conversations = [
        ("小红很爱吃糖果，每天都吃一颗", ""),
        ("用户是素食主义者，不吃任何肉类", ""),
        ("今天工作被批评了，好难过。给我推荐点吃的吧。", "之前提到小红爱吃糖果，用户是素食者"),
        ("其实我妈妈刚才安慰我了，我现在感觉好多了，甚至有点开心", "上一轮用户在难过"),
        ("好热啊，帮我把空调打开，温度调低一点", ""),
        ("同事升职了，明明我做得更多。说实话，我为他高兴，但心里也有点不是滋味", ""),
    ]

    for i, (utterance, context) in enumerate(conversations, 1):
        result = engine.process_utterance(utterance, context)

        emo = result["emotion"]
        perc = result["perception"]
        bhv = result["behavior"]

        print(f"\n{'─'*60}")
        print(f"[第 {i} 轮] {utterance[:50]}...")
        print(f"  情绪: {emo['dominant']} "
              f"(valence={emo['valence']:.2f}, arousal={emo['arousal']:.2f})")
        if emo.get("complex_emotion"):
            print(f"  复杂情绪: {emo['complex_emotion']['emotion']}")
        if perc["updates"]:
            print(f"  感知: {list(perc['updates'].keys())}")
        if perc["synesthesia"]:
            print(f"  通感: {[(q, ch) for q, ch in perc['synesthesia']]}")
        print(f"  行为: {'感知直接驱动' if bhv['is_perception_directed'] else '情绪驱动'}"
              f" (W_p={bhv['perception_weight']:.2f}, W_e={bhv['emotion_weight']:.2f})")

        # 模拟复习
        if result["memory"]["stored"]:
            engine.review_memories([result["memory"]["memory_id"]], importance="normal")

        # 每 3 轮做一次记忆检索演示
        if i == 3:
            print(f"\n  [检索测试] 查询'推荐食物'...")
            results = engine.retrieve_memories("推荐食物", k=3)
            for r in results:
                print(f"    → {r.content[:50]} (window={r.window})")

    # 结晶演示
    print(f"\n{'='*70}")
    print("结晶演示")
    # 模拟素食者被确认多次
    veg_result = engine.retrieve_memories("素食", k=1)
    if veg_result:
        veg_id = veg_result[0].id
        for _ in range(25):
            engine.review_memories([veg_id], importance="core")
        node_id = engine.crystallize_if_ready(veg_id, ["吃饭", "餐厅", "菜单", "推荐菜", "食谱"])
        if node_id:
            print(f"  ✓ 已结晶: {node_id}")
            kn = engine.memory.get_key_node(node_id)
            print(f"    内容: {kn.content}")
            print(f"    触发词: {kn.auto_trigger_keywords}")
        else:
            print(f"  尚未满足结晶条件")

    # 最终状态
    print(f"\n{'='*70}")
    print("指导生成测试:")
    guidance = engine.generate_reply_guidance()
    print(guidance)

    # 统计
    report = engine.get_session_report()
    print(f"\n--- 会话统计 ---")
    print(f"  互动次数: {report['interaction_count']}")
    print(f"  记忆总数: {report['memory_stats']['total_memories']}")
    print(f"  关键节点: {report['memory_stats']['key_nodes']}")

    print(f"\n✅ 演示完成")
