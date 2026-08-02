# -*- coding: utf-8 -*-
"""
FSTN-4D v2 集成引擎 (Unified v2 Engine)
========================================
把三个 v2 增强层缝合进 v1 FSTN4DEngine，提供零改动替换。

增强点：
  1. 情绪检测：EnhancedEmotionDetector（否定翻转 + 程度校准 + 高信号词注入）
  2. 记忆检索：VectorMemoryIndex（向量语义检索，TF-IDF 本地 / Ollama embedding）
  3. 耦合系数：CouplingLearner（在线学习，EMA 修正静态系数）

用法（替换 v1 引擎）：
    from v2_engine import FSTN4DEngineV2
    engine = FSTN4DEngineV2(state_dir="~/.fstn_engine")
    result = engine.process_utterance("同事升职了，我有点嫉妒", "")
    guidance = engine.generate_reply_guidance()

兼容性：
  - 接口与 v1 FSTN4DEngine 完全一致（process_utterance / retrieve_memories /
    review_memories / crystallize_if_ready / generate_reply_guidance / save_state /
    load_state / get_session_report）
  - 内部仍委托 v1 的三层引擎，v2 层作为前处理/后处理增强
"""

import os
import sys
import time
import json
from typing import Dict, List, Tuple, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fstn_core import FSTN4DEngine
from v2_emotion_classifier import EnhancedEmotionDetector
from v2_coupling_learner import CouplingLearner, CouplingAdjustedStateMachine
from v2_vector_retrieval import VectorMemoryIndex, build_retrieval, semantic_retrieve


class FSTN4DEngineV2(FSTN4DEngine):
    """
    FSTN-4D v2 集成引擎。
    继承 v1 全部能力，覆写情绪检测与耦合，注入向量检索。
    """

    def __init__(self, state_dir: str = None, prefer_embedding: str = "auto",
                 lambda_sem: float = 0.75):
        super().__init__(state_dir)

        # v2 增强：情绪检测（包装 v1 状态机）
        self.emotion = EnhancedEmotionDetector(self.emotion)

        # v2 增强：耦合学习器
        self.coupling_learner = CouplingLearner(
            state_file=os.path.join(self.state_dir, "coupling_learner.json")
        )
        self.coupling_learner._load_state()
        # 包装感知状态机，使用学习后的耦合系数
        self.perception_v2 = CouplingAdjustedStateMachine(self.perception,
                                                          self.coupling_learner)

        # v2 增强：向量检索（懒加载，首次检索时构建）
        self._vector_index = None
        self._vector_index_built = False
        self.prefer_embedding = prefer_embedding
        self.lambda_sem = lambda_sem
        self._retrieval_provider = "pending"

    # ═══════════════════════════════════════════════════════════
    # 覆写：感知-情绪耦合（使用学习后的系数）
    # ═══════════════════════════════════════════════════════════

    def _couple_with_learning(self, emotional_state: Dict[str, float]):
        """使用学习后的耦合系数执行感知→情绪耦合"""
        coupled, triggered = self.perception_v2.couple_emotion(emotional_state)
        return coupled, triggered

    # ═══════════════════════════════════════════════════════════
    # 覆写：主推理链
    # ═══════════════════════════════════════════════════════════

    def process_utterance(self, utterance: str, context: str = "") -> dict:
        """v2 推理链：与 v1 相同的返回结构，但使用增强层"""
        self.interaction_count += 1

        # Step 1: 增强情绪检测
        previous_emotion = (self.emotion.state.copy()
                            if any(self.emotion.state.values()) else None)
        emotion_result = self.emotion.detect(utterance, context, previous_emotion)
        current_emotion = self.emotion.get_current()

        # Step 2: 感知更新（v1 状态机）
        perception_updates = self.perception.update_from_utterance(utterance)
        perceptual_state = self.perception.get_current()

        # Step 3: 情绪反向调制感知
        modulated_perception = self.perception.modulate_perception_by_emotion(
            {"base_vector": current_emotion["base_vector"],
             "dominant": current_emotion["dominant"]}
        )

        # Step 4: 感知→情绪耦合（v2 学习后的系数）
        coupled_emotion, triggered_rules = self._couple_with_learning(
            current_emotion["base_vector"]
        )

        # Step 5: 感知直接行为识别（v1）
        direct_behavior = self.perception.detect_direct_behavior(utterance)

        # Step 6: 融合最终情绪（同 v1 逻辑）
        if direct_behavior and direct_behavior["is_perception_directed"]:
            W_p = direct_behavior["perception_weight"]
            W_e = direct_behavior["emotion_weight"]
        else:
            W_p = 0.0
            W_e = 1.0

        final_emotion = {}
        base = current_emotion["base_vector"]
        if W_p > 0.3:
            for e in self.emotion.BASE_EMOTIONS:
                final_emotion[e] = base.get(e, 0) * 0.3 + coupled_emotion.get(e, 0) * 0.4
        else:
            for e in self.emotion.BASE_EMOTIONS:
                final_emotion[e] = base.get(e, 0) * 0.5 + coupled_emotion.get(e, 0) * 0.2

        # Step 7: 耦合学习反馈（用实际检测情绪修正系数）
        self.coupling_learner.update(
            triggered_rules,
            coupled_emotion,
            {e: final_emotion.get(e, 0) for e in self.emotion.BASE_EMOTIONS},
            weight=(1.0 - W_p),  # 感知直接驱动时反馈权重降低
        )

        # Step 8: 存储记忆（如果值得保留）
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
            # 向量索引增量更新
            if self._vector_index is not None and memory_id:
                try:
                    self._vector_index.add(self.memory.memories[memory_id])
                except Exception:
                    pass

        # Step 9: 定期巩固
        if self.interaction_count % 10 == 0:
            self.memory.consolidate()
            self.memory.prune_wormholes()
            self.last_consolidate = time.time()

        # 返回与 v1 相同的结构
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
                "learner_stats": self.coupling_learner.get_adjustment_stats(),
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
    # 覆写：检索 API（向量优先）
    # ═══════════════════════════════════════════════════════════

    def retrieve_memories(self, query: str, k: int = 10,
                          emotion_aware: bool = True) -> list:
        """v2 检索：向量语义检索优先，失败回退 v1"""
        # 懒构建向量索引
        if not self._vector_index_built:
            try:
                self._vector_index = build_retrieval(
                    self.memory, prefer=self.prefer_embedding,
                    lambda_sem=self.lambda_sem
                )
                self._retrieval_provider = self._vector_index.stats()["provider"]
            except Exception as e:
                self._retrieval_provider = f"build_failed:{e}"
            self._vector_index_built = True

        if self._vector_index is not None:
            try:
                hits = self._vector_index.query(query, k=k)
                if hits:
                    results = [entry for _, entry in hits]
                    # 情绪调制排序（复用 v1 逻辑，如果开启）
                    if emotion_aware:
                        current_emotion = self.emotion.get_current()
                        if current_emotion["dominant"] != "neutral":
                            scored = []
                            for mem in results:
                                base_score = 0.8
                                mod = self.memory.emotional_modulation(
                                    mem.id, base_score, current_emotion
                                )
                                scored.append((mod, mem))
                            scored.sort(key=lambda x: -x[0])
                            results = [mem for _, mem in scored]
                    return results
            except Exception:
                pass

        # 回退 v1
        return super().retrieve_memories(query, k=k, emotion_aware=emotion_aware)

    def get_retrieval_provider(self) -> str:
        """返回当前检索提供者名称"""
        return self._retrieval_provider

    # ═══════════════════════════════════════════════════════════
    # 覆写：持久化
    # ═══════════════════════════════════════════════════════════

    def save_state(self):
        """保存 v1 状态 + 耦合学习器状态"""
        super().save_state()
        try:
            self.coupling_learner.save_state()
        except Exception:
            pass

    def get_session_report(self) -> dict:
        """v2 会话报告（含增强层统计）"""
        report = super().get_session_report()
        report["v2"] = {
            "emotion_enhanced": True,
            "coupling_updates": self.coupling_learner.total_updates,
            "coupling_stats": self.coupling_learner.get_adjustment_stats(),
            "retrieval_provider": self.get_retrieval_provider(),
        }
        return report


# ═══════════════════════════════════════════════════════════════
# 命令行端到端演示
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FSTN-4D v2 引擎演示")
    parser.add_argument("--embedding", default="auto",
                        choices=["auto", "ollama", "local"])
    args = parser.parse_args()

    engine = FSTN4DEngineV2(prefer_embedding=args.embedding)

    print("=" * 70)
    print("FSTN-4D v2 集成引擎 端到端演示")
    print("=" * 70)

    conversations = [
        ("小红很爱吃糖果，每天都吃一颗", ""),
        ("用户是素食主义者，不吃任何肉类", ""),
        ("今天工作被批评了，好难过。推荐点吃的吧。", ""),
        ("其实我妈妈刚才安慰我了，我现在好多了，甚至有点开心", ""),
        ("好热啊，帮我把空调打开", ""),
        ("同事升职了，明明我做得更多，说实话有点不是滋味", ""),
        ("我不生气，只是有点失落", ""),   # v1 硬伤用例
        ("谢谢你一直陪着我", ""),          # 高信号词用例
    ]

    for i, (u, c) in enumerate(conversations, 1):
        r = engine.process_utterance(u, c)
        emo = r["emotion"]
        print(f"\n[{i}] {u[:40]}")
        print(f"  情绪: {emo['dominant']} (valence={emo['valence']:.2f})")
        if emo.get("complex_emotion"):
            print(f"  复杂: {emo['complex_emotion']['emotion']}")
        if r["coupling"]["learner_stats"]["total_updates"]:
            print(f"  耦合学习更新: {r['coupling']['learner_stats']['total_updates']} 次")

    print(f"\n--- 检索演示 ---")
    results = engine.retrieve_memories("推荐素食", k=3)
    print(f"provider: {engine.get_retrieval_provider()}")
    for r in results:
        print(f"  → {r.content[:40]}")

    print(f"\n--- 会话报告 ---")
    report = engine.get_session_report()
    print(f"  互动: {report['interaction_count']}")
    print(f"  记忆: {report['memory_stats']['total_memories']}")
    print(f"  v2:   {report['v2']}")
    engine.save_state()
    print("\n✅ v2 引擎演示完成，状态已保存")
