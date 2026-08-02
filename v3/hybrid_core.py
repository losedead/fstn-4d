"""
FSTN-4D V3 整合核心 (Hybrid Core Engine)
=========================================
神经-符号混合架构的统一入口。接口与 V1 FSTN4DEngine 完全兼容，
可无缝替换（Hermes 适配器、技能加载器不用改调用方式）。

V3 推理链（对比 V1 的增强点）：
  1. 情绪检测   → 双层检测器（关键词快速 + LLM 语义兜底）        [升级]
  2. 感知更新   → 复用 V1 七维感知状态机                          [保留]
  3. 感知→情绪  → 自适应耦合矩阵（可在线学习）                    [升级]
  4. 情绪→感知  → 复用 V1 反向调制逻辑                            [保留]
  5. 行为识别   → 感知直接驱动 (W_p=0.85) vs 情绪驱动             [保留]
  6. 记忆存储   → 向量化记忆（TF-IDF + 情绪标签 + 感知指纹）      [升级]
  7. 延迟反馈   → 感知状态入队，与后续实际情绪配对修正耦合        [新增]
  8. 情绪状态   → 双指数衰减 + 干扰规则（时间保持器）             [保留]

对外提供与 V1 相同的 API：
  process_utterance / retrieve_memories / review_memories /
  crystallize_if_ready / get_emotional_modulation_context /
  generate_reply_guidance / get_session_report / save_state / load_state
"""

import time
import json
import os
import math
import sys
from typing import Dict, List, Optional, Any

# 复用 V1 引擎（感知状态机、衰减配置）
_ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine")
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)
from fstn_perception import PerceptualStateMachine

# V3 同目录模块（兼容 python -m 与直接运行两种方式）
_V3_DIR = os.path.dirname(os.path.abspath(__file__))
if _V3_DIR not in sys.path:
    sys.path.insert(0, _V3_DIR)
from neural_detector import DualLayerEmotionDetector, LLMEmotionDetector, BASE_EMOTIONS
from vector_memory import VectorMemoryEngine, VRetrievalHit
from adaptive_coupling import AdaptiveCouplingMatrix


# ═══════════════════════════════════════════════════════════════
# 情绪状态保持器（双指数衰减 + 干扰规则）
# ═══════════════════════════════════════════════════════════════

class EmotionStateKeeper:
    """跨轮情绪状态管理：衰减 + 叠加 + 反转（复用 V1 的衰减配置）"""

    # 与 V1 一致的差异化双指数衰减（分钟单位→秒）
    DECAY_PROFILES = {
        "surprise":  {"tau_fast": 5*60,   "tau_slow": 30*60,  "alpha": 0.80},
        "disgust":   {"tau_fast": 10*60,  "tau_slow": 60*60,  "alpha": 0.70},
        "fear":      {"tau_fast": 15*60,  "tau_slow": 120*60, "alpha": 0.70},
        "anger":     {"tau_fast": 20*60,  "tau_slow": 180*60, "alpha": 0.60},
        "joy":       {"tau_fast": 15*60,  "tau_slow": 120*60, "alpha": 0.70},
        "sadness":   {"tau_fast": 30*60,  "tau_slow": 360*60, "alpha": 0.50},
    }

    def __init__(self):
        self.state: Dict[str, float] = {e: 0.0 for e in BASE_EMOTIONS}
        self.last_update: float = time.time()
        self.history: List[Dict] = []

    def _decay(self, vec: Dict[str, float], elapsed: float) -> Dict[str, float]:
        out = {}
        for e, v in vec.items():
            if v < 0.01:
                out[e] = 0.0
                continue
            p = self.DECAY_PROFILES[e]
            d = p["alpha"] * math.exp(-elapsed / p["tau_fast"]) + \
                (1 - p["alpha"]) * math.exp(-elapsed / p["tau_slow"])
            out[e] = max(0.0, min(1.0, v * d))
        return out

    def update(self, new_vec: Dict[str, float]):
        """融合新检测：先衰减旧状态，再按干扰规则融合"""
        now = time.time()
        elapsed = now - self.last_update
        old = self._decay(self.state, elapsed)

        # 干扰规则：同维叠加 / 反向覆盖
        old_valence = self._valence(old)
        new_valence = self._valence(new_vec)
        merged = {}
        for e in BASE_EMOTIONS:
            o, n = old.get(e, 0), new_vec.get(e, 0)
            if o < 0.05 and n < 0.05:
                merged[e] = 0.0
            elif o > 0.1 and n > 0.1:
                # 同维叠加（V3 修正：原公式过猛会把残留+新值推满格，
                # 改为温和叠加，保留旧值阴影而非放大）
                merged[e] = min(1.0, o * 0.5 + n * (1 + 0.1 * o))
            elif abs(old_valence - new_valence) > 1.0 and n > o * 0.8:
                # 反向覆盖：旧情绪快速压制（残留 20% 阴影）
                merged[e] = max(n, o * math.exp(-3 * n))
            else:
                merged[e] = max(o * 0.3, n)
            merged[e] = max(0.0, min(1.0, merged[e]))

        self.state = merged
        self.last_update = now
        self.history.append({
            "vector": merged.copy(), "timestamp": now,
            "dominant": self.dominant(),
        })
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def get_current(self) -> Dict:
        now = time.time()
        vec = self._decay(self.state, now - self.last_update)
        return {
            "base_vector": vec,
            "dominant": self.dominant(vec),
            "valence": self._valence(vec),
            "arousal": self._arousal(vec),
        }

    @staticmethod
    def _valence(vec: Dict[str, float]) -> float:
        v = (vec.get("joy", 0) * 0.9
             - vec.get("anger", 0) * 0.8
             - vec.get("disgust", 0) * 0.7
             - vec.get("fear", 0) * 0.9
             - vec.get("sadness", 0) * 0.8)
        return max(-1.0, min(1.0, v / 2.5))

    @staticmethod
    def _arousal(vec: Dict[str, float]) -> float:
        a = (vec.get("anger", 0) * 0.9 + vec.get("fear", 0) * 0.85
             + vec.get("surprise", 0) * 0.9 + vec.get("joy", 0) * 0.5
             + vec.get("sadness", 0) * 0.3)
        return max(0.0, min(1.0, a / 4.0))

    def dominant(self, vec: Optional[Dict] = None) -> str:
        v = vec or self.state
        if max(v.values()) < 0.1:
            return "neutral"
        return max(v, key=v.get)

    def get_trajectory(self) -> Dict[str, List[float]]:
        traj = {e: [] for e in BASE_EMOTIONS}
        for h in self.history:
            for e in BASE_EMOTIONS:
                traj[e].append(h["vector"].get(e, 0))
        return traj


# ═══════════════════════════════════════════════════════════════
# 整合核心引擎
# ═══════════════════════════════════════════════════════════════

class FSTN4DEngineV3:
    """FSTN-4D V3 神经-符号混合引擎（接口兼容 V1）"""

    def __init__(self, state_dir: str = None, use_llm: bool = True):
        self.state_dir = state_dir or os.path.expanduser("~/.fstn_engine_v3")
        os.makedirs(self.state_dir, exist_ok=True)

        # 三层升级引擎
        self.detector = DualLayerEmotionDetector(
            llm_detector=LLMEmotionDetector() if use_llm else None
        )
        self.memory = VectorMemoryEngine(
            state_file=os.path.join(self.state_dir, "memory_state.json")
        )
        self.coupling = AdaptiveCouplingMatrix(
            state_file=os.path.join(self.state_dir, "coupling_state.json")
        )
        self.perception = PerceptualStateMachine()  # 复用 V1
        self.keeper = EmotionStateKeeper()

        # 会话统计
        self.session_start = time.time()
        self.interaction_count = 0
        self.last_consolidate = time.time()

    # ═══════════════════════════════════════════════════════════
    # 核心推理链
    # ═══════════════════════════════════════════════════════════

    def process_utterance(self, utterance: str, context: str = "") -> dict:
        """完整的感知-情绪-行为推理链（V3 版本）"""
        self.interaction_count += 1

        # 1. 双层情绪检测
        det = self.detector.detect(utterance, context)
        self.keeper.update(det["base_vector"])
        current_emotion = self.keeper.get_current()

        # 2. 感知更新（V1 状态机）
        perception_updates = self.perception.update_from_utterance(utterance)
        perceptual_state = self.perception.get_current()

        # 3. 感知→情绪耦合（自适应矩阵预测 + 陈旧感知弱化）
        active_states = self.perception.get_active_coupling_states()
        fresh_senses = set(perception_updates.keys())
        # 本轮提及的感知维度 → 满强度；前几轮残留 → 0.3 强度
        coupling_states = [
            (sense, state, 1.0 if sense in fresh_senses else 0.3)
            for sense, state in active_states
        ]
        coupled_delta, triggered_rules = self.coupling.predict(coupling_states)

        # 4. 情绪→感知反向调制（V1 逻辑）
        modulated_perception = self.perception.modulate_perception_by_emotion(
            {"base_vector": current_emotion["base_vector"],
             "dominant": current_emotion["dominant"]}
        )

        # 5. 感知直接行为识别（V1 逻辑）
        direct_behavior = self.perception.detect_direct_behavior(utterance)

        # 6. 融合最终情绪 = 检测情绪 + 耦合增量
        base = current_emotion["base_vector"]
        if direct_behavior and direct_behavior["is_perception_directed"]:
            W_p = direct_behavior["perception_weight"]
            W_e = direct_behavior["emotion_weight"]
        else:
            W_p, W_e = 0.0, 1.0
        final_emotion = {}
        for e in BASE_EMOTIONS:
            final_emotion[e] = min(1.0, base.get(e, 0) * W_e
                                   + coupled_delta.get(e, 0) * 0.6 * (1 - W_p))

        # 7. 存储记忆（向量化 + 情绪 + 感知指纹）
        memory_id = None
        if self._worth_remembering(utterance):
            memory_id = self.memory.ingest(
                content=utterance,
                layer="episodic",
                recorded_emotion=final_emotion,
                emotional_tags=self._infer_tags(current_emotion["dominant"]),
                perceptual_signature=self.perception.build_perceptual_fingerprint(utterance),
                pending_confirmation=any(v > 0.8 for v in final_emotion.values()),
            )

        # 8. 延迟反馈学习：把感知状态入队，等后续实际情绪配对
        for sense, state in active_states:
            self.coupling.enqueue_feedback(
                sense, state, final_emotion, intensity=0.8, ttl=300.0
            )

        # 9. 定期巩固
        if self.interaction_count % 10 == 0:
            self.memory.consolidate()
            self.coupling.save_state()
            self.memory.save_state()
            self.last_consolidate = time.time()

        return {
            "emotion": {
                "dominant": current_emotion["dominant"],
                "base_vector": current_emotion["base_vector"],
                "valence": current_emotion["valence"],
                "arousal": current_emotion["arousal"],
                "detector_source": det["source"],
                "detector_confidence": det["confidence"],
                "complex": self._detect_complex(final_emotion),
                "final_emotion": final_emotion,
                "coupled_delta": coupled_delta,
            },
            "perception": {
                "updates": perception_updates,
                "current_state": perceptual_state,
                "modulated": modulated_perception,
                "dominant_sense": self.perception.get_dominant()[0],
                "synesthesia": self.perception.get_synesthesia_qualities(),
                "active_coupling": triggered_rules,
            },
            "behavior": {
                "is_perception_directed": direct_behavior is not None
                if direct_behavior else False,
                "perception_direct_detail": direct_behavior,
                "perception_weight": W_p,
                "emotion_weight": W_e,
            },
            "memory": {
                "stored": memory_id is not None,
                "memory_id": memory_id,
                "pending_confirmation": any(v > 0.8 for v in final_emotion.values()),
            },
            "learning": {
                "coupling_updates_pending": len(self.coupling.feedback_queue),
            },
        }

    # ═══════════════════════════════════════════════════════════
    # 检索 API（接口对齐 V1）
    # ═══════════════════════════════════════════════════════════

    def retrieve_memories(self, query: str, k: int = 10,
                          emotion_aware: bool = True) -> List[VRetrievalHit]:
        cur = self.keeper.get_current()
        if emotion_aware:
            return self.memory.retrieve_emotion_aware(query, k=k, current_emotion=cur)
        return self.memory.retrieve(query, k=k)

    def review_memories(self, memory_ids: List[str], importance: str = "normal"):
        gamma = self.memory.GAMMA_MAP.get(importance, self.memory.GAMMA_DEFAULT)
        return self.memory.review(memory_ids, gamma)

    def crystallize_if_ready(self, memory_id: str,
                             trigger_keywords: List[str] = None) -> Optional[str]:
        return self.memory.crystallize(memory_id, trigger_keywords,
                                       current_emotion=self.keeper.get_current())

    def drain_coupling_feedback(self) -> int:
        """处理延迟反馈队列（在下一轮对话开始时调用）"""
        cur = self.keeper.get_current()
        logs = self.coupling.drain_feedback(cur["base_vector"])
        if logs:
            self.coupling.save_state()
        return len(logs)

    # ═══════════════════════════════════════════════════════════
    # 推理辅助（对齐 V1）
    # ═══════════════════════════════════════════════════════════

    def get_emotional_modulation_context(self) -> dict:
        cur = self.keeper.get_current()
        dom = cur["dominant"]
        return {
            "dominant_emotion": dom,
            "valence": cur["valence"],
            "arousal": cur["arousal"],
            "action_bias": self._get_action_bias(cur),
            "window_access": self._get_window_access(cur),
            "perceptual_dominant": self.perception.get_dominant()[0],
            "active_key_nodes": [
                {"id": n.id, "content": n.content,
                 "triggers": n.auto_trigger_keywords}
                for n in self.memory.get_all_key_nodes()
            ],
            "should_avoid": self._get_avoidance(cur),
            "detector_source": self.detector.stats,
        }

    def generate_reply_guidance(self) -> str:
        ctx = self.get_emotional_modulation_context()
        cur = self.keeper.get_current()
        lines = ["[FSTN-4D V3 引擎状态]"]
        dom = cur["dominant"]
        if dom != "neutral":
            lines.append(f"  当前情绪: {dom} (效价={cur['valence']:.2f}, "
                         f"唤醒度={cur['arousal']:.2f})")
            lines.append(f"  行为偏置: {ctx['action_bias']}")
            lines.append(f"  窗口访问: {ctx['window_access']}")
        complex_e = self._detect_complex(cur["base_vector"])
        if complex_e:
            lines.append(f"  复杂情绪: {complex_e['emotion']} "
                         f"(强度={complex_e['intensity']:.3f})")
        if ctx["perceptual_dominant"]:
            lines.append(f"  主导感知: {ctx['perceptual_dominant']}")
        if ctx["active_key_nodes"]:
            lines.append(f"  关键节点激活: {len(ctx['active_key_nodes'])} 个")
        if ctx["should_avoid"]:
            lines.append(f"  应避免: {', '.join(ctx['should_avoid'])}")
        lines.append(f"  检测器统计: {self.detector.get_stats()}")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 统计 & 持久化（对齐 V1）
    # ═══════════════════════════════════════════════════════════

    def get_session_report(self) -> dict:
        return {
            "version": "3.0",
            "session_duration_seconds": time.time() - self.session_start,
            "interaction_count": self.interaction_count,
            "memory_stats": self.memory.get_statistics(),
            "emotion_trajectory": self.keeper.get_trajectory(),
            "emotion_history": [
                {"dominant": h["dominant"],
                 "valence": self.keeper._valence(h["vector"])}
                for h in self.keeper.history[-10:]
            ],
            "coupling_stats": {
                "rules": len(self.coupling.rules),
                "total_updates": sum(r.updates for r in self.coupling.rules.values()),
                "feedback_pending": len(self.coupling.feedback_queue),
                "learned_diffs": len(self.coupling.diff_report()),
            },
            "detector_stats": self.detector.get_stats(),
        }

    def save_state(self):
        self.memory.save_state()
        self.coupling.save_state()
        state = {
            "interaction_count": self.interaction_count,
            "session_start": self.session_start,
        }
        with open(os.path.join(self.state_dir, "engine_v3_state.json"),
                  "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self) -> bool:
        path = os.path.join(self.state_dir, "engine_v3_state.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
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
        if len(utterance) < 4:
            return False
        pure = ["继续", "下一步", "好的", "ok", "嗯", "哦", "对", "是的"]
        if utterance.strip().lower() in pure:
            return False
        return True

    @staticmethod
    def _infer_tags(dominant: str) -> List[str]:
        tag_map = {
            "anger": ["confront", "negative"], "fear": ["threat", "negative"],
            "sadness": ["social_support", "negative"], "joy": ["positive", "pleasure"],
            "disgust": ["avoid", "negative"], "surprise": ["novelty"],
        }
        return tag_map.get(dominant, [])

    @staticmethod
    def _detect_complex(vec: Dict[str, float]) -> Optional[Dict]:
        recipes = {
            "jealousy": {"sadness": 0.4, "anger": 0.4, "fear": 0.2},
            "shame": {"sadness": 0.5, "fear": 0.3, "disgust": 0.2},
            "guilt": {"sadness": 0.6, "fear": 0.3, "anger": 0.1},
            "empathy": {"sadness": 0.5, "joy": 0.3, "surprise": 0.2},
            "love": {"joy": 0.6, "sadness": 0.2, "fear": 0.2},
            "gratitude": {"joy": 0.8, "sadness": 0.2},
            "anxiety": {"fear": 0.6, "sadness": 0.3, "surprise": 0.1},
            "resentment": {"anger": 0.5, "sadness": 0.4, "disgust": 0.1},
            "nostalgia": {"sadness": 0.4, "joy": 0.4, "surprise": 0.2},
        }
        # ── V3 升级 1：矛盾情感检测（混合情绪优先于误报具体标签）──
        # 正效价与负效价同时强激活（如 joy 与 sadness/anger 同 >0.4）→ ambivalent
        positive = max(vec.get("joy", 0), 0)
        negatives = [vec.get("sadness", 0), vec.get("anger", 0),
                     vec.get("fear", 0), vec.get("disgust", 0)]
        top_neg = max(negatives) if negatives else 0.0
        if positive > 0.4 and top_neg > 0.4:
            return {
                "emotion": "ambivalent",
                "intensity": round((positive + top_neg) / 2, 3),
                "conflict": [max(["joy"], key=lambda e: vec.get(e, 0)),
                             max(["sadness", "anger", "fear", "disgust"],
                                 key=lambda e: vec.get(e, 0))],
            }

        # ── V3 升级 2：配方核心维度门槛（消除单维高分误报）──
        best, best_score = None, 0.0
        for name, recipe in recipes.items():
            # 配方主维度须显著激活（≥0.25），次维度可弱激活（≥0.1）
            # 例：gratitude(joy 0.8, sadness 0.2) 允许纯感激(joy 高, sadness 低)
            top_dims = sorted(recipe, key=lambda e: -recipe[e])[:2]
            if vec.get(top_dims[0], 0) < 0.25:
                continue
            if vec.get(top_dims[1], 0) < 0.1:
                continue
            score = sum(vec.get(e, 0) * w for e, w in recipe.items())
            if score > best_score and score > 0.45:
                best, best_score = name, score
        active = sum(1 for v in vec.values() if v > 0.1)
        if active < 2 or best is None:
            return None
        return {"emotion": best, "intensity": round(best_score, 3)}

    @staticmethod
    def _get_action_bias(cur: dict) -> str:
        vec = cur["base_vector"]
        if vec.get("joy", 0) > 0.6: return "explore"
        if vec.get("fear", 0) > 0.6: return "avoid"
        if vec.get("anger", 0) > 0.6: return "confront"
        if vec.get("sadness", 0) > 0.6: return "seek_support"
        if vec.get("disgust", 0) > 0.6: return "avoid"
        if vec.get("surprise", 0) > 0.7: return "reset_attention"
        return "neutral"

    @staticmethod
    def _get_window_access(cur: dict) -> str:
        vec = cur["base_vector"]
        if vec.get("fear", 0) > 0.6: return "restricted"
        if vec.get("joy", 0) > 0.6: return "full"
        return "full"

    @staticmethod
    def _get_avoidance(cur: dict) -> List[str]:
        avoid = []
        vec = cur["base_vector"]
        if vec.get("sadness", 0) > 0.6:
            avoid += ["过于乐观的建议", "催促决策"]
        if vec.get("anger", 0) > 0.6:
            avoid += ["妥协方案", "回避问题"]
        if vec.get("fear", 0) > 0.6:
            avoid += ["冒险建议", "忽视安全"]
        return avoid


# ── 命令行演示（端到端） ─────────────────────────────────────────
if __name__ == "__main__":
    engine = FSTN4DEngineV3()
    print("=" * 66)
    print("FSTN-4D V3 神经-符号混合引擎 端到端演示")
    print("=" * 66)

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
        print(f"\n{'─'*64}")
        print(f"[第 {i} 轮] {utterance[:46]}...")
        print(f"  情绪: {emo['dominant']:10s} 检测来源: {emo['detector_source']:9s} "
              f"(conf={emo['detector_confidence']:.2f})")
        print(f"        效价={emo['valence']:.2f} 唤醒度={emo['arousal']:.2f}")
        if emo.get("complex"):
            print(f"  复杂情绪: {emo['complex']['emotion']}")
        if perc["active_coupling"]:
            print(f"  耦合触发: {perc['active_coupling'][:3]}")
        if bhv["is_perception_directed"]:
            print(f"  行为: 感知直接驱动 (W_p={bhv['perception_weight']:.2f})")
        if result["memory"]["stored"]:
            engine.review_memories([result["memory"]["memory_id"]], importance="normal")

    # 检索演示
    print(f"\n{'='*66}")
    print("向量化检索演示（'推荐食物'）")
    for hit in engine.retrieve_memories("推荐食物", k=3):
        print(f"  {hit.score:.3f} w={hit.window} {hit.content[:40]}")
        print(f"        命中词: {hit.top_terms}")

    print(f"\n{'='*66}")
    print("指导生成测试:")
    print(engine.generate_reply_guidance())

    report = engine.get_session_report()
    print(f"\n--- 会话统计 ---")
    print(f"  记忆总数: {report['memory_stats']['total_memories']}")
    print(f"  耦合规则: {report['coupling_stats']['rules']} 条, "
          f"学习修正 {report['coupling_stats']['total_updates']} 次")
    print(f"  检测器: {report['detector_stats']}")
    print(f"\n✅ V3 引擎演示完成")
