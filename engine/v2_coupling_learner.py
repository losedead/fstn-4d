# -*- coding: utf-8 -*-
"""
FSTN-4D v2 感知-情绪耦合系数在线学习器 (Coupling Coefficient Learner)
====================================================================
v1 的感知→情绪耦合矩阵系数是手调静态常量（如 thermal:too_hot → anger+0.4）。

v2 在线学习：
- 每次对话后，用「用户后续话语中实际表达的情绪」作为反馈信号
- 若实际情绪显著高于耦合预测 → 系数上调（该感知状态确实更易引发此情绪）
- 若实际情绪显著低于耦合预测 → 系数下调
- 用指数滑动平均（EMA）平滑，避免单次噪声干扰

设计：
- 保留 v1 静态矩阵作为先验/冷启动值
- 学习器维护 delta 修正表，最终系数 = 静态系数 * (1 + delta)
- delta ∈ [-0.5, +1.0]，EMA 学习率 η=0.15

用法：
    from v2_coupling_learner import CouplingLearner
    learner = CouplingLearner()
    # 每轮对话后：
    learner.update(triggered_rules, predicted_coupled, actual_emotion)
    # 取修正后的系数（提供给 couple_emotion 使用）：
    learner.get_adjusted_coefficients()  # -> {"thermal:too_hot": {"anger": 0.45, "disgust": 0.22}, ...}
"""

import json
import time
import os
from typing import Dict, List, Tuple, Optional, Any


class CouplingLearner:
    """
    感知→情绪耦合系数的在线学习器。
    修正 v1 的 COUPLING_RULES 静态系数。
    """

    # EMA 学习率
    ETA = 0.15
    # delta 边界
    DELTA_MIN = -0.5
    DELTA_MAX = 1.0
    # 需要多少条样本才开始影响（冷启动保护）
    MIN_SAMPLES = 5

    def __init__(self, state_file: Optional[str] = None):
        # rule -> emotion -> delta
        self.deltas: Dict[str, Dict[str, float]] = {}
        # rule -> emotion -> 样本数
        self.samples: Dict[str, Dict[str, int]] = {}
        # rule -> emotion -> 累计误差（用于统计）
        self.errors: Dict[str, Dict[str, float]] = {}
        self.state_file = state_file
        self.total_updates = 0
        if state_file:
            self._load_state()

    # ── 学习 ───────────────────────────────────────────────

    def update(self, triggered_rules: List[str],
               predicted_coupled: Dict[str, float],
               actual_emotion: Dict[str, float],
               weight: float = 1.0):
        """
        用实际情绪修正耦合系数。

        Args:
            triggered_rules: couple_emotion 返回的触发规则列表（如 ["thermal:too_hot"]）
            predicted_coupled: couple_emotion 返回的耦合后情绪
            actual_emotion: 增强检测器得到的实际情绪向量（或用户后续表达）
            weight: 样本权重（感知直接驱动时可降低，W_p 越大越不可靠）
        """
        if not triggered_rules:
            return

        for rule in triggered_rules:
            for emotion in self._emotions():
                pred = predicted_coupled.get(emotion, 0.0)
                actual = actual_emotion.get(emotion, 0.0)
                # 误差：实际 - 预测（仅当 pred>0 或 actual>0 时有效）
                if pred < 0.05 and actual < 0.05:
                    continue
                err = (actual - pred) * weight
                # 归一化误差到 delta 更新量（err ∈ [-1,1]，乘以学习率）
                d = self.ETA * err

                self.deltas.setdefault(rule, {}).setdefault(emotion, 0.0)
                self.samples.setdefault(rule, {}).setdefault(emotion, 0)
                self.errors.setdefault(rule, {}).setdefault(emotion, 0.0)

                cur = self.deltas[rule][emotion]
                self.deltas[rule][emotion] = max(
                    self.DELTA_MIN, min(self.DELTA_MAX, cur + d)
                )
                self.samples[rule][emotion] += 1
                self.errors[rule][emotion] += abs(err)
                self.total_updates += 1

    # ── 应用 ───────────────────────────────────────────────

    def get_adjusted_coefficients(self,
                                  static_rules: Dict[Tuple[str, str], Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        返回修正后的系数表（与 v1 COUPLING_RULES 同结构）。

        Args:
            static_rules: v1 的 COUPLING_RULES（(sense, state) -> {emotion: coef}）
        """
        adjusted: Dict[str, Dict[str, float]] = {}
        for (sense, state), coefs in static_rules.items():
            rule_key = f"{sense}:{state}"
            for emotion, c in coefs.items():
                new_c = c
                if rule_key in self.deltas and emotion in self.deltas[rule_key]:
                    n = self.samples[rule_key][emotion]
                    if n >= self.MIN_SAMPLES:
                        new_c = c * (1 + self.deltas[rule_key][emotion])
                adjusted.setdefault(rule_key, {})[emotion] = round(max(0.0, new_c), 3)
        return adjusted

    def get_adjustment_stats(self) -> Dict[str, Any]:
        """返回学习进度摘要"""
        summary = {}
        for rule, emo_deltas in self.deltas.items():
            items = []
            for emotion, d in emo_deltas.items():
                n = self.samples[rule].get(emotion, 0)
                if n > 0:
                    items.append({
                        "emotion": emotion,
                        "delta": round(d, 3),
                        "samples": n,
                        "avg_abs_error": round(self.errors[rule].get(emotion, 0) / n, 3),
                    })
            if items:
                summary[rule] = items
        return {"total_updates": self.total_updates, "rules": summary}

    def _emotions(self) -> List[str]:
        return ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    # ── 持久化 ─────────────────────────────────────────────

    def save_state(self):
        if not self.state_file:
            return
        state = {
            "deltas": self.deltas,
            "samples": self.samples,
            "errors": self.errors,
            "total_updates": self.total_updates,
            "saved_at": time.time(),
        }
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load_state(self) -> bool:
        if not self.state_file or not os.path.exists(self.state_file):
            return False
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.deltas = state.get("deltas", {})
            self.samples = state.get("samples", {})
            self.errors = state.get("errors", {})
            self.total_updates = state.get("total_updates", 0)
            return True
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════
# 便捷集成：把学习器接入 v1 FSTN4DEngine 的耦合步骤
# ═══════════════════════════════════════════════════════════════

class CouplingAdjustedStateMachine:
    """
    包装 v1 PerceptualStateMachine，使其使用学习后的耦合系数。
    用法：将 engine.perception.couple_emotion 替换为调整版。
    """

    def __init__(self, base_perception, learner: CouplingLearner):
        self.base = base_perception
        self.learner = learner
        self._adjusted = None

    def couple_emotion(self, emotional_state: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
        """使用学习后的系数执行耦合（与 v1 接口一致）"""
        from fstn_perception import PerceptualStateMachine
        # 计算学习后的系数表
        static = PerceptualStateMachine.COUPLING_RULES
        self._adjusted = self.learner.get_adjusted_coefficients(static)

        # 复制 v1 的耦合逻辑，但使用调整后的系数
        delta = {e: 0.0 for e in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]}
        triggered = []

        s = self.base.state
        rules = [
            ("thermal:too_hot",  s.thermal_comfort > 0.5),
            ("thermal:too_cold", s.thermal_comfort < -0.5),
            ("thermal:comfortable", abs(s.thermal_comfort) < 0.2),
            ("gustatory:bitter", s.gust_bitter > 0.6),
            ("gustatory:sweet",  s.gust_sweet > 0.6),
            ("interoceptive:hungry",  s.int_hunger > 0.7),
            ("interoceptive:thirsty", s.int_thirst > 0.7),
            ("interoceptive:tired",   s.int_fatigue > 0.7),
            ("tactile:pain", 0.4 < s.tactile_pain <= 0.8),
            ("tactile:severe_pain", s.tactile_pain > 0.8),
            ("auditory:noisy", s.aud_loudness > 0.7),
            ("auditory:quiet", s.aud_loudness < 0.2),
            ("visual:dark", s.vis_brightness < 0.2),
            ("olfactory:foul", s.olf_pleasant < -0.5),
            ("olfactory:fragrant", s.olf_pleasant > 0.5),
        ]

        for rule_key, active in rules:
            if not active:
                continue
            triggered.append(rule_key)
            coefs = self._adjusted.get(rule_key, {})
            for emotion, c in coefs.items():
                if emotion in delta:
                    delta[emotion] += c

        # 与 v1 相同：耦合强度 0.6 缩放
        coupled = {}
        for e in delta:
            base_v = emotional_state.get(e, 0)
            coupled[e] = min(1.0, base_v + delta[e] * 0.6)
        return coupled, triggered


# ═══════════════════════════════════════════════════════════════
# 命令行自测
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    learner = CouplingLearner()
    print("=" * 60)
    print("FSTN-4D v2 耦合系数在线学习 自测")
    print("=" * 60)

    # 模拟：用户反复在"太热"场景下表达愤怒
    for i in range(12):
        learner.update(
            ["thermal:too_hot"],
            {"anger": 0.24, "disgust": 0.12},   # 静态系数预测 (0.4, 0.2) * 0.6
            {"anger": 0.85, "disgust": 0.20},   # 实际：愤怒显著更高
            weight=0.8,
        )

    print("\n[场景1] 用户频繁在热天表达高愤怒（12 次样本）:")
    stats = learner.get_adjustment_stats()
    for rule, items in stats["rules"].items():
        for it in items:
            print(f"  {rule}.{it['emotion']}: delta={it['delta']:+.3f} 样本={it['samples']}")

    # 对照：另一个规则没有样本
    print("\n[场景2] 未学习的规则保持静态系数:")
    from fstn_perception import PerceptualStateMachine
    adjusted = learner.get_adjusted_coefficients(PerceptualStateMachine.COUPLING_RULES)
    print(f"  thermal:too_hot -> {adjusted.get('thermal:too_hot')}")
    print(f"  olfactory:foul  -> {adjusted.get('olfactory:foul')}  (未学习, 不变)")
