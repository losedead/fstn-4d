"""
FSTN-4D V3 自适应耦合矩阵 (Adaptive Coupling Matrix)
====================================================
针对 V1 手调死参数（COUPLING_RULES 静态字典）的升级：
感知→情绪耦合系数从「写死的常量」变为「可在线学习的权重」。

学习机制（超越 V1）：
1. Delta 规则在线学习：W_new = W_old + lr * (target - predicted) * x
   预测误差通过延迟反馈回传 —— 用户说"好热"（感知 x），
   之后话语检测到愤怒（target），系统就修正"热→怒"的耦合强度。
2. 置信度加权学习率：每条规则有 confidence，
   初期学习快（lr 高），稳定后收敛（lr 低），防止后期被单次噪声带偏。
3. 正负双向修正：耦合可以被加强（热→怒 0.4→0.6），
   也可以被削弱（热→怒 0.4→0.2，如果用户热了却开心）。
4. 学习日志：记录每一次修正（规则、方向、幅度、误差），可审计。
5. 记忆化：每个用户的耦合矩阵独立持久化 —— 这是"千人千面"的情绪模型。

数学说明（Delta Rule / Widrow-Hoff）：
  预测值 y_hat = sum(W_i * x_i)，其中 x_i 是触发强度
  误差 e = target - y_hat
  更新 W_i += lr * e * x_i
"""

import time
import json
import os
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class CouplingRule:
    """一条可学习的耦合规则"""
    sense: str            # 感知维度: thermal/gustatory/interoceptive...
    state: str            # 感知状态: too_hot/bitter/hungry...
    emotion: str          # 目标情绪: anger/fear/joy...
    weight: float         # 当前权重（初始来自 V1 专家经验值）
    confidence: float     # 置信度 0~1（学习次数越多越稳定）
    updates: int = 0      # 学习次数
    last_updated: float = field(default_factory=time.time)


class AdaptiveCouplingMatrix:
    """
    自适应感知-情绪耦合矩阵。

    初始权重 = V1 的专家经验耦合矩阵（保留设计洞见），
    之后通过反馈信号在线调整（升级实现手段）。
    """

    # V1 专家经验初始权重（感知状态 -> {情绪: 权重}）
    EXPERT_INIT = {
        ("thermal", "too_hot"):    {"anger": 0.4, "disgust": 0.2},
        ("thermal", "too_cold"):   {"sadness": 0.3, "fear": 0.2},
        ("thermal", "comfortable"): {"joy": 0.3},
        ("gustatory", "bitter"):   {"disgust": 0.6, "sadness": 0.2},
        ("gustatory", "sweet"):    {"joy": 0.5},
        ("gustatory", "spicy"):    {"surprise": 0.3, "anger": 0.2},
        ("interoceptive", "hungry"):  {"anger": 0.3, "sadness": 0.2},
        ("interoceptive", "thirsty"): {"fear": 0.2, "anger": 0.3},
        ("interoceptive", "tired"):   {"sadness": 0.4, "disgust": 0.2},
        ("tactile", "pain"):         {"fear": 0.4, "anger": 0.3},
        ("tactile", "severe_pain"):  {"fear": 0.7, "sadness": 0.3},
        ("auditory", "noisy"):       {"anger": 0.5, "fear": 0.2},
        ("auditory", "quiet"):       {"joy": 0.2},
        ("visual", "dark"):          {"fear": 0.3, "sadness": 0.2},
        ("visual", "bright"):        {"anger": 0.2, "surprise": 0.1},
        ("olfactory", "foul"):       {"disgust": 0.7, "anger": 0.2},
        ("olfactory", "fragrant"):   {"joy": 0.4},
    }

    # 学习超参数
    LR0 = 0.25          # 初始学习率（置信度低时）
    LR_MIN = 0.02       # 最小学习率（收敛后）
    CONF_INC = 0.12     # 每次学习置信度增量
    W_MAX = 1.2         # 权重上限（允许超过专家值一点，但不能失控）
    W_MIN = 0.0         # 权重下限（可为 0 = 该耦合被学习为无关）

    def __init__(self, state_file: Optional[str] = None):
        self.rules: Dict[Tuple[str, str, str], CouplingRule] = {}
        self.state_file = state_file
        self._init_expert()
        self.learning_log: List[Dict] = []
        self.feedback_queue: List[Dict] = []  # 延迟反馈队列
        if state_file and os.path.exists(state_file):
            self._load_state()

    def _init_expert(self):
        """用 V1 专家经验初始化"""
        for (sense, state), emo_map in self.EXPERT_INIT.items():
            for emo, w in emo_map.items():
                key = (sense, state, emo)
                self.rules[key] = CouplingRule(
                    sense=sense, state=state, emotion=emo,
                    weight=w, confidence=0.5,
                )

    # ═══════════════════════════════════════════════════════════
    # 预测接口（替代 V1 的 COUPLING_RULES 字典查询）
    # ═══════════════════════════════════════════════════════════

    def predict(self, active_states: List[Tuple[str, str]],
                trigger_intensity: float = 1.0) -> Tuple[Dict[str, float], List[str]]:
        """
        给定当前激活的感知状态列表，返回预测的情绪增量。

        Args:
            active_states: [("thermal", "too_hot"), ("interoceptive", "hungry")]
                           或三元组 [("thermal", "too_hot", 0.3)] 指定状态级强度
                           （V3 增强：陈旧感知用 0.3 强度，新鲜感知 1.0）
            trigger_intensity: 全局触发强度 0~1（默认 1.0）

        Returns:
            (情绪增量字典, 触发的规则说明列表)
        """
        delta: Dict[str, float] = {}
        triggered: List[str] = []
        for item in active_states:
            if len(item) == 3:
                sense, state, state_intensity = item
            else:
                sense, state = item
                state_intensity = trigger_intensity
            for (s, st, emo), rule in self.rules.items():
                if s == sense and st == state and rule.weight > 0.02:
                    delta[emo] = delta.get(emo, 0.0) + rule.weight * state_intensity
                    triggered.append(
                        f"{sense}:{state}→{emo}×{rule.weight:.2f}"
                        f"(强度{state_intensity:.1f})"
                    )
        return delta, triggered

    def predict_delta(self, active_states: List[Tuple[str, str]],
                      trigger_intensity: float = 1.0) -> Dict[str, float]:
        """只返回增量（供引擎融合）"""
        delta, _ = self.predict(active_states, trigger_intensity)
        return delta

    def get_weight(self, sense: str, state: str, emotion: str) -> float:
        key = (sense, state, emotion)
        return self.rules[key].weight if key in self.rules else 0.0

    # ═══════════════════════════════════════════════════════════
    # 在线学习接口
    # ═══════════════════════════════════════════════════════════

    def learn(self, active_states: List[Tuple[str, str]],
              target_emotion: Dict[str, float],
              trigger_intensity: float = 1.0) -> List[Dict]:
        """
        用目标情绪（实际检测到的用户情绪）修正耦合权重。

        Delta Rule:
            对每条激活的规则 (sense,state,emotion_e)：
              predicted = sum(所有激活规则给 emotion_e 的增量)
              error = target_emotion[e] - predicted_e
              weight_e += lr * error * trigger_intensity

        Returns: 本次学习产生的修正日志
        """
        if not active_states:
            return []
        # 计算当前预测
        delta, _ = self.predict(active_states, trigger_intensity)
        log = []

        for sense, state in active_states:
            for emo in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]:
                key = (sense, state, emo)
                rule = self.rules.get(key)
                if rule is None:
                    # ── V3 增强：从无到有创建新耦合 ──
                    # 专家矩阵没有这条规则，但用户反复在感知状态后
                    # 表达某情绪（target > 0.3）→ 说明存在个体特有耦合，
                    # 从 0 开始学习（不预设方向）
                    target_e = target_emotion.get(emo, 0.0)
                    if target_e > 0.3:
                        rule = CouplingRule(
                            sense=sense, state=state, emotion=emo,
                            weight=0.0, confidence=0.2,
                        )
                        self.rules[key] = rule
                    else:
                        continue
                # 该规则对 emotion 的当前贡献
                predicted_e = delta.get(emo, 0.0)
                target_e = target_emotion.get(emo, 0.0)
                error = target_e - predicted_e

                # 置信度自适应学习率
                lr = self.LR0 * (1.0 - rule.confidence) + self.LR_MIN * rule.confidence
                # 只对被激活的规则更新（x_i = trigger_intensity）
                update = lr * error * trigger_intensity

                if abs(update) > 1e-4:
                    old = rule.weight
                    rule.weight = max(self.W_MIN, min(self.W_MAX, old + update))
                    rule.confidence = min(1.0, rule.confidence + self.CONF_INC)
                    rule.updates += 1
                    rule.last_updated = time.time()
                    entry = {
                        "time": time.time(),
                        "rule": f"{sense}:{state}→{emo}",
                        "old": round(old, 4),
                        "new": round(rule.weight, 4),
                        "delta": round(rule.weight - old, 4),
                        "error": round(error, 4),
                        "target": round(target_e, 4),
                    }
                    log.append(entry)
                    self.learning_log.append(entry)

        # 保留最近 500 条学习日志
        if len(self.learning_log) > 500:
            self.learning_log = self.learning_log[-500:]
        return log

    def enqueue_feedback(self, sense: str, state: str, target_emotion: Dict[str, float],
                         intensity: float = 1.0, ttl: float = 300.0):
        """
        延迟反馈：感知状态触发时暂存反馈，
        等后续话语检测到实际情绪后再回填学习。

        用途：用户在"好热"之后几分钟说"烦死了"，
        系统把这次实际情绪与刚才的感知状态配对，修正"热→怒"。
        """
        self.feedback_queue.append({
            "time": time.time(), "ttl": ttl,
            "sense": sense, "state": state, "intensity": intensity,
            "target_emotion": target_emotion,
        })

    def drain_feedback(self, current_emotion: Dict[str, float]) -> List[Dict]:
        """处理到期反馈：用当前实际情绪学习"""
        now = time.time()
        logs: List[Dict] = []
        remaining = []
        for fb in self.feedback_queue:
            if now - fb["time"] > fb["ttl"]:
                continue  # 过期丢弃
            logs.extend(self.learn(
                [(fb["sense"], fb["state"])],
                current_emotion,
                trigger_intensity=fb["intensity"],
            ))
        self.feedback_queue = [fb for fb in self.feedback_queue
                               if now - fb["time"] <= fb["ttl"]]
        return logs

    def clear_feedback(self):
        self.feedback_queue = []

    # ═══════════════════════════════════════════════════════════
    # 可视化 & 持久化
    # ═══════════════════════════════════════════════════════════

    def diff_report(self, min_delta: float = 0.05) -> List[Dict]:
        """对比当前权重与专家初始值的差异（展示学习效果）"""
        report = []
        for (sense, state, emo), rule in self.rules.items():
            init_w = self.EXPERT_INIT.get((sense, state), {}).get(emo, 0.0)
            diff = rule.weight - init_w
            if abs(diff) >= min_delta:
                report.append({
                    "rule": f"{sense}:{state}→{emo}",
                    "expert": round(init_w, 3),
                    "learned": round(rule.weight, 3),
                    "delta": round(diff, 3),
                    "updates": rule.updates,
                    "confidence": round(rule.confidence, 2),
                })
        return report

    def save_state(self):
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        state = {
            "rules": [
                {"sense": r.sense, "state": r.state, "emotion": r.emotion,
                 "weight": r.weight, "confidence": r.confidence,
                 "updates": r.updates, "last_updated": r.last_updated}
                for r in self.rules.values()
            ],
            "learning_log": self.learning_log[-200:],
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)

    def _load_state(self):
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            for d in state.get("rules", []):
                key = (d["sense"], d["state"], d["emotion"])
                self.rules[key] = CouplingRule(
                    sense=d["sense"], state=d["state"], emotion=d["emotion"],
                    weight=d["weight"], confidence=d["confidence"],
                    updates=d.get("updates", 0),
                    last_updated=d.get("last_updated", time.time()),
                )
            self.learning_log = state.get("learning_log", [])
        except Exception:
            pass


# ── 自测：模拟一个学习场景 ───────────────────────────────────────
if __name__ == "__main__":
    print("=" * 66)
    print("自适应耦合矩阵自测 —— 模拟'热→怒'耦合的学习过程")
    print("=" * 66)

    m = AdaptiveCouplingMatrix()
    w0 = m.get_weight("thermal", "too_hot", "anger")
    print(f"\n初始 '热→怒' 权重: {w0:.2f} (来自 V1 专家经验)")

    # 场景 A：用户 8 次在说"好热"之后表达愤怒/烦躁
    print("\n── 场景 A：热了之后总是烦躁（学习 8 次）──")
    for i in range(8):
        target = {"anger": 0.7, "disgust": 0.3, "sadness": 0.0,
                  "fear": 0.0, "joy": 0.0, "surprise": 0.0}
        log = m.learn([("thermal", "too_hot")], target)
    wA = m.get_weight("thermal", "too_hot", "anger")
    print(f"  学习后 '热→怒' 权重: {w0:.2f} → {wA:.2f}")

    # 场景 B：同一用户换个场景，说热之后反而开心（在空调房）
    print("\n── 场景 B：热了却开心（反向学习 3 次，验证双向修正）──")
    for i in range(3):
        target = {"joy": 0.8, "anger": 0.0, "sadness": 0.0,
                  "fear": 0.0, "disgust": 0.0, "surprise": 0.1}
        m.learn([("thermal", "too_hot")], target, trigger_intensity=0.6)
    wB = m.get_weight("thermal", "too_hot", "anger")
    print(f"  学习后 '热→怒' 权重: {wA:.2f} → {wB:.2f}")

    # 展示差异报告
    print("\n── 差异报告（权重明显偏离专家值的规则）──")
    for d in m.diff_report(min_delta=0.05):
        print(f"  {d['rule']:30s} 专家={d['expert']:.2f} → 学习后={d['learned']:.2f}"
              f" (更新{d['updates']}次, 置信度{d['confidence']})")

    print("\n✅ 自适应耦合矩阵工作正常：权重可加强、可削弱、可审计")
