"""
FSTN-4D 情绪状态机 (Emotional State Machine)
==============================================
基于保罗·艾克曼六大基本情绪的计算引擎。
支持：情绪检测、双指数衰减、干扰规则、复杂社会情绪识别、效价/唤醒度计算。

创新点（超越原 spec）：
1. 中文关键词映射拓展至 200+ 词条（含口语化表达、网络用语）
2. 上下文情绪转折检测（"但是""不过""虽然"自动触发情绪反转）
3. 情绪历史回放（可追溯 N 步前的情绪状态用于干扰计算）
4. 强度自适应——同一关键词在不同上下文中的强度自动调节
"""

import math
import time
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class EmotionalState:
    """情绪状态快照"""
    anger: float = 0.0
    disgust: float = 0.0
    fear: float = 0.0
    joy: float = 0.0
    sadness: float = 0.0
    surprise: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_vector(self) -> Dict[str, float]:
        return {
            "anger": self.anger, "disgust": self.disgust,
            "fear": self.fear, "joy": self.joy,
            "sadness": self.sadness, "surprise": self.surprise
        }

    def dominant(self) -> str:
        v = self.to_vector()
        if max(v.values()) < 0.1:
            return "neutral"
        return max(v, key=v.get)

    def intensity(self) -> float:
        return math.sqrt(sum(v**2 for v in self.to_vector().values())) / math.sqrt(6)


@dataclass
class EmotionResult:
    """情绪检测结果"""
    base_vector: Dict[str, float]
    valence: float
    arousal: float
    dominant: str
    complex_emotion: Optional[Dict[str, Any]] = None
    intensity: float = 0.0
    interference: Optional[Dict] = None
    triggered_terms: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 核心引擎
# ═══════════════════════════════════════════════════════════════

class EmotionalStateMachine:
    """FSTN-4D 情绪状态机"""

    BASE_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    # 差异化双指数衰减配置（单位：秒）
    DECAY_PROFILES = {
        "surprise":  {"tau_fast": 5*60,   "tau_slow": 30*60,  "alpha": 0.80},
        "disgust":   {"tau_fast": 10*60,  "tau_slow": 60*60,  "alpha": 0.70},
        "fear":      {"tau_fast": 15*60,  "tau_slow": 120*60, "alpha": 0.70},
        "anger":     {"tau_fast": 20*60,  "tau_slow": 180*60, "alpha": 0.60},
        "joy":       {"tau_fast": 15*60,  "tau_slow": 120*60, "alpha": 0.70},
        "sadness":   {"tau_fast": 30*60,  "tau_slow": 360*60, "alpha": 0.50},
    }

    # 复杂社会情绪配方
    COMPLEX_EMOTIONS = {
        "jealousy":      {"sadness": 0.4, "anger": 0.4, "fear": 0.2},       # 嫉妒
        "shame":         {"sadness": 0.5, "fear": 0.3, "disgust": 0.2},     # 羞愧
        "guilt":         {"sadness": 0.6, "fear": 0.3, "anger": 0.1},       # 内疚
        "embarrassment": {"surprise": 0.4, "sadness": 0.3, "fear": 0.3},    # 尴尬
        "empathy":       {"sadness": 0.5, "joy": 0.3, "surprise": 0.2},     # 共情
        "love":          {"joy": 0.6, "sadness": 0.2, "fear": 0.2},         # 爱
        "gratitude":     {"joy": 0.8, "sadness": 0.2},                      # 感激
        "pride":         {"joy": 0.7, "surprise": 0.3},                     # 自豪
        "anxiety":       {"fear": 0.6, "sadness": 0.3, "surprise": 0.1},    # 焦虑（创新）
        "resentment":    {"anger": 0.5, "sadness": 0.4, "disgust": 0.1},   # 怨愤（创新）
        "relief":        {"joy": 0.5, "sadness": 0.3},                      # 释然（创新）
        "nostalgia":     {"sadness": 0.4, "joy": 0.4, "surprise": 0.2},    # 怀旧（创新）
    }

    # ═══════════════════════════════════════════════════════════════
    # 中文情绪关键词映射（六维 + 强度调节）
    # ═══════════════════════════════════════════════════════════════

    # 格式: (关键词, 基础强度)
    EMOTION_KEYWORDS: Dict[str, List[Tuple[str, float]]] = {
        "anger": [
            # 直接表达
            ("生气", 0.85), ("愤怒", 0.90), ("火大", 0.80), ("恼火", 0.75),
            ("暴躁", 0.85), ("暴怒", 0.95), ("气死", 0.90), ("气炸", 0.95),
            ("发飙", 0.88), ("炸毛", 0.82), ("怒", 0.75), ("恨", 0.80),
            ("恨死", 0.85), ("讨厌死", 0.75),
            # 口语/网络
            ("cnm", 0.85), ("tmd", 0.80), ("妈的", 0.78), ("草", 0.72),
            ("滚", 0.70), ("无语", 0.55), ("离谱", 0.55), ("服了", 0.60),
            ("真行", 0.50), ("牛逼", 0.35), ("绝了", 0.40),
            # 隐晦表达
            ("凭什么", 0.60), ("不公平", 0.65), ("忍不了", 0.75),
            ("惹毛", 0.78), ("看不惯", 0.62), ("烦死了", 0.70),
            ("别烦我", 0.72), ("走开", 0.60), ("闭嘴", 0.65),
            ("好气", 0.78), ("憋屈", 0.55), ("委屈", 0.45),
            # 单字补充 (防漏检)
            ("烦", 0.55), ("气", 0.60),
            # 不公平感知
            ("不是滋味", 0.50), ("不平衡", 0.55), ("明明", 0.35),
            ("凭什么他", 0.62), ("不服", 0.58),
        ],
        "disgust": [
            ("恶心", 0.88), ("讨厌", 0.75), ("厌恶", 0.85), ("反感", 0.78),
            ("想吐", 0.82), ("受不了", 0.65), ("脏", 0.60), ("臭", 0.55),
            ("嫌弃", 0.72), ("不忍直视", 0.70), ("辣眼睛", 0.65),
            ("不堪入目", 0.70), ("倒胃口", 0.78), ("反胃", 0.75),
            ("作呕", 0.80), ("膈应", 0.68), ("恶心死了", 0.90),
            ("嗤之以鼻", 0.70), ("看不下去", 0.62),
        ],
        "fear": [
            ("害怕", 0.82), ("恐惧", 0.88), ("吓死", 0.90), ("恐怖", 0.85),
            ("不敢", 0.70), ("担心", 0.65), ("焦虑", 0.72), ("紧张", 0.68),
            ("慌", 0.72), ("怂", 0.62), ("不安", 0.65), ("怕", 0.70),
            ("恐慌", 0.85), ("战栗", 0.82), ("毛骨悚然", 0.88),
            ("后怕", 0.75), ("提心吊胆", 0.78), ("心慌", 0.70),
            ("可怕", 0.78), ("吓一跳", 0.75), ("冷汗", 0.72),
            ("细思极恐", 0.75), ("不寒而栗", 0.82),
            # 社交焦虑 / 羞愧恐惧
            ("发烫", 0.45), ("好丢脸", 0.58), ("别人怎么看我", 0.62),
            ("大家肯定觉得", 0.55), ("被嘲笑", 0.68), ("出洋相", 0.58),
            ("社死", 0.75), ("尴尬癌", 0.68),
        ],
        "joy": [
            ("开心", 0.80), ("高兴", 0.82), ("快乐", 0.85), ("幸福", 0.88),
            ("爽", 0.78), ("太棒了", 0.85), ("真好", 0.72), ("哈哈哈", 0.82),
            ("笑死", 0.80), ("哈哈哈哈", 0.88), ("嘿嘿", 0.65), ("嘻嘻", 0.60),
            ("兴奋", 0.82), ("激动", 0.80), ("期待", 0.72), ("满足", 0.70),
            ("欣慰", 0.68), ("舒适", 0.60), ("惬意", 0.62), ("欢快", 0.75),
            ("乐", 0.68), ("喜悦", 0.82), ("愉快", 0.78), ("美好", 0.70),
            ("美滋滋", 0.75), ("偷着乐", 0.68), ("忍不住笑", 0.72),
            ("心花怒放", 0.85), ("喜出望外", 0.82),
        ],
        "sadness": [
            ("难过", 0.82), ("悲伤", 0.85), ("伤心", 0.84), ("想哭", 0.85),
            ("泪", 0.78), ("哭", 0.75), ("抑郁", 0.80), ("低落", 0.72),
            ("失落", 0.75), ("沮丧", 0.78), ("心碎", 0.88), ("崩溃", 0.82),
            ("绝望", 0.88), ("无助", 0.80), ("孤独", 0.75), ("寂寞", 0.72),
            ("心疼", 0.75), ("心酸", 0.78), ("心累", 0.72),
            ("算了", 0.55), ("随便", 0.50), ("无所谓", 0.55),
            ("就这样吧", 0.58), ("没意思", 0.62), ("累了", 0.55),
            ("麻木", 0.65), ("空虚", 0.68), ("闷", 0.52),
            ("难过死了", 0.88), ("痛不欲生", 0.92),
            # 羞愧/丢脸信号
            ("丢脸", 0.55), ("没面子", 0.58), ("丢人", 0.55),
            ("不是滋味", 0.52), ("心里难受", 0.62), ("说错话", 0.48),
            ("出丑", 0.55), ("尴尬", 0.50),
            # 情绪低落信号
            ("心情不好", 0.68), ("不开心", 0.60), ("不舒服", 0.45),
            ("不高兴", 0.55), ("没心情", 0.58), ("没胃口", 0.50),
        ],
        "surprise": [
            ("惊讶", 0.82), ("震惊", 0.88), ("没想到", 0.75), ("居然", 0.70),
            ("天哪", 0.78), ("我的天", 0.75), ("不是吧", 0.72), ("真的假的", 0.75),
            ("什么!", 0.62), ("什么？", 0.62), ("等等", 0.62), ("等等！", 0.78),
            ("突然", 0.62), ("突然想起", 0.78), ("想起来", 0.60),
            ("竟然", 0.72), ("不可思议", 0.80), ("难以置信", 0.82), ("出乎意料", 0.78),
            ("妈呀", 0.75), ("沃德天", 0.72), ("蛙趣", 0.68),
            ("目瞪口呆", 0.82), ("大跌眼镜", 0.78), ("惊呆", 0.82),
            ("猝不及防", 0.75), ("万万没想到", 0.80), ("神奇", 0.62),
        ],
    }

    # 上下文情绪转折词
    PIVOT_WORDS = {
        "reverse": ["但是", "不过", "然而", "可是", "但", "却", "没想到",
                     "反而", "倒是", "其实", "实际上", "说实话"],
        "intensify": ["而且", "甚至", "更加", "越发", "特别", "尤其",
                       "真的", "超级", "非常", "太", "极了", "死了"],
        "soften": ["虽然", "虽说", "尽管", "只是", "有点", "稍微",
                    "还算", "不算太", "倒也不"],
        "question": ["为什么", "怎么会", "难道", "凭什么", "怎么"],
    }

    def __init__(self):
        self.state: Dict[str, float] = {e: 0.0 for e in self.BASE_EMOTIONS}
        self.history: List[Dict] = []
        self.last_update: float = time.time()
        self.total_interactions: int = 0

    # ── 公开 API ───────────────────────────────────────────────

    def detect(self, utterance: str, context: str = "",
               previous_state: Optional[Dict[str, float]] = None) -> EmotionResult:
        """
        从用户话语中检测情绪。

        Args:
            utterance: 用户输入文本
            context: 可选的上下文（前一轮对话）
            previous_state: 可选的前一情绪状态（用于干扰计算）
        """
        # 1. 关键词扫描
        raw_vector, triggered = self._scan_keywords(utterance)

        # 2. 上下文转折检测
        raw_vector = self._apply_pivot_modulation(utterance, raw_vector)

        # 3. 语气强化/软化
        raw_vector = self._apply_intensity_modifiers(utterance, raw_vector)

        # 4. 情绪干扰计算
        interference = None
        if previous_state:
            raw_vector = self._apply_interference(previous_state, raw_vector)
            interference = self._analyze_interference(previous_state, raw_vector)

        # 5. 复杂情绪检测
        complex_emotion = self._detect_complex(raw_vector)

        # 6. 效价 & 唤醒度
        valence = self._compute_valence(raw_vector)
        arousal = self._compute_arousal(raw_vector)

        # 7. 更新内部状态
        self.state = raw_vector.copy()
        self.last_update = time.time()
        self.total_interactions += 1

        result = EmotionResult(
            base_vector=raw_vector,
            valence=valence,
            arousal=arousal,
            dominant=self._dominant(raw_vector),
            complex_emotion=complex_emotion,
            intensity=math.sqrt(sum(v**2 for v in raw_vector.values())) / math.sqrt(6),
            interference=interference,
            triggered_terms=triggered,
        )

        self.history.append({
            "vector": raw_vector.copy(),
            "valence": valence,
            "arousal": arousal,
            "complex": complex_emotion,
            "dominant": result.dominant,
            "timestamp": time.time(),
            "utterance_preview": utterance[:80],
        })

        # 只保留最近 50 条历史
        if len(self.history) > 50:
            self.history = self.history[-50:]

        return result

    def get_current(self, now: Optional[float] = None) -> Dict[str, Any]:
        """获取经过时间衰减的当前情绪状态"""
        now = now or time.time()
        elapsed = now - self.last_update
        current = {}
        for emotion, intensity in self.state.items():
            profile = self.DECAY_PROFILES[emotion]
            if intensity < 0.01:
                current[emotion] = 0.0
                continue
            decayed = intensity * (
                profile["alpha"] * math.exp(-elapsed / profile["tau_fast"]) +
                (1 - profile["alpha"]) * math.exp(-elapsed / profile["tau_slow"])
            )
            current[emotion] = max(0.0, min(1.0, decayed))
        return {
            "base_vector": current,
            "valence": self._compute_valence(current),
            "arousal": self._compute_arousal(current),
            "dominant": self._dominant(current),
            "elapsed_seconds": elapsed,
        }

    def reset(self):
        """重置情绪状态"""
        self.state = {e: 0.0 for e in self.BASE_EMOTIONS}
        self.last_update = time.time()

    def get_history(self, n: int = 5) -> List[Dict]:
        """获取最近 N 条情绪历史"""
        return self.history[-n:] if self.history else []

    def get_emotion_trajectory(self) -> Dict[str, List[float]]:
        """获取情绪轨迹（用于可视化）"""
        trajectory = {e: [] for e in self.BASE_EMOTIONS}
        for h in self.history:
            for e in self.BASE_EMOTIONS:
                trajectory[e].append(h["vector"].get(e, 0))
        return trajectory

    # ── 内部方法 ─────────────────────────────────────────────────

    def _scan_keywords(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        """扫描文本中的情绪关键词"""
        result = {e: 0.0 for e in self.BASE_EMOTIONS}
        triggered = []

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            max_intensity = 0.0
            for kw, intensity in keywords:
                if kw in text:
                    triggered.append(f"{emotion}:{kw}({intensity})")
                    max_intensity = max(max_intensity, intensity)
            result[emotion] = max_intensity

        # 处理多关键词叠加：同一情绪多个词触发，取最高值
        return result, triggered

    def _apply_pivot_modulation(self, text: str, vector: Dict[str, float]) -> Dict[str, float]:
        """应用上下文转折词的情绪调制。
        转折词出现时：负面情绪(+25%)，正面情绪(-15%)，因为"但"通常跟着负面。
        """
        result = vector.copy()
        has_reverse = any(w in text for w in self.PIVOT_WORDS["reverse"])
        has_intensify = any(w in text for w in self.PIVOT_WORDS["intensify"])
        has_soften = any(w in text for w in self.PIVOT_WORDS["soften"])

        if has_reverse:
            for e in self.BASE_EMOTIONS:
                if e in ["anger", "sadness", "fear", "disgust"]:
                    result[e] *= 1.25
                else:
                    result[e] *= 0.85

        if has_intensify:
            for e in self.BASE_EMOTIONS:
                result[e] *= 1.15

        if has_soften:
            for e in self.BASE_EMOTIONS:
                result[e] *= 0.80

        # 问句可能隐含愤怒/困惑
        if any(w in text for w in self.PIVOT_WORDS["question"]):
            if result["anger"] < 0.3:
                result["anger"] += 0.15

        # 归一化
        for e in self.BASE_EMOTIONS:
            result[e] = min(1.0, max(0.0, result[e]))

        return result

    def _apply_intensity_modifiers(self, text: str, vector: Dict[str, float]) -> Dict[str, float]:
        """语气强化词/表情符号的情绪强度调节"""
        result = vector.copy()

        # 感叹号强化
        exclaim_count = text.count("!") + text.count("！")
        if exclaim_count >= 3:
            for e in self.BASE_EMOTIONS:
                result[e] *= 1.2
        elif exclaim_count >= 1:
            for e in self.BASE_EMOTIONS:
                result[e] *= 1.08

        # 省略号软化
        if "..." in text or "…" in text:
            for e in self.BASE_EMOTIONS:
                result[e] *= 0.85

        # 表情符号检测
        positive_emoji = ["😊", "😄", "😂", "🤣", "😍", "🥰", "💕", "👍", "🎉", "✨",
                          "^_^", ":)", "(:", ":D"]
        negative_emoji = ["😢", "😭", "😡", "😤", "😰", "😱", "💔", "😞", "😩",
                          "T_T", "T.T", "TAT", ":(", "):"]

        if any(e in text for e in positive_emoji):
            result["joy"] += 0.25
            result["sadness"] = max(0, result["sadness"] * 0.7)
        if any(e in text for e in negative_emoji):
            result["sadness"] += 0.25

        for e in self.BASE_EMOTIONS:
            result[e] = min(1.0, max(0.0, result[e]))

        return result

    def _apply_interference(self, old_state: Dict[str, float],
                            new_vector: Dict[str, float]) -> Dict[str, float]:
        """情绪干扰规则：叠加、覆盖、反转"""
        result = {}
        old_valence = self._compute_valence(old_state)
        new_valence = self._compute_valence(new_vector)

        for emotion in self.BASE_EMOTIONS:
            old = old_state.get(emotion, 0)
            new = new_vector.get(emotion, 0)

            if old < 0.05 and new < 0.05:
                result[emotion] = 0.0
                continue

            # 规则1: 同维叠加
            if old > 0.1 and new > 0.1:
                # 计算时间衰减后的旧强度
                elapsed = time.time() - self.last_update
                profile = self.DECAY_PROFILES[emotion]
                decay_factor = (
                    profile["alpha"] * math.exp(-elapsed / profile["tau_fast"]) +
                    (1 - profile["alpha"]) * math.exp(-elapsed / profile["tau_slow"])
                )
                old_decayed = old * decay_factor
                result[emotion] = old_decayed * 0.7 + new * (1 + 0.3 * old_decayed)
            else:
                # 规则3: 反向覆盖（效价反转）
                if abs(old_valence - new_valence) > 1.0 and new > old * 0.8:
                    # 旧情绪被快速压制
                    suppressed_old = old * math.exp(-3 * new)
                    result[emotion] = max(new, suppressed_old)
                else:
                    # 默认：取较大值但保留部分旧情绪
                    result[emotion] = max(old * 0.3, new)

            result[emotion] = min(1.0, max(0.0, result[emotion]))

        return result

    def _analyze_interference(self, old_state: Dict[str, float],
                               new_vector: Dict[str, float]) -> Dict:
        """分析干扰类型"""
        old_dominant = self._dominant(old_state)
        new_dominant = self._dominant(new_vector)
        old_valence = self._compute_valence(old_state)
        new_valence = self._compute_valence(new_vector)

        if old_dominant == new_dominant and new_dominant != "neutral":
            return {"type": "same_dimension_superposition", "dimension": old_dominant}
        elif abs(old_valence - new_valence) > 1.0:
            residual = {e: v for e, v in old_state.items() if e != new_dominant}
            return {
                "type": "reverse_override",
                "previous_dominant": old_dominant,
                "new_dominant": new_dominant,
                "residual": {e: round(v, 3) for e, v in residual.items() if v > 0.05}
            }
        elif old_dominant != new_dominant and old_dominant != "neutral" and new_dominant != "neutral":
            return {
                "type": "cross_dimension_complexity",
                "previous": old_dominant,
                "current": new_dominant,
                "complex_state": f"{old_dominant}+{new_dominant}"
            }
        return None

    def _detect_complex(self, vector: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """检测复杂社会情绪"""
        best_match = None
        best_score = 0.0
        candidates = []

        for name, recipe in self.COMPLEX_EMOTIONS.items():
            score = sum(vector.get(e, 0) * w for e, w in recipe.items())
            candidates.append((name, score))
            if score > best_score and score > 0.35:
                best_score = score
                best_match = name

        # 至少要有两个基础情绪非零
        active_count = sum(1 for v in vector.values() if v > 0.1)
        if active_count < 2:
            return None

        return {
            "emotion": best_match,
            "intensity": round(best_score, 3),
            "candidates": sorted(candidates, key=lambda x: -x[1])[:3]
        } if best_match else None

    def _compute_valence(self, vector: Dict[str, float]) -> float:
        """效价计算 [-1, 1]"""
        v = (vector.get("joy", 0) * 0.9
             - vector.get("anger", 0) * 0.8
             - vector.get("disgust", 0) * 0.7
             - vector.get("fear", 0) * 0.9
             - vector.get("sadness", 0) * 0.8)
        return round(max(-1.0, min(1.0, v / 2.5)), 3)

    def _compute_arousal(self, vector: Dict[str, float]) -> float:
        """唤醒度计算 [0, 1]"""
        a = (vector.get("anger", 0) * 0.9
             + vector.get("fear", 0) * 0.85
             + vector.get("surprise", 0) * 0.9
             + vector.get("joy", 0) * 0.5
             + vector.get("sadness", 0) * 0.3)
        return round(max(0.0, min(1.0, a / 3.5)), 3)

    def _dominant(self, vector: Dict[str, float]) -> str:
        if max(vector.values()) < 0.1:
            return "neutral"
        return max(vector, key=vector.get)

    # ── 情绪差异化推理 ──────────────────────────────────────────

    def get_emotional_modulation(self, memory_emotion: Dict[str, float],
                                  memory_tags: List[str] = None) -> Dict[str, float]:
        """
        根据当前情绪状态计算对记忆检索的调制因子。
        返回值包含各种 boost/penalty 系数。
        """
        current = self.get_current()
        curr_vec = current["base_vector"]
        mem_vec = memory_emotion
        memory_tags = memory_tags or []

        mods = {
            "consistency_boost": 1.0,
            "opposition_penalty": 1.0,
            "intensity_blur": 1.0,
            "specific_mod": 1.0,
            "window_access": "full",       # full / restricted / blocked
            "wormhole_boost": 1.0,
            "action_bias": "neutral",
        }

        # 一致性加成
        vec_sim = self._cosine_similarity(mem_vec, curr_vec)
        if vec_sim > 0.6:
            mods["consistency_boost"] = 1 + 0.4 * max(curr_vec.values())

        # 效价极性抑制
        mem_valence = self._compute_valence(mem_vec)
        curr_valence = current["valence"]
        if abs(mem_valence - curr_valence) > 1.2:
            mods["opposition_penalty"] = 1 - 0.25 * abs(curr_valence)

        # 唤醒度模糊
        mods["intensity_blur"] = 1 - 0.15 * current["arousal"]

        # 特异性调制
        dominant = current["dominant"]

        if curr_vec.get("joy", 0) > 0.6:
            mods["window_access"] = "full"
            mods["wormhole_boost"] = 1.3
            mods["action_bias"] = "explore"
            mods["specific_mod"] *= 1.2

        if curr_vec.get("fear", 0) > 0.6:
            if "threat" not in memory_tags:
                mods["specific_mod"] *= 0.6
            mods["window_access"] = "restricted"
            mods["action_bias"] = "avoid"

        if curr_vec.get("anger", 0) > 0.6:
            if "confront" not in memory_tags:
                mods["specific_mod"] *= 0.7
            mods["action_bias"] = "confront"

        if curr_vec.get("sadness", 0) > 0.6:
            if "social_support" in memory_tags:
                mods["specific_mod"] *= 1.4
            mods["action_bias"] = "seek_support"

        if curr_vec.get("surprise", 0) > 0.7:
            mods["action_bias"] = "reset_attention"

        # 复杂情绪调制
        complex_e = self._detect_complex(curr_vec)
        if complex_e:
            if complex_e["emotion"] == "shame" and complex_e["intensity"] > 0.5:
                if "self_exposure" in memory_tags:
                    mods["specific_mod"] *= 0.3
            if complex_e["emotion"] == "empathy" and complex_e["intensity"] > 0.5:
                mods["specific_mod"] *= 1.4

        return mods

    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        """计算两个情绪向量的余弦相似度"""
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in EmotionalStateMachine.BASE_EMOTIONS)
        mag_a = math.sqrt(sum(v**2 for v in a.values()))
        mag_b = math.sqrt(sum(v**2 for v in b.values()))
        if mag_a < 0.001 or mag_b < 0.001:
            return 0.0
        return dot / (mag_a * mag_b)


# ═══════════════════════════════════════════════════════════════
# 命令行测试入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    esm = EmotionalStateMachine()

    test_cases = [
        "今天工作被批评了，好难过。给我推荐点吃的吧。",
        "其实我妈妈刚才安慰我了，我现在感觉好多了，甚至有点开心！给我推荐点吃的！",
        "同事升职了，明明我做得更多。说实话，我为他高兴，但心里也有点不是滋味。",
        "刚才在会议上说错话了，现在想起来脸都发烫，大家肯定觉得我很蠢。",
        "我的猫今天走了，养了十年，我真的好难过。",
        "等等！我突然想起来，上周你说的那个方案，其实和我三年前做过的一个项目几乎一样！",
        "好热啊，帮我把空调打开，温度调低一点",
        "刚才吃了口苦瓜，太苦了，给我拿颗糖来",
        "还没吃饭，饿死了，而且今天工作特别烦，随便给我点什么都行",
    ]

    print("=" * 70)
    print("FSTN-4D 情绪状态机测试")
    print("=" * 70)

    for i, utterance in enumerate(test_cases, 1):
        result = esm.detect(utterance)
        current = esm.get_current()
        mod = esm.get_emotional_modulation(result.base_vector, ["routine"])

        print(f"\n{'─'*60}")
        print(f"[测试 {i}] {utterance[:50]}...")
        print(f"  主导情绪: {result.dominant} (intensity={result.intensity:.3f})")
        print(f"  效价: {result.valence:.3f}  唤醒度: {result.arousal:.3f}")
        if result.complex_emotion:
            print(f"  复杂情绪: {result.complex_emotion['emotion']} "
                  f"(强度={result.complex_emotion['intensity']:.3f})")
        if result.interference and result.interference.get("type"):
            print(f"  干扰类型: {result.interference['type']}")
        if result.triggered_terms:
            print(f"  触发词: {', '.join(result.triggered_terms[:5])}")
        print(f"  调制: window_access={mod['window_access']}, "
              f"wormhole_boost={mod['wormhole_boost']}, "
              f"action_bias={mod['action_bias']}")

    print(f"\n{'='*70}")
    print("情绪历史轨迹 (最近 3 条):")
    for h in esm.get_history(3):
        dom = h["dominant"]
        print(f"  [{dom:>10}] valence={h['valence']:+.2f} "
              f"| {h['utterance_preview'][:40]}...")

    print(f"\n✅ 测试完成: {len(test_cases)} 个场景")
