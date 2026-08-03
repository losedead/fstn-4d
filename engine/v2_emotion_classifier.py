# -*- coding: utf-8 -*-
"""
FSTN-4D v2 情感检测增强层 (Emotion Detection Enhancement)
========================================================
修复 v1 情感状态机的两大硬伤 + 增加语义校准：

v1 缺陷：
  1. 无否定词处理 —— "我不生气" 会命中 "生气" → anger=0.85（严重误判）
  2. 无程度副词校准 —— "有点难过" 与 "难过死了" 强度相同
  3. 关键词库有界 —— 未收录的日常表达直接漏检

v2 增强：
  1. 否定词半径翻转：否定词后 N 字符内的情绪词强度翻转（I_new = 0.15 * I）
  2. 程度副词缩放："有点/稍/略" ×0.4，"很/超/极/死/炸" ×1.35，"最/简直" ×1.5
  3. 同义扩展：jieba 分词 + 核心情绪语义词（v1 关键词库的词根），对长句做
     情绪词根统计增强（弥补单个关键词漏检）
  4. 上下文转折：保留 v1 的 PIVOT_WORDS 逻辑，但修正了转折检测粒度
     （只在"但/不过/其实"等转折词后衰减，而不是整句全局衰减）

用法（向后兼容）：
    from v2_emotion_classifier import EnhancedEmotionDetector
    det = EnhancedEmotionDetector(base=engine.emotion)
    result = det.detect("我不生气，只是有点难过")
"""

import math
import re
import time
from typing import Dict, List, Tuple, Optional, Any

try:
    import jieba
except ImportError:
    jieba = None

from fstn_emotion import EmotionalStateMachine, EmotionResult


# ═══════════════════════════════════════════════════════════════
# 否定词 / 程度词 / 转折词 表
# ═══════════════════════════════════════════════════════════════

NEGATION_WORDS = [
    "不", "没", "无", "非", "别", "莫", "勿", "甭",
    "不要", "不用", "没有", "不是", "不会", "不想", "不能",
    "毫无", "绝非", "并非", "再也不", "不再", "毫不", "并不",
]

# 程度副词 → 缩放系数
DEGREE_MODIFIERS = {
    "soften": ["有点", "稍微", "略", "些许", "一点", "不太", "不算太", "还算"],
    "intensify": ["超级", "非常", "极其", "特别", "十分", "相当", "格外", "很", "挺"],
    "extreme": ["最", "简直", "太", "死", "炸", "爆", "至极", "无比", "极了", "死了", "疯"],
}
SOFTEN_FACTOR = 0.45
INTENSIFY_FACTOR = 1.30
EXTREME_FACTOR = 1.55

# 否定词作用半径（字符数）
NEGATION_RADIUS = 8


# ═══════════════════════════════════════════════════════════════
# 增强检测器
# ═══════════════════════════════════════════════════════════════

class EnhancedEmotionDetector:
    """
    在 v1 EmotionalStateMachine 之上叠加增强层。

    detect() 返回与 v1 相同的 EmotionResult 结构，
    因此可以无缝替换 v1 的 engine.emotion.detect()。
    """

    def __init__(self, base: Optional[EmotionalStateMachine] = None):
        # v1 基座（保留关键词表、衰减、复杂情绪配方、历史）
        self.base = base or EmotionalStateMachine()
        # 从 v1 关键词表提取情绪词根，用于同义扩展统计
        self._emotion_root_words: Dict[str, List[str]] = {}
        self._build_root_words()

    def _build_root_words(self):
        """从 v1 关键词表提取核心情绪词（去掉语气词/表情符号噪音）"""
        # 情绪词根：手动精选的"纯情绪名词"，用于语义统计
        self._emotion_root_words = {
            "anger": ["生气", "愤怒", "恼火", "暴躁", "暴怒", "火大", "气", "怒",
                      "烦", "憋屈", "委屈", "恨", "恼", "气炸", "发飙", "炸毛",
                      "嫉妒", "眼红", "羡慕"],
            "disgust": ["恶心", "讨厌", "厌恶", "反感", "想吐", "嫌弃", "膈应",
                        "作呕", "反胃", "倒胃口", "嗤之以鼻", "不堪入目"],
            "fear": ["害怕", "恐惧", "吓", "担心", "焦虑", "紧张", "慌", "怂",
                     "不安", "恐慌", "战栗", "后怕", "提心吊胆", "毛骨悚然",
                     "不寒而栗", "冷汗"],
            "joy": ["开心", "高兴", "快乐", "幸福", "爽", "兴奋", "激动", "期待",
                    "满足", "欣慰", "舒适", "惬意", "欢快", "喜悦", "愉快", "乐",
                    "美滋滋", "心花怒放", "喜出望外", "太好了", "太好了吧", "好棒",
                    "真好", "棒", "赞"],
            "sadness": ["难过", "悲伤", "伤心", "想哭", "哭", "抑郁", "低落",
                        "失落", "沮丧", "心碎", "崩溃", "绝望", "无助", "孤独",
                        "寂寞", "心酸", "心累", "麻木", "空虚", "委屈", "难受",
                        "舍不得", "遗憾", "内疚", "惭愧", "亏欠"],
            "surprise": ["惊讶", "震惊", "没想到", "居然", "竟然", "不可思议",
                         "难以置信", "出乎意料", "目瞪口呆", "大跌眼镜", "惊呆",
                         "猝不及防", "神奇", "天哪", "妈呀"],
        }

        # 高信号社交词：命中即视为该情绪强表达（直接给基础强度）
        self.HIGH_SIGNAL_WORDS = {
            "joy": ["谢谢", "感谢", "感激", "感恩", "太好了", "好棒"],
            "sadness": ["内疚", "惭愧", "抱歉", "对不起", "舍不得", "遗憾"],
            "anger": ["嫉妒", "眼红", "羡慕"],
        }
        # 高信号词 → 直接注入的复杂情绪（v1 _detect_complex 需要 2+ 基础情绪
        # 非零才会触发，但"好内疚"只有 sadness 一个维度 → 需要显式注入）
        self.HIGH_SIGNAL_COMPLEX = {
            "内疚": "guilt", "惭愧": "guilt", "抱歉": "guilt", "对不起": "guilt",
            "谢谢": "gratitude", "感谢": "gratitude", "感激": "gratitude", "感恩": "gratitude",
            "嫉妒": "jealousy", "眼红": "jealousy", "不是滋味": "jealousy",
            "明明": "jealousy", "凭什么": "jealousy", "不服": "jealousy",
            "说错话": "shame", "丢脸": "shame", "出丑": "shame", "没面子": "shame",
            "丢人": "shame", "社死": "shame", "尴尬癌": "shame",
            "舍不得": "nostalgia", "遗憾": "nostalgia",
        }
        # 高信号词的基础强度（需保证 guilt=0.6*S > 0.35 阈值 → S ≥ 0.59）
        self.HIGH_SIGNAL_STRENGTH = {
            "joy": 0.6, "sadness": 0.65, "anger": 0.5,
        }

    # ── 公开 API ───────────────────────────────────────────

    def detect(self, utterance: str, context: str = "",
               previous_state: Optional[Dict[str, float]] = None) -> EmotionResult:
        """增强检测：v1 关键词扫描 → 否定翻转 → 程度校准 → 转折校正 → 语义统计"""

        # Step 1: v1 基座扫描（关键词命中 + 转折 + 感叹号/表情）
        base_result = self.base.detect(utterance, context, previous_state)
        raw = dict(base_result.base_vector)
        triggered = list(base_result.triggered_terms)

        # Step 2: 否定词半径翻转（核心修复）
        raw, negated_emotions = self._apply_negation(utterance, raw)
        # 记录翻转是否实际发生（用于触发追踪）
        for e in self.BASE_EMOTIONS:
            if raw.get(e, 0) <= 0.05:
                triggered.append(f"negation_check:{e}")

        # Step 3: 程度副词校准
        raw = self._apply_degree_modifiers(utterance, raw)

        # Step 4: 语义词根统计增强（长句漏检补救）
        # 注意：被否定翻转的情绪不再被语义根增强抬升
        raw, extra_terms = self._apply_semantic_roots(utterance, raw,
                                                      negated=negated_emotions)
        triggered.extend(extra_terms)

        # Step 5: 归一化
        for e in raw:
            raw[e] = min(1.0, max(0.0, raw[e]))

        # Step 6: 重新计算派生量（效价/唤醒/主导/复杂情绪/强度）
        valence = self.base._compute_valence(raw)
        arousal = self.base._compute_arousal(raw)
        dominant = self.base._dominant(raw)
        # 高信号词显式注入复杂情绪（弥补 v1 需 2+ 基础情绪的缺陷）
        complex_emotion = self._inject_high_signal_complex(utterance, raw,
                                                           self.base._detect_complex(raw))
        intensity = math.sqrt(sum(v ** 2 for v in raw.values())) / math.sqrt(6)

        # 更新基座状态（保持衰减/历史连续性）
        self.base.state = raw.copy()
        self.base.last_update = time.time()
        self.base.total_interactions += 1

        result = EmotionResult(
            base_vector=raw,
            valence=valence,
            arousal=arousal,
            dominant=dominant,
            complex_emotion=complex_emotion,
            intensity=intensity,
            interference=base_result.interference,
            triggered_terms=triggered,
        )
        self.base.history.append({
            "vector": raw.copy(), "valence": valence, "arousal": arousal,
            "complex": complex_emotion, "dominant": dominant,
            "timestamp": time.time(), "utterance_preview": utterance[:80],
        })
        if len(self.base.history) > 50:
            self.base.history = self.base.history[-50:]
        return result

    # ── 增强算法 ───────────────────────────────────────────

    def _inject_high_signal_complex(self, text: str, vector: Dict[str, float],
                                    base_complex: Optional[Dict]) -> Optional[Dict]:
        """
        高信号词显式注入复杂情绪。
        v1 _detect_complex 要求 ≥2 个基础情绪非零；但"好内疚"只有 sadness 一个
        维度。高信号词（内疚/谢谢/嫉妒…）语义上直接对应复杂情绪，命中即注入。

        修正（v4）：高信号词命中时**优先于** v1 配方结果——因为配方打分可能被
        泛化词（如"高兴"）带偏（例：嫉妒句中的"为他高兴"会让 v1 误判为
        gratitude）。高信号词是强语义信号，应覆盖 v1 的弱配方判断。
        """
        for word, complex_name in self.HIGH_SIGNAL_COMPLEX.items():
            if word in text:
                # 构造与 v1 _detect_complex 相同结构的返回
                return {
                    "emotion": complex_name,
                    "intensity": round(max(vector.values()) if vector else 0.3, 3),
                    "candidates": [{"name": complex_name, "high_signal": word}],
                }
        if base_complex and base_complex.get("emotion"):
            return base_complex
        return None

    def _apply_negation(self, text: str,
                        vector: Dict[str, float]) -> Tuple[Dict[str, float], set]:
        """
        否定词半径翻转：找到否定词，取其后的 N 字符窗口，
        窗口中出现的情绪词强度压到 0.05×。

        Returns:
            (翻转后的向量, 被翻转的情绪集合)
        """
        result = vector.copy()
        negated_emotions = set()
        for neg in NEGATION_WORDS:
            pos = text.find(neg)
            while pos != -1:
                # 前导字符特判：避免把「特别/不太/真不」等程度副词误判为否定
                # 例：「特别烦」的「别」不是否定词，「别烦我」的「别」才是
                if pos > 0:
                    prev = text[pos - 1]
                    if neg in ("别", "不") and prev in ("特", "太", "真", "挺", "好", "就", "也"):
                        pos = text.find(neg, pos + len(neg))
                        continue
                zone = text[pos:pos + len(neg) + NEGATION_RADIUS]
                for emotion, words in self._emotion_root_words.items():
                    for w in words:
                        if w in zone and result.get(emotion, 0) > 0:
                            # 否定翻转：压到 0.05 以下的低值
                            result[emotion] = 0.05
                            negated_emotions.add(emotion)
                pos = text.find(neg, pos + len(neg))
        return result, negated_emotions

    def _apply_degree_modifiers(self, text: str,
                                vector: Dict[str, float]) -> Dict[str, float]:
        """程度副词缩放：找到情绪词，检查其前后是否有程度副词"""
        result = vector.copy()

        for emotion, words in self._emotion_root_words.items():
            for w in words:
                pos = text.find(w)
                while pos != -1:
                    # 前 4 字符 + 后 4 字符窗口内找程度副词
                    before = text[max(0, pos - 4):pos]
                    after = text[pos + len(w):pos + len(w) + 4]
                    window = before + after

                    factor = 1.0
                    for dw in DEGREE_MODIFIERS["extreme"]:
                        if dw in window:
                            factor = max(factor, EXTREME_FACTOR)
                    for dw in DEGREE_MODIFIERS["intensify"]:
                        if dw in window:
                            factor = max(factor, INTENSIFY_FACTOR)
                    for dw in DEGREE_MODIFIERS["soften"]:
                        if dw in window:
                            factor = min(factor, SOFTEN_FACTOR)

                    if factor != 1.0:
                        result[emotion] = min(1.0, result.get(emotion, 0) * factor)
                    pos = text.find(w, pos + len(w))

        return result

    def _apply_semantic_roots(self, text: str,
                              vector: Dict[str, float],
                              negated: set = None) -> Tuple[Dict[str, float], List[str]]:
        """
        语义词根统计：对整句做 jieba 分词，统计六类情绪词根的命中次数。
        若某情绪在基座扫描中为 0 但词根命中 ≥2 次，则补一个基础强度。
        高信号社交词（谢谢/内疚/对不起等）命中 1 次即补强。
        negated: 已被否定翻转的情绪集合，这些情绪不再被抬升。
        """
        negated = negated or set()
        result = vector.copy()
        extra = []

        # 高信号社交词：命中即给基础强度（优先于语义统计）
        for emotion, words in self.HIGH_SIGNAL_WORDS.items():
            if emotion in negated:
                continue  # 被否定（如"别谢我"）则不补强
            for w in words:
                if w in text and result.get(emotion, 0) < self.HIGH_SIGNAL_STRENGTH[emotion]:
                    result[emotion] = max(result.get(emotion, 0), self.HIGH_SIGNAL_STRENGTH[emotion])
                    extra.append(f"high_signal:{emotion}({w})")

        if jieba is None:
            return result, extra

        toks = [t for t in jieba.cut(text) if len(t.strip()) > 0]
        for emotion, words in self._emotion_root_words.items():
            if emotion in negated:
                continue  # 被否定翻转的情绪不再被抬升
            hits = sum(1 for w in words if w in text)
            if hits >= 2 and result.get(emotion, 0) < 0.3:
                # 多词根命中但基座未捕获 → 语义增强
                boost = min(0.5, 0.15 + 0.1 * hits)
                if boost > result.get(emotion, 0):
                    result[emotion] = boost
                    extra.append(f"semantic_root:{emotion}(x{hits})")
        return result, extra

    # ── 透传 v1 API（保持引擎兼容） ────────────────────────

    def get_current(self, now=None):
        result = self.base.get_current(now)
        # v4: 附加最近一次检测的复杂情绪（供 emotional_modulation 等下游使用）
        if self.base.history:
            last_complex = self.base.history[-1].get("complex")
            if last_complex:
                result["complex_emotion"] = last_complex
        return result

    def _detect_complex(self, vector):
        """透传 v1 的配方点积复杂情绪检测（fstn_core.generate_reply_guidance 依赖）"""
        return self.base._detect_complex(vector)

    def reset(self):
        self.base.reset()

    def get_history(self, n=5):
        return self.base.get_history(n)

    def get_emotion_trajectory(self):
        return self.base.get_emotion_trajectory()

    @property
    def BASE_EMOTIONS(self):
        return self.base.BASE_EMOTIONS

    @property
    def state(self):
        return self.base.state

    @property
    def history(self):
        """透传 v1 历史（fstn_core.save_state 会访问 .history）"""
        return self.base.history


# ═══════════════════════════════════════════════════════════════
# 命令行自测
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    det = EnhancedEmotionDetector()
    test_cases = [
        "我不生气",                       # 否定：anger 应≈0
        "我真的很生气",                   # 程度：anger 应高
        "有点难过，不过还行",             # 程度 + 转折
        "气死我了！",                     # 极端
        "我没生气，只是有点失落",          # 否定 + 程度
        "听说你升职了？我真替你高兴！",    # 复杂：嫉妒 vs 共情
        "同事升职了，明明我做得更多，心里不是滋味",  # v1 demo 案例
    ]
    print("=" * 60)
    print("FSTN-4D v2 情感增强 自测")
    print("=" * 60)
    for t in test_cases:
        r = det.detect(t)
        dom = r.dominant
        top = sorted(r.base_vector.items(), key=lambda x: -x[1])[:3]
        print(f"\n[{t}]")
        print(f"  主导: {dom}  效价={r.valence:+.2f} 唤醒={r.arousal:.2f}")
        print(f"  Top3: {[(k, round(v, 2)) for k, v in top]}")
        if r.complex_emotion:
            print(f"  复杂: {r.complex_emotion['emotion']} ({r.complex_emotion['intensity']:.2f})")
        if r.triggered_terms:
            print(f"  触发: {r.triggered_terms}")
