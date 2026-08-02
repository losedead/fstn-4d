"""
FSTN-4D 感知状态机 (Perceptual State Machine)
===============================================
七维感知状态追踪 + 感知-情绪耦合矩阵 + 情绪反向调制感知。
支持：话语提取感知更新、感知直接行为识别、跨模态通感联想。

创新点（超越原 spec）：
1. 感知状态自然衰减（长时间未提及的感知维度自动回落基线）
2. 内感受(interoceptive)与情绪的双向耦合增强——饥饿/疲劳/口渴不仅是感知，
   也是情绪前兆，实现"饿怒"(hangry)检测
3. 通感质量因子 (sharpness/warmth/heaviness/freshness) 跨模态映射
4. 感知历史回溯——可查询"用户上一次感到热是什么时候"
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class PerceptualState:
    """七维感知状态快照"""
    # 体感 (thermal)
    thermal_comfort: float = 0.0    # -1(极冷) ~ 0(舒适) ~ +1(极热)
    sweating: bool = False
    shivering: bool = False

    # 触觉 (tactile)
    tactile_pressure: float = 0.0   # 0(无接触) ~ 1(强压)
    tactile_texture: str = ""
    tactile_pain: float = 0.0       # 0 ~ 1
    tactile_itch: float = 0.0

    # 味觉 (gustatory)
    gust_sweet: float = 0.0
    gust_sour: float = 0.0
    gust_salty: float = 0.0
    gust_bitter: float = 0.0
    gust_umami: float = 0.0

    # 嗅觉 (olfactory)
    olf_pleasant: float = 0.0       # -1(恶臭) ~ 0(中性) ~ +1(芳香)
    olf_intensity: float = 0.0
    olf_familiarity: float = 0.0
    olf_triggers: List[str] = field(default_factory=list)

    # 视觉 (visual)
    vis_brightness: float = 0.5     # 0(全暗) ~ 1(刺眼)
    vis_color_temp: float = 5000    # 色温(K)
    vis_clutter: float = 0.0
    vis_nature_ratio: float = 0.3

    # 听觉 (auditory)
    aud_loudness: float = 0.3       # 0(静音) ~ 1(震耳)
    aud_pitch: float = 0.5
    aud_rhythm: float = 0.0
    aud_speech_ratio: float = 0.5

    # 内感受 (interoceptive)
    int_hunger: float = 0.0         # 0(饱) ~ 1(极度饥饿)
    int_thirst: float = 0.0
    int_fatigue: float = 0.0
    int_nausea: float = 0.0
    int_fullness: float = 0.5       # 0(空) ~ 1(撑)

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "thermal": {"thermal_comfort": self.thermal_comfort,
                        "sweating": self.sweating, "shivering": self.shivering},
            "tactile": {"pressure": self.tactile_pressure, "texture": self.tactile_texture,
                        "pain": self.tactile_pain, "itch": self.tactile_itch},
            "gustatory": {"sweet": self.gust_sweet, "sour": self.gust_sour,
                          "salty": self.gust_salty, "bitter": self.gust_bitter,
                          "umami": self.gust_umami},
            "olfactory": {"pleasant": self.olf_pleasant, "intensity": self.olf_intensity,
                          "familiarity": self.olf_familiarity, "triggers": self.olf_triggers},
            "visual": {"brightness": self.vis_brightness, "color_temp": self.vis_color_temp,
                       "clutter": self.vis_clutter, "nature_ratio": self.vis_nature_ratio},
            "auditory": {"loudness": self.aud_loudness, "pitch": self.aud_pitch,
                         "rhythm": self.aud_rhythm, "speech_ratio": self.aud_speech_ratio},
            "interoceptive": {"hunger": self.int_hunger, "thirst": self.int_thirst,
                              "fatigue": self.int_fatigue, "nausea": self.int_nausea,
                              "fullness": self.int_fullness},
        }


# ═══════════════════════════════════════════════════════════════
# 核心引擎
# ═══════════════════════════════════════════════════════════════

class PerceptualStateMachine:
    """FSTN-4D 感知状态机"""

    SENSES = ["thermal", "tactile", "gustatory", "olfactory",
              "visual", "auditory", "interoceptive"]

    # 感知衰减半衰期（秒）——长时间未提及的感知自动回落
    SENSE_DECAY_HALFLIFE = {
        "thermal": 1800,        # 30分钟
        "tactile": 600,          # 10分钟（疼痛消退快）
        "gustatory": 900,        # 15分钟（味觉余韵）
        "olfactory": 300,        # 5分钟（嗅觉适应快）
        "visual": 3600,          # 1小时
        "auditory": 1800,        # 30分钟
        "interoceptive": 3600,   # 1小时（饥饿/疲劳变化慢）
    }

    # 中文感知关键词映射
    PERCEPTION_KEYWORDS = {
        "thermal": {
            "hot": ["热", "出汗", "闷", "烤", "烫", "炎热", "闷热", "火炉", "暴晒", "中暑",
                     "开空调", "风扇", "降温", "凉快一下"],
            "cold": ["冷", "冻", "凉", "发抖", "哆嗦", "寒风", "冰", "雪", "颤抖",
                      "加衣服", "取暖", "暖气", "被窝"],
            "comfort": ["温暖", "舒适", "暖和", "凉爽", "宜人"],
        },
        "gustatory": {
            "bitter": ["苦", "苦瓜", "中药", "黄连"],
            "sweet": ["甜", "好吃", "美味", "糖果", "巧克力", "蛋糕", "冰淇淋",
                       "甜品", "甜食", "蜜", "糖"],
            "sour": ["酸", "柠檬", "醋", "酸味"],
            "salty": ["咸", "盐", "齁"],
            "spicy": ["辣", "麻", "辣椒", "花椒", "麻辣"],
            "umami": ["鲜", "鲜美", "高汤", "味精"],
        },
        "interoceptive": {
            "hungry": ["饿", "肚子叫", "想吃东西", "饥", "空腹", "饿死", "饿坏"],
            "thirsty": ["渴", "口干", "想喝水", "缺水", "干燥"],
            "tired": ["累", "困", "没劲", "疲劳", "倦", "乏", "打哈欠",
                       "不想动", "疲惫", "没精神"],
            "full": ["饱", "撑", "吃不下了", "吃饱"],
            "nauseous": ["恶心", "想吐", "反胃", "呕吐"],
        },
        "tactile": {
            "pain": ["疼", "痛", "不舒服", "受伤", "伤口", "流血", "刺痛", "酸疼",
                      "擦破", "撞", "摔", "扭"],
            "itch": ["痒", "挠", "抓"],
            "soft": ["软", "柔", "滑", "细腻", "绵", "舒服的触感"],
            "rough": ["粗糙", "硌", "磨", "硬邦邦"],
            "pressure": ["压", "挤", "紧绷", "压迫"],
        },
        "auditory": {
            "noisy": ["吵", "噪音", "喧闹", "嘈杂", "震耳", "轰鸣", "电钻", "施工",
                       "大声", "闹", "烦死了(声音)"],
            "quiet": ["安静", "静", "没声音", "寂静", "悄无声息", "轻声"],
            "rhythmic": ["节奏", "节拍", "鼓点", "律动", "循环", "心跳声"],
            "music": ["音乐", "歌", "曲子", "旋律", "和弦"],
        },
        "visual": {
            "bright": ["亮", "刺眼", "阳光", "灯光", "明亮", "白", "闪", "耀眼"],
            "dark": ["暗", "黑", "看不清", "昏暗", "漆黑", "阴影", "阴暗"],
            "colorful": ["彩色", "缤纷", "鲜艳", "绚丽"],
            "cluttered": ["乱", "堆满", "杂乱", "凌乱"],
        },
        "olfactory": {
            "fragrant": ["香", "芳香", "花香", "咖啡香", "香水", "好闻", "清香",
                          "美味的气味", "芬芳"],
            "foul": ["臭", "难闻", "恶臭", "腐烂", "酸臭", "刺鼻", "异味", "臭味"],
            "smoky": ["烟味", "焦味", "烧焦", "烟火"],
        },
    }

    # 感知-情绪耦合矩阵
    COUPLING_RULES = {
        ("thermal", "too_hot"): {"anger": 0.4, "disgust": 0.2},
        ("thermal", "too_cold"): {"sadness": 0.3, "fear": 0.2},
        ("thermal", "comfortable"): {"joy": 0.3},
        ("gustatory", "bitter"): {"disgust": 0.6, "sadness": 0.2},
        ("gustatory", "sweet"): {"joy": 0.5},
        ("gustatory", "spicy"): {"surprise": 0.3, "anger": 0.2},
        ("interoceptive", "hungry"): {"anger": 0.3, "sadness": 0.2},
        ("interoceptive", "thirsty"): {"fear": 0.2, "anger": 0.3},
        ("interoceptive", "tired"): {"sadness": 0.4, "disgust": 0.2},
        ("tactile", "pain"): {"fear": 0.4, "anger": 0.3},
        ("tactile", "severe_pain"): {"fear": 0.7, "sadness": 0.3},
        ("auditory", "noisy"): {"anger": 0.5, "fear": 0.2},
        ("auditory", "quiet"): {"joy": 0.2},
        ("visual", "dark"): {"fear": 0.3, "sadness": 0.2},
        ("visual", "bright"): {"anger": 0.2, "surprise": 0.1},
        ("olfactory", "foul"): {"disgust": 0.7, "anger": 0.2},
        ("olfactory", "fragrant"): {"joy": 0.4},
    }

    # 情绪→感知反向调制
    EMOTIONAL_PERCEPTION_MODULATION = {
        "joy": {
            "pain_threshold_multiplier": 1.5,
            "fatigue_perceived_reduction": 0.4,
            "hunger_perceived_reduction": 0.3,
        },
        "anger": {
            "pain_threshold_multiplier": 1.3,
            "auditory_sensitivity_boost": 1.4,
        },
        "fear": {
            "auditory_sensitivity_boost": 1.5,
            "visual_sensitivity_boost": 1.3,
            "pain_threshold_multiplier": 0.8,
        },
        "sadness": {
            "gustatory_sensitivity_reduction": 0.5,
            "fatigue_perceived_boost": 0.3,
        },
        "disgust": {
            "gustatory_sensitivity_boost": 1.6,
            "olfactory_sensitivity_boost": 1.5,
        },
        "surprise": {
            "auditory_sensitivity_boost": 1.8,
            "visual_sensitivity_boost": 1.6,
        },
    }

    # 通感质量因子
    SYNESTHESIA_QUALITIES = {
        "sharpness": {
            "auditory": {"condition": lambda s: s.aud_pitch > 0.7 and s.aud_loudness > 0.6},
            "tactile": {"condition": lambda s: s.tactile_pain > 0.5},
            "gustatory": {"condition": lambda s: s.gust_bitter > 0.6 or (s.gust_bitter + s.gust_sour) > 0.8},
            "visual": {"condition": lambda s: s.vis_brightness > 0.8},
            "associated_emotion": {"surprise": 0.4, "fear": 0.3},
        },
        "warmth": {
            "thermal": {"condition": lambda s: -0.3 < s.thermal_comfort < 0.3},
            "visual": {"condition": lambda s: s.vis_color_temp < 3500},
            "olfactory": {"condition": lambda s: s.olf_pleasant > 0.5},
            "tactile": {"condition": lambda s: "软" in s.tactile_texture or "柔" in s.tactile_texture},
            "associated_emotion": {"joy": 0.4},
        },
        "heaviness": {
            "tactile": {"condition": lambda s: s.tactile_pressure > 0.6},
            "auditory": {"condition": lambda s: s.aud_pitch < 0.3 and s.aud_loudness > 0.5},
            "visual": {"condition": lambda s: s.vis_brightness < 0.3},
            "associated_emotion": {"sadness": 0.5, "fear": 0.2},
        },
        "freshness": {
            "olfactory": {"condition": lambda s: s.olf_pleasant > 0.6},
            "gustatory": {"condition": lambda s: 0.3 < s.gust_sour < 0.6 and s.gust_sweet > 0.3},
            "visual": {"condition": lambda s: s.vis_brightness > 0.6 and s.vis_nature_ratio > 0.5},
            "associated_emotion": {"joy": 0.3, "surprise": 0.2},
        },
    }

    def __init__(self):
        self.state = PerceptualState()
        self.history: List[Dict] = []
        self.dominant_sense: Optional[str] = None

    # ── 公开 API ─────────────────────────────────────────────────

    def update_from_utterance(self, utterance: str) -> Dict[str, Any]:
        """从用户话语中提取感知线索并更新状态"""
        self._apply_natural_decay()
        updates = {}

        # 体感（文档 §5.2 伪代码 + §2.4.1 示例：热=负舒适，冷=正舒适）
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["thermal"]["hot"]):
            self.state.thermal_comfort = -0.7
            self.state.sweating = True
            self.state.shivering = False
            updates["thermal"] = {"thermal_comfort": -0.7, "sweating": True}
        elif any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["thermal"]["cold"]):
            self.state.thermal_comfort = 0.7
            self.state.sweating = False
            self.state.shivering = True
            updates["thermal"] = {"thermal_comfort": 0.7, "shivering": True}
        elif any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["thermal"]["comfort"]):
            self.state.thermal_comfort = 0.0
            self.state.sweating = False
            self.state.shivering = False
            updates["thermal"] = {"thermal_comfort": 0.0}

        # 味觉
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["gustatory"]["bitter"]):
            self.state.gust_bitter = 0.8
            updates["gustatory"] = {"bitter": 0.8}
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["gustatory"]["sweet"]):
            self.state.gust_sweet = 0.7
            updates.setdefault("gustatory", {})["sweet"] = 0.7
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["gustatory"]["spicy"]):
            self.state.tactile_pain = 0.5  # 辣本质是痛觉
            updates.setdefault("gustatory", {})["spicy"] = 0.5
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["gustatory"]["sour"]):
            self.state.gust_sour = 0.7
            updates.setdefault("gustatory", {})["sour"] = 0.7

        # 内感受
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["interoceptive"]["hungry"]):
            self.state.int_hunger = 0.8
            updates["interoceptive"] = {"hunger": 0.8}
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["interoceptive"]["thirsty"]):
            self.state.int_thirst = 0.8
            updates.setdefault("interoceptive", {})["thirst"] = 0.8
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["interoceptive"]["tired"]):
            self.state.int_fatigue = 0.7
            updates.setdefault("interoceptive", {})["fatigue"] = 0.7
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["interoceptive"]["nauseous"]):
            self.state.int_nausea = 0.7
            updates.setdefault("interoceptive", {})["nausea"] = 0.7

        # 触觉
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["tactile"]["pain"]):
            # 根据语气判断严重程度
            severity = 0.8 if any(w in utterance for w in ["剧痛", "钻心", "疼死"]) else 0.5
            self.state.tactile_pain = severity
            updates["tactile"] = {"pain": severity}
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["tactile"]["soft"]):
            self.state.tactile_texture = "soft"
            updates.setdefault("tactile", {})["texture"] = "soft"

        # 听觉
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["auditory"]["noisy"]):
            self.state.aud_loudness = 0.8
            updates["auditory"] = {"loudness": 0.8}
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["auditory"]["quiet"]):
            self.state.aud_loudness = 0.1
            updates.setdefault("auditory", {})["loudness"] = 0.1

        # 视觉
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["visual"]["bright"]):
            self.state.vis_brightness = 0.9
            updates["visual"] = {"brightness": 0.9}
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["visual"]["dark"]):
            self.state.vis_brightness = 0.1
            updates.setdefault("visual", {})["brightness"] = 0.1

        # 嗅觉
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["olfactory"]["fragrant"]):
            self.state.olf_pleasant = 0.7
            updates["olfactory"] = {"pleasant": 0.7}
        if any(kw in utterance for kw in self.PERCEPTION_KEYWORDS["olfactory"]["foul"]):
            self.state.olf_pleasant = -0.7
            updates.setdefault("olfactory", {})["pleasant"] = -0.7

        self.state.timestamp = time.time()

        # 更新主导感知
        if updates:
            self.dominant_sense = self._get_dominant_sense()
            self.history.append({
                "updates": updates,
                "dominant_sense": self.dominant_sense,
                "timestamp": time.time(),
            })
            if len(self.history) > 100:
                self.history = self.history[-50:]

        return updates

    def get_current(self) -> Dict:
        """获取当前感知状态（带自然衰减）"""
        self._apply_natural_decay()
        return self.state.to_dict()

    def get_dominant(self) -> Tuple[Optional[str], float]:
        """获取当前主导感知维度"""
        sense = self._get_dominant_sense()
        score = self._get_dominant_score()
        return sense, score

    def get_active_coupling_states(self) -> List[Tuple[str, str]]:
        """
        返回当前激活的感知-情绪耦合状态对列表。
        （V3 新增：供自适应耦合矩阵使用，判定逻辑与 couple_emotion 一致）
        例: [("thermal", "too_hot"), ("interoceptive", "hungry")]
        """
        active = []
        s = self.state
        # 体感（热=负舒适，冷=正舒适，与 update_from_utterance 一致）
        if s.thermal_comfort < -0.5:
            active.append(("thermal", "too_hot"))
        elif s.thermal_comfort > 0.5:
            active.append(("thermal", "too_cold"))
        elif abs(s.thermal_comfort) < 0.2:
            active.append(("thermal", "comfortable"))
        # 味觉
        if s.gust_bitter > 0.6:
            active.append(("gustatory", "bitter"))
        if s.gust_sweet > 0.6:
            active.append(("gustatory", "sweet"))
        # 内感受
        if s.int_hunger > 0.7:
            active.append(("interoceptive", "hungry"))
        if s.int_thirst > 0.7:
            active.append(("interoceptive", "thirsty"))
        if s.int_fatigue > 0.7:
            active.append(("interoceptive", "tired"))
        # 触觉
        if s.tactile_pain > 0.8:
            active.append(("tactile", "severe_pain"))
        elif s.tactile_pain > 0.4:
            active.append(("tactile", "pain"))
        # 听觉
        if s.aud_loudness > 0.7:
            active.append(("auditory", "noisy"))
        elif s.aud_loudness < 0.2:
            active.append(("auditory", "quiet"))
        # 视觉
        if s.vis_brightness < 0.2:
            active.append(("visual", "dark"))
        # 嗅觉
        if s.olf_pleasant < -0.5:
            active.append(("olfactory", "foul"))
        elif s.olf_pleasant > 0.5:
            active.append(("olfactory", "fragrant"))
        return active

    def couple_emotion(self, emotional_state: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
        """
        感知→情绪耦合。
        返回: (情绪增量, 触发的规则列表)
        """
        delta = {e: 0.0 for e in ["anger", "disgust", "fear", "joy", "sadness", "surprise"]}
        triggered = []

        # 体感（热=负舒适，冷=正舒适）
        if self.state.thermal_comfort < -0.5:
            self._apply_delta(delta, self.COUPLING_RULES[("thermal", "too_hot")])
            triggered.append("thermal:too_hot")
        elif self.state.thermal_comfort > 0.5:
            self._apply_delta(delta, self.COUPLING_RULES[("thermal", "too_cold")])
            triggered.append("thermal:too_cold")
        elif abs(self.state.thermal_comfort) < 0.2:
            self._apply_delta(delta, self.COUPLING_RULES[("thermal", "comfortable")])
            triggered.append("thermal:comfortable")

        # 味觉
        if self.state.gust_bitter > 0.6:
            self._apply_delta(delta, self.COUPLING_RULES[("gustatory", "bitter")])
            triggered.append("gustatory:bitter")
        if self.state.gust_sweet > 0.6:
            self._apply_delta(delta, self.COUPLING_RULES[("gustatory", "sweet")])
            triggered.append("gustatory:sweet")

        # 内感受
        if self.state.int_hunger > 0.7:
            self._apply_delta(delta, self.COUPLING_RULES[("interoceptive", "hungry")])
            triggered.append("interoceptive:hungry")
        if self.state.int_thirst > 0.7:
            self._apply_delta(delta, self.COUPLING_RULES[("interoceptive", "thirsty")])
            triggered.append("interoceptive:thirsty")
        if self.state.int_fatigue > 0.7:
            self._apply_delta(delta, self.COUPLING_RULES[("interoceptive", "tired")])
            triggered.append("interoceptive:tired")

        # 触觉
        if self.state.tactile_pain > 0.8:
            self._apply_delta(delta, self.COUPLING_RULES[("tactile", "severe_pain")])
            triggered.append("tactile:severe_pain")
        elif self.state.tactile_pain > 0.4:
            self._apply_delta(delta, self.COUPLING_RULES[("tactile", "pain")])
            triggered.append("tactile:pain")

        # 听觉
        if self.state.aud_loudness > 0.7:
            self._apply_delta(delta, self.COUPLING_RULES[("auditory", "noisy")])
            triggered.append("auditory:noisy")
        elif self.state.aud_loudness < 0.2:
            self._apply_delta(delta, self.COUPLING_RULES[("auditory", "quiet")])
            triggered.append("auditory:quiet")

        # 视觉
        if self.state.vis_brightness < 0.2:
            self._apply_delta(delta, self.COUPLING_RULES[("visual", "dark")])
            triggered.append("visual:dark")

        # 嗅觉
        if self.state.olf_pleasant < -0.5:
            self._apply_delta(delta, self.COUPLING_RULES[("olfactory", "foul")])
            triggered.append("olfactory:foul")
        elif self.state.olf_pleasant > 0.5:
            self._apply_delta(delta, self.COUPLING_RULES[("olfactory", "fragrant")])
            triggered.append("olfactory:fragrant")

        # 应用耦合强度 (默认 0.6)
        coupled = {}
        for e, d in delta.items():
            base = emotional_state.get(e, 0)
            coupled[e] = min(1.0, base + d * 0.6)

        return coupled, triggered

    def modulate_perception_by_emotion(self, emotional_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        情绪→感知反向调制。
        如：快乐时痛觉阈值提升、悲伤时味觉迟钝。
        """
        modulated = self.state.to_dict()
        base_vec = emotional_state.get("base_vector", emotional_state)
        dominant_emotion = max(base_vec, key=base_vec.get) if base_vec else "neutral"
        intensity = base_vec.get(dominant_emotion, 0)

        if dominant_emotion not in self.EMOTIONAL_PERCEPTION_MODULATION or intensity < 0.4:
            return modulated

        rules = self.EMOTIONAL_PERCEPTION_MODULATION[dominant_emotion]

        # 痛觉阈值调整
        if "pain_threshold_multiplier" in rules:
            factor = rules["pain_threshold_multiplier"] * intensity
            modulated["tactile"]["pain"] /= max(0.1, factor)

        # 听觉敏感度
        if "auditory_sensitivity_boost" in rules:
            factor = rules["auditory_sensitivity_boost"] * intensity
            modulated["auditory"]["loudness"] *= factor

        # 视觉敏感度
        if "visual_sensitivity_boost" in rules:
            factor = rules["visual_sensitivity_boost"] * intensity
            modulated["visual"]["brightness"] *= factor

        # 味觉敏感度降低
        if "gustatory_sensitivity_reduction" in rules:
            factor = rules["gustatory_sensitivity_reduction"] * intensity
            for key in ["sweet", "sour", "salty", "bitter", "umami"]:
                modulated["gustatory"][key] *= (1 - factor)

        # 味觉敏感度提升
        if "gustatory_sensitivity_boost" in rules:
            factor = rules["gustatory_sensitivity_boost"] * intensity
            for key in ["sweet", "sour", "salty", "bitter", "umami"]:
                modulated["gustatory"][key] *= factor

        # 疲劳感知
        if "fatigue_perceived_reduction" in rules:
            modulated["interoceptive"]["fatigue"] *= (1 - rules["fatigue_perceived_reduction"] * intensity)
        if "fatigue_perceived_boost" in rules:
            modulated["interoceptive"]["fatigue"] *= (1 + rules["fatigue_perceived_boost"] * intensity)

        # 饥饿感知
        if "hunger_perceived_reduction" in rules:
            modulated["interoceptive"]["hunger"] *= (1 - rules["hunger_perceived_reduction"] * intensity)

        return modulated

    def detect_direct_behavior(self, utterance: str) -> Optional[Dict]:
        """
        检测感知直接驱动行为（反射性行为）。
        返回 None 或 {is_perception_directed, dominant_sense, W_p, W_e, type}
        """
        # 热→开空调
        if any(kw in utterance for kw in ["热", "出汗", "闷"]) and \
           any(kw in utterance for kw in ["空调", "风扇", "降温", "凉快"]):
            return {
                "is_perception_directed": True,
                "dominant_sense": "thermal",
                "perception_weight": 0.85,
                "emotion_weight": 0.15,
                "type": "direct_reflex",
                "trigger": "heat → cooling",
            }

        # 冷→加衣
        if any(kw in utterance for kw in ["冷", "冻", "凉"]) and \
           any(kw in utterance for kw in ["衣服", "被子", "暖气", "取暖"]):
            return {
                "is_perception_directed": True,
                "dominant_sense": "thermal",
                "perception_weight": 0.85,
                "emotion_weight": 0.15,
                "type": "direct_reflex",
                "trigger": "cold → warm_up",
            }

        # 饿→吃饭
        if any(kw in utterance for kw in ["饿", "肚子叫"]) and \
           any(kw in utterance for kw in ["吃饭", "点餐", "外卖", "吃东西"]):
            return {
                "is_perception_directed": True,
                "dominant_sense": "interoceptive",
                "perception_weight": 0.80,
                "emotion_weight": 0.20,
                "type": "direct_reflex",
                "trigger": "hunger → eat",
            }

        return None

    def get_synesthesia_qualities(self) -> List[Tuple[str, List[str]]]:
        """检测当前感知状态中激活的通感质量因子"""
        active = []
        for quality, channels in self.SYNESTHESIA_QUALITIES.items():
            matched_channels = []
            for channel, info in channels.items():
                if channel == "associated_emotion":
                    continue
                if info["condition"](self.state):
                    matched_channels.append(channel)
            if len(matched_channels) >= 2:  # 至少两个通道激活
                active.append((quality, matched_channels))
        return active

    def reset(self):
        """重置感知状态"""
        self.state = PerceptualState()
        self.history = []

    # ── 内部方法 ─────────────────────────────────────────────────

    def _apply_natural_decay(self):
        """感知自然衰减"""
        elapsed = time.time() - self.state.timestamp
        for sense, halflife in self.SENSE_DECAY_HALFLIFE.items():
            if elapsed > halflife:
                decay_factor = math.exp(-0.693 * (elapsed - halflife) / halflife)
                self._decay_sense(sense, decay_factor)

    def _decay_sense(self, sense: str, factor: float):
        """对单个感知维度应用衰减"""
        if sense == "thermal":
            self.state.thermal_comfort *= factor
        elif sense == "tactile":
            self.state.tactile_pain *= factor
            self.state.tactile_pressure *= factor
        elif sense == "gustatory":
            for attr in ["gust_sweet", "gust_sour", "gust_salty", "gust_bitter", "gust_umami"]:
                setattr(self.state, attr, getattr(self.state, attr) * factor)
        elif sense == "olfactory":
            self.state.olf_pleasant *= factor
            self.state.olf_intensity *= factor
        elif sense == "visual":
            self.state.vis_brightness = 0.5 + (self.state.vis_brightness - 0.5) * factor
        elif sense == "auditory":
            self.state.aud_loudness = 0.3 + (self.state.aud_loudness - 0.3) * factor
        elif sense == "interoceptive":
            for attr in ["int_hunger", "int_thirst", "int_fatigue", "int_nausea"]:
                setattr(self.state, attr, getattr(self.state, attr) * factor)

    def _get_dominant_sense(self) -> Optional[str]:
        """计算主导感知维度"""
        scores = {
            "thermal": abs(self.state.thermal_comfort),
            "tactile": self.state.tactile_pain,
            "gustatory": max(self.state.gust_sweet, self.state.gust_bitter,
                             self.state.gust_sour, self.state.gust_salty, self.state.gust_umami),
            "olfactory": abs(self.state.olf_pleasant),
            "visual": abs(self.state.vis_brightness - 0.5) * 2,
            "auditory": self.state.aud_loudness,
            "interoceptive": max(self.state.int_hunger, self.state.int_thirst,
                                 self.state.int_fatigue, self.state.int_nausea),
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0.3 else None

    def _get_dominant_score(self) -> float:
        scores = {
            "thermal": abs(self.state.thermal_comfort),
            "tactile": self.state.tactile_pain,
            "gustatory": max(self.state.gust_sweet, self.state.gust_bitter,
                             self.state.gust_sour, self.state.gust_salty, self.state.gust_umami),
            "olfactory": abs(self.state.olf_pleasant),
            "visual": abs(self.state.vis_brightness - 0.5) * 2,
            "auditory": self.state.aud_loudness,
            "interoceptive": max(self.state.int_hunger, self.state.int_thirst,
                                 self.state.int_fatigue, self.state.int_nausea),
        }
        return max(scores.values())

    @staticmethod
    def _apply_delta(delta: Dict[str, float], rule: Dict[str, float]):
        for emotion, value in rule.items():
            delta[emotion] += value

    def build_perceptual_fingerprint(self, content: str) -> Dict[str, Any]:
        """
        从文本内容推断感知指纹（用于感知记忆存储）。
        返回各通道的 dominant_imagery。
        """
        fingerprint = {}

        for sense, groups in self.PERCEPTION_KEYWORDS.items():
            imagery = []
            max_intensity = 0.0
            for quality, keywords in groups.items():
                matched = [kw for kw in keywords if kw in content]
                if matched:
                    imagery.extend(matched)
                    # 估算强度
                    if quality in ["hot", "cold", "pain", "bitter", "sweet", "hungry",
                                   "thirsty", "tired", "noisy", "bright", "dark",
                                   "fragrant", "foul"]:
                        max_intensity = max(max_intensity, 0.7)
                    else:
                        max_intensity = max(max_intensity, 0.4)

            if imagery:
                channel_name = {
                    "thermal": "thermal", "gustatory": "gustatory",
                    "interoceptive": "interoceptive", "tactile": "tactile",
                    "auditory": "auditory", "visual": "visual",
                    "olfactory": "olfactory",
                }.get(sense, sense)

                fingerprint[channel_name] = {
                    "dominant_imagery": list(set(imagery)),
                    "intensity": max_intensity,
                    "valence": self._estimate_perceptual_valence(sense, max_intensity),
                }

        return fingerprint

    @staticmethod
    def _estimate_perceptual_valence(sense: str, intensity: float) -> float:
        """估算感知效价"""
        positive_senses = ["gustatory_sweet", "olfactory_fragrant", "thermal_comfort"]
        negative_senses = ["pain", "bitter", "foul", "hungry", "thirsty", "tired"]
        # 简化：根据感知类型判断
        return 0.0  # 中性默认，由具体上下文决定


# ═══════════════════════════════════════════════════════════════
# 命令行测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    psm = PerceptualStateMachine()

    test_utterances = [
        "好热啊，帮我把空调打开",
        "刚才吃了口苦瓜，太苦了，给我拿颗糖来",
        "还没吃饭，饿死了，而且今天工作特别烦",
        "刚才打球太投入了，现在才发现膝盖擦破了一大块",
        "最近心情不好，吃什么都没味道",
        "这个房间的灯光太刺眼了，跟刚才那个装修电钻声一样烦",
        "又冷又难过，想回家了",
    ]

    print("=" * 70)
    print("FSTN-4D 感知状态机测试")
    print("=" * 70)

    for i, utterance in enumerate(test_utterances, 1):
        updates = psm.update_from_utterance(utterance)
        dominant, score = psm.get_dominant()

        print(f"\n{'─'*60}")
        print(f"[测试 {i}] {utterance[:45]}...")
        print(f"  感知更新: {updates}")
        print(f"  主导感知: {dominant} (score={score:.2f})")

        # 耦合情绪
        neutral_emotion = {"anger": 0.0, "disgust": 0.0, "fear": 0.0,
                           "joy": 0.0, "sadness": 0.0, "surprise": 0.0}
        coupled, triggers = psm.couple_emotion(neutral_emotion)
        if triggers:
            print(f"  耦合情绪: {', '.join(triggers)}")
            significant = {e: round(v, 2) for e, v in coupled.items() if v > 0.1}
            if significant:
                print(f"  情绪增量: {significant}")

        # 直接行为检测
        direct = psm.detect_direct_behavior(utterance)
        if direct:
            print(f"  行为类型: 感知直接驱动 (W_p={direct['perception_weight']})")

        # 通感质量
        syn = psm.get_synesthesia_qualities()
        if syn:
            for quality, channels in syn:
                print(f"  通感质量: {quality} → {channels}")

        # 重置部分状态以模拟独立对话
        psm.reset()

    print(f"\n{'='*70}")

    # 感知指纹测试
    print("\n--- 感知指纹推断 ---")
    test_texts = [
        "昨天去了温泉，水滑滑的，硫磺味很重，泡完皮肤红红的",
        "篝火晚会上，红色的火光闪烁，木柴噼啪作响，空气中弥漫着烟味",
    ]
    for text in test_texts:
        fp = psm.build_perceptual_fingerprint(text)
        print(f"  {text[:30]}...")
        for channel, info in fp.items():
            print(f"    {channel}: imagery={info['dominant_imagery'][:3]}, "
                  f"intensity={info['intensity']:.2f}")

    print(f"\n✅ 测试完成")
