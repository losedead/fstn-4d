# -*- coding: utf-8 -*-
"""
fstn5/models.py — 数据模型

领域无关的核心数据结构：经验 / 策略 / 用户状态向量。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import time
import uuid

# 策略状态
STRATEGY_ACTIVE = "active"
STRATEGY_FROZEN = "frozen"       # 结晶：已验证有效的策略，不可淘汰
STRATEGY_DEPRECATED = "deprecated"  # 淘汰：效果衰减被弃用


@dataclass
class Strategy:
    """策略：引擎要学习/优化的对象。

    任何领域的具体做法都可以包装成策略：
    - Agent 服务：'优先用搜索工具' / '先读文档再写代码'
    - 数据处理：'先清洗再分析' / '用流式管道'
    """
    name: str
    description: str = ""
    domain: str = "generic"          # agent_service / data_pipeline / ...
    weight: float = 1.0              # Bandit 选择权重（可学习）
    reward_ema: float = 0.0          # 平均奖励 EMA
    trials: int = 0                  # 尝试次数
    status: str = STRATEGY_ACTIVE
    parent_id: Optional[str] = None  # 演化来源（谱系）
    rationale: str = ""              # 生成理由（LLM 演化时记录，可解释性）
    origin: str = "manual"           # manual / llm / rule
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    last_trial_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Strategy":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Experience:
    """经验：一次【任务-策略-结果】的完整记录。

    这是引擎"自我学习"的最小单元——记住了什么策略在什么任务上
    对什么用户产生了什么结果。

    v2（4D 存储机制回归）：扩展为完整记忆条目——
    - 分层（layer）：episodic/semantic/subconscious
    - 情绪（recorded_emotion）：存储时记录的用户情绪状态
    - 感知指纹（perceptual_signature）：五感标注（4D 协议一）
    - 心理时间（t_psych）：复习更新，驱动斐波那契遗忘窗口
    - 复习统计（review_count/review_count_30d/avg_gamma）：结晶条件
    - 虫洞关联：记忆之间的语义虫洞
    """
    task_text: str                 # 任务描述（人类可读）
    strategy_id: str               # 使用的策略
    reward: float                  # 结果评分 [-1, 1]
    user_vector: Dict[str, float] = field(default_factory=dict)  # 用户状态（可空）
    task_tags: List[str] = field(default_factory=list)           # 任务标签（聚类用）
    feedback_note: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    # ── 4D 存储机制字段 ──
    layer: str = "episodic"        # episodic | semantic | subconscious
    recorded_emotion: Dict[str, float] = field(default_factory=dict)  # 六维情绪向量
    emotional_tags: List[str] = field(default_factory=list)
    perceptual_signature: Dict = field(default_factory=dict)     # 五感感知指纹
    keywords: List[str] = field(default_factory=list)            # 提取的关键词
    t_psych: float = field(default_factory=time.time)            # 心理时间
    review_count: int = 0
    review_count_30d: int = 0
    avg_gamma: float = 0.0
    crystallized_to: Optional[str] = None   # 结晶到的关键节点 id
    frozen: bool = False                    # 冻结（不可复习/遗忘）
    confidence: float = 1.0

    @property
    def age(self) -> float:
        return time.time() - self.t_psych

    @property
    def window(self) -> int:
        """所属斐波那契窗口（0=新生，越大越旧）"""
        from .memory_engine import find_window
        return find_window(self.age)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Experience":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class UserVector:
    """用户状态向量：驱动个性化的多维描述。

    维度可插拔（适配器提供）：
    - 情绪维度（复用 FSTN-4D）：joy/sadness/anger/...
    - 偏好维度：喜欢简洁/喜欢详细、技术背景、任务偏好
    - 任何适配器认为与"选择策略"相关的维度
    """
    emotion: Dict[str, float] = field(default_factory=dict)   # 情绪六维
    traits: Dict[str, float] = field(default_factory=dict)    # 稳定特质
    context: Dict[str, Any] = field(default_factory=dict)     # 场景上下文

    def to_vector(self) -> Dict[str, float]:
        """展平成策略路由用的向量"""
        v = dict(self.emotion)
        v.update(self.traits)
        return v

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserVector":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """两个稀疏向量余弦相似度（用户状态/任务特征通用）"""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = sum(x * x for x in a.values()) ** 0.5
    nb = sum(x * x for x in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
