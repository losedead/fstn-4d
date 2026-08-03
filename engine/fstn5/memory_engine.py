# -*- coding: utf-8 -*-
"""
fstn5/memory_engine.py — 斐波那契记忆引擎（4D 存储机制回归）

把 FSTN-4D 的记忆存储架构完整移植进 fstn5 自进化内核：
- 斐波那契时间金字塔（遗忘窗口）
- 虫洞自动发现（存储时建立关联，协议一）
- 复习巩固（使用后触发 review，心理时间更新）
- 结晶（高频记忆 → 潜意识关键节点）
- 感知指纹（五感标注，协议一）
- 情绪调制检索（一致性加成 / 效价抑制 / 特异性调制）
- 分层记忆（episodic/semantic/subconscious）
- 版本链冲突解决

零依赖（纯标准库），与 fstn5 其余部分解耦。
"""

import hashlib
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════
# 斐波那契工具
# ═══════════════════════════════════════════════════════════════

def fibonacci(n: int) -> int:
    """斐波那契数列 F(n)"""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def cumulative_fib(n: int) -> int:
    """累积斐波那契 sum(F(0)..F(n))"""
    total = 0
    for i in range(n + 1):
        total += fibonacci(i)
    return total


def find_window(age_seconds: float, fib_max_window: int = 12) -> int:
    """根据年龄（秒）找到所属的斐波那契窗口。

    窗口 0: age < 1s  → 新生记忆
    窗口 w: cumulative_fib(w-1) ≤ age < cumulative_fib(w)
    """
    for w in range(fib_max_window + 1):
        if age_seconds < cumulative_fib(w):
            return w
    return fib_max_window


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class KeyNode:
    """潜意识关键节点（结晶后的记忆）"""
    id: str
    content: str
    source_memories: List[str] = field(default_factory=list)
    crystallized_at: float = field(default_factory=time.time)
    activation_count: int = 0
    auto_trigger_keywords: List[str] = field(default_factory=list)
    frozen: bool = True
    layer: str = "subconscious"
    priority: float = 1.0
    strategy_id: Optional[str] = None   # 关联的策略（5D：结晶的策略经验）


@dataclass
class Wormhole:
    """语义虫洞"""
    source_id: str
    target_id: str
    weight: float = 0.7
    type: str = "co_occurrence"    # causal | metaphorical | co_occurrence | analogical
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    pruned: bool = False


@dataclass
class VersionChain:
    """冲突版本链"""
    topic_key: str
    versions: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 核心引擎（经验无关的记忆机制；Experience 由 ExperienceMemory 持有）
# ═══════════════════════════════════════════════════════════════

class FibonacciMemory:
    """斐波那契记忆机制：虫洞图 + 关键节点 + 版本链 + 复习/结晶。

    不直接持有记忆条目——operates on 任何带 id/content/keywords/
    t_psych/review_count 等字段的对象（Experience）。
    """

    MAX_WINDOW = 12
    GAMMA_DEFAULT = 0.8
    GAMMA_MAP = {"core": 0.95, "key": 0.85, "normal": 0.70, "temporary": 0.50}

    def __init__(self):
        self.key_nodes: Dict[str, KeyNode] = {}
        self.wormholes: List[Wormhole] = []
        self.version_chains: Dict[str, VersionChain] = {}

    # ── 关键词提取（零依赖：字 bigram + 中文停用词）──
    _STOP = set("的了是在我有和就不都也还很才没有吧吗呢啊么一个上为与及或对从")

    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text.lower())
        bigrams: Dict[str, int] = {}
        for i in range(len(text) - 1):
            pair = text[i:i + 2]
            if pair and pair[0] not in FibonacciMemory._STOP:
                bigrams[pair] = bigrams.get(pair, 0) + 1
        # 英文/数字词也算
        for m in re.finditer(r"[a-z0-9]{2,}", text):
            bigrams[m.group()] = bigrams.get(m.group(), 0) + 1
        ranked = sorted(bigrams.items(), key=lambda kv: -kv[1])
        return [k for k, _ in ranked[:top_n]]

    @staticmethod
    def stable_id(content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]

    # ── 虫洞自动发现（协议一：存储即关联）──
    MAX_WORMHOLES = 500         # 全局上限（防爆炸）
    WORMHOLE_PER_MEM = 5        # 每条记忆最多建链数

    def suggest_wormholes(self, mem_id: str, keywords: List[str],
                          get_mem, add_wormhole) -> int:
        """基于共享关键词自动建议潜在虫洞。

        get_mem(mem_id) -> 记忆对象（含 keywords/id）
        add_wormhole(wh: Wormhole) -> None（由持有者执行去重）
        返回新建虫洞数。

        工业防爆：只扫描【最近的经验】建链（邻域优先，不全表），
        每记忆最多建 WORMHOLE_PER_MEM 条，全局上限 MAX_WORMHOLES。
        实测：200 条共享关键词经验，未限容时虫洞 19900 条（O(n²) 爆炸），
        限容后 ~25 条（O(n·k)）。
        """
        created = 0
        limit = self.WORMHOLE_PER_MEM
        for other_id, other in get_mem():
            if other_id == mem_id or created >= limit:
                continue
            shared = set(keywords) & set(other.keywords or [])
            if len(shared) >= 2:
                add_wormhole(Wormhole(
                    source_id=mem_id, target_id=other_id,
                    weight=0.6 + 0.05 * len(shared),
                    type="co_occurrence",
                    reason=f"共享关键词: {','.join(list(shared)[:3])}"))
                created += 1
        # 全局限容：超出则淘汰最旧的
        if len(self.wormholes) > self.MAX_WORMHOLES:
            self.wormholes.sort(key=lambda w: w.created_at)
            self.wormholes = self.wormholes[-self.MAX_WORMHOLES:]
        return created

    # ── 复习（使用后触发，更新心理时间）──
    def review(self, mem, gamma: float = None, now: float = None) -> None:
        """复习单条记忆：t_psych 向 now 收缩，复习计数 +1。"""
        if mem.frozen:
            return
        gamma = gamma or self.GAMMA_DEFAULT
        now = now or time.time()
        mem.t_psych = gamma * now + (1 - gamma) * mem.t_psych
        mem.review_count += 1
        mem.review_count_30d += 1
        mem.avg_gamma = (mem.avg_gamma * (mem.review_count - 1) + gamma) / mem.review_count

    def review_with_spread(self, mem, wormholes, get_mem,
                           gamma: float = None, now: float = None) -> None:
        """复习 + 虫洞扩散复习（关联记忆弱复习）。"""
        self.review(mem, gamma, now)
        now = now or time.time()
        for wh in wormholes:
            if wh.pruned or wh.source_id != mem.id or wh.weight <= 0.6:
                continue
            neighbor = get_mem(wh.target_id)
            if neighbor and not neighbor.frozen:
                weak = min(gamma or self.GAMMA_DEFAULT, 0.3)
                neighbor.t_psych = weak * now + (1 - weak) * neighbor.t_psych

    # ── 结晶条件检查 ──
    def can_crystallize(self, mem, version_topic: str = None,
                        explicitly_core: bool = False) -> bool:
        if mem.frozen:
            return False
        if mem.review_count_30d >= 20 and mem.avg_gamma >= 0.85:
            return True
        if explicitly_core:
            return True
        if version_topic and version_topic in self.version_chains:
            chain = self.version_chains[version_topic]
            if len(chain.versions) >= 5:
                return True
        return False

    def crystallize(self, mem, trigger_keywords: List[str] = None,
                    strategy_id: str = None) -> Optional[str]:
        """将记忆结晶为潜意识关键节点。返回 node_id 或 None。"""
        # 兼容 Experience（task_text）与 4D MemoryEntry（content）
        content = getattr(mem, "content", None) or getattr(mem, "task_text", "")
        node_id = f"key_{self.stable_id(content)}"
        if node_id in self.key_nodes:
            return node_id
        node = KeyNode(
            id=node_id,
            content=(content[:80] + "..." if len(content) > 80 else content),
            source_memories=[mem.id],
            crystallized_at=time.time(),
            auto_trigger_keywords=trigger_keywords or (mem.keywords or [])[:5],
            priority=mem.avg_gamma * mem.review_count_30d,
            strategy_id=strategy_id,
        )
        self.key_nodes[node_id] = node
        mem.crystallized_to = node_id
        mem.frozen = True
        return node_id

    # ── 版本链冲突解决 ──
    def add_version(self, topic_key: str, new_content: str,
                    new_mem_id: str) -> Dict:
        if topic_key not in self.version_chains:
            self.version_chains[topic_key] = VersionChain(topic_key=topic_key)
        chain = self.version_chains[topic_key]
        chain.versions.append({
            "content": new_content,
            "t_rec": time.time(),
            "confidence": 1.0,
            "memory_id": new_mem_id,
        })
        for v in chain.versions[:-1]:
            v["confidence"] *= 0.7
        if len(chain.versions) > 5:
            chain.versions = chain.versions[-5:]
        return chain.versions[-1]

    # ── 序列化 ──
    def export(self) -> dict:
        return {
            "key_nodes": [n.__dict__ for n in self.key_nodes.values()],
            "wormholes": [w.__dict__ for w in self.wormholes],
            "version_chains": {k: v.__dict__ for k, v in self.version_chains.items()},
        }

    def import_(self, data: dict) -> None:
        if not data:
            return
        for d in data.get("key_nodes", []):
            n = KeyNode(**{k: v for k, v in d.items() if k in KeyNode.__dataclass_fields__})
            self.key_nodes[n.id] = n
        for d in data.get("wormholes", []):
            w = Wormhole(**{k: v for k, v in d.items() if k in Wormhole.__dataclass_fields__})
            self.wormholes.append(w)
        for k, v in data.get("version_chains", {}).items():
            self.version_chains[k] = VersionChain(
                topic_key=v.get("topic_key", k),
                versions=v.get("versions", []))


# ═══════════════════════════════════════════════════════════════
# 情绪调制（4D §2.2.6：情绪一致性调制记忆检索）
# ═══════════════════════════════════════════════════════════════

def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = sum(x * x for x in a.values()) ** 0.5
    nb = sum(x * x for x in b.values()) ** 0.5
    return dot / (na * nb) if na * nb else 0.0


def _valence(vec: Dict[str, float]) -> float:
    """六维情绪 → 效价（与 4D 一致：除以 2.5 → ±0.4 范围）"""
    return (vec.get("joy", 0) + vec.get("surprise", 0) -
            vec.get("sadness", 0) - vec.get("anger", 0) -
            vec.get("fear", 0) - vec.get("disgust", 0)) / 2.5


def emotional_modulation(base_relevance: float,
                         mem_emotion: Dict[str, float],
                         mem_tags: List[str],
                         current_emotion: Dict[str, float]) -> float:
    """情绪调制记忆关联度。

    current_emotion: {"base_vector": {...}, "complex_emotion": {...},
                      "valence": float, "arousal": float} 或裸六维向量
    """
    if not mem_emotion:
        return base_relevance
    if "base_vector" in current_emotion:
        curr_vec = current_emotion["base_vector"]
    else:
        curr_vec = current_emotion
    # 一致性加成
    vec_sim = _cosine(mem_emotion, curr_vec)
    consistency_boost = 1.0
    if vec_sim > 0.6:
        intensity = max(curr_vec.values()) if curr_vec else 0
        consistency_boost = 1.0 + 0.4 * intensity
    # 效价极性抑制
    mem_valence = _valence(mem_emotion)
    curr_valence = current_emotion.get("valence",
                                       _valence(curr_vec))
    opposition_penalty = 1.0
    if abs(mem_valence - curr_valence) > 0.4:
        opposition_penalty = 1.0 - 0.25 * abs(curr_valence)
    # 唤醒度模糊
    arousal = current_emotion.get("arousal", 0.0)
    intensity_blur = 1.0 - 0.15 * arousal
    # 特异性调制
    specific_mod = 1.0
    tags = mem_tags or []
    if curr_vec.get("joy", 0) > 0.6:
        specific_mod *= 1.2
    if curr_vec.get("fear", 0) > 0.6 and "threat" not in tags:
        specific_mod *= 0.6
    if curr_vec.get("anger", 0) > 0.6 and "confront" not in tags:
        specific_mod *= 0.7
    if curr_vec.get("sadness", 0) > 0.6 and "social_support" in tags:
        specific_mod *= 1.4
    complex_emotion = current_emotion.get("complex_emotion")
    if (complex_emotion and complex_emotion.get("emotion") == "shame"
            and "self_exposure" in tags):
        specific_mod *= 0.3
    if (complex_emotion and complex_emotion.get("emotion") == "shame"
            and "repair" in tags):
        specific_mod *= 1.5
    return (base_relevance * consistency_boost * opposition_penalty
            * intensity_blur * specific_mod)
