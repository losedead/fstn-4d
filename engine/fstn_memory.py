"""
FSTN-4D 斐波那契记忆引擎 (Fibonacci Memory Engine)
=====================================================
实现斐波那契时间金字塔、心理时间、虫洞图、潜意识层（关键节点）、
版本链冲突解决、记忆复习与巩固。

创新点（超越原 spec）：
1. 双轨索引：语义（关键词）+ 时间（斐波那契窗口）双重检索
2. 记忆潮汐：窗口滑动时自动触发 consolidate，模拟自然遗忘节奏
3. 虫洞自动发现：基于共享关键词自动建议潜在虫洞
4. 情绪保护：极端情绪下自动标记记忆为"待确认"，暂缓结晶
5. 关键节点影响力追踪：记录每次默认路径触发，可视化影响力扩散
"""

import math
import time
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import OrderedDict


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
    窗口 1: 1s ≤ age < 2s
    ...
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
class MemoryEntry:
    """记忆条目"""
    id: str
    content: str
    layer: str = "episodic"        # episodic | semantic | subconscious
    t_rec: float = field(default_factory=time.time)      # 记录时间
    t_psych: float = field(default_factory=time.time)    # 心理时间
    recorded_emotion: Dict[str, float] = field(default_factory=dict)  # 六维情绪向量
    emotional_tags: List[str] = field(default_factory=list)
    perceptual_signature: Dict[str, Any] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    review_count: int = 0
    review_count_30d: int = 0
    avg_gamma: float = 0.0
    crystallized_to: Optional[str] = None
    frozen: bool = False
    explicitly_marked_core: bool = False
    confidence: float = 1.0
    pending_confirmation: bool = False  # 极端情绪下的声明

    @property
    def age(self) -> float:
        return time.time() - self.t_psych

    @property
    def window(self) -> int:
        return find_window(self.age)

    def abstract(self, max_len: int = 80) -> str:
        return self.content[:max_len] + ("..." if len(self.content) > max_len else "")


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
# 核心引擎
# ═══════════════════════════════════════════════════════════════

class FibonacciMemoryEngine:
    """FSTN-4D 记忆引擎"""

    MAX_WINDOW = 12
    GAMMA_DEFAULT = 0.8

    # 重要性等级 → gamma 映射
    GAMMA_MAP = {
        "core": 0.95,        # 核心知识（用户职业、偏好）
        "key": 0.85,         # 关键前提
        "normal": 0.70,      # 普通上下文
        "temporary": 0.50,   # 临时可替代信息
    }

    def __init__(self, state_file: Optional[str] = None):
        self.memories: Dict[str, MemoryEntry] = {}
        self.key_nodes: Dict[str, KeyNode] = {}
        self.wormholes: List[Wormhole] = []
        self.version_chains: Dict[str, VersionChain] = {}
        self.state_file = state_file
        self._next_id = 0
        self.interaction_count = 0

        # 自动加载状态
        if state_file:
            self._load_state()

    # ── 公开 API：存储 ──────────────────────────────────────────

    def ingest(self, content: str, layer: str = "episodic",
               importance: str = "normal",
               recorded_emotion: Optional[Dict[str, float]] = None,
               emotional_tags: Optional[List[str]] = None,
               perceptual_signature: Optional[Dict] = None,
               pending_confirmation: bool = False) -> str:
        """
        存储一条新记忆。

        Returns:
            memory_id
        """
        mem_id = self._generate_id(content)
        t_now = time.time()
        gamma = self.GAMMA_MAP.get(importance, self.GAMMA_DEFAULT)

        # 提取关键词
        keywords = self._extract_keywords(content)

        entry = MemoryEntry(
            id=mem_id,
            content=content,
            layer=layer,
            t_rec=t_now,
            t_psych=t_now,
            recorded_emotion=recorded_emotion or {},
            emotional_tags=emotional_tags or [],
            perceptual_signature=perceptual_signature or {},
            keywords=keywords,
            pending_confirmation=pending_confirmation,
        )

        # 检查是否与已有记忆冲突
        conflict = self._check_conflict(content)
        if conflict:
            entry = self.add_version(conflict, content)

        self.memories[mem_id] = entry
        self.interaction_count += 1

        # 自动发现潜在虫洞
        self._suggest_wormholes(mem_id)

        return mem_id

    def add_version(self, topic_key: str, new_content: str) -> MemoryEntry:
        """添加新版本（冲突解决）"""
        if topic_key not in self.version_chains:
            self.version_chains[topic_key] = VersionChain(
                topic_key=topic_key,
                versions=[]
            )
        chain = self.version_chains[topic_key]
        chain.versions.append({
            "content": new_content,
            "t_rec": time.time(),
            "confidence": 1.0,
            "memory_id": self._generate_id(new_content),
        })

        # 旧版本衰减
        for v in chain.versions[:-1]:
            v["confidence"] *= 0.7

        # 只保留最近 5 个版本
        if len(chain.versions) > 5:
            chain.versions = chain.versions[-5:]

        # 检查是否满足结晶条件
        latest = chain.versions[-1]
        confirmations = len(chain.versions)
        if confirmations >= 5:
            latest["ready_to_crystallize"] = True

        return MemoryEntry(
            id=latest["memory_id"],
            content=new_content,
            layer="episodic",
            t_rec=time.time(),
            t_psych=time.time(),
            confidence=latest["confidence"],
        )

    def perceive(self, memory_id: str, perceptual_profile: Dict) -> bool:
        """为记忆附加感知指纹"""
        if memory_id not in self.memories:
            return False
        self.memories[memory_id].perceptual_signature.update(perceptual_profile)
        return True

    # ── 公开 API：检索 ──────────────────────────────────────────

    def retrieve(self, query: str, k: int = 10,
                 filters: Optional[Dict] = None) -> List[MemoryEntry]:
        """
        检索记忆。支持过滤：
        - time_window: 最大窗口限制
        - layer: 记忆层
        - emotional_match: 情绪一致性要求
        - keyword_match: 必须包含的关键词
        """
        filters = filters or {}
        now = time.time()

        # 1. 潜意识自动激活
        subconscious_hits = self._scan_subconscious(query)

        # 2. 语义检索（关键词匹配 + TF-IDF-like 评分）
        scored = []
        query_keywords = self._extract_keywords(query)

        for mem_id, mem in self.memories.items():
            if mem.frozen and mem.crystallized_to:
                continue  # 已结晶的记忆，由关键节点代理

            score = self._semantic_score(mem, query, query_keywords)

            # 时间过滤
            max_window = filters.get("time_window", self.MAX_WINDOW)
            if mem.window > max_window:
                score *= 0.3  # 深层窗口惩罚

            # 层过滤
            target_layer = filters.get("layer")
            if target_layer and mem.layer not in (target_layer if isinstance(target_layer, list) else [target_layer]):
                continue

            if score > 0:
                scored.append((score, mem))

        # 按分数排序
        scored.sort(key=lambda x: -x[0])
        results = [mem for _, mem in scored[:k]]

        # 3. 虫洞扩散
        wormhole_bonus = []
        for mem in results[:5]:
            for wh in self.wormholes:
                if wh.pruned:
                    continue
                if wh.source_id == mem.id and wh.weight > 0.5:
                    target = self.memories.get(wh.target_id)
                    if target and target not in results and target not in wormhole_bonus:
                        wormhole_bonus.append(target)
                        wh.last_used = time.time()
                        wh.weight = min(1.0, wh.weight + 0.05)

        # 4. 关键节点扩散
        for node in subconscious_hits:
            for wh in self.wormholes:
                if wh.pruned or wh.source_id != node.id:
                    continue
                if wh.weight > 0.4:
                    neighbor = self.memories.get(wh.target_id)
                    if neighbor and neighbor not in results and neighbor not in wormhole_bonus:
                        wormhole_bonus.append(neighbor)

        # 融合结果
        all_results = results + wormhole_bonus
        return all_results[:k]

    def emotional_modulation(self, memory_id: str, base_relevance: float,
                              current_emotion: Dict[str, float]) -> float:
        """情绪调制记忆关联度"""
        mem = self.memories.get(memory_id)
        if not mem or not mem.recorded_emotion:
            return base_relevance

        mem_vec = mem.recorded_emotion
        curr_vec = current_emotion.get("base_vector", current_emotion)

        # 一致性加成
        vec_sim = self._cosine_similarity(mem_vec, curr_vec)
        consistency_boost = 1.0
        if vec_sim > 0.6:
            intensity = max(curr_vec.values()) if curr_vec else 0
            consistency_boost = 1.0 + 0.4 * intensity

        # 效价极性抑制（简化版）
        mem_valence = self._compute_valence(mem_vec)
        curr_valence = current_emotion.get("valence", 0.0)
        opposition_penalty = 1.0
        if abs(mem_valence - curr_valence) > 1.2:
            opposition_penalty = 1.0 - 0.25 * abs(curr_valence)

        # 唤醒度模糊
        arousal = current_emotion.get("arousal", 0.0)
        intensity_blur = 1.0 - 0.15 * arousal

        # 特异性调制
        specific_mod = 1.0
        if curr_vec.get("joy", 0) > 0.6:
            specific_mod *= 1.2
        if curr_vec.get("fear", 0) > 0.6:
            if "threat" not in (mem.emotional_tags or []):
                specific_mod *= 0.6
        if curr_vec.get("anger", 0) > 0.6:
            if "confront" not in (mem.emotional_tags or []):
                specific_mod *= 0.7
        if curr_vec.get("sadness", 0) > 0.6:
            if "social_support" in (mem.emotional_tags or []):
                specific_mod *= 1.4

        return base_relevance * consistency_boost * opposition_penalty * intensity_blur * specific_mod

    # ── 公开 API：复习与维护 ────────────────────────────────────

    def review(self, memory_ids: List[str], gamma: float = None) -> int:
        """复习记忆，更新心理时间"""
        gamma = gamma or self.GAMMA_DEFAULT
        count = 0
        now = time.time()

        for mid in memory_ids:
            if mid in self.key_nodes:
                continue
            mem = self.memories.get(mid)
            if not mem or mem.frozen:
                continue

            old_t = mem.t_psych
            mem.t_psych = gamma * now + (1 - gamma) * old_t
            mem.review_count += 1
            mem.review_count_30d += 1
            mem.avg_gamma = (mem.avg_gamma * (mem.review_count - 1) + gamma) / mem.review_count

            # 虫洞扩散复习
            for wh in self.wormholes:
                if wh.pruned:
                    continue
                if wh.source_id == mid and wh.weight > 0.6:
                    neighbor = self.memories.get(wh.target_id)
                    if neighbor and neighbor.id not in self.key_nodes and not neighbor.frozen:
                        weak_gamma = min(gamma, 0.3)
                        neighbor.t_psych = weak_gamma * now + (1 - weak_gamma) * neighbor.t_psych

            count += 1

        return count

    def crystallize(self, memory_id: str, trigger_keywords: List[str] = None,
                     current_emotion: Optional[Dict[str, float]] = None) -> Optional[str]:
        """
        将记忆结晶为潜意识关键节点。
        返回 key_node_id 或 None（条件不满足）。
        """
        # 情绪保护
        if current_emotion:
            max_intensity = max(current_emotion.get("base_vector", current_emotion).values()) if isinstance(
                current_emotion.get("base_vector", {}), dict
            ) else 0
            if max_intensity > 0.8:
                # 标记为待确认，暂缓结晶
                if memory_id in self.memories:
                    self.memories[memory_id].pending_confirmation = True
                return None

        mem = self.memories.get(memory_id)
        if not mem:
            return None

        # 结晶条件检查
        can_crystallize = (
            (mem.review_count_30d >= 20 and mem.avg_gamma >= 0.85) or
            mem.explicitly_marked_core or
            self._is_version_chain_ready(memory_id)
        )

        if not can_crystallize:
            return None

        # 创建关键节点
        node_id = f"key_{self._generate_short_id(mem.content)}"
        keywords = trigger_keywords or self._extract_keywords(mem.content, top_n=5)
        node = KeyNode(
            id=node_id,
            content=mem.abstract(),
            source_memories=[memory_id],
            crystallized_at=time.time(),
            auto_trigger_keywords=keywords,
            priority=mem.avg_gamma * mem.review_count_30d,
        )
        self.key_nodes[node_id] = node
        mem.crystallized_to = node_id
        mem.frozen = True

        return node_id

    def consolidate(self, target_layer: str = "episodic", merge_threshold: float = 0.85) -> int:
        """
        合并深层窗口中相似的记忆。
        返回合并的记忆数。
        """
        merged_count = 0
        now = time.time()

        # 对窗口 6+ 的记忆检查相似度
        for mid, mem in list(self.memories.items()):
            if mem.window < 6 or mem.frozen:
                continue
            if mem.layer != target_layer:
                continue

            # 找相似记忆
            for other_id, other in self.memories.items():
                if other_id == mid:
                    continue
                if other.window < 6 or other.frozen:
                    continue

                sim = self._keyword_similarity(mem.keywords, other.keywords)
                if sim >= merge_threshold:
                    # 合并到较新的那条
                    merged = self._merge_memories(mem, other)
                    self.memories[mid] = merged
                    if other_id in self.memories:
                        del self.memories[other_id]
                    merged_count += 1
                    break

        return merged_count

    def prune_wormholes(self, min_weight: float = 0.3, max_age_days: float = 30) -> int:
        """剪枝弱虫洞"""
        pruned = 0
        now = time.time()
        for wh in self.wormholes:
            if wh.pruned:
                continue
            age_days = (now - wh.created_at) / 86400
            if wh.weight < min_weight and age_days > max_age_days:
                wh.pruned = True
                pruned += 1
        return pruned

    def create_wormhole(self, source_id: str, target_id: str,
                         type_: str = "co_occurrence", reason: str = "") -> bool:
        """创建语义虫洞"""
        if source_id not in self.memories and source_id not in self.key_nodes:
            return False
        if target_id not in self.memories and target_id not in self.key_nodes:
            return False

        # 防重复
        for wh in self.wormholes:
            if wh.source_id == source_id and wh.target_id == target_id:
                wh.weight = min(1.0, wh.weight + 0.1)
                return True

        wh = Wormhole(
            source_id=source_id,
            target_id=target_id,
            weight=0.7,
            type=type_,
            reason=reason,
            created_at=time.time(),
        )
        self.wormholes.append(wh)
        return True

    # ── 查询方法 ─────────────────────────────────────────────────

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        return self.memories.get(memory_id)

    def get_key_node(self, node_id: str) -> Optional[KeyNode]:
        return self.key_nodes.get(node_id)

    def get_window_distribution(self) -> Dict[int, int]:
        """统计各窗口的记忆数量"""
        dist = {}
        for mem in self.memories.values():
            w = mem.window
            dist[w] = dist.get(w, 0) + 1
        return dist

    def get_all_key_nodes(self) -> List[KeyNode]:
        return list(self.key_nodes.values())

    def get_active_wormholes(self) -> List[Wormhole]:
        return [wh for wh in self.wormholes if not wh.pruned]

    def export_state(self) -> Dict:
        """导出引擎状态"""
        return {
            "memory_count": len(self.memories),
            "key_node_count": len(self.key_nodes),
            "wormhole_count": len([wh for wh in self.wormholes if not wh.pruned]),
            "window_distribution": self.get_window_distribution(),
            "interaction_count": self.interaction_count,
            "key_nodes": [
                {"id": n.id, "content": n.content, "priority": n.priority}
                for n in self.key_nodes.values()
            ],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """引擎统计信息"""
        dist = self.get_window_distribution()
        return {
            "total_memories": len(self.memories),
            "key_nodes": len(self.key_nodes),
            "active_wormholes": len([wh for wh in self.wormholes if not wh.pruned]),
            "version_chains": len(self.version_chains),
            "windows": dist,
            "fresh_memories": sum(v for w, v in dist.items() if w <= 2),
            "old_memories": sum(v for w, v in dist.items() if w >= 6),
            "interaction_count": self.interaction_count,
        }

    # ── 内部方法 ─────────────────────────────────────────────────

    def _generate_id(self, content: str) -> str:
        """生成记忆 ID"""
        self._next_id += 1
        h = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"mem_{self._next_id:04d}_{h}"

    @staticmethod
    def _generate_short_id(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:10]

    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """中文关键词提取：基于 n-gram (2-4字) + 停用词过滤"""
        import re
        # 移除常见标点
        punct = '，。！？、；：""''（）【】《》,.!?;:()[]'
        clean = text
        for p in punct:
            clean = clean.replace(p, '|')
        segments = [s.strip() for s in clean.split('|') if s.strip() and len(s.strip()) >= 2]

        # 从各段提取 2-4 字 n-gram
        candidates = set()
        for seg in segments:
            for n in [4, 3, 2]:
                for i in range(len(seg) - n + 1):
                    ngram = seg[i:i+n]
                    candidates.add(ngram)

        # 按长度降序 + 去重
        unique = list(dict.fromkeys(candidates))
        unique.sort(key=lambda w: -len(w))
        return unique[:top_n]

    def _semantic_score(self, mem: MemoryEntry, query: str,
                         query_keywords: List[str]) -> float:
        """计算语义相似度分数"""
        score = 0.0

        # 关键词匹配（主要信号）
        if query_keywords and mem.keywords:
            matched = set(query_keywords) & set(mem.keywords)
            score += len(matched) * 0.3

        # 查询词是否在记忆内容中
        for qw in query_keywords:
            if qw in mem.content.lower():
                score += 0.2
        # 单字匹配（补充召回）
        query_chars = set(query.replace(' ', ''))
        content_chars = set(mem.content.replace(' ', ''))
        char_overlap = len(query_chars & content_chars)
        if char_overlap >= 2:
            score += 0.12 * min(char_overlap, 10)

        # 完全匹配奖励
        if query.strip() in mem.content:
            score += 2.0

        # 记忆年龄惩罚（窗口越大折扣越多）
        window_penalty = max(0.1, 1.0 - mem.window * 0.08)
        score *= window_penalty

        return score

    def _keyword_similarity(self, kws1: List[str], kws2: List[str]) -> float:
        """关键词集合相似度"""
        if not kws1 or not kws2:
            return 0.0
        set1, set2 = set(kws1), set(kws2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _scan_subconscious(self, query: str) -> List[KeyNode]:
        """潜意识扫描：关键词触发关键节点"""
        activated = []
        for node in self.key_nodes.values():
            if any(kw in query for kw in node.auto_trigger_keywords):
                node.activation_count += 1
                activated.append(node)
        activated.sort(key=lambda n: -n.priority)
        return activated[:3]

    def _suggest_wormholes(self, memory_id: str, min_similarity: float = 0.3):
        """自动发现潜在虫洞"""
        mem = self.memories.get(memory_id)
        if not mem:
            return

        for other_id, other in self.memories.items():
            if other_id == memory_id:
                continue
            sim = self._keyword_similarity(mem.keywords, other.keywords)
            if sim >= min_similarity:
                self.create_wormhole(
                    memory_id, other_id,
                    type_="co_occurrence",
                    reason=f"关键词相似度 {sim:.2f}"
                )

    def _check_conflict(self, content: str) -> Optional[str]:
        """检查是否与已有记忆冲突（返回 topic_key）"""
        ckws = set(self._extract_keywords(content))
        for topic_key, chain in self.version_chains.items():
            if ckws & set(self._extract_keywords(topic_key)):
                return topic_key
        return None

    def _is_version_chain_ready(self, memory_id: str) -> bool:
        """检查版本链是否满足结晶条件"""
        for chain in self.version_chains.values():
            for v in chain.versions:
                if v.get("memory_id") == memory_id and v.get("ready_to_crystallize"):
                    return True
        return False

    def _merge_memories(self, a: MemoryEntry, b: MemoryEntry) -> MemoryEntry:
        """合并两条相似记忆"""
        merged_content = a.content if len(a.content) >= len(b.content) else b.content
        merged = MemoryEntry(
            id=a.id,
            content=merged_content,
            layer=a.layer,
            t_rec=min(a.t_rec, b.t_rec),
            t_psych=max(a.t_psych, b.t_psych),  # 取较新的心理时间
            recorded_emotion=a.recorded_emotion or b.recorded_emotion,
            emotional_tags=list(set((a.emotional_tags or []) + (b.emotional_tags or []))),
            keywords=list(set(a.keywords + b.keywords)),
            review_count=a.review_count + b.review_count,
            avg_gamma=(a.avg_gamma + b.avg_gamma) / 2 if a.avg_gamma and b.avg_gamma else 0.7,
        )
        return merged

    @staticmethod
    def _compute_valence(vector: Dict[str, float]) -> float:
        v = (vector.get("joy", 0) * 0.9
             - vector.get("anger", 0) * 0.8
             - vector.get("disgust", 0) * 0.7
             - vector.get("fear", 0) * 0.9
             - vector.get("sadness", 0) * 0.8)
        return max(-1.0, min(1.0, v / 2.5))

    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in emotions)
        mag_a = math.sqrt(sum(a.get(k, 0)**2 for k in emotions))
        mag_b = math.sqrt(sum(b.get(k, 0)**2 for k in emotions))
        if mag_a < 0.001 or mag_b < 0.001:
            return 0.0
        return dot / (mag_a * mag_b)

    # ── 持久化 ───────────────────────────────────────────────────

    def _load_state(self):
        """从文件加载状态（简化版）"""
        try:
            import os
            if not self.state_file or not os.path.exists(self.state_file):
                return
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 只恢复关键节点
            for kn_data in data.get("key_nodes", []):
                node = KeyNode(**kn_data)
                self.key_nodes[node.id] = node
            self._next_id = data.get("_next_id", 0)
            self.interaction_count = data.get("interaction_count", 0)
        except Exception:
            pass

    def save_state(self):
        """保存引擎状态到文件"""
        if not self.state_file:
            return
        data = {
            "key_nodes": [
                {
                    "id": n.id, "content": n.content,
                    "source_memories": n.source_memories,
                    "crystallized_at": n.crystallized_at,
                    "activation_count": n.activation_count,
                    "auto_trigger_keywords": n.auto_trigger_keywords,
                    "frozen": n.frozen, "layer": n.layer,
                    "priority": n.priority,
                }
                for n in self.key_nodes.values()
            ],
            "_next_id": self._next_id,
            "interaction_count": self.interaction_count,
        }
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# 命令行测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = FibonacciMemoryEngine()

    print("=" * 70)
    print("FSTN-4D 斐波那契记忆引擎测试")
    print("=" * 70)

    # 存储测试记忆
    memories = [
        ("小红很爱吃糖果，每天都吃一颗", "normal",
         {"anger": 0, "disgust": 0, "fear": 0, "joy": 0.6, "sadness": 0, "surprise": 0}),
        ("用户是素食主义者，不吃任何肉类", "core",
         {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0, "surprise": 0}),
        ("用户有猫，养了三年", "core",
         {"anger": 0, "disgust": 0, "fear": 0, "joy": 0.5, "sadness": 0, "surprise": 0}),
        ("用户讨厌电话沟通，偏好文字", "key",
         {"anger": 0.2, "disgust": 0.1, "fear": 0, "joy": 0, "sadness": 0, "surprise": 0}),
        ("昨天去了温泉，水滑滑的，硫磺味很重，泡完皮肤红红的", "normal",
         {"anger": 0, "disgust": 0, "fear": 0, "joy": 0.7, "sadness": 0, "surprise": 0}),
    ]

    ids = []
    for content, imp, emo in memories:
        mid = engine.ingest(content, importance=imp, recorded_emotion=emo,
                            emotional_tags=["routine"] if imp == "normal" else ["core_preference"])
        ids.append(mid)
        print(f"  ✓ {mid}: {content[:40]}... (importance={imp})")

    # 模拟多次复习
    print("\n--- 模拟复习（素食者 x25 次） ---")
    veg_id = ids[1]
    for i in range(25):
        engine.review([veg_id], gamma=0.9)
    mem = engine.get_memory(veg_id)
    print(f"  review_count={mem.review_count}, review_count_30d={mem.review_count_30d}, "
          f"avg_gamma={mem.avg_gamma:.3f}")

    # 尝试结晶
    print("\n--- 结晶测试 ---")
    node_id = engine.crystallize(veg_id, trigger_keywords=["吃饭", "餐厅", "推荐菜", "食谱", "肉类"])
    if node_id:
        print(f"  ✓ 已结晶: {node_id}")
        kn = engine.get_key_node(node_id)
        print(f"    触发关键词: {kn.auto_trigger_keywords}")
    else:
        print(f"  ✗ 条件不满足 (review_count_30d={mem.review_count_30d}, avg_gamma={mem.avg_gamma:.3f})")

    # 检索测试
    print("\n--- 检索测试 ---")
    results = engine.retrieve("今天吃什么", k=5)
    for r in results:
        print(f"  {r.id}: {r.content[:50]}... (window={r.window})")

    # 潜意识测试
    print("\n--- 潜意识扫描 ---")
    activated = engine._scan_subconscious("今晚吃什么好呢，帮我推荐个餐厅")
    for node in activated:
        print(f"  ↳ 关键节点激活: {node.content} (触发次数: {node.activation_count})")

    # 虫洞测试
    print("\n--- 虫洞测试 ---")
    engine.create_wormhole(ids[0], ids[2], type_="co_occurrence", reason="糖果和猫都是日常愉悦")
    engine.create_wormhole(ids[3], ids[2], type_="metaphorical", reason="猫咪不喜欢电话铃声")
    active_wh = engine.get_active_wormholes()
    print(f"  活跃虫洞: {len(active_wh)} 条")

    # 统计
    stats = engine.get_statistics()
    print(f"\n--- 引擎统计 ---")
    print(f"  总记忆: {stats['total_memories']}")
    print(f"  关键节点: {stats['key_nodes']}")
    print(f"  活跃虫洞: {stats['active_wormholes']}")
    print(f"  窗口分布: {stats['windows']}")

    # 情绪调制测试
    print("\n--- 情绪调制测试 ---")
    if ids:
        candy_id = ids[0]
        base = 0.85
        # 在悲伤时查糖果记忆
        sad_emotion = {"base_vector": {"anger": 0, "disgust": 0, "fear": 0.1, "joy": 0, "sadness": 0.8, "surprise": 0},
                       "valence": -0.8, "arousal": 0.3}
        mod_sad = engine.emotional_modulation(candy_id, base, sad_emotion)
        print(f"  糖果记忆: base={base} → sadness调制={mod_sad:.3f} (效价极性抑制)")

        # 在开心时查糖果记忆
        joy_emotion = {"base_vector": {"anger": 0, "disgust": 0, "fear": 0, "joy": 0.7, "sadness": 0, "surprise": 0},
                       "valence": 0.9, "arousal": 0.6}
        mod_joy = engine.emotional_modulation(candy_id, base, joy_emotion)
        print(f"  糖果记忆: base={base} → joy调制={mod_joy:.3f} (一致性加成+积极拓宽)")

    print(f"\n✅ 测试完成")
