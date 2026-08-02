"""
FSTN-4D V3 向量化记忆引擎 (Vector Memory Engine)
=================================================
针对 V1 关键词匹配检索的升级：真·TF-IDF 向量检索 + 余弦相似度。

核心升级（超越 V1）：
1. 中文分词：字符 bigram + unigram 混合特征（零依赖，不装 jieba）
2. TF-IDF 加权：idf = log((N+1)/(df+1)) + 1，检索时用全局最新 IDF 现算
3. 增量索引：ingest 时更新文档频率，不需要全量重算
4. 保留 FSTN 特色：斐波那契时间窗口（心理时间）+ 情绪调制 + 关键节点结晶
5. 检索可解释：返回每条记忆的相似度得分、命中的高权重词

接口与 V1 FibonacciMemoryEngine 对齐（ingest/retrieve/review/crystallize/consolidate），
hybrid_core 可以无缝替换。
"""

import time
import math
import json
import re
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════
# 斐波那契时间窗口（复用 V1 逻辑）
# ═══════════════════════════════════════════════════════════════

def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def cumulative_fib(n: int) -> int:
    return sum(fibonacci(i) for i in range(n + 1))

def find_window(age_seconds: float, fib_max_window: int = 12) -> int:
    for w in range(fib_max_window + 1):
        if age_seconds < cumulative_fib(w):
            return w
    return fib_max_window


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class VectorMemoryEntry:
    """向量化记忆条目"""
    id: str
    content: str
    layer: str = "episodic"
    t_rec: float = field(default_factory=time.time)
    t_psych: float = field(default_factory=time.time)
    recorded_emotion: Dict[str, float] = field(default_factory=dict)
    emotional_tags: List[str] = field(default_factory=list)
    perceptual_signature: Dict[str, Any] = field(default_factory=dict)
    tf: Dict[str, float] = field(default_factory=dict)       # 词频向量（归一化）
    tokens: List[str] = field(default_factory=list)          # 全部特征词
    review_count: int = 0
    review_count_30d: int = 0
    avg_gamma: float = 0.0
    crystallized_to: Optional[str] = None
    frozen: bool = False
    confidence: float = 1.0
    pending_confirmation: bool = False

    @property
    def age(self) -> float:
        return time.time() - self.t_psych

    @property
    def window(self) -> int:
        return find_window(self.age)

    def abstract(self, max_len: int = 80) -> str:
        return self.content[:max_len] + ("..." if len(self.content) > max_len else "")


@dataclass
class VKeyNode:
    """潜意识关键节点"""
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
class VRetrievalHit:
    """检索结果（带可解释信息）"""
    id: str
    content: str
    score: float
    window: int
    layer: str
    top_terms: List[str]          # 命中的高权重词（解释为什么检索到）
    emotional_tag: str = ""


# ═══════════════════════════════════════════════════════════════
# 中文分词：bigram + unigram 混合特征
# ═══════════════════════════════════════════════════════════════

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")

def tokenize(text: str) -> List[str]:
    """
    零依赖中文分词：
    - 汉字序列：unigram + bigram
    - 连续英文/数字：按单词
    - 标点/空白：忽略
    例: "推荐餐厅" -> ["推","荐","餐","厅","推荐","荐餐","餐厅"]
    """
    tokens: List[str] = []
    han_run = ""
    en_run = ""
    for ch in text:
        if CJK_RE.match(ch):
            han_run += ch
            if en_run:
                tokens.append(en_run.lower())
                en_run = ""
        elif ch.isalnum():
            en_run += ch
            if han_run:
                tokens.extend(_han_tokens(han_run))
                han_run = ""
        else:
            if han_run:
                tokens.extend(_han_tokens(han_run))
                han_run = ""
            if en_run:
                tokens.append(en_run.lower())
                en_run = ""
    if han_run:
        tokens.extend(_han_tokens(han_run))
    if en_run:
        tokens.append(en_run.lower())
    return tokens


def _han_tokens(run: str) -> List[str]:
    toks = list(run)  # unigram
    for i in range(len(run) - 1):
        toks.append(run[i:i + 2])  # bigram
    return toks


def build_tf(tokens: List[str]) -> Dict[str, float]:
    """词频（L2 归一化，消除文本长度影响）"""
    if not tokens:
        return {}
    freq: Dict[str, float] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in freq.values()))
    if norm == 0:
        return {}
    return {t: v / norm for t, v in freq.items()}


# ═══════════════════════════════════════════════════════════════
# 核心引擎
# ═══════════════════════════════════════════════════════════════

class VectorMemoryEngine:
    """FSTN-4D V3 向量化记忆引擎"""

    MAX_WINDOW = 12
    GAMMA_MAP = {"core": 0.95, "key": 0.85, "normal": 0.70, "temporary": 0.50}
    GAMMA_DEFAULT = 0.70

    def __init__(self, state_file: Optional[str] = None):
        self.memories: Dict[str, VectorMemoryEntry] = {}
        self.key_nodes: Dict[str, VKeyNode] = {}
        self.state_file = state_file
        self._next_id = 0
        # 全局 IDF 统计
        self._doc_freq: Dict[str, int] = {}   # 词 -> 出现文档数
        self._num_docs = 0
        self._idf_cache: Dict[str, float] = {}
        self._idf_dirty = True
        self.interaction_count = 0
        if state_file and os.path.exists(state_file):
            self._load_state()

    # ═══════════════════════════════════════════════════════════
    # 词法 & IDF
    # ═══════════════════════════════════════════════════════════

    def _idf(self, term: str) -> float:
        """idf = log((N+1)/(df+1)) + 1 —— 平滑，避免除以零"""
        if self._idf_dirty:
            self._idf_cache = {}
            self._idf_dirty = False
        if term not in self._idf_cache:
            df = self._doc_freq.get(term, 0)
            self._idf_cache[term] = math.log((self._num_docs + 1) / (df + 1)) + 1.0
        return self._idf_cache[term]

    def _register_tokens(self, tokens: List[str]):
        """ingest 时注册词频（增量更新 IDF）"""
        seen = set()
        for t in tokens:
            if t not in seen:
                seen.add(t)
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
        self._num_docs += 1
        self._idf_dirty = True

    # ═══════════════════════════════════════════════════════════
    # 存储
    # ═══════════════════════════════════════════════════════════

    def ingest(self, content: str, layer: str = "episodic",
               importance: str = "normal",
               recorded_emotion: Optional[Dict[str, float]] = None,
               emotional_tags: Optional[List[str]] = None,
               perceptual_signature: Optional[Dict] = None,
               pending_confirmation: bool = False,
               keywords: Optional[List[str]] = None) -> str:
        """存储一条记忆，返回 memory_id"""
        mem_id = f"mem_{int(time.time()*1000)}_{self._next_id}"
        self._next_id += 1
        tokens = tokenize(content)
        self._register_tokens(tokens)

        entry = VectorMemoryEntry(
            id=mem_id,
            content=content,
            layer=layer,
            t_rec=time.time(),
            t_psych=time.time(),
            recorded_emotion=recorded_emotion or {},
            emotional_tags=emotional_tags or [],
            perceptual_signature=perceptual_signature or {},
            tf=build_tf(tokens),
            tokens=tokens,
            pending_confirmation=pending_confirmation,
        )
        self.memories[mem_id] = entry
        self.interaction_count += 1
        return mem_id

    def perceive(self, memory_id: str, perceptual_profile: Dict) -> bool:
        if memory_id not in self.memories:
            return False
        self.memories[memory_id].perceptual_signature.update(perceptual_profile)
        return True

    # ═══════════════════════════════════════════════════════════
    # 检索（真·向量检索）
    # ═══════════════════════════════════════════════════════════

    def retrieve(self, query: str, k: int = 10,
                 filters: Optional[Dict] = None) -> List[VRetrievalHit]:
        """TF-IDF 余弦相似度检索。返回带解释的命中列表。"""
        filters = filters or {}
        q_tokens = tokenize(query)
        q_tf = build_tf(q_tokens)
        if not q_tf:
            return []

        q_vec = {t: w * self._idf(t) for t, w in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return []

        max_window = filters.get("time_window", self.MAX_WINDOW)
        target_layer = filters.get("layer")

        scored: List[Tuple[float, VectorMemoryEntry, List[str]]] = []
        for mid, mem in self.memories.items():
            if mem.frozen and mem.crystallized_to:
                continue
            if not mem.tf:
                continue
            if target_layer and mem.layer not in (
                target_layer if isinstance(target_layer, list) else [target_layer]
            ):
                continue
            if mem.window > max_window:
                continue  # 深层窗口直接排除（时间过滤）

            # 余弦相似度（TF-IDF 加权）
            dot = 0.0
            top_terms: List[str] = []
            for t, qw in q_vec.items():
                if t in mem.tf:
                    contrib = qw * mem.tf[t] * self._idf(t)
                    dot += qw * mem.tf[t]
                    if contrib > 0.05:
                        top_terms.append(t)
            if dot <= 0:
                continue
            mem_norm = math.sqrt(sum(
                (w * self._idf(t)) ** 2 for t, w in mem.tf.items()
            ))
            if mem_norm == 0:
                continue
            score = dot / (q_norm * mem_norm)

            # 浅层窗口小加成（新鲜度）
            if mem.window <= 2:
                score *= 1.05
            scored.append((score, mem, sorted(set(top_terms))[:5]))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, mem, top_terms in scored[:k]:
            results.append(VRetrievalHit(
                id=mem.id, content=mem.content, score=round(score, 4),
                window=mem.window, layer=mem.layer, top_terms=top_terms,
            ))
        return results

    # ═══════════════════════════════════════════════════════════
    # 情绪调制检索（V3 特色：向量一致性 + 情绪极性）
    # ═══════════════════════════════════════════════════════════

    def emotional_modulation(self, memory_id: str, base_relevance: float,
                             current_emotion: Dict[str, float]) -> float:
        """情绪调制：与当前情绪一致的记忆升权，相反的记忆降权"""
        mem = self.memories.get(memory_id)
        if not mem or not mem.recorded_emotion:
            return base_relevance

        mem_vec = mem.recorded_emotion
        curr_vec = current_emotion.get("base_vector", current_emotion)

        # 一致性加成：余弦相似度
        dot = sum(mem_vec.get(e, 0) * curr_vec.get(e, 0) for e in
                  ["anger", "disgust", "fear", "joy", "sadness", "surprise"])
        m_norm = math.sqrt(sum(v * v for v in mem_vec.values())) or 1.0
        c_norm = math.sqrt(sum(v * v for v in curr_vec.values())) or 1.0
        vec_sim = dot / (m_norm * c_norm)

        consistency_boost = 1.0
        if vec_sim > 0.3:
            intensity = max(curr_vec.values()) if curr_vec else 0
            consistency_boost = 1.0 + 0.5 * intensity * vec_sim

        # 效价极性抑制
        mem_valence = self._compute_valence(mem_vec)
        curr_valence = self._compute_valence(curr_vec)
        opposition_penalty = 1.0
        if abs(mem_valence - curr_valence) > 1.0:
            opposition_penalty = max(0.5, 1.0 - 0.3 * abs(curr_valence))

        # 情绪特异性调制
        specific_mod = 1.0
        if curr_vec.get("joy", 0) > 0.6:
            specific_mod *= 1.2
        if curr_vec.get("sadness", 0) > 0.6:
            if "social_support" in (mem.emotional_tags or []):
                specific_mod *= 1.4

        return base_relevance * consistency_boost * opposition_penalty * specific_mod

    @staticmethod
    def _compute_valence(vec: Dict[str, float]) -> float:
        v = (vec.get("joy", 0) * 0.9
             - vec.get("anger", 0) * 0.8
             - vec.get("disgust", 0) * 0.7
             - vec.get("fear", 0) * 0.9
             - vec.get("sadness", 0) * 0.8)
        return max(-1.0, min(1.0, v / 2.5))

    def retrieve_emotion_aware(self, query: str, k: int = 10,
                               current_emotion: Optional[Dict] = None,
                               filters: Optional[Dict] = None) -> List[VRetrievalHit]:
        """情绪感知检索：先向量检索取前 2k，再用情绪调制重排取 k"""
        base = self.retrieve(query, k=2 * k, filters=filters)
        if not current_emotion or current_emotion.get("dominant") in (None, "neutral"):
            return base[:k]

        scored = []
        for hit in base:
            mod = self.emotional_modulation(hit.id, hit.score, current_emotion)
            scored.append((mod, hit))
        scored.sort(key=lambda x: -x[0])
        return [hit for _, hit in scored[:k]]

    # ═══════════════════════════════════════════════════════════
    # 复习 & 结晶
    # ═══════════════════════════════════════════════════════════

    def review(self, memory_ids: List[str], gamma: float = None) -> int:
        """复习：更新心理时间（拉回浅窗口）"""
        gamma = gamma or self.GAMMA_DEFAULT
        now = time.time()
        count = 0
        for mid in memory_ids:
            if mid in self.key_nodes:
                continue
            mem = self.memories.get(mid)
            if not mem or mem.frozen:
                continue
            mem.t_psych = gamma * now + (1 - gamma) * mem.t_psych
            mem.review_count += 1
            mem.review_count_30d += 1
            mem.avg_gamma = (mem.avg_gamma * (mem.review_count - 1) + gamma) \
                / mem.review_count
            count += 1
        return count

    def crystallize(self, memory_id: str, trigger_keywords: List[str] = None,
                    current_emotion: Optional[Dict] = None) -> Optional[str]:
        """结晶为潜意识关键节点（情绪保护 + 高频条件）"""
        # 情绪保护：极端情绪不结晶
        if current_emotion:
            vec = current_emotion.get("base_vector", current_emotion)
            if isinstance(vec, dict) and vec:
                if max(vec.values()) > 0.8:
                    if memory_id in self.memories:
                        self.memories[memory_id].pending_confirmation = True
                    return None

        mem = self.memories.get(memory_id)
        if not mem:
            return None

        can = (
            (mem.review_count_30d >= 20 and mem.avg_gamma >= 0.85)
            or mem.confidence >= 1.0  # 预留：显式声明接口
        )
        if not can:
            return None

        node_id = f"key_{len(self.key_nodes)}_{int(time.time())}"
        kws = trigger_keywords or self._top_keywords(mem, n=5)
        node = VKeyNode(
            id=node_id,
            content=mem.abstract(),
            source_memories=[memory_id],
            crystallized_at=time.time(),
            auto_trigger_keywords=kws,
            priority=mem.avg_gamma * max(1, mem.review_count_30d),
        )
        self.key_nodes[node_id] = node
        mem.crystallized_to = node_id
        mem.frozen = True
        return node_id

    def _top_keywords(self, mem: VectorMemoryEntry, n: int = 5) -> List[str]:
        """取 TF-IDF 权重最高的 n 个词作为触发词"""
        if not mem.tf:
            return []
        scored = sorted(
            ((t, w * self._idf(t)) for t, w in mem.tf.items()),
            key=lambda x: -x[1],
        )
        return [t for t, _ in scored[:n]]

    # ═══════════════════════════════════════════════════════════
    # 巩固 & 统计
    # ═══════════════════════════════════════════════════════════

    def consolidate(self, merge_threshold: float = 0.88) -> int:
        """合并深层窗口中的高度相似记忆（向量相似度）"""
        merged = 0
        mems = [m for m in self.memories.values()
                if m.window >= 6 and not m.frozen]
        for i, a in enumerate(mems):
            for b in mems[i + 1:]:
                sim = self._vector_sim(a, b)
                if sim >= merge_threshold:
                    # 合并到较新者
                    if a.t_psych >= b.t_psych:
                        keep, drop = a, b
                    else:
                        keep, drop = b, a
                    keep.content = keep.content + " / " + drop.content
                    keep.tf = build_tf(tokenize(keep.content))
                    keep.tokens = tokenize(keep.content)
                    self.memories.pop(drop.id, None)
                    merged += 1
                    break
        return merged

    def _vector_sim(self, a: VectorMemoryEntry, b: VectorMemoryEntry) -> float:
        if not a.tf or not b.tf:
            return 0.0
        dot = sum(w * b.tf.get(t, 0) for t, w in a.tf.items())
        a_n = math.sqrt(sum(v * v for v in a.tf.values()))
        b_n = math.sqrt(sum(v * v for v in b.tf.values()))
        if a_n == 0 or b_n == 0:
            return 0.0
        return dot / (a_n * b_n)

    def get_statistics(self) -> Dict:
        return {
            "total_memories": len(self.memories),
            "key_nodes": len(self.key_nodes),
            "total_terms": len(self._doc_freq),
            "interactions": self.interaction_count,
        }

    def get_key_node(self, node_id: str) -> Optional[VKeyNode]:
        return self.key_nodes.get(node_id)

    def get_all_key_nodes(self) -> List[VKeyNode]:
        return list(self.key_nodes.values())

    # ═══════════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════════

    def save_state(self):
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        state = {
            "memories": [
                {
                    "id": m.id, "content": m.content, "layer": m.layer,
                    "t_rec": m.t_rec, "t_psych": m.t_psych,
                    "recorded_emotion": m.recorded_emotion,
                    "emotional_tags": m.emotional_tags,
                    "perceptual_signature": m.perceptual_signature,
                    "tf": m.tf, "tokens": m.tokens,
                    "review_count": m.review_count,
                    "review_count_30d": m.review_count_30d,
                    "avg_gamma": m.avg_gamma,
                    "crystallized_to": m.crystallized_to,
                    "frozen": m.frozen, "confidence": m.confidence,
                    "pending_confirmation": m.pending_confirmation,
                }
                for m in self.memories.values()
            ],
            "key_nodes": [
                {
                    "id": n.id, "content": n.content,
                    "source_memories": n.source_memories,
                    "crystallized_at": n.crystallized_at,
                    "activation_count": n.activation_count,
                    "auto_trigger_keywords": n.auto_trigger_keywords,
                    "priority": n.priority,
                }
                for n in self.key_nodes.values()
            ],
            "doc_freq": self._doc_freq,
            "num_docs": self._num_docs,
            "interaction_count": self.interaction_count,
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)

    def _load_state(self):
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            for d in state.get("memories", []):
                m = VectorMemoryEntry(id=d["id"], content=d["content"])
                for k, v in d.items():
                    setattr(m, k, v)
                self.memories[m.id] = m
            for d in state.get("key_nodes", []):
                n = VKeyNode(id=d["id"], content=d["content"])
                for k, v in d.items():
                    setattr(n, k, v)
                self.key_nodes[n.id] = n
            self._doc_freq = state.get("doc_freq", {})
            self._num_docs = state.get("num_docs", 0)
            self.interaction_count = state.get("interaction_count", 0)
            self._idf_dirty = True
        except Exception:
            pass


# ── 自测 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    eng = VectorMemoryEngine()
    print("=" * 60)
    print("向量化记忆引擎自测")
    print("=" * 60)

    eng.ingest("小红很爱吃糖果，每天都吃一颗", recorded_emotion={"joy": 0.6})
    eng.ingest("用户是素食主义者，不吃任何肉类", importance="core")
    eng.ingest("用户工作到凌晨，经常熬夜", importance="core")
    eng.ingest("用户有一只叫奶盖的猫", importance="key")

    print("\n[检索] '推荐食物'")
    for hit in eng.retrieve("推荐食物", k=3):
        print(f"  {hit.score:.3f} w={hit.window} {hit.content[:30]}")
        print(f"        命中词: {hit.top_terms}")

    print("\n[检索] '今晚吃什么好'")
    for hit in eng.retrieve("今晚吃什么好", k=3):
        print(f"  {hit.score:.3f} w={hit.window} {hit.content[:30]}")
        print(f"        命中词: {hit.top_terms}")

    print("\n[检索] '熬夜对身体不好'")
    for hit in eng.retrieve("熬夜对身体不好", k=3):
        print(f"  {hit.score:.3f} w={hit.window} {hit.content[:30]}")

    print("\n[情绪调制检索] 查询'吃的'，当前情绪 joy=0.8")
    cur = {"base_vector": {"joy": 0.8, "anger": 0.0, "sadness": 0.0,
                           "fear": 0.0, "disgust": 0.0, "surprise": 0.0},
           "dominant": "joy"}
    for hit in eng.retrieve_emotion_aware("吃的", k=3, current_emotion=cur):
        print(f"  {hit.score:.3f} {hit.content[:30]}")

    print(f"\n统计: {eng.get_statistics()}")
