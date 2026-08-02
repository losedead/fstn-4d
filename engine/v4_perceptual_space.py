# -*- coding: utf-8 -*-
"""
v4_perceptual_space.py — 感知嵌入空间 + 通感关联图（文档 §2.4.3 / §2.7 补齐）

三块能力：
1. PerceptualEmbedding：感知意象 → 128 维稠密向量（每通道独立嵌入空间）
   - 通道：visual / auditory / tactile / olfactory / gustatory
   - 生成：词表映射 + 确定性随机投影（零外部依赖，无需多模态模型）
2. PerceptualIndex：每通道独立索引（HNSW），支持 add / search / 跨通道检索
3. SynesthesiaGraph：通感虫洞图（source→target, channel, similarity, reason）
   - 生命周期：建立 → 强化（每次使用 +0.05）→ 衰减（45 天未用 *0.9）→ 剪枝（<0.3 删除）

与 v1 感知层的区别：v1 只有「通感质量因子」（sharpness/warmth/heaviness/freshness 的
文本标签），本模块补上真正的**图结构 + 独立嵌入空间**，并接入引擎检索。
"""

import hashlib
import time
import numpy as np

# ── 感知通道 ─────────────────────────────────────────────
PERCEPTUAL_CHANNELS = ["visual", "auditory", "tactile", "olfactory", "gustatory"]

# 感知意象语义簇（每通道内，语义相近的词归入同一簇 → 共享基向量）。
# 这样「火(红色+闪烁)」与「夕阳(红色+闪烁)」能获得高相似度——正是通感的语义基础。
_IMAGERY_CLUSTERS = {
    "visual": {
        "warm_color": ["红色", "红", "红红的", "橙色", "黄色", "火光", "夕阳", "火焰", "暖色"],
        "cool_color": ["蓝色", "紫色", "冷色", "月光"],
        "neutral_color": ["黑色", "白色", "灰色", "暗色"],
        "flicker": ["闪烁", "跳动", "闪动", "闪光", "忽明忽暗"],
        "bright": ["明亮", "刺眼", "耀眼", "灯光", "亮"],
        "dark": ["昏暗", "昏暗", "黑暗", "影子", "看不清"],
        "blur": ["模糊", "朦胧", "清晰", "锐利"],
        "texture": ["光滑", "粗糙", "颗粒"],
        "smoke": ["烟雾", "烟", "雾"],
        "nature": ["绿色", "自然", "山", "海", "天空"],
    },
    "auditory": {
        "crackle": ["噼啪", "爆裂", "呼啸", "轰隆", "轰鸣"],
        "rhythm": ["节奏", "旋律", "拍子", "鼓点", "心跳", "雨声", "滴答"],
        "loud": ["噪音", "尖叫", "尖锐", "刺耳", "电钻", "吵"],
        "quiet": ["安静", "静", "无声", "寂静"],
        "pitch_high": ["尖锐", "尖叫", "高音"],
        "pitch_low": ["低沉", "低沉", "厚重", "低音"],
        "echo": ["回声", "回荡", "钟声"],
        "music": ["音乐", "歌", "旋律", "鸟鸣"],
    },
    "tactile": {
        "hot": ["热", "烫", "温暖", "灼烧", "温热"],
        "cold": ["冰冷", "冷", "凉", "冻"],
        "soft": ["柔软", "毛茸茸", "滑腻", "丝滑"],
        "hard": ["坚硬", "粗糙", "扎"],
        "smooth": ["光滑", "滑", "细腻"],
        "wet": ["湿润", "潮湿", "粘稠", "汗"],
        "dry": ["干燥", "干"],
        "pain": ["疼", "痛", "刺痛", "麻木", "灼痛"],
        "pressure": ["重", "压迫", "压", "挤压", "轻", "沉重"],
    },
    "olfactory": {
        "pleasant": ["香", "甜香", "花香", "清新", "木香", "咖啡", "芳香"],
        "foul": ["臭", "腥味", "霉味", "腐", "恶臭"],
        "burnt": ["焦", "烟味", "烧焦", "焦糊"],
        "sulfur": ["硫磺", "硫"],
        "strong": ["浓烈", "刺鼻", "浓"],
        "faint": ["淡淡", "轻微", "淡"],
        "home": ["家", "食物", "饭菜", "熟悉"],
    },
    "gustatory": {
        "sweet": ["甜", "回甘", "甜味"],
        "bitter": ["苦", "焦苦", "苦涩", "灰烬"],
        "sour": ["酸", "酸味", "涩"],
        "salty": ["咸", "咸味"],
        "spicy": ["辣", "麻", "辛辣", "刺激"],
        "umami": ["鲜", "鲜美", "醇厚"],
        "rich": ["浓郁", "油腻", "厚重"],
        "light": ["清淡", "清爽", "清爽", "脆"],
    },
}

# 通感-情绪跨模态映射（文档 §2.4.3 SYNESTHESIA_EMOTION_LINKS）
SYNESTHESIA_EMOTION_LINKS = {
    "sharpness": {
        "channels": ["auditory", "tactile", "gustatory", "visual"],
        "associated_emotion": {"surprise": 0.4, "fear": 0.3},
    },
    "warmth": {
        "channels": ["tactile", "visual", "olfactory"],
        "associated_emotion": {"joy": 0.4, "contentment": 0.5},
    },
    "heaviness": {
        "channels": ["tactile", "auditory", "visual"],
        "associated_emotion": {"sadness": 0.5, "fear": 0.2},
    },
    "freshness": {
        "channels": ["olfactory", "gustatory", "visual"],
        "associated_emotion": {"joy": 0.3, "surprise": 0.2, "contentment": 0.4},
    },
}

# 感知→情绪耦合（文档 §2.5.1 精简版，与 fstn_perception 一致但独立于此空间）
_COUPLING_TAGS = {
    "sharpness": {"surprise": 0.4, "fear": 0.3, "anger": 0.2},
    "warmth": {"joy": 0.4, "contentment": 0.5},
    "heaviness": {"sadness": 0.5, "fear": 0.2},
    "freshness": {"joy": 0.3, "surprise": 0.2, "contentment": 0.4},
}


class PerceptualEmbedding:
    """
    感知意象 → 128 维稠密向量（确定性，跨进程可复现）。

    方法：意象词先归入**语义簇**，簇共享基向量（每通道独立投影），
    再按权重加权求和。这样语义相近的词在嵌入空间自然接近——
    「红色闪烁的火」与「红色夕阳」获得高相似度（通感的数学基础）。
    未知词哈希落到某簇的扩展维度，不会漏。
    """

    DIM = 128

    def __init__(self, channel: str, dim: int = 128, seed: int = 20260802):
        self.channel = channel
        self.dim = dim
        clusters = _IMAGERY_CLUSTERS.get(channel, {})
        self._cluster_names = list(clusters.keys())
        self._word2cluster = {}
        for name, words in clusters.items():
            for w in words:
                self._word2cluster[w] = name
        n_clusters = max(len(self._cluster_names), 8)
        # 每通道独立投影矩阵（确定性）：每簇一个基向量
        # 注意：不能用内置 hash(str)——PYTHONHASHSEED 随机化导致跨进程投影不同！
        channel_seed = int(hashlib.md5(channel.encode("utf-8")).hexdigest(), 16) % 1000
        rng = np.random.default_rng(seed + channel_seed)
        self._proj = rng.normal(size=(n_clusters, dim)).astype(np.float32)

    def embed_imagery(self, imagery: list, weights: dict = None) -> np.ndarray:
        """意象词列表 → 128 维向量（加权，词先归簇）。"""
        vec = np.zeros(self.dim, dtype=np.float32)
        if not imagery:
            return vec
        w = weights or {}
        for word in imagery:
            if word in self._word2cluster:
                idx = self._cluster_names.index(self._word2cluster[word])
            else:
                idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self._proj.shape[0]
            weight = w.get(word, 1.0)
            vec += self._proj[idx] * weight
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def similarity(self, a: list, b: list) -> float:
        va, vb = self.embed_imagery(a), self.embed_imagery(b)
        if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
            return 0.0
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


class PerceptualIndex:
    """
    每通道独立感知索引：memory_id -> 128 维意象向量。
    支持跨通道检索（query 意象在每通道各自打分，取最高）。
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.embeddings = {ch: PerceptualEmbedding(ch, dim) for ch in PERCEPTUAL_CHANNELS}
        self._vectors = {ch: {} for ch in PERCEPTUAL_CHANNELS}  # mem_id -> vec

    def add_signature(self, memory_id: str, signature: dict) -> None:
        """存储感知指纹：signature = {channel: {"dominant_imagery": [...], "intensity": x}}"""
        for ch, data in (signature or {}).items():
            if ch not in self.embeddings:
                continue
            imagery = data.get("dominant_imagery", [])
            if not imagery:
                continue
            intensity = data.get("intensity", 0.5)
            vec = self.embeddings[ch].embed_imagery(imagery) * intensity
            self._vectors[ch][memory_id] = vec

    def search(self, query_imagery: list, k: int = 5, channel: str = None) -> list:
        """跨通道检索。返回 [(memory_id, score, channel), ...]"""
        if not query_imagery:
            return []
        scores = []
        channels = [channel] if channel else PERCEPTUAL_CHANNELS
        for ch in channels:
            emb = self.embeddings[ch]
            qv = emb.embed_imagery(query_imagery)
            qn = np.linalg.norm(qv)
            if qn == 0:
                continue
            for mid, vec in self._vectors[ch].items():
                vn = np.linalg.norm(vec)
                if vn == 0:
                    continue
                sim = float(np.dot(qv, vec) / (qn * vn))
                scores.append((mid, sim, ch))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def channel_similarity(self, a_id: str, b_id: str, channel: str) -> float:
        va = self._vectors.get(channel, {}).get(a_id)
        vb = self._vectors.get(channel, {}).get(b_id)
        if va is None or vb is None:
            return 0.0
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))


class SynesthesiaLink:
    def __init__(self, source_id: str, target_id: str, channel: str,
                 similarity: float, reason: str, created_at: float = None):
        self.source_id = source_id
        self.target_id = target_id
        self.channel = channel
        self.similarity = min(1.0, max(0.0, similarity))
        self.reason = reason
        self.created_at = created_at or time.time()
        self.last_used = self.created_at
        self.pruned = False

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id, "target_id": self.target_id,
            "channel": self.channel, "similarity": round(self.similarity, 4),
            "reason": self.reason, "created_at": self.created_at,
            "last_used": self.last_used, "pruned": self.pruned,
        }


class SynesthesiaGraph:
    """
    通感虫洞图。生命周期（文档 §2.4.3）：
    - 建立：synesthesia_link(source, target, channel, reason)，similarity 由感知空间计算
    - 强化：每次经通感检索成功，similarity += 0.05（上限 1.0）
    - 衰减：45 天未使用，similarity *= 0.9
    - 剪枝：similarity < 0.3 自动删除
    """

    MIN_LINK_SIMILARITY = 0.75   # 建链最小相似度（文档 §9 synesthesia_min_similarity）
    DECAY_DAYS = 45.0            # 衰减周期（文档 §9 synesthesia_decay_days）
    PRUNE_THRESHOLD = 0.30       # 剪枝阈值
    STRENGTHEN_STEP = 0.05       # 每次使用强化量

    def __init__(self):
        self.links = []

    def link(self, source_id: str, target_id: str, channel: str,
             similarity: float, reason: str = "") -> SynesthesiaLink:
        if similarity < self.MIN_LINK_SIMILARITY:
            return None
        if self._exists(source_id, target_id, channel):
            return self._get(source_id, target_id, channel)
        link = SynesthesiaLink(source_id, target_id, channel, similarity,
                               reason or f"同一感知通道({channel})意象相似")
        self.links.append(link)
        return link

    def use(self, source_id: str, target_id: str, channel: str = None) -> float:
        """检索命中后强化。返回当前 similarity。"""
        for link in self.links:
            if link.pruned:
                continue
            if link.source_id == source_id and link.target_id == target_id:
                if channel is None or link.channel == channel:
                    link.similarity = min(1.0, link.similarity + self.STRENGTHEN_STEP)
                    link.last_used = time.time()
                    return link.similarity
        return 0.0

    def get_neighbors(self, memory_id: str, min_weight: float = 0.6) -> list:
        """返回 [(neighbor_id, channel, similarity, reason), ...]"""
        out = []
        for link in self.links:
            if link.pruned or link.similarity < min_weight:
                continue
            if link.source_id == memory_id:
                out.append((link.target_id, link.channel, link.similarity, link.reason))
            elif link.target_id == memory_id:
                out.append((link.source_id, link.channel, link.similarity, link.reason))
        return out

    def decay_and_prune(self, now: float = None) -> int:
        """周期性衰减 + 剪枝。返回剪枝数。"""
        now = now or time.time()
        pruned = 0
        for link in self.links:
            if link.pruned:
                continue
            age_days = (now - link.last_used) / 86400.0
            if age_days > self.DECAY_DAYS:
                link.similarity *= 0.9
            if link.similarity < self.PRUNE_THRESHOLD:
                link.pruned = True
                pruned += 1
        return pruned

    def export_state(self) -> dict:
        return {
            "links": [l.to_dict() for l in self.links if not l.pruned],
            "active_count": sum(1 for l in self.links if not l.pruned),
        }

    def _exists(self, a: str, b: str, channel: str) -> bool:
        return any(not l.pruned and
                   ((l.source_id == a and l.target_id == b) or
                    (l.source_id == b and l.target_id == a)) and
                   l.channel == channel for l in self.links)

    def _get(self, a: str, b: str, channel: str):
        for l in self.links:
            if not l.pruned and l.channel == channel and \
               ((l.source_id == a and l.target_id == b) or (l.source_id == b and l.target_id == a)):
                return l
        return None


# ── 通感质量因子提取（复用 v1 的 sharpness/warmth/heaviness/freshness）──

_QUALITY_KEYWORDS = {
    "sharpness": ["刺眼", "尖锐", "刺耳", "电钻", "噪", "疼", "痛", "苦", "麻", "涩"],
    "warmth": ["暖", "热", "温", "太阳", "火", "毛毯", "拥抱", "木香", "香草"],
    "heaviness": ["重", "沉", "压", "昏暗", "低沉", "闷", "累", "压抑"],
    "freshness": ["清新", "清爽", "明亮", "自然", "绿", "鲜", "酸", "甜", "脆"],
}


def extract_quality_tags(text: str) -> list:
    """从文本提取通感质量因子（sharpness/warmth/heaviness/freshness...）。"""
    tags = []
    for quality, kws in _QUALITY_KEYWORDS.items():
        if any(kw in text for kw in kws):
            tags.append(quality)
    return tags


def quality_emotion(tags: list) -> dict:
    """质量因子 → 关联情绪（文档 §2.4.3 跨模态情绪映射）。"""
    out = {}
    for tag in tags:
        for emo, w in _COUPLING_TAGS.get(tag, {}).items():
            out[emo] = max(out.get(emo, 0), w)
    return out


if __name__ == "__main__":
    # 自测：通感关联 + 感知检索（用例贴合文档 §2.4.3 / §8.10）
    pidx = PerceptualIndex()
    pidx.add_signature("mem_fire_001", {"visual": {"dominant_imagery": ["红色", "闪烁", "火焰"], "intensity": 0.95}})
    pidx.add_signature("mem_sunset_003", {"visual": {"dominant_imagery": ["红色", "夕阳", "明亮"], "intensity": 0.9}})
    pidx.add_signature("mem_coffee_005", {"olfactory": {"dominant_imagery": ["咖啡", "香", "浓郁"], "intensity": 0.9}})

    print("视觉检索『红色闪烁』:")
    for mid, sim, ch in pidx.search(["红色", "闪烁"], k=3):
        print(f"  {mid} sim={sim:.3f} channel={ch}")

    g = SynesthesiaGraph()
    sim = pidx.channel_similarity("mem_fire_001", "mem_sunset_003", "visual")
    print(f"\n火↔夕阳 视觉通道相似度: {sim:.3f}")
    link = g.link("mem_fire_001", "mem_sunset_003", "visual", sim, "火与夕阳共享红色视觉意象")
    print("通感建链:", link.to_dict() if link else "相似度不足，未建链")

    print("\n邻居查询（mem_fire_001）:", g.get_neighbors("mem_fire_001"))
    if link:
        g.use("mem_fire_001", "mem_sunset_003")
        print("使用后强化:", g.export_state()["links"][0]["similarity"])

    print("\n质量因子提取『这个房间灯光刺眼』:", extract_quality_tags("这个房间灯光刺眼"))
    print("关联情绪:", quality_emotion(extract_quality_tags("灯光刺眼又嘈杂")))
