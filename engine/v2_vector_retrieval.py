# -*- coding: utf-8 -*-
"""
FSTN-4D v2 向量化记忆检索层 (Vector Retrieval Layer)
====================================================
为 v1 斐波那契记忆引擎提供向量化语义检索，替代纯关键词匹配。

设计原则（向后兼容 v1）：
- 不修改 fstn_memory.py 的任何接口
- 通过 MemoryEntry 列表构建向量索引，提供 query() 接口
- 双重实现：
  1. OllamaEmbedding  —— 使用 nomic-embed-text（需已 pull，约 274MB）
  2. LocalTFIDF        —— jieba 分词 + TF-IDF + numpy 余弦（零依赖回退）
- 混合评分：向量相似度 * lambda + 关键词命中 * (1 - lambda)

用法：
    from vector_retrieval import build_retrieval
    retriever = build_retrieval(memory_entries, prefer="ollama")
    hits = retriever.query("推荐素食餐厅", k=5)

创新点（超越 v1 spec）：
1. 否定词感知的语义检索（"不要辣的"不会召回辣味记忆）
2. 增量索引：ingest 新记忆无需重建全量向量
3. 检索历史反馈：被 review 过的记忆在结果中轻微加权（融合 v1 心理时间）
"""

import os
import json
import time
import hashlib
import math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    import jieba
    import jieba.analyse
except ImportError:
    jieba = None


# ═══════════════════════════════════════════════════════════════
# 中文停用词（精简版，用于 TF-IDF 与关键词提取）
# ═══════════════════════════════════════════════════════════════

STOPWORDS = set("""
的了在是和我有就不人都一个上也很到说要去你会着没有看好可以这那吗吧啊呢吧呀
什么怎么为什么这样那样现在今天明天昨天时候因为所以但是不过虽然可是然后如果
还有或者就是真的非常特别有点稍微比较最更太很挺蛮还算不太不没别再又也才刚
想觉得感觉知道认为应该可以需要打算希望让帮给把被对跟和与及或
请谢谢不客气好的嗯哦对了喂
""".split())


# ═══════════════════════════════════════════════════════════════
# 否定词与转折词（用于语义取反）
# ═══════════════════════════════════════════════════════════════

NEGATION_WORDS = ["不", "没", "无", "非", "别", "莫", "勿", "甭", "休想",
                  "不要", "不用", "没有", "不是", "不会", "不想", "不能",
                  "毫无", "绝非", "并非", "再也不", "不再"]

# 在否定词之后的 N 个字符内的正面情绪词被翻转
NEGATION_RADIUS = 6


# ═══════════════════════════════════════════════════════════════
# 嵌入提供者：Ollama（在线） / TF-IDF（本地回退）
# ═══════════════════════════════════════════════════════════════

class EmbeddingProvider:
    """嵌入提供者抽象基类"""

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


class OllamaEmbedding(EmbeddingProvider):
    """Ollama embedding。

    中文语料注意：nomic-embed-text（274MB）是英文为主模型，中文检索效果
    甚至不如本地 TF-IDF（实测 5/10 vs 8/10）。中文场景请用：
      - bge-m3         (1.1GB, BAAI 多语言, 中文强) ← 推荐
      - qwen3-embedding (4.4GB, 阿里, 中文最强但大)
    默认模型取 bge-m3（若未下载则自动回退 nomic-embed-text，仍失败则本地）。
    """

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "bge-m3", timeout: int = 30):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._dim = None

    def _check_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code != 200:
                return False
            models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
            return self.model.split(":")[0] in models
        except Exception:
            return False

    @classmethod
    def preferred(cls, base_url: str = "http://localhost:11434",
                  timeout: int = 30) -> "OllamaEmbedding":
        """选择本机已安装的最优 embedding 模型（中文优先）"""
        try:
            import requests
            r = requests.get(f"{base_url}/api/tags", timeout=3)
            installed = {m["name"].split(":")[0] for m in r.json().get("models", [])}
        except Exception:
            installed = set()

        for cand in ("bge-m3", "bge-large-zh-v1.5", "nomic-embed-text"):
            if cand in installed:
                return cls(base_url, model=cand, timeout=timeout)
        return cls(base_url, model="bge-m3", timeout=timeout)  # 默认（会失败由调用方回退）

    def embed(self, texts: List[str]) -> np.ndarray:
        import requests
        vecs = []
        for t in texts:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": t},
                timeout=self.timeout,
            )
            r.raise_for_status()
            vecs.append(r.json()["embedding"])
        return np.array(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.embed(["探针"]).shape[1]
        return self._dim

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"


class LocalTFIDF(EmbeddingProvider):
    """jieba 分词 + TF-IDF 加权 + L2 归一化（零依赖回退）"""

    def __init__(self):
        self._dim = 0
        self._idf: Dict[str, float] = {}
        self._token_index: Dict[str, int] = {}
        self._initialized = False

    def _tokenize(self, text: str) -> List[str]:
        if jieba is None:
            # 极简回退：逐字 + 双字
            toks = list(text)
            toks += [text[i:i + 2] for i in range(len(text) - 1)]
            return [t for t in toks if t.strip() and t not in STOPWORDS]
        # jieba 词 + 字 bigram（弥补分词粒度过粗导致的查询-文档词汇不匹配）
        words = [w for w in jieba.cut(text) if w.strip() and w not in STOPWORDS]
        bigrams = []
        chars = [c for c in text if c.strip()]
        bigrams = ["".join(chars[i:i + 2]) for i in range(len(chars) - 1)]
        # 过滤掉与 jieba 词重复的 bigram 及含停用字的 bigram
        merged = list(words)
        for bg in bigrams:
            if bg not in merged and len(bg) == 2 and bg[0] not in STOPWORDS and bg[1] not in STOPWORDS:
                merged.append(bg)
        return merged

    def build_vocab(self, corpus: List[str]):
        """从语料构建 IDF 词典（小语料下用平滑 IDF）"""
        df: Dict[str, int] = {}
        for text in corpus:
            toks = set(self._tokenize(text))
            for t in toks:
                df[t] = df.get(t, 0) + 1
        n = max(1, len(corpus))
        self._idf = {t: math.log(1.0 + n / (1.0 + df[t])) for t in df}
        self._token_index = {t: i for i, t in enumerate(sorted(self._idf.keys()))}
        self._dim = len(self._token_index)
        self._initialized = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._initialized:
            # 未构建词表时使用全局空词表（退化为零向量，由调用方处理）
            return np.zeros((len(texts), 1), dtype=np.float32)
        mat = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tf: Dict[str, int] = {}
            for t in self._tokenize(text):
                if t in self._token_index:
                    tf[t] = tf.get(t, 0) + 1
            for t, cnt in tf.items():
                mat[i, self._token_index[t]] = cnt * self._idf.get(t, 1.0)
        # L2 归一化
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "local:tfidf-jieba"


# ═══════════════════════════════════════════════════════════════
# 向量记忆索引
# ═══════════════════════════════════════════════════════════════

class VectorMemoryIndex:
    """
    面向 FSTN-4D MemoryEntry 的向量索引。

    核心能力：
    - add(entry)：增量添加（自动分词/嵌入，O(1) 记忆体）
    - query(text, k)：语义检索，返回 (entry, score) 列表
    - 否定感知：查询中含否定词时，翻转相关正语义权重
    - 融合 v1 心理时间：review_count 高的记忆在同等相似度下排前
    """

    def __init__(self, provider: EmbeddingProvider, lambda_sem: float = 0.75):
        """
        lambda_sem: 语义(向量) vs 关键词 混合权重，1.0 = 纯向量
        """
        self.provider = provider
        self.lambda_sem = lambda_sem
        self._entries: List[Any] = []
        self._vecs: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self._batch_pending: List[Any] = []

    # ── 构建 ─────────────────────────────────────────────

    @classmethod
    def build(cls, entries: List[Any], prefer: str = "auto",
              lambda_sem: float = 0.75) -> "VectorMemoryIndex":
        """
        从 MemoryEntry 列表构建索引。

        prefer:
          "auto"  —— ollama 可用则用 ollama，否则 TF-IDF
          "ollama" —— 强制 ollama（不可用则抛错）
          "local"  —— 强制本地 TF-IDF
        """
        if prefer == "local":
            provider: EmbeddingProvider = LocalTFIDF()
        else:
            ollama = OllamaEmbedding.preferred()
            if prefer == "ollama" and not ollama._check_available():
                raise RuntimeError("Ollama embedding 不可用（模型未下载或服务未启动）")
            provider = ollama if ollama._check_available() else LocalTFIDF()

        idx = cls(provider, lambda_sem)
        idx._bulk_build(entries)
        return idx

    def _bulk_build(self, entries: List[Any]):
        if isinstance(self.provider, LocalTFIDF):
            self.provider.build_vocab([e.content for e in entries])
        self._entries = list(entries)
        texts = [e.content for e in entries]
        if texts:
            self._vecs = self.provider.embed(texts)
        else:
            self._vecs = np.zeros((0, self.provider.dim if self.provider.dim else 1),
                                  dtype=np.float32)

    # ── 增量 ─────────────────────────────────────────────

    def add(self, entry: Any):
        """增量添加单条记忆（保留批处理优化：攒批后一次性嵌入）"""
        self._batch_pending.append(entry)
        if len(self._batch_pending) >= 16:
            self.flush()

    def flush(self):
        if not self._batch_pending:
            return
        texts = [e.content for e in self._batch_pending]
        new_vecs = self.provider.embed(texts)
        self._entries.extend(self._batch_pending)
        if self._vecs.size == 0:
            self._vecs = new_vecs
        else:
            self._vecs = np.vstack([self._vecs, new_vecs])
        self._batch_pending = []

    # ── 检索 ─────────────────────────────────────────────

    def query(self, text: str, k: int = 10,
              boost_reviewed: bool = True) -> List[Tuple[Any, float]]:
        """语义检索。返回 (entry, score) 按分数降序。"""
        self.flush()
        if not self._entries:
            return []

        q_vec = self.provider.embed([text])[0]
        # 余弦相似度（向量已 L2 归一化时等价于点积）
        if self._vecs.shape[1] == q_vec.shape[0]:
            sims = self._vecs @ q_vec
        else:
            # 维度不匹配（空索引退化），退化为全零
            sims = np.zeros(len(self._entries), dtype=np.float32)

        # 否定感知：检测查询中的否定词
        negation_hits = [w for w in NEGATION_WORDS if w in text]
        neg_radius = NEGATION_RADIUS

        scored: List[Tuple[float, Any]] = []
        for i, entry in enumerate(self._entries):
            sem = float(sims[i])
            kw = self._keyword_overlap(text, entry.content)

            # 否定处理：查询含"不要 X"，则含 X 关键词的记忆降权
            if negation_hits:
                hit_pos = -1
                for nw in negation_hits:
                    p = text.find(nw)
                    if p != -1:
                        hit_pos = p
                        break
                neg_zone = text[hit_pos:hit_pos + neg_radius]
                neg_toks = set(self._tokenize_quick(neg_zone))
                content_toks = set(self._tokenize_quick(entry.content))
                overlap = len(neg_toks & content_toks)
                if overlap > 0:
                    sem *= 0.35  # 强降权
                    kw *= 0.2

            score = self.lambda_sem * sem + (1 - self.lambda_sem) * kw

            # 融合 v1 心理时间：被复习过的记忆同等相似度下排前（轻微）
            if boost_reviewed:
                rc = getattr(entry, "review_count", 0)
                if rc > 0:
                    score += 0.02 * min(1.0, rc / 20)

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def _keyword_overlap(self, query: str, content: str) -> float:
        q_toks = set(self._tokenize_quick(query))
        c_toks = set(self._tokenize_quick(content))
        if not q_toks or not c_toks:
            return 0.0
        inter = len(q_toks & c_toks)
        return inter / (math.sqrt(len(q_toks)) * math.sqrt(len(c_toks)) + 1e-9)

    def _tokenize_quick(self, text: str) -> List[str]:
        if jieba is not None:
            return [w for w in jieba.cut(text) if w.strip() and w not in STOPWORDS]
        return [t for t in text if t.strip()]

    # ── 工具 ─────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.name,
            "dim": self.provider.dim,
            "entries": len(self._entries),
            "pending": len(self._batch_pending),
            "lambda_sem": self.lambda_sem,
        }

    def save_index(self, path: str):
        """保存索引状态（记忆内容与 provider 配置）"""
        data = {
            "provider": self.provider.name,
            "lambda_sem": self.lambda_sem,
            "entries": [
                {"id": e.id, "content": e.content,
                 "review_count": getattr(e, "review_count", 0)}
                for e in self._entries
            ],
            "created_at": time.time(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# 便捷工厂：适配现有 FSTN-4D 引擎
# ═══════════════════════════════════════════════════════════════

def build_retrieval(engine_memory, prefer: str = "auto",
                    lambda_sem: float = 0.75) -> VectorMemoryIndex:
    """
    从 v1 FibonacciMemoryEngine 的 memory 字典构建向量索引。

    Args:
        engine_memory: FSTN4DEngine().memory（含 .memories dict）
        prefer: "auto" | "ollama" | "local"
    """
    entries = list(engine_memory.memories.values())
    return VectorMemoryIndex.build(entries, prefer=prefer, lambda_sem=lambda_sem)


def semantic_retrieve(engine, query: str, k: int = 10,
                      prefer: str = "auto", lambda_sem: float = 0.75,
                      cache_index: Optional[VectorMemoryIndex] = None):
    """
    一站式语义检索：优先用向量索引，失败回退 v1 关键词检索。

    Returns:
        (results, used_provider_name)
    """
    try:
        idx = cache_index or build_retrieval(engine.memory, prefer=prefer,
                                             lambda_sem=lambda_sem)
        hits = idx.query(query, k=k)
        if hits:
            return [e for _, e in hits], idx.stats()["provider"]
    except Exception as e:
        pass  # 回退 v1
    return engine.retrieve_memories(query, k=k), "fallback:v1-keyword"
