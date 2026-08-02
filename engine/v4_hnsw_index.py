# -*- coding: utf-8 -*-
"""
v4_hnsw_index.py — HNSW 近似最近邻索引（文档 §2.7 补齐）

背景：v1/v2 的语义检索是「线性扫描」（对全部记忆逐个算相似度），
     文档要求 HNSW 近似最近邻（ANN），支持百万级条目毫秒检索。

实现：faiss IndexHNSWFlat（余弦 ≈ 归一化向量 + L2）。
设计：
  - 内存索引 + 外部 id 映射（memory_id 字符串 ↔ faiss 内部序号）
  - 增量 add / 批量 rebuild / search
  - 优雅降级：faiss 不可用 → 纯 numpy 线性扫描兜底（接口一致）
  - 向量进索引前 L2 归一化 → 检索距离可换算为余弦相似度

依赖：faiss-cpu（可选），numpy（必需）。
"""

import time
import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """按行 L2 归一化（零向量保持为零）"""
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class HNSWMemoryIndex:
    """
    HNSW 向量索引，面向 FSTN-4D 的记忆条目。

    >>> idx = HNSWMemoryIndex(dim=128)
    >>> idx.add("mem_1", vec1)
    >>> idx.search(vec_q, k=5) -> [("mem_3", 0.92), ...]
    """

    def __init__(self, dim: int = 768, M: int = 32,
                 ef_construction: int = 200, ef_search: int = 64):
        self.dim = int(dim)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._ext_ids: list = []          # 外部 memory_id，顺序即 faiss 内部序号
        self._id2pos: dict = {}           # memory_id -> faiss 内部 pos
        self._index = None
        if _FAISS_AVAILABLE:
            self._build_faiss()
        else:
            self._vecs = np.zeros((0, self.dim), dtype=np.float32)

    # ── 构建 ─────────────────────────────────────────────

    def _build_faiss(self):
        idx = faiss.IndexHNSWFlat(self.dim, self.M)
        idx.hnsw.efConstruction = self.ef_construction
        idx.hnsw.efSearch = self.ef_search
        self._index = idx

    def add(self, ext_id: str, vec: np.ndarray) -> None:
        """增量添加单条。"""
        if ext_id in self._id2pos:
            # 更新：先删旧位（HNSW 不支持真删，标记后重建）
            return self._update_existing(ext_id, vec)
        v = _l2_normalize(np.asarray([vec], dtype=np.float32))
        if _FAISS_AVAILABLE:
            self._index.add(v)
        else:
            self._vecs = np.vstack([self._vecs, v])
        pos = len(self._ext_ids)
        self._ext_ids.append(ext_id)
        self._id2pos[ext_id] = pos

    def _update_existing(self, ext_id: str, vec: np.ndarray) -> None:
        """更新已有向量：faiss 不支持 remove，采用『重建』策略（低频操作，可接受）。"""
        pos = self._id2pos[ext_id]
        v = _l2_normalize(np.asarray([vec], dtype=np.float32))
        if _FAISS_AVAILABLE:
            # faiss IndexHNSWFlat 不支持单点 remove；收集全部重建
            self.rebuild_all()
        else:
            self._vecs[pos] = v

    def add_batch(self, pairs) -> None:
        """批量添加 [(ext_id, vec), ...]"""
        for ext_id, vec in pairs:
            self.add(ext_id, vec)

    def rebuild_all(self) -> None:
        """用当前数据全量重建 faiss 索引（增量 add 后 HNSW 性能会退化，定期重建）。

        注意顺序：必须先 collect（从旧索引 reconstruct），再建新索引并 add。
        否则 _build_faiss() 已把 self._index 换成空索引，reconstruct 全为零。
        """
        if not _FAISS_AVAILABLE:
            return
        n = len(self._ext_ids)
        if n == 0:
            self._build_faiss()
            return
        vecs = self._collect_vecs()
        old_index = self._index
        self._build_faiss()
        if vecs is not None and len(vecs):
            self._index.add(np.asarray(vecs, dtype=np.float32))

    def _collect_vecs(self) -> np.ndarray:
        if _FAISS_AVAILABLE:
            # 从当前索引 reconstruct 所有向量（供全量重建）
            n = len(self._ext_ids)
            out = np.zeros((n, self.dim), dtype=np.float32)
            for i in range(n):
                try:
                    out[i] = self._index.reconstruct(i)
                except Exception:
                    pass
            return out
        return self._vecs

    # ── 检索 ─────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, k: int = 10) -> list:
        """
        返回 [(ext_id, cosine_similarity), ...]，按相似度降序。
        cosine ≈ 归一化后 L2 距离的换算：sim = 1 - d²/2
        """
        if not self._ext_ids:
            return []
        q = _l2_normalize(np.asarray([query_vec], dtype=np.float32))
        k = min(k, len(self._ext_ids))
        if _FAISS_AVAILABLE:
            D, I = self._index.search(q, k)
            dists = D[0]
            ids = I[0]
            results = []
            for dist, pos in zip(dists, ids):
                if pos == -1 or pos >= len(self._ext_ids):
                    continue
                sim = max(0.0, min(1.0, 1.0 - float(dist) ** 2 / 2.0))
                results.append((self._ext_ids[pos], sim))
            return results
        # 纯 numpy 兜底：线性扫描（接口一致）
        dots = self._vecs @ q.T
        sims = dots[:, 0]
        top = np.argsort(-sims)[:k]
        return [(self._ext_ids[i], float(sims[i])) for i in top]

    # ── 状态 ─────────────────────────────────────────────

    def size(self) -> int:
        return len(self._ext_ids)

    def clear(self) -> None:
        self._ext_ids = []
        self._id2pos = {}
        if _FAISS_AVAILABLE:
            self._build_faiss()
        else:
            self._vecs = np.zeros((0, self.dim), dtype=np.float32)

    def export_state(self) -> dict:
        return {
            "dim": self.dim, "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "size": len(self._ext_ids),
            "backend": "faiss" if _FAISS_AVAILABLE else "numpy-linear",
        }

    # ── 性能基准 ─────────────────────────────────────────

    def benchmark(self, n_queries: int = 100) -> dict:
        """对当前索引做性能自测，返回平均检索耗时。"""
        if self.size() == 0:
            return {"error": "empty index"}
        vecs = self._collect_vecs()
        rng = np.random.default_rng(42)
        idxs = rng.integers(0, len(vecs), n_queries)
        t0 = time.perf_counter()
        for i in idxs:
            self.search(vecs[i], k=10)
        elapsed = time.perf_counter() - t0
        return {
            "size": self.size(),
            "queries": n_queries,
            "avg_ms": elapsed / n_queries * 1000.0,
            "backend": "faiss" if _FAISS_AVAILABLE else "numpy-linear",
        }


if __name__ == "__main__":
    # 自测：随机 1 万条 128 维向量，验证召回正确性 + 计时
    rng = np.random.default_rng(0)
    idx = HNSWMemoryIndex(dim=128)
    vecs = rng.normal(size=(10000, 128)).astype(np.float32)
    for i, v in enumerate(vecs):
        idx.add(f"mem_{i}", v)
    idx.rebuild_all()
    print("索引大小:", idx.size(), "| 后端:", idx.export_state()["backend"])

    # 正确性：最近邻应与线性扫描一致（同一归一化口径，top1 命中率）
    linear_hits = 0
    total = 200
    for i in range(total):
        q = vecs[rng.integers(0, 10000)]
        hnsw_top = idx.search(q, k=1)[0][0]
        dots = idx._collect_vecs() @ q  # 与 HNSW 同一批归一化向量
        linear_top = f"mem_{int(np.argmax(dots))}"
        if hnsw_top == linear_top:
            linear_hits += 1
    print(f"top1 与线性扫描一致率: {linear_hits}/{total} = {linear_hits/total:.1%}")

    bench = idx.benchmark(n_queries=500)
    print(f"检索耗时: {bench['avg_ms']:.4f} ms/query（{bench['size']} 条）")
