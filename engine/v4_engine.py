# -*- coding: utf-8 -*-
"""
v4_engine.py — FSTN-4D v4 集成引擎（文档 Ultimate 版全量补齐）

在 v2 基础上新增（对应文档缺口）：
1. HNSW 语义索引（v4_hnsw_index）替换线性扫描
2. 感知嵌入空间 + 跨通道检索（v4_perceptual_space.PerceptualIndex）
3. 通感关联图（v4_perceptual_space.SynesthesiaGraph）+ 通感扩散检索
4. 四路融合排序：语义 0.5 + 感知 0.3 + 虫洞 0.1 + 通感 0.1（文档 §5.5）

接口与 v1/v2 完全兼容：FSTN4DEngineV4 可 drop-in 替换 FSTN4DEngineV2。
"""

import os
import time
from typing import Dict, List, Optional

from fstn_core import FSTN4DEngine
from v2_engine import FSTN4DEngineV2
from v4_hnsw_index import HNSWMemoryIndex
from v4_perceptual_space import (
    PERCEPTUAL_CHANNELS,
    PerceptualIndex,
    SynesthesiaGraph,
    extract_quality_tags,
    quality_emotion,
)


class FSTN4DEngineV4(FSTN4DEngineV2):
    """v4 集成引擎：HNSW + 感知空间 + 通感图 + 四路融合检索。"""

    def __init__(self, state_dir: str = None, prefer_embedding: str = "auto",
                 lambda_sem: float = 0.6, hnsw_dim: int = 128,
                 hnsw_m: int = 32, hnsw_ef: int = 64):
        super().__init__(state_dir, prefer_embedding, lambda_sem)

        # ── v4: HNSW 语义索引（懒构建）──
        self.hnsw_dim = hnsw_dim
        self._hnsw_index: Optional[HNSWMemoryIndex] = None
        self._hnsw_built = False

        # ── v4: 感知嵌入空间 + 通感图 ──
        self.perceptual_index = PerceptualIndex(dim=hnsw_dim)
        self.synesthesia_graph = SynesthesiaGraph()
        self._perceptual_built = False

        # 检索融合权重（文档 §5.5 fuse_results）
        self.fuse_weights = {"semantic": 0.5, "perceptual": 0.3,
                             "wormhole": 0.1, "synesthesia": 0.1}

    # ═══════════════════════════════════════════════════════════
    # v4: HNSW 索引构建
    # ═══════════════════════════════════════════════════════════

    def _ensure_hnsw(self):
        """懒构建 HNSW 语义索引（用 v2 的 embedding provider 生成向量）。"""
        if self._hnsw_built:
            return
        self._hnsw_built = True
        try:
            # 复用 v2 的 provider（ollama bge-m3 或本地 TF-IDF）
            if self._vector_index is not None:
                provider = self._vector_index.provider
            else:
                from v2_vector_retrieval import VectorMemoryIndex
                idx = VectorMemoryIndex.build(
                    list(self.memory.memories.values()),
                    prefer=self.prefer_embedding, lambda_sem=self.lambda_sem
                )
                self._vector_index = idx
                provider = idx.provider

            dim = provider.dim if provider.dim else self.hnsw_dim
            self._hnsw_index = HNSWMemoryIndex(dim=dim, M=32, ef_search=64)
            for mem in self.memory.memories.values():
                vec = provider.embed([mem.content])[0]
                self._hnsw_index.add(mem.id, vec)
            self._hnsw_index.rebuild_all()
        except Exception as e:
            self._retrieval_provider = f"hnsw_build_failed:{e}"

    def _ensure_perceptual(self):
        """懒构建感知索引：从已有记忆的感知指纹填充。"""
        if self._perceptual_built:
            return
        self._perceptual_built = True
        for mem in self.memory.memories.values():
            sig = getattr(mem, "perceptual_signature", None)
            if sig:
                self.perceptual_index.add_signature(mem.id, sig)

    # ═══════════════════════════════════════════════════════════
    # v4: 覆写检索 —— 四路融合
    # ═══════════════════════════════════════════════════════════

    def retrieve_memories(self, query: str, k: int = 10,
                          emotion_aware: bool = True) -> list:
        self._ensure_hnsw()
        self._ensure_perceptual()

        scores: Dict[str, float] = {}   # memory_id -> 加权分

        # ── 路 1: HNSW 语义检索 ──
        semantic_hits = []
        if self._hnsw_index is not None:
            try:
                if self._vector_index is not None:
                    qv = self._vector_index.provider.embed([query])[0]
                    semantic_hits = self._hnsw_index.search(qv, k=k * 2)
            except Exception:
                semantic_hits = []
        if not semantic_hits:
            # 回退 v2 向量检索
            try:
                hits = self._vector_index.query(query, k=k * 2)
                semantic_hits = [(e.id, s) for s, e in hits]
            except Exception:
                pass
        for mid, sim in semantic_hits:
            scores[mid] = scores.get(mid, 0) + self.fuse_weights["semantic"] * sim

        # ── 路 2: 感知检索（意象词从 query 提取）──
        perceptual_hits = self.perceptual_index.search(
            _extract_imagery_words(query), k=k
        )
        for mid, sim, ch in perceptual_hits:
            scores[mid] = scores.get(mid, 0) + self.fuse_weights["perceptual"] * sim

        # ── 路 3: 虫洞扩散（复用 v1 虫洞图）──
        for mem_id in list(scores.keys())[:5]:
            for wh in self.memory.wormholes:
                if wh.pruned or wh.weight < 0.5:
                    continue
                if wh.source_id == mem_id:
                    scores[wh.target_id] = scores.get(wh.target_id, 0) + \
                        self.fuse_weights["wormhole"] * wh.weight

        # ── 路 4: 通感扩散 ──
        syn_neighbors = []
        for mem_id in list(scores.keys())[:5]:
            for nid, ch, sim, reason in self.synesthesia_graph.get_neighbors(mem_id, 0.6):
                syn_neighbors.append((nid, sim))
                self.synesthesia_graph.use(mem_id, nid, ch)  # 使用即强化
        for nid, sim in syn_neighbors:
            scores[nid] = scores.get(nid, 0) + self.fuse_weights["synesthesia"] * sim

        # ── 融合排序 ──
        if not scores:
            return super().retrieve_memories(query, k=k, emotion_aware=emotion_aware)

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        results = []
        for mid, score in ranked:
            mem = self.memory.memories.get(mid)
            if mem and mem not in results:
                results.append(mem)

        # 情绪调制（开启时）
        if emotion_aware:
            current_emotion = self.emotion.get_current()
            if current_emotion["dominant"] != "neutral":
                results.sort(
                    key=lambda m: self.memory.emotional_modulation(
                        m.id, 0.8, current_emotion), reverse=True
                )
        return results

    def get_retrieval_provider(self) -> str:
        self._ensure_hnsw()
        if self._hnsw_index is not None:
            return f"hnsw({self._hnsw_index.export_state()['backend']})"
        return self._retrieval_provider

    # ═══════════════════════════════════════════════════════════
    # v4: 覆写 process_utterance —— 增量维护感知空间 + 通感建链
    # ═══════════════════════════════════════════════════════════

    def process_utterance(self, utterance: str, context: str = "") -> dict:
        result = super().process_utterance(utterance, context)

        # 新记忆 → 感知索引（用 v4 语义簇词表提取，覆盖 v1 词表之外的意象）
        memory_id = result["memory"].get("memory_id")
        if memory_id and memory_id in self.memory.memories:
            sig = _build_v4_fingerprint(utterance)
            if sig:
                self.perceptual_index.add_signature(memory_id, sig)
                # 回填到记忆，供后续检索复用
                mem = self.memory.memories[memory_id]
                merged = dict(getattr(mem, "perceptual_signature", {}) or {})
                merged.update(sig)
                mem.perceptual_signature = merged

        # 通感建链：新记忆与已有记忆在同通道意象相似时自动建立
        if memory_id:
            self._auto_synesthesia(memory_id)

        # 周期清理通感图
        if self.interaction_count % 10 == 0:
            pruned = self.synesthesia_graph.decay_and_prune()
            if pruned:
                self._retrieval_provider = f"hnsw({pruned} syn-pruned)"

        return result

    def _auto_synesthesia(self, memory_id: str):
        """新记忆与其他记忆在任一感知通道相似度 ≥ 阈值时自动建链。"""
        mem = self.memory.memories.get(memory_id)
        if not mem:
            return
        sig = getattr(mem, "perceptual_signature", None)
        if not sig:
            return
        for ch, data in sig.items():
            if ch not in PERCEPTUAL_CHANNELS or not data:
                continue
            imagery = data.get("dominant_imagery", [])
            if not imagery:
                continue
            for other_id, other in self.memory.memories.items():
                if other_id == memory_id:
                    continue
                other_sig = getattr(other, "perceptual_signature", None)
                if not other_sig or ch not in other_sig:
                    continue
                other_imagery = other_sig[ch].get("dominant_imagery", [])
                if not other_imagery:
                    continue
                sim = self.perceptual_index.embeddings[ch].similarity(
                    imagery, other_imagery
                )
                if sim >= self.synesthesia_graph.MIN_LINK_SIMILARITY:
                    self.synesthesia_graph.link(
                        memory_id, other_id, ch, sim,
                        reason=f"{ch}通道意象相似（{imagery[:2]} vs {other_imagery[:2]}）"
                    )

    # ═══════════════════════════════════════════════════════════
    # v4: 通感-情绪（质量因子 → 情绪耦合）
    # ═══════════════════════════════════════════════════════════

    def get_synesthesia_emotion(self, text: str) -> Dict[str, float]:
        """从文本提取通感质量因子并映射到情绪增量。"""
        tags = extract_quality_tags(text)
        return quality_emotion(tags)

    # ═══════════════════════════════════════════════════════════
    # v4: 持久化
    # ═══════════════════════════════════════════════════════════

    def save_state(self):
        super().save_state()
        # v4: 完整持久化记忆（v1/v2 只存 key_nodes，普通记忆重启即失）
        try:
            import json
            state = {
                "memories": [
                    {
                        "id": m.id, "content": m.content, "layer": m.layer,
                        "t_rec": m.t_rec, "t_psych": m.t_psych,
                        "recorded_emotion": m.recorded_emotion,
                        "emotional_tags": m.emotional_tags,
                        "perceptual_signature": m.perceptual_signature,
                        "keywords": m.keywords,
                        "review_count": m.review_count,
                        "review_count_30d": m.review_count_30d,
                        "avg_gamma": m.avg_gamma,
                        "crystallized_to": m.crystallized_to,
                        "frozen": m.frozen,
                        "explicitly_marked_core": m.explicitly_marked_core,
                        "confidence": m.confidence,
                        "pending_confirmation": m.pending_confirmation,
                    }
                    for m in self.memory.memories.values()
                ],
                "version_chains": {
                    k: {"topic_key": c.topic_key, "versions": c.versions}
                    for k, c in self.memory.version_chains.items()
                },
            }
            mem_file = os.path.join(self.state_dir, "memories_full.json")
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        try:
            syn_file = os.path.join(self.state_dir, "synesthesia_graph.json")
            with open(syn_file, "w", encoding="utf-8") as f:
                import json
                json.dump(self.synesthesia_graph.export_state(), f,
                          ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_state(self) -> bool:
        ok = super().load_state()
        # v4: 恢复完整记忆
        try:
            import json
            mem_file = os.path.join(self.state_dir, "memories_full.json")
            if os.path.isfile(mem_file):
                with open(mem_file, encoding="utf-8") as f:
                    state = json.load(f)
                from fstn_memory import MemoryEntry
                for md in state.get("memories", []):
                    entry = MemoryEntry(
                        id=md["id"], content=md["content"], layer=md.get("layer", "episodic"),
                        t_rec=md.get("t_rec", time.time()),
                        t_psych=md.get("t_psych", time.time()),
                        recorded_emotion=md.get("recorded_emotion", {}),
                        emotional_tags=md.get("emotional_tags", []),
                        perceptual_signature=md.get("perceptual_signature", {}),
                        keywords=md.get("keywords", []),
                        review_count=md.get("review_count", 0),
                        review_count_30d=md.get("review_count_30d", 0),
                        avg_gamma=md.get("avg_gamma", 0.0),
                        crystallized_to=md.get("crystallized_to"),
                        frozen=md.get("frozen", False),
                        explicitly_marked_core=md.get("explicitly_marked_core", False),
                        confidence=md.get("confidence", 1.0),
                        pending_confirmation=md.get("pending_confirmation", False),
                    )
                    self.memory.memories[entry.id] = entry
                for topic_key, chain_data in state.get("version_chains", {}).items():
                    from fstn_memory import VersionChain
                    chain = VersionChain(topic_key=topic_key,
                                         versions=chain_data.get("versions", []))
                    self.memory.version_chains[topic_key] = chain
        except Exception:
            pass
        # v4: 恢复通感图
        try:
            import json
            syn_file = os.path.join(self.state_dir, "synesthesia_graph.json")
            if os.path.isfile(syn_file):
                with open(syn_file, encoding="utf-8") as f:
                    data = json.load(f)
                for link in data.get("links", []):
                    self.synesthesia_graph.link(
                        link["source_id"], link["target_id"], link["channel"],
                        link["similarity"], link.get("reason", "")
                    )
        except Exception:
            pass
        return ok

    def get_session_report(self) -> dict:
        report = super().get_session_report()
        self._ensure_hnsw()
        self._ensure_perceptual()
        report["v4"] = {
            "hnsw": self._hnsw_index.export_state() if self._hnsw_index else None,
            "perceptual_entries": sum(
                len(v) for v in self.perceptual_index._vectors.values()
            ),
            "synesthesia_links": self.synesthesia_graph.export_state(),
            "fuse_weights": self.fuse_weights,
        }
        return report


def _extract_imagery_words(text: str) -> list:
    """从文本中提取感知意象词（用于感知检索）。"""
    words = []
    for ch, clusters in _cluster_lexicon().items():
        for cluster_words in clusters.values():
            for w in cluster_words:
                if w in text and w not in words:
                    words.append(w)
    return words[:12]


def _cluster_lexicon():
    from v4_perceptual_space import _IMAGERY_CLUSTERS
    return _IMAGERY_CLUSTERS


def _build_v4_fingerprint(text: str) -> dict:
    """用 v4 语义簇词表从文本提取感知指纹（格式与 v1 兼容）。"""
    from v4_perceptual_space import _IMAGERY_CLUSTERS
    fingerprint = {}
    for ch, clusters in _IMAGERY_CLUSTERS.items():
        imagery = []
        for words in clusters.values():
            for w in words:
                if w in text and w not in imagery:
                    imagery.append(w)
        if imagery:
            fingerprint[ch] = {
                "dominant_imagery": imagery,
                "intensity": 0.7,
                "valence": 0.0,
            }
    return fingerprint


if __name__ == "__main__":
    import sys, os, tempfile

    tmp = tempfile.mkdtemp(prefix="hermes-fstn-v4-")
    eng = FSTN4DEngineV4(state_dir=tmp, prefer_embedding="local")

    # 1. 感知记忆 + 通感建链
    eng.process_utterance("昨天去看了篝火，红色的火焰在黑暗里闪烁跳跃")
    eng.process_utterance("傍晚的夕阳把天空染成一片红色，还有闪烁的光")
    eng.process_utterance("那家咖啡店有浓郁的咖啡香，坐一下午很舒服")

    # 2. 检索：语义
    hits = eng.retrieve_memories("火", k=3)
    print("检索『火』:", [h.content[:18] for h in hits])

    # 3. 检索：感知（意象词）——应命中篝火/夕阳
    phits = eng.perceptual_index.search(["红色", "闪烁"], k=3)
    print("感知检索『红色闪烁』:", [(m, round(s, 3)) for m, s, _ in phits])

    # 4. 通感图状态
    report = eng.get_session_report()
    print("通感链接数:", report["v4"]["synesthesia_links"]["active_count"])
    for link in report["v4"]["synesthesia_links"]["links"]:
        print("  ", link["source_id"], "↔", link["target_id"],
              f"({link['channel']}, {link['similarity']:.2f})")

    # 5. 通感-情绪
    print("通感情绪『灯光刺眼又吵』:", eng.get_synesthesia_emotion("灯光刺眼又吵"))

    eng.save_state()
    print("\nstate 保存 OK")
    eng2 = FSTN4DEngineV4(state_dir=tmp, prefer_embedding="local")
    eng2.load_state()
    print("state 加载 OK，通感链接:", eng2.synesthesia_graph.export_state()["active_count"])
