# -*- coding: utf-8 -*-
"""
fstn5/experience_memory.py — 经验记忆（4D 存储机制回归版）

存储【任务-策略-结果】三元组。v2 起接入 FSTN-4D 完整存储架构：
- 斐波那契时间金字塔：经验按心理时间沉入遗忘窗口
- 虫洞自动发现：存储时建立关联（4D 协议一）
- 复习巩固：被检索的经验自动 review（使用后触发）
- 感知指纹：存储时附加五感标注
- 情绪调制：检索时按当前情绪调整关联度
- 结晶：高频经验固化为潜意识关键节点（默认习惯）

同时保留 v1 的 bandit 接口（query_scored/best_strategy_for_task），
供策略路由使用——记忆层升级不影响决策层。
"""

import re
import time
from typing import Dict, List, Optional

from .models import Experience, similarity
from .memory_engine import (
    FibonacciMemory, KeyNode, Wormhole, VersionChain,
    find_window, emotional_modulation,
)


class ExperienceMemory:
    def __init__(self, max_experiences: int = 10000):
        self._experiences: List[Experience] = []
        self.max_experiences = max_experiences
        self.fib = FibonacciMemory()
        # 记录 emotion 调制用：最近一次当前情绪快照
        self._last_emotion: Dict[str, float] = {}

    # ══════════════ 写入（4D 协议一：存储即关联）══════════════

    def add(self, exp: Experience,
            layer: str = "episodic",
            recorded_emotion: Optional[Dict[str, float]] = None,
            emotional_tags: Optional[List[str]] = None,
            perceptual_signature: Optional[Dict] = None) -> Experience:
        """存储一条经验。自动：提取关键词 → 发现虫洞 → 检查版本冲突。"""
        exp.layer = layer
        exp.recorded_emotion = dict(recorded_emotion or {})
        exp.emotional_tags = list(emotional_tags or [])
        exp.perceptual_signature = dict(perceptual_signature or {})
        exp.keywords = self.fib.extract_keywords(exp.task_text)
        exp.t_psych = time.time()

        # 版本链冲突：同任务同策略的高置信经验视为"确认"
        topic = f"{exp.task_text[:30]}::{exp.strategy_id}"
        self.fib.add_version(topic, exp.task_text, exp.id)

        self._experiences.append(exp)
        if len(self._experiences) > self.max_experiences:
            self._experiences = self._experiences[-self.max_experiences:]

        # 虫洞自动发现（协议一）——只扫描最近 100 条（邻域优先，防 O(n²)）
        recent = self._experiences[-100:]
        self.fib.suggest_wormholes(
            exp.id, exp.keywords,
            get_mem=lambda: ((e.id, e) for e in recent),
            add_wormhole=lambda wh: self._add_wormhole_unique(wh))
        return exp

    def _add_wormhole_unique(self, wh: Wormhole) -> None:
        for w in self.fib.wormholes:
            if (w.source_id == wh.source_id and w.target_id == wh.target_id
                    and not w.pruned):
                w.weight = min(1.0, w.weight + 0.1)
                return
        self.fib.wormholes.append(wh)

    def perceive(self, exp_id: str, perceptual_profile: Dict) -> bool:
        """为经验附加感知指纹（4D 协议一）"""
        exp = self.get(exp_id)
        if exp is None:
            return False
        exp.perceptual_signature.update(perceptual_profile)
        return True

    # ══════════════ 检索（4D 存储 + v1 bandit 兼容）══════════════

    def get(self, exp_id: str) -> Optional[Experience]:
        for e in self._experiences:
            if e.id == exp_id:
                return e
        return None

    def query_scored(self, task_text: str,
                     user_vector: Dict[str, float] = None,
                     k: int = 5,
                     user_weight: float = 0.5,
                     task_tags: List[str] = None,
                     current_emotion: Optional[Dict[str, float]] = None,
                     apply_time_decay: bool = True,
                     apply_wormhole: bool = True,
                     apply_emotion: bool = True) -> List[tuple]:
        """返回 (score, exp) 排序列表。

        4D 机制融合：
        - 时间衰减：斐波那契窗口越深，分数越低（自然遗忘）
        - 情绪调制：当前情绪与记忆情绪一致性调整
        - 虫洞扩散：命中记忆的关联记忆获得加分
        """
        task_feats = _text_features(task_text)
        scored = []
        for exp in self._experiences:
            exp_feats = _text_features(exp.task_text)
            task_sim = _jaccard(task_feats, exp_feats)
            tag_sim = 0.0
            if task_tags and exp.task_tags:
                common = len(set(task_tags) & set(exp.task_tags))
                union = len(set(task_tags) | set(exp.task_tags))
                tag_sim = common / union if union else 0.0
            if user_vector and exp.user_vector:
                user_sim = similarity(user_vector, exp.user_vector)
                score = (task_sim * 0.5 + tag_sim * 0.3 + user_sim * 0.2)
            elif task_tags:
                score = task_sim * 0.6 + tag_sim * 0.4
            else:
                score = task_sim

            # 4D：时间衰减（斐波那契窗口）
            if apply_time_decay:
                w = exp.window
                if w > 3:
                    score *= max(0.3, 1.0 - 0.15 * (w - 3))

            # 4D：情绪调制
            if apply_emotion and current_emotion and exp.recorded_emotion:
                score = emotional_modulation(
                    score, exp.recorded_emotion, exp.emotional_tags,
                    current_emotion)

            if score > 0:
                scored.append((score, exp))

        scored.sort(key=lambda x: -x[0])
        top = scored[:k]

        # 4D：虫洞扩散（命中记忆的关联记忆加分）
        if apply_wormhole and self.fib.wormholes:
            hit_ids = {e.id for _, e in top}
            bonus = []
            for _, e in top:
                for wh in self.fib.wormholes:
                    if wh.pruned or wh.source_id != e.id or wh.weight <= 0.5:
                        continue
                    target = self.get(wh.target_id)
                    if target and target.id not in hit_ids:
                        bonus.append((wh.weight * 0.8, target))
                        wh.last_used = time.time()
                        wh.weight = min(1.0, wh.weight + 0.05)
            for sc, e in bonus:
                top.append((sc, e))
            top.sort(key=lambda x: -x[0])
            top = top[:k]
        return top

    def query(self, task_text: str, user_vector: Dict[str, float] = None,
              k: int = 5, user_weight: float = 0.3) -> List[Experience]:
        """按任务相似度 + 用户相似度排序返回历史经验。"""
        return [e for _, e in self.query_scored(task_text, user_vector, k, user_weight)]

    def best_strategy_for_task(self, task_text: str,
                               user_vector: Dict[str, float] = None,
                               min_samples: int = 3,
                               sim_threshold: float = 0.6,
                               task_tags: List[str] = None) -> Optional[dict]:
        """对相似任务：返回加权 reward 最高的策略 + 置信度。

        只聚合【高度相似】的经验（score > sim_threshold=0.6）——文本
        相似度不足 0.6 时视为"不同任务"，经验路由不适用，退回 UCB。
        """
        hits = self.query_scored(task_text, user_vector, k=50,
                                 task_tags=task_tags,
                                 apply_time_decay=True,
                                 apply_emotion=False)
        similar = [(s, h) for s, h in hits if s > sim_threshold]
        if len(similar) < min_samples:
            return None
        agg_sum: Dict[str, float] = {}
        agg_n: Dict[str, int] = {}
        for score, h in similar:
            agg_sum[h.strategy_id] = agg_sum.get(h.strategy_id, 0.0) + score * (h.reward + 1) / 2
            agg_n[h.strategy_id] = agg_n.get(h.strategy_id, 0) + 1
        if not agg_sum:
            return None
        candidates = {sid for sid, n in agg_n.items() if n >= min_samples}
        if not candidates:
            return None
        import math as _m
        total_n = sum(agg_n[sid] for sid in candidates)
        def quality(sid):
            avg = agg_sum[sid] / agg_n[sid]
            lcb = 0.35 * _m.sqrt(_m.log(total_n + 1) / agg_n[sid])
            return avg - lcb
        best_id = max(candidates, key=quality)
        total_quality = sum(quality(sid) for sid in candidates)
        top_quality = quality(best_id)
        conf = 0.4 * min(1.0, len(similar) / 10.0) + 0.6 * (top_quality / total_quality if total_quality > 0 else 0.5)
        return {"strategy_id": best_id,
                "confidence": round(min(1.0, conf), 3),
                "n": agg_n[best_id],
                "similar_samples": len(similar)}

    # ══════════════ 复习与结晶（4D 维护）══════════════

    def review(self, exp_ids: List[str], gamma: float = None) -> int:
        """复习经验（使用后触发）：更新心理时间 + 虫洞扩散。"""
        count = 0
        for mid in exp_ids:
            exp = self.get(mid)
            if exp is None or exp.frozen:
                continue
            self.fib.review_with_spread(
                exp, self.fib.wormholes,
                get_mem=lambda eid: self.get(eid), gamma=gamma)
            count += 1
        return count

    def crystallize(self, exp_id: str,
                    trigger_keywords: List[str] = None,
                    strategy_id: str = None) -> Optional[str]:
        """将高频经验结晶为潜意识关键节点（默认习惯）。"""
        exp = self.get(exp_id)
        if exp is None:
            return None
        topic = f"{exp.task_text[:30]}::{exp.strategy_id}"
        if not self.fib.can_crystallize(exp, version_topic=topic):
            return None
        return self.fib.crystallize(exp, trigger_keywords, strategy_id)

    def scan_subconscious(self, query: str) -> List[KeyNode]:
        """潜意识自动激活：查询命中关键节点触发词。"""
        q_kw = self.fib.extract_keywords(query)
        hits = []
        for node in self.fib.key_nodes.values():
            if any(kw in node.auto_trigger_keywords for kw in q_kw) or \
               any(kw in query for kw in node.auto_trigger_keywords):
                node.activation_count += 1
                hits.append(node)
        return hits

    def consolidate(self, merge_threshold: float = 0.85) -> int:
        """合并深层窗口相似经验（记忆巩固）。"""
        merged = 0
        deep = [e for e in self._experiences
                if e.window >= 6 and not e.frozen]
        for i in range(len(deep)):
            for j in range(i + 1, len(deep)):
                a, b = deep[i], deep[j]
                sim = _jaccard(set(a.keywords), set(b.keywords))
                if sim >= merge_threshold and a.strategy_id == b.strategy_id:
                    # 保留 reward 较高的，合并统计
                    keep, drop = (a, b) if a.reward >= b.reward else (b, a)
                    keep.review_count += drop.review_count
                    if drop.id in [e.id for e in self._experiences]:
                        self._experiences = [e for e in self._experiences
                                             if e.id != drop.id]
                        merged += 1
        return merged

    # ══════════════ 统计与序列化 ══════════════

    def count(self) -> int:
        return len(self._experiences)

    def reward_by_strategy(self) -> Dict[str, float]:
        """每个策略的平均经验奖励（用于演化择优）"""
        agg: Dict[str, List[float]] = {}
        for e in self._experiences:
            agg.setdefault(e.strategy_id, []).append(e.reward)
        return {sid: sum(v) / len(v) for sid, v in agg.items()}

    def export(self) -> dict:
        return {
            "experiences": [e.to_dict() for e in self._experiences],
            "fib": self.fib.export(),
        }

    def import_(self, data: dict) -> None:
        for d in data.get("experiences", []):
            self._experiences.append(Experience.from_dict(d))
        self.fib.import_(data.get("fib", {}))


# ── 轻量中文文本特征（零依赖：字 bigram + 关键词）──
_STOP = set("的了是在我有和就不都也还很才没有吧吗呢啊么一个上为与及或对从")

def _text_features(text: str) -> set:
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text.lower())
    feats = set()
    for i in range(len(text) - 1):
        pair = text[i:i+2]
        if pair and pair[0] not in _STOP:
            feats.add(pair)
    return feats


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
