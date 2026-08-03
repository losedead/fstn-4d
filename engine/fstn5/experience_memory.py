# -*- coding: utf-8 -*-
"""
fstn5/experience_memory.py — 经验记忆

存储【任务-策略-结果】三元组。泛化自 FSTN-4D 记忆层：
- 按任务文本相似度检索"类似任务当时用了什么策略、效果如何"
- 按用户状态过滤"对类似用户什么策略有效"
- 遗忘：经验是学习素材，保留全部；但策略层面的遗忘在 PolicyLibrary
"""

import re
import time
from typing import Dict, List, Optional

from .models import Experience, similarity


class ExperienceMemory:
    def __init__(self, max_experiences: int = 10000):
        self._experiences: List[Experience] = []
        self.max_experiences = max_experiences

    # ── 写入 ──
    def add(self, exp: Experience) -> Experience:
        self._experiences.append(exp)
        if len(self._experiences) > self.max_experiences:
            # 淘汰最旧的一半（保留最近经验）
            self._experiences = self._experiences[-self.max_experiences:]
        return exp

    # ── 检索：相似任务 + 相似用户 ──
    def query_scored(self, task_text: str,
                     user_vector: Dict[str, float] = None,
                     k: int = 5,
                     user_weight: float = 0.5,
                     task_tags: List[str] = None) -> List[tuple]:
        """返回 (score, exp) 排序列表（保留相似度分数供加权用）。

        user_weight 只在【提供用户向量】时生效；无用户向量时纯任务相似度
        （实测 bug：空 user_vector 时 score 被 0.5 权重稀释，完全相同
        文本只有 0.5 分，低于 sim_threshold=0.6 导致经验路由永远不触发）。

        task_tags：结构化任务标签（如 ['large','file']）。提供时，标签
        重合度参与相似度（工业推荐——纯文本对"50MB vs 100MB"这类数字
        差异不敏感，tags 才是可靠的相似度信号）。
        """
        task_feats = _text_features(task_text)
        scored = []
        for exp in self._experiences:
            exp_feats = _text_features(exp.task_text)
            task_sim = _jaccard(task_feats, exp_feats)
            # 标签相似度（可选的强信号）
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
            scored.append((score, exp))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

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
        这个阈值很关键：过低会让不同任务互相污染路由（实测 bug：
        5 个任务文本相似度 0.4-0.5，经验路由把全局最优 D 导向了
        各任务自己的历史最优，UCB 的高分被经验加成压过）。

        返回 {strategy_id, confidence, n} 或 None。
        """
        hits = self.query_scored(task_text, user_vector, k=50,
                                 task_tags=task_tags)
        # 只保留高度相似的经验
        similar = [(s, h) for s, h in hits if s > sim_threshold]
        if len(similar) < min_samples:
            return None
        # 用【平均质量 + 置信下界正则】而非裸平均——
        # 样本少的策略会因噪声拿到高"平均质量"（实测 bug：试过 1 次
        # 拿高分的新策略压过试过 100 次的老策略）。加 LCB 探索项：
        #   quality = avg - c * sqrt(ln(N) / n)
        # 样本越少，减分越多——防止过拟合单次幸运。
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

    # ── 统计 ──
    def count(self) -> int:
        return len(self._experiences)

    def reward_by_strategy(self) -> Dict[str, float]:
        """每个策略的平均经验奖励（用于演化择优）"""
        agg: Dict[str, List[float]] = {}
        for e in self._experiences:
            agg.setdefault(e.strategy_id, []).append(e.reward)
        return {sid: sum(v) / len(v) for sid, v in agg.items()}

    # ── 序列化 ──
    def export(self) -> dict:
        return {"experiences": [e.to_dict() for e in self._experiences]}

    def import_(self, data: dict) -> None:
        for d in data.get("experiences", []):
            self._experiences.append(Experience.from_dict(d))


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
