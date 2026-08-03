# -*- coding: utf-8 -*-
"""
fstn5/personalizer.py — 个性化路由层

用户状态向量（含情感维度）→ 影响策略选择的个性化因子。

机制：
- 为每个用户维护"策略亲和力"（从该用户的历史经验中统计）
- recommend 时：UCB 分数 + 用户亲和力加成
- 情感维度参与路由：例如高焦虑用户偏向"稳妥型"策略（若适配器标注了该特质）
"""

from typing import Dict, List, Optional

from .models import Experience, similarity
from .policy_library import PolicyLibrary


class Personalizer:
    def __init__(self, affinity_boost: float = 0.25, min_exp: int = 3):
        self.affinity_boost = affinity_boost
        self.min_exp = min_exp
        self._user_affinity: Dict[str, Dict[str, float]] = {}
        # user_key -> {strategy_id: EMA reward}

    def note_experience(self, user_key: str, exp: Experience) -> None:
        """从一次带用户状态的经验中更新亲和力"""
        if not user_key:
            return
        aff = self._user_affinity.setdefault(user_key, {})
        prev = aff.get(exp.strategy_id, 0.0)
        # 简单 EMA
        aff[exp.strategy_id] = prev * 0.8 + exp.reward * 0.2

    def user_strategy_boost(self, user_key: str,
                            strategy_id: str) -> float:
        """该用户对该策略的个性化加成"""
        aff = self._user_affinity.get(user_key, {})
        if strategy_id in aff:
            return aff[strategy_id] * self.affinity_boost
        return 0.0

    def recommend_personalized(self, library: PolicyLibrary,
                               learner, user_key: str,
                               base_scores: Dict[str, float]) -> str:
        """base_scores: UCB 分数 {strategy_id: score} → 加成后取最高"""
        best_id = None
        best_score = -1e9
        for sid, score in base_scores.items():
            boosted = score + self.user_strategy_boost(user_key, sid)
            if boosted > best_score:
                best_score = boosted
                best_id = sid
        return best_id

    def export(self) -> dict:
        return {"user_affinity": self._user_affinity}

    def import_(self, data: dict) -> None:
        self._user_affinity.update(data.get("user_affinity", {}))
