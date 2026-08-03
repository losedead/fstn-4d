# -*- coding: utf-8 -*-
"""
fstn5/learner.py — 反馈学习层（Bandit 选择）

引擎"越来越聪明"的核心机制：多臂老虎机（UCB1 + 汤普森风格混合）。

- UCB 公式：exploit 好策略 + explore 未充分尝试的策略
  score = reward_ema + sqrt(2 * ln(total_trials) / trials)
- 结晶策略（frozen）有永久加成（用户已验证的习惯）
- 通过反馈更新权重，选择概率自动收敛到最优策略
"""

import math
from typing import Dict, List, Optional

from .models import Strategy, STRATEGY_FROZEN
from .policy_library import PolicyLibrary


class BanditLearner:
    def __init__(self, exploration: float = 1.2, frozen_bonus: float = 0.15):
        self.exploration = exploration   # UCB 探索系数
        self.frozen_bonus = frozen_bonus  # 结晶策略加成

    def choose(self, library: PolicyLibrary,
               domain: Optional[str] = None,
               exclude: Optional[List[str]] = None,
               user_key: str = "") -> Strategy:
        """选择下一个要尝试的策略（UCB）

        user_key 提供时，用该用户的专属 EMA/trials 计算 UCB 分数——
        个性化学习的核心：每个用户有自己的策略权重。
        """
        candidates = library.active(domain)
        if exclude:
            candidates = [s for s in candidates if s.id not in exclude]
        if not candidates:
            raise RuntimeError("无可用策略")

        # 用户专属统计
        def stats(s):
            if user_key:
                return library.user_ema(user_key, s.id)
            return s.reward_ema, s.trials

        total_trials = sum(stats(s)[1] for s in candidates) + 1
        best = None
        best_score = -1e9
        for s in candidates:
            ema, trials = stats(s)
            if trials == 0:
                score = 1e6 + self.exploration  # 从未尝试：优先探索
            else:
                mean = ema
                explore = self.exploration * math.sqrt(
                    math.log(total_trials) / trials)
                score = mean + explore
            if s.status == STRATEGY_FROZEN:
                score += self.frozen_bonus
            if score > best_score:
                best_score = score
                best = s
        return best

    def weights(self, library: PolicyLibrary,
                domain: Optional[str] = None) -> Dict[str, float]:
        """各策略的选择权重（用于可视化/报告）"""
        out = {}
        for s in library.active(domain):
            if s.trials == 0:
                out[s.name] = self.exploration + 1.0
            else:
                out[s.name] = s.reward_ema + self.exploration * math.sqrt(
                    math.log(sum(x.trials for x in library.active(domain)) + 1)
                    / s.trials)
            if s.status == STRATEGY_FROZEN:
                out[s.name] += self.frozen_bonus
        return out

    def summary(self, library: PolicyLibrary,
                domain: Optional[str] = None) -> List[dict]:
        """策略学习状态（供报告/API）"""
        rows = []
        for s in library.all(domain):
            rows.append({
                "name": s.name,
                "domain": s.domain,
                "status": s.status,
                "reward_ema": round(s.reward_ema, 4),
                "trials": s.trials,
                "weight": round(
                    self.weights(library, domain).get(s.name, 0.0), 4),
            })
        rows.sort(key=lambda r: -r["weight"])
        return rows
