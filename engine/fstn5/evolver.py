# -*- coding: utf-8 -*-
"""
fstn5/evolver.py — 策略演化层

"自我创新"的落地：周期性产生新策略变体 → 小流量验证 → 择优固化 / 淘汰。

生成来源（可插拔）：
1. LLM 生成器：给一个"策略摘要"，让 LLM 提出改进变体（需外部 LLM）
2. 规则模板：领域内的参数变异（纯本地，零依赖）
3. 经验归纳：从高奖励经验中提取策略（纯本地）

演化流程：
  evolve_once()
    → 对每个 active 策略生成 1 个变体（或由外部注入）
    → 变体进入策略库（weight 低，会被 UCB 探索）
    → 经过 N 次尝试后，若 reward 低于父策略 → 自动淘汰（遗忘）
    → 若 reward 高于父策略 → 保留，成为新习惯候选
"""

import time
from typing import Callable, Dict, List, Optional

from .models import Strategy, STRATEGY_ACTIVE, STRATEGY_DEPRECATED
from .policy_library import PolicyLibrary


class Evolver:
    def __init__(self, library: PolicyLibrary,
                 generator: Optional[Callable[[str, str], str]] = None):
        """
        generator: 可选的外部策略生成器。
        签名: (strategy_name, strategy_description) -> 新策略名
        不提供时用规则变异（在名称后加变体标记）。
        """
        self.library = library
        self.generator = generator
        self._variant_counter = 0

    def generate_variant(self, parent: Strategy) -> Strategy:
        """基于父策略产生一个变体"""
        self._variant_counter += 1
        if self.generator is not None:
            try:
                new_name = self.generator(parent.name, parent.description)
            except Exception:
                new_name = f"{parent.name}·变体{self._variant_counter}"
        else:
            # 规则变异：参数化变体（领域无关）
            new_name = f"{parent.name}·V{self._variant_counter}"
            delta = "规则变异（同策略新参数组合）"
        return self.library.mutate(
            parent.id, new_name,
            delta_desc=delta if 'delta' in dir() else "规则变异")

    def evolve_once(self, parents: Optional[List[Strategy]] = None) -> List[Strategy]:
        """对候选父策略各生成一个变体。返回新变体列表。"""
        parents = parents or self.library.active()
        variants = []
        for p in parents[:5]:  # 每轮最多 5 个变体，控制策略数量
            v = self.generate_variant(p)
            variants.append(v)
        return variants

    def prune_failed_variants(self, min_trials: int = 10,
                              parent_ratio: float = 0.7) -> int:
        """淘汰劣质变体：试了够多次且 reward 明显低于父策略 → 弃用"""
        removed = 0
        for s in self.library.all():
            if s.parent_id is None:
                continue
            parent = self.library.get(s.parent_id)
            if parent is None:
                continue
            if (s.trials >= min_trials
                    and s.reward_ema < parent.reward_ema * parent_ratio):
                s.status = STRATEGY_DEPRECATED
                removed += 1
        return removed
