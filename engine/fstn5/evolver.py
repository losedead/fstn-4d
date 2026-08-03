# -*- coding: utf-8 -*-
"""
fstn5/evolver.py — 策略演化层

"自我创新"的落地：周期性产生新策略变体 → 小流量验证 → 择优固化 / 淘汰。

生成来源（可插拔）：
1. LLM 生成器（LLMStrategyGenerator）：给"策略池+经验统计+失败案例"，
   让 LLM 提出全新策略（真创新，推荐）
2. 规则变异（Evolver）：领域内的参数变异（纯本地，零依赖，向后兼容）

演化流程：
  evolve_with_llm() → LLM 生成全新策略 → 进策略库 → UCB 小流量探索
    → 效果好固化 / 差淘汰
  evolve_once()（规则）→ 对 active 策略生成变体 → 同上
"""

import time
from typing import Callable, Dict, List, Optional

from .models import Strategy, STRATEGY_ACTIVE, STRATEGY_DEPRECATED
from .policy_library import PolicyLibrary
from .llm_client import LLMClient


class LLMStrategyGenerator:
    """LLM 驱动的策略生成器：生成【全新思路】的策略，而非复制改名。

    输入：领域 + 策略池摘要 + 经验统计 + 失败案例
    输出：LLM 提出的新策略（带 rationale）
    """

    def __init__(self, client: LLMClient = None, count: int = 2):
        self.client = client or LLMClient()
        self.count = count

    def generate(self, domain: str, library: PolicyLibrary,
                 experience_summary: dict = None) -> List[Strategy]:
        """生成并注册新策略，返回新 Strategy 列表。"""
        strategies = [{"name": s.name,
                       "description": s.description,
                       "reward_ema": round(s.reward_ema, 3),
                       "trials": s.trials,
                       "status": s.status}
                      for s in library.active(domain)]
        failures = []
        exp = experience_summary or {}
        for item in exp.get("recent", [])[:8]:
            if item.get("reward", 0) < 0.3:
                failures.append({
                    "task": item.get("task_text", ""),
                    "strategy": item.get("strategy_name", ""),
                    "reward": round(item.get("reward", 0), 3),
                })
        stats = {}
        for s in library.active(domain):
            stats[s.name] = {"reward_ema": round(s.reward_ema, 3),
                             "trials": s.trials}
        try:
            proposals = self.client.generate_strategies(
                domain, {
                    "strategies": strategies,
                    "experience_stats": stats,
                    "failures": failures,
                }, count=self.count)
        except Exception as e:
            print(f"[evolver] LLM 生成失败: {e}")
            return []
        created = []
        for p in proposals:
            try:
                s = library.add(p["name"], p["description"], domain)
            except Exception:
                continue
            s.rationale = p.get("rationale", "")
            s.origin = "llm"
            created.append(s)
        return created


class Evolver:
    """规则变异演化器（纯本地，向后兼容）。

    新代码建议用 LLMStrategyGenerator 做真创新；Evolver 保留
    作为零依赖的备选（复制+参数变异）。
    """

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
        delta = "规则变异（同策略新参数组合）"
        if self.generator is not None:
            try:
                new_name = self.generator(parent.name, parent.description)
            except Exception:
                new_name = f"{parent.name}·变体{self._variant_counter}"
        else:
            new_name = f"{parent.name}·V{self._variant_counter}"
        return self.library.mutate(
            parent.id, new_name, delta_desc=delta)

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
