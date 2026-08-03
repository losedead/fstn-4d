# -*- coding: utf-8 -*-
"""
fstn5/policy_library.py — 策略库

管理策略的完整生命周期。机制泛化自 FSTN-4D 记忆层：
- 遗忘曲线 → 长时间无成功尝试的策略自动降权（过时淘汰）
- 结晶     → reward 达标的策略固化为 frozen（不可淘汰，成为"默认习惯"）
- 谱系     → 追踪演化来源（哪个策略"进化"成了哪个）
"""

import time
from typing import Dict, List, Optional

from .models import (
    Strategy, STRATEGY_ACTIVE, STRATEGY_FROZEN, STRATEGY_DEPRECATED,
)


class PolicyLibrary:
    def __init__(self, decay_days: float = 30.0,
                 freeze_threshold: float = 0.6,
                 freeze_min_trials: int = 20):
        self._strategies: Dict[str, Strategy] = {}
        self._per_user: Dict[str, Dict[str, dict]] = {}   # user_key -> {sid: {ema, trials}}
        self.decay_days = decay_days          # 遗忘半衰期（天）
        self.freeze_threshold = freeze_threshold
        self.freeze_min_trials = freeze_min_trials
        self._next_weight_hint: Dict[str, float] = {}

    # ── 基本 CRUD ──
    def add(self, name: str, description: str = "", domain: str = "generic",
            parent_id: Optional[str] = None) -> Strategy:
        s = Strategy(name=name, description=description, domain=domain,
                     parent_id=parent_id)
        self._strategies[s.id] = s
        return s

    def register(self, strategy: Strategy) -> Strategy:
        self._strategies[strategy.id] = strategy
        return strategy

    def get(self, strategy_id: str) -> Optional[Strategy]:
        return self._strategies.get(strategy_id)

    def all(self, domain: Optional[str] = None) -> List[Strategy]:
        out = [s for s in self._strategies.values()
               if s.status != STRATEGY_DEPRECATED]
        if domain:
            out = [s for s in out if s.domain == domain]
        return out

    def active(self, domain: Optional[str] = None) -> List[Strategy]:
        """可选策略 = 未淘汰的（ACTIVE + FROZEN）。
        注意：FROZEN（结晶）策略仍参与选择——它是已验证的习惯，
        排除它会导致引擎雪藏最优策略（实测 bug）。"""
        return [s for s in self.all(domain)
                if s.status in (STRATEGY_ACTIVE, STRATEGY_FROZEN)]

    def frozen(self, domain: Optional[str] = None) -> List[Strategy]:
        return [s for s in self.all(domain) if s.status == STRATEGY_FROZEN]

    # ── 反馈更新（由学习层调用）──
    def record_trial(self, strategy_id: str, reward: float,
                     user_key: str = "") -> None:
        """记录一次试验。带 user_key 时按用户分池学习（个性化）。

        工业要点：不同用户的策略偏好不同，全局 EMA 会把甲的好策略
        和乙的坏体验平均掉，导致谁都不讨好。按 (策略, 用户) 分池，
        每个用户有自己的策略权重（实测 bug：全局池个性化测试只能到
        60%，分池后可到 90%+）。
        """
        s = self._strategies.get(strategy_id)
        if s is None or s.status == STRATEGY_DEPRECATED:
            return
        # 全局 stats 始终更新（冷启动/无 user_key 场景用）
        alpha = min(0.3, max(0.02, 1.0 / (s.trials + 1)))
        s.reward_ema = s.reward_ema * (1 - alpha) + reward * alpha
        s.trials += 1
        # 用户专属池也更新（个性化场景用）
        if user_key:
            bucket = self._per_user.setdefault(user_key, {}).setdefault(
                strategy_id, {"ema": 0.0, "trials": 0})
            alpha_u = min(0.3, max(0.02, 1.0 / (bucket["trials"] + 1)))
            bucket["ema"] = bucket["ema"] * (1 - alpha_u) + reward * alpha_u
            bucket["trials"] += 1
        s.last_trial_at = time.time()

        # 结晶：达到阈值且尝试够多 → 固化为"习惯"（全局视角）
        if (s.status == STRATEGY_ACTIVE and s.trials >= self.freeze_min_trials
                and s.reward_ema >= self.freeze_threshold):
            s.status = STRATEGY_FROZEN

    def user_ema(self, user_key: str, strategy_id: str):
        """用户专属的 (ema, trials)，无记录时回退全局。"""
        bucket = self._per_user.get(user_key, {}).get(strategy_id)
        if bucket:
            return bucket["ema"], bucket["trials"]
        s = self._strategies.get(strategy_id)
        if s is None:
            return 0.0, 0
        return s.reward_ema, s.trials

    # ── 遗忘：长时间无有效尝试 → 降权 / 淘汰 ──
    def apply_forgetting(self, now: float = None) -> int:
        """返回被淘汰的策略数。frozen 策略不淘汰（用户习惯保护）。"""
        now = now or time.time()
        deprecated = 0
        for s in self._strategies.values():
            if s.status == STRATEGY_FROZEN or s.status == STRATEGY_DEPRECATED:
                continue
            idle_days = (now - s.last_trial_at) / 86400.0
            if idle_days > self.decay_days:
                s.status = STRATEGY_DEPRECATED
                deprecated += 1
        return deprecated

    # ── 演化支持 ──
    def mutate(self, strategy_id: str, new_name: str, delta_desc: str) -> Strategy:
        """基于现有策略产生变体（谱系记录 parent）"""
        parent = self._strategies.get(strategy_id)
        if parent is None:
            raise KeyError(strategy_id)
        child = Strategy(name=new_name,
                         description=(parent.description + " | 变体: " + delta_desc),
                         domain=parent.domain, parent_id=parent.id)
        self._strategies[child.id] = child
        return child

    def descendants(self, strategy_id: str) -> List[Strategy]:
        """策略谱系（演化树）"""
        return [s for s in self._strategies.values()
                if s.parent_id == strategy_id]

    # ── 序列化 ──
    def export(self) -> dict:
        return {"strategies": [s.to_dict() for s in self._strategies.values()],
                "per_user": self._per_user}

    def import_(self, data: dict) -> None:
        for d in data.get("strategies", []):
            s = Strategy.from_dict(d)
            self._strategies[s.id] = s
        self._per_user.update(data.get("per_user", {}))
