# -*- coding: utf-8 -*-
"""
fstn5/core.py — FSTN5Core 主引擎

把 经验记忆 / 策略库 / Bandit学习 / 个性化路由 / 演化 串成自改进闭环：

  recommend(task, user) → strategy     （决策）
  record_feedback(exp)  → 更新权重    （学习）
  evolve()              → 生成变体    （创新）
  status()              → 自证报告    （"变聪明"的证据）

独立部署：不依赖 Hermes / FSTN-4D / 任何外部服务。
"""

import json
import os
import time
from typing import Dict, List, Optional

from .models import (
    Experience, Strategy, UserVector,
    STRATEGY_ACTIVE, STRATEGY_FROZEN, STRATEGY_DEPRECATED,
)
from .policy_library import PolicyLibrary
from .experience_memory import ExperienceMemory
from .learner import BanditLearner
from .personalizer import Personalizer
from .evolver import Evolver


class FSTN5Core:
    def __init__(self, state_dir: Optional[str] = None,
                 exploration: float = 1.2,
                 decay_days: float = 30.0,
                 freeze_threshold: float = 0.6,
                 freeze_min_trials: int = 20,
                 generator=None):
        self.state_dir = state_dir
        self.library = PolicyLibrary(decay_days=decay_days,
                                     freeze_threshold=freeze_threshold,
                                     freeze_min_trials=freeze_min_trials)
        self.memory = ExperienceMemory()
        self.learner = BanditLearner(exploration=exploration)
        self.personalizer = Personalizer()
        self.evolver = Evolver(self.library, generator=generator)
        self._next_exp_id = 1
        self._user_keys: Dict[str, str] = {}  # user_vector_hash -> user_key

        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
            self.load()

    # ══════════════ 策略管理 ══════════════

    def add_strategy(self, name: str, description: str = "",
                     domain: str = "generic") -> str:
        """注册一个新策略。返回 strategy_id。"""
        s = self.library.add(name, description=description, domain=domain)
        return s.id

    def strategies(self, domain: Optional[str] = None) -> List[dict]:
        """全部策略（含状态/学习指标）"""
        return self.learner.summary(self.library, domain)

    # ══════════════ 决策闭环 ══════════════

    def recommend(self, task_text: str,
                  user: Optional[UserVector] = None,
                  user_key: str = "",
                  domain: Optional[str] = None,
                  exclude: Optional[List[str]] = None,
                  task_tags: Optional[List[str]] = None) -> dict:
        """为任务推荐策略。

        融合三条路径：
        1. 经验记忆：相似任务的历史最优策略（经验驱动，支持 task_tags）
        2. UCB 学习：策略库全局学习权重（学习驱动）
        3. 个性化：该用户的策略亲和力（个性化驱动）
        """
        user_vec = user.to_vector() if user else {}
        result = {"task": task_text, "user_key": user_key, "domain": domain}

        # 路径 1：经验记忆推荐（task-aware：相似任务的历史最优策略）
        exp_best = self.memory.best_strategy_for_task(
            task_text, user_vec, task_tags=task_tags)
        result["experience_recommendation"] = (
            exp_best.get("strategy_id") if exp_best else None)
        result["experience_confidence"] = (
            exp_best.get("confidence", 0.0) if exp_best else 0.0)

        strategies = self.library.active(domain)
        if exclude:
            strategies = [s for s in strategies if s.id not in exclude]
        if not strategies:
            raise RuntimeError("该领域无可用策略")

        total_trials = sum(
            self.library.user_ema(user_key, s.id)[1] if user_key else s.trials
            for s in strategies) + 1
        scores = {}
        for s in strategies:
            ema, trials = (self.library.user_ema(user_key, s.id)
                           if user_key else (s.reward_ema, s.trials))
            if trials == 0:
                score = 1e6 + self.learner.exploration
            else:
                score = (ema + self.learner.exploration * (
                    __import__("math").log(total_trials) / trials) ** 0.5)
            if s.status == STRATEGY_FROZEN:
                score += self.learner.frozen_bonus
            scores[s.id] = score

        # 经验优先：相似任务有足够历史时，用 task-aware 推荐
        # （这才能区分"10GB用流式"vs"100MB用批量"——全局 UCB 做不到）
        # 加成封顶 0.5：经验是先验偏置，不是决策主导——
        # 环境突变时 UCB 仍能纠正过时经验。
        exp_best_id = exp_best.get("strategy_id") if exp_best else None
        if exp_best_id is not None and exp_best_id in scores:
            exp_conf = min(0.5, exp_best.get("confidence", 0.0))
            scores[exp_best_id] += exp_conf
            result["routing"] = "experience"
            result["experience_boost"] = exp_conf
            result["experience_n"] = exp_best.get("n", 0)
        else:
            result["routing"] = "ucb"

        chosen_id = self.personalizer.recommend_personalized(
            self.library, self.learner, user_key, scores)
        chosen = self.library.get(chosen_id)

        result["strategy_id"] = chosen.id
        result["strategy_name"] = chosen.name
        result["strategy_status"] = chosen.status
        return result

    def _resolve_strategy(self, strategy_id_or_name: str) -> Optional[str]:
        """容忍传 id 或 name（工业 API 友好性：用户常传名字）"""
        if self.library.get(strategy_id_or_name) is not None:
            return strategy_id_or_name
        for s in self.library.all():
            if s.name == strategy_id_or_name:
                return s.id
        return None

    def record_feedback(self, task_text: str, strategy_id: str, reward: float,
                        user: Optional[UserVector] = None,
                        user_key: str = "",
                        task_tags: Optional[List[str]] = None,
                        feedback_note: str = "") -> dict:
        """记录一次任务结果反馈 → 更新学习权重。

        reward ∈ [-1, 1]：正 = 策略有效，负 = 无效。
        strategy_id 可传 id 或策略名（工业 API 友好）。
        这是"自我学习"的入口。
        """
        sid = self._resolve_strategy(strategy_id)
        if sid is None:
            return {"ok": False, "error": f"策略不存在: {strategy_id}"}
        reward = max(-1.0, min(1.0, float(reward)))
        exp = Experience(
            task_text=task_text, strategy_id=sid,
            reward=reward,
            user_vector=user.to_vector() if user else {},
            task_tags=task_tags or [], feedback_note=feedback_note)
        self.memory.add(exp)
        self.library.record_trial(sid, reward, user_key=user_key)
        if user_key:
            self.personalizer.note_experience(user_key, exp)
        return {"ok": True, "experience_id": exp.id, "reward": reward,
                "strategy_id": sid}

    # ══════════════ 自改进 ══════════════

    def evolve(self, parents: Optional[List[Strategy]] = None) -> List[str]:
        """自我创新：生成策略变体进入学习池。"""
        variants = self.evolver.evolve_once(parents)
        return [v.id for v in variants]

    def prune(self, min_trials: int = 10, parent_ratio: float = 0.7) -> int:
        """淘汰劣质变体（自我改进的负向操作）"""
        return self.evolver.prune_failed_variants(min_trials, parent_ratio)

    def apply_forgetting(self) -> int:
        """遗忘：淘汰长期无效的策略"""
        return self.library.apply_forgetting()

    def settle(self, now: float = None) -> int:
        """定期清理：遗忘 + 淘汰劣质变体"""
        n1 = self.apply_forgetting()
        n2 = self.prune()
        return n1 + n2

    # ══════════════ 自证报告 ══════════════

    def status(self, domain: Optional[str] = None) -> dict:
        """引擎当前状态 + 学习进度（"变聪明"的证据）"""
        strategies = self.library.all(domain)
        active = [s for s in strategies if s.status == STRATEGY_ACTIVE]
        frozen = [s for s in strategies if s.status == STRATEGY_FROZEN]
        deprecated = [s for s in strategies if s.status == STRATEGY_DEPRECATED]
        return {
            "strategies_total": len(strategies),
            "strategies_active": len(active),
            "strategies_frozen": len(frozen),
            "strategies_deprecated": len(deprecated),
            "experiences": self.memory.count(),
            "total_trials": sum(s.trials for s in strategies),
            "avg_reward": round(
                sum(s.reward_ema for s in active) / len(active), 4) if active else 0.0,
            "learning_summary": self.learner.summary(self.library, domain),
        }

    # ══════════════ 持久化 ══════════════

    def save(self) -> str:
        if not self.state_dir:
            return ""
        data = {
            "library": self.library.export(),
            "memory": self.memory.export(),
            "personalizer": self.personalizer.export(),
        }
        path = os.path.join(self.state_dir, "fstn5_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def load(self) -> bool:
        if not self.state_dir:
            return False
        path = os.path.join(self.state_dir, "fstn5_state.json")
        if not os.path.isfile(path):
            return False
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.library.import_(data.get("library", {}))
        self.memory.import_(data.get("memory", {}))
        self.personalizer.import_(data.get("personalizer", {}))
        return True
