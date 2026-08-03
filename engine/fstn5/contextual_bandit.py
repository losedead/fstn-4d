# -*- coding: utf-8 -*-
"""
fstn5/contextual_bandit.py — 深度上下文 Bandit（LinUCB）

"选得准"的实现：用任务特征向量预测每个策略的奖励，选 UCB 置信上界最高的。

对比非上下文 BanditLearner：
  非上下文：score = 平均奖励 + 探索（不管任务是什么）
  上下文：   score = x·θ + α√(xᵀA⁻¹x)（x 是任务特征，θ 是策略参数）

LinUCB（Li et al. 2010）：
  - 每个策略 s 维护 A_s = XᵀX + λI（d×d）和 b_s = Xᵀy
  - θ_s = A_s⁻¹ b_s（岭回归解）
  - 奖励预测 = xᵀθ_s，置信上界 = α√(xᵀ A_s⁻¹ x)
  - 未尝试策略给探索先验（等价于观察一次伪反馈）

零新依赖（纯 Python + math），特征维度 d 通常 ≤ 200。
"""

import math
from typing import Dict, List, Optional

from .models import Strategy, STRATEGY_FROZEN
from .policy_library import PolicyLibrary


def _solve_ridge(A, b, lam=1.0):
    """解 (A + λI)θ = b（高斯消元，纯 Python）。

    A: d×d 列表，b: d 列表。返回 θ: d 列表。
    """
    d = len(b)
    if d == 0:
        return []
    # 构造增广矩阵 M = [A + λI | b]
    M = []
    for i in range(d):
        row = [A[i][j] + (lam if i == j else 0.0) for j in range(d)]
        row.append(b[i])
        M.append(row)
    # 高斯消元（部分主元）
    for col in range(d):
        # 找主元
        pivot = col
        maxv = abs(M[col][col])
        for r in range(col + 1, d):
            if abs(M[r][col]) > maxv:
                maxv = abs(M[r][col])
                pivot = r
        if maxv < 1e-12:
            continue
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        for j in range(col, d + 1):
            M[col][j] /= pv
        for r in range(d):
            if r != col and abs(M[r][col]) > 1e-12:
                factor = M[r][col]
                for j in range(col, d + 1):
                    M[r][j] -= factor * M[col][j]
    return [M[i][d] for i in range(d)]


class ContextualBanditLearner:
    """LinUCB 上下文 Bandit"""

    def __init__(self, alpha: float = 1.0, lam: float = 1.0,
                 frozen_bonus: float = 0.15,
                 feature_dim: int = 64):
        self.alpha = alpha          # 置信上界系数（探索强度）
        self.lam = lam              # 岭回归正则
        self.frozen_bonus = frozen_bonus
        self.feature_dim = feature_dim
        # 每策略参数：A（d×d）, b（d）, n（样本数）
        self._params: Dict[str, dict] = {}

    def _params_for(self, sid: str, d: int) -> dict:
        if sid not in self._params:
            self._params[sid] = {
                "A": [[0.0] * d for _ in range(d)],
                "A_inv": [[1.0 / self.lam if i == j else 0.0 for j in range(d)]
                          for i in range(d)],  # (λI)⁻¹
                "b": [0.0] * d,
                "n": 0,
            }
        return self._params[sid]

    def observe(self, strategy_id: str, feature: List[float],
                reward: float) -> None:
        """记录一次 (特征, 奖励) 反馈。A⁻¹ 用 Sherman-Morrison 增量更新。"""
        d = len(feature)
        p = self._params_for(strategy_id, d)
        # A += x xᵀ
        for i in range(d):
            for j in range(d):
                p["A"][i][j] += feature[i] * feature[j]
        # A_inv 增量更新：A_inv' = A_inv - (A_inv x xᵀ A_inv) / (1 + xᵀ A_inv x)
        # 计算 t = A_inv · x
        t = [sum(p["A_inv"][i][j] * feature[j] for j in range(d))
             for i in range(d)]
        denom = 1.0 + sum(feature[i] * t[i] for i in range(d))
        if abs(denom) > 1e-12:
            scale = 1.0 / denom
            for i in range(d):
                row_i = p["A_inv"][i]
                ti = t[i]
                for j in range(d):
                    row_i[j] -= scale * ti * t[j]
        # b += x * reward
        for i in range(d):
            p["b"][i] += feature[i] * reward
        p["n"] += 1

    def _score(self, p: dict, feature: List[float]) -> float:
        """xᵀθ + α√(xᵀ A⁻¹ x)。用缓存的 A_inv，O(d²)。"""
        d = len(feature)
        if p["n"] == 0:
            return float("inf")
        # θ = A⁻¹ b（用缓存逆）
        theta = [sum(p["A_inv"][i][j] * p["b"][j] for j in range(d))
                 for i in range(d)]
        pred = sum(theta[i] * feature[i] for i in range(d)) if theta else 0.0
        # 不确定性 √(xᵀ A⁻¹ x)
        v = [sum(p["A_inv"][i][j] * feature[j] for j in range(d))
             for i in range(d)]
        unc = math.sqrt(max(0.0, sum(feature[i] * v[i] for i in range(d))))
        return pred + self.alpha * unc

    def choose(self, library: PolicyLibrary,
               feature: List[float],
               domain: Optional[str] = None,
               exclude: Optional[List[str]] = None) -> Strategy:
        """基于任务特征选择策略。"""
        candidates = library.active(domain)
        if exclude:
            candidates = [s for s in candidates if s.id not in exclude]
        if not candidates:
            raise RuntimeError("无可用策略")
        best = None
        best_score = -1e9
        for s in candidates:
            p = self._params.get(s.id)
            if p is None or p["n"] == 0:
                score = 1e6 + self.alpha  # 未尝试：优先探索
            else:
                score = self._score(p, feature)
            if s.status == STRATEGY_FROZEN:
                score += self.frozen_bonus
            if score > best_score:
                best_score = score
                best = s
        return best

    def predict_rewards(self, library: PolicyLibrary,
                        feature: List[float],
                        domain: Optional[str] = None) -> Dict[str, float]:
        """各策略对当前特征的预测奖励（可解释/调试）。"""
        out = {}
        for s in library.active(domain):
            p = self._params.get(s.id)
            if p is None or p["n"] == 0:
                out[s.name] = 0.0
                continue
            d = len(feature)
            theta = [sum(p["A_inv"][i][j] * p["b"][j] for j in range(d))
                     for i in range(d)]
            out[s.name] = round(
                sum(theta[i] * feature[i] for i in range(d))
                if theta else 0.0, 4)
        return out

    # ── 序列化 ──
    def export(self) -> dict:
        return {"alpha": self.alpha, "lam": self.lam,
                "feature_dim": self.feature_dim, "params": self._params}

    def import_(self, data: dict) -> None:
        if not data:
            return
        self.alpha = data.get("alpha", self.alpha)
        self.lam = data.get("lam", self.lam)
        self.feature_dim = data.get("feature_dim", self.feature_dim)
        self._params = data.get("params", {})
        # 旧状态无 A_inv：按当前 A 补算（或退化为对角）
        for p in self._params.values():
            if "A_inv" not in p:
                d = len(p.get("b", []))
                p["A_inv"] = [[1.0 / self.lam if i == j else 0.0
                               for j in range(d)] for i in range(d)]
                # 用 A 重建：对每个策略完整算一次（一次性，可接受）
                A = p.get("A")
                if A and d:
                    # 解 A v = e_i 得到逆矩阵列
                    cols = []
                    for i in range(d):
                        e = [1.0 if j == i else 0.0 for j in range(d)]
                        cols.append(_solve_ridge(A, e, self.lam))
                    p["A_inv"] = [[cols[j][i] for j in range(d)]
                                  for i in range(d)]
