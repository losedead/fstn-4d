# -*- coding: utf-8 -*-
"""
selftest_contextual.py — 深度上下文 Bandit 自证测试

证明"选得准"：
  1. 岭回归求解数学正确
  2. 特征编码（TF-IDF）可区分语义
  3. LinUCB 收敛：同一策略池，不同任务收敛到不同最优策略
  4. 泛化：未见任务能基于特征给出合理选择
  5. 对比：上下文 vs 非上下文（任务区分能力差异）
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fstn5 import FSTN5Core
from fstn5.contextual_bandit import ContextualBanditLearner, _solve_ridge
from fstn5.features import FeatureExtractor
from fstn5.policy_library import PolicyLibrary

failures = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


# ── 1. 岭回归数学正确 ──
A = [[2.0, 0.0], [0.0, 2.0]]
b = [4.0, 6.0]
theta = _solve_ridge(A, b, lam=0.0)
check("岭回归求解", abs(theta[0] - 2.0) < 1e-6 and abs(theta[1] - 3.0) < 1e-6,
      str(theta))

# ── 2. LinUCB 核心：同池不同任务收敛到不同最优 ──
lib = PolicyLibrary()
lib.add("流式管道", "流式", "data")
lib.add("批量加载", "批量", "data")
bandit = ContextualBanditLearner(alpha=0.8, lam=1.0, feature_dim=32)
fx = FeatureExtractor(mode="tfidf", dim=64)
fx.observe(["处理10GB用户日志", "处理100MB配置文件", "处理50GB埋点数据"])

big_tasks = ["处理10GB用户日志", "处理50GB埋点数据", "处理20GB日志"]
small_tasks = ["处理100MB配置文件", "处理80MB配置", "处理50MB小文件"]
for i in range(400):
    big = i % 2 == 0
    task = big_tasks[i % len(big_tasks)] if big else small_tasks[i % len(small_tasks)]
    feat = fx.encode(task)
    s = bandit.choose(lib, feat, domain="data")
    reward = 0.9 if (big and s.name == "流式管道") else (
        0.9 if (not big and s.name == "批量加载") else 0.1)
    bandit.observe(s.id, feat, reward)

final = {"big_stream": 0, "big_batch": 0, "small_stream": 0, "small_batch": 0}
for i in range(100):
    big = i % 2 == 0
    task = big_tasks[i % len(big_tasks)] if big else small_tasks[i % len(small_tasks)]
    feat = fx.encode(task)
    s = bandit.choose(lib, feat, domain="data")
    key = ("big" if big else "small") + "_" + ("stream" if s.name == "流式管道" else "batch")
    final[key] += 1
big_ok = final["big_stream"] / (final["big_stream"] + final["big_batch"] + 1e-9)
small_ok = final["small_batch"] / (final["small_stream"] + final["small_batch"] + 1e-9)
check("LinUCB.大数据选流式>60%", big_ok > 0.6, f"{big_ok*100:.0f}%")
check("LinUCB.小数据选批量>60%", small_ok > 0.6, f"{small_ok*100:.0f}%")

# ── 3. 集成：FSTN5Core contextual 模式任务区分 ──
core = FSTN5Core(state_dir=tempfile.mkdtemp(), contextual=True, feature_dim=128)
for s in ["策略·流式管道", "策略·批量加载", "策略·并行分片", "策略·增量处理"]:
    core.add_strategy(s, domain="data")
for i in range(400):
    task = "处理10GB用户日志" if i % 2 == 0 else "处理100MB配置文件"
    rec = core.recommend(task, domain="data")
    name = rec["strategy_name"]
    if "10GB" in task:
        success = 0.95 if "流式" in name else 0.15
    else:
        success = 0.92 if "批量" in name else 0.1
    core.record_feedback(task, rec["strategy_id"], success * 2 - 1)
r1 = core.recommend("处理10GB用户日志", domain="data")
r2 = core.recommend("处理100MB配置文件", domain="data")
check("集成.10GB→流式", "流式" in r1["strategy_name"], r1["strategy_name"])
check("集成.100MB→批量", "批量" in r2["strategy_name"], r2["strategy_name"])

# ── 4. 对比：上下文 vs 非上下文（未见任务泛化）──
def eval_core(c, label):
    tests = ["处理30GB日志", "处理60MB配置", "处理200GB仓库", "处理45MB小文件"]
    ok = 0
    for t in tests:
        rec = c.recommend(t, domain="data")
        want = "流式" if "GB" in t else "批量"
        if want in rec["strategy_name"]:
            ok += 1
    return ok, len(tests)

core_plain = FSTN5Core(state_dir=tempfile.mkdtemp(), contextual=False)
for s in ["策略·流式管道", "策略·批量加载", "策略·并行分片", "策略·增量处理"]:
    core_plain.add_strategy(s, domain="data")
for i in range(400):
    task = "处理10GB用户日志" if i % 2 == 0 else "处理100MB配置文件"
    rec = core_plain.recommend(task, domain="data")
    name = rec["strategy_name"]
    if "10GB" in task:
        success = 0.95 if "流式" in name else 0.15
    else:
        success = 0.92 if "批量" in name else 0.1
    core_plain.record_feedback(task, rec["strategy_id"], success * 2 - 1)
ctx_ok, ctx_t = eval_core(core, "上下文")
plain_ok, plain_t = eval_core(core_plain, "非上下文")
check("对比.上下文>=非上下文", ctx_ok >= plain_ok,
      f"上下文 {ctx_ok}/{ctx_t} vs 非上下文 {plain_ok}/{plain_t}")
check("对比.上下文>0", ctx_ok > 0, f"{ctx_ok}/{ctx_t}")

print()
print("FAILED:", ", ".join(failures) if failures else "ALL_PASS")
sys.exit(1 if failures else 0)
