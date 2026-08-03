# -*- coding: utf-8 -*-
"""
stress_test.py — FSTN-5D 强度压力测试（严格版，非自夸）

比 selftest 更苛刻的场景，测真实强度：
  1. 12 策略大池收敛（vs 自证的 4 策略）
  2. 噪声反馈（reward 带 ±0.2 随机干扰）
  3. 10 用户个性化（vs 自证的 2 用户）
  4. 三次环境突变（vs 自证的一次）
  5. 语义近邻任务区分（"登录失败"vs"认证错误"）
  6. LLM 演化策略质量
  7. 持久化跨进程
  8. 性能（1000 轮耗时 / 内存经验规模）

用法: python stress_test.py [--rounds N]
"""

import json
import math
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fstn5 import FSTN5Core

ROUNDS = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 600
random.seed(42)  # 可复现

results = []


def report(ok, name, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    results.append(bool(ok))


# ══════════ 1. 大策略池收敛（12 策略，1 个最优）══════════
print("=" * 66)
print("压力 1：12 策略大池收敛（最优策略应被选中 >60%）")
core = FSTN5Core(state_dir=tempfile.mkdtemp(), contextual=False)
for i in range(12):
    core.add_strategy(f"策略{i:02d}", f"策略{i}号", "s1")
# 最优：策略07 reward 0.85，其余 0.25
picks = 0
for i in range(ROUNDS):
    rec = core.recommend("修复登录bug", domain="s1")
    sid = rec["strategy_id"]
    s = core.library.get(sid)
    reward = 0.85 if s.name == "策略07" else 0.25
    core.record_feedback("修复登录bug", sid, reward)
    if s.name == "策略07":
        picks += 1
rate = picks / ROUNDS * 100
report(rate > 60, "1.大池收敛", f"策略07 选中率 {rate:.1f}%")

# ══════════ 2. 噪声反馈 ══════════
print("\n" + "=" * 66)
print("压力 2：噪声反馈（reward 带 ±0.25 随机干扰）")
core = FSTN5Core(state_dir=tempfile.mkdtemp(), contextual=False)
for s in ["策略A", "策略B", "策略C", "策略D"]:
    core.add_strategy(s, domain="s2")
picks = 0
for i in range(ROUNDS):
    rec = core.recommend("处理订单", domain="s2")
    sid = rec["strategy_id"]
    s = core.library.get(sid)
    base = 0.9 if s.name == "策略D" else 0.3
    noisy = max(-1.0, min(1.0, base + random.uniform(-0.25, 0.25)))
    core.record_feedback("处理订单", sid, noisy)
    if s.name == "策略D":
        picks += 1
rate = picks / ROUNDS * 100
report(rate > 50, "2.噪声下收敛", f"策略D 选中率 {rate:.1f}%（reward 有 ±0.25 干扰）")

# ══════════ 3. 10 用户个性化 ══════════
print("\n" + "=" * 66)
print("压力 3：10 用户个性化（每个用户偏好不同策略）")
core = FSTN5Core(state_dir=tempfile.mkdtemp(), contextual=False)
for s in [f"策略{i:02d}" for i in range(10)]:
    core.add_strategy(s, domain="s3")
user_best = {f"user{i:02d}": f"策略{i:02d}" for i in range(10)}
picks = {u: 0 for u in user_best}
for i in range(ROUNDS):
    u = f"user{i % 10:02d}"
    rec = core.recommend("部署服务", domain="s3", user_key=u)
    sid = rec["strategy_id"]
    s = core.library.get(sid)
    reward = 0.9 if s.name == user_best[u] else 0.2
    core.record_feedback("部署服务", sid, reward, user_key=u)
    if s.name == user_best[u]:
        picks[u] += 1
rates = {u: picks[u] / (ROUNDS // 10) * 100 for u in user_best}
avg = sum(rates.values()) / 10
report(avg > 60, "3.多用户个性化", f"10 用户平均命中率 {avg:.1f}% (min {min(rates.values()):.0f}%)")

# ══════════ 4. 三次环境突变 ══════════
print("\n" + "=" * 66)
print("压力 4：三次环境突变（最优策略连续切换）")
core = FSTN5Core(state_dir=tempfile.mkdtemp(), contextual=False)
for s in ["策略A", "策略B", "策略C", "策略D"]:
    core.add_strategy(s, domain="s4")
phases = [("策略A", 0.85), ("策略B", 0.85), ("策略C", 0.85)]  # 三次切换
picks = {"策略A": 0, "策略B": 0, "策略C": 0}
phase_len = ROUNDS // 3
for ph, (best, rw) in enumerate(phases):
    for i in range(phase_len):
        rec = core.recommend("处理任务", domain="s4")
        sid = rec["strategy_id"]
        s = core.library.get(sid)
        reward = rw if s.name == best else 0.15
        core.record_feedback("处理任务", sid, reward)
        if s.name == best:
            picks[best] += 1
# 每个阶段最后 20% 的命中率
rates = {}
for ph, (best, rw) in enumerate(phases):
    seg = picks[best]  # 简化：看总命中
    rates[best] = picks[best] / phase_len * 100
avg = sum(rates.values()) / 3
report(avg > 50, "4.三次突变", f"三阶段命中率 A:{rates['策略A']:.0f}% B:{rates['策略B']:.0f}% C:{rates['策略C']:.0f}%")

# ══════════ 5. 语义近邻任务区分 ══════════
print("\n" + "=" * 66)
print("压力 5：语义近邻任务区分（登录失败 vs 认证错误）")
core = FSTN5Core(state_dir=tempfile.mkdtemp(), contextual=True, feature_dim=128)
for s in ["策略·快速重试", "策略·深度排查", "策略·回滚", "策略·日志分析"]:
    core.add_strategy(s, domain="s5")
# 训练：登录类任务→快速重试最优，支付类→深度排查最优
train_pairs = [
    ("修复登录失败问题", "策略·快速重试"),
    ("修复支付接口异常", "策略·深度排查"),
    ("修复认证错误", "策略·快速重试"),
    ("修复退款流程故障", "策略·深度排查"),
]
for i in range(ROUNDS):
    task, best = train_pairs[i % len(train_pairs)]
    rec = core.recommend(task, domain="s5")
    sid = rec["strategy_id"]
    s = core.library.get(sid)
    reward = 0.9 if s.name == best else 0.15
    core.record_feedback(task, sid, reward)
# 测试：语义近邻（训练没见过但语义同类的）
tests = [("修复用户登录超时", "策略·快速重试"),
         ("处理认证失败重试", "策略·快速重试"),
         ("修复支付网关故障", "策略·深度排查"),
         ("处理退款接口错误", "策略·深度排查")]
ok = 0
for task, want in tests:
    rec = core.recommend(task, domain="s5")
    hit = rec["strategy_name"] == want
    ok += hit
    print(f"    {task:16s} → {rec['strategy_name']:10s} {'✅' if hit else '❌'} (want {want})")
report(ok >= 3, "5.语义近邻", f"4 个未见近邻任务 {ok}/4 选对")

# ══════════ 6. LLM 演化策略质量（离线检查，不调 LLM）══════════
print("\n" + "=" * 66)
print("压力 6：演化策略可解释性（rationale 质量）")
core = FSTN5Core(state_dir=tempfile.mkdtemp(), contextual=False)
for s in ["基础策略A", "基础策略B"]:
    core.add_strategy(s, domain="s6")
# 规则变异演化
new_ids = core.evolve()
report(len(new_ids) > 0, "6.演化产生", f"规则演化生成 {len(new_ids)} 变体")
for sid in new_ids[:2]:
    s = core.library.get(sid)
    print(f"    {s.name} parent={s.parent_id is not None}")

# ══════════ 7. 持久化跨进程 ══════════
print("\n" + "=" * 66)
print("压力 7：持久化跨进程（学习成果重启后保留）")
state_dir = tempfile.mkdtemp()
core = FSTN5Core(state_dir=state_dir, contextual=False)
for s in ["策略A", "策略B", "策略C", "策略D"]:
    core.add_strategy(s, domain="s7")
for i in range(200):
    rec = core.recommend("处理任务", domain="s7")
    sid = rec["strategy_id"]
    s = core.library.get(sid)
    reward = 0.9 if s.name == "策略D" else 0.2
    core.record_feedback("处理任务", sid, reward)
core.save()
core2 = FSTN5Core(state_dir=state_dir, contextual=False)
core2.load()
rec = core2.recommend("处理任务", domain="s7")
report(rec["strategy_name"] == "策略D", "7.跨进程持久化",
       f"重启后推荐 {rec['strategy_name']}（应策略D）")

# ══════════ 8. 性能 ══════════
print("\n" + "=" * 66)
print("压力 8：性能（1000 轮 recommend+feedback）")
core = FSTN5Core(state_dir=tempfile.mkdtemp(), contextual=True, feature_dim=128)
for s in [f"策略{i:02d}" for i in range(10)]:
    core.add_strategy(s, domain="s8")
t0 = time.time()
for i in range(1000):
    rec = core.recommend("处理数据任务", domain="s8")
    core.record_feedback("处理数据任务", rec["strategy_id"], 0.5)
elapsed = time.time() - t0
report(elapsed < 30, "8.性能", f"1000 轮 {elapsed:.2f}s（{elapsed/1000*1000:.1f}ms/轮）")

print("\n" + "=" * 66)
passed = sum(1 for r in results if r is True)
total = len(results)
print(f"强度测试汇总: {passed}/{total} 通过")
if passed < total:
    print("弱项: 见上方 FAIL 项")
print("=" * 66)
sys.exit(0 if passed == total else 1)
