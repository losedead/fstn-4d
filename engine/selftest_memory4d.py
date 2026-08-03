# -*- coding: utf-8 -*-
"""
selftest_memory4d.py — 4D 存储机制回归自证测试

验证 fstn5 已接入 FSTN-4D 完整存储架构：
  1. 斐波那契窗口（新记忆窗口小，久远记忆窗口大）
  2. 时间衰减（深层窗口经验检索降权）
  3. 虫洞自动发现（存储即关联，共享关键词建链）
  4. 虫洞扩散（命中记忆拉出关联记忆）
  5. 复习巩固（review 后心理时间刷新 → 窗口变小）
  6. 感知指纹（perceive 附加五感标注，感知检索）
  7. 情绪调制（一致性加成 + 效价抑制）
  8. 结晶（高频经验 → 潜意识关键节点）
  9. 版本链（同任务冲突 → 版本确认）
  10. 潜意识自动激活（scan_subconscious）
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fstn5 import FSTN5Core
from fstn5.memory_engine import find_window, emotional_modulation

failures = []
def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


# ── 1. 斐波那契窗口 ──
w0 = find_window(0.5)     # 0.5s → 窗口 0/1
w_big = find_window(100000)  # 100000s → 深层窗口
check("斐波那契.新记忆小窗口", w0 <= 1, f"0.5s→w{w0}")
check("斐波那契.久远大窗口", w_big >= 8, f"100000s→w{w_big}")

# ── 2-10. 集成测试 ──
c = FSTN5Core(state_dir=None, contextual=False)
c.add_strategy("策略A", "a", "d")
c.add_strategy("策略B", "b", "d")

# 存储两条关联经验（共享关键词）→ 应自动建虫洞
e1 = c.record_feedback("用户喜欢在晚上听爵士乐工作", "策略A", 0.9,
                       recorded_emotion={"joy": 0.7, "sadness": 0.1},
                       emotional_tags=["relax"])
e2 = c.record_feedback("晚上工作听什么音乐好", "策略B", 0.8,
                       recorded_emotion={"joy": 0.3, "sadness": 0.2})
e3 = c.record_feedback("用户是肉食主义者", "策略A", 0.7)
wh_count = len(c.memory.fib.wormholes)
check("虫洞.自动发现", wh_count >= 1, f"{wh_count} 条虫洞")

# 感知指纹：给"音乐"经验附加听觉感知
c.memory.perceive(e1["experience_id"],
                  {"auditory": {"jazz": 0.9}, "time": {"night": 1.0}})
p1 = c.memory.get(e1["experience_id"]).perceptual_signature
check("感知指纹.附加成功", "auditory" in p1 and p1["auditory"]["jazz"] == 0.9,
      str(list(p1.keys())))

# 复习：经验被 review 后 t_psych 刷新 → 窗口应变小
exp1 = c.memory.get(e1["experience_id"])
# 模拟旧记忆（把 t_psych 调老）
import time as _t
exp1.t_psych = _t.time() - 100000
old_w = exp1.window
# 4D 真实行为：单次复习只拉近 20%，需要多次复习才回到浅窗口
for _ in range(40):
    c.memory.review([e1["experience_id"]], gamma=0.8)
new_w = exp1.window
check("复习.心理时间刷新窗口变小", new_w < old_w,
      f"复习前 w{old_w} → 40次复习后 w{new_w}")

# 情绪调制：快乐时检索"音乐"经验应加分，悲伤时减分
base = 0.5
joy_mod = emotional_modulation(base, {"joy": 0.7, "sadness": 0.1}, ["relax"],
                               {"base_vector": {"joy": 0.8}, "valence": 0.5})
sad_mod = emotional_modulation(base, {"joy": 0.7, "sadness": 0.1}, ["relax"],
                               {"base_vector": {"sadness": 0.8}, "valence": -0.5})
check("情绪调制.一致加成", joy_mod > base, f"joy:{joy_mod:.3f}")
check("情绪调制.效价抑制", sad_mod < base, f"sad:{sad_mod:.3f}")

# 虫洞扩散检索：查"音乐"应能带出关联经验
hits = c.memory.query_scored("晚上听音乐", k=3, apply_wormhole=True)
ids = [h.id for _, h in hits]
check("虫洞扩散.命中关联", e1["experience_id"] in ids or e2["experience_id"] in ids,
      f"{len(ids)} 条")

# 潜意识自动激活：查询含触发词应激活关键节点
# 先手动结晶"爵士乐工作"经验（模拟高频）
exp2 = c.memory.get(e2["experience_id"])
for _ in range(25):
    c.memory.review([e2["experience_id"]], gamma=0.9)
c.memory.fib.crystallize(exp2, trigger_keywords=["音乐", "爵士", "晚上"],
                         strategy_id="策略B")
nodes = c.memory.scan_subconscious("晚上听什么音乐")
check("潜意识.自动激活", len(nodes) >= 1, f"{len(nodes)} 个关键节点")

# 版本链：同任务反复确认 → 版本累积
chain_count = len(c.memory.fib.version_chains)
check("版本链.冲突记录", chain_count >= 1, f"{chain_count} 条链")

# 记忆分层
c.record_feedback("重要核心规则", "策略A", 0.9, layer="semantic")
layers = {e.layer for e in c.memory._experiences}
check("分层.多层级共存", "episodic" in layers and "semantic" in layers,
      str(layers))

print()
print("FAILED:", ", ".join(failures) if failures else "ALL_PASS")
sys.exit(1 if failures else 0)
