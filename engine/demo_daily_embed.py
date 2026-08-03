# -*- coding: utf-8 -*-
"""
demo_daily_embed.py — FSTN-5D 日常嵌入演示

模拟 Hermes 一天的真实对话，验证"嵌入日常"后的行为：
  1. 每轮 analyze() → 情绪/感知/策略推荐
  2. 根据用户反馈 learn() → 引擎学习
  3. 多次交互后：策略推荐开始区分情绪（悲伤→共情，开心→跳跃）
  4. 4D 记忆积累：虫洞/关键节点/经验
  5. 状态持久化：重启后学习成果保留

用法：python demo_daily_embed.py
"""

import os
import sys
import tempfile

# 用临时 state 隔离（不污染真实状态）
_tmp = tempfile.mkdtemp(prefix="fstn_daily_")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FSTN_DAILY_STATE", _tmp)

import builtins
# 强制使用临时 state
import auto_loader
auto_loader.STATE_DIR = _tmp
if hasattr(builtins, "_fstn_v5"):
    del builtins._fstn_v5

print("═" * 60)
print("FSTN-5D 日常嵌入演示（state 隔离于临时目录）")
print("═" * 60)

# 一天的对话：情绪各异，反馈各异
convo = [
    # (用户话语, 反馈 reward)  reward: 正=推荐策略有效，负=无效
    ("今天好累，什么都不想干", 0.9),      # 悲伤/疲惫 → 共情
    ("帮我跑一下那个量化回测脚本", 0.8),   # 中性任务 → 直接执行
    ("烦死了，代码又报错", 0.7),           # 愤怒 → 冷静分析
    ("哈哈哈终于跑通了！", 0.9),           # 开心 → 跳跃陪伴
    ("有点焦虑，怕做不完", 0.8),           # 焦虑 → 安全感
    ("又失败了，我好没用", 0.9),           # 羞愧/悲伤 → 共情
    ("明天要答辩，好紧张", 0.7),           # 焦虑 → 安全感
    ("今天心情不错，聊点别的？", 0.8),     # 开心 → 跳跃陪伴
]

print("\n── 第一天对话 ──")
for i, (text, rw) in enumerate(convo):
    g = auto_loader.analyze(text)
    # 提取策略行
    strat_line = [l for l in g.splitlines() if l.startswith("[策略]")]
    emo_line = [l for l in g.splitlines() if l.startswith("[情绪]")]
    print(f"[{i+1}] {text[:18]:<20} | {emo_line[0][:30] if emo_line else ''}")
    print(f"      {strat_line[0] if strat_line else ''}")
    # 用户接受 → 学习
    r = auto_loader.learn(rw, "用户接受该服务方式")
    if not r.get("ok"):
        print(f"      learn 失败: {r}")

print("\n── 学习后记忆状态 ──")
snap = auto_loader.memory_snapshot()
print(f"4D 记忆: {snap['memories']} 条 | 虫洞: {snap['wormholes']} | "
      f"关键节点: {snap['key_nodes']}")
print(f"经验: {snap['experiences']} 条 | 策略: {snap['strategies']} 个")
rep = snap.get("self_report", {})
print(f"平均奖励: {rep.get('avg_reward', 0):.3f} | 尝试次数: {rep.get('total_trials', 0)}")

print("\n── 重启模拟（新进程加载同一 state）──")
# 模拟重启：删除内存实例，重新 analyze（状态从磁盘加载）
if hasattr(builtins, "_fstn_v5"):
    del builtins._fstn_v5
for text, rw in convo[:3]:
    g = auto_loader.analyze(text)
    strat_line = [l for l in g.splitlines() if l.startswith("[策略]")]
    print(f"  {text[:18]:<20} → {strat_line[0][7:] if strat_line else '?'}")

print("\n── 策略学习报告（哪些策略在涨/在跌）──")
rep = auto_loader.memory_snapshot().get("self_report", {})
for s in rep.get("learning_summary", []):
    print(f"  {s['name']:<10} ema={s['reward_ema']:+.3f} trials={s['trials']} "
          f"weight={s['weight']:.2f}")

print("\n✅ 日常嵌入演示完成")
