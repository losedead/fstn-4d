"""
FSTN-4D V5 Hermes 自动加载器（日常嵌入版）
===========================================
将 FSTN4DEngineV5（4D 情绪记忆 + 5D 自进化学习）嵌入 Hermes 每轮对话：

每轮调用 analyze()：
  1. 情绪检测/感知追踪（4D 能力）→ 回复指导
  2. self_recommend：结合用户情绪状态推荐服务策略（5D 学习层）
  3. 学习闭环：推荐被使用后，调用 learn() 记录结果
  4. 状态自动持久化（重启不丢）

用法（Hermes execute_code / skill / cronjob）：
  import sys; sys.path.insert(0, r'<engine 目录>')
  from auto_loader import analyze, learn
  guidance = analyze(USER_MESSAGE)        # 每轮对话开头调用
  # ... Agent 按 guidance 调整回复 ...
  learn(USER_MESSAGE, chosen_strategy, reward)   # 回复被接受/拒绝后调用
"""

import builtins
import os
import sys

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

STATE_DIR = os.path.join(ENGINE_DIR, ".fstn_hermes_state")
os.makedirs(STATE_DIR, exist_ok=True)


def auto_init():
    """初始化或获取全局 v5 引擎实例（状态持久化到 engine/.fstn_hermes_state）"""
    if not hasattr(builtins, "_fstn_v5"):
        from v5_engine import FSTN4DEngineV5
        eng = FSTN4DEngineV5(state_dir=STATE_DIR)
        eng.load_state()
        builtins._fstn_v5 = eng
        builtins._fstn_v5._user_key = "hermes_default"
        _ensure_default_strategies(eng)
    return builtins._fstn_v5


DEFAULT_STRATEGIES = [
    # (名称, 描述, 领域, 适用情绪标签)
    ("共情优先回应", "先接住情绪再给方案；悲伤/焦虑时最有效", "agent_service",
     ["sadness", "anxiety", "shame", "fear"]),
    ("直接高效执行", "用户情绪中性/积极时，直接干活不绕弯", "agent_service",
     ["neutral", "joy", "anger"]),
    ("安全感建立", "焦虑/恐慌时给确定性、给边界、给兜底方案", "agent_service",
     ["anxiety", "fear"]),
    ("冷静分析模式", "愤怒时先不评判，复述事实+给可操作步骤", "agent_service",
     ["anger"]),
    ("适度跳跃陪伴", "用户开心时接住情绪，可轻松延伸话题", "agent_service",
     ["joy"]),
]


def _ensure_default_strategies(eng):
    """预置 agent_service 领域策略池（无则补，已存在不重复）"""
    existing = {s.name for s in eng.self_core.library.all()}
    for name, desc, domain, tags in DEFAULT_STRATEGIES:
        if name not in existing:
            sid = eng.self_add_strategy(name, desc, domain)
            # 记录情绪标签到策略描述（供演化/解释用）
            try:
                s = eng.self_core.library.get(sid)
                if s is not None and hasattr(s, "description"):
                    s.description = desc + f"  [适用情绪: {','.join(tags)}]"
            except Exception:
                pass
    eng.save_state()


def analyze(utterance: str, context: str = "", domain: str = "agent_service",
            traits: dict = None) -> str:
    """分析用户话语，返回指导文本（静默融入 Agent 思考过程）。

    v5 版：在 v1 情绪/感知指导之上，叠加 self_recommend 策略推荐——
    "这个用户此刻的情绪+上下文，该用什么服务策略"。
    """
    engine = auto_init()
    # 1. 4D 能力：情绪/感知/记忆检索
    result = engine.process_utterance(utterance, context)
    guidance = engine.generate_reply_guidance()

    lines = [guidance]
    emo = result["emotion"]
    bhv = result["behavior"]

    emo_line = f"[情绪] {emo['dominant']} V={emo['valence']:+.2f} A={emo['arousal']:.2f}"
    if emo.get("complex_emotion"):
        emo_line += f" 复杂:{emo['complex_emotion']['emotion']}"
    lines.append(emo_line)

    if bhv["is_perception_directed"]:
        lines.append(f"[行为] 感知直接驱动 W_p={bhv['perception_weight']:.2f}")
    else:
        lines.append("[行为] 情绪驱动")

    # 2. 5D 学习层：self_recommend（情绪状态 → 服务策略）
    try:
        rec = engine.self_recommend(utterance, domain=domain, traits=traits)
        lines.append(f"[策略] 推荐: {rec.get('strategy_name', '?')} "
                     f"来源: {rec.get('routing', '?')}")
        # 记住本次推荐（供 learn() 使用）
        engine._last_rec = (utterance, rec["strategy_id"], domain)
    except Exception as e:
        lines.append(f"[策略] (推荐失败: {e})")

    # 3. 持久化（每轮存一次，重启不丢）
    try:
        engine.save_state()
    except Exception:
        pass

    return "\n".join(lines)


def learn(reward: float, feedback_note: str = "", task_tags: list = None) -> dict:
    """学习闭环：记录上一次推荐的实际结果。

    reward ∈ [-1, 1]：正 = 推荐策略有效（用户接受了/结果好）
                       负 = 无效（用户拒绝了/结果差）
    引擎会自动：更新 bandit 权重 + 4D 记忆（带情绪/感知指纹）+
    个性化亲和力。这是"越用越懂你"的核心。
    """
    engine = auto_init()
    if not hasattr(engine, "_last_rec") or engine._last_rec is None:
        return {"ok": False, "error": "无待学习的推荐（先调用 analyze）"}
    task_text, sid, domain = engine._last_rec
    # 附带当前情绪快照（4D 存储机制：记忆带情绪）
    try:
        cur = engine.emotion.get_current()
        emo_snapshot = dict(cur.get("base_vector", {}) or {})
    except Exception:
        emo_snapshot = {}
    r = engine.self_learn(task_text, sid, reward=reward,
                          user_key=engine._user_key,
                          note=feedback_note,
                          task_tags=task_tags,
                          recorded_emotion=emo_snapshot)
    engine._last_rec = None
    engine.save_state()
    return r


def memory_snapshot() -> dict:
    """当前引擎记忆状态速查（供 Agent 了解用户画像/记忆层）"""
    engine = auto_init()
    snap = {
        "memories": len(engine.memory.memories),
        "key_nodes": len(engine.memory.key_nodes),
        "wormholes": len(engine.memory.wormholes),
        "strategies": len(engine.self_core.library.all()),
        "experiences": engine.self_core.memory.count(),
    }
    try:
        snap["self_report"] = engine.self_report()
    except Exception:
        pass
    return snap


if __name__ == "__main__":
    # 命令行自测
    g = analyze("今天好累啊，什么都不想干")
    print("=== analyze ===")
    print(g)
    print("\n=== learn(0.9) ===")
    print(learn(0.9, "用户接受推荐"))
    print("\n=== memory_snapshot ===")
    print(memory_snapshot())
