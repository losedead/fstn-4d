# -*- coding: utf-8 -*-
"""
v5_engine.py — FSTN4DEngineV5：自进化版 FSTN-4D

在 FSTN-4D（v4）之上长出自我学习/自我改进/自我创新能力——
不是另起炉灶，而是 4D 的完整升级：

┌─ 保留 4D 全部能力 ──────────────────────────────┐
│  emotion（情绪检测/复杂情绪/衰减）                │
│  perception（七通道感知/耦合/直接行为）           │
│  memory（分层记忆/遗忘/结晶/虫洞）                │
│  perceptual_index（感知嵌入）+ synesthesia（通感）│
│  HNSW 检索 / process_utterance 完整管线           │
├─ 新增 5D 自进化层（复用 fstn5_core 机制）─────────┤
│  self_learn:     任务→策略→结果，自动学习最优      │
│  self_recommend: 结合用户情感状态推荐策略          │
│  self_evolve:    生成新策略变体，择优淘汰          │
│  self_report:    学习进度报告（"变聪明"的证据）    │
└──────────────────────────────────────────────────┘

关键设计：情感是"用户独特性"的感知层——
  FSTN-4D 的情绪检测结果自动构成 UserVector 的情感维度，
  感知/记忆状态构成上下文维度，用于个性化路由。

用法：
  from v5_engine import FSTN4DEngineV5
  e = FSTN4DEngineV5(state_dir="state")     # 4D 全部能力可用
  rec = e.self_recommend("修复登录bug", domain="agent_service")
  e.self_learn("修复登录bug", rec["strategy_id"], reward=0.9)
  e.self_evolve()
"""

import os
import sys
import time

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)
# 复用 fstn5_core 的机制（若在独立目录，则尝试导入）
_FSTN5 = None
for cand in [os.path.join(ENGINE_DIR, "fstn5"),
             os.path.join(os.path.dirname(ENGINE_DIR), "fstn5_core", "fstn5")]:
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)
try:
    from fstn5 import FSTN5Core, UserVector
    _FSTN5 = True
except ImportError:
    _FSTN5 = False

from v4_engine import FSTN4DEngineV4


class FSTN4DEngineV5(FSTN4DEngineV4):
    """自进化版 FSTN-4D：4D 能力 + 5D 学习层"""

    def __init__(self, state_dir: str = None, prefer_embedding: str = "auto",
                 **kw):
        super().__init__(state_dir=state_dir,
                         prefer_embedding=prefer_embedding, **kw)
        if not _FSTN5:
            raise ImportError(
                "需要 fstn5 核心包（fstn5_core）。"
                "把 fstn5 目录放到 engine/ 下或设置 sys.path。")
        # 自进化内核（状态存引擎同目录下的 .fstn5_state）
        state5 = os.path.join(state_dir or ENGINE_DIR, ".fstn5_state") \
            if state_dir else os.path.join(ENGINE_DIR, ".fstn5_state")
        self.self_core = FSTN5Core(state_dir=state5)

    # ══════════════ 核心桥梁：FSTN-4D 状态 → UserVector ══════════════

    def build_user_vector(self, user_key: str = "",
                          traits: dict = None) -> UserVector:
        """用 FSTN-4D 当前状态构建用户向量。

        情感维度 = 引擎情绪检测结果（艾克曼六维 + 复杂情绪）
        上下文   = 感知主导通道 + 活跃耦合 + 记忆特征
        这是"情感用于分辨用户独特性"的实现点。
        """
        cur = self.emotion.get_current()
        emotion = {k: float(v) for k, v in cur.get("base_vector", {}).items()
                   if float(v) > 0.05}
        # 复杂情绪也并入情感维度
        ce = cur.get("complex_emotion")
        if isinstance(ce, dict) and ce.get("emotion"):
            emotion[ce["emotion"]] = emotion.get(ce["emotion"], 0.0) + 0.3
        ctx = {"perceptual_dominant": self.perception.get_dominant()[0] or "",
               "active_couplings": [f"{a}:{b}" for a, b in
                                    self.perception.get_active_coupling_states()]}
        uv = UserVector(emotion=emotion,
                        traits=dict(traits or {}),
                        context=ctx)
        if user_key:
            uv.context["user_key"] = user_key
        return uv

    # ══════════════ 5D 自进化接口 ══════════════

    def self_recommend(self, task_text: str, domain: str = "generic",
                       user_key: str = "", traits: dict = None,
                       task_tags=None) -> dict:
        """推荐策略（结合 FSTN-4D 当前情感/感知状态做个性化）。"""
        uv = self.build_user_vector(user_key, traits)
        return self.self_core.recommend(
            task_text, user=uv, user_key=user_key,
            domain=domain, task_tags=task_tags)

    def self_learn(self, task_text: str, strategy_id: str, reward: float,
                   user_key: str = "", traits: dict = None,
                   task_tags=None, note: str = "",
                   sync_perception: bool = False,
                   recorded_emotion: dict = None,
                   emotional_tags: list = None,
                   perceptual_signature: dict = None,
                   layer: str = "episodic") -> dict:
        """记录反馈 → 自我学习。

        默认【不】用任务文本污染用户状态——任务文本的情绪检测
        是"任务的情绪"不是"用户的情绪"，混入会导致学习失真
        （实测 bug：任务文本触发 anger 让所有任务路由偏向同一策略）。
        需要同步感知时显式传 sync_perception=True。

        recorded_emotion/emotional_tags/perceptual_signature：
        4D 存储机制——经验记忆附带情绪/感知指纹（协议一：存储即关联）。
        """
        if sync_perception:
            self.process_utterance(task_text)
        uv = self.build_user_vector(user_key, traits)
        return self.self_core.record_feedback(
            task_text, strategy_id, reward,
            user=uv, user_key=user_key, task_tags=task_tags,
            feedback_note=note,
            recorded_emotion=recorded_emotion,
            emotional_tags=emotional_tags,
            perceptual_signature=perceptual_signature,
            layer=layer)

    def self_evolve(self, parents=None, use_llm: bool = False,
                    domain: str = None, count: int = 2,
                    client=None) -> list:
        """自我创新：生成新策略变体。

        use_llm=True 时用 LLM 基于策略池+经验+失败案例生成【全新思路】
        策略（真创新）；False 时用规则变异（复制+参数，零依赖）。
        """
        if use_llm:
            return self.self_core.evolve_with_llm(
                domain=domain, count=count, client=client)
        return self.self_core.evolve(parents)

    def self_prune(self, min_trials: int = 10, parent_ratio: float = 0.7) -> int:
        """淘汰劣质变体"""
        return self.self_core.prune(min_trials, parent_ratio)

    def self_settle(self) -> int:
        """定期清理：遗忘 + 淘汰"""
        return self.self_core.settle()

    def self_report(self, domain: str = None) -> dict:
        """学习进度报告（"变聪明"的证据）"""
        return self.self_core.status(domain)

    def self_add_strategy(self, name: str, description: str = "",
                          domain: str = "generic") -> str:
        """注册策略"""
        return self.self_core.add_strategy(name, description, domain)

    # ══════════════ 持久化 ══════════════

    def save_state(self):
        super().save_state()
        self.self_core.save()

    def load_state(self):
        ok = super().load_state()
        self.self_core.load()
        return ok
