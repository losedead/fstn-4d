"""
FSTN-4D ←→ Hermes Memory Bridge
==================================
将 FSTN-4D 引擎的"潜意识层"（关键节点）同步到 Hermes 的 memory 工具，
使其在每次对话中自动注入，无需显式加载 Skill。

Architecture:
  FSTN-4D Engine (Python)          Hermes Memory (persistent)
  ┌─────────────────────┐          ┌──────────────────────┐
  │ Key Nodes (crystallized) │◄──sync──►│ memory entries       │
  │ Subconscious Layer       │          │ (auto-injected/turn) │
  └─────────────────────┘          └──────────────────────┘

Usage (from Hermes execute_code):
  bridge = MemoryBridge()
  bridge.sync_to_hermes()   # push key nodes → Hermes memory
  bridge.sync_from_hermes() # pull Hermes memory → engine
"""

import sys
import os
import json
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from fstn_core import FSTN4DEngine


class MemoryBridge:
    """
    FSTN-4D ←→ Hermes Memory 双向同步桥。
    在 Hermes 的 execute_code 上下文中运行时，
    通过调用 hermes_tools 的 memory/write_file 操作持久化状态。
    """

    def __init__(self, engine: FSTN4DEngine = None):
        self.engine = engine or FSTN4DEngine()
        self.engine.load_state()

    def get_sync_snapshot(self) -> dict:
        """获取需要同步到 Hermes memory 的关键数据快照"""
        knodes = self.engine.memory.get_all_key_nodes()

        if not knodes:
            return {"status": "empty", "key_nodes": [], "stats": self.engine.memory.get_statistics()}

        node_data = []
        for node in knodes:
            node_data.append({
                "id": node.id,
                "content": node.content,
                "triggers": node.auto_trigger_keywords,
                "priority": round(node.priority, 2),
                "activation_count": node.activation_count,
            })

        return {
            "status": "ok",
            "key_nodes": node_data,
            "stats": self.engine.memory.get_statistics(),
            "emotion_dominant": self.engine.emotion.get_current()["dominant"],
        }

    def build_memory_payloads(self) -> list:
        """生成可写入 Hermes memory 的条目列表"""
        snapshot = self.get_sync_snapshot()
        payloads = []

        # 每个关键节点生成一条记忆
        for node in snapshot.get("key_nodes", []):
            payload = (
                f"[FSTN-KN] {node['content']} "
                f"触发:{'|'.join(node['triggers'][:5])} "
                f"优先级:{node['priority']}"
            )
            payloads.append(payload)

        # 引擎状态概要
        stats = snapshot.get("stats", {})
        state_payload = (
            f"[FSTN-STATE] 记忆:{stats.get('total_memories',0)}个 "
            f"节点:{stats.get('key_nodes',0)}个 "
            f"虫洞:{stats.get('active_wormholes',0)}条 "
            f"情绪:{snapshot.get('emotion_dominant','neutral')}"
        )
        payloads.append(state_payload)

        return payloads

    def restore_from_payloads(self, payloads: list):
        """从 Hermes memory 条目恢复关键节点到引擎"""
        for payload in payloads:
            if not payload.startswith("[FSTN-KN]"):
                continue
            # 解析: "[FSTN-KN] 内容 触发:kw1|kw2|kw3 优先级:1.5"
            try:
                content_part = payload.replace("[FSTN-KN] ", "")
                # 简单解析
                if "触发:" in content_part:
                    content, rest = content_part.split("触发:", 1)
                    triggers_str = rest.split("优先级:")[0] if "优先级:" in rest else rest
                    triggers = [t.strip() for t in triggers_str.split("|") if t.strip()]
                else:
                    content = content_part
                    triggers = []

                # 在引擎中创建关键节点
                node_id = f"key_restored_{hashlib.md5(content.encode()).hexdigest()[:10]}"
                from fstn_memory import KeyNode
                node = KeyNode(
                    id=node_id,
                    content=content.strip(),
                    auto_trigger_keywords=triggers,
                    priority=1.0,
                )
                self.engine.memory.key_nodes[node_id] = node
            except Exception:
                continue

        return len(self.engine.memory.key_nodes)

    def generate_startup_script(self) -> str:
        """
        生成 Hermes 会话启动时可执行的 Python 代码。
        将此脚本嵌入 Skill 或 memory，在每轮对话前自动运行。
        """
        return """
# FSTN-4D Auto-Init
import sys, os
sys.path.insert(0, os.path.expanduser('~/Desktop/fstn_enhancement/engine'))
from fstn_core import FSTN4DEngine

# 全局单例（同一 execute_code 调用内有效）
if '_fstn_engine' not in dir():
    _fstn_engine = FSTN4DEngine()
    _fstn_engine.load_state()

# 分析当前输入
_utterance = '''__USER_MESSAGE__'''
_result = _fstn_engine.process_utterance(_utterance)
print(_fstn_engine.generate_reply_guidance())

# 输出情绪分析结果供 Agent 参考
_emo = _result['emotion']
print(f"EMOTION:{_emo['dominant']}|V:{_emo['valence']:.2f}|A:{_emo['arousal']:.2f}")
if _emo.get('complex_emotion'):
    print(f"COMPLEX:{_emo['complex_emotion']['emotion']}")
_bhv = _result['behavior']
if _bhv['is_perception_directed']:
    print(f"PERCEPTION_DIRECT:W_p={_bhv['perception_weight']:.2f}")
""".strip()


# ═══════════════════════════════════════════════════════════════
# 命令行接口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FSTN-4D Memory Bridge")
    parser.add_argument("--snapshot", action="store_true", help="Print sync snapshot as JSON")
    parser.add_argument("--payloads", action="store_true", help="Print memory payloads for Hermes")
    parser.add_argument("--script", action="store_true", help="Print auto-init startup script")
    parser.add_argument("--save", action="store_true", help="Save engine state to disk")
    args = parser.parse_args()

    bridge = MemoryBridge()

    if args.snapshot:
        print(json.dumps(bridge.get_sync_snapshot(), ensure_ascii=False, indent=2))

    if args.payloads:
        for p in bridge.build_memory_payloads():
            print(p)

    if args.script:
        print(bridge.generate_startup_script())

    if args.save:
        bridge.engine.save_state()
        print('{"status": "saved"}')
