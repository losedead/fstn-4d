# -*- coding: utf-8 -*-
"""
e2e_v4.py — v4 引擎端到端回归测试（补齐工程缺口后的完整链路）

验证：
  1. 情绪检测（基础+复杂+干扰）        2. 感知追踪+耦合+反向调制
  3. 记忆（存储/复习/结晶/版本/遗忘）   4. HNSW 检索
  5. 感知嵌入+通感图+跨通道检索         6. 四路融合检索
  7. 状态持久化（save/load）           8. 性能自测（HNSW vs 线性）
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def main():
    print("=" * 60)
    print("FSTN-4D v4 端到端回归")
    print("=" * 60)

    from v4_engine import FSTN4DEngineV4
    from v4_hnsw_index import HNSWMemoryIndex
    from v4_perceptual_space import PerceptualIndex, SynesthesiaGraph

    tmp = tempfile.mkdtemp(prefix="hermes-fstn-e2e-")
    eng = FSTN4DEngineV4(state_dir=tmp, prefer_embedding="local")

    # ── 1. 情绪 ──
    print("\n[1] 情绪检测")
    r = eng.process_utterance("我不生气，只是有点难过")
    check("否定翻转", r["emotion"]["dominant"] in ("sadness", "neutral"),
          f"dominant={r['emotion']['dominant']}")
    r = eng.emotion.detect("同事升职了，我为他高兴但心里不是滋味")
    ce = r.complex_emotion or {}
    check("复杂情绪-嫉妒", "jealousy" in str(ce.get("emotion", "")),
          f"complex={ce.get('emotion')}")

    # ── 2. 感知 ──
    print("\n[2] 感知追踪与耦合")
    eng.process_utterance("好热啊，把空调打开")
    cur = eng.perception.get_current()
    tc = cur.get("thermal", {}).get("thermal_comfort")
    check("感知-热(负舒适)", tc == -0.7, f"thermal_comfort={tc}")
    db = eng.perception.detect_direct_behavior("好热，开空调")
    check("感知直接行为", db is not None and db.get("perception_weight", 0) >= 0.8,
          f"W_p={db.get('perception_weight') if db else 'N/A'}")

    # ── 3. 记忆 ──
    print("\n[3] 记忆管理")
    mid = eng.memory.ingest("用户每天早上喝咖啡", recorded_emotion={"joy": 0.4})
    for _ in range(25):
        eng.memory.review([mid], gamma=0.9)
    node = eng.memory.crystallize(mid, trigger_keywords=["咖啡", "早餐", "习惯"])
    check("结晶", node is not None, f"key_node={node}")
    v = eng.memory.add_version("饮食偏好", "用户开始吃鱼肉了")
    chain = eng.memory.version_chains.get("饮食偏好")
    check("版本链", chain is not None and len(chain.versions) >= 1)

    # ── 4. HNSW ──
    print("\n[4] HNSW 检索")
    eng.process_utterance("用户喜欢爵士乐，工作到深夜")
    eng.process_utterance("用户养了一只猫叫豆包")
    eng.process_utterance("用户是素食主义者")
    eng._ensure_hnsw()
    check("HNSW 后端", eng._hnsw_index is not None and eng._hnsw_index.size() > 0,
          f"size={eng._hnsw_index.size() if eng._hnsw_index else 0}")
    hits = eng.retrieve_memories("喝咖啡的习惯", k=3)
    check("HNSW 检索命中", any("咖啡" in h.content for h in hits),
          f"top={hits[0].content[:15] if hits else 'N/A'}")

    # ── 5. 感知嵌入 + 通感 ──
    print("\n[5] 感知嵌入与通感图")
    eng.process_utterance("昨天看了篝火，红色的火焰在黑暗里闪烁")
    eng.process_utterance("傍晚的夕阳把天空染成红色，还有闪烁的光")
    phits = eng.perceptual_index.search(["红色", "闪烁"], k=3)
    check("感知检索", len(phits) >= 2, f"命中 {len(phits)} 条")
    syn_count = eng.synesthesia_graph.export_state()["active_count"]
    check("通感自动建链", syn_count >= 1, f"通感链接 {syn_count} 条")
    emo = eng.get_synesthesia_emotion("灯光刺眼又吵")
    check("通感-情绪", "anger" in emo or "fear" in emo, f"emotion={emo}")

    # ── 6. 四路融合 ──
    print("\n[6] 四路融合检索")
    fused = eng.retrieve_memories("红色闪烁的东西", k=3)
    check("融合检索返回", len(fused) >= 2, f"返回 {len(fused)} 条")

    # ── 7. 持久化 ──
    print("\n[7] 状态持久化")
    eng.save_state()
    eng2 = FSTN4DEngineV4(state_dir=tmp, prefer_embedding="local")
    eng2.load_state()
    check("save/load", len(eng2.memory.memories) == len(eng.memory.memories),
          f"{len(eng2.memory.memories)} 条记忆")
    check("通感图持久化",
          eng2.synesthesia_graph.export_state()["active_count"] == syn_count,
          f"{eng2.synesthesia_graph.export_state()['active_count']} 条")

    # ── 8. 性能 ──
    print("\n[8] 性能自测")
    idx = HNSWMemoryIndex(dim=64)
    import numpy as np
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(50000, 64)).astype(np.float32)
    for i, v in enumerate(vecs[:50000]):
        idx.add(f"m{i}", v)
    idx.rebuild_all()
    bench = idx.benchmark(n_queries=200)
    check("HNSW 5万条毫秒级", bench["avg_ms"] < 5.0,
          f"{bench['avg_ms']:.3f} ms/query")

    print("\n" + "=" * 60)
    print(f"端到端结果: {PASS} 通过 / {FAIL} 失败")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
