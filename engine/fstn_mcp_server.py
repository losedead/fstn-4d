# -*- coding: utf-8 -*-
"""
fstn_mcp_server.py — FSTN-4D 全量 MCP 服务器

把 FSTN-4D v4 引擎的【完整能力面】封装成标准 MCP 服务（不是单一情感功能），
对外暴露 5 类 24 个工具，任何 MCP 客户端（Claude Desktop / Cursor / Hermes /
自研 client）均可按 JSON-RPC 调用。

┌─ 情绪层（emotion_*）───────────────────────────────────────┐
│ emotion_detect       检测文本情绪（六维+复杂+主导）            │
│ emotion_current      当前情绪状态（含衰减/残留）               │
│ emotion_trajectory   情绪轨迹（历史变化）                     │
│ emotion_reset        重置情绪状态机                           │
│ emotion_modulate     记忆的情绪调制（一致性/对立/特异性）       │
├─ 感知层（perception_*）─────────────────────────────────────┤
│ perception_update    从话语更新七通道感知状态                  │
│ perception_current   当前感知状态（热/冷/苦/饿/疼/噪音/疲劳）    │
│ perception_couple    感知→情绪耦合（返回耦合后情绪+触发项）     │
│ perception_behavior  感知直接行为检测（W_p 驱动）              │
│ perception_modulate  情绪→感知反向调制（亢奋不知疼等）          │
│ perception_fingerprint 构建感知指纹（通感质量因子）            │
├─ 记忆层（memory_*）─────────────────────────────────────────┤
│ memory_ingest        写入记忆（层/重要度/情绪/感知签名）        │
│ memory_retrieve      混合检索（语义+感知+虫洞+通感融合）        │
│ memory_review        复习（强化 gamma）                      │
│ memory_crystallize   结晶为关键节点（潜意识）                 │
│ memory_scan_subconscious 潜意识扫描（触发关键节点）            │
│ memory_wormhole      创建虫洞联想                             │
│ memory_stats         记忆统计（分布/窗口/结晶）                │
├─ 通感层（synesthesia_*）────────────────────────────────────┤
│ synesthesia_link     建立通感关联（跨通道意象）                │
│ synesthesia_emotion  文本的通感情绪映射                        │
│ synesthesia_qualities 当前感知质量因子（尖锐/温暖/沉重/新鲜）   │
├─ 对话层（chat_*）───────────────────────────────────────────┤
│ chat_reply           微调模型情感感知对话回复                  │
│ chat_utterance       完整处理管线（情绪→感知→记忆→回复指引）    │
│ chat_session_report  会话报告（统计/轨迹/建议）                │
└────────────────────────────────────────────────────────────┘

运行：
  python fstn_mcp_server.py                # stdio（MCP 客户端默认）
  python fstn_mcp_server.py --transport sse --port 8765  # SSE 远程

依赖：mcp, torch(cuda), transformers, peft（+ 引擎本体 v4_*）
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ENGINE_DIR)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("需要 mcp 包: pip install mcp", file=sys.stderr)
    sys.exit(1)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = os.path.join(ENGINE_DIR, "fstn_lora_out")

# ── 全局懒加载 ──
_engine = None
_model = None
_tokenizer = None


def get_engine():
    """懒加载 FSTN-4D v4 引擎（情绪/感知/记忆/通感全模块）"""
    global _engine
    if _engine is None:
        from v4_engine import FSTN4DEngineV4
        import tempfile
        state_dir = os.path.join(ENGINE_DIR, ".fstn_mcp_state")
        os.makedirs(state_dir, exist_ok=True)
        _engine = FSTN4DEngineV4(state_dir=state_dir, prefer_embedding="local")
        _engine.load_state()
        print(f"[fstn-mcp] FSTN-4D v4 引擎就绪 (state={state_dir})", file=sys.stderr)
    return _engine


def get_model():
    """懒加载微调模型（4bit + LoRA）"""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    print("[fstn-mcp] 加载微调模型 (4bit+LoRA)...", file=sys.stderr)
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if os.path.isdir(ADAPTER_DIR):
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    else:
        print(f"[fstn-mcp] ⚠️ adapter 缺失，用基座", file=sys.stderr)
    model.eval()
    print(f"[fstn-mcp] 模型就绪 ({time.time()-t0:.1f}s)", file=sys.stderr)
    _model, _tokenizer = model, tokenizer
    return model, tokenizer


def _safe(fn, *a, **kw):
    """统一异常包装 → JSON 可序列化"""
    try:
        return fn(*a, **kw)
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


def _to_jsonable(obj):
    """MemoryEntry/KeyNode/Wormhole 等对象 → dict"""
    if hasattr(obj, "__dict__"):
        d = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return {k: (_to_jsonable(v) if hasattr(v, "__dict__") else v)
                for k, v in d.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


mcp = FastMCP("fstn-4d",
              instructions="FSTN-4D 情感记忆引擎全量接口：情绪检测/感知追踪/记忆管理/通感/对话")


# ══════════════ 情绪层 ══════════════

@mcp.tool()
def emotion_detect(text: str) -> str:
    """检测文本情绪：艾克曼六维向量 + 复杂情绪 + 主导情绪 + 置信度。"""
    eng = get_engine()
    r = eng.emotion.detect(text)
    vec = {k: round(float(v), 2) for k, v in r.base_vector.items() if v > 0.05}
    complex_em = (r.complex_emotion or {}).get("emotion", "none")
    return json.dumps({
        "text": text, "dominant": r.dominant, "complex_emotion": complex_em,
        "vector": vec, "confidence": round(float(max(r.base_vector.values())), 2),
    }, ensure_ascii=False)


@mcp.tool()
def emotion_current() -> str:
    """当前情绪状态（含自然衰减后的残留，7 维感知耦合后的完整状态）。"""
    eng = get_engine()
    return json.dumps(eng.emotion.get_current(), ensure_ascii=False, default=str)


@mcp.tool()
def emotion_trajectory() -> str:
    """情绪轨迹：六维情绪随时间的变化序列。"""
    eng = get_engine()
    return json.dumps(eng.emotion.get_emotion_trajectory(), ensure_ascii=False, default=str)


@mcp.tool()
def emotion_modulate(memory_emotion: str, memory_tags: str = "[]") -> str:
    """计算记忆在当前情绪下的调制系数（一致性/效价对立/特异性）。
    参数 memory_emotion: 记忆情绪的 JSON 对象，如 {"joy":0.6};
    memory_tags: 记忆标签 JSON 数组。返回 modulation 各因子。"""
    import json as _j
    mem_em = _j.loads(memory_emotion)
    tags = _j.loads(memory_tags) if memory_tags else []
    eng = get_engine()
    return json.dumps(eng.emotion.get_emotional_modulation(mem_em, tags),
                      ensure_ascii=False)


@mcp.tool()
def emotion_reset() -> str:
    """重置情绪状态机（清空历史与当前状态）。"""
    eng = get_engine()
    eng.emotion.reset()
    return json.dumps({"ok": True}, ensure_ascii=False)


# ══════════════ 感知层 ══════════════

@mcp.tool()
def perception_update(utterance: str) -> str:
    """从话语更新七通道感知（热/冷/苦/饿/疼/噪音/疲劳），返回各通道变化。"""
    eng = get_engine()
    return json.dumps(eng.perception.update_from_utterance(utterance),
                      ensure_ascii=False, default=str)


@mcp.tool()
def perception_current() -> str:
    """当前感知状态 + 活跃耦合触发项。"""
    eng = get_engine()
    state = eng.perception.get_current()
    active = eng.perception.get_active_coupling_states()
    return json.dumps({"state": state, "active_couplings": active},
                      ensure_ascii=False, default=str)


@mcp.tool()
def perception_couple(emotional_state: str) -> str:
    """感知→情绪耦合：输入当前情绪 JSON，返回耦合后情绪增量 + 触发项。
    如热 → anger 增加、冷 → fear 增加。"""
    import json as _j
    em = _j.loads(emotional_state)
    eng = get_engine()
    delta, triggered = eng.perception.couple_emotion(em)
    return json.dumps({"delta": delta, "triggered": triggered},
                      ensure_ascii=False, default=str)


@mcp.tool()
def perception_behavior(utterance: str) -> str:
    """感知直接行为检测：如『好热』→ 开空调（W_p=0.85 优先级）。"""
    eng = get_engine()
    return json.dumps(eng.perception.detect_direct_behavior(utterance),
                      ensure_ascii=False, default=str)


@mcp.tool()
def perception_modulate(emotional_state: str) -> str:
    """情绪→感知反向调制：如亢奋抑制痛觉、悲伤降低味觉灵敏度。"""
    import json as _j
    em = _j.loads(emotional_state)
    eng = get_engine()
    return json.dumps(eng.perception.modulate_perception_by_emotion(em),
                      ensure_ascii=False, default=str)


@mcp.tool()
def perception_fingerprint(content: str) -> str:
    """构建感知指纹（含通感质量因子 sharpness/warmth/heaviness/freshness）。"""
    eng = get_engine()
    return json.dumps(eng.perception.build_perceptual_fingerprint(content),
                      ensure_ascii=False, default=str)


# ══════════════ 记忆层 ══════════════

@mcp.tool()
def memory_ingest(content: str, layer: str = "episodic", importance: str = "normal",
                  recorded_emotion: str = "{}", emotional_tags: str = "[]",
                  perceptual_signature: str = "{}") -> str:
    """写入记忆。layer: episodic/semantic/procedural; importance: high/normal/low;
    recorded_emotion: 六维情绪 JSON; emotional_tags: 标签数组; perceptual_signature: 感知签名。"""
    import json as _j
    eng = get_engine()
    mid = eng.memory.ingest(
        content, layer=layer, importance=importance,
        recorded_emotion=_j.loads(recorded_emotion),
        emotional_tags=_j.loads(emotional_tags) if emotional_tags else None,
        perceptual_signature=_j.loads(perceptual_signature) or None)
    return json.dumps({"memory_id": mid, "ok": True}, ensure_ascii=False)


@mcp.tool()
def memory_retrieve(query: str, k: int = 10, emotion_aware: bool = True) -> str:
    """混合检索记忆：语义(HNSW) + 感知 + 虫洞 + 通感 四路融合（v4）。"""
    eng = get_engine()
    results = eng.retrieve_memories(query, k=k, emotion_aware=emotion_aware)
    out = []
    for item in results:
        entry = item[0] if isinstance(item, tuple) else item
        score = item[1] if isinstance(item, tuple) else 0.0
        out.append({
            "memory_id": getattr(entry, "memory_id", getattr(entry, "id", "")),
            "content": getattr(entry, "content", ""),
            "layer": getattr(entry, "layer", ""),
            "importance": getattr(entry, "importance", ""),
            "score": round(float(score), 4) if isinstance(score, (int, float)) else 0.0,
            "review_count": getattr(entry, "review_count", 0),
        })
    return json.dumps({"query": query, "results": out}, ensure_ascii=False)


@mcp.tool()
def memory_review(memory_ids: str, gamma: float = None) -> str:
    """复习记忆（强化）。memory_ids: JSON 数组；gamma: 复习质量 0-1。"""
    import json as _j
    ids = _j.loads(memory_ids)
    eng = get_engine()
    n = eng.memory.review(ids, gamma=gamma)
    return json.dumps({"reviewed": n}, ensure_ascii=False)


@mcp.tool()
def memory_crystallize(memory_id: str, trigger_keywords: str = "[]") -> str:
    """结晶记忆为关键节点（潜意识自动触发）。trigger_keywords: 触发词 JSON 数组。"""
    import json as _j
    eng = get_engine()
    node_id = eng.memory.crystallize(
        memory_id, trigger_keywords=_j.loads(trigger_keywords) or None)
    return json.dumps({"crystallized": node_id is not None,
                       "key_node_id": node_id}, ensure_ascii=False)


@mcp.tool()
def memory_scan_subconscious(text: str) -> str:
    """潜意识扫描：输入文本是否触发已结晶的关键节点。"""
    eng = get_engine()
    node = eng.memory._scan_subconscious(text)
    if node is None:
        return json.dumps({"triggered": False}, ensure_ascii=False)
    return json.dumps({"triggered": True,
                       "node_id": getattr(node, "node_id", ""),
                       "content": getattr(node, "content", "")},
                      ensure_ascii=False)


@mcp.tool()
def memory_wormhole(source_id: str, target_id: str, type_: str = "co_occurrence",
                    reason: str = "") -> str:
    """创建虫洞联想（跨记忆连接）。"""
    eng = get_engine()
    ok = eng.memory.create_wormhole(source_id, target_id, type_=type_, reason=reason)
    return json.dumps({"ok": ok}, ensure_ascii=False)


@mcp.tool()
def memory_stats() -> str:
    """记忆统计：总量/分布/窗口/结晶节点/虫洞。"""
    eng = get_engine()
    stats = eng.memory.get_statistics()
    key_nodes = eng.memory.get_all_key_nodes()
    wormholes = eng.memory.get_active_wormholes()
    return json.dumps({
        "statistics": stats,
        "key_nodes": len(key_nodes),
        "wormholes": len(wormholes),
    }, ensure_ascii=False, default=str)


# ══════════════ 通感层 ══════════════

@mcp.tool()
def synesthesia_link(memory_id_a: str, memory_id_b: str, channel: str = "visual",
                     reason: str = "") -> str:
    """建立通感关联：两条记忆在同一感知通道上的意象相似则建链。"""
    eng = get_engine()
    idx = eng.perceptual_index
    sim = 0.0
    try:
        sim = idx.channel_similarity(memory_id_a, memory_id_b, channel)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    graph = eng.synesthesia_graph
    link = graph.link(memory_id_a, memory_id_b, channel, sim, reason)
    return json.dumps({"ok": link is not None, "similarity": round(float(sim), 4),
                       "channel": channel}, ensure_ascii=False)


@mcp.tool()
def synesthesia_emotion(text: str) -> str:
    """文本的通感情绪映射（跨通道意象触发的情绪）。"""
    eng = get_engine()
    return json.dumps(eng.get_synesthesia_emotion(text),
                      ensure_ascii=False, default=str)


@mcp.tool()
def synesthesia_qualities() -> str:
    """当前感知质量因子：sharpness/warmth/heaviness/freshness。"""
    eng = get_engine()
    q = eng.perception.get_synesthesia_qualities()
    return json.dumps([{"channel": c, "qualities": qs} for c, qs in q],
                      ensure_ascii=False, default=str)


# ══════════════ 对话层 ══════════════

@mcp.tool()
def chat_reply(text: str, max_new_tokens: int = 120) -> str:
    """用微调模型生成情感感知回复（带最近上下文）。"""
    import torch
    model, tokenizer = get_model()
    eng = get_engine()
    # 先走引擎：更新情绪+感知
    emo = eng.emotion.detect(text)
    eng.perception.update_from_utterance(text)
    dominant = emo.dominant
    prompt = (f"你是搭载 FSTN-4D 情感记忆引擎的陪伴型 Agent。"
              f"当前检测到用户情绪：{dominant}。\n\n用户：{text}\n[回复]")
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=0.8, top_p=0.9,
                             pad_token_id=tokenizer.pad_token_id)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    for sep in ["用户：", "助手：", "[回复]"]:
        resp = resp.split(sep)[0]
    return resp.strip()


@mcp.tool()
def chat_utterance(utterance: str, context: str = "") -> str:
    """完整处理管线：情绪检测→感知追踪→耦合→记忆检索→回复指引。
    返回引擎认为应该如何回应的完整上下文。"""
    eng = get_engine()
    result = eng.process_utterance(utterance, context=context)
    guidance = eng.generate_reply_guidance()
    return json.dumps({"engine_result": result, "guidance": guidance},
                      ensure_ascii=False, default=str)


@mcp.tool()
def chat_session_report() -> str:
    """会话报告：情绪轨迹、记忆统计、耦合状态、调制上下文。"""
    eng = get_engine()
    return json.dumps(eng.get_session_report(), ensure_ascii=False, default=str)


@mcp.tool()
def chat_save_state() -> str:
    """持久化引擎状态（记忆/关键节点/虫洞/通感图到磁盘）。"""
    eng = get_engine()
    eng.save_state()
    return json.dumps({"ok": True, "state_dir": os.path.join(ENGINE_DIR, ".fstn_mcp_state")},
                      ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"[fstn-mcp] FSTN-4D 全量 MCP 服务器启动 transport={args.transport}", file=sys.stderr)
    if args.transport == "sse":
        mcp.run(transport="sse", host="127.0.0.1", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
