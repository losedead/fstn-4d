# -*- coding: utf-8 -*-
"""
FSTN-4D v2 检索提供者对比 (Ollama embedding vs 本地 TF-IDF)
============================================================
在同一个测试集上对比两种向量提供者的检索质量：
  - nomic-embed-text (Ollama, 274MB, 需已 pull)
  - jieba+TF-IDF (本地, 零依赖)

用例设计侧重"跨词面语义检索"——查询与目标记忆没有共享关键词，
只有 TF-IDF 向量和 embedding 才能召回：
  - 语义相关但无共词（"今晚吃什么" → "爱吃糖果"）
  - 上位词/下位词（"饮品" → "咖啡"）
  - 场景推理（"要睡觉了" → "失眠严重"）

用法：
    python v2_retrieval_compare.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fstn_memory import FibonacciMemoryEngine
from v2_vector_retrieval import VectorMemoryIndex, OllamaEmbedding, LocalTFIDF


# 检索测试集（侧重语义，非字面）
COMPARE_CASES = [
    # (查询, 期望命中的记忆片段, 类型)
    ("今晚吃什么好", "小红很爱吃糖果", "语义相关无共词"),
    ("有什么饮品推荐", "用户喜欢喝咖啡", "上位词"),
    ("晚上睡不着怎么办", "失眠严重", "场景推理"),
    ("快下班了，去哪个会议室开会", "项目Alpha的截止日期", "场景相关"),
    ("推荐一家素食餐厅", "素食主义者", "字面+语义"),
    ("下个月要出差一周", "上海出差三天", "语义扩展"),
    ("找个能专心写代码的地方", "安静的咖啡馆工作", "场景推理"),
    ("预算还够不够", "项目预算在五万以内", "语义相关"),
    ("周末想放松一下", "想学钢琴", "兴趣相关"),
    ("妈妈身体怎么样了", "妈妈上周做了手术", "人物关联"),
]


def run_compare():
    corpus = [
        "用户是素食主义者，不吃任何肉类",
        "小红很爱吃糖果，每天都吃一颗",
        "项目Alpha的截止日期是下周五",
        "用户最近失眠严重，晚上很难入睡",
        "用户喜欢喝咖啡，尤其是美式",
        "项目预算在五万以内",
        "下周二要去上海出差三天",
        "用户喜欢在安静的咖啡馆工作",
        "用户有一只叫团子的猫",
        "用户妈妈上周做了手术，恢复中",
        "用户在工作上遇到了瓶颈，考虑转行",
        "用户想学钢琴，已经报了班",
        "用户想学钢琴是因为想参加公司的年会表演",
        "用户昨晚加班到凌晨三点",
        "用户对海鲜过敏",
        "用户养了一只金毛叫大黄",
        "用户的电脑最近很卡，想换新的",
        "用户喜欢在周末爬山",
    ]

    mem = FibonacciMemoryEngine()
    for c in corpus:
        mem.ingest(c, layer="episodic")

    entries = list(mem.memories.values())

    # 本地 TF-IDF
    local_idx = VectorMemoryIndex.build(entries, prefer="local")

    # Ollama（若可用，中文优先选择 bge-m3）
    ollama_ok = OllamaEmbedding.preferred()._check_available()
    ollama_idx = None
    if ollama_ok:
        try:
            ollama_idx = VectorMemoryIndex.build(entries, prefer="ollama")
        except Exception as e:
            print(f"[warn] Ollama 构建失败: {e}")

    print("=" * 72)
    print("FSTN-4D v2 检索提供者对比")
    print(f"本地: TF-IDF+jieba    Ollama: {'可用' if ollama_ok else '不可用'}")
    print("=" * 72)

    local_hits = ollama_hits = 0
    total = len(COMPARE_CASES)

    for query, expect, ctype in COMPARE_CASES:
        l_hits = local_idx.query(query, k=5)
        l_contents = [e.content for _, e in l_hits]
        l_ok = any(expect in c for c in l_contents)

        o_ok = None
        o_contents = []
        if ollama_idx:
            o_hits = ollama_idx.query(query, k=5)
            o_contents = [e.content for _, e in o_hits]
            o_ok = any(expect in c for c in o_contents)
            if o_ok:
                ollama_hits += 1

        if l_ok:
            local_hits += 1

        lm = "✓" if l_ok else "✗"
        om = ("✓" if o_ok else "✗") if o_ok is not None else "-"
        print(f"\n[{ctype}] 查询: {query}  期望: {expect}")
        print(f"  local {lm}: {l_contents[0][:28] if l_contents else '∅'}")
        if o_contents:
            print(f"  ollama {om}: {o_contents[0][:28] if o_contents else '∅'}")

    print("\n" + "=" * 72)
    print(f"本地 TF-IDF 命中: {local_hits}/{total}")
    if ollama_idx:
        print(f"Ollama 命中:      {ollama_hits}/{total}")
    print("=" * 72)


if __name__ == "__main__":
    run_compare()
