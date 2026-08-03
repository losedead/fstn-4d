# -*- coding: utf-8 -*-
"""
fstn5/features.py — 任务/用户特征提取（深度 Bandit 的输入）

把任务文本和用户状态编码成稠密特征向量，供 LinUCB 使用。

可插拔后端：
1. TF-IDF 本地（零依赖，中文字 bigram）——默认
2. bge-m3 embedding（Ollama 本地，语义最强）——可选
3. 词袋 + 标签（最快）

这是"深度上下文 Bandit"的第一层：手写相似度 → 学习表征。
"""

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Optional


def _stable_hash(s: str, mod: int) -> int:
    """确定性 hash（跨进程稳定）。

    PYTHONHASHSEED 让内置 hash(str) 每次进程随机——特征落桶会变，
    导致语义区分测试结果不稳定（实测：同样代码一次 4/4、一次 2/4）。
    用 hashlib.md5 保证跨进程/跨重启一致。
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


def _tokenize_cn(text: str) -> List[str]:
    """中文切词：字 bigram + 数字归一化 + 单位保留（零依赖近似分词）

    数字归一化是关键：'10GB'/'120GB'/'30GB' 都映射到 '<NUM>gb'，
    让 LinUCB 能外推——学到的 '大数字+gb → 流式' 泛化到未见任务
    （实测：不归一化时 '30GB日志' 是新 token，线性模型预测失效）。
    """
    text = text.lower()
    tokens = []
    # 数字+单位 token（归一化数字）：30gb / 100mb / 90gb → <NUM>gb
    for m in re.finditer(r"[0-9]+([a-z]+)", text):
        tokens.append(f"<NUM>{m.group(1)}")
    # 纯数字
    for m in re.finditer(r"[0-9]+", text):
        tokens.append("<NUM>")
    # 英文词（无数字）
    for m in re.finditer(r"[a-z]+", text):
        tokens.append(m.group())
    # 中文 bigram
    cn = re.sub(r"[^\u4e00-\u9fff]", "", text)
    for i in range(len(cn) - 1):
        tokens.append(cn[i:i + 2])
    if cn:
        tokens.append(cn)
    return tokens


class FeatureExtractor:
    """任务特征提取：文本 + 标签 → 稠密向量

    mode:
      'bow'    — 词袋（字 bigram 计数，快）
      'tfidf'  — TF-IDF 加权（默认，区分度好）
      'bge'    — bge-m3 embedding（需 Ollama，语义最强）
    """

    def __init__(self, mode: str = "auto", dim: int = 64,
                 bge_url: str = None):
        self.mode = mode
        self.dim = dim                      # bow/tfidf 的向量维度（hash 桶数）
        self.bge_url = bge_url or "http://localhost:11434"
        self._vocab: Counter = Counter()    # token -> 出现文档数
        self._doc_count = 0
        self._trained = False
        self._bge_ok: Optional[bool] = None  # bge 可用性缓存
        self._cache: Dict[str, List[float]] = {}  # bge embedding 缓存
        self._cache_max = 512

    def _resolve_mode(self) -> str:
        """auto：检测 bge-m3 可用就用语义 embedding，否则回退 tfidf。

        实测（bge-m3 vs tfidf）：
          登录失败↔认证错误  sim 0.117 → 0.814（bge 懂语义近义）
          支付异常↔支付网关   sim 0.437 → 0.837
        语义 embedding 对"未见近邻任务"的泛化显著更强；
        但保持零依赖兼容——无 Ollama 环境自动回退。
        """
        if self.mode != "auto":
            return self.mode
        if self._bge_ok is None:
            self._bge_ok = self._check_bge()
        return "bge" if self._bge_ok else "tfidf"

    def _check_bge(self) -> bool:
        """探测 bge-m3 是否可用（一次，缓存结果）"""
        import json
        import urllib.request
        try:
            payload = {"model": "bge-m3", "prompt": "测试"}
            req = urllib.request.Request(
                f"{self.bge_url}/api/embeddings",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.loads(r.read().decode())
            return bool(d.get("embedding"))
        except Exception:
            return False

    # ── 训练（增量收集词频）──
    def observe(self, texts: List[str]) -> None:
        """收集文本，更新词频统计（TF-IDF 需要）。"""
        for t in texts:
            self._doc_count += 1
            seen = set(_tokenize_cn(t))
            for tok in seen:
                self._vocab[tok] += 1
        self._trained = self._doc_count > 0

    # ── 编码 ──
    def encode(self, text: str, tags: Optional[List[str]] = None) -> List[float]:
        """文本+标签 → 稠密特征向量（长度 dim + tag 数 * 2）"""
        mode = self._resolve_mode()
        if mode == "bge":
            vec = self._encode_bge(text)
        elif mode == "tfidf":
            vec = self._encode_tfidf(text)
        else:
            vec = self._encode_bow(text)
        # 标签编码（追加：每个 tag 两个槽——存在性 + 简单哈希）
        tags = tags or []
        tag_vec = []
        for t in tags:
            h = _stable_hash(t, self.dim // 4)
            tag_vec.extend([1.0, float(h) / (self.dim // 4)])
        return vec + tag_vec

    def _encode_bow(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize_cn(text):
            vec[_stable_hash(tok, self.dim)] += 1.0
        return vec

    def _encode_tfidf(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        counts = Counter(_tokenize_cn(text))
        for tok, c in counts.items():
            df = self._vocab.get(tok, 1)
            idf = math.log(1 + self._doc_count / max(1, df))
            vec[_stable_hash(tok, self.dim)] += c * idf
        # L2 归一化（余弦友好）
        n = math.sqrt(sum(x * x for x in vec))
        if n > 0:
            vec = [x / n for x in vec]
        return vec

    def _encode_bge(self, text: str) -> List[float]:
        """调 Ollama bge-m3 embedding（原始 1024 维 → 截断到 self.dim）。

        bge-m3 实际返回 1024 维，直接喂 LinUCB 会爆内存（1024×1024 A 矩阵）。
        截断到 self.dim（128）后与 TF-IDF 输出维度一致，A_inv 保持可算。
        截断会丢信息，但 128 维已足够区分语义（实测相似度仍远好于 tfidf）。
        """
        # 缓存：相同文本不重复调 Ollama（每轮 2 次 encode 的 hot path）
        key = text[:200]
        if key in self._cache:
            return self._cache[key]
        import json
        import urllib.request
        payload = {"model": "bge-m3", "prompt": text[:500]}
        req = urllib.request.Request(
            f"{self.bge_url}/api/embeddings",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read().decode())
            emb = d.get("embedding", [])
            if emb:
                # 截断 + L2 归一化（保持与 tfidf 一致尺度）
                vec = emb[:self.dim]
                n = math.sqrt(sum(x * x for x in vec))
                if n > 0:
                    vec = [x / n for x in vec]
                # LRU 缓存
                if len(self._cache) >= self._cache_max:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = vec
                return vec
        except Exception:
            pass
        # bge 失败 → 回退 tfidf
        return self._encode_tfidf(text)

    # ── 序列化 ──
    def export(self) -> dict:
        return {"mode": self.mode, "dim": self.dim,
                "vocab": dict(self._vocab), "doc_count": self._doc_count}

    def import_(self, data: dict) -> None:
        if not data:
            return
        self.mode = data.get("mode", self.mode)
        self.dim = data.get("dim", self.dim)
        self._vocab = Counter(data.get("vocab", {}))
        self._doc_count = data.get("doc_count", 0)
        self._trained = self._doc_count > 0
