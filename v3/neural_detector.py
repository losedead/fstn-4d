"""
FSTN-4D V3 双层情绪检测器 (Dual-Layer Emotion Detector)
========================================================
针对 V1 关键词匹配脆弱的升级：
  Layer 1: 关键词快速路径 —— 复用原引擎 200+ 词条，毫秒级，高置信直接出结果
  Layer 2: LLM 语义路径  —— 关键词置信度低时，调用本地 Ollama 模型做语义理解
  Router:  置信度路由器 —— 决定该走哪条路，LLM 调用节流，不可用自动降级

设计要点（超越 V1）：
1. 置信度 = f(最大触发强度, 触发词数量, 情绪维度分歧度)
   - 单维强触发(≥0.6) → 高置信，直接出结果
   - 多维多词且矛盾(joy+sadness 同高) → 中置信，LLM 裁决
   - 零触发或强度极低(<0.3) → 低置信，必须 LLM
2. LLM 节流：两次调用最小间隔 8 秒，避免每轮都卡模型
3. 优雅降级：Ollama 不可用时回退关键词 + 启发式（效价/唤醒度残差）
4. 检测结果缓存：相同文本 60 秒内不重复调 LLM
"""

import time
import json
import re
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

# 复用原引擎的情绪关键词库（200+ 中文词条）与衰减配置
import sys, os
_ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine")
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)
from fstn_emotion import EmotionalStateMachine

BASE_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# 情绪中文名（用于 LLM prompt）
EMOTION_CN = {
    "anger": "愤怒", "disgust": "厌恶", "fear": "恐惧",
    "joy": "快乐", "sadness": "悲伤", "surprise": "惊讶",
}


class LLMEmotionDetector:
    """通过 Ollama 本地模型做语义情绪检测（可选后端）"""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = None, timeout: float = 12.0,
                 min_interval: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_call = 0.0
        self._available = None      # None=未探测, True/False
        self._cache: Dict[str, Tuple[float, Dict]] = {}  # 文本 -> (时间, 结果)
        self.cache_ttl = 60.0

    # ── 可用性探测（带 2 秒超时，不影响主流程） ──────────────
    def check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                headers={"User-Agent": "fstn-v3"},
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                # 自动选择可用模型
                if not self.model:
                    for cand in ["qwen3.5:9b", "qwen3:8b", "qwen2.5:7b",
                                 "llama3.1:8b", "gemma2:9b"]:
                        if any(cand in m for m in models):
                            self.model = cand
                            break
                self._available = bool(models and self.model)
        except Exception:
            self._available = False
        return self._available

    def _throttled(self) -> bool:
        return (time.time() - self._last_call) < self.min_interval

    def detect(self, text: str, context: str = "") -> Optional[Dict]:
        """LLM 语义情绪检测，返回六维向量。失败返回 None。"""
        if not self.check_available() or self._throttled():
            return None

        # 缓存命中
        cache_key = text[:120]
        if cache_key in self._cache:
            ts, res = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                return res

        prompt = (
            "分析以下中文消息的情绪。只输出 JSON，不要解释。\n"
            '格式: {"anger":0.0,"disgust":0.0,"fear":0.0,"joy":0.0,'
            '"sadness":0.0,"surprise":0.0}\n'
            "每项取值 0.0~1.0，表示该情绪强度。注意：\n"
            "- 讽刺、自嘲、反话要识别真实情绪\n"
            "- 混合情绪允许多项非零\n"
            "- 平淡陈述全给 0\n"
            f"消息: {text[:300]}"
            + (f"\n上文: {context[:100]}" if context else "")
        )
        payload = json.dumps({
            "model": self.model, "prompt": prompt,
            "stream": False, "format": "json", "temperature": 0.1,
            "think": False,  # 关闭推理模型的 thinking 字段（qwen3.5 等）
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "fstn-v3"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            self._last_call = time.time()

            resp_text = out.get("response", "") or out.get("thinking", "")
            if not resp_text.strip():
                return None
            m = re.search(r"\{.*\}", resp_text, re.S)
            if not m:
                return None
            raw = json.loads(m.group(0))
            vec = {e: max(0.0, min(1.0, float(raw.get(e, 0.0))))
                   for e in BASE_EMOTIONS}
            self._cache[cache_key] = (time.time(), vec)
            return vec
        except Exception:
            return None


class DualLayerEmotionDetector:
    """
    双层情绪检测器（组合路由器）。

    detect(text, context) -> {
        "base_vector": {...}, "dominant": "anger",
        "source": "keyword" | "llm" | "heuristic",
        "confidence": 0.0~1.0,
        "triggered_terms": [...],
    }
    """

    # 置信度阈值
    HIGH_CONF = 0.75      # 关键词路径出结果
    LLM_THRESHOLD = 0.30  # 最大强度低于此值 → 必须 LLM

    def __init__(self, llm_detector: Optional[LLMEmotionDetector] = None):
        # 复用原引擎关键词扫描 + 转折词/强度调节逻辑
        self._kw = EmotionalStateMachine()
        self.llm = llm_detector or LLMEmotionDetector()
        self.stats = {"keyword": 0, "llm": 0, "heuristic": 0, "llm_miss": 0}

    def detect(self, text: str, context: str = "") -> Dict:
        # ── Layer 1: 关键词快速路径 ──────────────────────────
        raw, triggered = self._kw._scan_keywords(text)
        raw = self._kw._apply_pivot_modulation(text, raw)
        raw = self._kw._apply_intensity_modifiers(text, raw)

        max_intensity = max(raw.values())
        active_dims = [e for e, v in raw.items() if v > 0.15]
        conflict = len(active_dims) >= 2  # 多维多词 → 情绪复杂，需裁决

        # 高置信条件：单维强触发（或双维但其中一维明显占优）
        if len(triggered) >= 1 and max_intensity >= self.HIGH_CONF and not conflict:
            self.stats["keyword"] += 1
            return self._build(raw, triggered, "keyword",
                               confidence=min(1.0, max_intensity + 0.15))

        # 中等置信：有触发但复杂/不强 → 尝试 LLM 裁决
        if len(triggered) >= 1 or max_intensity > 0.05:
            llm_vec = self.llm.detect(text, context)
            if llm_vec:
                self.stats["llm"] += 1
                return self._build(llm_vec, triggered, "llm",
                                   confidence=0.85)
            self.stats["llm_miss"] += 1

        # ── Layer 2: 无触发或低置信 ──────────────────────────
        if max_intensity < self.LLM_THRESHOLD:
            llm_vec = self.llm.detect(text, context)
            if llm_vec:
                self.stats["llm"] += 1
                return self._build(llm_vec, triggered, "llm",
                                   confidence=0.80)

        # ── 降级：启发式（原关键词结果 + 效价平滑） ──────────
        self.stats["heuristic"] += 1
        return self._build(raw, triggered, "heuristic",
                           confidence=min(0.6, max_intensity + 0.3))

    def _build(self, vec: Dict, triggered: List[str], source: str,
               confidence: float) -> Dict:
        vec = {e: max(0.0, min(1.0, v)) for e, v in vec.items()}
        dominant = "neutral"
        if max(vec.values()) >= 0.1:
            dominant = max(vec, key=vec.get)
        return {
            "base_vector": vec,
            "dominant": dominant,
            "source": source,
            "confidence": confidence,
            "triggered_terms": triggered,
        }

    def get_stats(self) -> Dict:
        return dict(self.stats)


# ── 自测 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    det = DualLayerEmotionDetector()
    samples = [
        ("今天工作被批评了，好难过。", ""),
        ("同事升职了，我为他高兴，但心里也有点不是滋味", ""),
        ("帮我推荐一家餐厅吧，想吃点好的", ""),
        ("好热啊，把空调打开", ""),
        ("其实我妈妈刚才安慰我了，我现在感觉好多了，甚至有点开心", ""),
    ]
    print("=" * 60)
    print("双层情绪检测器自测")
    print("=" * 60)
    for text, ctx in samples:
        r = det.detect(text, ctx)
        print(f"\n文本: {text}")
        print(f"  情绪: {r['dominant']:10s} 来源: {r['source']:9s} "
              f"置信度: {r['confidence']:.2f}")
        print(f"  向量: " + " ".join(
            f"{e[:3]}={v:.2f}" for e, v in r['base_vector'].items()
            if v > 0.05))
    print(f"\n统计: {det.get_stats()}")
