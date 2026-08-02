# FSTN-4D v2 增强层 (Enhancement Layer)

针对 v1 引擎三大短板的增量升级，**完全向后兼容**，不修改 v1 任何代码。

## 架构总览

```
┌─────────────────────────────────────────────────────┐
│  FSTN4DEngineV2 (v2_engine.py)                      │
│  ┌─────────────────────────────────────────────┐    │
│  │ 情绪检测  EnhancedEmotionDetector           │    │
│  │   v1 关键词扫描 + 否定翻转 + 程度校准        │    │
│  │   + 语义词根增强 + 高信号词注入             │    │
│  └─────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────┐    │
│  │ 记忆检索  VectorMemoryIndex                 │    │
│  │   Ollama embedding (nomic-embed-text)       │    │
│  │   └─ 回退: jieba+TF-IDF 本地向量            │    │
│  │   + 否定感知 + 心理时间融合                 │    │
│  └─────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────┐    │
│  │ 耦合系数  CouplingLearner                   │    │
│  │   感知→情绪耦合系数在线学习 (EMA)           │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## 文件清单

| 文件 | 功能 |
|------|------|
| `v2_emotion_classifier.py` | 增强情绪检测（修复 v1 否定句/程度句硬伤） |
| `v2_vector_retrieval.py` | 向量化语义检索（Ollama / TF-IDF 双实现） |
| `v2_coupling_learner.py` | 耦合系数在线学习器 |
| `v2_engine.py` | 统一 v2 引擎（drop-in 替换 v1） |
| `v2_benchmark.py` | ablation 基准测试 |

## 快速开始

```bash
# 1. 拉取 embedding 模型（中文场景务必用 bge-m3，不要用 nomic-embed-text！）
ollama pull bge-m3        # 1.1GB, BAAI 多语言, 中文检索强
# ollama pull nomic-embed-text  # ❌ 英文为主，中文检索实测 5/10 不如本地 TF-IDF 8/10

# 2. 跑基准测试
python v2_benchmark.py

# 3. 跑端到端演示
python v2_engine.py --embedding auto   # auto: Ollama 可用则用，否则本地
python v2_engine.py --embedding local  # 强制本地 TF-IDF

# 4. 检索提供者对比（Ollama embedding vs 本地 TF-IDF）
python v2_retrieval_compare.py
```

## 替换 v1 引擎（应用层零改动）

```python
# 原来
from fstn_core import FSTN4DEngine
engine = FSTN4DEngine()

# 现在
from v2_engine import FSTN4DEngineV2
engine = FSTN4DEngineV2(prefer_embedding="auto")
```

接口与 v1 完全一致：`process_utterance` / `retrieve_memories` /
`review_memories` / `crystallize_if_ready` / `generate_reply_guidance` /
`save_state` / `load_state` / `get_session_report`。

## 基准结果（本机实测）

| 维度 | v1 | v2 | 提升 |
|------|----|----|------|
| 情感检测准确率 | 31% (5/16) | 100% (16/16) | +69pp |
| 记忆检索命中率 | 75% (6/8) | 100% (8/8) | +25pp |
| 耦合预测 MAE | 0.427 | 0.307 | -28.1% |

测试集覆盖：否定句（"我不生气"）、程度句（"有点难过"）、复杂社会情绪
（嫉妒/内疚/感激）、混合情绪（bittersweet）、跨词面语义检索。

## v2 修复的 v1 缺陷

### 1. 否定词翻转（v1 最严重硬伤）
- v1: `"我不生气"` → anger=0.85 ❌
- v2: `"我不生气"` → anger=0.05, dominant=neutral ✅
- 实现：否定词后 8 字符窗口内的情绪词强度压至 0.05；
  被否定翻转的情绪不再被语义词根增强抬升。

### 2. 程度副词校准
- v1: `"有点难过"` 与 `"难过死了"` 强度相同 ❌
- v2: 有点×0.45 / 很×1.30 / 死·炸·极×1.55 ✅

### 3. 高信号社交词注入
- v1: `"谢谢你"` → neutral（"谢谢"不在关键词表）❌
- v2: 谢谢→gratitude，内疚→guilt，嫉妒→jealousy ✅
- 实现：v1 复杂情绪检测要求 ≥2 个基础情绪非零，单维度高信号词
  需要显式注入（HIGH_SIGNAL_COMPLEX 表）。

### 4. 向量化语义检索（跨词面召回）
- v1: `"今晚吃什么好"` 检索不到 `"小红爱吃糖果"`（无共享关键词）❌
- v2: TF-IDF 向量余弦相似度命中 ✅
- 实现：jieba 分词 + 字 bigram（解决"素食主义者" vs "素食" 粒度不匹配）
  + TF-IDF 加权 + L2 归一化；查询含否定词时对否定窗口内词降权。

### 5. 耦合系数在线学习
- v1: 静态系数 `thermal:too_hot → anger 0.4` 永不变化
- v2: EMA 在线学习，用实际检测情绪反馈修正系数（MIN_SAMPLES=5 冷启动保护）

## 依赖

- Python 3.8+（本机实测 3.11）
- `numpy` — 向量计算
- `jieba` — 中文分词
- `requests` — Ollama API（可选，无则自动回退本地）

## 下一步（可选升级）

- [ ] 情绪检测换微调模型（distillBERT 情感分类，替代关键词表）
- [ ] 检索换 Faiss HNSW（百万级记忆毫秒检索）
- [ ] 耦合学习加状态记录（每轮样本持久化，便于离线分析）
- [ ] 基准扩展：加入真人标注语料 + 跨模型一致性评测
