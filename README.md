# FSTN-4D — Fibonacci Spacetime Network for Agents

**感知-情绪-记忆一体化的 Agent 认知引擎**

FSTN-4D 是 Agent 的**外接长期记忆与情感计算引擎**，核心设计：

```
调制层: 情绪调制层 (Ekman 六维 + 复杂社会情绪)
Layer 0: 潜意识层 (关键节点结晶, 默认路径激活)
Layer 1: 感知层   (七维感知状态机 + 通感关联)
Layer 2: 工作记忆 (会话上下文)
Layer 3: 情景记忆 (斐波那契时间金字塔 + 向量索引)
Layer 4: 语义记忆 (知识图谱 + 语义虫洞网络)
```

## 架构亮点

- **感知-情绪-行为三重推理链**：区分"感知直接驱动"(W_p=0.85) 与"情绪中介驱动"(W_e=0.6)，避免过度解读
- **斐波那契时间金字塔**：心理时间 `T_psych = γ·T_now + (1-γ)·T_psych_old`，复习提升、自然沉淀
- **语义虫洞**：跨域联想、通感关联（sharpness/warmth/heaviness/freshness 质量因子）
- **结晶机制**：高频记忆固化潜意识关键节点；极端情绪(>0.8)禁结晶

## 版本演进

| 版本 | 内容 |
|------|------|
| **v1** (engine/) | 符号化规则引擎：关键词情绪检测 + 斐波那契记忆 + 静态耦合矩阵 |
| **v2** (engine/v2_*.py) | 增强层：否定翻转、程度校准、高信号词注入、向量检索(本地TF-IDF/Ollama bge-m3)、耦合系数在线学习 |
| **v3** (v3/) | 实验分支：neural_detector、vector_memory、hybrid_core |

## v2 基准结果（本机实测）

| 维度 | v1 | v2 | 提升 |
|------|----|----|------|
| 情感检测准确率 | 31% (5/16) | 100% (16/16) | +69pp |
| 记忆检索命中率 | 75% (6/8) | 100% (8/8) | +25pp |
| 耦合预测 MAE | 0.427 | 0.307 | -28.1% |

## 快速开始

```bash
# 依赖
pip install numpy jieba requests

# 拉取 embedding 模型（中文场景务必用 bge-m3）
ollama pull bge-m3     # 1.1GB, 中英多语言
# ⚠️ 不要用 nomic-embed-text：英文为主，中文检索实测 5/10 不如本地 TF-IDF 8/10

# 跑基准
python engine/v2_benchmark.py

# 端到端演示
python engine/v2_engine.py --embedding auto

# 检索提供者对比（TF-IDF vs bge-m3）
python engine/v2_retrieval_compare.py
```

## 替换 v1 引擎（drop-in）

```python
from v2_engine import FSTN4DEngineV2
engine = FSTN4DEngineV2(prefer_embedding="auto")
```

接口与 v1 完全一致：`process_utterance` / `retrieve_memories` / `review_memories` / `crystallize_if_ready` / `generate_reply_guidance` / `save_state` / `load_state` / `get_session_report`。

## 目录结构

```
fstn_enhancement/
├── engine/          # v1 核心 + v2 增强层
│   ├── fstn_core.py / fstn_emotion.py / fstn_memory.py / fstn_perception.py
│   ├── v2_engine.py / v2_emotion_classifier.py / v2_vector_retrieval.py
│   ├── v2_coupling_learner.py / v2_benchmark.py / v2_retrieval_compare.py
│   └── hermes_adapter.py / auto_loader.py / memory_bridge.py
├── v3/              # v3 实验分支（神经检测/向量记忆/混合核心）
├── bench/           # 消融测试
├── demos/           # 端到端演示
├── skills/          # Agent 行为协议
└── v2_README.md     # v2 增强层详细文档
```

## 设计文档

- [终极整合规范](fstn_4d_agent_training_spec_ultimate.md) — 完整系统设计蓝图
- [v2 增强层文档](v2_README.md)
