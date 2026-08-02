# FSTN-4D — 斐波那契时空网络·情感记忆引擎

> **Fibonacci Spacetime Network**：一个带情绪感知、感知-情绪耦合、分层记忆与通感联想的 Agent 外接长期记忆引擎。纯 Python 实现，零训练依赖可跑，文档驱动、基准验证。

中文情感计算 + 认知架构的参考实现。引擎以"情绪状态机 × 感知状态机 × 分层记忆"为骨架，实现了一套可被任意 LLM Agent 外接的**情感记忆中间件**。

---

## 为什么做这个

主流 Agent 的记忆系统大多是"键值存储 + 向量检索"——没有情绪、没有感知、没有联想。FSTN-4D 尝试回答一个问题：

> 如果 Agent 也有"情绪记忆"，它会不会更懂用户？

它把认知科学里的**艾克曼情绪模型、感知-情绪耦合、斐波那契遗忘曲线、潜意识结晶、通感联想**落成可运行的代码，并配套一份完整的《Agent 记忆训练规范》文档（`fstn_4d_agent_training_spec_ultimate.md`，10 步检查清单全部实现）。

## 核心能力

| 层 | 模块 | 说明 |
|---|---|---|
| 🧠 情绪 | `fstn_emotion.py` | 艾克曼六维（joy/sadness/anger/fear/disgust/surprise）+ 复杂情绪（嫉妒/羞愧/内疚/感激）+ 效价/唤醒 + 自然衰减与残留 |
| 👁 感知 | `fstn_perception.py` | 七通道（热/冷/苦/饿/疼/噪音/疲劳）+ 感知指纹 + 通感质量因子（sharpness/warmth/heaviness/freshness） |
| 🔗 耦合 | `fstn_perception.py` + `v2_coupling_learner.py` | 感知→情绪（热→怒）+ 情绪→感知（亢奋抑制痛觉）+ EMA 在线学习耦合系数 |
| 🗂 记忆 | `fstn_memory.py` | 分层记忆（episodic/semantic/procedural）+ 斐波那契遗忘窗口 + 复习强化 + 潜意识结晶 + 版本链冲突解决 + 虫洞联想 |
| ⚡ 检索 | `v4_hnsw_index.py` | HNSW ANN 索引（faiss），万级 0.05ms/query；语义+感知+虫洞+通感四路融合 |
| 🎨 通感 | `v4_perceptual_space.py` | 每通道 128 维感知嵌入 + 语义簇意象 + 跨通道通感图 |
| 🧩 增强 | `v2_emotion_classifier.py` | 否定翻转、程度副词校准、高信号复杂情绪注入、转折窗口衰减 |
| 🎙 对话 | `v4_engine.py` | `FSTN4DEngineV4`：process_utterance 完整管线 + 回复指引 + 会话报告 |

## 快速开始

```bash
# 零依赖核心（只需 Python 3.10+）
cd engine
python -c "
from v4_engine import FSTN4DEngineV4
e = FSTN4DEngineV4(prefer_embedding='local')   # 本地 TF-IDF，零外部依赖
r = e.process_utterance('好热啊，而且今天工作特别烦')
print('情绪:', r['emotion'])
print('感知:', r['perception'])
"
```

**检索后端**：默认本地 TF-IDF（jieba 分词 + 字 bigram，中文召回 8/10）。接入 Ollama embedding 自动提升语义召回：

```bash
ollama pull bge-m3   # 中文最优 embedding（1.1GB）
python -c "
from v4_engine import FSTN4DEngineV4
e = FSTN4DEngineV4(prefer_embedding='auto')   # 自动探测 Ollama bge-m3
"
```

## 基准（32/32 通过）

```bash
cd engine
python v4_benchmark.py    # 文档 §8 的 8.1-8.32 全部 32 项
python e2e_v4.py          # 端到端回归 15/15
```

文档 8.1-8.32 覆盖：基础记忆 / 潜意识结晶 / 感知追踪 / 情绪-耦合-行为 / 复杂情绪 / 虫洞联想 / 通感关联 / 遗忘边界 / 双驱动。

### 引擎修复记录（v2/v4 累计修掉的真实 bug）
- 否定词误伤「特别」（"特**别**烦"被当否定句）
- 效价极性抑制阈值与实际效价范围不匹配（永不触发）
- 热/冷符号约定与规范文档相反
- 复杂情绪高信号词被弱配方结果覆盖
- HNSW rebuild 顺序错误导致全零向量

## 微调（可选，QLoRA + DPO）

引擎本身零训练依赖。若要让本地 LLM **内化**协议行为（非必须），项目附带完整训练管线：

```bash
cd engine
python build_training_data.py    # 400 条 SFT 样本（14 类场景）
python dpo_reward.py             # 400 对 DPO 偏好（奖励表）
# GPU（8GB 即可，实测 RTX 5060）：
python finetune_fstn_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --output ./fstn_lora_out
python finetune_fstn_dpo.py --base Qwen/Qwen2.5-1.5B-Instruct --data training_data/dpo_preferences.jsonl --output ./fstn_dpo_out
python eval_fstn_model.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter ./fstn_lora_out
```

实测效果（18 项协议探针）：基座 14/18 → SFT **16/18** → +DPO 15/18。微调让模型学会"耦合双输入""感知过滤机制"等协议语言。

## MCP 接入（可选）

```bash
python fstn_mcp_server.py   # stdio，暴露 24 个工具（情绪/感知/记忆/通感/对话）
```

## 目录结构

```
├── engine/                  # 引擎本体（v1 原引擎 + v2 增强 + v4 补齐）
│   ├── fstn_core.py         # 引擎核心（v1 接口，全版本兼容）
│   ├── fstn_emotion.py      # 情绪状态机
│   ├── fstn_memory.py       # 记忆引擎（斐波那契+结晶+虫洞）
│   ├── fstn_perception.py   # 感知状态机
│   ├── v2_*.py              # v2 增强层（drop-in 替换）
│   ├── v4_*.py              # v4 补齐（HNSW/感知空间/通感/基准/训练）
│   ├── build_training_data.py / finetune_*.py / dpo_reward.py   # 训练管线
│   └── fstn_mcp_server.py   # MCP 服务器
├── bench/                   # 旧基准（v2 时代）
├── demos/                   # 演示脚本
├── skills/                  # Agent 行为协议（Skill 格式）
├── v3/                      # 神经-符号混合实验分支
├── fstn_4d_agent_training_spec_ultimate.md  # 完整训练规范（10 步清单）
└── FEASIBILITY.md           # 可行性论证
```

## 设计哲学

1. **强设定驱动**：魅力来自世界规则本身的诡异与自洽，不靠堆数值。
2. **人物弧光大于战力弧光**：记忆的重点是"经历与改变"，不是"信息存储量"。
3. **情绪复杂化**：覆盖悬疑的毛骨悚然、智斗的紧张、选择的艰难、失去的隐痛——拒绝单一"爽"感。

## License

MIT

## 致谢

- 架构灵感：《FSTN-4D Agent 记忆训练规范》系列文档
- 中文 NLP：jieba 分词；faiss 提供 HNSW 索引
- 训练基座：Qwen2.5（Qwen Team）
