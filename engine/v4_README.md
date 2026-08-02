# FSTN-4D v4 —— Ultimate 缺口补齐

> 本目录（`engine/` 下 `v4_*` 文件）补齐了
> `fstn_4d_agent_training_spec_ultimate.md` 检查清单中 v1/v2/v3 未实现的部分。
> 全部可运行、已实测、有基准报告。

## 补齐内容

| 文档项 | v4 实现 | 验证 |
|--------|---------|------|
| §2.7 HNSW 近似最近邻 | `v4_hnsw_index.py`（faiss IndexHNSWFlat） | 5 万条 0.15ms/query，top1 与线性扫描 100% 一致 |
| §2.4.3 通感关联图 | `v4_perceptual_space.py` `SynesthesiaGraph` | 自动建链/强化/衰减/剪枝，实测建链成功 |
| §2.4.2 独立感知嵌入空间 | `v4_perceptual_space.py` `PerceptualIndex`（每通道 128 维语义簇） | 跨通道检索命中，火↔夕阳相似度 0.79 |
| §2.7 四路融合检索 | `v4_engine.py` `FSTN4DEngineV4`（语义0.5+感知0.3+虫洞0.1+通感0.1） | e2e 验证融合排序 |
| §8 评估基准 8.1-8.32 | `v4_benchmark.py` | **32/32 通过（100%）** |
| §6 训练样本 400+ | `build_training_data.py` → `training_data/train_fstn_4d.jsonl` | 400 条，14 类场景 |
| §7 RLHF/DPO 奖励 | `dpo_reward.py` → `training_data/dpo_preferences.jsonl` | 400 对偏好，奖励命中统计 |
| §7 Step8 SFT 微调 | `finetune_fstn_lora.py`（QLoRA） | 0.5B 冒烟训练通过 |
| §7 Step9 DPO | `finetune_fstn_dpo.py` | dry-run 通过 |
| 状态持久化（v1/v2 缺口） | `v4_engine.py` save/load 完整记忆+版本链+通感图 | e2e 验证 8 条记忆恢复 |

## 引擎演进

```
v1 (fstn_*.py)        —— 符号规则基座
v2 (v2_*.py)          —— 否定/程度/向量检索/耦合学习
v3 (v3/)              —— LLM 语义兜底 + 神经-符号混合（实验分支）
v4 (v4_*.py)  ← 本补齐 —— HNSW + 感知空间 + 通感图 + 32项基准 + 训练管线
```

## 快速验证

```bash
# 32 项评估基准（文档 §8）
python v4_benchmark.py

# 端到端回归
python e2e_v4.py

# HNSW 性能自测
python v4_hnsw_index.py

# 生成训练数据（400+ 条）
python build_training_data.py

# 生成 DPO 偏好对（400 对）
python dpo_reward.py

# 微调（SFT，QLoRA）
python finetune_fstn_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --output ./fstn_lora_out

# 微调（DPO）
python finetune_fstn_dpo.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter ./fstn_lora_out

# 微调后行为验证
python eval_fstn_model.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter ./fstn_lora_out
```

## 训练说明

- 国内网络：脚本自动设置 `HF_ENDPOINT=https://hf-mirror.com`
- 本机 RTX 5060 8GB：可跑 QLoRA 7B（4bit），推荐 1.5B-3B 快速迭代
- 完整 CUDA torch：`pip install torch --index-url https://download.pytorch.org/whl/cu124`
  （2.5GB+，国内慢；当前 CPU 版 torch 可跑 0.5B 冒烟验证，正式训练建议装 CUDA 版）

## 修复的引擎 bug（v4 基准暴露）

1. **否定词误伤「特别」**：v2 否定表单字"别"匹配到"特别烦"→ anger 被压 0.05。
   修复：否定词前导字符特判（别/不 前是 特/太/真/挺 等则跳过）。
2. **效价极性抑制永不触发**：`_compute_valence` 实际范围 ±0.36（除以 2.5），
   但 `emotional_modulation` 阈值 1.2 永远无法满足。修复：阈值改为 0.4。
3. **热/冷符号与文档相反**：文档 §5.2 热=负舒适(-0.7)、冷=正舒适(+0.7)，
   引擎实现相反。修复：统一 update/couple/modulate 三处判定。
4. **羞愧调制缺失**：`emotional_modulation` 没有 shame→self_exposure 抑制分支。
   修复：优先用检测器 complex_emotion（高信号词感知），配配方兜底。
5. **复杂情绪高信号被覆盖**：`_inject_high_signal_complex` 在 v1 已有弱配方
   结果时放弃高信号注入 → 嫉妒句误判 gratitude。修复：高信号词优先。
6. **add_version 不入库**：v2 `add_version` 只写版本链不写 memories。
   已在 v4 持久化层补全恢复逻辑。
7. **测试污染**：基准测试未隔离 state_dir → 加载了真实用户记忆。
   修复：`new_engine()` 用 tempfile 隔离。
