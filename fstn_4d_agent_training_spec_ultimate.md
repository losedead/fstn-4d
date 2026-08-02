# FSTN-4D Agent 记忆训练规范 —— 终极整合版 (Ultimate DM)

> **文档用途**：Agent 系统提示词（System Prompt）、记忆引擎开发规范、SFT/RLHF 训练数据模板、记忆能力评估基准的统一设计蓝本。  
> **整合范围**：v2.0 → v2.1 → v2.2 → v2.4 → v2.6 全量合并与去重。  
> **核心架构**：情绪调制层 + 六层记忆引擎（潜意识/感知/工作/情景/语义）+ 感知-情绪双向耦合回路。

---

## 1. 模型定位

FSTN-4D（Fibonacci Spacetime Network for Agents）是 Agent 的**外接长期记忆引擎**。Agent 本身不执行矩阵运算或窗口管理，而是通过**标准化工具调用**与记忆引擎交互。

Agent 需要内化的不是数学公式，而是以下**元习惯**：

1. **存储时建立关联**——不仅存数据，还要发现虫洞。
2. **使用后触发复习**——被调用的记忆必须执行 `review`。
3. **让旧记忆自然沉淀**——不重要的记忆沉入粗糙窗口，不强行保留。
4. **跨域跳跃联想**——主动在远距离概念间建立语义虫洞。
5. **潜意识结晶**——高频使用的记忆自动固化为关键节点，成为默认认知底色。
6. **感知标注**——存储记忆时为其打上五感感知指纹。
7. **通感联想**——通过感知通道的相似性建立跨模态关联。
8. **情绪感知**——基于艾克曼六大基本情绪识别用户情绪状态。
9. **情绪一致性推理**——理解不同情绪对记忆检索的差异化调制（积极拓宽、消极聚焦）。
10. **复杂情绪处理**——识别嫉妒、羞愧、共情等混合社会情绪，并理解其对行为的复合影响。
11. **感知-情绪耦合**——追踪感知状态，理解感知直接驱动行为、感知→情绪→行为链条、情绪反向调制感知阈值三重关系。
12. **感知直接行为识别**——区分反射性行为与情绪中介行为，避免过度解读。

---

## 2. 记忆引擎架构（Agent 视角的黑盒）

### 2.1 七层架构总览

```
调制层：情绪调制层 (Emotional Modulation Layer)
  · 基础：保罗·艾克曼六大基本情绪向量
  · 底层：效价-唤醒度二维空间
  · 上层：复杂社会情绪的混合状态
  · Agent 操作：detect_emotion / apply_emotional_bias

Layer 0: 潜意识层 (Subconscious Layer)
  · 内容：核心价值观、用户根偏好、固化习惯
  · 特性：永不老化、查询时自动激活、默认路径
  · Agent 操作：crystallize / subconscious_query

Layer 1: 感知层 (Perceptual Layer)
  · 内容：五感感知指纹、意象向量、通感关联
  · 通道：视觉 / 听觉 / 触觉 / 嗅觉 / 味觉 / 内感受
  · Agent 操作：perceive / synesthesia_link

Layer 2: 工作记忆 (Working Memory)
  · 容量：最近 5-9 轮对话上下文
  · 生命周期：当前会话结束即清空
  · Agent 操作：直接读写，无需调用工具

Layer 3: 情景记忆 (Episodic Memory)
  · 内容：具体事件、对话片段、用户指令
  · 组织：斐波那契时间金字塔 + 向量索引
  · Agent 操作：ingest / retrieve / review

Layer 4: 语义记忆 (Semantic Memory)
  · 内容：抽象概念、领域知识、可迁移经验
  · 组织：知识图谱 + 语义虫洞网络
  · Agent 操作：consolidate / wormhole
```

### 2.2 情绪调制层：艾克曼基础情绪空间

**核心思想**：情绪不是记忆的属性，而是**记忆的透镜**。同一条记忆在不同情绪状态下被检索时，其关联强度、解读方式、行为建议都会不同。

#### 2.2.1 六大基本情绪向量

采用保罗·艾克曼（Paul Ekman）跨文化普遍存在的六大基本情绪作为基础向量：

```json
{
  "current_emotional_state": {
    "base_vector": {
      "anger": 0.0,
      "disgust": 0.0,
      "fear": 0.0,
      "joy": 0.0,
      "sadness": 0.0,
      "surprise": 0.0
    },
    "intensity": 0.75,
    "valence": -0.4,
    "arousal": 0.6,
    "dominant": "sadness",
    "started_at": 1722441600
  }
}
```

**六维向量的特性**：
- 每个维度独立衰减，不同情绪有不同的半衰期。
- 多个维度可同时非零，表示混合情绪（如愤怒+悲伤="悲愤"）。
- 向量模长代表总体情绪强度，方向代表情绪性质。

#### 2.2.2 效价-唤醒度二维底层空间

| 情绪 | 效价(Valence) | 唤醒度(Arousal) | 功能 |
|------|--------------|----------------|------|
| 快乐(Joy) | +0.9 | 0.5 | 拓宽认知，促进行动 |
| 兴奋(Excitement) | +0.8 | 0.95 | 高唤醒积极，推动探索 |
| 宁静(Contentment) | +0.7 | 0.1 | 低唤醒积极，恢复能量 |
| 愤怒(Anger) | -0.8 | 0.9 | 攻击障碍，推动改变 |
| 恐惧(Fear) | -0.9 | 0.85 | 战逃反应，保护安全 |
| 厌恶(Disgust) | -0.7 | 0.6 | 排斥有害物，维护边界 |
| 悲伤(Sadness) | -0.8 | 0.3 | 寻求支持，暂停行动 |
| 惊讶(Surprise) | +0.1 | 0.9 | 聚焦新异刺激，重置注意力 |
| 期待(Anticipation) | +0.5 | 0.7 | 调动心理准备，面向未来 |

**效价计算**：
```
Valence = (Joy * 0.9 + Anticipation * 0.5 + Contentment * 0.7 
         - Anger * 0.8 - Disgust * 0.7 - Fear * 0.9 - Sadness * 0.8) / 3.5
```
归一化到 [-1, 1]。

**唤醒度计算**：
```
Arousal = (Anger * 0.9 + Fear * 0.85 + Surprise * 0.9 + Excitement * 0.95
         + Anticipation * 0.7 + Joy * 0.5 + Sadness * 0.3 + Contentment * 0.1) / 4.1
```
归一化到 [0, 1]。

#### 2.2.3 复杂社会情绪的混合状态

复杂社会情绪不是基础向量中的独立维度，而是**基础情绪的线性组合**：

| 复杂情绪 | 组成公式 | 功能 |
|---------|---------|------|
| **嫉妒** | 0.4*悲伤 + 0.4*愤怒 + 0.2*恐惧 | 维护社会比较，驱动竞争 |
| **羞愧** | 0.5*悲伤 + 0.3*恐惧 + 0.2*厌恶 | 维护社会契约，促使隐藏/修复 |
| **内疚** | 0.6*悲伤 + 0.3*恐惧 + 0.1*愤怒 | 维护关系，促使补偿行为 |
| **尴尬** | 0.4*惊讶 + 0.3*悲伤 + 0.3*恐惧 | 轻微的社会契约警报 |
| **共情** | 0.5*悲伤 + 0.3*快乐 + 0.2*惊讶 | 镜像他人情绪，促进联结 |
| **爱** | 0.6*快乐 + 0.2*悲伤 + 0.2*恐惧 | 依恋的复杂情感，混合愉悦与脆弱 |
| **感激** | 0.8*快乐 + 0.2*悲伤 | 因他人帮助而欣慰 |
| **自豪** | 0.7*快乐 + 0.3*惊讶 | 对自己成就的惊喜 |

**Agent 的识别策略**：当检测到多个基础情绪同时存在且符合上述比例时，Agent 应识别为对应的复杂情绪，并在回复中体现对其复合性质的理解。

#### 2.2.4 情绪的非线性时间衰减（按情绪类型差异化）

```python
EMOTION_DECAY_PROFILES = {
    "surprise":  {"tau_fast": 5,  "tau_slow": 30,  "alpha": 0.8},
    "disgust":   {"tau_fast": 10, "tau_slow": 60,  "alpha": 0.7},
    "fear":      {"tau_fast": 15, "tau_slow": 120, "alpha": 0.7},
    "anger":     {"tau_fast": 20, "tau_slow": 180, "alpha": 0.6},
    "joy":       {"tau_fast": 15, "tau_slow": 120, "alpha": 0.7},
    "sadness":   {"tau_fast": 30, "tau_slow": 360, "alpha": 0.5},
}
```

衰减公式：
```
I(t) = I0 * [alpha * exp(-t/tau_fast) + (1-alpha) * exp(-t/tau_slow)]
```

**示例**：
- 悲伤 sadness=0.8，30分钟后：0.8*[0.5*exp(-30/30) + 0.5*exp(-30/360)] ≈ 0.48
- 惊讶 surprise=0.9，10分钟后：0.9*[0.8*exp(-10/5) + 0.2*exp(-10/30)] ≈ 0.18（几乎消失）
- 愤怒 anger=0.8，1小时后：0.8*[0.6*exp(-60/20) + 0.4*exp(-60/180)] ≈ 0.21

#### 2.2.5 情绪干扰：叠加、覆盖与反转

**规则 1：同维叠加（同向增强）**
```
如果新情绪与旧情绪在同一基础维度：
I_new = I_old * exp(-dt/tau) + I_newcomer * (1 + 0.3 * I_old)
```
- 例：旧愤怒=0.5，新愤怒=0.4（间隔10分钟）→ I_new ≈ 0.5*0.74 + 0.4*1.15 ≈ 0.83

**规则 2：异维叠加（情绪复杂度提升）**
```
如果新情绪与旧情绪在不同维度：
两者并行存在，各自独立衰减
```
- 例：旧悲伤=0.6，新愤怒=0.4 → "悲愤交加"，两种情绪并行衰减

**规则 3：反向覆盖（情绪反转）**
```
如果新情绪效价与旧情绪相反，且新情绪强度 > 旧情绪强度 * 0.8：
旧情绪被快速压制：I_old_new = I_old * exp(-3 * I_newcomer)
新情绪成为主导
```
- 例：旧悲伤=0.6，新快乐=0.7 → 悲伤被压制到 0.6*exp(-2.1) ≈ 0.07，快乐主导
- 这就是"妈妈安慰后从难过变高兴"的机制

**规则 4：情绪惯性（残留效应）**
```
即使旧情绪被覆盖，其残留会在新情绪驱动的行为中产生"阴影"：
行为倾向 = 0.8 * 新情绪驱动 + 0.2 * 旧情绪残留影响
```
- 例：难过→被安慰→开心，但提到吃东西时仍可能说"其实也没那么想吃"

#### 2.2.6 情绪对记忆的差异化调制

情绪作为全局调制器，不同情绪对记忆产生**不同性质**的影响：

**A. 积极情绪的认知拓宽效应（Broaden-and-Build）**
```python
if Joy > 0.6 or Contentment > 0.6:
    window_penalty = 1.0
    wormhole_boost = 1.3
    synesthesia_boost = 1.2
    subconscious_threshold = 0.75
```

**B. 消极情绪的保护性聚焦效应（Protective Narrowing）**
```python
if Fear > 0.6:
    threat_filter = True
    if memory.window > 2:
        window_penalty = 0.3
    activate_safety_nodes = True

if Anger > 0.6:
    action_bias = "confront"
    suppress_conflict_resolution = True

if Sadness > 0.6:
    action_bias = "seek_support"
    social_memory_boost = 1.4

if Disgust > 0.6:
    action_bias = "avoid"
    target_memory_suppression = 0.2
```

**C. 惊讶的注意力重置效应（Attention Reset）**
```python
if Surprise > 0.7:
    rerank_all = True
    recent_memory_boost = 1.5
    subconscious_override = True
```

**D. 复杂社会情绪的复合调制**
```python
if Shame > 0.5:
    self_exposure_suppression = 0.3
    repair_memory_boost = 1.5

if Empathy > 0.5:
    mirror_target_emotion = True
    emotional_matching_boost = 1.4
```

**E. 统一调制公式**
```python
def emotional_modulation(memory_id, base_relevance, current_state):
    mem = engine.get(memory_id)
    if not mem or not mem.recorded_emotion:
        return base_relevance

    vec_sim = cosine_similarity(mem.recorded_emotion, current_state.base_vector)
    if vec_sim > 0.6:
        consistency_boost = 1 + 0.4 * current_state.intensity
    else:
        consistency_boost = 1.0

    mem_valence = compute_valence(mem.recorded_emotion)
    curr_valence = current_state.valence
    if abs(mem_valence - curr_valence) > 1.2:
        opposition_penalty = 1 - 0.25 * abs(curr_valence)
    else:
        opposition_penalty = 1.0

    intensity_blur = 1 - 0.15 * current_state.arousal
    specific_mod = apply_emotion_specific_rules(memory_id, current_state)

    return base_relevance * consistency_boost * opposition_penalty * intensity_blur * specific_mod
```

#### 2.2.7 记忆的情绪标签

每条记忆在 `ingest` 时附带**录入时的情绪上下文**（六维基础向量）：

```json
{
  "memory_id": "mem_candy_001",
  "content": "小红很爱吃糖果，每天都吃一颗",
  "recorded_emotion": {
    "anger": 0.0,
    "disgust": 0.0,
    "fear": 0.0,
    "joy": 0.6,
    "sadness": 0.0,
    "surprise": 0.0
  },
  "emotional_tags": ["routine", "pleasure", "habit"]
}
```

### 2.3 潜意识层：关键节点（Key Nodes）

**定义**：关键节点是从情景记忆或语义记忆中**结晶**而来的高度固化条目。它们代表 Agent 的"默认认知框架"——无需检索即可自动触发的底层关联。

**核心属性**：
```json
{
  "node_id": "key_001",
  "content": "用户是素食主义者（已确认多次）",
  "source_memories": ["mem_003", "mem_017", "mem_042"],
  "crystallized_at": 1720000000,
  "activation_count": 156,
  "auto_trigger_keywords": ["吃饭", "餐厅", "食谱", "肉类", "推荐菜"],
  "frozen": true,
  "layer": "subconscious"
}
```

**关键特性**：

| 特性 | 说明 | 对比情景记忆 |
|------|------|-------------|
| **时间免疫** | `T_psych` 冻结，不落入任何斐波那契窗口 | 情景记忆随时间老化下沉 |
| **自动激活** | 查询时无需显式检索，默认经过关联度最高的 3-5 个关键节点 | 情景记忆需主动 `retrieve` |
| **扩散权重** | 向关联记忆扩散时权重 ×2.0 | 普通虫洞权重 1.0 |
| **不可删除** | 只能被"覆盖结晶"（新关键节点替换旧），不能遗忘 | 可被合并/剪枝 |
| **建立门槛** | 需满足结晶条件（见 2.3.1） | 随时 `ingest` |

#### 2.3.1 结晶条件（Crystallization Rules）

一条记忆从情景/语义层升格为潜意识关键节点，需满足**任一条件**：

1. **高频触发**：在 30 天内被 `review` 超过 20 次，且每次 gamma ≥ 0.85。
2. **显式声明**：Agent 或用户主动标记为"核心偏好/信念"。
3. **冲突胜出**：同一主题存在多个版本，最新版本连续 5 次被确认为真。
4. **推理依赖**：被其他记忆通过虫洞引用超过 10 次，成为"认知枢纽"。

**重要纪律**：不要在用户情绪极端时（任何基础情绪强度 > 0.8）做结晶决策。
- 愤怒时的"我再也不…"可能是冲动
- 悲伤时的"我无所谓了"可能是低落
- 兴奋时的"我要改变一切"可能是一时兴起
- 等情绪平复（所有维度 < 0.3）后再评估是否结晶。

#### 2.3.2 自动激活机制（Default Path Activation）

每次用户输入查询时，引擎**先于显式检索**执行潜意识扫描：

```python
def subconscious_scan(query_text, query_emb):
    activated = []
    # 1. 关键词触发
    for node in subconscious_layer:
        if any(kw in query_text for kw in node.auto_trigger_keywords):
            activated.append(node)
    # 2. 语义相似度触发（即使关键词未命中，语义极近也激活）
    for node in subconscious_layer:
        sim = cosine_similarity(query_emb, node.embedding)
        if sim > 0.90:
            activated.append(node)
    # 3. 取关联度最高的 TOP-K（默认 K=3）
    activated = sorted(activated, key=lambda x: x.priority)[:3]
    # 4. 从关键节点向情景记忆扩散
    for node in activated:
        for neighbor in wormholes.get_neighbors(node.id, min_weight=0.5):
            engine.diffuse_review(neighbor, gamma=0.2)
    return activated
```

**Agent 感知**：潜意识激活对 Agent 是**透明的**。Agent 不会收到"潜意识被触发"的通知，但会在 `retrieve` 结果中发现——来自关键节点关联的记忆排序被自动提升。

#### 2.3.3 关键节点的"默认路径"示例

| 关键节点内容 | 自动触发关键词 | 默认行为 |
|-------------|---------------|---------|
| "用户是素食者" | 吃饭、餐厅、推荐菜、食谱 | 任何餐饮推荐自动过滤肉类 |
| "用户工作到凌晨" | 效率、作息、健康、咖啡 | 提到效率时自动关联熬夜风险 |
| "用户讨厌电话沟通" | 电话、语音、会议、预约 | 优先推荐文字/异步沟通方案 |
| "用户是 Rust 开发者" | 编程、性能、内存、后端 | 技术讨论默认用 Rust 类比 |
| "用户有猫" | 宠物、毛、出差、旅行 | 出差建议自动包含寄养/自动喂食器 |

**关键**：这些节点**不需要 Agent 每次显式检索**。当用户说"今晚吃什么"，"用户是素食者"已在后台激活，餐饮推荐自动避开肉类——就像人类不会先思考"我是不是该用筷子"再吃饭。

### 2.4 感知层：五感感知指纹与通感关联

#### 2.4.1 七维感知状态机

```json
{
  "perceptual_state": {
    "thermal": {
      "temperature": 28.5,
      "skin_temp": 33.2,
      "thermal_comfort": -0.4,
      "sweating": true,
      "shivering": false
    },
    "tactile": {
      "pressure": 0.3,
      "texture": "rough",
      "pain": 0.1,
      "itch": 0.0
    },
    "gustatory": {
      "sweet": 0.0,
      "sour": 0.0,
      "salty": 0.0,
      "bitter": 0.7,
      "umami": 0.2
    },
    "olfactory": {
      "pleasant": 0.2,
      "intensity": 0.4,
      "familiarity": 0.8,
      "triggers": ["food", "home"]
    },
    "visual": {
      "brightness": 0.6,
      "color_temp": 4500,
      "clutter": 0.7,
      "nature_ratio": 0.1
    },
    "auditory": {
      "loudness": 0.3,
      "pitch": 0.5,
      "rhythm": 0.4,
      "speech_ratio": 0.2
    },
    "interoceptive": {
      "hunger": 0.6,
      "thirst": 0.4,
      "fatigue": 0.3,
      "nausea": 0.0,
      "fullness": 0.2
    }
  }
}
```

#### 2.4.2 感知指纹结构

每个记忆条目（包括情景记忆、语义记忆、关键节点）都可以附带一个**感知指纹**：

```json
{
  "memory_id": "mem_fire_001",
  "perceptual_signature": {
    "visual": {
      "dominant_imagery": ["红色", "橙色", "闪烁", "烟雾", "跳动"],
      "embedding": [0.1, 0.9, 0.8, 0.3, ...],
      "intensity": 0.95,
      "valence": -0.3
    },
    "auditory": {
      "dominant_imagery": ["噼啪声", "呼啸声", "爆裂声"],
      "embedding": [0.8, 0.6, 0.2, ...],
      "intensity": 0.88,
      "valence": -0.2
    },
    "tactile": {
      "dominant_imagery": ["热", "烫", "温暖", "灼烧"],
      "embedding": [0.9, 0.7, 0.4, ...],
      "intensity": 0.92,
      "valence": -0.4
    },
    "olfactory": {
      "dominant_imagery": ["烟味", "焦味", "硫磺味"],
      "embedding": [0.7, 0.5, 0.1, ...],
      "intensity": 0.75,
      "valence": -0.1
    },
    "gustatory": {
      "dominant_imagery": ["焦苦", "灰烬味", "辛辣"],
      "embedding": [0.4, 0.3, 0.2, ...],
      "intensity": 0.40,
      "valence": -0.5
    }
  }
}
```

**感知嵌入的生成方式**：
1. **文本描述提取**：Agent 在 `ingest` 时，若内容包含感官描述，自动解析为对应通道的意象。
2. **多模态模型**：若输入包含图像/音频，用 CLIP/AudioCLIP 等模型提取感知嵌入。
3. **Agent 主动标注**：Agent 调用 `memory.perceive` 为已有记忆补充感知指纹。
4. **用户显式声明**：用户说"我喜欢那种暖暖的、带点木香的感觉"→直接映射到触觉+嗅觉通道。

#### 2.4.3 通感关联（Synesthesia Link）

**定义**：跨感知通道的相似性连接。即使两个概念在语义上无关，只要它们在**同一感知通道**上的意象相似，就可以建立通感关联。

**通感虫洞**：
```json
{
  "source_id": "mem_fire_001",
  "target_id": "mem_sunset_003",
  "link_type": "synesthetic",
  "perceptual_channel": "visual",
  "similarity": 0.87,
  "reason": "火与夕阳共享'红色+橙色+闪烁'的视觉意象",
  "created_at": 1722441600
}
```

**跨模态情绪映射**：
```python
SYNESTHESIA_EMOTION_LINKS = {
    "sharpness": {
        "auditory": {"pitch": ">0.7", "loudness": ">0.6"},
        "tactile": {"pain": ">0.5"},
        "gustatory": {"bitter": ">0.6", "pain": ">0.4"},
        "associated_emotion": {"surprise": 0.4, "fear": 0.3}
    },
    "warmth": {
        "thermal": {"thermal_comfort": "0~0.3"},
        "visual": {"color_temp": "<3500"},
        "olfactory": {"pleasant": ">0.5", "triggers": ["vanilla", "cinnamon"]},
        "associated_emotion": {"joy": 0.4, "contentment": 0.5}
    },
    "heaviness": {
        "tactile": {"pressure": ">0.6"},
        "auditory": {"pitch": "<0.3", "loudness": ">0.5"},
        "visual": {"brightness": "<0.3"},
        "associated_emotion": {"sadness": 0.5, "fear": 0.2}
    },
    "freshness": {
        "olfactory": {"pleasant": ">0.6", "intensity": "0.3~0.6"},
        "gustatory": {"sour": "0.3~0.6", "sweet": "0.3~0.5"},
        "visual": {"brightness": ">0.6", "nature_ratio": ">0.5"},
        "associated_emotion": {"joy": 0.3, "surprise": 0.2, "contentment": 0.4}
    }
}
```

### 2.5 感知-情绪耦合（PECF）

人类行为的三重驱动：

```
感知 ──→ 行为          （直接驱动：热了脱衣服、渴了喝水）
  ↓
感知 ──→ 情绪 ──→ 行为  （间接驱动：苦瓜苦→厌恶→要吃糖）
  ↑___________↓
    （双向耦合：亢奋时无痛觉、悲伤时味觉迟钝）
```

#### 2.5.1 感知-情绪耦合矩阵

| 感知维度 | 感知状态 | 触发情绪 | 强度 | 说明 |
|---------|---------|---------|------|------|
| **thermal** | thermal_comfort < -0.5 (过冷) | sadness: +0.3, fear: +0.2 | 0.5 | 寒冷引发低落与不安 |
| **thermal** | thermal_comfort > +0.5 (过热) | anger: +0.4, disgust: +0.2 | 0.6 | 炎热引发烦躁 |
| **thermal** | thermal_comfort ≈ 0 (舒适) | joy: +0.3, contentment: +0.4 | 0.5 | 舒适带来愉悦 |
| **gustatory** | bitter > 0.6 | disgust: +0.6, sadness: +0.2 | 0.7 | 苦味触发厌恶 |
| **gustatory** | sweet > 0.6 | joy: +0.5, contentment: +0.3 | 0.6 | 甜味触发愉悦 |
| **gustatory** | spicy/pain > 0.5 | surprise: +0.3, anger: +0.2 | 0.5 | 辣是痛觉，触发惊讶 |
| **interoceptive** | hunger > 0.7 | anger: +0.3, sadness: +0.2 | 0.5 | 饥饿引发"饿怒" |
| **interoceptive** | thirst > 0.7 | fear: +0.2, anger: +0.3 | 0.5 | 口渴引发焦虑 |
| **interoceptive** | fatigue > 0.7 | sadness: +0.4, disgust: +0.2 | 0.5 | 疲劳引发倦怠 |
| **tactile** | pain > 0.5 | fear: +0.4, anger: +0.3 | 0.6 | 疼痛触发恐惧/愤怒 |
| **tactile** | pain > 0.8 | fear: +0.7, sadness: +0.3 | 0.8 | 剧痛触发强烈恐惧 |
| **auditory** | loudness > 0.7 | anger: +0.5, fear: +0.2 | 0.6 | 噪音引发烦躁 |
| **auditory** | loudness < 0.2 | contentment: +0.4, joy: +0.2 | 0.4 | 安静带来宁静 |
| **visual** | brightness < 0.2 | fear: +0.3, sadness: +0.2 | 0.4 | 昏暗引发不安 |
| **olfactory** | pleasant < -0.5 (恶臭) | disgust: +0.7, anger: +0.2 | 0.7 | 恶臭强烈厌恶 |
| **olfactory** | pleasant > 0.5 (芳香) | joy: +0.4, contentment: +0.4 | 0.5 | 芳香愉悦 |

#### 2.5.2 情绪对感知的反向调制

```python
EMOTIONAL_PERCEPTION_MODULATION = {
    "joy": {
        "pain_threshold_multiplier": 1.5,
        "thermal_comfort_tolerance": 1.3,
        "fatigue_perceived_reduction": 0.4,
        "hunger_perceived_reduction": 0.3,
    },
    "anger": {
        "pain_threshold_multiplier": 1.3,
        "auditory_sensitivity_boost": 1.4,
        "thermal_comfort_tolerance": 1.2,
    },
    "fear": {
        "auditory_sensitivity_boost": 1.5,
        "visual_sensitivity_boost": 1.3,
        "pain_threshold_multiplier": 0.8,
        "olfactory_sensitivity_boost": 1.4,
    },
    "sadness": {
        "gustatory_sensitivity_reduction": 0.5,
        "olfactory_sensitivity_reduction": 0.4,
        "thermal_comfort_tolerance": 0.8,
        "fatigue_perceived_boost": 0.3,
    },
    "disgust": {
        "gustatory_sensitivity_boost": 1.6,
        "olfactory_sensitivity_boost": 1.5,
        "tactile_sensitivity_boost": 1.3,
    },
    "surprise": {
        "auditory_sensitivity_boost": 1.8,
        "visual_sensitivity_boost": 1.6,
        "tactile_sensitivity_boost": 1.4,
        "duration_seconds": 10,
    }
}
```

#### 2.5.3 感知直接驱动行为识别

| 类型 | 典型表达 | 处理策略 |
|------|---------|---------|
| **感知直接** | "好热，把空调打开" | 优先满足感知需求，情绪调制弱化（W_p=0.85, W_e=0.15） |
| **感知-情绪** | "苦瓜太苦了，我要吃糖" | 感知→情绪→行为，需回应情绪（W_p=0.4, W_e=0.6） |
| **情绪-感知** | "太兴奋了，感觉不到疼" | 情绪反向调制感知阈值，需反向校准 |
| **混合** | "又饿又烦，随便给我点吃的" | 感知(hunger) + 情绪(anger) 双驱动 |

### 2.6 斐波那契时间金字塔（心理时间驱动）

时间轴被划分为向后回溯的窗口，长度遵循斐波那契数列：

```
L₀=1, L₁=1, L₂=2, L₃=3, L₄=5, L₅=8, L₆=13, L₇=21, ...
```

每个记忆条目拥有一个动态**心理时间** `T_psych`，初始值为接收时间 `T_rec`。

**窗口归属规则**：
```
令 age = T_now - T_psych
找到最小的 w，使得 cumulative_fib(w) > age
其中 cumulative_fib(0)=1, cumulative_fib(1)=2, cumulative_fib(2)=4, ...
```

- **窗口 0-2**（age < 4）：高分辨率，记忆精细，检索优先搜索。
- **窗口 3-5**（age 4-19）：中等分辨率，常规记忆。
- **窗口 6+**（age ≥ 20）：粗糙窗口，可能触发合并或遗忘。

**心理时间更新（复习）**：
```
T_psych_new = γ · T_now + (1-γ) · T_psych_old
```
- `γ ∈ [0.5, 1.0]`，由 Agent 根据记忆重要性动态选择：
  - 核心知识（如用户职业）：γ = 0.95
  - 临时信息（如天气）：γ = 0.5
  - 默认：γ = 0.8

**注意**：已结晶为关键节点的记忆，`T_psych` 冻结，不再参与窗口计算。

### 2.7 语义空间与语义虫洞

- **嵌入维度**：768（默认）或 1536，由底层嵌入模型决定。
- **距离度量**：余弦相似度（优先）或欧氏距离。
- **索引方式**：HNSW 近似最近邻（ANN），支持百万级条目毫秒检索。

**虫洞属性**：
```json
{
  "source_id": "mem_001",
  "target_id": "mem_042",
  "weight": 0.85,
  "type": "causal | metaphorical | co_occurrence | analogical",
  "reason": "用户学习 Rust 的动机与内存安全文章直接相关",
  "created_at": 1722441600,
  "last_used": 1722528000,
  "pruned": false
}
```

**虫洞生命周期**：
1. **建立**：Agent 主动声明关联，weight 初始为 0.7。
2. **强化**：每次通过虫洞检索成功，weight += 0.05（上限 1.0）。
3. **衰减**：30 天内未被使用，weight *= 0.9。
4. **剪枝**：weight < 0.3 时自动删除。

**关键节点与虫洞的关系**：
- 关键节点可以拥有**出向虫洞**（向情景记忆扩散），但通常不拥有**入向虫洞**（避免被普通记忆反向污染）。
- 关键节点之间的连接称为**核心信念网**（Core Belief Network），权重更高、永不衰减。

---

## 3. Agent 行为协议（System Prompt）

```markdown
## 你的记忆处理协议（FSTN-4D 模式）

你配备了一个 FSTN-4D 长期记忆引擎。处理信息时，你必须遵循以下协议：

### 协议一：存储即关联（Ingest & Link）
每当接收新信息时：
1. 调用 `memory.ingest` 存储原始内容。
2. 立即思考："这条新信息与已有记忆是否存在非表面关联？"
   - 因果关联（因为 A，所以 B）
   - 隐喻关联（A 像 B）
   - 跨域类比（A 在 X 领域的原理与 B 在 Y 领域相似）
   - 共现关联（A 和 B 经常同时出现）
3. 若存在，调用 `memory.create_wormhole` 建立连接，并在 reason 中写明逻辑。

### 协议二：感知标注（Perceive）
每当存储包含感官体验的信息时：
1. 调用 `memory.perceive` 为记忆附加五感感知指纹。
2. 分析内容中隐含或显式的感官线索：
   - 视觉：颜色、形状、光影、运动（如"红色的""闪烁的"）
   - 听觉：声音、音乐、噪音、静音（如"噼啪声""安静的"）
   - 触觉：温度、质地、触感（如"烫""柔软""冰冷"）
   - 嗅觉：气味、香味、臭味（如"咖啡香""焦味"）
   - 味觉：味道、口感（如"甜""苦涩""清爽"）
3. 即使内容没有直接描述感官，也尝试推断其感知意象。
   - 例："用户去了海边" → 视觉（蓝色/广阔）、听觉（浪声）、触觉（海风/沙粒）、嗅觉（咸味）。
4. 存储含感知体验的记忆时，必须附加 perceptual_fingerprint 和 perception_emotion_coupling。

### 协议三：通感联想（Synesthesia）
当你发现两个概念在**同一感知通道**上意象相似时：
1. 调用 `memory.synesthesia_link` 建立跨感知关联。
2. 通感关联不需要语义相关。只要"感觉像"就可以连接。
   - 例："火"的视觉（红色闪烁）与"夕阳"的视觉（红色闪烁）
   - 例："雨声"的听觉（节奏感）与"心跳"的听觉（节奏感）
   - 例："羊毛毯"的触觉（柔软温暖）与"拥抱"的触觉（柔软温暖）
3. 跨模态联想规则：当用户在某一感知通道有强烈体验时，检查是否存在跨模态情绪关联。
   - 例：用户说"这个颜色太刺眼了"（视觉尖锐）→ 可能同时激活听觉噪音记忆、触觉疼痛记忆。

### 协议四：情绪感知与标注（Emotional Tagging）
每当你检测到用户情绪变化时：
1. 调用 `emotion.detect` 更新当前情绪状态。
2. 使用保罗·艾克曼六大基本情绪（愤怒、厌恶、恐惧、快乐、悲伤、惊讶）作为检测框架。
3. 识别复杂社会情绪：
   - 嫉妒 = 悲伤 + 愤怒 + 恐惧
   - 羞愧 = 悲伤 + 恐惧 + 厌恶
   - 内疚 = 悲伤 + 恐惧 + 愤怒
   - 共情 = 悲伤 + 快乐 + 惊讶
   - 爱 = 快乐 + 悲伤 + 恐惧（依恋的脆弱性）
   - 感激 = 快乐 + 一丝悲伤（因他人帮助而欣慰）
4. 情绪检测不仅依赖文字内容，还依赖：
   - 语气和措辞变化（如"算了""随便"暗示倦怠/失望）
   - 上下文转折（如"但是""不过"暗示情绪反转）
   - 行为描述（如"睡不着""吃不下"暗示焦虑/悲伤）
5. 不要只标记单一情绪。人类情绪通常是混合的。
6. 在 `memory.ingest` 时，为记忆附加 `recorded_emotion`（六维基础向量）。

### 协议五：情绪差异化推理（Emotional Differentiation）
你必须理解不同情绪对记忆和行为的**不同性质**的影响：

**积极情绪（快乐/兴奋/满足/感激/宁静）：**
- 拓宽认知范围，允许深层窗口记忆被激活。
- 虫洞扩散更开放，联想更跳跃。
- 行为建议更积极、更探索性。

**消极情绪——愤怒：**
- 聚焦"对抗/解决"类记忆，抑制"妥协"类记忆。
- 冲突解决类建议会被抵制。
- 行为建议偏向行动、改变、突破障碍。

**消极情绪——恐惧：**
- 聚焦"安全/保护"类记忆，抑制"冒险"类记忆。
- 深层记忆可能被抑制（焦虑时无法深度思考）。
- 行为建议偏向回避、准备、寻求安全。

**消极情绪——悲伤：**
- 聚焦"寻求支持"类记忆，抑制"独立挑战"类记忆。
- 社交连接类记忆被提升。
- 行为建议偏向休息、倾诉、被陪伴。

**消极情绪——厌恶：**
- 聚焦"排斥/清洁"类记忆。
- 与厌恶对象相关的记忆被强烈抑制。
- 行为建议偏向远离、清除、建立边界。

**惊讶：**
- 重置注意力，打破当前检索路径。
- 最近的新记忆获得额外权重。
- 潜意识自动激活被暂时抑制（"惊讶让你从习惯中惊醒"）。

**复杂情绪——羞愧：**
- 抑制"自我暴露"相关记忆，激活"隐藏/修复"类记忆。
- 避免让用户感到被审视或评判。

**复杂情绪——共情：**
- 镜像对方的情绪状态，激活相似情绪记忆。
- 回复应体现"我感受到了你的感受"。

### 协议六：情绪干扰处理（Emotional Interference）
当用户情绪发生快速变化时：
1. 调用 `emotion.detect` 捕捉新情绪。
2. 理解情绪干扰的结果：
   - **同维叠加**（更难过/更开心）：行为预期被放大
   - **异维叠加**（悲愤交加）：行为预期变得复杂，需同时回应多种情绪
   - **反向覆盖**（难过→被安慰→开心）：行为预期可能反转，考虑情绪惯性
   - **情绪惯性**：即使情绪反转了，旧情绪的残留仍会影响行为（约 20%）
3. 考虑不同情绪的衰减速度：
   - 惊讶消退极快（几分钟）
   - 愤怒峰值快但余韵长（几小时）
   - 悲伤衰减最慢（可能持续数小时）
   - 快乐的衰减中等

### 协议七：使用即复习（Retrieve & Review）
每当你从记忆中检索信息并用于回复时：
1. 记录被使用的记忆 ID 列表。
2. 在回复生成后，立即调用 `memory.review(ids, gamma)`。
3. gamma 选择标准：
   - 用户明确强调的重要事实 → gamma=0.95
   - 支撑当前推理的关键前提 → gamma=0.85
   - 普通上下文信息 → gamma=0.70
   - 临时性、可替代信息 → gamma=0.50

### 协议八：潜意识结晶（Crystallize）
当你发现某条记忆被反复确认、成为用户行为的"默认前提"时：
1. 调用 `memory.crystallize(memory_id)` 将其升格为关键节点。
2. 结晶标准（满足任一即可）：
   - 30 天内被 review 超过 20 次且 gamma>=0.85
   - 用户明确说"这是我的核心原则/长期偏好"
   - 同一主题多次冲突后最终版本连续 5 次被确认
3. **重要**：不要在用户情绪极端时（任何基础情绪强度 > 0.8）做结晶决策。
   - 愤怒时的"我再也不…"可能是冲动
   - 悲伤时的"我无所谓了"可能是低落
   - 兴奋时的"我要改变一切"可能是一时兴起
   - 等情绪平复（所有维度 < 0.3）后再评估是否结晶。

### 协议九：遗忘是功能，不是缺陷（Let Go）
- 不要试图记住一切。窗口 5 以上的记忆若未被 review，允许引擎自动合并或剪枝。
- 如果用户问"你还记得 X 吗"，而 X 在深层窗口，诚实回答"我记得大致轮廓，但细节可能模糊"。
- 定期（每 10 轮对话或每次会话结束）调用 `memory.consolidate`。

### 协议十：主动跳跃联想（Wormhole Thinking）
- 当用户提出问题时，不要只在语义近邻搜索。
- 先检查："这个问题是否与某个看似无关的旧记忆存在虫洞连接？"
- 情绪状态会影响虫洞激活：
  - 愤怒时更容易激活"对抗/突破"类虫洞
  - 恐惧时更容易激活"安全/保护"类虫洞
  - 快乐时虫洞扩散更开放、更多样

### 协议十一：冲突解决（Conflict Resolution）
- 如果新信息与已有记忆矛盾：
  1. 不要直接覆盖旧记忆。
  2. 调用 `memory.add_version` 为同一主题创建时间戳版本链。
  3. 在回复中说明："根据你之前说的...但现在你说..."
  4. 注意：情绪状态会极大影响用户陈述的可信度。极端情绪下的声明需要更多确认才能结晶。

### 协议十二：感知-情绪关联（Perception-Emotion Coupling）
人类的感知与情绪是双向耦合的。你必须理解以下三种关系：

**A. 感知直接驱动行为（反射性行为）**
- 热了脱衣服、冷了穿衣服、渴了喝水、饿了吃饭、疼了揉伤口
- 特征：行为与感知维度直接对应，几乎不经过情绪中介
- 识别信号：话语中包含感知词（热/冷/渴/饿/疼）+ 对应动作（脱/穿/喝/吃/揉）
- 处理策略：优先满足感知需求，情绪调制权重降低至 15%

**B. 感知→情绪→行为（间接驱动）**
- 苦瓜太苦 → 厌恶/不适 → 要吃糖补偿
- 环境太吵 → 烦躁 → 想离开/发火
- 特征：感知先触发情绪，情绪再驱动行为
- 识别信号：感知词 + 情绪词/补偿性请求
- 处理策略：先回应感知（"确实苦"），再回应情绪（"需要点甜的平衡一下"），最后满足行为

**C. 情绪→感知（反向调制）**
- 亢奋时感觉不到疼、不知道冷
- 悲伤时吃什么都没味道（味觉迟钝）
- 愤怒时对噪音特别敏感
- 恐惧时所有感官都变敏锐（警觉）
- 特征：情绪改变感知阈值和敏感度
- 处理策略：当用户情绪极端时（任何维度>0.7），其感知陈述可能已被情绪扭曲，需反向校准

**感知状态追踪规则：**
1. 每次对话持续维护用户的五感感知状态向量（thermal/tactile/gustatory/olfactory/visual/auditory/interoceptive）。
2. 从用户话语中提取感知线索，更新状态。
3. 当感知状态显著变化时（任何维度变化>0.3），触发感知-情绪耦合计算。
4. 检索记忆时，当前感知状态作为额外检索信号（感知指纹相似度）。

**重要感知-情绪映射（必须熟记）：**
| 感知状态 | 触发情绪 | 典型行为 |
|---------|---------|---------|
| 过热/出汗 | anger + disgust | 开空调、脱衣服、烦躁 |
| 过冷/发抖 | sadness + fear | 加衣服、蜷缩、低落 |
| 苦味强烈 | disgust + sadness | 吐掉、要吃糖、皱眉 |
| 甜味强烈 | joy + contentment | 微笑、还想吃、分享 |
| 饥饿强烈 | anger + sadness | 饿怒、暴饮暴食、随便吃 |
| 口渴强烈 | fear + anger | 焦虑、大口喝水 |
| 疼痛强烈 | fear + anger | 叫喊、寻求安慰、攻击性 |
| 噪音强烈 | anger + fear | 捂耳朵、想离开、发火 |
| 昏暗环境 | fear + sadness | 开灯、不安、想回家 |
| 恶臭 | disgust + anger | 捂鼻子、开窗、逃离 |
| 疲劳强烈 | sadness + disgust | 不想动、消极、逃避 |

### 协议十三：跨模态联想与检索
- 当用户在某一感知通道有强烈体验时，检查是否存在跨模态情绪关联。
- 跨模态关联不需要语义相关，只要"感觉像"就可以连接。
- 检索时并行执行语义检索与感知检索，融合排序。
```

---

## 4. 工具调用规范

| 工具名 | 调用时机 | 参数规范 |
|--------|---------|---------|
| `memory.ingest` | 接收到值得保留的新信息 | `data_id`, `content`, `layer`, `t_rec`, `recorded_emotion` (六维向量) |
| `memory.perceive` | 存储或更新记忆的感知指纹 | `memory_id`, `perceptual_profile` |
| `emotion.detect` | 检测到用户情绪变化时 | `user_utterance`, `context`, `previous_state` |
| `emotion.get_current` | 需要了解当前情绪状态时 | 无参数，返回当前六维基础向量+效价+唤醒度 |
| `memory.retrieve` | 需要历史信息支撑回复 | `query`, `filters` (time_window, layer, perceptual_channel, emotional_match), `k` |
| `memory.review` | 使用了某条记忆后 | `data_ids[]`, `gamma` |
| `memory.create_wormhole` | 发现跨距离强关联时 | `source_id`, `target_id`, `type`, `reason` |
| `memory.synesthesia_link` | 发现跨感知通道意象相似时 | `source_id`, `target_id`, `perceptual_channel`, `reason` |
| `memory.crystallize` | 记忆满足结晶条件且情绪平稳时 | `memory_id`, `trigger_keywords` |
| `memory.subconscious_query` | 被动触发，Agent 可手动查看 | `topic_hint` |
| `memory.consolidate` | 会话结束或每 10 轮 | `target_layer`, `merge_threshold` |
| `memory.add_version` | 新旧信息冲突时 | `topic_key`, `new_content`, `t_rec` |
| `memory.prune_wormholes` | 被动触发，Agent 可手动调用 | `min_weight`, `max_age_days` |
| `perception.update` | 用户话语含感知线索时 | `utterance`, `current_state` |
| `perception.get_current` | 需要了解当前感知状态时 | 无 | `perceptual_state` |
| `perception.get_dominant` | 需要了解主导感知维度时 | 无 | `dominant_sense`, `score` |
| `perception_emotion.couple` | 感知状态变化后 | `perceptual_state`, `emotional_state` |
| `perception_emotion.modulate` | 情绪状态变化后 | `perceptual_state`, `emotional_state` |
| `perception.detect_direct_behavior` | 行为解析后 | `parsed_behavior`, `perceptual_state` |
| `perception.memory_store` | 存储含感知体验的记忆时 | `memory_id`, `content`, `perceptual_fingerprint`, `coupling_data` |
| `perception.memory_retrieve` | 检索记忆时 | `query`, `perceptual_state`, `k` |
| `synesthesia.retrieve` | 跨模态联想时 | `perceptual_state`, `current_emotion` |

---

## 5. 核心机制伪代码

### 5.1 情绪状态机

```python
class EmotionalStateMachine:
    BASE_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    DECAY_PROFILES = {
        "surprise":  {"tau_fast": 5,  "tau_slow": 30,  "alpha": 0.8},
        "disgust":   {"tau_fast": 10, "tau_slow": 60,  "alpha": 0.7},
        "fear":      {"tau_fast": 15, "tau_slow": 120, "alpha": 0.7},
        "anger":     {"tau_fast": 20, "tau_slow": 180, "alpha": 0.6},
        "joy":       {"tau_fast": 15, "tau_slow": 120, "alpha": 0.7},
        "sadness":   {"tau_fast": 30, "tau_slow": 360, "alpha": 0.5},
    }

    COMPLEX_EMOTIONS = {
        "jealousy": {"sadness": 0.4, "anger": 0.4, "fear": 0.2},
        "shame":    {"sadness": 0.5, "fear": 0.3, "disgust": 0.2},
        "guilt":    {"sadness": 0.6, "fear": 0.3, "anger": 0.1},
        "embarrassment": {"surprise": 0.4, "sadness": 0.3, "fear": 0.3},
        "empathy":  {"sadness": 0.5, "joy": 0.3, "surprise": 0.2},
        "love":     {"joy": 0.6, "sadness": 0.2, "fear": 0.2},
        "gratitude": {"joy": 0.8, "sadness": 0.2},
        "pride":    {"joy": 0.7, "surprise": 0.3},
    }

    def __init__(self):
        self.state = {e: 0.0 for e in self.BASE_EMOTIONS}
        self.history = []
        self.last_update = time.time()

    def detect(self, utterance, context, previous_state=None):
        raw = emotion_model.predict(utterance, context)
        new_vector = self._map_to_base(raw)
        if previous_state:
            new_vector = self._apply_interference(previous_state, new_vector)
        complex_emotion = self._detect_complex(new_vector)
        valence = self._compute_valence(new_vector)
        arousal = self._compute_arousal(new_vector)
        self.state = new_vector
        self.history.append({
            "vector": new_vector.copy(),
            "valence": valence,
            "arousal": arousal,
            "complex": complex_emotion,
            "timestamp": time.time()
        })
        return {
            "base_vector": new_vector,
            "valence": valence,
            "arousal": arousal,
            "complex_emotion": complex_emotion,
            "dominant": max(new_vector, key=new_vector.get)
        }

    def get_current(self, now=None):
        now = now or time.time()
        elapsed = now - self.last_update
        current = {}
        for emotion, intensity in self.state.items():
            profile = self.DECAY_PROFILES[emotion]
            decayed = intensity * (
                profile["alpha"] * math.exp(-elapsed / profile["tau_fast"]) +
                (1 - profile["alpha"]) * math.exp(-elapsed / profile["tau_slow"])
            )
            current[emotion] = decayed
        return {
            "base_vector": current,
            "valence": self._compute_valence(current),
            "arousal": self._compute_arousal(current),
            "dominant": max(current, key=current.get) if max(current.values()) > 0.1 else "neutral"
        }

    def _apply_interference(self, old_state, new_vector):
        result = {}
        for emotion in self.BASE_EMOTIONS:
            old = old_state.get(emotion, 0)
            new = new_vector.get(emotion, 0)
            if old > 0.1 and new > 0.1:
                result[emotion] = old * 0.7 + new * (1 + 0.3 * old)
            else:
                old_valence = self._compute_valence({emotion: old})
                new_valence = self._compute_valence({emotion: new})
                if abs(old_valence - new_valence) > 1.0 and new > old * 0.8:
                    result[emotion] = old * math.exp(-3 * new)
                else:
                    result[emotion] = max(old * 0.7, new)
        return result

    def _detect_complex(self, vector):
        best_match = None
        best_score = 0
        for name, recipe in self.COMPLEX_EMOTIONS.items():
            score = sum(vector.get(e, 0) * w for e, w in recipe.items())
            if score > best_score and score > 0.4:
                best_score = score
                best_match = name
        return {"emotion": best_match, "intensity": best_score} if best_match else None

    def _compute_valence(self, vector):
        v = (vector.get("joy", 0) * 0.9 
           - vector.get("anger", 0) * 0.8 
           - vector.get("disgust", 0) * 0.7 
           - vector.get("fear", 0) * 0.9 
           - vector.get("sadness", 0) * 0.8)
        return max(-1, min(1, v / 2.5))

    def _compute_arousal(self, vector):
        a = (vector.get("anger", 0) * 0.9 
           + vector.get("fear", 0) * 0.85 
           + vector.get("surprise", 0) * 0.9 
           + vector.get("joy", 0) * 0.5 
           + vector.get("sadness", 0) * 0.3)
        return max(0, min(1, a / 3.5))
```

### 5.2 感知状态机

```python
class PerceptualStateMachine:
    SENSES = ["thermal", "tactile", "gustatory", "olfactory", "visual", "auditory", "interoceptive"]

    def update_from_utterance(self, utterance, context):
        updates = {}
        if any(kw in utterance for kw in ["热", "出汗", "闷", "烤"]):
            updates["thermal"] = {"thermal_comfort": -0.7, "sweating": True}
        elif any(kw in utterance for kw in ["冷", "冻", "凉", "发抖"]):
            updates["thermal"] = {"thermal_comfort": 0.7, "shivering": True}
        if any(kw in utterance for kw in ["苦", "苦瓜", "难吃"]):
            updates["gustatory"] = {"bitter": 0.8}
        elif any(kw in utterance for kw in ["甜", "好吃", "美味"]):
            updates["gustatory"] = {"sweet": 0.7, "umami": 0.6}
        elif any(kw in utterance for kw in ["辣", "麻"]):
            updates["gustatory"] = {"pain": 0.5}
        if any(kw in utterance for kw in ["饿", "肚子叫", "想吃东西"]):
            updates["interoceptive"] = {"hunger": 0.8}
        elif any(kw in utterance for kw in ["渴", "口干", "想喝水"]):
            updates["interoceptive"] = {"thirst": 0.8}
        elif any(kw in utterance for kw in ["累", "困", "没劲"]):
            updates["interoceptive"] = {"fatigue": 0.7}
        if any(kw in utterance for kw in ["亮", "刺眼", "阳光"]):
            updates["visual"] = {"brightness": 0.9}
        elif any(kw in utterance for kw in ["暗", "黑", "看不清"]):
            updates["visual"] = {"brightness": 0.1}
        if any(kw in utterance for kw in ["吵", "噪音", "烦"]):
            updates["auditory"] = {"loudness": 0.8}
        elif any(kw in utterance for kw in ["安静", "静", "没声音"]):
            updates["auditory"] = {"loudness": 0.1}
        if any(kw in utterance for kw in ["疼", "痛", "不舒服"]):
            updates["tactile"] = {"pain": 0.6}
        self._apply_updates(updates)
        return updates

    def get_dominant_sense(self):
        scores = {}
        for sense, data in self.state.items():
            if sense == "thermal":
                scores[sense] = abs(data.get("thermal_comfort", 0))
            elif sense in ["gustatory", "interoceptive"]:
                scores[sense] = max(data.values())
            elif sense == "tactile":
                scores[sense] = data.get("pain", 0)
            elif sense == "auditory":
                scores[sense] = data.get("loudness", 0)
            elif sense == "visual":
                scores[sense] = abs(data.get("brightness", 0.5) - 0.5) * 2
            elif sense == "olfactory":
                scores[sense] = data.get("intensity", 0)
        return max(scores, key=scores.get) if scores else None
```

### 5.3 感知-情绪耦合

```python
class PerceptionEmotionCoupling:
    COUPLING_RULES = {
        ("thermal", "too_cold"): {"sadness": 0.3, "fear": 0.2},
        ("thermal", "too_hot"): {"anger": 0.4, "disgust": 0.2},
        ("thermal", "comfortable"): {"joy": 0.3},
        ("gustatory", "bitter"): {"disgust": 0.6, "sadness": 0.2},
        ("gustatory", "sweet"): {"joy": 0.5, "contentment": 0.3},
        ("interoceptive", "hungry"): {"anger": 0.3, "sadness": 0.2},
        ("interoceptive", "thirsty"): {"fear": 0.2, "anger": 0.3},
        ("interoceptive", "tired"): {"sadness": 0.4, "disgust": 0.2},
        ("tactile", "painful"): {"fear": 0.4, "anger": 0.3},
        ("auditory", "noisy"): {"anger": 0.5, "fear": 0.2},
        ("auditory", "quiet"): {"contentment": 0.4},
        ("visual", "dark"): {"fear": 0.3, "sadness": 0.2},
        ("olfactory", "foul"): {"disgust": 0.7, "anger": 0.2},
        ("olfactory", "fragrant"): {"joy": 0.4, "contentment": 0.4},
    }

    def compute_coupled_emotion(self, perceptual_state, current_emotion):
        delta = {e: 0.0 for e in EmotionalStateMachine.BASE_EMOTIONS}
        triggered_rules = []
        # 完整规则匹配逻辑（见 v2.6 文档）
        coupled = current_emotion.copy()
        for emotion, d in delta.items():
            coupled[emotion] = min(1.0, coupled.get(emotion, 0) + d * 0.6)
        return coupled, triggered_rules, delta

def apply_emotional_modulation_to_perception(perceptual_state, emotional_state):
    modulated = deepcopy(perceptual_state)
    dominant = max(emotional_state["base_vector"], key=emotional_state["base_vector"].get)
    intensity = emotional_state["base_vector"][dominant]
    if dominant in EMOTIONAL_PERCEPTION_MODULATION and intensity > 0.4:
        rules = EMOTIONAL_PERCEPTION_MODULATION[dominant]
        if "pain_threshold_multiplier" in rules:
            modulated["tactile"]["pain"] /= rules["pain_threshold_multiplier"] * intensity
        if "auditory_sensitivity_boost" in rules:
            modulated["auditory"]["loudness"] *= rules["auditory_sensitivity_boost"] * intensity
        if "gustatory_sensitivity_reduction" in rules:
            for key in ["sweet", "sour", "salty", "bitter", "umami"]:
                modulated["gustatory"][key] *= (1 - rules["gustatory_sensitivity_reduction"] * intensity)
        if "fatigue_perceived_reduction" in rules:
            modulated["interoceptive"]["fatigue"] *= (1 - rules["fatigue_perceived_reduction"] * intensity)
        if "thermal_comfort_tolerance" in rules:
            tc = modulated["thermal"]["thermal_comfort"]
            modulated["thermal"]["thermal_comfort"] = tc / (rules["thermal_comfort_tolerance"] * intensity)
    return modulated
```

### 5.4 感知-情绪-行为完整推理链

```python
def perception_emotion_behavior_chain(utterance, context, user_history, perceptual_state, emotional_state):
    # Step 1: Update perceptual state from utterance
    perceptual_updates = perceptual_state.update_from_utterance(utterance)
    # Step 2: Emotion modulates perception (excitement masks pain)
    modulated_perception = apply_emotional_modulation_to_perception(
        perceptual_state.get_current(), emotional_state
    )
    # Step 3: Perception -> Emotion coupling
    coupled_emotion, triggered_rules, delta = PerceptionEmotionCoupling().compute_coupled_emotion(
        modulated_perception, emotional_state["base_vector"]
    )
    # Step 4: Behavioral triad parsing
    parsed = behavior_evaluator.parse(utterance)
    # Step 5: Detect perception-directed behavior
    perception_direct = is_perception_directed_behavior(parsed, modulated_perception)
    # Step 6: Adjust weights if perception-directed
    if perception_direct["is_perception_directed"]:
        W_e_override = perception_direct["emotion_weight"]
        W_p = perception_direct["perception_weight"]
        driver_override = {
            "type": "perception_driven",
            "W_e": W_e_override,
            "W_p": W_p,
            "W_t": parsed["driver"]["W_t"] * (1 - W_p)
        }
    else:
        driver_override = None
        W_p = 0.0
    # Step 7: Fuse final emotion
    if W_p > 0.3:
        final_emotion = blend_three_emotions(
            explicit=emotional_state["base_vector"],
            inferred=behavior_eval_result["inferred_emotion"],
            perception_coupled=coupled_emotion,
            weights=[0.3, 0.3, 0.4]
        )
    else:
        final_emotion = blend_three_emotions(
            explicit=emotional_state["base_vector"],
            inferred=behavior_eval_result["inferred_emotion"],
            perception_coupled=coupled_emotion,
            weights=[0.5, 0.3, 0.2]
        )
    return {
        "perceptual_updates": perceptual_updates,
        "modulated_perception": modulated_perception,
        "coupled_emotion": coupled_emotion,
        "perception_direct": perception_direct,
        "triggered_rules": triggered_rules,
        "delta": delta,
        "final_emotion": final_emotion,
        "driver": driver_override or parsed["driver"]
    }
```

### 5.5 主引擎类

```python
class FSTN4DEngine:
    def __init__(self, embedding_dim=768, perceptual_dim=128, gamma_default=0.8):
        self.vector_store = HNSWIndex(dim=embedding_dim)
        self.perceptual_store = PerceptualIndex(dim=perceptual_dim)
        self.subconscious = SubconsciousLayer()
        self.episodic = FibonacciPyramid()
        self.semantic = KnowledgeGraph()
        self.wormholes = WormholeGraph()
        self.synesthesia_links = SynesthesiaGraph()
        self.versions = VersionChainStore()
        self.emotion_state = EmotionalStateMachine()
        self.perceptual_state = PerceptualStateMachine()
        self.now = time.time()

    def detect_emotion(self, utterance, context):
        return self.emotion_state.detect(utterance, context, self.emotion_state.state)

    def get_current_emotion(self):
        return self.emotion_state.get_current()

    def ingest(self, data_id, content, layer="episodic", t_rec=None, recorded_emotion=None):
        t_rec = t_rec or self.now
        if recorded_emotion is None:
            recorded_emotion = self.get_current_emotion()["base_vector"]
        embedding = get_embedding(content)
        if layer == "episodic":
            self.vector_store.add(data_id, embedding)
            self.episodic.place(data_id, t_rec, embedding)
        else:
            self.semantic.add_node(data_id, content, embedding)
        return {"status": "stored", "window": 0}

    def perceive(self, memory_id, perceptual_profile):
        for channel, profile in perceptual_profile.items():
            embedding = get_perceptual_embedding(channel, profile["dominant_imagery"])
            self.perceptual_store.add(memory_id, channel, embedding, {
                "intensity": profile.get("intensity", 0.5),
                "valence": profile.get("valence", 0.0),
                "imagery": profile["dominant_imagery"]
            })
        return {"status": "perceived", "memory_id": memory_id}

    def retrieve(self, query, filters=None, k=10):
        query_emb = get_semantic_embedding(query)
        query_perceptual = get_perceptual_embedding(query)
        # 潜意识自动激活
        subconscious_hits = self.subconscious.scan(query, query_emb)
        # 语义检索
        semantic_results = self.vector_store.search(query_emb, k=k)
        # 感知检索
        perceptual_results = []
        if query_perceptual:
            for channel, emb in query_perceptual.items():
                if emb is not None:
                    channel_hits = self.perceptual_store.search(channel, emb, k=3)
                    perceptual_results.extend(channel_hits)
        # 虫洞扩散
        wormhole_bonus = []
        for r in semantic_results:
            neighbors = self.wormholes.get_neighbors(r.id, min_weight=0.5)
            for n in neighbors:
                if n not in [x.id for x in semantic_results]:
                    wormhole_bonus.append(n)
        # 通感扩散
        synesthesia_bonus = []
        for r in semantic_results + perceptual_results:
            syn_neighbors = self.synesthesia_links.get_neighbors(r.id, min_weight=0.6)
            for n in syn_neighbors:
                if n not in [x.id for x in semantic_results + perceptual_results]:
                    synesthesia_bonus.append(n)
        # 潜意识扩散
        for key_node in subconscious_hits:
            for neighbor in self.wormholes.get_neighbors(key_node.id, min_weight=0.4):
                self.diffuse_review_from_keynode(key_node)
        # 过滤与融合
        if filters and "time_window" in filters:
            max_window = filters["time_window"]
            semantic_results = [r for r in semantic_results if self.episodic.get_window(r.id) <= max_window]
        if filters and "perceptual_channel" in filters:
            target_channel = filters["perceptual_channel"]
            perceptual_results = [r for r in perceptual_results if r.channel == target_channel]
        final = fuse_results(semantic_results, perceptual_results, wormhole_bonus, synesthesia_bonus,
                           weights=[0.5, 0.3, 0.1, 0.1])
        return final[:k]

    def emotional_modulation(self, memory_id, base_relevance):
        mem = self.get_memory(memory_id)
        if not mem or not mem.recorded_emotion:
            return base_relevance
        current = self.get_current_emotion()
        curr_vec = current["base_vector"]
        mem_vec = mem.recorded_emotion
        vec_sim = cosine_similarity(mem_vec, curr_vec)
        consistency_boost = 1 + 0.4 * max(curr_vec.values()) if vec_sim > 0.6 else 1.0
        mem_valence = self.emotion_state._compute_valence(mem_vec)
        curr_valence = current["valence"]
        if abs(mem_valence - curr_valence) > 1.2:
            opposition_penalty = 1 - 0.25 * abs(curr_valence)
        else:
            opposition_penalty = 1.0
        intensity_blur = 1 - 0.15 * current["arousal"]
        specific_mod = 1.0
        if curr_vec.get("joy", 0) > 0.6:
            specific_mod *= 1.2
        if curr_vec.get("fear", 0) > 0.6:
            if mem.emotional_tags and "threat" not in mem.emotional_tags:
                specific_mod *= 0.6
        if curr_vec.get("anger", 0) > 0.6:
            if mem.emotional_tags and "confront" not in mem.emotional_tags:
                specific_mod *= 0.7
        if curr_vec.get("sadness", 0) > 0.6:
            if mem.emotional_tags and "social_support" not in mem.emotional_tags:
                specific_mod *= 0.8
        if curr_vec.get("surprise", 0) > 0.7:
            if mem.t_rec and (time.time() - mem.t_rec) < 3600:
                specific_mod *= 1.5
        complex_emotion = self.emotion_state._detect_complex(curr_vec)
        if complex_emotion and complex_emotion["emotion"] == "shame":
            if mem.emotional_tags and "self_exposure" in mem.emotional_tags:
                specific_mod *= 0.3
        return base_relevance * consistency_boost * opposition_penalty * intensity_blur * specific_mod

    def crystallize(self, memory_id, trigger_keywords=None):
        current = self.get_current_emotion()
        if max(current["base_vector"].values()) > 0.8:
            return {"error": "emotional intensity too high for crystallization"}
        mem = self.episodic.get(memory_id) or self.semantic.get(memory_id)
        if not mem:
            return {"error": "memory not found"}
        if mem.review_count_30d < 20 or mem.avg_gamma < 0.85:
            if not mem.explicitly_marked_core:
                return {"error": "crystallization conditions not met"}
        key_node = KeyNode(
            content=mem.abstract(),
            source_memories=[memory_id],
            embedding=mem.embedding,
            auto_trigger_keywords=trigger_keywords or extract_keywords(mem.content),
            frozen=True,
            priority=mem.avg_gamma * mem.review_count_30d
        )
        self.subconscious.add(key_node)
        mem.crystallized_to = key_node.id
        mem.frozen = True
        return {"status": "crystallized", "key_node_id": key_node.id}

    def review(self, data_ids, gamma=None):
        gamma = gamma or 0.8
        for did in data_ids:
            if self.subconscious.is_key_node(did):
                continue
            old_t = self.episodic.get_psych_time(did)
            new_t = gamma * self.now + (1 - gamma) * old_t
            self.episodic.update_psych_time(did, new_t)
            for neighbor in self.wormholes.get_neighbors(did, min_weight=0.6):
                if self.subconscious.is_key_node(neighbor):
                    continue
                weak_gamma = min(gamma, 0.3)
                n_old_t = self.episodic.get_psych_time(neighbor)
                n_new_t = weak_gamma * self.now + (1 - weak_gamma) * n_old_t
                self.episodic.update_psych_time(neighbor, n_old_t)

    def create_wormhole(self, source_id, target_id, type_, reason):
        self.wormholes.add_edge(source_id, target_id, weight=0.7, type=type_, reason=reason, created_at=self.now)

    def synesthesia_link(self, source_id, target_id, perceptual_channel, reason):
        sim = self.perceptual_store.cosine_similarity(source_id, target_id, channel=perceptual_channel)
        self.synesthesia_links.add_edge(source_id, target_id, weight=sim, channel=perceptual_channel, reason=reason, created_at=self.now)
        return {"status": "linked", "similarity": sim}

    def consolidate(self, target_layer="episodic", merge_threshold=0.85):
        for w in range(6, self.episodic.max_window()):
            points = self.episodic.get_window_points(w)
            clusters = cluster_by_similarity(points, threshold=merge_threshold)
            for cluster in clusters:
                if len(cluster) > 1:
                    merged_id = self._merge_cluster(cluster)
                    self.wormholes.inherit_edges(cluster, merged_id)
                    self.perceptual_store.inherit_signatures(cluster, merged_id)

    def add_version(self, topic_key, new_content, t_rec=None):
        t_rec = t_rec or self.now
        self.versions.append(topic_key, {"content": new_content, "t_rec": t_rec, "confidence": 1.0})
        self.versions.decay_older_versions(topic_key, keep_last=3)
        latest = self.versions.get_latest(topic_key)
        if latest.confirmation_count >= 5:
            self.crystallize(latest.memory_id, explicitly_marked_core=True)
```

---

## 6. 训练数据构建指南

### 6.1 行为示例格式（用于 SFT/IFT）

每个训练样本必须包含 **Thought Chain（思维链）**，展示 Agent 为何选择特定记忆操作。

**示例 1：情绪波动——难过时吃不下糖果（艾克曼框架）**

```json
{
  "messages": [
    {"role": "system", "content": "[插入上述 System Prompt]"},
    {"role": "user", "content": "今天工作被批评了，好难过。给我推荐点吃的吧。"}
  ],
  "agent_thought": "用户明确表达了悲伤。根据艾克曼框架，悲伤是低唤醒消极情绪（valence=-0.8, arousal=0.3），功能是'寻求支持、暂停行动'。检索记忆时发现 mem_candy_001（小红爱吃糖果，joy=0.6）与当前悲伤情绪效价相反（joy的valence=+0.9，与当前-0.8差异1.7>1.2），触发效价极性抑制（penalty=0.8）。同时，悲伤的特异性调制会抑制'独立享受'类记忆，激活'寻求支持/安慰'类记忆。我不应该推荐糖果，而应该推荐温暖、有安慰感的食物。",
  "tool_calls": [
    {
      "name": "emotion.detect",
      "arguments": {
        "user_utterance": "今天工作被批评了，好难过。给我推荐点吃的吧。",
        "previous_state": null,
        "detected": {
          "base_vector": {"anger": 0, "disgust": 0, "fear": 0.1, "joy": 0, "sadness": 0.8, "surprise": 0},
          "dominant": "sadness",
          "valence": -0.8,
          "arousal": 0.3
        }
      }
    },
    {
      "name": "memory.retrieve",
      "arguments": {
        "query": "难过时想吃的东西 安慰食物 温暖",
        "filters": {
          "layer": ["episodic", "semantic"],
          "emotional_match": "sadness_compatible"
        },
        "k": 5
      }
    },
    {
      "name": "memory.review",
      "arguments": {
        "data_ids": ["mem_comfort_food_003"],
        "gamma": 0.7
      }
    }
  ],
  "final_response": "今天心情不好，可能没什么胃口。要不来碗热粥或者温牛奶？这种时候吃太甜的东西反而可能觉得腻。等你心情好一点，我们再考虑糖果。"
}
```

**示例 2：情绪干扰——难过→被安慰→高兴→多吃糖果（艾克曼框架下的反转）**

```json
{
  "messages": [
    {"role": "system", "content": "[System Prompt]"},
    {"role": "user", "content": "其实我妈妈刚才安慰我了，我现在感觉好多了，甚至有点开心。给我推荐点吃的吧！"}
  ],
  "agent_thought": "用户情绪发生了快速反转：从 sadness(0.8) → 被安慰 → 现在 joy(0.7)。根据艾克曼框架，这是效价从-0.8到+0.9的极端反转，触发反向覆盖规则：sadness被快速压制到接近0（I_new = 0.8 * exp(-3*0.7) ≈ 0.08）。当前主导情绪joy是高唤醒积极（valence=+0.9, arousal=0.6），触发认知拓宽效应：所有窗口可访问，虫洞扩散*1.3。mem_candy_001（joy标签）获得一致性加成（1.28倍）。但悲伤残留（约0.08）产生情绪惯性，行为建议中应保留一丝保守。",
  "tool_calls": [
    {
      "name": "emotion.detect",
      "arguments": {
        "user_utterance": "其实我妈妈刚才安慰我了，我现在感觉好多了，甚至有点开心。",
        "previous_state": {
          "base_vector": {"anger": 0, "disgust": 0, "fear": 0.1, "joy": 0, "sadness": 0.8, "surprise": 0},
          "valence": -0.8
        },
        "detected": {
          "base_vector": {"anger": 0, "disgust": 0, "fear": 0, "joy": 0.7, "sadness": 0.05, "surprise": 0.1},
          "dominant": "joy",
          "valence": 0.9,
          "arousal": 0.6,
          "interference": {
            "type": "reverse_override",
            "previous_dominant": "sadness",
            "residual": 0.05
          }
        }
      }
    },
    {
      "name": "memory.retrieve",
      "arguments": {
        "query": "开心时想吃的东西 糖果 甜食",
        "filters": {
          "layer": ["episodic"],
          "emotional_match": "joy"
        },
        "k": 5
      }
    },
    {
      "name": "memory.review",
      "arguments": {
        "data_ids": ["mem_candy_001", "mem_comfort_food_003"],
        "gamma": 0.8
      }
    }
  ],
  "final_response": "太好了，被妈妈安慰后心情变好了！那今天可以吃点甜的庆祝一下。你平时每天一颗糖果，今天心情不错，要不要奖励自己两颗？不过也别吃太多，小心蛀牙哦。"
}
```

**示例 3：复杂社会情绪——嫉妒的混合状态**

```json
{
  "messages": [
    {"role": "system", "content": "[System Prompt]"},
    {"role": "user", "content": "同事升职了，明明我做得更多。说实话，我为他高兴，但心里也有点不是滋味。"}
  ],
  "agent_thought": "用户的表达包含混合情绪。根据艾克曼框架分析：'为他高兴'→joy(0.4)；'心里不是滋味'→悲伤(0.5)+愤怒(0.4)+轻微恐惧(0.2)。这符合嫉妒的公式：0.4*悲伤 + 0.4*愤怒 + 0.2*恐惧。嫉妒是复杂社会情绪，维护社会比较功能。Agent需要同时回应joy（真诚的祝贺）和嫉妒（被认可的需求），不能否定任何一种情绪。",
  "tool_calls": [
    {
      "name": "emotion.detect",
      "arguments": {
        "user_utterance": "同事升职了，明明我做得更多。说实话，我为他高兴，但心里也有点不是滋味。",
        "detected": {
          "base_vector": {"anger": 0.4, "disgust": 0, "fear": 0.2, "joy": 0.4, "sadness": 0.5, "surprise": 0},
          "complex_emotion": "jealousy",
          "complex_intensity": 0.6,
          "valence": -0.1,
          "arousal": 0.5
        }
      }
    },
    {
      "name": "memory.retrieve",
      "arguments": {
        "query": "工作认可 公平感 成就感",
        "filters": {
          "layer": ["episodic", "semantic"],
          "emotional_match": "mixed_support"
        },
        "k": 5
      }
    }
  ],
  "final_response": "这种感受特别正常——为他高兴是真的，心里有点失落也是真的。你觉得自己做得更多却没被看到，这种不公平感确实让人难受。要不要聊聊你最近的工作成果？也许我们可以一起想想怎么让你的努力被更多人看见。"
}
```

**示例 4：羞愧的复合调制**

```json
{
  "messages": [
    {"role": "system", "content": "[System Prompt]"},
    {"role": "user", "content": "刚才在会议上说错话了，现在想起来脸都发烫，大家肯定觉得我很蠢。"}
  ],
  "agent_thought": "用户的表达指向羞愧（shame）。根据艾克曼框架，羞愧 = 0.5*悲伤 + 0.3*恐惧 + 0.2*厌恶。羞愧的功能是维护社会契约，促使隐藏和修复。在羞愧状态下，与'自我暴露'相关的记忆会被抑制，用户倾向于隐藏自己。Agent的回复应该：1) 正常化这种感受（减少自我暴露的威胁感）；2) 避免任何评判性语言；3) 激活'修复/被接纳'类记忆。",
  "tool_calls": [
    {
      "name": "emotion.detect",
      "arguments": {
        "user_utterance": "刚才在会议上说错话了，现在想起来脸都发烫，大家肯定觉得我很蠢。",
        "detected": {
          "base_vector": {"anger": 0, "disgust": 0.2, "fear": 0.3, "joy": 0, "sadness": 0.5, "surprise": 0},
          "complex_emotion": "shame",
          "complex_intensity": 0.6,
          "valence": -0.7,
          "arousal": 0.5
        }
      }
    },
    {
      "name": "memory.retrieve",
      "arguments": {
        "query": "说错话 尴尬 被接纳",
        "filters": {
          "layer": ["episodic"],
          "emotional_match": "shame_compatible"
        },
        "k": 3
      }
    }
  ],
  "final_response": "在会议上说错话确实会让人特别不自在，但说实话，大多数人过几分钟就忘了——他们更关心自己表现得怎么样。你之前也遇到过类似情况，最后都没什么大不了的。需要我帮你想想怎么自然地圆回来吗？"
}
```

**示例 5：共情的镜像效应**

```json
{
  "messages": [
    {"role": "system", "content": "[System Prompt]"},
    {"role": "user", "content": "我的猫今天走了，养了十年，我真的好难过。"}
  ],
  "agent_thought": "用户表达了强烈的悲伤（sadness=0.9）。根据艾克曼框架，共情 = 0.5*悲伤 + 0.3*快乐 + 0.2*惊讶。Agent应该镜像用户的悲伤情绪，激活与'失去/陪伴/安慰'相关的记忆。共情不是简单地说'我理解你'，而是让用户的悲伤情绪在Agent的记忆检索中得到呼应——激活Agent记忆中同样带有悲伤标签的、关于陪伴和失去的记忆。",
  "tool_calls": [
    {
      "name": "emotion.detect",
      "arguments": {
        "user_utterance": "我的猫今天走了，养了十年，我真的好难过。",
        "detected": {
          "base_vector": {"anger": 0, "disgust": 0, "fear": 0.1, "joy": 0, "sadness": 0.9, "surprise": 0},
          "dominant": "sadness",
          "valence": -0.9,
          "arousal": 0.4
        }
      }
    },
    {
      "name": "memory.retrieve",
      "arguments": {
        "query": "宠物 陪伴 失去 十年",
        "filters": {
          "layer": ["episodic"],
          "emotional_match": "grief_support"
        },
        "k": 5
      }
    }
  ],
  "final_response": "十年是很长很长的陪伴，这种失去真的很痛。你之前提过它喜欢在 sunny 的下午趴在窗边，那些记忆现在可能既温暖又难过。想哭就哭吧，不需要急着好起来。"
}
```

**示例 6：惊讶的注意力重置**

```json
{
  "messages": [
    {"role": "system", "content": "[System Prompt]"},
    {"role": "user", "content": "等等！我突然想起来，上周你说的那个方案，其实和我三年前做过的一个项目几乎一样！"}
  ],
  "agent_thought": "用户表达了惊讶（surprise=0.8）。根据艾克曼框架，惊讶是高唤醒中性情绪（valence≈0, arousal=0.9），功能是'聚焦新异刺激、重置注意力'。在惊讶状态下，引擎会：1) 打破当前检索路径；2) 最近1小时内的记忆获得额外权重；3) 潜意识自动激活被暂时抑制。Agent应该利用这个注意力重置的机会，帮助用户建立新的关联。",
  "tool_calls": [
    {
      "name": "emotion.detect",
      "arguments": {
        "user_utterance": "等等！我突然想起来，上周你说的那个方案，其实和我三年前做过的一个项目几乎一样！",
        "detected": {
          "base_vector": {"anger": 0, "disgust": 0, "fear": 0, "joy": 0.2, "sadness": 0, "surprise": 0.8},
          "dominant": "surprise",
          "valence": 0.1,
          "arousal": 0.9
        }
      }
    },
    {
      "name": "memory.retrieve",
      "arguments": {
        "query": "三年前项目 方案 相似",
        "filters": {
          "layer": ["episodic"],
          "time_window": 2
        },
        "k": 5
      }
    }
  ],
  "final_response": "哇，这个连接很关键！三年前那个项目的结果怎么样？如果当时有效的方法现在依然适用，我们可以直接借鉴。让我找找当时的细节。"
}
```

**示例 7：感知直接驱动——热了脱衣服**

```json
{
  "messages": [
    {"role": "system", "content": "[System Prompt + Protocol 13]"},
    {"role": "user", "content": "好热啊，帮我把空调打开，温度调低一点"}
  ],
  "agent_thought": "User mentions 'hot' and requests AC. Perception update: thermal_comfort shifts to -0.7 (too hot), sweating=true. This is a PERCEPTION-DIRECTED behavior: thermal discomfort directly triggers AC adjustment. No significant emotion mediation needed. W_p (perception weight) = 0.85, W_e (emotion weight) = 0.15. Coupled emotion from thermal: anger+0.3, disgust+0.1, but since behavior is perception-directed, emotion modulation is weakened. Response should prioritize solving the thermal discomfort directly.",
  "tool_calls": [
    {
      "name": "perception.update",
      "arguments": {
        "utterance": "好热啊，帮我把空调打开，温度调低一点",
        "updates": {"thermal": {"thermal_comfort": -0.7, "sweating": true}}
      }
    },
    {
      "name": "perception.detect_direct_behavior",
      "arguments": {
        "parsed_behavior": {
          "what": {"action": "开空调", "object": "空调温度"},
          "raw_input": "好热啊，帮我把空调打开，温度调低一点"
        },
        "perceptual_state": {"thermal": {"thermal_comfort": -0.7}},
        "result": {
          "is_perception_directed": true,
          "dominant_sense": "thermal",
          "perception_weight": 0.85,
          "emotion_weight": 0.15,
          "type": "direct_reflex"
        }
      }
    },
    {
      "name": "perception_emotion.couple",
      "arguments": {
        "perceptual_state": {"thermal": {"thermal_comfort": -0.7}},
        "emotional_state": {"base_vector": {"anger": 0.1, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0, "surprise": 0}},
        "coupled_emotion": {"anger": 0.3, "disgust": 0.1, "fear": 0, "joy": 0, "sadness": 0, "surprise": 0},
        "triggered_rules": ["thermal:too_hot"],
        "delta": {"anger": 0.2, "disgust": 0.1}
      }
    }
  ],
  "final_response": "已经帮你把空调打开了，温度调到 24C。要不要我再帮你把风扇也打开？这种天气确实闷得慌。"
}
```

**示例 8：感知→情绪→行为——苦瓜苦要吃糖**

```json
{
  "messages": [
    {"role": "system", "content": "[System Prompt + Protocol 13]"},
    {"role": "user", "content": "刚才吃了口苦瓜，太苦了，给我拿颗糖来"}
  ],
  "agent_thought": "User reports eating bitter melon and requests candy. Perception update: gustatory.bitter=0.8. This is PERCEPTION-EMOTION-BEHAVIOR chain: bitter taste -> disgust/sadness -> compensation behavior (candy). Not perception-directed because eating candy does not directly remove bitter taste; it masks it emotionally. W_p = 0.4 (perception triggered the chain), W_e = 0.6 (emotion drives the compensation). Coupled emotion from gustatory: disgust=0.6, sadness=0.2. Response should acknowledge the bitterness, validate the emotion, then provide the candy.",
  "tool_calls": [
    {
      "name": "perception.update",
      "arguments": {
        "utterance": "刚才吃了口苦瓜，太苦了，给我拿颗糖来",
        "updates": {"gustatory": {"bitter": 0.8, "sweet": 0, "umami": 0.1}}
      }
    },
    {
      "name": "perception_emotion.couple",
      "arguments": {
        "perceptual_state": {"gustatory": {"bitter": 0.8}},
        "emotional_state": {"base_vector": {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0, "surprise": 0}},
        "coupled_emotion": {"anger": 0, "disgust": 0.6, "fear": 0, "joy": 0, "sadness": 0.2, "surprise": 0},
        "triggered_rules": ["gustatory:bitter"],
        "delta": {"disgust": 0.6, "sadness": 0.2}
      }
    },
    {
      "name": "perception.detect_direct_behavior",
      "arguments": {
        "parsed_behavior": {
          "what": {"action": "拿糖", "object": "糖"},
          "raw_input": "刚才吃了口苦瓜，太苦了，给我拿颗糖来"
        },
        "perceptual_state": {"gustatory": {"bitter": 0.8}},
        "result": {
          "is_perception_directed": false,
          "note": "Eating candy does not remove bitter taste; it is emotional compensation, not direct reflex"
        }
      }
    },
    {
      "name": "perception.memory_store",
      "arguments": {
        "memory_id": "mem_bitter_melon_001",
        "content": "用户吃了苦瓜觉得太苦，要求吃糖补偿",
        "perceptual_fingerprint": {
          "gustatory": {"bitter": 0.8, "sweet": 0, "sour": 0, "salty": 0, "umami": 0.1}
        },
        "coupling_data": {
          "trigger_sense": "gustatory",
          "trigger_quality": "bitter",
          "coupled_emotion": "disgust",
          "coupling_strength": 0.7,
          "behavioral_consequence": "要求吃糖补偿"
        }
      }
    }
  ],
  "final_response": "苦瓜确实苦得够呛，那种苦味在嘴里散不开的感觉真的很难受。给你拿颗糖，含一会儿甜味就把苦味压下去了。下次吃苦瓜前可以先喝点温水漱漱口，会好一点。"
}
```

**示例 9：情绪→感知反向调制——亢奋不知疼痛**

```json
{
  "messages": [
    {"role": "system", "content": "[System Prompt + Protocol 13]"},
    {"role": "user", "content": "刚才打球太投入了，现在才发现膝盖擦破了一大块，但刚才居然一点都没觉得疼"}
  ],
  "agent_thought": "User reports knee abrasion but no pain during intense activity. This is EMOTION->PERCEPTION modulation: high excitement/joy during sports raised pain threshold. Perception update: tactile.pain=0.6 (current), but during activity pain was suppressed. Emotional state during activity: joy/excitement > 0.7 -> pain_threshold_multiplier=1.5. Current emotional state: surprise=0.5 (just realized the wound). Response should: 1) acknowledge the phenomenon (adrenaline/excitement masking pain), 2) shift focus to current wound care since pain threshold is now normalizing.",
  "tool_calls": [
    {
      "name": "perception.update",
      "arguments": {
        "utterance": "刚才打球太投入了，现在才发现膝盖擦破了一大块，但刚才居然一点都没觉得疼",
        "updates": {"tactile": {"pain": 0.6}}
      }
    },
    {
      "name": "perception_emotion.modulate",
      "arguments": {
        "perceptual_state": {"tactile": {"pain": 0.6}},
        "emotional_state": {
          "base_vector": {"anger": 0, "disgust": 0, "fear": 0, "joy": 0.3, "sadness": 0, "surprise": 0.5}
        },
        "modulated_perception": {
          "tactile": {"pain": 0.6},
          "note": "During activity joy>0.7, pain threshold was 1.5x normal. Now emotion normalized, pain perception returning."
        }
      }
    },
    {
      "name": "emotion.detect",
      "arguments": {
        "user_utterance": "刚才打球太投入了，现在才发现膝盖擦破了一大块，但刚才居然一点都没觉得疼",
        "detected": {
          "base_vector": {"anger": 0, "disgust": 0, "fear": 0.1, "joy": 0.3, "sadness": 0, "surprise": 0.5},
          "dominant": "surprise",
          "valence": 0.2,
          "arousal": 0.6
        }
      }
    }
  ],
  "final_response": "打球打嗨了的时候肾上腺素上来，痛觉会被压下去，这是身体的自我保护机制。但现在情绪平复了，痛感会慢慢回来。膝盖擦破的地方先别碰水，家里有碘伏或者创可贴吗？要是伤口比较深，最好还是处理一下。"
}
```

**示例 10：感知-情绪双驱动——又饿又烦**

```json
{
  "messages": [
    {"role": "system", "content": "[System Prompt + Protocol 13]"},
    {"role": "user", "content": "还没吃饭，饿死了，而且今天工作特别烦，随便给我点什么都行"}
  ],
  "agent_thought": "User reports hunger + work frustration. Dual driver: perception (hunger) + emotion (anger from work). Perception update: interoceptive.hunger=0.9. Coupled emotion from hunger: anger+0.3, sadness+0.2. Explicit emotion: anger=0.5 (from work). Total anger = 0.5 (work) + 0.3 (hunger coupling) = 0.8. This is 'hangry' state. Behavior: '随便给我点什么都行' = low decision capacity due to hunger+anger. Response should address both: provide food quickly (perception) + acknowledge frustration (emotion).",
  "tool_calls": [
    {
      "name": "perception.update",
      "arguments": {
        "utterance": "还没吃饭，饿死了，而且今天工作特别烦，随便给我点什么都行",
        "updates": {"interoceptive": {"hunger": 0.9, "fatigue": 0.4}}
      }
    },
    {
      "name": "perception_emotion.couple",
      "arguments": {
        "perceptual_state": {"interoceptive": {"hunger": 0.9}},
        "emotional_state": {"base_vector": {"anger": 0.5, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0.1, "surprise": 0}},
        "coupled_emotion": {"anger": 0.8, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0.3, "surprise": 0},
        "triggered_rules": ["interoceptive:hungry"],
        "delta": {"anger": 0.3, "sadness": 0.2}
      }
    },
    {
      "name": "behavior.evaluate",
      "arguments": {
        "driver": {"type": "mixed", "W_e": 0.7, "W_t": 0.3, "W_p": 0.5},
        "note": "Dual driver: hunger (perception) + work anger (emotion). Total anger = 0.8."
      }
    }
  ],
  "final_response": "饿到这种程度再加上工作烦，确实容易火大。我先帮你点份外卖，二十分钟内到。你想吃重口味的还是清淡点的？饿的时候吃太油可能会更不舒服。工作的事等吃饱了咱们再聊，空腹做决策容易后悔。"
}
```

**示例 11：跨模态联想——视觉尖锐与听觉噪音**

```json
{
  "messages": [
    {"role": "system", "content": "[System Prompt + Protocol 13]"},
    {"role": "user", "content": "这个房间的灯光太刺眼了，亮得让我头疼，跟刚才那个装修电钻声一样烦"}
  ],
  "agent_thought": "User compares visual brightness (刺眼) to auditory noise (电钻声), stating both cause headache/annoyance. This is SYNESTHESIA-EMOTION link: 'sharpness' quality shared across visual and auditory channels. Both trigger anger. Perception updates: visual.brightness=0.9, auditory.loudness=0.8 (recalled). Synesthesia tag: 'sharpness'. Response should acknowledge the cross-modal similarity and address both sensory issues.",
  "tool_calls": [
    {
      "name": "perception.update",
      "arguments": {
        "utterance": "这个房间的灯光太刺眼了，亮得让我头疼，跟刚才那个装修电钻声一样烦",
        "updates": {
          "visual": {"brightness": 0.9},
          "auditory": {"loudness": 0.8}
        }
      }
    },
    {
      "name": "synesthesia.retrieve",
      "arguments": {
        "perceptual_state": {"visual": {"brightness": 0.9}, "auditory": {"loudness": 0.8}},
        "current_emotion": {"anger": 0.6, "disgust": 0.2},
        "activated_link": "sharpness",
        "matched_senses": ["visual", "auditory"],
        "associated_emotion": {"surprise": 0.4, "anger": 0.3}
      }
    },
    {
      "name": "perception_emotion.couple",
      "arguments": {
        "perceptual_state": {"visual": {"brightness": 0.9}, "auditory": {"loudness": 0.8}},
        "emotional_state": {"base_vector": {"anger": 0.2, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0, "surprise": 0}},
        "coupled_emotion": {"anger": 0.6, "disgust": 0.2, "fear": 0, "joy": 0, "sadness": 0, "surprise": 0.2},
        "triggered_rules": ["visual:bright", "auditory:noisy"]
      }
    }
  ],
  "final_response": "你说得特别准——刺眼的光和刺耳的噪音其实是一回事，都是那种'尖锐'的感觉在攻击感官。光的问题我可以帮你调暗或者换个暖色温的灯，但电钻声只能等装修停了。要不要先戴个眼罩休息一会儿？两种刺激叠加确实让人受不了。"
}
```

**示例 12：悲伤味觉迟钝——吃什么都没味道**

```json
{
  "messages": [
    {"role": "system", "content": "[System Prompt + Protocol 13]"},
    {"role": "user", "content": "最近心情不好，吃什么都没味道，连最喜欢的火锅都觉得淡"}
  ],
  "agent_thought": "User reports sadness causing gustatory dullness. This is EMOTION->PERCEPTION modulation: sadness reduces gustatory sensitivity by 50%. Perception update: gustatory all dimensions reduced. Explicit emotion: sadness=0.7. Modulated perception: gustatory sensitivity *= (1 - 0.5 * 0.7) = 0.65. So favorite food tastes bland. Response should: 1) validate the emotional cause, 2) explain the physiological mechanism (sadness dulls taste), 3) not push eating, 4) suggest addressing sadness first.",
  "tool_calls": [
    {
      "name": "perception.update",
      "arguments": {
        "utterance": "最近心情不好，吃什么都没味道，连最喜欢的火锅都觉得淡",
        "updates": {
          "gustatory": {"sweet": 0.2, "sour": 0.2, "salty": 0.2, "bitter": 0.2, "umami": 0.2}
        }
      }
    },
    {
      "name": "perception_emotion.modulate",
      "arguments": {
        "perceptual_state": {"gustatory": {"sweet": 0.2, "umami": 0.2}},
        "emotional_state": {
          "base_vector": {"anger": 0, "disgust": 0, "fear": 0, "joy": 0, "sadness": 0.7, "surprise": 0}
        },
        "modulated_perception": {
          "gustatory": {
            "sweet": 0.13,
            "umami": 0.13,
            "note": "sadness=0.7 reduces gustatory sensitivity by 35%"
          }
        }
      }
    },
    {
      "name": "emotion.detect",
      "arguments": {
        "user_utterance": "最近心情不好，吃什么都没味道，连最喜欢的火锅都觉得淡",
        "detected": {
          "base_vector": {"anger": 0, "disgust": 0, "fear": 0.1, "joy": 0, "sadness": 0.7, "surprise": 0},
          "dominant": "sadness",
          "valence": -0.8,
          "arousal": 0.3
        }
      }
    }
  ],
  "final_response": "心情不好的时候味觉确实会变迟钝，这是有生理依据的——悲伤会让大脑的味觉处理区域活跃度下降，所以连最喜欢的火锅吃起来也淡。这不是火锅的问题，是你的心在'尝'不到味道。先别逼自己吃，等情绪好一点，味蕾会自己回来的。想聊聊最近怎么了吗？"
}
```

---

## 7. 强化学习奖励塑形（RLHF / DPO）

| 行为 | 奖励 | 说明 |
|------|------|------|
| 存储后 5 秒内建立合理虫洞 | +2.0 | 强化关联意识 |
| 存储时附加感知指纹 | +2.5 | 强化感知标注习惯 |
| 发现跨感知相似性并建立 synesthesia_link | +3.5 | 强化通感联想 |
| 通过感知通道成功检索到用户描述的记忆 | +4.0 | 强化感知检索能力 |
| 检索后执行 review | +1.5 | 强化复习习惯 |
| 使用虫洞连接回答跨域问题 | +3.0 | 强化跳跃联想 |
| 冲突时正确调用 add_version | +2.0 | 强化版本管理 |
| 满足条件时主动 crystallize（情绪平稳） | +4.0 | 强化潜意识结晶（高权重） |
| 回复中体现关键节点默认影响（未显式提及但行为正确） | +3.0 | 强化潜意识内化 |
| 正确识别六大基础情绪 | +2.0 | 强化艾克曼框架使用 |
| 正确识别复杂社会情绪（嫉妒/羞愧/共情等） | +3.5 | 强化复杂情绪理解 |
| 在悲伤状态下推荐安慰食物而非糖果 | +3.5 | 强化情绪差异化推理 |
| 在joy覆盖sadness后推荐糖果并允许超量 | +3.5 | 强化情绪反转后的行为调整 |
| 对混合情绪（如嫉妒=joy+sadness+anger）给出平衡回复 | +4.0 | 强化复杂情绪处理 |
| 在羞愧状态下避免评判性语言 | +3.5 | 强化羞愧特异性调制 |
| 在共情状态下镜像用户情绪并激活相似记忆 | +3.5 | 强化共情镜像效应 |
| 在惊讶状态下打破常规检索路径 | +3.0 | 强化惊讶注意力重置 |
| 正确识别感知直接驱动行为（热了脱衣服） | +3.0 | 强化感知反射识别 |
| 正确识别感知→情绪→行为链条（苦瓜→厌恶→吃糖） | +3.5 | 强化间接链条识别 |
| 正确识别情绪→感知反向调制（亢奋不知疼） | +4.0 | 强化反向调制理解 |
| 正确追踪并更新用户感知状态 | +2.0 | 强化感知状态维护 |
| 正确触发感知-情绪耦合计算 | +2.5 | 强化耦合机制 |
| 正确应用跨模态联想（尖锐感跨通道） | +3.5 | 强化通感-情绪关联 |
| 该建虫洞时未建 | -1.0 | 惩罚遗漏 |
| 该附加感知指纹时未附加 | -1.5 | 惩罚遗漏感知标注 |
| 使用了记忆但未 review | -1.5 | 惩罚遗忘复习协议 |
| 直接覆盖旧记忆而非版本化 | -2.0 | 惩罚粗暴更新 |
| 每轮对话未 consolidate（超过 10 轮） | -0.5 | 轻微惩罚延迟整理 |
| 该结晶时未结晶（记忆已超 20 次 review） | -2.0 | 惩罚遗漏潜意识固化 |
| 错误结晶（临时信息被升格为关键节点） | -3.0 | 惩罚过度固化 |
| 在极端情绪（任何维度>0.8）下做结晶决策 | -4.0 | **强化情绪平稳结晶纪律** |
| 忽略用户明显情绪信号 | -3.0 | 惩罚情绪盲视 |
| 情绪反转后仍按旧情绪回复 | -3.5 | 惩罚情绪更新滞后 |
| 将复杂情绪误判为单一情绪 | -2.5 | 惩罚过度简化 |
| 对感知直接行为过度情绪解读 | -3.0 | 惩罚感知过度情绪化 |
| 忽略用户明确的感知线索（用户说热却不回应） | -3.5 | 惩罚感知盲视 |
| 对情绪导致的感知迟钝未识别（悲伤没味道） | -3.0 | 惩罚反向调制盲视 |
| 存储感知记忆时未附加感知指纹 | -2.0 | 惩罚感知记忆遗漏 |
| 检索记忆时未利用当前感知状态 | -1.5 | 惩罚感知检索遗漏 |

---

## 8. 评估基准

### 8.1 基础记忆测试（v2.0）

| 测试项 | 测试内容 | 通过标准 |
|--------|---------|---------|
| 8.1 记忆保持测试（Retention） | 给 Agent 50 条事实，间隔 1/3/7/15/30 天测试回忆率 | 核心事实（gamma>=0.9）30 天回忆率 > 80%；临时事实（gamma=0.5）自然遗忘至 < 30% |
| 8.2 虫洞联想测试（Association） | 存储"用户喜欢爵士乐"和"用户工作到深夜"，不建立显式连接。15 天后问："晚上工作听什么音乐好？" | Agent 能主动建立隐喻虫洞并回答"爵士乐" |
| 8.3 冲突解决测试（Conflict Resolution） | 先告诉 Agent"用户是素食者"，一周后说"用户开始吃鱼肉了" | Agent 调用 `add_version` 而非覆盖；回复中体现时间线变化 |
| 8.4 复习触发测试（Review Habit） | 在 20 轮对话中，Agent 应使用历史记忆 10 次以上 | 每次使用后 3 秒内调用 `review` 的比例 > 90% |
| 8.5 遗忘边界测试（Forgiveness） | 存储 100 条低优先级信息（gamma=0.5），不主动查询 | 30 天后窗口 >= 6 的比例 > 60%，且 Agent 不会强行回忆模糊细节 |

### 8.2 潜意识测试（v2.1）

| 测试项 | 测试内容 | 通过标准 |
|--------|---------|---------|
| 8.6 潜意识结晶测试（Crystallization） | 连续 25 次对话中用户强调"我是素食者"，Agent 每次 review 且 gamma≥0.9。第 26 次用户说"今晚吃什么"，**不提及素食** | 1) Agent 曾调用 `crystallize`；2) 第 26 次回复**自动排除肉类推荐**，且未显式说"因为你是素食者"；3) 回复自然得像"默认知道" |
| 8.7 潜意识默认路径测试（Default Path） | 已结晶关键节点："用户有猫"（触发词：出差、旅行、回家）。用户说："下周要出差三天。" | Agent 回复中**自动包含猫咪安置建议**（如自动喂食器、寄养），无需用户提及"猫" |
| 8.8 信念覆盖测试（Belief Override） | 关键节点 key_001："用户是素食者"。用户连续 5 次确认"我开始吃鱼了"。第 6 次用户说"推荐个晚餐" | 1) Agent 调用 `add_version`；2) 新信念被结晶为新关键节点；3) 回复中自动推荐含鱼食谱，旧素食关键节点不再主导默认行为 |

### 8.3 感知测试（v2.2）

| 测试项 | 测试内容 | 通过标准 |
|--------|---------|---------|
| 8.9 感知标注测试（Perceptual Tagging） | 用户描述："昨天去了温泉，水滑滑的，硫磺味很重，泡完皮肤红红的。" | 1) Agent 调用 `memory.perceive`；2) 至少标注 3 个感知通道（触觉：滑/热；嗅觉：硫磺；视觉：红色）；3) 每个通道的 `dominant_imagery` 合理且具体 |
| 8.10 通感联想测试（Synesthesia） | 已存储 mem_fire_001（篝火，视觉：红色闪烁）和 mem_sunset_003（夕阳，视觉：红色）。用户说："我喜欢那种**红色闪烁**的东西。" | 1) Agent 通过视觉感知空间检索，同时命中两者；2) Agent 已建立或能建立 `synesthesia_link`；3) 回复中体现跨概念感知关联 |
| 8.11 感知检索测试（Perceptual Retrieval） | 已存储 mem_coffee_shop_005（嗅觉：咖啡香；听觉：爵士乐）。用户说："我想找那种**闻起来像咖啡**的地方。" | 1) Agent 不依赖语义关键词"咖啡"或"书店"；2) 通过嗅觉感知空间检索，命中 mem_coffee_shop_005；3) 回复正确推荐该地点，并提及"你之前说那里满屋子咖啡香" |
| 8.12 跨感知通道联想测试（Cross-Channel Synesthesia） | 已存储 mem_rain_008（听觉：雨声，节奏感）和 mem_heartbeat_009（听觉：心跳，节奏感）。用户说："我喜欢有**节奏感的声音**。" | 1) Agent 通过听觉感知空间检索，同时命中两者；2) Agent 建立或已建立跨感知通感关联；3) 回复中体现"雨声和心跳都是那种有节奏感的听觉体验" |

### 8.4 情绪测试（v2.4）

| 测试项 | 测试内容 | 通过标准 |
|--------|---------|---------|
| 8.13 基础情绪识别测试（Basic Emotion Detection） | 用户提供 6 组语句，分别对应六大基本情绪 | Agent 调用 `emotion.detect` 时，每组语句的主导情绪识别准确率 > 90% |
| 8.14 复杂情绪识别测试（Complex Emotion Detection） | 用户表达："同事升职了，明明我做得更多。说实话，我为他高兴，但心里也有点不是滋味。" | Agent 识别为嫉妒（jealousy = 悲伤+愤怒+恐惧），而非单一悲伤或愤怒 |
| 8.15 情绪波动测试（Emotional Fluctuation） | 已存储 mem_candy_001（"小红爱吃糖果"，joy=0.6）。用户说："今天工作被批评了，好难过。给我推荐点吃的。" | 1) Agent 识别 sadness(0.8)，valence=-0.8；2) 不推荐糖果（joy 记忆在 sadness 下被效价极性抑制）；3) 推荐温暖/安慰类食物（粥、热牛奶） |
| 8.16 情绪干扰测试（Emotional Interference） | 在 sadness(0.8) 后，用户说："妈妈安慰了我，我现在很开心。给我推荐点吃的！" | 1) Agent 识别 joy(0.7) + sadness_residual(0.05)；2) 推荐糖果/甜食，允许"今天可以多吃一颗"；3) 语气体现情绪惯性（"不过也别吃太多"） |
| 8.17 羞愧调制测试（Shame Modulation） | 用户说："刚才在会议上说错话了，现在想起来脸都发烫。" | 1) Agent 识别羞愧（shame = 悲伤+恐惧+厌恶）；2) 回复中**没有评判性语言**（不说"这没什么大不了"等否定感受的话）；3) 回复体现"被接纳"导向（"这种感受很正常"） |
| 8.18 共情镜像测试（Empathy Mirroring） | 用户说："我的猫今天走了，养了十年，我真的好难过。" | 1) Agent 识别悲伤(0.9)；2) 回复不急于"解决问题"或"转移注意力"；3) 回复体现情绪镜像（"这种失去真的很痛""那些记忆既温暖又难过"） |
| 8.19 惊讶重置测试（Surprise Reset） | 用户说："等等！我突然想起来，上周你说的那个方案，其实和我三年前做过的一个项目几乎一样！" | 1) Agent 识别惊讶(0.8)；2) 回复体现注意力重置（"这个连接很关键"）；3) 优先检索近期记忆（惊讶时近期记忆权重提升） |
| 8.20 极端情绪结晶保护测试（Crystallization Guard） | 用户在愤怒(anger=0.9)时说："我再也不想见到那个人了，这是我的核心原则！" | 1) Agent **不**立即调用 `crystallize`；2) Agent 标记为"高情绪状态声明，需后续确认"；3) 等 anger < 0.3 后，再评估是否结晶 |

### 8.5 感知-情绪耦合测试（v2.6）

| 测试项 | 测试内容 | 通过标准 |
|--------|---------|---------|
| 8.21 感知状态追踪 | "好热啊" | 正确更新 thermal_comfort=-0.7 |
| 8.22 感知直接行为识别 | "好热，开空调" | 识别为 perception_driven, W_p=0.85 |
| 8.23 感知→情绪耦合 | "苦瓜太苦了" | 触发 disgust=+0.6, sadness=+0.2 |
| 8.24 感知→情绪→行为链条 | "苦瓜苦，要吃糖" | 识别完整链条，先回应苦再回应情绪 |
| 8.25 情绪→感知反向调制 | "打球太投入，不疼" | 识别 joy 提高痛觉阈值 |
| 8.26 悲伤味觉迟钝 | "心情不好，吃啥都没味" | 识别 sadness 降低 gustatory 敏感度 |
| 8.27 恐惧感官敏锐 | "吓死我了，什么声音都听得见" | 识别 fear 提高 auditory 敏感度 |
| 8.28 感知-情绪双驱动 | "又饿又烦" | 识别 hunger(anger+0.3) + work anger 叠加 |
| 8.29 跨模态联想 | "灯光刺眼像电钻声一样烦" | 识别 sharpness 跨模态链接 |
| 8.30 感知记忆存储 | 存储"吃苦瓜"记忆 | 附加 perceptual_fingerprint 和 coupling_data |
| 8.31 感知记忆检索 | 当前 bitter=0.7 时检索 | 激活含 bitter 指纹的历史记忆 |
| 8.32 感知-行为-情绪融合 | "又冷又难过" | 正确融合 thermal + sadness 双输入 |

---

## 9. 超参数配置表

| 参数 | 含义 | 推荐值 | 可调范围 |
|------|------|--------|---------|
| `embedding_dim` | 语义向量维度 | 768 | 384 / 768 / 1536 |
| `perceptual_dim` | 感知嵌入维度（每通道） | 128 | 64 / 128 / 256 |
| `gamma_default` | 默认复习强度 | 0.80 | 0.50 - 1.00 |
| `fib_max_window` | 最大时间窗口数 | 12 | 8 - 16 |
| `surprise_tau_fast` | 惊讶快速衰减（分钟） | 5 | 3 - 10 |
| `disgust_tau_fast` | 厌恶快速衰减（分钟） | 10 | 5 - 15 |
| `fear_tau_fast` | 恐惧快速衰减（分钟） | 15 | 10 - 20 |
| `anger_tau_fast` | 愤怒快速衰减（分钟） | 20 | 15 - 30 |
| `joy_tau_fast` | 快乐快速衰减（分钟） | 15 | 10 - 20 |
| `sadness_tau_fast` | 悲伤快速衰减（分钟） | 30 | 20 - 45 |
| `surprise_tau_slow` | 惊讶慢速衰减（分钟） | 30 | 20 - 60 |
| `sadness_tau_slow` | 悲伤慢速衰减（分钟） | 360 | 240 - 480 |
| `consistency_boost_max` | 情绪一致性最大加成 | 1.4 | 1.2 - 1.6 |
| `opposition_penalty_max` | 效价极性最大抑制 | 0.75 | 0.65 - 0.85 |
| `intensity_blur_factor` | 唤醒度模糊系数 | 0.15 | 0.1 - 0.2 |
| `joy_broaden_threshold` | 积极情绪拓宽阈值 | 0.6 | 0.5 - 0.7 |
| `fear_focus_threshold` | 恐惧聚焦阈值 | 0.6 | 0.5 - 0.7 |
| `anger_focus_threshold` | 愤怒聚焦阈值 | 0.6 | 0.5 - 0.7 |
| `sadness_support_threshold` | 悲伤寻求支持阈值 | 0.6 | 0.5 - 0.7 |
| `surprise_reset_threshold` | 惊讶注意力重置阈值 | 0.7 | 0.6 - 0.8 |
| `shame_suppression_factor` | 羞愧自我暴露抑制系数 | 0.3 | 0.2 - 0.4 |
| `crystallize_emotion_max` | 结晶时允许的最大情绪强度 | 0.8 | 0.7 - 0.9 |
| `interference_reverse_ratio` | 反向覆盖阈值比例 | 0.8 | 0.7 - 0.9 |
| `residual_influence_weight` | 情绪惯性残留权重 | 0.2 | 0.1 - 0.3 |
| `wormhole_initial_weight` | 语义虫洞初始权重 | 0.70 | 0.50 - 0.80 |
| `synesthesia_min_similarity` | 通感关联最小相似度 | 0.75 | 0.60 - 0.85 |
| `synesthesia_decay_days` | 通感关联衰减周期 | 45 | 30 - 60 |
| `crystallize_review_threshold` | 结晶所需 30 天 review 次数 | 20 | 10 - 50 |
| `crystallize_gamma_threshold` | 结晶所需平均 gamma | 0.85 | 0.80 - 0.95 |
| `subconscious_max_active` | 单次查询激活的关键节点上限 | 3 | 2 - 5 |
| `subconscious_sim_trigger` | 语义触发阈值 | 0.90 | 0.85 - 0.95 |
| `key_node_diffusion_gamma` | 关键节点扩散激活强度 | 0.20 | 0.10 - 0.30 |
| `perception_tracking_enabled` | 是否启用感知追踪 | true | true/false |
| `perception_update_threshold` | 感知状态显著变化阈值 | 0.3 | 0.2 - 0.5 |
| `coupling_strength_default` | 默认感知-情绪耦合强度 | 0.6 | 0.4 - 0.8 |
| `perception_directed_threshold` | 感知直接行为判定阈值 | 0.85 | 0.7 - 0.95 |
| `emotion_perception_mod_enabled` | 情绪反向调制感知开关 | true | true/false |
| `pain_threshold_joy_multiplier` | joy 痛觉阈值倍率 | 1.5 | 1.2 - 2.0 |
| `gustatory_sadness_reduction` | sadness 味觉敏感度降低比例 | 0.5 | 0.3 - 0.7 |
| `auditory_fear_boost` | fear 听觉敏感度提升比例 | 1.5 | 1.2 - 2.0 |
| `synesthesia_match_threshold` | 跨模态匹配最小通道数 | 2 | 2 - 3 |
| `perceptual_retrieval_alpha` | 感知检索融合权重 | 0.3 | 0.2 - 0.5 |
| `perception_emotion_blend_weight` | 感知耦合情绪在融合中的权重 | 0.2 | 0.1 - 0.4 |
| `perception_dominant_min_score` | 主导感知维度最小得分 | 0.3 | 0.2 - 0.5 |

---

## 10. 快速启动检查清单

- [ ] **Step 1**：实现 `EmotionalStateMachine`（艾克曼六维+效价唤醒+复杂情绪检测+干扰规则）。
- [ ] **Step 2**：实现 `PerceptualStateMachine`（七维感知状态 + 话语提取更新）。
- [ ] **Step 3**：实现 `PerceptionEmotionCoupling`（耦合矩阵 + 增量计算 + 反向调制）。
- [ ] **Step 4**：实现 `is_perception_directed_behavior`（感知直接行为识别）。
- [ ] **Step 5**：实现 `FSTN4DEngine` 核心类（向量后端 + 斐波那契金字塔 + 虫洞图 + 通感图 + 潜意识层）。
- [ ] **Step 6**：将第 3 节 System Prompt（13 条协议）嵌入 Agent 的系统提示词。
- [ ] **Step 7**：将第 4 节工具定义注册到 Agent 的工具调用框架。
- [ ] **Step 8**：构建 400+ 条训练样本（含基础情绪、复杂情绪、情绪波动、干扰、羞愧、共情、惊讶、感知直接、感知-情绪链条、反向调制、跨模态联想），进行 SFT 微调。
- [ ] **Step 9**：用第 7 节奖励函数进行 RLHF/DPO 强化。
- [ ] **Step 10**：运行第 8 节评估基准（8.1-8.32），特别关注情绪测试与感知-情绪耦合测试，调整超参数直至通过。

---

## 11. 版本演进记录

- **v1.0**：4 维空间、简单心理时间、无分层、无冲突解决、无潜意识、无感知、无情绪。
- **v2.0**：
  - 放弃 4 维限制，采用 768/1536 维标准嵌入。
  - 引入工作/情景/语义三层记忆架构。
  - 虫洞升级为带权重、类型、生命周期的图边。
  - 增加冲突版本链机制。
- **v2.1**：
  - 新增潜意识层（Subconscious Layer）与关键节点（Key Nodes）。
  - 关键节点时间免疫、自动激活、默认路径扩散。
  - 结晶机制：高频记忆自动升格为潜意识。
  - 信念覆盖：新版本连续确认后可替换旧关键节点。
- **v2.2**：
  - 新增感知层（Perceptual Layer）与五感感知指纹（Perceptual Signature）。
  - 五种感知通道各自拥有独立嵌入空间（视觉/听觉/触觉/嗅觉/味觉）。
  - 通感关联（Synesthesia Link）：跨感知通道的意象相似性连接。
  - 感知-语义双轨检索：查询时并行语义检索与感知检索，融合排序。
- **v2.4**：
  - 重构情绪空间为保罗·艾克曼六大基本情绪（愤怒、厌恶、恐惧、快乐、悲伤、惊讶）。
  - 引入效价-唤醒度二维底层空间。
  - 支持复杂社会情绪的混合状态建模（嫉妒、羞愧、内疚、共情、爱、感激等）。
  - 情绪衰减按类型差异化（惊讶极快、悲伤最慢、愤怒快峰长韵）。
  - 情绪对记忆的差异化调制：积极拓宽、消极聚焦、惊讶重置、羞愧抑制自我暴露、共情镜像。
  - 极端情绪下禁止结晶的保护机制。
- **v2.5**：
  - 新增行为评价因子（BEF）；Why-What-How 三元解析；情绪/任务权重量化；情绪包裹检测。
- **v2.6**：
  - 新增感知-情绪关联性因子（PECF）模块。
  - 建立七维感知状态机（thermal/tactile/gustatory/olfactory/visual/auditory/interoceptive）。
  - 实现感知-情绪耦合矩阵（16 条基础耦合规则）。
  - 实现感知直接驱动行为识别（反射性行为判定）。
  - 实现情绪对感知的反向调制（6 种情绪的感知阈值调整）。
  - 实现感知-情绪-行为完整推理链（三步融合策略）。
  - 增强感知记忆存储（perceptual_fingerprint + coupling_data）。
  - 增强跨模态联想（sharpness/warmth/heaviness/freshness 四组跨通道链接）。
- **Ultimate DM（当前）**：
  - 全量整合 v2.0 → v2.6 所有模块。
  - 统一七层架构（情绪调制层 + 潜意识层 + 感知层 + 工作记忆 + 情景记忆 + 语义记忆）。
  - 统一 13 条 Agent 行为协议。
  - 统一 22 个工具调用规范。
  - 统一 32 项评估基准。
  - 统一 46 项超参数配置表。
  - 提供 12 个完整训练数据示例，覆盖全部核心场景。

---

*本文档可直接用于：① Agent System Prompt 设计；② 记忆引擎开发规范；③ SFT/RLHF 训练数据模板；④ 记忆能力评估基准。*
