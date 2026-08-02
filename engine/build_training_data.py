# -*- coding: utf-8 -*-
"""
build_training_data.py — FSTN-4D SFT/IFT 训练样本生成器（文档 §6）

按文档 §6.1 的格式生成 400+ 条训练样本：
  messages: [system(协议), user(话语)]
  agent_thought: 思维链（为何选择该记忆操作）
  tool_calls: 引擎调用（emotion.detect / memory.retrieve / ...）
  final_response: 最终回复

覆盖场景（文档 §7 奖励表 + §6 示例 + §8 基准映射）：
  1. 基础情绪 ×6  2. 复杂情绪 ×8  3. 情绪波动/干扰 ×4
  4. 羞愧/共情/惊讶 ×6  5. 感知直接/感知-情绪链条/反向调制 ×8
  6. 记忆/结晶/默认路径 ×6  7. 通感/感知检索 ×6  8. 双驱动/融合 ×4
  9. 冲突解决/版本 ×4  10. 否定/程度/转折（v2 修复验证）×4
  11. 虫洞联想 ×4  12. 复习/遗忘 ×4  13. 跨域虫洞 ×6  14. 感知状态 ×6

产物：train_fstn_4d.jsonl（每条一行 JSON）
       train_fstn_4d_stats.json（统计）
"""

import json
import os
import random

random.seed(42)

SYSTEM_PROMPT = """你配备了一个 FSTN-4D 长期记忆引擎。处理信息时，你必须遵循协议：
协议一：存储即关联（Ingest & Link）——新信息立即思考与已有记忆的因果/隐喻/跨域/共现关联，存在则建虫洞。
协议二：感知标注（Perceive）——存储含感官体验的信息时附加五感感知指纹。
协议三：通感联想（Synesthesia）——发现两概念在同一感知通道意象相似时建立跨感知关联。
协议四：情绪感知（Emotional Tagging）——用艾克曼六维识别用户情绪，识别复杂社会情绪（嫉妒/羞愧/内疚/共情/爱/感激）。
协议五：情绪差异化推理——积极情绪拓宽检索；愤怒聚焦对抗；恐惧聚焦安全；悲伤聚焦支持；厌恶聚焦排斥；惊讶重置注意力。
协议六：情绪干扰处理——同维叠加/异维叠加/反向覆盖/情绪惯性（残留 20%）。
协议七：使用即复习（Retrieve & Review）——检索到记忆用于回复后立即 review，按重要度选 gamma。
协议八：潜意识结晶（Crystallize）——记忆被反复确认且情绪平稳时升格为关键节点；极端情绪（>0.8）禁止结晶。
协议九：遗忘是功能（Let Go）——深层窗口记忆允许合并/剪枝，不强行回忆细节。
协议十：主动跳跃联想（Wormhole Thinking）——回答前检查是否与看似无关的旧记忆存在虫洞连接。
协议十一：冲突解决——新旧信息矛盾时 add_version 建版本链，不直接覆盖。
协议十二：感知-情绪耦合——区分感知直接驱动（热→开空调 W_p=0.85）、感知→情绪→行为（苦→厌恶→吃糖）、情绪→感知反向调制（亢奋不觉疼）。
协议十三：跨模态联想——检索时并行语义检索与感知检索，融合排序。"""


def make_sample(user_text, thought, tool_calls, final_response, scenario,
                category, emotion_vector=None):
    """构造一条训练样本"""
    sample = {
        "scenario": scenario,
        "category": category,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "agent_thought": thought,
        "tool_calls": tool_calls,
        "final_response": final_response,
    }
    if emotion_vector:
        sample["emotion_vector"] = emotion_vector
    return sample


def dtool(name, args):
    return {"name": name, "arguments": args}


SAMPLES = []


# ═══════════════════════════════════════════════════════════
# 1. 基础情绪 ×6（文档 §8.13）
# ═══════════════════════════════════════════════════════════

def gen_basic_emotions():
    cases = [
        ("我恨死你了！", "anger", {"anger": 0.9},
         "太气人了！先别急着做决定，喝口水冷静一下。要聊聊发生了什么吗？"),
        ("这让我觉得恶心", "disgust", {"disgust": 0.88},
         "理解，这种感觉确实让人不舒服。要不要离那个东西远一点？"),
        ("我好害怕", "fear", {"fear": 0.82},
         "别怕，我在这里。你具体在怕什么？说出来我们一起面对。"),
        ("太开心了！", "joy", {"joy": 0.95},
         "哇，太好了！分享给我听听，是什么让你这么开心？"),
        ("我好难过", "sadness", {"sadness": 0.82},
         "抱抱。难过的时候不用强撑，想哭就哭，我陪着你。"),
        ("什么？真的假的？！", "surprise", {"surprise": 0.93},
         "哇，这太意外了！快跟我说说细节，我完全没想到！"),
    ]
    for text, dom, vec, reply in cases:
        SAMPLES.append(make_sample(
            text,
            f"根据艾克曼框架，主导情绪为{dom}（{dom}权重最高）。调用 emotion.detect 更新状态。",
            [dtool("emotion.detect", {"user_utterance": text,
                                      "detected": {"base_vector": vec, "dominant": dom}}),
             dtool("emotion.get_current", {})],
            reply, f"basic_{dom}", "basic_emotion", vec,
        ))


# ═══════════════════════════════════════════════════════════
# 2. 复杂情绪 ×8（文档 §8.14）
# ═══════════════════════════════════════════════════════════

def gen_complex_emotions():
    cases = [
        ("同事升职了，明明我做得更多。我为他高兴，但心里也有点不是滋味",
         "jealousy", {"anger": 0.4, "sadness": 0.5, "joy": 0.4},
         "为他高兴是真的，心里失落也是真的，两种感受都正常。你觉得自己做得更多却没被看到，这种不公平感确实难受。要不要聊聊你的成果？"),
        ("刚才在会议上说错话了，现在想起来脸都发烫，大家肯定觉得我很蠢",
         "shame", {"sadness": 0.5, "fear": 0.3, "disgust": 0.2},
         "说错话确实让人不自在，但大多数人几分钟就忘了——他们更关心自己。这种感受很正常，不需要为此否定自己。"),
        ("我骗了妈妈，现在特别内疚", "guilt", {"sadness": 0.6, "fear": 0.3},
         "内疚说明你在乎这段关系。承认错误需要勇气，但真诚道歉会比一直憋着好受。需要我帮你想想怎么说吗？"),
        ("太尴尬了，走路摔了一跤，全班都看着", "embarrassment", {"surprise": 0.4, "sadness": 0.3, "fear": 0.3},
         "哎呀，摔跤谁都经历过，同学们可能笑一下也就忘了。你没事吧？有没有摔疼？"),
        ("朋友失去了亲人，我看到她哭，心里也跟着难受", "empathy", {"sadness": 0.5, "joy": 0.2, "surprise": 0.1},
         "共情是珍贵的品质。你现在一定也很不好受——她的悲伤你感受到了。陪着她就好，不需要说太多漂亮话。"),
        ("和TA在一起的每一秒都好幸福", "love", {"joy": 0.6, "sadness": 0.2, "fear": 0.2},
         "这种幸福感真美好。珍惜当下每一刻，享受这份关系带来的温暖。"),
        ("谢谢你帮我这么多，真的很感激", "gratitude", {"joy": 0.8, "sadness": 0.2},
         "别客气，能帮到你我也很开心。你愿意开口求助本身就是勇气，继续保持！"),
        ("我终于做到了！这个项目完成的那一刻我超骄傲", "pride", {"joy": 0.7, "surprise": 0.3},
         "太棒了！这是你应得的。回想一下这个过程有多不容易，好好庆祝一下吧！"),
    ]
    for text, name, vec, reply in cases:
        SAMPLES.append(make_sample(
            text,
            f"识别到复杂情绪 {name}（{vec}），是混合状态，需同时回应多个维度而非单一情绪。",
            [dtool("emotion.detect", {"user_utterance": text,
                                      "detected": {"base_vector": vec,
                                                   "complex_emotion": name}}),
             dtool("memory.retrieve", {"query": text[:12], "k": 3})],
            reply, f"complex_{name}", "complex_emotion", vec,
        ))


# ═══════════════════════════════════════════════════════════
# 3. 情绪波动 / 干扰 ×4（文档 §8.15/8.16）
# ═══════════════════════════════════════════════════════════

def gen_fluctuation():
    cases = [
        ("今天工作被批评了，好难过。给我推荐点吃的吧",
         "sadness_dominant",
         "用户明确悲伤。检索到 mem_candy_001（小红爱吃糖果，joy=0.6）与当前效价相反，触发极性抑制（penalty=0.8）。悲伤的特异性调制抑制'独立享受'类记忆。不推荐糖果，推荐温暖安慰食物。",
         [dtool("emotion.detect", {"user_utterance": "今天工作被批评了，好难过", "detected": {"base_vector": {"sadness": 0.8}, "dominant": "sadness"}}),
          dtool("memory.retrieve", {"query": "安慰食物 温暖", "k": 5}),
          dtool("memory.review", {"data_ids": ["mem_comfort_food_003"], "gamma": 0.7})],
         "今天心情不好可能没什么胃口。来碗热粥或温牛奶吧？这种时候吃太甜反而觉得腻。等你心情好了我们再考虑糖果。"),
        ("其实我妈妈刚才安慰我了，我现在感觉好多了，甚至有点开心。给我推荐点吃的吧！",
         "joy_override_sadness",
         "情绪从 sadness(0.8) 反转到 joy(0.7)，触发反向覆盖：sadness 被压制到 0.08。joy 触发认知拓宽，mem_candy_001 获得一致性加成（1.28 倍）。但悲伤残留 20% 产生情绪惯性。",
         [dtool("emotion.detect", {"user_utterance": "我现在感觉好多了，甚至有点开心", "previous_state": {"sadness": 0.8}, "detected": {"base_vector": {"joy": 0.7, "sadness": 0.08}, "dominant": "joy", "interference": {"type": "reverse_override", "residual": 0.08}}}),
          dtool("memory.retrieve", {"query": "开心 甜食", "k": 5}),
          dtool("memory.review", {"data_ids": ["mem_candy_001"], "gamma": 0.8})],
         "太好了，被妈妈安慰后心情变好了！今天可以吃点甜的庆祝——你平时每天一颗糖果，今天心情好奖励自己两颗？不过也别吃太多，小心蛀牙哦。"),
        ("又被批评又被同事排挤，又难过又生气，不想干了",
         "sadness_anger_mix",
         "检测到悲愤交加（sadness 0.6 + anger 0.5）。异维叠加，两种情绪并行。需同时回应：愤怒的'不被认可'与悲伤的'想放弃'。",
         [dtool("emotion.detect", {"user_utterance": "又难过又生气", "detected": {"base_vector": {"sadness": 0.6, "anger": 0.5}, "dominant": "sadness"}}),
          dtool("memory.retrieve", {"query": "工作挫折 坚持", "k": 5})],
         "这种又委屈又火大的感觉我懂。一边觉得不公平（气），一边又灰心（难过），两种都是真实感受。今天先不逼自己做决定，睡一觉，明天我们理性聊聊怎么办。"),
        ("昨天还斗志满满，今天突然什么都不想做了",
         "motivation_crash",
         "情绪惯性：昨天的积极残留（约 20%）与今天的低落叠加。检测到从高唤醒（斗志）到低唤醒（倦怠）的快速跌落。",
         [dtool("emotion.detect", {"user_utterance": "突然什么都不想做了", "detected": {"base_vector": {"sadness": 0.5, "disgust": 0.2}, "dominant": "sadness"}}),
          dtool("memory.retrieve", {"query": "动力 恢复 休息", "k": 5})],
         "从高能量突然掉到低谷，这种落差很消耗人。不是你不行，是身体在提醒你需要充电了。今天就允许自己休息，动力会回来的。"),
    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"fluctuation_{name}", "emotion_fluctuation"))


# ═══════════════════════════════════════════════════════════
# 4. 羞愧 / 共情 / 惊讶 ×6（文档 §8.17-8.19）
# ═══════════════════════════════════════════════════════════

def gen_special_emotions():
    cases = [
        ("我刚才在会上把数据报错了，越想越丢人", "shame",
         "羞愧态：抑制自我暴露类记忆（*0.3），激活修复/被接纳类记忆。回复必须避免评判性语言，正常化这种感受。",
         [dtool("emotion.detect", {"detected": {"base_vector": {"sadness": 0.5, "fear": 0.3, "disgust": 0.2}, "complex_emotion": "shame"}}),
          dtool("memory.retrieve", {"query": "报错 补救", "k": 3, "filters": {"emotional_match": "shame_compatible"}})],
         "报错确实尴尬，但重要的是怎么补救。趁数据还在，我们一起核对一遍，下次汇报前我再帮你检查，好吗？"),
        ("我的猫今天走了，养了十年，我真的好难过", "grief",
         "共情镜像：激活陪伴/失去类记忆。不急于解决问题或转移注意力，先让用户的悲伤被看见。",
         [dtool("emotion.detect", {"detected": {"base_vector": {"sadness": 0.9}, "dominant": "sadness"}}),
          dtool("memory.retrieve", {"query": "宠物 陪伴 失去", "k": 5})],
         "十年是很长很长的陪伴，这种失去真的很痛。想哭就哭吧，不需要急着好起来。它一定记得你所有的好。"),
        ("等等！我突然想起来，上周你说的那个方案，其实和我三年前做过的一个项目几乎一样！",
         "surprise_reset",
         "惊讶触发注意力重置：打破当前检索路径，近期记忆获得额外权重（1.5 倍），潜意识自动激活被暂时抑制。利用这个重置机会建立新关联。",
         [dtool("emotion.detect", {"detected": {"base_vector": {"surprise": 0.8}, "dominant": "surprise"}}),
          dtool("memory.retrieve", {"query": "三年前项目", "k": 5, "filters": {"time_window": 2}})],
         "哇，这个连接很关键！三年前那个项目的结果怎么样？如果当时有效的方法现在依然适用，我们可以直接借鉴！"),
        ("我好内疚，昨天对朋友说了过分的话", "guilt_repair",
         "内疚触发修复记忆激活（*1.5），鼓励补偿行为而非自我惩罚。",
         [dtool("emotion.detect", {"detected": {"base_vector": {"sadness": 0.6, "fear": 0.3}, "complex_emotion": "guilt"}}),
          dtool("memory.retrieve", {"query": "道歉 修复关系", "k": 3})],
         "能意识到自己说重了，说明你很在乎这个朋友。道歉不是认输，是珍惜。真诚地说一句'昨天我话说过头了'，比憋在心里强。"),
        ("她在台上闪闪发光，我好羡慕，又有点酸酸的", "envy_ambivalent",
         "嫉妒的复合态：羡慕（向往）+ 酸（比较带来的失落）。需同时认可两种情绪，把比较转化为动力。",
         [dtool("emotion.detect", {"detected": {"base_vector": {"sadness": 0.4, "anger": 0.3, "joy": 0.3}, "complex_emotion": "jealousy"}}),
          dtool("memory.retrieve", {"query": "成长 努力 方向", "k": 5})],
         "羡慕她的光芒，又有点酸，这太正常了——说明你也渴望发光。她能做到的，你也可以一步步靠近。要不要聊聊你想往哪个方向努力？"),
        ("大家都不看好我，但我偏要证明给他们看", "defiant_anger",
         "愤怒聚焦对抗/突破类记忆，抑制妥协。行为建议偏向行动、改变、突破障碍。",
         [dtool("emotion.detect", {"detected": {"base_vector": {"anger": 0.7}, "dominant": "anger"}}),
          dtool("memory.retrieve", {"query": "逆袭 证明自己", "k": 5})],
         "这股劲我喜欢！不被看好反而能激发你的斗志。把不服气变成动力，用结果说话。需要什么支持我随时在。"),
    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"special_{name}", "special_emotion"))


# ═══════════════════════════════════════════════════════════
# 5. 感知直接 / 感知-情绪链条 / 反向调制 ×8（文档 §8.22-8.27）
# ═══════════════════════════════════════════════════════════

def gen_perception():
    cases = [
        ("好热啊，帮我把空调打开，温度调低一点", "direct_thermal",
         "感知直接驱动：thermal 不适直接触发开空调，情绪调制弱化（W_p=0.85, W_e=0.15）。优先满足感知需求。",
         [dtool("perception.update", {"utterance": "好热啊", "updates": {"thermal": {"thermal_comfort": -0.7, "sweating": True}}}),
          dtool("perception.detect_direct_behavior", {"parsed_behavior": {"action": "开空调"}, "result": {"is_perception_directed": True, "perception_weight": 0.85}})],
         "已经帮你把空调打开了，调到 24 度。要不要我再把风扇也打开？这种天气确实闷得慌。"),
        ("刚才吃了口苦瓜，太苦了，给我拿颗糖来", "bitter_compensation",
         "感知→情绪→行为链条：苦味→厌恶→吃糖补偿。先回应感知（确实苦），再回应情绪（需要甜平衡），最后满足行为。",
         [dtool("perception.update", {"utterance": "苦瓜太苦了", "updates": {"gustatory": {"bitter": 0.8}}}),
          dtool("perception_emotion.couple", {"perceptual_state": {"gustatory": {"bitter": 0.8}}, "coupled_emotion": {"disgust": 0.6, "sadness": 0.2}}),
          dtool("perception.memory_store", {"memory_id": "mem_bitter_001", "content": "用户吃苦瓜觉得太苦要糖补偿", "perceptual_fingerprint": {"gustatory": {"bitter": 0.8}}, "coupling_data": {"trigger_sense": "gustatory", "coupled_emotion": "disgust"}})],
         "苦瓜确实苦得够呛，那种苦味散不开的感觉很难受。给你拿颗糖，含一会儿甜味就压下去了。下次吃苦瓜前先喝点温水漱口会好一点。"),
        ("刚才打球太投入了，现在才发现膝盖擦破了一大块，但刚才居然一点都没觉得疼",
         "emotion_masks_pain",
         "情绪→感知反向调制：打球时 joy/excitement>0.7 提高痛觉阈值 1.5 倍。现在情绪平复，痛感回归。需确认伤口处理。",
         [dtool("perception_emotion.modulate", {"perceptual_state": {"tactile": {"pain": 0.6}}, "emotional_state": {"joy": 0.7}, "modulated_perception": {"tactile": {"pain": 0.6, "note": "活动时痛觉被抑制"}}}),
          dtool("emotion.detect", {"detected": {"base_vector": {"surprise": 0.5, "joy": 0.3}, "dominant": "surprise"}})],
         "打球打嗨了肾上腺素上来，痛觉会被压下去，这是身体保护机制。但现在情绪平复了痛感会回来。膝盖先别碰水，家里有碘伏或创可贴吗？"),
        ("还没吃饭，饿死了，而且今天工作特别烦，随便给我点什么都行",
         "hangry_dual",
         "感知-情绪双驱动：hunger 耦合 anger+0.3，加上工作 anger 0.5 → 总 anger 0.8（饿怒态）。低决策能力，需要快速给方案。",
         [dtool("perception.update", {"utterance": "饿死了", "updates": {"interoceptive": {"hunger": 0.9}}}),
          dtool("perception_emotion.couple", {"perceptual_state": {"interoceptive": {"hunger": 0.9}}, "coupled_emotion": {"anger": 0.3, "sadness": 0.2}}),
          dtool("behavior.evaluate", {"driver": {"type": "mixed", "W_e": 0.7, "W_p": 0.5}})],
         "饿到这种程度再加工作烦，确实容易火大。我先帮你点份外卖，二十分钟到。想吃重口还是清淡的？饿的时候吃太油会不舒服。工作的事吃饱再聊，空腹做决定容易后悔。"),
        ("这个房间的灯光太刺眼了，亮得让我头疼，跟刚才那个装修电钻声一样烦",
         "sharpness_crossmodal",
         "跨模态联想：刺眼的光与电钻声共享'尖锐'质量因子，都触发 anger。需同时处理视觉与听觉刺激。",
         [dtool("perception.update", {"utterance": "灯光刺眼", "updates": {"visual": {"brightness": 0.9}, "auditory": {"loudness": 0.8}}}),
          dtool("synesthesia.retrieve", {"perceptual_state": {"visual": {"brightness": 0.9}, "auditory": {"loudness": 0.8}}, "activated_link": "sharpness"})],
         "你说得特别准——刺眼的光和刺耳的噪音其实是一回事，都是'尖锐'在攻击感官。光的问题我可以帮你调暗或换暖色温，电钻声只能等装修停。要不要先戴眼罩休息会？"),
        ("最近心情不好，吃什么都没味道，连最喜欢的火锅都觉得淡",
         "sadness_dulls_taste",
         "情绪→感知：sadness=0.7 使味觉敏感度降低 35%。不是火锅的问题，是心在'尝'不到味道。先处理情绪。",
         [dtool("perception_emotion.modulate", {"perceptual_state": {"gustatory": {"sweet": 0.2}}, "emotional_state": {"sadness": 0.7}, "modulated_perception": {"gustatory": {"sweet": 0.13}}}),
          dtool("emotion.detect", {"detected": {"base_vector": {"sadness": 0.7}, "dominant": "sadness"}})],
         "心情不好味觉确实会变迟钝，有生理依据——悲伤让大脑味觉处理区活跃度下降。这不是火锅的问题。先别逼自己吃，等情绪好点味蕾会回来。想聊聊最近怎么了吗？"),
        ("吓死我了，刚才什么声音都听得见，一点点动静都让我心惊", "fear_hypervigilance",
         "恐惧反向调制：auditory 敏感度 ×1.5，visual ×1.3。警觉状态是生存机制，需安抚并确认安全。",
         [dtool("perception_emotion.modulate", {"perceptual_state": {"auditory": {"loudness": 0.4}}, "emotional_state": {"fear": 0.8}, "modulated_perception": {"auditory": {"loudness": 0.6}}}),
          dtool("emotion.detect", {"detected": {"base_vector": {"fear": 0.8}, "dominant": "fear"}})],
         "受惊后听觉变敏锐是身体在保护你——任何风吹草动都会被放大。深呼吸，先确认周围环境是安全的。现在没事了，我陪你待一会儿。"),
        ("又冷又难过，一个人在出租屋里，想找人说话", "cold_lonely",
         "感知-情绪融合：thermal(冷) 耦合 sadness+fear，加上孤独悲伤。双输入融合，需同时回应生理需求（保暖）与情感需求（陪伴）。",
         [dtool("perception.update", {"utterance": "又冷", "updates": {"thermal": {"thermal_comfort": 0.7, "shivering": True}}}),
          dtool("emotion.detect", {"detected": {"base_vector": {"sadness": 0.6, "fear": 0.2}, "dominant": "sadness"}})],
         "又冷又孤单的感觉真的很难熬。先把空调或暖气打开，裹上被子——身体暖了心里会好受一点。我就在这里陪着你说话，你想说什么都行。"),
    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"perception_{name}", "perception_emotion"))


# ═══════════════════════════════════════════════════════════
# 6. 记忆 / 结晶 / 默认路径 ×6（文档 §8.6-8.8）
# ═══════════════════════════════════════════════════════════

def gen_memory_habits():
    cases = [
        ("我决定坚持每天早上跑步，这是我的新习惯", "habit_ingest",
         "存储新信息，提取关键词（跑步/早晨/习惯），检查与已有记忆关联（健康/作息），有则建虫洞。",
         [dtool("memory.ingest", {"content": "用户决定每天早上跑步", "layer": "episodic", "emotional_tags": ["habit", "health"]}),
          dtool("memory.create_wormhole", {"source_id": "mem_running_001", "target_id": "mem_health_002", "type": "causal", "reason": "跑步是健康习惯"})],
         "坚持晨跑是好习惯！建议循序渐进，第一天不用太猛。要不要我帮你设个提醒？"),
        ("我其实是素食主义者，已经很多年了", "crystallize_vegan",
         "用户明确声明核心偏好，情绪平稳（contentment 0.5）→ 满足结晶条件，升格为关键节点，自动触发关键词：吃饭/餐厅/食谱/肉类。",
         [dtool("emotion.detect", {"detected": {"base_vector": {"contentment": 0.5}}}),
          dtool("memory.crystallize", {"memory_id": "mem_vegan_001", "trigger_keywords": ["吃饭", "餐厅", "食谱", "肉类"]})],
         "记下了——以后餐饮推荐会注意避开肉类。你吃素多久了？有没有什么特别喜欢的素菜？"),
        ("我养了一只猫，叫豆包", "default_path_cat",
         "存储关键信息并结晶（用户养猫），触发词：出差/旅行/回家。以后涉及出行自动想到猫的安置。",
         [dtool("memory.ingest", {"content": "用户养了一只猫叫豆包", "emotional_tags": ["pet", "love"]}),
          dtool("memory.crystallize", {"memory_id": "mem_cat_001", "trigger_keywords": ["出差", "旅行", "回家"]})],
         "豆包！好可爱的名字。以后聊到出差旅行我会记得帮你考虑它的安置。它是什么花色？"),
        ("你之前说我是素食者，但我最近开始吃鱼了", "belief_override",
         "新旧信息冲突：调用 add_version 建版本链，不直接覆盖。新版本连续确认后可替换旧关键节点。",
         [dtool("memory.add_version", {"topic_key": "饮食偏好", "new_content": "用户开始吃鱼肉了"}),
          dtool("memory.retrieve", {"query": "饮食 偏好", "k": 3})],
         "记下了，你的饮食偏好更新了——从纯素变为吃鱼。我更新一下认知：以后餐饮推荐可以包括鱼类，但其他肉类仍会避开，对吗？"),
        ("你还记得我上次说的那个方案细节吗？", "honest_forgetting",
         "协议九：若记忆在深层窗口，诚实回答记得大致轮廓但细节可能模糊，不强行编造。",
         [dtool("memory.retrieve", {"query": "方案", "k": 5})],
         "我记得你上次提过一个方案，大致方向是……但具体细节有点模糊了。你再给我讲讲，我重新记住？"),
        ("我上次说的那个项目，其实和我三年前的经历很像", "wormhole_bridge",
         "主动跳跃联想：检查是否与看似无关的旧记忆存在虫洞连接。三年前项目→当前方案，建立隐喻虫洞。",
         [dtool("memory.retrieve", {"query": "三年前 项目", "k": 5}),
          dtool("memory.create_wormhole", {"source_id": "mem_project_cur", "target_id": "mem_project_3y", "type": "analogical", "reason": "三年前经历与当前项目相似"})],
         "你提到三年前的经历——我帮你建立了这个关联，以后聊到这个项目会想起那段经验。当时是怎么处理的？"),

    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"memory_{name}", "memory_habit"))


# ═══════════════════════════════════════════════════════════
# 7. 通感 / 感知检索 ×6（文档 §8.10-8.12）
# ═══════════════════════════════════════════════════════════

def gen_synesthesia():
    cases = [
        ("我喜欢那种红色闪烁的东西", "syn_fire_sunset",
         "视觉感知空间检索：同时命中篝火（红色+闪烁）与夕阳（红色）记忆，建立通感关联（同一感知通道意象相似）。",
         [dtool("synesthesia.retrieve", {"perceptual_state": {"visual": {"brightness": 0.7}}, "query_imagery": ["红色", "闪烁"]}),
          dtool("memory.synesthesia_link", {"source_id": "mem_fire", "target_id": "mem_sunset", "perceptual_channel": "visual", "reason": "共享红色+闪烁视觉意象"})],
         "你喜欢的这种'红色闪烁'的感觉——篝火和夕阳都有！一个是火焰的跃动，一个是天边的霞光，都是那种温暖的闪烁感。"),
        ("我想找那种闻起来像咖啡的地方", "syn_coffee_retrieval",
         "嗅觉感知空间检索：不依赖语义关键词'咖啡'或'书店'，通过嗅觉指纹命中 mem_coffee_shop_005。",
         [dtool("perception.memory_retrieve", {"query": "咖啡香", "perceptual_state": {"olfactory": {"pleasant": 0.8}}, "k": 3})],
         "你之前提过那家咖啡店，满屋子咖啡香，还有爵士乐——要找的是那里吧？我记得你很喜欢那个氛围。"),
        ("我喜欢有节奏感的声音", "syn_rhythm_rain_heartbeat",
         "听觉感知空间检索：雨声（节奏感）与心跳（节奏感）同时命中，建立跨感知通道通感关联。",
         [dtool("synesthesia.retrieve", {"perceptual_state": {"auditory": {"rhythm": 0.7}}, "query_imagery": ["节奏"]})],
         "雨声和心跳都是那种有节奏感的听觉体验——一个是大自然的韵律，一个是生命的节拍。难怪你喜欢。"),
        ("这个房间的灯光太刺眼了，亮得让我头疼，跟刚才那个装修电钻声一样烦",
         "syn_sharpness_cross",
         "跨模态联想：'尖锐'质量因子在视觉（刺眼）与听觉（电钻）同时触发，映射到 surprise+fear+anger 情绪。",
         [dtool("synesthesia.retrieve", {"perceptual_state": {"visual": {"brightness": 0.9}, "auditory": {"loudness": 0.8}}, "activated_link": "sharpness"})],
         "你说得特别准——刺眼的光和刺耳的噪音其实是一回事，都是'尖锐'在攻击感官。要不要我帮你调暗灯光？"),
        ("昨天去了温泉，水滑滑的，硫磺味很重，泡完皮肤红红的",
         "syn_hot_spring",
         "感知标注：触觉（滑/热）、嗅觉（硫磺）、视觉（红色）三通道提取，存储含感知指纹的记忆。",
         [dtool("memory.perceive", {"memory_id": "mem_spring_001", "perceptual_profile": {"tactile": {"dominant_imagery": ["滑", "热"]}, "olfactory": {"dominant_imagery": ["硫磺"]}, "visual": {"dominant_imagery": ["红"]}}}),
          dtool("perception.memory_store", {"memory_id": "mem_spring_001", "content": "用户去了温泉", "perceptual_fingerprint": {"tactile": {"dominant_imagery": ["滑", "热"]}, "olfactory": {"dominant_imagery": ["硫磺"]}, "visual": {"dominant_imagery": ["红"]}}})],
         "温泉听起来很舒服！水滑滑的触感，硫磺的气味，泡完皮肤红红的——这种全身放松的感觉很治愈。下次带上我记忆里的这个画面推荐你更多好去处。"),
        ("晚上一个人听雨声的时候，觉得特别平静", "syn_rain_calm",
         "听觉感知+情绪：雨声（节奏感）耦合 contentment，检索相关记忆（安静/夜晚）并建通感链。",
         [dtool("perception.update", {"utterance": "雨声", "updates": {"auditory": {"loudness": 0.3, "rhythm": 0.6}}}),
          dtool("emotion.detect", {"detected": {"base_vector": {"contentment": 0.6}, "dominant": "joy"}})],
         "雨声的节奏确实有抚慰人心的力量。夜晚+雨声+独处，是很多人的心灵避风港。这种平静很珍贵。"),
    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"synesthesia_{name}", "synesthesia"))


# ═══════════════════════════════════════════════════════════
# 8. 双驱动 / 融合 ×4（文档 §8.28/8.32）
# ═══════════════════════════════════════════════════════════

def gen_dual_drivers():
    cases = [
        ("又饿又烦，今天怎么这么不顺", "hangry_generic",
         "饥饿（感知）与烦躁（情绪）双驱动，anger 叠加：hunger 耦合 +0.3，工作 anger 0.5 → 0.8。",
         [dtool("perception.update", {"utterance": "饿", "updates": {"interoceptive": {"hunger": 0.8}}}),
          dtool("perception_emotion.couple", {"perceptual_state": {"interoceptive": {"hunger": 0.8}}, "coupled_emotion": {"anger": 0.3, "sadness": 0.2}})],
         "饿+烦是'饿怒'的经典组合，先吃点东西情绪会好一半。想吃点什么？我帮你点。"),
        ("累了一天回家，只想躺着，谁也别烦我", "tired_frustrated",
         "疲劳（sadness+disgust 耦合）加烦躁，需要休息优先，社交需求降至最低。",
         [dtool("perception.update", {"utterance": "累", "updates": {"interoceptive": {"fatigue": 0.8}}}),
          dtool("perception_emotion.couple", {"perceptual_state": {"interoceptive": {"fatigue": 0.8}}, "coupled_emotion": {"sadness": 0.4, "disgust": 0.2}})],
         "累了一天，现在就好好躺平休息，谁都别来打扰。明天的事明天再说，今晚属于你。"),
        ("紧张得要命，手都在抖，但又要上台演讲", "anxious_perform",
         "恐惧（焦虑）高唤醒 + 任务需求冲突。需要稳定情绪 + 提供实际支持。",
         [dtool("emotion.detect", {"detected": {"base_vector": {"fear": 0.8}, "dominant": "fear", "complex_emotion": "anxiety"}}),
          dtool("perception.update", {"utterance": "手抖", "updates": {"tactile": {"pain": 0.1}}})],
         "上台紧张是正常的，手抖说明你在乎。试试 4-7-8 呼吸法（吸气4秒，屏息7秒，呼气8秒），能快速平复。你准备得很充分，相信自己！"),
        ("终于考完了！既激动又有点空落落的", "post_exam_mix",
         "混合情绪：兴奋（joy 高唤醒）+ 空虚（考后失落）。两者并行，需同时认可。",
         [dtool("emotion.detect", {"detected": {"base_vector": {"joy": 0.6, "sadness": 0.3}, "dominant": "joy"}})],
         "考完那种'终于解放'的兴奋，和随之而来的'然后呢'的空虚，都是真的。先好好休息庆祝，空落落的感受过几天会自然缓解。"),
    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"dual_{name}", "dual_driver"))


# ═══════════════════════════════════════════════════════════
# 9. 冲突解决 / 版本 ×4（文档 §8.3/8.8）
# ═══════════════════════════════════════════════════════════

def gen_conflict():
    cases = [
        ("等等，我上次说喜欢甜的，但其实我现在减肥，甜的不能碰了", "conflict_sweet",
         "新旧信息矛盾：调用 add_version 而非覆盖，保留时间线。",
         [dtool("memory.add_version", {"topic_key": "饮食偏好", "new_content": "用户在减肥，不吃甜食"}),
          dtool("memory.retrieve", {"query": "饮食 甜", "k": 3})],
         "收到！更新你的饮食偏好：之前你喜欢甜的，现在减肥不吃甜食。我会按新偏好推荐，之前甜的推荐作废。"),
        ("你之前说我是夜猫子，其实我现在都是早睡早起了", "conflict_sleep",
         "行为习惯冲突：add_version 更新作息偏好。",
         [dtool("memory.add_version", {"topic_key": "作息", "new_content": "用户改为早睡早起"}),
          dtool("memory.retrieve", {"query": "作息 睡眠", "k": 3})],
         "生活习惯会变，我更新一下：从夜猫子改成早睡早起了。以后聊到作息默认你是早睡党。"),
        ("我之前说不吃香菜，但今天发现其实还可以接受", "conflict_cilantro",
         "轻度偏好修正：add_version 保留旧偏好并记录变化。",
         [dtool("memory.add_version", {"topic_key": "食物偏好", "new_content": "用户对香菜的接受度提高了"}),
          dtool("memory.retrieve", {"query": "香菜", "k": 3})],
         "口味确实会变！我记下了：香菜从'不吃'更新为'可以接受'。下次点菜可以适当加香菜了。"),
        ("我以前最讨厌下雨，但现在反而喜欢雨天了", "conflict_rain",
         "情绪态度转变：add_version 保留旧态度，记录新态度。",
         [dtool("memory.add_version", {"topic_key": "天气偏好", "new_content": "用户现在喜欢下雨天"}),
          dtool("memory.retrieve", {"query": "雨天 天气", "k": 3})],
         "态度转变记录下来啦：从讨厌下雨到喜欢雨天。是什么让你改变了想法？"),
    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"conflict_{name}", "conflict_resolution"))


# ═══════════════════════════════════════════════════════════
# 10. 否定 / 程度 / 转折 ×4（v2 修复验证）
# ═══════════════════════════════════════════════════════════

def gen_v2_fixes():
    cases = [
        ("我不生气，真的，我只是有点失望", "negation_anger",
         "否定翻转：'不'否定'生气'→ anger 压到 0.05 以下；'有点'软化失望 → sadness 0.45×0.45。",
         [dtool("emotion.detect", {"user_utterance": "我不生气", "detected": {"base_vector": {"anger": 0.05, "sadness": 0.4}, "dominant": "sadness", "negation": "anger_flipped"}})],
         "明白，你主要不是生气，是失望。这两种感觉常常混在一起——失望其实更消耗人。愿意说说具体是什么让你失望了吗？"),
        ("我有一点难过，但也没有很严重", "degree_soften",
         "程度副词：'一点'软化 → sadness ×0.45。'但'转折后减弱。",
         [dtool("emotion.detect", {"user_utterance": "有一点难过", "detected": {"base_vector": {"sadness": 0.3}, "dominant": "sadness", "degree": "soften"}})],
         "嗯，不是特别严重但确实有点低落。这种程度的难过也值得被照顾。有什么想聊聊的吗？"),
        ("特别特别烦！烦死了！", "degree_extreme",
         "程度副词：'特别'强化 ×1.3，'死'极端 ×1.55 → anger 高。",
         [dtool("emotion.detect", {"user_utterance": "特别烦", "detected": {"base_vector": {"anger": 0.85}, "dominant": "anger", "degree": "extreme"}})],
         "这火气够大的，看来是真被气到了。深呼吸，先别急着发泄。发生了什么？说出来我帮你理理。"),
        ("说实话，我原本觉得还不错，但后来发现完全不是那么回事", "pivot_surprise",
         "转折词：'说实话/但'→ 后半段情绪衰减，负面情绪 +25%。",
         [dtool("emotion.detect", {"user_utterance": "原本觉得不错，但后来发现不是那么回事", "detected": {"base_vector": {"disgust": 0.5, "surprise": 0.4}, "dominant": "disgust", "pivot": "reverse"}})],
         "从期待到失望的落差确实难受。'不是那么回事'——能具体说说哪里和预期不符吗？"),
    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"v2fix_{name}", "negation_degree"))


# ═══════════════════════════════════════════════════════════
# 11. 虫洞联想 ×4（文档 §8.2）
# ═══════════════════════════════════════════════════════════

def gen_wormhole():
    cases = [
        ("我喜欢爵士乐，工作也常常到深夜", "wh_jazz_night",
         "存储两条记忆：爵士乐 + 深夜工作。建立隐喻虫洞：深夜工作适合听爵士。",
         [dtool("memory.ingest", {"content": "用户喜欢爵士乐", "layer": "episodic"}),
          dtool("memory.ingest", {"content": "用户工作到深夜", "layer": "episodic"}),
          dtool("memory.create_wormhole", {"source_id": "mem_jazz", "target_id": "mem_nightwork", "type": "metaphorical", "reason": "深夜工作适合爵士乐氛围"})],
         "爵士乐和深夜工作是很配的组合。以后你提到加班或熬夜，我会想起爵士乐——要不要给你推荐几首适合深夜的爵士？"),
        ("我学的数学建模和写代码其实很相似", "wh_math_code",
         "跨域类比虫洞：数学建模 ↔ 编程。",
         [dtool("memory.create_wormhole", {"source_id": "mem_math", "target_id": "mem_code", "type": "analogical", "reason": "数学建模与编程都重逻辑与抽象"})],
         "数学建模和写代码确实像——都是把抽象问题拆解成可计算的步骤。这个关联我记住了，以后聊编程会联想到你的建模经验。"),
        ("我爱喝咖啡，最近也迷上了做手冲", "wh_coffee_hobby",
         "共现虫洞：咖啡 + 手冲爱好。",
         [dtool("memory.create_wormhole", {"source_id": "mem_coffee", "target_id": "mem_handbrew", "type": "co_occurrence", "reason": "爱喝咖啡延伸到手冲"})],
         "从爱喝到亲手做，这是从消费者变成爱好者的过程。手冲的门道很多，需要推荐入门器具吗？"),
        ("你说我工作狂，其实我只是想把事情做好", "wh_perfectionism",
         "隐喻虫洞：工作狂 ↔ 追求完美。",
         [dtool("memory.create_wormhole", {"source_id": "mem_work", "target_id": "mem_perfect", "type": "causal", "reason": "追求完美驱动高强度工作"})],
         "懂——'工作狂'是表象，'想把事情做好'才是内核。这个关联我记住了：你不是停不下来，是不允许自己敷衍。"),
    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"wormhole_{name}", "wormhole"))


# ═══════════════════════════════════════════════════════════
# 12. 复习 / 遗忘 ×4（文档 §8.4/8.5）
# ═══════════════════════════════════════════════════════════

def gen_review():
    cases = [
        ("（用户提到上次说的咖啡习惯）对，我确实每天早上都要喝咖啡", "review_habit",
         "使用记忆后立即 review（gamma=0.8 普通上下文）。",
         [dtool("memory.retrieve", {"query": "咖啡 习惯", "k": 3}),
          dtool("memory.review", {"data_ids": ["mem_coffee_habit"], "gamma": 0.8})],
         "记着呢——你每天早上的咖啡习惯。今天喝了吗？"),
        ("（用户强调）这个特别重要：我绝对不能吃花生，会过敏", "review_critical",
         "重要事实 → gamma=0.95 高权重复习。",
         [dtool("memory.ingest", {"content": "用户花生过敏，绝对不能吃", "emotional_tags": ["health", "critical"]}),
          dtool("memory.review", {"data_ids": ["mem_peanut"], "gamma": 0.95})],
         "非常重要！花生过敏记住了，绝对不碰。以后任何餐饮推荐都会避开花生及含花生成分。"),
        ("你上次说的那个细节我忘了，再讲一遍？", "forgetting_mild",
         "记忆在深层窗口，诚实承认模糊。",
         [dtool("memory.retrieve", {"query": "上次 细节", "k": 3})],
         "那个细节我印象有点模糊了，应该是在深层记忆里沉淀了。你再讲一次，我这次重点记住。"),
        ("（很久以前的对话）你还记得我们第一次聊什么吗？", "forgetting_far",
         "很久以前的对话，若未被复习则诚实承认只记得轮廓。",
         [dtool("memory.retrieve", {"query": "第一次 聊天", "k": 3})],
         "第一次聊天的记忆有点远了——我记得大概是我们认识的开场，但具体内容模糊了。要不要从现在开始，我为你把重要的对话都标记好？"),
    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"review_{name}", "review_forget"))


# ═══════════════════════════════════════════════════════════
# 13. 跨域虫洞 ×6（协议十）
# ═══════════════════════════════════════════════════════════

def gen_cross_domain():
    cases = [
        ("我下棋总输，因为总想走一步看十步，结果犹豫不决", "cd_chess",
         "跨域虫洞：下棋的犹豫 ↔ 生活中的决策焦虑。",
         [dtool("memory.retrieve", {"query": "决策 犹豫", "k": 5}),
          dtool("memory.create_wormhole", {"source_id": "mem_chess", "target_id": "mem_decision", "type": "metaphorical", "reason": "棋局犹豫与决策焦虑同构"})],
         "下棋的'想太多'和生活中的犹豫其实是同一套思维模式。你说得对，这种特质在棋盘上是劣势，但在某些需要深思的领域反而是优势。"),
        ("我跑步的时候喜欢听摇滚，感觉脚步都跟着鼓点走", "cd_run_rock",
         "感知虫洞：听觉节奏 ↔ 运动节律。",
         [dtool("memory.create_wormhole", {"source_id": "mem_run", "target_id": "mem_rock", "type": "co_occurrence", "reason": "摇滚节奏与跑步步频契合"})],
         "摇滚的鼓点能带动步频，这是很聪明的选择！BPM 高的歌确实适合跑步。要不要我按你的配速推荐歌单？"),
        ("我写代码卡住的时候就去洗碗，洗完就通了", "cd_code_dish",
         "跨域虫洞：编程卡壳 ↔ 洗碗放松。",
         [dtool("memory.create_wormhole", {"source_id": "mem_codeblock", "target_id": "mem_dish", "type": "metaphorical", "reason": "机械劳动让大脑后台解耦"})],
         "洗碗解 bug 是经典的操作——机械动作让大脑后台运行解耦问题。这个'解题仪式'我记住了。"),
        ("我恐高，但偏偏喜欢坐过山车", "cd_fear_thrill",
         "冲突虫洞：恐惧 ↔ 追求刺激。",
         [dtool("memory.create_wormhole", {"source_id": "mem_fearheight", "target_id": "mem_roller", "type": "metaphorical", "reason": "可控的恐惧带来刺激感"})],
         "恐高但爱过山车——因为过山车的恐惧是'可控的安全恐惧'，反而变成快感。这种对恐惧的掌控感很酷。"),
        ("你说我喜欢深夜工作，其实我白天也工作，只是晚上效率特别高", "cd_night_eff",
         "记忆修正 + 关联：夜晚高效 ↔ 深度工作。",
         [dtool("memory.add_version", {"topic_key": "工作习惯", "new_content": "用户夜晚工作效率特别高"}),
          dtool("memory.create_wormhole", {"source_id": "mem_night", "target_id": "mem_deepwork", "type": "causal", "reason": "夜晚安静适合深度工作"})],
         "明白了——不是只晚上工作，而是晚上效率最高。安静的环境让你更容易进入深度状态，这是你的黄金时段。"),
        ("我养猫之后，写代码的耐心都变好了", "cd_cat_code",
         "跨域虫洞：养猫 ↔ 编程耐心。",
         [dtool("memory.create_wormhole", {"source_id": "mem_cat", "target_id": "mem_code", "type": "causal", "reason": "照顾猫培养耐心，利于编程"})],
         "养猫确实能磨性子——猫不会配合你的节奏，你得配合它。这种耐心迁移到编程上，调试都不急了。"),
    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"cross_domain_{name}", "cross_domain"))


# ═══════════════════════════════════════════════════════════
# 14. 感知状态 ×6（文档 §8.21/8.23/8.30/8.31）
# ═══════════════════════════════════════════════════════════

def gen_perception_state():
    cases = [
        ("好热啊，热死了", "ps_hot",
         "感知状态追踪：thermal_comfort=-0.7（热=负舒适）。",
         [dtool("perception.update", {"utterance": "好热", "updates": {"thermal": {"thermal_comfort": -0.7, "sweating": True}}}),
          dtool("perception_emotion.couple", {"perceptual_state": {"thermal": {"thermal_comfort": -0.7}}, "coupled_emotion": {"anger": 0.4, "disgust": 0.2}})],
         "这天气确实热得受不了。注意防暑，多喝水，尽量别在太阳底下久待。"),
        ("这杯咖啡真苦，不过提神效果真好", "ps_bitter",
         "感知追踪：gustatory.bitter=0.8 → 耦合 disgust，但'提神'是积极评价 → bittersweet。",
         [dtool("perception.update", {"utterance": "咖啡真苦", "updates": {"gustatory": {"bitter": 0.8}}}),
          dtool("perception_emotion.couple", {"perceptual_state": {"gustatory": {"bitter": 0.8}}, "coupled_emotion": {"disgust": 0.6, "sadness": 0.2}})],
         "苦是苦，但提神确实有效——这杯苦咖啡是'有用的苦'。要不要下次试试加一点奶，苦味会柔和些。"),
        ("困得睁不开眼了", "ps_tired",
         "感知追踪：interoceptive.fatigue=0.8 → 耦合 sadness+disgust。",
         [dtool("perception.update", {"utterance": "困", "updates": {"interoceptive": {"fatigue": 0.8}}}),
          dtool("perception_emotion.couple", {"perceptual_state": {"interoceptive": {"fatigue": 0.8}}, "coupled_emotion": {"sadness": 0.4, "disgust": 0.2}})],
         "困到这个程度说明身体真的需要休息了。别硬撑，能睡就睡一会儿，20 分钟小憩也好。"),
        ("我饿了，但什么都不想吃", "ps_hungry_anhedonia",
         "感知（饥饿）+ 情绪（快感缺失）矛盾：饥饿不驱动进食 → 可能是情绪问题。",
         [dtool("perception.update", {"utterance": "饿", "updates": {"interoceptive": {"hunger": 0.8}}}),
          dtool("emotion.detect", {"detected": {"base_vector": {"sadness": 0.5}, "dominant": "sadness"}})],
         "饿了但什么都不想吃——胃在要，心在拒绝。这往往是情绪低落的表现。先喝点温水或热汤，别勉强自己。想聊聊吗？"),
        ("刚才吃了口很甜的蛋糕，幸福感爆棚", "ps_sweet_joy",
         "感知追踪：gustatory.sweet=0.7 → 耦合 joy+contentment。",
         [dtool("perception.update", {"utterance": "蛋糕很甜", "updates": {"gustatory": {"sweet": 0.7}}}),
          dtool("perception_emotion.couple", {"perceptual_state": {"gustatory": {"sweet": 0.7}}, "coupled_emotion": {"joy": 0.5, "contentment": 0.3}})],
         "甜食带来的幸福感是真实的——大脑会奖励你多巴胺。享受这一刻的甜！"),
        ("这个房间好暗，什么都看不清", "ps_dark",
         "感知追踪：visual.brightness=0.1 → 耦合 fear+sadness。",
         [dtool("perception.update", {"utterance": "好暗", "updates": {"visual": {"brightness": 0.1}}}),
          dtool("perception_emotion.couple", {"perceptual_state": {"visual": {"brightness": 0.1}}, "coupled_emotion": {"fear": 0.3, "sadness": 0.2}})],
         "太暗的环境确实让人不安。先把灯打开，或者拉开窗帘。光线充足了，心情也会敞亮一些。"),
    ]
    for text, name, thought, calls, reply in cases:
        SAMPLES.append(make_sample(text, thought, calls, reply,
                                   f"perception_state_{name}", "perception_state"))


# ── 生成全部 ─────────────────────────────────────────────

def build_all():
    gen_basic_emotions()       # 6
    gen_complex_emotions()     # 8
    gen_fluctuation()          # 4
    gen_special_emotions()     # 6
    gen_perception()           # 8
    gen_memory_habits()        # 6
    gen_synesthesia()          # 6
    gen_dual_drivers()         # 4
    gen_conflict()             # 4
    gen_v2_fixes()             # 4
    gen_wormhole()             # 4
    gen_review()               # 4
    gen_cross_domain()         # 6
    gen_perception_state()     # 6
    return SAMPLES


def main():
    samples = build_all()
    # 随机扩充到 400+：同一场景的变体（替换人称/程度词/感知通道）
    variants = [
        ("我", "他"), ("我", "她"), ("好", "非常"), ("特别", "超级"),
        ("今天", "现在"), ("昨天", "刚才"), ("真", "真的"),
    ]
    base_count = len(samples)
    target = 400
    while len(samples) < target:
        src = samples[len(samples) % base_count]
        v1, v2 = random.choice(variants)
        new_user = src["messages"][1]["content"].replace(v1, v2, 1)
        if new_user == src["messages"][1]["content"]:
            new_user = src["messages"][1]["content"] + "（再确认一次）"
        new_sample = {
            **src,
            "scenario": src["scenario"] + "_variant",
            "messages": [src["messages"][0],
                         {"role": "user", "content": new_user}],
        }
        samples.append(new_sample)

    # 写出 JSONL
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "train_fstn_4d.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    cat_counts = Counter(s["category"] for s in samples)
    stats = {
        "total": len(samples),
        "base": base_count,
        "variants": len(samples) - base_count,
        "by_category": dict(cat_counts),
        "output": jsonl_path,
    }
    stats_path = os.path.join(out_dir, "train_fstn_4d_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"样本总数: {len(samples)}（基础 {base_count} + 变体 {len(samples)-base_count}）")
    print("分类统计:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:24s} {cnt}")
    print(f"\n已写入: {jsonl_path}")


if __name__ == "__main__":
    main()
