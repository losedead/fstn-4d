# -*- coding: utf-8 -*-
"""
fstn5/llm_client.py — LLM 客户端（OpenAI 兼容 API）

用于策略自动生成闭环。零第三方依赖（urllib 标准库）。
支持：DeepSeek / OpenAI / 任何 OpenAI 兼容端点；ollama 本地也兼容。

配置优先级：
  1. 构造参数（base_url, api_key, model）
  2. 环境变量 FSTN_LLM_BASE / FSTN_LLM_KEY / FSTN_LLM_MODEL
  3. 默认 DeepSeek 端点（key 必须由环境变量提供）
"""

import json
import os
import urllib.request
import urllib.error


class LLMClient:
    def __init__(self, base_url: str = None, api_key: str = None,
                 model: str = None, timeout: int = 60):
        self.base_url = (base_url or os.environ.get(
            "FSTN_LLM_BASE", "https://api.deepseek.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("FSTN_LLM_KEY", "")
        self.model = model or os.environ.get("FSTN_LLM_MODEL", "deepseek-chat")
        self.timeout = timeout

    def chat(self, messages: list, temperature: float = 0.7,
             max_tokens: int = 1024) -> str:
        """调用 chat/completions。返回 assistant 文本。

        兼容两种后端：
        - OpenAI 兼容端点（DeepSeek/OpenAI）：/v1/chat/completions
        - Ollama 本地：/api/generate（对 qwen3.5:9b 这类推理模型，
          chat 接口 content 为空，必须走 generate 的 response 字段）
        """
        # Ollama 后端走原生 /api/generate（推理模型最可靠）
        if "11434" in self.base_url or "localhost" in self.base_url:
            return self._ollama_generate(messages, temperature, max_tokens)
        return self._chat_completions(messages, temperature, max_tokens)

    def _ollama_generate(self, messages: list, temperature: float,
                         max_tokens: int) -> str:
        """Ollama 原生 generate：拼接 messages → prompt，取 response 字段。"""
        prompt = ""
        for m in messages:
            role = m["role"]
            if role == "system":
                prompt += f"指令: {m['content']}\n"
            elif role == "user":
                prompt += f"{m['content']}\n"
        payload = {
            "model": self.model,
            "prompt": prompt.strip(),
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        req = urllib.request.Request(
            f"{self.base_url.replace('/v1', '')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            content = data.get("response") or ""
            if not content.strip():
                # 兜底：思考字段（极少用，qwen 的 response 一般有内容）
                content = data.get("thinking") or data.get("reasoning") or ""
            return content
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Ollama HTTP {e.code}: {body}")

    def _chat_completions(self, messages: list, temperature: float,
                          max_tokens: int) -> str:
        """标准 OpenAI 兼容 chat/completions。"""
        if not self.api_key:
            raise RuntimeError(
                "未配置 LLM API key：设 FSTN_LLM_KEY 环境变量，"
                "或传 api_key 参数。")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            # 推理模型（如 qwen3.5:9b）content 为空、思考在 reasoning 字段
            if not content.strip():
                content = msg.get("reasoning") or msg.get(
                    "reasoning_content") or ""
            return content
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"LLM HTTP {e.code}: {body}")

    def generate_strategies(self, domain: str, context: dict,
                            count: int = 2) -> list:
        """根据上下文生成 count 个新策略。

        context 建议包含：
          - strategies: 现有策略摘要
          - experience_stats: 各策略的 reward/trials 统计
          - failures: 失败经验（供 LLM 针对性改进）
        返回 [{"name": str, "description": str, "rationale": str}, ...]
        """
        sys_prompt = (
            "你是资深系统架构师，为Agent策略库设计全新策略。"
            "严格输出JSON数组，不要任何其他文字、不要markdown、不要解释。\n"
            "每个元素格式: {\"name\":\"策略名\",\"description\":\"具体怎么做\","
            "\"rationale\":\"为什么有效(50字内)\"}\n"
            "要求: 策略必须有实际新思路，不能是现有策略的改名。")
        user_prompt = f"""领域: {domain}

现有策略池:
{json.dumps(context.get('strategies', []), ensure_ascii=False, indent=1)}

经验统计（reward ∈ [-1,1]，越高越好）:
{json.dumps(context.get('experience_stats', {}), ensure_ascii=False, indent=1)}

失败经验（低 reward 的尝试，可据此改进）:
{json.dumps(context.get('failures', []), ensure_ascii=False, indent=1)}

生成 {count} 个新策略。只输出JSON数组。"""
        resp = self.chat([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.8, max_tokens=2048)
        return self._parse_json(resp)

    @staticmethod
    def _parse_json(resp: str) -> list:
        """容忍 markdown 代码块包裹 / 前导文字 / 尾随解释 / 坏结构。

        返回能解析的 dict 列表；坏条目跳过而非整体失败。
        """
        text = resp.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            text = text[start:end + 1]
        # 宽容解析：逐对象提取（容忍键缺失/顺序错乱/单引号）
        out = []
        decoder = json.JSONDecoder()
        idx = 0
        n = len(text)
        while idx < n:
            ch = text[idx]
            if ch == "{":
                try:
                    obj, nxt = decoder.raw_decode(text, idx)
                    if isinstance(obj, dict) and obj.get("name"):
                        out.append({
                            "name": str(obj["name"]),
                            "description": str(obj.get("description", "")),
                            "rationale": str(obj.get("rationale", "")),
                        })
                    idx = nxt
                    continue
                except (json.JSONDecodeError, ValueError):
                    idx += 1
                    continue
            idx += 1
        return out
