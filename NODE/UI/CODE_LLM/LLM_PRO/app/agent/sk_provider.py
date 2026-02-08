#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKHynixProvider - nanobot LLMProvider 상속
API + GGUF 폴백 지원
"""

import re
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# nanobot 사용 가능 여부
try:
    from nanobot.providers.base import LLMProvider as NanobotLLMProvider, LLMResponse, ToolCallRequest
    HAS_NANOBOT = True
except ImportError:
    HAS_NANOBOT = False
    # 더미 클래스
    class NanobotLLMProvider:
        def __init__(self): pass
    class LLMResponse:
        def __init__(self, **kw):
            self.content = kw.get("content", "")
            self.tool_calls = kw.get("tool_calls", [])
            self.finish_reason = kw.get("finish_reason", "stop")
            self.usage = kw.get("usage", {})
            self.has_tool_calls = bool(self.tool_calls)
    class ToolCallRequest:
        def __init__(self, **kw):
            self.id = kw.get("id", "")
            self.name = kw.get("name", "")
            self.arguments = kw.get("arguments", {})


class SKHynixProvider(NanobotLLMProvider):
    """SK Hynix LLM API - nanobot LLMProvider 상속"""

    def __init__(self, config, llm_provider_instance):
        if HAS_NANOBOT:
            super().__init__()
        self.config = config
        self.llm_provider = llm_provider_instance

    def _call_gguf(self, messages, max_tokens=1024, temperature=0.7):
        """GGUF 로컬 모델로 대화"""
        if self.llm_provider.local_llm is None:
            return None

        # messages → Qwen3 ChatML 프롬프트
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n/no_think\n")
        full_prompt = "\n".join(prompt_parts)

        try:
            output = self.llm_provider.local_llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=1.1,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False
            )
            content = output["choices"][0]["text"].strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return LLMResponse(content=content, tool_calls=[], finish_reason="stop", usage={})
        except Exception as e:
            logger.error(f"GGUF 오류: {e}")
            return None

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.3):
        """LLM 채팅 (API 우선, GGUF 폴백)"""
        # API 토큰 없으면 GGUF 폴백
        if not self.config.api_token:
            gguf_resp = self._call_gguf(messages, max_tokens, temperature)
            if gguf_resp:
                return gguf_resp
            return LLMResponse(content="API 토큰도 없고 GGUF 모델도 없음", finish_reason="error")

        import httpx
        headers = {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model or self.config.api_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            async with httpx.AsyncClient(timeout=300) as c:
                resp = await c.post(self.config.api_url, headers=headers, json=payload)
            if resp.status_code != 200:
                return LLMResponse(content=f"API {resp.status_code}", finish_reason="error")

            result = resp.json()
            ch = result["choices"][0]
            msg = ch["message"]
            content = re.sub(
                r'<think>.*?</think>', '', msg.get("content", "") or "",
                flags=re.DOTALL
            ).strip()

            tc_list = []
            for tc in (msg.get("tool_calls") or []):
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {"raw": args}
                tc_list.append(ToolCallRequest(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=args
                ))

            usage = {}
            if result.get("usage"):
                usage = {
                    k: result["usage"].get(k, 0)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                }

            return LLMResponse(
                content=content, tool_calls=tc_list,
                finish_reason=ch.get("finish_reason", "stop"), usage=usage
            )
        except Exception as e:
            logger.error(f"LLM 오류: {e}")
            return LLMResponse(content=f"오류: {e}", finish_reason="error")

    def get_default_model(self):
        if self.config.api_token:
            return self.config.api_model or "Qwen3-Coder-30B-A3B-Instruct"
        return self.config.AVAILABLE_GGUF_MODELS.get(
            self.config.current_gguf_model, {}
        ).get("name", "GGUF-Local")
