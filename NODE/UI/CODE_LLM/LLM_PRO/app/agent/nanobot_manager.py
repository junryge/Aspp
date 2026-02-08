#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
마기(MAGI) 에이전트 매니저 - nanobot 통합
nanobot-ai 미설치 시에도 기본 기능 제공
"""

import re
import json
import time
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

WORKSPACE = Path("nanobot_workspace")
KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge"
DOMAIN_KNOWLEDGE_FILE = Path(__file__).parent.parent.parent / "domain_knowledge.txt"
RALPH_MAX_RETRY = 3
MAX_ITERATIONS = 15

# nanobot 사용 가능 여부
try:
    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.context import ContextBuilder
    from nanobot.agent.memory import MemoryStore
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
    from nanobot.agent.tools.shell import ExecTool
    from nanobot.session.manager import SessionManager
    HAS_NANOBOT = True
except ImportError:
    HAS_NANOBOT = False
    logger.warning("nanobot-ai 미설치 - 에이전트 기능 제한")

from .tools import ValidateCodeTool, SaveCodeTool


class NanobotManager:
    """nanobot 기반 코딩 에이전트"""

    def __init__(self, config, llm_provider):
        self.config = config
        self.llm_provider = llm_provider
        self.provider = None
        self.sessions = None
        self.memory = None
        self.tools = None
        self._init = False

    @property
    def available(self):
        return HAS_NANOBOT

    def initialize(self):
        """에이전트 초기화"""
        if self._init:
            return
        if not HAS_NANOBOT:
            logger.warning("nanobot-ai 미설치 - 에이전트 초기화 건너뜀")
            return

        WORKSPACE.mkdir(exist_ok=True)
        (WORKSPACE / "output").mkdir(exist_ok=True)
        (WORKSPACE / "memory").mkdir(exist_ok=True)

        from .sk_provider import SKHynixProvider
        self.provider = SKHynixProvider(self.config, self.llm_provider)
        self.sessions = SessionManager(WORKSPACE)
        self.memory = MemoryStore(WORKSPACE)
        self.tools = ToolRegistry()

        for t in [ReadFileTool(), WriteFileTool(), EditFileTool(), ListDirTool()]:
            self.tools.register(t)
        self.tools.register(ExecTool(working_dir=str(WORKSPACE), timeout=30, restrict_to_workspace=False))
        self.tools.register(ValidateCodeTool())
        self.tools.register(SaveCodeTool())

        self._init = True
        logger.info(f"마기(MAGI) 에이전트 초기화 완료 - 도구 {len(self.tools)}개")

    def _build_system_prompt(self):
        """시스템 프롬프트 생성"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        ws = str(WORKSPACE.resolve())

        prompt = f"""# 마기(MAGI) 코딩 에이전트
당신은 '마기(MAGI)' 코딩 전문 AI 에이전트입니다.
## 현재: {now}  |  워크스페이스: {ws}
## 규칙
1. 코드 생성 -> validate_code 검증 -> save_code 저장
2. 검증 실패 -> 수정 후 재검증 (최대 {RALPH_MAX_RETRY}회)
3. 파일 작업은 도구 사용 (read_file, write_file, list_dir, exec)
4. 한국어 응답"""

        if self.memory:
            mem = self.memory.get_memory_context()
            if mem:
                prompt += f"\n## 메모리\n{mem}"

        return prompt

    async def process(self, content: str, session_id: str = "desktop:default") -> dict:
        """메시지 처리"""
        if not self._init:
            self.initialize()

        if not HAS_NANOBOT:
            # nanobot 없으면 기본 LLM 호출로 처리
            result = self.llm_provider.call(content, "당신은 코딩 전문 AI 에이전트입니다. 한국어로 답변하세요.")
            return {
                "success": result.get("success", False),
                "response": result.get("content", result.get("error", "오류")),
                "steps": [],
                "session_id": session_id,
                "usage": {},
                "iterations": 1,
                "total_time": 0
            }

        t0 = time.time()
        steps = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        session = self.sessions.get_or_create(session_id)

        # API 모드: 도구 사용 가능
        msgs = [{"role": "system", "content": self._build_system_prompt()}]
        msgs.extend(session.get_history(max_messages=20))
        msgs.append({"role": "user", "content": content})

        it, final = 0, None
        while it < MAX_ITERATIONS:
            it += 1
            ts = time.time()
            resp = await self.provider.chat(messages=msgs, tools=self.tools.get_definitions())

            if resp.usage:
                for k in usage:
                    usage[k] += resp.usage.get(k, 0)

            if resp.has_tool_calls:
                tc_dicts = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name,
                                  "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                    for tc in resp.tool_calls
                ]
                msgs.append({"role": "assistant", "content": resp.content or "", "tool_calls": tc_dicts})

                for tc in resp.tool_calls:
                    tt = time.time()
                    result = await self.tools.execute(tc.name, tc.arguments)
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result})
                    steps.append({
                        "type": "tool",
                        "detail": f"{tc.name}: {result[:120]}",
                        "duration": round(time.time() - tt, 2)
                    })
            else:
                final = resp.content
                break

        if not final:
            final = "처리 완료."

        session.add_message("user", content)
        session.add_message("assistant", final)
        self.sessions.save(session)

        tt = round(time.time() - t0, 2)
        return {
            "success": True,
            "response": final,
            "steps": steps,
            "session_id": session_id,
            "usage": usage,
            "iterations": it,
            "total_time": tt
        }
