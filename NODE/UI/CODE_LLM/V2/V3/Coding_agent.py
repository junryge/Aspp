#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coding_agent.py
나나봇 모드 - nanobot 스타일 코딩 에이전트
pc_assistant.py에 import하여 라우터로 연결

사용법 (pc_assistant.py 맨 아래에 추가):
    from coding_agent import agent_router
    app.include_router(agent_router)

nanobot 아키텍처 차용:
  - Agent Loop: LLM ↔ Tool 실행 반복 사이클
  - Skills: 코드생성/검증/저장 모듈
  - Memory: 세션 컨텍스트 유지
  - Ralph Loop: 자기검증 + 자동재시도
"""

import os
import re
import json
import time
import uuid
import subprocess
import tempfile
import datetime
import traceback
from typing import Optional, List, Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel
import logging

logger = logging.getLogger("NanoBot")

# ============================================================
#  pc_assistant에서 가져올 것들 (lazy import)
# ============================================================
_pc = None  # pc_assistant 모듈 참조

def _get_pc():
    """pc_assistant 모듈을 lazy import"""
    global _pc
    if _pc is None:
        import pc_assistant as _pc
    return _pc


def call_llm_bridge(prompt: str, system_prompt: str = "", max_tokens: int = 4096) -> dict:
    """pc_assistant의 call_llm을 브릿지로 호출"""
    pc = _get_pc()
    return pc.call_llm(prompt, system_prompt, max_tokens)


# ============================================================
#  CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_WORKSPACE = os.path.join(BASE_DIR, "nanobot_workspace")
AGENT_OUTPUT = os.path.join(AGENT_WORKSPACE, "output")
AGENT_MEMORY_FILE = os.path.join(AGENT_WORKSPACE, "memory.json")
os.makedirs(AGENT_OUTPUT, exist_ok=True)

# Agent Loop 설정
MAX_LOOP_STEPS = 5       # 최대 에이전트 루프 반복
RALPH_MAX_RETRY = 3      # Ralph Loop 최대 재시도
SUPPORTED_LANGS = {
    "python":     {"ext": ".py",   "icon": "🐍"},
    "javascript": {"ext": ".js",   "icon": "📜"},
    "typescript": {"ext": ".ts",   "icon": "📘"},
    "java":       {"ext": ".java", "icon": "☕"},
    "csharp":     {"ext": ".cs",   "icon": "🔷"},
    "html":       {"ext": ".html", "icon": "🌐"},
    "css":        {"ext": ".css",  "icon": "🎨"},
    "sql":        {"ext": ".sql",  "icon": "🗄️"},
    "bash":       {"ext": ".sh",   "icon": "🐚"},
}

# ============================================================
#  MEMORY - nanobot/agent/memory.py 패턴
# ============================================================
class AgentMemory:
    """세션별 대화 메모리 (nanobot memory.py 경량 버전)"""

    def __init__(self):
        self.sessions: Dict[str, List[dict]] = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(AGENT_MEMORY_FILE):
                with open(AGENT_MEMORY_FILE, 'r', encoding='utf-8') as f:
                    self.sessions = json.load(f)
        except:
            self.sessions = {}

    def _save(self):
        try:
            with open(AGENT_MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.sessions, f, ensure_ascii=False, indent=2)
        except:
            pass

    def add(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "time": datetime.datetime.now().isoformat()
        })
        # 최근 20턴만 유지
        self.sessions[session_id] = self.sessions[session_id][-40:]
        self._save()

    def get_context(self, session_id: str, max_turns: int = 5) -> str:
        msgs = self.sessions.get(session_id, [])
        recent = msgs[-(max_turns * 2):]
        lines = []
        for m in recent:
            role = "사용자" if m["role"] == "user" else "에이전트"
            lines.append(f"[{role}] {m['content'][:300]}")
        return "\n".join(lines)

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)
        self._save()

    def list_sessions(self) -> List[dict]:
        result = []
        for sid, msgs in self.sessions.items():
            if msgs:
                result.append({
                    "session_id": sid,
                    "messages": len(msgs),
                    "last_time": msgs[-1].get("time", ""),
                    "preview": msgs[-1].get("content", "")[:80]
                })
        return sorted(result, key=lambda x: x["last_time"], reverse=True)


memory = AgentMemory()


# ============================================================
#  SKILLS - nanobot/skills/ 패턴 (코딩 특화)
# ============================================================

def skill_detect_language(code: str, hint: str = "") -> str:
    """코드 또는 힌트에서 언어 감지"""
    hint_lower = hint.lower()
    for lang in SUPPORTED_LANGS:
        if lang in hint_lower:
            return lang

    patterns = {
        "python": [r"^(import |from |def |class |print\()", r"\.py$"],
        "javascript": [r"(const |let |var |function |=>|console\.log)", r"\.js$"],
        "typescript": [r"(interface |type |:\s*(string|number|boolean))", r"\.ts$"],
        "java": [r"(public class |System\.out|void main)", r"\.java$"],
        "csharp": [r"(namespace |using System|Console\.Write)", r"\.cs$"],
        "html": [r"(<html|<div|<body|<!DOCTYPE)", r"\.html$"],
        "css": [r"(\{[^}]*:[^}]*\}|@media|\.[\w-]+\s*\{)", r"\.css$"],
        "sql": [r"(SELECT |INSERT |CREATE TABLE|ALTER )", r"\.sql$"],
        "bash": [r"(#!/bin/|echo |export |if \[)", r"\.sh$"],
    }
    for lang, pats in patterns.items():
        for pat in pats:
            if re.search(pat, code, re.MULTILINE | re.IGNORECASE):
                return lang
    return "python"


def skill_validate_syntax(code: str, language: str) -> dict:
    """코드 문법 검증 (Ralph Loop의 검증 단계)"""
    result = {"valid": True, "errors": [], "warnings": []}

    if language == "python":
        try:
            compile(code, "<agent>", "exec")
        except SyntaxError as e:
            result["valid"] = False
            result["errors"].append(f"Line {e.lineno}: {e.msg}")

    elif language in ("javascript", "typescript"):
        # 기본 괄호 매칭 검사
        opens = code.count('{') + code.count('(') + code.count('[')
        closes = code.count('}') + code.count(')') + code.count(']')
        if opens != closes:
            result["valid"] = False
            result["errors"].append(f"괄호 불일치: 열림={opens} 닫힘={closes}")

    elif language == "java":
        if "class " not in code:
            result["warnings"].append("class 키워드가 없습니다")
        opens = code.count('{')
        closes = code.count('}')
        if opens != closes:
            result["valid"] = False
            result["errors"].append(f"중괄호 불일치: {{{opens} }}{closes}")

    elif language == "html":
        open_tags = len(re.findall(r'<(\w+)[^/>]*>', code))
        close_tags = len(re.findall(r'</(\w+)>', code))
        self_close = len(re.findall(r'<\w+[^>]*/>', code))
        if open_tags > close_tags + self_close + 3:
            result["warnings"].append(f"닫히지 않은 태그가 있을 수 있습니다 (열림:{open_tags} 닫힘:{close_tags})")

    elif language == "sql":
        upper = code.upper()
        if not any(kw in upper for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]):
            result["warnings"].append("SQL 키워드가 감지되지 않았습니다")

    # 공통 검사
    if len(code.strip()) < 10:
        result["valid"] = False
        result["errors"].append("코드가 너무 짧습니다")

    if code.count('```') % 2 != 0:
        result["warnings"].append("마크다운 코드블록이 제대로 닫히지 않았습니다")

    return result


def skill_save_code(code: str, language: str, filename: str = "") -> dict:
    """생성된 코드를 파일로 저장"""
    lang_info = SUPPORTED_LANGS.get(language, {"ext": ".txt", "icon": "📄"})

    if not filename:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_{ts}{lang_info['ext']}"

    if not filename.endswith(lang_info['ext']):
        filename += lang_info['ext']

    filepath = os.path.join(AGENT_OUTPUT, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        return {
            "success": True,
            "filepath": filepath,
            "filename": filename,
            "language": language,
            "size": len(code)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def skill_list_files() -> List[dict]:
    """워크스페이스 파일 목록"""
    files = []
    for f in sorted(os.listdir(AGENT_OUTPUT)):
        fp = os.path.join(AGENT_OUTPUT, f)
        if os.path.isfile(fp):
            files.append({
                "filename": f,
                "size": os.path.getsize(fp),
                "modified": datetime.datetime.fromtimestamp(
                    os.path.getmtime(fp)
                ).strftime("%Y-%m-%d %H:%M"),
            })
    return files


def skill_read_file(filename: str) -> dict:
    """워크스페이스 파일 읽기"""
    filepath = os.path.join(AGENT_OUTPUT, filename)
    if not os.path.exists(filepath):
        return {"success": False, "error": f"파일 없음: {filename}"}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"success": True, "content": content, "filename": filename}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
#  AGENT LOOP - nanobot/agent/loop.py 패턴
# ============================================================

AGENT_SYSTEM_PROMPT = """당신은 '나나봇'이라는 코딩 전문 AI 에이전트입니다.

★★★ 핵심 규칙 ★★★

[1] 코드 생성 요청을 받으면 반드시 아래 JSON 형식으로 응답하세요:
{
  "action": "generate_code",
  "language": "python",
  "filename": "파일명.py",
  "description": "코드 설명",
  "code": "실제 코드 내용"
}

[2] 코드를 읽거나 수정해야 할 때:
{"action": "read_file", "filename": "파일명.py"}
{"action": "edit_code", "filename": "파일명.py", "description": "수정 내용", "code": "수정된 전체 코드"}

[3] 파일 목록 확인:
{"action": "list_files"}

[4] 일반 대화/설명/질문일 때:
{"action": "reply", "message": "답변 내용"}

★★★ 코드 작성 규칙 ★★★
- 실행 가능한 완전한 코드를 작성하세요
- 주석은 한국어로 작성
- 타입 힌트 사용
- 에러 핸들링 포함
- import문 누락 없이

★★★ 항상 JSON만 출력하세요. 다른 텍스트 없이 JSON 하나만! ★★★"""


def extract_agent_action(text: str) -> Optional[dict]:
    """LLM 응답에서 에이전트 액션 JSON 추출"""
    # 1. 직접 JSON 파싱
    text = text.strip()
    # <think> 태그 제거
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 코드블록 안의 JSON
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass

    # 중괄호로 시작하는 JSON
    m = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]+?"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass

    # 전체를 JSON으로 시도 (코드가 포함된 큰 JSON)
    brace_start = text.find('{')
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i+1])
                    except:
                        break

    return None


def agent_loop(user_message: str, session_id: str = "default") -> dict:
    """
    나나봇 Agent Loop (nanobot loop.py 패턴)

    흐름:
    1. 사용자 메시지 수신
    2. LLM에게 컨텍스트 + 메시지 전달
    3. LLM이 액션 JSON 반환
    4. 액션 실행 (generate_code → validate → save)
    5. 결과를 다시 LLM에 전달 (필요시 반복)
    6. 최종 응답 반환

    Ralph Loop:
    - generate_code 시 자동 문법 검증
    - 실패하면 에러 피드백과 함께 재생성 (최대 3회)
    """
    steps = []  # 에이전트 작업 로그
    memory.add(session_id, "user", user_message)

    # 컨텍스트 구성 (nanobot context.py 패턴)
    context = memory.get_context(session_id, max_turns=4)
    full_prompt = f"""[이전 대화 컨텍스트]
{context}

[현재 요청]
{user_message}

JSON 액션으로 응답하세요."""

    for step in range(MAX_LOOP_STEPS):
        steps.append({"step": step + 1, "type": "llm_call", "status": "진행중"})

        result = call_llm_bridge(full_prompt, AGENT_SYSTEM_PROMPT, max_tokens=6000)
        if not result["success"]:
            steps[-1]["status"] = "실패"
            steps[-1]["error"] = result.get("error", "LLM 호출 실패")
            return _build_response(
                f"❌ LLM 오류: {result.get('error', '?')}",
                steps, session_id
            )

        raw_text = result["content"]
        steps[-1]["raw"] = raw_text[:300]

        action = extract_agent_action(raw_text)

        # 액션 추출 실패 → 텍스트 그대로 반환
        if not action:
            steps[-1]["status"] = "텍스트 응답"
            # LLM이 그냥 텍스트로 답한 경우
            clean = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            return _build_response(clean, steps, session_id)

        action_type = action.get("action", "")
        steps[-1]["action"] = action_type
        steps[-1]["status"] = "실행중"

        # ─── reply: 일반 답변 ───
        if action_type == "reply":
            msg = action.get("message", raw_text)
            steps[-1]["status"] = "완료"
            return _build_response(msg, steps, session_id)

        # ─── list_files: 파일 목록 ───
        elif action_type == "list_files":
            files = skill_list_files()
            steps[-1]["status"] = "완료"
            steps[-1]["files"] = len(files)
            if not files:
                return _build_response(
                    "📂 워크스페이스가 비어있습니다. 코드 생성을 요청해보세요!",
                    steps, session_id
                )
            lines = ["📂 **나나봇 워크스페이스 파일 목록**\n"]
            for f in files:
                lines.append(f"- `{f['filename']}` ({f['size']}B, {f['modified']})")
            return _build_response("\n".join(lines), steps, session_id)

        # ─── read_file: 파일 읽기 ───
        elif action_type == "read_file":
            fname = action.get("filename", "")
            read_result = skill_read_file(fname)
            steps[-1]["status"] = "완료"
            if read_result["success"]:
                content = read_result["content"]
                lang = skill_detect_language(content, fname)
                return _build_response(
                    f"📄 **{fname}**\n```{lang}\n{content}\n```",
                    steps, session_id
                )
            else:
                return _build_response(
                    f"❌ {read_result['error']}",
                    steps, session_id
                )

        # ─── generate_code / edit_code: 코드 생성 + Ralph Loop ───
        elif action_type in ("generate_code", "edit_code"):
            code = action.get("code", "")
            language = action.get("language", "python")
            filename = action.get("filename", "")
            description = action.get("description", "")

            if not code:
                steps[-1]["status"] = "코드 없음"
                return _build_response("❌ LLM이 코드를 생성하지 못했습니다.", steps, session_id)

            # ─── Ralph Loop: 검증 → 재시도 ───
            for attempt in range(RALPH_MAX_RETRY):
                ralph_step = {
                    "step": f"ralph_{attempt+1}",
                    "type": "validate",
                    "language": language,
                    "attempt": attempt + 1
                }

                validation = skill_validate_syntax(code, language)
                ralph_step["valid"] = validation["valid"]
                ralph_step["errors"] = validation["errors"]
                ralph_step["warnings"] = validation["warnings"]

                if validation["valid"]:
                    ralph_step["status"] = "✅ 통과"
                    steps.append(ralph_step)
                    break
                else:
                    ralph_step["status"] = "❌ 실패 → 재시도"
                    steps.append(ralph_step)

                    if attempt < RALPH_MAX_RETRY - 1:
                        # 에러 피드백으로 재생성 요청
                        fix_prompt = f"""이전에 생성한 코드에 문법 오류가 있습니다:

[오류 목록]
{json.dumps(validation["errors"], ensure_ascii=False)}

[원본 코드]
```{language}
{code}
```

오류를 수정한 전체 코드를 다시 JSON으로 응답하세요:
{{"action": "generate_code", "language": "{language}", "filename": "{filename}", "description": "{description}", "code": "수정된 코드"}}"""

                        fix_result = call_llm_bridge(fix_prompt, AGENT_SYSTEM_PROMPT, max_tokens=6000)
                        if fix_result["success"]:
                            fix_action = extract_agent_action(fix_result["content"])
                            if fix_action and fix_action.get("code"):
                                code = fix_action["code"]
                                continue
                    # 최종 실패
                    ralph_step["status"] = "⚠️ 최대 재시도 도달"

            # 코드 저장
            save_result = skill_save_code(code, language, filename)
            save_step = {
                "step": "save",
                "type": "save_file",
                "filename": save_result.get("filename", "?"),
                "status": "✅ 저장" if save_result["success"] else "❌ 저장 실패"
            }
            steps.append(save_step)

            # 최종 응답 구성
            lang_info = SUPPORTED_LANGS.get(language, {"icon": "📄"})
            icon = lang_info["icon"]

            resp_lines = []
            resp_lines.append(f"{icon} **코드 {'수정' if action_type == 'edit_code' else '생성'} 완료**")
            if description:
                resp_lines.append(f"> {description}\n")

            # 검증 결과 요약
            if validation["valid"]:
                resp_lines.append("✅ **문법 검증 통과**")
            else:
                resp_lines.append(f"⚠️ **검증 경고** ({len(validation['errors'])}개 오류)")
                for err in validation["errors"]:
                    resp_lines.append(f"  - {err}")

            if validation["warnings"]:
                for w in validation["warnings"]:
                    resp_lines.append(f"  💡 {w}")

            # 저장 결과
            if save_result["success"]:
                resp_lines.append(f"\n💾 `{save_result['filename']}` 저장됨 ({save_result['size']}B)")
            else:
                resp_lines.append(f"\n❌ 저장 실패: {save_result.get('error')}")

            # 코드 표시
            resp_lines.append(f"\n```{language}\n{code}\n```")

            return _build_response("\n".join(resp_lines), steps, session_id)

        # ─── 알 수 없는 액션 → 텍스트로 반환 ───
        else:
            steps[-1]["status"] = "알 수 없는 액션"
            clean = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            return _build_response(clean, steps, session_id)

    # 루프 한도 초과
    return _build_response("⚠️ 에이전트 루프 한도에 도달했습니다.", steps, session_id)


def _build_response(message: str, steps: list, session_id: str) -> dict:
    """최종 응답 포맷 구성"""
    memory.add(session_id, "assistant", message[:500])
    return {
        "success": True,
        "response": message,
        "steps": steps,
        "session_id": session_id,
        "mode": "nanobot"
    }


# ============================================================
#  ROUTER - pc_assistant에 붙일 API 엔드포인트
# ============================================================
agent_router = APIRouter(prefix="/assistant", tags=["nanobot-agent"])


class AgentChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@agent_router.post("/api/agent/chat")
async def agent_chat(request: AgentChatRequest):
    """나나봇 코딩 에이전트 채팅"""
    try:
        result = agent_loop(request.message, request.session_id)
        return result
    except Exception as e:
        logger.error(f"❌ Agent error: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "response": f"❌ 에이전트 오류: {e}",
            "steps": [],
            "mode": "nanobot"
        }


@agent_router.get("/api/agent/status")
async def agent_status():
    """나나봇 상태"""
    files = skill_list_files()
    sessions = memory.list_sessions()
    return {
        "success": True,
        "mode": "nanobot",
        "workspace_files": len(files),
        "sessions": len(sessions),
        "supported_languages": list(SUPPORTED_LANGS.keys()),
        "max_loop_steps": MAX_LOOP_STEPS,
        "ralph_max_retry": RALPH_MAX_RETRY
    }


@agent_router.get("/api/agent/files")
async def agent_files():
    """워크스페이스 파일 목록"""
    return {"success": True, "files": skill_list_files()}


@agent_router.get("/api/agent/files/{filename}")
async def agent_read_file(filename: str):
    """워크스페이스 파일 내용 읽기"""
    result = skill_read_file(filename)
    return result


@agent_router.delete("/api/agent/files/{filename}")
async def agent_delete_file(filename: str):
    """워크스페이스 파일 삭제"""
    filepath = os.path.join(AGENT_OUTPUT, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return {"success": True, "message": f"'{filename}' 삭제됨"}
    return {"success": False, "error": "파일 없음"}


@agent_router.get("/api/agent/sessions")
async def agent_sessions():
    """세션 목록"""
    return {"success": True, "sessions": memory.list_sessions()}


@agent_router.delete("/api/agent/sessions/{session_id}")
async def agent_clear_session(session_id: str):
    """세션 초기화"""
    memory.clear(session_id)
    return {"success": True, "message": f"세션 '{session_id}' 초기화됨"}


logger.info("🐈 나나봇 코딩 에이전트 모듈 로드 완료")