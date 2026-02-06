"""
⚡ 감마봇 코딩 에이전트 - nanobot-ai 직접 import 통합
pip install nanobot-ai 후 사용
"""
import asyncio, json, os, re, time, traceback
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

# nanobot 내부 모듈 직접 import
from nanobot.agent.loop import AgentLoop
from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.bus.queue import MessageBus
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.manager import SessionManager

# 설정
WORKSPACE = Path("nanobot_workspace")
WORKSPACE.mkdir(exist_ok=True)
(WORKSPACE / "output").mkdir(exist_ok=True)
(WORKSPACE / "memory").mkdir(exist_ok=True)
RALPH_MAX_RETRY = 3
MAX_ITERATIONS = 15


class SKHynixProvider(LLMProvider):
    """SK Hynix LLM API - nanobot LLMProvider 상속"""
    def __init__(self):
        super().__init__()
        self._token = self._url = self._model = None
        self._sync()

    def _sync(self):
        try:
            import pc_assistant as pa
            self._token = getattr(pa, 'API_TOKEN', None)
            self._url = getattr(pa, 'API_URL', None)
            self._model = getattr(pa, 'API_MODEL', None)
        except ImportError:
            tp = Path("token.txt")
            if tp.exists(): self._token = tp.read_text().strip()
            self._url = "http://dev.assistant.llm.skhynix.com/v1/chat/completions"
            self._model = "Qwen3-Coder-30B-A3B-Instruct"

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.3):
        import httpx
        self._sync()
        if not self._token:
            return LLMResponse(content="❌ API 토큰 없음", finish_reason="error")

        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        payload = {"model": model or self._model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            async with httpx.AsyncClient(timeout=300) as c:
                resp = await c.post(self._url, headers=headers, json=payload)
            if resp.status_code != 200:
                return LLMResponse(content=f"❌ API {resp.status_code}", finish_reason="error")

            result = resp.json()
            ch = result["choices"][0]
            msg = ch["message"]
            content = re.sub(r'<think>.*?</think>', '', msg.get("content","") or "", flags=re.DOTALL).strip()

            tc_list = []
            for tc in (msg.get("tool_calls") or []):
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except: args = {"raw": args}
                tc_list.append(ToolCallRequest(id=tc["id"], name=tc["function"]["name"], arguments=args))

            usage = {}
            if result.get("usage"):
                usage = {k: result["usage"].get(k,0) for k in ("prompt_tokens","completion_tokens","total_tokens")}
                try:
                    import pc_assistant as pa
                    tu = getattr(pa, 'TOKEN_USAGE', None)
                    if tu:
                        for k in usage: tu[k] += usage[k]
                        tu["call_count"] += 1
                except: pass

            return LLMResponse(content=content, tool_calls=tc_list,
                             finish_reason=ch.get("finish_reason","stop"), usage=usage)
        except Exception as e:
            logger.error(f"LLM 오류: {e}")
            return LLMResponse(content=f"❌ {e}", finish_reason="error")

    def get_default_model(self):
        self._sync()
        return self._model or "Qwen3-Coder-30B-A3B-Instruct"


class ValidateCodeTool(Tool):
    @property
    def name(self): return "validate_code"
    @property
    def description(self): return "코드 구문 검증 (Python/JS/HTML)"
    @property
    def parameters(self):
        return {"type":"object","properties":{
            "code":{"type":"string","description":"검증할 코드"},
            "language":{"type":"string","description":"언어"}}, "required":["code","language"]}

    async def execute(self, code="", language="python", **kw):
        errors, lang = [], language.lower()
        if len(code.strip()) < 5:
            return json.dumps({"valid":False,"errors":["코드가 너무 짧음"]})
        if lang == "python":
            try: compile(code, "<agent>", "exec")
            except SyntaxError as e: errors.append(f"Line {e.lineno}: {e.msg}")
        elif lang in ("javascript","typescript"):
            stack, pairs = [], {'{':'}','(':')','[':']'}
            for i,ch in enumerate(code):
                if ch in pairs: stack.append((ch,i))
                elif ch in pairs.values():
                    if not stack: errors.append(f"Pos {i}: '{ch}' 대응없음")
                    else:
                        o,_ = stack.pop()
                        if pairs[o]!=ch: errors.append(f"Pos {i}: '{pairs[o]}' 예상")
            for o,p in stack: errors.append(f"Pos {p}: '{o}' 미닫힘")
        elif lang == "html":
            void = {'br','hr','img','input','meta','link','area','base','col','embed','source','track','wbr'}
            opens = [t.lower() for t in re.findall(r'<([a-zA-Z]\w*)[^>]*(?<!/)>',code) if t.lower() not in void]
            closes = [t.lower() for t in re.findall(r'</([a-zA-Z]\w*)>',code)]
            if len(opens)!=len(closes): errors.append(f"태그 불일치: 열기{len(opens)} 닫기{len(closes)}")
        return json.dumps({"valid":not errors,"errors":errors,"language":lang}, ensure_ascii=False)


class SaveCodeTool(Tool):
    @property
    def name(self): return "save_code"
    @property
    def description(self): return "코드를 워크스페이스에 저장"
    @property
    def parameters(self):
        return {"type":"object","properties":{
            "filename":{"type":"string","description":"파일명"},
            "code":{"type":"string","description":"코드"},
            "language":{"type":"string","description":"언어"}}, "required":["filename","code"]}

    async def execute(self, filename="", code="", language="", **kw):
        out = WORKSPACE / "output"; out.mkdir(exist_ok=True)
        safe = "".join(c for c in filename if c.isalnum() or c in '.-_').strip()
        if not safe:
            ext = {"python":".py","javascript":".js","typescript":".ts","java":".java",
                   "html":".html","css":".css","sql":".sql","bash":".sh"}.get(language.lower(),".txt")
            safe = f"code_{int(time.time())}{ext}"
        fp = out / safe; fp.write_text(code, encoding="utf-8")
        return json.dumps({"saved":True,"path":str(fp),"filename":safe,"lines":len(code.split('\n'))}, ensure_ascii=False)


class NanobotManager:
    """nanobot 모듈 래핑 - FastAPI 통합"""
    def __init__(self):
        self.provider = None
        self.sessions = None
        self.memory = None
        self.tools = None
        self._init = False

    def initialize(self):
        if self._init: return
        logger.info("⚡ 감마봇 초기화...")
        self.provider = SKHynixProvider()
        self.sessions = SessionManager(WORKSPACE)
        self.memory = MemoryStore(WORKSPACE)
        self.tools = ToolRegistry()
        for t in [ReadFileTool(), WriteFileTool(), EditFileTool(), ListDirTool()]:
            self.tools.register(t)
        self.tools.register(ExecTool(working_dir=str(WORKSPACE), timeout=30, restrict_to_workspace=False))
        self.tools.register(ValidateCodeTool())
        self.tools.register(SaveCodeTool())
        self._init = True
        logger.info(f"⚡ 감마봇 준비 - 도구 {len(self.tools)}개")

    def _sys_prompt(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        ws = str(WORKSPACE.resolve())
        p = f"""# ⚡ 감마봇 코딩 에이전트
당신은 '감마봇' 코딩 전문 AI 에이전트입니다.
## 현재: {now}  |  워크스페이스: {ws}
## 규칙
1. 코드 생성 → validate_code 검증 → save_code 저장
2. 검증 실패 → 수정 후 재검증 (최대 {RALPH_MAX_RETRY}회)
3. 파일 작업은 도구 사용 (read_file, write_file, list_dir, exec)
4. 한국어 응답
## Ralph Loop: validate_code → (실패시 수정) → save_code"""
        mem = self.memory.get_memory_context() if self.memory else ""
        if mem: p += f"\n## 메모리\n{mem}"
        return p

    async def process(self, content, session_id="web:default"):
        if not self._init: self.initialize()
        t0 = time.time()
        steps, usage = [], {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}

        session = self.sessions.get_or_create(session_id)
        msgs = [{"role":"system","content":self._sys_prompt()}]
        msgs.extend(session.get_history(max_messages=20))
        msgs.append({"role":"user","content":content})

        steps.append({"type":"context","icon":"📋","detail":f"히스토리 {len(session.get_history())}개","duration":round(time.time()-t0,2)})

        it, final = 0, None
        while it < MAX_ITERATIONS:
            it += 1; ts = time.time()
            resp = await self.provider.chat(messages=msgs, tools=self.tools.get_definitions())
            if resp.usage:
                for k in usage: usage[k] += resp.usage.get(k,0)

            if resp.has_tool_calls:
                tc_dicts = [{"id":tc.id,"type":"function",
                    "function":{"name":tc.name,"arguments":json.dumps(tc.arguments,ensure_ascii=False)}}
                    for tc in resp.tool_calls]
                msgs.append({"role":"assistant","content":resp.content or "","tool_calls":tc_dicts})

                for tc in resp.tool_calls:
                    tt = time.time()
                    result = await self.tools.execute(tc.name, tc.arguments)
                    msgs.append({"role":"tool","tool_call_id":tc.id,"name":tc.name,"content":result})
                    icons = {"validate_code":"✅","save_code":"💾","exec":"⚡","read_file":"📖",
                             "write_file":"✏️","edit_file":"✏️","list_dir":"📁"}
                    steps.append({"type":"tool","icon":icons.get(tc.name,"🔧"),
                                  "detail":f"{tc.name}: {result[:120]}","duration":round(time.time()-tt,2)})
                steps.append({"type":"llm","icon":"🧠","detail":f"#{it} 도구 {len(resp.tool_calls)}개",
                              "duration":round(time.time()-ts,2)})
            else:
                final = resp.content
                steps.append({"type":"llm","icon":"🧠","detail":f"#{it} 최종","duration":round(time.time()-ts,2)})
                break

        if not final: final = "처리 완료."
        session.add_message("user", content)
        session.add_message("assistant", final)
        self.sessions.save(session)

        tt = round(time.time()-t0,2)
        steps.append({"type":"complete","icon":"🏁","detail":f"{it}회 {tt}초","duration":tt})
        return {"response":final,"steps":steps,"session_id":session_id,"usage":usage,"iterations":it,"total_time":tt}


# 싱글톤
nanobot_manager = NanobotManager()

# FastAPI 라우터
agent_router = APIRouter(prefix="/assistant", tags=["nanobot"])

class ChatReq(BaseModel):
    message: str
    session_id: str = "web:default"

@agent_router.post("/api/agent/chat")
async def agent_chat(req: ChatReq):
    try: return await nanobot_manager.process(req.message, req.session_id)
    except Exception as e:
        logger.error(traceback.format_exc()); raise HTTPException(500, str(e))

@agent_router.get("/api/agent/status")
async def agent_status():
    try:
        m = nanobot_manager
        if not m._init:
            m.initialize()
        # provider 이름 가져오기
        provider_name = "?"
        try:
            import pc_assistant as pa
            env = getattr(pa, 'CURRENT_ENV', '?')
            provider_name = getattr(pa, 'ENV_CONFIG', {}).get(env, {}).get("name", env)
        except: pass
        return {
            "success": True,
            "initialized": m._init,
            "tools": m.tools.tool_names if m.tools else [],
            "model": m.provider.get_default_model() if m.provider else None,
            "provider": provider_name,
            "max_iterations": MAX_ITERATIONS,
            "nanobot_version": _ver(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "initialized": False,
        }

@agent_router.get("/api/agent/files")
async def agent_files():
    d = WORKSPACE/"output"
    if not d.exists(): return {"files":[]}
    return {"files":[{"name":f.name,"size":f.stat().st_size,
            "modified":datetime.fromtimestamp(f.stat().st_mtime).isoformat(),"ext":f.suffix}
            for f in sorted(d.iterdir(),key=lambda x:x.stat().st_mtime,reverse=True) if f.is_file()]}

@agent_router.get("/api/agent/files/{fn}")
async def agent_read(fn:str):
    fp = WORKSPACE/"output"/fn
    if not fp.exists(): raise HTTPException(404)
    return {"filename":fn,"content":fp.read_text(encoding="utf-8"),"size":fp.stat().st_size}

@agent_router.delete("/api/agent/files/{fn}")
async def agent_del(fn:str):
    fp = WORKSPACE/"output"/fn
    if fp.exists(): fp.unlink(); return {"deleted":True}
    raise HTTPException(404)

@agent_router.get("/api/agent/sessions")
async def agent_sessions():
    if not nanobot_manager._init: nanobot_manager.initialize()
    return {"sessions":nanobot_manager.sessions.list_sessions()}

@agent_router.delete("/api/agent/sessions/{sid}")
async def agent_del_session(sid:str):
    if not nanobot_manager._init: nanobot_manager.initialize()
    return {"deleted":nanobot_manager.sessions.delete(sid)}

@agent_router.get("/api/agent/memory")
async def agent_mem():
    if not nanobot_manager._init: nanobot_manager.initialize()
    m = nanobot_manager.memory
    return {"long_term":m.read_long_term(),"today":m.read_today(),
            "files":[f.name for f in m.list_memory_files()[:10]]}

def _ver():
    try:
        import importlib.metadata; return importlib.metadata.version("nanobot-ai")
    except: return "?"

def get_agent_router(): return agent_router

logger.info("⚡ coding_agent.py 로드 (nanobot-ai 직접 import)")