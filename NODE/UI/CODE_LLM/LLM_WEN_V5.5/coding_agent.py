"""
⚡ 마기(main1_First) 코딩 에이전트 - nanobot-ai 직접 import 통합
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
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"  # 지식베이스 폴더 (MD)
DOMAIN_KNOWLEDGE_FILE = Path(__file__).parent / "domain_knowledge.txt"  # 도메인 지식 (TXT)
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

    def _get_local_llm(self):
        """pc_assistant에서 로드된 GGUF 모델 가져오기"""
        try:
            import pc_assistant as pa
            return getattr(pa, 'LOCAL_LLM', None)
        except:
            return None

    def _call_gguf(self, messages, max_tokens=1024, temperature=0.7):
        """GGUF 로컬 모델로 대화 (tool_calls 미지원, 텍스트 응답만)"""
        local_llm = self._get_local_llm()
        if not local_llm:
            return None

        # 현재 모델 확인
        is_gemma = False
        try:
            import pc_assistant as pa
            is_gemma = getattr(pa, 'CURRENT_LOCAL_MODEL', '') == 'gemma3-12b'
        except:
            pass

        # messages → 프롬프트 변환 (모델별)
        prompt_parts = []
        if is_gemma:
            # Gemma 3: system을 user에 합침
            sys_content = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    sys_content = content + "\n\n"
                elif role == "user":
                    prompt_parts.append(f"<start_of_turn>user\n{sys_content}{content}<end_of_turn>")
                    sys_content = ""
                elif role == "assistant":
                    prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
            prompt_parts.append("<start_of_turn>model\n")
            stop_tokens = ["<end_of_turn>", "<start_of_turn>"]
        else:
            # Qwen3 (기본): ChatML
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n/no_think\n")
            stop_tokens = ["<|im_end|>", "<|im_start|>"]
        full_prompt = "\n".join(prompt_parts)

        try:
            output = local_llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=1.1,
                stop=stop_tokens,
                echo=False
            )
            content = output["choices"][0]["text"].strip()
            # think 블록 혹시 남아있으면 제거
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return LLMResponse(content=content, tool_calls=[], finish_reason="stop", usage={})
        except Exception as e:
            logger.error(f"GGUF 오류: {e}")
            return None

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.3):
        self._sync()

        # API 토큰 없으면 GGUF 폴백
        if not self._token:
            gguf_resp = self._call_gguf(messages, max_tokens, temperature)
            if gguf_resp:
                return gguf_resp
            return LLMResponse(content="❌ API 토큰도 없고 GGUF 모델도 없음", finish_reason="error")

        import httpx
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
        if self._token:
            return self._model or "Qwen3-Coder-30B-A3B-Instruct"
        # GGUF 모드일 때 실제 모델명 반환
        try:
            import pc_assistant as pa
            model_key = getattr(pa, 'CURRENT_LOCAL_MODEL', '?')
            models = getattr(pa, 'AVAILABLE_MODELS', {})
            return models.get(model_key, {}).get("name", model_key)
        except:
            return "GGUF-Local"


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
        logger.info("⚡ 마기(main1_First) 초기화...")
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
        logger.info(f"⚡ 마기(main1_First) 준비 - 도구 {len(self.tools)}개")

    def _sys_prompt(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        ws = str(WORKSPACE.resolve())
        kb = str(KNOWLEDGE_DIR.resolve())
        # 지식베이스 파일 목록 (MD + TXT)
        kb_files = ""
        if KNOWLEDGE_DIR.exists():
            files = [f.name for f in KNOWLEDGE_DIR.iterdir() if f.is_file() and f.suffix.lower() in ('.md', '.txt')]
            if files:
                kb_files = "\n".join(f"  - {kb}/{f}" for f in files)
        # 도메인 지식 TXT 로드
        domain_knowledge = ""
        if DOMAIN_KNOWLEDGE_FILE.exists():
            try:
                raw = DOMAIN_KNOWLEDGE_FILE.read_text(encoding="utf-8")
                # 주석(#으로 시작) 제거한 실제 내용만
                lines = [l for l in raw.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
                if lines:
                    domain_knowledge = "\n".join(lines)
            except:
                pass
        p = f"""# ⚡ 마기(main1_First) 코딩 에이전트
당신은 '마기(main1_First)' 코딩 전문 AI 에이전트입니다.
## 현재: {now}  |  워크스페이스: {ws}
## 지식베이스: {kb}
{f'사용 가능한 문서:{chr(10)}{kb_files}' if kb_files else '문서 없음'}
- 도메인 지식 파일: {DOMAIN_KNOWLEDGE_FILE.resolve()}
{f'## 도메인 지식 (약어/용어 참고용){chr(10)}{domain_knowledge}' if domain_knowledge else ''}
## 규칙
1. 코드 생성 → validate_code 검증 → save_code 저장
2. 검증 실패 → 수정 후 재검증 (최대 {RALPH_MAX_RETRY}회)
3. 파일 작업은 도구 사용 (read_file, write_file, list_dir, exec)
4. 지식/도메인 질문 → **반드시** 먼저 지식베이스 문서를 read_file로 읽고, 문서 내용만으로 답변. 도메인 지식만으로 상세 답변 금지.
5. 한국어 응답
6. 문서에 없는 내용을 지어내지 마세요. 없으면 "문서에 해당 내용이 없습니다"라고 답하세요.
## Ralph Loop: validate_code → (실패시 수정) → save_code"""
        mem = self.memory.get_memory_context() if self.memory else ""
        if mem: p += f"\n## 메모리\n{mem}"
        return p

    def _is_gguf_mode(self):
        """현재 GGUF 로컬 모드인지 확인"""
        self.provider._sync()
        return not self.provider._token

    def _search_knowledge(self, query, max_chars=1500):
        """사용자 질문에서 키워드 추출 → 지식베이스 파일에서 관련 내용 검색"""
        if not KNOWLEDGE_DIR.exists():
            return ""
        # 키워드 추출 (한글/영문 단어, 2글자 이상)
        keywords = [w for w in re.findall(r'[가-힣a-zA-Z0-9_]{2,}', query.lower())]
        stop_words = {'어떻게','무엇','알려줘','설명','해줘','뭐야','있나요','대해','관련','마기(main1_First)','코딩','에이전트','부탁'}
        keywords = [k for k in keywords if k not in stop_words]
        if not keywords:
            return ""

        results = []
        for fp in KNOWLEDGE_DIR.iterdir():
            if not fp.is_file() or fp.suffix.lower() not in ('.md', '.txt'):
                continue
            try:
                text = fp.read_text(encoding="utf-8")
            except:
                continue
            # 파일명도 매칭 대상
            fname_lower = fp.stem.lower()
            score = sum(1 for k in keywords if k in fname_lower) * 3  # 파일명 매칭 가중치
            score += sum(1 for k in keywords if k in text.lower())
            if score > 0:
                results.append((score, fp.name, text))

        if not results:
            return ""

        # 점수 높은 순 정렬
        results.sort(key=lambda x: x[0], reverse=True)

        # 상위 파일에서 관련 줄 추출 (max_chars 제한)
        output = []
        total = 0
        for score, fname, text in results[:2]:  # 최대 2개 파일
            lines = text.split("\n")
            matched = []
            for i, line in enumerate(lines):
                if any(k in line.lower() for k in keywords):
                    # 매칭된 줄 ± 2줄 컨텍스트
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    for j in range(start, end):
                        if lines[j] not in matched:
                            matched.append(lines[j])
            if not matched:
                # 키워드가 파일명에만 매칭 → 앞부분 가져오기
                matched = lines[:15]
            chunk = "\n".join(matched)
            if total + len(chunk) > max_chars:
                chunk = chunk[:max_chars - total]
            if chunk:
                output.append(f"[{fname}]\n{chunk}")
                total += len(chunk)
            if total >= max_chars:
                break

        return "\n\n".join(output)

    def _sys_prompt_light(self, query=""):
        """GGUF 모드용 시스템 프롬프트 (도구 사용 가능)"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        kb_dir = str(KNOWLEDGE_DIR.resolve()) if KNOWLEDGE_DIR.exists() else ""
        # 도메인 지식 로드
        domain = ""
        if DOMAIN_KNOWLEDGE_FILE.exists():
            try:
                raw = DOMAIN_KNOWLEDGE_FILE.read_text(encoding="utf-8")
                lines = [l for l in raw.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
                if lines:
                    domain = "\n".join(lines[:60])
            except:
                pass
        # 질문 관련 지식베이스 검색
        kb_context = self._search_knowledge(query) if query else ""

        return f"""당신은 '마기(main1_First)' 지식기반 AI 비서입니다. 현재: {now}.
지식베이스 경로: {kb_dir}
{f'## 도메인 지식{chr(10)}{domain}' if domain else ''}
{f'## 참고 지식{chr(10)}{kb_context}' if kb_context else ''}
## 사용 가능한 도구
도구를 사용하려면 반드시 아래 JSON 형식으로만 응답하세요:
{{"tool":"도구명","args":{{"파라미터":"값"}}}}

- list_dir: 폴더 내 파일 목록. args: {{"path":"경로"}}
- read_file: 파일 읽기. args: {{"path":"파일경로"}}

## 규칙
- 도구가 필요하면 JSON만 출력 (다른 텍스트 금지)
- 도구가 필요없으면 바로 텍스트로 답변
- 코드를 절대 작성하지 마라
- 참고 지식을 정리해서 텍스트로만 답변
- "문서를 참조하세요" 같은 떠넘기기 금지. 직접 답변
- 한국어로 간결하게 답변
- 답변 마지막에 반드시 출처를 표시:
  도메인 지식에서 답변했으면 → 📌 출처: domain_knowledge.txt
  지식베이스 MD에서 답변했으면 → 📌 출처: [파일명.md]
  도구로 파일을 읽었으면 → 📌 출처: [읽은 파일명]
  본인 지식으로 답변했으면 → 📌 출처: AI 일반 지식"""

    def _parse_tool_call(self, text):
        """GGUF 응답에서 도구 호출 JSON 파싱"""
        if not text:
            return None
        text = text.strip()
        # "tool" 키워드가 없으면 도구 호출 아님
        if '"tool"' not in text and "'tool'" not in text:
            return None
        try:
            # ```json ... ``` 블록
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if m:
                return json.loads(m.group(1))
            # 텍스트에서 첫 번째 { ... } 추출
            start = text.find('{')
            if start >= 0:
                brace = 0
                for i in range(start, len(text)):
                    if text[i] == '{': brace += 1
                    elif text[i] == '}': brace -= 1
                    if brace == 0:
                        parsed = json.loads(text[start:i+1])
                        if "tool" in parsed:
                            return parsed
                        break
        except Exception as e:
            logger.warning(f"도구 파싱 실패: {e}, 원문: {text[:200]}")
        return None

    async def process(self, content, session_id="web:default"):
        if not self._init: self.initialize()
        t0 = time.time()
        steps, usage = [], {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}

        gguf_mode = self._is_gguf_mode()

        session = self.sessions.get_or_create(session_id)

        if gguf_mode:
            # GGUF: 도구 사용 가능 (최대 3회 반복)
            msgs = [{"role":"system","content":self._sys_prompt_light(query=content)}]
            msgs.extend(session.get_history(max_messages=4))
            msgs.append({"role":"user","content":content})
            steps.append({"type":"context","icon":"📋","detail":"GGUF 모드","duration":round(time.time()-t0,2)})

            final = None
            for gguf_it in range(3):  # 최대 3회 도구 호출
                ts = time.time()
                resp = await self.provider.chat(messages=msgs, tools=None, max_tokens=512)
                dur = round(time.time()-ts,2)
                raw = resp.content or ""
                steps.append({"type":"llm","icon":"🧠","detail":f"GGUF #{gguf_it+1} {dur}초","duration":dur})

                # 도구 호출 감지
                logger.info(f"🔍 GGUF 원문: [{raw[:300]}]")
                tool_call = self._parse_tool_call(raw)
                logger.info(f"🔍 파싱 결과: {tool_call}")
                if tool_call and "tool" in tool_call:
                    tool_name = tool_call["tool"]
                    tool_args = tool_call.get("args", {})
                    logger.info(f"🔧 GGUF 도구 호출: {tool_name}({tool_args})")
                    try:
                        result = await self.tools.execute(tool_name, tool_args)
                        # 결과가 너무 길면 자르기
                        if len(result) > 2000:
                            result = result[:2000] + "\n... (이하 생략)"
                        steps.append({"type":"tool","icon":"🔧","detail":f"{tool_name}: {result[:100]}","duration":round(time.time()-ts,2)})
                        # 도구 결과를 대화에 추가하고 다시 LLM 호출
                        msgs.append({"role":"assistant","content":raw})
                        msgs.append({"role":"user","content":f"[도구 결과: {tool_name}]\n{result}\n\n위 결과를 바탕으로 사용자 질문에 한국어로 답변하세요."})
                    except Exception as e:
                        logger.error(f"도구 실행 실패: {e}")
                        final = f"도구 실행 오류: {e}"
                        break
                else:
                    # 도구 호출 아님 → 최종 답변
                    final = raw
                    break

            if not final:
                final = "처리 완료."
            # 도구 JSON은 히스토리에 저장하지 않음
            if not self._parse_tool_call(final):
                session.add_message("user", content)
                session.add_message("assistant", final)
            self.sessions.save(session)
            tt = round(time.time()-t0,2)
            steps.append({"type":"complete","icon":"🏁","detail":f"GGUF {tt}초","duration":tt})
            return {"success":True,"response":final,"steps":steps,"session_id":session_id,"usage":usage,"iterations":1,"total_time":tt}

        # API 모드: 기존 로직 (도구 사용 가능)
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
        return {"success":True,"response":final,"steps":steps,"session_id":session_id,"usage":usage,"iterations":it,"total_time":tt}


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
        # 실제 서버 모드/모델 정보 가져오기
        provider_name = "?"
        current_model = m.provider.get_default_model() if m.provider else None
        try:
            import pc_assistant as pa
            llm_mode = getattr(pa, 'LLM_MODE', '?')
            if llm_mode == "local":
                model_key = getattr(pa, 'CURRENT_LOCAL_MODEL', '?')
                models = getattr(pa, 'AVAILABLE_MODELS', {})
                model_info = models.get(model_key, {})
                current_model = model_info.get("name", model_key)
                provider_name = f"LOCAL({current_model})"
            else:
                env = getattr(pa, 'CURRENT_ENV', '?')
                provider_name = getattr(pa, 'ENV_CONFIG', {}).get(env, {}).get("name", env)
        except: pass
        return {
            "success": True,
            "initialized": m._init,
            "tools": m.tools.tool_names if m.tools else [],
            "model": current_model,
            "provider": provider_name,
            "llm_mode": llm_mode if 'llm_mode' in dir() else "?",
            "max_iterations": MAX_ITERATIONS,
            "nanobot_version": _ver(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "initialized": False,
        }

@agent_router.get("/api/agent/history")
async def agent_history(session_id: str = "web:default"):
    """마기(main1_First) 세션 히스토리 반환"""
    try:
        m = nanobot_manager
        if not m._init:
            m.initialize()
        session = m.sessions.get_or_create(session_id)
        history = session.get_history(max_messages=50)
        return {"success": True, "history": history}
    except Exception as e:
        return {"success": True, "history": []}

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
#ㅁ
def _ver():
    try:
        import importlib.metadata; return importlib.metadata.version("nanobot-ai")
    except: return "?"

def get_agent_router(): return agent_router

logger.info("⚡ coding_agent.py 로드 (nanobot-ai 직접 import)")