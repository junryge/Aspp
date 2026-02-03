#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
코딩 LLM + Self-Correction 통합 서버 (v2.0)
- 기존 코드 생성/리뷰/디버그 기능
- LangGraph Self-Correction 루프 추가
- 엑셀 데이터 처리 도구 포함
"""

import os
import re
import json
import requests
import tempfile
import threading
from typing import TypedDict, Literal, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangGraph 임포트
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("⚠️ langgraph 미설치. Self-Correction 비활성화")

app = FastAPI(title="Coding LLM + Self-Correction")

# ========================================
# Global Variables
# ========================================
API_TOKEN = None
llm = None

LLM_MODE = "api"
ENV_MODE = "common"

active_requests = {}
request_lock = threading.Lock()

LOCAL_MODEL_PATH = "Qwen3-14B-Q4_K_M.gguf"

ENV_CONFIG = {
    "dev": {
        "url": "http://dev.assistant.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-Coder-30B-A3B-Instruct",
        "name": "DEV(30B)"
    },
    "prod": {
        "url": "http://summary.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-Next-80B-A3B-Instruct",
        "name": "PROD(80B)"
    },
    "common": {
        "url": "http://common.llm.skhynix.com/v1/chat/completions",
        "model": "gpt-oss-20b",
        "name": "COMMON(20B)"
    }
}

API_URL = ENV_CONFIG["common"]["url"]
API_MODEL = ENV_CONFIG["common"]["model"]

# ========================================
# 시스템 프롬프트
# ========================================
SYSTEM_PROMPTS = {
    "generate": """당신은 숙련된 소프트웨어 개발자입니다.
사용자의 요구사항에 맞는 고품질 코드를 생성합니다.
규칙: 깔끔한 코드, 적절한 주석, 에러 처리 포함.
코드 블록은 ```언어명 으로 감싸기. 한국어로 설명.""",

    "review": """당신은 시니어 코드 리뷰어입니다.
검토 항목: 코드 품질, 버그, 성능, 보안, 베스트 프랙티스.
한국어로 상세히 피드백하고 점수(1-10)도 매겨주세요.""",

    "debug": """당신은 디버깅 전문가입니다.
에러 분석 → 버그 원인 파악 → 수정 코드 → 설명 → 예방법.
한국어로 설명하고 수정된 코드를 제공하세요.""",

    "explain": """당신은 프로그래밍 교사입니다.
코드 목적, 부분별 동작, 알고리즘, 실행 흐름을 설명합니다.
초보자도 이해할 수 있게 한국어로 친절하게 설명하세요.""",

    "refactor": """당신은 리팩토링 전문가입니다.
가독성 향상, 중복 제거, 단일 책임, 네이밍 개선, 성능 최적화.
원본 기능 유지하면서 개선된 코드를 제공하세요.""",

    "convert": """당신은 다국어 프로그래머입니다.
원본 로직 유지, 대상 언어 관용적 표현, 베스트 프랙티스 준수.
한국어로 설명하고 변환된 코드를 제공하세요.""",

    "test": """당신은 테스트 전문가입니다.
단위 테스트, 경계값, 예외 상황 테스트를 작성합니다.
적절한 프레임워크 사용.""",

    "general": """당신은 친절한 프로그래밍 도우미입니다.
정확하고 실용적인 정보, 예제 코드 포함. 한국어로 친절하게.""",
    
    # Self-Correction용 프롬프트
    "sc_generate": """당신은 데이터 분석 전문가입니다.
사용자의 질문에 정확하고 구체적으로 답변하세요.
규칙:
1. 데이터가 제공되면 반드시 참조하여 답변
2. 숫자나 통계는 정확하게 인용
3. 불확실한 내용은 명시
4. 간결하고 명확하게 답변""",

    "sc_review": """당신은 엄격한 품질 검토자입니다.
생성된 답변을 검토하고 문제점을 찾아주세요.
검토 기준:
1. 질문에 정확히 답변했는가?
2. 제공된 데이터를 올바르게 사용했는가?
3. 논리적 오류나 모순이 있는가?
4. 숫자/통계가 정확한가?
출력 형식: 첫 줄에 PASS 또는 FAIL, 이후 상세 피드백"""
}


# ========================================
# 토큰 로드
# ========================================
def load_api_token():
    global API_TOKEN
    paths = ["token.txt", "../token.txt", os.path.expanduser("~/token.txt")]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding='utf-8') as f:
                    API_TOKEN = f.read().strip()
                if API_TOKEN and "REPLACE" not in API_TOKEN:
                    logger.info(f"✅ 토큰 로드: {p}")
                    return True
            except Exception as e:
                logger.error(f"❌ 토큰 로드 실패: {e}")
    return False


# ========================================
# LLM API 호출
# ========================================
def call_llm_api(prompt: str, system_prompt: str = "", max_tokens: int = 4000, request_id: str = None) -> dict:
    global API_TOKEN
    
    if not API_TOKEN:
        return {"success": False, "error": "API 토큰 없음"}
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": API_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    cancel_event = None
    if request_id:
        cancel_event = threading.Event()
        with request_lock:
            active_requests[request_id] = cancel_event
    
    try:
        if cancel_event and cancel_event.is_set():
            return {"success": False, "error": "요청 취소됨", "cancelled": True}
        
        response = requests.post(API_URL, headers=headers, json=data, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return {"success": True, "content": content}
        else:
            return {"success": False, "error": f"API 오류: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if request_id:
            with request_lock:
                active_requests.pop(request_id, None)


def call_llm(prompt: str, system_prompt: str = "", max_tokens: int = 4000, request_id: str = None) -> dict:
    return call_llm_api(prompt, system_prompt, max_tokens, request_id)


# ========================================
# Self-Correction 상태 및 그래프
# ========================================
if LANGGRAPH_AVAILABLE:
    class AgentState(TypedDict):
        question: str
        context: str
        answer: str
        review: str
        is_valid: bool
        retry_count: int
        final_answer: str

    def sc_generate_answer(state: AgentState) -> AgentState:
        """답변 생성"""
        logger.info(f"🔄 [생성] 시도 {state['retry_count'] + 1}")
        
        prompt = f"질문: {state['question']}"
        if state.get('context'):
            prompt = f"[데이터]\n{state['context']}\n\n{prompt}"
        if state['retry_count'] > 0 and state.get('review'):
            prompt += f"\n\n[이전 피드백]\n{state['review']}\n\n위 피드백을 반영하여 다시 답변해주세요."
        
        result = call_llm(prompt, SYSTEM_PROMPTS["sc_generate"], 2000)
        answer = result.get("content", result.get("error", "오류"))
        
        return {**state, "answer": answer, "retry_count": state['retry_count'] + 1}

    def sc_review_answer(state: AgentState) -> AgentState:
        """답변 검토"""
        logger.info("🔍 [검토]")
        
        prompt = f"[원본 질문]\n{state['question']}\n\n[생성된 답변]\n{state['answer']}"
        if state.get('context'):
            prompt = f"[참조 데이터]\n{state['context']}\n\n{prompt}"
        
        result = call_llm(prompt, SYSTEM_PROMPTS["sc_review"], 500)
        review = result.get("content", "검토 실패")
        is_valid = review.strip().upper().startswith("PASS")
        
        logger.info(f"   결과: {'✅ PASS' if is_valid else '❌ FAIL'}")
        return {**state, "review": review, "is_valid": is_valid}

    def sc_finalize(state: AgentState) -> AgentState:
        """최종 확정"""
        logger.info("✅ [완료]")
        return {**state, "final_answer": state['answer']}

    def sc_should_retry(state: AgentState) -> Literal["retry", "finalize"]:
        if state['retry_count'] >= 3:
            return "finalize"
        return "finalize" if state['is_valid'] else "retry"

    def build_sc_graph():
        graph = StateGraph(AgentState)
        graph.add_node("generate", sc_generate_answer)
        graph.add_node("review", sc_review_answer)
        graph.add_node("finalize", sc_finalize)
        graph.set_entry_point("generate")
        graph.add_edge("generate", "review")
        graph.add_conditional_edges("review", sc_should_retry, {"retry": "generate", "finalize": "finalize"})
        graph.add_edge("finalize", END)
        return graph.compile()

    SC_GRAPH = build_sc_graph()
else:
    SC_GRAPH = None


# ========================================
# 엑셀 도구
# ========================================
def read_excel_data(file_path: str) -> dict:
    try:
        import pandas as pd
        if not os.path.exists(file_path):
            return {"success": False, "error": f"파일 없음: {file_path}"}
        df = pd.read_excel(file_path)
        return {
            "success": True,
            "columns": df.columns.tolist(),
            "rows": len(df),
            "preview": df.head(10).to_dict('records'),
            "summary": df.describe().to_dict() if df.select_dtypes(include='number').columns.any() else {}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========================================
# Pydantic Models
# ========================================
class CodingQuery(BaseModel):
    question: str
    mode: str = "general"
    language: str = "python"
    code: Optional[str] = ""
    request_id: Optional[str] = None

class SCQuery(BaseModel):
    question: str
    context: Optional[str] = ""


# ========================================
# Startup
# ========================================
@app.on_event("startup")
async def startup():
    global LLM_MODE
    if load_api_token():
        LLM_MODE = "api"
        logger.info("✅ API 모드")
    else:
        logger.warning("⚠️ 토큰 없음")
    
    logger.info(f"🚀 서버 시작: {ENV_MODE}, SC={'✅' if LANGGRAPH_AVAILABLE else '❌'}")


# ========================================
# API Endpoints
# ========================================
@app.get("/")
async def home():
    return FileResponse("index_coding.html")


@app.get("/api/status")
async def api_status():
    return {
        "llm_mode": LLM_MODE,
        "api_available": API_TOKEN is not None,
        "env": ENV_MODE,
        "model": API_MODEL,
        "env_name": ENV_CONFIG[ENV_MODE]["name"],
        "self_correction": LANGGRAPH_AVAILABLE
    }


@app.post("/api/set_env")
async def set_env(data: dict):
    global ENV_MODE, API_URL, API_MODEL
    new_env = data.get("env", "common")
    if new_env not in ENV_CONFIG:
        return {"success": False, "message": "잘못된 환경"}
    ENV_MODE = new_env
    API_URL = ENV_CONFIG[new_env]["url"]
    API_MODEL = ENV_CONFIG[new_env]["model"]
    return {"success": True, "env": ENV_MODE, "model": API_MODEL, "name": ENV_CONFIG[new_env]["name"]}


@app.post("/api/ask")
async def ask(query: CodingQuery):
    """기존 코딩 질문 처리"""
    mode = query.mode
    question = query.question.strip()
    code = query.code.strip() if query.code else ""
    language = query.language
    request_id = query.request_id
    
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])
    
    if mode == "generate":
        prompt = f"다음 요구사항에 맞는 {language} 코드를 생성해주세요:\n\n{question}"
    elif mode in ["review", "debug", "explain", "refactor"]:
        if not code:
            return {"success": False, "answer": "❌ 코드를 입력해주세요!"}
        prompt = f"다음 {language} 코드를 분석해주세요:\n\n```{language}\n{code}\n```\n\n{question if question else ''}"
    elif mode == "test":
        if not code:
            return {"success": False, "answer": "❌ 테스트할 코드를 입력해주세요!"}
        prompt = f"다음 {language} 코드에 대한 테스트 코드를 작성해주세요:\n\n```{language}\n{code}\n```"
    else:
        prompt = question
        if code:
            prompt += f"\n\n```{language}\n{code}\n```"
    
    result = call_llm(prompt, system_prompt, request_id=request_id)
    
    if result["success"]:
        return {"success": True, "answer": result["content"]}
    return {"success": False, "answer": f"❌ {result['error']}"}


# ========================================
# Self-Correction Endpoints
# ========================================
@app.post("/api/sc/ask")
async def sc_ask(query: SCQuery):
    """Self-Correction 루프로 질문 처리"""
    if not LANGGRAPH_AVAILABLE:
        return {"success": False, "error": "langgraph 미설치. pip install langgraph"}
    
    initial_state = {
        "question": query.question,
        "context": query.context or "",
        "answer": "",
        "review": "",
        "is_valid": False,
        "retry_count": 0,
        "final_answer": ""
    }
    
    result = SC_GRAPH.invoke(initial_state)
    
    return {
        "success": True,
        "answer": result['final_answer'],
        "retry_count": result['retry_count'],
        "is_valid": result['is_valid'],
        "review": result['review']
    }


@app.post("/api/sc/ask_excel")
async def sc_ask_excel(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """엑셀 + Self-Correction"""
    if not LANGGRAPH_AVAILABLE:
        return {"success": False, "error": "langgraph 미설치"}
    
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        excel_data = read_excel_data(tmp_path)
        if not excel_data['success']:
            return {"success": False, "error": excel_data['error']}
        
        context = f"엑셀 컬럼: {excel_data['columns']}\n"
        context += f"행 수: {excel_data['rows']}\n"
        context += f"데이터:\n{json.dumps(excel_data['preview'], ensure_ascii=False, indent=2)}"
        
        initial_state = {
            "question": question,
            "context": context,
            "answer": "",
            "review": "",
            "is_valid": False,
            "retry_count": 0,
            "final_answer": ""
        }
        
        result = SC_GRAPH.invoke(initial_state)
        
        return {
            "success": True,
            "answer": result['final_answer'],
            "retry_count": result['retry_count'],
            "is_valid": result['is_valid'],
            "excel_info": {"columns": excel_data['columns'], "rows": excel_data['rows']}
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/sc/excel_read")
async def sc_excel_read(file: UploadFile = File(...)):
    """엑셀 미리보기"""
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        return read_excel_data(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)