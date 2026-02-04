#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
코딩 LLM + Self-Correction 통합 서버 (v2.1)
- 모든 모드에서 Self-Correction 옵션 지원
- 코드 생성/리뷰/디버그/설명/리팩토링/테스트/일반 + SC
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
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Coding LLM + Self-Correction v2.1")

# ========================================
# Global Variables
# ========================================
API_TOKEN = None
LLM_MODE = "api"
ENV_MODE = "common"

active_requests = {}
request_lock = threading.Lock()

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
}

# Self-Correction 검토용 프롬프트
SC_REVIEW_PROMPT = """당신은 엄격한 품질 검토자입니다.
생성된 답변을 검토하고 문제점을 찾아주세요.

검토 기준:
1. 질문/요청에 정확히 답변했는가?
2. 코드가 있다면 문법 오류나 버그가 없는가?
3. 논리적 오류나 모순이 있는가?
4. 설명이 명확하고 이해하기 쉬운가?
5. 누락된 중요 정보가 있는가?

출력 형식: 첫 줄에 PASS 또는 FAIL, 이후 상세 피드백"""


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
def call_llm_api(prompt: str, system_prompt: str = "", max_tokens: int = 4000) -> dict:
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
    
    try:
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


# ========================================
# Self-Correction 로직
# ========================================
def run_self_correction(prompt: str, system_prompt: str, max_retries: int = 3) -> dict:
    """Self-Correction 루프 실행"""
    
    answer = ""
    review = ""
    is_valid = False
    attempt = 0
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"🔄 [SC] 시도 {attempt}/{max_retries}")
        
        # 1. 답변 생성
        if attempt == 1:
            gen_prompt = prompt
        else:
            # 재시도: 이전 피드백 반영
            gen_prompt = f"{prompt}\n\n[이전 검토 피드백]\n{review}\n\n위 피드백을 반영하여 다시 답변해주세요."
        
        result = call_llm_api(gen_prompt, system_prompt)
        if not result["success"]:
            return {"success": False, "error": result["error"], "retry_count": attempt}
        
        answer = result["content"]
        
        # 2. 답변 검토
        review_prompt = f"[원본 요청]\n{prompt}\n\n[생성된 답변]\n{answer}"
        review_result = call_llm_api(review_prompt, SC_REVIEW_PROMPT, max_tokens=500)
        
        if not review_result["success"]:
            # 검토 실패해도 답변은 반환
            return {
                "success": True,
                "answer": answer,
                "retry_count": attempt,
                "is_valid": False,
                "review": "검토 실패"
            }
        
        review = review_result["content"]
        is_valid = review.strip().upper().startswith("PASS")
        
        logger.info(f"   결과: {'✅ PASS' if is_valid else '❌ FAIL'}")
        
        if is_valid:
            break
    
    return {
        "success": True,
        "answer": answer,
        "retry_count": attempt,
        "is_valid": is_valid,
        "review": review
    }


# ========================================
# 프롬프트 빌더
# ========================================
def build_prompt(mode: str, question: str, code: str, language: str) -> str:
    """모드에 따른 프롬프트 생성"""
    
    if mode == "generate":
        return f"다음 요구사항에 맞는 {language} 코드를 생성해주세요:\n\n{question}"
    
    elif mode in ["review", "debug", "explain", "refactor"]:
        prompt = f"다음 {language} 코드를 분석해주세요:\n\n```{language}\n{code}\n```"
        if question:
            prompt += f"\n\n{question}"
        return prompt
    
    elif mode == "test":
        prompt = f"다음 {language} 코드에 대한 테스트 코드를 작성해주세요:\n\n```{language}\n{code}\n```"
        if question:
            prompt += f"\n\n{question}"
        return prompt
    
    else:  # general
        prompt = question
        if code:
            prompt += f"\n\n```{language}\n{code}\n```"
        return prompt


# ========================================
# Pydantic Models
# ========================================
class CodingQuery(BaseModel):
    question: str
    mode: str = "general"
    language: str = "python"
    code: Optional[str] = ""
    use_sc: bool = False  # ★ Self-Correction 사용 여부
    request_id: Optional[str] = None


class ConvertRequest(BaseModel):
    code: str
    from_lang: str
    to_lang: str
    use_sc: bool = False


class QuickRequest(BaseModel):
    action: str
    code: str
    language: str = "python"
    use_sc: bool = False


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
    
    logger.info(f"🚀 서버 시작: {ENV_MODE}")


# ========================================
# API Endpoints
# ========================================
@app.get("/")
async def home():
    return FileResponse("index_coding_v2.html")


@app.get("/api/status")
async def api_status():
    return {
        "llm_mode": LLM_MODE,
        "api_available": API_TOKEN is not None,
        "env": ENV_MODE,
        "model": API_MODEL,
        "env_name": ENV_CONFIG[ENV_MODE]["name"],
        "self_correction": True
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


@app.post("/api/reload_token")
async def reload_token():
    success = load_api_token()
    return {"success": success, "message": "토큰 리로드 완료" if success else "토큰 리로드 실패"}


@app.post("/api/ask")
async def ask(query: CodingQuery):
    """메인 질문 처리 - Self-Correction 옵션 포함"""
    mode = query.mode
    question = query.question.strip()
    code = query.code.strip() if query.code else ""
    language = query.language
    use_sc = query.use_sc
    
    # 코드 필요한 모드 체크
    if mode in ["review", "debug", "explain", "refactor", "test"] and not code:
        return {"success": False, "answer": "❌ 코드를 입력해주세요!"}
    
    if mode == "generate" and not question:
        return {"success": False, "answer": "❌ 요청사항을 입력해주세요!"}
    
    # 프롬프트 생성
    prompt = build_prompt(mode, question, code, language)
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])
    
    # ★ Self-Correction 사용 여부에 따라 분기
    if use_sc:
        logger.info(f"🔄 Self-Correction 모드: {mode}")
        result = run_self_correction(prompt, system_prompt)
        
        if result["success"]:
            return {
                "success": True,
                "answer": result["answer"],
                "use_sc": True,
                "retry_count": result["retry_count"],
                "is_valid": result["is_valid"],
                "review": result.get("review", "")
            }
        else:
            return {"success": False, "answer": f"❌ {result['error']}"}
    else:
        # 일반 모드 (1회 호출)
        result = call_llm_api(prompt, system_prompt)
        
        if result["success"]:
            return {"success": True, "answer": result["content"], "use_sc": False}
        return {"success": False, "answer": f"❌ {result['error']}"}


@app.post("/api/convert")
async def convert_code(request: ConvertRequest):
    """코드 언어 변환"""
    system_prompt = SYSTEM_PROMPTS["convert"]
    prompt = f"""다음 {request.from_lang} 코드를 {request.to_lang}로 변환해주세요:

```{request.from_lang}
{request.code}
```

변환 시 {request.to_lang}의 관용적인 표현과 베스트 프랙티스를 따라주세요."""
    
    if request.use_sc:
        result = run_self_correction(prompt, system_prompt)
        if result["success"]:
            return {
                "success": True,
                "answer": result["answer"],
                "use_sc": True,
                "retry_count": result["retry_count"],
                "is_valid": result["is_valid"]
            }
        return {"success": False, "answer": f"❌ {result['error']}"}
    else:
        result = call_llm_api(prompt, system_prompt)
        if result["success"]:
            return {"success": True, "answer": result["content"], "use_sc": False}
        return {"success": False, "answer": f"❌ {result['error']}"}


@app.post("/api/quick")
async def quick_action(request: QuickRequest):
    """빠른 작업"""
    action = request.action
    code = request.code
    language = request.language
    use_sc = request.use_sc
    
    if not code:
        return {"success": False, "answer": "❌ 코드를 입력해주세요!"}
    
    action_prompts = {
        "add_comments": "이 코드에 한국어 주석을 추가해주세요. 각 함수와 중요한 로직에 설명을 달아주세요.",
        "improve_names": "이 코드의 변수명과 함수명을 더 명확하고 의미있게 개선해주세요.",
        "optimize": "이 코드의 성능을 최적화해주세요. 불필요한 연산을 줄이고 효율적인 알고리즘을 사용해주세요.",
        "simplify": "이 코드를 더 간결하고 읽기 쉽게 단순화해주세요.",
        "type_hints": "이 Python 코드에 타입 힌트를 추가해주세요.",
        "docstring": "이 코드의 모든 함수와 클래스에 docstring을 추가해주세요."
    }
    
    if action not in action_prompts:
        return {"success": False, "answer": "❌ 알 수 없는 작업입니다."}
    
    prompt = f"{action_prompts[action]}\n\n```{language}\n{code}\n```"
    system_prompt = SYSTEM_PROMPTS["refactor"]
    
    if use_sc:
        result = run_self_correction(prompt, system_prompt)
        if result["success"]:
            return {
                "success": True,
                "answer": result["answer"],
                "use_sc": True,
                "retry_count": result["retry_count"],
                "is_valid": result["is_valid"]
            }
        return {"success": False, "answer": f"❌ {result['error']}"}
    else:
        result = call_llm_api(prompt, system_prompt)
        if result["success"]:
            return {"success": True, "answer": result["content"], "use_sc": False}
        return {"success": False, "answer": f"❌ {result['error']}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)