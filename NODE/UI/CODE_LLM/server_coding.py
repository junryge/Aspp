#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
코딩 전용 LLM 서버 (v1.0)
- 코드 생성, 리뷰, 버그 수정, 설명, 리팩토링
- SK Hynix 내부 API 연동
"""

import os
import re
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Coding LLM Tool")

# ========================================
# Global Variables
# ========================================
API_TOKEN = None

# 개발/운영 환경 설정
ENV_MODE = "dev"

ENV_CONFIG = {
    "dev": {
        "url": "http://dev.assistant.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-Coder-30B-A3B-Instruct",
        "name": "개발(30B)"
    },
    "prod": {
        "url": "http://summary.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-Next-80B-A3B-Instruct",
        "name": "운영(80B)"
    }
}

API_URL = ENV_CONFIG["dev"]["url"]
API_MODEL = ENV_CONFIG["dev"]["model"]

# ========================================
# 모드별 시스템 프롬프트
# ========================================
SYSTEM_PROMPTS = {
    "generate": """당신은 숙련된 소프트웨어 개발자입니다.
사용자의 요구사항에 맞는 고품질 코드를 생성합니다.

규칙:
1. 깔끔하고 읽기 쉬운 코드 작성
2. 적절한 주석 포함
3. 에러 처리 포함
4. 베스트 프랙티스 준수
5. 코드 블록은 ```언어명 으로 감싸기

한국어로 설명하고, 코드는 요청된 언어로 작성하세요.""",

    "review": """당신은 시니어 코드 리뷰어입니다.
제출된 코드를 철저히 검토하고 개선점을 제안합니다.

검토 항목:
1. 코드 품질 및 가독성
2. 버그 또는 잠재적 문제
3. 성능 최적화 가능성
4. 보안 취약점
5. 베스트 프랙티스 준수 여부
6. 테스트 가능성

한국어로 상세히 피드백하세요. 점수(1-10)도 매겨주세요.""",

    "debug": """당신은 디버깅 전문가입니다.
버그를 찾아 수정하고 원인을 설명합니다.

접근 방식:
1. 에러 메시지 분석
2. 버그 원인 파악
3. 수정된 코드 제공
4. 왜 버그가 발생했는지 설명
5. 향후 예방법 제안

한국어로 설명하고, 수정된 코드를 제공하세요.""",

    "explain": """당신은 프로그래밍 교사입니다.
코드를 쉽게 이해할 수 있도록 설명합니다.

설명 포함 사항:
1. 전체 코드 목적
2. 각 부분별 동작 설명
3. 사용된 알고리즘/패턴
4. 핵심 개념 설명
5. 실행 흐름

초보자도 이해할 수 있게 한국어로 친절하게 설명하세요.""",

    "refactor": """당신은 리팩토링 전문가입니다.
기존 코드를 더 좋은 코드로 개선합니다.

리팩토링 원칙:
1. 가독성 향상
2. 중복 제거 (DRY)
3. 단일 책임 원칙
4. 적절한 함수/클래스 분리
5. 네이밍 개선
6. 성능 최적화

원본 코드의 기능은 유지하면서 개선된 코드를 제공하세요.""",

    "convert": """당신은 다국어 프로그래머입니다.
코드를 다른 프로그래밍 언어로 변환합니다.

변환 원칙:
1. 원본 로직 완벽 유지
2. 대상 언어의 관용적 표현 사용
3. 대상 언어의 베스트 프랙티스 준수
4. 필요한 라이브러리 명시
5. 언어별 차이점 설명

한국어로 설명하고, 변환된 코드를 제공하세요.""",

    "test": """당신은 테스트 전문가입니다.
주어진 코드에 대한 테스트 코드를 작성합니다.

테스트 작성 원칙:
1. 단위 테스트 작성
2. 경계값 테스트
3. 예외 상황 테스트
4. 모킹이 필요한 경우 명시
5. 테스트 커버리지 고려

적절한 테스트 프레임워크를 사용하여 테스트 코드를 작성하세요.""",

    "general": """당신은 친절한 프로그래밍 도우미입니다.
코딩 관련 모든 질문에 답변합니다.

답변 원칙:
1. 정확하고 실용적인 정보 제공
2. 예제 코드 포함
3. 관련 참고 자료 언급
4. 초보자도 이해할 수 있게 설명

한국어로 친절하게 답변하세요."""
}

# ========================================
# API Token 관리
# ========================================
def load_api_token():
    """API 토큰 로드"""
    global API_TOKEN
    
    # 여러 경로에서 토큰 파일 찾기
    token_paths = [
        "token.txt",
        "../token.txt",
        os.path.expanduser("~/token.txt")
    ]
    
    for token_path in token_paths:
        if os.path.exists(token_path):
            try:
                with open(token_path, "r", encoding='utf-8') as f:
                    API_TOKEN = f.read().strip()
                if API_TOKEN and "REPLACE" not in API_TOKEN:
                    logger.info(f"✅ API 토큰 로드 완료: {token_path}")
                    return True
                else:
                    logger.warning(f"⚠️ 토큰 파일이 기본값: {token_path}")
            except Exception as e:
                logger.error(f"❌ 토큰 로드 실패: {e}")
    
    logger.warning("⚠️ 토큰 파일을 찾을 수 없습니다")
    return False


# ========================================
# LLM API 호출
# ========================================
def call_llm_api(prompt: str, system_prompt: str = "", max_tokens: int = 4000) -> dict:
    """LLM API 호출"""
    global API_TOKEN
    
    if not API_TOKEN:
        return {"success": False, "error": "API 토큰이 설정되지 않았습니다. token.txt를 확인하세요."}
    
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
    
    for attempt in range(2):
        try:
            logger.info(f"🚀 API 호출 중... (시도 {attempt + 1}/2)")
            response = requests.post(API_URL, headers=headers, json=data, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # <think> 태그 제거 (Qwen3 특성)
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                content = content.strip()
                
                return {"success": True, "content": content}
            else:
                logger.error(f"❌ API 오류: {response.status_code}")
                return {"success": False, "error": f"API 오류: {response.status_code}\n{response.text}"}
                
        except requests.exceptions.Timeout:
            logger.warning(f"⏱️ API 타임아웃 (시도 {attempt + 1}/2)")
            if attempt == 1:
                return {"success": False, "error": "API 요청 시간 초과 (5분). 서버 상태를 확인하세요."}
        except Exception as e:
            logger.error(f"❌ API 호출 실패: {e}")
            return {"success": False, "error": f"API 호출 실패: {str(e)}"}
    
    return {"success": False, "error": "API 호출 실패"}


# ========================================
# Pydantic Models
# ========================================
class CodingQuery(BaseModel):
    question: str
    mode: str = "general"
    language: str = "python"
    code: Optional[str] = ""


class ConvertRequest(BaseModel):
    code: str
    from_lang: str
    to_lang: str


# ========================================
# FastAPI Startup
# ========================================
@app.on_event("startup")
async def startup():
    """서버 시작 시 초기화"""
    load_api_token()
    logger.info(f"🚀 코딩 LLM 서버 시작! 환경: {ENV_MODE}")


# ========================================
# API Endpoints
# ========================================
@app.get("/")
async def home():
    """메인 페이지"""
    return FileResponse("index_coding.html")


@app.get("/api/status")
async def api_status():
    """API 상태 확인"""
    return {
        "api_available": API_TOKEN is not None,
        "env": ENV_MODE,
        "model": API_MODEL,
        "env_name": ENV_CONFIG[ENV_MODE]["name"]
    }


@app.post("/api/reload_token")
async def reload_token():
    """토큰 리로드"""
    success = load_api_token()
    return {
        "success": success,
        "message": "토큰 리로드 완료" if success else "토큰 리로드 실패"
    }


@app.post("/api/set_env")
async def set_env(data: dict):
    """환경 전환"""
    global ENV_MODE, API_URL, API_MODEL
    
    new_env = data.get("env", "dev")
    if new_env not in ENV_CONFIG:
        return {"success": False, "message": "잘못된 환경입니다."}
    
    ENV_MODE = new_env
    API_URL = ENV_CONFIG[new_env]["url"]
    API_MODEL = ENV_CONFIG[new_env]["model"]
    
    logger.info(f"🔄 환경 전환: {new_env} ({API_MODEL})")
    return {
        "success": True,
        "env": ENV_MODE,
        "model": API_MODEL,
        "name": ENV_CONFIG[new_env]["name"]
    }


@app.post("/api/ask")
async def ask(query: CodingQuery):
    """메인 질문 처리"""
    mode = query.mode
    question = query.question.strip()
    code = query.code.strip() if query.code else ""
    language = query.language
    
    # 시스템 프롬프트 선택
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])
    
    # 프롬프트 구성
    if mode == "generate":
        prompt = f"다음 요구사항에 맞는 {language} 코드를 생성해주세요:\n\n{question}"
    elif mode in ["review", "debug", "explain", "refactor"]:
        if not code:
            return {"success": False, "answer": "❌ 코드를 입력해주세요!"}
        prompt = f"다음 {language} 코드를 분석해주세요:\n\n```{language}\n{code}\n```\n\n{question if question else ''}"
    elif mode == "test":
        if not code:
            return {"success": False, "answer": "❌ 테스트할 코드를 입력해주세요!"}
        prompt = f"다음 {language} 코드에 대한 테스트 코드를 작성해주세요:\n\n```{language}\n{code}\n```\n\n{question if question else ''}"
    else:
        prompt = question
        if code:
            prompt += f"\n\n```{language}\n{code}\n```"
    
    # LLM 호출
    result = call_llm_api(prompt, system_prompt)
    
    if result["success"]:
        return {"success": True, "answer": result["content"]}
    else:
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
    
    result = call_llm_api(prompt, system_prompt)
    
    if result["success"]:
        return {"success": True, "answer": result["content"]}
    else:
        return {"success": False, "answer": f"❌ {result['error']}"}


@app.post("/api/quick")
async def quick_action(data: dict):
    """빠른 작업 (주석 추가, 변수명 개선 등)"""
    action = data.get("action", "")
    code = data.get("code", "")
    language = data.get("language", "python")
    
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
    
    prompt = f"""{action_prompts[action]}

```{language}
{code}
```"""
    
    result = call_llm_api(prompt, SYSTEM_PROMPTS["refactor"])
    
    if result["success"]:
        return {"success": True, "answer": result["content"]}
    else:
        return {"success": False, "answer": f"❌ {result['error']}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)