#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pc_assistant.py
MAGI (main1_First AI) - PC 비서 + 지식기반 AI v0.2
- 스크린샷: 전용 폴더 저장 + 웹 인라인 표시
- 파일 탐색기/메모장 실행 제거
"""

import os
import re
import json
import base64
import subprocess
import platform
import psutil
import tempfile
import datetime
import time
import webbrowser
import fnmatch
import requests
import shutil
import pandas as pd
from typing import Optional, List
import asyncio
from fastapi import FastAPI, APIRouter, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PCAssistant")

router = APIRouter(prefix="/assistant", tags=["assistant"])
app = FastAPI(title="MAGI (main1_First AI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Global Configuration
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ★ 로컬 GGUF 모델 설정 (생성 파라미터 포함)
AVAILABLE_MODELS = {
    "qwen3-14b": {
        "path": os.path.join(BASE_DIR, "Qwen3-14B-Q4_K_M.gguf"),
        "name": "Qwen3-14B (Q4_K_M)",
        "desc": "한글 최적화 ⭐추천 (풀 GPU)",
        "ctx": 8192,
        "gpu_layers": 50,
        "chat_format": "chatml",
        "repeat_penalty": 1.1,
        "temperature": 0.7,
        "korean_support": True
    },
    "qwen3-8b": {
        "path": os.path.join(BASE_DIR, "Qwen3-8B-Q6_K.gguf"),
        "name": "Qwen3-8B (Q6_K)",
        "desc": "경량 모델, 빠른 응답 (풀 GPU)",
        "ctx": 4096,
        "gpu_layers": 50,
        "chat_format": "chatml",
        "repeat_penalty": 1.1,
        "temperature": 0.7,
        "korean_support": True
    },
    "gemma3-12b": {
        "path": os.path.join(BASE_DIR, "gemma-3-12b-it-q4_k_m.gguf"),
        "name": "Gemma3-12B (Q4_K_M)",
        "desc": "Google 12B, 균형잡힌 성능 (풀 GPU)",
        "ctx": 4096,
        "gpu_layers": 50,
        "chat_format": "gemma",
        "repeat_penalty": 1.1,
        "temperature": 0.7,
        "korean_support": True
    },
    "oh-dcft-claude": {
        "path": os.path.join(BASE_DIR, "oh-dcft-v3.1-claude-3-5-sonnet-20241022.Q8_0.gguf"),
        "name": "OH-DCFT Claude-Sonnet (Q8_0)",
        "desc": "Claude 스타일 응답, 고품질 (GPU+CPU)",
        "ctx": 8192,
        "gpu_layers": 35,
        "chat_format": "chatml",
        "repeat_penalty": 1.1,
        "temperature": 0.7,
        "korean_support": True
    },
    "qwen25-7b": {
        "path": os.path.join(BASE_DIR, "Qwen2.5-7B-Instruct-Q8_0.gguf"),
        "name": "Qwen2.5-7B (Q8_0)",
        "desc": "Qwen2.5 경량 고정밀, 빠른 응답 (풀 GPU)",
        "ctx": 4096,
        "gpu_layers": 50,
        "chat_format": "chatml",
        "repeat_penalty": 1.1,
        "temperature": 0.7,
        "korean_support": True
    },
    "llama-korean-8b": {
        "path": os.path.join(BASE_DIR, "Llama-3.1-Korean-8B-Instruct.Q8_0.gguf"),
        "name": "Llama-3.1-Korean-8B (Q8_0)",
        "desc": "한국어 특화 Llama, 한글 최적화 (풀 GPU)",
        "ctx": 8192,
        "gpu_layers": 50,
        "chat_format": "llama-3",
        "repeat_penalty": 1.1,
        "temperature": 0.7,
        "korean_support": True
    },
}
CURRENT_LOCAL_MODEL = "qwen3-8b"
GGUF_MODEL_PATH = AVAILABLE_MODELS[CURRENT_LOCAL_MODEL]["path"]

LOCAL_LLM = None
GENERATION_CANCELLED = False  # ★ LLM 생성 중지 플래그
CHAT_HISTORY = []
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
HISTORY_MAX = 500  # ★ 히스토리 최대 건수 (설정 가능)

# ★ 세션 관리
CHAT_SESSIONS_DIR = os.path.join(BASE_DIR, "chat_sessions")
os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)
CURRENT_SESSION_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ★ 토큰 사용량 추적
TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "call_count": 0
}

# ★ 스크린샷 전용 폴더
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ★ 스크린샷 비밀번호 인증
SCREENSHOT_PASSWORD = "1234"
screenshot_authenticated = False  # 인증 상태 (1회용)

# ★ 리소스 폴더 (HTML 구성도 등 정적 파일)
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)

# ★ 지식베이스(MD 문서) 폴더
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# ★ 과거지식 보관 폴더
KNOWLEDGE_ARCHIVE_DIR = os.path.join(BASE_DIR, "knowledge_archive")
os.makedirs(KNOWLEDGE_ARCHIVE_DIR, exist_ok=True)

# ★ TF-IDF + BM25 검색 인덱스 (지식베이스)
TFIDF_INDEX = {
    "vectorizer": None,
    "matrix": None,
    "bm25": None,  # ★ BM25 인덱스 추가
    "tokenized_docs": [],  # ★ BM25용 토큰화된 문서
    "filenames": [],
    "contents": [],
    "built_at": None
}

# ★ 한국어 조사/어미 분리 패턴 (확장)
_KOREAN_PARTICLE_PATTERN = re.compile(
    r'(은|는|이|가|을|를|의|에|에서|로|으로|도|와|과|랑|이랑|부터|까지|만|라고|이라고|에게|한테|께|보다|처럼|같이|마다|대로|밖에|조차|뿐)'
    r'$'
)

def korean_tokenize(text: str) -> List[str]:
    """한국어 텍스트를 토큰으로 분리 (조사 분리, 복합명사 분리)"""
    tokens = []
    # 영어/숫자 토큰
    for match in re.finditer(r'[a-zA-Z0-9_]+', text):
        word = match.group().lower()
        if len(word) >= 2:
            tokens.append(word)

    # 한국어 토큰 (2글자 이상)
    for match in re.finditer(r'[가-힣]{2,}', text):
        word = match.group()
        tokens.append(word)
        # 조사 분리 시도
        stripped = _KOREAN_PARTICLE_PATTERN.sub('', word)
        if stripped and stripped != word and len(stripped) >= 2:
            tokens.append(stripped)

    return list(set(tokens))


def _get_knowledge_fingerprint() -> str:
    """knowledge 디렉토리의 파일 목록+수정시간 해시 (변경 감지용)"""
    try:
        entries = []
        for f in sorted(os.listdir(KNOWLEDGE_DIR)):
            if f.endswith(('.md', '.txt')):
                filepath = os.path.join(KNOWLEDGE_DIR, f)
                mtime = os.path.getmtime(filepath)
                entries.append(f"{f}:{mtime}")
        return "|".join(entries)
    except Exception:
        return ""


def build_tfidf_index():
    """지식베이스 파일들로 TF-IDF + BM25 인덱스 구축"""
    filenames = []
    contents = []
    for f in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not f.endswith(('.md', '.txt')):
            continue
        filepath = os.path.join(KNOWLEDGE_DIR, f)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read()
                if text.strip():
                    filenames.append(f)
                    contents.append(text)
        except (OSError, PermissionError):
            continue

    if not contents:
        TFIDF_INDEX["vectorizer"] = None
        TFIDF_INDEX["matrix"] = None
        TFIDF_INDEX["bm25"] = None
        TFIDF_INDEX["tokenized_docs"] = []
        TFIDF_INDEX["filenames"] = []
        TFIDF_INDEX["contents"] = []
        TFIDF_INDEX["built_at"] = None
        return

    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        max_features=10000,
        sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(contents)

    # ★ BM25 인덱스 구축
    bm25_instance = None
    tokenized_docs = []
    try:
        from rank_bm25 import BM25Okapi
        tokenized_docs = [korean_tokenize(doc) for doc in contents]
        bm25_instance = BM25Okapi(tokenized_docs)
        logger.info(f"📊 BM25 인덱스 구축 완료: {len(tokenized_docs)}개 문서")
    except ImportError:
        logger.warning("⚠️ rank_bm25 미설치 → TF-IDF만 사용")
    except Exception as e:
        logger.warning(f"⚠️ BM25 인덱스 구축 실패: {e}")

    TFIDF_INDEX["vectorizer"] = vectorizer
    TFIDF_INDEX["matrix"] = matrix
    TFIDF_INDEX["bm25"] = bm25_instance
    TFIDF_INDEX["tokenized_docs"] = tokenized_docs
    TFIDF_INDEX["filenames"] = filenames
    TFIDF_INDEX["contents"] = contents
    TFIDF_INDEX["built_at"] = datetime.datetime.now().isoformat()
    TFIDF_INDEX["fingerprint"] = _get_knowledge_fingerprint()
    logger.info(f"📊 TF-IDF 인덱스 구축: {len(filenames)}개 문서, {matrix.shape[1]}개 특성")


def tfidf_search(query: str, top_k: int = 5) -> List[dict]:
    """TF-IDF + BM25 하이브리드 검색 (점수 결합)"""
    # ★ 파일 변경 감지 → 자동 리빌드
    current_fp = _get_knowledge_fingerprint()
    if TFIDF_INDEX["vectorizer"] is None or TFIDF_INDEX["matrix"] is None or TFIDF_INDEX.get("fingerprint") != current_fp:
        logger.info("📊 knowledge 디렉토리 변경 감지 → TF-IDF 인덱스 리빌드")
        build_tfidf_index()

    if TFIDF_INDEX["vectorizer"] is None:
        return []

    num_docs = len(TFIDF_INDEX["filenames"])

    # ★ TF-IDF 스코어 (0~1 정규화)
    query_vec = TFIDF_INDEX["vectorizer"].transform([query])
    tfidf_scores = cosine_similarity(query_vec, TFIDF_INDEX["matrix"]).flatten()

    # ★ BM25 스코어 (있으면 결합)
    bm25_scores = np.zeros(num_docs)
    if TFIDF_INDEX["bm25"] is not None:
        try:
            query_tokens = korean_tokenize(query)
            raw_bm25 = TFIDF_INDEX["bm25"].get_scores(query_tokens)
            # 0~1로 정규화
            max_bm25 = max(raw_bm25) if max(raw_bm25) > 0 else 1.0
            bm25_scores = np.array(raw_bm25) / max_bm25
        except Exception as e:
            logger.warning(f"⚠️ BM25 검색 오류: {e}")

    # ★ 하이브리드 점수: TF-IDF 40% + BM25 60% (BM25가 키워드 매칭에 더 효과적)
    if TFIDF_INDEX["bm25"] is not None:
        combined_scores = tfidf_scores * 0.4 + bm25_scores * 0.6
    else:
        combined_scores = tfidf_scores

    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        score = float(combined_scores[idx])
        if score < 0.01:
            continue
        content = TFIDF_INDEX["contents"][idx]
        # 스니펫: 쿼리 키워드가 포함된 부분 추출
        snippet = ""
        query_tokens_for_snippet = korean_tokenize(query)
        content_lower = content.lower()
        for token in query_tokens_for_snippet:
            pos = content_lower.find(token.lower())
            if pos >= 0:
                snippet = content[max(0, pos-30):min(len(content), pos+100)].replace('\n', ' ').strip()
                break
        if not snippet and len(content) > 0:
            snippet = content[:100].replace('\n', ' ').strip()

        search_method = "tfidf+bm25" if TFIDF_INDEX["bm25"] is not None else "tfidf"
        results.append({
            "filename": TFIDF_INDEX["filenames"][idx],
            "snippet": f"...{snippet}..." if snippet else "",
            "score": round(score * 100, 1),
            "method": search_method
        })

    return results


# ★ 시스템 프롬프트 / 도메인 지식 TXT 파일
SYSTEM_PROMPT_FILE = os.path.join(BASE_DIR, "system_prompt.txt")
DOMAIN_KNOWLEDGE_FILE = os.path.join(BASE_DIR, "domain_knowledge.txt")

# ★ LLM 생성 파라미터 (UI에서 조절 가능)
LLM_PARAMS = {
    "temperature": 0.7,
    "repeat_penalty": 1.1,
    "max_tokens": 4096,
    "task_auto_mode": False,   # 태스크별 자동 분류 ON/OFF
}

# ★ 태스크별 파라미터 프로필 (답변 품질 최적화)
TASK_PARAM_PROFILES = {
    "knowledge_qa": {"temperature": 0.15, "top_p": 0.8, "max_tokens": 2048, "repeat_penalty": 1.0},
    "general_chat": {"temperature": 0.5, "top_p": 0.9, "max_tokens": 2048},
    "tool_call":    {"temperature": 0.1, "top_p": 0.7, "max_tokens": 1024},
}

def classify_task_type(user_message: str) -> str:
    """사용자 메시지를 분석하여 태스크 유형 분류"""
    msg = user_message.lower().strip()

    # tool_call: PC 제어 관련 키워드
    tool_keywords = ["프로그램", "실행", "종료", "프로세스", "스크린샷", "캡처", "화면",
                     "시스템", "cpu", "메모리", "디스크", "몇시", "시간", "날짜",
                     "파일 찾", "파일 검색", "검색해", "구글", "뉴스", "폴더", "디렉토리"]
    if any(kw in msg for kw in tool_keywords):
        return "tool_call"

    # knowledge_qa: 지식/문서 관련 키워드 또는 구체적 질문
    knowledge_keywords = ["문서", "아키텍처", "설계", "구조", "모델", "예측", "컬럼",
                          "스펙", "사양", "가이드", "매뉴얼", "규칙", "정책", "프로젝트",
                          "알려줘", "설명해", "뭐야", "어떻게", "왜", "무엇",
                          "이야기해", "에 대해", "에대해", "관련", "내용", "정리해",
                          "요약해", "변경", "수정", "업데이트", "히스토리"]
    if any(kw in msg for kw in knowledge_keywords):
        return "knowledge_qa"

    # 영문/언더스코어 포함 키워드 (파일명, 모듈명 등) → knowledge_qa
    if re.search(r'[A-Za-z_]{3,}', user_message):
        return "knowledge_qa"

    return "general_chat"

def get_task_params(task_type: str) -> dict:
    """태스크 유형에 맞는 LLM 파라미터 반환 (자동 분류 모드시 프로필 사용)"""
    if LLM_PARAMS.get("task_auto_mode"):
        # 자동 분류 ON → 태스크별 프로필 사용
        profile = TASK_PARAM_PROFILES.get(task_type, TASK_PARAM_PROFILES["general_chat"])
        return {
            "temperature": profile["temperature"],
            "top_p": profile["top_p"],
            "max_tokens": profile["max_tokens"],
            "repeat_penalty": profile.get("repeat_penalty", LLM_PARAMS.get("repeat_penalty", 1.1)),
        }
    else:
        # 자동 분류 OFF → UI 수동 설정 사용
        return {
            "temperature": LLM_PARAMS.get("temperature", 0.7),
            "top_p": 0.9,
            "max_tokens": LLM_PARAMS.get("max_tokens", 4096),
            "repeat_penalty": LLM_PARAMS.get("repeat_penalty", 1.1),
        }

LLM_MODE = "local"
API_TOKEN = None

ENV_CONFIG = {
    "dev": {
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-Coder-480B-A35B-Instruct",
        "name": "DEV(480B)"
    },
    "prod": {
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-235B-A22B-Instruct-2507",
        "name": "PROD(235B)"
    },
    "common": {
        "url": "http://dev.hcp.llm.skhynix.com/v1/chat/completions",
        "model": "gpt-oss-120b",
        "name": "COMMON(120B)"
    }
}
CURRENT_ENV = "common"
API_URL = ENV_CONFIG["common"]["url"]
API_MODEL = ENV_CONFIG["common"]["model"]

# ========================================
# System Prompt (파일 기반)
# ========================================
DEFAULT_SYSTEM_PROMPT = """당신은 'MAGI (main1_First AI)'이라는 PC AI 비서입니다.

★★★ 질문 유형 구분 (중요!) ★★★

[1] PC 작업 요청 → 바로 PC 도구 사용
다음 키워드가 포함되면 지식베이스 검색 없이 바로 해당 도구 실행:
- "프로그램", "실행", "종료", "프로세스", "목록" → list_processes, run_program, kill_program
- "스크린샷", "캡처", "화면" → screenshot
- "시스템", "CPU", "메모리", "디스크" → get_system_info
- "몇시", "시간", "날짜" → get_time
- "파일 찾아", "파일 검색" → search_files
- "검색해줘", "구글" → google_search
- "뉴스" → latest_news
- "폴더", "디렉토리" → list_directory

[2] 지식/정보 질문 → 지식베이스 먼저 검색
프로젝트, 코드, 기술문서, 업무 관련 질문:
- 먼저 search_knowledge로 검색
- 문서 있으면 → 내용 기반 답변
- 문서 없으면 → 일반 지식으로 답변

[3] 일반 대화 → 그냥 대화
인사, 잡담, 일반 질문은 도구 없이 바로 답변

★★★ 도구 호출 규칙 (매우 중요!) ★★★
도구를 사용할 때는 반드시 아래 형식의 JSON만 출력하세요.
JSON 앞뒤에 설명 텍스트를 절대 붙이지 마세요.

[도구 호출 예시]
예시1) 사용자: "메모리 얼마나 쓰고 있어?"
올바른 응답: {"tool": "get_system_info"}

예시2) 사용자: "M14 프로젝트 구조 알려줘"
올바른 응답: {"tool": "search_knowledge", "keyword": "M14 프로젝트 구조"}

[PC 도구]
- 프로세스목록: {"tool": "list_processes", "sort_by": "memory"}
- 시스템정보: {"tool": "get_system_info"}
- 스크린샷: {"tool": "screenshot"}
- 현재시간: {"tool": "get_time"}
- 프로그램실행: {"tool": "run_program", "program": "notepad"}
- 프로그램종료: {"tool": "kill_program", "name": "notepad"}
- 파일검색: {"tool": "search_files", "keyword": "문서", "path": "C:/"}
- 폴더보기: {"tool": "list_directory", "path": "C:/Users"}
- 웹검색: {"tool": "google_search", "query": "검색어"}
- 최신뉴스: {"tool": "latest_news"}

[지식베이스 도구]
- 지식검색: {"tool": "search_knowledge", "keyword": "키워드"}
- 지식목록: {"tool": "list_knowledge"}
- 지식읽기: {"tool": "read_knowledge", "filename": "파일명.md"}

★★★ 답변 품질 규칙 ★★★
1. 확실하지 않은 정보는 추측하지 말고 "정확하지 않을 수 있습니다" 또는 "확인이 필요합니다"라고 표시하세요.
2. 복잡한 질문은 단계별로 분석한 후 답변하세요.
3. 이전 대화 맥락을 참고하여 일관된 답변을 유지하세요.
4. 일반 대화는 한국어로 자연스럽게 답변하세요."""

# ★ 지식베이스 답변용 공통 시스템 프롬프트 (CoT + 정확도 강화)
KNOWLEDGE_QA_SYSTEM_PROMPT = """당신은 기술 문서 전문가입니다. 제공된 문서 내용만으로 답변합니다.

[최우선 규칙]
★ 제공된 문서에 있는 내용만 답변하세요.
★ 문서에 없는 내용을 절대 지어내지 마세요. 당신이 알고 있는 지식도 사용하지 마세요.
★ 문서에 언급되지 않은 시스템명, 프로토콜, 기술명을 추가하지 마세요.
★ 문서에 해당 내용이 없으면 "문서에 해당 내용이 없습니다"라고만 답하세요.
★ 도메인 지식이나 사전 학습 내용으로 답변을 보충하지 마세요.

[사고 과정 - 반드시 따르세요]
1) 먼저 사용자의 질문에서 핵심 키워드와 의도를 파악하세요.
2) 제공된 문서에서 관련 내용을 찾으세요.
3) 문서에 있는 내용만 근거로 답변하세요.

[답변 형식]
**📋 핵심 요약**
질문에 대한 핵심 답변을 2~3줄로 요약

**📝 상세 내용**
문서에서 중요한 내용을 충분히 자세하게 정리:
- 주요 기능/목적
- 구성 요소 및 관계
- 동작 방식/흐름
- 중요한 설정이나 파라미터
- 주의사항이나 특이사항

[답변 규칙]
1. 문서 내용을 근거로 정확하고 **충분히 상세하게** 답변하세요.
2. 상세 내용은 최소 10줄 이상 작성하세요. 문서에 있는 중요 정보는 빠뜨리지 마세요.
3. 소스코드 원본은 보여주지 말고, 코드의 기능/역할/동작을 설명하세요.
4. 마크다운 표(| --- |) 사용 금지. "- 항목: 값" 형태로 나열하세요.
5. ## ### 대제목 헤더 대신 **볼드**와 이모지를 사용하세요.
6. 한국어로 답변하세요.
7. 절대 JSON을 출력하거나 도구를 호출하지 마세요.
8. 확실하지 않은 정보는 추측하지 말고 "정확하지 않을 수 있습니다"라고 표시하세요."""

DEFAULT_DOMAIN_KNOWLEDGE = """# 도메인 지식
# 이 파일에 AI가 참고할 도메인 지식을 작성하세요.
# 저장하면 즉시 시스템 프롬프트에 반영됩니다.
#
# 예시:
# [프로젝트 정보]
# - 프로젝트명: OOO
# - 사용 기술: FastAPI, Python, React
# - 아키텍처: 마이크로서비스
#
# [코딩 규칙]
# - Python 3.10+ 사용
# - 타입 힌트 필수
# - docstring 필수
#
# [내부 API]
# - 엔드포인트: http://xxx.xxx.com/v1/
# - 인증: Bearer Token
"""

# ★ 시스템 프롬프트 & 도메인 지식 로드/저장
def load_prompt_file(filepath: str, default_content: str) -> str:
    """TXT 파일에서 내용 로드. 파일 없으면 기본값으로 생성"""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            if content.strip():
                return content
        except Exception as e:
            logger.error(f"❌ 파일 로드 실패 ({filepath}): {e}")
    # 파일 없거나 비어있으면 기본값으로 생성
    save_prompt_file(filepath, default_content)
    return default_content


def save_prompt_file(filepath: str, content: str) -> bool:
    """TXT 파일에 내용 저장"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"✅ 파일 저장: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ 파일 저장 실패 ({filepath}): {e}")
        return False


def get_effective_system_prompt() -> str:
    """시스템 프롬프트 + 도메인 지식을 합쳐서 반환"""
    system_prompt = load_prompt_file(SYSTEM_PROMPT_FILE, DEFAULT_SYSTEM_PROMPT)
    domain_knowledge = load_prompt_file(DOMAIN_KNOWLEDGE_FILE, DEFAULT_DOMAIN_KNOWLEDGE)

    # 도메인 지식에서 주석(#으로 시작하는 줄) 제거한 실제 내용 확인
    dk_lines = [line for line in domain_knowledge.strip().split('\n')
                if line.strip() and not line.strip().startswith('#')]
    has_domain_knowledge = len(dk_lines) > 0

    if has_domain_knowledge:
        effective = f"""{system_prompt}

★★★ 도메인 지식 (참고용) ★★★
{domain_knowledge}

[주의] 도메인 지식은 약어/용어 확인용입니다. 구체적 기술 내용은 반드시 지식베이스 문서(search_knowledge)를 검색해서 확인하세요. 도메인 지식만으로 상세 답변을 지어내지 마세요."""
        logger.info(f"📚 도메인 지식 적용됨 ({len(dk_lines)}줄)")
        return effective
    else:
        return system_prompt


# 전역 변수 (호환성 유지)
SYSTEM_PROMPT = get_effective_system_prompt()


# ========================================
# LLM Functions
# ========================================
def load_local_model(model_key: str = None):
    """로컬 GGUF 모델 로드 (model_key로 모델 선택)"""
    global LOCAL_LLM, CURRENT_LOCAL_MODEL, GGUF_MODEL_PATH

    if model_key is None:
        model_key = CURRENT_LOCAL_MODEL

    if model_key not in AVAILABLE_MODELS:
        logger.error(f"알 수 없는 모델: {model_key}")
        return None

    model_config = AVAILABLE_MODELS[model_key]
    model_path = model_config["path"]

    if not os.path.exists(model_path):
        logger.error(f"GGUF 파일 없음: {model_path}")
        return None

    try:
        from llama_cpp import Llama
        logger.info(f"🔄 모델 로딩 중: {model_config['name']}...")
        llm = Llama(
            model_path=model_path,
            n_ctx=model_config.get("ctx", 8192),
            n_threads=8,
            n_gpu_layers=model_config.get("gpu_layers", 50),
            n_batch=512,
            verbose=False
        )
        CURRENT_LOCAL_MODEL = model_key
        GGUF_MODEL_PATH = model_path
        logger.info(f"✅ 모델 로드 완료: {model_config['name']}")
        return llm
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return None


def get_available_local_models() -> List[dict]:
    """사용 가능한 로컬 모델 목록 반환"""
    models = []
    for key, config in AVAILABLE_MODELS.items():
        exists = os.path.exists(config["path"])
        models.append({
            "key": key,
            "name": config["name"],
            "desc": config.get("desc", ""),
            "available": exists,
            "current": key == CURRENT_LOCAL_MODEL,
            "korean_support": config.get("korean_support", True)
        })
    return models


def load_api_token():
    global API_TOKEN
    paths = [
        os.path.join(BASE_DIR, "token.txt"),
        os.path.join(BASE_DIR, "api_token.txt"),
        "token.txt",
        "../token.txt",
        os.path.expanduser("~/token.txt")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    API_TOKEN = f.read().strip()
                if API_TOKEN and "REPLACE" not in API_TOKEN:
                    logger.info(f"✅ API 토큰 로드: {p}")
                    return True
            except Exception as e:
                logger.error(f"❌ 토큰 로드 실패: {e}")
    logger.warning("⚠️ API 토큰 파일 없음")
    return False


def call_local_llm(prompt: str, system_prompt: str = "", max_tokens: int = 4096, task_type: str = "") -> dict:
    global GENERATION_CANCELLED
    global LOCAL_LLM, CURRENT_LOCAL_MODEL, LLM_PARAMS
    if LOCAL_LLM is None:
        logger.info("⚡ LOCAL_LLM이 None → 자동 재로드 시도")
        LOCAL_LLM = load_local_model()
        if LOCAL_LLM is None:
            return {"success": False, "error": "로컬 모델이 로드되지 않았습니다. GGUF 파일 확인 필요."}

    # ★ 태스크별 파라미터 적용 (task_type이 있으면 프로필 우선, 없으면 기존 방식)
    if task_type:
        params = get_task_params(task_type)
        temperature = params["temperature"]
        repeat_penalty = params["repeat_penalty"]
        actual_max_tokens = params["max_tokens"]
    else:
        temperature = LLM_PARAMS.get("temperature", 0.7)
        repeat_penalty = LLM_PARAMS.get("repeat_penalty", 1.1)
        actual_max_tokens = LLM_PARAMS.get("max_tokens", max_tokens)

    # ★ 모델별 프롬프트 형식
    model_config = AVAILABLE_MODELS.get(CURRENT_LOCAL_MODEL, {})
    chat_format = model_config.get("chat_format", "chatml")

    if chat_format == "gemma":
        # Gemma 3: Google 형식
        full_prompt = f"""<start_of_turn>user
{system_prompt}

{prompt}<end_of_turn>
<start_of_turn>model
"""
        stop_tokens = ["<end_of_turn>", "<start_of_turn>"]
    elif chat_format == "llama-3":
        # Llama 3.1: Meta 형식
        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        stop_tokens = ["<|eot_id|>", "<|end_of_text|>"]
    else:
        # ChatML 형식 (Qwen3, Qwen2.5, OH-DCFT 등 기본)
        full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        stop_tokens = ["<|im_end|>", "<|im_start|>"]

    # ★ 재시도 로직 (일시적 오류 시 1회 재시도)
    for attempt in range(2):
        try:
            GENERATION_CANCELLED = False
            # ★ 스트리밍 모드로 생성 → 토큰마다 중지 체크 가능
            content_parts = []
            cancelled = False
            for chunk in LOCAL_LLM(
                full_prompt,
                max_tokens=actual_max_tokens,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
                stop=stop_tokens,
                echo=False,
                stream=True
            ):
                if GENERATION_CANCELLED:
                    cancelled = True
                    logger.info("🛑 LLM 생성 중지됨 (사용자 요청)")
                    break
                token_text = chunk["choices"][0]["text"]
                content_parts.append(token_text)

            if cancelled:
                GENERATION_CANCELLED = False
                partial = "".join(content_parts).strip()
                return {"success": True, "content": partial + "\n\n🛑 *응답이 중지되었습니다.*"}

            content = "".join(content_parts).strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            # ★ 빈 응답 감지 → 1회 재생성
            if len(content) < 10 and attempt == 0:
                logger.warning(f"⚠️ 로컬 LLM 빈/짧은 응답 ({len(content)}자) → 재시도")
                continue

            return {"success": True, "content": content}
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "ggml" in error_msg.lower():
                logger.error(f"GGUF 메모리 부족: {e}")
                return {"success": False, "error": "GPU/CPU 메모리 부족 - 더 작은 모델로 전환하거나 max_tokens를 줄여주세요."}
            if attempt == 0:
                logger.warning(f"⚠️ GGUF 런타임 오류 → 1회 재시도: {e}")
                time.sleep(1)
                continue
            logger.error(f"GGUF 런타임 오류: {e}", exc_info=True)
            return {"success": False, "error": f"로컬 모델 추론 오류: {error_msg}"}
        except Exception as e:
            logger.error(f"GGUF 추론 오류: {e}", exc_info=True)
            return {"success": False, "error": f"로컬 모델 오류: {type(e).__name__} - {str(e)}"}

    return {"success": False, "error": "로컬 모델 재시도 한도 초과"}


def call_api_llm(prompt: str, system_prompt: str = "", max_tokens: int = 4096, task_type: str = "") -> dict:
    global API_TOKEN, GENERATION_CANCELLED
    if not API_TOKEN:
        return {"success": False, "error": "API 토큰 없음"}

    # ★ 태스크별 파라미터 적용
    if task_type:
        params = get_task_params(task_type)
        api_temperature = params["temperature"]
        api_max_tokens = params["max_tokens"]
    else:
        api_temperature = 0.3
        api_max_tokens = max_tokens

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
        "max_tokens": api_max_tokens,
        "temperature": api_temperature,
        "stream": True  # ★ 스트리밍으로 중지 체크 가능
    }
    # ★ 재시도 로직 (429, 503, Timeout 시 최대 2회 재시도)
    max_retries = 2
    api_timeout = 120  # 기본 120초 (코드생성시 200초)
    if task_type == "coding":
        api_timeout = 200

    GENERATION_CANCELLED = False

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=api_timeout, stream=True)
            if response.status_code == 200:
                # ★ 응답이 스트리밍인지 JSON인지 판별 (사내 서버 호환성)
                content_type = response.headers.get('content-type', '')
                if 'text/event-stream' in content_type or 'stream' in content_type:
                    # ★ SSE 스트리밍 모드: 토큰마다 중지 체크
                    content_parts = []
                    cancelled = False
                    for line in response.iter_lines(decode_unicode=True):
                        if GENERATION_CANCELLED:
                            cancelled = True
                            logger.info("🛑 API LLM 생성 중지됨 (사용자 요청)")
                            response.close()
                            break
                        if not line:
                            continue
                        if line.startswith("data: "):
                            line_data = line[6:]
                            if line_data.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(line_data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                token_text = delta.get("content", "")
                                if token_text:
                                    content_parts.append(token_text)
                            except json.JSONDecodeError:
                                continue

                    if cancelled:
                        GENERATION_CANCELLED = False
                        partial = "".join(content_parts).strip()
                        return {"success": True, "content": partial + "\n\n🛑 *응답이 중지되었습니다.*"}

                    content = "".join(content_parts).strip()
                else:
                    # ★ 비스트리밍 JSON 응답 (사내 서버가 stream 미지원 시 fallback)
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    # 토큰 사용량
                    usage = result.get("usage", {})
                    if usage:
                        TOKEN_USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        TOKEN_USAGE["completion_tokens"] += usage.get("completion_tokens", 0)
                        TOKEN_USAGE["total_tokens"] += usage.get("total_tokens", 0)

                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                # ★ 빈 응답 감지 → 1회 재생성
                if len(content) < 10 and attempt < max_retries:
                    logger.warning(f"⚠️ 빈/짧은 응답 감지 ({len(content)}자) → 재시도 {attempt+1}/{max_retries}")
                    time.sleep(1)
                    continue

                TOKEN_USAGE["call_count"] += 1
                return {"success": True, "content": content}
            elif response.status_code == 401:
                return {"success": False, "error": "API 인증 실패 (401) - 토큰이 만료되었거나 유효하지 않습니다."}
            elif response.status_code in (429, 503):
                if attempt < max_retries:
                    wait_time = 2 ** (attempt + 1)  # 2초, 4초
                    logger.warning(f"⚠️ API {response.status_code} → {wait_time}초 후 재시도 ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                if response.status_code == 429:
                    return {"success": False, "error": "API 요청 한도 초과 (429) - 재시도 후에도 실패했습니다."}
                return {"success": False, "error": f"LLM 서버 응답 없음 (503) - {CURRENT_ENV} 서버 상태를 확인해주세요."}
            else:
                return {"success": False, "error": f"API 오류 ({response.status_code}) - {response.text[:200]}"}
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                logger.warning(f"⚠️ API 타임아웃 → 재시도 ({attempt+1}/{max_retries})")
                continue
            return {"success": False, "error": f"API 요청 시간 초과 ({api_timeout}초) - 네트워크를 확인하거나 로컬 모드로 전환해주세요."}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                logger.warning(f"⚠️ 연결 오류 → 2초 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(2)
                continue
            return {"success": False, "error": f"LLM 서버 연결 실패 - {API_URL} 에 접속할 수 없습니다."}
        except Exception as e:
            logger.error(f"API LLM 호출 오류: {e}", exc_info=True)
            return {"success": False, "error": f"LLM 호출 오류: {type(e).__name__} - {str(e)}"}

    return {"success": False, "error": "API 호출 재시도 한도 초과"}


def call_llm(prompt: str, system_prompt: str = "", max_tokens: int = 4096, task_type: str = "") -> dict:
    if LLM_MODE == "local":
        return call_local_llm(prompt, system_prompt, max_tokens, task_type=task_type)
    else:
        return call_api_llm(prompt, system_prompt, max_tokens, task_type=task_type)


# ========================================
# Tool Functions
# ========================================
def search_files(keyword: str, path: str = "C:/", limit: int = 50) -> List[dict]:
    results = []
    logger.info(f"파일 검색: '{keyword}' in '{path}'")
    try:
        for root, dirs, files in os.walk(path):
            for name in files + dirs:
                if keyword.lower() in name.lower():
                    full_path = os.path.join(root, name)
                    is_dir = os.path.isdir(full_path)
                    try:
                        size = os.path.getsize(full_path) if not is_dir else 0
                        size_str = f"{size / (1024**3):.2f}GB" if size > 1024**3 else f"{size / (1024**2):.1f}MB" if size > 1024**2 else f"{size}B"
                    except (OSError, PermissionError):
                        size_str = "?"
                    results.append({
                        "name": name, "path": full_path,
                        "type": "폴더" if is_dir else "파일", "size": size_str
                    })
                    if len(results) >= limit:
                        return results
    except Exception as e:
        logger.error(f"검색 오류: {e}")
    return results


def search_content(keyword: str, path: str = "C:/", limit: int = 30) -> List[dict]:
    results = []
    extensions = ['.txt', '.py', '.md', '.json', '.html', '.css', '.js', '.csv', '.log']
    try:
        for root, dirs, files in os.walk(path):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in extensions:
                    full_path = os.path.join(root, name)
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(50000)
                            if keyword.lower() in content.lower():
                                idx = content.lower().find(keyword.lower())
                                snippet = content[max(0, idx-30):min(len(content), idx+70)].replace('\n', ' ')
                                results.append({"name": name, "path": full_path, "snippet": f"...{snippet}..."})
                                if len(results) >= limit:
                                    return results
                    except (OSError, PermissionError, UnicodeDecodeError):
                        continue
    except Exception as e:
        logger.error(f"내용 검색 오류: {e}")
    return results


def get_system_info() -> dict:
    drives = []
    for p in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(p.mountpoint)
            drives.append({"drive": p.device, "total": f"{usage.total / (1024**3):.1f}GB", "used": f"{usage.percent}%"})
        except (OSError, PermissionError):
            pass
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": f"{psutil.cpu_count()}코어, {psutil.cpu_percent()}%",
        "memory": f"{psutil.virtual_memory().total // (1024**3)}GB, {psutil.virtual_memory().percent}%",
        "drives": drives
    }


def list_directory(path: str) -> List[dict]:
    items = []
    try:
        for name in os.listdir(path)[:50]:
            full_path = os.path.join(path, name)
            is_dir = os.path.isdir(full_path)
            try:
                size = os.path.getsize(full_path) if not is_dir else 0
                modified = datetime.datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M")
            except (OSError, PermissionError):
                size = 0
                modified = "?"
            items.append({"name": name, "type": "폴더" if is_dir else "파일", "size": f"{size:,}" if not is_dir else "-", "modified": modified})
    except Exception as e:
        return [{"error": str(e)}]
    return items


def read_file(path: str, max_chars: int = 5000) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_chars)
            if len(content) == max_chars:
                content += "\n... (파일이 너무 커서 일부만 표시)"
            return content
    except Exception as e:
        return f"파일 읽기 오류: {e}"


def run_program(program: str) -> str:
    try:
        subprocess.Popen(program, shell=True)
        return f"'{program}' 실행됨"
    except Exception as e:
        return f"실행 오류: {e}"


def kill_program(name: str) -> str:
    try:
        killed = 0
        for proc in psutil.process_iter(['name']):
            if name.lower() in proc.info['name'].lower():
                proc.kill()
                killed += 1
        return f"{killed}개 프로세스 종료됨"
    except Exception as e:
        return f"종료 오류: {e}"


def open_web(url: str) -> str:
    if not url.startswith('http'):
        url = 'https://' + url
    webbrowser.open(url)
    return f"'{url}' 열림"


def google_search(query: str) -> str:
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)
    return f"'{query}' 검색 중..."


def get_time() -> str:
    now = datetime.datetime.now()
    return f"{now.strftime('%Y년 %m월 %d일 %A %H시 %M분 %S초')}"


# ★ 스크린샷: 전용 폴더 저장 + URL 반환
def take_screenshot() -> dict:
    """스크린샷 찍고 전용 폴더에 저장, 웹 표시용 URL 반환"""
    global screenshot_authenticated
    
    # ★ 비밀번호 인증 체크
    if not screenshot_authenticated:
        return {"success": False, "auth_required": True, "error": "🔒 스크린샷은 비밀번호 인증이 필요합니다."}
    
    # 인증 후 1회 사용 → 자동 잠금
    screenshot_authenticated = False
    
    try:
        from PIL import ImageGrab
        filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        img = ImageGrab.grab()
        img.save(filepath)
        logger.info(f"📸 스크린샷 저장: {filepath}")
        return {
            "success": True,
            "filename": filename,
            "path": filepath,
            "url": f"/assistant/screenshots/{filename}"
        }
    except ImportError:
        return {"success": False, "error": "PIL(Pillow) 미설치. pip install Pillow"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ★ 최신뉴스: 독립 브라우저 창 열기 → 스크린샷 → 그 창만 닫기
def latest_news() -> dict:
    """구글뉴스를 독립 브라우저로 열고, 스크린샷 찍고, 그 창만 닫기"""
    global screenshot_authenticated
    
    # ★ 비밀번호 인증 체크
    if not screenshot_authenticated:
        return {"success": False, "auth_required": True, "error": "🔒 뉴스 스크린샷은 비밀번호 인증이 필요합니다."}
    
    screenshot_authenticated = False  # 1회 사용 후 잠금

    import time

    news_proc = None
    temp_profile = None
    
    try:
        news_url = "https://news.google.com/home?hl=ko&gl=KR&ceid=KR:ko"
        
        # 임시 프로필 폴더 (독립 Chrome 인스턴스용)
        temp_profile = os.path.join(tempfile.gettempdir(), "chrome_news_temp")
        
        # 1. Chrome 찾기
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
        ]
        
        chrome_exe = None
        for p in chrome_paths:
            if os.path.exists(p):
                chrome_exe = p
                break
        
        if chrome_exe:
            # 독립 Chrome 인스턴스 (기존 Chrome과 별개, 전체화면)
            news_proc = subprocess.Popen([
                chrome_exe,
                f"--user-data-dir={temp_profile}",
                "--no-first-run",
                "--no-default-browser-check",
                "--start-maximized",
                "--disable-extensions",
                "--disable-sync",
                "--disable-translate",
                news_url
            ])
            logger.info(f"📰 구글뉴스 독립 창 열기 (PID: {news_proc.pid})")
        else:
            webbrowser.open(news_url)
            logger.info("📰 구글뉴스 열기 (기본 브라우저)")
        
        # 2. 초기 로딩 대기 (임시 프로필 첫 실행은 느림)
        time.sleep(3)
        
        # 2.5. 강제 전체화면 (임시 프로필은 최대화 무시할 수 있음)
        try:
            import ctypes
            import ctypes.wintypes
            
            # 가장 앞에 있는 Chrome 창 찾아서 최대화
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if hwnd:
                SW_MAXIMIZE = 3
                user32.ShowWindow(hwnd, SW_MAXIMIZE)
                logger.info(f"🔲 뉴스 창 최대화 완료 (hwnd: {hwnd})")
        except Exception as e:
            logger.warning(f"⚠️ 최대화 실패 (무시): {e}")
        
        # 3. 뉴스 페이지 완전히 로딩될 때까지 충분히 대기
        logger.info("⏳ 뉴스 페이지 로딩 대기 중... (8초)")
        time.sleep(8)
        
        # 3. 스크린샷 찍기
        from PIL import ImageGrab
        filename = f"news_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        img = ImageGrab.grab()
        img.save(filepath)
        logger.info(f"📸 뉴스 스크린샷: {filepath}")
        
        # 4. 독립 Chrome만 종료
        time.sleep(0.5)
        if news_proc and news_proc.poll() is None:
            # 자식 프로세스 포함 전체 종료
            try:
                parent = psutil.Process(news_proc.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
                logger.info(f"🔒 뉴스 창 닫기 완료 (PID: {news_proc.pid})")
            except psutil.NoSuchProcess:
                pass
        
        # 5. 임시 프로필 정리 (백그라운드)
        try:
            if temp_profile and os.path.exists(temp_profile):
                shutil.rmtree(temp_profile, ignore_errors=True)
        except OSError:
            pass

        return {
            "success": True,
            "filename": filename,
            "path": filepath,
            "url": f"/assistant/screenshots/{filename}"
        }
    except Exception as e:
        # 에러 시에도 프로세스 정리
        if news_proc and news_proc.poll() is None:
            try:
                news_proc.terminate()
            except (OSError, psutil.NoSuchProcess):
                pass
        return {"success": False, "error": str(e)}


# ★ 프로세스 목록 조회
def list_processes(sort_by: str = "memory", limit: int = 30) -> List[dict]:
    """실행 중인 프로세스 목록 반환"""
    processes = []
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'status']):
            try:
                info = proc.info
                mem = info.get('memory_info')
                mem_mb = mem.rss / (1024 * 1024) if mem else 0
                processes.append({
                    "pid": info['pid'],
                    "name": info['name'],
                    "cpu": proc.cpu_percent(interval=0),
                    "memory_mb": round(mem_mb, 1),
                    "status": info.get('status', '?')
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # 정렬
        if sort_by == "cpu":
            processes.sort(key=lambda x: x['cpu'], reverse=True)
        else:
            processes.sort(key=lambda x: x['memory_mb'], reverse=True)

        return processes[:limit]
    except Exception as e:
        logger.error(f"프로세스 목록 오류: {e}")
        return [{"error": str(e)}]


# ★ 지식베이스 함수들
def list_knowledge() -> List[dict]:
    """지식베이스 파일 목록"""
    files = []
    try:
        for f in sorted(os.listdir(KNOWLEDGE_DIR)):
            if f.endswith(('.md', '.txt')):
                filepath = os.path.join(KNOWLEDGE_DIR, f)
                size = os.path.getsize(filepath)
                modified = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M")
                files.append({"filename": f, "size": f"{size:,}B", "modified": modified})
    except Exception as e:
        logger.error(f"지식 목록 오류: {e}")
    return files


def normalize_keyword(keyword: str) -> List[str]:
    """키워드를 여러 변형으로 확장 (언더스코어, 공백, 하이픈, 한국어 조사 분리)"""
    keyword = keyword.strip().lower()
    variants = [keyword]

    # 언더스코어 <-> 공백 <-> 하이픈 변환
    if '_' in keyword:
        variants.append(keyword.replace('_', ' '))
        variants.append(keyword.replace('_', '-'))
        variants.append(keyword.replace('_', ''))
    if ' ' in keyword:
        variants.append(keyword.replace(' ', '_'))
        variants.append(keyword.replace(' ', '-'))
        variants.append(keyword.replace(' ', ''))
    if '-' in keyword:
        variants.append(keyword.replace('-', '_'))
        variants.append(keyword.replace('-', ' '))
        variants.append(keyword.replace('-', ''))

    # 대소문자 변형 추가
    variants.append(keyword.upper())
    variants.append(keyword.title())

    # ★ 한국어 조사 분리 → 원형 추가
    stripped = _KOREAN_PARTICLE_PATTERN.sub('', keyword)
    if stripped and stripped != keyword and len(stripped) >= 2:
        variants.append(stripped)

    # ★ 한국어 토큰도 개별 변형으로 추가
    for token in korean_tokenize(keyword):
        if token not in variants and len(token) >= 2:
            variants.append(token)

    return list(set(variants))


def calculate_relevance_score(filename: str, content: str, keyword: str, variants: List[str]) -> int:
    """문서의 관련성 점수 계산 (높을수록 관련성 높음)"""
    score = 0
    filename_lower = filename.lower()
    content_lower = content.lower()

    for variant in variants:
        v_lower = variant.lower()

        # 파일명에 키워드 포함 (+50점)
        if v_lower in filename_lower:
            score += 50
            # 파일명이 키워드로 시작하면 추가 점수
            if filename_lower.startswith(v_lower):
                score += 30

        # 내용에서 키워드 등장 횟수 (최대 100점)
        count = content_lower.count(v_lower)
        score += min(count * 5, 100)

        # 제목/헤더에 키워드 있으면 추가 점수
        lines = content.split('\n')[:20]  # 상위 20줄만 확인
        for line in lines:
            if line.startswith('#') and v_lower in line.lower():
                score += 40
                break

    return score


def search_knowledge(keyword: str) -> List[dict]:
    """지식베이스에서 키워드로 파일 검색 (TF-IDF + 키워드 매칭 하이브리드)"""
    results = []
    seen_files = set()
    variants = normalize_keyword(keyword)

    # ★ 키워드를 토큰으로 분리 (공백, 언더스코어 등으로)
    keyword_tokens = re.split(r'[\s_\-\.]+', keyword.strip().lower())
    keyword_tokens = [t for t in keyword_tokens if len(t) > 1]

    logger.info(f"🔍 지식검색: '{keyword}' → 변형: {variants[:5]}, 토큰: {keyword_tokens}")

    # ========================================
    # 1차: TF-IDF 코사인 유사도 검색 (의미적 유사성)
    # ========================================
    try:
        tfidf_results = tfidf_search(keyword, top_k=5)
        for r in tfidf_results:
            seen_files.add(r["filename"])
            results.append(r)
        if tfidf_results:
            logger.info(f"📊 TF-IDF 결과: {len(tfidf_results)}개 (상위: {[r['filename'] for r in tfidf_results[:3]]})")
    except Exception as e:
        logger.warning(f"TF-IDF 검색 실패 (키워드 매칭으로 대체): {e}")

    # ========================================
    # 2차: 키워드 매칭 (TF-IDF에서 못 찾은 파일 보완)
    # ========================================
    try:
        for f in os.listdir(KNOWLEDGE_DIR):
            if not f.endswith(('.md', '.txt')) or f in seen_files:
                continue
            filepath = os.path.join(KNOWLEDGE_DIR, f)

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                    content = fh.read()
            except (OSError, PermissionError):
                continue

            matched = False
            snippet = ""
            for variant in variants:
                v_lower = variant.lower()
                if v_lower in f.lower() or v_lower in content.lower():
                    matched = True
                    idx = content.lower().find(v_lower)
                    if idx >= 0:
                        snippet = content[max(0, idx-50):min(len(content), idx+100)].replace('\n', ' ').strip()
                    break

            if not matched and keyword_tokens:
                f_lower = f.lower()
                content_lower = content.lower()
                token_matches = sum(1 for t in keyword_tokens if t in f_lower or t in content_lower)
                if token_matches >= max(1, len(keyword_tokens) * 0.5):
                    matched = True
                    for token in keyword_tokens:
                        idx = content_lower.find(token)
                        if idx >= 0:
                            snippet = content[max(0, idx-50):min(len(content), idx+100)].replace('\n', ' ').strip()
                            break

            if matched:
                score = calculate_relevance_score(f, content, keyword, variants)
                if keyword_tokens:
                    for token in keyword_tokens:
                        if token in f.lower():
                            score += 30
                results.append({
                    "filename": f,
                    "snippet": f"...{snippet}..." if snippet else "(키워드 매칭)",
                    "score": score,
                    "method": "keyword"
                })

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        logger.info(f"📊 최종 검색 결과: {len(results)}개 (상위: {[r['filename'] for r in results[:3]]})")

    except Exception as e:
        logger.error(f"지식 검색 오류: {e}")
    return results


def generate_guided_questions(user_query: str) -> dict:
    """지식베이스 검색 실패 시 LLM이 파일 목록을 보고 역질문을 생성"""
    try:
        # 1. 현재 지식베이스 파일 목록 가져오기
        kb_files = []
        for f in os.listdir(KNOWLEDGE_DIR):
            if f.endswith(('.md', '.txt')):
                filepath = os.path.join(KNOWLEDGE_DIR, f)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                        # 첫 500자만 읽어서 힌트 추출
                        preview = fh.read(500)
                        # 헤더/제목 추출
                        headers = [line.strip('# ').strip() for line in preview.split('\n')[:10] 
                                   if line.startswith('#')]
                    kb_files.append({
                        "filename": f,
                        "headers": headers[:3]
                    })
                except (OSError, UnicodeDecodeError):
                    kb_files.append({"filename": f, "headers": []})

        if not kb_files:
            return {
                "success": False,
                "message": "지식베이스에 등록된 문서가 없습니다.",
                "suggestions": []
            }

        # 2. 파일 목록 문자열 생성
        file_list_str = "\n".join([
            f"- {f['filename']}" + (f" (주요 내용: {', '.join(f['headers'])})" if f['headers'] else "")
            for f in kb_files
        ])

        # 3. LLM에게 역질문 생성 요청
        guide_prompt = f"""사용자가 "{user_query}"라고 질문했지만, 지식베이스에서 정확히 매칭되는 문서를 찾지 못했습니다.

현재 지식베이스에 등록된 파일 목록:
{file_list_str}

위 파일 목록을 분석해서, 사용자의 의도에 맞는 **구체적인 추천 질문 3~5개**를 생성해주세요.

[규칙]
1. 파일명에서 프로젝트명, 버전, 키워드를 추출해서 구체적인 질문으로 만드세요.
2. 사용자의 원래 질문과 관련성 높은 파일을 우선 추천하세요.
3. 관련 파일이 없으면, 가장 유사한 파일 기반으로 질문을 만드세요.
4. 각 질문은 지식베이스에서 검색 가능한 키워드를 포함해야 합니다.

[출력 형식 - 반드시 이 JSON 형식으로만 출력]
{{"guide_message": "관련 문서를 찾지 못했습니다. 혹시 이런 내용을 찾으시나요?", "suggestions": ["질문1", "질문2", "질문3"]}}"""

        guide_system = """당신은 질문 유도 전문가입니다.
사용자의 모호한 질문을 분석하고, 지식베이스 파일 목록을 참고하여 더 구체적인 질문을 추천합니다.
반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 출력하지 마세요."""

        result = call_llm(guide_prompt, guide_system, max_tokens=1000)

        if result["success"]:
            content = result["content"].strip()
            # JSON 추출 시도
            try:
                # ```json ``` 블록 제거
                content = re.sub(r'```(?:json)?\s*', '', content)
                content = content.strip('`').strip()
                # JSON 파싱
                guide_data = json.loads(content)
                return {
                    "success": True,
                    "message": guide_data.get("guide_message", "관련 문서를 찾지 못했습니다."),
                    "suggestions": guide_data.get("suggestions", []),
                    "kb_files": [f["filename"] for f in kb_files]
                }
            except json.JSONDecodeError:
                logger.warning(f"⚠️ 역질문 JSON 파싱 실패: {content[:200]}")
                # JSON 파싱 실패 시 파일 목록 기반 기본 추천
                pass

        # 4. LLM 실패 시 파일명 기반 기본 추천 생성
        suggestions = []
        for f in kb_files[:5]:
            fname = f["filename"].replace('.md', '').replace('.txt', '')
            # 파일명에서 의미있는 키워드 추출
            parts = re.split(r'[_\-\.]', fname)
            clean_name = ' '.join([p for p in parts if len(p) > 1])
            if clean_name:
                suggestions.append(f"{clean_name} 알려줘")

        return {
            "success": True,
            "message": f"'{user_query}'에 대한 정확한 문서를 찾지 못했습니다. 다음 중 찾으시는 내용이 있나요?",
            "suggestions": suggestions,
            "kb_files": [f["filename"] for f in kb_files]
        }

    except Exception as e:
        logger.error(f"역질문 생성 오류: {e}")
        return {"success": False, "message": "역질문 생성 실패", "suggestions": []}


def read_knowledge(filename: str) -> str:
    """지식베이스 MD 파일 읽기"""
    filepath = os.path.join(KNOWLEDGE_DIR, filename)
    if not os.path.exists(filepath):
        for f in os.listdir(KNOWLEDGE_DIR):
            if filename.lower() in f.lower():
                filepath = os.path.join(KNOWLEDGE_DIR, f)
                break
        else:
            return f"❌ '{filename}' 파일을 찾을 수 없습니다."

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(30000)
            if len(content) == 30000:
                content += "\n\n... (문서가 길어서 일부만 표시)"
            return content
    except Exception as e:
        return f"파일 읽기 오류: {e}"


def extract_relevant_sections(doc_content: str, query: str, max_chars: int = 3000) -> str:
    """문서에서 질문과 관련된 섹션만 추출 (작은 LLM용 핵심 컨텍스트 축소)"""
    query_lower = query.lower()
    query_tokens = re.split(r'[\s_\-\.]+', query_lower)
    query_tokens = [t for t in query_tokens if len(t) > 1]

    # 마크다운 섹션 분리 (##~#### 기준)
    sections = re.split(r'\n(?=#{1,4}\s)', doc_content)
    if len(sections) <= 1:
        # 섹션 구분 없는 문서 → 원본 그대로
        return doc_content[:max_chars]

    scored_sections = []
    for section in sections:
        section_lower = section.lower()
        score = 0
        for token in query_tokens:
            count = section_lower.count(token)
            if count > 0:
                score += count * 10
            # 섹션 제목(첫줄)에 있으면 가중치
            first_line = section_lower.split('\n')[0]
            if token in first_line:
                score += 50
        scored_sections.append((score, section))

    # 점수순 정렬, 상위 섹션 선택
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    result_parts = []
    total = 0
    # 항상 첫 섹션(참조/약어 등 문서 헤더) 포함
    header = sections[0]
    if len(header) < 500:
        result_parts.append(header)
        total += len(header)

    for score, section in scored_sections:
        if score == 0:
            continue
        if section in result_parts:
            continue
        if total + len(section) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                result_parts.append(section[:remaining] + "\n... (생략)")
            break
        result_parts.append(section)
        total += len(section)

    if not result_parts:
        return doc_content[:max_chars]

    return "\n\n".join(result_parts)


def analyze_data(path: str) -> str:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(path, encoding='utf-8', errors='ignore')
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            return f"지원하지 않는 형식: {ext}"
        result = []
        result.append(f"파일: {os.path.basename(path)}")
        result.append(f"크기: {len(df):,}행 x {len(df.columns)}열")
        result.append(f"컬럼: {', '.join(df.columns.tolist()[:20])}")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().to_string()
            result.append(f"통계:\n{stats}")
        result.append(f"샘플:\n{df.head(5).to_string()}")
        return "\n".join(result)
    except Exception as e:
        return f"분석 오류: {e}"


# ========================================
# ★ 자동 리소스 첨부 (키워드 매칭 시 관련 파일 링크 자동 추가)
# ========================================
AUTO_RESOURCES = [
    {
        "keywords": ["amhs", "amos", "oht", "mcs", "stk", "cnv", "lft", "inv",
                     "foup", "pdt", "rtc", "fio", "반송", "스토커", "컨베이어",
                     "리프트", "인버터", "물류", "반송차량", "구성도", "시스템 구성"],
        "filename": "Amhs_시스템구성도.html",
        "label": "📊 AMHS 시스템 구성도",
        "desc": "인터랙티브 구성도 (클릭하면 새 탭에서 열림)"
    }
]


def auto_attach_resources(user_message: str, response: str) -> str:
    """사용자 질문에 관련 리소스가 있으면 응답 끝에 링크 자동 추가"""
    msg_lower = user_message.lower()
    attached = []

    for res in AUTO_RESOURCES:
        # 키워드 매칭 (2개 이상 매칭되거나, 핵심 키워드 1개 매칭)
        matched = [kw for kw in res["keywords"] if kw in msg_lower]
        core_keywords = ["amhs", "amos", "구성도", "시스템 구성"]
        core_match = any(kw in msg_lower for kw in core_keywords)

        if len(matched) >= 1 or core_match:
            filepath = os.path.join(RESOURCES_DIR, res["filename"])
            if os.path.exists(filepath):
                attached.append(res)

    if attached:
        links = []
        for res in attached:
            url = f"/assistant/resources/{res['filename']}"
            links.append(f"\n\n---\n🔗 **[{res['label']}]({url})** - {res['desc']}")
        response += "".join(links)

    return response


# ========================================
# Tool 실행기
# ========================================
def execute_tool(tool_data: dict) -> str:
    tool_name = tool_data.get("tool")

    if tool_name == "search_files":
        results = search_files(tool_data.get("keyword", ""), tool_data.get("path", "C:/"))
        return json.dumps(results[:20], ensure_ascii=False, indent=2)

    elif tool_name == "search_content":
        results = search_content(tool_data.get("keyword", ""), tool_data.get("path", "C:/"))
        return json.dumps(results[:10], ensure_ascii=False, indent=2)

    elif tool_name == "get_system_info":
        return json.dumps(get_system_info(), ensure_ascii=False, indent=2)

    elif tool_name == "list_directory":
        results = list_directory(tool_data.get("path", "C:/"))
        return json.dumps(results, ensure_ascii=False, indent=2)

    elif tool_name == "read_file":
        return read_file(tool_data.get("path", ""))

    elif tool_name == "run_program":
        return run_program(tool_data.get("program", ""))

    elif tool_name == "kill_program":
        return kill_program(tool_data.get("name", ""))

    elif tool_name == "open_web":
        return open_web(tool_data.get("url", ""))

    elif tool_name == "google_search":
        return google_search(tool_data.get("query", ""))

    elif tool_name == "get_time":
        return get_time()

    # ★ 스크린샷 - JSON 반환
    elif tool_name == "screenshot":
        result = take_screenshot()
        return json.dumps(result, ensure_ascii=False)

    # ★ 최신뉴스 - 구글뉴스 열고 스크린샷 찍고 닫기
    elif tool_name == "latest_news":
        result = latest_news()
        return json.dumps(result, ensure_ascii=False)

    elif tool_name == "analyze_data":
        return analyze_data(tool_data.get("path", ""))

    # ★ 지식베이스 도구들
    elif tool_name == "list_knowledge":
        results = list_knowledge()
        return json.dumps(results, ensure_ascii=False, indent=2)

    elif tool_name == "search_knowledge":
        results = search_knowledge(tool_data.get("keyword", ""))
        return json.dumps(results, ensure_ascii=False, indent=2)

    elif tool_name == "read_knowledge":
        return read_knowledge(tool_data.get("filename", ""))

    # ★ 프로세스 목록
    elif tool_name == "list_processes":
        results = list_processes(tool_data.get("sort_by", "memory"), tool_data.get("limit", 30))
        return json.dumps(results, ensure_ascii=False, indent=2)

    return "알 수 없는 도구"


# ========================================
# JSON 감지
# ========================================
def extract_tool_json(text: str) -> Optional[dict]:
    # 패턴 1: ```json 코드블록
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

    # 패턴 2: 인라인 {"tool": "..."}
    match = re.search(r'(\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\})', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

    # 패턴 3: 전체가 JSON
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            data = json.loads(stripped)
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

    # 패턴 4: 멀티라인 JSON
    match = re.search(r'\{\s*"tool"\s*:.*?\}', text, re.DOTALL)
    if match:
        try:
            json_str = re.sub(r'[\n\r\t]', ' ', match.group(0))
            json_str = re.sub(r'\s+', ' ', json_str)
            data = json.loads(json_str)
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


# ========================================
# ========================================
# Chat Processing
# ========================================
# ★ 대화 요약 캐시
_conversation_summary_cache = {
    "summary": "",
    "summarized_up_to": 0,  # 요약된 마지막 인덱스
}

def get_recent_context(max_turns: int = 4) -> str:
    """최근 대화 기록을 문자열로 반환 (맥락 유지용) - 동적 컨텍스트 관리"""
    if not CHAT_HISTORY:
        return ""

    # ★ 동적 컨텍스트 크기 계산
    if LLM_MODE == "local":
        model_ctx = AVAILABLE_MODELS.get(CURRENT_LOCAL_MODEL, {}).get("ctx", 4096)
        max_context_chars = int(model_ctx * 1.5)  # 약 40% 할당 (1토큰 ≈ 3.5자 한글)
    else:
        max_context_chars = 6000  # API 모드: 넉넉히

    context_parts = []
    total_chars = 0

    # ★ 8턴 초과시 이전 대화 요약 활용
    history_len = len(CHAT_HISTORY)
    if history_len > 16:  # 8턴 = 16메시지 (user+assistant)
        summary = _get_or_create_summary()
        if summary:
            summary_text = f"[이전 대화 요약]\n{summary}\n"
            context_parts.append(summary_text)
            total_chars += len(summary_text)

    # ★ 최근 대화: 최근 2턴은 전문, 그 이전은 1000자까지
    recent = CHAT_HISTORY[-(max_turns * 2):]
    for i, msg in enumerate(recent):
        role = "사용자" if msg["role"] == "user" else "비서"
        is_recent_2_turns = i >= len(recent) - 4  # 마지막 4개 (2턴)
        char_limit = 2000 if is_recent_2_turns else 1000
        content = msg["content"][:char_limit]
        line = f"[{role}]: {content}"

        if total_chars + len(line) > max_context_chars:
            break
        context_parts.append(line)
        total_chars += len(line)

    return "\n".join(context_parts)


def _get_or_create_summary() -> str:
    """대화 요약을 생성하거나 캐시에서 반환"""
    global _conversation_summary_cache
    history_len = len(CHAT_HISTORY)

    # 이미 최신 요약이 있으면 캐시 반환
    if (_conversation_summary_cache["summary"] and
        _conversation_summary_cache["summarized_up_to"] >= history_len - 20):
        return _conversation_summary_cache["summary"]

    # 요약할 대화 범위: 처음부터 최근 8턴 이전까지
    end_idx = max(0, history_len - 16)
    if end_idx < 4:  # 요약할 게 별로 없으면 스킵
        return ""

    messages_to_summarize = CHAT_HISTORY[:end_idx]
    # 최대 20개 메시지만 요약 대상
    if len(messages_to_summarize) > 20:
        messages_to_summarize = messages_to_summarize[-20:]

    summary_input = "\n".join([
        f"{'사용자' if m['role'] == 'user' else '비서'}: {m['content'][:300]}"
        for m in messages_to_summarize
    ])

    summary_prompt = f"""다음 대화 내용을 3줄 이내로 핵심만 요약하세요.
주요 주제, 결정사항, 중요 정보만 포함하세요.

{summary_input}

요약:"""

    result = call_llm(summary_prompt, "대화 내용을 간결하게 요약하는 도우미입니다. 3줄 이내로 핵심만 요약하세요.", max_tokens=300, task_type="general_chat")
    if result["success"] and len(result["content"]) > 10:
        _conversation_summary_cache["summary"] = result["content"].strip()
        _conversation_summary_cache["summarized_up_to"] = end_idx
        logger.info(f"📝 대화 요약 생성 완료 ({end_idx}개 메시지 → {len(result['content'])}자)")
        return _conversation_summary_cache["summary"]

    return ""


# ========================================
# ★ AMHS 데이터 분석 엔진
# ========================================
import sys
import io

# data_poi 폴더를 sys.path에 추가하여 전처리기 import
DATA_POI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_poi")
if DATA_POI_DIR not in sys.path:
    sys.path.insert(0, DATA_POI_DIR)

try:
    from fabjob_preprocessor import analyze_fabjob, is_fabjob_data
    from oht_preprocessor import analyze_oht, analyze_oht_multi, is_oht_data
    from conveyor_preprocessor import analyze_conveyor, analyze_conveyor_multi, is_conveyor_data
    from lifter_preprocessor import analyze_lifter, analyze_lifter_multi, is_lifter_data
    AMHS_PREPROCESSORS_AVAILABLE = True
    logger.info("✅ AMHS 전처리기 4개 로드 완료")
except ImportError as e:
    AMHS_PREPROCESSORS_AVAILABLE = False
    logger.warning(f"⚠️ AMHS 전처리기 로드 실패: {e}")

# AMHS 세션 데이터 캐시 (후속 질문 지원)
AMHS_SESSION_DATA = {
    "active": False,
    "files": [],            # [{name, equipment_type, preprocess_text, basic_info}]
    "system_prompt": "",
    "equipment_type": "",
    "last_analysis_response": "",  # ★ 이전 LLM 분석 응답 캐시 (후속 질문 맥락 유지)
}

# 프롬프트 디렉토리
EQUIP_PROMPT_DIR = os.path.join(DATA_POI_DIR, "prompts")


def detect_equipment_type(df: pd.DataFrame) -> str:
    """MESSAGENAME 패턴으로 설비 타입 감지"""
    if 'MESSAGENAME' not in df.columns:
        return 'UNKNOWN'
    messages = df['MESSAGENAME'].dropna().astype(str).tolist()
    counts = {"OHT": 0, "CONVEYOR": 0, "LIFTER": 0, "FABJOB": 0}
    for msg in messages:
        msg_upper = str(msg).upper()
        if msg_upper.startswith("RAIL-") and "INTERRAIL" not in msg_upper:
            counts["OHT"] += 1
        elif msg_upper.startswith("INTERRAIL-"):
            counts["CONVEYOR"] += 1
        elif msg_upper.startswith("STORAGE-"):
            counts["LIFTER"] += 1
        elif msg_upper.startswith("VM-"):
            counts["FABJOB"] += 1

    if counts["FABJOB"] >= 3:
        return "FABJOB"
    max_type = max(counts, key=counts.get)
    return max_type if counts[max_type] >= 3 else "UNKNOWN"


def get_equipment_prompts(equipment_type: str) -> str:
    """설비별 프롬프트 로드 (common + system + fewshot)"""
    parts = []
    # 공통 프롬프트
    common_path = os.path.join(EQUIP_PROMPT_DIR, "BASE", "common.txt")
    if os.path.exists(common_path):
        with open(common_path, 'r', encoding='utf-8') as f:
            parts.append(f.read().strip())

    # 설비별 프롬프트
    equip_dir = os.path.join(EQUIP_PROMPT_DIR, equipment_type)
    if os.path.isdir(equip_dir):
        for fname in ["system.txt", "fewshot.txt"]:
            fpath = os.path.join(equip_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8') as f:
                    parts.append(f.read().strip())

    return "\n\n".join(parts) if parts else ""


def parse_csv_content(file_content_b64: str, file_name: str = "") -> pd.DataFrame:
    """base64 CSV 내용 → DataFrame 변환 (인코딩 자동감지)"""
    raw_bytes = base64.b64decode(file_content_b64)
    for enc in ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']:
        try:
            text = raw_bytes.decode(enc)
            df = pd.read_csv(io.StringIO(text))
            if len(df.columns) > 1:
                return df
        except:
            continue
    raise ValueError(f"CSV 파싱 실패: {file_name}")


def analyze_csv_basic(df: pd.DataFrame) -> str:
    """CSV 기본 정보 분석"""
    info_lines = [f"행 수: {len(df)}", f"컬럼: {', '.join(df.columns.tolist())}"]
    if 'MESSAGENAME' in df.columns:
        msg_counts = df['MESSAGENAME'].dropna().value_counts()
        info_lines.append(f"메시지 유형 수: {len(msg_counts)}")
        top_msgs = msg_counts.head(5)
        info_lines.append("주요 메시지: " + ", ".join(f"{m}({c})" for m, c in top_msgs.items()))
    if 'CARRIER' in df.columns:
        carriers = df['CARRIER'].dropna().unique()
        info_lines.append(f"캐리어: {', '.join(str(c).strip().strip(chr(39)) for c in carriers[:5])}")
    if 'TIME_EX' in df.columns:
        times = df['TIME_EX'].dropna()
        if len(times) > 0:
            info_lines.append(f"시간 범위: {times.iloc[0][:30]} ~ {times.iloc[-1][:30]}")
    return "\n".join(info_lines)


def analyze_amhs_data(user_message: str, files: list) -> str:
    """AMHS 데이터 분석 메인 함수 (다중 파일 지원)"""
    global AMHS_SESSION_DATA

    if not AMHS_PREPROCESSORS_AVAILABLE:
        return "❌ AMHS 전처리기를 불러올 수 없습니다. data_poi 폴더를 확인하세요."

    all_results = []

    for file_data in files:
        try:
            df = parse_csv_content(file_data.file_content, file_data.file_name)
            equip_type = detect_equipment_type(df)
            basic_info = analyze_csv_basic(df)

            # 설비별 전처리기 실행
            preprocess_text = ""
            if equip_type == "FABJOB":
                result = analyze_fabjob(df)
                preprocess_text = result.get('preprocessed_text', '')
            elif equip_type == "OHT":
                result = analyze_oht_multi(df)
                preprocess_text = result.get('preprocessed_text', '')
            elif equip_type == "CONVEYOR":
                result = analyze_conveyor_multi(df)
                preprocess_text = result.get('preprocessed_text', '')
            elif equip_type == "LIFTER":
                result = analyze_lifter_multi(df)
                preprocess_text = result.get('preprocessed_text', '')
            else:
                preprocess_text = f"⚠️ 설비 타입 감지 실패 (UNKNOWN)\n기본정보:\n{basic_info}"

            all_results.append({
                'name': file_data.file_name,
                'equipment_type': equip_type,
                'preprocess_text': preprocess_text,
                'basic_info': basic_info
            })
            logger.info(f"📊 AMHS 분석: {file_data.file_name} → {equip_type}")

        except Exception as e:
            logger.error(f"AMHS 분석 오류 ({file_data.file_name}): {e}", exc_info=True)
            all_results.append({
                'name': file_data.file_name,
                'equipment_type': 'ERROR',
                'preprocess_text': f"❌ 분석 오류: {str(e)}",
                'basic_info': ''
            })

    if not all_results:
        return "❌ 분석할 파일이 없습니다."

    # 세션 데이터 캐싱
    primary_type = all_results[0]['equipment_type']
    equip_system_prompt = get_equipment_prompts(primary_type)

    AMHS_SESSION_DATA = {
        "active": True,
        "files": all_results,
        "system_prompt": equip_system_prompt,
        "equipment_type": primary_type,
        "last_analysis_response": "",  # generate_amhs_response에서 채워짐
    }

    return generate_amhs_response(user_message, AMHS_SESSION_DATA)


def generate_amhs_response(user_message: str, session_data: dict) -> str:
    """AMHS 세션 데이터 기반 LLM 응답 생성"""
    # 전처리 결과 조합
    preprocess_parts = []
    for f in session_data['files']:
        preprocess_parts.append(f"[파일: {f['name']}] (설비: {f['equipment_type']})\n{f['preprocess_text']}")

    preprocess_combined = "\n\n".join(preprocess_parts)

    # 대화 이력 맥락 추가 (AMHS 관련 대화만, 충분한 길이로)
    recent_history = ""
    if len(CHAT_HISTORY) > 0:
        # 최근 6개 항목 (3턴) — AMHS 분석 맥락 유지를 위해 충분한 길이 허용
        recent = CHAT_HISTORY[-6:]
        history_parts = []
        for h in recent:
            role = "사용자" if h['role'] == 'user' else "AI"
            # ★ 이전 분석 응답은 1500자까지 (200자는 너무 짧아 맥락 소실)
            content_limit = 1500 if h['role'] == 'assistant' else 500
            truncated = h['content'][:content_limit]
            if len(h['content']) > content_limit:
                truncated += "...(생략)"
            history_parts.append(f"{role}: {truncated}")
        recent_history = "\n".join(history_parts)

    # ★ 이전 분석 결과도 세션에 캐시 (후속 질문에 중요)
    prev_analysis = session_data.get('last_analysis_response', '')

    # 프롬프트 조립
    analysis_prompt = f"""다음은 AMHS 설비 로그를 전처리한 분석 결과입니다.

{preprocess_combined}
"""

    if recent_history:
        analysis_prompt += f"""
[이전 대화 맥락]
{recent_history}
"""

    if prev_analysis and user_message != prev_analysis[:50]:
        analysis_prompt += f"""
[이전 분석 결과 요약]
{prev_analysis[:2000]}
"""

    analysis_prompt += f"""
사용자 질문: {user_message}

위 전처리 결과와 이전 대화 맥락을 바탕으로 사용자 질문에 정확하게 답변해주세요.
구간별 소요시간, 지연 원인, HCACK 코드, 경로 정보 등을 참고하여 구체적으로 분석해주세요.
이전 대화에서 이미 분석한 내용이 있으면 그것을 참고하여 일관성 있게 답변하세요."""

    system_prompt = session_data.get('system_prompt', '')

    result = call_llm(analysis_prompt, system_prompt=system_prompt, max_tokens=4096, task_type="analysis")

    if result.get("success"):
        # ★ 마지막 분석 응답을 세션에 캐싱 (후속 질문에서 참조)
        session_data['last_analysis_response'] = result["content"]
        return result["content"]
    else:
        return f"❌ LLM 분석 실패: {result.get('error', '알 수 없는 오류')}\n\n📊 전처리 결과:\n{preprocess_combined[:2000]}"


def process_chat(user_message: str, files: list = None) -> str:
    global LOCAL_LLM, LLM_MODE, AMHS_SESSION_DATA

    if LLM_MODE == "local" and LOCAL_LLM is None:
        return "❌ 로컬 모델이 로드되지 않았습니다."
    if LLM_MODE != "local" and not API_TOKEN:
        return "❌ API 토큰이 없습니다."

    try:
        # ★ AMHS 파일 분석 모드
        if files and len(files) > 0:
            return analyze_amhs_data(user_message, files)

        # ★ AMHS 세션 활성 시 후속 질문 처리
        if AMHS_SESSION_DATA.get("active"):
            # 일반 대화로 전환하는 키워드 체크
            reset_keywords = ["초기화", "리셋", "새로운 대화", "다른 질문", "일반 모드"]
            if any(kw in user_message for kw in reset_keywords):
                AMHS_SESSION_DATA = {"active": False, "files": [], "system_prompt": "", "equipment_type": "", "last_analysis_response": ""}
                return "✅ AMHS 분석 모드를 종료했습니다. 일반 대화 모드로 전환합니다."
            return generate_amhs_response(user_message, AMHS_SESSION_DATA)

        # ★ 태스크 유형 분류 (파라미터 최적화용)
        task_type = classify_task_type(user_message)
        logger.info(f"📊 태스크 유형: {task_type} | 메시지: {user_message[:50]}")

        # ★ 도메인 키워드 → LLM 우회, 직접 지식베이스 검색
        amhs_keywords = ["amhs", "amos", "구성도", "시스템 구성", "oht", "mcs", "stk", "cnv", "lft", "inv",
                         "foup", "pdt", "rtc", "fio", "반송", "스토커", "컨베이어", "리프트", "인버터",
                         "통신", "프로토콜", "atlas", "smartstar", "logpresso", "tibco",
                         "아키텍처", "컬럼사전", "예측모델", "hubroom", "hid",
                         "접속", "url", "시뮬레이션", "컬럼", "m14", "m16", "모니터링",
                         "fab", "quwa", "strate", "inpos", "sorter", "emptyfoup",
                         "c2", "c2f", "m10", "m10a", "m10b", "m14a", "m14b", "m16a", "m16e", "r3"]
        msg_lower = user_message.lower()
        amhs_matched = [kw for kw in amhs_keywords if kw in msg_lower]
        if amhs_matched:
            logger.info(f"🔀 AMHS 키워드 감지 ({amhs_matched}) → 지식베이스 강제 검색")
            # ★ 전체 사용자 메시지로 검색 (첫 키워드만 쓰면 엉뚱한 문서 반환됨)
            search_result = execute_tool({"tool": "search_knowledge", "keyword": user_message})
            if search_result and not search_result.startswith("❌"):
                try:
                    sr_data = json.loads(search_result)
                    sr_list = sr_data if isinstance(sr_data, list) else sr_data.get("results", []) if isinstance(sr_data, dict) else []
                    if sr_list and isinstance(sr_list[0], dict):
                        best_file = sr_list[0]["filename"]
                        doc_content = execute_tool({"tool": "read_knowledge", "filename": best_file})
                        if doc_content and not doc_content.startswith("❌"):
                            doc_limit = 12000 if LLM_MODE == "api" else 3000
                            # ★ 관련 섹션만 추출 (작은 LLM이 긴 문서 못 읽는 문제 해결)
                            if LLM_MODE != "api" and len(doc_content) > 1500:
                                doc_content = extract_relevant_sections(doc_content, user_message, max_chars=doc_limit)
                            elif len(doc_content) > doc_limit:
                                doc_content = doc_content[:doc_limit] + "\n\n... (이하 생략)"
                            # ★ 최근 대화 맥락 (후속 질문 지원, 짧게)
                            brief_context = ""
                            if CHAT_HISTORY:
                                last_turns = CHAT_HISTORY[-4:]  # 최근 2턴만
                                ctx_lines = []
                                for m in last_turns:
                                    role = "사용자" if m["role"] == "user" else "비서"
                                    ctx_lines.append(f"{role}: {m['content'][:200]}")
                                brief_context = f"[이전 대화 (참고만)]\n" + "\n".join(ctx_lines) + "\n\n"
                            follow_up = f"""아래 문서 내용을 읽고, 이 문서 내용만으로 질문에 답하세요.

=== 문서 시작 ({best_file}) ===
{doc_content}
=== 문서 끝 ===

{brief_context}[질문] {user_message}

[규칙]
- 위 문서에 적힌 내용만 답하세요. 문서에 없는 내용은 절대 추가하지 마세요.
- 이전 대화에서 나온 내용이라도 문서에 없으면 사용하지 마세요.
- 문서에 없으면 "문서에 해당 내용이 없습니다"라고만 답하세요."""
                            result2 = call_llm(follow_up, KNOWLEDGE_QA_SYSTEM_PROMPT, max_tokens=4096 if LLM_MODE == "api" else 1024, task_type="knowledge_qa")
                            if result2["success"]:
                                content2 = result2["content"].strip()
                                source_info = f"\n\n---\n📚 **참조 문서**: {best_file}"
                                return content2 + source_info
                except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
                    pass
            # 검색 실패 시 일반 LLM 흐름으로 fallback

        # ★ 대화 맥락 추가
        recent_context = get_recent_context(max_turns=3)
        if recent_context:
            context_prompt = f"""[이전 대화]
{recent_context}

[현재 질문]
{user_message}"""
        else:
            context_prompt = user_message

        result = call_llm(context_prompt, get_effective_system_prompt(), task_type=task_type)
        if not result["success"]:
            return f"❌ LLM 오류: {result.get('error', '알 수 없는 오류')}"

        text = result["content"]
        logger.info(f"📝 LLM 응답: {text[:200]}")

        tool_data = extract_tool_json(text)

        # ★ JSON 파싱 실패했지만 tool 호출 의도가 보이는 경우 → 1회 재요청
        if tool_data is None and '"tool"' in text and task_type == "tool_call":
            logger.warning("⚠️ JSON 파싱 실패 → 재요청")
            retry_prompt = f"{context_prompt}\n\n[주의] 이전 응답에서 JSON 형식이 잘못되었습니다. 유효한 JSON만 출력하세요. 예: {{\"tool\": \"get_system_info\"}}"
            retry_result = call_llm(retry_prompt, get_effective_system_prompt(), task_type="tool_call")
            if retry_result["success"]:
                retry_data = extract_tool_json(retry_result["content"])
                if retry_data:
                    tool_data = retry_data
                    text = retry_result["content"]
                    logger.info(f"✅ JSON 재파싱 성공: {tool_data}")

        if tool_data:
            try:
                if "keyword" in tool_data:
                    kw = tool_data["keyword"].replace("*", "").replace(".", "").strip()
                    if not kw:
                        return "❌ 검색 키워드가 비어있습니다."
                    tool_data["keyword"] = kw

                logger.info(f"🔧 도구 실행: {tool_data}")
                tool_result = execute_tool(tool_data)
                logger.info(f"📊 도구 결과: {tool_result[:300]}")

                tool_name = tool_data.get("tool")

                # ★ 스크린샷: 직접 포맷팅 (LLM 2차 호출 불필요)
                if tool_name == "screenshot":
                    try:
                        sc_data = json.loads(tool_result)
                        if sc_data.get("auth_required"):
                            return "🔒 **스크린샷 인증 필요**\n\n비밀번호를 입력해야 스크린샷을 찍을 수 있습니다.\n\n<!--AUTH_REQUIRED:screenshot-->"
                        if sc_data.get("success"):
                            return f"📸 스크린샷을 찍었습니다!\n\n![스크린샷]({sc_data['url']})\n\n저장 위치: `{sc_data['path']}`"
                        else:
                            return f"❌ 스크린샷 실패: {sc_data.get('error', '?')}"
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"스크린샷 결과 파싱 오류: {e}")
                        return f"❌ 스크린샷 처리 오류: {e}"

                # ★ 최신뉴스: 직접 포맷팅
                if tool_name == "latest_news":
                    try:
                        news_data = json.loads(tool_result)
                        if news_data.get("auth_required"):
                            return "🔒 **뉴스 스크린샷 인증 필요**\n\n비밀번호를 입력해야 뉴스를 확인할 수 있습니다.\n\n<!--AUTH_REQUIRED:news-->"
                        if news_data.get("success"):
                            return f"📰 **최신 뉴스** (구글뉴스)\n\n![뉴스]({news_data['url']})\n\n브라우저를 닫았습니다."
                        else:
                            return f"❌ 뉴스 확인 실패: {news_data.get('error', '?')}"
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"뉴스 결과 파싱 오류: {e}")
                        return f"❌ 뉴스 처리 오류: {e}"

                # ========================================
                # ★ 지식베이스 핸들러 (3가지 구조)
                # ========================================

                # 1) read_knowledge → "이 문서를 참고해서 질문에 정확히 답변해라" + 기술 문서 전문가
                if tool_name == "read_knowledge":
                    if tool_result.startswith("❌"):
                        return tool_result
                    
                    # 문서 길이 제한 (API vs GGUF) + 관련 섹션 추출
                    doc_limit = 12000 if LLM_MODE == "api" else 3000
                    if LLM_MODE != "api" and len(tool_result) > 1500:
                        doc_content = extract_relevant_sections(tool_result, user_message, max_chars=doc_limit)
                    elif len(tool_result) > doc_limit:
                        doc_content = tool_result[:doc_limit] + "\n\n... (이하 생략)"
                    else:
                        doc_content = tool_result
                    
                    follow_up_prompt = f"""아래 문서 내용을 읽고, 이 문서 내용만으로 질문에 답하세요.

=== 문서 시작 ===
{doc_content}
=== 문서 끝 ===

[질문] {user_message}

[규칙]
- 위 문서에 적힌 내용만 답하세요. 문서에 없는 내용은 절대 추가하지 마세요.
- 이전 대화에서 나온 내용이라도 문서에 없으면 사용하지 마세요.
- 문서에 없으면 "문서에 해당 내용이 없습니다"라고만 답하세요."""

                    result2_tokens = 4096 if LLM_MODE == "api" else 1024
                    result2 = call_llm(follow_up_prompt, KNOWLEDGE_QA_SYSTEM_PROMPT, max_tokens=result2_tokens, task_type="knowledge_qa")
                    if result2["success"]:
                        content = result2["content"].strip()
                        logger.info(f"📝 지식읽기 2차 응답: {content[:200] if content else '(빈 응답)'}")
                        if content and not extract_tool_json(content):
                            return content
                    # fallback: 문서 내용 직접 반환
                    logger.info("⚠️ 2차 LLM 응답 없음 → 문서 직접 반환")
                    return f"📄 **문서 내용:**\n\n{doc_content[:5000]}"

                # 2) search_knowledge → 관련성 높은 문서만 사용
                if tool_name == "search_knowledge":
                    try:
                        search_results = json.loads(tool_result)
                        if not search_results:
                            # ★ 역질문 유도: LLM이 파일 목록 보고 추천 질문 생성
                            guide = generate_guided_questions(user_message)
                            if guide["success"] and guide["suggestions"]:
                                lines = [f"🔍 **{guide['message']}**\n"]
                                for i, suggestion in enumerate(guide["suggestions"], 1):
                                    # <!--SUGGEST:질문--> 마커로 프론트엔드에서 클릭 버튼 생성
                                    lines.append(f"<!--SUGGEST:{suggestion}-->")
                                lines.append(f"\n\n💡 위 추천 질문을 클릭하거나, 더 구체적인 키워드로 다시 질문해보세요.")
                                if guide.get("kb_files"):
                                    lines.append(f"\n📚 현재 등록된 문서: {', '.join(guide['kb_files'][:5])}")
                                return "\n".join(lines)
                            else:
                                return "🔍 관련 문서를 찾지 못했습니다. 지식베이스에 문서를 먼저 등록해주세요."

                        # ★ 관련성 점수 기반으로 문서 선택 (상위 3개까지)
                        MAX_TOTAL_LENGTH = 15000 if LLM_MODE == "api" else 3000
                        MAX_DOCS = 3  # ★ 상위 3개 문서까지만
                        merged_docs = []
                        total_length = 0
                        doc_names = []

                        # search_results items이 dict인지 방어 체크
                        first_item = search_results[0]
                        if not isinstance(first_item, dict):
                            logger.warning(f"⚠️ search_results[0] is {type(first_item).__name__}, not dict")
                            return "🔍 검색 결과 형식 오류입니다. 다시 시도해주세요."
                        top_score = first_item.get("score", 100)

                        for i, result in enumerate(search_results[:MAX_DOCS]):
                            if not isinstance(result, dict):
                                continue
                            filename = result["filename"]
                            score = result.get("score", 0)

                            # 1위 문서와 점수 차이가 50% 이상이면 제외
                            if i > 0 and score < top_score * 0.5:
                                logger.info(f"⏭️ 점수 낮아 제외: {filename} (점수: {score}, 1위: {top_score})")
                                break

                            doc_content = read_knowledge(filename)

                            if doc_content.startswith("❌"):
                                continue

                            # 남은 공간에 맞게 자르기
                            remaining = MAX_TOTAL_LENGTH - total_length
                            if remaining <= 1000:
                                break

                            if len(doc_content) > remaining:
                                doc_content = doc_content[:remaining] + "\n\n... (문서 일부 생략)"

                            merged_docs.append(f"📄 **[{filename}]**\n{doc_content}")
                            doc_names.append(filename)
                            total_length += len(doc_content)

                        if not merged_docs:
                            return "🔍 문서를 읽을 수 없습니다."

                        combined_content = "\n\n---\n\n".join(merged_docs)
                        doc_list = ", ".join(doc_names)
                        logger.info(f"📚 참조 문서: {doc_list} (총 {total_length}자)")

                        follow_up_prompt = f"""아래 문서 내용을 읽고, 이 문서 내용만으로 질문에 답하세요.

=== 문서 시작 ({doc_list}) ===
{combined_content}
=== 문서 끝 ===

[질문] {user_message}

[규칙]
- 위 문서에 적힌 내용만 답하세요. 문서에 없는 내용은 절대 추가하지 마세요.
- 이전 대화에서 나온 내용이라도 문서에 없으면 사용하지 마세요.
- 문서에 없으면 "문서에 해당 내용이 없습니다"라고만 답하세요."""

                        result2_tokens = 4096 if LLM_MODE == "api" else 1024
                        result2 = call_llm(follow_up_prompt, KNOWLEDGE_QA_SYSTEM_PROMPT, max_tokens=result2_tokens, task_type="knowledge_qa")
                        if result2["success"]:
                            content = result2["content"].strip()
                            logger.info(f"📝 지식검색 2차 응답: {content[:200] if content else '(빈 응답)'}")
                            if content and not extract_tool_json(content):
                                # 참조 문서 목록 추가
                                source_info = f"\n\n---\n📚 **참조 문서**: {doc_list}"
                                return content + source_info
                        # fallback: 문서 내용 직접 반환
                        logger.info("⚠️ 2차 LLM 응답 없음 → 문서 직접 반환")
                        return f"📄 **참조 문서:**\n\n{combined_content[:5000]}"
                    except Exception as e:
                        logger.error(f"지식검색 처리 오류: {e}")
                        pass

                # 3) list_knowledge → LLM 호출 없이 직접 포맷팅 (API 낭비 방지)
                if tool_name == "list_knowledge":
                    try:
                        files = json.loads(tool_result)
                        if not files:
                            return "📭 지식베이스에 등록된 문서가 없습니다.\n\n📚 지식베이스 버튼으로 MD/TXT 파일을 업로드하세요."
                        lines = [f"## 📚 지식베이스 문서 ({len(files)}개)\n"]
                        for f in files:
                            lines.append(f"- 📄 **{f['filename']}** ({f['size']}, {f['modified']})")
                        lines.append(f"\n💡 문서 내용이 궁금하면 파일명으로 질문하세요. (예: \"HID_INOUT 알려줘\")")
                        return "\n".join(lines)
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.error(f"지식 목록 파싱 오류: {e}")
                        pass

                # 기타 도구: 2차 LLM으로 해석
                follow_up_prompt = f"""사용자 질문: {user_message}

도구 실행 결과:
{tool_result}

위 결과를 사용자가 이해하기 쉽게 한국어로 정리해서 답변하세요.
- JSON 원본을 보여주지 말고 핵심만 정리
- 도구를 다시 호출하지 마세요 (JSON 출력 금지)
- 마크다운 형식으로 보기 좋게"""

                follow_up_system = """당신은 MAGI (main1_First AI) 비서입니다.
도구 실행 결과를 한국어로 친절하게 설명합니다.
절대 JSON을 출력하지 마세요. 자연어로만 답변하세요."""

                result2 = call_llm(follow_up_prompt, follow_up_system, task_type="tool_call")
                if result2["success"]:
                    response = result2["content"]
                    if extract_tool_json(response):
                        return format_tool_result_fallback(tool_data, tool_result)
                    return response
                else:
                    return format_tool_result_fallback(tool_data, tool_result)

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON 파싱 오류: {e}")
                return "❌ 명령 처리 중 오류가 발생했습니다."

        # ★★★ LLM이 도구를 호출하지 않은 경우 → 자동 지식베이스 탐색 ★★★
        # PC 명령어/인사가 아닌데 도구를 안 불렀으면 = 지식베이스를 놓친 것
        skip_keywords = ["프로그램", "실행", "종료", "프로세스", "스크린샷", "캡처",
                         "시스템", "cpu", "메모리", "디스크", "몇시", "시간", "날짜",
                         "파일 찾", "파일 검색", "검색해", "구글", "뉴스", "폴더", "디렉토리"]
        greeting_patterns = ["안녕", "하이", "헬로", "hi", "hello", "ㅎㅇ", "반가",
                             "고마워", "감사", "ㄱㅅ", "ㅋㅋ", "ㅎㅎ", "네", "응", "ㅇㅇ",
                             "아니", "뭐해", "심심", "잘자", "바이"]

        msg_lower = user_message.lower().strip()
        is_pc_cmd = any(kw in msg_lower for kw in skip_keywords)
        is_greeting = any(msg_lower.startswith(g) or msg_lower == g for g in greeting_patterns)
        is_short = len(msg_lower) <= 4

        # 지식베이스 파일이 있고, PC명령/인사/짧은말이 아닌 경우 → 자동 검색
        kb_has_files = False
        try:
            kb_has_files = any(f.endswith(('.md', '.txt')) for f in os.listdir(KNOWLEDGE_DIR))
        except OSError:
            pass

        if kb_has_files and not is_pc_cmd and not is_greeting and not is_short:
            logger.info(f"🔄 자동 지식베이스 탐색: '{user_message}'")

            # 키워드 추출 (조사/어미 제거)
            clean_msg = re.sub(r'(알려줘|설명해줘|뭐야|뭐에요|해줘|할래|에 대해|에대해|좀|줘|요|는|은|이|가|을|를|의|로|으로|에서|부터|까지|이랑|랑|하고|그리고|또는|이나|나|이든)', '', msg_lower).strip()
            if not clean_msg:
                clean_msg = msg_lower

            auto_results = search_knowledge(clean_msg)

            # dict 항목만 필터링 (비정상 데이터 방어)
            if auto_results:
                auto_results = [r for r in auto_results if isinstance(r, dict)]

            if auto_results:
                # 검색 성공 → 문서 기반 답변 생성 (기존 search_knowledge 핸들러 동일 로직)
                logger.info(f"✅ 자동 검색 성공: {[r['filename'] for r in auto_results[:3]]}")
                MAX_TOTAL_LENGTH = 15000 if LLM_MODE == "api" else 3000
                MAX_DOCS = 3  # ★ 상위 3개 문서까지만
                merged_docs = []
                total_length = 0
                doc_names = []
                top_score = auto_results[0].get("score", 100)

                for i, res_item in enumerate(auto_results[:MAX_DOCS]):
                    filename = res_item["filename"]
                    score = res_item.get("score", 0)
                    if i > 0 and score < top_score * 0.5:
                        break
                    doc_content = read_knowledge(filename)
                    if doc_content.startswith("❌"):
                        continue
                    remaining = MAX_TOTAL_LENGTH - total_length
                    if remaining <= 1000:
                        break
                    if len(doc_content) > remaining:
                        doc_content = doc_content[:remaining] + "\n\n... (문서 일부 생략)"
                    merged_docs.append(f"📄 **[{filename}]**\n{doc_content}")
                    doc_names.append(filename)
                    total_length += len(doc_content)

                if merged_docs:
                    combined_content = "\n\n---\n\n".join(merged_docs)
                    doc_list = ", ".join(doc_names)

                    follow_up_prompt = f"""아래 문서 내용을 읽고, 이 문서 내용만으로 질문에 답하세요.

=== 문서 시작 ({doc_list}) ===
{combined_content}
=== 문서 끝 ===

[질문] {user_message}

[규칙]
- 위 문서에 적힌 내용만 답하세요. 문서에 없는 내용은 절대 추가하지 마세요.
- 이전 대화에서 나온 내용이라도 문서에 없으면 사용하지 마세요.
- 문서에 없으면 "문서에 해당 내용이 없습니다"라고만 답하세요."""

                    result2_tokens = 4096 if LLM_MODE == "api" else 1024
                    result2 = call_llm(follow_up_prompt, KNOWLEDGE_QA_SYSTEM_PROMPT, max_tokens=result2_tokens, task_type="knowledge_qa")
                    if result2["success"]:
                        content2 = result2["content"].strip()
                        if content2 and not extract_tool_json(content2):
                            source_info = f"\n\n---\n📚 **참조 문서**: {doc_list}"
                            return content2 + source_info
            else:
                # ★ 검색 실패 → 역질문 유도
                logger.info(f"🔄 자동 검색 실패 → 역질문 유도")
                guide = generate_guided_questions(user_message)
                if guide["success"] and guide["suggestions"]:
                    lines = [f"🔍 **{guide['message']}**\n"]
                    for suggestion in guide["suggestions"]:
                        lines.append(f"<!--SUGGEST:{suggestion}-->")
                    lines.append(f"\n\n💡 위 추천 질문을 클릭하거나, 더 구체적인 키워드로 다시 질문해보세요.")
                    if guide.get("kb_files"):
                        lines.append(f"\n📚 현재 등록된 문서: {', '.join(guide['kb_files'][:5])}")
                    return "\n".join(lines)

        # 위 모든 경우에 해당하지 않으면 LLM 원래 응답 반환
        return text

    except Exception as e:
        logger.error(f"❌ 처리 오류: {e}")
        return f"❌ 오류: {e}"


# Fallback 포맷터
def format_tool_result_fallback(tool_data: dict, tool_result: str) -> str:
    tool_name = tool_data.get("tool", "")
    try:
        if tool_name == "get_system_info":
            info = json.loads(tool_result)
            lines = ["## 💻 시스템 정보", f"- **OS**: {info.get('os', '?')}", f"- **CPU**: {info.get('cpu', '?')}", f"- **메모리**: {info.get('memory', '?')}"]
            for d in info.get('drives', []):
                lines.append(f"- **{d['drive']}**: {d['total']} (사용률 {d['used']})")
            return "\n".join(lines)

        elif tool_name == "get_time":
            return f"🕐 현재 시간: {tool_result}"

        elif tool_name in ["search_files", "search_content"]:
            results = json.loads(tool_result)
            if not results:
                return f"🔍 '{tool_data.get('keyword', '')}' 검색 결과가 없습니다."
            lines = [f"🔍 검색 결과: **{len(results)}개** 발견\n"]
            for r in results[:10]:
                if "snippet" in r:
                    lines.append(f"- 📄 `{r['name']}` → {r['snippet']}")
                else:
                    lines.append(f"- {'📁' if r.get('type') == '폴더' else '📄'} `{r['name']}` ({r.get('size', '?')})")
            return "\n".join(lines)

        elif tool_name == "list_directory":
            items = json.loads(tool_result)
            lines = [f"📂 `{tool_data.get('path', '')}` 내용:\n"]
            for item in items[:20]:
                icon = "📁" if item.get("type") == "폴더" else "📄"
                lines.append(f"- {icon} `{item['name']}` ({item.get('size', '-')})")
            return "\n".join(lines)

        elif tool_name == "read_file":
            return f"📄 **파일 내용:**\n```\n{tool_result}\n```"

        elif tool_name in ["run_program", "kill_program", "open_web", "google_search"]:
            return f"✅ {tool_result}"

        elif tool_name == "list_processes":
            procs = json.loads(tool_result)
            if not procs or "error" in procs[0]:
                return "❌ 프로세스 목록을 가져올 수 없습니다."
            lines = [f"## 📋 실행 중인 프로세스 (상위 {len(procs)}개)\n"]
            lines.append("| 이름 | PID | 메모리(MB) | CPU% | 상태 |")
            lines.append("|------|-----|-----------|------|------|")
            for p in procs:
                lines.append(f"| {p['name']} | {p['pid']} | {p['memory_mb']} | {p['cpu']}% | {p['status']} |")
            return "\n".join(lines)

        elif tool_name == "analyze_data":
            return f"📊 **데이터 분석:**\n```\n{tool_result}\n```"

    except Exception as e:
        logger.error(f"포맷팅 오류: {e}")

    return f"📋 **결과:**\n```\n{tool_result}\n```"


# ========================================
# 대화 기록 (인코딩 저장 + 세션 관리)
# ========================================
def _encode_history(data: list) -> str:
    """대화 기록을 base64 인코딩하여 저장 (평문 노출 방지)"""
    raw = json.dumps(data, ensure_ascii=False)
    return base64.b64encode(raw.encode('utf-8')).decode('ascii')


def _decode_history(encoded: str) -> list:
    """base64 인코딩된 대화 기록 복원"""
    raw = base64.b64decode(encoded.encode('ascii')).decode('utf-8')
    return json.loads(raw)


def save_history():
    try:
        data_to_save = CHAT_HISTORY[-HISTORY_MAX:]
        encoded = _encode_history(data_to_save)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({"v": 2, "session": CURRENT_SESSION_ID, "data": encoded}, f)
    except (OSError, TypeError) as e:
        logger.warning(f"대화 기록 저장 실패: {e}")


def load_history():
    global CHAT_HISTORY
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # v2 인코딩 형식
            if isinstance(raw, dict) and raw.get("v") == 2:
                CHAT_HISTORY = _decode_history(raw["data"])
            # v1 레거시 (평문 리스트) → 자동 마이그레이션
            elif isinstance(raw, list):
                CHAT_HISTORY = raw
                save_history()  # v2로 재저장
                logger.info("대화 기록 v1→v2 마이그레이션 완료")
            else:
                CHAT_HISTORY = []
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"대화 기록 로드 실패 (초기화): {e}")
        CHAT_HISTORY = []


def save_session(session_name: str = None) -> dict:
    """현재 대화를 세션 파일로 저장"""
    global CURRENT_SESSION_ID
    if not CHAT_HISTORY:
        return {"success": False, "error": "저장할 대화가 없습니다."}

    if not session_name:
        session_name = CURRENT_SESSION_ID

    # 파일명 안전하게
    safe_name = re.sub(r'[^\w가-힣\-_]', '_', session_name)
    filepath = os.path.join(CHAT_SESSIONS_DIR, f"{safe_name}.json")

    try:
        encoded = _encode_history(CHAT_HISTORY)
        session_data = {
            "v": 2,
            "session_id": safe_name,
            "session_name": session_name,
            "created": datetime.datetime.now().isoformat(),
            "message_count": len(CHAT_HISTORY),
            "data": encoded
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False)
        return {"success": True, "session_id": safe_name, "message_count": len(CHAT_HISTORY)}
    except OSError as e:
        return {"success": False, "error": str(e)}


def load_session(session_id: str) -> dict:
    """저장된 세션 불러오기"""
    global CHAT_HISTORY, CURRENT_SESSION_ID
    filepath = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")

    if not os.path.exists(filepath):
        return {"success": False, "error": f"세션 없음: {session_id}"}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        CHAT_HISTORY = _decode_history(session_data["data"])
        CURRENT_SESSION_ID = session_id
        save_history()
        return {"success": True, "session_id": session_id, "message_count": len(CHAT_HISTORY)}
    except (OSError, json.JSONDecodeError, ValueError) as e:
        return {"success": False, "error": str(e)}


def list_sessions() -> list:
    """저장된 세션 목록"""
    sessions = []
    try:
        for f in sorted(os.listdir(CHAT_SESSIONS_DIR), reverse=True):
            if f.endswith('.json'):
                filepath = os.path.join(CHAT_SESSIONS_DIR, f)
                try:
                    with open(filepath, 'r', encoding='utf-8') as fh:
                        meta = json.load(fh)
                    sessions.append({
                        "session_id": f.replace('.json', ''),
                        "session_name": meta.get("session_name", f),
                        "created": meta.get("created", ""),
                        "message_count": meta.get("message_count", 0)
                    })
                except (json.JSONDecodeError, OSError):
                    continue
    except OSError:
        pass
    return sessions


def search_history(keyword: str, limit: int = 30) -> list:
    """대화 기록에서 키워드 검색"""
    results = []
    kw_lower = keyword.lower()
    for i, msg in enumerate(CHAT_HISTORY):
        if kw_lower in msg.get("content", "").lower():
            results.append({
                "index": i,
                "role": msg["role"],
                "content": msg["content"][:200],
                "time": msg.get("time", "")
            })
            if len(results) >= limit:
                break
    return results


def export_history(format_type: str = "json") -> dict:
    """대화 기록 내보내기 (json/txt/md)"""
    if not CHAT_HISTORY:
        return {"success": False, "error": "내보낼 대화가 없습니다."}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if format_type == "txt":
        lines = []
        for msg in CHAT_HISTORY:
            role = "사용자" if msg["role"] == "user" else "MAGI"
            time_str = msg.get("time", "")
            lines.append(f"[{role}] ({time_str})")
            lines.append(msg["content"])
            lines.append("")
        content = "\n".join(lines)
        filename = f"chat_export_{timestamp}.txt"

    elif format_type == "md":
        lines = [f"# MAGI 대화 기록\n", f"내보내기: {timestamp}\n", f"총 {len(CHAT_HISTORY)}개 메시지\n", "---\n"]
        for msg in CHAT_HISTORY:
            role = "**사용자**" if msg["role"] == "user" else "**MAGI**"
            time_str = msg.get("time", "")
            lines.append(f"### {role} ({time_str})\n")
            lines.append(msg["content"])
            lines.append("\n---\n")
        content = "\n".join(lines)
        filename = f"chat_export_{timestamp}.md"

    else:  # json
        content = json.dumps(CHAT_HISTORY, ensure_ascii=False, indent=2)
        filename = f"chat_export_{timestamp}.json"

    # 파일로 저장
    export_dir = os.path.join(BASE_DIR, "chat_exports")
    os.makedirs(export_dir, exist_ok=True)
    filepath = os.path.join(export_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "filename": filename, "path": filepath, "format": format_type}
    except OSError as e:
        return {"success": False, "error": str(e)}


# ========================================
# API Models
# ========================================
class FileData(BaseModel):
    file_name: str
    file_content: str   # base64 인코딩된 파일 내용
    file_type: str      # "csv" 또는 "xml"

class ChatRequest(BaseModel):
    message: str
    files: Optional[List[FileData]] = None  # 다중 파일 지원

class SearchRequest(BaseModel):
    keyword: str
    path: str = "C:/"
    file_content: bool = False

class EnvRequest(BaseModel):
    env: str

class ModelRequest(BaseModel):
    model_key: str


# ========================================
# Endpoints
# ========================================
def init_assistant():
    global LOCAL_LLM, LLM_MODE
    load_history()
    if load_api_token():
        LLM_MODE = "api"
        logger.info("✅ 비서: API 모드")
    else:
        # 서버(Coding_llm_server_v2)에서 이미 로드한 GGUF 모델 재사용
        try:
            import __main__ as server_main
            server_llm = getattr(server_main, 'LOCAL_LLM', None)
            if server_llm is not None:
                LOCAL_LLM = server_llm
                LLM_MODE = "local"
                logger.info("✅ 비서: LOCAL 모드 (서버 모델 공유)")
                return
        except (ImportError, AttributeError):
            pass
        LOCAL_LLM = load_local_model()
        if LOCAL_LLM:
            LLM_MODE = "local"
            logger.info("✅ 비서: LOCAL 모드")

    # ★ TF-IDF 인덱스 구축
    build_tfidf_index()


@router.get("/")
async def assistant_home():
    return FileResponse(os.path.join(BASE_DIR, "assistant_ui.html"))

@router.get("/magi.png")
async def magi_icon():
    return FileResponse(os.path.join(BASE_DIR, "magi.png"), media_type="image/png")

@router.get("/magi_f.png")
async def magi_f_icon():
    return FileResponse(os.path.join(BASE_DIR, "magi_f.png"), media_type="image/png")


# ★ 스크린샷 이미지 서빙
@router.get("/screenshots/{filename}")
async def serve_screenshot(filename: str):
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png")
    return {"error": "파일 없음"}


# ★ 리소스 파일 서빙 (HTML 구성도 등)
@router.get("/resources/{filename}")
async def serve_resource(filename: str):
    filepath = os.path.join(RESOURCES_DIR, filename)
    if os.path.exists(filepath):
        # 확장자별 MIME 타입
        ext = os.path.splitext(filename)[1].lower()
        mime_types = {
            ".html": "text/html",
            ".htm": "text/html",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
        }
        media_type = mime_types.get(ext, "application/octet-stream")
        return FileResponse(filepath, media_type=media_type)
    return JSONResponse(status_code=404, content={"error": "파일 없음"})


# ★ 스크린샷 비밀번호 인증 API
@router.post("/api/screenshot/auth")
async def screenshot_auth(data: dict):
    global screenshot_authenticated
    password = data.get("password", "")
    if password == SCREENSHOT_PASSWORD:
        screenshot_authenticated = True
        logger.info("🔓 스크린샷 인증 성공")
        return {"success": True, "message": "인증 성공"}
    else:
        logger.warning("🔒 스크린샷 인증 실패")
        return {"success": False, "message": "비밀번호가 틀렸습니다."}


# ★ 스크린샷 목록
@router.get("/api/screenshots")
async def list_screenshots():
    files = []
    if os.path.exists(SCREENSHOT_DIR):
        for f in sorted(os.listdir(SCREENSHOT_DIR), reverse=True)[:20]:
            if f.endswith('.png'):
                filepath = os.path.join(SCREENSHOT_DIR, f)
                size = os.path.getsize(filepath)
                files.append({
                    "filename": f,
                    "url": f"/assistant/screenshots/{f}",
                    "size": f"{size / 1024:.0f}KB",
                    "time": datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M:%S")
                })
    return {"screenshots": files}


@router.get("/api/status")
async def assistant_status():
    # 현재 로컬 모델 정보
    current_model_name = "LOCAL"
    if LLM_MODE == "local" and CURRENT_LOCAL_MODEL in AVAILABLE_MODELS:
        current_model_name = AVAILABLE_MODELS[CURRENT_LOCAL_MODEL]["name"]
    elif LLM_MODE != "local":
        current_model_name = ENV_CONFIG.get(CURRENT_ENV, {}).get("name", "API")

    return {
        "mode": LLM_MODE,
        "env": CURRENT_ENV if LLM_MODE != "local" else "local",
        "model_loaded": LOCAL_LLM is not None if LLM_MODE == "local" else API_TOKEN is not None,
        "model_name": current_model_name,
        "current_local_model": CURRENT_LOCAL_MODEL,
        "system": get_system_info(),
        "history_count": len(CHAT_HISTORY),
        "token_usage": TOKEN_USAGE
    }


# ★ 로컬 모델 목록 API
@router.get("/api/models")
async def list_local_models():
    """사용 가능한 로컬 GGUF 모델 목록"""
    return {
        "success": True,
        "models": get_available_local_models(),
        "current": CURRENT_LOCAL_MODEL
    }


# ★ 로컬 모델 변경 API
@router.post("/api/models/switch")
async def switch_local_model(request: ModelRequest):
    """로컬 GGUF 모델 변경"""
    global LOCAL_LLM, LLM_MODE

    model_key = request.model_key

    if model_key not in AVAILABLE_MODELS:
        return {"success": False, "error": f"알 수 없는 모델: {model_key}"}

    if not os.path.exists(AVAILABLE_MODELS[model_key]["path"]):
        return {"success": False, "error": f"모델 파일 없음: {AVAILABLE_MODELS[model_key]['name']}"}

    # 기존 모델 해제
    if LOCAL_LLM is not None:
        del LOCAL_LLM
        LOCAL_LLM = None
        import gc
        gc.collect()

    # 새 모델 로드
    LOCAL_LLM = load_local_model(model_key)
    if LOCAL_LLM:
        LLM_MODE = "local"
        return {
            "success": True,
            "model_key": model_key,
            "model_name": AVAILABLE_MODELS[model_key]["name"]
        }
    else:
        return {"success": False, "error": "모델 로드 실패"}


# ★ 토큰 사용량 API
@router.get("/api/tokens")
async def assistant_tokens():
    return {
        "success": True,
        "prompt_tokens": TOKEN_USAGE["prompt_tokens"],
        "completion_tokens": TOKEN_USAGE["completion_tokens"],
        "total_tokens": TOKEN_USAGE["total_tokens"],
        "call_count": TOKEN_USAGE["call_count"]
    }


@router.post("/api/tokens/reset")
async def assistant_reset_tokens():
    TOKEN_USAGE["prompt_tokens"] = 0
    TOKEN_USAGE["completion_tokens"] = 0
    TOKEN_USAGE["total_tokens"] = 0
    TOKEN_USAGE["call_count"] = 0
    return {"success": True, "message": "토큰 카운터 초기화됨"}


# ★ 파라미터 조회 API
@router.get("/api/params")
async def get_params():
    return {"success": True, "params": LLM_PARAMS}


# ★ 파라미터 변경 API
@router.post("/api/params")
async def set_params(request: dict):
    global LLM_PARAMS
    if "temperature" in request:
        LLM_PARAMS["temperature"] = float(request["temperature"])
    if "repeat_penalty" in request:
        LLM_PARAMS["repeat_penalty"] = float(request["repeat_penalty"])
    if "max_tokens" in request:
        LLM_PARAMS["max_tokens"] = int(request["max_tokens"])
    logger.info(f"⚙️ 파라미터 변경: {LLM_PARAMS}")
    return {"success": True, "params": LLM_PARAMS}


# ★ 태스크별 자동 분류 프로필 API
@router.get("/api/task_profiles")
async def get_task_profiles():
    return {
        "success": True,
        "auto_mode": LLM_PARAMS.get("task_auto_mode", False),
        "profiles": TASK_PARAM_PROFILES
    }


@router.post("/api/task_profiles")
async def set_task_profiles(request: dict):
    global TASK_PARAM_PROFILES
    if "auto_mode" in request:
        LLM_PARAMS["task_auto_mode"] = bool(request["auto_mode"])
    if "profiles" in request:
        for key, val in request["profiles"].items():
            if key in TASK_PARAM_PROFILES and "temperature" in val:
                TASK_PARAM_PROFILES[key]["temperature"] = float(val["temperature"])
    logger.info(f"🎯 태스크 자동 분류: {'ON' if LLM_PARAMS.get('task_auto_mode') else 'OFF'} | {TASK_PARAM_PROFILES}")
    return {
        "success": True,
        "auto_mode": LLM_PARAMS.get("task_auto_mode", False),
        "profiles": TASK_PARAM_PROFILES
    }


@router.post("/api/set_env")
async def assistant_set_env(request: EnvRequest):
    global LLM_MODE, LOCAL_LLM, CURRENT_ENV, API_URL, API_MODEL
    env = request.env.lower()

    if env == "local":
        if LOCAL_LLM is None:
            LOCAL_LLM = load_local_model()
        if LOCAL_LLM:
            LLM_MODE = "local"
            return {"success": True, "env": "local", "name": "LOCAL(14B-GGUF)"}
        return {"success": False, "error": "로컬 모델 로드 실패"}

    elif env in ENV_CONFIG:
        if not API_TOKEN and not load_api_token():
            return {"success": False, "error": "API 토큰 없음"}
        LLM_MODE = "api"
        CURRENT_ENV = env
        API_URL = ENV_CONFIG[env]["url"]
        API_MODEL = ENV_CONFIG[env]["model"]
        return {"success": True, "env": env, "name": ENV_CONFIG[env]["name"]}

    return {"success": False, "error": f"알 수 없는 환경: {env}"}


@router.post("/api/chat")
async def assistant_chat(request: ChatRequest):
    user_msg = request.message.strip()
    CHAT_HISTORY.append({"role": "user", "content": user_msg, "time": datetime.datetime.now().isoformat()})
    response = process_chat(user_msg, files=request.files)
    # ★ 자동 리소스 첨부 (AMHS 관련 질문 시 구성도 링크 등)
    if not request.files:
        response = auto_attach_resources(user_msg, response)
    CHAT_HISTORY.append({"role": "assistant", "content": response, "time": datetime.datetime.now().isoformat()})
    save_history()
    return {"success": True, "response": response}


@router.post("/api/chat/stream")
async def assistant_chat_stream(request: ChatRequest):
    """SSE 스트리밍 채팅 - 진행 단계 + 타이핑 효과"""
    user_msg = request.message.strip()
    files_to_send = request.files
    CHAT_HISTORY.append({"role": "user", "content": user_msg, "time": datetime.datetime.now().isoformat()})

    async def event_generator():
        # 1단계: 생각 중
        status_msg = '📊 AMHS 데이터 분석 중...' if files_to_send else '생각 중...'
        yield f"data: {json.dumps({'type': 'status', 'message': status_msg}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.1)

        # 2단계: LLM 호출 (동기 함수를 비동기로 실행)
        loop = asyncio.get_event_loop()
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': '응답 생성 중...'}, ensure_ascii=False)}\n\n"
            response = await loop.run_in_executor(None, lambda: process_chat(user_msg, files=files_to_send))
            if not files_to_send:
                response = auto_attach_resources(user_msg, response)

            # 3단계: 응답을 청크로 나눠서 전송 (타이핑 효과)
            yield f"data: {json.dumps({'type': 'status', 'message': '답변 작성 중...'}, ensure_ascii=False)}\n\n"

            # 응답을 문장 단위로 분할하여 스트리밍
            chunks = []
            current = ""
            for char in response:
                current += char
                if char in '.!?\n' and len(current) >= 10:
                    chunks.append(current)
                    current = ""
            if current:
                chunks.append(current)

            if not chunks:
                chunks = [response]

            sent_content = []
            stream_cancelled = False
            for chunk in chunks:
                if GENERATION_CANCELLED:
                    stream_cancelled = True
                    logger.info("🛑 스트리밍 전송 중지됨 (사용자 요청)")
                    break
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
                sent_content.append(chunk)
                await asyncio.sleep(0.03)

            # 4단계: 완료
            final_content = "".join(sent_content)
            if stream_cancelled:
                final_content += "\n\n🛑 *응답이 중지되었습니다.*"
                stop_msg = json.dumps({"type": "chunk", "content": "\n\n🛑 *응답이 중지되었습니다.*"}, ensure_ascii=False)
                yield f"data: {stop_msg}\n\n"
            CHAT_HISTORY.append({"role": "assistant", "content": final_content, "time": datetime.datetime.now().isoformat()})
            save_history()
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"스트리밍 채팅 오류: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/search")
async def assistant_search(request: SearchRequest):
    if request.file_content:
        results = search_content(request.keyword, request.path)
    else:
        results = search_files(request.keyword, request.path)
    return {"success": True, "results": results, "count": len(results)}


@router.get("/api/drives")
async def assistant_drives():
    drives = []
    for p in psutil.disk_partitions():
        drives.append({"device": p.device, "mountpoint": p.mountpoint})
    return {"success": True, "drives": drives}


@router.get("/api/history")
async def assistant_get_history(limit: int = 50):
    return {"history": CHAT_HISTORY[-limit:], "total": len(CHAT_HISTORY), "session": CURRENT_SESSION_ID}


@router.delete("/api/history")
async def assistant_clear_history():
    global CHAT_HISTORY, CURRENT_SESSION_ID, _conversation_summary_cache, AMHS_SESSION_DATA
    CHAT_HISTORY = []
    CURRENT_SESSION_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # ★ 대화 요약 캐시도 초기화 (할루시네이션 전파 방지)
    _conversation_summary_cache = {"summary": "", "summarized_up_to": 0}
    # ★ AMHS 세션 데이터 초기화
    AMHS_SESSION_DATA = {"active": False, "files": [], "system_prompt": "", "equipment_type": "", "last_analysis_response": ""}
    save_history()
    return {"success": True}


# ★ LLM 생성 중지 API
@router.post("/api/stop")
async def assistant_stop_generation():
    """LLM 생성을 중지합니다."""
    global GENERATION_CANCELLED
    GENERATION_CANCELLED = True
    logger.info("🛑 LLM 생성 중지 요청 수신")
    return {"success": True, "message": "생성 중지 요청됨"}


# ★ 대화 검색
@router.get("/api/history/search")
async def assistant_search_history(keyword: str, limit: int = 30):
    """대화 기록에서 키워드 검색"""
    results = search_history(keyword, limit)
    return {"success": True, "results": results, "count": len(results)}


# ★ 대화 내보내기
@router.get("/api/history/export")
async def assistant_export_history(format: str = "json"):
    """대화 기록 내보내기 (json/txt/md)"""
    if format not in ("json", "txt", "md"):
        return {"success": False, "error": "지원 형식: json, txt, md"}
    result = export_history(format)
    if result["success"]:
        return FileResponse(result["path"], filename=result["filename"], media_type="application/octet-stream")
    return result


# ★ 히스토리 최대 건수 설정
@router.post("/api/history/config")
async def assistant_history_config(data: dict):
    """히스토리 최대 건수 변경"""
    global HISTORY_MAX
    new_max = data.get("max", HISTORY_MAX)
    if isinstance(new_max, int) and 50 <= new_max <= 2000:
        HISTORY_MAX = new_max
        return {"success": True, "max": HISTORY_MAX}
    return {"success": False, "error": "50~2000 범위만 가능"}


# ★ 세션 관리 API
@router.get("/api/sessions")
async def assistant_list_sessions():
    """저장된 세션 목록"""
    sessions = list_sessions()
    return {"success": True, "sessions": sessions, "current": CURRENT_SESSION_ID}


@router.post("/api/sessions/save")
async def assistant_save_session(data: dict = None):
    """현재 대화를 세션으로 저장"""
    name = data.get("name", "") if data else ""
    result = save_session(name if name else None)
    return result


@router.post("/api/sessions/load")
async def assistant_load_session(data: dict):
    """저장된 세션 불러오기"""
    session_id = data.get("session_id", "")
    if not session_id:
        return {"success": False, "error": "session_id 필요"}
    result = load_session(session_id)
    return result


@router.delete("/api/sessions/{session_id}")
async def assistant_delete_session(session_id: str):
    """세션 삭제"""
    filepath = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return {"success": True, "message": f"세션 '{session_id}' 삭제됨"}
    return {"success": False, "error": "세션 없음"}


# ★ 지식베이스 역질문 추천 API
@router.post("/api/knowledge/suggest")
async def api_suggest_questions(request: dict):
    """지식베이스 검색 실패 시 역질문 추천"""
    query = request.get("query", "")
    if not query:
        return {"success": False, "error": "검색어가 비어있습니다."}
    guide = generate_guided_questions(query)
    return guide


# ★ 지식베이스 API
@router.get("/api/knowledge")
async def api_list_knowledge():
    """지식베이스 문서 목록"""
    files = list_knowledge()
    return {"success": True, "files": files, "count": len(files)}


@router.post("/api/knowledge/upload")
async def api_upload_knowledge(file: UploadFile = File(...)):
    """지식 문서 업로드 (MD/TXT/PDF/Excel 지원, 버전 관리)"""
    allowed_ext = ('.md', '.txt', '.pdf', '.xlsx', '.xls', '.csv')
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_ext:
        return {"success": False, "error": f"지원 형식: {', '.join(allowed_ext)}"}

    try:
        content = await file.read()
        target_filename = file.filename

        # PDF/Excel → MD 변환
        if ext == '.pdf':
            try:
                import tempfile as _tf
                with _tf.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(tmp_path)
                    text_parts = []
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(f"## 페이지 {i+1}\n\n{page_text}")
                    md_content = f"# {file.filename}\n\n" + "\n\n".join(text_parts)
                except ImportError:
                    return {"success": False, "error": "PyPDF2 미설치. pip install PyPDF2"}
                finally:
                    os.unlink(tmp_path)
                target_filename = os.path.splitext(file.filename)[0] + '.md'
                content = md_content.encode('utf-8')
            except Exception as e:
                return {"success": False, "error": f"PDF 변환 실패: {e}"}

        elif ext in ('.xlsx', '.xls', '.csv'):
            try:
                import tempfile as _tf
                import io
                with _tf.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    if ext == '.csv':
                        df = pd.read_csv(tmp_path, encoding='utf-8', errors='ignore')
                    else:
                        df = pd.read_excel(tmp_path)
                    md_lines = [f"# {file.filename}\n"]
                    md_lines.append(f"- 행: {len(df):,}개, 열: {len(df.columns)}개\n")
                    md_lines.append("## 컬럼 목록\n")
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        md_lines.append(f"- **{col}** ({dtype})")
                    md_lines.append("\n## 데이터 미리보기 (상위 20행)\n")
                    md_lines.append(df.head(20).to_markdown(index=False))
                    numeric_cols = df.select_dtypes(include='number').columns
                    if len(numeric_cols) > 0:
                        md_lines.append("\n## 수치 통계\n")
                        md_lines.append(df[numeric_cols].describe().to_markdown())
                    md_content = "\n".join(md_lines)
                except ImportError:
                    return {"success": False, "error": "pandas 미설치. pip install pandas openpyxl tabulate"}
                finally:
                    os.unlink(tmp_path)
                target_filename = os.path.splitext(file.filename)[0] + '.md'
                content = md_content.encode('utf-8')
            except Exception as e:
                return {"success": False, "error": f"Excel/CSV 변환 실패: {e}"}

        # ★ 버전 관리: 기존 파일이 있으면 버전 백업
        filepath = os.path.join(KNOWLEDGE_DIR, target_filename)
        version_info = None
        if os.path.exists(filepath):
            ver_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base, fext = os.path.splitext(target_filename)
            backup_name = f"{base}_v{ver_ts}{fext}"
            backup_path = os.path.join(KNOWLEDGE_ARCHIVE_DIR, backup_name)
            shutil.copy2(filepath, backup_path)
            version_info = {"backup": backup_name, "message": "기존 버전이 과거지식에 백업됨"}
            logger.info(f"지식문서 버전 백업: {target_filename} → {backup_name}")

        with open(filepath, 'wb') as f:
            f.write(content)

        # ★ TF-IDF 인덱스 재구축
        build_tfidf_index()

        result = {"success": True, "filename": target_filename, "size": len(content),
                  "original": file.filename, "converted": ext != os.path.splitext(target_filename)[1]}
        if version_info:
            result["version"] = version_info
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/api/knowledge/{filename}")
async def api_delete_knowledge(filename: str):
    """지식베이스 문서 삭제 (자동으로 과거지식에 백업)"""
    filepath = os.path.join(KNOWLEDGE_DIR, filename)
    if os.path.exists(filepath):
        # 삭제 전 자동 백업
        ver_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(filename)
        backup_name = f"{base}_del{ver_ts}{ext}"
        backup_path = os.path.join(KNOWLEDGE_ARCHIVE_DIR, backup_name)
        try:
            shutil.copy2(filepath, backup_path)
        except OSError:
            pass
        os.remove(filepath)
        # ★ TF-IDF 인덱스 재구축
        build_tfidf_index()
        return {"success": True, "message": f"'{filename}' 삭제됨 (백업: {backup_name})"}
    return {"success": False, "error": "파일 없음"}


@router.get("/api/knowledge/download/{filename}")
async def api_download_knowledge(filename: str):
    """지식베이스 문서 다운로드"""
    filepath = os.path.join(KNOWLEDGE_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, filename=filename, media_type="application/octet-stream")
    return JSONResponse(status_code=404, content={"error": "파일 없음"})


# ========================================
# 과거지식 보관소 API
# ========================================
@router.get("/api/knowledge/archive")
async def api_list_archive():
    """과거지식 문서 목록"""
    files = []
    try:
        for f in sorted(os.listdir(KNOWLEDGE_ARCHIVE_DIR)):
            if f.lower().endswith(('.md', '.txt')):
                filepath = os.path.join(KNOWLEDGE_ARCHIVE_DIR, f)
                size = os.path.getsize(filepath)
                modified = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M")
                size_str = f"{size / 1024:.1f}KB" if size > 1024 else f"{size}B"
                files.append({"filename": f, "size": size_str, "modified": modified})
    except Exception as e:
        logger.error(f"과거지식 목록 오류: {e}")
    return {"success": True, "files": files, "count": len(files)}


@router.post("/api/knowledge/archive/{filename}")
async def api_archive_knowledge(filename: str):
    """지식베이스 → 과거지식으로 이동"""
    src = os.path.join(KNOWLEDGE_DIR, filename)
    dst = os.path.join(KNOWLEDGE_ARCHIVE_DIR, filename)
    if not os.path.exists(src):
        return {"success": False, "error": f"'{filename}' 파일 없음"}
    try:
        shutil.move(src, dst)
        return {"success": True, "message": f"'{filename}' → 과거지식으로 이동됨"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/api/knowledge/restore/{filename}")
async def api_restore_knowledge(filename: str):
    """과거지식 → 지식베이스로 복원"""
    src = os.path.join(KNOWLEDGE_ARCHIVE_DIR, filename)
    dst = os.path.join(KNOWLEDGE_DIR, filename)
    if not os.path.exists(src):
        return {"success": False, "error": f"'{filename}' 파일 없음"}
    try:
        shutil.move(src, dst)
        return {"success": True, "message": f"'{filename}' → 지식베이스로 복원됨"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/api/knowledge/archive/{filename}")
async def api_delete_archive(filename: str):
    """과거지식 문서 완전 삭제"""
    filepath = os.path.join(KNOWLEDGE_ARCHIVE_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return {"success": True, "message": f"'{filename}' 완전 삭제됨"}
    return {"success": False, "error": "파일 없음"}


# ★ 지식문서 버전 히스토리 조회
@router.get("/api/knowledge/versions/{filename}")
async def api_knowledge_versions(filename: str):
    """특정 문서의 버전 히스토리 (아카이브에서 같은 이름의 백업 찾기)"""
    base = os.path.splitext(filename)[0]
    versions = []
    try:
        for f in sorted(os.listdir(KNOWLEDGE_ARCHIVE_DIR), reverse=True):
            if f.startswith(base) and f != filename and not f.endswith(".meta"):
                filepath = os.path.join(KNOWLEDGE_ARCHIVE_DIR, f)
                size = os.path.getsize(filepath)
                modified = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M")
                # 메모 메타데이터 읽기
                memo = ""
                meta_path = filepath + ".meta"
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as mf:
                            meta = json.load(mf)
                            memo = meta.get("memo", "")
                    except Exception:
                        pass
                versions.append({"filename": f, "size": f"{size / 1024:.1f}KB", "modified": modified, "memo": memo})
    except OSError:
        pass
    return {"success": True, "document": filename, "versions": versions, "count": len(versions)}


@router.post("/api/knowledge/versions/{filename}")
async def api_create_version(filename: str, data: dict = None):
    """현재 문서의 스냅샷을 수동으로 버전 저장"""
    filepath = os.path.join(KNOWLEDGE_DIR, filename)
    if not os.path.exists(filepath):
        return {"success": False, "error": f"파일 없음: {filename}"}

    memo = data.get("memo", "") if data else ""
    ver_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(filename)
    backup_name = f"{base}_v{ver_ts}{ext}"
    backup_path = os.path.join(KNOWLEDGE_ARCHIVE_DIR, backup_name)

    try:
        shutil.copy2(filepath, backup_path)
        # 메모가 있으면 메타 파일로 저장
        if memo:
            meta_path = backup_path + ".meta"
            with open(meta_path, 'w', encoding='utf-8') as mf:
                json.dump({"memo": memo, "created": ver_ts, "source": filename}, mf, ensure_ascii=False)
        logger.info(f"📋 수동 버전 저장: {filename} → {backup_name} (메모: {memo})")
        return {"success": True, "version": backup_name, "memo": memo}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========================================
# ★ 설비별 프롬프트 편집 API (AMHS)
# ========================================
@router.get("/api/equip_prompts")
async def api_get_equip_prompts():
    """설비별 프롬프트 전체 조회"""
    prompts = {}
    # 공통 프롬프트
    common_path = os.path.join(EQUIP_PROMPT_DIR, "BASE", "common.txt")
    if os.path.exists(common_path):
        with open(common_path, 'r', encoding='utf-8') as f:
            prompts['common'] = f.read()

    # 설비별 프롬프트
    for equip in ['OHT', 'CONVEYOR', 'LIFTER', 'FABJOB']:
        equip_dir = os.path.join(EQUIP_PROMPT_DIR, equip)
        if os.path.isdir(equip_dir):
            for ptype in ['system', 'fewshot']:
                fpath = os.path.join(equip_dir, f"{ptype}.txt")
                if os.path.exists(fpath):
                    with open(fpath, 'r', encoding='utf-8') as f:
                        prompts[f"{equip}_{ptype}"] = f.read()

    return {"success": True, "prompts": prompts}


@router.post("/api/save_equip_prompt")
async def api_save_equip_prompt(data: dict):
    """설비별 프롬프트 저장"""
    equip_type = data.get('equipment_type', '')
    prompt_type = data.get('prompt_type', '')
    content = data.get('content', '')

    if equip_type == 'BASE' or prompt_type == 'common':
        target_path = os.path.join(EQUIP_PROMPT_DIR, "BASE", "common.txt")
    else:
        target_dir = os.path.join(EQUIP_PROMPT_DIR, equip_type)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, f"{prompt_type}.txt")

    try:
        # 백업
        if os.path.exists(target_path):
            backup_path = target_path + f".bak_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(target_path, backup_path)

        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"📝 설비 프롬프트 저장: {equip_type}/{prompt_type}")
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/api/reset_equip_prompt")
async def api_reset_equip_prompt(data: dict):
    """설비별 프롬프트 초기화 (백업에서 복원 또는 빈 파일)"""
    equip_type = data.get('equipment_type', '')
    prompt_type = data.get('prompt_type', '')

    if equip_type == 'BASE' or prompt_type == 'common':
        target_path = os.path.join(EQUIP_PROMPT_DIR, "BASE", "common.txt")
    else:
        target_path = os.path.join(EQUIP_PROMPT_DIR, equip_type, f"{prompt_type}.txt")

    try:
        # 기존 백업 중 가장 오래된 것(원본)으로 복원
        bak_files = sorted([f for f in os.listdir(os.path.dirname(target_path))
                           if f.startswith(os.path.basename(target_path) + ".bak_")])
        if bak_files:
            original_bak = os.path.join(os.path.dirname(target_path), bak_files[0])
            shutil.copy2(original_bak, target_path)
        else:
            # 백업 없으면 빈 파일
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write('')
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========================================
# ★ 시스템 프롬프트 & 도메인 지식 편집 API
# ========================================
@router.get("/api/prompt/system")
async def api_get_system_prompt():
    """시스템 프롬프트 조회"""
    content = load_prompt_file(SYSTEM_PROMPT_FILE, DEFAULT_SYSTEM_PROMPT)
    return {
        "success": True,
        "content": content,
        "filepath": SYSTEM_PROMPT_FILE,
        "char_count": len(content)
    }


@router.post("/api/prompt/system")
async def api_save_system_prompt(request: dict):
    """시스템 프롬프트 저장"""
    content = request.get("content", "")
    if not content.strip():
        return {"success": False, "error": "내용이 비어있습니다"}
    success = save_prompt_file(SYSTEM_PROMPT_FILE, content)
    if success:
        return {"success": True, "message": "시스템 프롬프트 저장 완료", "char_count": len(content)}
    return {"success": False, "error": "저장 실패"}


@router.post("/api/prompt/system/reset")
async def api_reset_system_prompt():
    """시스템 프롬프트 기본값 복원"""
    success = save_prompt_file(SYSTEM_PROMPT_FILE, DEFAULT_SYSTEM_PROMPT)
    if success:
        return {"success": True, "message": "기본값으로 복원됨", "content": DEFAULT_SYSTEM_PROMPT}
    return {"success": False, "error": "복원 실패"}


@router.get("/api/prompt/domain")
async def api_get_domain_knowledge():
    """도메인 지식 조회"""
    content = load_prompt_file(DOMAIN_KNOWLEDGE_FILE, DEFAULT_DOMAIN_KNOWLEDGE)
    active_lines = [l for l in content.strip().split('\n')
                    if l.strip() and not l.strip().startswith('#')]
    return {
        "success": True,
        "content": content,
        "filepath": DOMAIN_KNOWLEDGE_FILE,
        "char_count": len(content),
        "active_lines": len(active_lines)
    }


@router.post("/api/prompt/domain")
async def api_save_domain_knowledge(request: dict):
    """도메인 지식 저장"""
    content = request.get("content", "")
    success = save_prompt_file(DOMAIN_KNOWLEDGE_FILE, content)
    if success:
        active_lines = [l for l in content.strip().split('\n')
                        if l.strip() and not l.strip().startswith('#')]
        return {
            "success": True,
            "message": "도메인 지식 저장 완료",
            "char_count": len(content),
            "active_lines": len(active_lines)
        }
    return {"success": False, "error": "저장 실패"}


@router.post("/api/prompt/domain/reset")
async def api_reset_domain_knowledge():
    """도메인 지식 기본 템플릿 복원"""
    success = save_prompt_file(DOMAIN_KNOWLEDGE_FILE, DEFAULT_DOMAIN_KNOWLEDGE)
    if success:
        return {"success": True, "message": "기본 템플릿으로 복원됨", "content": DEFAULT_DOMAIN_KNOWLEDGE}
    return {"success": False, "error": "복원 실패"}


@router.get("/api/prompt/preview")
async def api_preview_effective_prompt():
    """현재 합성된 최종 시스템 프롬프트 미리보기"""
    effective = get_effective_system_prompt()
    return {
        "success": True,
        "effective_prompt": effective,
        "total_chars": len(effective)
    }


# ★ 마기(main1_First) 코딩 에이전트 라우터 연결 (모듈 로드 시점에 등록)
try:
    from coding_agent import agent_router
    app.include_router(agent_router)
    print("⚡ 마기(main1_First) 코딩 에이전트 라우터 연결 완료")
except ImportError as e:
    print(f"⚠️ coding_agent.py 없음 → 마기(main1_First) 모드 비활성: {e}")
except Exception as e:
    print(f"⚠️ 마기(main1_First) 로드 오류: {e}")


if __name__ == "__main__":
    import uvicorn
    app.include_router(router)

    @app.on_event("startup")
    async def standalone_startup():
        init_assistant()

    uvicorn.run(app, host="0.0.0.0", port=10002)
