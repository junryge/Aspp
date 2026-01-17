#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M14 히스토리 데이터 LLM 분석 서버 (v1.0)
- M14 히스토리 데이터 로드
- LLM을 통한 데이터 분석
- SK Hynix API + 로컬 GGUF 지원
"""

import os
import re
import requests
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from io import StringIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="M14 LLM Analysis")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Global Variables
# ========================================
API_TOKEN = None
llm = None  # 로컬 LLM (GGUF)

# LLM 모드: "api" 또는 "local"
LLM_MODE = "local"

# 환경 설정
ENV_MODE = "dev"

# 로컬 모델 경로
LOCAL_MODEL_PATH = "Qwen3-14B-Q4_K_M.gguf"

# M14 데이터 폴더 경로 (나중에 설정 가능)
M14_DATA_DIR = "../NODE/UI/data"

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
# 시스템 프롬프트 (편집 가능)
# ========================================
SYSTEM_PROMPT_ANALYSIS = """당신은 M14 반송 큐 예측 모델 분석 전문가입니다.

## 분석 원칙
1. **결론 먼저** - 핵심 수치와 판단을 먼저 제시
2. **간결하게** - 불필요한 설명 없이 핵심만
3. **숫자 중심** - 정확한 수치와 비율 제시
4. **문제점 명확히** - 무엇이 문제인지 직접적으로
5. **개선안 구체적** - 실행 가능한 제안

## 출력 형식
- 테이블은 마크다운 형식, 최소한의 컬럼만
- 이모지 적절히 사용 (📊 ✅ ⚠️ 🔴 💡)
- 섹션은 ## 헤더로 구분
- 긴 설명 대신 핵심 bullet point

## M14 기준값
- 위험 임계값: TOTALCNT >= 1700
- 경고 임계값: TOTALCNT >= 1600
- 예측 모델: XGB_타겟, XGB_중요, XGB_보조, XGB_PDT, XGB_Job
- 성능 지표: TP(적중), TN(정상예측), FP(오탐), FN(놓침)"""

# 분석 유형별 프롬프트 템플릿
PROMPT_TEMPLATES = {
    "summary": """위 데이터를 분석해주세요.

다음 형식으로 출력:
## 📊 핵심 요약
- 정확도, 재현율, 정밀도 (테이블)
- TP/TN/FP/FN 수치
  - TP(적중): 1700+ 예측 → 실제 1700+
  - TN(정상): 정상 예측 → 실제 정상
  - FP(오탐): 1700+ 예측 → 실제 정상
  - FN(놓침): 정상 예측 → 실제 1700+

## ✅ 좋은 점
- (1-2개)

## ⚠️ 문제점
- 핵심 문제 (수치 포함)
- 원인 분석

## 💡 개선 제안
- 구체적 조치 (수치 기준 포함)""",

    "pattern": """시간대별/일별 패턴을 분석해주세요.

다음 형식으로 출력:
## 📈 패턴 분석
- 피크 시간대 (몇시, 평균값)
- 안정 시간대 (몇시, 평균값)

## ⚠️ 주의 시간대
- 1700+ 자주 발생하는 시간
- 패턴의 원인 추정

## 💡 운영 제안
- 피크 시간 대응 방안
- 모니터링 강화 시간대""",

    "prediction": """예측 모델 성능을 분석해주세요.

다음 형식으로 출력:
## 🎯 예측 성능
| 지표 | 값 |
- 정확도, 재현율, 정밀도

## ⚠️ 예측 문제점
- FN (놓침): 원인 분석
- FP (오탐): 원인 분석

## 💡 모델 개선 방향
- threshold 조정 제안
- 재학습 필요 여부"""
}


# ========================================
# API Token 관리
# ========================================
def load_api_token():
    """API 토큰 로드"""
    global API_TOKEN
    
    token_paths = ["token.txt", "../token.txt", os.path.expanduser("~/token.txt")]
    
    for token_path in token_paths:
        if os.path.exists(token_path):
            try:
                with open(token_path, "r", encoding='utf-8') as f:
                    API_TOKEN = f.read().strip()
                if API_TOKEN and "REPLACE" not in API_TOKEN:
                    logger.info(f"✅ API 토큰 로드 완료: {token_path}")
                    return True
            except Exception as e:
                logger.error(f"❌ 토큰 로드 실패: {e}")
    
    logger.warning("⚠️ 토큰 파일 없음")
    return False


# ========================================
# 로컬 LLM 로드 (GGUF)
# ========================================
def load_local_llm():
    """로컬 GGUF 모델 로드"""
    global llm
    
    # 여러 경로에서 모델 찾기
    model_paths = [
        LOCAL_MODEL_PATH,
        f"../{LOCAL_MODEL_PATH}",
        f"../../{LOCAL_MODEL_PATH}",
        os.path.expanduser(f"~/{LOCAL_MODEL_PATH}")
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        logger.warning(f"⚠️ 로컬 모델 없음: {LOCAL_MODEL_PATH}")
        return False
    
    try:
        from llama_cpp import Llama
        logger.info(f"📦 로컬 LLM 로딩 중: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=32768,
            n_gpu_layers=-1,
            verbose=False
        )
        logger.info("✅ 로컬 LLM 로드 완료!")
        return True
    except ImportError:
        logger.warning("⚠️ llama-cpp-python 미설치")
        return False
    except Exception as e:
        logger.error(f"❌ 로컬 LLM 로드 실패: {e}")
        return False


# ========================================
# LLM 호출 함수들
# ========================================
def call_local_llm(prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> dict:
    """로컬 GGUF 모델 호출"""
    global llm
    
    if llm is None:
        return {"success": False, "error": "로컬 LLM이 로드되지 않았습니다."}
    
    try:
        if not system_prompt:
            system_prompt = SYSTEM_PROMPT_ANALYSIS
        
        formatted_prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
        
        logger.info("🖥️ 로컬 LLM 호출 중...")
        response = llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            stop=["<|im_end|>", "\n\n\n"]
        )
        
        content = response['choices'][0]['text'].strip()
        
        # <think> 태그 제거
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        content = content.strip()
        
        return {"success": True, "content": content}
        
    except Exception as e:
        logger.error(f"❌ 로컬 LLM 호출 실패: {e}")
        return {"success": False, "error": f"로컬 LLM 호출 실패: {str(e)}"}


def stream_local_llm(prompt: str, system_prompt: str = "", max_tokens: int = 2000):
    """로컬 LLM 스트리밍 호출 (Generator)"""
    global llm
    
    if llm is None:
        yield "data: [ERROR] 로컬 LLM이 로드되지 않았습니다.\n\n"
        return
    
    try:
        formatted_prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
        
        logger.info("🖥️ 로컬 LLM 스트리밍 시작...")
        
        in_think = False
        buffer = ""
        
        for chunk in llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            stop=["<|im_end|>", "\n\n\n"],
            stream=True
        ):
            token = chunk['choices'][0]['text']
            buffer += token
            
            # <think> 태그 처리
            if '<think>' in buffer and not in_think:
                in_think = True
                # <think> 이전 텍스트 전송
                before_think = buffer.split('<think>')[0]
                if before_think:
                    safe_before = before_think.replace('\n', '⏎')
                    yield f"data: {safe_before}\n\n"
                buffer = buffer.split('<think>', 1)[1] if '<think>' in buffer else ""
                continue
            
            if in_think:
                if '</think>' in buffer:
                    in_think = False
                    buffer = buffer.split('</think>', 1)[1] if '</think>' in buffer else ""
                continue
            
            # 정상 토큰 전송 (줄바꿈을 특수 문자로 치환)
            if token and not in_think:
                safe_token = token.replace('\n', '⏎')
                yield f"data: {safe_token}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"❌ 스트리밍 오류: {e}")
        yield f"data: [ERROR] {str(e)}\n\n"


def call_llm_api(prompt: str, system_prompt: str = "", max_tokens: int = 4000) -> dict:
    """LLM API 호출"""
    global API_TOKEN
    
    if not API_TOKEN:
        return {"success": False, "error": "API 토큰이 없습니다."}
    
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
        logger.info(f"🚀 API 호출 중...")
        response = requests.post(API_URL, headers=headers, json=data, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            return {"success": True, "content": content.strip()}
        else:
            return {"success": False, "error": f"API 오류: {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": f"API 호출 실패: {str(e)}"}


def call_llm(prompt: str, system_prompt: str = "", max_tokens: int = 4000) -> dict:
    """통합 LLM 호출"""
    global LLM_MODE
    
    if LLM_MODE == "api":
        result = call_llm_api(prompt, system_prompt, max_tokens)
        if not result["success"] and llm is not None:
            logger.info("⚠️ API 실패 → 로컬 폴백")
            result = call_local_llm(prompt, system_prompt, min(max_tokens, 2000))
        return result
    else:
        if llm is None:
            return {"success": False, "error": "로컬 LLM이 로드되지 않았습니다."}
        return call_local_llm(prompt, system_prompt, min(max_tokens, 2000))


# ========================================
# M14 데이터 로드 함수
# ========================================
def load_m14_data(date_str: str) -> dict:
    """M14 데이터 로드"""
    data_file = os.path.join(M14_DATA_DIR, f'm14_data_{date_str}.csv')
    pred_file = os.path.join(M14_DATA_DIR, f'm14_pred_{date_str}.csv')
    alert_file = os.path.join(M14_DATA_DIR, f'm14_alert_{date_str}.csv')
    
    result = {
        "date": date_str,
        "data": None,
        "alerts_10": [],
        "alerts_30": [],
        "stats": {}
    }
    
    # 데이터 파일 로드
    if os.path.exists(pred_file):
        try:
            df = pd.read_csv(pred_file)
            result["data"] = df.to_dict('records')
            
            # 기본 통계
            if 'TOTALCNT' in df.columns:
                result["stats"] = {
                    "count": len(df),
                    "avg": round(df['TOTALCNT'].mean(), 1),
                    "max": int(df['TOTALCNT'].max()),
                    "min": int(df['TOTALCNT'].min()),
                    "over_1700": int((df['TOTALCNT'] >= 1700).sum()),
                    "over_1600": int((df['TOTALCNT'] >= 1600).sum())
                }
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
    elif os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file)
            result["data"] = df.to_dict('records')
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
    
    # 알람 파일 로드
    if os.path.exists(alert_file):
        try:
            df_alert = pd.read_csv(alert_file)
            df_alert['TYPE'] = df_alert['TYPE'].astype(str)
            result["alerts_10"] = df_alert[df_alert['TYPE'] == '10'].to_dict('records')
            result["alerts_30"] = df_alert[df_alert['TYPE'] == '30'].to_dict('records')
        except Exception as e:
            logger.error(f"알람 로드 실패: {e}")
    
    return result


def get_available_dates() -> List[str]:
    """사용 가능한 날짜 목록"""
    dates = []
    
    if not os.path.exists(M14_DATA_DIR):
        return dates
    
    for f in os.listdir(M14_DATA_DIR):
        if f.startswith('m14_data_') and f.endswith('.csv'):
            date_str = f.replace('m14_data_', '').replace('.csv', '')
            if len(date_str) == 8 and date_str.isdigit():
                dates.append(date_str)
    
    return sorted(dates, reverse=True)


# ========================================
# Pydantic Models
# ========================================
class AnalysisRequest(BaseModel):
    date_start: str
    date_end: str
    time_start: str = "0000"
    time_end: str = "2359"
    question: str = ""
    analysis_type: str = "summary"  # summary, pattern, prediction, custom


# ========================================
# FastAPI Startup
# ========================================
@app.on_event("startup")
async def startup():
    """서버 시작 시 초기화"""
    global LLM_MODE, M14_DATA_DIR
    
    # M14 데이터 폴더 확인
    if not os.path.exists(M14_DATA_DIR):
        # 다른 경로 시도
        alt_paths = ["data", "../data", "NODE/UI/data"]
        for path in alt_paths:
            if os.path.exists(path):
                M14_DATA_DIR = path
                break
    
    logger.info(f"📁 M14 데이터 폴더: {M14_DATA_DIR}")
    
    # API 토큰 로드
    if load_api_token():
        LLM_MODE = "api"
    else:
        LLM_MODE = "local"
    
    # 로컬 LLM 로드
    if load_local_llm():
        logger.info("✅ 로컬 LLM 준비 완료")
        if not API_TOKEN:
            LLM_MODE = "local"
    
    logger.info(f"🚀 M14 LLM 분석 서버 시작! 모드: {LLM_MODE}")


# ========================================
# API Endpoints
# ========================================
@app.get("/")
async def home():
    """메인 페이지"""
    return FileResponse("llm_analysis.html")


@app.get("/api/status")
async def api_status():
    """서버 상태"""
    return {
        "llm_mode": LLM_MODE,
        "api_available": API_TOKEN is not None,
        "local_available": llm is not None,
        "env": ENV_MODE,
        "model": API_MODEL if LLM_MODE == "api" else LOCAL_MODEL_PATH,
        "data_dir": M14_DATA_DIR,
        "data_exists": os.path.exists(M14_DATA_DIR)
    }


@app.get("/api/dates")
async def get_dates():
    """사용 가능한 날짜 목록"""
    dates = get_available_dates()
    return {"dates": dates}


@app.get("/api/data/{date_str}")
async def get_data(date_str: str):
    """특정 날짜 데이터 조회"""
    result = load_m14_data(date_str)
    if result["data"] is None:
        return JSONResponse(status_code=404, content={"error": f"{date_str} 데이터 없음"})
    return result


@app.post("/api/set_mode")
async def set_mode(data: dict):
    """LLM 모드 전환"""
    global LLM_MODE
    
    new_mode = data.get("mode", "local")
    
    if new_mode == "local" and llm is None:
        return {"success": False, "message": "로컬 LLM이 없습니다."}
    if new_mode == "api" and API_TOKEN is None:
        return {"success": False, "message": "API 토큰이 없습니다."}
    
    LLM_MODE = new_mode
    return {"success": True, "mode": LLM_MODE}


@app.post("/api/set_data_dir")
async def set_data_dir(data: dict):
    """데이터 폴더 경로 설정"""
    global M14_DATA_DIR
    
    new_dir = data.get("path", "")
    if os.path.exists(new_dir):
        M14_DATA_DIR = new_dir
        return {"success": True, "path": M14_DATA_DIR}
    else:
        return {"success": False, "message": "경로가 존재하지 않습니다."}


@app.get("/api/prompts")
async def get_prompts():
    """현재 프롬프트 조회"""
    return {
        "system_prompt": SYSTEM_PROMPT_ANALYSIS,
        "templates": PROMPT_TEMPLATES
    }


@app.post("/api/prompts")
async def set_prompts(data: dict):
    """프롬프트 수정"""
    global SYSTEM_PROMPT_ANALYSIS, PROMPT_TEMPLATES
    
    if "system_prompt" in data:
        SYSTEM_PROMPT_ANALYSIS = data["system_prompt"]
    
    if "templates" in data:
        for key, value in data["templates"].items():
            if key in PROMPT_TEMPLATES:
                PROMPT_TEMPLATES[key] = value
    
    return {"success": True, "message": "프롬프트 저장됨"}


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    """데이터 분석 요청"""
    
    # 데이터 로드
    all_data = []
    all_alerts_10 = []
    all_alerts_30 = []
    
    start = datetime.strptime(request.date_start, "%Y%m%d")
    end = datetime.strptime(request.date_end, "%Y%m%d")
    
    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        result = load_m14_data(date_str)
        
        if result["data"]:
            all_data.extend(result["data"])
            all_alerts_10.extend(result["alerts_10"])
            all_alerts_30.extend(result["alerts_30"])
        
        current += timedelta(days=1)
    
    if not all_data:
        return {"success": False, "answer": "❌ 해당 기간에 데이터가 없습니다."}
    
    # 시간 필터링
    start_dt = request.date_start + request.time_start
    end_dt = request.date_end + request.time_end
    
    filtered_data = [
        row for row in all_data
        if start_dt <= str(row.get('CURRTIME', ''))[:12] <= end_dt
    ]
    
    if not filtered_data:
        return {"success": False, "answer": "❌ 선택한 시간 범위에 데이터가 없습니다."}
    
    # 데이터 요약 생성
    df = pd.DataFrame(filtered_data)
    
    summary_stats = {
        "period": f"{request.date_start} {request.time_start} ~ {request.date_end} {request.time_end}",
        "total_count": len(df),
        "avg_totalcnt": round(df['TOTALCNT'].mean(), 1) if 'TOTALCNT' in df.columns else 0,
        "max_totalcnt": int(df['TOTALCNT'].max()) if 'TOTALCNT' in df.columns else 0,
        "min_totalcnt": int(df['TOTALCNT'].min()) if 'TOTALCNT' in df.columns else 0,
        "over_1700_count": int((df['TOTALCNT'] >= 1700).sum()) if 'TOTALCNT' in df.columns else 0,
        "over_1600_count": int((df['TOTALCNT'] >= 1600).sum()) if 'TOTALCNT' in df.columns else 0,
        "alerts_10_count": len(all_alerts_10),
        "alerts_30_count": len(all_alerts_30)
    }
    
    # 분석 유형별 프롬프트 생성
    if request.analysis_type == "summary":
        prompt = f"""다음 M14 반송 큐 모니터링 데이터를 분석해주세요.

## 데이터 요약
- 기간: {summary_stats['period']}
- 총 데이터 수: {summary_stats['total_count']}개
- 평균 TOTALCNT: {summary_stats['avg_totalcnt']}
- 최대 TOTALCNT: {summary_stats['max_totalcnt']}
- 최소 TOTALCNT: {summary_stats['min_totalcnt']}
- 1700+ 초과 횟수: {summary_stats['over_1700_count']}회
- 1600+ 경고 횟수: {summary_stats['over_1600_count']}회
- 10분 예측 알람: {summary_stats['alerts_10_count']}회
- 30분 예측 알람: {summary_stats['alerts_30_count']}회

전반적인 상황 분석과 개선점을 제안해주세요."""

    elif request.analysis_type == "pattern":
        # 시간대별 평균 계산
        if 'CURRTIME' in df.columns:
            df['hour'] = df['CURRTIME'].astype(str).str[8:10].astype(int)
            hourly_avg = df.groupby('hour')['TOTALCNT'].mean().round(1).to_dict()
        else:
            hourly_avg = {}
        
        prompt = f"""M14 반송 큐의 시간대별 패턴을 분석해주세요.

## 데이터 요약
- 기간: {summary_stats['period']}
- 총 데이터: {summary_stats['total_count']}개

## 시간대별 평균 TOTALCNT
{hourly_avg}

시간대별 패턴, 피크 타임, 주의가 필요한 시간대를 분석해주세요."""

    elif request.analysis_type == "prediction":
        # 예측 정확도 계산 시도
        pred_accuracy = "데이터 부족"
        if 'PREDICT_10' in df.columns and 'TOTALCNT' in df.columns:
            # 예측 1700+ vs 실제 1700+
            pred_over = (df['PREDICT_10'] >= 1700).sum()
            actual_over = (df['TOTALCNT'] >= 1700).sum()
            pred_accuracy = f"예측 1700+: {pred_over}회, 실제 1700+: {actual_over}회"
        
        prompt = f"""M14 반송 큐 예측 성능을 분석해주세요.

## 데이터 요약
- 기간: {summary_stats['period']}
- 총 데이터: {summary_stats['total_count']}개
- 10분 예측 알람: {summary_stats['alerts_10_count']}회
- 30분 예측 알람: {summary_stats['alerts_30_count']}회
- 예측 정확도 참고: {pred_accuracy}

예측 모델의 성능과 개선 방향을 분석해주세요."""

    else:  # custom
        if not request.question:
            return {"success": False, "answer": "❌ 질문을 입력해주세요."}
        
        prompt = f"""M14 반송 큐 모니터링 데이터에 대해 답변해주세요.

## 데이터 요약
- 기간: {summary_stats['period']}
- 총 데이터: {summary_stats['total_count']}개
- 평균 TOTALCNT: {summary_stats['avg_totalcnt']}
- 최대 TOTALCNT: {summary_stats['max_totalcnt']}
- 1700+ 초과: {summary_stats['over_1700_count']}회

## 질문
{request.question}"""

    # LLM 호출
    result = call_llm(prompt, SYSTEM_PROMPT_ANALYSIS)
    
    if result["success"]:
        return {
            "success": True,
            "answer": result["content"],
            "stats": summary_stats,
            "mode": LLM_MODE
        }
    else:
        return {"success": False, "answer": f"❌ {result['error']}"}


@app.post("/api/quick_analyze")
async def quick_analyze(data: dict):
    """빠른 분석 (간단한 질문)"""
    question = data.get("question", "")
    date_str = data.get("date", datetime.now().strftime("%Y%m%d"))
    
    if not question:
        return {"success": False, "answer": "❌ 질문을 입력해주세요."}
    
    # 데이터 로드
    result = load_m14_data(date_str)
    
    if result["data"] is None:
        return {"success": False, "answer": f"❌ {date_str} 데이터가 없습니다."}
    
    stats = result["stats"]
    
    prompt = f"""M14 반송 큐 데이터 ({date_str})에 대해 답변해주세요.

## 오늘 데이터 요약
- 데이터 수: {stats.get('count', 0)}개
- 평균 TOTALCNT: {stats.get('avg', 0)}
- 최대 TOTALCNT: {stats.get('max', 0)}
- 1700+ 초과: {stats.get('over_1700', 0)}회

## 질문
{question}"""

    llm_result = call_llm(prompt, SYSTEM_PROMPT_ANALYSIS, max_tokens=1500)
    
    if llm_result["success"]:
        return {"success": True, "answer": llm_result["content"], "stats": stats}
    else:
        return {"success": False, "answer": f"❌ {llm_result['error']}"}


# ========================================
# CSV 업로드 분석
# ========================================
@app.post("/api/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """CSV 파일 업로드 및 파싱"""
    try:
        content = await file.read()
        
        # 여러 인코딩 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin1']
        content_str = None
        
        for enc in encodings:
            try:
                content_str = content.decode(enc)
                logger.info(f"CSV 인코딩 감지: {enc}")
                break
            except:
                continue
        
        if content_str is None:
            return {"success": False, "error": "파일 인코딩을 인식할 수 없습니다"}
        
        df = pd.read_csv(StringIO(content_str))
        
        # 컬럼명 정리 (공백 제거)
        df.columns = df.columns.str.strip()
        
        # 기본 통계 계산
        stats = {
            "filename": file.filename,
            "row_count": len(df),
            "columns": list(df.columns)
        }
        
        # 주요 컬럼별 통계
        if '현재TOTALCNT' in df.columns:
            stats["avg_totalcnt"] = round(df['현재TOTALCNT'].mean(), 1)
            stats["max_totalcnt"] = int(df['현재TOTALCNT'].max())
            stats["min_totalcnt"] = int(df['현재TOTALCNT'].min())
        
        if '실제위험(1700+)' in df.columns:
            stats["danger_count"] = int(df['실제위험(1700+)'].sum())
        
        if '최종판정' in df.columns:
            stats["pred_danger_count"] = int(df['최종판정'].sum())
        
        if '예측상태' in df.columns:
            status_counts = df['예측상태'].value_counts().to_dict()
            stats["status_counts"] = status_counts
            
            # TP, TN, FP, FN 계산
            stats["TP"] = sum(1 for s in df['예측상태'] if 'TP' in str(s))
            stats["TN"] = sum(1 for s in df['예측상태'] if 'TN' in str(s))
            stats["FP"] = sum(1 for s in df['예측상태'] if 'FP' in str(s))
            stats["FN"] = sum(1 for s in df['예측상태'] if 'FN' in str(s))
        
        # 데이터 샘플 (처음 10행)
        sample_data = df.head(10).to_dict('records')
        
        return {
            "success": True,
            "stats": stats,
            "sample": sample_data,
            "data": df.to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"CSV 업로드 실패: {e}")
        return {"success": False, "error": f"CSV 파싱 실패: {str(e)}"}


@app.post("/api/analyze_csv")
async def analyze_csv(data: dict):
    """업로드된 CSV 데이터 LLM 분석"""
    csv_data = data.get("data", [])
    stats = data.get("stats", {})
    question = data.get("question", "")
    analysis_type = data.get("analysis_type", "summary")
    
    if not csv_data:
        return {"success": False, "answer": "❌ CSV 데이터가 없습니다."}
    
    df = pd.DataFrame(csv_data)
    
    # 통계 요약 생성
    summary_text = f"""## CSV 데이터 요약
- 파일명: {stats.get('filename', 'unknown')}
- 데이터 수: {stats.get('row_count', len(df))}개
- 컬럼: {', '.join(stats.get('columns', [])[:10])}{'...' if len(stats.get('columns', [])) > 10 else ''}
"""
    
    if 'avg_totalcnt' in stats:
        summary_text += f"""
## TOTALCNT 통계
- 평균: {stats.get('avg_totalcnt')}
- 최대: {stats.get('max_totalcnt')}
- 최소: {stats.get('min_totalcnt')}
- 1700+ 실제 위험: {stats.get('danger_count', 0)}회
- 1700+ 예측 위험: {stats.get('pred_danger_count', 0)}회
"""
    
    if 'TP' in stats:
        total = stats.get('TP', 0) + stats.get('TN', 0) + stats.get('FP', 0) + stats.get('FN', 0)
        accuracy = round((stats.get('TP', 0) + stats.get('TN', 0)) / total * 100, 2) if total > 0 else 0
        recall = round(stats.get('TP', 0) / (stats.get('TP', 0) + stats.get('FN', 0)) * 100, 2) if (stats.get('TP', 0) + stats.get('FN', 0)) > 0 else 0
        precision = round(stats.get('TP', 0) / (stats.get('TP', 0) + stats.get('FP', 0)) * 100, 2) if (stats.get('TP', 0) + stats.get('FP', 0)) > 0 else 0
        
        summary_text += f"""
## 예측 성능
- TP (정확한 위험 예측): {stats.get('TP', 0)}
- TN (정확한 정상 예측): {stats.get('TN', 0)}
- FP (오탐 - 잘못된 위험 예측): {stats.get('FP', 0)}
- FN (놓침 - 위험 미탐지): {stats.get('FN', 0)}
- 정확도: {accuracy}%
- 재현율: {recall}%
- 정밀도: {precision}%
"""
    
    if 'status_counts' in stats:
        summary_text += f"""
## 예측상태 분포
"""
        for status, count in stats.get('status_counts', {}).items():
            summary_text += f"- {status}: {count}개\n"
    
    # 분석 유형별 프롬프트 (PROMPT_TEMPLATES 사용)
    if analysis_type == "summary":
        template = PROMPT_TEMPLATES.get("summary", "위 데이터를 분석해주세요.")
        prompt = f"""{summary_text}

{template}"""

    elif analysis_type == "model":
        # 모델별 예측값 통계
        model_cols = ['XGB_타겟', 'XGB_중요', 'XGB_보조', 'XGB_PDT', 'XGB_Job']
        model_stats = ""
        for col in model_cols:
            if col in df.columns:
                over_1700 = (df[col] >= 1700).sum()
                model_stats += f"- {col}: 평균 {df[col].mean():.1f}, 최대 {df[col].max():.1f}, 1700+ 예측 {over_1700}개 ({over_1700/len(df)*100:.2f}%)\n"
        
        template = PROMPT_TEMPLATES.get("model", "모델별 성능을 분석해주세요.")
        prompt = f"""{summary_text}

## 모델별 예측값 통계
{model_stats}

{template}"""

    elif analysis_type == "error":
        # FP, FN 케이스 분석
        fp_cases = df[df['예측상태'].str.contains('FP', na=False)] if '예측상태' in df.columns else pd.DataFrame()
        fn_cases = df[df['예측상태'].str.contains('FN', na=False)] if '예측상태' in df.columns else pd.DataFrame()
        
        error_info = f"""
## 오류 케이스 분석
- FP (오탐) 케이스: {len(fp_cases)}개
- FN (미탐) 케이스: {len(fn_cases)}개
"""
        if len(fp_cases) > 0 and '현재TOTALCNT' in fp_cases.columns:
            error_info += f"- FP 평균 TOTALCNT: {fp_cases['현재TOTALCNT'].mean():.1f}\n"
        if len(fn_cases) > 0 and '현재TOTALCNT' in fn_cases.columns:
            error_info += f"- FN 평균 TOTALCNT: {fn_cases['현재TOTALCNT'].mean():.1f}\n"
        
        template = PROMPT_TEMPLATES.get("error", "FP/FN 오류를 분석해주세요.")
        prompt = f"""{summary_text}
{error_info}

{template}"""

    else:  # custom
        if not question:
            return {"success": False, "answer": "❌ 질문을 입력해주세요."}
        
        prompt = f"""{summary_text}

## 질문
{question}"""

    # LLM 호출
    result = call_llm(prompt, SYSTEM_PROMPT_ANALYSIS)
    
    if result["success"]:
        return {
            "success": True,
            "answer": result["content"],
            "stats": stats,
            "mode": LLM_MODE
        }
    else:
        return {"success": False, "answer": f"❌ {result['error']}"}


def build_csv_prompt(csv_data, stats, analysis_type, question=""):
    """CSV 분석용 프롬프트 생성 - PROMPT_TEMPLATES 사용"""
    df = pd.DataFrame(csv_data)
    
    # 통계 요약 생성
    summary_text = f"""## CSV 데이터 요약
- 파일명: {stats.get('filename', 'unknown')}
- 데이터 수: {stats.get('row_count', len(df))}개
"""
    
    if 'avg_totalcnt' in stats:
        summary_text += f"""
## TOTALCNT 통계
| 항목 | 값 |
|------|-----|
| 평균 | {stats.get('avg_totalcnt')} |
| 최대 | {stats.get('max_totalcnt')} |
| 최소 | {stats.get('min_totalcnt')} |
| 1700+ 실제 | {stats.get('danger_count', 0)}회 |
| 1700+ 예측 | {stats.get('pred_danger_count', 0)}회 |
"""
    
    if 'TP' in stats:
        total = stats.get('TP', 0) + stats.get('TN', 0) + stats.get('FP', 0) + stats.get('FN', 0)
        accuracy = round((stats.get('TP', 0) + stats.get('TN', 0)) / total * 100, 2) if total > 0 else 0
        recall = round(stats.get('TP', 0) / (stats.get('TP', 0) + stats.get('FN', 0)) * 100, 2) if (stats.get('TP', 0) + stats.get('FN', 0)) > 0 else 0
        precision = round(stats.get('TP', 0) / (stats.get('TP', 0) + stats.get('FP', 0)) * 100, 2) if (stats.get('TP', 0) + stats.get('FP', 0)) > 0 else 0
        
        summary_text += f"""
## 예측 성능
| 지표 | 값 |
|------|-----|
| TP | {stats.get('TP', 0)} |
| TN | {stats.get('TN', 0)} |
| FP (오탐) | {stats.get('FP', 0)} |
| FN (놓침) | {stats.get('FN', 0)} |
| 정확도 | {accuracy}% |
| 재현율 | {recall}% |
| 정밀도 | {precision}% |
"""
    
    if 'status_counts' in stats:
        summary_text += "\n## 예측상태 분포\n| 상태 | 개수 |\n|------|------|\n"
        for status, count in stats.get('status_counts', {}).items():
            summary_text += f"| {status} | {count} |\n"
    
    # 분석 유형별 프롬프트 (PROMPT_TEMPLATES 사용)
    if analysis_type == "summary":
        template = PROMPT_TEMPLATES.get("summary", "종합 분석해주세요.")
        prompt = f"{summary_text}\n\n{template}"

    elif analysis_type == "pattern":
        # 시간대별 통계 추가
        hour_stats = "\n## 시간대별 통계\n| 시간 | 평균 | 최대 | 1700+ |\n|------|------|------|-------|\n"
        if 'CURRTIME' in df.columns or '현재시간' in df.columns:
            time_col = 'CURRTIME' if 'CURRTIME' in df.columns else '현재시간'
            try:
                df['hour'] = df[time_col].astype(str).str[11:13].astype(int)
                for h in sorted(df['hour'].unique()):
                    h_data = df[df['hour'] == h]
                    tc_col = 'TOTALCNT' if 'TOTALCNT' in df.columns else '현재TOTALCNT'
                    if tc_col in df.columns:
                        avg = h_data[tc_col].mean()
                        max_val = h_data[tc_col].max()
                        over = (h_data[tc_col] >= 1700).sum()
                        hour_stats += f"| {h:02d}시 | {avg:.0f} | {max_val:.0f} | {over}회 |\n"
            except:
                hour_stats = "\n## 시간대별 통계\n(시간 파싱 실패)\n"
        
        template = PROMPT_TEMPLATES.get("pattern", "패턴 분석해주세요.")
        prompt = f"{summary_text}{hour_stats}\n{template}"

    elif analysis_type == "prediction":
        # 예측 성능 추가 정보
        pred_info = "\n## 예측 상세\n"
        if '예측상태' in df.columns:
            fp_count = df['예측상태'].str.contains('FP', na=False).sum()
            fn_count = df['예측상태'].str.contains('FN', na=False).sum()
            pred_info += f"| FP (오탐) | {fp_count}건 |\n| FN (놓침) | {fn_count}건 |\n"
        
        template = PROMPT_TEMPLATES.get("prediction", "예측 분석해주세요.")
        prompt = f"{summary_text}{pred_info}\n{template}"

    else:  # custom
        prompt = f"{summary_text}\n\n## 질문\n{question}"
    
    return prompt


@app.post("/api/analyze_csv_stream")
async def analyze_csv_stream(data: dict):
    """스트리밍 CSV 분석"""
    csv_data = data.get("data", [])
    stats = data.get("stats", {})
    question = data.get("question", "")
    analysis_type = data.get("analysis_type", "summary")
    
    if not csv_data:
        async def error_gen():
            yield "data: [ERROR] CSV 데이터가 없습니다.\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")
    
    if analysis_type == "custom" and not question:
        async def error_gen():
            yield "data: [ERROR] 질문을 입력해주세요.\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")
    
    prompt = build_csv_prompt(csv_data, stats, analysis_type, question)
    
    # 로컬 LLM만 스트리밍 지원
    if LLM_MODE == "local" and llm is not None:
        return StreamingResponse(
            stream_local_llm(prompt, SYSTEM_PROMPT_ANALYSIS),
            media_type="text/event-stream"
        )
    else:
        # API 모드는 일반 응답 후 한번에 전송
        async def api_response_gen():
            result = call_llm(prompt, SYSTEM_PROMPT_ANALYSIS)
            if result["success"]:
                # 한글자씩 전송 (스트리밍 효과) - 줄바꿈 이스케이프
                content = result["content"].replace('\n', '⏎')
                for char in content:
                    yield f"data: {char}\n\n"
                yield "data: [DONE]\n\n"
            else:
                yield f"data: [ERROR] {result['error']}\n\n"
        
        return StreamingResponse(api_response_gen(), media_type="text/event-stream")


# ============================================================================
# 마크다운 -> HTML 변환 API
# ============================================================================

@app.post("/api/markdown_to_html")
async def markdown_to_html(request: Request):
    """마크다운 텍스트를 HTML로 변환"""
    try:
        import markdown
        from markdown.extensions.tables import TableExtension
        from markdown.extensions.fenced_code import FencedCodeExtension
        from markdown.extensions.nl2br import Nl2BrExtension
        
        body = await request.json()
        md_text = body.get("text", "")
        
        if not md_text:
            return {"html": ""}
        
        # 마크다운 전처리: 붙어있는 형식 분리
        md_text = md_text.replace('---####', '---\n\n####')
        md_text = md_text.replace('---###', '---\n\n###')
        md_text = md_text.replace('---##', '---\n\n##')
        md_text = md_text.replace('---#', '---\n\n#')
        
        # 테이블 앞뒤 줄바꿈 보장
        import re
        md_text = re.sub(r'([^\n])(\n\|)', r'\1\n\2', md_text)
        md_text = re.sub(r'(\|[^\n]*\n)([^\|])', r'\1\n\2', md_text)
        
        # 마크다운 -> HTML 변환
        md = markdown.Markdown(extensions=[
            TableExtension(),
            FencedCodeExtension(),
            Nl2BrExtension(),
            'md_in_html'
        ])
        
        html = md.convert(md_text)
        
        return {"html": html}
        
    except ImportError:
        # markdown 라이브러리 없으면 간단한 변환
        logger.warning("markdown 라이브러리 없음, 기본 변환 사용")
        return {"html": f"<pre>{md_text}</pre>"}
    except Exception as e:
        logger.error(f"마크다운 변환 오류: {e}")
        return {"html": f"<pre>{body.get('text', '')}</pre>"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
