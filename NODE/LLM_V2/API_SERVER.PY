#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 직접 검색 RAG 서버 (Qwen3-Coder API 연동 버전)
"""

import os
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import logging
from datetime import datetime
import json

# CSV 검색 모듈
import csv_searcher

# STAR DB 검색 모듈
import star_searcher

# MongoDB/Logpresso 검색 모듈
import mongo_searcher

# M14 예측 모듈
import m14_predictor

# HUB 예측 모듈
import hub_predictor_numerical
import hub_predictor_categorical

# LLM 후처리 모듈
from llm_postprocessor import clean_llm_response, get_llm_analysis, get_prediction_llm_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 전역 변수
COLUMN_DEFINITIONS = ""

# ========== Qwen3-Coder API 설정 ==========
QWEN_API_URL = "http://리눅스서버IP:8000"  # ← 여기에 리눅스 서버 IP 입력


def call_qwen_api(message: str, system: str = "You are a helpful assistant. 한국어로 답변해주세요.", max_tokens: int = 256) -> str:
    """Qwen3-Coder API 호출"""
    try:
        resp = requests.post(
            f"{QWEN_API_URL}/chat",
            json={"message": message, "system": system},
            timeout=180
        )
        if resp.status_code == 200:
            return resp.json().get('response', '')
        else:
            logger.error(f"Qwen API 에러: {resp.status_code}")
            return None
    except Exception as e:
        logger.error(f"Qwen API 호출 실패: {e}")
        return None


def is_qwen_available() -> bool:
    """Qwen API 서버 상태 확인"""
    try:
        resp = requests.get(f"{QWEN_API_URL}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False


def load_column_definitions():
    """컬럼 정의 파일 로드"""
    try:
        with open("column_definitions_short.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        try:
            with open("column_definitions.txt", "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"컬럼 정의 로드 실패: {e}")
            return ""


@app.on_event("startup")
async def startup():
    """서버 시작 시 초기화"""
    global COLUMN_DEFINITIONS
    
    # 0. 컬럼 정의 로드
    COLUMN_DEFINITIONS = load_column_definitions()
    logger.info("✅ 컬럼 정의 로드 완료")
    
    # 1. CSV 로드
    CSV_PATH = "./csv/with.csv"
    if os.path.exists(CSV_PATH):
        if csv_searcher.load_csv(CSV_PATH):
            logger.info("✅ CSV 로드 완료 (csv_searcher)")
        else:
            logger.error("❌ CSV 로드 실패")
    else:
        logger.error(f"❌ CSV 파일 없음: {CSV_PATH}")
    
    # 1.5. STAR DB 문서 로드
    if star_searcher.load_md():
        logger.info("✅ STAR DB 문서 로드 완료")
    else:
        logger.warning("⚠️ STAR DB 문서 로드 실패 (STAR_READ.md 없음)")
    
    # 1.6. MongoDB/Logpresso 문서 로드
    if mongo_searcher.load_md():
        logger.info("✅ MongoDB/Logpresso 문서 로드 완료")
    else:
        logger.warning("⚠️ MongoDB/Logpresso 문서 로드 실패 (MOGO_Read.md 없음)")
    
    # 2. Qwen3-Coder API 연결 확인
    if is_qwen_available():
        logger.info("✅ Qwen3-Coder API 연결됨")
    else:
        logger.warning("⚠️ Qwen3-Coder API 연결 안됨 - 리눅스 서버 확인 필요")


def get_llm_summary(text: str, prompt_suffix: str) -> str:
    """Qwen API로 요약 생성"""
    prompt = f"{text}\n\n{prompt_suffix}"
    result = call_qwen_api(prompt, system="반드시 한국어로만 답변하세요. 간결하게 답변하세요.")
    return result if result else None


class Query(BaseModel):
    question: str
    mode: str = "search"


class PredictQuery(BaseModel):
    mode: str
    data: str


class CodeQuery(BaseModel):
    question: str
    system: str = "You are a helpful coding assistant. 한국어로 답변해주세요."


@app.get("/")
async def home():
    """메인 페이지"""
    return FileResponse("index.html")


@app.get("/columns")
async def get_columns():
    """컬럼 목록 반환"""
    return {"columns": csv_searcher.get_columns()}


@app.get("/stats/{column}")
async def get_column_stats(column: str):
    """컬럼 통계 반환"""
    return csv_searcher.get_statistics(column)


@app.post("/code")
async def code_assistant(query: CodeQuery):
    """Qwen3-Coder로 코드 생성"""
    try:
        if not is_qwen_available():
            return {"answer": "❌ Qwen3-Coder API 서버가 연결되지 않았습니다."}
        
        logger.info(f"Qwen3-Coder 요청: {query.question[:50]}...")
        response = call_qwen_api(query.question, query.system)
        
        if response:
            return {"answer": f"🤖 Qwen3-Coder:\n\n{response}"}
        else:
            return {"answer": "❌ Qwen3-Coder 응답 실패"}
    
    except Exception as e:
        logger.error(f"코드 생성 실패: {e}")
        return {"answer": f"❌ 오류: {str(e)}"}


@app.post("/ask")
async def ask(query: Query):
    """RAG 질문 처리"""
    global COLUMN_DEFINITIONS
    
    try:
        logger.info(f"질문: {query.question} | 모드: {query.mode}")
        
        if query.mode == "search":
            
            # ⭐ MongoDB/Logpresso 쿼리 먼저 체크
            if mongo_searcher.is_mongo_query(query.question):
                logger.info("MongoDB/Logpresso 검색 감지")
                section_key, answer = mongo_searcher.search(query.question)
                
                # Qwen API로 요약
                if is_qwen_available():
                    summary = get_llm_summary(answer[:500], "위 접속 정보를 한국어 1문장으로 요약하세요.")
                    if summary:
                        import re
                        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
                    
                    if not summary or len(summary) < 5:
                        # 템플릿 폴백
                        if 'MongoDB' in answer and '이천' in answer:
                            summary = "이천 MongoDB 클러스터 접속 정보입니다."
                        elif 'MongoDB' in answer and '청주' in answer:
                            summary = "청주 MongoDB 클러스터 접속 정보입니다."
                        elif 'Logpresso' in answer and '이천' in answer:
                            summary = "이천 Logpresso 로그 서버 접속 정보입니다."
                        elif 'Logpresso' in answer and '청주' in answer:
                            summary = "청주 Logpresso 로그 서버 접속 정보입니다."
                        elif 'Logpresso' in answer and '우시' in answer:
                            summary = "우시 Logpresso 로그 서버 접속 정보입니다."
                        else:
                            summary = "MongoDB/Logpresso 접속 정보입니다."
                    
                    answer += f"\n---\n🤖 요약: {summary}"
                
                return {"answer": answer}
            
            # ⭐ STAR DB 쿼리 체크
            if star_searcher.is_star_query(query.question):
                logger.info("STAR DB 검색 감지")
                section_key, answer = star_searcher.search(query.question)
                
                # Qwen API로 요약
                if is_qwen_available():
                    summary = get_llm_summary(answer, "위 DB 접속 정보를 한국어 1문장으로 요약하세요.")
                    if summary:
                        import re
                        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
                    
                    if not summary or len(summary) < 5:
                        # 템플릿 폴백
                        if '청주' in answer and '운영' in answer:
                            summary = "청주 운영 환경 Oracle RAC DB 접속 정보입니다."
                        elif '청주' in answer and 'QA' in answer:
                            summary = "청주 QA 환경 Oracle RAC DB 접속 정보입니다."
                        elif '이천' in answer and '운영' in answer:
                            summary = "이천 운영 환경 Oracle RAC DB 접속 정보입니다."
                        elif '이천' in answer and 'QA' in answer:
                            summary = "이천 QA 환경 Oracle RAC DB 접속 정보입니다."
                        else:
                            summary = "STAR DB 접속 정보입니다."
                    
                    answer += f"\n---\n🤖 요약: {summary}"
                
                return {"answer": answer}
            
            # csv_searcher로 검색
            result, data_text = csv_searcher.search_csv(query.question)
            
            if result is None:
                return {"answer": data_text}
            
            # 1. 정확한 데이터 먼저
            answer = f"📊 검색 결과\n{data_text}\n"
            
            # 2. Qwen API로 분석 추가
            if is_qwen_available():
                analysis_prompt = f"{data_text}\n\n위 데이터를 한국어 3-4문장으로 분석하세요."
                analysis = call_qwen_api(analysis_prompt, system="You are a data analyst. 한국어로 간결하게 분석하세요.")
                if analysis:
                    answer += f"\n---\n🤖 LLM 분석\n{analysis}"
            
            return {"answer": answer}
        
        elif query.mode == "m14":
            data_text = "M14 예측 기능은 준비 중입니다.\n현재는 데이터 검색만 가능합니다."
            return {"answer": data_text}
        
        elif query.mode == "hub":
            data_text = "HUB 예측 기능은 준비 중입니다.\n현재는 데이터 검색만 가능합니다."
            return {"answer": data_text}
        
        else:
            result, data_text = csv_searcher.search_csv(query.question)
            if result is None:
                return {"answer": data_text}
            return {"answer": data_text}
        
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"answer": f"❌ 오류: {str(e)}"}


@app.post("/predict")
async def predict(query: PredictQuery):
    """M14/HUB 예측 처리"""
    try:
        logger.info(f"예측 요청: 모드={query.mode}")
        
        if query.mode == "m14":
            result = m14_predictor.predict_m14(query.data)
            
            if 'error' in result:
                return JSONResponse(content=result, status_code=400)
            
            # HTML 대시보드 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dashboard_filename = f'M14_Dashboard_{timestamp}.html'
            dashboard_path = os.path.join('dashboards', dashboard_filename)
            
            os.makedirs('dashboards', exist_ok=True)
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(result['dashboard_html'])
            
            logger.info(f"대시보드 저장: {dashboard_filename}")
            
            # 요약 생성
            summary = generate_prediction_summary(result)
            
            # Qwen API로 분석
            llm_analysis = ""
            if is_qwen_available():
                llm_analysis = generate_llm_analysis_api(result)
            
            return {
                "success": True,
                "summary": summary,
                "llm_analysis": llm_analysis,
                "dashboard_url": f"/dashboard/{dashboard_filename}",
                "predictions": result['predictions'],
                "current_value": result['current_value'],
                "current_status": result['current_status']
            }
        
        elif query.mode == "hub":
            result_numerical = hub_predictor_numerical.predict_hub_numerical(query.data)
            result_categorical = hub_predictor_categorical.predict_hub_categorical(query.data)
            
            if 'error' in result_numerical:
                return JSONResponse(content=result_numerical, status_code=400)
            
            if 'error' in result_categorical:
                return JSONResponse(content=result_categorical, status_code=400)
            
            # HTML 대시보드 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            dashboard_numerical_filename = f'HUB_Numerical_{timestamp}.html'
            dashboard_numerical_path = os.path.join('dashboards', dashboard_numerical_filename)
            
            dashboard_categorical_filename = f'HUB_Categorical_{timestamp}.html'
            dashboard_categorical_path = os.path.join('dashboards', dashboard_categorical_filename)
            
            os.makedirs('dashboards', exist_ok=True)
            
            with open(dashboard_numerical_path, 'w', encoding='utf-8') as f:
                f.write(result_numerical['dashboard_html'])
            
            with open(dashboard_categorical_path, 'w', encoding='utf-8') as f:
                f.write(result_categorical['dashboard_html'])
            
            logger.info(f"대시보드 저장: {dashboard_numerical_filename}, {dashboard_categorical_filename}")
            
            # 요약 생성
            summary = generate_hub_summary(result_numerical, result_categorical)
            
            # Qwen API로 분석
            llm_analysis = ""
            if is_qwen_available():
                llm_analysis = generate_hub_llm_analysis_api(result_numerical, result_categorical)
            
            return {
                "success": True,
                "summary": summary,
                "llm_analysis": llm_analysis,
                "dashboard_numerical_url": f"/dashboard/{dashboard_numerical_filename}",
                "dashboard_categorical_url": f"/dashboard/{dashboard_categorical_filename}",
                "predictions_numerical": result_numerical['predictions'],
                "predictions_categorical": result_categorical['predictions'],
                "current_value": result_numerical['current_value']
            }
        
        else:
            return {
                "error": "Invalid mode",
                "message": "mode는 'm14' 또는 'hub'여야 합니다."
            }
        
    except Exception as e:
        logger.error(f"예측 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"error": "Prediction failed", "message": str(e)},
            status_code=500
        )


def generate_prediction_summary(result):
    """M14 예측 결과 간단 요약"""
    predictions = result['predictions']
    current_val = result['current_value']
    current_status = result['current_status']
    
    summary = f"📊 현재: {current_val:,} ({current_status})\n\n"
    summary += "🔮 예측:\n"
    
    for pred in predictions:
        status_emoji = {
            'LOW': '✅',
            'NORMAL': '🟢',
            'CAUTION': '⚠️',
            'CRITICAL': '🚨'
        }.get(pred['status'], '❓')
        
        summary += f"• {pred['horizon']}분: {pred['prediction']:,} {status_emoji} (위험 {pred['danger_probability']}%)\n"
    
    return summary


def generate_hub_summary(result_numerical, result_categorical):
    """HUB 예측 결과 간단 요약"""
    current_val = result_numerical['current_value']
    
    pred_num = result_numerical['predictions']
    pred_cat = result_categorical['predictions']
    
    summary = f"📊 현재: {current_val:,.1f}\n\n"
    summary += "🔢 수치형 예측:\n"
    
    for pred in pred_num:
        status_emoji = {
            'NORMAL': '✅',
            'CAUTION': '⚠️',
            'WARNING': '🟠',
            'CRITICAL': '🚨'
        }.get(pred['status'], '❓')
        
        summary += f"• {pred['horizon']}분: {pred['pred_min']:.1f} ~ {pred['pred_max']:.1f} {status_emoji}\n"
    
    summary += "\n🎯 범주형 예측:\n"
    
    for pred in pred_cat:
        status_emoji = {
            'LOW': '✅',
            'MEDIUM': '⚠️',
            'HIGH': '🟠',
            'CRITICAL': '🚨'
        }.get(pred['status'], '❓')
        
        summary += f"• {pred['horizon']}분: {pred['class_name']} (급증 {pred['prob2']:.1f}%) {status_emoji}\n"
    
    return summary


def generate_llm_analysis_api(result):
    """Qwen API로 M14 예측 결과 분석"""
    predictions = result['predictions']
    current_val = result['current_value']
    current_status = result['current_status']
    
    current_m14b = result.get('current_m14b', 0)
    current_m14bsum = result.get('current_m14bsum', 0)
    current_gap = result.get('current_gap', 0)
    current_trans = result.get('current_trans', 0)
    
    max_danger = max(p['danger_probability'] for p in predictions)
    
    pred_text = ""
    for pred in predictions:
        pred_text += f"{pred['horizon']}분 후: {pred['prediction']:,} (위험도 {pred['danger_probability']}%)\n"
    
    prompt = f"""현재 AMHS 물류 상황:
- TOTALCNT: {current_val:,} ({current_status})
- M14AM14B: {current_m14b:.0f} (임계값: 517주의/520심각)
- M14AM14BSUM: {current_m14bsum:.0f} (임계값: 576주의/588심각)
- queue_gap: {current_gap:.0f} (임계값: 300위험)
- TRANSPORT: {current_trans:.0f} (임계값: 151주의/180심각)

예측 결과:
{pred_text}

위 데이터를 바탕으로 한국어 3-4문장으로 분석하세요:
1. 왜 위험도가 높은지 구체적 이유
2. 예측되는 추세
3. 권장 조치사항"""
    
    response = call_qwen_api(prompt, system="You are an AMHS logistics analyst. 한국어로 간결하게 분석하세요.")
    
    if response:
        return response
    else:
        # 템플릿 폴백
        return f"현재 TOTALCNT {current_val:,}이며, 최대 위험도 {max_danger}%입니다. 지속적인 모니터링이 필요합니다."


def generate_hub_llm_analysis_api(result_numerical, result_categorical):
    """Qwen API로 HUB 예측 결과 분석"""
    current_val = result_numerical['current_value']
    
    pred_num = result_numerical['predictions']
    pred_cat = result_categorical['predictions']
    
    max_surge_prob = max(p['prob2'] for p in pred_cat)
    max_pred_value = max(p['pred_max'] for p in pred_num)
    
    pred_text = ""
    for pred in pred_num:
        pred_text += f"{pred['horizon']}분 후: {pred['pred_min']:.1f} ~ {pred['pred_max']:.1f}\n"
    
    for pred in pred_cat:
        pred_text += f"{pred['horizon']}분 후 급증확률: {pred['prob2']:.1f}%\n"
    
    prompt = f"""현재 HUB 물류 상황:
- 현재값: {current_val:.1f} (임계값: 270정상/280주의/300심각)
- 최대 예측값: {max_pred_value:.1f}
- 최대 급증 확률: {max_surge_prob:.1f}%

예측 결과:
{pred_text}

위 데이터를 바탕으로 한국어 3-4문장으로 분석하세요:
1. 왜 위험한지 구체적 이유
2. 예측 추세
3. 권장 조치사항"""
    
    response = call_qwen_api(prompt, system="You are a HUB logistics analyst. 한국어로 간결하게 분석하세요.")
    
    if response:
        return response
    else:
        # 템플릿 폴백
        return f"현재값 {current_val:.1f}, 최대 예측값 {max_pred_value:.1f}, 급증 확률 {max_surge_prob:.1f}%입니다. 모니터링이 필요합니다."


@app.get("/dashboard/{filename}")
async def get_dashboard(filename: str):
    """생성된 HTML 대시보드 반환"""
    filepath = os.path.join("dashboards", filename)
    
    if not os.path.exists(filepath):
        return JSONResponse(
            content={"error": "File not found"},
            status_code=404
        )
    
    return FileResponse(filepath)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)