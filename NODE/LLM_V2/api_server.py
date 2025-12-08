#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 직접 검색 RAG 서버 (csv_searcher 모듈 사용)
v2.4 - 로컬/API LLM 선택 기능 추가
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

# ========================================
# 전역 변수
# ========================================
llm = None  # 로컬 LLM
COLUMN_DEFINITIONS = ""

# LLM 설정
LLM_MODE = "api"  # "local" 또는 "api"
API_URL = "http://dev.assistant.llm.skhynix.com/v1/chat/completions"
API_MODEL = "Qwen3-Coder-30B-A3B-Instruct"
API_TOKEN = None

# ========================================
# API LLM 함수
# ========================================
def load_api_token():
    """토큰 파일에서 API 토큰 로드"""
    global API_TOKEN
    token_path = "token.txt"
    
    if os.path.exists(token_path):
        try:
            with open(token_path, "r") as f:
                API_TOKEN = f.read().strip()
            logger.info("✅ API 토큰 로드 완료")
            return True
        except Exception as e:
            logger.error(f"❌ API 토큰 로드 실패: {e}")
            return False
    else:
        logger.warning(f"⚠️ 토큰 파일 없음: {token_path}")
        return False

def call_api_llm(prompt: str, system_prompt: str = "한국어로 답변해주세요.", max_tokens: int = 200) -> str:
    """API LLM 호출"""
    global API_TOKEN
    
    if not API_TOKEN:
        logger.warning("API 토큰이 없습니다.")
        return ""
    
    try:
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": API_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logger.info(f"✅ API LLM 응답: {answer[:100]}...")
            return answer
        else:
            logger.error(f"❌ API 오류: {response.status_code} - {response.text}")
            return ""
            
    except Exception as e:
        logger.error(f"❌ API 호출 실패: {e}")
        return ""

def call_local_llm(prompt: str, max_tokens: int = 200) -> str:
    """로컬 LLM 호출"""
    global llm
    
    if llm is None:
        return ""
    
    try:
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            stop=["<|im_end|>", "\n\n\n"]
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"❌ 로컬 LLM 호출 실패: {e}")
        return ""

def get_llm_response(prompt: str, system_prompt: str = "한국어로만 답변하세요.", max_tokens: int = 200) -> str:
    """LLM 응답 (모드에 따라 로컬/API 선택)"""
    global LLM_MODE
    
    if LLM_MODE == "api":
        return call_api_llm(prompt, system_prompt, max_tokens)
    else:
        # 로컬 LLM용 ChatML 프롬프트
        chatml_prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
        return call_local_llm(chatml_prompt, max_tokens)


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
    global llm, COLUMN_DEFINITIONS, LLM_MODE
    
    # 0. 컬럼 정의 로드
    COLUMN_DEFINITIONS = load_column_definitions()
    logger.info("✅ 컬럼 정의 로드 완료")
    
    # 1. CSV 로드 (csv_searcher 사용)
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
    
    # 2. API 토큰 로드 (API 모드용)
    if load_api_token():
        LLM_MODE = "api"
        logger.info("✅ LLM 모드: API")
    else:
        LLM_MODE = "local"
        logger.info("⚠️ API 토큰 없음 → 로컬 모드 시도")
    
    # 3. 로컬 LLM 로드 (로컬 모드용 또는 백업)
    MODEL_PATH = "Qwen3-4B-Q8_0.gguf"
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"LLM 로드 시작: {MODEL_PATH}")
        
        try:
            from llama_cpp import Llama
            
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=3000,
                n_batch=512,
                n_gpu_layers=0,
                n_threads=8,
                verbose=False
            )
            
            logger.info("✅ 로컬 LLM 로드 성공!")
            
            # API 토큰 없으면 로컬 모드
            if not API_TOKEN:
                LLM_MODE = "local"
                
        except Exception as e:
            logger.error(f"❌ 로컬 LLM 로드 실패: {e}")
            if not API_TOKEN:
                logger.warning("⚠️ LLM 없이 실행 (템플릿 기반)")
    else:
        logger.warning(f"⚠️ 로컬 LLM 모델 없음: {MODEL_PATH}")
    
    logger.info(f"🚀 최종 LLM 모드: {LLM_MODE}")


def format_star_result(section_key: str, context: str) -> str:
    """STAR 검색 결과를 보기 좋게 포맷팅"""
    import re
    
    section_titles = {
        '청주_운영': '🔵 청주 운영 환경',
        '청주_QA': '🟡 청주 QA 환경',
        '이천_운영': '🔵 이천 운영 환경',
        '이천_QA': '🟡 이천 QA 환경',
        '계정': '👤 공통 계정 정보',
        '요약': '📊 전체 요약',
        'Failover': '🔧 Failover 설정'
    }
    
    title = section_titles.get(section_key, f'📂 {section_key}')
    
    result = f"{title}\n"
    result += "=" * 45 + "\n\n"
    
    lines = context.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line or line.startswith('|--') or line.startswith('---'):
            continue
        
        if line.startswith('#'):
            continue
        
        if line.startswith('|') and line.endswith('|'):
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if len(cells) >= 2:
                key = cells[0]
                value = cells[1]
                
                if key in ['항목', '사이트'] or value in ['값', '환경']:
                    continue
                
                if 'Service' in key:
                    result += f"📌 {key}: {value}\n"
                elif 'Node' in key:
                    result += f"   🖥️ {key}: {value}\n"
                elif '계정' in key:
                    result += f"👤 {key}: {value}\n"
                elif '비밀번호' in key:
                    result += f"🔑 {key}: {value}\n"
                elif '사이트' in key or '청주' in key or '이천' in key:
                    if len(cells) >= 4:
                        result += f"📍 {cells[0]} {cells[1]}: {cells[2]} ({cells[3]})\n"
                    else:
                        result += f"📍 {key}: {value}\n"
                else:
                    result += f"   {key}: {value}\n"
        
        elif line.startswith('*'):
            item = line[1:].strip()
            result += f"  • {item}\n"
    
    return result


class Query(BaseModel):
    question: str
    mode: str = "search"

class PredictQuery(BaseModel):
    mode: str
    data: str

class LLMConfigQuery(BaseModel):
    llm_mode: str  # "local" 또는 "api"

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

@app.get("/llm_status")
async def get_llm_status():
    """현재 LLM 상태 반환"""
    global LLM_MODE, llm, API_TOKEN
    
    return {
        "mode": LLM_MODE,
        "local_available": llm is not None,
        "api_available": API_TOKEN is not None,
        "api_model": API_MODEL if API_TOKEN else None
    }

@app.post("/set_llm_mode")
async def set_llm_mode(config: LLMConfigQuery):
    """LLM 모드 변경"""
    global LLM_MODE, llm, API_TOKEN
    
    new_mode = config.llm_mode.lower()
    
    if new_mode == "api":
        if API_TOKEN:
            LLM_MODE = "api"
            logger.info("✅ LLM 모드 변경: API")
            return {"success": True, "mode": "api", "message": "API 모드로 변경됨"}
        else:
            return {"success": False, "mode": LLM_MODE, "message": "API 토큰이 없습니다"}
    
    elif new_mode == "local":
        if llm is not None:
            LLM_MODE = "local"
            logger.info("✅ LLM 모드 변경: Local")
            return {"success": True, "mode": "local", "message": "로컬 모드로 변경됨"}
        else:
            return {"success": False, "mode": LLM_MODE, "message": "로컬 LLM이 로드되지 않았습니다"}
    
    else:
        return {"success": False, "mode": LLM_MODE, "message": "유효하지 않은 모드"}

@app.post("/ask")
async def ask(query: Query):
    """RAG 질문 처리"""
    global COLUMN_DEFINITIONS, LLM_MODE
    
    try:
        logger.info(f"질문: {query.question} | 모드: {query.mode} | LLM: {LLM_MODE}")
        
        if query.mode == "search":
            
            # ⭐ MongoDB/Logpresso 쿼리 먼저 체크
            if mongo_searcher.is_mongo_query(query.question):
                logger.info("MongoDB/Logpresso 검색 감지")
                section_key, answer = mongo_searcher.search(query.question)
                
                # LLM 요약 추가
                try:
                    summary = get_llm_response(
                        f"{answer[:500]}\n\n위 접속 정보를 한국어 1문장으로 요약하세요.",
                        "반드시 한국어로만 답변하세요. 생각 과정 없이 바로 답변하세요.",
                        60
                    )
                    
                    import re
                    summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
                    
                    if not summary or len(summary) < 5:
                        if 'MongoDB' in answer and '이천' in answer:
                            summary = "이천 MongoDB 클러스터 접속 정보입니다."
                        elif 'MongoDB' in answer and '청주' in answer:
                            summary = "청주 MongoDB 클러스터 접속 정보입니다."
                        elif 'Logpresso' in answer:
                            summary = "Logpresso 로그 서버 접속 정보입니다."
                        else:
                            summary = "MongoDB/Logpresso 접속 정보입니다."
                    
                    answer += f"\n---\n🤖 요약: {summary}"
                except Exception as e:
                    logger.warning(f"LLM 요약 실패: {e}")
                
                return {"answer": answer}
            
            # ⭐ STAR DB 쿼리 체크
            if star_searcher.is_star_query(query.question):
                logger.info("STAR DB 검색 감지")
                section_key, answer = star_searcher.search(query.question)
                
                # LLM 요약 추가
                try:
                    summary = get_llm_response(
                        f"{answer}\n\n위 DB 접속 정보를 한국어 1문장으로 요약하세요.",
                        "반드시 한국어로만 답변하세요. 생각 과정 없이 바로 답변하세요.",
                        60
                    )
                    
                    import re
                    summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
                    
                    if not summary or len(summary) < 5:
                        if '청주' in answer and '운영' in answer:
                            summary = "청주 운영 환경 Oracle RAC DB 접속 정보입니다."
                        elif '이천' in answer:
                            summary = "이천 Oracle RAC DB 접속 정보입니다."
                        else:
                            summary = "STAR DB 접속 정보입니다."
                    
                    answer += f"\n---\n🤖 요약: {summary}"
                except Exception as e:
                    logger.warning(f"LLM 요약 실패: {e}")
                
                return {"answer": answer}
            
            # CSV 검색
            result, data_text = csv_searcher.search_csv(query.question)
            
            if result is None:
                return {"answer": data_text}
            
            answer = f"📊 검색 결과\n{data_text}\n"
            
            # LLM 분석 추가
            if "🔮 예측 분석" in data_text:
                analysis = get_prediction_llm_analysis(data_text, llm if LLM_MODE == "local" else None)
            else:
                data_type = "hub" if "HUB" in data_text else "m14"
                analysis = get_llm_analysis(data_text, llm if LLM_MODE == "local" else None, data_type)
            
            # API 모드면 추가 분석
            if LLM_MODE == "api" and analysis and "템플릿" not in analysis:
                pass  # 이미 처리됨
            elif LLM_MODE == "api":
                api_analysis = get_llm_response(
                    f"{data_text[:800]}\n\n위 데이터를 한국어 2-3문장으로 분석하세요.",
                    "당신은 AMHS 물류 분석 전문가입니다. 한국어로만 답변하세요.",
                    150
                )
                if api_analysis and len(api_analysis) > 10:
                    analysis = api_analysis
            
            answer += f"\n---\n🤖 LLM 분석 ({LLM_MODE.upper()})\n{analysis}"
            
            return {"answer": answer}
        
        elif query.mode == "m14":
            return {"answer": "M14 예측 기능은 데이터 입력 섹션을 사용해주세요."}
        
        elif query.mode == "hub":
            return {"answer": "HUB 예측 기능은 데이터 입력 섹션을 사용해주세요."}
        
        elif query.mode == "general":
            # 일반 대화
            if LLM_MODE == "api":
                answer = get_llm_response(
                    query.question,
                    "당신은 AMHS 물류 AI 어시스턴트입니다. 친절하고 전문적으로 한국어로 답변하세요.",
                    400
                )
                if answer:
                    return {"answer": answer}
                else:
                    return {"answer": "❌ API 호출에 실패했습니다."}
            
            elif llm is not None:
                # 로컬 LLM
                context_parts = []
                
                if star_searcher.is_star_query(query.question):
                    star_content = star_searcher.get_full_content() if hasattr(star_searcher, 'get_full_content') else ""
                    if star_content:
                        context_parts.append(f"[STAR DB 정보]\n{star_content[:500]}")
                
                if mongo_searcher.is_mongo_query(query.question):
                    mongo_content = mongo_searcher.get_full_content() if hasattr(mongo_searcher, 'get_full_content') else ""
                    if mongo_content:
                        context_parts.append(f"[MongoDB 정보]\n{mongo_content[:500]}")
                
                if context_parts:
                    data_context = "\n\n".join(context_parts)
                    system_prompt = f"당신은 AMHS 물류 AI 어시스턴트입니다.\n참고:\n{data_context}"
                else:
                    system_prompt = "당신은 AMHS 물류 AI 어시스턴트입니다. 한국어로 짧게 답변하세요."
                
                answer = get_llm_response(query.question, system_prompt, 300)
                
                if answer:
                    import re
                    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
                    return {"answer": answer}
                else:
                    return {"answer": "❌ LLM 응답 생성에 실패했습니다."}
            
            else:
                return {"answer": "❌ LLM 모델이 로드되지 않았습니다."}
        
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
    global LLM_MODE
    
    try:
        logger.info(f"예측 요청: 모드={query.mode}")
        
        if query.mode == "m14":
            result = m14_predictor.predict_m14(query.data)
            
            if 'error' in result:
                return JSONResponse(content=result, status_code=400)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dashboard_filename = f'M14_Dashboard_{timestamp}.html'
            dashboard_path = os.path.join('dashboards', dashboard_filename)
            
            os.makedirs('dashboards', exist_ok=True)
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(result['dashboard_html'])
            
            logger.info(f"대시보드 저장: {dashboard_filename}")
            
            summary = generate_prediction_summary(result)
            
            llm_analysis = ""
            try:
                if LLM_MODE == "api":
                    llm_analysis = generate_llm_analysis_api(result)
                elif llm is not None:
                    llm_analysis = generate_llm_analysis(result)
            except Exception as e:
                logger.warning(f"LLM 분석 실패: {e}")
            
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
            
            summary = generate_hub_summary(result_numerical, result_categorical)
            
            llm_analysis = ""
            try:
                if LLM_MODE == "api":
                    llm_analysis = generate_hub_llm_analysis_api(result_numerical, result_categorical)
                elif llm is not None:
                    llm_analysis = generate_hub_llm_analysis(result_numerical, result_categorical)
            except Exception as e:
                logger.warning(f"LLM 분석 실패: {e}")
            
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
    """API LLM으로 M14 예측 결과 분석"""
    predictions = result['predictions']
    current_val = result['current_value']
    current_status = result['current_status']
    
    current_m14b = result.get('current_m14b', 0)
    current_m14bsum = result.get('current_m14bsum', 0)
    current_gap = result.get('current_gap', 0)
    current_trans = result.get('current_trans', 0)
    
    pred_text = ""
    for pred in predictions:
        pred_text += f"{pred['horizon']}분 후: {pred['prediction']:,} (위험도 {pred['danger_probability']}%)\n"
    
    prompt = f"""현재 AMHS 물류 상황:
- TOTALCNT: {current_val:,} ({current_status})
- M14AM14B: {current_m14b:.0f}
- M14AM14BSUM: {current_m14bsum:.0f}
- queue_gap: {current_gap:.0f}
- TRANSPORT: {current_trans:.0f}

예측 결과:
{pred_text}

위 데이터를 바탕으로 한국어 3-4문장으로 분석하세요. 위험도가 높은 이유와 권장 조치사항을 포함하세요."""
    
    analysis = get_llm_response(prompt, "당신은 AMHS 물류 분석 전문가입니다. 한국어로만 답변하세요.", 250)
    
    if not analysis or len(analysis) < 20:
        return generate_m14_template_analysis(result, [], max(p['danger_probability'] for p in predictions))
    
    return analysis

def generate_hub_llm_analysis_api(result_numerical, result_categorical):
    """API LLM으로 HUB 예측 결과 분석"""
    current_val = result_numerical['current_value']
    pred_num = result_numerical['predictions']
    pred_cat = result_categorical['predictions']
    
    max_surge_prob = max(p['prob2'] for p in pred_cat)
    max_pred_value = max(p['pred_max'] for p in pred_num)
    
    pred_num_text = ""
    for pred in pred_num:
        pred_num_text += f"{pred['horizon']}분 후: {pred['pred_min']:.1f} ~ {pred['pred_max']:.1f}\n"
    
    pred_cat_text = ""
    for pred in pred_cat:
        pred_cat_text += f"{pred['horizon']}분 후: 급증 확률 {pred['prob2']:.1f}%\n"
    
    prompt = f"""현재 HUB 물류 상황:
- 현재값: {current_val:.1f}
- 최대 예측값: {max_pred_value:.1f}
- 최대 급증 확률: {max_surge_prob:.1f}%

수치형 예측:
{pred_num_text}

범주형 예측:
{pred_cat_text}

위 데이터를 바탕으로 한국어 3-4문장으로 분석하세요. 위험 시간대와 권장 조치사항을 포함하세요."""
    
    analysis = get_llm_response(prompt, "당신은 AMHS 물류 분석 전문가입니다. 한국어로만 답변하세요.", 250)
    
    if not analysis or len(analysis) < 20:
        return generate_hub_template_analysis(result_numerical, result_categorical, [], max_surge_prob, max_pred_value)
    
    return analysis

def generate_hub_llm_analysis(result_numerical, result_categorical):
    """로컬 LLM으로 HUB 예측 결과 분석"""
    current_val = result_numerical['current_value']
    pred_num = result_numerical['predictions']
    pred_cat = result_categorical['predictions']
    
    max_surge_prob = max(p['prob2'] for p in pred_cat)
    max_pred_value = max(p['pred_max'] for p in pred_num)
    
    risk_factors = []
    
    if max_pred_value >= 300:
        risk_factors.append(f"예측 최대값({max_pred_value:.0f})이 심각 임계값(300) 초과 예상")
    elif max_pred_value >= 280:
        risk_factors.append(f"예측 최대값({max_pred_value:.0f})이 주의 임계값(280) 초과 예상")
    
    if max_surge_prob >= 70:
        risk_factors.append(f"급증 확률({max_surge_prob:.1f}%)이 매우 높음")
    elif max_surge_prob >= 50:
        risk_factors.append(f"급증 확률({max_surge_prob:.1f}%)이 높음")
    
    pred_num_text = ""
    for pred in pred_num:
        pred_num_text += f"{pred['horizon']}분 후: {pred['pred_min']:.1f} ~ {pred['pred_max']:.1f}\n"
    
    pred_cat_text = ""
    for pred in pred_cat:
        pred_cat_text += f"{pred['horizon']}분 후: {pred['class_name']} (급증 {pred['prob2']:.1f}%)\n"
    
    risk_text = "\n- ".join(risk_factors) if risk_factors else "현재 위험 요인 없음"
    
    prompt = f"""현재 HUB 물류:
- 현재값: {current_val:.1f}
- 최대 예측값: {max_pred_value:.1f}
- 최대 급증 확률: {max_surge_prob:.1f}%

수치형 예측:
{pred_num_text}

범주형 예측:
{pred_cat_text}

위험 요인:
- {risk_text}

한국어 3-4문장으로 분석하세요."""
    
    try:
        response = llm(
            f"<|im_start|>system\n한국어로만 답변하세요.\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n",
            max_tokens=250,
            temperature=0.2,
            stop=["<|im_end|>"]
        )
        
        raw_answer = response['choices'][0]['text'].strip()
        cleaned = clean_llm_response(raw_answer)
        
        if not cleaned or len(cleaned) < 20:
            return generate_hub_template_analysis(result_numerical, result_categorical, risk_factors, max_surge_prob, max_pred_value)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"LLM 분석 실패: {e}")
        return generate_hub_template_analysis(result_numerical, result_categorical, risk_factors, max_surge_prob, max_pred_value)

def generate_hub_template_analysis(result_numerical, result_categorical, risk_factors, max_surge_prob, max_pred_value):
    """템플릿 기반 HUB 분석"""
    current_val = result_numerical['current_value']
    pred_num = result_numerical['predictions']
    pred_cat = result_categorical['predictions']
    
    max_horizon = max(pred_num, key=lambda x: x['pred_max'])
    
    if max_pred_value < 280 and max_surge_prob < 30:
        return f"현재값 {current_val:.1f}로 정상 범위입니다. 급증 확률이 낮아 안정적인 상태입니다."
    
    analysis = f"⚠️ 위험 분석:\n\n"
    
    analysis += f"🔢 수치형 예측:\n"
    for p in pred_num:
        if p['pred_max'] >= 300:
            analysis += f"  🚨 {p['horizon']}분 후: {p['pred_min']:.0f} ~ {p['pred_max']:.0f} (심각)\n"
        elif p['pred_max'] >= 280:
            analysis += f"  ⚠️ {p['horizon']}분 후: {p['pred_min']:.0f} ~ {p['pred_max']:.0f} (주의)\n"
        else:
            analysis += f"  ✅ {p['horizon']}분 후: {p['pred_min']:.0f} ~ {p['pred_max']:.0f} (정상)\n"
    
    analysis += f"\n🎯 범주형 근거:\n"
    for p in pred_cat:
        if p['prob2'] >= 70:
            analysis += f"  🚨 {p['horizon']}분 후: 급증 확률 {p['prob2']:.1f}%\n"
        elif p['prob2'] >= 50:
            analysis += f"  ⚠️ {p['horizon']}분 후: 급증 확률 {p['prob2']:.1f}%\n"
        elif p['prob2'] >= 30:
            analysis += f"  🟡 {p['horizon']}분 후: 급증 확률 {p['prob2']:.1f}%\n"
    
    analysis += f"\n📋 결론:\n"
    if max_pred_value >= 300 and max_surge_prob >= 70:
        analysis += f"  → {max_horizon['horizon']}분 후 {max_pred_value:.0f}까지 상승 예측, 즉시 조치 필요!"
    elif max_pred_value >= 280 or max_surge_prob >= 50:
        analysis += f"  → 모니터링 강화 필요"
    else:
        analysis += f"  → 현재 안정적"
    
    return analysis

def generate_llm_analysis(result):
    """로컬 LLM으로 M14 예측 결과 분석"""
    predictions = result['predictions']
    current_val = result['current_value']
    current_status = result['current_status']
    
    current_m14b = result.get('current_m14b', 0)
    current_m14bsum = result.get('current_m14bsum', 0)
    current_gap = result.get('current_gap', 0)
    current_trans = result.get('current_trans', 0)
    
    max_danger = max(p['danger_probability'] for p in predictions)
    
    risk_factors = []
    if current_m14b > 520:
        risk_factors.append(f"M14AM14B({current_m14b:.0f}) 임계값 초과")
    if current_m14bsum > 588:
        risk_factors.append(f"M14AM14BSUM({current_m14bsum:.0f}) 임계값 초과")
    if current_gap > 300:
        risk_factors.append(f"queue_gap({current_gap:.0f}) 임계값 초과")
    if current_trans > 151:
        risk_factors.append(f"TRANSPORT({current_trans:.0f}) 임계값 초과")
    
    pred_text = ""
    for pred in predictions:
        pred_text += f"{pred['horizon']}분 후: {pred['prediction']:,} (위험도 {pred['danger_probability']}%)\n"
    
    risk_text = "\n- ".join(risk_factors) if risk_factors else "모든 지표 정상"
    
    prompt = f"""현재 AMHS 물류:
- TOTALCNT: {current_val:,} ({current_status})
- M14AM14B: {current_m14b:.0f}
- M14AM14BSUM: {current_m14bsum:.0f}
- queue_gap: {current_gap:.0f}
- TRANSPORT: {current_trans:.0f}

위험 요인:
- {risk_text}

예측:
{pred_text}

한국어 3-4문장으로 분석하세요."""
    
    try:
        response = llm(
            f"<|im_start|>system\n한국어로만 답변하세요.\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n",
            max_tokens=250,
            temperature=0.2,
            stop=["<|im_end|>"]
        )
        
        raw_answer = response['choices'][0]['text'].strip()
        cleaned = clean_llm_response(raw_answer)
        
        if not cleaned or len(cleaned) < 20:
            return generate_m14_template_analysis(result, risk_factors, max_danger)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"LLM 분석 실패: {e}")
        return generate_m14_template_analysis(result, risk_factors, max_danger)

def generate_m14_template_analysis(result, risk_factors, max_danger):
    """템플릿 기반 M14 분석"""
    predictions = result['predictions']
    current_val = result['current_value']
    max_pred = max(p['prediction'] for p in predictions)
    
    analysis = ""
    
    if risk_factors:
        analysis += f"⚠️ 현재 지표 위험 요인:\n"
        for factor in risk_factors:
            analysis += f"  🚨 {factor}\n"
        analysis += "\n"
    
    critical_preds = [p for p in predictions if p['prediction'] >= 1700]
    
    if critical_preds:
        analysis += f"🔮 예측 기반 위험:\n"
        for p in predictions:
            if p['prediction'] >= 1700:
                analysis += f"  🚨 {p['horizon']}분 후: {p['prediction']:,} (CRITICAL)\n"
            elif p['prediction'] >= 1650:
                analysis += f"  ⚠️ {p['horizon']}분 후: {p['prediction']:,} (CAUTION)\n"
    
    analysis += f"\n📋 결론:\n"
    if max_pred >= 1700:
        analysis += f"  → 예측 최대값 {max_pred:,}으로 CRITICAL 상태 예상, 즉시 조치 필요!"
    elif max_pred >= 1650:
        analysis += f"  → 예측 최대값 {max_pred:,}으로 CAUTION 상태 예상"
    else:
        analysis += f"  → 현재 안정적, 모니터링 권장"
    
    if not analysis.strip():
        return "현재 모든 지표가 정상 범위입니다."
    
    return analysis

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