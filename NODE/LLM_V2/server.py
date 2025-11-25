#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 직접 검색 RAG 서버 (csv_searcher 모듈 사용)
"""

import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import logging
from datetime import datetime
import json

# CSV 검색 모듈
import csv_searcher

# M14 예측 모듈
import m14_predictor

# HUB 예측 모듈
import hub_predictor_numerical
import hub_predictor_categorical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 전역 변수
llm = None
COLUMN_DEFINITIONS = ""

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
    global llm, COLUMN_DEFINITIONS
    
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
    
    # 2. LLM 로드
    MODEL_PATH = "models/Qwen3-1.7B-Q8_0.gguf"
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"LLM 로드 시작: {MODEL_PATH}")
        
        try:
            from llama_cpp import Llama
            
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=3000,
                n_batch=256,
                n_gpu_layers=0,
                n_threads=6,
                verbose=False
            )
            
            logger.info("✅ LLM 로드 성공!")
            
        except Exception as e:
            logger.error(f"❌ LLM 로드 실패: {e}")
    else:
        logger.warning(f"⚠️ LLM 모델 없음: {MODEL_PATH}")

class Query(BaseModel):
    question: str
    mode: str = "search"

class PredictQuery(BaseModel):
    mode: str
    data: str

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

@app.post("/ask")
async def ask(query: Query):
    """RAG 질문 처리"""
    global COLUMN_DEFINITIONS
    
    try:
        logger.info(f"질문: {query.question} | 모드: {query.mode}")
        
        # 모드별 처리
        if query.mode == "search":
            # csv_searcher로 검색
            result, data_text = csv_searcher.search_csv(query.question)
            
            if result is None:
                return {"answer": data_text}
            
            # 1. 정확한 데이터 먼저
            answer = f"📊 검색 결과\n{data_text}\n"
            
            # 2. LLM 분석 추가 (있으면)
            if llm is not None:
                try:
                    prompt = f"""You MUST answer in Korean only. 
아래 데이터를 분석해주세요. 데이터 값은 절대 바꾸지 마세요.

컬럼 정의:
{COLUMN_DEFINITIONS}

검색된 데이터:
{data_text}

위 데이터를 바탕으로 현재 상태를 간단히 분석해주세요 (2-3문장):
- 정상/주의/위험 상태인지
- 특이사항이 있는지

분석 (한국어, 간결하게):"""
                    
                    response = llm(
                        prompt,
                        max_tokens=150,
                        temperature=0.3,
                        top_p=0.85,
                        repeat_penalty=1.5,
                        stop=["질문:", "검색된", "\n\n\n"]
                    )
                    
                    analysis = response['choices'][0]['text'].strip()
                    
                    # 반복 제거
                    lines = analysis.split('\n')
                    seen = set()
                    unique_lines = []
                    for line in lines:
                        line_clean = line.strip()
                        if line_clean and line_clean not in seen:
                            seen.add(line_clean)
                            unique_lines.append(line)
                    
                    analysis = '\n'.join(unique_lines[:4])
                    
                    if analysis:
                        answer += f"---\n🤖 분석\n{analysis}"
                    
                except Exception as e:
                    logger.warning(f"LLM 분석 실패: {e}")
            
            return {"answer": answer}
        
        elif query.mode == "m14":
            data_text = "M14 예측 기능은 준비 중입니다.\n현재는 데이터 검색만 가능합니다."
            return {"answer": data_text}
        
        elif query.mode == "hub":
            data_text = "HUB 예측 기능은 준비 중입니다.\n현재는 데이터 검색만 가능합니다."
            return {"answer": data_text}
        
        else:
            # 기본값: 검색
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
            # M14 예측 실행
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
            
            # 간단 요약 생성
            summary = generate_prediction_summary(result)
            
            # LLM 해석 (있으면)
            llm_analysis = ""
            if llm is not None:
                try:
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
            # HUB 예측 실행 (수치형 + 범주형)
            result_numerical = hub_predictor_numerical.predict_hub_numerical(query.data)
            result_categorical = hub_predictor_categorical.predict_hub_categorical(query.data)
            
            if 'error' in result_numerical:
                return JSONResponse(content=result_numerical, status_code=400)
            
            if 'error' in result_categorical:
                return JSONResponse(content=result_categorical, status_code=400)
            
            # HTML 대시보드 저장 (2개)
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
            
            # 간단 요약 생성
            summary = generate_hub_summary(result_numerical, result_categorical)
            
            # LLM 해석 (있으면)
            llm_analysis = ""
            if llm is not None:
                try:
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

def generate_hub_llm_analysis(result_numerical, result_categorical):
    """LLM으로 HUB 예측 결과 분석"""
    current_val = result_numerical['current_value']
    
    pred_num = result_numerical['predictions']
    pred_cat = result_categorical['predictions']
    
    # 최대 급증 확률
    max_surge_prob = max(p['prob2'] for p in pred_cat)
    
    # 최대 예측값
    max_pred_value = max(p['pred_max'] for p in pred_num)
    
    # 프롬프트 구성
    pred_num_text = ""
    for pred in pred_num:
        pred_num_text += f"{pred['horizon']}분 후: {pred['pred_min']:.1f} ~ {pred['pred_max']:.1f} (상태 {pred['status']})\n"
    
    pred_cat_text = ""
    for pred in pred_cat:
        pred_cat_text += f"{pred['horizon']}분 후: {pred['class_name']} (급증 {pred['prob2']:.1f}%, 상태 {pred['status']})\n"
    
    prompt = f"""You MUST answer in Korean only. Be concise and professional.

현재 HUB 물류 상황:
- 현재 CURRENT_M16A_3F_JOB_2: {current_val:,.1f}

수치형 예측 결과:
{pred_num_text}

범주형 예측 결과:
{pred_cat_text}

최대 급증 확률: {max_surge_prob:.1f}%
최대 예측값: {max_pred_value:.1f}

위 예측 결과를 바탕으로 다음을 한국어로 간결하게 설명하세요 (3-4문장):
1. 현재 상황 평가
2. 예측되는 추세 (증가/감소/안정)
3. 권장 조치사항

답변 (한국어):"""
    
    try:
        response = llm(
            prompt,
            max_tokens=200,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.5,
            stop=["질문:", "\n\n\n"]
        )
        
        answer = response['choices'][0]['text'].strip()
        return answer
        
    except Exception as e:
        logger.error(f"LLM 분석 실패: {e}")
        return ""

def generate_llm_analysis(result):
    """LLM으로 M14 예측 결과 분석"""
    predictions = result['predictions']
    current_val = result['current_value']
    current_status = result['current_status']
    
    # 최대 위험도
    max_danger = max(p['danger_probability'] for p in predictions)
    
    # 프롬프트 구성
    pred_text = ""
    for pred in predictions:
        pred_text += f"{pred['horizon']}분 후: {pred['prediction']:,} (위험도 {pred['danger_probability']}%, 상태 {pred['status']})\n"
    
    prompt = f"""You MUST answer in Korean only. Be concise and professional.

현재 AMHS 물류 상황:
- 현재 TOTALCNT: {current_val:,} ({current_status})

예측 결과:
{pred_text}

최대 위험도: {max_danger}%

위 예측 결과를 바탕으로 다음을 한국어로 간결하게 설명하세요 (3-4문장):
1. 현재 상황 평가
2. 예측되는 추세 (증가/감소/안정)
3. 권장 조치사항

답변 (한국어):"""
    
    try:
        response = llm(
            prompt,
            max_tokens=200,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.5,
            stop=["질문:", "\n\n\n"]
        )
        
        answer = response['choices'][0]['text'].strip()
        return answer
        
    except Exception as e:
        logger.error(f"LLM 분석 실패: {e}")
        return ""

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