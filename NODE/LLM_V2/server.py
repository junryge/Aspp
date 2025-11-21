#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 직접 검색 RAG 서버 (벡터DB 없음)
"""

import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import logging
import pandas as pd
import re
import numpy as np
import pickle
from datetime import datetime, timedelta
from io import StringIO
import json

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
df = None
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
    global llm, df, COLUMN_DEFINITIONS
    
    # 0. 컬럼 정의 로드
    COLUMN_DEFINITIONS = load_column_definitions()
    logger.info("✅ 컬럼 정의 로드 완료")
    
    # 1. CSV 로드
    CSV_PATH = "./CSV/2025_DATA.CSV"
    
    if os.path.exists(CSV_PATH):
        logger.info(f"CSV 로드 중: {CSV_PATH}")
        
        try:
            df = pd.read_csv(CSV_PATH, encoding='utf-8')
            logger.info(f"✅ CSV 로드 완료: {len(df)}행, {len(df.columns)}컬럼")
            logger.info(f"컬럼: {list(df.columns[:5])}...")
            
            # STAT_DT가 있는지 확인
            if 'STAT_DT' not in df.columns:
                logger.error("❌ STAT_DT 컬럼이 없습니다!")
            
        except Exception as e:
            logger.error(f"❌ CSV 로드 실패: {e}")
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

def search_csv(query):
    """CSV에서 직접 검색"""
    if df is None:
        return None, "CSV 파일이 로드되지 않았습니다."
    
    # 시간 패턴 추출 (202509210013 형식)
    time_pattern = r'(\d{12})'
    time_match = re.search(time_pattern, query)
    
    if time_match:
        stat_dt = time_match.group(1)
        logger.info(f"시간 검색: {stat_dt}")
        
        # STAT_DT로 정확히 매칭
        result = df[df['STAT_DT'].astype(str) == stat_dt]
        
        if not result.empty:
            # 첫 번째 매칭 행 반환
            row = result.iloc[0]
            
            # 데이터 포맷팅 (주요 컬럼만)
            data_text = f"시간: {stat_dt}\n"
            
            # 주요 컬럼만 표시
            important_cols = [
                'CURRENT_M16A_3F_JOB', 'CURRENT_M16A_3F_JOB_2',
                'M16A_3F_STORAGE_UTIL', 'HUBROOMTOTAL',
                'M16HUB.QUE.ALL.CURRENTQCNT', 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN',
                'M14A_3F_TO_HUB_JOB2', 'M16A_3F_TO_M14A_3F_JOB'
            ]
            
            for col in important_cols:
                if col in row.index:
                    data_text += f"{col}: {row[col]}\n"
            
            return row, data_text
        else:
            return None, f"시간 {stat_dt}에 해당하는 데이터가 없습니다."
    
    # 시간이 없으면 컬럼명으로 검색
    col_pattern = r'([A-Z_\.]+)'
    col_matches = re.findall(col_pattern, query)
    
    if col_matches:
        # 최근 5개 데이터 요약
        recent_data = df.tail(5)
        data_text = f"최근 5개 데이터:\n"
        
        for idx, row in recent_data.iterrows():
            stat_dt = row['STAT_DT'] if 'STAT_DT' in row.index else idx
            data_text += f"\n[{stat_dt}]\n"
            
            for col in col_matches:
                if col in row.index:
                    data_text += f"  {col}: {row[col]}\n"
        
        return recent_data, data_text
    
    return None, "검색 조건을 찾을 수 없습니다. 시간(예: 202509210013) 또는 컬럼명을 포함해주세요."

class Query(BaseModel):
    question: str
    mode: str = "search"  # 기본값: search

class PredictQuery(BaseModel):
    mode: str  # "m14" or "hub"
    data: str  # CSV 데이터

@app.get("/")
async def home():
    """메인 페이지"""
    return FileResponse("index.html")

@app.post("/ask")
async def ask(query: Query):
    """RAG 질문 처리"""
    global COLUMN_DEFINITIONS
    
    if llm is None:
        return {"answer": "❌ LLM이 로드되지 않았습니다."}
    
    try:
        logger.info(f"질문: {query.question} | 모드: {query.mode}")
        
        # 모드별 처리
        if query.mode == "search":
            # 데이터 검색 모드
            result, data_text = search_csv(query.question)
            
            if result is None:
                return {"answer": data_text}
        
        elif query.mode == "m14":
            # M14 예측 모드
            data_text = "M14 예측 기능은 준비 중입니다.\n현재는 데이터 검색만 가능합니다."
            return {"answer": data_text}
        
        elif query.mode == "hub":
            # HUB 예측 모드
            data_text = "HUB 예측 기능은 준비 중입니다.\n현재는 데이터 검색만 가능합니다."
            return {"answer": data_text}
        
        else:
            # 기본값: 검색
            result, data_text = search_csv(query.question)
            
            if result is None:
                return {"answer": data_text}
        
        # 2. 프롬프트 구성
        prompt = f"""You MUST answer in Korean only. Be concise.
당신은 AMHS 전문가입니다. 한국어로 간결하게 답변하세요.

컬럼 정의:
{COLUMN_DEFINITIONS}

검색된 데이터:
{data_text}

질문: {query.question}

답변 (한국어, 간결하게):"""
        
        # 3. LLM 호출
        response = llm(
            prompt,
            max_tokens=150,
            temperature=0.2,
            top_p=0.85,
            repeat_penalty=1.5,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            stop=["질문:", "Question:", "[", "추정값"]
        )
        
        answer = response['choices'][0]['text'].strip()
        
        # 반복 패턴 제거
        lines = answer.split('\n')
        seen = set()
        unique_lines = []
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                seen.add(line_clean)
                unique_lines.append(line)
        
        answer = '\n'.join(unique_lines[:5])
        
        logger.info(f"답변 생성 완료")
        
        return {"answer": answer.strip()}
        
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
    """예측 결과 간단 요약"""
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
    """LLM으로 예측 결과 분석"""
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