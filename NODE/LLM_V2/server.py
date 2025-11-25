#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 직접 검색 RAG 서버 (csv_searcher 모듈 사용)
"""

import os
import re
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import logging
from datetime import datetime
import json

import csv_searcher
import m14_predictor
import hub_predictor_numerical
import hub_predictor_categorical

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

llm = None
COLUMN_DEFINITIONS = ""

def load_column_definitions():
    try:
        with open("column_definitions_short.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        try:
            with open("column_definitions.txt", "r", encoding="utf-8") as f:
                return f.read()
        except:
            return ""

def clean_llm_response(text):
    """LLM 응답 정제"""
    if not text:
        return ""
    text = re.sub(r'\[+[^\]]*\]+', '', text)
    text = re.sub(r'#{1,6}\s*.*', '', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'\*{1,2}', '', text)
    text = re.sub(r'^\d+[\.\)]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*•]\s*', '', text, flags=re.MULTILINE)
    for word in ['문제', '요약', '설명하세요', '바탕으로', '간략히', '분석 결과', '답변:', '결과:']:
        text = text.replace(word, '')
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 3]
    result = ' '.join(lines)
    sentences = re.split(r'[.。!]\s*', result)
    valid = [s.strip() for s in sentences if len(s.strip()) > 5][:2]
    return '. '.join(valid) + '.' if valid else ""

@app.on_event("startup")
async def startup():
    global llm, COLUMN_DEFINITIONS
    COLUMN_DEFINITIONS = load_column_definitions()
    logger.info("✅ 컬럼 정의 로드 완료")
    
    CSV_PATH = "./csv/with.csv"
    if os.path.exists(CSV_PATH):
        if csv_searcher.load_csv(CSV_PATH):
            logger.info("✅ CSV 로드 완료")
    
    MODEL_PATH = "models/Qwen3-1.7B-Q8_0.gguf"
    if os.path.exists(MODEL_PATH):
        try:
            from llama_cpp import Llama
            llm = Llama(model_path=MODEL_PATH, n_ctx=3000, n_batch=256, n_gpu_layers=0, n_threads=6, verbose=False)
            logger.info("✅ LLM 로드 성공!")
        except Exception as e:
            logger.error(f"❌ LLM 로드 실패: {e}")

class Query(BaseModel):
    question: str
    mode: str = "search"

class PredictQuery(BaseModel):
    mode: str
    data: str

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.get("/columns")
async def get_columns():
    return {"columns": csv_searcher.get_columns()}

@app.get("/stats/{column}")
async def get_column_stats(column: str):
    return csv_searcher.get_statistics(column)

@app.post("/ask")
async def ask(query: Query):
    try:
        logger.info(f"질문: {query.question} | 모드: {query.mode}")
        
        if query.mode == "search":
            result, data_text = csv_searcher.search_csv(query.question)
            
            if result is None:
                return {"answer": data_text}
            
            # HTML 출력 (<br> 유지)
            answer = f"📊 검색 결과<br>{data_text}"
            
            if llm is not None:
                try:
                    data_plain = data_text.replace('<br>', ' ')
                    prompt = f"데이터: {data_plain}\n\n위 데이터 상태를 한국어 2문장으로 평가해. 포맷 없이 평문만."
                    response = llm(prompt, max_tokens=60, temperature=0.1, repeat_penalty=2.0,
                                   stop=["\n\n", "[[", "[", "1.", "###", "```", "데이터:"])
                    raw = response['choices'][0]['text'].strip()
                    analysis = clean_llm_response(raw)
                    if analysis and len(analysis) > 10:
                        answer += f"<br><br>---<br>🤖 분석<br>{analysis}"
                except Exception as e:
                    logger.warning(f"LLM 분석 실패: {e}")
            
            return {"answer": answer}
        
        elif query.mode == "m14":
            return {"answer": "M14 예측 기능은 준비 중입니다.\n현재는 데이터 검색만 가능합니다."}
        elif query.mode == "hub":
            return {"answer": "HUB 예측 기능은 준비 중입니다.\n현재는 데이터 검색만 가능합니다."}
        else:
            result, data_text = csv_searcher.search_csv(query.question)
            return {"answer": data_text if result else data_text}
        
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        return {"answer": f"❌ 오류: {str(e)}"}

@app.post("/predict")
async def predict(query: PredictQuery):
    try:
        logger.info(f"예측 요청: 모드={query.mode}")
        
        if query.mode == "m14":
            result = m14_predictor.predict_m14(query.data)
            if 'error' in result:
                return JSONResponse(content=result, status_code=400)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dashboard_filename = f'M14_Dashboard_{timestamp}.html'
            os.makedirs('dashboards', exist_ok=True)
            with open(f'dashboards/{dashboard_filename}', 'w', encoding='utf-8') as f:
                f.write(result['dashboard_html'])
            
            summary = generate_prediction_summary(result)
            llm_analysis = generate_llm_analysis(result) if llm else ""
            
            return {
                "success": True, "summary": summary, "llm_analysis": llm_analysis,
                "dashboard_url": f"/dashboard/{dashboard_filename}",
                "predictions": result['predictions'],
                "current_value": result['current_value'],
                "current_status": result['current_status']
            }
        
        elif query.mode == "hub":
            result_num = hub_predictor_numerical.predict_hub_numerical(query.data)
            result_cat = hub_predictor_categorical.predict_hub_categorical(query.data)
            
            if 'error' in result_num:
                return JSONResponse(content=result_num, status_code=400)
            if 'error' in result_cat:
                return JSONResponse(content=result_cat, status_code=400)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs('dashboards', exist_ok=True)
            
            with open(f'dashboards/HUB_Numerical_{timestamp}.html', 'w', encoding='utf-8') as f:
                f.write(result_num['dashboard_html'])
            with open(f'dashboards/HUB_Categorical_{timestamp}.html', 'w', encoding='utf-8') as f:
                f.write(result_cat['dashboard_html'])
            
            summary = generate_hub_summary(result_num, result_cat)
            llm_analysis = generate_hub_llm_analysis(result_num, result_cat) if llm else ""
            
            return {
                "success": True, "summary": summary, "llm_analysis": llm_analysis,
                "dashboard_numerical_url": f"/dashboard/HUB_Numerical_{timestamp}.html",
                "dashboard_categorical_url": f"/dashboard/HUB_Categorical_{timestamp}.html",
                "predictions_numerical": result_num['predictions'],
                "predictions_categorical": result_cat['predictions'],
                "current_value": result_num['current_value']
            }
        
        return {"error": "Invalid mode", "message": "mode는 'm14' 또는 'hub'"}
        
    except Exception as e:
        logger.error(f"예측 실패: {e}")
        return JSONResponse(content={"error": "Prediction failed", "message": str(e)}, status_code=500)

def generate_prediction_summary(result):
    preds = result['predictions']
    summary = f"📊 현재: {result['current_value']:,} ({result['current_status']})\n\n🔮 예측:\n"
    for p in preds:
        emoji = {'LOW':'✅','NORMAL':'🟢','CAUTION':'⚠️','CRITICAL':'🚨'}.get(p['status'],'❓')
        summary += f"• {p['horizon']}분: {p['prediction']:,} {emoji} (위험 {p['danger_probability']}%)\n"
    return summary

def generate_hub_summary(result_num, result_cat):
    summary = f"📊 현재: {result_num['current_value']:,.1f}\n\n🔢 수치형:\n"
    for p in result_num['predictions']:
        emoji = {'NORMAL':'✅','CAUTION':'⚠️','WARNING':'🟠','CRITICAL':'🚨'}.get(p['status'],'❓')
        summary += f"• {p['horizon']}분: {p['pred_min']:.1f}~{p['pred_max']:.1f} {emoji}\n"
    summary += "\n🎯 범주형:\n"
    for p in result_cat['predictions']:
        emoji = {'LOW':'✅','MEDIUM':'⚠️','HIGH':'🟠','CRITICAL':'🚨'}.get(p['status'],'❓')
        summary += f"• {p['horizon']}분: {p['class_name']} (급증 {p['prob2']:.1f}%) {emoji}\n"
    return summary

def generate_hub_llm_analysis(result_num, result_cat):
    try:
        prompt = f"현재:{result_num['current_value']:.1f}, 최대예측:{max(p['pred_max'] for p in result_num['predictions']):.1f}, 급증확률:{max(p['prob2'] for p in result_cat['predictions']):.1f}%\n한국어 2문장 평가. 포맷 없이 평문만."
        response = llm(prompt, max_tokens=60, temperature=0.1, repeat_penalty=2.0, stop=["\n\n","[[","[","1."])
        return clean_llm_response(response['choices'][0]['text'].strip())
    except:
        return ""

def generate_llm_analysis(result):
    try:
        prompt = f"현재:{result['current_value']:,}({result['current_status']}), 최대예측:{max(p['prediction'] for p in result['predictions']):,}, 위험도:{max(p['danger_probability'] for p in result['predictions'])}%\n한국어 2문장 평가. 포맷 없이 평문만."
        response = llm(prompt, max_tokens=60, temperature=0.1, repeat_penalty=2.0, stop=["\n\n","[[","[","1."])
        return clean_llm_response(response['choices'][0]['text'].strip())
    except:
        return ""

@app.get("/dashboard/{filename}")
async def get_dashboard(filename: str):
    filepath = f"dashboards/{filename}"
    if not os.path.exists(filepath):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    return FileResponse(filepath)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)