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

# LLM 후처리 모듈
from llm_postprocessor import clean_llm_response, get_llm_analysis

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
            
            # 2. LLM 분석 추가
            # 데이터 타입 감지
            data_type = "hub" if "HUB" in data_text else "m14"
            analysis = get_llm_analysis(data_text, llm, data_type)
            answer += f"\n---\n🤖 LLM 분석\n{analysis}"
            
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
    
    # 위험 요인 분석 (예측 기반)
    risk_factors = []
    
    # 수치형 예측 기반 위험
    if max_pred_value >= 300:
        risk_factors.append(f"예측 최대값({max_pred_value:.0f})이 심각 임계값(300) 초과 예상")
    elif max_pred_value >= 280:
        risk_factors.append(f"예측 최대값({max_pred_value:.0f})이 주의 임계값(280) 초과 예상")
    
    # 범주형 예측 기반 위험
    if max_surge_prob >= 70:
        risk_factors.append(f"급증 확률({max_surge_prob:.1f}%)이 매우 높음 (70% 이상)")
    elif max_surge_prob >= 50:
        risk_factors.append(f"급증 확률({max_surge_prob:.1f}%)이 높음 (50% 이상)")
    elif max_surge_prob >= 30:
        risk_factors.append(f"급증 확률({max_surge_prob:.1f}%)이 주의 수준 (30% 이상)")
    
    # 시간별 추세 분석
    for pred in pred_cat:
        if pred['prob2'] >= 50:
            risk_factors.append(f"{pred['horizon']}분 후 급증 확률 {pred['prob2']:.1f}% - {pred['class_name']}")
    
    # 프롬프트 구성
    pred_num_text = ""
    for pred in pred_num:
        pred_num_text += f"{pred['horizon']}분 후: {pred['pred_min']:.1f} ~ {pred['pred_max']:.1f} (상태 {pred['status']})\n"
    
    pred_cat_text = ""
    for pred in pred_cat:
        pred_cat_text += f"{pred['horizon']}분 후: {pred['class_name']} (급증 {pred['prob2']:.1f}%)\n"
    
    risk_text = "\n- ".join(risk_factors) if risk_factors else "현재 위험 요인 없음"
    
    prompt = f"""You MUST answer in Korean only. Be concise and professional.

현재 HUB 물류 상황:
- 현재값: {current_val:.1f} (임계값: 270정상/280주의/300심각)
- 최대 예측값: {max_pred_value:.1f}
- 최대 급증 확률: {max_surge_prob:.1f}%

수치형 예측:
{pred_num_text}

범주형 예측:
{pred_cat_text}

⚠️ 위험 요인:
- {risk_text}

위 데이터 바탕으로 한국어 3-4문장으로 분석하세요:
1. 왜 위험한지 구체적 이유 (어떤 시간대에 급증 예상인지)
2. 예측 추세 (증가/감소/급증)
3. 권장 조치사항

답변:"""
    
    try:
        response = llm(
            prompt,
            max_tokens=250,
            temperature=0.2,
            top_p=0.85,
            repeat_penalty=1.5,
            stop=["질문:", "\n\n\n", "이미지:", "http"]
        )
        
        raw_answer = response['choices'][0]['text'].strip()
        cleaned = clean_llm_response(raw_answer)
        
        # 영어/이상한 문자 감지 → 템플릿 사용
        english_patterns = ['please', 'answer', 'korean', 'following', 'response', 'analysis', 'the ', 'is ', 'are ', 'this ']
        has_english = any(p in cleaned.lower() for p in english_patterns)
        
        # 이상한 문자 감지 (중국어 등)
        has_weird = any(ord(c) > 0x4E00 and ord(c) < 0x9FFF for c in cleaned)
        
        # 지표 언급 확인
        indicator_keywords = ['급증', '확률', '%', '분 후', '예측', '증가', '임계']
        has_indicator = any(k in cleaned for k in indicator_keywords)
        
        # 부적절하면 템플릿 사용
        if not cleaned or len(cleaned) < 20 or has_english or has_weird or (risk_factors and not has_indicator):
            logger.warning(f"HUB LLM 응답 부적절, 템플릿 사용")
            return generate_hub_template_analysis(result_numerical, result_categorical, risk_factors, max_surge_prob, max_pred_value)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"LLM 분석 실패: {e}")
        return generate_hub_template_analysis(result_numerical, result_categorical, risk_factors, max_surge_prob, max_pred_value)

def generate_hub_template_analysis(result_numerical, result_categorical, risk_factors, max_surge_prob, max_pred_value):
    """LLM 실패시 템플릿 기반 HUB 분석 - 수치형 예측 + 범주형 뒷받침"""
    current_val = result_numerical['current_value']
    pred_num = result_numerical['predictions']
    pred_cat = result_categorical['predictions']
    
    # 가장 위험한 시간대 찾기
    max_horizon = max(pred_num, key=lambda x: x['pred_max'])
    max_cat = next((p for p in pred_cat if p['horizon'] == max_horizon['horizon']), pred_cat[-1])
    
    if max_pred_value < 280 and max_surge_prob < 30:
        return f"현재값 {current_val:.1f}로 정상 범위입니다. 급증 확률이 낮아 안정적인 상태입니다. 지속적인 모니터링을 권장합니다."
    
    analysis = f"⚠️ 위험 분석:\n\n"
    
    # 1. 수치형 예측 (메인)
    analysis += f"🔢 수치형 예측:\n"
    for p in pred_num:
        if p['pred_max'] >= 300:
            analysis += f"  🚨 {p['horizon']}분 후: {p['pred_min']:.0f} ~ {p['pred_max']:.0f} (심각)\n"
        elif p['pred_max'] >= 280:
            analysis += f"  ⚠️ {p['horizon']}분 후: {p['pred_min']:.0f} ~ {p['pred_max']:.0f} (주의)\n"
        else:
            analysis += f"  ✅ {p['horizon']}분 후: {p['pred_min']:.0f} ~ {p['pred_max']:.0f} (정상)\n"
    
    # 2. 범주형 뒷받침 (근거)
    analysis += f"\n🎯 범주형 근거 (발생 확률):\n"
    for p in pred_cat:
        if p['prob2'] >= 70:
            analysis += f"  🚨 {p['horizon']}분 후: 급증 확률 {p['prob2']:.1f}%\n"
        elif p['prob2'] >= 50:
            analysis += f"  ⚠️ {p['horizon']}분 후: 급증 확률 {p['prob2']:.1f}%\n"
        elif p['prob2'] >= 30:
            analysis += f"  🟡 {p['horizon']}분 후: 급증 확률 {p['prob2']:.1f}%\n"
    
    # 3. 결론
    analysis += f"\n📋 결론:\n"
    if max_pred_value >= 300 and max_surge_prob >= 70:
        analysis += f"  → {max_horizon['horizon']}분 후 {max_pred_value:.0f}까지 상승 예측, 발생 확률 {max_surge_prob:.1f}%로 매우 높음\n"
        analysis += f"  → HUB 용량 확보 및 유입량 조절 즉시 필요!"
    elif max_pred_value >= 280 or max_surge_prob >= 50:
        analysis += f"  → {max_horizon['horizon']}분 후 {max_pred_value:.0f}까지 상승 가능, 급증 확률 {max_surge_prob:.1f}%\n"
        analysis += f"  → 물류 흐름 모니터링 강화 필요"
    else:
        analysis += f"  → 현재 안정적이나 지속적인 모니터링 필요"
    
    return analysis

def generate_llm_analysis(result):
    """LLM으로 M14 예측 결과 분석"""
    predictions = result['predictions']
    current_val = result['current_value']
    current_status = result['current_status']
    
    # 현재 지표 데이터
    current_m14b = result.get('current_m14b', 0)
    current_m14bsum = result.get('current_m14bsum', 0)
    current_gap = result.get('current_gap', 0)
    current_trans = result.get('current_trans', 0)
    
    # 최대 위험도
    max_danger = max(p['danger_probability'] for p in predictions)
    
    # 위험 요인 분석
    risk_factors = []
    if current_m14b > 520:
        risk_factors.append(f"M14AM14B({current_m14b:.0f})가 심각 임계값(520) 초과")
    elif current_m14b > 517:
        risk_factors.append(f"M14AM14B({current_m14b:.0f})가 주의 임계값(517) 초과")
    
    if current_m14bsum > 588:
        risk_factors.append(f"M14AM14BSUM({current_m14bsum:.0f})이 심각 임계값(588) 초과")
    elif current_m14bsum > 576:
        risk_factors.append(f"M14AM14BSUM({current_m14bsum:.0f})이 주의 임계값(576) 초과")
    
    if current_gap > 300:
        risk_factors.append(f"queue_gap({current_gap:.0f})이 위험 임계값(300) 초과")
    
    if current_trans > 180:
        risk_factors.append(f"TRANSPORT({current_trans:.0f})가 심각 임계값(180) 초과")
    elif current_trans > 151:
        risk_factors.append(f"TRANSPORT({current_trans:.0f})가 주의 임계값(151) 초과")
    
    # 프롬프트 구성
    pred_text = ""
    for pred in predictions:
        pred_text += f"{pred['horizon']}분 후: {pred['prediction']:,} (위험도 {pred['danger_probability']}%)\n"
    
    risk_text = "\n- ".join(risk_factors) if risk_factors else "모든 지표 정상 범위"
    
    prompt = f"""You MUST answer in Korean only. Be concise and professional.

현재 AMHS 물류 상황:
- TOTALCNT: {current_val:,} ({current_status})
- M14AM14B: {current_m14b:.0f} (임계값: 517주의/520심각)
- M14AM14BSUM: {current_m14bsum:.0f} (임계값: 576주의/588심각)
- queue_gap: {current_gap:.0f} (임계값: 300위험)
- TRANSPORT: {current_trans:.0f} (임계값: 151주의/180심각)

⚠️ 현재 위험 요인:
- {risk_text}

예측 결과:
{pred_text}

위 데이터를 바탕으로 한국어 3-4문장으로 분석하세요:
1. 왜 위험도가 높은지 구체적 이유 (어떤 지표가 임계값 초과했는지)
2. 예측되는 추세
3. 권장 조치사항

답변:"""
    
    try:
        response = llm(
            prompt,
            max_tokens=250,
            temperature=0.2,
            top_p=0.85,
            repeat_penalty=1.5,
            stop=["질문:", "\n\n\n", "이미지:", "http"]
        )
        
        raw_answer = response['choices'][0]['text'].strip()
        cleaned = clean_llm_response(raw_answer)
        
        # 영어 패턴 감지 → 템플릿 사용
        english_patterns = ['please', 'answer', 'korean', 'following', 'response', 'analysis', 'the ', 'is ', 'are ', 'this ']
        has_english = any(p in cleaned.lower() for p in english_patterns)
        
        # 지표 언급 확인 (위험 요인 있는데 지표 안 말하면 템플릿)
        indicator_keywords = ['M14AM14B', 'M14AM14BSUM', 'queue_gap', 'TRANSPORT', '임계값', '초과', '병목', '적체']
        has_indicator = any(k in cleaned for k in indicator_keywords)
        
        # LLM 응답이 없거나, 짧거나, 영어 섞이거나, 위험요인 있는데 지표 안 말하면 → 템플릿
        if not cleaned or len(cleaned) < 20 or has_english or (risk_factors and not has_indicator):
            logger.warning(f"LLM 응답 부적절, 템플릿 사용: {cleaned[:50] if cleaned else 'empty'}")
            return generate_m14_template_analysis(result, risk_factors, max_danger)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"LLM 분석 실패: {e}")
        return generate_m14_template_analysis(result, risk_factors, max_danger)

def generate_m14_template_analysis(result, risk_factors, max_danger):
    """LLM 실패시 템플릿 기반 M14 분석"""
    predictions = result['predictions']
    max_pred = max(p['prediction'] for p in predictions)
    
    if not risk_factors:
        return "현재 모든 지표가 정상 범위입니다. 예측 결과를 모니터링하며 상황을 지켜보세요."
    
    analysis = f"⚠️ 위험도 {max_danger}% 원인:\n"
    for factor in risk_factors:
        analysis += f"🚨 {factor}\n"
    
    if max_pred >= 1700:
        analysis += f"\n예측 최대값 {max_pred:,}으로 CRITICAL 상태 진입 예상. 즉시 물류 분산 조치가 필요합니다."
    else:
        analysis += f"\n예측 최대값 {max_pred:,}. 지속적인 모니터링이 필요합니다."
    
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