#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 직접 검색 RAG 서버 (벡터DB 없음)
"""

import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
import pandas as pd
import re

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
    CSV_PATH = "./CSV/2025.CSV"
    
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
    MODEL_PATH = "./models/QWEN3-1.7B-18_0.GGUF"
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)