#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터DB + 컬럼 정의 통합 RAG 서버
"""

import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
import pickle
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 전역 변수
llm = None
vectordb = None
embedding_model = None
COLUMN_DEFINITIONS = ""

def load_column_definitions():
    """컬럼 정의 파일 로드"""
    try:
        with open("column_definitions.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"컬럼 정의 로드 실패: {e}")
        return "컬럼 정의 파일을 찾을 수 없습니다."

@app.on_event("startup")
async def startup():
    """서버 시작 시 초기화"""
    global llm, vectordb, embedding_model, COLUMN_DEFINITIONS
    
    # 0. 컬럼 정의 로드
    COLUMN_DEFINITIONS = load_column_definitions()
    logger.info("✅ 컬럼 정의 로드 완료")
    
    # 1. LLM 로드
    MODEL_PATH = "./models/QWEN3-1.7B-18_0.GGUF"
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"LLM 로드 시작: {MODEL_PATH}")
        
        try:
            from llama_cpp import Llama
            
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=1024,
                n_batch=128,
                n_gpu_layers=0,
                n_threads=6,
                verbose=False
            )
            
            logger.info("✅ LLM 로드 성공!")
            
        except Exception as e:
            logger.error(f"❌ LLM 로드 실패: {e}")
    else:
        logger.warning(f"⚠️ LLM 모델 없음: {MODEL_PATH}")
    
    # 2. 벡터DB 로드
    DB_PATH = "./vector_db/vectordb.pkl"
    
    if os.path.exists(DB_PATH):
        logger.info(f"벡터DB 로드 중: {DB_PATH}")
        
        try:
            with open(DB_PATH, 'rb') as f:
                vectordb = pickle.load(f)
            
            logger.info(f"✅ 벡터DB 로드 완료: {len(vectordb['documents'])}개 문서")
            
        except Exception as e:
            logger.error(f"❌ 벡터DB 로드 실패: {e}")
    else:
        logger.warning(f"⚠️ 벡터DB 없음: {DB_PATH}")
    
    # 3. 임베딩 모델 로드
    EMB_PATH = "./embeddings/all-MiniLM-L6-v2"
    
    try:
        from sentence_transformers import SentenceTransformer
        
        if os.path.exists(EMB_PATH):
            logger.info(f"임베딩 모델 로드: {EMB_PATH}")
            embedding_model = SentenceTransformer(EMB_PATH)
        else:
            logger.info("온라인에서 임베딩 모델 다운로드...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("✅ 임베딩 모델 로드 완료")
        
    except Exception as e:
        logger.error(f"❌ 임베딩 모델 로드 실패: {e}")

def search_similar(query, k=3):
    """벡터DB에서 유사 문서 검색"""
    if vectordb is None or embedding_model is None:
        return []
    
    try:
        # 쿼리 임베딩
        query_embedding = embedding_model.encode([query])[0]
        
        # 코사인 유사도 계산
        similarities = []
        for i, doc_embedding in enumerate(vectordb['embeddings']):
            sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((i, sim))
        
        # 상위 k개 선택
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        results = []
        for idx, score in top_k:
            doc = vectordb['documents'][idx]
            results.append({
                'content': doc['content'],
                'stat_dt': doc.get('stat_dt', 'Unknown'),
                'score': float(score)
            })
        
        return results
        
    except Exception as e:
        logger.error(f"검색 실패: {e}")
        return []

class Query(BaseModel):
    question: str

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
        logger.info(f"질문: {query.question}")
        
        # 1. 벡터DB 검색
        search_results = search_similar(query.question, k=3)
        
        # 2. 검색 결과 포맷팅
        context = ""
        if search_results:
            context = "\n\n📈 관련 데이터:\n"
            for i, result in enumerate(search_results, 1):
                context += f"\n[{i}] {result['stat_dt']}\n"
                # 너무 길면 일부만 표시
                content_preview = result['content'][:200]
                context += f"{content_preview}...\n"
        
        # 3. 프롬프트 구성
        prompt = f"""당신은 반도체 제조 AMHS 전문가입니다. 한국어로 답변하세요.

{COLUMN_DEFINITIONS}
{context}

질문: {query.question}
답변:"""
        
        # 4. LLM 호출
        response = llm(
            prompt,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["질문:", "\n\n", "📊"]
        )
        
        answer = response['choices'][0]['text'].strip()
        logger.info(f"답변 생성 완료")
        
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"answer": f"❌ 오류: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)