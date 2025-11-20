#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터DB 검색 전용 서버
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
vectordb = None
embedding_model = None

@app.on_event("startup")
async def startup():
    """서버 시작 시 벡터DB만 로드"""
    global vectordb, embedding_model
    
    # 1. 벡터DB 로드
    DB_PATH = "./vector_db/vectordb.pkl"
    
    if os.path.exists(DB_PATH):
        logger.info(f"벡터DB 로드 중: {DB_PATH}")
        
        try:
            with open(DB_PATH, 'rb') as f:
                vectordb = pickle.load(f)
            
            logger.info(f"✅ 벡터DB 로드 완료: {vectordb['total_docs']}개 문서")
            
        except Exception as e:
            logger.error(f"❌ 벡터DB 로드 실패: {e}")
    else:
        logger.error(f"❌ 벡터DB 없음: {DB_PATH}")
        logger.info("먼저 create_vectordb.py를 실행하세요!")
    
    # 2. 임베딩 모델 로드 (검색용)
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

def search_vectordb(query, k=5):
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
        
        # 상위 k개
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        results = []
        for idx, score in top_k:
            doc = vectordb['documents'][idx]
            results.append({
                'stat_dt': doc.get('stat_dt', 'Unknown'),
                'content': doc['content'],
                'score': round(float(score), 4)
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
    """벡터DB 검색"""
    
    if vectordb is None:
        return {"answer": "❌ 벡터DB가 로드되지 않았습니다."}
    
    try:
        logger.info(f"질문: {query.question}")
        
        # 벡터DB 검색
        results = search_vectordb(query.question, k=5)
        
        if not results:
            return {"answer": "검색 결과가 없습니다."}
        
        # 결과 포맷팅
        answer = f"🔍 '{query.question}' 검색 결과 (상위 {len(results)}개):\n\n"
        
        for i, result in enumerate(results, 1):
            answer += f"[{i}] {result['stat_dt']} (유사도: {result['score']})\n"
            answer += f"{result['content'][:300]}...\n\n"
        
        logger.info(f"검색 완료: {len(results)}개")
        
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        return {"answer": f"❌ 오류: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)