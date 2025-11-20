#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 백엔드 서버
"""

import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# LLM 전역 변수
llm = None

@app.on_event("startup")
async def startup():
    """서버 시작 시 LLM 로드"""
    global llm
    
    MODEL_PATH = "./models/QWEN3-1.7B-18_0.GGUF"
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ 모델 파일 없음: {MODEL_PATH}")
        logger.error("models 폴더 확인:")
        if os.path.exists("./models"):
            for f in os.listdir("./models"):
                if f.endswith(".gguf"):
                    logger.error(f"  → {f}")
        return
    
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
        import traceback
        logger.error(traceback.format_exc())

class Query(BaseModel):
    question: str

@app.get("/")
async def home():
    """메인 페이지 - HTML 파일 반환"""
    return FileResponse("index.html")

@app.post("/ask")
async def ask(query: Query):
    """LLM에게 질문"""
    
    if llm is None:
        return {"answer": "❌ LLM이 로드되지 않았습니다. 서버 로그를 확인하세요."}
    
    try:
        logger.info(f"질문: {query.question}")
        
        # 프롬프트 구성
        prompt = f"""당신은 친절한 AI 어시스턴트입니다. 항상 한국어로 답변하세요.

질문: {query.question}
답변:"""
        
        # LLM 호출
        response = llm(
            prompt,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["질문:", "\n\n"]
        )
        
        answer = response['choices'][0]['text'].strip()
        logger.info(f"답변: {answer[:100]}...")
        
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        return {"answer": f"❌ 오류: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)