#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM만 사용하는 초간단 웹 서버
"""

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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
    
    MODEL_PATH = "./models/qwen3-7b-instruct-q4_k_m.gguf"
    
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
            n_ctx=512,
            n_batch=64,
            n_gpu_layers=0,
            n_threads=4,
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
    """메인 페이지"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LLM 채팅</title>
    <style>
        body {
            font-family: Arial;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        .input-box {
            margin: 20px 0;
        }
        input {
            width: 70%;
            padding: 10px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background: #45a049;
        }
        #result {
            margin-top: 20px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 5px;
            white-space: pre-wrap;
            min-height: 100px;
        }
        .loading {
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h1>🤖 LLM 채팅</h1>
    
    <div class="input-box">
        <input type="text" id="question" placeholder="질문을 입력하세요...">
        <button onclick="ask()">전송</button>
    </div>
    
    <div id="result">여기에 답변이 표시됩니다...</div>
    
    <script>
        async function ask() {
            const question = document.getElementById('question').value;
            const resultDiv = document.getElementById('result');
            
            if (!question.trim()) {
                resultDiv.textContent = '질문을 입력하세요!';
                return;
            }
            
            resultDiv.innerHTML = '<span class="loading">답변 생성 중...</span>';
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });
                
                const data = await response.json();
                resultDiv.textContent = data.answer;
                
            } catch (error) {
                resultDiv.textContent = '❌ 오류: ' + error;
            }
        }
        
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') ask();
        });
    </script>
</body>
</html>
    """)

@app.post("/ask")
async def ask(query: Query):
    """LLM에게 질문"""
    
    if llm is None:
        return {"answer": "❌ LLM이 로드되지 않았습니다. 서버 로그를 확인하세요."}
    
    try:
        logger.info(f"질문: {query.question}")
        
        # LLM 호출
        response = llm(
            query.question,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            stop=["\n\n"]
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