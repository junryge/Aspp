#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최소한의 FastAPI 웹 서버
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

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
    <title>RAG 시스템</title>
    <style>
        body {
            font-family: Arial;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }
        input {
            width: 70%;
            padding: 10px;
            font-size: 16px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
        }
        #result {
            margin-top: 20px;
            padding: 20px;
            background: #f5f5f5;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <h1>RAG 시스템</h1>
    
    <div>
        <input type="text" id="question" placeholder="질문 입력">
        <button onclick="ask()">질문하기</button>
    </div>
    
    <div id="result"></div>
    
    <script>
        async function ask() {
            const question = document.getElementById('question').value;
            const resultDiv = document.getElementById('result');
            
            resultDiv.textContent = '처리 중...';
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });
                
                const data = await response.json();
                resultDiv.textContent = data.answer;
            } catch (error) {
                resultDiv.textContent = '오류: ' + error;
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
    """질문 처리"""
    # 그냥 질문을 되돌려줌
    return {"answer": f"받은 질문: {query.question}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)