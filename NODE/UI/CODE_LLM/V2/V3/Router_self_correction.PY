#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 라우터: Self-Correction 엔드포인트
기존 coding_llm_server.py에 include 하여 사용

사용법:
    from router_self_correction import router as sc_router
    app.include_router(sc_router, prefix="/api/sc")
"""

import os
import json
import tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

# Self-Correction 모듈 임포트
from langgraph_self_correction import (
    SelfCorrectionAgent,
    read_excel_data,
    analyze_excel_column
)

router = APIRouter(tags=["Self-Correction"])

# 에이전트 인스턴스 (싱글톤)
agent = None

def get_agent():
    global agent
    if agent is None:
        agent = SelfCorrectionAgent()
    return agent


# ========================================
# Request Models
# ========================================
class SCRequest(BaseModel):
    question: str
    context: Optional[str] = ""


class ExcelColumnRequest(BaseModel):
    file_path: str
    column: str


# ========================================
# Endpoints
# ========================================
@router.post("/ask")
async def self_correction_ask(request: SCRequest):
    """
    Self-Correction 루프로 질문 처리
    
    - 답변 생성 → 검토 → (필요시) 재생성
    - 최대 3회 재시도
    """
    agent = get_agent()
    result = agent.ask(request.question, request.context)
    return result


@router.post("/ask_with_excel")
async def ask_with_excel(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """
    엑셀 파일 업로드 후 Self-Correction으로 분석
    
    - 엑셀 데이터를 컨텍스트로 제공
    - Self-Correction 루프 실행
    """
    # 임시 파일로 저장
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        agent = get_agent()
        result = agent.ask_with_excel(tmp_path, question)
        return result
    finally:
        # 임시 파일 정리
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/excel/read")
async def read_excel(file: UploadFile = File(...)):
    """
    엑셀 파일 읽기 (미리보기)
    """
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = read_excel_data(tmp_path)
        return result
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/excel/analyze_column")
async def analyze_column(
    file: UploadFile = File(...),
    column: str = Form(...)
):
    """
    엑셀 특정 컬럼 상세 분석
    """
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = analyze_excel_column(tmp_path, column)
        return result
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/status")
async def sc_status():
    """Self-Correction 상태 확인"""
    return {
        "enabled": True,
        "max_retries": 3,
        "model": "gpt-oss-20b (COMMON 20B)"
    }