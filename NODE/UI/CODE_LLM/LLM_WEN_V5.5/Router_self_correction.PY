#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 라우터: Self-Correction 엔드포인트 (v2.0)
- 범용 질문 지원 (코드, 데이터, 일반, 지식)
- 교정 히스토리/로그 추적
- 사용자 피드백 (좋아요/싫어요 + 재교정)

사용법:
    from Router_self_correction import router as sc_router
    app.include_router(sc_router, prefix="/api/sc")
"""

import os
import json
import tempfile
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

# Self-Correction 모듈 임포트
from Langgraph_self_correction import (
    SelfCorrectionAgent,
    read_excel_data,
    analyze_excel_column,
    get_correction_history,
    get_correction_by_id,
    update_feedback,
    re_correct,
    get_feedback_stats
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
    question_type: Optional[str] = None  # auto-detect if None
    max_retries: Optional[int] = 3


class FeedbackRequest(BaseModel):
    correction_id: str
    feedback: str  # "good" or "bad"
    feedback_text: Optional[str] = ""


class ReCorrectRequest(BaseModel):
    correction_id: str
    feedback_text: str


class ExcelColumnRequest(BaseModel):
    file_path: str
    column: str


# ========================================
# Endpoints
# ========================================
@router.post("/ask")
async def self_correction_ask(request: SCRequest):
    """
    Self-Correction 루프로 질문 처리 (범용)

    - 질문 유형 자동 감지 (general/code/data/knowledge)
    - 답변 생성 → 검토 → (필요시) 재생성
    - 최대 N회 재시도 (기본 3회)
    - 교정 히스토리 자동 저장
    """
    sc_agent = get_agent()
    result = sc_agent.ask(
        request.question,
        request.context or "",
        request.question_type,
        request.max_retries or 3
    )
    return result


@router.post("/ask_with_excel")
async def ask_with_excel(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """
    엑셀 파일 업로드 후 Self-Correction으로 분석

    - 엑셀 데이터를 컨텍스트로 제공
    - 질문 유형 자동: data
    """
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        sc_agent = get_agent()
        result = sc_agent.ask_with_excel(tmp_path, question)
        return result
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ========================================
# 피드백 API
# ========================================
@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    교정 결과에 대한 사용자 피드백

    - feedback: "good" (좋아요) 또는 "bad" (싫어요)
    - feedback_text: 추가 의견 (선택)
    """
    if request.feedback not in ("good", "bad"):
        return {"success": False, "error": "feedback은 'good' 또는 'bad'만 가능합니다."}

    success = update_feedback(
        request.correction_id,
        request.feedback,
        request.feedback_text or ""
    )

    if success:
        return {"success": True, "message": f"피드백 저장 완료: {request.feedback}"}
    else:
        return {"success": False, "error": f"교정 기록 없음: {request.correction_id}"}


@router.post("/re_correct")
async def re_correct_endpoint(request: ReCorrectRequest):
    """
    기존 교정 결과에 대해 사용자 피드백을 반영하여 재교정

    - 원본 질문 + 사용자 피드백 텍스트를 반영
    - 새로운 교정 ID 발급
    """
    if not request.feedback_text.strip():
        return {"success": False, "error": "피드백 내용을 입력해주세요."}

    result = re_correct(request.correction_id, request.feedback_text)
    return result


# ========================================
# 히스토리 API
# ========================================
@router.get("/history")
async def get_history(limit: int = 20, question: str = None):
    """
    교정 히스토리 조회

    - limit: 최대 조회 건수 (기본 20)
    - question: 질문 키워드 필터 (선택)
    """
    history = get_correction_history(limit=limit, question_filter=question)
    return {
        "success": True,
        "count": len(history),
        "history": history
    }


@router.get("/history/{correction_id}")
async def get_history_detail(correction_id: str):
    """특정 교정 기록 상세 조회"""
    record = get_correction_by_id(correction_id)
    if record:
        return {"success": True, "record": record}
    return {"success": False, "error": f"교정 기록 없음: {correction_id}"}


@router.get("/stats")
async def get_stats():
    """
    교정 통계 (피드백 요약)

    - 총 교정 횟수, 좋아요/싫어요 수
    - 평균 재시도 횟수, 1회 통과율
    """
    stats = get_feedback_stats()
    return {"success": True, "stats": stats}


# ========================================
# 엑셀 도구 API
# ========================================
@router.post("/excel/read")
async def read_excel(file: UploadFile = File(...)):
    """엑셀 파일 읽기 (미리보기)"""
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
    """엑셀 특정 컬럼 상세 분석"""
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


# ========================================
# 상태 API
# ========================================
@router.get("/status")
async def sc_status():
    """Self-Correction 상태 확인"""
    stats = get_feedback_stats()
    return {
        "enabled": True,
        "version": "2.0",
        "features": [
            "범용 질문 지원 (general/code/data/knowledge)",
            "질문 유형 자동 감지",
            "교정 히스토리 추적",
            "사용자 피드백 (좋아요/싫어요)",
            "피드백 반영 재교정"
        ],
        "max_retries": 3,
        "model": "gpt-oss-20b (COMMON 20B)",
        "stats": stats
    }
