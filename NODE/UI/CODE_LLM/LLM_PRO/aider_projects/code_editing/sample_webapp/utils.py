"""유틸리티 함수"""

import json
import os
from datetime import datetime


def load_json(filepath: str) -> dict:
    """JSON 파일 로드"""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filepath: str, data: dict):
    """JSON 파일 저장"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_datetime(dt: datetime) -> str:
    """날짜 포맷팅"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def validate_email(email: str) -> bool:
    """이메일 유효성 검사 (간단)"""
    return "@" in email and "." in email.split("@")[-1]


def truncate(text: str, max_len: int = 50) -> str:
    """텍스트 자르기"""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
