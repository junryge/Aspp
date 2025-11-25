#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 응답 후처리 모듈
- 마크다운 제거
- URL/이미지 환각 제거
- 프롬프트 반복 제거
"""

import re
from typing import Optional

def clean_llm_response(text: str, max_lines: int = 3) -> str:
    """
    LLM 응답 후처리: 마크다운, URL, 환각 제거
    
    Args:
        text: LLM 원본 응답
        max_lines: 최대 반환 줄 수 (기본 3줄)
    
    Returns:
        정제된 텍스트
    """
    if not text:
        return ""
    
    # 1. 마크다운/특수문법 제거
    text = re.sub(r'```[\s\S]*?```', '', text)  # 코드 블록
    text = re.sub(r'`[^`]+`', '', text)  # 인라인 코드
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)  # 헤더
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 볼드
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # 이탤릭
    text = re.sub(r'\[{1,2}[^\]]*\]{1,2}', '', text)  # [태그], [[태그]]
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)  # 리스트
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # 숫자 리스트
    text = re.sub(r'[=\-]{3,}', '', text)  # ===, --- 구분선
    
    # 2. 프롬프트 반복 제거
    text = re.sub(r'데이터에서 주어진.*', '', text)
    text = re.sub(r'위 데이터를.*', '', text)
    text = re.sub(r'분석\s*\(한국어.*', '', text)
    text = re.sub(r'답변\s*\(한국어.*', '', text)
    
    # 3. 환각 제거: URL, 이미지 참조
    text = re.sub(r'https?://[^\s\)]+', '', text)  # URL 제거
    text = re.sub(r'www\.[^\s\)]+', '', text)  # www. 제거
    text = re.sub(r'\([^)]*https?[^)]*\)', '', text)  # (URL) 제거
    text = re.sub(r'이미지\s*:\s*.*', '', text, flags=re.MULTILINE)  # 이미지: 라인 제거
    text = re.sub(r'그림\s*:\s*.*', '', text, flags=re.MULTILINE)  # 그림: 라인 제거
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # ![alt](url) 제거
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # [text](url) 제거
    
    # 4. 빈 괄호 제거
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    
    # 5. 반복 라인 제거 및 정리
    lines = text.split('\n')
    seen = set()
    unique_lines = []
    for line in lines:
        line_clean = line.strip()
        if line_clean and line_clean not in seen and len(line_clean) > 2:
            seen.add(line_clean)
            unique_lines.append(line_clean)
    
    return '\n'.join(unique_lines[:max_lines])


def remove_markdown(text: str) -> str:
    """마크다운 문법만 제거"""
    if not text:
        return ""
    
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[=\-]{3,}', '', text)
    
    return text.strip()


def remove_hallucinations(text: str) -> str:
    """URL, 이미지 등 환각 제거"""
    if not text:
        return ""
    
    text = re.sub(r'https?://[^\s\)]+', '', text)
    text = re.sub(r'www\.[^\s\)]+', '', text)
    text = re.sub(r'\([^)]*https?[^)]*\)', '', text)
    text = re.sub(r'이미지\s*:\s*.*', '', text, flags=re.MULTILINE)
    text = re.sub(r'그림\s*:\s*.*', '', text, flags=re.MULTILINE)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    
    return text.strip()


def remove_duplicates(text: str, max_lines: Optional[int] = None) -> str:
    """중복 라인 제거"""
    if not text:
        return ""
    
    lines = text.split('\n')
    seen = set()
    unique_lines = []
    
    for line in lines:
        line_clean = line.strip()
        if line_clean and line_clean not in seen and len(line_clean) > 2:
            seen.add(line_clean)
            unique_lines.append(line_clean)
    
    if max_lines:
        unique_lines = unique_lines[:max_lines]
    
    return '\n'.join(unique_lines)


# 테스트
if __name__ == "__main__":
    test_text = """
    **분석 결과**
    
    정상 상태입니다. 현재 큐 수는 1304개로, 병목 지표에 따라 심각도가 낮습니다.
    이미지: (https://s7-98dxdyjzqf.s3.ap-northeast-1.amazonaws.com/queue.png)
    정상 상태입니다. 현재 큐 수는 1,400이 넘지 않으며 병목 지표에 따라 심각도가 낮습니다.
    
    - 리스트 항목 1
    - 리스트 항목 2
    
    위 데이터를 분석하면...
    """
    
    print("원본:")
    print(test_text)
    print("\n" + "="*50 + "\n")
    print("정제 후:")
    print(clean_llm_response(test_text))