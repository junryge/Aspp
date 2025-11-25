#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 응답 후처리 모듈
- LLM 분석 호출
- 마크다운 제거
- URL/이미지 환각 제거
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_llm_analysis(data_text: str, llm, data_type: str = "m14") -> str:
    """
    LLM 분석 호출 + 후처리
    
    Args:
        data_text: 분석할 데이터 텍스트 (상태 분석 포함)
        llm: LLM 모델 객체
        data_type: "m14" 또는 "hub"
    
    Returns:
        정제된 분석 결과
    """
    if llm is None:
        return "⚠️ LLM 모델이 로드되지 않았습니다."
    
    try:
        # 상태 분석 부분 추출
        if "📊 상태 분석" in data_text:
            status_part = data_text.split("📊 상태 분석")[1][:300]
        else:
            status_part = data_text[:300]
        
        prompt = f"""/no_think
상태 분석 결과:
{status_part}

위 상태 분석을 보고 한국어 2문장으로 요약하세요.
- 정상 항목과 주의 항목을 구분
- 수치 언급

요약:"""
        
        response = llm(
            prompt,
            max_tokens=100,
            temperature=0.5,
            stop=["\n\n\n", "---", "상태 분석"]
        )
        
        raw_analysis = response['choices'][0]['text'].strip()
        logger.info(f"LLM 원본: {raw_analysis[:200]}")
        
        # 후처리
        analysis = clean_llm_response(raw_analysis, max_lines=3)
        logger.info(f"LLM 후처리: {analysis[:200] if analysis else '없음'}")
        
        if analysis:
            return analysis
        elif raw_analysis:
            simple = raw_analysis.replace('```', '').replace('[', '').replace(']', '').strip()
            return simple[:200] if simple else "(분석 생성 실패)"
        else:
            return "(분석 생성 실패)"
            
    except Exception as e:
        logger.warning(f"LLM 분석 실패: {e}")
        return f"⚠️ 분석 실패: {str(e)[:50]}"


def clean_llm_response(text: str, max_lines: int = 5) -> str:
    """
    LLM 응답 후처리: 마크다운, URL, 환각 제거
    
    Args:
        text: LLM 원본 응답
        max_lines: 최대 반환 줄 수 (기본 5줄)
    
    Returns:
        정제된 텍스트
    """
    if not text:
        return ""
    
    # 1. 마크다운 기호만 제거 (내용은 보존!)
    text = text.replace('```', '')  # 코드블록 기호만 제거
    text = text.replace('`', '')  # 인라인 코드 기호만 제거
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)  # 헤더
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 볼드 → 내용만
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # 이탤릭 → 내용만
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)  # [태그] → 내용만
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)  # 리스트 기호 제거
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # 숫자 리스트
    text = re.sub(r'[=\-]{3,}', '', text)  # ===, --- 구분선
    
    # 2. 프롬프트 반복 제거
    text = re.sub(r'데이터에서 주어진.*', '', text)
    text = re.sub(r'위 데이터를.*', '', text)
    text = re.sub(r'분석\s*\(한국어.*', '', text)
    text = re.sub(r'답변\s*\(한국어.*', '', text)
    text = re.sub(r'데이터 분석 결과', '', text)  # 추가
    
    # 3. 환각 제거: URL, 이미지 참조
    text = re.sub(r'https?://[^\s\)]+', '', text)  # URL 제거
    text = re.sub(r'www\.[^\s\)]+', '', text)  # www. 제거
    text = re.sub(r'\([^)]*https?[^)]*\)', '', text)  # (URL) 제거
    text = re.sub(r'이미지\s*:\s*.*', '', text, flags=re.MULTILINE)  # 이미지: 라인 제거
    text = re.sub(r'그림\s*:\s*.*', '', text, flags=re.MULTILINE)  # 그림: 라인 제거
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # ![alt](url) 제거
    
    # 4. 빈 괄호 제거
    text = re.sub(r'\(\s*\)', '', text)
    
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