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


def generate_status_summary(status_text: str) -> str:
    """
    상태 분석 텍스트를 파싱해서 상세 설명 자동 생성
    """
    lines = status_text.split('\n')
    
    normal_items = []
    caution_items = []  # 🟡 관심
    warning_items = []  # ⚠️ 주의
    critical_items = [] # 🔴 위험
    
    for line in lines:
        line = line.strip()
        if '✅' in line and '종합' not in line:
            # ✅ M14AM14B: 262.0 → 정상
            match = re.search(r'✅\s*([^:]+):\s*([\d.]+)', line)
            if match:
                normal_items.append((match.group(1).strip(), match.group(2)))
        elif '🟡' in line:
            # 🟡 TRANSPORT: 149.0 → 관심 (≥ 145)
            match = re.search(r'🟡\s*([^:]+):\s*([\d.]+).*?≥\s*([\d.]+)', line)
            if match:
                caution_items.append((match.group(1).strip(), match.group(2), match.group(3)))
        elif '⚠️' in line:
            match = re.search(r'⚠️\s*([^:]+):\s*([\d.]+).*?≥\s*([\d.]+)', line)
            if match:
                warning_items.append((match.group(1).strip(), match.group(2), match.group(3)))
        elif '🚨' in line and '종합' not in line:
            # 🚨 M14AM14BSUM: 614.0 → 심각 (≥ 588)
            match = re.search(r'🚨\s*([^:]+):\s*([\d.]+).*?≥\s*([\d.]+)', line)
            if match:
                critical_items.append((match.group(1).strip(), match.group(2), match.group(3)))
        elif '🔴' in line:
            match = re.search(r'🔴\s*([^:]+):\s*([\d.]+).*?≥\s*([\d.]+)', line)
            if match:
                critical_items.append((match.group(1).strip(), match.group(2), match.group(3)))
    
    # 설명 생성 (심각한 것부터!)
    parts = []
    
    # 위험/심각 항목 (가장 먼저!)
    if critical_items:
        for name, value, threshold in critical_items:
            parts.append(f"🚨 {name}({value})이 심각 구간({threshold} 이상)! 즉시 조치 필요!")
    
    # 주의 항목
    for name, value, threshold in warning_items:
        parts.append(f"⚠️ {name}({value})이 주의 구간({threshold} 이상). 점검 필요.")
    
    # 관심 항목
    for name, value, threshold in caution_items:
        parts.append(f"{name}({value})이 기준값({threshold}) 이상으로 관심 구간 진입. 모니터링 권장.")
    
    # 심각/주의 항목 있으면 → 정상 항목 생략, 경고 마무리
    if critical_items:
        parts.append(f"⚠️ 총 {len(critical_items)}개 항목이 심각 상태입니다. 즉시 점검하세요!")
    elif warning_items:
        parts.append(f"⚠️ 총 {len(warning_items)}개 항목이 주의 상태입니다. 점검이 필요합니다.")
    elif normal_items and not caution_items:
        # 전부 정상일 때만 정상 언급
        names = ', '.join([item[0] for item in normal_items[:3]])
        if len(normal_items) > 3:
            names += f" 등 {len(normal_items)}개"
        parts.append(f"{names} 모두 정상 범위입니다.")
    
    if not parts:
        return "모든 항목이 정상 범위 내에 있습니다."
    
    return ' '.join(parts)


def get_llm_analysis(data_text: str, llm, data_type: str = "m14") -> str:
    """
    LLM 분석 호출 + 후처리
    """
    if llm is None:
        return "⚠️ LLM 모델이 로드되지 않았습니다."
    
    try:
        # 상태 분석 부분 추출
        if "📊 상태 분석" in data_text:
            status_part = data_text.split("📊 상태 분석")[1][:300]
        else:
            status_part = data_text[:300]
        
        prompt = f"""<|im_start|>system
한국어로만 답변하세요. 영어 금지. 생각 과정 없이 바로 답변하세요.
<|im_end|>
<|im_start|>user
{status_part}

위 상태를 한국어 2문장으로 요약하세요.
<|im_end|>
<|im_start|>assistant
"""
        
        response = llm(
            prompt,
            max_tokens=80,
            temperature=0.3,
            stop=["<|im_end|>", "\n\n", "---"]
        )
        
        raw_analysis = response['choices'][0]['text'].strip()
        logger.info(f"LLM 원본: {raw_analysis[:200]}")
        
        # <think> 태그 제거
        raw_analysis = re.sub(r'<think>.*?</think>', '', raw_analysis, flags=re.DOTALL).strip()
        raw_analysis = re.sub(r'<[^>]+>', '', raw_analysis).strip()  # 모든 태그 제거
        
        # 빈 응답 또는 영어 thinking 감지 → 템플릿 기반 응답
        if not raw_analysis or len(raw_analysis) < 5:
            return generate_status_summary(status_part)
        
        if "let me" in raw_analysis.lower() or "okay" in raw_analysis.lower():
            return generate_status_summary(status_part)
        
        # 후처리
        analysis = clean_llm_response(raw_analysis, max_lines=3)
        logger.info(f"LLM 후처리: {analysis[:200] if analysis else '없음'}")
        
        if analysis and len(analysis) > 10:
            return analysis
        else:
            return generate_status_summary(status_part)
            
    except Exception as e:
        logger.warning(f"LLM 분석 실패: {e}")
        # 상태 분석 부분 추출 시도
        if "📊 상태 분석" in data_text:
            status_part = data_text.split("📊 상태 분석")[1][:300]
            return generate_status_summary(status_part)
        return "상태 분석을 확인해주세요."


def get_prediction_llm_analysis(data_text: str, llm) -> str:
    """
    예측 분석용 LLM 호출 + 후처리
    """
    # 예측 분석 부분 추출
    if "🔮 예측 분석" not in data_text:
        return "예측 데이터가 없습니다."
    
    pred_part = data_text.split("🔮 예측 분석")[1][:500]
    
    # 핵심 정보 추출
    import re
    
    current_match = re.search(r'현재TOTALCNT:\s*([\d,]+)', pred_part)
    pred_match = re.search(r'보정예측:\s*([\d,]+)', pred_part)
    actual_match = re.search(r'실제값:\s*([\d,]+)', pred_part)
    error_match = re.search(r'예측 오차:\s*([+\-]?[\d,]+)', pred_part)
    error_rate_match = re.search(r'오차율:\s*([\d.]+)%', pred_part)
    direction_match = re.search(r'(과소예측|과대예측|정확)', pred_part)
    
    current_val = current_match.group(1) if current_match else "?"
    pred_val = pred_match.group(1) if pred_match else "?"
    actual_val = actual_match.group(1) if actual_match else "?"
    error_val = error_match.group(1) if error_match else "?"
    error_rate = error_rate_match.group(1) if error_rate_match else "?"
    direction = direction_match.group(1) if direction_match else "?"
    
    # LLM 없으면 템플릿
    if llm is None:
        return generate_prediction_summary(current_val, pred_val, actual_val, error_val, error_rate, direction)
    
    try:
        prompt = f"""<|im_start|>system
한국어로만 답변하세요. 영어 금지. 바로 답변하세요.
<|im_end|>
<|im_start|>user
예측 결과:
- 현재값: {current_val}
- 예측값: {pred_val}
- 실제값: {actual_val}
- 오차: {error_val} ({error_rate}%)
- 방향: {direction}

위 예측 결과를 한국어 2문장으로 해석하세요.
<|im_end|>
<|im_start|>assistant
"""
        
        response = llm(
            prompt,
            max_tokens=100,
            temperature=0.3,
            stop=["<|im_end|>", "\n\n"]
        )
        
        raw = response['choices'][0]['text'].strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = re.sub(r'<[^>]+>', '', raw).strip()
        
        # 영어 감지 → 템플릿 폴백
        english_patterns = ['let me', 'let\'s', 'okay', 'the ', 'this ', 'user', 'want', 'given', 'result']
        has_english = any(p in raw.lower() for p in english_patterns)
        
        if not raw or len(raw) < 10 or has_english:
            return generate_prediction_summary(current_val, pred_val, actual_val, error_val, error_rate, direction)
        
        return clean_llm_response(raw, max_lines=3)
        
    except Exception as e:
        logger.warning(f"예측 LLM 분석 실패: {e}")
        return generate_prediction_summary(current_val, pred_val, actual_val, error_val, error_rate, direction)


def generate_prediction_summary(current_val, pred_val, actual_val, error_val, error_rate, direction) -> str:
    """예측 분석 템플릿"""
    try:
        error_rate_f = float(error_rate.replace(',', ''))
    except:
        error_rate_f = 0
    
    if error_rate_f <= 2:
        accuracy = "우수"
        emoji = "✅"
    elif error_rate_f <= 5:
        accuracy = "양호"
        emoji = "🟡"
    else:
        accuracy = "개선 필요"
        emoji = "⚠️"
    
    result = f"{emoji} 예측 정확도 {accuracy} (오차 {error_rate}%). "
    
    if direction == "과소예측":
        result += f"실제값({actual_val})이 예측({pred_val})보다 높음. 예상보다 물량 증가!"
    elif direction == "과대예측":
        result += f"실제값({actual_val})이 예측({pred_val})보다 낮음. 예상보다 물량 감소."
    else:
        result += f"예측이 정확했습니다."
    
    return result


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