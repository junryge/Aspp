#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 검색 모듈
server.py에서 import하여 사용
"""

import os
import pandas as pd
import re
import json
import logging
from typing import Tuple, Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수
_df = None
_csv_path = None
_column_config = None

# 스크립트 위치 기준 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_column_config() -> dict:
    """컬럼 설정 JSON 로드"""
    global _column_config
    
    config_path = os.path.join(SCRIPT_DIR, 'sort', 'column_config.json')
    
    if not os.path.exists(config_path):
        logger.warning(f"⚠️ 컬럼 설정 파일 없음: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            _column_config = json.load(f)
        logger.info(f"✅ 컬럼 설정 로드 완료: {config_path}")
        return _column_config
    except Exception as e:
        logger.error(f"❌ 컬럼 설정 로드 실패: {e}")
        return {}

def get_column_config() -> dict:
    """현재 컬럼 설정 반환"""
    global _column_config
    if _column_config is None:
        load_column_config()
    return _column_config or {}

def detect_data_type(columns: List[str]) -> str:
    """데이터 타입 자동 감지 (m14/hub)"""
    config = get_column_config()
    
    # 기본 감지 컬럼 (config 없을 때 사용)
    DEFAULT_DETECT = {
        "m14": ["M14AM14B", "M14AM14BSUM", "TOTALCNT", "현재TOTALCNT", "queue_gap", "TRANSPORT"],
        "hub": ["CURRENT_M16A_3F_JOB_2", "HUBROOMTOTAL", "M16A_3F_STORAGE_UTIL", "M16HUB.QUE.ALL.CURRENTQCNT"]
    }
    
    if config:
        m14_detect = config.get('m14', {}).get('detect_columns', DEFAULT_DETECT["m14"])
        hub_detect = config.get('hub', {}).get('detect_columns', DEFAULT_DETECT["hub"])
    else:
        m14_detect = DEFAULT_DETECT["m14"]
        hub_detect = DEFAULT_DETECT["hub"]
    
    m14_count = sum(1 for col in m14_detect if col in columns)
    hub_count = sum(1 for col in hub_detect if col in columns)
    
    if m14_count > hub_count:
        return "m14"
    elif hub_count > 0:
        return "hub"
    else:
        return "unknown"

def has_prediction_data(row: pd.Series) -> bool:
    """예측 데이터가 있는지 확인"""
    required = ['보정예측', '실제값']
    return all(col in row.index and pd.notna(row[col]) for col in required)


def analyze_prediction(row: pd.Series) -> str:
    """
    예측 분석 텍스트 생성
    
    Returns:
        예측 분석 텍스트 (현재값 → 예측값 → 실제값 → 오차)
    """
    if not has_prediction_data(row):
        return ""
    
    # 컬럼 추출
    current_time = row.get('현재시간', 'N/A')
    current_value = row.get('현재TOTALCNT', 0)
    pred_time = row.get('예측시점', 'N/A')
    actual_time = row.get('실제시점', pred_time)
    actual_value = row.get('실제값', 0)
    raw_pred = row.get('원본예측', 0)
    adj_pred = row.get('보정예측', 0)
    error = row.get('오차', 0)
    error_rate = row.get('오차율(%)', 0)
    
    # 숫자 변환
    try:
        current_value = float(current_value) if pd.notna(current_value) else 0
        actual_value = float(actual_value) if pd.notna(actual_value) else 0
        raw_pred = float(raw_pred) if pd.notna(raw_pred) else 0
        adj_pred = float(adj_pred) if pd.notna(adj_pred) else 0
        error = float(error) if pd.notna(error) else 0
        error_rate = float(error_rate) if pd.notna(error_rate) else 0
    except:
        return ""
    
    # 변화량 계산
    change = actual_value - current_value
    change_rate = (change / current_value * 100) if current_value > 0 else 0
    
    # 예측 오차 방향
    if error > 0:
        error_direction = "과소예측"
        error_emoji = "📈"
    elif error < 0:
        error_direction = "과대예측"
        error_emoji = "📉"
    else:
        error_direction = "정확"
        error_emoji = "✅"
    
    # 텍스트 생성
    text = "\n" + "=" * 50 + "\n"
    text += "🔮 예측 분석\n"
    text += "=" * 50 + "\n\n"
    
    # 1. 현재 시점
    text += f"🕐 현재 시점: {current_time}\n"
    text += f"   현재TOTALCNT: {current_value:,.0f}\n\n"
    
    # 2. 예측
    text += f"🎯 예측 시점: {pred_time}\n"
    text += f"   보정예측: {adj_pred:,.0f}\n"
    if raw_pred > 0:
        text += f"   (원본예측: {raw_pred:,.0f})\n"
    text += "\n"
    
    # 3. 실제 결과
    text += f"📊 실제 결과: {actual_time}\n"
    text += f"   실제값: {actual_value:,.0f}\n\n"
    
    # 4. 오차 분석
    text += f"📐 오차 분석\n"
    text += f"   예측 오차: {error:+,.0f} ({error_emoji} {error_direction})\n"
    text += f"   오차율: {abs(error_rate):.1f}%\n\n"
    
    # 5. 변화 분석
    text += f"📈 변화 분석\n"
    text += f"   현재값 → 실제값: {current_value:,.0f} → {actual_value:,.0f}\n"
    text += f"   변화량: {change:+,.0f} ({change_rate:+.1f}%)\n\n"
    
    # 6. 종합 평가
    text += "💡 종합 평가\n"
    
    if abs(error_rate) <= 2:
        text += f"   ✅ 예측 정확도 우수 (오차 {abs(error_rate):.1f}%)\n"
    elif abs(error_rate) <= 5:
        text += f"   🟡 예측 정확도 양호 (오차 {abs(error_rate):.1f}%)\n"
    else:
        text += f"   ⚠️ 예측 정확도 개선 필요 (오차 {abs(error_rate):.1f}%)\n"
    
    if error > 0:
        text += f"   → 실제값({actual_value:,.0f})이 예측({adj_pred:,.0f})보다 {abs(error):,.0f} 높음\n"
        text += f"   → 예상보다 물량 증가!\n"
    elif error < 0:
        text += f"   → 실제값({actual_value:,.0f})이 예측({adj_pred:,.0f})보다 {abs(error):,.0f} 낮음\n"
        text += f"   → 예상보다 물량 감소\n"
    else:
        text += f"   → 예측이 정확했습니다!\n"
    
    # 1700 임계값 체크
    if actual_value >= 1700:
        text += f"\n   🚨 실제값 {actual_value:,.0f} → CRITICAL 상태 (1700 이상)!\n"
    elif actual_value >= 1650:
        text += f"\n   ⚠️ 실제값 {actual_value:,.0f} → CAUTION 상태 (1650 이상)\n"
    elif actual_value >= 1600:
        text += f"\n   🟡 실제값 {actual_value:,.0f} → 주의 구간 (1600 이상)\n"
    
    return text


def analyze_status(row: pd.Series, data_type: str) -> str:
    """임계값 기반 상태 분석"""
    config = get_column_config()
    
    # 기본 임계값 (config 없을 때 사용)
    DEFAULT_THRESHOLDS = {
        "m14": {
            "현재TOTALCNT": {"normal": 1600, "caution": 1650, "critical": 1700},
            "TOTALCNT": {"normal": 1600, "caution": 1650, "critical": 1700},
            "M14AM14B": {"normal": 497, "caution": 517, "critical": 520},
            "M14AM14BSUM": {"normal": 566, "caution": 576, "critical": 588},
            "queue_gap": {"normal": 200, "caution": 300, "critical": 400},
            "TRANSPORT": {"normal": 145, "caution": 151, "critical": 180},
            "OHT_UTIL": {"normal": 83.6, "caution": 84.6, "critical": 85.6}
        },
        "hub": {
            "CURRENT_M16A_3F_JOB_2": {"normal": 270, "caution": 280, "critical": 300},
            "HUBROOMTOTAL": {"normal": 620, "caution": 610, "critical": 590},
            "M16A_3F_STORAGE_UTIL": {"normal": 205, "caution": 206, "critical": 207},
            "CD_M163FSTORAGEUTIL": {"normal": 7, "caution": 8, "critical": 10},
            "M16HUB.QUE.ALL.CURRENTQCNT": {"normal": 1200, "caution": 1300, "critical": 1400}
        }
    }
    
    # config 있으면 사용, 없으면 기본값
    if config and data_type in config:
        thresholds = config[data_type].get('thresholds', {})
    else:
        thresholds = DEFAULT_THRESHOLDS.get(data_type, {})
    
    if not thresholds:
        return ""
    
    results = []
    warnings = 0
    criticals = 0
    
    for col, limits in thresholds.items():
        if col not in row.index or pd.isna(row[col]):
            continue
        
        value = float(row[col])
        
        # HUBROOMTOTAL은 낮을수록 위험 (반대 로직)
        if col == 'HUBROOMTOTAL':
            critical = limits.get('critical', 0)
            caution = limits.get('caution', 0)
            normal = limits.get('normal', 0)
            
            if value < critical:
                results.append(f"🚨 {col}: {value} → 심각 (< {critical})")
                criticals += 1
            elif value < caution:
                results.append(f"⚠️ {col}: {value} → 주의 (< {caution})")
                warnings += 1
            else:
                results.append(f"✅ {col}: {value} → 정상")
        else:
            # 일반 컬럼: 높을수록 위험
            critical = limits.get('critical', float('inf'))
            caution = limits.get('caution', float('inf'))
            normal = limits.get('normal', float('inf'))
            
            if value >= critical:
                results.append(f"🚨 {col}: {value} → 심각 (≥ {critical})")
                criticals += 1
            elif value >= caution:
                results.append(f"⚠️ {col}: {value} → 주의 (≥ {caution})")
                warnings += 1
            elif value >= normal:
                results.append(f"🟡 {col}: {value} → 관심 (≥ {normal})")
            else:
                results.append(f"✅ {col}: {value} → 정상")
    
    if not results:
        return ""
    
    # 종합 판단
    if criticals > 0:
        summary = f"🚨 종합: 심각 ({criticals}개 항목 임계값 초과)"
    elif warnings > 0:
        summary = f"⚠️ 종합: 주의 ({warnings}개 항목 주의 필요)"
    else:
        summary = "✅ 종합: 정상"
    
    analysis_text = "\n📊 상태 분석\n"
    analysis_text += "\n".join(results)
    analysis_text += f"\n\n{summary}"
    
    return analysis_text

def load_csv(csv_path: str) -> bool:
    """CSV 파일 로드"""
    global _df, _csv_path
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ CSV 파일 없음: {csv_path}")
        return False
    
    try:
        _df = pd.read_csv(csv_path, encoding='utf-8')
        _csv_path = csv_path
        logger.info(f"✅ CSV 로드 완료: {len(_df)}행, {len(_df.columns)}컬럼")
        logger.info(f"컬럼: {list(_df.columns[:5])}...")
        return True
    except Exception as e:
        logger.error(f"❌ CSV 로드 실패: {e}")
        return False

def get_df() -> Optional[pd.DataFrame]:
    """현재 로드된 DataFrame 반환"""
    return _df

def get_columns() -> List[str]:
    """컬럼 목록 반환"""
    if _df is None:
        return []
    return list(_df.columns)

def convert_time_format(time_str: str) -> List[str]:
    """
    다양한 시간 형식을 변환하여 검색 가능한 형식 목록 반환
    
    입력 예시:
    - 202509210014 → ['2025-09-21 00:14', '2025-09-21 0:14']
    - 2025-10-14 04:39 → ['2025-10-14 04:39', '2025-10-14 4:39']
    - 202510140439 → ['2025-10-14 04:39', '2025-10-14 4:39']
    """
    formats = []
    time_str = time_str.strip()
    
    # 1. YYYYMMDDHHMM 형식 (12자리 숫자)
    if re.match(r'^\d{12}$', time_str):
        year = time_str[0:4]
        month = time_str[4:6]
        day = time_str[6:8]
        hour = time_str[8:10]
        minute = time_str[10:12]
        
        # 여러 형식 생성
        formats.append(f"{year}-{month}-{day} {int(hour):02d}:{minute}")  # 2025-10-14 04:39
        formats.append(f"{year}-{month}-{day} {int(hour)}:{minute}")      # 2025-10-14 4:39
        formats.append(time_str)  # 원본도 포함
    
    # 2. YYYY-MM-DD HH:MM 형식
    elif re.match(r'^\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}$', time_str):
        formats.append(time_str)
        # 시간 부분 정규화
        date_part, time_part = time_str.rsplit(' ', 1)
        if ':' in time_part:
            h, m = time_part.split(':')
            formats.append(f"{date_part} {int(h):02d}:{m}")
            formats.append(f"{date_part} {int(h)}:{m}")
    
    # 3. 그 외 형식은 원본 그대로
    else:
        formats.append(time_str)
    
    # 중복 제거
    return list(set(formats))


def search_by_time(time_str: str) -> Tuple[Optional[pd.Series], str]:
    """
    시간으로 검색 (YYYYMMDDHHMM 또는 YYYY-MM-DD HH:MM 형식)
    
    Returns:
        (매칭된 행, 설명 텍스트)
    """
    if _df is None:
        return None, "CSV 파일이 로드되지 않았습니다."
    
    # 시간 컬럼 후보
    time_cols = ['현재시간', 'STAT_DT', 'CURRTIME', '시간', 'TIME', 'DATETIME']
    time_col = None
    
    for col in time_cols:
        if col in _df.columns:
            time_col = col
            break
    
    if time_col is None:
        return None, "시간 컬럼을 찾을 수 없습니다."
    
    # 검색 (여러 방식 시도)
    time_col_str = _df[time_col].astype(str)
    
    # 시간 형식 변환 (여러 형식으로)
    search_formats = convert_time_format(time_str)
    logger.info(f"시간 검색 형식: {search_formats}")
    
    result = pd.DataFrame()
    
    # 각 형식으로 검색 시도
    for fmt in search_formats:
        matched = _df[time_col_str.str.contains(fmt, na=False, regex=False)]
        if not matched.empty:
            result = matched
            break
    
    if result.empty:
        # 유사한 시간 제안
        sample_times = time_col_str.head(5).tolist()
        return None, f"시간 '{time_str}'에 해당하는 데이터가 없습니다.\n변환 시도: {search_formats}\n예시: {sample_times}"
    
    row = result.iloc[0]
    
    # JSON 설정에서 컬럼 로드
    config = get_column_config()
    
    m14_config = config.get('m14', {})
    hub_config = config.get('hub', {})
    
    m14_display = m14_config.get('display_columns', [])
    hub_display = hub_config.get('display_columns', [])
    m14_detect = m14_config.get('detect_columns', [])
    hub_detect = hub_config.get('detect_columns', [])
    
    # 데이터 타입 자동 감지
    m14_found = sum(1 for col in m14_detect if col in row.index and pd.notna(row[col]))
    hub_found = sum(1 for col in hub_detect if col in row.index and pd.notna(row[col]))
    
    # 결과 포맷팅
    data_text = f"시간: {row[time_col]}\n"
    data_text += "-" * 40 + "\n"
    
    if m14_found > hub_found and m14_display:
        # M14 데이터
        data_type = "m14"
        icon = m14_config.get('icon', '📦')
        name = m14_config.get('name', 'M14')
        data_text += f"{icon} [{name}]\n"
        for col in m14_display:
            if col in row.index and pd.notna(row[col]):
                data_text += f"{col}: {row[col]}\n"
    elif hub_found > 0 and hub_display:
        # HUB 데이터
        data_type = "hub"
        icon = hub_config.get('icon', '🏭')
        name = hub_config.get('name', 'HUB')
        data_text += f"{icon} [{name}]\n"
        for col in hub_display:
            if col in row.index and pd.notna(row[col]):
                data_text += f"{col}: {row[col]}\n"
    else:
        # 둘 다 아니면 전체 표시
        data_type = "unknown"
        data_text += "📊 [전체 데이터]\n"
        for col in row.index:
            if col != time_col and pd.notna(row[col]):
                data_text += f"{col}: {row[col]}\n"
    
    # ⭐ 예측 분석 추가 (보정예측, 실제값 있으면)
    if has_prediction_data(row):
        pred_analysis = analyze_prediction(row)
        data_text += pred_analysis
    
    # 상태 분석 추가
    analysis = analyze_status(row, data_type)
    if analysis:
        data_text += "\n" + analysis
    
    return row, data_text

def search_by_columns(col_names: List[str], n_rows: int = 5) -> Tuple[Optional[pd.DataFrame], str]:
    """
    컬럼명으로 검색 (최근 n개 데이터)
    
    Args:
        col_names: 검색할 컬럼명 리스트
        n_rows: 반환할 행 수
    
    Returns:
        (매칭된 DataFrame, 설명 텍스트)
    """
    if _df is None:
        return None, "CSV 파일이 로드되지 않았습니다."
    
    # 유효한 컬럼만 필터
    valid_cols = [c for c in col_names if c in _df.columns]
    
    if not valid_cols:
        return None, f"유효한 컬럼이 없습니다. 사용 가능: {list(_df.columns[:10])}..."
    
    recent = _df.tail(n_rows)
    data_text = f"최근 {n_rows}개 데이터:\n\n"
    
    # 시간 컬럼
    time_cols = ['현재시간', 'STAT_DT', 'CURRTIME', '시간']
    time_col = None
    for tc in time_cols:
        if tc in _df.columns:
            time_col = tc
            break
    
    for idx, row in recent.iterrows():
        if time_col:
            data_text += f"[{row[time_col]}]\n"
        else:
            data_text += f"[Row {idx}]\n"
        
        for col in valid_cols:
            data_text += f"  {col}: {row[col]}\n"
        data_text += "\n"
    
    return recent[valid_cols], data_text

def search_csv(query: str) -> Tuple[Optional[Any], str]:
    """
    자연어 쿼리로 CSV 검색 (기존 server.py 함수와 호환)
    
    Args:
        query: 검색 쿼리 (시간 또는 컬럼명 포함)
    
    Returns:
        (결과 데이터, 설명 텍스트)
    """
    if _df is None:
        return None, "CSV 파일이 로드되지 않았습니다."
    
    # 0. 시간 범위 패턴 먼저 체크 (2025-10-14 4:45 ~ 2025-10-14 5:50)
    range_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})\s*[~\-]\s*(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})'
    range_match = re.search(range_pattern, query)
    
    if range_match:
        start_time = range_match.group(1)
        end_time = range_match.group(2)
        logger.info(f"시간 범위 검색: {start_time} ~ {end_time}")
        return search_time_range(start_time, end_time)
    
    # 0-1. 날짜 + 조건 패턴 체크 (2025-10-14 에서 1700이상)
    # 패턴: 날짜 + (이상|이하|초과|미만) + 숫자  또는  날짜 + 숫자 + (이상|이하|초과|미만)
    date_condition_patterns = [
        r'(\d{4}-\d{2}-\d{2}).*?(\d+(?:\.\d+)?)\s*(이상|이하|초과|미만)',  # 2025-10-14 1700 이상
        r'(\d{4}-\d{2}-\d{2}).*?(이상|이하|초과|미만)\s*(\d+(?:\.\d+)?)',  # 2025-10-14 이상 1700
    ]
    
    for pattern in date_condition_patterns:
        match = re.search(pattern, query)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                date_str = groups[0]
                if groups[1] in ['이상', '이하', '초과', '미만']:
                    value = float(groups[0] if groups[0].replace('.','').isdigit() else groups[2])
                    operator = groups[1]
                else:
                    value = float(groups[1])
                    operator = groups[2]
                
                # 컬럼 추출 (지정 안 하면 기본 타겟)
                col_pattern = r'([A-Z][A-Z0-9_\.]+)'
                col_match = re.search(col_pattern, query, re.IGNORECASE)
                target_col = col_match.group(1) if col_match and col_match.group(1) in _df.columns else None
                
                logger.info(f"날짜+조건 검색: {date_str}, {operator} {value}, 컬럼={target_col}")
                return search_date_condition(date_str, operator, value, target_col)
    
    # 1. 시간 패턴 추출 (202509210013 또는 2025-10-14 4:39 형식)
    time_patterns = [
        r'(\d{12})',  # 202509210013
        r'(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})',  # 2025-10-14 4:39
    ]
    
    time_str = None
    for pattern in time_patterns:
        match = re.search(pattern, query)
        if match:
            time_str = match.group(1)
            break
    
    # 2. 컬럼명 추출 (시간 부분 제외하고)
    query_without_time = query
    if time_str:
        query_without_time = query.replace(time_str, '')
    
    # 컬럼 패턴: 영문대문자, 한글, 숫자, _, ., (, ) 포함
    col_pattern = r'([A-Z가-힣][A-Z가-힣0-9_\.\(\)]+)'
    col_matches = re.findall(col_pattern, query_without_time, re.IGNORECASE)
    
    # 실제 존재하는 컬럼만 필터
    valid_cols = [c for c in col_matches if c in _df.columns]
    
    # 요청했지만 없는 컬럼 찾기 (영문 포함된 것만 - "시점의" 같은 한글 제외)
    invalid_cols = [c for c in col_matches 
                    if c not in _df.columns 
                    and len(c) > 2 
                    and re.search(r'[A-Z]', c, re.IGNORECASE)]  # 영문 포함된 것만
    
    # 3. 시간 + 컬럼 둘 다 있으면 → 해당 시간의 특정 컬럼값만
    if time_str and valid_cols:
        logger.info(f"시간+컬럼 검색: {time_str}, {valid_cols}")
        row, _ = search_by_time(time_str)
        
        if row is None:
            return None, f"시간 '{time_str}'에 해당하는 데이터가 없습니다."
        
        # 해당 컬럼값만 반환
        data_text = f"시간: {time_str}\n"
        for col in valid_cols:
            if col in row.index:
                data_text += f"{col}: {row[col]}\n"
        
        return row, data_text
    
    # 3-1. 시간 + 영문 컬럼 요청했는데 컬럼이 없으면 → 에러
    if time_str and invalid_cols and not valid_cols:
        logger.info(f"컬럼 없음: {invalid_cols}")
        # 유사 컬럼 추천
        similar_cols = []
        for inv_col in invalid_cols:
            for col in _df.columns:
                if inv_col.upper() in col.upper() or col.upper() in inv_col.upper():
                    similar_cols.append(col)
        
        error_msg = f"❌ 컬럼 '{', '.join(invalid_cols)}'이(가) 현재 로드된 CSV에 없습니다.\n"
        error_msg += f"\n⚠️ DB 또는 CSV 파일을 확인해주세요.\n"
        if similar_cols:
            error_msg += f"\n💡 유사한 컬럼: {', '.join(similar_cols[:5])}\n"
        error_msg += f"\n📋 현재 CSV 컬럼 ({len(_df.columns)}개):\n"
        error_msg += ", ".join(list(_df.columns)[:15]) + "..."
        
        return None, error_msg
    
    # 4. 시간만 있으면 → 전체 행 데이터
    if time_str:
        logger.info(f"시간 검색: {time_str}")
        return search_by_time(time_str)
    
    # 5. 컬럼만 있으면 → 최근 데이터
    if valid_cols:
        logger.info(f"컬럼 검색: {valid_cols}")
        return search_by_columns(valid_cols)
    
    return None, "검색 조건을 찾을 수 없습니다. 시간(예: 2025-10-14 4:39) 또는 컬럼명을 포함해주세요."

def search_date_condition(date_str: str, operator: str, value: float, target_col: str = None) -> Tuple[Optional[pd.DataFrame], str]:
    """
    날짜 + 조건으로 검색 (예: 2025-10-14에서 1700 이상)
    
    Args:
        date_str: 날짜 (YYYY-MM-DD)
        operator: 이상/이하/초과/미만
        value: 비교값
        target_col: 대상 컬럼 (없으면 자동 감지)
    
    Returns:
        (결과 DataFrame, 설명 텍스트)
    """
    if _df is None:
        return None, "CSV 파일이 로드되지 않았습니다."
    
    # 시간 컬럼 찾기
    time_cols = ['현재시간', 'STAT_DT', 'CURRTIME', '시간']
    time_col = None
    for tc in time_cols:
        if tc in _df.columns:
            time_col = tc
            break
    
    if time_col is None:
        return None, "시간 컬럼을 찾을 수 없습니다."
    
    # 타겟 컬럼 결정
    if target_col is None:
        # 데이터 타입 감지해서 기본 타겟 설정
        data_type = detect_data_type(list(_df.columns))
        if data_type == "m14":
            target_col = "현재TOTALCNT" if "현재TOTALCNT" in _df.columns else "TOTALCNT"
        elif data_type == "hub":
            target_col = "CURRENT_M16A_3F_JOB_2"
        else:
            # 숫자 컬럼 중 첫번째
            for col in _df.columns:
                if _df[col].dtype in ['int64', 'float64']:
                    target_col = col
                    break
    
    if target_col not in _df.columns:
        return None, f"❌ 컬럼 '{target_col}'이(가) 없습니다."
    
    try:
        df_copy = _df.copy()
        df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')
        df_copy[target_col] = pd.to_numeric(df_copy[target_col], errors='coerce')
        
        # 날짜 필터
        target_date = pd.to_datetime(date_str).date()
        date_mask = df_copy[time_col].dt.date == target_date
        
        # 조건 필터
        op_map = {
            '이상': '>=',
            '이하': '<=',
            '초과': '>',
            '미만': '<'
        }
        op = op_map.get(operator, '>=')
        
        if op == '>=':
            cond_mask = df_copy[target_col] >= value
        elif op == '<=':
            cond_mask = df_copy[target_col] <= value
        elif op == '>':
            cond_mask = df_copy[target_col] > value
        elif op == '<':
            cond_mask = df_copy[target_col] < value
        else:
            cond_mask = df_copy[target_col] >= value
        
        result = _df[date_mask & cond_mask].copy()
        
        if result.empty:
            return None, f"❌ {date_str}에서 {target_col} {value} {operator} 데이터가 없습니다."
        
        # 데이터 타입 감지
        data_type = detect_data_type(list(result.columns))
        config = get_column_config()
        
        if data_type == "m14":
            icon = "📦"
            name = "M14 물류"
        elif data_type == "hub":
            icon = "🏭"
            name = "HUB 물류"
        else:
            icon = "📊"
            name = "데이터"
        
        # 결과 텍스트 생성
        data_text = f"{icon} [{name}] 조건 검색\n"
        data_text += f"📅 {date_str} | {target_col} {value} {operator} ({len(result)}건)\n"
        data_text += "-" * 50 + "\n"
        
        # 각 행 출력
        for idx, row in result.iterrows():
            time_val = row[time_col]
            if pd.notna(time_val):
                if isinstance(time_val, str):
                    time_str_fmt = time_val
                else:
                    time_str_fmt = time_val.strftime('%Y-%m-%d %H:%M') if hasattr(time_val, 'strftime') else str(time_val)
                
                val = row[target_col]
                data_text += f"{time_str_fmt} : {target_col} = {val}\n"
        
        # 통계
        vals = result[target_col].dropna()
        data_text += "\n" + "=" * 50 + "\n"
        data_text += f"📈 통계\n"
        data_text += f"  건수: {len(result)}건\n"
        data_text += f"  최소: {vals.min():.1f}\n"
        data_text += f"  최대: {vals.max():.1f}\n"
        data_text += f"  평균: {vals.mean():.1f}\n"
        
        # 마지막 행 상태 분석
        last_row = result.iloc[-1]
        last_time = last_row[time_col]
        if isinstance(last_time, str):
            last_time_str = last_time
        else:
            last_time_str = last_time.strftime('%Y-%m-%d %H:%M') if hasattr(last_time, 'strftime') else str(last_time)
        
        data_text += "\n" + "=" * 50 + "\n"
        data_text += f"📊 상태 분석 ({last_time_str} 마지막 데이터)\n"
        
        analysis = analyze_status(last_row, data_type)
        if analysis:
            analysis = analysis.replace("\n📊 상태 분석\n", "\n")
            data_text += analysis
        
        return result, data_text
        
    except Exception as e:
        logger.error(f"날짜+조건 검색 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"❌ 검색 오류: {e}"

def search_time_range(start_time: str, end_time: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    시간 범위로 검색 (요약 + 상태 분석)
    
    Args:
        start_time: 시작 시간
        end_time: 종료 시간
    
    Returns:
        (결과 DataFrame, 설명 텍스트)
    """
    if _df is None:
        return None, "CSV 파일이 로드되지 않았습니다."
    
    # 시간 컬럼 찾기
    time_cols = ['현재시간', 'STAT_DT', 'CURRTIME', '시간']
    time_col = None
    for tc in time_cols:
        if tc in _df.columns:
            time_col = tc
            break
    
    if time_col is None:
        return None, "시간 컬럼을 찾을 수 없습니다."
    
    try:
        df_copy = _df.copy()
        df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')
        
        # 시간 형식 변환
        start_formats = convert_time_format(start_time)
        end_formats = convert_time_format(end_time)
        
        start_dt = pd.to_datetime(start_formats[0])
        end_dt = pd.to_datetime(end_formats[0])
        
        mask = (df_copy[time_col] >= start_dt) & (df_copy[time_col] <= end_dt)
        result = _df[mask].copy()
        
        if result.empty:
            return None, f"❌ {start_time} ~ {end_time} 범위에 데이터가 없습니다."
        
        # 데이터 타입 감지
        data_type = detect_data_type(list(result.columns))
        config = get_column_config()
        
        # 타겟 컬럼 결정
        if data_type == "m14":
            target_col = "현재TOTALCNT" if "현재TOTALCNT" in result.columns else "TOTALCNT"
            icon = config.get('m14', {}).get('icon', '📦')
            name = config.get('m14', {}).get('name', 'M14 물류')
        elif data_type == "hub":
            target_col = "CURRENT_M16A_3F_JOB_2"
            icon = config.get('hub', {}).get('icon', '🏭')
            name = config.get('hub', {}).get('name', 'HUB 물류')
        else:
            target_col = result.columns[2] if len(result.columns) > 2 else result.columns[0]
            icon = "📊"
            name = "데이터"
        
        # 결과 텍스트 생성
        data_text = f"{icon} [{name}]\n"
        data_text += f"📅 {start_time} ~ {end_time} ({len(result)}건)\n"
        data_text += "-" * 40 + "\n"
        
        # 각 행의 시간: 타겟값 출력
        for idx, row in result.iterrows():
            time_val = row[time_col]
            if pd.notna(time_val):
                if isinstance(time_val, str):
                    time_str = time_val
                else:
                    time_str = time_val.strftime('%Y-%m-%d %H:%M') if hasattr(time_val, 'strftime') else str(time_val)
                
                if target_col in row.index and pd.notna(row[target_col]):
                    data_text += f"{time_str} : {target_col} = {row[target_col]}\n"
                else:
                    data_text += f"{time_str}\n"
        
        # 마지막 행 상태 분석
        last_row = result.iloc[-1]
        last_time = last_row[time_col]
        if isinstance(last_time, str):
            last_time_str = last_time
        else:
            last_time_str = last_time.strftime('%Y-%m-%d %H:%M') if hasattr(last_time, 'strftime') else str(last_time)
        
        data_text += "\n" + "=" * 40 + "\n"
        data_text += f"📊 상태 분석 ({last_time_str} 마지막 데이터)\n"
        
        analysis = analyze_status(last_row, data_type)
        if analysis:
            # "📊 상태 분석" 제목 제거 (중복 방지)
            analysis = analysis.replace("\n📊 상태 분석\n", "\n")
            data_text += analysis
        else:
            data_text += "✅ 상태 분석 정보 없음"
        
        return result, data_text
        
    except Exception as e:
        logger.error(f"시간 범위 검색 오류: {e}")
        return None, f"❌ 시간 파싱 오류: {e}"

def search_range(start_time: str, end_time: str) -> Tuple[Optional[pd.DataFrame], str]:
    """
    시간 범위로 검색
    
    Args:
        start_time: 시작 시간
        end_time: 종료 시간
    
    Returns:
        (결과 DataFrame, 설명 텍스트)
    """
    if _df is None:
        return None, "CSV 파일이 로드되지 않았습니다."
    
    # 시간 컬럼 찾기
    time_cols = ['STAT_DT', '현재시간', 'CURRTIME', '시간']
    time_col = None
    for tc in time_cols:
        if tc in _df.columns:
            time_col = tc
            break
    
    if time_col is None:
        return None, "시간 컬럼을 찾을 수 없습니다."
    
    try:
        df_copy = _df.copy()
        df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')
        
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        mask = (df_copy[time_col] >= start_dt) & (df_copy[time_col] <= end_dt)
        result = _df[mask]
        
        if result.empty:
            return None, f"{start_time} ~ {end_time} 범위에 데이터가 없습니다."
        
        data_text = f"검색 결과: {len(result)}건 ({start_time} ~ {end_time})\n"
        return result, data_text
        
    except Exception as e:
        return None, f"시간 파싱 오류: {e}"

def search_condition(column: str, operator: str, value: Any) -> Tuple[Optional[pd.DataFrame], str]:
    """
    조건으로 검색
    
    Args:
        column: 컬럼명
        operator: 연산자 ('>', '<', '>=', '<=', '==', '!=')
        value: 비교값
    
    Returns:
        (결과 DataFrame, 설명 텍스트)
    """
    if _df is None:
        return None, "CSV 파일이 로드되지 않았습니다."
    
    if column not in _df.columns:
        return None, f"컬럼 '{column}'이 없습니다."
    
    try:
        col_data = pd.to_numeric(_df[column], errors='coerce')
        value = float(value)
        
        if operator == '>':
            mask = col_data > value
        elif operator == '<':
            mask = col_data < value
        elif operator == '>=':
            mask = col_data >= value
        elif operator == '<=':
            mask = col_data <= value
        elif operator == '==':
            mask = col_data == value
        elif operator == '!=':
            mask = col_data != value
        else:
            return None, f"지원하지 않는 연산자: {operator}"
        
        result = _df[mask]
        
        if result.empty:
            return None, f"{column} {operator} {value} 조건에 맞는 데이터가 없습니다."
        
        data_text = f"검색 결과: {len(result)}건 ({column} {operator} {value})\n"
        return result, data_text
        
    except Exception as e:
        return None, f"조건 검색 오류: {e}"

def get_statistics(column: str) -> Dict[str, Any]:
    """
    컬럼 통계 정보
    
    Args:
        column: 컬럼명
    
    Returns:
        통계 딕셔너리
    """
    if _df is None:
        return {'error': 'CSV 파일이 로드되지 않았습니다.'}
    
    if column not in _df.columns:
        return {'error': f"컬럼 '{column}'이 없습니다."}
    
    try:
        col_data = pd.to_numeric(_df[column], errors='coerce').dropna()
        
        return {
            'count': len(col_data),
            'mean': float(col_data.mean()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'median': float(col_data.median()),
            'q25': float(col_data.quantile(0.25)),
            'q75': float(col_data.quantile(0.75))
        }
    except Exception as e:
        return {'error': str(e)}

def format_row(row: pd.Series, important_cols: List[str] = None) -> str:
    """
    행 데이터를 보기 좋게 포맷팅
    
    Args:
        row: pandas Series
        important_cols: 우선 표시할 컬럼 (없으면 전체)
    
    Returns:
        포맷팅된 문자열
    """
    text = ""
    
    if important_cols:
        for col in important_cols:
            if col in row.index and pd.notna(row[col]):
                text += f"{col}: {row[col]}\n"
    else:
        for col in row.index:
            if pd.notna(row[col]):
                text += f"{col}: {row[col]}\n"
    
    return text


# 테스트는 별도 파일에서 실행
# python -c "import csv_searcher; csv_searcher.load_csv('./csv/with.csv'); print(csv_searcher.get_columns())"