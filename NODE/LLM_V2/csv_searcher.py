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
    
    if not config:
        return "unknown"
    
    m14_detect = config.get('m14', {}).get('detect_columns', [])
    hub_detect = config.get('hub', {}).get('detect_columns', [])
    
    m14_count = sum(1 for col in m14_detect if col in columns)
    hub_count = sum(1 for col in hub_detect if col in columns)
    
    if m14_count > hub_count:
        return "m14"
    elif hub_count > 0:
        return "hub"
    else:
        return "unknown"

def analyze_status(row: pd.Series, data_type: str) -> str:
    """임계값 기반 상태 분석"""
    config = get_column_config()
    
    if not config or data_type not in config:
        return ""
    
    thresholds = config[data_type].get('thresholds', {})
    
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
    
    # 1. 정확히 포함
    result = _df[time_col_str.str.contains(time_str, na=False, regex=False)]
    
    # 2. 시간 정규화 후 비교 (4:39 vs 04:39)
    if result.empty:
        # 입력 시간에서 날짜와 시간 분리
        if ' ' in time_str:
            date_part, time_part = time_str.rsplit(' ', 1)
            # 시간 부분 정규화 (4:39 -> 04:39, 04:39 -> 4:39)
            if ':' in time_part:
                h, m = time_part.split(':')
                # 두 가지 형식으로 검색
                time_str_padded = f"{date_part} {int(h):02d}:{m}"
                time_str_unpadded = f"{date_part} {int(h)}:{m}"
                
                result = _df[time_col_str.str.contains(time_str_padded, na=False, regex=False) |
                            time_col_str.str.contains(time_str_unpadded, na=False, regex=False)]
    
    if result.empty:
        # 유사한 시간 제안
        sample_times = time_col_str.head(5).tolist()
        return None, f"시간 '{time_str}'에 해당하는 데이터가 없습니다.\n예시: {sample_times}"
    
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
    
    # 4. 시간만 있으면 → 전체 행 데이터
    if time_str:
        logger.info(f"시간 검색: {time_str}")
        return search_by_time(time_str)
    
    # 5. 컬럼만 있으면 → 최근 데이터
    if valid_cols:
        logger.info(f"컬럼 검색: {valid_cols}")
        return search_by_columns(valid_cols)
    
    return None, "검색 조건을 찾을 수 없습니다. 시간(예: 2025-10-14 4:39) 또는 컬럼명을 포함해주세요."

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