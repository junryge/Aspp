#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIFTER 로그 전처리 모듈 V2.0 (R3 스타일)
- STORAGE-* 메시지 기반 분석
- 층간 이동 구간별 소요시간 정확히 계산
- STB 용어 통일
"""

import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

THRESHOLDS = {
    'entry_wait': (10, 30, 60),      # 입구 대기
    'crane': (15, 30, 60),           # 크레인 동작
    'floor_move': (60, 120, 240),    # 층간 이동: 1분/2분/4분
    'exit_wait': (10, 30, 60),       # 출구 대기
    'total': (120, 240, 420),        # 전체: 2분/4분/7분
}


def parse_time_ex(time_str: str) -> Optional[datetime]:
    if pd.isna(time_str):
        return None
    match = re.search(r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\]', str(time_str))
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S.%f')
        except:
            pass
    return None


def extract_xml_value(text: str, tag: str) -> Optional[str]:
    if pd.isna(text):
        return None
    pattern = rf'<\s*{tag}\s*>\s*([^<]*?)\s*<\s*/\s*{tag}\s*>'
    match = re.search(pattern, str(text), re.IGNORECASE)
    return match.group(1).strip() if match else None


def format_duration(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}초"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}분 {secs}초"


def get_status(seconds: float, seg_type: str) -> tuple:
    normal, caution, critical = THRESHOLDS.get(seg_type, (60, 120, 300))
    if seconds <= normal:
        return ("정상", "✅")
    elif seconds <= caution:
        return ("주의", "🟡")
    elif seconds <= critical:
        return ("경고", "⚠️")
    return ("지연", "🔴")


def get_floor_from_unit(unit_name: str) -> str:
    """유닛명에서 층 추출: _AI323 → 3F, _AO621 → 6F"""
    if not unit_name:
        return ''
    match = re.search(r'_A[IO](\d)', unit_name)
    return f"{match.group(1)}F" if match else ''


def get_fab_info(machine_name: str) -> str:
    if not machine_name:
        return ''
    return 'M14A' if machine_name[0] == '4' else ('M16A' if machine_name[0] == '6' else '')


def analyze_lifter(df: pd.DataFrame) -> Dict:
    result = {
        'carrier_id': None, 'machine_name': None,
        'source_floor': None, 'dest_floor': None,
        'segments': [], 'delays': [],
        'total_duration_sec': 0, 'final_status': 'UNKNOWN',
        'start_time': None, 'end_time': None,
        'direction': '', 'preprocessed_text': ''
    }
    
    df = df.copy()
    df['parsed_time'] = df['TIME_EX'].apply(parse_time_ex)
    df = df.dropna(subset=['parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    
    if df.empty:
        result['preprocessed_text'] = "❌ 시간 파싱 실패"
        return result
    
    # 핵심 이벤트 시간
    times = {
        'entry': None, 'crane_start': None, 'crane_end': None,
        'transfer_start': None, 'transfer_end': None, 'exit': None
    }
    floors = []
    
    for _, row in df.iterrows():
        msg = str(row.get('MESSAGENAME', ''))
        text = str(row.get('TEXT', ''))
        t = row['parsed_time']
        
        if not result['carrier_id']:
            c = row.get('CARRIER')
            if pd.notna(c):
                result['carrier_id'] = str(c).strip().strip("'")
        
        if not result['machine_name']:
            m = row.get('MACHINENAME')
            if pd.notna(m):
                result['machine_name'] = str(m)
        
        # 층 추출
        unit = extract_xml_value(text, 'UNITNAME') or extract_xml_value(text, 'CURRENTUNITNAME')
        if unit:
            fl = get_floor_from_unit(unit)
            if fl and (not floors or floors[-1] != fl):
                floors.append(fl)
        
        # 이벤트 추출
        if 'CARRIERIDREAD' in msg or 'CARRIERWAITIN' in msg:
            if not times['entry']:
                times['entry'] = t
                result['start_time'] = t
        
        if 'CRANEACTIVE' in msg:
            if not times['crane_start']:
                times['crane_start'] = t
        
        if 'CRANEIDLE' in msg:
            times['crane_end'] = t
        
        if 'TRANSFERINITIATED' in msg:
            times['transfer_start'] = t
        
        if 'TRANSFERCOMPLETED' in msg:
            times['transfer_end'] = t
            result['final_status'] = 'COMPLETED'
        
        if 'CARRIERREMOVED' in msg or 'CARRIERWAITOUT' in msg:
            times['exit'] = t
            if not result['end_time']:
                result['end_time'] = t
    
    if not result['end_time'] and times['transfer_end']:
        result['end_time'] = times['transfer_end']
    
    if floors:
        result['source_floor'] = floors[0]
        result['dest_floor'] = floors[-1]
        # 방향 결정
        try:
            src_num = int(floors[0].replace('F', ''))
            dst_num = int(floors[-1].replace('F', ''))
            result['direction'] = '⬆️ 상승' if dst_num > src_num else ('⬇️ 하강' if dst_num < src_num else '')
        except:
            pass
    
    if result['start_time'] and result['end_time']:
        result['total_duration_sec'] = (result['end_time'] - result['start_time']).total_seconds()
    
    # 구간 생성
    segments = []
    seg_defs = [
        ('entry', 'crane_start', '입구 대기', 'entry_wait'),
        ('crane_start', 'crane_end', '크레인 동작', 'crane'),
        ('transfer_start', 'transfer_end', '층간 이동', 'floor_move'),
        ('transfer_end', 'exit', '출구 대기', 'exit_wait'),
    ]
    
    for from_key, to_key, name, thresh_key in seg_defs:
        if times.get(from_key) and times.get(to_key):
            sec = (times[to_key] - times[from_key]).total_seconds()
            if sec < 0:
                continue
            status, emoji = get_status(sec, thresh_key)
            seg = {
                'name': name,
                'start_str': times[from_key].strftime('%H:%M:%S.%f')[:-3],
                'end_str': times[to_key].strftime('%H:%M:%S.%f')[:-3],
                'duration_sec': sec, 'duration_str': format_duration(sec),
                'status': status, 'emoji': emoji,
                'is_delay': status in ['경고', '지연']
            }
            segments.append(seg)
            if seg['is_delay']:
                causes = {'입구 대기': '내부 점유', '크레인 동작': '크레인 오류', '층간 이동': '속도 저하', '출구 대기': '포트 점유'}
                result['delays'].append({'segment': name, 'duration_str': format_duration(sec), 'cause': causes.get(name, '소요시간 초과')})
    
    result['segments'] = segments
    result['preprocessed_text'] = generate_prompt_text(result)
    return result


def generate_prompt_text(analysis: Dict) -> str:
    lines = ["=" * 60, "🔼 LIFTER 이송 분석 리포트", "=" * 60]
    
    fab = get_fab_info(analysis.get('machine_name', ''))
    lines.append(f"\n📍 캐리어: {analysis.get('carrier_id', 'N/A')}")
    lines.append(f"🏭 장비: {fab} Lifter ({analysis.get('machine_name', 'N/A')})")
    
    src = analysis.get('source_floor', 'N/A')
    dst = analysis.get('dest_floor', 'N/A')
    direction = analysis.get('direction', '')
    lines.append(f"📍 층간: {src} → {dst} {direction}")
    
    total = analysis.get('total_duration_sec', 0)
    lines.append(f"\n⏱️ 총 소요시간: {format_duration(total)} (정상: 2분 이내)")
    lines.append(f"📌 상태: {analysis.get('final_status', 'UNKNOWN')}")
    lines.append(f"{'🔴 지연 발생' if total > 120 else '✅ 정상 완료'}")
    
    if analysis.get('segments'):
        lines.append("\n" + "-" * 60)
        lines.append("### 🕒 구간별 소요시간")
        lines.append("\n| # | 구간 | 시작 | 종료 | 소요시간 | 상태 |")
        lines.append("|---|------|------|------|----------|------|")
        for i, s in enumerate(analysis['segments'], 1):
            m = "🔴 " if s.get('is_delay') else ""
            lines.append(f"| {i} | {m}{s['name']} | {s['start_str']} | {s['end_str']} | {s['duration_str']} | {s['emoji']} {s['status']} |")
    
    if analysis.get('delays'):
        lines.append("\n### 🔴 주요 지연")
        for d in analysis['delays']:
            lines.append(f"- {d['segment']}: {d['duration_str']} ({d['cause']})")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def is_lifter_data(df: pd.DataFrame) -> bool:
    if 'MESSAGENAME' not in df.columns:
        return False
    msgs = df['MESSAGENAME'].dropna().astype(str).tolist()
    return sum(1 for m in msgs if m.startswith('STORAGE-')) >= 5
