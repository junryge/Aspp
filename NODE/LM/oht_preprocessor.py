#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OHT 로그 전처리 모듈 V2.0 (R3 스타일)
- RAIL-* 메시지 기반 분석
- 구간별 소요시간 정확히 계산
- HCACK=2 지연 원인
- STB 용어 통일
"""

import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

HCACK_MEANINGS = {
    '0': ('성공', '명령 수락'),
    '2': ('거부', 'OHT 반송 거절'),
    '4': ('시작', '실행 중'),
    '6': ('실패', '이송 불가'),
}

THRESHOLDS = {
    'command_to_assign': (10, 30, 60),
    'assign_to_pickup': (30, 60, 120),
    'pickup': (10, 20, 40),
    'transfer': (60, 120, 240),
    'deposit': (10, 20, 40),
    'total': (120, 180, 300),
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


def analyze_oht(df: pd.DataFrame) -> Dict:
    result = {
        'carrier_id': None, 'vehicle_id': None,
        'source_port': None, 'dest_port': None,
        'hcack_events': [], 'segments': [], 'delays': [],
        'total_duration_sec': 0, 'final_status': 'UNKNOWN',
        'start_time': None, 'end_time': None, 'preprocessed_text': ''
    }
    
    df = df.copy()
    df['parsed_time'] = df['TIME_EX'].apply(parse_time_ex)
    df = df.dropna(subset=['parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    
    if df.empty:
        result['preprocessed_text'] = "❌ 시간 파싱 실패"
        return result
    
    # 핵심 이벤트 시간 추출
    times = {
        'command': None, 'assigned': None, 'pickup_start': None,
        'pickup_end': None, 'deposit_start': None, 'deposit_end': None, 'complete': None
    }
    
    for _, row in df.iterrows():
        msg = str(row.get('MESSAGENAME', ''))
        text = str(row.get('TEXT', ''))
        t = row['parsed_time']
        
        if not result['carrier_id']:
            c = row.get('CARRIER')
            if pd.notna(c):
                result['carrier_id'] = str(c).strip().strip("'")
        
        if 'CARRIERTRANSFER' in msg and 'REPLY' not in msg:
            if not times['command']:
                times['command'] = t
                result['source_port'] = extract_xml_value(text, 'SOURCEPORT')
                result['dest_port'] = extract_xml_value(text, 'DESTPORT')
                result['start_time'] = t
        
        if 'CARRIERTRANSFERREPLY' in msg:
            hcack = extract_xml_value(text, 'HCACK')
            if hcack:
                result['hcack_events'].append({
                    'time': t, 'time_str': t.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'hcack': hcack
                })
        
        if 'VEHICLEASSIGNED' in msg and not times['assigned']:
            times['assigned'] = t
            result['vehicle_id'] = extract_xml_value(text, 'VEHICLEID')
        
        if 'ACQUIRESTARTED' in msg and not times['pickup_start']:
            times['pickup_start'] = t
        if 'ACQUIRECOMPLETED' in msg or 'CARRIERINSTALLED' in msg:
            times['pickup_end'] = t
        if 'DEPOSITSTARTED' in msg and not times['deposit_start']:
            times['deposit_start'] = t
        if 'DEPOSITCOMPLETED' in msg or 'CARRIERREMOVED' in msg:
            times['deposit_end'] = t
        if 'TRANSFERCOMPLETED' in msg:
            times['complete'] = t
            result['end_time'] = t
            result['final_status'] = 'COMPLETED'
    
    if result['start_time'] and result['end_time']:
        result['total_duration_sec'] = (result['end_time'] - result['start_time']).total_seconds()
    
    # 구간 생성
    segments = []
    seg_defs = [
        ('command', 'assigned', '명령 → 차량할당', 'command_to_assign'),
        ('assigned', 'pickup_start', '차량할당 → 픽업시작', 'assign_to_pickup'),
        ('pickup_start', 'pickup_end', '픽업', 'pickup'),
        ('pickup_end', 'deposit_start', '이송', 'transfer'),
        ('deposit_start', 'deposit_end', '하역', 'deposit'),
    ]
    
    for from_key, to_key, name, thresh_key in seg_defs:
        if times.get(from_key) and times.get(to_key):
            sec = (times[to_key] - times[from_key]).total_seconds()
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
                cause = 'HCACK=2 거절' if any(h['hcack']=='2' for h in result['hcack_events']) else '소요시간 초과'
                result['delays'].append({'segment': name, 'duration_str': format_duration(sec), 'cause': cause})
    
    result['segments'] = segments
    result['preprocessed_text'] = generate_prompt_text(result)
    return result


def generate_prompt_text(analysis: Dict) -> str:
    lines = ["=" * 60, "🚃 OHT 이송 분석 리포트", "=" * 60]
    
    lines.append(f"\n📍 캐리어: {analysis.get('carrier_id', 'N/A')}")
    lines.append(f"🚃 차량: {analysis.get('vehicle_id', 'N/A')}")
    lines.append(f"📍 경로: {analysis.get('source_port', 'N/A')} → {analysis.get('dest_port', 'N/A')}")
    
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
    
    rej = [h for h in analysis.get('hcack_events', []) if h['hcack'] == '2']
    if rej:
        lines.append(f"\n### ⚠️ HCACK=2 거절 {len(rej)}회")
        lines.append("→ OHT Vehicle 할당 문제 또는 Rail Cut")
    
    if analysis.get('delays'):
        lines.append("\n### 🔴 주요 지연")
        for d in analysis['delays']:
            lines.append(f"- {d['segment']}: {d['duration_str']} ({d['cause']})")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def is_oht_data(df: pd.DataFrame) -> bool:
    if 'MESSAGENAME' not in df.columns:
        return False
    msgs = df['MESSAGENAME'].dropna().astype(str).tolist()
    return sum(1 for m in msgs if m.startswith('RAIL-') and 'INTERRAIL' not in m) >= 5
