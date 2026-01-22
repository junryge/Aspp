#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONVEYOR 로그 전처리 모듈 V2.0 (R3 스타일)
- INTERRAIL-* 메시지 기반 분석
- 구간별 소요시간 정확히 계산
- STB 용어 통일
"""

import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

THRESHOLDS = {
    'entry': (10, 30, 60),           # 진입
    'command': (5, 10, 30),          # 명령 응답
    'transfer': (300, 480, 900),     # 컨베이어 이송: 5분/8분/15분
    'exit': (10, 30, 60),            # 퇴장
    'total': (300, 600, 1200),       # 전체: 5분/10분/20분
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


def get_fab_info(machine_name: str) -> str:
    if not machine_name:
        return ''
    return 'M14' if machine_name[0] == '4' else ('M16' if machine_name[0] == '6' else '')


def analyze_conveyor(df: pd.DataFrame) -> Dict:
    result = {
        'carrier_id': None, 'machine_name': None,
        'source_zone': None, 'dest_zone': None,
        'segments': [], 'delays': [],
        'total_duration_sec': 0, 'final_status': 'UNKNOWN',
        'start_time': None, 'end_time': None, 'preprocessed_text': ''
    }
    
    df = df.copy()
    df['parsed_time'] = df['TIME_EX'].apply(parse_time_ex)
    df = df.dropna(subset=['parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    
    if df.empty:
        result['preprocessed_text'] = "❌ 시간 파싱 실패"
        return result
    
    # 핵심 이벤트 시간
    times = {
        'entry': None, 'command': None, 'reply': None,
        'transfer_start': None, 'transfer_end': None, 'exit': None
    }
    zones = []
    
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
        
        # 이벤트 추출
        if 'CARRIERIDREAD' in msg or 'CARRIERINSTALLED' in msg:
            if not times['entry']:
                times['entry'] = t
                result['start_time'] = t
        
        if 'CARRIERTRANSFER' in msg and 'REPLY' not in msg:
            if not times['command']:
                times['command'] = t
        
        if 'CARRIERTRANSFERREPLY' in msg:
            times['reply'] = t
        
        if 'TRANSFERINITIATED' in msg:
            times['transfer_start'] = t
        
        if 'TRANSFERRING' in msg:
            zone = extract_xml_value(text, 'CARRIERZONENAME')
            if zone and (not zones or zones[-1] != zone):
                zones.append(zone)
        
        if 'TRANSFERCOMPLETED' in msg:
            times['transfer_end'] = t
            result['end_time'] = t
            result['final_status'] = 'COMPLETED'
        
        if 'CARRIERREMOVED' in msg:
            times['exit'] = t
            if not result['end_time']:
                result['end_time'] = t
    
    if zones:
        result['source_zone'] = zones[0]
        result['dest_zone'] = zones[-1]
    
    if result['start_time'] and result['end_time']:
        result['total_duration_sec'] = (result['end_time'] - result['start_time']).total_seconds()
    
    # 구간 생성
    segments = []
    seg_defs = [
        ('entry', 'command', '진입 → 명령', 'entry'),
        ('command', 'reply', '명령 → 응답', 'command'),
        ('transfer_start', 'transfer_end', '컨베이어 이송', 'transfer'),
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
                result['delays'].append({'segment': name, 'duration_str': format_duration(sec), 'cause': '컨베이어 적체 또는 센서 문제'})
    
    result['segments'] = segments
    result['preprocessed_text'] = generate_prompt_text(result)
    return result


def generate_prompt_text(analysis: Dict) -> str:
    lines = ["=" * 60, "➡️ CONVEYOR 이송 분석 리포트", "=" * 60]
    
    fab = get_fab_info(analysis.get('machine_name', ''))
    lines.append(f"\n📍 캐리어: {analysis.get('carrier_id', 'N/A')}")
    lines.append(f"🏭 장비: {fab}쪽 Conveyor ({analysis.get('machine_name', 'N/A')})")
    lines.append(f"📍 Zone: {analysis.get('source_zone', 'N/A')} → {analysis.get('dest_zone', 'N/A')}")
    
    total = analysis.get('total_duration_sec', 0)
    lines.append(f"\n⏱️ 총 소요시간: {format_duration(total)} (정상: 5분 이내)")
    lines.append(f"📌 상태: {analysis.get('final_status', 'UNKNOWN')}")
    lines.append(f"{'🔴 지연 발생' if total > 300 else '✅ 정상 완료'}")
    
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


def is_conveyor_data(df: pd.DataFrame) -> bool:
    if 'MESSAGENAME' not in df.columns:
        return False
    msgs = df['MESSAGENAME'].dropna().astype(str).tolist()
    return sum(1 for m in msgs if m.startswith('INTERRAIL-')) >= 5
