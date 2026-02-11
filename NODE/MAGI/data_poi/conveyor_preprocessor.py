#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONVEYOR 로그 전처리 모듈 V3.0 (R3 스타일)
- INTERRAIL-* 메시지 기반 분석
- 구간별 소요시간 정확히 계산
- SOURCEUNIT/DESTUNIT 추출
- CARRIERLOC 기반 경로 추적
- 퇴출(exit) 구간 추가
- HCACK 코드 분석
- 다중 캐리어 지원
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
    '2': ('거부', '컨베이어 반송 거절'),
    '4': ('시작', '실행 중'),
    '6': ('실패', '이송 불가'),
}

THRESHOLDS = {
    'entry': (10, 30, 60),           # 진입
    'command': (5, 10, 30),          # 명령 응답
    'transfer': (300, 480, 900),     # 컨베이어 이송: 5분/8분/15분
    'exit': (10, 30, 60),            # 퇴출
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
        'carrier_id': None, 'machine_name': None, 'fab': '',
        'source_unit': None, 'dest_unit': None,
        'source_zone': None, 'dest_zone': None,
        'carrier_locations': [],
        'hcack_events': [],
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
    carrier_locs = []

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
                result['fab'] = get_fab_info(str(m))

        # CARRIERLOC 추적
        loc = extract_xml_value(text, 'CARRIERLOC')
        if loc and loc.strip() and (not carrier_locs or carrier_locs[-1]['loc'] != loc):
            carrier_locs.append({'loc': loc, 'time': t, 'time_str': t.strftime('%H:%M:%S.%f')[:-3]})

        # CARRIERZONENAME 추적
        zone = extract_xml_value(text, 'CARRIERZONENAME')
        if zone and zone.strip() and (not zones or zones[-1] != zone):
            zones.append(zone)

        # 이벤트 추출
        if 'CARRIERIDREAD' in msg or 'CARRIERINSTALLED' in msg:
            if not times['entry']:
                times['entry'] = t
                result['start_time'] = t

        if 'CARRIERTRANSFER' in msg and 'REPLY' not in msg and 'TRANSFERRING' not in msg:
            if not times['command']:
                times['command'] = t
                # SOURCEUNIT/DESTUNIT 추출
                src = extract_xml_value(text, 'SOURCEUNIT')
                dst = extract_xml_value(text, 'DESTUNIT')
                if src:
                    result['source_unit'] = src
                if dst:
                    result['dest_unit'] = dst

        if 'CARRIERTRANSFERREPLY' in msg:
            times['reply'] = t
            # HCACK 코드 분석
            hcack = extract_xml_value(text, 'HCACK')
            if not hcack:
                # SECSII에서 HCACK 추출 시도
                hcack_match = re.search(r"\[HCACK\]\s*'(\d+)'", text)
                if hcack_match:
                    hcack = hcack_match.group(1)
            if hcack:
                meaning = HCACK_MEANINGS.get(hcack, ('알수없음', ''))
                result['hcack_events'].append({
                    'time': t, 'time_str': t.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'hcack': hcack, 'status': meaning[0], 'desc': meaning[1]
                })
            # REPLY에서도 SOURCEUNIT/DESTUNIT fallback
            if not result['source_unit']:
                src = extract_xml_value(text, 'SOURCEUNIT')
                if src:
                    result['source_unit'] = src
            if not result['dest_unit']:
                dst = extract_xml_value(text, 'DESTUNIT')
                if dst:
                    result['dest_unit'] = dst

        if 'TRANSFERINITIATED' in msg:
            times['transfer_start'] = t

        if 'TRANSFERRING' in msg:
            pass  # zone은 위에서 이미 추적

        if 'TRANSFERCOMPLETED' in msg:
            times['transfer_end'] = t
            result['final_status'] = 'COMPLETED'
            # TRANSFERCOMPLETED에서 SOURCEPORT/DESTPORT 추출
            final_src = extract_xml_value(text, 'SOURCEPORT')
            final_dst = extract_xml_value(text, 'DESTPORT')
            if final_src and not result['source_unit']:
                result['source_unit'] = final_src
            if final_dst and not result['dest_unit']:
                result['dest_unit'] = final_dst

        if 'CARRIERREMOVED' in msg:
            times['exit'] = t
            if not result['end_time']:
                result['end_time'] = t

    # end_time 결정
    if not result['end_time']:
        if times['exit']:
            result['end_time'] = times['exit']
        elif times['transfer_end']:
            result['end_time'] = times['transfer_end']

    result['carrier_locations'] = carrier_locs

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
        ('transfer_end', 'exit', '퇴출 대기', 'exit'),
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
                cause = 'HCACK=2 거절' if any(h['hcack'] == '2' for h in result['hcack_events']) else '컨베이어 적체 또는 센서 문제'
                result['delays'].append({'segment': name, 'duration_str': format_duration(sec), 'cause': cause})

    result['segments'] = segments
    result['preprocessed_text'] = generate_prompt_text(result)
    return result


def analyze_conveyor_multi(df: pd.DataFrame) -> Dict:
    """다중 캐리어 지원: CARRIER 컬럼 기준 그룹핑 후 각각 분석"""
    if 'CARRIER' not in df.columns:
        return analyze_conveyor(df)

    carriers = df['CARRIER'].dropna().astype(str).str.strip().str.strip("'")
    unique_carriers = [c for c in carriers.unique() if c and c != 'nan']

    if len(unique_carriers) <= 1:
        return analyze_conveyor(df)

    all_results = []
    combined_text_lines = []

    for carrier_id in unique_carriers:
        carrier_df = df[df['CARRIER'].astype(str).str.strip().str.strip("'") == carrier_id]
        if len(carrier_df) < 3:
            continue
        r = analyze_conveyor(carrier_df)
        all_results.append(r)
        combined_text_lines.append(r.get('preprocessed_text', ''))

    if not all_results:
        return analyze_conveyor(df)

    if len(all_results) == 1:
        return all_results[0]

    combined = {
        'carrier_id': ', '.join(r['carrier_id'] or 'N/A' for r in all_results),
        'machine_name': all_results[0].get('machine_name'),
        'fab': all_results[0].get('fab', ''),
        'source_unit': all_results[0].get('source_unit'),
        'dest_unit': all_results[-1].get('dest_unit'),
        'source_zone': all_results[0].get('source_zone'),
        'dest_zone': all_results[-1].get('dest_zone'),
        'carrier_locations': [],
        'hcack_events': [],
        'segments': [], 'delays': [],
        'total_duration_sec': sum(r['total_duration_sec'] for r in all_results),
        'final_status': 'COMPLETED' if all(r['final_status'] == 'COMPLETED' for r in all_results) else 'PARTIAL',
        'start_time': all_results[0].get('start_time'),
        'end_time': all_results[-1].get('end_time'),
        'multi_carrier': True,
        'carrier_results': all_results,
        'preprocessed_text': f"\n{'=' * 60}\n📦 다중 캐리어 CONVEYOR 분석 ({len(all_results)}건)\n{'=' * 60}\n\n" + "\n\n".join(combined_text_lines)
    }
    return combined


def generate_prompt_text(analysis: Dict) -> str:
    lines = ["=" * 60, "➡️ CONVEYOR 이송 분석 리포트", "=" * 60]

    fab = analysis.get('fab', '')
    lines.append(f"\n📍 캐리어: {analysis.get('carrier_id', 'N/A')}")
    if fab:
        lines.append(f"🏭 FAB: {fab} Conveyor ({analysis.get('machine_name', 'N/A')})")
    else:
        lines.append(f"🏭 장비: {analysis.get('machine_name', 'N/A')}")

    # 포트 정보
    src_unit = analysis.get('source_unit', 'N/A')
    dst_unit = analysis.get('dest_unit', 'N/A')
    lines.append(f"📍 포트: {src_unit} → {dst_unit}")

    # Zone 정보
    src_zone = analysis.get('source_zone')
    dst_zone = analysis.get('dest_zone')
    if src_zone or dst_zone:
        lines.append(f"📍 Zone: {src_zone or 'N/A'} → {dst_zone or 'N/A'}")

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

    # CARRIERLOC 이동 경로
    if analysis.get('carrier_locations'):
        lines.append("\n### 📍 캐리어 위치 이동 경로")
        loc_strs = [cl['loc'] for cl in analysis['carrier_locations']]
        lines.append(f"경로: {' → '.join(loc_strs)}")

    # HCACK 이벤트
    all_hcack = analysis.get('hcack_events', [])
    if all_hcack:
        lines.append(f"\n### 📋 HCACK 응답 이력")
        for h in all_hcack:
            status_str = h.get('status', '')
            desc_str = h.get('desc', '')
            lines.append(f"- [{h['time_str']}] HCACK={h['hcack']} ({status_str}: {desc_str})")

    rej = [h for h in all_hcack if h['hcack'] == '2']
    if rej:
        lines.append(f"\n### ⚠️ HCACK=2 거절 {len(rej)}회")
        lines.append("→ 컨베이어 이송 거절 또는 경로 차단")

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
