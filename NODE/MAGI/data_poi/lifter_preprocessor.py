#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIFTER 로그 전처리 모듈 V3.0 (R3 스타일)
- STORAGE-* 메시지 기반 분석
- 층간 이동 구간별 소요시간 정확히 계산
- CARRIERLOC 파싱으로 정확한 층 추출 (AI311→3F, AI623→6F)
- CARRIERLOCATIONCHANGED 이벤트로 중간 층까지 추적
- RM(내부 이동/크레인) 감지
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
    '2': ('거부', '리프터 반송 거절'),
    '4': ('시작', '실행 중'),
    '6': ('실패', '이송 불가'),
}

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


def get_floor_from_carrierloc(carrierloc: str) -> str:
    """CARRIERLOC에서 층 추출
    예: 6ABL6022_AI311 → 3F, 6ABL6022_OP623 → 6F, 6ABL6022_AI623_B1 → 6F
    패턴: _AI[층][포트][번호], _AO[층][번호], _OP[층][번호]
    """
    if not carrierloc:
        return ''
    # RM은 크레인 내부 → 층 모름
    if carrierloc.endswith('RM'):
        return 'RM'
    # _AI311, _AO623, _OP311 등에서 첫 숫자가 층
    match = re.search(r'_(?:AI|AO|OP)(\d)', carrierloc)
    if match:
        return f"{match.group(1)}F"
    # _AOG6 패턴 (G=Ground? 목적지)
    match = re.search(r'_AOG(\d)', carrierloc)
    if match:
        return f"{match.group(1)}F"
    return ''


def get_location_type(carrierloc: str) -> str:
    """CARRIERLOC에서 위치 타입 판별"""
    if not carrierloc:
        return ''
    if carrierloc.endswith('RM'):
        return '크레인(RM)'
    if '_AI' in carrierloc:
        if '_B' in carrierloc:
            return '입구 버퍼'
        return '입구(AI)'
    if '_AO' in carrierloc:
        return '출구(AO)'
    if '_OP' in carrierloc:
        return '조작위치(OP)'
    return ''


def get_fab_info(machine_name: str) -> str:
    if not machine_name:
        return ''
    return 'M14A' if machine_name[0] == '4' else ('M16A' if machine_name[0] == '6' else '')


def analyze_lifter(df: pd.DataFrame) -> Dict:
    result = {
        'carrier_id': None, 'machine_name': None, 'fab': '',
        'source_floor': None, 'dest_floor': None,
        'source_unit': None, 'dest_unit': None,
        'carrier_locations': [],
        'location_changes': [],
        'hcack_events': [],
        'segments': [], 'delays': [],
        'total_duration_sec': 0, 'final_status': 'UNKNOWN',
        'start_time': None, 'end_time': None,
        'direction': '', 'has_rm': False,
        'preprocessed_text': ''
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
    carrier_locs = []
    location_changes = []

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

        # CARRIERLOC 추적 (XML TEXT 내부)
        loc = extract_xml_value(text, 'CARRIERLOC')
        if loc and loc.strip():
            if not carrier_locs or carrier_locs[-1]['loc'] != loc:
                floor = get_floor_from_carrierloc(loc)
                loc_type = get_location_type(loc)
                carrier_locs.append({
                    'loc': loc, 'floor': floor, 'type': loc_type,
                    'time': t, 'time_str': t.strftime('%H:%M:%S.%f')[:-3]
                })
                if loc.endswith('RM'):
                    result['has_rm'] = True
                if floor and floor != 'RM' and (not floors or floors[-1] != floor):
                    floors.append(floor)

        # CARRIERLOCATIONCHANGED 이벤트 추적
        if 'LOCATIONCHANGED' in msg:
            loc_changed = extract_xml_value(text, 'CARRIERLOC')
            if loc_changed:
                floor_ch = get_floor_from_carrierloc(loc_changed)
                location_changes.append({
                    'loc': loc_changed, 'floor': floor_ch,
                    'time': t, 'time_str': t.strftime('%H:%M:%S.%f')[:-3],
                    'type': get_location_type(loc_changed)
                })

        # 층 추출 (UNITNAME/CURRENTUNITNAME fallback)
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

        if 'CARRIERTRANSFER' in msg and 'REPLY' not in msg and 'TRANSFERRING' not in msg and 'COMPLETED' not in msg and 'INITIATED' not in msg:
            # SOURCEUNIT/DESTUNIT 추출
            src = extract_xml_value(text, 'SOURCEUNIT')
            dst = extract_xml_value(text, 'DESTUNIT')
            if src and not result['source_unit']:
                result['source_unit'] = src
            if dst and not result['dest_unit']:
                result['dest_unit'] = dst

        if 'CARRIERTRANSFERREPLY' in msg:
            # HCACK 코드 분석
            hcack = extract_xml_value(text, 'HCACK')
            if not hcack:
                hcack_match = re.search(r"\[HCACK\]\s*'(\d+)'", text)
                if hcack_match:
                    hcack = hcack_match.group(1)
            if hcack:
                meaning = HCACK_MEANINGS.get(hcack, ('알수없음', ''))
                result['hcack_events'].append({
                    'time': t, 'time_str': t.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'hcack': hcack, 'status': meaning[0], 'desc': meaning[1]
                })

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

    result['carrier_locations'] = carrier_locs
    result['location_changes'] = location_changes

    # 층 결정 (CARRIERLOC 기반 우선, UNITNAME fallback)
    if floors:
        result['source_floor'] = floors[0]
        result['dest_floor'] = floors[-1]
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
                causes = {
                    '입구 대기': '내부 점유',
                    '크레인 동작': '크레인 오류',
                    '층간 이동': '속도 저하',
                    '출구 대기': '포트 점유'
                }
                result['delays'].append({
                    'segment': name,
                    'duration_str': format_duration(sec),
                    'cause': causes.get(name, '소요시간 초과')
                })

    result['segments'] = segments
    result['preprocessed_text'] = generate_prompt_text(result)
    return result


def analyze_lifter_multi(df: pd.DataFrame) -> Dict:
    """다중 캐리어 지원: CARRIER 컬럼 기준 그룹핑 후 각각 분석"""
    if 'CARRIER' not in df.columns:
        return analyze_lifter(df)

    carriers = df['CARRIER'].dropna().astype(str).str.strip().str.strip("'")
    unique_carriers = [c for c in carriers.unique() if c and c != 'nan']

    if len(unique_carriers) <= 1:
        return analyze_lifter(df)

    all_results = []
    combined_text_lines = []

    for carrier_id in unique_carriers:
        carrier_df = df[df['CARRIER'].astype(str).str.strip().str.strip("'") == carrier_id]
        if len(carrier_df) < 3:
            continue
        r = analyze_lifter(carrier_df)
        all_results.append(r)
        combined_text_lines.append(r.get('preprocessed_text', ''))

    if not all_results:
        return analyze_lifter(df)

    if len(all_results) == 1:
        return all_results[0]

    combined = {
        'carrier_id': ', '.join(r['carrier_id'] or 'N/A' for r in all_results),
        'machine_name': all_results[0].get('machine_name'),
        'fab': all_results[0].get('fab', ''),
        'source_floor': all_results[0].get('source_floor'),
        'dest_floor': all_results[-1].get('dest_floor'),
        'source_unit': all_results[0].get('source_unit'),
        'dest_unit': all_results[-1].get('dest_unit'),
        'carrier_locations': [],
        'location_changes': [],
        'hcack_events': [],
        'segments': [], 'delays': [],
        'total_duration_sec': sum(r['total_duration_sec'] for r in all_results),
        'final_status': 'COMPLETED' if all(r['final_status'] == 'COMPLETED' for r in all_results) else 'PARTIAL',
        'start_time': all_results[0].get('start_time'),
        'end_time': all_results[-1].get('end_time'),
        'direction': all_results[0].get('direction', ''),
        'has_rm': any(r.get('has_rm') for r in all_results),
        'multi_carrier': True,
        'carrier_results': all_results,
        'preprocessed_text': f"\n{'=' * 60}\n📦 다중 캐리어 LIFTER 분석 ({len(all_results)}건)\n{'=' * 60}\n\n" + "\n\n".join(combined_text_lines)
    }
    return combined


def generate_prompt_text(analysis: Dict) -> str:
    lines = ["=" * 60, "🔼 LIFTER 이송 분석 리포트", "=" * 60]

    fab = analysis.get('fab', '')
    lines.append(f"\n📍 캐리어: {analysis.get('carrier_id', 'N/A')}")
    if fab:
        lines.append(f"🏭 FAB: {fab} Lifter ({analysis.get('machine_name', 'N/A')})")
    else:
        lines.append(f"🏭 장비: {analysis.get('machine_name', 'N/A')}")

    src = analysis.get('source_floor', 'N/A')
    dst = analysis.get('dest_floor', 'N/A')
    direction = analysis.get('direction', '')
    lines.append(f"📍 층간: {src} → {dst} {direction}")

    # 포트 정보
    src_unit = analysis.get('source_unit')
    dst_unit = analysis.get('dest_unit')
    if src_unit or dst_unit:
        lines.append(f"📍 포트: {src_unit or 'N/A'} → {dst_unit or 'N/A'}")

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

    # CARRIERLOC 이동 경로 (상세)
    if analysis.get('carrier_locations'):
        lines.append("\n### 📍 캐리어 위치 이동 경로")
        for cl in analysis['carrier_locations']:
            floor_str = f" ({cl['floor']})" if cl.get('floor') else ""
            type_str = f" [{cl['type']}]" if cl.get('type') else ""
            lines.append(f"  [{cl['time_str']}] {cl['loc']}{floor_str}{type_str}")

    # RM 감지
    if analysis.get('has_rm'):
        lines.append("\n### 🏗️ 크레인 내부이동(RM) 감지됨")

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
        lines.append("→ 리프터 이송 거절 또는 크레인 점유")

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
