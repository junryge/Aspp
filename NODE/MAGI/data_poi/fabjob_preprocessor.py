#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FABJOB 로그 전처리 모듈 V3.0 (R3 피드백 반영)

핵심 수정:
1. STB로 통일 (Stocker 사용 안함)
2. VM-TRANSPORTJOBLOCATIONCHANGED 시간 기준 정확한 구간 계산
3. JOB 시작 → 첫 OHT = STB→OHT 구간 (32분 지연 정확히 캐치)
4. HCACK=2 거절 → OHT 반송 거절 원인

R3 피드백 정답:
1. M14A STB → M14A OHT: 11:12:03 → 11:44:36 = 32분 (지연!)
2. M14A OHT → M14 Conveyor: 11:44:36 → 11:46:27 = 2분 (정상)
3. M14 Conveyor → M16 Conveyor: 11:46:27 → 11:51:31 = 6분 (정상)
4. M16 Conveyor → M16 Bridge OHT: 11:51:31 → 11:52:21 = 1분 (정상)
5. M16 Bridge OHT → M16 Lifter 3F: 11:52:21 → 11:56:04 = 4분 (정상)
6. M16 Lifter 3F → M16 Lifter 6F: 11:56:04 → 11:59:45 = 3분 (정상)
7. M16 Lifter 6F → M16A OHT: 11:59:45 → 12:00:25 = 1분 (정상)
8. M16A OHT → M16 STB: 12:00:25 → 12:01:22 = 2분 (정상)
"""

import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 장비 유형 분류 (STB 통일)
# ============================================================================
EQUIPMENT_TYPES = {
    'STB': {
        'patterns': ['ANZ'],  # 4ANZ40G1, 6ANZ0202
        'description': 'STB (캐리어 보관)'
    },
    'OHT': {
        'patterns': ['ACM', 'ECM'],  # 4ACM4701, 6ACM3901, 6ECMB101
        'description': 'OHT (천장 이송)'
    },
    'Conveyor': {
        'patterns': ['AFC'],  # 4AFC3301
        'description': 'Conveyor (바닥 이송)'
    },
    'Lifter': {
        'patterns': ['ABL'],  # 6ABL0121
        'description': 'Lifter (층간 이송)'
    }
}


def get_equipment_type(machine_name: str) -> str:
    """장비명에서 유형 추출"""
    if not machine_name:
        return 'Unknown'
    
    machine_upper = machine_name.upper()
    
    for eq_type, info in EQUIPMENT_TYPES.items():
        for pattern in info['patterns']:
            if pattern in machine_upper:
                return eq_type
    
    return 'Unknown'


def get_fab_info(machine_name: str) -> str:
    """장비명에서 FAB 추출 (4xxx=M14A, 6xxx=M16A)"""
    if not machine_name or len(machine_name) < 1:
        return ''
    
    first_char = machine_name[0]
    if first_char == '4':
        return 'M14A'
    elif first_char == '6':
        return 'M16A'
    return ''


def get_floor_from_unit(unit_name: str) -> str:
    """유닛명에서 층 추출 (예: 6ABL0121_AI323 → 3F)"""
    if not unit_name:
        return ''
    
    # Lifter: _AI323 → 3F, _AO621 → 6F
    floor_match = re.search(r'_A[IO](\d)', unit_name)
    if floor_match:
        return f"{floor_match.group(1)}F"
    return ''


def format_location(machine_name: str, unit_name: str = '') -> str:
    """장비명을 읽기 좋은 형식으로 변환"""
    if not machine_name:
        return 'Unknown'
    
    eq_type = get_equipment_type(machine_name)
    fab = get_fab_info(machine_name)
    floor = get_floor_from_unit(unit_name)
    
    if eq_type == 'STB':
        return f"{fab} STB({machine_name})"
    
    elif eq_type == 'OHT':
        # Bridge OHT 구분 (ECM 또는 BV 포함)
        if 'ECM' in machine_name.upper() or (unit_name and 'BV' in unit_name):
            return f"{fab} Bridge OHT({machine_name})"
        floor_str = f" {floor}" if floor else ""
        return f"{fab}{floor_str} OHT({machine_name})"
    
    elif eq_type == 'Conveyor':
        fab_short = 'M14' if fab == 'M14A' else ('M16' if fab == 'M16A' else fab)
        return f"{fab_short}쪽 Conveyor({machine_name})"
    
    elif eq_type == 'Lifter':
        floor_str = f" {floor}" if floor else ""
        return f"{fab} Lifter{floor_str}({machine_name})"
    
    return machine_name


def format_location_short(machine_name: str, unit_name: str = '') -> str:
    """간단한 위치 (구간 표시용)"""
    if not machine_name:
        return 'Unknown'
    
    eq_type = get_equipment_type(machine_name)
    fab = get_fab_info(machine_name)
    floor = get_floor_from_unit(unit_name)
    
    if eq_type == 'STB':
        return f"{fab} STB"
    elif eq_type == 'OHT':
        if 'ECM' in machine_name.upper():
            return f"{fab} Bridge OHT"
        floor_str = f" {floor}" if floor else ""
        return f"{fab}{floor_str} OHT"
    elif eq_type == 'Conveyor':
        fab_short = 'M14' if fab == 'M14A' else ('M16' if fab == 'M16A' else fab)
        return f"{fab_short}쪽 Conveyor"
    elif eq_type == 'Lifter':
        floor_str = f" {floor}" if floor else ""
        return f"{fab} Lifter{floor_str}"
    
    return machine_name


# ============================================================================
# 시간/XML 파싱
# ============================================================================
def parse_time_ex(time_str: str) -> Optional[datetime]:
    """TIME_EX에서 시간 추출: [2026-01-21 11:12:03.885]"""
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
    """XML 태그에서 값 추출"""
    if pd.isna(text):
        return None
    
    pattern = rf'<\s*{tag}\s*>\s*([^<]*?)\s*<\s*/\s*{tag}\s*>'
    match = re.search(pattern, str(text), re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        return val if val else None
    return None


def format_duration(seconds: float) -> str:
    """초 → 읽기 좋은 형식"""
    if seconds < 0:
        return "N/A"
    
    if seconds < 60:
        return f"{seconds:.0f}초"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}분 {secs}초"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}시간 {mins}분"


def get_duration_status(seconds: float, segment_type: str) -> tuple:
    """구간별 소요시간 상태 → (상태, 이모지)"""
    thresholds = {
        'stb_to_oht': (180, 300, 600),       # STB→OHT: 3분/5분/10분 (R3: 32분 = 지연)
        'oht_to_conveyor': (180, 300, 480),  # OHT→Conveyor: 3분/5분/8분
        'conveyor': (480, 600, 900),         # Conveyor: 8분/10분/15분
        'oht_to_lifter': (300, 420, 600),    # OHT→Lifter: 5분/7분/10분
        'lifter': (240, 360, 480),           # Lifter 층간: 4분/6분/8분
        'oht_to_stb': (180, 300, 480),       # OHT→STB: 3분/5분/8분
        'default': (180, 300, 600),
        'total': (1200, 1800, 2700),         # 전체: 20분/30분/45분
    }
    
    normal, caution, critical = thresholds.get(segment_type, thresholds['default'])
    
    if seconds <= normal:
        return ("정상", "✅")
    elif seconds <= caution:
        return ("주의", "🟡")
    elif seconds <= critical:
        return ("경고", "⚠️")
    else:
        return ("지연", "🔴")


def get_segment_type(from_eq: str, to_eq: str) -> str:
    """두 장비 유형 → 구간 타입"""
    key = f"{from_eq}_{to_eq}"
    mapping = {
        'STB_OHT': 'stb_to_oht',
        'OHT_Conveyor': 'oht_to_conveyor',
        'Conveyor_Conveyor': 'conveyor',
        'Conveyor_OHT': 'oht_to_conveyor',
        'OHT_Lifter': 'oht_to_lifter',
        'Lifter_Lifter': 'lifter',
        'Lifter_OHT': 'oht_to_lifter',
        'OHT_STB': 'oht_to_stb',
        'OHT_OHT': 'default',
    }
    return mapping.get(key, 'default')


# ============================================================================
# HCACK 분석
# ============================================================================
HCACK_MEANINGS = {
    '0': ('성공', '명령 수락'),
    '2': ('거절', 'OHT 반송 거절 - Vehicle 할당 실패'),
    '4': ('시작', '명령 수락 후 실행'),
    '6': ('실패', '이송 불가'),
}


def analyze_hcack(hcack_events: List[dict]) -> dict:
    """HCACK 이벤트 분석"""
    result = {
        'rejections': [],
        'first_reject_time': None,
        'first_success_time': None,
        'rejection_count': 0,
        'delay_seconds': 0,
    }
    
    for h in hcack_events:
        if h['hcack'] == '2':
            result['rejections'].append(h)
            if not result['first_reject_time']:
                result['first_reject_time'] = h['time']
        elif h['hcack'] == '4':
            if not result['first_success_time']:
                result['first_success_time'] = h['time']
    
    result['rejection_count'] = len(result['rejections'])
    
    if result['first_reject_time'] and result['first_success_time']:
        result['delay_seconds'] = (result['first_success_time'] - result['first_reject_time']).total_seconds()
    
    return result


# ============================================================================
# 메인 분석 함수
# ============================================================================
def analyze_fabjob(df: pd.DataFrame) -> Dict:
    """
    FABJOB 로그 분석 (R3 피드백 반영)
    
    핵심:
    1. JOB 시작 시간 → 첫 OHT LOCATIONCHANGED = STB→OHT 구간
    2. LOCATIONCHANGED 간 시간 차이 = 각 구간 소요시간
    3. HCACK=2 거절 → 지연 원인
    """
    result = {
        'carrier_id': None,
        'lot_id': None,
        'source': {},
        'destination': {},
        'location_changes': [],
        'hcack_events': [],
        'segments': [],
        'delays': [],
        'total_duration_sec': 0,
        'final_status': 'UNKNOWN',
        'start_time': None,
        'end_time': None,
        'preprocessed_text': ''
    }
    
    # 1. 시간 파싱 및 정렬
    df = df.copy()
    df['parsed_time'] = df['TIME_EX'].apply(parse_time_ex)
    df = df.dropna(subset=['parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    
    if df.empty:
        result['preprocessed_text'] = "❌ 시간 정보를 파싱할 수 없습니다."
        return result
    
    # 2. 기본 정보 추출
    for _, row in df.iterrows():
        text = str(row.get('TEXT', ''))
        msg = str(row.get('MESSAGENAME', ''))
        
        # 캐리어 ID
        if not result['carrier_id']:
            carrier = row.get('CARRIER')
            if pd.notna(carrier) and carrier:
                result['carrier_id'] = str(carrier).strip().strip("'")
        
        # JOB 생성
        if 'FABTRANSPORTJOBCREATED' in msg:
            if not result['start_time']:
                result['start_time'] = row['parsed_time']
            if not result['lot_id']:
                result['lot_id'] = extract_xml_value(text, 'LOTID')
            if not result['source'].get('machine'):
                result['source'] = {
                    'fab': extract_xml_value(text, 'SOURCEFABNAME'),
                    'floor': extract_xml_value(text, 'SOURCEFLOORNAME'),
                    'machine': extract_xml_value(text, 'SOURCEMACHINENAME'),
                }
            if not result['destination'].get('machine'):
                result['destination'] = {
                    'fab': extract_xml_value(text, 'DESTFABNAME'),
                    'floor': extract_xml_value(text, 'DESTFLOORNAME'),
                    'machine': extract_xml_value(text, 'DESTMACHINENAME'),
                }
        
        # JOB 완료
        if 'TRANSPORTJOBCOMPLETED' in msg:
            result['end_time'] = row['parsed_time']
            result['final_status'] = extract_xml_value(text, 'STATE') or 'COMPLETED'
    
    # 3. HCACK 이벤트 추출 (RAIL-CARRIERTRANSFERREPLY)
    for _, row in df.iterrows():
        msg = str(row.get('MESSAGENAME', ''))
        if 'CARRIERTRANSFERREPLY' in msg:
            text = str(row.get('TEXT', ''))
            hcack = extract_xml_value(text, 'HCACK')
            if hcack:
                result['hcack_events'].append({
                    'time': row['parsed_time'],
                    'time_str': row['parsed_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'hcack': hcack,
                    'machine': row.get('MACHINENAME', ''),
                })
    
    # 4. VM-TRANSPORTJOBLOCATIONCHANGED 추출 (핵심!)
    location_changes = []
    for _, row in df.iterrows():
        msg = str(row.get('MESSAGENAME', ''))
        if 'LOCATIONCHANGED' in msg:
            text = str(row.get('TEXT', ''))
            machine = extract_xml_value(text, 'CURRENTMACHINENAME')
            unit = extract_xml_value(text, 'CURRENTUNITNAME')
            
            if machine:
                loc = {
                    'time': row['parsed_time'],
                    'time_str': row['parsed_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'machine': machine,
                    'unit': unit or '',
                    'location_str': format_location(machine, unit),
                    'location_short': format_location_short(machine, unit),
                    'eq_type': get_equipment_type(machine),
                }
                
                # 중복 제거: 같은 machine+unit 연속이면 스킵
                if location_changes:
                    last = location_changes[-1]
                    # 완전 동일하면 스킵
                    if last['machine'] == machine and last['unit'] == unit:
                        continue
                
                location_changes.append(loc)
    
    result['location_changes'] = location_changes
    
    # 5. 전체 시간
    if result['start_time'] and result['end_time']:
        result['total_duration_sec'] = (result['end_time'] - result['start_time']).total_seconds()
    
    # 6. 구간별 분석 (R3 피드백 핵심!)
    segments = []
    
    # 6-1. 첫 구간: JOB 시작 → 첫 위치변경 = STB → OHT (32분 지연 구간!)
    if result['start_time'] and location_changes:
        first_loc = location_changes[0]
        first_seg_sec = (first_loc['time'] - result['start_time']).total_seconds()
        
        # 출발지 = source machine (STB)
        src_machine = result['source'].get('machine', '')
        src_short = format_location_short(src_machine)
        
        seg_type = get_segment_type('STB', first_loc['eq_type'])
        status, emoji = get_duration_status(first_seg_sec, seg_type)
        
        segment = {
            'name': f"{src_short} → {first_loc['location_short']}",
            'start_time': result['start_time'],
            'start_str': result['start_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'end_time': first_loc['time'],
            'end_str': first_loc['time_str'],
            'duration_sec': first_seg_sec,
            'duration_str': format_duration(first_seg_sec),
            'status': status,
            'emoji': emoji,
            'is_delay': status in ['경고', '지연'],
            'from_eq': 'STB',
            'to_eq': first_loc['eq_type'],
        }
        
        # HCACK=2 지연 원인
        hcack_analysis = analyze_hcack(result['hcack_events'])
        if hcack_analysis['rejection_count'] > 0:
            segment['delay_cause'] = f"HCACK=2 (OHT 반송 거절) {hcack_analysis['rejection_count']}회"
            if first_seg_sec > 300:  # 5분 이상
                segment['is_delay'] = True
                result['delays'].append({
                    'segment': segment['name'],
                    'duration_sec': first_seg_sec,
                    'duration_str': format_duration(first_seg_sec),
                    'cause': f"HCACK=2 (OHT 반송 거절) {hcack_analysis['rejection_count']}회 - OHT Vehicle 할당 문제 또는 Rail Cut",
                    'hcack_events': hcack_analysis['rejections'],
                })
        
        segments.append(segment)
    
    # 6-2. 나머지 구간: 위치변경 간 시간 차이
    for i in range(len(location_changes) - 1):
        curr = location_changes[i]
        next_loc = location_changes[i + 1]
        
        seg_sec = (next_loc['time'] - curr['time']).total_seconds()
        seg_type = get_segment_type(curr['eq_type'], next_loc['eq_type'])
        status, emoji = get_duration_status(seg_sec, seg_type)
        
        segment = {
            'name': f"{curr['location_short']} → {next_loc['location_short']}",
            'start_time': curr['time'],
            'start_str': curr['time_str'],
            'end_time': next_loc['time'],
            'end_str': next_loc['time_str'],
            'duration_sec': seg_sec,
            'duration_str': format_duration(seg_sec),
            'status': status,
            'emoji': emoji,
            'is_delay': status in ['경고', '지연'],
            'from_eq': curr['eq_type'],
            'to_eq': next_loc['eq_type'],
        }
        segments.append(segment)
    
    result['segments'] = segments
    
    # 7. 텍스트 생성
    result['preprocessed_text'] = generate_prompt_text(result)
    
    return result


# ============================================================================
# 텍스트 생성
# ============================================================================
def generate_prompt_text(analysis: Dict) -> str:
    """분석 결과 → LLM 프롬프트용 텍스트"""
    
    lines = []
    
    # 헤더
    lines.append("=" * 70)
    lines.append("📦 FABJOB 이송 분석 리포트")
    lines.append("=" * 70)
    
    # 기본 정보
    carrier = analysis.get('carrier_id', 'N/A')
    src = analysis.get('source', {})
    location_changes = analysis.get('location_changes', [])
    
    src_str = format_location_short(src.get('machine', ''))
    if location_changes:
        dst_str = location_changes[-1]['location_short']
    else:
        dst = analysis.get('destination', {})
        dst_str = format_location_short(dst.get('machine', ''))
    
    lines.append(f"\n📍 캐리어: {carrier}")
    lines.append(f"📍 전체 경로: {src_str} → {dst_str}")
    
    # 시간 정보
    total_sec = analysis.get('total_duration_sec', 0)
    total_str = format_duration(total_sec)
    status, emoji = get_duration_status(total_sec, 'total')
    
    lines.append(f"\n⏱️ 총 소요시간: {total_str} (정상: 20분 이내)")
    lines.append(f"📌 최종 상태: {analysis.get('final_status', 'UNKNOWN')}")
    
    if total_sec > 1200:
        lines.append(f"🔴 결과: **지연 발생**")
    else:
        lines.append(f"✅ 결과: **정상 완료**")
    
    # 구간별 분석 (핵심!)
    segments = analysis.get('segments', [])
    if segments:
        lines.append("\n" + "-" * 70)
        lines.append("### 🕒 구간별 소요시간 분석")
        lines.append("")
        lines.append("| # | 구간 | 시작 시간 | 종료 시간 | 소요시간 | 상태 |")
        lines.append("|---|------|-----------|-----------|----------|------|")
        
        for i, seg in enumerate(segments, 1):
            name = seg['name']
            if len(name) > 35:
                name = name[:32] + "..."
            
            start_t = seg['start_str'].split()[1] if ' ' in seg['start_str'] else seg['start_str']
            end_t = seg['end_str'].split()[1] if ' ' in seg['end_str'] else seg['end_str']
            
            delay_mark = "🔴 " if seg.get('is_delay') else ""
            lines.append(f"| {i} | {delay_mark}{name} | {start_t} | {end_t} | {seg['duration_str']} | {seg['emoji']} {seg['status']} |")
    
    # 주요 지연 구간
    delays = analysis.get('delays', [])
    if delays:
        lines.append("\n" + "-" * 70)
        lines.append("### ⚠️ 주요 문제점")
        for d in delays:
            lines.append(f"\n**지연 구간**: {d['segment']}")
            lines.append(f"**소요시간**: {d['duration_str']}")
            lines.append(f"**원인**: {d['cause']}")
    
    # HCACK 분석
    hcack_events = analysis.get('hcack_events', [])
    rejections = [h for h in hcack_events if h['hcack'] == '2']
    
    if rejections:
        lines.append("\n" + "-" * 70)
        lines.append("### 🔍 지연 원인 분석")
        lines.append("")
        lines.append(f"**HCACK=2 (OHT 반송 거절) {len(rejections)}회 발생**")
        lines.append("")
        lines.append("| 시간 | HCACK | 의미 |")
        lines.append("|------|-------|------|")
        
        for h in rejections[:5]:  # 최대 5개
            t = h['time_str'].split()[1] if ' ' in h['time_str'] else h['time_str']
            lines.append(f"| {t} | 2 | ❌ OHT 반송 거절 |")
        
        lines.append("")
        lines.append("**추정 원인**:")
        lines.append("- OHT Vehicle 할당 문제 (차량 부족 또는 점유)")
        lines.append("- OHT Rail Cut 문제 (경로 차단)")
    
    # 위치 변경 타임라인
    if location_changes:
        lines.append("\n" + "-" * 70)
        lines.append("### 📍 위치 변경 타임라인")
        lines.append("")
        for loc in location_changes:
            eq_emoji = {'STB': '📦', 'OHT': '🚃', 'Conveyor': '➡️', 'Lifter': '🔼'}.get(loc['eq_type'], '📍')
            t = loc['time_str'].split()[1] if ' ' in loc['time_str'] else loc['time_str']
            lines.append(f"- {t} | {eq_emoji} {loc['location_str']}")
    
    # 결론
    lines.append("\n" + "=" * 70)
    lines.append("### 📌 결론")
    
    if delays:
        main_delay = max(delays, key=lambda x: x['duration_sec'])
        lines.append(f"\n이 이송 JOB은 총 **{total_str}** 소요되어 정상 범위(20분)를 초과했습니다.")
        lines.append(f"**주요 지연 구간**: {main_delay['segment']} ({main_delay['duration_str']})")
        lines.append(f"**지연 원인**: {main_delay['cause']}")
        
        lines.append("\n### 💡 권장 조치")
        lines.append("1. **OHT 가용성 점검**: 해당 시간대 OHT 차량 상태 확인")
        lines.append("2. **Rail Cut 여부 확인**: Rail 차단으로 인한 우회 가능성")
        lines.append("3. **OHT 작업 부하 분석**: 동시간대 다른 JOB의 OHT 점유 현황")
    else:
        lines.append(f"\n이 이송 JOB은 **정상적으로 완료**되었습니다.")
        lines.append(f"총 소요시간: {total_str}")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


# ============================================================================
# 유틸리티
# ============================================================================
def is_fabjob_data(df: pd.DataFrame) -> bool:
    """FABJOB 데이터인지 판단"""
    if 'MESSAGENAME' not in df.columns:
        return False
    
    messages = df['MESSAGENAME'].dropna().astype(str).tolist()
    vm_count = sum(1 for m in messages if m.startswith('VM-'))
    
    return vm_count >= 5


# ============================================================================
# 테스트
# ============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        
        for enc in ['utf-8', 'cp949', 'euc-kr']:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except:
                continue
        
        if is_fabjob_data(df):
            result = analyze_fabjob(df)
            print(result['preprocessed_text'])
        else:
            print("FABJOB 데이터가 아닙니다.")
    else:
        print("Usage: python fabjob_preprocessor.py <csv_file>")
