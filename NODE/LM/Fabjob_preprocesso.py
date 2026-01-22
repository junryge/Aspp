#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FABJOB 로그 전처리 모듈 V2.0
- VM-TRANSPORTJOBLOCATIONCHANGED 기반 정확한 구간 계산
- HCACK 에러 코드 분석 (지연 원인 파악)
- STB vs Stocker 정확한 구분
- server.py의 analyze_amhs_log()에서 FABJOB 감지 시 호출
"""

import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 장비 유형 분류
# ============================================================================
EQUIPMENT_TYPES = {
    'STB': {
        'patterns': ['ANZ'],  # 4ANZ40G1, 6ANZ0202
        'description': 'Stocker Bay (캐리어 임시 보관)'
    },
    'OHT': {
        'patterns': ['ACM', 'ECM'],  # 4ACM4701, 6ACM3901, 6ECMB101
        'description': 'Overhead Hoist Transport (천장 이송)'
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


def get_fab_floor(machine_name: str) -> str:
    """장비명에서 FAB/층 추출 (예: 4ANZ40G1 -> M14A, 6ABL0121 -> M16A)"""
    if not machine_name or len(machine_name) < 1:
        return ''
    
    first_char = machine_name[0]
    if first_char == '4':
        return 'M14A'
    elif first_char == '6':
        return 'M16A'
    return ''


def format_location(machine_name: str, unit_name: str = '') -> str:
    """장비명을 읽기 좋은 형식으로 변환"""
    if not machine_name:
        return 'Unknown'
    
    eq_type = get_equipment_type(machine_name)
    fab = get_fab_floor(machine_name)
    
    if eq_type == 'STB':
        return f"{fab} STB({machine_name})"
    elif eq_type == 'OHT':
        # V00138 같은 차량 ID가 unit에 있으면 포함
        if unit_name and unit_name.startswith('V'):
            return f"{fab} OHT({machine_name}, 차량:{unit_name})"
        elif unit_name and 'BV' in unit_name:
            return f"{fab} Bridge OHT({machine_name})"
        return f"{fab} OHT({machine_name})"
    elif eq_type == 'Conveyor':
        # Conveyor 입구/출구 구분
        port_info = ''
        if unit_name:
            if '_IN' in unit_name:
                port_info = ' 입구'
            elif '_OUT' in unit_name:
                port_info = ' 출구'
        return f"{fab} Conveyor({machine_name}){port_info}"
    elif eq_type == 'Lifter':
        # Lifter 층/위치 구분 (AI=입구, AO=출구, RM=내부)
        floor_info = ''
        if unit_name:
            # 층 번호 추출 (예: 6ABL0121_AI323 -> 3층, 6ABL0121_AO621 -> 6층)
            if '_AI' in unit_name:
                floor_match = re.search(r'_AI(\d)', unit_name)
                if floor_match:
                    floor_info = f" {floor_match.group(1)}F 입구"
                else:
                    floor_info = ' 입구'
            elif '_AO' in unit_name:
                floor_match = re.search(r'_AO(\d)', unit_name)
                if floor_match:
                    floor_info = f" {floor_match.group(1)}F 출구"
                else:
                    floor_info = ' 출구'
            elif 'RM' in unit_name:
                floor_info = ' 내부(이동중)'
        return f"{fab} Lifter({machine_name}){floor_info}"
    
    return f"{machine_name}"


# ============================================================================
# 시간/XML 파싱
# ============================================================================
def parse_time_ex(time_str: str) -> Optional[datetime]:
    """TIME_EX 컬럼에서 시간 추출: [2026-01-21 11:12:03.885]"""
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
    """XML 태그에서 값 추출 (공백 허용): < TAG > value < /TAG >"""
    if pd.isna(text):
        return None
    
    # 공백 포함한 패턴
    pattern = rf'<\s*{tag}\s*>\s*([^<]*?)\s*<\s*/\s*{tag}\s*>'
    match = re.search(pattern, str(text), re.IGNORECASE)
    if match:
        val = match.group(1).strip()
        return val if val else None
    return None


def format_duration(seconds: float) -> str:
    """초를 읽기 좋은 형식으로 변환"""
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
    """구간별 소요시간 상태 판단 -> (상태, 이모지)"""
    thresholds = {
        'stb_to_oht': (60, 180, 300),      # STB→OHT: 1분/3분/5분
        'oht_transfer': (120, 300, 600),    # OHT 이송: 2분/5분/10분
        'oht_to_conveyor': (120, 240, 480), # OHT→Conveyor: 2분/4분/8분
        'conveyor': (300, 480, 900),        # Conveyor: 5분/8분/15분
        'lifter': (180, 300, 600),          # Lifter: 3분/5분/10분
        'total': (1200, 1800, 2700),        # 전체: 20분/30분/45분
    }
    
    normal, caution, critical = thresholds.get(segment_type, (180, 300, 600))
    
    if seconds <= normal:
        return ("정상", "✅")
    elif seconds <= caution:
        return ("주의", "🟡")
    elif seconds <= critical:
        return ("경고", "⚠️")
    else:
        return ("지연", "🔴")


# ============================================================================
# HCACK 에러 코드
# ============================================================================
HCACK_MEANINGS = {
    '0': ('성공', '이송 명령 정상 수락'),
    '2': ('거부', 'OHT 차량 부재/점유 (⚠️ 지연 원인)'),
    '4': ('시작됨', '이송 명령 수락 후 실행'),
    '6': ('실패', '이송 불가 - 장비 오류'),
}


def analyze_hcack_delays(hcack_events: List[dict]) -> dict:
    """HCACK 이벤트에서 지연 분석"""
    result = {
        'rejections': [],
        'success_time': None,
        'first_reject_time': None,
        'delay_seconds': 0,
        'rejection_count': 0
    }
    
    for h in hcack_events:
        if h['hcack'] == '2':
            result['rejections'].append(h)
            if not result['first_reject_time']:
                result['first_reject_time'] = h['time']
        elif h['hcack'] == '4':
            if not result['success_time']:
                result['success_time'] = h['time']
    
    result['rejection_count'] = len(result['rejections'])
    
    if result['first_reject_time'] and result['success_time']:
        result['delay_seconds'] = (result['success_time'] - result['first_reject_time']).total_seconds()
    
    return result


# ============================================================================
# 메인 분석 함수
# ============================================================================
def analyze_fabjob(df: pd.DataFrame) -> Dict:
    """
    FABJOB 로그 분석 메인 함수 V2.0
    
    핵심 개선:
    1. VM-TRANSPORTJOBLOCATIONCHANGED 기반 정확한 구간 계산
    2. HCACK=2 지연 원인 명확히 분석
    3. STB vs Stocker 정확한 구분
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
        result['preprocessed_text'] = "❌ 시간 정보(TIME_EX)를 파싱할 수 없습니다."
        return result
    
    # 2. 기본 정보 추출 (JOB 생성 메시지)
    for _, row in df.iterrows():
        text = str(row.get('TEXT', ''))
        msg = str(row.get('MESSAGENAME', ''))
        
        if not result['carrier_id']:
            carrier = row.get('CARRIER')
            if pd.notna(carrier) and carrier:
                result['carrier_id'] = str(carrier).strip().strip("'")
        
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
        
        if 'TRANSPORTJOBCOMPLETED' in msg:
            result['end_time'] = row['parsed_time']
            result['final_status'] = extract_xml_value(text, 'STATE') or 'COMPLETED'
    
    # 3. HCACK 이벤트 추출
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
                    'message': msg,
                    'machine': row.get('MACHINENAME', ''),
                })
    
    # 4. VM-TRANSPORTJOBLOCATIONCHANGED 기반 위치 변경 추출
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
                    'eq_type': get_equipment_type(machine),
                }
                
                # 중복 제거 로직 개선:
                # - 같은 machine이면서 같은 unit이면 스킵
                # - 같은 machine이라도 unit이 다르면 (Conveyor IN→OUT, Lifter 층간) 추가
                if not location_changes:
                    location_changes.append(loc)
                else:
                    last = location_changes[-1]
                    # 완전히 같으면 스킵
                    if last['machine'] == machine and last['unit'] == unit:
                        continue
                    # 같은 machine이지만 unit이 다르면 (중요한 상태 변화)
                    # Conveyor: IN → OUT (입구→출구)
                    # Lifter: AI → RM → AO (입구→내부→출구)
                    if last['machine'] == machine:
                        # Conveyor나 Lifter의 경우 unit 변경은 의미있는 이동
                        eq_type = get_equipment_type(machine)
                        if eq_type in ['Conveyor', 'Lifter']:
                            location_changes.append(loc)
                        # 그 외는 스킵 (같은 OHT 컨트롤러 내 중복)
                    else:
                        location_changes.append(loc)
    
    result['location_changes'] = location_changes
    
    # 5. 전체 시간 계산
    if result['start_time'] and result['end_time']:
        result['total_duration_sec'] = (result['end_time'] - result['start_time']).total_seconds()
    
    # 6. 구간별 분석 (핵심!)
    segments = []
    
    # 6-1. HCACK 지연 분석 (STB → OHT 구간)
    hcack_analysis = analyze_hcack_delays(result['hcack_events'])
    
    if hcack_analysis['rejection_count'] > 0 and result['start_time']:
        # 첫 위치 변경이 OHT 획득 시점
        first_oht_time = None
        if location_changes:
            first_loc = location_changes[0]
            if first_loc['eq_type'] == 'OHT':
                first_oht_time = first_loc['time']
        
        # JOB 시작 → 첫 OHT 획득까지가 "STB → OHT" 구간
        if first_oht_time:
            stb_to_oht_sec = (first_oht_time - result['start_time']).total_seconds()
            status, emoji = get_duration_status(stb_to_oht_sec, 'stb_to_oht')
            
            src_machine = result['source'].get('machine', 'Unknown')
            first_oht_machine = location_changes[0]['machine'] if location_changes else 'Unknown'
            
            segment = {
                'name': f"{format_location(src_machine)} → {format_location(first_oht_machine, location_changes[0].get('unit', ''))}",
                'start_time': result['start_time'],
                'start_str': result['start_time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'end_time': first_oht_time,
                'end_str': first_oht_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'duration_sec': stb_to_oht_sec,
                'duration_str': format_duration(stb_to_oht_sec),
                'status': status,
                'emoji': emoji,
                'is_delay': stb_to_oht_sec > 180,  # 3분 이상이면 지연
            }
            
            # HCACK=2로 인한 지연이면 원인 추가
            if hcack_analysis['delay_seconds'] > 60:
                segment['delay_cause'] = f"HCACK=2 (OHT 명령 거부) {hcack_analysis['rejection_count']}회"
                segment['is_delay'] = True
                
                result['delays'].append({
                    'segment': segment['name'],
                    'duration_sec': stb_to_oht_sec,
                    'duration_str': format_duration(stb_to_oht_sec),
                    'cause': segment['delay_cause'],
                    'hcack_events': hcack_analysis['rejections']
                })
            
            segments.append(segment)
    
    # 6-2. 위치 변경 기반 구간 분석
    for i in range(len(location_changes) - 1):
        curr = location_changes[i]
        next_loc = location_changes[i + 1]
        
        duration_sec = (next_loc['time'] - curr['time']).total_seconds()
        
        # 구간 유형 결정
        seg_type = 'oht_transfer'
        if curr['eq_type'] == 'Conveyor' or next_loc['eq_type'] == 'Conveyor':
            seg_type = 'conveyor'
        elif curr['eq_type'] == 'Lifter' or next_loc['eq_type'] == 'Lifter':
            seg_type = 'lifter'
        elif curr['eq_type'] == 'OHT' and next_loc['eq_type'] == 'Conveyor':
            seg_type = 'oht_to_conveyor'
        
        status, emoji = get_duration_status(duration_sec, seg_type)
        
        segment = {
            'name': f"{curr['location_str']} → {next_loc['location_str']}",
            'start_time': curr['time'],
            'start_str': curr['time_str'],
            'end_time': next_loc['time'],
            'end_str': next_loc['time_str'],
            'duration_sec': duration_sec,
            'duration_str': format_duration(duration_sec),
            'status': status,
            'emoji': emoji,
            'is_delay': duration_sec > 300,  # 5분 이상이면 지연
        }
        segments.append(segment)
    
    result['segments'] = segments
    
    # 7. 프롬프트용 텍스트 생성
    result['preprocessed_text'] = generate_prompt_text(result)
    
    return result


def generate_prompt_text(analysis: Dict) -> str:
    """분석 결과를 LLM 프롬프트용 텍스트로 변환 - R3.TXT 형식에 맞게"""
    
    lines = []
    
    # 헤더
    lines.append("=" * 70)
    lines.append("📦 FABJOB 이송 분석 리포트")
    lines.append("=" * 70)
    
    # 기본 정보
    carrier = analysis.get('carrier_id', 'N/A')
    src = analysis.get('source', {})
    location_changes = analysis.get('location_changes', [])
    
    # 출발지
    src_str = format_location(src.get('machine', ''))
    
    # 목적지: 마지막 위치 변경에서 가져옴 (실제 도착 위치)
    if location_changes:
        last_loc = location_changes[-1]
        dst_str = last_loc['location_str']
    else:
        dst = analysis.get('destination', {})
        dst_str = format_location(dst.get('machine', ''))
    
    lines.append(f"\n📍 캐리어: {carrier}")
    lines.append(f"📍 전체 경로: {src_str} → {dst_str}")
    
    # 시간 정보
    total_sec = analysis.get('total_duration_sec', 0)
    total_str = format_duration(total_sec)
    status, emoji = get_duration_status(total_sec, 'total')
    
    lines.append(f"\n⏱️ 총 소요시간: {total_str} (정상: 20분 이내)")
    lines.append(f"📌 최종 상태: {analysis.get('final_status', 'UNKNOWN')}")
    
    if total_sec > 1200:  # 20분 초과
        lines.append(f"{emoji} 결과: **지연 발생**")
    else:
        lines.append(f"{emoji} 결과: **정상 완료**")
    
    # HCACK 분석 (중요!)
    hcack_events = analysis.get('hcack_events', [])
    if hcack_events:
        lines.append("\n" + "-" * 70)
        lines.append("### ⚠️ HCACK 응답 분석 (OHT 명령 응답)")
        lines.append("")
        lines.append("| 시간 | HCACK | 의미 | 장비 |")
        lines.append("|------|-------|------|------|")
        
        for h in hcack_events:
            hcack_val = h['hcack']
            meaning, desc = HCACK_MEANINGS.get(hcack_val, ('알수없음', ''))
            emoji = '❌' if hcack_val == '2' else ('✅' if hcack_val in ['0', '4'] else '⚠️')
            lines.append(f"| {h['time_str']} | {hcack_val} | {emoji} {meaning} | {h.get('machine', '')} |")
        
        # HCACK=2 분석
        rejections = [h for h in hcack_events if h['hcack'] == '2']
        if rejections:
            lines.append("")
            lines.append(f"🔴 **HCACK=2 (OHT 명령 거부) {len(rejections)}회 발생!**")
            lines.append("→ OHT 차량 할당 실패로 인한 대기 발생")
            
            # 첫 거절 ~ 성공까지 시간
            success = [h for h in hcack_events if h['hcack'] == '4']
            if success:
                delay_sec = (success[0]['time'] - rejections[0]['time']).total_seconds()
                lines.append(f"→ 거절 시작: {rejections[0]['time_str']}")
                lines.append(f"→ 성공 시점: {success[0]['time_str']}")
                lines.append(f"→ **OHT 대기 시간: {format_duration(delay_sec)}**")
    
    # 구간별 분석 (핵심!)
    segments = analysis.get('segments', [])
    if segments:
        lines.append("\n" + "-" * 70)
        lines.append("### 🕒 구간별 소요시간 분석")
        lines.append("")
        lines.append("| # | 구간 | 시작 | 종료 | 소요시간 | 상태 |")
        lines.append("|---|------|------|------|----------|------|")
        
        for i, seg in enumerate(segments, 1):
            name = seg['name']
            if len(name) > 50:
                name = name[:47] + "..."
            
            delay_mark = "🔴 " if seg.get('is_delay') else ""
            lines.append(f"| {i} | {delay_mark}{name} | {seg['start_str'].split()[1] if ' ' in seg['start_str'] else seg['start_str']} | {seg['end_str'].split()[1] if ' ' in seg['end_str'] else seg['end_str']} | {seg['duration_str']} | {seg['emoji']} {seg['status']} |")
    
    # 지연 구간 상세
    delays = analysis.get('delays', [])
    if delays:
        lines.append("\n" + "-" * 70)
        lines.append("### 🔴 지연 구간 상세")
        for d in delays:
            lines.append(f"\n**구간**: {d['segment']}")
            lines.append(f"**소요시간**: {d['duration_str']}")
            lines.append(f"**원인**: {d['cause']}")
    
    # 위치 변경 타임라인
    location_changes = analysis.get('location_changes', [])
    if location_changes:
        lines.append("\n" + "-" * 70)
        lines.append("### 📍 위치 변경 타임라인")
        lines.append("")
        for loc in location_changes:
            eq_emoji = {'STB': '📦', 'OHT': '🚃', 'Conveyor': '➡️', 'Lifter': '🔼'}.get(loc['eq_type'], '📍')
            lines.append(f"- {loc['time_str']} | {eq_emoji} {loc['location_str']}")
    
    # 결론
    lines.append("\n" + "=" * 70)
    lines.append("### 📌 분석 결론")
    
    if delays:
        main_delay = max(delays, key=lambda x: x['duration_sec'])
        lines.append(f"\n이 이송 JOB은 총 **{total_str}** 소요되어 정상 범위(20분)를 초과했습니다.")
        lines.append(f"**주요 지연 구간**: {main_delay['segment']}")
        lines.append(f"**지연 시간**: {main_delay['duration_str']}")
        lines.append(f"**지연 원인**: {main_delay['cause']}")
        
        # HCACK=2가 원인이면 권장 조치 추가
        if 'HCACK=2' in main_delay['cause']:
            lines.append("\n### 💡 권장 조치")
            lines.append("1. **OHT 가용성 점검**: 해당 시간대 OHT 차량 상태 확인")
            lines.append("2. **OHT Rail Cut 여부**: Rail 차단으로 인한 우회 발생 가능성")
            lines.append("3. **OHT 작업 부하 분석**: 동시간대 다른 JOB의 OHT 점유 현황")
    else:
        lines.append(f"\n이 이송 JOB은 **정상적으로 완료**되었습니다.")
        lines.append(f"총 소요시간: {total_str}")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


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
        print("Usage: python fabjob_preprocessor_v2.py <csv_file>")