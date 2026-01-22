#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FABJOB 로그 전처리 모듈
- CSV에서 시간/이벤트 추출
- 구간별 소요시간 자동 계산  
- HCACK 에러 분석
- LLM 프롬프트용 구조화된 텍스트 생성

server.py의 analyze_amhs_log()에서 FABJOB 감지 시 호출
"""

import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


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
    """XML 태그에서 값 추출: < TAG > value < /TAG >"""
    if pd.isna(text):
        return None
    
    pattern = rf'<\s*{tag}\s*>\s*([^<]*?)\s*<\s*/{tag}\s*>'
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
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}분 {secs}초"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}시간 {mins}분"


def get_duration_status(seconds: float, segment_type: str) -> str:
    """구간별 소요시간 상태 판단"""
    # 정상 기준 (초)
    thresholds = {
        'oht_wait': (60, 180, 300),      # OHT 대기: 1분/3분/5분
        'oht_transfer': (120, 300, 600),  # OHT 이송: 2분/5분/10분
        'conveyor': (300, 480, 900),      # 컨베이어: 5분/8분/15분
        'lifter': (180, 300, 600),        # 리프터: 3분/5분/10분
        'total': (1200, 1800, 2700),      # 전체: 20분/30분/45분
    }
    
    normal, caution, critical = thresholds.get(segment_type, (300, 600, 1200))
    
    if seconds <= normal:
        return "✅ 정상"
    elif seconds <= caution:
        return "🟡 주의"
    elif seconds <= critical:
        return "⚠️ 경고"
    else:
        return "🔴 심각"


def analyze_fabjob(df: pd.DataFrame) -> Dict:
    """
    FABJOB 로그 분석 메인 함수
    
    Returns:
        Dict with 'preprocessed_text' for LLM prompt
    """
    result = {
        'carrier_id': None,
        'lot_id': None,
        'source': {},
        'destination': {},
        'events': [],
        'hcack_events': [],
        'segments': [],
        'delays': [],
        'total_duration_sec': 0,
        'final_status': 'UNKNOWN',
        'preprocessed_text': ''
    }
    
    # 1. 시간 파싱 및 정렬
    df = df.copy()
    df['parsed_time'] = df['TIME_EX'].apply(parse_time_ex)
    df = df.dropna(subset=['parsed_time']).sort_values('parsed_time').reset_index(drop=True)
    
    if df.empty:
        result['preprocessed_text'] = "❌ 시간 정보(TIME_EX)를 파싱할 수 없습니다."
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
        
        # FABTRANSPORTJOBCREATED에서 출발지/목적지 추출
        if 'FABTRANSPORTJOBCREATED' in msg:
            if not result['lot_id']:
                result['lot_id'] = extract_xml_value(text, 'LOTID')
            if not result['source']:
                result['source'] = {
                    'fab': extract_xml_value(text, 'SOURCEFABNAME'),
                    'floor': extract_xml_value(text, 'SOURCEFLOORNAME'),
                    'machine': extract_xml_value(text, 'SOURCEMACHINENAME'),
                }
            if not result['destination']:
                result['destination'] = {
                    'fab': extract_xml_value(text, 'DESTFABNAME'),
                    'floor': extract_xml_value(text, 'DESTFLOORNAME'),
                    'machine': extract_xml_value(text, 'DESTMACHINENAME'),
                }
    
    # 3. 주요 이벤트 추출
    key_messages = [
        'VM-FABTRANSPORTJOBCREATED',
        'RAIL-CARRIERTRANSFERREPLY',
        'RAIL-TRANSFERINITIATED',
        'RAIL-TRANSFERRING',
        'RAIL-VEHICLEACQUIRESTARTED',
        'RAIL-CARRIERINSTALLED',
        'RAIL-TRANSFERCOMPLETED',
        'INTERRAIL-CARRIERTRANSFERREPLY',
        'INTERRAIL-TRANSFERINITIATED',
        'INTERRAIL-TRANSFERRING',
        'INTERRAIL-TRANSFERCOMPLETED',
        'STORAGE-TRANSFERCOMPLETED',
        'VM-TRANSPORTJOBLOCATIONCHANGED',
        'VM-TRANSPORTJOBCOMPLETED',
    ]
    
    for _, row in df.iterrows():
        msg = str(row.get('MESSAGENAME', ''))
        text = str(row.get('TEXT', ''))
        time = row['parsed_time']
        machine = row.get('MACHINENAME', '')
        
        # 주요 메시지만 필터
        if not any(km in msg for km in key_messages):
            continue
        
        event = {
            'time': time,
            'time_str': time.strftime('%H:%M:%S.%f')[:-3],
            'message': msg,
            'machine': machine if pd.notna(machine) else '',
        }
        
        # HCACK 추출
        if 'CARRIERTRANSFERREPLY' in msg:
            hcack = extract_xml_value(text, 'HCACK')
            if hcack:
                event['hcack'] = hcack
                result['hcack_events'].append({
                    'time': time,
                    'time_str': time.strftime('%H:%M:%S'),
                    'hcack': hcack,
                    'message': msg,
                    'machine': machine,
                })
        
        # 위치 변경 시 현재 위치
        if 'LOCATIONCHANGED' in msg:
            event['current_machine'] = extract_xml_value(text, 'CURRENTMACHINENAME')
            event['current_unit'] = extract_xml_value(text, 'CURRENTUNITNAME')
        
        # 완료 상태
        if 'COMPLETED' in msg:
            state = extract_xml_value(text, 'STATE')
            if state:
                event['state'] = state
                if 'TRANSPORTJOBCOMPLETED' in msg:
                    result['final_status'] = state
        
        # 차량 ID
        if 'VEHICLEACQUIRESTARTED' in msg or 'CARRIERINSTALLED' in msg:
            event['vehicle'] = extract_xml_value(text, 'VEHICLEID')
        
        result['events'].append(event)
    
    # 4. 시간 범위 계산
    if result['events']:
        start_time = result['events'][0]['time']
        end_time = result['events'][-1]['time']
        result['start_time'] = start_time
        result['end_time'] = end_time
        result['total_duration_sec'] = (end_time - start_time).total_seconds()
    
    # 5. 구간별 분석
    segments = []
    
    # HCACK 지연 분석 (가장 중요!)
    hcack_events = result['hcack_events']
    if len(hcack_events) >= 2:
        first_hcack = hcack_events[0]
        
        # HCACK=2 (거부) 찾기
        rejections = [h for h in hcack_events if h['hcack'] == '2']
        # HCACK=4 (성공) 찾기
        success = [h for h in hcack_events if h['hcack'] == '4']
        
        if rejections and success:
            first_reject = rejections[0]
            first_success = success[0]
            wait_sec = (first_success['time'] - first_reject['time']).total_seconds()
            
            if wait_sec > 60:  # 1분 이상 대기면 지연으로 판단
                segments.append({
                    'name': 'OHT 차량 대기 (HCACK=2 → 4)',
                    'start': first_reject['time_str'],
                    'end': first_success['time_str'],
                    'duration_sec': wait_sec,
                    'duration_str': format_duration(wait_sec),
                    'status': get_duration_status(wait_sec, 'oht_wait'),
                    'is_delay': True,
                    'detail': f"HCACK=2 거부 {len(rejections)}회 발생"
                })
                result['delays'].append({
                    'segment': 'OHT 차량 대기',
                    'duration_sec': wait_sec,
                    'duration_str': format_duration(wait_sec),
                    'cause': f'HCACK=2 (OHT 명령 거부) {len(rejections)}회'
                })
    
    # 이송 구간 분석
    transfer_events = [
        ('RAIL-TRANSFERINITIATED', 'RAIL-TRANSFERCOMPLETED', 'M14 OHT 이송', 'oht_transfer'),
        ('INTERRAIL-TRANSFERINITIATED', 'INTERRAIL-TRANSFERCOMPLETED', 'Conveyor 이송', 'conveyor'),
        ('STORAGE-TRANSFERINITIATED', 'STORAGE-TRANSFERCOMPLETED', 'Lifter 이송', 'lifter'),
    ]
    
    for start_msg, end_msg, name, seg_type in transfer_events:
        start_evt = next((e for e in result['events'] if start_msg in e['message']), None)
        end_evt = next((e for e in result['events'] if end_msg in e['message']), None)
        
        if start_evt and end_evt:
            dur = (end_evt['time'] - start_evt['time']).total_seconds()
            segments.append({
                'name': name,
                'start': start_evt['time_str'],
                'end': end_evt['time_str'],
                'duration_sec': dur,
                'duration_str': format_duration(dur),
                'status': get_duration_status(dur, seg_type),
                'machine': start_evt.get('machine', ''),
                'is_delay': False,
            })
    
    # 위치 변경 기반 구간 분석
    location_events = [e for e in result['events'] if 'LOCATIONCHANGED' in e['message']]
    for i in range(len(location_events) - 1):
        curr = location_events[i]
        next_evt = location_events[i + 1]
        dur = (next_evt['time'] - curr['time']).total_seconds()
        
        curr_machine = curr.get('current_machine', 'Unknown')
        next_machine = next_evt.get('current_machine', 'Unknown')
        
        segments.append({
            'name': f"{curr_machine} → {next_machine}",
            'start': curr['time_str'],
            'end': next_evt['time_str'],
            'duration_sec': dur,
            'duration_str': format_duration(dur),
            'status': get_duration_status(dur, 'oht_transfer'),
            'is_delay': dur > 300,  # 5분 이상이면 지연
        })
    
    result['segments'] = segments
    
    # 6. LLM 프롬프트용 텍스트 생성
    result['preprocessed_text'] = generate_prompt_text(result)
    
    return result


def generate_prompt_text(analysis: Dict) -> str:
    """분석 결과를 LLM 프롬프트용 텍스트로 변환"""
    
    lines = []
    lines.append("=" * 60)
    lines.append("📊 FABJOB 로그 자동 분석 결과 (전처리 완료)")
    lines.append("=" * 60)
    
    # 기본 정보
    lines.append("\n## 1. 기본 정보")
    lines.append(f"- 캐리어: {analysis.get('carrier_id', 'N/A')}")
    lines.append(f"- LOT ID: {analysis.get('lot_id', 'N/A')}")
    
    src = analysis.get('source', {})
    dst = analysis.get('destination', {})
    lines.append(f"- 출발지: {src.get('fab', '')} {src.get('floor', '')} {src.get('machine', '')}")
    lines.append(f"- 목적지: {dst.get('fab', '')} {dst.get('floor', '')} {dst.get('machine', '')}")
    
    # 시간 정보
    lines.append("\n## 2. 시간 정보")
    if analysis.get('start_time'):
        lines.append(f"- 시작: {analysis['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    if analysis.get('end_time'):
        lines.append(f"- 종료: {analysis['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_sec = analysis.get('total_duration_sec', 0)
    total_str = format_duration(total_sec)
    total_status = get_duration_status(total_sec, 'total')
    lines.append(f"- 전체 소요시간: {total_str} {total_status}")
    lines.append(f"- 정상 기준: 20분 이내")
    lines.append(f"- 최종 상태: {analysis.get('final_status', 'UNKNOWN')}")
    
    # HCACK 에러 분석
    hcack_events = analysis.get('hcack_events', [])
    if hcack_events:
        lines.append("\n## 3. HCACK 응답 분석 (⚠️ 핵심!)")
        lines.append("| 시간 | HCACK | 의미 | 장비 |")
        lines.append("|------|-------|------|------|")
        
        hcack_meanings = {
            '0': '✅ 성공',
            '2': '❌ 거부 (차량 부재/점유)',
            '4': '✅ 이송 시작됨',
            '6': '❌ 실패 (장비 오류)',
        }
        
        for h in hcack_events:
            meaning = hcack_meanings.get(h['hcack'], f"코드 {h['hcack']}")
            lines.append(f"| {h['time_str']} | {h['hcack']} | {meaning} | {h.get('machine', '')} |")
        
        # HCACK=2 분석
        rejections = [h for h in hcack_events if h['hcack'] == '2']
        if rejections:
            lines.append(f"\n⚠️ HCACK=2 (명령 거부) {len(rejections)}회 발생!")
            lines.append("→ OHT 차량 할당 실패로 인한 대기 발생")
    
    # 지연 구간
    delays = analysis.get('delays', [])
    if delays:
        lines.append("\n## 4. 🔴 지연 구간 분석")
        for d in delays:
            lines.append(f"- {d['segment']}: {d['duration_str']}")
            lines.append(f"  원인: {d['cause']}")
    
    # 구간별 소요시간
    segments = analysis.get('segments', [])
    if segments:
        lines.append("\n## 5. 구간별 소요시간")
        lines.append("| 구간 | 시작 | 종료 | 소요시간 | 상태 |")
        lines.append("|------|------|------|----------|------|")
        
        for seg in segments:
            delay_mark = "🔴" if seg.get('is_delay') else ""
            lines.append(f"| {delay_mark}{seg['name']} | {seg['start']} | {seg['end']} | {seg['duration_str']} | {seg['status']} |")
    
    # 주요 이벤트 타임라인
    events = analysis.get('events', [])
    if events:
        lines.append("\n## 6. 주요 이벤트 타임라인")
        for e in events[:20]:  # 최대 20개
            extra = ""
            if 'hcack' in e:
                extra = f" [HCACK={e['hcack']}]"
            if 'vehicle' in e:
                extra = f" [차량: {e['vehicle']}]"
            if 'current_machine' in e:
                extra = f" [위치: {e['current_machine']}]"
            if 'state' in e:
                extra = f" [{e['state']}]"
            
            lines.append(f"- {e['time_str']} | {e['message']}{extra}")
    
    lines.append("\n" + "=" * 60)
    lines.append("위 분석 결과를 바탕으로 한국어로 상세히 설명해주세요.")
    lines.append("특히 지연 원인과 개선 방안을 제시해주세요.")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def is_fabjob_data(df: pd.DataFrame) -> bool:
    """FABJOB 데이터인지 판단"""
    if 'MESSAGENAME' not in df.columns:
        return False
    
    messages = df['MESSAGENAME'].dropna().astype(str).tolist()
    vm_count = sum(1 for m in messages if m.startswith('VM-'))
    
    # VM- 메시지가 5개 이상이면 FABJOB
    return vm_count >= 5


# 테스트
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        df = pd.read_csv(csv_path)
        
        if is_fabjob_data(df):
            result = analyze_fabjob(df)
            print(result['preprocessed_text'])
        else:
            print("FABJOB 데이터가 아닙니다.")
    else:
        print("Usage: python fabjob_preprocessor.py <csv_file>")