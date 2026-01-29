#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hid_zone_csv_cre.py - layout.xml에서 HID_Zone_Master.csv 생성

사용법:
    1. 모듈로 import:
       from hid_zone_csv_cre import generate_hid_zone_csv
       generate_hid_zone_csv(xml_path, output_csv_path)

    2. 직접 실행:
       python hid_zone_csv_cre.py [layout.xml 또는 layout.zip 경로] [출력 CSV 경로]
"""

import os
import re
import csv
import zipfile
from typing import Dict, List, Tuple


def safe_int(value, default=0):
    """안전하게 정수로 변환 (실패 시 기본값 반환)"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def parse_mcp_zones_from_xml(xml_content: str) -> List[Dict]:
    """
    layout.xml에서 McpZone 정보 추출 (라인 단위 파싱)

    Returns:
        zones: [{
            'no': int,
            'id': int,
            'vehicle_max': int,
            'vehicle_precaution': int,
            'type': int,
            'entries': [{'start': int, 'end': int, 'zcu': str}, ...],
            'exits': [{'start': int, 'end': int}, ...],
            'cut_lanes': [{'start': int, 'end': int}, ...]
        }, ...]
    """
    print("McpZone 파싱 시작...")

    zones = {}
    current_zone = None
    current_zone_params = {}
    current_entry = None
    current_exit = None
    current_cut_lane = None
    entry_params = {}
    exit_params = {}
    cut_lane_params = {}

    lines = xml_content.split('\n')
    total_lines = len(lines)
    print(f"  총 라인 수: {total_lines:,}")

    for line_no, line in enumerate(lines):
        if line_no % 500000 == 0:
            print(f"  처리 중: {line_no:,}/{total_lines:,} ({line_no*100//total_lines}%)")

        line = line.strip()

        # McpZone 그룹 시작
        if '<group name="McpZone' in line and 'mcpzone.McpZone"' in line:
            # 이전 zone 저장
            if current_zone is not None:
                zone_no = safe_int(current_zone_params.get('no', 0))
                if zone_no > 0:
                    zones[zone_no] = {
                        'no': zone_no,
                        'id': safe_int(current_zone_params.get('id', zone_no), zone_no),
                        'vehicle_max': safe_int(current_zone_params.get('vehicle-max', 0)),
                        'vehicle_precaution': safe_int(current_zone_params.get('vehicle-precaution', 0)),
                        'type': safe_int(current_zone_params.get('type', 0)),
                        'entries': current_zone_params.get('entries', []),
                        'exits': current_zone_params.get('exits', []),
                        'cut_lanes': current_zone_params.get('cut_lanes', [])
                    }

            current_zone_params = {'entries': [], 'exits': [], 'cut_lanes': []}
            current_zone = line
            current_entry = None
            current_exit = None
            current_cut_lane = None
            continue

        # Entry 그룹 시작
        if current_zone and '<group name="Entry' in line and 'mcpzone.Entry"' in line:
            current_entry = line
            entry_params = {}
            continue

        # Exit 그룹 시작
        if current_zone and '<group name="Exit' in line and 'mcpzone.Exit"' in line:
            current_exit = line
            exit_params = {}
            continue

        # CutLane 그룹 시작
        if current_zone and '<group name="CutLane' in line and 'mcpzone.CutLane"' in line:
            current_cut_lane = line
            cut_lane_params = {}
            continue

        # 그룹 종료
        if '</group>' in line:
            if current_entry:
                if 'start' in entry_params and 'end' in entry_params:
                    current_zone_params['entries'].append({
                        'start': int(entry_params.get('start', 0)),
                        'end': int(entry_params.get('end', 0)),
                        'zcu': entry_params.get('stop-zcu', '')
                    })
                current_entry = None
            elif current_exit:
                if 'start' in exit_params and 'end' in exit_params:
                    current_zone_params['exits'].append({
                        'start': int(exit_params.get('start', 0)),
                        'end': int(exit_params.get('end', 0))
                    })
                current_exit = None
            elif current_cut_lane:
                if 'start' in cut_lane_params and 'end' in cut_lane_params:
                    current_zone_params['cut_lanes'].append({
                        'start': int(cut_lane_params.get('start', 0)),
                        'end': int(cut_lane_params.get('end', 0))
                    })
                current_cut_lane = None
            continue

        # 파라미터 파싱
        if '<param ' in line and 'key="' in line and 'value="' in line:
            key_match = re.search(r'key="([^"]+)"', line)
            value_match = re.search(r'value="([^"]*)"', line)

            if key_match and value_match:
                key = key_match.group(1)
                value = value_match.group(1)

                if current_entry:
                    entry_params[key] = value
                elif current_exit:
                    exit_params[key] = value
                elif current_cut_lane:
                    cut_lane_params[key] = value
                elif current_zone:
                    current_zone_params[key] = value

    # 마지막 zone 저장
    if current_zone is not None:
        zone_no = safe_int(current_zone_params.get('no', 0))
        if zone_no > 0:
            zones[zone_no] = {
                'no': zone_no,
                'id': safe_int(current_zone_params.get('id', zone_no), zone_no),
                'vehicle_max': safe_int(current_zone_params.get('vehicle-max', 0)),
                'vehicle_precaution': safe_int(current_zone_params.get('vehicle-precaution', 0)),
                'type': safe_int(current_zone_params.get('type', 0)),
                'entries': current_zone_params.get('entries', []),
                'exits': current_zone_params.get('exits', []),
                'cut_lanes': current_zone_params.get('cut_lanes', [])
            }

    print(f"  총 McpZone: {len(zones)}개")

    return list(zones.values())


def derive_bay_zone(zone_no: int) -> str:
    """
    Zone 번호로부터 Bay_Zone 추정 (기존 CSV 패턴 기반)
    """
    # 기존 HID_Zone_Master.csv 패턴 분석:
    # Zone 1-2: B01, Zone 3: B02, Zone 4: B03, ...
    # 대략적인 매핑 (정확한 매핑은 외부 데이터 필요)
    bay_mapping = {
        1: 'B01', 2: 'B01',
        3: 'B02',
        4: 'B03',
        5: 'B04',
        6: 'B05', 7: 'B05',
        8: 'B06', 9: 'B06',
    }
    if zone_no in bay_mapping:
        return bay_mapping[zone_no]

    # 기본 추정: zone 10 이상은 패턴으로 추정
    # 실제 매핑은 외부 데이터 필요
    bay_num = (zone_no // 10) + 1
    return f'B{bay_num:02d}'


def generate_hid_zone_csv(zones: List[Dict], output_path: str,
                          project_name: str = "M14 Project Ph-1") -> None:
    """
    HID_Zone_Master.csv 파일 생성

    Args:
        zones: parse_mcp_zones_from_xml()에서 반환된 zone 목록
        output_path: 출력 CSV 파일 경로
        project_name: 프로젝트 이름 (기본값: M14 Project Ph-1)
    """
    print(f"HID_Zone_Master.csv 생성: {output_path}")

    # CSV 헤더
    headers = [
        'Zone_ID', 'HID_No', 'Bay_Zone', 'Sub_Region', 'Full_Name',
        'Territory', 'Type', 'IN_Count', 'OUT_Count', 'IN_Lanes', 'OUT_Lanes',
        'Vehicle_Max', 'Vehicle_Precaution', 'Project', 'ZCU', 'HID_Type'
    ]

    rows = []

    # zone을 no 기준으로 정렬
    sorted_zones = sorted(zones, key=lambda z: z['no'])

    for zone in sorted_zones:
        zone_no = zone['no']
        zone_id = zone['id']

        # IN/OUT Lanes 포맷
        in_lanes = '; '.join([f"{e['start']}→{e['end']}" for e in zone['entries']])
        out_lanes = '; '.join([f"{e['start']}→{e['end']}" for e in zone['exits']])

        # ZCU 추출 (첫 번째 entry의 stop-zcu)
        zcu = ''
        for entry in zone['entries']:
            if entry.get('zcu'):
                zcu = entry['zcu']
                break

        # Bay_Zone 및 Sub_Region 추정
        bay_zone = derive_bay_zone(zone_no)
        sub_region = ((zone_no - 1) % 2) + 1  # 대략적 추정

        # HID 타입 추정 (type 필드 기반)
        hid_type_map = {1: 'HID4', 2: 'HID3', 3: 'HID2'}
        hid_type = hid_type_map.get(zone['type'], 'HID4')

        row = {
            'Zone_ID': zone_id,
            'HID_No': f'HID-OHT-{zone_id:03d}',
            'Bay_Zone': bay_zone,
            'Sub_Region': sub_region,
            'Full_Name': f'HID-{bay_zone}-{sub_region}({zone_id:03d})',
            'Territory': 1,  # 기본값
            'Type': 'HID',
            'IN_Count': len(zone['entries']),
            'OUT_Count': len(zone['exits']),
            'IN_Lanes': in_lanes,
            'OUT_Lanes': out_lanes,
            'Vehicle_Max': zone['vehicle_max'],
            'Vehicle_Precaution': zone['vehicle_precaution'],
            'Project': project_name,
            'ZCU': zcu,
            'HID_Type': hid_type
        }
        rows.append(row)

    # CSV 파일 작성 (UTF-8 BOM)
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  완료: {len(rows)}개 Zone, {os.path.getsize(output_path):,} bytes")


def load_xml_content(source_path: str) -> str:
    """
    layout.xml 또는 layout.zip에서 XML 내용 로드
    """
    if source_path.endswith('.zip'):
        print(f"layout.zip에서 layout.xml 추출 중: {source_path}")
        with zipfile.ZipFile(source_path, 'r') as zf:
            with zf.open('layout.xml') as f:
                return f.read().decode('utf-8')
    else:
        print(f"layout.xml 읽는 중: {source_path}")
        with open(source_path, 'r', encoding='utf-8') as f:
            return f.read()


def create_hid_zone_csv(xml_or_zip_path: str, output_csv_path: str,
                        project_name: str = "M14 Project Ph-1") -> None:
    """
    layout.xml 또는 layout.zip에서 HID_Zone_Master.csv 생성

    Args:
        xml_or_zip_path: layout.xml 또는 layout.zip 파일 경로
        output_csv_path: 출력 CSV 파일 경로
        project_name: 프로젝트 이름
    """
    print("=" * 60)
    print("HID_Zone_Master.csv 생성")
    print("=" * 60)
    print(f"입력: {xml_or_zip_path}")
    print(f"출력: {output_csv_path}")
    print()

    # XML 내용 로드
    xml_content = load_xml_content(xml_or_zip_path)
    print(f"  XML 크기: {len(xml_content):,} bytes")

    # McpZone 파싱
    zones = parse_mcp_zones_from_xml(xml_content)

    # CSV 생성
    generate_hid_zone_csv(zones, output_csv_path, project_name)

    print()
    print("완료!")
    print("=" * 60)


def main():
    """
    커맨드라인 실행용 메인 함수
    """
    import sys
    import pathlib

    # 기본 경로
    script_dir = pathlib.Path(__file__).parent.resolve()

    # layout.xml 또는 layout.zip 찾기
    xml_path = script_dir / 'layout' / 'layout.xml'
    zip_path = script_dir / 'layout' / 'layout' / 'layout.zip'

    if xml_path.exists():
        default_input = str(xml_path)
    elif zip_path.exists():
        default_input = str(zip_path)
    else:
        default_input = str(zip_path)

    default_output = str(script_dir / 'HID_Zone_Master_Generated.csv')

    # 명령행 인자 처리
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output

    if not os.path.exists(input_path):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    create_hid_zone_csv(input_path, output_path)


if __name__ == '__main__':
    main()