#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hid_zone_csv_cre.py - layout.xml에서 HID_ZONE_Master.csv 생성

수정: XML에서 Address/Station/McpZone 모두 파싱하여 HID 매핑 직접 생성
     (외부 layout_HID_Zone_MCP_Mapping.csv 불필요)

사용법:
    1. 모듈로 import:
       from hid_zone_csv_cre import create_hid_zone_csv
       create_hid_zone_csv(xml_path, output_csv_path)

    2. 직접 실행:
       python hid_zone_csv_cre.py [layout.xml 또는 layout.zip 경로] [출력 CSV 경로]
"""

import os
import re
import csv
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


def safe_int(value, default=0):
    """안전하게 정수로 변환 (실패 시 기본값 반환)"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def parse_addresses_and_stations_iterparse(xml_file: str) -> Tuple[Dict, Dict]:
    """
    XML 파일에서 Address와 Station 정보 추출 (iterparse 사용 - 메모리 효율)

    Returns:
        addresses: {addr_no: {'x': float, 'y': float, 'stations': [stn_no, ...]}}
        stations: {stn_no: {'port_id': str, 'type': int}}
    """
    print("Address/Station 파싱 중 (iterparse)...")

    addresses = {}
    stations = {}

    current_addr = None
    current_addr_no = None
    current_station = None
    current_station_no = None
    current_next_addr = None
    addr_data = {}
    station_data = {}
    count = 0

    context = ET.iterparse(xml_file, events=('start', 'end'))

    for event, elem in context:
        if event == 'start':
            if elem.tag == 'group':
                name = elem.get('name', '')
                cls = elem.get('class', '')

                # Address 그룹 시작 (NextAddr 제외)
                if 'Addr' in name and 'address.Addr' in cls and 'NextAddr' not in name:
                    match = re.search(r'Addr(\d+)', name)
                    if match:
                        current_addr_no = int(match.group(1))
                        addr_data = {'x': 0, 'y': 0, 'stations': []}
                        current_addr = True

                # Station 그룹 시작
                elif 'Station' in name and 'Station' in cls:
                    match = re.search(r'Station(\d+)', name)
                    if match:
                        current_station_no = int(match.group(1))
                        station_data = {'port_id': '', 'type': 0}
                        current_station = True

                # NextAddr 그룹 시작
                elif 'NextAddr' in name and 'NextAddr' in cls:
                    current_next_addr = True

        elif event == 'end':
            if elem.tag == 'param':
                key = elem.get('key', '')
                value = elem.get('value', '')

                if current_addr and not current_station and not current_next_addr:
                    if key == 'draw-x':
                        try:
                            addr_data['x'] = float(value)
                        except:
                            pass
                    elif key == 'draw-y':
                        try:
                            addr_data['y'] = float(value)
                        except:
                            pass

                elif current_station:
                    if key == 'port-id':
                        station_data['port_id'] = value
                    elif key == 'type':
                        station_data['type'] = safe_int(value, 0)

            elif elem.tag == 'group':
                name = elem.get('name', '')

                if current_next_addr and 'NextAddr' in name:
                    current_next_addr = False

                elif current_station and 'Station' in name:
                    if current_station_no:
                        stations[current_station_no] = station_data.copy()
                        if current_addr_no:
                            addr_data['stations'].append(current_station_no)
                    current_station = False
                    current_station_no = None

                elif current_addr and 'Addr' in name and 'NextAddr' not in name:
                    if current_addr_no is not None:
                        addresses[current_addr_no] = addr_data.copy()
                        count += 1
                        if count % 5000 == 0:
                            print(f"  {count:,} addresses...")
                    current_addr = False
                    current_addr_no = None

            elem.clear()

    print(f"  완료: {len(addresses):,} addresses, {len(stations):,} stations")
    return addresses, stations


def parse_mcp_zones_from_content(xml_content: str) -> Dict:
    """
    XML 내용에서 McpZone 정보 추출 (라인 단위 파싱)

    Returns:
        mcp_zones: {zone_id: {
            'mcp_id': int,
            'zone_id': int,
            'vehicle_max': int,
            'vehicle_precaution': int,
            'type': int,
            'entries': [(start, end), ...],
            'exits': [(start, end), ...],
            'zcu': str
        }}
    """
    print("McpZone 파싱 중...")

    mcp_zones = {}

    current_mcp_zone = None
    current_mcp_id = None
    current_zone_id = None
    current_zone_params = {}
    current_entries = []
    current_exits = []
    current_zcu = ''

    in_entry = False
    in_exit = False
    entry_start = None
    entry_end = None
    entry_zcu = ''
    exit_start = None
    exit_end = None
    zone_depth = 0

    lines = xml_content.split('\n')
    total = len(lines)

    for i, line in enumerate(lines):
        if i % 500000 == 0 and i > 0:
            print(f"  {i:,}/{total:,} ({i*100//total}%)")

        line = line.strip()

        # McpZone 그룹 시작
        if '<group name="McpZone' in line and 'mcpzone.McpZone"' in line:
            # 이전 zone 저장
            if current_mcp_zone is not None and current_zone_id is not None:
                mcp_zones[current_zone_id] = {
                    'mcp_id': current_mcp_id,
                    'zone_id': current_zone_id,
                    'vehicle_max': safe_int(current_zone_params.get('vehicle-max', 0)),
                    'vehicle_precaution': safe_int(current_zone_params.get('vehicle-precaution', 0)),
                    'type': safe_int(current_zone_params.get('type', 0)),
                    'entries': current_entries.copy(),
                    'exits': current_exits.copy(),
                    'zcu': current_zcu
                }

            # McpZone name에서 숫자 추출
            match = re.search(r'McpZone(\d+)', line)
            current_mcp_id = int(match.group(1)) if match else None

            current_mcp_zone = True
            current_zone_id = None
            current_zone_params = {}
            current_entries = []
            current_exits = []
            current_zcu = ''
            in_entry = False
            in_exit = False
            zone_depth = 1
            continue

        if current_mcp_zone is None:
            continue

        # Entry 그룹 시작
        if '<group name="Entry' in line and 'mcpzone.Entry"' in line:
            in_entry = True
            entry_start = None
            entry_end = None
            entry_zcu = ''
            zone_depth += 1
            continue

        # Exit 그룹 시작
        if '<group name="Exit' in line and 'mcpzone.Exit"' in line:
            in_exit = True
            exit_start = None
            exit_end = None
            zone_depth += 1
            continue

        # CutLane 그룹 시작
        if '<group name="CutLane' in line and 'mcpzone.CutLane"' in line:
            zone_depth += 1
            continue

        # 다른 그룹 시작
        if '<group ' in line and '>' in line and '/>' not in line:
            zone_depth += 1
            continue

        # 파라미터 파싱
        if '<param ' in line and 'key="' in line and 'value="' in line:
            key_match = re.search(r'key="([^"]+)"', line)
            value_match = re.search(r'value="([^"]*)"', line)

            if key_match and value_match:
                key = key_match.group(1)
                value = value_match.group(1)

                if in_entry:
                    if key == 'start':
                        entry_start = safe_int(value)
                    elif key == 'end':
                        entry_end = safe_int(value)
                    elif key == 'stop-zcu':
                        entry_zcu = value

                elif in_exit:
                    if key == 'start':
                        exit_start = safe_int(value)
                    elif key == 'end':
                        exit_end = safe_int(value)

                elif zone_depth == 1:  # McpZone 직속 파라미터
                    if key == 'id':
                        current_zone_id = safe_int(value)
                    else:
                        current_zone_params[key] = value

        # 그룹 종료
        if '</group>' in line:
            if in_entry:
                if entry_start is not None and entry_end is not None:
                    current_entries.append((entry_start, entry_end))
                if entry_zcu and not current_zcu:
                    current_zcu = entry_zcu
                in_entry = False
                zone_depth -= 1
                continue

            elif in_exit:
                if exit_start is not None and exit_end is not None:
                    current_exits.append((exit_start, exit_end))
                in_exit = False
                zone_depth -= 1
                continue

            zone_depth -= 1

            # McpZone 그룹 종료
            if zone_depth == 0:
                if current_zone_id is not None:
                    mcp_zones[current_zone_id] = {
                        'mcp_id': current_mcp_id,
                        'zone_id': current_zone_id,
                        'vehicle_max': safe_int(current_zone_params.get('vehicle-max', 0)),
                        'vehicle_precaution': safe_int(current_zone_params.get('vehicle-precaution', 0)),
                        'type': safe_int(current_zone_params.get('type', 0)),
                        'entries': current_entries.copy(),
                        'exits': current_exits.copy(),
                        'zcu': current_zcu
                    }
                current_mcp_zone = None

    print(f"  완료: {len(mcp_zones):,} zones")
    return mcp_zones


def build_addr_to_zone_mapping(mcp_zones: Dict) -> Dict:
    """
    McpZone Entry/Exit 주소로 Address → Zone 매핑 생성

    Returns:
        {addr_no: {'zone_id': int, 'mcp_id': int}}
    """
    print("Address ↔ Zone 매핑 생성 중...")

    addr_to_zone = {}

    for zone_id, zone_data in mcp_zones.items():
        mcp_id = zone_data.get('mcp_id', '')

        for entry_start, entry_end in zone_data.get('entries', []):
            addr_to_zone[entry_start] = {'zone_id': zone_id, 'mcp_id': mcp_id}
            addr_to_zone[entry_end] = {'zone_id': zone_id, 'mcp_id': mcp_id}

        for exit_start, exit_end in zone_data.get('exits', []):
            addr_to_zone[exit_start] = {'zone_id': zone_id, 'mcp_id': mcp_id}
            addr_to_zone[exit_end] = {'zone_id': zone_id, 'mcp_id': mcp_id}

    print(f"  {len(addr_to_zone):,}개 주소 매핑됨")
    return addr_to_zone


def build_zone_hid_mapping(addresses: Dict, stations: Dict, addr_to_zone: Dict) -> Dict:
    """
    Zone별 HID 매핑 생성

    Returns:
        {zone_id: [{'HID_ID': str, 'Addr_No': str, 'Station_No': str}, ...]}
    """
    print("Zone ↔ HID 매핑 생성 중...")

    zone_hid_map = {}

    for addr_no, addr_data in addresses.items():
        # 이 주소가 어느 Zone에 속하는지 확인
        if addr_no not in addr_to_zone:
            continue

        zone_info = addr_to_zone[addr_no]
        zone_id = zone_info['zone_id']

        # 이 주소에 연결된 Station들의 HID 추출
        for stn_no in addr_data.get('stations', []):
            if stn_no not in stations:
                continue

            hid_id = stations[stn_no].get('port_id', '')
            if not hid_id:
                continue

            if zone_id not in zone_hid_map:
                zone_hid_map[zone_id] = []

            zone_hid_map[zone_id].append({
                'HID_ID': hid_id,
                'Addr_No': str(addr_no),
                'Station_No': str(stn_no)
            })

    total_hids = sum(len(v) for v in zone_hid_map.values())
    print(f"  {len(zone_hid_map):,}개 Zone, {total_hids:,}개 HID 매핑됨")
    return zone_hid_map


def derive_bay_zone(zone_no: int) -> str:
    """Zone 번호에서 Bay_Zone 추정"""
    bay_mapping = {
        1: 'B01', 2: 'B01', 3: 'B02', 4: 'B03', 5: 'B04',
        6: 'B05', 7: 'B05', 8: 'B06', 9: 'B06',
    }
    if zone_no in bay_mapping:
        return bay_mapping[zone_no]
    bay_num = (zone_no // 10) + 1
    return f'B{bay_num:02d}'


def generate_hid_zone_csv(mcp_zones: Dict, zone_hid_map: Dict, output_path: str,
                          project_name: str = "M14 Project Ph-1") -> None:
    """
    HID_ZONE_Master.csv 파일 생성

    Args:
        mcp_zones: McpZone 정보
        zone_hid_map: Zone별 HID 매핑
        output_path: 출력 CSV 파일 경로
        project_name: 프로젝트 이름
    """
    print(f"HID_ZONE_Master.csv 생성: {output_path}")

    headers = [
        'Zone_ID', 'HID_No', 'Bay_Zone', 'Sub_Region', 'Full_Name',
        'Territory', 'Type', 'IN_Count', 'OUT_Count', 'IN_Lanes', 'OUT_Lanes',
        'Vehicle_Max', 'Vehicle_Precaution', 'Project', 'ZCU', 'HID_Type',
        'HID_ID', 'Zone_ID2', 'Addr_No', 'Station_No'
    ]

    rows = []
    sorted_zone_ids = sorted(mcp_zones.keys())

    for zone_id in sorted_zone_ids:
        zone_data = mcp_zones[zone_id]
        zone_no = zone_id

        # Entry/Exit lanes
        entries = zone_data.get('entries', [])
        exits = zone_data.get('exits', [])

        in_lanes = '; '.join([f"{e[0]}→{e[1]}" for e in entries])
        out_lanes = '; '.join([f"{e[0]}→{e[1]}" for e in exits])

        # ZCU
        zcu = zone_data.get('zcu', '')

        # Bay_Zone 및 Sub_Region 추정
        bay_zone = derive_bay_zone(zone_no)
        sub_region = ((zone_no - 1) % 2) + 1

        # HID 타입
        hid_type_map = {1: 'HID4', 2: 'HID3', 3: 'HID2'}
        hid_type = hid_type_map.get(zone_data.get('type', 0), 'HID4')

        # HID 맵핑 정보
        hid_list = zone_hid_map.get(zone_id, [])

        if hid_list:
            # 맵핑된 HID가 있으면 각 HID별로 행 생성
            for hid_info in hid_list:
                row = {
                    'Zone_ID': zone_id,
                    'HID_No': f'HID-OHT-{zone_id:03d}',
                    'Bay_Zone': bay_zone,
                    'Sub_Region': sub_region,
                    'Full_Name': f'HID-{bay_zone}-{sub_region}({zone_id:03d})',
                    'Territory': 1,
                    'Type': 'HID',
                    'IN_Count': len(entries),
                    'OUT_Count': len(exits),
                    'IN_Lanes': in_lanes,
                    'OUT_Lanes': out_lanes,
                    'Vehicle_Max': zone_data.get('vehicle_max', 0),
                    'Vehicle_Precaution': zone_data.get('vehicle_precaution', 0),
                    'Project': project_name,
                    'ZCU': zcu,
                    'HID_Type': hid_type,
                    'HID_ID': hid_info['HID_ID'],
                    'Zone_ID2': zone_id,
                    'Addr_No': hid_info['Addr_No'],
                    'Station_No': hid_info['Station_No']
                }
                rows.append(row)
        else:
            # 맵핑된 HID가 없으면 빈 값으로 1행만 생성
            row = {
                'Zone_ID': zone_id,
                'HID_No': f'HID-OHT-{zone_id:03d}',
                'Bay_Zone': bay_zone,
                'Sub_Region': sub_region,
                'Full_Name': f'HID-{bay_zone}-{sub_region}({zone_id:03d})',
                'Territory': 1,
                'Type': 'HID',
                'IN_Count': len(entries),
                'OUT_Count': len(exits),
                'IN_Lanes': in_lanes,
                'OUT_Lanes': out_lanes,
                'Vehicle_Max': zone_data.get('vehicle_max', 0),
                'Vehicle_Precaution': zone_data.get('vehicle_precaution', 0),
                'Project': project_name,
                'ZCU': zcu,
                'HID_Type': hid_type,
                'HID_ID': '',
                'Zone_ID2': '',
                'Addr_No': '',
                'Station_No': ''
            }
            rows.append(row)

    # CSV 파일 작성 (UTF-8 BOM)
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  완료: {len(rows):,}개 행, {os.path.getsize(output_path):,} bytes")


def load_xml_content(source_path: str) -> Tuple[str, str]:
    """
    layout.xml 또는 layout.zip에서 XML 내용 로드

    Returns:
        (xml_content, xml_file_path)
    """
    if source_path.lower().endswith('.zip'):
        print(f"layout.zip에서 XML 추출 중: {source_path}")
        # 임시 파일로 추출
        import tempfile
        with zipfile.ZipFile(source_path, 'r') as zf:
            # ZIP 내 파일 목록 확인
            file_list = zf.namelist()
            print(f"  ZIP 내 파일 목록: {file_list}")

            # XML 파일 찾기 (layout.xml 또는 *.xml)
            xml_file = None
            for name in file_list:
                lower_name = name.lower()
                if lower_name == 'layout.xml':
                    xml_file = name
                    break
                elif lower_name.endswith('.xml'):
                    xml_file = name  # 첫 번째 XML 파일 사용

            if not xml_file:
                raise FileNotFoundError(f"ZIP 파일 내에 XML 파일이 없습니다: {file_list}")

            print(f"  사용할 XML 파일: {xml_file}")
            with zf.open(xml_file) as f:
                content = f.read().decode('utf-8')

            # iterparse용 임시 파일 생성
            temp_dir = tempfile.mkdtemp()
            temp_xml = os.path.join(temp_dir, 'layout.xml')
            with open(temp_xml, 'w', encoding='utf-8') as f:
                f.write(content)

            return content, temp_xml
    else:
        print(f"layout.xml 읽는 중: {source_path}")
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, source_path


def create_hid_zone_csv(xml_or_zip_path: str, output_csv_path: str,
                        project_name: str = "M14 Project Ph-1") -> None:
    """
    layout.xml 또는 layout.zip에서 HID_ZONE_Master.csv 생성

    Args:
        xml_or_zip_path: layout.xml 또는 layout.zip 파일 경로
        output_csv_path: 출력 CSV 파일 경로
        project_name: 프로젝트 이름
    """
    print("=" * 60)
    print("HID_ZONE_Master.csv 생성")
    print("=" * 60)
    print(f"입력: {xml_or_zip_path}")
    print(f"출력: {output_csv_path}")
    print()

    # XML 내용 로드
    xml_content, xml_file_path = load_xml_content(xml_or_zip_path)
    print(f"  XML 크기: {len(xml_content):,} bytes")
    print()

    # 1. Address/Station 파싱 (iterparse)
    addresses, stations = parse_addresses_and_stations_iterparse(xml_file_path)
    print()

    # 2. McpZone 파싱 (라인 단위)
    mcp_zones = parse_mcp_zones_from_content(xml_content)
    print()

    # 3. Address → Zone 매핑 (Entry/Exit 주소 기반)
    addr_to_zone = build_addr_to_zone_mapping(mcp_zones)
    print()

    # 4. Zone → HID 매핑 생성
    zone_hid_map = build_zone_hid_mapping(addresses, stations, addr_to_zone)
    print()

    # 5. CSV 생성
    generate_hid_zone_csv(mcp_zones, zone_hid_map, output_csv_path, project_name)

    print()
    print("완료!")
    print("=" * 60)


def get_fab_paths(script_dir, fab_name: str, layout_prefix: str = "A"):
    """
    FAB별 파일 경로를 반환

    실제 폴더 구조:
        MAP/{FAB}/{prefix}.layout.zip
        MAP/{FAB}/{prefix}.layout.xml
        MAP/{FAB}/{prefix}.station.dat
        MAP/{FAB}/HID_Zone_Master_{FAB}_{prefix}.csv

    Args:
        script_dir: 스크립트 디렉토리
        fab_name: FAB 이름 (예: "M14A", "M16A", "M16B")
        layout_prefix: 레이아웃 파일 접두사 (예: "A", "BR", "E", "B")

    Returns:
        dict: 각 파일 경로를 담은 딕셔너리
    """
    map_base_dir = script_dir / "MAP"
    fab_dir = map_base_dir / fab_name
    prefix = layout_prefix.upper()

    return {
        # MAP/{FAB}/ 경로 - 모든 파일이 여기에 있음
        "layout_zip": str(fab_dir / f"{prefix}.layout.zip"),
        "layout_xml": str(fab_dir / f"{prefix}.layout.xml"),
        "station_dat": str(fab_dir / f"{prefix}.station.dat"),
        # HID Zone 마스터 파일 (FAB별, 같은 폴더에 생성)
        "hid_zone_csv": str(fab_dir / f"HID_Zone_Master_{fab_name}_{prefix}.csv"),
    }


def create_hid_zone_csv_for_fab(script_dir, fab_name: str, layout_prefix: str = "A",
                                 project_name: str = None):
    """
    FAB별 HID_Zone_Master.csv 생성

    Args:
        script_dir: 스크립트 디렉토리
        fab_name: FAB 이름 (예: "M14", "M16")
        layout_prefix: 레이아웃 파일 접두사 (예: "A", "BR", "E")
        project_name: 프로젝트 이름 (None이면 FAB 기반 자동 생성)
    """
    paths = get_fab_paths(script_dir, fab_name, layout_prefix)

    # 프로젝트 이름 자동 생성
    if project_name is None:
        project_name = f"{fab_name} Project"

    # layout.xml 또는 layout.zip 찾기
    if os.path.exists(paths["layout_xml"]):
        input_path = paths["layout_xml"]
    elif os.path.exists(paths["layout_zip"]):
        input_path = paths["layout_zip"]
    else:
        raise FileNotFoundError(
            f"FAB {fab_name}의 레이아웃 파일을 찾을 수 없습니다: "
            f"{paths['layout_xml']} 또는 {paths['layout_zip']}"
        )

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(paths["hid_zone_csv"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    create_hid_zone_csv(input_path, paths["hid_zone_csv"], project_name)


def main():
    """
    커맨드라인 실행용 메인 함수

    사용법:
        python hid_zone_csv_cre.py [layout.xml 또는 layout.zip 경로] [출력 CSV 경로]
        python hid_zone_csv_cre.py --fab M14 --layout A
    """
    import sys
    import pathlib

    # 기본 경로
    script_dir = pathlib.Path(__file__).parent.resolve()

    # FAB 모드 확인
    if "--fab" in sys.argv:
        fab_idx = sys.argv.index("--fab")
        fab_name = sys.argv[fab_idx + 1] if fab_idx + 1 < len(sys.argv) else "M14"

        layout_prefix = "A"
        if "--layout" in sys.argv:
            layout_idx = sys.argv.index("--layout")
            layout_prefix = sys.argv[layout_idx + 1] if layout_idx + 1 < len(sys.argv) else "A"

        project_name = None
        if "--project" in sys.argv:
            project_idx = sys.argv.index("--project")
            project_name = sys.argv[project_idx + 1] if project_idx + 1 < len(sys.argv) else None

        print("=" * 60)
        print(f"hid_zone_csv_cre.py - FAB 모드: {fab_name}, 레이아웃: {layout_prefix}")
        print("=" * 60)

        create_hid_zone_csv_for_fab(script_dir, fab_name, layout_prefix, project_name)
        return

    # 기존 모드 (직접 경로 지정)
    # layout.xml 또는 layout.zip 찾기
    xml_path = script_dir / 'layout' / 'layout.xml'
    zip_path = script_dir / 'layout' / 'layout' / 'layout.zip'

    if xml_path.exists():
        default_input = str(xml_path)
    elif zip_path.exists():
        default_input = str(zip_path)
    else:
        default_input = str(zip_path)

    default_output = str(script_dir / 'HID_ZONE_Master.csv')

    # 명령행 인자 처리
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output

    if not os.path.exists(input_path):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    create_hid_zone_csv(input_path, output_path)


if __name__ == '__main__':
    main()