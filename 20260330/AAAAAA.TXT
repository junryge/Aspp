#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_oht_parser.py - 실제 OHT 메시지 파싱 + layout.xml 좌표 변환

실시간으로 들어오는 OHT VHL_STATE_REPORT 메시지를 파싱하고,
layout.xml의 Address 노드맵을 이용하여 x,y 좌표로 변환한 뒤
CSV 파일로 저장한다.

사용법:
    1. 단독 실행 (샘플 데이터 테스트):
       python real_oht_parser.py

    2. import해서 사용:
       from real_oht_parser import RealOHTParser
       parser = RealOHTParser("MAP/M14A/A.layout.zip")
       results = parser.parse_messages(["2,OHT,V00795,..."])
       parser.save_csv(results, "output/real_oht_data.csv")
"""

import os
import sys
import math
import csv
import json
import zipfile
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# layout_map_cre.py에서 XML 파싱 함수 재사용
from layout_map_cre import parse_layout_xml_content

# ============================================================
# Enum 매핑 (sim_server import 없이 독립적으로 정의)
# ============================================================
VHL_STATE_MAP = {
    "1": "RUN",           # 운전 중
    "2": "STOP",          # 정지 중
    "3": "ABNORMAL",      # 상태 이상
    "4": "MANUAL",        # 수동 조치
    "5": "REMOVING",      # 분리 및 제거 중
    "6": "OBS_BZ_STOP",   # OBS-STOP/BZ-STOP
    "7": "JAM",           # 정체
    "8": "HT_STOP",       # HT-STOP
    "9": "E84_TIMEOUT",   # E84 time out
}

RUN_CYCLE_MAP = {
    "0": "NONE",              # 사이클 없음
    "1": "POSITION_DETECT",   # 위치 확인
    "2": "MOVING",            # 이동 중
    "3": "ACQUIRE",           # 물품 집어올리기
    "4": "DEPOSIT",           # 물품 내려놓기
    "5": "SAMPLING",          # 샘플링
    "9": "FLOOR_TRANS",       # 층간 이동
    "21": "WHEELDRIVE",       # 바퀴 주행
    "22": "MANUAL_CONTROL",   # 수동 조작
    "23": "DRIVE_TEACHING",   # 주행 학습
    "24": "TRANS_TEACHING",   # 이재부 학습
    "2E": "BUILDING_TRANS",   # 동 간 이동
    "2F": "EVACUATION",       # 대피 이동
}

VHL_CYCLE_MAP = {
    "0": "NONE",           # 실행 사이클 없음
    "1": "MOVING",         # 이동 중
    "2": "ACQUIRE_MOVING", # 구원 이동
    "3": "ACQUIRING",      # 구원 이송 중
    "4": "DEPOSIT_MOVING", # 도매 이동
    "5": "DEPOSITING",     # 도매 이송 중
    "6": "MAINT_MOVING",   # 유지 이동 중
    "7": "WAITING",        # 대기
    "8": "INPUT",          # 투입 중
}

VHL_DET_STATE_MAP = {
    "0": "NONE",
    "1": "WAIT",
    "2": "STAGE_WAIT",
    "3": "STANDBY_WAIT",
    "4": "DEPOSIT_SIG_WAIT",
    "5": "ACQ_WAIT",
    "6": "MAP_WAIT",
    "101": "MOVING",
    "102": "PARKING_UTS_MOVING",
    "103": "STAGE_MOVING",
    "104": "STANDBY_MOVING",
    "105": "BALANCE_MOVING",
    "106": "PARKING_MOVING",
}

# 거리 단위: UDP 메시지는 100mm 단위
UDP_DISTANCE_UNIT = 100  # 1 = 100mm


class RealOHTParser:
    """실제 OHT 데이터 파서 + 좌표 변환기"""

    def __init__(self, layout_source: str):
        """
        Args:
            layout_source: layout.zip 경로 또는 layout.xml 경로
        """
        self.nodes: Dict[int, dict] = {}       # {addr_no: {'no', 'x', 'y'}}
        self.connections: List[List] = []       # [[from, to], ...]
        self.edge_dist: Dict[Tuple[int, int], float] = {}  # {(from, to): distance}

        self._load_layout(layout_source)

    def _load_layout(self, layout_source: str):
        """layout.xml 로딩 (ZIP 또는 XML 직접)"""
        if not os.path.exists(layout_source):
            print(f"[오류] 파일을 찾을 수 없습니다: {layout_source}")
            return

        xml_content = None

        if layout_source.lower().endswith('.zip'):
            print(f"ZIP에서 layout.xml 추출: {layout_source}")
            with zipfile.ZipFile(layout_source, 'r') as zf:
                file_list = zf.namelist()
                xml_file = None
                for name in file_list:
                    if name.lower() == 'layout/layout.xml':
                        xml_file = name
                        break
                    elif name.lower() == 'layout.xml' and xml_file is None:
                        xml_file = name

                if not xml_file:
                    print(f"[오류] ZIP 내 layout.xml을 찾을 수 없습니다")
                    return

                with zf.open(xml_file) as f:
                    xml_content = f.read().decode('utf-8')
        else:
            print(f"layout.xml 읽기: {layout_source}")
            with open(layout_source, 'r', encoding='utf-8') as f:
                xml_content = f.read()

        # parse_layout_xml_content는 list와 list 반환
        nodes_list, self.connections = parse_layout_xml_content(xml_content)

        # 리스트 → 딕셔너리 변환 {addr_no: {no, x, y}}
        for node in nodes_list:
            self.nodes[node['no']] = node

        # edge 거리 계산 (보간용)
        for conn in self.connections:
            from_no, to_no = conn[0], conn[1]
            if from_no in self.nodes and to_no in self.nodes:
                n1 = self.nodes[from_no]
                n2 = self.nodes[to_no]
                dist = math.sqrt((n2['x'] - n1['x'])**2 + (n2['y'] - n1['y'])**2)
                self.edge_dist[(from_no, to_no)] = dist

        print(f"노드맵 로딩 완료: {len(self.nodes)}개 노드, {len(self.edge_dist)}개 엣지")

    def parse_oht_message(self, line: str) -> Optional[dict]:
        """
        OHT VHL_STATE_REPORT 메시지 1건 파싱

        메시지 포맷 (23개 필드, 콤마 구분):
        [0]MessageId [1]McpName [2]VehicleId [3]State [4]IsFull [5]ErrorCode [6]IsOnline
        [7]CurrentAddress [8]Distance(100mm) [9]NextAddress [10]RunCycle [11]VhlCycle
        [12]CarrierId [13]Destination [14]EMState [15]GroupId [16]SourcePort [17]DestPort
        [18]Priority [19]DetailState [20]RunDistance

        Returns:
            파싱된 딕셔너리 또는 None (파싱 실패 시)
        """
        line = line.strip()
        if not line:
            return None

        fields = line.split(',')
        if len(fields) < 19:
            print(f"[경고] 필드 부족 ({len(fields)}개): {line[:50]}...")
            return None

        try:
            state_code = fields[3]
            run_cycle_code = fields[10]
            vhl_cycle_code = fields[11]
            detail_state_code = fields[19] if len(fields) > 19 else "0"

            return {
                'messageId': fields[0],
                'mcpName': fields[1],
                'vehicleId': fields[2],
                'state': state_code,
                'stateName': VHL_STATE_MAP.get(state_code, f"UNKNOWN({state_code})"),
                'isFull': int(fields[4]),
                'errorCode': fields[5],
                'isOnline': int(fields[6]),
                'currentAddress': int(fields[7]),
                'distance_100mm': int(fields[8]),
                'distance_mm': int(fields[8]) * UDP_DISTANCE_UNIT,
                'nextAddress': int(fields[9]),
                'runCycle': run_cycle_code,
                'runCycleName': RUN_CYCLE_MAP.get(run_cycle_code, f"UNKNOWN({run_cycle_code})"),
                'vhlCycle': vhl_cycle_code,
                'vhlCycleName': VHL_CYCLE_MAP.get(vhl_cycle_code, f"UNKNOWN({vhl_cycle_code})"),
                'carrierId': fields[12],
                'destination': int(fields[13]) if fields[13].isdigit() else 0,
                'emState': fields[14] if len(fields) > 14 else "00000000",
                'groupId': fields[15] if len(fields) > 15 else "0000",
                'sourcePort': fields[16] if len(fields) > 16 else "",
                'destPort': fields[17] if len(fields) > 17 else "",
                'priority': int(fields[18]) if len(fields) > 18 else 50,
                'detailState': detail_state_code,
                'detailStateName': VHL_DET_STATE_MAP.get(detail_state_code, f"UNKNOWN({detail_state_code})"),
                'runDistance': int(fields[20]) if len(fields) > 20 else 0,
            }
        except (ValueError, IndexError) as e:
            print(f"[오류] 메시지 파싱 실패: {e} - {line[:50]}...")
            return None

    def get_vehicle_position(self, parsed: dict) -> Tuple[float, float, bool]:
        """
        파싱된 메시지에서 x,y 좌표 계산 (CurrentAddress + NextAddress + Distance 보간)

        Returns:
            (x, y, found) - found=True면 노드맵에서 찾음
        """
        current_addr = parsed['currentAddress']
        next_addr = parsed['nextAddress']
        distance_mm = parsed['distance_mm']

        n1 = self.nodes.get(current_addr)
        if not n1:
            return 0.0, 0.0, False

        n2 = self.nodes.get(next_addr)
        if not n2 or current_addr == next_addr:
            # NextAddress가 없거나 동일하면 CurrentAddress 위치 그대로
            return n1['x'], n1['y'], True

        # 엣지 거리 (좌표 단위)
        edge_dist = self.edge_dist.get((current_addr, next_addr))
        if edge_dist is None:
            # 연결 정보가 없으면 직접 계산
            edge_dist = math.sqrt((n2['x'] - n1['x'])**2 + (n2['y'] - n1['y'])**2)

        if edge_dist <= 0:
            return n1['x'], n1['y'], True

        # 좌표 거리를 mm로 변환 (layout 좌표 = mm 단위 가정)
        # distance_mm / (edge_dist * 1000) 이 아니라
        # distance_100mm 값 * 100mm = distance_mm
        # edge_dist는 좌표 거리 (draw-x, draw-y 단위)
        # sim_server에서는: ratio = positionRatio (0~1), distance = edge_dist * ratio * 1000
        # 역산: ratio = distance_mm / (edge_dist * 1000)
        ratio = distance_mm / (edge_dist * 1000) if edge_dist > 0 else 0.0
        ratio = min(1.0, max(0.0, ratio))

        x = n1['x'] + (n2['x'] - n1['x']) * ratio
        y = n1['y'] + (n2['y'] - n1['y']) * ratio

        return round(x, 2), round(y, 2), True

    def process_message(self, line: str) -> Optional[dict]:
        """
        메시지 1건을 파싱 + 좌표 변환하여 완성된 결과 반환

        Returns:
            가공된 딕셔너리 (x, y 좌표 포함) 또는 None
        """
        parsed = self.parse_oht_message(line)
        if not parsed:
            return None

        x, y, found = self.get_vehicle_position(parsed)
        parsed['x'] = x
        parsed['y'] = y
        parsed['coordFound'] = found
        parsed['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        return parsed

    def parse_messages(self, lines: List[str]) -> List[dict]:
        """여러 메시지를 한꺼번에 파싱"""
        results = []
        for line in lines:
            result = self.process_message(line)
            if result:
                results.append(result)
        return results

    def to_map_json(self, results: List[dict]) -> dict:
        """
        기존 시뮬레이터 get_state() 포맷 호환 JSON 생성

        프론트엔드 MAP에서 바로 사용할 수 있는 형식
        """
        vehicles = []
        for r in results:
            vehicles.append({
                'vehicleId': r['vehicleId'],
                'x': r['x'],
                'y': r['y'],
                'state': int(r['state']),
                'stateName': r['stateName'],
                'isLoaded': r['isFull'],
                'velocity': 0,  # 실제 데이터에는 속도 없음
                'currentNode': r['currentAddress'],
                'nextNode': r['nextAddress'],
                'destination': r['destination'],
                'carrierId': r['carrierId'],
                'path': [],
                'runCycle': r['runCycle'],
                'runCycleName': r['runCycleName'],
                'vhlCycle': r['vhlCycle'],
                'vhlCycleName': r['vhlCycleName'],
                'distance': r['distance_mm'],
                'detailState': r['detailStateName'],
                'sourcePort': r['sourcePort'],
                'destPort': r['destPort'],
                'priority': r['priority'],
            })

        # 통계
        total = len(vehicles)
        running = sum(1 for v in vehicles if v['stateName'] == 'RUN')
        stopped = sum(1 for v in vehicles if v['stateName'] == 'STOP')
        jammed = sum(1 for v in vehicles if v['stateName'] == 'JAM')
        loaded = sum(1 for v in vehicles if v['isLoaded'] == 1)

        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'dataSource': 'REAL',
            'vehicles': vehicles,
            'stats': {
                'total': total,
                'running': running,
                'stopped': stopped,
                'jammed': jammed,
                'loaded': loaded,
                'empty': total - loaded,
            }
        }

    def save_csv(self, results: List[dict], output_path: str):
        """가공 결과를 CSV로 저장"""
        if not results:
            print("[경고] 저장할 데이터가 없습니다")
            return

        columns = [
            'timestamp', 'vehicleId', 'x', 'y', 'coordFound',
            'state', 'stateName', 'isFull', 'currentAddress', 'nextAddress',
            'distance_mm', 'carrierId', 'destination',
            'sourcePort', 'destPort',
            'runCycle', 'runCycleName', 'vhlCycle', 'vhlCycleName',
            'priority', 'detailState', 'detailStateName',
            'errorCode', 'emState',
        ]

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)

        print(f"CSV 저장 완료: {output_path} ({len(results)}건)")


# ============================================================
# 단독 실행 - 샘플 데이터 테스트
# ============================================================
def main():
    print("=" * 70)
    print("real_oht_parser.py - 실제 OHT 데이터 파싱 + 좌표 변환 테스트")
    print("=" * 70)

    # 샘플 데이터 (실제 메신저에서 받은 메시지)
    SAMPLE_MESSAGES = [
        "2,OHT,V00795,1,1,0000,1,12340,14,12341,4,4,4PDMV608,36073,00000000,0000,4ABL3301A_OUT04,4ANZ19-701,90,0,0",
        "2,OHT,V00564,6,1,0000,1,2115,9,2116,4,4,4PDD0270,7449,00000000,0000,4ANZ25-205,4KCW3301_3,50,0,0",
        "2,OHT,V00975,1,1,0000,1,6033,0,3294,4,4,6PDB1402,34128,00000000,0000,4EPR5301_3,4ANZ03-302,50,0,0",
        "2,OHT,V00313,1,1,0000,1,1071,0,1072,4,4,6PDN4064,21980,00000000,0000,4CSC1603_3,4AFZ47-328,50,0,0",
        "2,OHT,V00649,1,1,0000,1,1448,0,1449,4,4,4NDNA076,17160,00000000,0000,4ALFE001_AO31,4AFZ15-160,50,0,0",
    ]

    # layout 소스 찾기 (FAB별 ZIP 또는 레거시 경로)
    script_dir = pathlib.Path(__file__).parent.resolve()
    layout_source = None

    # 1순위: MAP/M14A/A.layout.zip (멀티FAB 구조)
    zip_path = script_dir / "MAP" / "M14A" / "A.layout.zip"
    if zip_path.exists():
        layout_source = str(zip_path)
    else:
        # 2순위: layout/layout.zip (레거시 구조)
        legacy_zip = script_dir / "layout" / "layout.zip"
        if legacy_zip.exists():
            layout_source = str(legacy_zip)
        else:
            # 3순위: layout/layout.xml
            xml_path = script_dir / "layout" / "layout.xml"
            if xml_path.exists():
                layout_source = str(xml_path)
            else:
                # 4순위: MAP 폴더 안에서 아무 ZIP 찾기
                map_dir = script_dir / "MAP"
                if map_dir.exists():
                    for fab_dir in sorted(map_dir.iterdir()):
                        if fab_dir.is_dir():
                            for f in fab_dir.iterdir():
                                if f.suffix.lower() == '.zip' and 'layout' in f.name.lower():
                                    layout_source = str(f)
                                    break
                        if layout_source:
                            break

    if not layout_source:
        print("[오류] layout.xml 또는 layout.zip을 찾을 수 없습니다")
        print("  확인할 경로:")
        print(f"    - {script_dir / 'MAP' / 'M14A' / 'A.layout.zip'}")
        print(f"    - {script_dir / 'layout' / 'layout.zip'}")
        sys.exit(1)

    print(f"\n레이아웃 소스: {layout_source}")
    print()

    # 파서 초기화 (XML 로딩)
    parser = RealOHTParser(layout_source)
    print()

    # 샘플 메시지 파싱
    print("-" * 70)
    print("샘플 메시지 파싱 결과:")
    print("-" * 70)

    results = []
    for msg in SAMPLE_MESSAGES:
        result = parser.process_message(msg)
        if result:
            results.append(result)
            found_mark = "O" if result['coordFound'] else "X"
            print(f"  [{found_mark}] {result['vehicleId']} | "
                  f"State={result['stateName']:12s} | "
                  f"Addr={result['currentAddress']:>5d} -> {result['nextAddress']:>5d} | "
                  f"Dist={result['distance_mm']:>5d}mm | "
                  f"({result['x']:>8.2f}, {result['y']:>8.2f}) | "
                  f"Carrier={result['carrierId']:10s} | "
                  f"{result['sourcePort']} -> {result['destPort']}")

    print()

    # CSV 저장
    output_dir = script_dir / "output"
    csv_path = str(output_dir / f"real_oht_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    parser.save_csv(results, csv_path)

    # JSON 출력 (MAP 호환 포맷)
    map_json = parser.to_map_json(results)
    json_path = str(output_dir / f"real_oht_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(map_json, f, indent=2, ensure_ascii=False)
    print(f"JSON 저장 완료: {json_path}")

    print()
    print(f"통계: 전체 {map_json['stats']['total']}대 | "
          f"RUN {map_json['stats']['running']} | "
          f"STOP {map_json['stats']['stopped']} | "
          f"JAM {map_json['stats']['jammed']} | "
          f"적재 {map_json['stats']['loaded']} | "
          f"빈차 {map_json['stats']['empty']}")

    # 노드맵 미매칭 확인
    not_found = [r for r in results if not r['coordFound']]
    if not_found:
        print(f"\n[주의] 노드맵에서 못 찾은 Address ({len(not_found)}건):")
        for r in not_found:
            print(f"  - {r['vehicleId']}: CurrentAddress={r['currentAddress']}")

    print()
    print("=" * 70)
    print("완료!")
    print("=" * 70)


if __name__ == '__main__':
    main()
