#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_5min_data.py - OHT 이동 데이터 TXT 생성

MAP/M14A/A.layout/layout.xml 에서 레이아웃을 읽어
5대 차량의 이동 데이터를 0.5초 간격으로 생성합니다.

사용법:
    python generate_5min_data.py       → 5분 데이터 생성
    python generate_5min_data.py 10    → 10분 데이터 생성

생성 후:
    python data_sender.py → 1번 선택하면 0.5초마다 재생
"""

import math
import os
import random
import re
import sys
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "oht_5v_data.txt")

# sim_server 규칙과 동일한 경로
FAB_NAME = "M14A"
LAYOUT_PREFIX = "A"
MAP_DIR = os.path.join(SCRIPT_DIR, "MAP")
LAYOUT_XML = os.path.join(MAP_DIR, FAB_NAME, f"{LAYOUT_PREFIX}.layout", "layout.xml")
LAYOUT_ZIP = os.path.join(MAP_DIR, FAB_NAME, f"{LAYOUT_PREFIX}.layout.zip")
MAP_ZIP = os.path.join(SCRIPT_DIR, "MAP.zip")

INTERVAL = 0.5

VEHICLES = [
    {"id": "V00795", "cur": 12340, "next": 12341, "dist": 14, "state": 1, "isFull": 1,
     "equipId": "4PDMV608", "src": "4ABL3301A_OUT04", "dst": "4ANZ19-701", "vel": 90},
    {"id": "V00564", "cur": 2115,  "next": 2116,  "dist": 9,  "state": 1, "isFull": 1,
     "equipId": "4PDD0270", "src": "4ANZ25-205",      "dst": "4KCW3301_3", "vel": 50},
    {"id": "V00975", "cur": 6033,  "next": 3294,  "dist": 0,  "state": 1, "isFull": 1,
     "equipId": "6PDB1402", "src": "4EPR5301_3",      "dst": "4ANZ03-302", "vel": 50},
    {"id": "V00313", "cur": 1071,  "next": 1072,  "dist": 0,  "state": 1, "isFull": 1,
     "equipId": "6PDN4064", "src": "4CSC1603_3",      "dst": "4AFZ47-328", "vel": 50},
    {"id": "V00649", "cur": 1448,  "next": 1449,  "dist": 0,  "state": 1, "isFull": 1,
     "equipId": "4NDNA076", "src": "4ALFE001_AO31",   "dst": "4AFZ15-160", "vel": 50},
]


def make_msg(v):
    return (
        f"2,OHT,{v['id']},{v['state']},{v['isFull']},0000,1,"
        f"{v['cur']},{v['dist']},{v['next']},"
        f"4,4,{v['equipId']},{v['dst']},00000000,0000,"
        f"{v['src']},{v['dst']},{v['vel']},0,0"
    )


def parse_layout_xml(xml_path):
    """
    layout.xml 파싱 - sim_server_3d_D5_D7.py 의 parse_layout_xml_direct 와 동일 로직

    <group name="Addr..."> 안에서:
      key="address"  → 노드 번호
      key="draw-x"   → X 좌표
      key="draw-y"   → Y 좌표
    <group name="NextAddr..."> 안에서:
      key="next-address" → 연결 노드
    """
    node_xy = {}
    connections = []

    current_addr = None
    current_addr_params = {}
    in_next_addr = False
    next_addr_params = {}

    print(f"  XML 파싱: {xml_path}")
    with open(xml_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            line_count += 1
            if line_count % 500000 == 0:
                print(f"    {line_count:,} 라인...")

            line = line.strip()

            # Addr 그룹 시작
            if '<group name="Addr' in line and 'class=' in line and 'address.Addr"' in line:
                if current_addr is not None and 'address' in current_addr_params:
                    addr_no = int(current_addr_params.get('address', 0))
                    if addr_no > 0:
                        x = float(current_addr_params.get('draw-x', 0))
                        y = float(current_addr_params.get('draw-y', 0))
                        node_xy[addr_no] = (x, y)
                current_addr_params = {}
                current_addr = line
                in_next_addr = False
                continue

            # NextAddr 그룹 시작
            if '<group name="NextAddr' in line and 'class=' in line and 'NextAddr"' in line:
                in_next_addr = True
                next_addr_params = {}
                continue

            # NextAddr 그룹 종료
            if in_next_addr and '</group>' in line:
                if 'address' in current_addr_params and 'next-address' in next_addr_params:
                    from_addr = int(current_addr_params.get('address', 0))
                    try:
                        to_addr = int(next_addr_params.get('next-address', '0'))
                        if from_addr > 0 and to_addr > 0:
                            connections.append((from_addr, to_addr))
                    except ValueError:
                        pass
                in_next_addr = False
                continue

            # 파라미터 파싱
            if '<param ' in line and 'key="' in line and 'value="' in line:
                key_match = re.search(r'key="([^"]+)"', line)
                value_match = re.search(r'value="([^"]*)"', line)
                if key_match and value_match:
                    key = key_match.group(1)
                    value = value_match.group(1)
                    if in_next_addr:
                        next_addr_params[key] = value
                    elif current_addr is not None:
                        current_addr_params[key] = value

    # 마지막 Addr 저장
    if current_addr is not None and 'address' in current_addr_params:
        addr_no = int(current_addr_params.get('address', 0))
        if addr_no > 0:
            x = float(current_addr_params.get('draw-x', 0))
            y = float(current_addr_params.get('draw-y', 0))
            node_xy[addr_no] = (x, y)

    # 인접맵 + 엣지 거리
    adj = {}
    edge_dist = {}
    for from_n, to_n in connections:
        adj.setdefault(from_n, []).append(to_n)
        if from_n in node_xy and to_n in node_xy:
            x1, y1 = node_xy[from_n]
            x2, y2 = node_xy[to_n]
            dist = int(math.sqrt((x2-x1)**2 + (y2-y1)**2) * 10)
            edge_dist[(from_n, to_n)] = max(dist, 10)

    print(f"  노드: {len(node_xy)}개  엣지: {len(edge_dist)}개")
    return node_xy, adj, edge_dist


def load_layout():
    """레이아웃 로드 - sim_server 규칙"""

    # 1) layout.xml 직접
    if os.path.exists(LAYOUT_XML):
        return parse_layout_xml(LAYOUT_XML)

    # 2) A.layout.zip 에서 추출
    if os.path.exists(LAYOUT_ZIP):
        print(f"  A.layout.zip 추출 중...")
        extract_dir = os.path.join(MAP_DIR, FAB_NAME, f"{LAYOUT_PREFIX}.layout")
        with zipfile.ZipFile(LAYOUT_ZIP, 'r') as z:
            z.extractall(extract_dir)
        xml_in_zip = os.path.join(extract_dir, "layout", "layout.xml")
        if os.path.exists(xml_in_zip):
            return parse_layout_xml(xml_in_zip)
        if os.path.exists(LAYOUT_XML):
            return parse_layout_xml(LAYOUT_XML)

    # 3) MAP.zip 풀기
    if os.path.exists(MAP_ZIP):
        print(f"  MAP.zip 추출 중...")
        with zipfile.ZipFile(MAP_ZIP, 'r') as z:
            z.extractall(MAP_DIR)
        if os.path.exists(LAYOUT_XML):
            return parse_layout_xml(LAYOUT_XML)

    print("오류: 레이아웃 파일 없음")
    sys.exit(1)


def main():
    duration_min = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    duration_sec = duration_min * 60
    total_ticks = int(duration_sec / INTERVAL)

    print("=" * 50)
    print(f"OHT 데이터 생성: {duration_min}분 ({total_ticks}틱, {INTERVAL}초 간격)")
    print("=" * 50)

    node_xy, adj, edge_dist = load_layout()

    # 차량 초기화
    vehicles = []
    for v in VEHICLES:
        vv = dict(v)
        if vv["cur"] not in node_xy:
            vv["cur"] = random.choice(list(node_xy.keys()))
            print(f"  [!] {vv['id']}: 시작노드 변경 → {vv['cur']}")
        neighbors = adj.get(vv["cur"], [])
        if vv["next"] not in node_xy and neighbors:
            vv["next"] = neighbors[0]
        vv["edge_max"] = edge_dist.get((vv["cur"], vv["next"]), 100)
        vehicles.append(vv)
        print(f"  {vv['id']}: {vv['cur']}→{vv['next']} edge={vv['edge_max']}")

    # 데이터 생성
    print(f"\n생성 중...")
    batches = []
    for tick in range(total_ticks):
        batch = []
        for v in vehicles:
            v["dist"] += 50
            if v["dist"] >= v["edge_max"]:
                v["dist"] = 0
                v["cur"] = v["next"]
                neighbors = adj.get(v["cur"], [])
                if neighbors:
                    v["next"] = random.choice(neighbors)
                    v["edge_max"] = edge_dist.get((v["cur"], v["next"]), 100)
            batch.append(make_msg(v))
        batches.append("\n".join(batch))
        if tick % 100 == 0:
            print(f"  {tick}/{total_ticks}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(batches) + "\n")

    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\n완료: {OUTPUT_FILE}")
    print(f"  {len(vehicles)}대 × {total_ticks}틱 ({size_kb:.0f} KB)")
    print(f"\n→ python data_sender.py → 1번 (0.5초마다 재생)")


if __name__ == "__main__":
    main()
