#!/usr/bin/env python3
"""
HID Zone별 IN/OUT Lane 개수 추출 스크립트 (v2 - 중복제거)
"""

import re
import csv

def extract_hid_zones(config_file):
    """mcp75.cfg에서 HID Zone 정보 추출 (중복 제거)"""

    with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    zones = {}  # 딕셔너리로 중복 방지

    # <ZONE>:숫자:숫자 패턴으로 Zone 시작점 찾기
    zone_starts = list(re.finditer(r'<ZONE>:(\d+):(\d+);', content))

    for i, match in enumerate(zone_starts):
        zone_id = int(match.group(1))
        territory = int(match.group(2))

        # 고유 키
        key = (zone_id, territory)
        if key in zones:
            continue  # 중복 건너뛰기

        # 블록 추출
        start_pos = match.end()
        if i + 1 < len(zone_starts):
            end_pos = zone_starts[i + 1].start()
        else:
            next_section = re.search(r'\n<[A-Z_]+>', content[start_pos:])
            if next_section:
                end_pos = start_pos + next_section.start()
            else:
                end_pos = len(content)

        block = content[start_pos:end_pos]

        # TYPE 확인
        type_match = re.search(r'TYPE\s*=\s*"(\w+)"', block)
        zone_type = type_match.group(1) if type_match else "UNKNOWN"

        # HID Zone만 처리
        if zone_type != "HID":
            continue

        # LOOP_ENTRY (IN) 개수
        in_entries = re.findall(r'LOOP_ENTRY\s*=\s*(\d+)\s*,\s*(\d+)', block)
        in_count = len(in_entries)

        # EXIT (OUT) 개수
        out_exits = re.findall(r'EXIT\s*=\s*(\d+)\s*,\s*(\d+)', block)
        out_count = len(out_exits)

        # VEHICLE_MAX
        vhl_max_match = re.search(r'VEHICLE_MAX\s*=\s*(\d+)', block)
        vhl_max = int(vhl_max_match.group(1)) if vhl_max_match else 0

        # VEHICLE_PRECAUTION
        vhl_precaution_match = re.search(r'VEHICLE_PRECAUTION\s*=\s*(\d+)', block)
        vhl_precaution = int(vhl_precaution_match.group(1)) if vhl_precaution_match else 0

        # IN/OUT Lane 상세
        in_lanes = [f"{e[0]}→{e[1]}" for e in in_entries]
        out_lanes = [f"{e[0]}→{e[1]}" for e in out_exits]

        zones[key] = {
            'zone_id': zone_id,
            'territory': territory,
            'type': zone_type,
            'in_count': in_count,
            'out_count': out_count,
            'in_lanes': '; '.join(in_lanes),
            'out_lanes': '; '.join(out_lanes),
            'vehicle_max': vhl_max,
            'vehicle_precaution': vhl_precaution
        }

    return list(zones.values())


def save_to_csv(zones, output_file):
    """CSV 파일로 저장"""

    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        # 헤더
        writer.writerow([
            'Zone_ID',
            'Territory',
            'Type',
            'IN_Count',
            'OUT_Count',
            'IN_Lanes',
            'OUT_Lanes',
            'Vehicle_Max',
            'Vehicle_Precaution'
        ])

        # 데이터 (Zone ID 순서로 정렬)
        for zone in sorted(zones, key=lambda x: (x['zone_id'], x['territory'])):
            writer.writerow([
                zone['zone_id'],
                zone['territory'],
                zone['type'],
                zone['in_count'],
                zone['out_count'],
                zone['in_lanes'],
                zone['out_lanes'],
                zone['vehicle_max'],
                zone['vehicle_precaution']
            ])

    print(f"✅ CSV 파일 저장 완료: {output_file}")


def print_summary(zones):
    """요약 출력"""

    print("\n" + "="*70)
    print("📊 HID Zone IN/OUT 분석 결과")
    print("="*70)
    print(f"총 HID Zone 개수: {len(zones)}개\n")

    print(f"{'Zone ID':<10} {'Territory':<10} {'IN개수':<10} {'OUT개수':<10} {'최대OHT':<10}")
    print("-"*50)

    total_in = 0
    total_out = 0

    for zone in sorted(zones, key=lambda x: (x['zone_id'], x['territory'])):
        print(f"{zone['zone_id']:<10} {zone['territory']:<10} {zone['in_count']:<10} {zone['out_count']:<10} {zone['vehicle_max']:<10}")
        total_in += zone['in_count']
        total_out += zone['out_count']

    print("-"*50)
    print(f"{'합계':<20} {total_in:<10} {total_out:<10}")
    print("="*70)


if __name__ == "__main__":
    config_file = "/home/user/ASAS/OHT2/mcp75.cfg"
    output_csv = "/home/user/ASAS/OHT2/HID_Zone_IN_OUT.csv"

    print("🔍 HID Zone 데이터 추출 시작...")

    zones = extract_hid_zones(config_file)

    if zones:
        save_to_csv(zones, output_csv)
        print_summary(zones)
    else:
        print("❌ HID Zone을 찾을 수 없습니다.")