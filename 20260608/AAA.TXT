#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_near_hid4.py - 리프터 -> 근처 HID4 구간 매핑 CSV 자동 생성

각 리프터에 '경계(lane)가 가장 가까운 HID4 구역(1~37)' 을 찾아 매핑.
HID4 는 HID_INOUT 로그에 기록되는 구역이라, 이 매핑으로 HID_INOUT 만으로
리프터별 차량수를 셀 수 있다.

입력: BR.layout.zip(또는 .xml), BR.station.dat, HID_Zone_Master_M16A_BR.csv
출력: 리프터_근처HID4.csv  (Lifter, FAB, 근처HID4)

사용법:
  python gen_near_hid4.py BR.layout.zip BR.station.dat HID_Zone_Master_M16A_BR.csv 리프터_근처HID4.csv

※ make_map.py 와 같은 폴더에서 실행 (parse_layout/parse_lifters 재사용).
"""
import sys, os, csv, re, math, zipfile
from collections import defaultdict

# make_map.py 의 파서 재사용
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_map import load_xml, parse_layout, parse_lifters


def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    layout, station, hid_master = sys.argv[1:4]
    out = sys.argv[4] if len(sys.argv) > 4 else "리프터_근처HID4.csv"

    nodes, _ = parse_layout(load_xml(layout))
    lift = parse_lifters(station)
    lpts = defaultdict(list)
    for a, p in lift.items():
        if a in nodes:
            lpts[p.split("_")[0]].append(nodes[a])

    # HID4(1~37) lane 좌표
    hid4 = defaultdict(list)
    with open(hid_master, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            zid = r["Zone_ID"].strip()
            if not (zid.isdigit() and 1 <= int(zid) <= 37):
                continue
            for fld in ("IN_Lanes", "OUT_Lanes"):
                for seg in (r.get(fld) or "").split(";"):
                    m = re.match(r'\s*(\d+)\s*→\s*(\d+)', seg)
                    if m:
                        for a in (int(m.group(1)), int(m.group(2))):
                            if a in nodes:
                                hid4[zid].append(nodes[a])

    # 리프터 -> 최근접 HID4 (경계 lane점)
    rows = []
    for lf in sorted(lpts):
        best, bd = None, 1e18
        for z, pts in hid4.items():
            for px, py in pts:
                for lx, ly in lpts[lf]:
                    d = (lx - px) ** 2 + (ly - py) ** 2
                    if d < bd:
                        bd, best = d, z
        rows.append((lf, "M16" if lf[0] == "6" else "M14", best, round(math.sqrt(bd))))

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Lifter", "FAB", "근처HID4"])
        for lf, fab, z, d in rows:
            w.writerow([lf, fab, z])

    print(f"리프터 {len(rows)}기 -> 근처 HID4 매핑")
    for lf, fab, z, d in rows:
        print(f"  {lf:10} -> HID{z:3} (경계 {d}mm)")
    print(f"\n저장: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
