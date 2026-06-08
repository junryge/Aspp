#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_lifter_inout.py - 리프터 근처 HID 구간 차량수 (1분단위, 17기, HID_INOUT 파일만)

입력: LOGPRESSO_HID_INOUT_20260421.csv  (이 파일 하나만)
      리프터_근처HID4.csv                (리프터 -> 근처 HID4 구간 매핑)

규칙: 1분 단위 / 리프터별 / 차량(VHL_ID) 중복제거
      각 리프터의 '근처 HID4 구간'에 그 분(分)에 진입(TO_HIDID)한 차량 수.
      ※ 같은 HID4 구역에 속한 리프터는 같은 값 (구역 단위).

사용법:
  python count_lifter_inout.py LOGPRESSO_HID_INOUT_20260421.csv 리프터_근처HID4.csv [출력.csv]
  python count_lifter_inout.py ... 리프터_근처HID4.csv --at "2026-04-21 14:04"
"""
import sys, os, csv
from collections import defaultdict


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    inout, map_csv = sys.argv[1], sys.argv[2]
    at = None; out = "리프터근처_차량수_1분.csv"
    if "--at" in sys.argv:
        at = sys.argv[sys.argv.index("--at") + 1]
    elif len(sys.argv) > 3 and not sys.argv[3].startswith("--"):
        out = sys.argv[3]

    # 리프터 -> 근처 HID4, 경계mm
    lifter_zone = {}
    lifter_mm = {}
    with open(map_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lifter_zone[r["Lifter"]] = r["근처HID4"].strip()
            lifter_mm[r["Lifter"]] = (r.get("경계mm") or "").strip()

    # (분, HID) -> 진입 차량집합
    zone_veh = defaultdict(set)
    with open(inout, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            m = r["_time"][:16]
            if at and m != at:
                continue
            zone_veh[(m, r["TO_HIDID"].strip())].add(r["VHL_ID"])

    minutes = sorted(set(m for m, _ in zone_veh))
    fab = lambda lf: "M16" if lf[0] == "6" else ("M14" if lf[0] == "4" else "?")

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["시각", "Lifter", "FAB", "근처HID", "경계mm", "근처차량수"])
        for m in minutes:
            for lf in sorted(lifter_zone):
                z = lifter_zone[lf]
                w.writerow([m, lf, fab(lf), z, lifter_mm.get(lf, ""), len(zone_veh.get((m, z), ()))])

    if at:
        print(f"=== {at} · 리프터 근처 HID 구간 차량수 (17기, 중복제거) ===")
        rows = [(lf, lifter_zone[lf], lifter_mm.get(lf, ""), len(zone_veh.get((at, lifter_zone[lf]), ()))) for lf in lifter_zone]
        for lf, z, mm, c in sorted(rows, key=lambda x: -x[3]):
            print(f"  {lf:10} 근처HID{z:3} (경계 {mm}mm) -> {c:3}대")
    else:
        print(f"분 구간 {len(minutes)}개 ({minutes[0]} ~ {minutes[-1]}) · 리프터 {len(lifter_zone)}기")
    print(f"\n저장: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
