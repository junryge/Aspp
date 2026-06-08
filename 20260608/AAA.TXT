#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_lifter_inout.py - 리프터 근처 HID 결과 CSV (1분단위, 17기)
  진입개수 + 점유 + MAX_VHL + 포화도 를 한 파일에 다 출력.

입력:
  LOGPRESSO_HID_INOUT_*.csv      (FROM_HIDID/TO_HIDID/VHL_ID/_time)
  리프터_근처HID4.csv             (Lifter -> 근처HID4, 경계mm)
  HID_Zone_Master_*.csv          (Zone_ID -> Vehicle_Max, Vehicle_Precaution)  ※용량/포화도용

출력 컬럼:
  시각, Lifter, FAB, 근처HID, 경계mm, 진입개수, 점유차량수, MAX_VHL, 포화도%
   - 진입개수 = 그 1분에 그 HID로 들어온 차량(TO_HIDID, 중복제거)
   - 점유차량수 = 그 시점 HID에 머무는 차량(IN-OUT 추적 peak)
   - MAX_VHL = 그 HID 최대 수용(Vehicle_Max)
   - 포화도% = 점유차량수 / MAX_VHL * 100

사용법:
  python count_lifter_inout.py LOGPRESSO_HID_INOUT_*.csv 리프터_근처HID4.csv HID_Zone_Master_M16A_BR.csv 결과.csv
  python count_lifter_inout.py ... HID_Zone_Master_M16A_BR.csv --at "2026-04-21 14:04"
"""
import sys, os, csv
from collections import defaultdict


def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    inout, map_csv, hid_master = sys.argv[1:4]
    at = None; out = "결과.csv"
    if "--at" in sys.argv:
        at = sys.argv[sys.argv.index("--at") + 1]
    elif len(sys.argv) > 4 and not sys.argv[4].startswith("--"):
        out = sys.argv[4]

    # 리프터 -> HID, 경계mm
    lifter_zone = {}; lifter_mm = {}
    with open(map_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lifter_zone[r["Lifter"]] = r["근처HID4"].strip()
            lifter_mm[r["Lifter"]] = (r.get("경계mm") or "").strip()

    # HID -> 용량(Vehicle_Max)
    zmax = {}
    with open(hid_master, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try: zmax[r["Zone_ID"].strip()] = int(r.get("Vehicle_Max") or 0)
            except ValueError: pass

    # 이벤트 시간순 -> 진입개수(분별) + 점유 추적(peak)
    events = []
    with open(inout, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            events.append((r["_time"], r["FROM_HIDID"].strip(), r["TO_HIDID"].strip(), r["VHL_ID"]))
    events.sort()

    in_veh = defaultdict(set)               # (minute, hid) -> 진입차량 집합
    occ = defaultdict(set)                  # hid -> 현재 점유 차량
    peak = defaultdict(lambda: defaultdict(int))  # minute -> hid -> peak 점유
    for t, fr, to, v in events:
        m = t[:16]
        if to:
            in_veh[(m, to)].add(v)
            occ[to].add(v)
        if fr:
            occ[fr].discard(v)
        for hid in (fr, to):
            if hid:
                c = len(occ[hid])
                if c > peak[m][hid]:
                    peak[m][hid] = c

    minutes = sorted(set(m for m, _ in in_veh) | set(peak))
    fab = lambda lf: "M16" if lf[0] == "6" else ("M14" if lf[0] == "4" else "?")
    def sat(c, z):
        mx = zmax.get(z, 0)
        return round(100.0 * c / mx, 1) if mx else ""

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["시각", "Lifter", "FAB", "근처HID", "경계mm", "진입개수", "점유차량수", "MAX_VHL", "포화도%"])
        for m in minutes:
            for lf in sorted(lifter_zone):
                z = lifter_zone[lf]
                nin = len(in_veh.get((m, z), ()))
                pk = peak[m].get(z, 0)
                w.writerow([m, lf, fab(lf), z, lifter_mm.get(lf, ""), nin, pk, zmax.get(z, ""), sat(pk, z)])

    if at:
        print(f"=== {at} · 리프터 근처 HID (진입개수 / 점유·포화도) ===")
        rows = [(lf, lifter_zone[lf], len(in_veh.get((at, lifter_zone[lf]), ())),
                 peak.get(at, {}).get(lifter_zone[lf], 0)) for lf in lifter_zone]
        for lf, z, nin, pk in sorted(rows, key=lambda x: -x[2]):
            print(f"  {lf:10} HID{z:3}  진입 {nin:3}대  점유 {pk:3}/{zmax.get(z,'?')} ({sat(pk,z)}%)")
    else:
        print(f"분 구간 {len(minutes)}개 ({minutes[0]} ~ {minutes[-1]}) · 리프터 {len(lifter_zone)}기")
    print(f"\n저장: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
