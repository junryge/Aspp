#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hid_count.py - 특정 시각(분) 스냅샷: 리프터 근방 HID 차량 개수 (17기 전부)

원리: OHT 위치 로그(oht_data)의 EDGE 가 각 HID 의 진입 lane(HID_Zone_Master IN_Lanes)
      과 일치하면 그 차량이 그 HID 로 진입한 것. 대상 1분 구간에서 차량(중복제거) 카운트.
      -> HID4·HID3 전부 커버하므로 리프터 17기 전부 가능.

사용법:
  python hid_count.py <oht_data.zip|csv> <HID_Zone_Master.csv> <리프터_HID.csv> <YYYY-MM-DD HH:MM>

예:
  python hid_count.py 20260421.zip HID_Zone_Master_M16A_BR.csv 리프터_HID.csv "2026-04-21 14:04"
"""
import sys, os, re, csv, zipfile, io
from collections import defaultdict


def hid_entry_edges(hid_master_csv):
    """HID번호 -> 진입 lane edge 집합 {'1788_1789', ...}"""
    edges = defaultdict(set)
    with open(hid_master_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            zid = (r.get("Zone_ID") or "").strip()
            for seg in (r.get("IN_Lanes") or "").split(";"):
                m = re.match(r'\s*(\d+)\s*→\s*(\d+)', seg)
                if m:
                    edges[zid].add(f"{m.group(1)}_{m.group(2)}")
    return edges


def lifter_hid_map(lifter_hid_csv):
    """리프터 -> [HID번호,...]"""
    out = {}
    with open(lifter_hid_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["Lifter"]] = [h.strip() for h in r["근방HID_Zone번호"].split(";") if h.strip()]
    return out


def open_oht(path):
    """oht_data 로그 행 reader (zip 내부 csv 자동 탐색, m16BR/EDGE 보유 파일 우선)"""
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        cands = [n for n in zf.namelist() if "oht_data" in n.lower() and n.endswith(".csv")]
        if not cands:
            raise FileNotFoundError("zip 안에 oht_data csv 없음")
        # EDGE 컬럼이 있는 파일 선택 (없으면 첫번째)
        name = None
        for n in cands:
            head = io.TextIOWrapper(zf.open(n), "utf-8").readline()
            if "EDGE" in head:
                name = n
                break
        name = name or cands[0]
        return csv.reader(io.TextIOWrapper(zf.open(name), "utf-8"))
    return csv.reader(open(path, encoding="utf-8", errors="replace"))


def main():
    if len(sys.argv) < 5:
        print(__doc__); sys.exit(1)
    oht_path, hid_master, lifter_csv, target = sys.argv[1:5]

    edges = hid_entry_edges(hid_master)
    lhmap = lifter_hid_map(lifter_csv)

    # 관심 HID 의 엣지 -> HID 역매핑
    target_hids = set(h for hs in lhmap.values() for h in hs)
    edge2hid = defaultdict(set)
    for hid in target_hids:
        for e in edges.get(hid, ()):
            edge2hid[e].add(hid)

    # oht_data 에서 target 분 + 해당 엣지 통과 차량 수집
    hid_veh = defaultdict(set)
    r = open_oht(oht_path)
    h = next(r)
    ie, it, iv = h.index("EDGE"), h.index("_time"), h.index("VEHICLE")
    for row in r:
        if len(row) <= ie:
            continue
        if not row[it].startswith(target):
            continue
        for hid in edge2hid.get(row[ie], ()):
            hid_veh[hid].add(row[iv])

    # 리프터별 집계 + CSV
    out_csv = f"리프터근방_차량수_{target.replace(':','').replace(' ','_').replace('-','')}.csv"
    rows = []
    for lf in sorted(lhmap):
        hids = lhmap[lf]
        tot = set()
        detail = []
        for hid in hids:
            v = hid_veh.get(hid, set())
            tot |= v
            detail.append(f"HID{hid}:{len(v)}")
        rows.append((lf, "M16" if lf[0] == "6" else "M14", ",".join(hids), len(tot), " ".join(detail)))

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["시각", "Lifter", "FAB", "근방HID", "근방차량수", "HID별내역"])
        for lf, fab, hids, tot, det in rows:
            w.writerow([target, lf, fab, hids, tot, det])

    print(f"=== {target} · 리프터 근방 HID 차량수 (17기) ===")
    for lf, fab, hids, tot, det in sorted(rows, key=lambda x: -x[3]):
        print(f"  {lf:10} {tot:3}대   ({det})")
    print(f"\n저장: {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()
