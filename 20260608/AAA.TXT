#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_near_hid4.py - 리프터 -> 근처 HID4 구간 매핑 CSV 생성 (단독 실행, 의존 없음)

각 리프터에 '경계(lane)가 가장 가까운 HID4 구역(1~37)' 을 찾아 매핑.
make_map.py 등 다른 파일 필요 없음. 아래 3개 입력만 있으면 됨.

입력:
  1) BR.layout.zip (또는 .xml)         - 주소->좌표
  2) BR.station.dat                    - 리프터 포트->주소
  3) HID_Zone_Master_M16A_BR.csv       - HID4 구역 lane

출력: 리프터_근처HID4.csv  (Lifter, FAB, 근처HID4)

사용법:
  python gen_near_hid4.py BR.layout.zip BR.station.dat HID_Zone_Master_M16A_BR.csv 리프터_근처HID4.csv
"""
import sys, os, csv, re, math, zipfile


def load_xml(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            name = next((n for n in zf.namelist() if n.lower().endswith("layout.xml")), None)
            if not name:
                raise FileNotFoundError("zip 안에 layout.xml 없음")
            return zf.read(name).decode("utf-8", "replace")
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_addr_xy(xml):
    """layout.xml -> {address: (x,y)}"""
    nodes = {}
    cur = None
    key_re = re.compile(r'key="([^"]+)"'); val_re = re.compile(r'value="([^"]*)"')
    for line in xml.split("\n"):
        line = line.strip()
        if '<group name="Addr' in line and 'address.Addr"' in line:
            if cur and "address" in cur:
                try:
                    a = int(cur["address"])
                    if a > 0:
                        nodes[a] = (float(cur.get("draw-x", 0)), float(cur.get("draw-y", 0)))
                except ValueError:
                    pass
            cur = {}
            continue
        if cur is not None and line.startswith("<param"):
            k = key_re.search(line); v = val_re.search(line)
            if k and v:
                cur[k.group(1)] = v.group(1)
    if cur and "address" in cur:
        try:
            a = int(cur["address"])
            if a > 0:
                nodes[a] = (float(cur.get("draw-x", 0)), float(cur.get("draw-y", 0)))
        except ValueError:
            pass
    return nodes


def parse_lifter_ports(station_path):
    """station.dat -> {address: lifter_id}  (리프터 *ABL* 포트)"""
    out = {}
    for line in open(station_path, encoding="utf-8", errors="replace"):
        if "ABL" not in line:
            continue
        m = re.search(r'STATION\s*=\s*(.+)', line)
        if not m:
            continue
        parts = [p.strip().strip('"') for p in m.group(1).split(",")]
        try:
            port, addr = parts[3], int(parts[6])
        except (IndexError, ValueError):
            continue
        if re.match(r'\dABL', port) and ("_AI" in port or "_AO" in port):
            out[addr] = port.split("_")[0]
    return out


def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    layout, station, hid_master = sys.argv[1:4]
    out = sys.argv[4] if len(sys.argv) > 4 else "리프터_근처HID4.csv"

    for f in (layout, station, hid_master):
        if not os.path.exists(f):
            print(f"[오류] 입력 파일 없음: {f}"); sys.exit(1)

    print(f"[1/3] 레이아웃 파싱: {layout}")
    nodes = parse_addr_xy(load_xml(layout))
    print(f"      주소 {len(nodes)}개")

    print(f"[2/3] station.dat 파싱: {station}")
    ports = parse_lifter_ports(station)
    from collections import defaultdict
    lpts = defaultdict(list)
    for a, lf in ports.items():
        if a in nodes:
            lpts[lf].append(nodes[a])
    print(f"      리프터 {len(lpts)}기")
    if not lpts:
        print("      [경고] 리프터 0기! station.dat 가 올바른지 확인하세요 (정상=6ABL/4ABL 포트 포함, 약 113KB)")

    print(f"[3/3] HID4 구역 lane 파싱: {hid_master}")
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
    print(f"      HID4 구역 {len(hid4)}개")

    rows = []
    for lf in sorted(lpts):
        best, bd = None, 1e18
        for z, pts in hid4.items():
            for px, py in pts:
                for lx, ly in lpts[lf]:
                    d = (lx - px) ** 2 + (ly - py) ** 2
                    if d < bd:
                        bd, best = d, z
        rows.append((lf, "M16" if lf[0] == "6" else "M14", best, round(math.sqrt(bd)) if best else -1))

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Lifter", "FAB", "근처HID4"])
        for lf, fab, z, d in rows:
            w.writerow([lf, fab, z])

    print(f"\n=== 생성됨: {out} (리프터 {len(rows)}기) ===")
    for lf, fab, z, d in rows:
        print(f"  {lf:10} -> HID{z}  (경계 {d}mm)")
    print(f"\n저장 위치: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
