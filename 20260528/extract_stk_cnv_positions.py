#!/usr/bin/env python3
"""station.dat 의 CNV/STK 포트 station + layout.xml Addr 좌표를 결합해서
   CNV/STK 포트의 고정 위치(X, Y) CSV 출력.
python3 extract_stk_cnv_positions.py --station /tmp/A.station.dat --layout-zip /tmp/A.layout.zip --output stk_cnv_positions_M14A.csv
사용:
    python extract_stk_cnv_positions.py
    python extract_stk_cnv_positions.py \\
        --station /home/user/ASAS/OHT2/station.dat \\
        --layout-zip /home/user/ASAS/OHT2/layout/layout/layout.zip \\
        --output stk_cnv_positions.csv
"""

import argparse
import csv
import sys
import time
import zipfile
from pathlib import Path


PORT_STATION_TYPES = {"ACQUIRE", "DEPOSIT", "DUAL_ACCESS"}


def parse_station_line(line):
    """station.dat 한 줄 파싱.
    형식: STATION = id,"TYPE",f2,"NAME",flags,f5,address,f7..f11,"ACCESS",f13..f15,slide,...
    """
    if "STATION" not in line or "=" not in line:
        return None

    after_eq = line.split("=", 1)[1].strip()
    fields, current, in_quote = [], "", False
    for ch in after_eq:
        if ch == '"':
            in_quote = not in_quote
            current += ch
        elif ch == "," and not in_quote:
            fields.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        fields.append(current.strip())

    if len(fields) < 17:
        return None

    try:
        return {
            "station_id": int(fields[0]),
            "station_type": fields[1].strip('"'),
            "name": fields[3].strip('"'),
            "address": int(fields[6]),
            "access_mode": fields[12].strip('"'),
            "slide_mm": int(fields[16]),
        }
    except (ValueError, IndexError):
        return None


def parse_stations(station_dat_path):
    stations = []
    with open(station_dat_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = parse_station_line(line)
            if rec and rec["station_type"] in PORT_STATION_TYPES:
                stations.append(rec)
    return stations


def parse_layout_addresses(layout_zip_path):
    """layout.xml 에서 Addr -> (cad-x, cad-y, cad-z, draw-x, draw-y) 추출.

    layout.xml 구조:
        <group name="AddrNNNNN" class="...address.Addr">
            <param value="..." key="address"/>
            <param value="..." key="cad-x"/>
            ...
            <group name="NextAddrXX" class="...NextAddr">  ← 이 안의 param 은 건너뜀
                ...
            </group>
        </group>
    """
    addrs = {}

    with zipfile.ZipFile(layout_zip_path) as zf:
        xml_name = next(
            (n for n in zf.namelist() if n.endswith("layout.xml")),
            None,
        )
        if xml_name is None:
            raise FileNotFoundError(f"layout.xml not found in {layout_zip_path}")

        with zf.open(xml_name) as f:
            in_addr = False
            in_next = False
            params = {}

            for raw in f:
                line = raw.decode("utf-8", errors="replace")

                if '<group name="Addr' in line and 'address.Addr"' in line:
                    if in_addr and params.get("address"):
                        try:
                            a = int(params["address"])
                            addrs[a] = {
                                "cad_x": float(params.get("cad-x", 0)),
                                "cad_y": float(params.get("cad-y", 0)),
                                "cad_z": float(params.get("cad-z", 0)),
                                "draw_x": float(params.get("draw-x", 0)),
                                "draw_y": float(params.get("draw-y", 0)),
                            }
                        except ValueError:
                            pass
                    in_addr = True
                    in_next = False
                    params = {}
                    continue

                if not in_addr:
                    continue

                if '<group name="NextAddr' in line:
                    in_next = True
                    continue
                if in_next and "</group>" in line:
                    in_next = False
                    continue
                if in_next:
                    continue

                if "<param " in line and 'key="' in line and 'value="' in line:
                    k_start = line.index('key="') + 5
                    k_end = line.index('"', k_start)
                    v_start = line.index('value="') + 7
                    v_end = line.index('"', v_start)
                    params[line[k_start:k_end]] = line[v_start:v_end]

            if in_addr and params.get("address"):
                try:
                    a = int(params["address"])
                    addrs[a] = {
                        "cad_x": float(params.get("cad-x", 0)),
                        "cad_y": float(params.get("cad-y", 0)),
                        "cad_z": float(params.get("cad-z", 0)),
                        "draw_x": float(params.get("draw-x", 0)),
                        "draw_y": float(params.get("draw-y", 0)),
                    }
                except ValueError:
                    pass

    return addrs


def equipment_id(port_name):
    return port_name.split("_", 1)[0] if "_" in port_name else port_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="/home/user/ASAS/OHT2/station.dat")
    ap.add_argument("--layout-zip", default="/home/user/ASAS/OHT2/layout/layout/layout.zip")
    ap.add_argument("--output", default="stk_cnv_positions.csv")
    args = ap.parse_args()

    for p in (args.station, args.layout_zip):
        if not Path(p).exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            sys.exit(1)

    t0 = time.time()
    print(f"[1/3] station.dat 파싱: {args.station}")
    stations = parse_stations(args.station)
    print(f"      포트 station {len(stations)}개 ({'/'.join(sorted(PORT_STATION_TYPES))})")

    print(f"[2/3] layout.xml 파싱 (대용량, 시간 소요): {args.layout_zip}")
    t1 = time.time()
    addrs = parse_layout_addresses(args.layout_zip)
    print(f"      Addr {len(addrs)}개 좌표 추출 ({time.time() - t1:.1f}s)")

    print(f"[3/3] CSV 생성: {args.output}")
    rows, missing = [], 0
    for st in stations:
        a = addrs.get(st["address"])
        if a is None:
            missing += 1
            continue
        rows.append({
            "station_id": st["station_id"],
            "station_type": st["station_type"],
            "access_mode": st["access_mode"],
            "port_name": st["name"],
            "equipment_id": equipment_id(st["name"]),
            "address": st["address"],
            "slide_mm": st["slide_mm"],
            "cad_x": a["cad_x"],
            "cad_y": a["cad_y"],
            "cad_z": a["cad_z"],
            "draw_x": a["draw_x"],
            "draw_y": a["draw_y"],
        })

    rows.sort(key=lambda r: (r["equipment_id"], r["port_name"]))

    fields = [
        "station_id", "station_type", "access_mode",
        "port_name", "equipment_id", "address", "slide_mm",
        "cad_x", "cad_y", "cad_z", "draw_x", "draw_y",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n완료: {len(rows)}개 포트 → {args.output} ({time.time() - t0:.1f}s 총)")
    if missing:
        print(f"      ⚠ Addr 미발견 {missing}개 (layout.xml 에 없는 주소)")

    by_type = {}
    eqs = set()
    for r in rows:
        by_type[r["station_type"]] = by_type.get(r["station_type"], 0) + 1
        eqs.add(r["equipment_id"])
    print(f"\n  유형별: {by_type}")
    print(f"  고유 장비 ID: {len(eqs)}개")


if __name__ == "__main__":
    main()
