# -*- coding: utf-8 -*-
"""
M16A MCS - MAXCAPACITY 조정/원복 추적
- UI-MACHINE-STORAGE-CAPACITY 메시지의 TEXT(XML)에서 MAXCAPACITY 추출
- MACHINENAME 별 시간순 정렬 -> 값이 바뀌는 순간(변경점)만 추출
- 변경점을 조정(상향) -> 원복(복귀) 로 페어링
- 조정/원복 페어 CSV 1개 출력

사용법:
    python maxcapa_track.py --from "2026-07-28 00:00:00" --to "2026-07-29 00:00:00"
    python maxcapa_track.py --days 1
"""
import os
import re
import sys
import csv
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from mcs_query import load_config, connect, run, to_str
except ImportError:
    sys.exit("[ERR] mcs_query.py 를 같은 폴더에 둬라")

MSG = "UI-MACHINE-STORAGE-CAPACITY"

SQL = """
SELECT TIME, PROCESSNAME, TEXT
  FROM NT_L_LOGMESSAGE
 WHERE COMMUNICATIONMESSAGENAME = :msg
   AND TIME >= TO_DATE(:dt_from, 'YYYY-MM-DD HH24:MI:SS')
   AND TIME <  TO_DATE(:dt_to,   'YYYY-MM-DD HH24:MI:SS')
 ORDER BY TIME
"""

# XML 태그 뽑기 (정규식, 파서 부담 없음)
TAGS = ("MACHINENAME", "USERNAME", "STATE", "MAXCAPACITY", "CURRENTCAPACITY",
        "HIGHWATERMARK", "LOWWATERMARK", "FULLUP", "TSCSTATE", "CONTROLSTATE")


def tag(text, name):
    m = re.search(r"<%s>(.*?)</%s>" % (name, name), text, re.S)
    return m.group(1).strip() if m else ""


def to_int(v):
    try:
        return int(str(v).strip())
    except Exception:
        return None


def parse(text):
    d = {t: tag(text, t) for t in TAGS}
    d["MAXCAPACITY"] = to_int(d["MAXCAPACITY"])
    d["CURRENTCAPACITY"] = to_int(d["CURRENTCAPACITY"])
    return d


# ------------------------------------------------- 변경점 추출
def extract_changes(rows, cols):
    """MACHINE 별 시간순으로 훑으며 MAXCAPACITY 가 바뀐 순간만 뽑는다"""
    idx = {c: i for i, c in enumerate(cols)}
    hist = defaultdict(list)

    for r in rows:
        text = to_str(r[idx["TEXT"]])
        if not text:
            continue
        d = parse(text)
        mach = d["MACHINENAME"]
        if not mach or d["MAXCAPACITY"] is None:
            continue
        hist[mach].append({
            "time": r[idx["TIME"]],
            "proc": to_str(r[idx["PROCESSNAME"]]),
            "max": d["MAXCAPACITY"],
            "cur": d["CURRENTCAPACITY"],
            "user": d["USERNAME"],
            "state": d["STATE"],
            "fullup": d["FULLUP"],
        })

    changes = []       # 변경점
    baseline = {}      # MACHINE 별 최초 관측값 = 기준값
    lastval = {}       # MACHINE 별 최종값

    for mach, evs in hist.items():
        evs.sort(key=lambda e: e["time"])
        baseline[mach] = evs[0]["max"]
        lastval[mach] = evs[-1]

        prev = evs[0]
        for e in evs[1:]:
            if e["max"] != prev["max"]:
                changes.append({
                    "machine": mach,
                    "time": e["time"],
                    "before": prev["max"],
                    "after": e["max"],
                    "diff": e["max"] - prev["max"],
                    "direction": "상향" if e["max"] > prev["max"] else "하향",
                    "baseline": baseline[mach],
                    "cur_carrier": e["cur"],
                    "user": e["user"],
                    "state": e["state"],
                    "fullup": e["fullup"],
                    "prev_time": prev["time"],
                    "hold_h": (e["time"] - prev["time"]).total_seconds() / 3600.0,
                })
                prev = e
            else:
                prev = e   # 값 같으면 시각만 갱신

    changes.sort(key=lambda c: (c["machine"], c["time"]))
    return changes, baseline, lastval, hist


# ------------------------------------------------- 조정/원복 페어링
def make_pairs(changes, baseline, lastval, dt_to_dt):
    """기준값에서 벗어난 시점 = 조정 시작, 기준값으로 돌아온 시점 = 원복"""
    by_mach = defaultdict(list)
    for c in changes:
        by_mach[c["machine"]].append(c)

    pairs = []
    for mach, cs in by_mach.items():
        base = baseline[mach]
        open_c = None
        for c in cs:
            if open_c is None:
                if c["after"] != base:                 # 기준 이탈 = 조정 시작
                    open_c = c
            else:
                if c["after"] == base:                 # 기준 복귀 = 원복
                    pairs.append({
                        "machine": mach, "baseline": base,
                        "adj_time": open_c["time"], "adj_from": open_c["before"],
                        "adj_to": open_c["after"], "peak": open_c["after"],
                        "rst_time": c["time"], "rst_to": c["after"],
                        "hold_h": (c["time"] - open_c["time"]).total_seconds() / 3600.0,
                        "status": "원복완료", "steps": 1,
                    })
                    open_c = None
                else:
                    open_c["after"] = max(open_c["after"], c["after"])  # 계단식 상향 추적

        if open_c is not None:                          # 아직 안 돌아옴
            pairs.append({
                "machine": mach, "baseline": base,
                "adj_time": open_c["time"], "adj_from": open_c["before"],
                "adj_to": open_c["after"], "peak": open_c["after"],
                "rst_time": None, "rst_to": lastval[mach]["max"],
                "hold_h": (dt_to_dt - open_c["time"]).total_seconds() / 3600.0,
                "status": "미원복", "steps": 1,
            })

    pairs.sort(key=lambda p: (0 if p["status"] == "미원복" else 1, -p["hold_h"]))
    return pairs


# ------------------------------------------------- CSV
def w(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        c = csv.writer(f)
        c.writerow(header)
        c.writerows(rows)
    print("[CSV] %s (%d rows)" % (os.path.abspath(path), len(rows)))


def fmt(t):
    return t.strftime("%Y-%m-%d %H:%M:%S") if t else ""


def save_pairs(pairs, path):
    w(path,
      ["MACHINE", "상태", "기준값", "조정시각", "조정 전", "조정 후(최대)",
       "원복시각", "원복값", "유지시간(h)"],
      [[p["machine"], p["status"], p["baseline"], fmt(p["adj_time"]),
        p["adj_from"], p["adj_to"], fmt(p["rst_time"]), p["rst_to"],
        "%.1f" % p["hold_h"]] for p in pairs])


# ------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="MAXCAPACITY 조정/원복 추적")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--from", dest="dt_from", default=None)
    ap.add_argument("--to", dest="dt_to", default=None)
    ap.add_argument("--machine", default=None, help="특정 MACHINE 만 필터")
    ap.add_argument("--csv", default="maxcapa_pairs.csv")
    args = ap.parse_args()

    now = datetime.now()
    dt_to = args.dt_to or now.strftime("%Y-%m-%d %H:%M:%S")
    dt_from = args.dt_from or (now - timedelta(days=args.days)).strftime("%Y-%m-%d 00:00:00")
    dt_to_dt = datetime.strptime(dt_to, "%Y-%m-%d %H:%M:%S")

    print("[RANGE] %s ~ %s" % (dt_from, dt_to))
    conn = connect(load_config())
    try:
        cols, rows = run(conn, SQL, {"msg": MSG, "dt_from": dt_from, "dt_to": dt_to})
    finally:
        conn.close()

    if not rows:
        print("[OUT] 데이터 없다")
        return

    changes, baseline, lastval, hist = extract_changes(rows, cols)

    if args.machine:
        m = args.machine.upper()
        changes = [c for c in changes if m in c["machine"].upper()]
        baseline = {k: v for k, v in baseline.items() if m in k.upper()}
        lastval = {k: v for k, v in lastval.items() if m in k.upper()}

    pairs = make_pairs(changes, baseline, lastval, dt_to_dt)
    notrst = [p for p in pairs if p["status"] == "미원복"]

    print("\n" + "=" * 78)
    print(" 관측 MACHINE %d개 | 로그 %d건 | MAXCAPACITY 변경 %d회 | 조정건 %d (미원복 %d)"
          % (len(baseline), len(rows), len(changes), len(pairs), len(notrst)))
    print("=" * 78)

    print("\n[ 조치 필요 - 미원복 ]")
    if not notrst:
        print("  없음")
    for p in notrst:
        print("  * %-10s  %s -> %s  (기준 %s)  %s 부터 %.1fh 경과"
              % (p["machine"], p["adj_from"], p["adj_to"], p["baseline"],
                 p["adj_time"].strftime("%m-%d %H:%M"), p["hold_h"]))

    print("\n[ 조정 -> 원복 완료 ]")
    done = [p for p in pairs if p["status"] == "원복완료"]
    if not done:
        print("  없음")
    for p in done:
        print("  - %-10s  %s -> %s -> %s  | %s ~ %s (%.1fh)"
              % (p["machine"], p["adj_from"], p["adj_to"], p["rst_to"],
                 p["adj_time"].strftime("%m-%d %H:%M"),
                 p["rst_time"].strftime("%m-%d %H:%M"), p["hold_h"]))
    print("=" * 78 + "\n")

    save_pairs(pairs, args.csv)


if __name__ == "__main__":
    main()
