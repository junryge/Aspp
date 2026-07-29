# -*- coding: utf-8 -*-
"""
M16A MCS - 포트 MAXCAPACITY 수동 조정 내역 추출 (최종판)

  1단계: UI-UNIT-PORT-MAXCAPACITY-UPDATE 로 TRANSACTIONID + USERNAME 수집
  2단계: TRANSACTIONID 로 재조회 -> OPERATION=compareAndUpdatePortMaxCapacity 의
         TEXT 를 파싱해서 전 -> 후 추출

  TEXT 두 가지 패턴:
    변경됨 : port{6ABL6031_AI612}.maxCapacity was changed to {1}
    동일   : Current PortMaxCapacity is 4. Input maxCapacity is 4. same to value

사용법:
    python maxcapa_final.py --from "2026-07-27 00:00:00" --to "2026-07-30 00:00:00"
    python maxcapa_final.py --days 7 --all      # 미변경(same) 건도 포함
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

MSG = "UI-UNIT-PORT-MAXCAPACITY-UPDATE"

SQL1 = """
SELECT /*+ INDEX(NT_L_LOGMESSAGE NT_L_LOGMESSAGE_IX2) */
       TRANSACTIONID, TIME, PROCESSNAME, PARTITIONID
  FROM NT_L_LOGMESSAGE
 WHERE COMMUNICATIONMESSAGENAME = :msg
   AND TIME >= TO_DATE(:dt_from, 'YYYY-MM-DD HH24:MI:SS')
   AND TIME <  TO_DATE(:dt_to,   'YYYY-MM-DD HH24:MI:SS')
 ORDER BY TIME
"""

SQL2 = """
SELECT TIME, COMMUNICATIONMESSAGENAME, OPERATIONNAME,
       MACHINENAME, UNITNAME, TEXT
  FROM NT_L_LOGMESSAGE
 WHERE TRANSACTIONID = :txid
   AND PARTITIONID = :pid
 ORDER BY TIME
"""

# 파싱 패턴
RE_CHANGED = re.compile(r"port\{([^}]+)\}\.maxCapacity was changed to \{(-?\d+)\}")
RE_SAME = re.compile(r"Current PortMaxCapacity is (-?\d+)\.\s*Input maxCapacity is (-?\d+)\.\s*same")
RE_USER = re.compile(r"<USERNAME>(.*?)</USERNAME>", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dt_from", default=None)
    ap.add_argument("--to", dest="dt_to", default=None)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--csv", default="maxcapa_final.csv")
    ap.add_argument("--all", action="store_true", help="변경없음(same) 건도 CSV에 포함")
    ap.add_argument("--snapall", action="store_true", help="(내부용)")
    args = ap.parse_args()

    now = datetime.now()
    dt_to = args.dt_to or now.strftime("%Y-%m-%d %H:%M:%S")
    dt_from = args.dt_from or (now - timedelta(days=args.days)).strftime("%Y-%m-%d 00:00:00")

    conn = connect(load_config())
    try:
        print("\n[1단계] TRANSACTIONID 수집 (%s ~ %s)" % (dt_from, dt_to))
        c1, r1 = run(conn, SQL1, {"msg": MSG, "dt_from": dt_from, "dt_to": dt_to})
        i1 = {c: i for i, c in enumerate(c1)}

        txs = []
        for r in r1:
            tid = to_str(r[i1["TRANSACTIONID"]])
            if tid:
                txs.append({"txid": tid, "time": r[i1["TIME"]],
                            "proc": to_str(r[i1["PROCESSNAME"]]),
                            "pid": to_str(r[i1["PARTITIONID"]])})
        print("   조작 %d건" % len(txs))
        if not txs:
            print("   해당 기간 조정 없음")
            return

        print("\n[2단계] 트랜잭션별 상세 파싱")
        recs = []
        for t in txs:
            c2, r2 = run(conn, SQL2, {"txid": t["txid"], "pid": t["pid"]})
            i2 = {c: i for i, c in enumerate(c2)}

            user = ""
            changed, same = [], []

            for r in r2:
                text = to_str(r[i2["TEXT"]])
                if not text:
                    continue
                if not user:
                    m = RE_USER.search(text)
                    if m and m.group(1).strip():
                        user = m.group(1).strip()

                op = to_str(r[i2["OPERATIONNAME"]])
                if "compareAndUpdatePortMaxCapacity" not in op:
                    continue

                mach = to_str(r[i2["MACHINENAME"]])
                unit = to_str(r[i2["UNITNAME"]])
                tm = r[i2["TIME"]]

                mc = RE_CHANGED.search(text)
                if mc:
                    changed.append({"time": tm, "machine": mach,
                                    "port": mc.group(1), "after": int(mc.group(2))})
                    continue
                ms = RE_SAME.search(text)
                if ms:
                    same.append({"time": tm, "machine": mach, "port": unit,
                                 "before": int(ms.group(1)), "after": int(ms.group(2))})

            print("\n  === %s | %s | %s | 사번 %s ==="
                  % (t["time"].strftime("%m-%d %H:%M:%S"), t["proc"], t["txid"],
                     user or "-"))
            print("      변경 %d포트 / 동일 %d포트" % (len(changed), len(same)))
            for c in changed:
                print("        [변경] %-22s -> %s" % (c["port"], c["after"]))

            # same 건에는 전/후 값이 다 있다 -> 그 시점 현재값 스냅샷으로 활용
            snap = {}
            for c in same:
                if c["port"]:
                    snap[c["port"]] = c["before"]

            for c in changed:
                recs.append({"time": t["time"], "proc": t["proc"], "user": user,
                             "machine": c["machine"], "port": c["port"],
                             "after": c["after"], "kind": "변경",
                             "nch": len(changed), "txid": t["txid"],
                             "snap": snap})
            if args.all:
                for c in same:
                    recs.append({"time": t["time"], "proc": t["proc"], "user": user,
                                 "machine": c["machine"], "port": c["port"],
                                 "before": c["before"], "after": c["after"],
                                 "kind": "변경없음", "nch": len(changed),
                                 "txid": t["txid"], "snap": snap})
    finally:
        conn.close()

    # ---------- 전(before) 값 채우기 ----------
    recs.sort(key=lambda r: r["time"])
    lastval = {}     # port -> 마지막 후값
    baseline = {}    # port -> 최초 기준값

    for r in recs:
        port = r["port"]
        if "before" not in r or r["before"] == "":
            if port in lastval:
                r["before"] = lastval[port]          # 직전 조작의 후값
                r["src"] = "직전이력"
            elif port in r["snap"]:
                r["before"] = r["snap"][port]        # 같은 TX 의 same 로그
                r["src"] = "동일TX"
            else:
                r["before"] = ""
                r["src"] = "미확인"
        else:
            r["src"] = "로그"

        if port not in baseline and r["before"] != "":
            baseline[port] = r["before"]
        lastval[port] = r["after"]

    # ---------- 원복 판정 ----------
    for r in recs:
        b = baseline.get(r["port"], "")
        r["base"] = b
        if b == "" or r["kind"] != "변경":
            r["state"] = ""
        elif r["after"] == b:
            r["state"] = "원복"
        else:
            r["state"] = "조정"

    # ---------- CSV ----------
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["조작시각", "PROCESS", "조작자사번", "MACHINE", "PORT",
                    "전(before)", "후(after)", "증감", "판정", "기준값",
                    "전값출처", "구분", "동일TX변경포트수", "TRANSACTIONID"])
        for r in recs:
            diff = ""
            if r["before"] != "":
                try:
                    diff = "%+d" % (int(r["after"]) - int(r["before"]))
                except Exception:
                    diff = ""
            w.writerow([r["time"].strftime("%Y-%m-%d %H:%M:%S"), r["proc"], r["user"],
                        r["machine"], r["port"], r["before"], r["after"], diff,
                        r["state"], r["base"], r["src"], r["kind"], r["nch"], r["txid"]])
    print("\n[CSV] %s (%d rows)" % (os.path.abspath(args.csv), len(recs)))

    # ---------- 요약 ----------
    ch = [r for r in recs if r["kind"] == "변경"]
    if ch:
        print("\n" + "=" * 84)
        print(" 조정 요약 : 조작 %d회 / 변경 포트 %d개" % (len(txs), len(ch)))
        print("=" * 84)
        print(" %-19s %-22s %6s %6s %6s  %s" % ("시각", "PORT", "전", "후", "증감", "판정"))
        print("-" * 84)
        for r in ch:
            d = ""
            if r["before"] != "":
                try:
                    d = "%+d" % (int(r["after"]) - int(r["before"]))
                except Exception:
                    pass
            print(" %-19s %-22s %6s %6s %6s  %s"
                  % (r["time"].strftime("%m-%d %H:%M:%S"), r["port"],
                     r["before"] if r["before"] != "" else "?", r["after"], d, r["state"]))
        print("-" * 84)
        nof = sum(1 for r in ch if r["before"] == "")
        print(" 전값 확인 %d / 미확인 %d" % (len(ch) - nof, nof))
        byuser = defaultdict(int)
        for r in ch:
            byuser[r["user"] or "-"] += 1
        for u, n in sorted(byuser.items(), key=lambda x: -x[1]):
            print(" 사번 %-10s : %d포트" % (u, n))
        print("=" * 84)


if __name__ == "__main__":
    main()
