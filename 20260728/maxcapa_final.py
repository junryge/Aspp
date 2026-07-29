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

            for c in changed:
                recs.append([t["time"].strftime("%Y-%m-%d %H:%M:%S"), t["proc"], user,
                             c["machine"], c["port"], "", c["after"], "변경",
                             len(changed), t["txid"]])
            if args.all:
                for c in same:
                    recs.append([t["time"].strftime("%Y-%m-%d %H:%M:%S"), t["proc"], user,
                                 c["machine"], c["port"], c["before"], c["after"],
                                 "변경없음", len(changed), t["txid"]])
    finally:
        conn.close()

    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["조작시각", "PROCESS", "조작자사번", "MACHINE", "PORT",
                    "전(before)", "후(after)", "구분", "동일TX변경포트수", "TRANSACTIONID"])
        w.writerows(recs)
    print("\n[CSV] %s (%d rows)" % (os.path.abspath(args.csv), len(recs)))

    ch = [r for r in recs if r[7] == "변경"]
    if ch:
        print("\n" + "=" * 74)
        print(" 조정 요약 : 조작 %d회 / 변경 포트 %d개" % (len(txs), len(ch)))
        print("=" * 74)
        byuser = defaultdict(int)
        for r in ch:
            byuser[r[2] or "-"] += 1
        for u, n in sorted(byuser.items(), key=lambda x: -x[1]):
            print("   사번 %-10s : %d포트" % (u, n))
        print("=" * 74)


if __name__ == "__main__":
    main()
