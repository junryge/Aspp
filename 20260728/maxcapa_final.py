# -*- coding: utf-8 -*-
"""
M16A MCS - 포트 MAXCAPACITY 수동 조정 내역 (최종)

  1단계: UI-UNIT-PORT-MAXCAPACITY-UPDATE -> TRANSACTIONID + 조작자 사번
  2단계: TRANSACTIONID 재조회 -> compareAndUpdatePortMaxCapacity TEXT 파싱

  TEXT 패턴:
    port{6ABL6031_AI612}.maxCapacity was changed to {1}    <- 실제 변경 (후값만)
    Current PortMaxCapacity is 4. Input maxCapacity is 4.  <- 값 동일 (변경 아님)

  전(before) 값: 같은 포트의 직전 변경건 후값에서 가져온다.
                 기간 내 첫 변경건은 알 수 없다 (로그에 없음) -> 빈칸

사용법:
    python maxcapa_v3.py --from "2026-06-01 00:00:00" --to "2026-07-30 00:00:00"
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
SELECT TIME, OPERATIONNAME, MACHINENAME, UNITNAME, TEXT
  FROM NT_L_LOGMESSAGE
 WHERE TRANSACTIONID = :txid
   AND PARTITIONID = :pid
 ORDER BY TIME
"""

RE_CHANGED = re.compile(r"port\{([^}]+)\}\.maxCapacity was changed to \{(-?\d+)\}")
RE_USER = re.compile(r"<USERNAME>\s*(\S+?)\s*</USERNAME>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dt_from", default=None)
    ap.add_argument("--to", dest="dt_to", default=None)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--csv", default="maxcapa_v3.csv")
    args = ap.parse_args()

    now = datetime.now()
    dt_to = args.dt_to or now.strftime("%Y-%m-%d %H:%M:%S")
    dt_from = args.dt_from or (now - timedelta(days=args.days)).strftime("%Y-%m-%d 00:00:00")

    conn = connect(load_config())
    try:
        print("\n[1단계] TRANSACTIONID 수집 (%s ~ %s)" % (dt_from, dt_to))
        c1, r1 = run(conn, SQL1, {"msg": MSG, "dt_from": dt_from, "dt_to": dt_to})
        i1 = {c: i for i, c in enumerate(c1)}
        txs = [{"txid": to_str(r[i1["TRANSACTIONID"]]), "time": r[i1["TIME"]],
                "proc": to_str(r[i1["PROCESSNAME"]]), "pid": to_str(r[i1["PARTITIONID"]])}
               for r in r1 if to_str(r[i1["TRANSACTIONID"]])]
        print("   조작 %d회" % len(txs))
        if not txs:
            print("   해당 기간 조정 없음")
            return

        print("\n[2단계] 상세 파싱")
        recs = []
        for t in txs:
            c2, r2 = run(conn, SQL2, {"txid": t["txid"], "pid": t["pid"]})
            i2 = {c: i for i, c in enumerate(c2)}

            user = ""
            seen = set()                       # 중복 로그 제거용
            got = []

            for r in r2:
                text = to_str(r[i2["TEXT"]])
                if not text:
                    continue
                if not user:
                    m = RE_USER.search(text)
                    if m:
                        user = m.group(1)
                if "compareAndUpdatePortMaxCapacity" not in to_str(r[i2["OPERATIONNAME"]]):
                    continue
                mc = RE_CHANGED.search(text)
                if not mc:
                    continue                   # same to value = 변경 아님, 버림
                port, after = mc.group(1), int(mc.group(2))
                if port in seen:               # 같은 TX 안 중복 제거
                    continue
                seen.add(port)
                got.append({"port": port, "after": after,
                            "machine": to_str(r[i2["MACHINENAME"]])})

            print("   %s | %s | 사번 %-8s | 변경 %d포트"
                  % (t["time"].strftime("%m-%d %H:%M:%S"), t["proc"],
                     user or "-", len(got)))

            for g in got:
                recs.append({"time": t["time"], "proc": t["proc"], "user": user,
                             "machine": g["machine"] or g["port"].split("_")[0],
                             "port": g["port"], "after": g["after"],
                             "nch": len(got), "txid": t["txid"]})
    finally:
        conn.close()

    # ---------------- 전(before) 채우기 ----------------
    recs.sort(key=lambda r: (r["time"], r["port"]))
    lastval = {}
    for r in recs:
        r["before"] = lastval.get(r["port"], "")
        lastval[r["port"]] = r["after"]

    # 증감 / 방향
    for r in recs:
        if r["before"] == "":
            r["diff"] = ""
            r["dir"] = ""
        else:
            d = r["after"] - r["before"]
            r["diff"] = "%+d" % d
            r["dir"] = "상향" if d > 0 else "하향"

    # 원복 판정: 이전에 가졌던 값으로 되돌아왔는지
    held = defaultdict(list)
    for r in recs:
        if r["before"] != "" and r["after"] in held[r["port"]]:
            r["dir"] = r["dir"] + "(원복)"
        if r["before"] != "":
            held[r["port"]].append(r["before"])

    # ---------------- CSV ----------------
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["조작시각", "조작자사번", "MACHINE", "PORT",
                    "전(before)", "후(after)", "증감", "방향",
                    "PROCESS", "동일조작변경포트수", "TRANSACTIONID"])
        for r in recs:
            w.writerow([r["time"].strftime("%Y-%m-%d %H:%M:%S"), r["user"],
                        r["machine"], r["port"], r["before"], r["after"],
                        r["diff"], r["dir"], r["proc"], r["nch"], r["txid"]])
    print("\n[CSV] %s (%d rows)" % (os.path.abspath(args.csv), len(recs)))

    # ---------------- 요약 ----------------
    print("\n" + "=" * 86)
    print(" 조작 %d회 / 포트변경 %d건" % (len(txs), len(recs)))
    print("=" * 86)
    print(" %-17s %-10s %-22s %5s %5s %6s %s"
          % ("시각", "사번", "PORT", "전", "후", "증감", "방향"))
    print("-" * 86)
    for r in recs:
        print(" %-17s %-10s %-22s %5s %5s %6s %s"
              % (r["time"].strftime("%m-%d %H:%M:%S"), r["user"] or "-", r["port"],
                 r["before"] if r["before"] != "" else "?", r["after"],
                 r["diff"] or "-", r["dir"] or "-"))
    print("-" * 86)
    unk = sum(1 for r in recs if r["before"] == "")
    print(" 전값 확인 %d / 미확인 %d  (미확인 = 기간 내 첫 변경, 기간 넓히면 채워짐)"
          % (len(recs) - unk, unk))
    byuser = defaultdict(int)
    for r in recs:
        byuser[r["user"] or "-"] += 1
    for u, n in sorted(byuser.items(), key=lambda x: -x[1]):
        print(" 사번 %-10s : %d건" % (u, n))
    print("=" * 86)


if __name__ == "__main__":
    main()
