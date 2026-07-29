# -*- coding: utf-8 -*-
"""
M16A MCS - MAXCAPACITY 변경내역 추출 (TRANSACTIONID 2단계 조회)

  1단계: UI-UNIT-PORT-MAXCAPACITY-UPDATE 로 TRANSACTIONID 수집
  2단계: 그 TRANSACTIONID 로 전체 메시지 재조회 -> TEXT(XML) 파싱

사용법:
    python maxcapa_tx.py --from "2026-07-27 00:00:00" --to "2026-07-30 00:00:00"
    python maxcapa_tx.py --from "2026-07-28 00:00:00" --to "2026-07-29 00:00:00" --raw
"""
import os
import re
import sys
import csv
import argparse
from datetime import datetime, timedelta

try:
    from mcs_query import load_config, connect, run, to_str
except ImportError:
    sys.exit("[ERR] mcs_query.py 를 같은 폴더에 둬라")

MSG = "UI-UNIT-PORT-MAXCAPACITY-UPDATE"

# 1단계: TRANSACTIONID 수집
SQL1 = """
SELECT /*+ INDEX(NT_L_LOGMESSAGE NT_L_LOGMESSAGE_IX2) */
       TRANSACTIONID, TIME, PROCESSNAME, PARTITIONID
  FROM NT_L_LOGMESSAGE
 WHERE COMMUNICATIONMESSAGENAME = :msg
   AND TIME >= TO_DATE(:dt_from, 'YYYY-MM-DD HH24:MI:SS')
   AND TIME <  TO_DATE(:dt_to,   'YYYY-MM-DD HH24:MI:SS')
 ORDER BY TIME
"""

# 2단계: TRANSACTIONID 로 전체 메시지 (파티션 + 시간범위로 좁힘)
SQL2 = """
SELECT TIME, COMMUNICATIONMESSAGENAME, PROCESSNAME, OPERATIONNAME,
       MACHINENAME, UNITNAME, TEXT
  FROM NT_L_LOGMESSAGE
 WHERE TRANSACTIONID = :txid
   AND PARTITIONID = :pid
 ORDER BY TIME
"""

TAGS = ["MACHINENAME", "UNITNAME", "USERNAME", "ORIGINATEDTYPE", "ORIGINATEDNAME",
        "MAXCAPACITY", "OLDMAXCAPACITY", "NEWMAXCAPACITY", "CAPACITY",
        "QUEUSESIZE", "ZONENAME", "ZONECAPACITY", "STATE", "ACCESSMODE",
        "MANUAL", "PORTTYPE", "COMMENTS"]


def tag(text, name):
    m = re.search(r"<%s>(.*?)</%s>" % (name, name), text, re.S)
    return m.group(1).strip() if m else ""


def all_tags(text):
    """XML 안의 모든 태그를 dict 로. 어떤 필드가 있는지 모를 때 대비"""
    out = {}
    for k, v in re.findall(r"<([A-Z][A-Z0-9_]*)>([^<>]*)</\1>", text):
        v = v.strip()
        if v and k not in out:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dt_from", default=None)
    ap.add_argument("--to", dest="dt_to", default=None)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--csv", default="maxcapa_tx.csv")
    ap.add_argument("--raw", action="store_true", help="XML 원문도 화면에 출력")
    args = ap.parse_args()

    now = datetime.now()
    dt_to = args.dt_to or now.strftime("%Y-%m-%d %H:%M:%S")
    dt_from = args.dt_from or (now - timedelta(days=args.days)).strftime("%Y-%m-%d 00:00:00")

    conn = connect(load_config())
    try:
        # ---------------- 1단계
        print("\n[1단계] %s 조회 (%s ~ %s)" % (MSG, dt_from, dt_to))
        c1, r1 = run(conn, SQL1, {"msg": MSG, "dt_from": dt_from, "dt_to": dt_to})
        i1 = {c: i for i, c in enumerate(c1)}

        txs = []
        for r in r1:
            txid = to_str(r[i1["TRANSACTIONID"]])
            if txid:
                txs.append({
                    "txid": txid,
                    "time": r[i1["TIME"]],
                    "proc": to_str(r[i1["PROCESSNAME"]]),
                    "pid": to_str(r[i1["PARTITIONID"]]),
                })
        print("   TRANSACTIONID %d개 확보" % len(txs))
        for t in txs:
            print("     %s  %s  %s" % (t["time"].strftime("%m-%d %H:%M:%S"),
                                       t["proc"], t["txid"]))
        if not txs:
            print("   해당 기간에 없다. 기간 바꿔봐라.")
            return

        # ---------------- 2단계
        print("\n[2단계] TRANSACTIONID 별 전체 메시지 조회")
        rows_out = []
        for t in txs:
            c2, r2 = run(conn, SQL2, {"txid": t["txid"], "pid": t["pid"]})
            i2 = {c: i for i, c in enumerate(c2)}
            print("\n  === %s (%s) : %d개 메시지 ==="
                  % (t["txid"], t["time"].strftime("%m-%d %H:%M:%S"), len(r2)))

            for r in r2:
                text = to_str(r[i2["TEXT"]])
                msgname = to_str(r[i2["COMMUNICATIONMESSAGENAME"]])
                tg = all_tags(text) if text else {}

                print("    - %-42s len=%d" % (msgname, len(text)))
                # 값이 있는 태그만 요약 출력
                pick = {k: v for k, v in tg.items()
                        if any(w in k for w in ("CAPACITY", "CAPA", "QUEUE", "QUEUS",
                                                "USERNAME", "UNITNAME", "MACHINENAME",
                                                "ORIGINATED", "ZONE"))}
                if pick:
                    print("        " + ", ".join("%s=%s" % (k, v) for k, v in pick.items()))
                if args.raw and text:
                    print("      " + "-" * 60)
                    print(text[:3000])
                    print("      " + "-" * 60)

                rows_out.append([
                    t["txid"],
                    r[i2["TIME"]].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    msgname,
                    to_str(r[i2["PROCESSNAME"]]),
                    to_str(r[i2["OPERATIONNAME"]]),
                    tg.get("ORIGINATEDTYPE", ""),
                    tg.get("USERNAME", ""),
                    tg.get("MACHINENAME", "") or to_str(r[i2["MACHINENAME"]]),
                    tg.get("UNITNAME", "") or to_str(r[i2["UNITNAME"]]),
                    tg.get("MAXCAPACITY", ""),
                    tg.get("OLDMAXCAPACITY", ""),
                    tg.get("NEWMAXCAPACITY", ""),
                    tg.get("QUEUSESIZE", ""),
                    tg.get("ZONENAME", ""),
                    tg.get("ZONECAPACITY", ""),
                    len(text),
                    text.replace("\n", " ")[:4000],
                ])
    finally:
        conn.close()

    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["TRANSACTIONID", "TIME", "MESSAGENAME", "PROCESS", "OPERATION",
                    "ORIGINATEDTYPE", "USERNAME", "MACHINENAME", "UNITNAME",
                    "MAXCAPACITY", "OLD", "NEW", "QUEUSESIZE",
                    "ZONENAME", "ZONECAPACITY", "TEXT_LEN", "TEXT"])
        w.writerows(rows_out)
    print("\n[CSV] %s (%d rows)" % (os.path.abspath(args.csv), len(rows_out)))


if __name__ == "__main__":
    main()
