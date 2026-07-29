# -*- coding: utf-8 -*-
"""
M16A MCS (M16MCSPP) NT_L_LOGMESSAGE 조회 툴
- config.ini 에서 접속정보 읽음 (CWD 우선)
- oracledb thin mode (Oracle Client 설치 불필요)

사용법:
    python mcs_query.py
    python mcs_query.py --from "2026-07-27 00:00:00" --to "2026-07-28 00:00:00"
    python mcs_query.py --msg UI-UNIT-PORT-MAXCAPACITY-UPDATE --limit 500 --csv out.csv
    python mcs_query.py --sql "select count(*) from NT_L_LOGMESSAGE"
"""
import os
import sys
import csv
import argparse
import configparser
from datetime import datetime

try:
    import oracledb
except ImportError:
    sys.exit("[ERR] pip install oracledb 먼저 해라")

CONFIG_NAME = "config.ini"


# ---------------------------------------------------------------- config
def find_config(name=CONFIG_NAME):
    """CWD 우선, 없으면 스크립트 폴더"""
    for p in (os.path.join(os.getcwd(), name),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), name)):
        if os.path.exists(p):
            return p
    sys.exit("[ERR] %s 못 찾겠다" % name)


def load_config():
    path = find_config()
    cp = configparser.ConfigParser()
    cp.read(path, encoding="utf-8")
    print("[CFG] %s" % path)
    return cp


def build_dsn(cfg):
    """TNS DESCRIPTION 문자열 생성 (RAC FAILOVER 포함)"""
    o = cfg["oracle"]
    hosts = [h.strip() for h in o["hosts"].split(",") if h.strip()]
    port = o.get("port", "1521")

    addr = "".join(
        "(ADDRESS=(PROTOCOL=TCP)(HOST=%s)(PORT=%s))" % (h, port) for h in hosts
    )
    dsn = (
        "(DESCRIPTION="
        "(FAILOVER=%s)"
        "%s"
        "(CONNECT_DATA=(SERVICE_NAME=%s)"
        "(FAILOVER_MODE=(TYPE=%s)(METHOD=%s)(RETRIES=%s)(DELAY=%s))))"
    ) % (
        o.get("failover", "on"),
        addr,
        o["service_name"],
        o.get("failover_type", "SELECT"),
        o.get("method", "BASIC"),
        o.get("retries", "5"),
        o.get("delay", "5"),
    )
    return dsn


def connect(cfg):
    o = cfg["oracle"]
    dsn = build_dsn(cfg)
    print("[CONN] %s@%s" % (o["user"], o["service_name"]))
    conn = oracledb.connect(user=o["user"], password=o["password"], dsn=dsn)
    print("[CONN] OK - version %s" % conn.version)
    return conn


# ---------------------------------------------------------------- query
def build_query(msg, dt_from, dt_to, limit):
    """NT_L_LOGMESSAGE 조회 SQL + 바인드 생성"""
    where = ["COMMUNICATIONMESSAGENAME = :msg"]
    binds = {"msg": msg}

    if dt_from:
        where.append("TIME >= TO_DATE(:dt_from, 'YYYY-MM-DD HH24:MI:SS')")
        binds["dt_from"] = dt_from
    if dt_to:
        where.append("TIME < TO_DATE(:dt_to, 'YYYY-MM-DD HH24:MI:SS')")
        binds["dt_to"] = dt_to

    sql = (
        "SELECT /*+ INDEX(NT_L_LOGMESSAGE NT_L_LOGMESSAGE_IX2) */ *\n"
        "  FROM NT_L_LOGMESSAGE\n"
        " WHERE " + "\n   AND ".join(where) + "\n"
        " ORDER BY TIME"
    )
    if limit and int(limit) > 0:
        sql = "SELECT * FROM (\n%s\n) WHERE ROWNUM <= :lmt" % sql
        binds["lmt"] = int(limit)
    return sql, binds


def run(conn, sql, binds=None, arraysize=1000):
    cur = conn.cursor()
    cur.arraysize = arraysize
    t0 = datetime.now()
    cur.execute(sql, binds or {})
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    print("[SQL] %d rows / %.2f sec" % (len(rows), (datetime.now() - t0).total_seconds()))
    return cols, rows


# ---------------------------------------------------------------- output
def to_str(v):
    if v is None:
        return ""
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, oracledb.LOB):
        return v.read()
    return str(v)


def print_rows(cols, rows, n=20, width=30):
    if not rows:
        print("[OUT] 데이터 없다")
        return
    head = " | ".join(c[:width].ljust(min(width, len(c))) for c in cols)
    print("-" * min(len(head), 200))
    print(head[:200])
    print("-" * min(len(head), 200))
    for r in rows[:n]:
        line = " | ".join(to_str(v)[:width] for v in r)
        print(line[:200])
    if len(rows) > n:
        print("... (총 %d건, 상위 %d건만 표시)" % (len(rows), n))


def save_csv(path, cols, rows):
    if not path:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([to_str(v) for v in r])
    print("[CSV] %s (%d rows)" % (os.path.abspath(path), len(rows)))


# ---------------------------------------------------------------- main
def main():
    cfg = load_config()
    q = cfg["query"] if cfg.has_section("query") else {}

    ap = argparse.ArgumentParser(description="M16A MCS NT_L_LOGMESSAGE 조회")
    ap.add_argument("--msg", default=q.get("message_name", "UI-UNIT-PORT-MAXCAPACITY-UPDATE"))
    ap.add_argument("--from", dest="dt_from", default=q.get("from_time", "").strip())
    ap.add_argument("--to", dest="dt_to", default=q.get("to_time", "").strip())
    ap.add_argument("--limit", type=int, default=int(q.get("limit", "1000") or 0))
    ap.add_argument("--csv", default=q.get("csv_path", "").strip())
    ap.add_argument("--sql", default=None, help="직접 SQL 실행 (다른 옵션 무시)")
    ap.add_argument("--show", type=int, default=20, help="화면 출력 건수")
    args = ap.parse_args()

    conn = connect(cfg)
    try:
        if args.sql:
            sql, binds = args.sql, {}
        else:
            sql, binds = build_query(args.msg, args.dt_from, args.dt_to, args.limit)
        print("[SQL]\n%s" % sql)
        cols, rows = run(conn, sql, binds)
        print_rows(cols, rows, args.show)
        save_csv(args.csv, cols, rows)
    finally:
        conn.close()
        print("[CONN] closed")


if __name__ == "__main__":
    main()
