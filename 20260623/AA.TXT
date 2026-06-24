#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWS_IDC_DATA_HIS — 일자 선택 raw 다운로드
================================================================
aws_idc_realtime_collector.py 의 일자 선택 버전.

- 매분 루프 X → **지정한 날짜의 24시간 raw 한 번에** 다운로드
- 265개 IDC 컬럼 (raw_columns.RAW_COLS_265 사용)
- 출력: ./raw/M16A_HUBROOM_PR_{YYYYMMDD}.csv

사용:
  python aws_idc_일자다운로드.py <YYYYMMDD> [-o <출력폴더>]
  python aws_idc_일자다운로드.py <YYYYMMDD> <YYYYMMDD> [-o <출력폴더>]  # 시작~끝 다일자

예:
  python aws_idc_일자다운로드.py 20260525                  # 5/25 24시간
  python aws_idc_일자다운로드.py 20260525 20260531 -o raw  # 5/25~5/31 (각 일자별 1파일)

DB 접속 정보 (환경변수로 오버라이드 가능):
  ORA_USER  (기본 STAREAD)
  ORA_PASS  (기본 Stareadadmin123!)
  ORA_DSN   (기본 10.40.41.103:1521/ICASTARPP)
"""
import csv
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# raw_columns 에서 265 컬럼 가져옴 (단일 소스)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_columns import RAW_COLS_265

try:
    import oracledb
except ImportError:
    print("❌ oracledb 미설치 — pip install oracledb")
    sys.exit(1)

# ==========================================================
# 설정 (aws_idc_realtime_collector.py 와 동일 — env 로 오버라이드)
# ==========================================================
ORACLE_USER     = os.getenv("ORA_USER", "STAREAD")
ORACLE_PASSWORD = os.getenv("ORA_PASS", "Stareadadmin123!")
ORACLE_DSN      = os.getenv("ORA_DSN",  "10.40.41.103:1521/ICASTARPP")

CSV_HEADER = ["CRT_TM"] + RAW_COLS_265

# ==========================================================
# 로깅
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("idc_일자다운로드")


# ==========================================================
# SQL — 일자 범위 PIVOT (SYSDATE 대신 명시 날짜)
# ==========================================================
def build_sql():
    pivot_cols = ",\n  ".join(
        f"MAX(CASE WHEN IDC_NM='{nm}' THEN IDC_VAL END) AS \"{nm}\""
        for nm in RAW_COLS_265
    )
    in_list = ",\n    ".join(f"'{nm}'" for nm in RAW_COLS_265)
    return f"""
SELECT
  TO_CHAR(CRT_TM, 'YYYY-MM-DD HH24:MI:SS') AS CRT_TM,
  {pivot_cols}
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM >= TO_DATE(:t_start, 'YYYY-MM-DD HH24:MI:SS')
  AND CRT_TM <  TO_DATE(:t_end,   'YYYY-MM-DD HH24:MI:SS')
  AND IDC_NM IN (
    {in_list}
  )
GROUP BY CRT_TM
ORDER BY CRT_TM
""".strip()


SQL_QUERY = build_sql()


# ==========================================================
# 일자 다운로드
# ==========================================================
def parse_yyyymmdd(s):
    s = s.strip()
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, '%Y%m%d')
    if len(s) == 10:  # YYYY-MM-DD
        return datetime.strptime(s, '%Y-%m-%d')
    raise ValueError(f"날짜 형식 오류: {s} (YYYYMMDD 또는 YYYY-MM-DD)")


def fetch_day(conn, day_dt, out_dir):
    """day_dt 의 00:00 ~ 24:00 24시간 raw 다운로드 → CSV 저장."""
    t_start = day_dt.strftime('%Y-%m-%d 00:00:00')
    t_end = (day_dt + timedelta(days=1)).strftime('%Y-%m-%d 00:00:00')

    log.info(f"  쿼리: {t_start} ≤ CRT_TM < {t_end}")
    with conn.cursor() as cur:
        cur.execute(SQL_QUERY, t_start=t_start, t_end=t_end)
        rows = cur.fetchall()
    log.info(f"  수신: {len(rows)}행")

    # 분 단위 머지 (aws_idc_realtime_collector 와 동일)
    merged = {}
    for r in rows:
        if not r or r[0] is None: continue
        t_str = str(r[0])
        minute_key = t_str[:16] + ':00' if len(t_str) >= 16 else t_str
        if minute_key not in merged:
            merged[minute_key] = [minute_key] + [None] * (len(r) - 1)
        for i, v in enumerate(r[1:], 1):
            if v is not None and v != '':
                merged[minute_key][i] = v

    sorted_rows = [merged[k] for k in sorted(merged.keys())]

    fname = f"M16A_HUBROOM_PR_{day_dt.strftime('%Y%m%d')}.csv"
    out_path = Path(out_dir) / fname
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(CSV_HEADER)
        for r in sorted_rows:
            writer.writerow(["" if v is None else v for v in r])
    log.info(f"  ✅ 저장: {out_path}  ({len(sorted_rows)}분 / {len(RAW_COLS_265)} 컬럼)")
    return len(sorted_rows)


# ==========================================================
# 메인
# ==========================================================
def main():
    if len(sys.argv) < 2:
        print(__doc__); return

    # 날짜 인자 파싱
    date_args = []
    out_dir = './raw'
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == '-o' and i+1 < len(sys.argv):
            out_dir = sys.argv[i+1]; i += 2; continue
        date_args.append(a); i += 1

    if not date_args:
        print(__doc__); return

    # 단일 일자 또는 범위
    if len(date_args) == 1:
        d_start = d_end = parse_yyyymmdd(date_args[0])
    else:
        d_start = parse_yyyymmdd(date_args[0])
        d_end = parse_yyyymmdd(date_args[1])
        if d_end < d_start:
            d_start, d_end = d_end, d_start

    os.makedirs(out_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("AWS_IDC_DATA_HIS — 일자 다운로드")
    log.info(f"  DSN     : {ORACLE_DSN}")
    log.info(f"  USER    : {ORACLE_USER}")
    log.info(f"  기간    : {d_start.strftime('%Y-%m-%d')} ~ {d_end.strftime('%Y-%m-%d')}")
    log.info(f"  컬럼    : {len(RAW_COLS_265)}개 IDC + CRT_TM")
    log.info(f"  출력    : {out_dir}/")
    log.info("=" * 60)

    log.info("Oracle 연결 시도...")
    conn = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)
    log.info("Oracle 연결 성공")

    total_min = 0
    d = d_start
    while d <= d_end:
        log.info(f"━━ {d.strftime('%Y-%m-%d')} ━━")
        try:
            total_min += fetch_day(conn, d, out_dir)
        except Exception as e:
            log.exception(f"  실패: {e}")
        d += timedelta(days=1)

    try: conn.close()
    except: pass
    log.info(f"🎉 완료 — 총 {total_min}분 / {(d_end - d_start).days + 1}일")


if __name__ == '__main__':
    main()
