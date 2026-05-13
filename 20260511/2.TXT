# -*- coding: utf-8 -*-
"""
AWS_IDC_DATA_HIS — M16HUB + M14/M14B/M16_PKT/M16_WT 통합 수집 v3
================================================================
- 매 분 00초 동기 호출
- 윈도우: SYSDATE 기준 과거 WINDOW_MIN 분
- 컬럼: 60개 IDC + CRT_TM
- 저장: ./predict/M16A_HUBROOM_PR.csv (덮어쓰기)
"""

import os
import sys
import time
import csv
import logging
from pathlib import Path

import oracledb

# ==========================================================
# 설정
# ==========================================================
ORACLE_USER     = os.getenv("ORA_USER", "STAREAD")
ORACLE_PASSWORD = os.getenv("ORA_PASS", "Stareadadmin123!")
ORACLE_DSN      = os.getenv("ORA_DSN",  "10.40.41.103:1521/ICASTARPP")

WINDOW_MIN      = 90
INTERVAL_SEC    = 60

OUTPUT_DIR      = Path(__file__).resolve().parent / "predict"
OUTPUT_FILE     = OUTPUT_DIR / "M16A_HUBROOM_PR.csv"
TMP_FILE        = OUTPUT_DIR / "M16A_HUBROOM_PR.csv.tmp"

# ==========================================================
# 로깅
# ==========================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "collector.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("idc_collector")

# ==========================================================
# IDC 컬럼 (v3 — 60개)
# ==========================================================
IDC_COLUMNS = [
    # M16HUB 기존 32
    "M16HUB.QUE.ALL.CURRENTQCNT",
    "M16HUB.QUE.ALL.CURRENTQCOMPLETED",
    "M16HUB.QUE.OHT.CURRENTOHTQCNT",
    "M16HUB.QUE.OHT.OHTUTIL",
    "M16HUB.QUE.LOAD.AVGLOADTIME",
    "M16HUB.QUE.ALL.TRANSPORT4MINOVERCNT",
    "M16HUB.QUE.ALL.TRANSPORT4MINOVERRATIO",
    "M16HUB.QUE.ALL.TRANSPORT4MINOVERTIMEAVG",
    "M16HUB.QUE.TIME.AVGTOTALTIME",
    "M16HUB.QUE.TIME.AVGTOTALTIME1MIN",
    "M16HUB.OHT.ALERT.OHTMCPALARMCNT",
    "M16HUB.QUE.M14TOM16.MESCURRENTQCNT",
    "M16HUB.QUE.M16TOM14.MESCURRENTQCNT",
    "M16HUB.QUE.M16TOM14A.MESCURRENTQCNT",
    "M16HUB.QUE.M16TOM14B.MESCURRENTQCNT",
    "M16HUB.QUE.ALL.FABTRANSJOBCNT",
    "M16HUB.QUE.M14ATOM16.MESCURRENTQCNT",
    "M16HUB.QUE.M14BTOM16.MESCURRENTQCNT",
    "M16HUB.LFT.6ABL6011.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL6012.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL6021.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL6022.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL6031.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL6032.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL0111.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL0112.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL0121.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL0122.TOTAL_CURRENTQCNT",
    "M16HUB.QUE.ALL.M16HUBTOM14MANUAL_CURRENTQCNT",
    "M16HUB.STRATE.STK.STORAGERATIO",
    "M16HUB.STRATE.ALL.FABSTORAGERATIO",
    # 32번째는 위에 다 들어감 - 31개. 원본 v3 헤더와 맞춤 (M16HUB 31개)
    # ★★★ M14 STATECNT (4)
    "M14.OHT.STATECNT.HTSTOP",
    "M14.OHT.STATECNT.CONGESTED",
    "M14.OHT.STATECNT.ABNORMAL",
    "M14.OHT.STATECNT.OBSANDBZSTOP",
    # ★★ M14B 트래픽/지연 (6)
    "M14B.QUE.OHT.OHTUTIL",
    "M14B.QUE.OHT.CURRENTOHTQCNT",
    "M14B.QUE.TIME.AVGTOTALTIME",
    "M14B.QUE.TIME.AVGTOTALTIME1MIN",
    "M14B.QUE.ABN.AOTRANSDELAY",
    "M14B.OHT.ALERT.OHTMCPALARMCNT",
    # ★★★ M14B 7F 리프터 4ABLDxxx (6)
    "M14B.LFT.4ABLD111.TOTAL_CURRENTQCNT",
    "M14B.LFT.4ABLD112.TOTAL_CURRENTQCNT",
    "M14B.LFT.4ABLD121.TOTAL_CURRENTQCNT",
    "M14B.LFT.4ABLD122.TOTAL_CURRENTQCNT",
    "M14B.LFT.4ABLD131.TOTAL_CURRENTQCNT",
    "M14B.LFT.4ABLD132.TOTAL_CURRENTQCNT",
    # ★★ M14B 7F→HUB (3)
    "M14B.QUE.ALL.7F_TO_HUB_JOB",
    "M14B.QUE.ALL.7F_TO_HUB_JOB_ALT",
    "M14B.QUE.OHT.7F_TO_HUB_CMD",
    # ★ M14B Send Fab (1)
    "M14B.LFT.SENDFAB.TO_M16HUB_CURRENTQCNT",
    # ★★ M16_PKT 브릿지 (4)
    "M16_PKT.QUE.OHT.OHTUTIL",
    "M16_PKT.QUE.TIME.AVGTOTALTIME1MIN",
    "M16_PKT.QUE.ABN.AOTRANSDELAY",
    "M16_PKT.OHT.ALERT.OHTMCPALARMCNT",
    # ★★ M16_WT 브릿지 (4)
    "M16_WT.QUE.OHT.OHTUTIL",
    "M16_WT.QUE.TIME.AVGTOTALTIME1MIN",
    "M16_WT.QUE.ABN.AOTRANSDELAY",
    "M16_WT.OHT.ALERT.OHTMCPALARMCNT",
]

CSV_HEADER = ["CRT_TM"] + IDC_COLUMNS

# ==========================================================
# SQL — v3 쿼리 (PIVOT MAX CASE WHEN)
# ==========================================================
def build_sql() -> str:
    pivot_cols = ",\n  ".join(
        f"MAX(CASE WHEN IDC_NM='{nm}' THEN IDC_VAL END) AS \"{nm}\""
        for nm in IDC_COLUMNS
    )
    in_list = ",\n    ".join(f"'{nm}'" for nm in IDC_COLUMNS)
    return f"""
SELECT
  TO_CHAR(CRT_TM, 'YYYY-MM-DD HH24:MI:SS') AS CRT_TM,
  {pivot_cols}
FROM AWS_IDC_DATA_HIS
WHERE CRT_TM BETWEEN SYSDATE - :window_min/1440 AND SYSDATE
  AND IDC_NM IN (
    {in_list}
  )
GROUP BY CRT_TM
ORDER BY CRT_TM
""".strip()


SQL_QUERY = build_sql()


# ==========================================================
# 수집
# ==========================================================
def fetch_and_save(conn) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with conn.cursor() as cur:
        cur.execute(SQL_QUERY, window_min=WINDOW_MIN)
        rows = cur.fetchall()

    with open(TMP_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(CSV_HEADER)
        for r in rows:
            writer.writerow(["" if v is None else v for v in r])

    # 윈도우 파일잠금 대비 재시도
    for attempt in range(10):
        try:
            os.replace(TMP_FILE, OUTPUT_FILE)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)
    return len(rows)


def sleep_until_next_minute():
    """다음 분 00초까지 대기."""
    now = time.time()
    wait = INTERVAL_SEC - (now % INTERVAL_SEC)
    if wait < 0.05:
        wait += INTERVAL_SEC
    time.sleep(wait)


# ==========================================================
# 메인
# ==========================================================
def main():
    log.info("=" * 60)
    log.info("AWS_IDC_DATA_HIS v3 실시간 수집기 시작")
    log.info(f"  DSN     : {ORACLE_DSN}")
    log.info(f"  USER    : {ORACLE_USER}")
    log.info(f"  WINDOW  : 과거 {WINDOW_MIN}분")
    log.info(f"  INTERVAL: {INTERVAL_SEC}초 (매분 00초 동기)")
    log.info(f"  OUTPUT  : {OUTPUT_FILE}")
    log.info(f"  COLUMNS : {len(IDC_COLUMNS)}개 IDC + CRT_TM")
    log.info("=" * 60)

    log.info("매 분 정각(00초)까지 대기 중...")
    sleep_until_next_minute()

    conn = None
    while True:
        cycle_start = time.time()
        try:
            if conn is None:
                log.info("Oracle 연결 시도...")
                conn = oracledb.connect(
                    user=ORACLE_USER,
                    password=ORACLE_PASSWORD,
                    dsn=ORACLE_DSN,
                )
                log.info("Oracle 연결 성공")

            n = fetch_and_save(conn)
            elapsed = time.time() - cycle_start
            log.info(f"저장 완료: {n}행, {elapsed:.2f}s → {OUTPUT_FILE.name}")

        except oracledb.DatabaseError as e:
            log.error(f"DB 오류: {e}")
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
            conn = None
            time.sleep(5)
            continue

        except KeyboardInterrupt:
            log.info("사용자 중단 (Ctrl+C)")
            break

        except Exception as e:
            log.exception(f"예상치 못한 오류: {e}")

        sleep_until_next_minute()

    if conn:
        try:
            conn.close()
        except Exception:
            pass
    log.info("종료")


if __name__ == "__main__":
    main()
