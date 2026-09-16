# -*- coding: utf-8 -*-
"""
AWS_IDC_DATA_HIS — 통합 실시간 수집기 v4.2 (265 + PIO 12 = 277개 컬럼)
================================================================
- 매 분 00초 동기 호출
- 윈도우: SYSDATE 기준 과거 WINDOW_MIN 분 (기본 90분)
- 컬럼: 265개 IDC + CRT_TM + PIO 12개
- 저장: ./predict/M16A_HUBROOM_PR.csv (덮어쓰기)

v4.1 → v4.2 변경 사항  (2026-09-16)
======================
★ PIO_ERROR 12개 컬럼 추가 — 경로별 DEPOSIT 반송실패 건수 (1분 단위)
    출처 : STA_TRANS_TIMEOUT_FAIL_HIS  (PIO_DATA_MAKE.py 의 SQL_FAIL 과 동일한 CASE)
    컬럼 : {경로}_PIOERROR_DEPOSITED  × 12

  · 기존 265개 컬럼의 순서·이름·값은 **하나도 바뀌지 않는다**. PIO 12개는 맨 뒤에만 붙는다.
  · PIO 조회가 실패해도 265컬럼 저장은 그대로 진행된다 (값만 0으로 채움).
    → PIO 때문에 수집기가 멈추는 일은 없다.
  · 끄고 싶으면 환경변수  PIO_ENABLE=0  하나면 된다. 코드를 되돌릴 필요 없다.
  · 실패 없는 분은 0 이다 (빈칸이 아니다).

v3.1 → v4.1 변경 사항
======================
- 64개 → 265개 컬럼으로 확장 (8영역 전체 수집)
- 추후 룰 추가/변경 시 수집기 수정 불필요
- SQL 자동 생성 (PIVOT MAX CASE WHEN), 컬럼 추가는 IDC_COLUMNS 리스트만 수정

영역별 컬럼 수:
  M16HUB  110개  (HUB 중심)
  M14B     42개  (M14B 7F)
  M14      41개  (M14A 3F)
  M16A     37개  (M16A 6F/2F)
  M16B     16개  (M16B 10F)
  M16      11개  (SFAB)
  M16_PKT   4개  (M16EUV)
  M16_WT    4개  (M16WT)
  ─────────
  합계    265개  (+ PIO 12개 = 277개)

v4.1 predictor (hubroom_predictor.py)가 사용하는 80+개 컬럼 모두 포함.
그 외 185개 컬럼도 함께 수집 → 추후 ML 학습/룰 추가 시 즉시 사용 가능.
"""

import os
import sys
import time
import csv
import logging
from datetime import datetime, timedelta      # ★ v4.2 PIO — 조회 구간 계산용
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
# ★ v4.2 PIO_ERROR 설정
# ==========================================================
#   PIO_ENABLE=0 이면 PIO 조회를 아예 하지 않는다.
#   그래도 12개 컬럼은 0 으로 채워 나간다 — 하류(예측기/영역분리)의 컬럼 수가 변하지 않게.
PIO_ENABLE = os.getenv("PIO_ENABLE", "1") != "0"

# 컬럼 순서 = PIO_DATA_MAKE.py 의 GUBUNS 와 동일 (고객 요청 순서)
PIO_GUBUNS = [
    'M16HUB->MLUD', 'M16HUB->M14B', 'M16HUB<-M14B',
    'M16HUB->M14A', 'M16HUB<-M14A',
    'M16HUB->M16A', 'M16HUB<-M16A',
    'M16A->M16B', 'M16B->M16A',
    'M14A->M14B', 'M14A<-M14B',
    'M14A->M10A',
]
PIO_SUFFIX  = '_PIOERROR_DEPOSITED'          # PIO_DATA_MAKE.py 와 같은 이름 규칙
PIO_COLUMNS = [g + PIO_SUFFIX for g in PIO_GUBUNS]

# PIO_DATA_MAKE.py 의 SQL_FAIL 그대로 (2026-09-16 CASE 수정본).
#   · EQP 컬럼은 쓰지 않으므로 뺐다 — GUBUN × 분 집계만 한다
#   · 구간은 COMPLT_TM 문자열(YYYYMMDDHH24MISS) 비교 — 인덱스를 그대로 탄다
#   · FC/FB/PN 별칭은 반드시 한 단계 아래 SELECT 에서 만든다.
#     같은 SELECT 안에서는 방금 만든 별칭을 참조할 수 없다 (ORA-00904).
SQL_PIO = """
SELECT GUBUN,
       TO_CHAR(GROUP1, 'YYYY-MM-DD HH24:MI') AS GROUP1,
       SUM(CASE WHEN FT = 'DEPOSIT' THEN 1 ELSE 0 END) AS DEPOSITED_FAIL_CNT
FROM (
    SELECT GROUP1, FT,
           CASE
               WHEN FC='M16' AND FB='M16HUB'             AND PN LIKE '6FIOB%' THEN 'M16HUB->MLUD'
               WHEN FC='M16' AND FB='M16HUB'             AND PN LIKE '4ABLD%' THEN 'M16HUB->M14B'
               WHEN FC='M14' AND FB='M14B'               AND PN LIKE '4ABLD%' THEN 'M16HUB<-M14B'
               WHEN FC='M16' AND FB='M16HUB'             AND PN LIKE '4AFC%'  THEN 'M16HUB->M14A'
               WHEN FC='M14' AND FB IN ('M14A','M14')    AND PN LIKE '4AFC%'  THEN 'M16HUB<-M14A'
               WHEN FC='M16' AND FB IN ('M16HUB','M14B') AND PN LIKE '6ABL%'  THEN 'M16HUB->M16A'
               WHEN FC='M16' AND FB='M16A'               AND PN LIKE '6ABL%'  THEN 'M16HUB<-M16A'
               WHEN FC='M16' AND FB='M16A'               AND PN LIKE '6ALF%'  THEN 'M16A->M16B'
               WHEN FC='M16' AND FB='M16B'               AND PN LIKE '6ALF%'  THEN 'M16B->M16A'
               WHEN FC='M14' AND FB IN ('M14A','M14')    AND PN LIKE '4ALF%'  THEN 'M14A->M14B'
               WHEN FC='M14' AND FB='M14B'               AND PN LIKE '4ALF%'  THEN 'M14A<-M14B'
               WHEN FC='M14' AND FB IN ('M10A','M10')    AND PN LIKE '4ABL%'  THEN 'M14A->M10A'
           END AS GUBUN
    FROM (
        SELECT TO_DATE(SUBSTR(A.COMPLT_TM, 1, 12), 'YYYYMMDDHH24MI') AS GROUP1,
               UPPER(TRIM(A.FAIL_TYP)) AS FT,
               UPPER(TRIM(A.FAC_ID))   AS FC,
               UPPER(TRIM(A.FAB_ID))   AS FB,
               UPPER(TRIM(A.PORT_NM))  AS PN
        FROM STA_TRANS_TIMEOUT_FAIL_HIS A
        WHERE A.COMPLT_TM >= :t_from
          AND A.COMPLT_TM <  :t_to
          AND A.FAC_ID IN ('M14', 'M16')
    )
)
WHERE GUBUN IS NOT NULL
GROUP BY GUBUN, GROUP1
ORDER BY GROUP1, GUBUN
"""

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
log = logging.getLogger("idc_collector_v42")

# ==========================================================
# IDC 컬럼 (v4.1 — 265개 전체) — v4.2 에서 변경 없음
# ==========================================================
IDC_COLUMNS = [
    # ===== M16HUB (110개) =====
    "M16HUB.CNV.SENDFAB.TO_M14A_CURRENTQCNT",
    "M16HUB.LFT.6ABL0111.2F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0111.2F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0111.3F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0111.3F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0111.6F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0111.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0111.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL0112.2F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0112.2F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0112.3F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0112.3F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0112.6F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0112.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0112.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL0121.2F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0121.2F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0121.3F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0121.3F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0121.6F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0121.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL0122.2F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0122.3F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0122.3F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0122.6F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0122.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL0122.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL6011.2F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6011.2F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6011.3F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6011.6F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6011.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6011.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL6012.2F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6012.2F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6012.3F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6012.3F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6012.6F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6012.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6012.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL6021.2F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6021.2F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6021.3F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6021.3F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6021.6F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6021.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6021.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL6022.2F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6022.2F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6022.3F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6022.3F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6022.6F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6022.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6022.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL6031.2F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6031.2F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6031.3F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6031.3F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6031.6F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6031.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6031.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.6ABL6032.2F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6032.3F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6032.3F_TO_6F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6032.6F_TO_2F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6032.6F_TO_3F_CURRENTQCNT",
    "M16HUB.LFT.6ABL6032.TOTAL_CURRENTQCNT",
    "M16HUB.LFT.SENDFAB.TO_M14B_CURRENTQCNT",
    "M16HUB.LFT.SENDFAB.TO_M16A_CURRENTQCNT",
    "M16HUB.LFT.SENDFAB.TO_M16E_CURRENTQCNT",
    "M16HUB.OHT.ALERT.OHTMCPALARMCNT",
    "M16HUB.QUE.ABN.AOTRANSDELAY",
    "M16HUB.QUE.ALL.3F_CMD",
    "M16HUB.QUE.ALL.3F_TO_3F_MLUD_JOB",
    "M16HUB.QUE.ALL.3F_TO_M14A_3F_JOB",
    "M16HUB.QUE.ALL.3F_TO_M14B_7F_JOB",
    "M16HUB.QUE.ALL.3F_TO_M16A_2F_JOB",
    "M16HUB.QUE.ALL.3F_TO_M16A_6F_JOB",
    "M16HUB.QUE.ALL.CURRENTQCNT",
    "M16HUB.QUE.ALL.CURRENTQCOMPLETED",
    "M16HUB.QUE.ALL.CURRENTQCREATED",
    "M16HUB.QUE.ALL.CURRENT_M16A_3F_JOB",
    "M16HUB.QUE.ALL.CURRENT_M16A_3F_JOB_2",
    "M16HUB.QUE.ALL.FABTRANSJOBCNT",
    "M16HUB.QUE.ALL.M16HUBTOM14MANUAL_CURRENTQCNT",
    "M16HUB.QUE.ALL.TRANSPORT4MINOVERCNT",
    "M16HUB.QUE.ALL.TRANSPORT4MINOVERRATIO",
    "M16HUB.QUE.ALL.TRANSPORT4MINOVERTIMEAVG",
    "M16HUB.QUE.CNV.3F_CNV_MAXCAPA",
    "M16HUB.QUE.CNV.3F_TO_M14A_CNV_AI_CMD",
    "M16HUB.QUE.LFT.3F_LFT_MAXCAPA",
    "M16HUB.QUE.LFT.3F_M14BLFT_MAXCAPA",
    "M16HUB.QUE.LFT.3F_TO_M14B_LFT_AI_CMD",
    "M16HUB.QUE.LFT.3F_TO_M16A_LFT_AI_CMD",
    "M16HUB.QUE.LOAD.AVGLOADTIME",
    "M16HUB.QUE.M14ATOM16.MESCURRENTQCNT",
    "M16HUB.QUE.M14BTOM16.MESCURRENTQCNT",
    "M16HUB.QUE.M14TOM16.MESCURRENTQCNT",
    "M16HUB.QUE.M16TOM14.MESCURRENTQCNT",
    "M16HUB.QUE.M16TOM14A.MESCURRENTQCNT",
    "M16HUB.QUE.M16TOM14B.MESCURRENTQCNT",
    "M16HUB.QUE.MLUD.3F_TO_M16A_MLUD_AI_CMD",
    "M16HUB.QUE.OHT.CURRENTOHTQCNT",
    "M16HUB.QUE.OHT.OHTUTIL",
    "M16HUB.QUE.STB.3F_TO_M16A_3F_STB_CMD",
    "M16HUB.QUE.TIME.AVGTOTALTIME",
    "M16HUB.QUE.TIME.AVGTOTALTIME1MIN",
    "M16HUB.STRATE.ALL.FABSTORAGERATIO",
    "M16HUB.STRATE.STB.3F_STORAGE_UTIL",
    "M16HUB.STRATE.STK.STORAGERATIO",
    # ===== M14 (41개) =====
    "M14.CNV.SENDFAB.TO_M16HUB_CURRENTQCNT",
    "M14.OHT.STATECNT.ABNORMAL",
    "M14.OHT.STATECNT.CONGESTED",
    "M14.OHT.STATECNT.HTSTOP",
    "M14.OHT.STATECNT.OBSANDBZSTOP",
    "M14.QUE.ABN.AOTRANSDELAY",
    "M14.QUE.ALL.3F_TO_HUB_JOB",
    "M14.QUE.ALL.3F_TO_HUB_JOB_ALT",
    "M14.QUE.ALL.CURRENTQCNT",
    "M14.QUE.ALL.CURRENTQCOMPLETED",
    "M14.QUE.ALL.CURRENTQCREATED",
    "M14.QUE.ALL.TOTALCNVCURRENTQCNT",
    "M14.QUE.ALL.TRANSPORT4MINOVERCNT",
    "M14.QUE.ALL.TRANSPORT4MINOVERRATIO",
    "M14.QUE.ALL.TRANSPORT4MINOVERTIMEAVG",
    "M14.QUE.CNV.3F_CNV_MAXCAPA",
    "M14.QUE.CNV.ALLTONORTHCNVCURRENTQCNT",
    "M14.QUE.CNV.ALLTOSOUTHCNVCURRENTQCNT",
    "M14.QUE.CNV.M14ATOM16ACURRNETQCNT",
    "M14.QUE.CNV.M14ATOM16CURRNETQCNT",
    "M14.QUE.CNV.M14ATONORTHCURRENTQCNT",
    "M14.QUE.CNV.M14ATOSOUTHCURRENTQCNT",
    "M14.QUE.CNV.NORTHCNVTOALLCURRENTQCNT",
    "M14.QUE.CNV.NORTHCNVTOM14TIME",
    "M14.QUE.CNV.NORTHCNVTOM14TIME1MIN",
    "M14.QUE.CNV.NORTHCURRENTQCNT",
    "M14.QUE.CNV.NORTHM14TOCNVTIME",
    "M14.QUE.CNV.NORTHM14TOCNVTIME1MIN",
    "M14.QUE.CNV.SOUTHCNVTOALLCURRENTQCNT",
    "M14.QUE.CNV.SOUTHCNVTOM14TIME",
    "M14.QUE.CNV.SOUTHCNVTOM14TIME1MIN",
    "M14.QUE.CNV.SOUTHCURRENTQCNT",
    "M14.QUE.CNV.SOUTHM14TOCNVTIME",
    "M14.QUE.CNV.SOUTHM14TOCNVTIME1MIN",
    "M14.QUE.LOAD.AVGLOADTIME",
    "M14.QUE.LOAD.AVGLOADTIME1MIN",
    "M14.QUE.OHT.3F_TO_HUB_CMD",
    "M14.QUE.OHT.OHTUTIL",
    "M14.QUE.SFAB.SENDTOM16",
    "M14.SORTER.ABN.CUSORTERWAITCOUNTOVER",
    "M14.SORTER.ABN.SORTERWAITCOUNTOVER",
    # ===== M14B (42개) =====
    "M14B.LFT.4ABLD111.4F_TO_7F_CURRENTQCNT",
    "M14B.LFT.4ABLD111.7F_TO_4F_CURRENTQCNT",
    "M14B.LFT.4ABLD111.TOTAL_CURRENTQCNT",
    "M14B.LFT.4ABLD112.4F_TO_7F_CURRENTQCNT",
    "M14B.LFT.4ABLD112.7F_TO_4F_CURRENTQCNT",
    "M14B.LFT.4ABLD112.TOTAL_CURRENTQCNT",
    "M14B.LFT.4ABLD121.4F_TO_7F_CURRENTQCNT",
    "M14B.LFT.4ABLD121.7F_TO_4F_CURRENTQCNT",
    "M14B.LFT.4ABLD121.TOTAL_CURRENTQCNT",
    "M14B.LFT.4ABLD122.4F_TO_7F_CURRENTQCNT",
    "M14B.LFT.4ABLD122.7F_TO_4F_CURRENTQCNT",
    "M14B.LFT.4ABLD122.TOTAL_CURRENTQCNT",
    "M14B.LFT.4ABLD131.4F_TO_7F_CURRENTQCNT",
    "M14B.LFT.4ABLD131.7F_TO_4F_CURRENTQCNT",
    "M14B.LFT.4ABLD131.TOTAL_CURRENTQCNT",
    "M14B.LFT.4ABLD132.4F_TO_7F_CURRENTQCNT",
    "M14B.LFT.4ABLD132.7F_TO_4F_CURRENTQCNT",
    "M14B.LFT.4ABLD132.TOTAL_CURRENTQCNT",
    "M14B.LFT.SENDFAB.TO_M14A_CURRENTQCNT",
    "M14B.LFT.SENDFAB.TO_M16HUB_CURRENTQCNT",
    "M14B.OHT.ALERT.OHTMCPALARMCNT",
    "M14B.QUE.ABN.AOTRANSDELAY",
    "M14B.QUE.ALL.7F_TO_HUB_JOB",
    "M14B.QUE.ALL.7F_TO_HUB_JOB_ALT",
    "M14B.QUE.ALL.CURRENTQCNT",
    "M14B.QUE.ALL.CURRENTQCOMPLETED",
    "M14B.QUE.ALL.CURRENTQCREATED",
    "M14B.QUE.LFT.ALLTOLFTCURRENTQCNT",
    "M14B.QUE.LFT.LFTTOALLCURRENTQCNT",
    "M14B.QUE.LFT.M14BTOM16ACURRNETQCNT",
    "M14B.QUE.LOAD.AVGLOADTIME",
    "M14B.QUE.LOAD.AVGLOADTIME1MIN",
    "M14B.QUE.LOAD.CURRENTLOADQCNT",
    "M14B.QUE.OHT.7F_TO_HUB_CMD",
    "M14B.QUE.OHT.CURRENTOHTQCNT",
    "M14B.QUE.OHT.OHTUTIL",
    "M14B.QUE.SENDFAB.VERTICALQUEUECOUNT",
    "M14B.QUE.TIME.AVGTOTALTIME",
    "M14B.QUE.TIME.AVGTOTALTIME1MIN",
    "M14B.SORTER.ABN.CUSORTERWAITCOUNTOVER",
    "M14B.SORTER.ABN.SORTERWAITCOUNTOVER",
    "M14B.SORTER.ABN.SORTERWAITCOUNTOVER_B01",
    # ===== M16A (37개) =====
    "M16A.LFT.SENDFAB.TO_M16B_CURRENTQCNT",
    "M16A.LFT.SENDFAB.TO_M16E_CURRENTQCNT",
    "M16A.LFT.SENDFAB.TO_M16HUB_CURRENTQCNT",
    "M16A.QUE.ABN.AOTRANSDELAY",
    "M16A.QUE.ALL.2F_TO_6F_JOB",
    "M16A.QUE.ALL.2F_TO_HUB_JOB",
    "M16A.QUE.ALL.2F_TO_HUB_JOB_ALT",
    "M16A.QUE.ALL.6F_TO_2F_JOB",
    "M16A.QUE.ALL.6F_TO_HUB_JOB",
    "M16A.QUE.ALL.6F_TO_HUB_JOB_ALT",
    "M16A.QUE.ALL.CURRENTQCNT",
    "M16A.QUE.ALL.CURRENTQCOMPLETED",
    "M16A.QUE.ALL.CURRENTQCREATED",
    "M16A.QUE.ALL.TRANSPORT4MINOVERCNT",
    "M16A.QUE.ALL.TRANSPORT4MINOVERRATIO",
    "M16A.QUE.ALL.TRANSPORT4MINOVERTIMEAVG",
    "M16A.QUE.CNV.ALLTONORTHCNVCURRENTQCNT",
    "M16A.QUE.CNV.ALLTOSOUTHCNVCURRENTQCNT",
    "M16A.QUE.CNV.M16ATOM14ACURRNETQCNT",
    "M16A.QUE.CNV.M16ATOM14BCURRNETQCNT",
    "M16A.QUE.CNV.M16TOM14ACURRNETQCNT",
    "M16A.QUE.CNV.M16TOM14BCURRNETQCNT",
    "M16A.QUE.CNV.NORTHCNVTOALLCURRENTQCNT",
    "M16A.QUE.CNV.SOUTHCNVTOALLCURRENTQCNT",
    "M16A.QUE.LFT.2F_LFT_MAXCAPA",
    "M16A.QUE.LFT.6F_LFT_MAXCAPA",
    "M16A.QUE.LFT.ALLTOLFTCURRENTQCNT",
    "M16A.QUE.LFT.LFTTOALLCURRENTQCNT",
    "M16A.QUE.LOAD.AVGFOUPLOADTIME",
    "M16A.QUE.LOAD.AVGLOADTIME1MIN",
    "M16A.QUE.LOAD.CURRENTLOADQCNT",
    "M16A.QUE.OHT.2F_TO_HUB_CMD",
    "M16A.QUE.OHT.6F_TO_HUB_CMD",
    "M16A.QUE.OHT.CURRENTOHTQCNT",
    "M16A.QUE.OHT.OHTUTIL",
    "M16A.SORTER.ABN.CUSORTERWAITCOUNTOVER",
    "M16A.SORTER.ABN.SORTERWAITCOUNTOVER",
    # ===== M16B (16개) =====
    "M16B.LFT.SENDFAB.TO_M16A_CURRENTQCNT",
    "M16B.QUE.ABN.AOTRANSDELAY",
    "M16B.QUE.ALL.10F_TO_HUB_JOB",
    "M16B.QUE.ALL.CURRENTQCNT",
    "M16B.QUE.ALL.CURRENTQCOMPLETED",
    "M16B.QUE.ALL.CURRENTQCREATED",
    "M16B.QUE.ALL.TRANSPORT4MINOVERCNT",
    "M16B.QUE.ALL.TRANSPORT4MINOVERRATIO",
    "M16B.QUE.ALL.TRANSPORT4MINOVERTIMEAVG",
    "M16B.QUE.LOAD.AVGFOUPLOADTIME",
    "M16B.QUE.LOAD.AVGLOADTIME1MIN",
    "M16B.QUE.LOAD.CURRENTLOADQCNT",
    "M16B.QUE.OHT.CURRENTOHTQCNT",
    "M16B.QUE.OHT.OHTUTIL",
    "M16B.SORTER.ABN.CUSORTERWAITCOUNTOVER",
    "M16B.SORTER.ABN.SORTERWAITCOUNTOVER",
    # ===== M16 (11개) =====
    "M16.CNV.SENDFAB.TO_M16WT_CURRENTQCNT",
    "M16.QUE.SFAB.COMPLETEQUEUETOTAL",
    "M16.QUE.SFAB.COMPLETETOM10",
    "M16.QUE.SFAB.COMPLETETOM14",
    "M16.QUE.SFAB.RECEIVEQUEUETOTAL",
    "M16.QUE.SFAB.RETURNQUEUETOTAL",
    "M16.QUE.SFAB.RETURNTOM10",
    "M16.QUE.SFAB.RETURNTOM14",
    "M16.QUE.SFAB.SENDQUEUETOTAL",
    "M16.QUE.SFAB.SENDTOM10",
    "M16.QUE.SFAB.SENDTOM14",
    # ===== M16_PKT (4개) =====
    "M16_PKT.OHT.ALERT.OHTMCPALARMCNT",
    "M16_PKT.QUE.ABN.AOTRANSDELAY",
    "M16_PKT.QUE.OHT.OHTUTIL",
    "M16_PKT.QUE.TIME.AVGTOTALTIME1MIN",
    # ===== M16_WT (4개) =====
    "M16_WT.OHT.ALERT.OHTMCPALARMCNT",
    "M16_WT.QUE.ABN.AOTRANSDELAY",
    "M16_WT.QUE.OHT.OHTUTIL",
    "M16_WT.QUE.TIME.AVGTOTALTIME1MIN",
]
# 총 265개
# ★ v4.2 — PIO 12개를 맨 뒤에만 붙인다. 기존 265개 위치는 그대로다.
CSV_HEADER = ["CRT_TM"] + IDC_COLUMNS + PIO_COLUMNS

# ==========================================================
# SQL — PIVOT MAX CASE WHEN
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
# ★ v4.2 PIO 조회
# ==========================================================
def fetch_pio(conn, minute_keys) -> dict:
    """
    minute_keys 구간의 경로별 DEPOSIT 실패 건수를 읽는다.

      반환 : {'YYYY-MM-DD HH:MM': {경로: 건수}}
             실패 없는 분은 키 자체가 없다 (호출부에서 0 으로 채운다)

    ★ 이 함수는 어떤 경우에도 예외를 밖으로 내보내지 않는다.
       PIO 쪽 문제로 265컬럼 수집이 멈추면 안 되기 때문이다.
       조회가 안 되면 빈 dict 를 돌려주고, 그 사이클의 PIO 값은 전부 0 이 된다.
    """
    if not PIO_ENABLE or not minute_keys:
        return {}

    cur = None
    try:
        # IDC 로 실제 들어온 분 구간에 맞춘다 (Python 시계와 DB SYSDATE 차이를 피한다)
        lo = min(minute_keys)[:16]          # 'YYYY-MM-DD HH:MM'
        hi = max(minute_keys)[:16]
        t_from = datetime.strptime(lo, '%Y-%m-%d %H:%M')
        t_to   = datetime.strptime(hi, '%Y-%m-%d %H:%M') + timedelta(minutes=1)

        cur = conn.cursor()
        cur.execute(SQL_PIO, {'t_from': t_from.strftime('%Y%m%d%H%M%S'),
                              't_to':   t_to.strftime('%Y%m%d%H%M%S')})
        out = {}
        n = 0
        for gubun, gmin, cnt in cur.fetchall():
            if not gubun or not gmin:
                continue
            out.setdefault(str(gmin)[:16], {})[str(gubun)] = int(cnt or 0)
            n += 1
        log.info(f"  PIO: {t_from:%m/%d %H:%M}~{t_to:%H:%M} → 실패 있는 (분×경로) {n}건")
        return out
    except Exception as e:
        # 여기서 삼킨다. 절대 위로 던지지 않는다.
        log.warning(f"  PIO 조회 실패 — 이번 사이클 PIO 값은 0 으로 둔다: {e}")
        return {}
    finally:
        try:
            if cur is not None:
                cur.close()
        except Exception:
            pass


# ==========================================================
# 수집
# ==========================================================
def fetch_and_save(conn) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with conn.cursor() as cur:
        cur.execute(SQL_QUERY, window_min=WINDOW_MIN)
        rows = cur.fetchall()

    # ★ Python 분 단위 머지 (SQL 이 혹시 여러 행 줘도 분당 1행 보장)
    # 같은 분 (CRT_TM 의 'YYYY-MM-DD HH:MM') 의 행들을 한 행으로 합침
    # 컬럼별로 비-NULL 값 채택 (None / '' 은 무시하고 다른 행 값 사용)
    merged = {}  # {minute_key: row_list}
    for r in rows:
        if not r or r[0] is None:
            continue
        # 분 단위 키 추출 ('2026-06-02 12:56:07' → '2026-06-02 12:56:00')
        t_str = str(r[0])
        if len(t_str) >= 16:
            minute_key = t_str[:16] + ':00'
        else:
            minute_key = t_str
        if minute_key not in merged:
            # 새 분 — 빈 행으로 시작
            merged[minute_key] = [minute_key] + [None] * (len(r) - 1)
        # 비-NULL 값으로 갱신
        for i, v in enumerate(r[1:], 1):
            if v is not None and v != '':
                merged[minute_key][i] = v

    # 분 키 정렬
    sorted_keys = sorted(merged.keys())

    # ★ v4.2 — PIO 12칸을 각 행 뒤에 붙인다. 조회 실패 시엔 전부 0.
    pio_by_min = fetch_pio(conn, sorted_keys)
    sorted_rows = []
    for k in sorted_keys:
        per = pio_by_min.get(k[:16]) or {}
        sorted_rows.append(merged[k] + [per.get(g, 0) for g in PIO_GUBUNS])

    with open(TMP_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(CSV_HEADER)
        for r in sorted_rows:
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
    log.info("AWS_IDC_DATA_HIS v4.2 통합 실시간 수집기 시작")
    log.info(f"  DSN     : {ORACLE_DSN}")
    log.info(f"  USER    : {ORACLE_USER}")
    log.info(f"  WINDOW  : 과거 {WINDOW_MIN}분")
    log.info(f"  INTERVAL: {INTERVAL_SEC}초 (매분 00초 동기)")
    log.info(f"  OUTPUT  : {OUTPUT_FILE}")
    log.info(f"  COLUMNS : {len(IDC_COLUMNS)}개 IDC + CRT_TM + PIO {len(PIO_COLUMNS)}개")
    log.info(f"  PIO     : {'ON' if PIO_ENABLE else 'OFF (PIO_ENABLE=0)'} "
             f"— STA_TRANS_TIMEOUT_FAIL_HIS")
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
