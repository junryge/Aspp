#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWS_IDC_DATA_HIS — 날짜 지정 분단위 피벗 추출 (백테스트용)
=========================================================
aws_idc_realtime_collector.py 를 기반으로, **실시간 윈도우(SYSDATE-90분)**
대신 **사용자가 지정한 날짜 범위** 로 한 번 추출하고 CSV 저장 후 종료.

특징:
 - collector 의 분단위 머지 로직 그대로 사용 (분당 1행 보장)
 - csv.writer 자동 quoting (RFC 4180) → Excel 함정 회피
 - 출력 = hubroom_predictor.py 입력으로 바로 사용 가능

[설치]
    pip install oracledb

[환경변수 (회사 PC 에 이미 설정돼 있으면 생략)]
    set ORA_USER=STAREAD
    set ORA_PASS=Stareadadmin123!
    set ORA_DSN=10.40.41.103:1521/ICASTARPP

[사용]
    # 하루치
    python aws_idc_AS.py 2026-05-15

    # 날짜 범위
    python aws_idc_AS.py 2026-05-01 2026-05-08

    # 출력 폴더 지정 (기본: ./predict)
    python aws_idc_AS.py 2026-05-15 -o C:\\DATA

[출력 파일명]
    2026_05_15_idc.csv  (하루치)
    20260501_20260508_idc.csv  (범위)
"""
import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    import oracledb
except ImportError:
    sys.exit("ERROR: oracledb 모듈 필요 → pip install oracledb")

# ==========================================================
# 설정 (collector 와 동일)
# ==========================================================
ORACLE_USER = os.getenv("ORA_USER", "STAREAD")
ORACLE_PASSWORD = os.getenv("ORA_PASS", "Stareadadmin123!")
ORACLE_DSN = os.getenv("ORA_DSN", "10.40.41.103:1521/ICASTARPP")

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "predict"

# ==========================================================
# 로깅
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("idc_AS")

# ==========================================================
# IDC 컬럼 (v4.1 — 265개 전체, collector 와 완전 동일)
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

CSV_HEADER = ["CRT_TM"] + IDC_COLUMNS


# ==========================================================
# SQL 빌드 — 날짜 범위 버전 (window_min → start_dt/end_dt)
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
WHERE CRT_TM >= TO_DATE(:start_dt, 'YYYY-MM-DD HH24:MI:SS')
  AND CRT_TM <  TO_DATE(:end_dt,   'YYYY-MM-DD HH24:MI:SS')
  AND IDC_NM IN (
    {in_list}
  )
GROUP BY CRT_TM
ORDER BY CRT_TM
""".strip()


SQL_QUERY = build_sql()


# ==========================================================
# 분단위 머지 (collector 와 동일 로직)
# ==========================================================
def merge_by_minute(rows):
    """같은 분 ('YYYY-MM-DD HH:MM') 의 행들을 1행으로 합침.
    컬럼별 비-NULL 값 채택 (collector fetch_and_save 와 동일)."""
    merged = {}  # {minute_key: row_list}
    for r in rows:
        if not r or r[0] is None:
            continue
        t_str = str(r[0])
        if len(t_str) >= 16:
            minute_key = t_str[:16] + ':00'
        else:
            minute_key = t_str
        if minute_key not in merged:
            merged[minute_key] = [minute_key] + [None] * (len(r) - 1)
        for i, v in enumerate(r[1:], 1):
            if v is not None and v != '':
                merged[minute_key][i] = v
    return [merged[k] for k in sorted(merged.keys())]


# ==========================================================
# 추출
# ==========================================================
def extract(start_dt: datetime, end_dt: datetime, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 출력 파일명
    one_day = (end_dt - start_dt) == timedelta(days=1) and start_dt.time() == datetime.min.time()
    if one_day:
        fname = f"{start_dt.strftime('%Y_%m_%d')}_idc.csv"
    else:
        fname = f"{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}_idc.csv"
    out_path = out_dir / fname

    log.info("=" * 60)
    log.info("AWS_IDC_DATA_HIS 날짜지정 추출")
    log.info(f"  DSN     : {ORACLE_DSN}")
    log.info(f"  기간    : {start_dt} ~ {end_dt}  (end 미포함)")
    log.info(f"  컬럼    : CRT_TM + {len(IDC_COLUMNS)}개 IDC")
    log.info(f"  출력    : {out_path}")
    log.info("=" * 60)

    t0 = time.time()
    log.info("[1/4] Oracle 접속...")
    conn = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)

    try:
        log.info(f"[2/4] 쿼리 실행 (예상 ~{int((end_dt-start_dt).total_seconds()/60)}분)")
        with conn.cursor() as cur:
            cur.arraysize = 1000
            cur.execute(SQL_QUERY,
                        start_dt=start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                        end_dt=end_dt.strftime('%Y-%m-%d %H:%M:%S'))
            rows = cur.fetchall()
        log.info(f"      → {len(rows):,}행 fetch ({time.time()-t0:.1f}s)")
    finally:
        conn.close()

    log.info("[3/4] 분단위 머지...")
    merged = merge_by_minute(rows)
    log.info(f"      → {len(merged):,}분 (분당 1행)")

    log.info(f"[4/4] CSV 저장 → {out_path}")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(CSV_HEADER)
        for r in merged:
            writer.writerow(["" if v is None else v for v in r])

    size_mb = out_path.stat().st_size / 1024 / 1024
    log.info(f"완료: {size_mb:.1f}MB, 총 {time.time()-t0:.1f}s")
    return out_path


# ==========================================================
# 날짜 파서
# ==========================================================
def parse_date(s: str) -> datetime:
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y%m%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    sys.exit(f"ERROR: 날짜 형식 오류: {s!r}  (예: 2026-05-15 또는 '2026-05-15 00:00:00')")


# ==========================================================
# 메인
# ==========================================================
def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('start', help="시작일 (예: 2026-05-15)")
    p.add_argument('end', nargs='?', help="종료일 (생략 시 하루치, 미포함 경계)")
    p.add_argument('-o', '--out', default=str(DEFAULT_OUT_DIR),
                   help=f"출력 폴더 (기본: {DEFAULT_OUT_DIR})")
    args = p.parse_args()

    start_dt = parse_date(args.start)
    if args.end:
        end_dt = parse_date(args.end)
    else:
        end_dt = start_dt + timedelta(days=1)

    if end_dt <= start_dt:
        sys.exit(f"ERROR: end({end_dt}) <= start({start_dt})")

    out_path = extract(start_dt, end_dt, Path(args.out))
    print()
    print(f"OK: {out_path}")
    print(f"다음: python ..\\hubroom_predictor.py \"{out_path}\" -o .\\predict_tobe")


if __name__ == "__main__":
    main()
