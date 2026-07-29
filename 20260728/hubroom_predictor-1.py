#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M16 HUBROOM 통합 이벤트 예측기 v4.1 (룰베이스 8영역)
======================================================
8개 FAB 영역 통합 룰베이스 데드락 예측기.
   대상 영역: M16HUB, M14, M14B, M16A, M16B, M16, M16_PKT, M16_WT
   학습 임계값: 2026-03-24 14:39 ~ 2026-04-30 정상분포 (p95/p99) 기반
   테스트 구간 : 2026-05-01 ~ (사용자가 5월로 검증 예정)

   ※ 데이터 검증 (사용자 지적 반영)
   1) 1~3월 초는 22개 핵심 컬럼 (MAXCAPA 6, HUB인플로 9, HUB출구 5, STB 1, AOTRANSDELAY 1)
      이 NULL. 2026-03-24 14:39 부터 수집 시작. 따라서 학습기간을 그 이후로 한정함.
      모든 룰은 safe_int/safe_float 가 None 으로 처리하므로 NULL 행은 자동 스킵됨.
   2) 현재 DB 에는 없지만 추후 추가될 수 있는 3개 컬럼은 graceful 처리 (있으면 사용):
        - M16HUB.SORTER.ABN.SORTERWAITCOUNTOVER
        - M16A.SORTER.ABN.SORTERTRANSFERFAIL
        - M16B.SORTER.ABN.SORTERTRANSFERFAIL
   3) R-A' 컬럼명 영역별 차이 (검증완료):
        - M16HUB / M14B / M16_PKT / M16_WT  →  QUE.TIME.AVGTOTALTIME1MIN
        - M14   / M16A  / M16B              →  QUE.LOAD.AVGLOADTIME1MIN
   4) 계획서 269컬럼 vs 실제 수집 265컬럼 차이는 폐기 19 / 추가 4(AOTRANSDELAY) /
      추가 11(v3 collector 호환) — 본 예측기는 실제 수집 컬럼만 사용.

수집기가 매분 ./predict/M16A_HUBROOM_PR.csv 덮어쓰면
본 스크립트는 ./predict_tobe/ 폴더에 날짜별 CSV 로 append.

사용법:
    # 일괄 (백테스트)
    python3 hubroom_predictor.py path/to/INPUT.csv -o ./predict_tobe

    # 실시간 감시 (수집기 동기)
    python3 hubroom_predictor.py --watch
"""
import csv, logging, os, sys, time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

# Logpresso 업로더 (선택 — 모듈 없거나 비활성 시 자동 스킵)
try:
    import Rule_LO as _logpresso
except Exception as _e:
    _logpresso = None

# Graph_LO — 발동이벤트 ≥주의 시 raw 90분 복사 + 그래프 자동 생성 (선택)
try:
    import Graph_LO as _graph
except Exception:
    _graph = None

# ============================================================
# 기본 경로 / 상수
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = BASE_DIR / "predict" / "M16A_HUBROOM_PR.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "predict_tobe"

SYNC_OFFSET_SEC = 5
INTERVAL_SEC = 60
# ============================================================
# 임계값 외부 설정 (thresholds.json) — 코드 수정 없이 조정
# ============================================================
#   · thresholds.json 이 있으면 그 값으로 덮어씀.
#   · 파일이 없거나 / 특정 키가 빠지면 → 아래 코드 기본값 사용 (절대 안 깨짐).
#   · dict 임계(TH_RA 등)는 영역별 부분 수정 가능 (빠진 영역은 기본값 유지).
#   · 수정 후 predictor 재시작 필요.
import json as _json

THRESHOLDS_FILE = BASE_DIR / "thresholds.json"


def _load_thresholds():
    if not THRESHOLDS_FILE.exists():
        return {}
    try:
        with open(THRESHOLDS_FILE, encoding='utf-8') as f:
            data = _json.load(f)
        return {k: v for k, v in data.items() if not str(k).startswith('_')}
    except Exception as e:  # 깨진 json 이라도 운영 멈추지 않음
        print(f"[thresholds] {THRESHOLDS_FILE} 로드 실패 ({e}) — 코드 기본값 사용")
        return {}


_TH = _load_thresholds()


def _T(key, default):
    """스칼라 임계: json 값 있으면 사용, 없으면 기본값."""
    return _TH.get(key, default)


def _TD(key, default_dict):
    """dict 임계: json 의 해당 영역만 덮어쓰고 나머지는 기본값 유지."""
    merged = dict(default_dict)
    override = _TH.get(key)
    if isinstance(override, dict):
        merged.update(override)
    return merged


WINDOW_MIN = _T('WINDOW_MIN', 90)
INCIDENT_END_GAP_MIN = _T('INCIDENT_END_GAP_MIN', 60)  # ★ v6: 10→60분, 진동(on/off) 흡수
PREDICT_LOOKBACK_MIN = _T('PREDICT_LOOKBACK_MIN', 60)
# ★ 사건 정의 = '점수 이 값 이상(경계+)' 으로 시작/종료. (전엔 stage 3 기준이라
#   점수 낮은(정상) 시각이 사건 시작으로 잡혀 9시간 가짜 사건이 생기던 문제 → 점수 기준으로 교정)
MIN_INCIDENT_SCORE = _T('MIN_INCIDENT_SCORE', 50)

# ============================================================
# 영역별 임계값 (2026-03-24 14:39 ~ 04-30 학습 분포 p95/p99 기반)
#   → thresholds.json 으로 운영 중 조정 (위 _T/_TD 로더 참조)
# ============================================================
TH_RA = _TD('TH_RA', {
    'M16HUB':  9.0, 'M14':     3.3, 'M14B':    5.0,
    'M16A':    3.2, 'M16B':    3.5, 'M16_PKT': 7.5, 'M16_WT':  2.8,
})
TH_RA_SUSTAINED_RATIO = _T('TH_RA_SUSTAINED_RATIO', 0.67)
TH_RA_SUSTAINED_COUNT = _T('TH_RA_SUSTAINED_COUNT', 3)

TH_RB_30 = _TD('TH_RB_30', {
    'M16HUB': 100, 'M14': 80, 'M14B': 150,
    'M16A': 80, 'M16B': 30, 'M16': 20,
})
# TH_RB_10 은 기본적으로 30분 임계의 30% 로 자동 산출. json 에 TH_RB_10 명시 시 그 값 우선.
TH_RB_10 = _TD('TH_RB_10', {k: max(10, int(v * 0.3)) for k, v in TH_RB_30.items()})

TH_RC_REVERSE = _T('TH_RC_REVERSE', 2)
TH_RD_FABSTORAGE = _T('TH_RD_FABSTORAGE', 25.0)
TH_RD_HUB_STB_UTIL = _T('TH_RD_HUB_STB_UTIL', 99.0)
TH_RD_OHT_UTIL = _T('TH_RD_OHT_UTIL', 95.0)

# ★ v5 신규 룰 임계
# R-MLUD: MLUD 잡(수동 이동) 누적 — 메신저 "MLUD 정체" 패턴 (1건 미탐 사례)
TH_MLUD_JOB = _T('TH_MLUD_JOB', 50)            # MLUD 잡 임계 (개)
TH_MLUD_MANUAL = _T('TH_MLUD_MANUAL', 30)      # 수동 이동 큐 임계 (개)
# R-CNVFULL: M16HUB CNV 충만도 — 메신저 "Conv 전체 Full" 패턴
TH_CNV_FULL_RATIO = _T('TH_CNV_FULL_RATIO', 0.85)  # cnv_to_m14a / cnv_capa ≥ N%

TH_SLA_RATIO = _TD('TH_SLA_RATIO', {'M16HUB': 5.0, 'M14': 25.0, 'M16A': 13.0, 'M16B': 18.0})

TH_SORTER_WAIT = _TD('TH_SORTER_WAIT', {
    'M14': 100, 'M14B': 75, 'M16A': 180, 'M16B': 90, 'M16HUB': 30,
})
TH_SORTER_TRANSFER_FAIL = _T('TH_SORTER_TRANSFER_FAIL', 1)

# ============================================================
# ★ 영역별 융합 가중 (unified_risk_score 전용)
# ------------------------------------------------------------
#   Layer3 점수 합산(layer1/flow/sla/sorter/mc) 시에만 곱한다.
#   · hot_area / affected_areas / propagation_chain / 단계(stage) 는
#     원본 area_score·트리거를 그대로 쓰므로 영향 없음.
#   · 값 1.0 = 현행 유지. 0.5 = 그 영역 점수 기여를 정확히 절반.
#   · thresholds.json 에 {"AREA_WEIGHT": {"M16B": 0.5}} 로 넣으면
#     코드 수정 없이 조정 가능 (재시작 필요).
# ============================================================
AREA_WEIGHT = _TD('AREA_WEIGHT', {'M16B': 0.5})


def _aw(area):
    """영역 융합 가중치 (미지정 영역은 1.0)."""
    return AREA_WEIGHT.get(area, 1.0)


MAXCAPA_NORMAL = {
    'M16HUB.QUE.LFT.3F_LFT_MAXCAPA':       (165, 100),
    'M16HUB.QUE.LFT.3F_M14BLFT_MAXCAPA':   (66, 50),
    'M16HUB.QUE.CNV.3F_CNV_MAXCAPA':       (129, 80),
    'M14.QUE.CNV.3F_CNV_MAXCAPA':          (244, 150),
    'M16A.QUE.LFT.2F_LFT_MAXCAPA':         (54, 40),
    'M16A.QUE.LFT.6F_LFT_MAXCAPA':         (149, 100),
}

FLOW_NODES = {
    'M14_CNV_TO_HUB':    ('M14',    'M14.QUE.CNV.M14ATOM16ACURRNETQCNT'),
    'M14_TO_HUB_JOB':    ('M14',    'M14.QUE.ALL.3F_TO_HUB_JOB'),
    'M14B_7F_TO_HUB':    ('M14B',   'M14B.QUE.ALL.7F_TO_HUB_JOB'),
    'M14B_LFT_4ABLD_SUM':        ('M14B',   'M14B.LFT.4ABLD_ALL.TOTAL_CURRENTQCNT_SUM'),
    'M14B_LFT_4ABLD_TO_HUB_SUM': ('M14B',   'M14B.LFT.4ABLD_ALL.7F_TO_4F_CURRENTQCNT_SUM'),
    'M16A_6F_TO_HUB':    ('M16A',   'M16A.QUE.ALL.6F_TO_HUB_JOB'),
    'M16A_2F_TO_HUB':    ('M16A',   'M16A.QUE.ALL.2F_TO_HUB_JOB'),
    'M16B_10F_TO_HUB':   ('M16B',   'M16B.QUE.ALL.10F_TO_HUB_JOB'),
    'HUB_OHT_QCNT':      ('M16HUB', 'M16HUB.QUE.OHT.CURRENTOHTQCNT'),
    'M14_TO_M16':        ('M16HUB', 'M16HUB.QUE.M14TOM16.MESCURRENTQCNT'),
}
TH_FLOW_X1_5 = _T('TH_FLOW_X1_5', 1.5)
TH_FLOW_X2_0 = _T('TH_FLOW_X2_0', 2.0)
TH_FLOW_X3_0 = _T('TH_FLOW_X3_0', 3.0)

LIFTER_IDS = [
    '6ABL6011', '6ABL6012', '6ABL6021', '6ABL6022',
    '6ABL6031', '6ABL6032', '6ABL0111', '6ABL0112',
    '6ABL0121', '6ABL0122',
]

RA_COL = {
    'M16HUB':  'M16HUB.QUE.TIME.AVGTOTALTIME1MIN',
    'M14':     'M14.QUE.LOAD.AVGLOADTIME1MIN',
    'M14B':    'M14B.QUE.TIME.AVGTOTALTIME1MIN',
    'M16A':    'M16A.QUE.LOAD.AVGLOADTIME1MIN',
    'M16B':    'M16B.QUE.LOAD.AVGLOADTIME1MIN',
    'M16_PKT': 'M16_PKT.QUE.TIME.AVGTOTALTIME1MIN',
    'M16_WT':  'M16_WT.QUE.TIME.AVGTOTALTIME1MIN',
}
RB_COL = {
    'M16HUB':  'M16HUB.QUE.M14TOM16.MESCURRENTQCNT',
    'M14':     'M14.QUE.ALL.3F_TO_HUB_JOB',
    'M14B':    'M14B.QUE.ALL.7F_TO_HUB_JOB',
    'M16A':    'M16A.QUE.ALL.6F_TO_HUB_JOB',
    'M16B':    'M16B.QUE.ALL.10F_TO_HUB_JOB',
    'M16':     'M16.QUE.SFAB.SENDQUEUETOTAL',
}
RD_OHT_COL = {
    'M16HUB': 'M16HUB.QUE.OHT.OHTUTIL',
    'M14':    'M14.QUE.OHT.OHTUTIL',
    'M14B':   'M14B.QUE.OHT.OHTUTIL',
    'M16A':   'M16A.QUE.OHT.OHTUTIL',
    'M16B':   'M16B.QUE.OHT.OHTUTIL',
}
SLA_COL = {
    'M16HUB': 'M16HUB.QUE.ALL.TRANSPORT4MINOVERRATIO',
    'M14':    'M14.QUE.ALL.TRANSPORT4MINOVERRATIO',
    'M16A':   'M16A.QUE.ALL.TRANSPORT4MINOVERRATIO',
    'M16B':   'M16B.QUE.ALL.TRANSPORT4MINOVERRATIO',
}
SORTER_COL = {
    'M14':    'M14.SORTER.ABN.SORTERWAITCOUNTOVER',
    'M14B':   'M14B.SORTER.ABN.SORTERWAITCOUNTOVER',
    'M16A':   'M16A.SORTER.ABN.SORTERWAITCOUNTOVER',
    'M16B':   'M16B.SORTER.ABN.SORTERWAITCOUNTOVER',
    'M16HUB': 'M16HUB.SORTER.ABN.SORTERWAITCOUNTOVER',
}
SORTER_FAIL_COL = {
    'M16A':   'M16A.SORTER.ABN.SORTERTRANSFERFAIL',
    'M16B':   'M16B.SORTER.ABN.SORTERTRANSFERFAIL',
}
HUB_OUT_COLS = [
    'M16HUB.QUE.ALL.3F_TO_M16A_6F_JOB',
    'M16HUB.QUE.ALL.3F_TO_M16A_2F_JOB',
    'M16HUB.QUE.ALL.3F_TO_M14A_3F_JOB',
    'M16HUB.QUE.ALL.3F_TO_M14B_7F_JOB',
    'M16HUB.QUE.ALL.3F_TO_3F_MLUD_JOB',
]


# ============================================================
# 로깅 & 유틸
# ============================================================
def setup_logger(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("predictor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(out_dir / "predictor.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def safe_float(v):
    try:
        return float(v) if v not in (None, '', 'null') else None
    except (ValueError, TypeError):
        return None


def safe_int(v):
    try:
        return int(float(v)) if v not in (None, '', 'null') else None
    except (ValueError, TypeError):
        return None


def parse_time(s):
    if not s:
        return None
    s = s.strip().strip('"').strip("'")
    if 'T' in s and '+' in s.split('T')[1]:
        s = s.split('+')[0]
    if s.endswith('Z'):
        s = s[:-1]
    if '.' in s:
        s = s.split('.')[0]
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M',
                '%Y%m%d%H%M%S', '%Y%m%d %H%M%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ============================================================
# 입력 CSV 로더 — 8개 영역 통합
# ============================================================
def iter_unified_rows(filepath):
    enc_list = ['utf-8-sig', 'utf-8', 'cp949']
    last_err = None
    for enc in enc_list:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = parse_time(row.get('CRT_TM', ''))
                    if not t:
                        continue
                    d = {'time': t}
                    g = row.get

                    d['M16HUB'] = {
                        'ra': safe_float(g(RA_COL['M16HUB'])),
                        'rb': safe_int(g(RB_COL['M16HUB'])),
                        'rd_fab': safe_float(g('M16HUB.STRATE.ALL.FABSTORAGERATIO')),
                        'rd_oht': safe_float(g(RD_OHT_COL['M16HUB'])),
                        'sla_ratio': safe_float(g(SLA_COL['M16HUB'])),
                        'sla_cnt': safe_int(g('M16HUB.QUE.ALL.TRANSPORT4MINOVERCNT')),
                        'sorter': safe_int(g(SORTER_COL['M16HUB'])),
                        'hub_stb_util': safe_float(g('M16HUB.STRATE.STB.3F_STORAGE_UTIL')),
                        'oht_qcnt': safe_int(g('M16HUB.QUE.OHT.CURRENTOHTQCNT')),
                        'oht_alarm': safe_int(g('M16HUB.OHT.ALERT.OHTMCPALARMCNT')),
                        'aotransdelay': safe_int(g('M16HUB.QUE.ABN.AOTRANSDELAY')),
                        # ★ v5 신규 — MLUD/CNV 룰용 메트릭
                        'mlud_job':      safe_int(g('M16HUB.QUE.ALL.3F_TO_3F_MLUD_JOB')),
                        'mlud_manual':   safe_int(g('M16HUB.QUE.ALL.M16HUBTOM14MANUAL_CURRENTQCNT')),
                        'cnv_capa':      safe_int(g('M16HUB.QUE.CNV.3F_CNV_MAXCAPA')),
                        'cnv_to_m14a':   safe_int(g('M16HUB.CNV.SENDFAB.TO_M14A_CURRENTQCNT')),
                        'lifters': {lid: safe_int(g(f'M16HUB.LFT.{lid}.TOTAL_CURRENTQCNT'))
                                    for lid in LIFTER_IDS},
                        'hub_outs': {col: safe_int(g(col)) for col in HUB_OUT_COLS},
                        'maxcapa': {
                            'M16HUB.QUE.LFT.3F_LFT_MAXCAPA': safe_int(g('M16HUB.QUE.LFT.3F_LFT_MAXCAPA')),
                            'M16HUB.QUE.LFT.3F_M14BLFT_MAXCAPA': safe_int(g('M16HUB.QUE.LFT.3F_M14BLFT_MAXCAPA')),
                            'M16HUB.QUE.CNV.3F_CNV_MAXCAPA': safe_int(g('M16HUB.QUE.CNV.3F_CNV_MAXCAPA')),
                        },
                    }
                    d['M16HUB']['lifters'] = {k: v for k, v in d['M16HUB']['lifters'].items() if v is not None}

                    d['M14'] = {
                        'ra': safe_float(g(RA_COL['M14'])),
                        'rb': safe_int(g(RB_COL['M14'])),
                        'rd_oht': safe_float(g(RD_OHT_COL['M14'])),
                        'sla_ratio': safe_float(g(SLA_COL['M14'])),
                        'sla_cnt': safe_int(g('M14.QUE.ALL.TRANSPORT4MINOVERCNT')),
                        'sorter': safe_int(g(SORTER_COL['M14'])),
                        'cnv_north': safe_int(g('M14.QUE.CNV.M14ATONORTHCURRENTQCNT')),
                        'cnv_south': safe_int(g('M14.QUE.CNV.M14ATOSOUTHCURRENTQCNT')),
                        'cnv_m14a_m16a': safe_int(g('M14.QUE.CNV.M14ATOM16ACURRNETQCNT')),
                        'inflow_alt': safe_int(g('M14.QUE.ALL.3F_TO_HUB_JOB_ALT')),
                        'oht_cmd': safe_int(g('M14.QUE.OHT.3F_TO_HUB_CMD')),
                        'htstop': safe_int(g('M14.OHT.STATECNT.HTSTOP')),
                        'congested': safe_int(g('M14.OHT.STATECNT.CONGESTED')),
                        'abnormal': safe_int(g('M14.OHT.STATECNT.ABNORMAL')),
                        'maxcapa': {'M14.QUE.CNV.3F_CNV_MAXCAPA': safe_int(g('M14.QUE.CNV.3F_CNV_MAXCAPA'))},
                    }

                    d['M14B'] = {
                        'ra': safe_float(g(RA_COL['M14B'])),
                        'rb': safe_int(g(RB_COL['M14B'])),
                        'rd_oht': safe_float(g(RD_OHT_COL['M14B'])),
                        'sorter': safe_int(g(SORTER_COL['M14B'])),
                        # M14B(7F) ↔ M16HUBROOM(3F) 리프터 6대 (도메인 4-4)
                        # 양방향 TOTAL 합산 (참고용) — None 안전
                        'lft_4abld_sum': sum(
                            (safe_int(g(f'M14B.LFT.4ABLD{lid}.TOTAL_CURRENTQCNT')) or 0)
                            for lid in ('111', '112', '121', '122', '131', '132')
                        ),
                        # M14B(7F) → M16HUB(3F) 하강 방향만 (정체 직접 신호) ★ — None 안전
                        'lft_4abld_to_hub_sum': sum(
                            (safe_int(g(f'M14B.LFT.4ABLD{lid}.7F_TO_4F_CURRENTQCNT')) or 0)
                            for lid in ('111', '112', '121', '122', '131', '132')
                        ),
                        # 122 단독값 — 운영 호환용 (메신저에서 자주 참조)
                        'lft_4abld122': safe_int(g('M14B.LFT.4ABLD122.TOTAL_CURRENTQCNT')),
                        'inflow_alt': safe_int(g('M14B.QUE.ALL.7F_TO_HUB_JOB_ALT')),
                        'oht_cmd': safe_int(g('M14B.QUE.OHT.7F_TO_HUB_CMD')),
                        'aotransdelay': safe_int(g('M14B.QUE.ABN.AOTRANSDELAY')),
                    }

                    d['M16A'] = {
                        'ra': safe_float(g(RA_COL['M16A'])),
                        'rb': safe_int(g(RB_COL['M16A'])),
                        'inflow_2f': safe_int(g('M16A.QUE.ALL.2F_TO_HUB_JOB')),
                        'rd_oht': safe_float(g(RD_OHT_COL['M16A'])),
                        'sla_ratio': safe_float(g(SLA_COL['M16A'])),
                        'sla_cnt': safe_int(g('M16A.QUE.ALL.TRANSPORT4MINOVERCNT')),
                        'sorter': safe_int(g(SORTER_COL['M16A'])),
                        'sorter_fail': safe_int(g(SORTER_FAIL_COL['M16A'])),
                        'oht_cmd_2f': safe_int(g('M16A.QUE.OHT.2F_TO_HUB_CMD')),
                        'oht_cmd_6f': safe_int(g('M16A.QUE.OHT.6F_TO_HUB_CMD')),
                        'maxcapa': {
                            'M16A.QUE.LFT.2F_LFT_MAXCAPA': safe_int(g('M16A.QUE.LFT.2F_LFT_MAXCAPA')),
                            'M16A.QUE.LFT.6F_LFT_MAXCAPA': safe_int(g('M16A.QUE.LFT.6F_LFT_MAXCAPA')),
                        },
                    }

                    d['M16B'] = {
                        'ra': safe_float(g(RA_COL['M16B'])),
                        'rb': safe_int(g(RB_COL['M16B'])),
                        'rd_oht': safe_float(g(RD_OHT_COL['M16B'])),
                        'sla_ratio': safe_float(g(SLA_COL['M16B'])),
                        'sla_cnt': safe_int(g('M16B.QUE.ALL.TRANSPORT4MINOVERCNT')),
                        'sorter': safe_int(g(SORTER_COL['M16B'])),
                        'sorter_fail': safe_int(g(SORTER_FAIL_COL['M16B'])),
                    }

                    d['M16'] = {
                        'rb': safe_int(g(RB_COL['M16'])),
                        'sfab_send': safe_int(g('M16.QUE.SFAB.SENDQUEUETOTAL')),
                        'sfab_recv': safe_int(g('M16.QUE.SFAB.RECEIVEQUEUETOTAL')),
                        'sfab_ret': safe_int(g('M16.QUE.SFAB.RETURNQUEUETOTAL')),
                    }

                    d['M16_PKT'] = {
                        'ra': safe_float(g(RA_COL['M16_PKT'])),
                        'rd_oht': safe_float(g('M16_PKT.QUE.OHT.OHTUTIL')),
                        'aotransdelay': safe_int(g('M16_PKT.QUE.ABN.AOTRANSDELAY')),
                    }

                    d['M16_WT'] = {
                        'ra': safe_float(g(RA_COL['M16_WT'])),
                        'rd_oht': safe_float(g('M16_WT.QUE.OHT.OHTUTIL')),
                        'aotransdelay': safe_int(g('M16_WT.QUE.ABN.AOTRANSDELAY')),
                    }

                    yield d
            return
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err


# ============================================================
# 영역별 4축 룰 평가
# ============================================================
def eval_area_rules(area, window):
    out = {
        'ra_trig': False, 'ra_sustained': False, 'ra_value': None, 'ra_count': 0,
        'rb_trig': False, 'rb_fast': False, 'rb_diff_30': 0, 'rb_diff_10': 0,
        'rc_trig': False, 'rev_count': 0, 'rev_lids': [], 'rc_trend': 0,
        'rd_trig': False, 'rd_fab': 0, 'rd_oht': 0,
        'sla_trig': False, 'sla_ratio': 0, 'sla_cnt': 0,
        'sorter_trig': False, 'sorter_val': 0,
        'maxcapa_changed': [], 'maxcapa_changed_n': 0,
        'area_score': 0, 'area_signals': [],
    }
    wlist = list(window)
    if not wlist:
        return out
    latest = wlist[-1]
    if not latest:
        return out

    # R-A'
    if area in TH_RA:
        th_ra = TH_RA[area]
        th_sus = th_ra * TH_RA_SUSTAINED_RATIO
        ra_vals = [w.get('ra') for w in wlist[-10:] if w and w.get('ra') is not None]
        if ra_vals:
            out['ra_value'] = ra_vals[-1]
            out['ra_count'] = sum(1 for v in ra_vals if v >= th_ra)
            out['ra_trig'] = out['ra_count'] >= 1
        last5 = [w.get('ra') for w in wlist[-5:] if w and w.get('ra') is not None]
        if len(last5) >= 3:
            out['ra_sustained'] = sum(1 for v in last5 if v >= th_sus) >= TH_RA_SUSTAINED_COUNT

    # R-B
    if area in TH_RB_30:
        th30 = TH_RB_30[area]
        th10 = TH_RB_10[area]
        rb_vals = [w.get('rb') for w in wlist if w]
        if len(rb_vals) >= 31 and rb_vals[-1] is not None and rb_vals[-31] is not None:
            out['rb_diff_30'] = rb_vals[-1] - rb_vals[-31]
            out['rb_trig'] = out['rb_diff_30'] >= th30
        if len(rb_vals) >= 11 and rb_vals[-1] is not None and rb_vals[-11] is not None:
            out['rb_diff_10'] = rb_vals[-1] - rb_vals[-11]
            out['rb_fast'] = out['rb_diff_10'] >= th10

    # R-C' (M16HUB lifter / M14 CNV skew)
    if area == 'M16HUB':
        lifters_list = [w.get('lifters', {}) for w in wlist if w]
        if len(lifters_list) >= 21 and lifters_list[-1] and lifters_list[-21]:
            now_l = lifters_list[-1]
            prev_l = lifters_list[-21]
            out['rc_trend'] = sum(now_l.values()) - sum(prev_l.values())
            for lid, v in now_l.items():
                if v is not None and v > (prev_l.get(lid) or 0):
                    out['rev_lids'].append(lid)
                    out['rev_count'] += 1
            out['rc_trig'] = out['rc_trend'] < 0 and out['rev_count'] >= TH_RC_REVERSE
    elif area == 'M14':
        n = latest.get('cnv_north') or 0
        s = latest.get('cnv_south') or 0
        if n + s > 0:
            ratio = max(n, s) / max(1, n + s)
            out['cnv_skew'] = ratio
            out['rc_trig'] = ratio >= 0.70

    # R-D
    if area == 'M16HUB':
        out['rd_fab'] = latest.get('rd_fab') or 0
        out['rd_oht'] = latest.get('rd_oht') or 0
        stb = latest.get('hub_stb_util') or 0
        out['hub_stb_util'] = stb
        # ★ v5 신규 — R-MLUD: MLUD 잡 누적 (메신저 "MLUD 정체" 패턴)
        mlud_job = latest.get('mlud_job') or 0
        mlud_manual = latest.get('mlud_manual') or 0
        out['mlud_job'] = mlud_job
        out['mlud_manual'] = mlud_manual
        out['mlud_trig'] = (mlud_job >= TH_MLUD_JOB) or (mlud_manual >= TH_MLUD_MANUAL)
        # ★ v5 신규 — R-CNVFULL: M16HUB CNV 충만도 (메신저 "Conv 전체 Full" 패턴)
        cnv_cur = latest.get('cnv_to_m14a') or 0
        cnv_capa = latest.get('cnv_capa') or 0
        cnv_ratio = (cnv_cur / cnv_capa) if cnv_capa > 0 else 0
        out['cnv_ratio'] = cnv_ratio
        out['cnv_full_trig'] = cnv_ratio >= TH_CNV_FULL_RATIO
        # R-D 통합 트리거 — 기존 FAB/STB + 신규 MLUD/CNV (R-D 우산 아래로)
        out['rd_trig'] = ((out['rd_fab'] >= TH_RD_FABSTORAGE)
                          or (stb >= TH_RD_HUB_STB_UTIL)
                          or out['mlud_trig']
                          or out['cnv_full_trig'])
    elif area in RD_OHT_COL:
        out['rd_oht'] = latest.get('rd_oht') or 0
        out['rd_trig'] = out['rd_oht'] >= TH_RD_OHT_UTIL

    # SLA
    if area in SLA_COL:
        ratio = latest.get('sla_ratio')
        cnt = latest.get('sla_cnt')
        out['sla_ratio'] = ratio or 0
        out['sla_cnt'] = cnt or 0
        if ratio is not None and ratio >= TH_SLA_RATIO[area]:
            out['sla_trig'] = True
        if len(wlist) >= 11:
            prev = wlist[-11]
            if prev and prev.get('sla_cnt') is not None and cnt is not None:
                if cnt - prev.get('sla_cnt') >= 20:
                    out['sla_trig'] = True

    # Sorter
    if area in TH_SORTER_WAIT:
        sv = latest.get('sorter')
        if sv is not None:
            out['sorter_val'] = sv
            out['sorter_trig'] = sv >= TH_SORTER_WAIT[area]
    sf = latest.get('sorter_fail')
    if sf is not None and sf >= TH_SORTER_TRANSFER_FAIL:
        out['sorter_fail_trig'] = True
        out['sorter_fail_val'] = sf
        out['sorter_trig'] = True

    # MAXCAPA
    mc = latest.get('maxcapa') or {}
    for col, val in mc.items():
        if val is None:
            continue
        normal, th = MAXCAPA_NORMAL.get(col, (None, None))
        if th is not None and val <= th:
            out['maxcapa_changed'].append(f"{col.split('.')[-1]}={val}(<={th})")
    out['maxcapa_changed_n'] = len(out['maxcapa_changed'])

    # 영역 점수 (0~50) — 룰별 분해 점수 함께 저장
    ra_pts      = 10 if out['ra_trig']      else 0
    ra_sus_pts  = 5  if out['ra_sustained'] else 0
    rb_pts      = 10 if out['rb_trig']      else 0
    rb_fast_pts = 5  if out['rb_fast']      else 0
    rc_pts      = 8  if out['rc_trig']      else 0
    rd_pts      = 7  if out['rd_trig']      else 0
    sla_pts     = 5  if out['sla_trig']     else 0
    sort_pts    = 3  if out['sorter_trig']  else 0
    mc_pts      = 10 * out['maxcapa_changed_n']

    sig = []
    if out['ra_trig']:        sig.append('RA')
    if out['ra_sustained']:   sig.append('RA_sus')
    if out['rb_trig']:        sig.append('RB')
    if out['rb_fast']:        sig.append('RB_fast')
    if out['rc_trig']:        sig.append('RC')
    if out['rd_trig']:        sig.append('RD')
    if out['sla_trig']:       sig.append('SLA')
    if out['sorter_trig']:    sig.append('SORT')
    if out['maxcapa_changed_n'] > 0:
        sig.append(f'MAXCAPA*{out["maxcapa_changed_n"]}')

    s = ra_pts + ra_sus_pts + rb_pts + rb_fast_pts + rc_pts + rd_pts + sla_pts + sort_pts + mc_pts
    out['area_score'] = min(50, s)
    out['area_score_raw'] = s  # 클리핑 전 원본 (분석용)
    out['area_signals'] = sig

    # 룰별 분해 점수 (고객 조합 분석용)
    out['pts_RA']      = ra_pts
    out['pts_RA_sus']  = ra_sus_pts
    out['pts_RB']      = rb_pts
    out['pts_RB_fast'] = rb_fast_pts
    out['pts_RC']      = rc_pts
    out['pts_RD']      = rd_pts
    out['pts_SLA']     = sla_pts
    out['pts_SORT']    = sort_pts
    out['pts_MAXCAPA'] = mc_pts
    return out


# ============================================================
# 흐름 룰 평가 (9개 노드)
# ============================================================
def eval_flow_rules(flow_history):
    result = {}
    if len(flow_history) < 11:
        return result
    latest = flow_history[-1] or {}
    avg_window = list(flow_history)[-30:] if len(flow_history) >= 30 else list(flow_history)
    for node, cur in latest.items():
        if cur is None:
            continue
        vals = [w.get(node) for w in avg_window if w and w.get(node) is not None]
        if not vals or len(vals) < 5:
            continue
        avg = sum(vals) / len(vals)
        if avg <= 0:
            continue
        ratio = cur / avg
        level = ''
        if ratio >= TH_FLOW_X3_0:
            level = '심각'
        elif ratio >= TH_FLOW_X2_0:
            level = '위험'
        elif ratio >= TH_FLOW_X1_5:
            level = '주의'
        if level:
            result[node] = {'ratio': ratio, 'level': level, 'current': cur, 'avg30': avg}
    return result


# ============================================================
# Layer 3 통합 융합
# ============================================================
def evaluate_unified(t, area_results, flow_result, propagation_history):
    # ★ AREA_WEIGHT: 점수 합산에만 영역 가중 적용 (아래 hot_area/affected_areas 는 원본 사용)
    layer1_total = round(sum(r.get('area_score', 0) * _aw(a)
                             for a, r in area_results.items()), 1)

    flow_score = 0
    flow_signals = []
    for node, info in flow_result.items():
        pts = 30 if info['level'] == '심각' else 15 if info['level'] == '위험' else \
              5 if info['level'] == '주의' else 0
        if pts:
            flow_score += pts * _aw(FLOW_NODES.get(node, (None,))[0])
        flow_signals.append(f"{node}={info['ratio']:.1f}x({info['level']})")
    flow_score = round(flow_score, 1)

    sla_score = round(sum(5 * _aw(a) for a, r in area_results.items() if r.get('sla_trig')), 1)
    sorter_score = round(sum(3 * _aw(a) for a, r in area_results.items() if r.get('sorter_trig')), 1)

    mc_score = 0
    mc_signals = []
    for area, r in area_results.items():
        n = r.get('maxcapa_changed_n', 0)
        if n > 0:
            mc_score += 10 * n * _aw(area)
            mc_signals.extend([f"{area}:{x}" for x in r.get('maxcapa_changed', [])])
    mc_score = round(mc_score, 1)

    # ★ 점수 정규화 — raw 합산 → 0~100 척도 (raw 220 = 100점 발동)
    raw_score = layer1_total + flow_score + sla_score + sorter_score + mc_score
    unified_risk_score = min(100, round(raw_score * 100 / 220))

    # ★ 위험도 등급 (회사 표준 50/71/85): 경계 50~70 / 위험 71~84 / 초위험 85~100. 50 미만은 등급 공란 (정상 라벨 없음).
    if unified_risk_score >= 85:
        unified_risk_level = '초위험'
    elif unified_risk_score >= 71:
        unified_risk_level = '위험'
    elif unified_risk_score >= 50:
        unified_risk_level = '경계'
    else:
        unified_risk_level = ''

    hot_area = None
    hot_score = 0
    for area, r in area_results.items():
        if r.get('area_score', 0) > hot_score:
            hot_score = r['area_score']
            hot_area = area

    triggered_areas = [a for a, r in area_results.items() if r.get('area_score', 0) >= 15]
    for a in triggered_areas:
        propagation_history.append({'time': t, 'area': a,
                                    'score': area_results[a]['area_score'],
                                    'signals': area_results[a]['area_signals']})

    any_ra = any(r.get('ra_trig') or r.get('ra_sustained') for r in area_results.values())
    any_rb = any(r.get('rb_trig') or r.get('rb_fast') for r in area_results.values())
    any_rd_or_sla = any(r.get('rd_trig') or r.get('sla_trig') for r in area_results.values())
    any_rc = area_results.get('M16HUB', {}).get('rc_trig', False)
    any_flow_severe = any(info['level'] in ('위험', '심각') for info in flow_result.values())

    unified_s3 = any_ra and (any_rd_or_sla or any_rc) and (any_rb or any_flow_severe)
    unified_s1 = any_ra or any(r.get('ra_sustained') for r in area_results.values())
    unified_s2 = any_rb

    chain = []
    seen = {}
    for ent in list(propagation_history):
        if (t - ent['time']).total_seconds() / 60.0 > PREDICT_LOOKBACK_MIN:
            continue
        a = ent['area']
        if a not in seen:
            seen[a] = ent
            chain.append(ent)
    chain.sort(key=lambda x: x['time'])
    propagation_chain = ' → '.join(
        f"{x['area']}({x['time'].strftime('%H:%M')},{'+'.join(x['signals']) or '-'})"
        for x in chain
    )

    return {
        'unified_risk_score': unified_risk_score,
        'unified_risk_level': unified_risk_level,
        'unified_s1': unified_s1,
        'unified_s2': unified_s2,
        'unified_s3': unified_s3,
        'hot_area': hot_area or '',
        'hot_score': hot_score,
        'propagation_chain': propagation_chain,
        'affected_areas': ';'.join(triggered_areas),
        'flow_signals': ';'.join(flow_signals),
        'maxcapa_signals': ';'.join(mc_signals),
        'layer1_total': layer1_total,
        'flow_score': flow_score,
        'sla_score': sla_score,
        'sorter_score': sorter_score,
        'mc_score': mc_score,
    }


# ============================================================
# ★ v6.3 — 예측 장애 유형: hot_area + 증상 키워드 (운영자 즉시 이해)
# 예: HUB-FAB정체 / HUB-리프터역증가 / M14-Sorter대기 / M16B-SLA초과 / 광역정체
# ============================================================
def _predict_fault_type(ctx):
    if not ctx:
        return ''
    ar = ctx.get('area_results', {}) or {}
    hot_area = ctx.get('hot_area', '')
    hot_score = ctx.get('hot_score', 0)
    if not hot_area or hot_score < 10:
        return ''
    # 광역 우선 — 4+ 영역 동시
    affected = [a for a in (ctx.get('affected_areas', '') or '').split(';') if a]
    if len(affected) >= 4:
        return '광역정체'
    r = ar.get(hot_area, {}) or {}
    if hot_area == 'M16HUB':
        if r.get('mlud_trig'):       return 'HUB-MLUD'
        if r.get('cnv_full_trig'):   return 'HUB-CNV가득'
        if r.get('rc_trig') and (r.get('rev_count', 0) or 0) >= 4: return 'HUB-리프터역증가'
        if r.get('rd_trig'):
            if (r.get('rd_fab', 0) or 0) >= TH_RD_FABSTORAGE: return 'HUB-FAB정체'
            if (r.get('hub_stb_util', 0) or 0) >= TH_RD_HUB_STB_UTIL: return 'HUB-STB정체'
        if r.get('rb_trig'):         return 'HUB-큐누적'
        if r.get('ra_trig'):         return 'HUB-반송지연'
        return 'HUB-신호'
    if r.get('sorter_trig'):         return f'{hot_area}-Sorter대기'
    if r.get('sla_trig'):            return f'{hot_area}-SLA초과'
    if r.get('rd_trig'):             return f'{hot_area}-OHT정체'
    if r.get('rc_trig') and hot_area == 'M14': return 'M14-CNV쏠림'
    if r.get('ra_trig'):             return f'{hot_area}-반송지연'
    if r.get('rb_trig'):             return f'{hot_area}-큐누적'
    return f'{hot_area}-신호'


# ============================================================
# 사건 추적 FSM
# ============================================================
class IncidentTracker:
    def __init__(self):
        self.state = 'IDLE'
        self.current = None
        self.incidents = []
        self.early_signals = deque(maxlen=PREDICT_LOOKBACK_MIN * 2)
        self.events = []
        self.last_stage = 0

    def _record_event(self, t, stage, ctx):
        self.events.append({
            'time': t, 'stage': stage, 'prev_stage': self.last_stage,
            'is_transition': stage != self.last_stage,
            'ctx': ctx,
        })
        self.last_stage = stage

    def update(self, t, s1, s2, s3, ctx):
        if s1 or s2:
            self.early_signals.append(t)
        cur_stage = 3 if s3 else (2 if s2 else (1 if s1 else 0))
        self._record_event(t, cur_stage, ctx)
        # ★ 사건 시작/지속/종료 = '점수 50 이상(경계+)' 기준. (전엔 stage 3 → 점수 낮은 시각이
        #   사건 시작으로 잡혀 9시간 가짜 사건이 생김. 이제 진짜 정체(점수)만 사건으로.)
        is_alarm = (ctx.get('unified_risk_score', 0) or 0) >= MIN_INCIDENT_SCORE
        if self.state == 'IDLE':
            if is_alarm:
                self._start_new(t, ctx)
        else:
            if is_alarm:
                last_alarm = self.current['last_s3_time']   # 필드명 유지(의미=마지막 경보 시각)
                if (t - last_alarm).total_seconds() / 60.0 >= INCIDENT_END_GAP_MIN:
                    self.current['refire_count'] += 1
                self._update_current(t, ctx)
            else:
                last_alarm = self.current['last_s3_time']
                if (t - last_alarm).total_seconds() / 60.0 >= INCIDENT_END_GAP_MIN:
                    self._end_current(t)
        # ★ v6 신규 — 발동이벤트 행에 사건 상태/지속성/예측유형 박기
        ev = self.events[-1]
        if self.state == 'IN_INCIDENT' and self.current is not None:
            ev['incident_state'] = 'IN_INCIDENT'
            ev['continuity_min'] = int((t - self.current['start_time']).total_seconds() / 60) + 1
            ev['refire_count'] = self.current.get('refire_count', 0)
        else:
            ev['incident_state'] = 'IDLE'
            ev['continuity_min'] = 0
            ev['refire_count'] = 0
        ev['predicted_fault_type'] = _predict_fault_type(ctx)

    def _start_new(self, t, ctx):
        cutoff = t - timedelta(minutes=PREDICT_LOOKBACK_MIN)
        early = [x for x in self.early_signals if x >= cutoff]
        predict_time = min(early) if early else t
        self.current = {
            'predict_time': predict_time, 'start_time': t,
            'last_s3_time': t, 'end_time': t, 'refire_count': 0,
            'max_risk_score': ctx.get('unified_risk_score', 0),
            'max_risk_time': t,   # ★ 최고점(진짜 몰림) 발생 시각
            'max_risk_level': ctx.get('unified_risk_level', ''),
            'hot_area': ctx.get('hot_area', ''),
            'affected_areas_union': set(ctx.get('affected_areas', '').split(';')) - {''},
            'propagation_chain': ctx.get('propagation_chain', ''),
            # ★ 사건 내 각 영역별 최대값 추적
            'area_max': {},        # {area: {ra,rb_diff_30,rev_count,rd_fab,sla_ratio,sorter,maxcapa}}
            'triggered_rules': {}, # {area: set of rule_names}
            'maxcapa_history': set(),  # 사건 내 변경된 MAXCAPA 컬럼들
            # ★ 신규 — 룰별 점수 max / sum 추적 (고객 조합 분석용)
            'area_pts_max': {},    # {area: {RA: max_pts, ...}} 9 룰
            'area_pts_sum': {},    # {area: {RA: total_pts, ...}} — 사건 동안 누적
            'unified_pts_max': {
                'layer1_total': 0, 'flow_score': 0,
                'sla_score': 0, 'sorter_score': 0, 'mc_score': 0,
            },
            'unified_pts_sum': {
                'layer1_total': 0, 'flow_score': 0,
                'sla_score': 0, 'sorter_score': 0, 'mc_score': 0,
            },
            # ★ B 보강 — 진단 키 max / sum 추적
            'area_diag_max': {},   # {area: {ra_count: max, rb_diff_10: max, ...}}
            'area_diag_sum': {},
        }
        self.state = 'IN_INCIDENT'
        self._merge_area_stats(ctx)

    def _update_current(self, t, ctx):
        c = self.current
        c['last_s3_time'] = t
        c['end_time'] = t
        if ctx.get('unified_risk_score', 0) > c['max_risk_score']:
            c['max_risk_score'] = ctx['unified_risk_score']
            c['max_risk_time'] = t   # ★ 최고점 갱신 시각도 기록
            c['max_risk_level'] = ctx['unified_risk_level']
        c['affected_areas_union'].update(set(ctx.get('affected_areas', '').split(';')) - {''})
        if ctx.get('propagation_chain'):
            c['propagation_chain'] = ctx['propagation_chain']
        self._merge_area_stats(ctx)

    def _merge_area_stats(self, ctx):
        """ 사건 내 영역별 max값 / 발동룰 누적 """
        c = self.current
        ar = ctx.get('area_results', {}) or {}
        for area, r in ar.items():
            if r.get('area_score', 0) == 0:
                continue
            am = c['area_max'].setdefault(area, {})
            for k in ('ra_value', 'rb_diff_30', 'rev_count',
                      'rd_fab', 'rd_oht', 'hub_stb_util',
                      'sla_ratio', 'sla_cnt', 'sorter_val'):
                v = r.get(k)
                if v is None:
                    continue
                cur = am.get(k)
                if cur is None or v > cur:
                    am[k] = v
            # 발동 룰 누적
            tr = c['triggered_rules'].setdefault(area, set())
            if r.get('ra_trig'):       tr.add('RA')
            if r.get('ra_sustained'):  tr.add('RA_sus')
            if r.get('rb_trig'):       tr.add('RB')
            if r.get('rb_fast'):       tr.add('RB_fast')
            if r.get('rc_trig'):       tr.add('RC')
            if r.get('rd_trig'):       tr.add('RD')
            if r.get('sla_trig'):      tr.add('SLA')
            if r.get('sorter_trig'):   tr.add('SORT')
            if r.get('sorter_fail_trig'): tr.add('SORT_FAIL')
            for x in r.get('maxcapa_changed', []) or []:
                c['maxcapa_history'].add(f"{area}:{x}")

            # ★ 룰별 점수 max / sum 누적 (9 룰)
            pm = c['area_pts_max'].setdefault(area, {})
            ps = c['area_pts_sum'].setdefault(area, {})
            for sig in ('RA', 'RA_sus', 'RB', 'RB_fast', 'RC', 'RD', 'SLA', 'SORT', 'MAXCAPA'):
                v = r.get(f'pts_{sig}', 0) or 0
                if v > pm.get(sig, 0):
                    pm[sig] = v
                ps[sig] = ps.get(sig, 0) + v

            # ★ B 보강 — 진단 키 max / sum 누적
            dm = c['area_diag_max'].setdefault(area, {})
            ds = c['area_diag_sum'].setdefault(area, {})
            for k in ('ra_count', 'rb_diff_10', 'rc_trend', 'rd_oht',
                      'sla_cnt', 'sorter_fail_val', 'cnv_skew', 'area_score_raw'):
                v = r.get(k)
                if v is None:
                    continue
                # 음수 trend 도 추적 (max 는 절댓값 큰 쪽)
                if k == 'rc_trend':
                    cur_max = dm.get(k)
                    if cur_max is None or abs(v) > abs(cur_max):
                        dm[k] = v
                else:
                    if v > dm.get(k, 0):
                        dm[k] = v
                ds[k] = ds.get(k, 0) + (v if isinstance(v, (int, float)) else 0)

        # 통합 점수 분해 max / sum 누적 (영역 루프 밖, ctx 1회)
        um = c['unified_pts_max']
        us = c['unified_pts_sum']
        for k in ('layer1_total', 'flow_score', 'sla_score', 'sorter_score', 'mc_score'):
            v = ctx.get(k, 0) or 0
            if v > um.get(k, 0):
                um[k] = v
            us[k] = us.get(k, 0) + v

    def _end_current(self, t):
        c = self.current
        c['end_time'] = c['last_s3_time']
        # ★ 사건 기록 기준 = 점수 50 이상(경계+). 시작 기준과 동일하게 통일.
        if c.get('max_risk_score', 0) >= MIN_INCIDENT_SCORE:
            self.incidents.append(c)
        self.current = None
        self.state = 'IDLE'

    def finalize(self, last_t):
        if self.state == 'IN_INCIDENT':
            self.current['end_time'] = self.current['last_s3_time']
            if self.current.get('max_risk_score', 0) >= MIN_INCIDENT_SCORE:
                self.incidents.append(self.current)
            self.current = None
            self.state = 'IDLE'


# ============================================================
# CSV 출력
# ============================================================
STAGE_LABEL = {0: '이벤트없음', 1: '1단계 조기경보', 2: '2단계 주의보', 3: '3단계 ⭐확정'}

EVENT_FIELDS = [
    'file', 'datetime', 'date', 'time',
    'stage', 'stage_name', 'prev_stage', 'transition',
    # ★ v6 신규 — 사건 지속성/재발생/예측유형
    'incident_state', 'continuity_min', 'refire_count', 'predicted_fault_type',
    'unified_risk_score', 'unified_risk_level', 'hot_area', 'hot_score',
    'affected_areas', 'propagation_chain',
    'flow_signals', 'maxcapa_signals',
    'M16HUB_score', 'M14_score', 'M14B_score', 'M16A_score', 'M16B_score',
    'M16_score', 'M16_PKT_score', 'M16_WT_score',
    'M16HUB_signals', 'M14_signals', 'M14B_signals', 'M16A_signals', 'M16B_signals',
    'M16HUB_ra', 'M14_ra', 'M14B_ra', 'M16A_ra', 'M16B_ra',
    'M16HUB_rb_diff30', 'M14_rb_diff30', 'M14B_rb_diff30', 'M16A_rb_diff30',
    'M16B_rb_diff30',                            # ★ A 보강: 대칭 복구
    'M16HUB_rd_fab', 'M16HUB_stb_util',
    'M16HUB_rev_count', 'M16HUB_rev_lids',
    'sla_M14', 'sla_M14B', 'sla_M16A', 'sla_M16B', 'sla_M16HUB',  # ★ A 보강: sla_M14B
    'sorter_M14', 'sorter_M14B', 'sorter_M16A', 'sorter_M16B',
    'sorter_M16HUB',                             # ★ A 보강: sorter_M16HUB
    'reason',
    # ─────────────────────────────────────────────────────────
    # ★ A 보강 — 룰별 분해 점수 (영역 5 × 9 룰 = 45 컬럼)
    # RA/RA_sus/RB/RB_fast/RC/RD/SLA/SORT/MAXCAPA 9 룰 모두 분해
    'M16HUB_pts_RA', 'M16HUB_pts_RA_sus', 'M16HUB_pts_RB', 'M16HUB_pts_RB_fast',
    'M16HUB_pts_RC', 'M16HUB_pts_RD', 'M16HUB_pts_SLA', 'M16HUB_pts_SORT', 'M16HUB_pts_MAXCAPA',
    'M14_pts_RA', 'M14_pts_RA_sus', 'M14_pts_RB', 'M14_pts_RB_fast',
    'M14_pts_RC', 'M14_pts_RD', 'M14_pts_SLA', 'M14_pts_SORT', 'M14_pts_MAXCAPA',
    'M14B_pts_RA', 'M14B_pts_RA_sus', 'M14B_pts_RB', 'M14B_pts_RB_fast',
    'M14B_pts_RC', 'M14B_pts_RD', 'M14B_pts_SLA', 'M14B_pts_SORT', 'M14B_pts_MAXCAPA',
    'M16A_pts_RA', 'M16A_pts_RA_sus', 'M16A_pts_RB', 'M16A_pts_RB_fast',
    'M16A_pts_RC', 'M16A_pts_RD', 'M16A_pts_SLA', 'M16A_pts_SORT', 'M16A_pts_MAXCAPA',
    'M16B_pts_RA', 'M16B_pts_RA_sus', 'M16B_pts_RB', 'M16B_pts_RB_fast',
    'M16B_pts_RC', 'M16B_pts_RD', 'M16B_pts_SLA', 'M16B_pts_SORT', 'M16B_pts_MAXCAPA',
    # 통합 점수 분해 (5 컬럼)
    'layer1_total', 'flow_score', 'sla_score_total', 'sorter_score_total', 'mc_score_total',
    # ─────────────────────────────────────────────────────────
    # ★ B 보강 — 룰 진단 키 (운영자 디버깅 + ML 피처)
    # RA 진단: 10분 중 임계 초과 횟수 (5 영역)
    'M16HUB_ra_count', 'M14_ra_count', 'M14B_ra_count', 'M16A_ra_count', 'M16B_ra_count',
    # RB 진단: 10분 변화량 (5 영역)
    'M16HUB_rb_diff10', 'M14_rb_diff10', 'M14B_rb_diff10', 'M16A_rb_diff10', 'M16B_rb_diff10',
    # RC 진단: M16HUB 리프터 20분 트렌드 + M14 CNV 편향
    'M16HUB_rc_trend', 'M14_cnv_skew',
    # RD 진단: OHT 가동률 (다른 영역 R-D 핵심)
    'M14_rd_oht', 'M14B_rd_oht', 'M16A_rd_oht', 'M16B_rd_oht',
    # SLA 진단: 4분 초과 카운트 (4 영역)
    'M16HUB_sla_cnt', 'M14_sla_cnt', 'M16A_sla_cnt', 'M16B_sla_cnt',
    # Sorter 실패 (M16A/M16B)
    'M16A_sorter_fail', 'M16B_sorter_fail',
    # 영역 점수 원본 (50 클리핑 전, 5 영역) — score 가 50 캡 됐는지 확인용
    'M16HUB_score_raw', 'M14_score_raw', 'M14B_score_raw', 'M16A_score_raw', 'M16B_score_raw',
]

INCIDENT_FIELDS = [
    'file', 'date', 'predict_time', 'start_time', 'end_time',
    'max_risk_time',   # ★ 최고점(진짜 몰림) 시각 — 보고서 대표 시각
    'lead_min', 'duration_min', 'refire_count',
    'max_risk_score', 'max_risk_level',
    # ★ v6.3 — 사건 대표 장애 유형 (예: HUB-FAB정체 / HUB-리프터역증가 / 광역정체)
    'predicted_fault_type',
    'hot_area', 'affected_areas', 'propagation_chain',
    # ★ 상세 트리거 정보 (어떤 룰/컬럼이 발동했는지)
    'triggered_rules',     # 영역별 발동 룰 (예: "M16HUB:RA+RC+RD; M14:RA_sus+SLA")
    'risk_factors',        # 핵심 위험요인 (예: "M16HUB.AVGTOTALTIME1MIN=12.5분(>=9), ...")
    'maxcapa_changes',     # 사건 내 변경된 운영자 변수
    'relation',            # 영역별 핵심 컬럼-값-임계값 상세
    # ─────────────────────────────────────────────────────────
    # ★ A 보강 — 룰별 점수 max (5 영역 × 9 룰 = 45)
    'M16HUB_max_pts_RA', 'M16HUB_max_pts_RA_sus', 'M16HUB_max_pts_RB',
    'M16HUB_max_pts_RB_fast', 'M16HUB_max_pts_RC', 'M16HUB_max_pts_RD',
    'M16HUB_max_pts_SLA', 'M16HUB_max_pts_SORT', 'M16HUB_max_pts_MAXCAPA',
    'M14_max_pts_RA', 'M14_max_pts_RA_sus', 'M14_max_pts_RB',
    'M14_max_pts_RB_fast', 'M14_max_pts_RC', 'M14_max_pts_RD',
    'M14_max_pts_SLA', 'M14_max_pts_SORT', 'M14_max_pts_MAXCAPA',
    'M14B_max_pts_RA', 'M14B_max_pts_RA_sus', 'M14B_max_pts_RB',
    'M14B_max_pts_RB_fast', 'M14B_max_pts_RC', 'M14B_max_pts_RD',
    'M14B_max_pts_SLA', 'M14B_max_pts_SORT', 'M14B_max_pts_MAXCAPA',
    'M16A_max_pts_RA', 'M16A_max_pts_RA_sus', 'M16A_max_pts_RB',
    'M16A_max_pts_RB_fast', 'M16A_max_pts_RC', 'M16A_max_pts_RD',
    'M16A_max_pts_SLA', 'M16A_max_pts_SORT', 'M16A_max_pts_MAXCAPA',
    'M16B_max_pts_RA', 'M16B_max_pts_RA_sus', 'M16B_max_pts_RB',
    'M16B_max_pts_RB_fast', 'M16B_max_pts_RC', 'M16B_max_pts_RD',
    'M16B_max_pts_SLA', 'M16B_max_pts_SORT', 'M16B_max_pts_MAXCAPA',
    # ★ A 보강 — 룰별 점수 sum (45 컬럼)
    'M16HUB_sum_pts_RA', 'M16HUB_sum_pts_RA_sus', 'M16HUB_sum_pts_RB',
    'M16HUB_sum_pts_RB_fast', 'M16HUB_sum_pts_RC', 'M16HUB_sum_pts_RD',
    'M16HUB_sum_pts_SLA', 'M16HUB_sum_pts_SORT', 'M16HUB_sum_pts_MAXCAPA',
    'M14_sum_pts_RA', 'M14_sum_pts_RA_sus', 'M14_sum_pts_RB',
    'M14_sum_pts_RB_fast', 'M14_sum_pts_RC', 'M14_sum_pts_RD',
    'M14_sum_pts_SLA', 'M14_sum_pts_SORT', 'M14_sum_pts_MAXCAPA',
    'M14B_sum_pts_RA', 'M14B_sum_pts_RA_sus', 'M14B_sum_pts_RB',
    'M14B_sum_pts_RB_fast', 'M14B_sum_pts_RC', 'M14B_sum_pts_RD',
    'M14B_sum_pts_SLA', 'M14B_sum_pts_SORT', 'M14B_sum_pts_MAXCAPA',
    'M16A_sum_pts_RA', 'M16A_sum_pts_RA_sus', 'M16A_sum_pts_RB',
    'M16A_sum_pts_RB_fast', 'M16A_sum_pts_RC', 'M16A_sum_pts_RD',
    'M16A_sum_pts_SLA', 'M16A_sum_pts_SORT', 'M16A_sum_pts_MAXCAPA',
    'M16B_sum_pts_RA', 'M16B_sum_pts_RA_sus', 'M16B_sum_pts_RB',
    'M16B_sum_pts_RB_fast', 'M16B_sum_pts_RC', 'M16B_sum_pts_RD',
    'M16B_sum_pts_SLA', 'M16B_sum_pts_SORT', 'M16B_sum_pts_MAXCAPA',
    # 통합 점수 분해 max + sum (10 컬럼)
    'max_layer1_total', 'max_flow_score', 'max_sla_score', 'max_sorter_score', 'max_mc_score',
    'sum_layer1_total', 'sum_flow_score', 'sum_sla_score', 'sum_sorter_score', 'sum_mc_score',
    # ─────────────────────────────────────────────────────────
    # ★ B 보강 — 진단 키 max + sum (사건 분석용)
    # ra_count / rb_diff10 / rc_trend / rd_oht / sla_cnt / sorter_fail / cnv_skew / score_raw
    'M16HUB_max_ra_count','M14_max_ra_count','M14B_max_ra_count','M16A_max_ra_count','M16B_max_ra_count',
    'M16HUB_max_rb_diff10','M14_max_rb_diff10','M14B_max_rb_diff10','M16A_max_rb_diff10','M16B_max_rb_diff10',
    'M16HUB_max_rc_trend','M14_max_cnv_skew',
    'M14_max_rd_oht','M14B_max_rd_oht','M16A_max_rd_oht','M16B_max_rd_oht',
    'M16HUB_max_sla_cnt','M14_max_sla_cnt','M16A_max_sla_cnt','M16B_max_sla_cnt',
    'M16A_max_sorter_fail','M16B_max_sorter_fail',
    'M16HUB_max_score_raw','M14_max_score_raw','M14B_max_score_raw','M16A_max_score_raw','M16B_max_score_raw',
    # sum 동일 패턴
    'M16HUB_sum_ra_count','M14_sum_ra_count','M14B_sum_ra_count','M16A_sum_ra_count','M16B_sum_ra_count',
    'M16HUB_sum_rb_diff10','M14_sum_rb_diff10','M14B_sum_rb_diff10','M16A_sum_rb_diff10','M16B_sum_rb_diff10',
    'M14_sum_rd_oht','M14B_sum_rd_oht','M16A_sum_rd_oht','M16B_sum_rd_oht',
    'M16HUB_sum_sla_cnt','M14_sum_sla_cnt','M16A_sum_sla_cnt','M16B_sum_sla_cnt',
    'M16A_sum_sorter_fail','M16B_sum_sorter_fail',
    'M16HUB_sum_score_raw','M14_sum_score_raw','M14B_sum_score_raw','M16A_sum_score_raw','M16B_sum_score_raw',
]


def append_rows_csv(path, fields, rows):
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    last_err = None
    for attempt in range(10):
        try:
            with open(path, 'a', encoding='utf-8-sig', newline='') as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(fields)
                for r in rows:
                    w.writerow(r)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.3)
    raise last_err


def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, float):
        return f"{v:.2f}"
    return v


def _build_reason(ctx):
    """ 분 단위 reason — 어느 컬럼/룰이 발동중인지 상세 """
    parts = []
    hot = ctx.get('hot_area')
    if hot:
        parts.append(f"hot_area={hot}")
    if ctx.get('unified_s3'):
        parts.append("S3확정")
    elif ctx.get('unified_s2'):
        parts.append("S2주의보")
    elif ctx.get('unified_s1'):
        parts.append("S1조기경보")
    # 영역별 발동한 룰 컬럼 상세
    ar = ctx.get('area_results', {}) or {}
    rule_parts = []
    for area, r in ar.items():
        if r.get('area_score', 0) == 0:
            continue
        sub = []
        if r.get('ra_trig'):
            v = r.get('ra_value')
            col = RA_COL.get(area, '?').split('.')[-1]
            sub.append(f"R-A'({col}={v:.2f}분/기준{TH_RA.get(area, '?')})")
        if r.get('ra_sustained'):
            sub.append('R-A_sus')
        if r.get('rb_trig'):
            d = r.get('rb_diff_30', 0)
            col = RB_COL.get(area, '?').split('.')[-1]
            sub.append(f"R-B({col}+{d}/30분/기준+{TH_RB_30.get(area, '?')})")
        if r.get('rb_fast'):
            d = r.get('rb_diff_10', 0)
            sub.append(f"R-B_fast(+{d}/10분)")
        if r.get('rc_trig'):
            if area == 'M16HUB':
                sub.append(f"R-C'(역증가{r.get('rev_count', 0)}개:{','.join(r.get('rev_lids') or [])})")
            else:
                sub.append("R-C'(CNV쏠림)")
        if r.get('rd_trig'):
            if area == 'M16HUB':
                sub.append(f"R-D(FAB저장={r.get('rd_fab', 0):.1f}%,STB={r.get('hub_stb_util', 0):.1f}%)")
            else:
                sub.append(f"R-D(OHT={r.get('rd_oht', 0):.1f}%)")
        if r.get('sla_trig'):
            sub.append(f"SLA({r.get('sla_ratio', 0):.1f}%4분초과)")
        if r.get('sorter_trig'):
            sub.append(f"Sorter({r.get('sorter_val', 0)}LOT)")
        if r.get('maxcapa_changed_n', 0) > 0:
            sub.append(f"MAXCAPA{r.get('maxcapa_changed_n')}개변경")
        if sub:
            rule_parts.append(f"{area}[{','.join(sub)}]")
    if rule_parts:
        parts.append('발동: ' + '; '.join(rule_parts))
    fs = ctx.get('flow_signals')
    if fs:
        parts.append(f"흐름:{fs}")
    mc = ctx.get('maxcapa_signals')
    if mc:
        parts.append(f"운영자조치:{mc}")
    return '; '.join(parts)


def event_to_row(ev, file_name):
    t = ev['time']
    ctx = ev.get('ctx', {})
    ar = ctx.get('area_results', {}) or {}
    stage = ev['stage']
    transition = f"{ev['prev_stage']}→{stage}" if ev.get('is_transition') and stage != 0 else ''
    reason = _build_reason(ctx) if stage > 0 else ''

    def A(area, key, default=''):
        return ar.get(area, {}).get(key, default)

    return [
        file_name, t.strftime('%Y-%m-%d %H:%M'), t.strftime('%Y-%m-%d'), t.strftime('%H:%M'),
        stage, STAGE_LABEL.get(stage, ''), ev['prev_stage'], transition,
        # ★ v6 신규 — 사건 지속성/재발생/예측유형
        ev.get('incident_state', 'IDLE'), ev.get('continuity_min', 0),
        ev.get('refire_count', 0), ev.get('predicted_fault_type', ''),
        ctx.get('unified_risk_score', 0), ctx.get('unified_risk_level', ''),
        ctx.get('hot_area', ''), ctx.get('hot_score', 0),
        ctx.get('affected_areas', ''), ctx.get('propagation_chain', ''),
        ctx.get('flow_signals', ''), ctx.get('maxcapa_signals', ''),
        A('M16HUB', 'area_score', 0), A('M14', 'area_score', 0),
        A('M14B', 'area_score', 0), A('M16A', 'area_score', 0),
        A('M16B', 'area_score', 0), A('M16', 'area_score', 0),
        A('M16_PKT', 'area_score', 0), A('M16_WT', 'area_score', 0),
        '+'.join(A('M16HUB', 'area_signals', []) or []),
        '+'.join(A('M14', 'area_signals', []) or []),
        '+'.join(A('M14B', 'area_signals', []) or []),
        '+'.join(A('M16A', 'area_signals', []) or []),
        '+'.join(A('M16B', 'area_signals', []) or []),
        _fmt(A('M16HUB', 'ra_value')), _fmt(A('M14', 'ra_value')),
        _fmt(A('M14B', 'ra_value')), _fmt(A('M16A', 'ra_value')),
        _fmt(A('M16B', 'ra_value')),
        A('M16HUB', 'rb_diff_30', 0), A('M14', 'rb_diff_30', 0),
        A('M14B', 'rb_diff_30', 0), A('M16A', 'rb_diff_30', 0),
        A('M16B', 'rb_diff_30', 0),                                  # ★ A 보강
        _fmt(A('M16HUB', 'rd_fab')), _fmt(A('M16HUB', 'hub_stb_util')),
        A('M16HUB', 'rev_count', 0), ','.join(A('M16HUB', 'rev_lids', []) or []),
        _fmt(A('M14', 'sla_ratio')), _fmt(A('M14B', 'sla_ratio')),   # ★ A 보강: M14B
        _fmt(A('M16A', 'sla_ratio')), _fmt(A('M16B', 'sla_ratio')),
        _fmt(A('M16HUB', 'sla_ratio')),
        A('M14', 'sorter_val', 0), A('M14B', 'sorter_val', 0),
        A('M16A', 'sorter_val', 0), A('M16B', 'sorter_val', 0),
        A('M16HUB', 'sorter_val', 0),                                # ★ A 보강
        reason,
        # ─────────────────────────────────────────────────────────
        # ★ A 보강 — 룰별 분해 점수 9 룰 (영역 5 × 9 = 45 컬럼)
        A('M16HUB','pts_RA',0), A('M16HUB','pts_RA_sus',0), A('M16HUB','pts_RB',0),
        A('M16HUB','pts_RB_fast',0), A('M16HUB','pts_RC',0), A('M16HUB','pts_RD',0),
        A('M16HUB','pts_SLA',0), A('M16HUB','pts_SORT',0), A('M16HUB','pts_MAXCAPA',0),
        A('M14','pts_RA',0), A('M14','pts_RA_sus',0), A('M14','pts_RB',0),
        A('M14','pts_RB_fast',0), A('M14','pts_RC',0), A('M14','pts_RD',0),
        A('M14','pts_SLA',0), A('M14','pts_SORT',0), A('M14','pts_MAXCAPA',0),
        A('M14B','pts_RA',0), A('M14B','pts_RA_sus',0), A('M14B','pts_RB',0),
        A('M14B','pts_RB_fast',0), A('M14B','pts_RC',0), A('M14B','pts_RD',0),
        A('M14B','pts_SLA',0), A('M14B','pts_SORT',0), A('M14B','pts_MAXCAPA',0),
        A('M16A','pts_RA',0), A('M16A','pts_RA_sus',0), A('M16A','pts_RB',0),
        A('M16A','pts_RB_fast',0), A('M16A','pts_RC',0), A('M16A','pts_RD',0),
        A('M16A','pts_SLA',0), A('M16A','pts_SORT',0), A('M16A','pts_MAXCAPA',0),
        A('M16B','pts_RA',0), A('M16B','pts_RA_sus',0), A('M16B','pts_RB',0),
        A('M16B','pts_RB_fast',0), A('M16B','pts_RC',0), A('M16B','pts_RD',0),
        A('M16B','pts_SLA',0), A('M16B','pts_SORT',0), A('M16B','pts_MAXCAPA',0),
        # 통합 점수 분해 (5 컬럼)
        ctx.get('layer1_total', 0),
        ctx.get('flow_score', 0),
        ctx.get('sla_score', 0),
        ctx.get('sorter_score', 0),
        ctx.get('mc_score', 0),
        # ─────────────────────────────────────────────────────────
        # ★ B 보강 — 룰 진단 키 (운영자 디버깅 + ML 피처)
        A('M16HUB','ra_count',0), A('M14','ra_count',0), A('M14B','ra_count',0),
        A('M16A','ra_count',0), A('M16B','ra_count',0),
        A('M16HUB','rb_diff_10',0), A('M14','rb_diff_10',0), A('M14B','rb_diff_10',0),
        A('M16A','rb_diff_10',0), A('M16B','rb_diff_10',0),
        A('M16HUB','rc_trend',0), _fmt(A('M14','cnv_skew')),
        _fmt(A('M14','rd_oht')), _fmt(A('M14B','rd_oht')),
        _fmt(A('M16A','rd_oht')), _fmt(A('M16B','rd_oht')),
        A('M16HUB','sla_cnt',0), A('M14','sla_cnt',0),
        A('M16A','sla_cnt',0), A('M16B','sla_cnt',0),
        A('M16A','sorter_fail_val',0), A('M16B','sorter_fail_val',0),
        A('M16HUB','area_score_raw',0), A('M14','area_score_raw',0),
        A('M14B','area_score_raw',0), A('M16A','area_score_raw',0),
        A('M16B','area_score_raw',0),
    ]


def _predict_fault_type_from_incident(c):
    """사건의 area_max + triggered_rules 보고 대표 장애 유형 결정 (v6.3 동일 로직)."""
    hot = c.get('hot_area', '')
    if not hot:
        return ''
    affected = c.get('affected_areas_union', set()) or set()
    if len(affected) >= 4:
        return '광역정체'
    am = (c.get('area_max', {}) or {}).get(hot, {}) or {}
    rules = (c.get('triggered_rules', {}) or {}).get(hot, set()) or set()
    if hot == 'M16HUB':
        rev = am.get('rev_count', 0) or 0
        if rev >= TH_RC_REVERSE:                        return 'HUB-리프터역증가'
        fab = am.get('rd_fab', 0) or 0
        if fab >= TH_RD_FABSTORAGE:                     return 'HUB-FAB정체'
        stb = am.get('hub_stb_util', 0) or 0
        if stb >= TH_RD_HUB_STB_UTIL:                   return 'HUB-STB정체'
        rb = am.get('rb_diff_30', 0) or 0
        if rb >= TH_RB_30.get('M16HUB', 100):           return 'HUB-큐누적'
        ra = am.get('ra_value', 0) or 0
        if ra >= TH_RA.get('M16HUB', 9):                return 'HUB-반송지연'
        return 'HUB-신호'
    if 'SORT' in rules:                                 return f'{hot}-Sorter대기'
    if 'SLA' in rules:                                  return f'{hot}-SLA초과'
    if 'RD' in rules:                                   return f'{hot}-OHT정체'
    if 'RC' in rules and hot == 'M14':                  return 'M14-CNV쏠림'
    if 'RA' in rules or 'RA_sus' in rules:              return f'{hot}-반송지연'
    if 'RB' in rules or 'RB_fast' in rules:             return f'{hot}-큐누적'
    return f'{hot}-신호'


def incident_to_row(c, file_name):
    duration_min = round((c['end_time'] - c['start_time']).total_seconds() / 60.0, 1)
    lead_min = round((c['start_time'] - c['predict_time']).total_seconds() / 60.0)

    # 영역별 발동 룰
    tr_parts = []
    for area in ['M16HUB', 'M14', 'M14B', 'M16A', 'M16B', 'M16', 'M16_PKT', 'M16_WT']:
        rules = c.get('triggered_rules', {}).get(area)
        if rules:
            tr_parts.append(f"{area}:{'+'.join(sorted(rules))}")
    triggered_rules_s = '; '.join(tr_parts)

    # 핵심 위험요인 (영역별 max값 / 임계값 비교)
    rf_parts = []
    rel_parts = []
    for area in ['M16HUB', 'M14', 'M14B', 'M16A', 'M16B', 'M16_PKT', 'M16_WT']:
        am = c.get('area_max', {}).get(area)
        if not am:
            continue
        ra_max = am.get('ra_value')
        if ra_max is not None and area in TH_RA and ra_max >= TH_RA[area]:
            col = RA_COL.get(area, '?')
            rf_parts.append(f"{col}={ra_max:.2f}분(>={TH_RA[area]})")
            rel_parts.append(f"[{area} R-A'] {col}={ra_max:.2f}분 (기준 {TH_RA[area]}분)")
        rb_max = am.get('rb_diff_30')
        if rb_max is not None and area in TH_RB_30 and rb_max >= TH_RB_30[area]:
            col = RB_COL.get(area, '?')
            rf_parts.append(f"{col} +{rb_max}/30분(>={TH_RB_30[area]})")
            rel_parts.append(f"[{area} R-B] {col} +{rb_max}/30분 (기준 +{TH_RB_30[area]})")
        if area == 'M16HUB':
            rev = am.get('rev_count') or 0
            if rev >= TH_RC_REVERSE:
                rf_parts.append(f"M16HUB 리프터 역증가 {rev}개(>={TH_RC_REVERSE})")
                rel_parts.append(f"[M16HUB R-C'] 리프터 역증가 {rev}개 (기준 {TH_RC_REVERSE})")
            rdf = am.get('rd_fab') or 0
            if rdf >= TH_RD_FABSTORAGE:
                rf_parts.append(f"M16HUB.FABSTORAGERATIO={rdf:.1f}%(>={TH_RD_FABSTORAGE}%)")
                rel_parts.append(f"[M16HUB R-D] FABSTORAGERATIO={rdf:.1f}% (기준 {TH_RD_FABSTORAGE}%)")
            stb = am.get('hub_stb_util') or 0
            if stb >= TH_RD_HUB_STB_UTIL:
                rf_parts.append(f"M16HUB.STB_STORAGE_UTIL={stb:.1f}%(>={TH_RD_HUB_STB_UTIL}%)")
                rel_parts.append(f"[M16HUB R-D] STB.3F_STORAGE_UTIL={stb:.1f}% (기준 {TH_RD_HUB_STB_UTIL}%)")
        else:
            oht = am.get('rd_oht') or 0
            if oht >= TH_RD_OHT_UTIL:
                col = RD_OHT_COL.get(area, '?')
                rf_parts.append(f"{col}={oht:.1f}%(>={TH_RD_OHT_UTIL}%)")
                rel_parts.append(f"[{area} R-D] {col}={oht:.1f}% (기준 {TH_RD_OHT_UTIL}%)")
        sla_max = am.get('sla_ratio')
        if sla_max is not None and area in TH_SLA_RATIO and sla_max >= TH_SLA_RATIO[area]:
            col = SLA_COL.get(area, '?')
            rf_parts.append(f"{col}={sla_max:.1f}%(>={TH_SLA_RATIO[area]}%)")
            rel_parts.append(f"[{area} SLA] {col}={sla_max:.1f}% 4분초과 (기준 {TH_SLA_RATIO[area]}%)")
        sort_max = am.get('sorter_val')
        if sort_max is not None and area in TH_SORTER_WAIT and sort_max >= TH_SORTER_WAIT[area]:
            col = SORTER_COL.get(area, '?')
            rf_parts.append(f"{col}={sort_max}(>={TH_SORTER_WAIT[area]})")
            rel_parts.append(f"[{area} Sorter] {col}={sort_max} LOT대기 (기준 {TH_SORTER_WAIT[area]})")

    risk_factors_s = '; '.join(rf_parts) if rf_parts else ''
    maxcapa_changes_s = '; '.join(sorted(c.get('maxcapa_history', []) or []))
    relation_s = ' | '.join(rel_parts) if rel_parts else ''

    # ★ 신규 — 룰별 점수 max / sum (9 룰 × 5 영역 × 2 = 90) + 통합 10 + 진단 max/sum
    def PM(area, sig):
        return (c.get('area_pts_max', {}).get(area, {}) or {}).get(sig, 0)
    def PS(area, sig):
        return (c.get('area_pts_sum', {}).get(area, {}) or {}).get(sig, 0)
    def DM(area, k):
        return (c.get('area_diag_max', {}).get(area, {}) or {}).get(k, 0)
    def DS(area, k):
        return (c.get('area_diag_sum', {}).get(area, {}) or {}).get(k, 0)

    SIGS = ('RA', 'RA_sus', 'RB', 'RB_fast', 'RC', 'RD', 'SLA', 'SORT', 'MAXCAPA')
    AREAS5 = ('M16HUB', 'M14', 'M14B', 'M16A', 'M16B')
    pts_max_vals = [PM(a, s) for a in AREAS5 for s in SIGS]
    pts_sum_vals = [PS(a, s) for a in AREAS5 for s in SIGS]
    um = c.get('unified_pts_max', {}) or {}
    us = c.get('unified_pts_sum', {}) or {}
    unified_max_vals = [um.get(k, 0) for k in
                        ('layer1_total', 'flow_score', 'sla_score', 'sorter_score', 'mc_score')]
    unified_sum_vals = [us.get(k, 0) for k in
                        ('layer1_total', 'flow_score', 'sla_score', 'sorter_score', 'mc_score')]

    # B 보강 — 진단 키 max / sum
    diag_max_vals = (
        [DM(a, 'ra_count')    for a in AREAS5] +
        [DM(a, 'rb_diff_10')  for a in AREAS5] +
        [DM('M16HUB', 'rc_trend'), DM('M14', 'cnv_skew')] +
        [DM(a, 'rd_oht')      for a in ('M14','M14B','M16A','M16B')] +
        [DM(a, 'sla_cnt')     for a in ('M16HUB','M14','M16A','M16B')] +
        [DM(a, 'sorter_fail_val') for a in ('M16A','M16B')] +
        [DM(a, 'area_score_raw') for a in AREAS5]
    )
    diag_sum_vals = (
        [DS(a, 'ra_count')    for a in AREAS5] +
        [DS(a, 'rb_diff_10')  for a in AREAS5] +
        [DS(a, 'rd_oht')      for a in ('M14','M14B','M16A','M16B')] +
        [DS(a, 'sla_cnt')     for a in ('M16HUB','M14','M16A','M16B')] +
        [DS(a, 'sorter_fail_val') for a in ('M16A','M16B')] +
        [DS(a, 'area_score_raw') for a in AREAS5]
    )

    return [
        file_name, c['start_time'].strftime('%Y-%m-%d'),
        c['predict_time'].strftime('%H:%M'), c['start_time'].strftime('%H:%M'),
        c['end_time'].strftime('%H:%M'),
        c.get('max_risk_time', c['start_time']).strftime('%H:%M'),   # ★ 최고점 시각
        lead_min, duration_min, c['refire_count'],
        c['max_risk_score'], c['max_risk_level'],
        _predict_fault_type_from_incident(c),   # ★ v6.3 신규
        c['hot_area'], ';'.join(sorted(c['affected_areas_union'])),
        c['propagation_chain'],
        triggered_rules_s, risk_factors_s, maxcapa_changes_s, relation_s,
    ] + pts_max_vals + pts_sum_vals + unified_max_vals + unified_sum_vals \
      + diag_max_vals + diag_sum_vals


def append_event_row(out_dir, ev, file_name):
    ymd = ev['time'].strftime('%Y%m%d')
    path = os.path.join(out_dir, f'{ymd}_발동이벤트.csv')
    row = event_to_row(ev, file_name)
    append_rows_csv(path, EVENT_FIELDS, [row])
    # Logpresso 적재 (file 컬럼은 Rule_LO 에서 'Rule_system' 으로 하드코딩)
    if _logpresso is not None:
        _logpresso.upload(EVENT_FIELDS, row)
    # Graph_LO — ≥주의 시 raw 복사 + 그래프 자동 생성
    if _graph is not None:
        _graph.trigger(EVENT_FIELDS, row)
    return path


def append_incident_row(out_dir, incident, file_name):
    ymd = incident['start_time'].strftime('%Y%m%d')
    path = os.path.join(out_dir, f'{ymd}_사건단위.csv')
    row = incident_to_row(incident, file_name)
    append_rows_csv(path, INCIDENT_FIELDS, [row])
    # ★ 신규 — Logpresso 사건단위 적재 (file='Rule_incident')
    if _logpresso is not None:
        try:
            _logpresso.upload_incident(INCIDENT_FIELDS, row)
        except AttributeError:
            # 구 버전 Rule_LO 호환 (upload_incident 없으면 일반 upload)
            _logpresso.upload(INCIDENT_FIELDS, row)
    return path


# ============================================================
# Predictor
# ============================================================
AREAS_ALL = ['M16HUB', 'M14', 'M14B', 'M16A', 'M16B', 'M16', 'M16_PKT', 'M16_WT']



class Predictor:
    def __init__(self, input_csv: Path, out_dir: Path, logger):
        self.input_csv = Path(input_csv)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.file_name = self.input_csv.name
        self.area_windows = {a: deque(maxlen=WINDOW_MIN) for a in AREAS_ALL}
        self.flow_history = deque(maxlen=WINDOW_MIN)
        self.propagation_history = deque(maxlen=PREDICT_LOOKBACK_MIN * 2)
        self.tracker = IncidentTracker()
        self.last_t = None
        self.last_event_count = 0
        self.last_incident_count = 0

    def tick(self):
        if not self.input_csv.exists():
            self.logger.warning(f"입력 CSV 없음: {self.input_csv}")
            return 0
        new_rows = 0
        try:
            for d in iter_unified_rows(self.input_csv):
                t = d['time']
                # 분 단위 정규화 — 같은 분에 초만 다른 입력 행 중복 방지
                t = t.replace(second=0, microsecond=0)
                if self.last_t is not None and t <= self.last_t:
                    continue
                self.last_t = t
                d['time'] = t
                new_rows += 1

                for a in AREAS_ALL:
                    self.area_windows[a].append(d.get(a) or {})

                flow_now = {}
                m14 = d.get('M14') or {}
                m14b = d.get('M14B') or {}
                m16a = d.get('M16A') or {}
                m16b = d.get('M16B') or {}
                m16hub = d.get('M16HUB') or {}
                flow_now['M14_CNV_TO_HUB'] = m14.get('cnv_m14a_m16a')
                flow_now['M14_TO_HUB_JOB'] = m14.get('rb')
                flow_now['M14B_7F_TO_HUB'] = m14b.get('rb')
                flow_now['M14B_LFT_4ABLD_SUM'] = m14b.get('lft_4abld_sum')
                flow_now['M14B_LFT_4ABLD_TO_HUB_SUM'] = m14b.get('lft_4abld_to_hub_sum')
                flow_now['M16A_6F_TO_HUB'] = m16a.get('rb')
                flow_now['M16A_2F_TO_HUB'] = m16a.get('inflow_2f')
                flow_now['M16B_10F_TO_HUB'] = m16b.get('rb')
                flow_now['HUB_OHT_QCNT'] = m16hub.get('oht_qcnt')
                flow_now['M14_TO_M16'] = m16hub.get('rb')
                self.flow_history.append(flow_now)

                if len(self.area_windows['M16HUB']) < 31:
                    continue

                area_results = {a: eval_area_rules(a, self.area_windows[a]) for a in AREAS_ALL}
                flow_result = eval_flow_rules(self.flow_history)
                unified = evaluate_unified(t, area_results, flow_result, self.propagation_history)
                unified['area_results'] = area_results
                unified['flow_result'] = flow_result

                self.tracker.update(t,
                                    unified['unified_s1'],
                                    unified['unified_s2'],
                                    unified['unified_s3'],
                                    unified)

                while self.last_event_count < len(self.tracker.events):
                    ev = self.tracker.events[self.last_event_count]
                    append_event_row(self.out_dir, ev, self.file_name)
                    self.last_event_count += 1
                    if ev['stage'] >= 1:
                        c = ev.get('ctx', {})
                        self.logger.info(
                            f"  ▶ {ev['time'].strftime('%m-%d %H:%M')} "
                            f"단계{ev['stage']} ({STAGE_LABEL[ev['stage']]}) "
                            f"score={c.get('unified_risk_score', 0)} "
                            f"hot={c.get('hot_area', '')} "
                            f"affected={c.get('affected_areas', '')}"
                        )

                while self.last_incident_count < len(self.tracker.incidents):
                    inc = self.tracker.incidents[self.last_incident_count]
                    p = append_incident_row(self.out_dir, inc, self.file_name)
                    self.last_incident_count += 1
                    self.logger.info(
                        f"  ★ 사건 종료 → {inc['start_time'].strftime('%Y-%m-%d %H:%M')}~"
                        f"{inc['end_time'].strftime('%H:%M')} hot={inc['hot_area']} "
                        f"max={inc['max_risk_score']} 저장: {os.path.basename(p)}"
                    )

        except Exception as e:
            self.logger.exception(f"tick 오류: {e}")
        return new_rows

    def finalize(self):
        self.tracker.finalize(self.last_t)
        while self.last_incident_count < len(self.tracker.incidents):
            inc = self.tracker.incidents[self.last_incident_count]
            append_incident_row(self.out_dir, inc, self.file_name)
            self.last_incident_count += 1


def sleep_until_next_minute(offset_sec=SYNC_OFFSET_SEC):
    now = time.time()
    sec_in_min = now % 60
    wait = (60 - sec_in_min) + offset_sec
    if wait > 60:
        wait -= 60
    if wait < 0.05:
        wait += 60
    time.sleep(wait)


def run_once(input_csv: Path, out_dir: Path, logger):
    # ★ 백테스트 모드는 Logpresso 적재 절대 안 함 (테스트 데이터가 운영 DB 에 들어가면 안 됨)
    global _logpresso
    if _logpresso is not None:
        logger.info("[백테스트] Logpresso 적재 비활성화 (운영 DB 보호)")
        _logpresso = None
    logger.info("=" * 70)
    logger.info("M16 HUBROOM 통합 예측기 v4.1 — 일괄 처리 모드")
    logger.info(f"  INPUT : {input_csv}")
    logger.info(f"  OUTPUT: {out_dir}")
    logger.info(f"  대상 영역: {', '.join(AREAS_ALL)}")
    logger.info("=" * 70)
    p = Predictor(input_csv, out_dir, logger)
    n = p.tick()
    p.finalize()
    logger.info(f"처리 완료: {n}행 / 이벤트 {p.last_event_count}건 / 사건 {p.last_incident_count}건")


def run_watch(input_csv: Path, out_dir: Path, logger):
    logger.info("=" * 70)
    logger.info("M16 HUBROOM 통합 예측기 v4.1 — 실시간 감시 모드 (00초+5초 offset)")
    logger.info(f"  INPUT : {input_csv}")
    logger.info(f"  OUTPUT: {out_dir}")
    logger.info(f"  대상 영역: {', '.join(AREAS_ALL)}")
    logger.info("=" * 70)
    if _logpresso is not None:
        _logpresso.start()
    if _graph is not None:
        _graph.start()
    p = Predictor(input_csv, out_dir, logger)
    logger.info("[INIT] 시작 시점 윈도우 채우기...")
    n0 = p.tick()
    logger.info(f"[INIT] 초기 {n0}행 처리. 다음 분 동기 대기 시작...")
    try:
        while True:
            sleep_until_next_minute(SYNC_OFFSET_SEC)
            cycle_start = time.time()
            n = p.tick()
            elapsed = time.time() - cycle_start
            now_s = datetime.now().strftime('%H:%M:%S')
            if n > 0:
                logger.info(f"[{now_s}] tick: 신규 {n}행, {elapsed:.2f}s")
            else:
                logger.info(f"[{now_s}] tick: 신규 행 없음, {elapsed:.2f}s")
    except KeyboardInterrupt:
        logger.info("사용자 중단 (Ctrl+C)")
        p.finalize()
        if _logpresso is not None:
            _logpresso.stop()
        if _graph is not None:
            _graph.stop()
        logger.info("종료")


def main():
    input_csv = DEFAULT_INPUT_CSV
    out_dir = DEFAULT_OUTPUT_DIR
    watch_mode = False
    args = sys.argv[1:]
    i = 0
    if i < len(args) and not args[i].startswith('-'):
        input_csv = Path(args[i])
        i += 1
    while i < len(args):
        a = args[i]
        if a == '-o' and i + 1 < len(args):
            out_dir = Path(args[i + 1])
            i += 2
        elif a == '--watch':
            watch_mode = True
            i += 1
        elif a in ('-h', '--help'):
            print(__doc__)
            return
        else:
            i += 1
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)
    if not Path(input_csv).exists() and not watch_mode:
        logger.error(f"입력 CSV 없음: {input_csv}")
        sys.exit(1)
    if watch_mode:
        run_watch(Path(input_csv), out_dir, logger)
    else:
        run_once(Path(input_csv), out_dir, logger)


if __name__ == '__main__':
    main()
