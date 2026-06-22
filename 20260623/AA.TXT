#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py - OHT 월드모델 시뮬레이션 설정
"""

import os
import re
import pathlib

# ============================================================
# 경로 설정 (개발/PyInstaller 빌드 양쪽 지원)
# ============================================================
try:
    from app_paths import runtime_dir
    _SCRIPT_DIR = runtime_dir()
except ImportError:
    _SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
_PROJECT_DIR = _SCRIPT_DIR.parent  # 상위 폴더 (구 구조 호환)


def _resolve_data_root(name: str) -> pathlib.Path:
    """OHT_MAP / OHS_DATA_MD 같은 데이터 폴더를 자동 탐색.
       1) 스크립트 폴더 안(파생/OHT_MAP/)  ← 권장
       2) 상위 폴더(WORLD_MODEL/OHT_MAP/)   ← 구 구조
       둘 다 없으면 1번 경로를 그대로 반환(없어도 main.py 에서 에러 처리).
    """
    here = _SCRIPT_DIR / name
    up   = _PROJECT_DIR / name
    if here.exists():
        return here
    if up.exists():
        return up
    return here   # 기본값(없으면 이 경로 기준으로 동작)


# 데이터 폴더 — 파생/ 안에 있으면 그걸, 없으면 상위 폴더 것 사용
DATA_DIR = _resolve_data_root("OHS_DATA_MD")
MAP_DIR  = _resolve_data_root("OHT_MAP")

# 레이아웃/설정 파일
LAYOUT_CACHE_JSON = MAP_DIR / "layout_cache.json"        # M14A 기본 캐시 (하위호환)
LAYOUT_CACHE_DIR = MAP_DIR / "cache"                      # FAB별 캐시 저장 폴더
FAB_CONFIG_JSON = MAP_DIR / "fab_config.json"
HID_ZONE_MASTER_CSV = MAP_DIR / "MAP" / "M14A" / "HID_Zone_Master_M14A_A.csv"

# 기본 선택
DEFAULT_FAB = "M14A"
DEFAULT_PREFIX = "A"


# ============================================================
# FAB 카탈로그 스캔 — MAP/<FAB>/<PREFIX>.layout.zip 을 훑어 목록화
# ============================================================
def scan_fab_catalog():
    """
    OHT_MAP/MAP/ 밑을 훑어 사용 가능한 FAB/prefix 목록 반환.
    Returns:
        [{"fab": "M16A", "prefix": "BR",
          "layout_zip": ".../BR.layout.zip",
          "layout_cache": ".../cache/M16A_BR_layout_cache.json",
          "hid_csv": ".../HID_Zone_Master_M16A_BR.csv"}, ...]
    """
    catalog = []
    map_root = MAP_DIR / "MAP"
    if not map_root.exists():
        return catalog

    for fab_dir in sorted(map_root.iterdir()):
        if not fab_dir.is_dir():
            continue
        fab = fab_dir.name  # e.g. "M14A", "M16A"

        for item in sorted(fab_dir.iterdir()):
            if not item.is_file():
                continue
            # <PREFIX>.layout.zip 만 매칭
            m = re.match(r'^([A-Za-z0-9]+)\.layout\.zip$', item.name)
            if not m:
                continue
            prefix = m.group(1)
            layout_zip = item
            hid_csv = fab_dir / f"HID_Zone_Master_{fab}_{prefix}.csv"
            cache_file = LAYOUT_CACHE_DIR / f"{fab}_{prefix}_layout_cache.json"

            catalog.append({
                "fab": fab,
                "prefix": prefix,
                "layout_zip": str(layout_zip),
                "layout_cache": str(cache_file),
                "hid_csv": str(hid_csv) if hid_csv.exists() else "",
            })

    return catalog


FAB_CATALOG = scan_fab_catalog()


def get_fab_entry(fab: str, prefix: str):
    """FAB/prefix에 해당하는 카탈로그 엔트리 반환 (없으면 None)"""
    for e in FAB_CATALOG:
        if e["fab"] == fab and e["prefix"] == prefix:
            return e
    return None


# ============================================================
# 날짜별 데이터 폴더 — FAB_PREFIX_DATA 패턴으로 스캔
# ============================================================
def _match_files(files, *keywords):
    """파일 목록에서 키워드(여러 후보) 중 하나가 포함된 첫 파일 반환 (대소문자 무시)"""
    for kw in keywords:
        if not kw:
            continue
        kw_l = kw.lower()
        for f in files:
            if kw_l in f.lower():
                return f
    return None


def _csv_has_header(folder, filename, *cols):
    """폴더 안 CSV의 첫 줄(헤더)에 지정 컬럼이 하나라도 포함되어 있나 확인."""
    try:
        with open(os.path.join(str(folder), filename), 'r', encoding='utf-8') as f:
            header = f.readline().lower()
    except OSError:
        return False
    return any(c.lower() in header for c in cols)


def _autopick_oht_csv(folder, files):
    """폴더 안 CSV 중 OHT 파일을 자동 식별.
       (1) 파일명에 OHT/oht 포함 우선
       (2) 그래도 없으면 헤더에 ADDRESS/NEXT_ADDRESS 또는 'line' 포함된 CSV.
       Returns: (parsed_filename or None, raw_filename or None)
       parsed: ADDRESS 컬럼 있는 25컬럼 파싱본
       raw:    'line' 컬럼만 있는 UDP 원본
    """
    csvs = [f for f in files if f.lower().endswith('.csv')]
    parsed = None
    raw = None
    # 1차 — 파일명 우선순위 + 헤더 확인
    for f in csvs:
        if _csv_has_header(folder, f, 'ADDRESS') and _csv_has_header(folder, f, 'NEXT_ADDRESS'):
            parsed = parsed or f
        elif _csv_has_header(folder, f, '"line"', ',line,', '"line'):
            raw = raw or f
    # 2차 — 'OHT' 키워드만 보고 fallback
    if not (parsed or raw):
        oht_like = next((f for f in csvs if 'oht' in f.lower()), None)
        if oht_like:
            if _csv_has_header(folder, oht_like, 'ADDRESS'):
                parsed = oht_like
            else:
                raw = oht_like
    return parsed, raw


def _build_meta_for_folder(folder, fab, prefix, name):
    """단일 폴더(날짜 서브폴더든 단일 폴더든)에서 메타 한 건 생성.
       OHT 파일이 없으면 None 반환."""
    fab_num_match = re.search(r'(\d+)', fab)
    fab_num = fab_num_match.group(1) if fab_num_match else ""
    merged = f"m{fab_num}{prefix}".lower()
    ts_resource_key = f"ts_resource_{merged}"
    oht_data_key = f"oht_data_{merged}"

    try:
        files = os.listdir(str(folder))
    except OSError:
        return None
    if not files:
        return None

    # 우선 기존 키워드 매칭
    oht_raw = _match_files(files, 'OHT_DATA', f'OHT_{name}', 'oht_raw')
    oht_data_m14a = _match_files(files, oht_data_key)

    # fallback — 파일명이 임의여도 헤더로 자동 식별 (1개 CSV만 들어오는 운영 구조)
    if not (oht_raw or oht_data_m14a):
        autop, autor = _autopick_oht_csv(folder, files)
        oht_data_m14a = oht_data_m14a or autop
        oht_raw       = oht_raw       or autor or autop  # raw 슬롯도 채워야 _load_oht_raw 진입
    if not (oht_raw or oht_data_m14a):
        return None

    hid_inout    = _match_files(files, 'HID_INOUT')
    rail_cut     = _match_files(files, 'RAIL_CUT')
    star         = _match_files(files, 'STAR_OHT', '컬럼수집')
    ts_resource  = _match_files(files, ts_resource_key, 'ts_resource')
    oht_time_avg = _match_files(files, 'oht_time_avg')
    csv_count = sum(1 for f in files if f.lower().endswith('.csv'))
    return {
        "dir": folder,
        "fab": fab,
        "prefix": prefix,
        "oht_raw": oht_raw or oht_data_m14a,
        "hid_inout": hid_inout,
        "rail_cut": rail_cut,
        "star": star,
        "ts_resource": ts_resource,
        "oht_data_m14a": oht_data_m14a,
        "oht_time_avg": oht_time_avg,
        "time_range": ("00:00:00", "23:59:59"),
        "description": (f"{name[:4]}-{name[4:6]}-{name[6:8]} ({csv_count}개 CSV)"
                        if re.match(r'^\d{8}$', name) else f"{name} ({csv_count}개 CSV)"),
    }


def _scan_date_folder(folder, fab, prefix):
    """<folder>/YYYYMMDD/*.csv 스캔 → {date_key: meta} 반환.
       날짜 서브폴더가 없는 1-CSV 운영 구조(<folder>/*.csv)도 지원."""
    result = {}
    if not folder.exists() or not folder.is_dir():
        return result

    # FAB+prefix에서 파일명 매칭용 키 생성
    # 예: M14A/A → fab_num=14, suffix=a → "oht_data_m14a", "ts_resource_m14a"
    fab_num_match = re.search(r'(\d+)', fab)
    fab_num = fab_num_match.group(1) if fab_num_match else ""
    merged = f"m{fab_num}{prefix}".lower()  # e.g. m14a, m16br, m16a, m16e
    oht_data_key = f"oht_data_{merged}"
    ts_resource_key = f"ts_resource_{merged}"

    saw_date_subfolder = False
    for date_folder in sorted(folder.iterdir()):
        if not date_folder.is_dir():
            continue
        name = date_folder.name
        if not re.match(r'^\d{6,8}$', name):
            continue
        saw_date_subfolder = True

        try:
            files = os.listdir(str(date_folder))
        except OSError:
            continue
        if not files:
            continue

        oht_raw = _match_files(files, 'OHT_DATA', f'OHT_{name}', 'oht_raw')
        oht_data_m14a = _match_files(files, oht_data_key)
        # fallback — 임의 파일명도 자동 식별
        if not (oht_raw or oht_data_m14a):
            autop, autor = _autopick_oht_csv(date_folder, files)
            oht_data_m14a = oht_data_m14a or autop
            oht_raw       = oht_raw       or autor or autop
        if not (oht_raw or oht_data_m14a):
            continue

        hid_inout = _match_files(files, 'HID_INOUT')
        rail_cut = _match_files(files, 'RAIL_CUT')
        star = _match_files(files, 'STAR_OHT', '컬럼수집')
        ts_resource = _match_files(files, ts_resource_key, 'ts_resource')
        oht_time_avg = _match_files(files, 'oht_time_avg')

        csv_count = sum(1 for f in files if f.lower().endswith('.csv'))

        result[name] = {
            "dir": date_folder,
            "fab": fab,
            "prefix": prefix,
            "oht_raw": oht_raw,
            "hid_inout": hid_inout,
            "rail_cut": rail_cut,
            "star": star,
            "ts_resource": ts_resource,
            "oht_data_m14a": oht_data_m14a,  # 키 이름 유지 — 값은 FAB별 파싱본
            "oht_time_avg": oht_time_avg,
            "time_range": ("00:00:00", "23:59:59"),
            "description": f"{name[:4]}-{name[4:6]}-{name[6:8]} ({csv_count}개 CSV)",
        }

    return result


def _scan_data_by_fab():
    """
    OHS_DATA_MD/ 하위를 훑어 FAB별 데이터 날짜 맵 생성.

    인식 패턴:
      1) OHS_DATA_MD/<FAB>_<PREFIX>_DATA/<YYYYMMDD>/   (신규, 권장)
      2) OHS_DATA_MD/<FAB>_DATA/<YYYYMMDD>/            → prefix는 'A'로 가정
      3) OHS_DATA_MD/<YYYYMMDD>/                        → DEFAULT_FAB/DEFAULT_PREFIX로 매핑 (하위호환)

    Returns:
        {"M14A/A": {"20260421": {meta}}, "M16A/BR": {...}, ...}
    """
    result = {}
    if not DATA_DIR.exists():
        return result

    for entry in sorted(DATA_DIR.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name

        # 패턴 3: 날짜 직접 → 기본 FAB (하위호환)
        if re.match(r'^\d{6,8}$', name):
            key = f"{DEFAULT_FAB}/{DEFAULT_PREFIX}"
            bucket = result.setdefault(key, {})
            files = os.listdir(str(entry))
            oht_raw = _match_files(files, 'OHT_DATA')
            if not oht_raw:
                continue
            fab_num = re.search(r'(\d+)', DEFAULT_FAB).group(1)
            merged = f"m{fab_num}{DEFAULT_PREFIX}".lower()
            bucket[name] = {
                "dir": entry,
                "fab": DEFAULT_FAB,
                "prefix": DEFAULT_PREFIX,
                "oht_raw": oht_raw,
                "hid_inout": _match_files(files, 'HID_INOUT'),
                "rail_cut": _match_files(files, 'RAIL_CUT'),
                "star": _match_files(files, 'STAR_OHT', '컬럼수집'),
                "ts_resource": _match_files(files, f'ts_resource_{merged}', 'ts_resource'),
                "oht_data_m14a": _match_files(files, f'oht_data_{merged}'),
                "oht_time_avg": _match_files(files, 'oht_time_avg'),
                "time_range": ("00:00:00", "23:59:59"),
                "description": f"{name[:4]}-{name[4:6]}-{name[6:8]} ({sum(1 for f in files if f.lower().endswith('.csv'))}개 CSV)",
            }
            continue

        # 패턴 1/2: <FAB>[_<PREFIX>]_DATA
        m = re.match(r'^([A-Z0-9]+?)(?:_([A-Za-z0-9]+))?_DATA$', name)
        if not m:
            continue
        fab = m.group(1)
        prefix = m.group(2) or DEFAULT_PREFIX

        key = f"{fab}/{prefix}"
        dates = _scan_date_folder(entry, fab, prefix)
        if dates:
            result.setdefault(key, {}).update(dates)

    return result


DATA_BY_FAB = _scan_data_by_fab()

# 하위호환 — 기본 FAB(M14A/A)의 날짜 맵을 DATA_DATES로 노출
DATA_DATES = DATA_BY_FAB.get(f"{DEFAULT_FAB}/{DEFAULT_PREFIX}", {})


def get_dates_for_fab(fab: str, prefix: str):
    """FAB/prefix의 날짜 맵 반환"""
    return DATA_BY_FAB.get(f"{fab}/{prefix}", {})


# ============================================================
# OHT 시스템 상수 (MD 문서 기반 검증된 값)
# ============================================================

# 차량
VEHICLE_COUNT_M14A = 450
VEHICLE_COUNT_TOTAL = 1033  # V-Vehicle (R-Vehicle 29대 별도)

# TAT (Turn Around Time) - ts_resource 분석 결과
TAT_AVERAGE_MIN = 2.88      # 평균 2.88분
TAT_MEDIAN_MIN = 2.57       # 중앙값 2.57분
TAT_UNDER_3MIN_PCT = 61.5   # 3분 이내 완료 비율
TAT_UNDER_5MIN_PCT = 91.7   # 5분 이내 완료 비율

# 물동량
THROUGHPUT_PER_HOUR = 20000  # 시간당 ~20,000건 (주야간 동일)

# 큐
QUEUE_NORMAL_RANGE = (1200, 1400)  # 정상 큐 범위
QUEUE_AVG = 1249

# OBS (장애물 정지)
OBS_NORMAL_RANGE = (120, 160)   # 정상 OBS 범위
OBS_WARNING_THRESHOLD = 180      # 주의 기준
OBS_DANGER_THRESHOLD = 200       # 위험 기준
OBS_DEADLOCK_THRESHOLD = 300     # 데드락 기준

# 가동률/적재율
UTILIZATION_PCT = 82.3
LOADED_PCT = 62.1
OBS_BZ_STOP_PCT = 14.3

# LineCost 가중치 (Dijkstra 아키텍처)
LINECOST_IDLE_VHL_PENALTY = 3000    # 놀고 있는 차량 1대당 +3,000ms
LINECOST_WORK_VHL_PENALTY = 5000    # 일하는 차량 1대당 +5,000ms
LINECOST_WORK_DEST_PENALTY = 5000   # 대기 작업 1건당 +5,000ms

# EMA 속도 갱신
EMA_OLD_WEIGHT = 0.6
EMA_NEW_WEIGHT = 0.4

# HID Zone
HID_ZONE_COUNT = 182
HID_VEHICLE_MAX = 37
HID_VEHICLE_PRECAUTION = 35
HID_CONGESTION_THRESHOLD_PCT = 70  # 70% 이상이면 혼잡

# 시뮬레이션
SIM_STEP_SEC = 5.0          # 시뮬레이션 스텝 (초)
DEFAULT_VELOCITY = 180.0     # 기본 차량 속도 (m/min)
PREDICTION_HORIZONS = [600, 1200]  # 예측 시간 (초): 10분, 20분

# OHT 메시지 상태 코드
STATE_NAMES = {
    1: "RUN",
    2: "STOP",
    3: "ACCEL",
    4: "DECEL",
    5: "CURVE",
    6: "OBS_BZ_STOP",
    7: "JAM",
    8: "HT_STOP",
    9: "E84_TIMEOUT",
}

# 서버 설정
SERVER_HOST = "0.0.0.0"     # 모든 네트워크 인터페이스에서 접속 허용 (외부 접속 가능)
SERVER_PORT = 10005
WS_INTERVAL = 0.5  # WebSocket 전송 간격 (초)
