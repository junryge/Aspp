#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HID_VHL_OHT.py — 5 FAB OHT 병목 HID 구간 찾기 (따로 실행)

  50초마다 로그프레소에서 차량별 상태 · 위치(ADDRESS)를 받아
    ① 멈춘 차 = 미보고(끊기기 직전 위치) · JAM(STATUS 7) · HT_STOP(STATUS 8)
    ② 주소 → HID 구역 (OHT_MAP 의 HID_Zone_Master + 레이아웃)
    ③ 그 50초 구간 값으로 FAB 알람 판정 — 경계(전조) · 위험(사건) · 초위험(사건 시작)
    ④ 알람이면 멈춘 차가 많은 HID 구역 상위 TOP_N 개를 저장

  저장  HID_BOTTLENECK/{FAB}/HID_BOTTLENECK_{FAB}_YYYYMMDD.csv  (FAB 마다 폴더 · 파일 따로, 실시간 1분마다 한 줄)
        날짜, 시간, HID_ZONE, {FAB}_OHT_report, {FAB}_OHT_missing, {FAB}_OHT_JAM, ALARM_KR, ALARM_EN, HID_section, …  (15칸)
          ALARM_KR = 정상 / 경계 / 위험 / 초위험,  ALARM_EN = NORMAL / WARNING / DANGER / CRITICAL
          ★HT_STOP 은 판정에만 쓰고 CSV · 문제맵 · 히트맵 · 로그에는 안 보인다 (숨김)
          report = 그 50초 구간에 보고한 차량 수
          missing/JAM/HT_STOP = FAB 전체 (그 분이 끝날 때까지 끝난 마지막 50초 구간)
          ALARM        = 경계 / 위험 / 초위험, 아니면 정상
          HID_ZONE     = 병목 HID 구역 번호 — 문서의 "HID 4 · HID 33" 과 같은 번호 (그 50초 구간 멈춘 차 최다), 알람 아니면 0
                         마스터 ZONE_ID · ZONE_ID2 중 참조표와 더 맞는 칸을 시작할 때 고른다 (HID_ZONE_COL)
          HID_section  = 그 구역의 Bay_Zone, 알람 아니면 0
          ZONE_STOP    = 그 구역 안 멈춘 차 (미보고 + JAM + HT, 한 대 한 번)          ┐
          ZONE_VHL     = 그 구역 안 전체 차량 (보고 차 + 미보고 차 마지막 위치)       │ 그 분의 마지막
          VHL_MAX      = 마스터 Vehicle_Max                                           │ 50초 구간,
          ZONE_OCC     = 점유율 % = ZONE_VHL / VHL_MAX × 100 (VHL_MAX 0 이면 0)       │ 알람 아니면 0
          HID_ZONE_2/3 = 2위 · 3위 구역 (HID_ZONE 과 같은 번호)                        ┘
          데이터가 없는 분은 숫자 0 · ALARM 정상

  알람 (그 50초 구간의 FAB 전체 값 — 과거 구간을 끌고 오지 않음)  — ★FAB 마다 HID_CONFIG.json 에서 바꾼다 (아래는 기본값)
    단계마다 숫자 5개, 넷 중 하나라도 맞으면 (0 = 그 조건 끔):
      1. missing + JAM : 같은 50초 구간에 2개가 같이 기준 이상
      2. missing_ONLY  : missing 1개만 기준 이상
      3. HT_STOP       : HT_STOP 1개만 기준 이상
      4. ZONE_STOP     : HID 구역 하나 안에 멈춘 차(미보고 + JAM + HT)가 기준 이상
    경계   전조       missing 10↑ + JAM 10↑    또는  구역 멈춘 차 10↑
    위험   사건       missing 30↑ 단독        또는  HT_STOP 10↑
    초위험 사건 시작  missing 100↑ 단독       또는  HT_STOP 30↑

  ★기존 수집 데이터(M16A_HUBROOM_PR.csv 등)와 별개 — 로그프레소에서 차량 보고를 직접 받아
    보고 · 미보고 · JAM · HT_STOP 을 여기서 새로 계산한다. 다른 .py 파일 필요 없음.

  필요한 것 (같은 폴더)
    hdi_api_key.txt                     로그프레소 키 (첫 줄, 두 서버 같은 키)
    HID_CONFIG.json                     FAB 별 경계 · 위험 · 초위험 기준 (없으면 기본값으로 만든다)
    OHT_MAP/                            월드모델파생의 OHT_MAP 폴더 그대로
        MAP/{M14A,M14B,M16A,M16B}/HID_Zone_Master_*.csv
        cache/*_layout_cache.json
    oht3d/                              월드모델파생 static/js/oht3d (oht3d.js · three.module.min.js) — 문제맵 아이소메트리
    HID_MAP_SETTINGS.json               문제맵 표시 기본값 (OHT 크기 · 색 · 테마 · 히트맵 …, 없으면 만든다)

  ★문제맵 (예전 HID_PROBLEM_MAP.py 를 합쳤다)
    실시간으로 1분 줄을 CSV 에 쓸 때, ALARM 이 MAP_MIN_LEVEL(경계) 이상이면 그 1분 지도를 바로 만든다.
    판정 · 숫자는 그 줄 그대로, OHT 차량은 그 줄과 같은 50초 구간 (메모리에 있는 것 + 방향 · 적재만 한 번 더 조회).
      → HID_BOTTLENECK/PROBLEM_MAP/{FAB}_{YYYYMMDD}/PROBLEM_MAP_{FAB}_{YYYYMMDD}_{HHMM}_{ALARM}.html
         (ALARM = WARNING 경계 · DANGER 위험 · CRITICAL 초위험 · NORMAL 정상)
    지도: 월드모델파생 2D 맵 모양 · [2D][유사 3D][아이소메트리] · 히트맵(처음엔 꺼짐) · ⚙ 설정(OHT 크기 등)

  실행
    python HID_VHL_OHT.py                       계속 돈다 (Ctrl+C 로 멈춤)
    python HID_VHL_OHT.py --test                구간표 확인 + 최근 10분 한 번 판정 (저장 안 함)
    python HID_VHL_OHT.py --map                 주소 → HID 구역표를 CSV 로 내보냄 (확인용)
    python HID_VHL_OHT.py M16HUB 20260929                  과거 판정 — FAB 하루
    python HID_VHL_OHT.py M16HUB 20260912 20260929         과거 판정 — FAB 날짜 ~ 날짜 (하루씩 파일)
    python HID_VHL_OHT.py ALL 20260912 20260929            FAB 5개 다
        → HID_BOTTLENECK/PAST/{FAB}/HID_BOTTLENECK_{FAB}_YYYYMMDD.csv   (FAB · 날짜별, 다시 돌리면 덮어씀)
        → HID_BOTTLENECK/PAST/SUMMARY_{FAB|ALL}_{시작}_{끝}.csv     (FAB · 날짜별 정상 · 경계 · 위험 · 초위험 분 수)
    python HID_VHL_OHT.py --range 202609290750 202609290830 --fab M16HUB   분 단위 구간 (검증용)
    python HID_VHL_OHT.py --no-problem-map                실시간인데 문제맵은 안 만듦
    python HID_VHL_OHT.py M16HUB 20260929 --problem-map   과거 판정 + 경계 이상 분마다 문제맵
    python HID_VHL_OHT.py --map-at M16HUB 202609291348    그 1분만 판정해서 문제맵 하나 (여러 시각 가능)
"""
import argparse
import csv
import heapq
import io
import json
import logging
import sys
import time
import urllib.parse
from collections import Counter, OrderedDict, defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ==========================================================
# 설정 — 로그프레소
# ==========================================================
SERVERS = {                     # 서버별 접속 (key 비우면 hdi_api_key.txt 첫 줄)
    "M16": {"host": "10.40.42.167", "port": 8888, "key": "", "remote": "icamcslogdt01"},
    "M14": {"host": "10.40.42.27",  "port": 8888, "key": "", "remote": "icamcslogdt01"},
}
KEY_FILE        = "hdi_api_key.txt"
CONNECT_TIMEOUT = 3             # 서버에 못 붙으면 3초 만에 포기
HTTP_TIMEOUT    = 30
BUCKET_SEC      = 50            # ★50초 구간
LAG_SEC         = 30            # 구간 끝나고 이만큼 지난 뒤에 받는다 (로그프레소 적재 지연)
FLEET_AUTO_MIN  = 100          # 미보고 = (최근 100분 안에 보고한 차량 수) − (그 50초에 보고한 차량 수)
FLEET_MIN_SEEN  = 2            # 전체 차량에는 50초 구간 2개 이상에서 보인 차만 센다
                               #   (9/22 10:03 처럼 한 구간만 보고 차량이 455대로 튀어도 미보고가 부풀지 않게)

# ==========================================================
# 설정 — 판정
# ==========================================================
FABS = OrderedDict([            # FAB → 테이블 · 서버(hid_aws_idc_realtime_collector.SERVERS) · 맵(폴더, 접두)
    ("M14",    {"table": "oht_data_m14a",  "server": "M14", "map": ("M14A", "A")}),
    ("M14B",   {"table": "oht_data_m14b",  "server": "M14", "map": ("M14B", "A")}),
    ("M16A",   {"table": "oht_data_m16a",  "server": "M16", "map": ("M16A", "A")}),
    ("M16B",   {"table": "oht_data_m16b",  "server": "M16", "map": ("M16B", "B")}),
    ("M16HUB", {"table": "oht_data_m16br", "server": "M16", "map": ("M16A", "BR")}),
])
# ★돌릴 FAB 은 HID_CONFIG.json 의 "RUN_FABS" 에서 정한다 (아래는 파일에 없을 때의 기본값).
#   실시간 · --test 는 그 FAB 만 맵을 읽고 조회한다. 과거 판정은 FAB 를 적으면 그 FAB, ALL 이면 5개 다.
DEFAULT_RUN_FABS = ["M16HUB"]
RUN_FABS = list(DEFAULT_RUN_FABS)

# ★알람 기준은 FAB 마다 따로 HID_CONFIG.json 에서 정한다 (없으면 아래 기본값으로 파일을 만든다).
#   기본값은 M16HUB 37일 데이터로 정한 값이라, 다른 FAB 는 운영하며 따로 맞춰야 한다.
#   고치면 다시 켜지 않아도 다음 50초 구간부터 적용된다. 숫자를 0 으로 두면 그 조건은 끈다.
CONFIG_FILE = HERE / "HID_CONFIG.json"
DEFAULT_LEVELS = {              # 단계마다 숫자 5개 — missing + JAM 은 같이, missing_ONLY · HT_STOP · ZONE_STOP 은 혼자
    "경계":   {"missing": 10, "JAM": 10, "missing_ONLY": 0,   "HT_STOP": 0,  "ZONE_STOP": 10},
    "위험":   {"missing": 0,  "JAM": 0,  "missing_ONLY": 30,  "HT_STOP": 10, "ZONE_STOP": 0},
    "초위험": {"missing": 0,  "JAM": 0,  "missing_ONLY": 100, "HT_STOP": 30, "ZONE_STOP": 0},
}
CONFIG_HELP = [
    "★ FAB 마다 따로 설정해야 합니다 — M14 · M14B · M16A · M16B · M16HUB 는 차량 수와 평소 수준이 달라서 같은 숫자를 쓰면 안 맞습니다.",
    "FAB 마다 경계 / 위험 / 초위험 기준 (그 50초 구간 값). 넷 중 하나라도 맞으면 그 단계:",
    "  1. missing + JAM : 같은 50초 구간에 missing 이상 AND JAM 이상 — 2개가 같이 발생해야 함",
    "  2. missing_ONLY  : missing 이상 — 미보고 1개만 발생해도 됨",
    "  3. HT_STOP       : HT_STOP 이상 — 1개만 발생해도 됨",
    "  4. ZONE_STOP     : HID 구역 하나 안에 멈춘 차(미보고 + JAM + HT)가 이 숫자 이상 — 1개만 발생해도 됨",
    "초위험 -> 위험 -> 경계 순으로 본다. 숫자 0 = 그 조건 끔.",
    "고치면 프로그램을 다시 켜지 않아도 다음 50초 구간부터 적용됩니다.",
    "RUN_FABS = 돌릴 FAB (실시간 · --test). 예) [\"M16HUB\"] · 다섯 다 [\"M14\", \"M14B\", \"M16A\", \"M16B\", \"M16HUB\"] — 이것만은 프로그램을 다시 켜야 적용됩니다.",
]
CFG = {}                        # FAB → {"경계", "위험", "초위험"}
_cfg_mtime = None
# HID_ZONE 에 쓸 마스터 칸 — 문서의 "HID 4 · HID 33" 번호 (HID IN/OUT 로그의 HID 번호) 와 같은 칸
#   "auto"     참조표(HID_REF_M16HUB.csv)와 ZONE_ID · ZONE_ID2 를 맞춰 보고 더 맞는 칸 (참조표 없으면 ZONE_ID)
#   "ZONE_ID" / "ZONE_ID2"   고정
HID_ZONE_COL = "auto"
REF_FILE     = HERE / "HID_REF_M16HUB.csv"     # 문서 만들 때 쓴 주소 → HID 구간 (4/21 HID IN/OUT 로그)
HID_ZONE_USE = "ZONE_ID"                            # 시작할 때 정해진다
# ★HID 구역 = 마스터 Vehicle_Max > 0 인 구역만 (2026-10)
#   M16A_BR 마스터: 1~37 (37개, 정원 있음) = HID / 4001~ · 5001~ · 10001~ (161개, 정원 0) = HID 아님
#   HID 아닌 구역은 구역 계산(HID_ZONE · ZONE_STOP · 점유율)과 문제맵(구역 · 레인)에서 뺀다.
#   False 면 예전처럼 마스터의 모든 구역.
HID_ONLY_VMAX = True

WINDOW_MIN = 10                 # 메모리에 남겨 둘 과거 분 (판정은 그 50초 구간 하나로 한다)
TOP_N      = 3                  # 알람 때 저장할 HID 구역 수 (멈춘 차 많은 순)
IDLE_MIN   = 30                 # 이보다 오래 안 보인 차는 위치에서 뺀다 (정비 · 이탈)
FIRST_MIN  = 12                 # 처음 켤 때 받을 분
RANGE_CHUNK_MIN = 60            # --range 때 한 번에 받을 분
RANGE_TIMEOUT   = 180

MAP_DIRS = [HERE / "OHT_MAP", Path.cwd() / "OHT_MAP"]     # 앞에서부터 찾는다

# ★문제맵 — 1분 줄의 ALARM 이 이 단계 이상이면 그 1분 지도(HTML)를 만든다 (월드모델파생 맵 · OHT 차량 · 2D/유사3D/아이소메트리)
MAP_MIN_LEVEL = "경계"          # "경계" | "위험" | "초위험"
MAP_REALTIME  = True            # 실시간(계속 도는 모드)에서 만들기 — 끄려면 --no-problem-map
MAP_PAST      = False           # 과거 판정(FAB 날짜 · --range)에서도 만들기 — --problem-map 으로 켠다
OUT_DIR  = HERE / "HID_BOTTLENECK"     # 결과 폴더 (영문)
MAP_OUT  = OUT_DIR / "PROBLEM_MAP"     # 문제맵 — PROBLEM_MAP/{FAB}_{YYYYMMDD}/PROBLEM_MAP_{FAB}_{YYYYMMDD}_{HHMM}_{ALARM}.html
PAST_DIR = OUT_DIR / "PAST"           # 과거 판정 · --range 결과 — 실시간 파일과 섞이지 않게 따로
LOG_FILE = HERE / "HID_VHL_OHT.log"

ALARM_ORDER = ["경계", "위험", "초위험"]

log = logging.getLogger("HID병목")
log.setLevel(logging.INFO)
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s [HID병목] %(message)s")
    for _h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        _h.setFormatter(_fmt)
        log.addHandler(_h)


# ==========================================================
# 로그프레소 조회
# ==========================================================
EPOCH_KST = datetime(1970, 1, 1, 9, 0, 0)       # datetrunc(_time, "50s") 와 같은 기준


def bucket_floor(t):
    sec = int((t - EPOCH_KST).total_seconds()) // BUCKET_SEC * BUCKET_SEC
    return EPOCH_KST + timedelta(seconds=sec)


def parse_time(s):
    s = (s or "").strip().strip('"')[:19]
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    return None


_key_warned = False


def api_key(server):
    global _key_warned
    if SERVERS[server].get("key"):
        return SERVERS[server]["key"]
    for base in (HERE, Path.cwd()):
        p = base / KEY_FILE
        if p.exists():
            lines = [ln.strip().strip('"').strip("'") for ln in p.read_text(encoding="utf-8-sig").splitlines()
                     if ln.strip()]
            return lines[0] if lines else ""
    if not _key_warned:
        log.warning(f"  {KEY_FILE} 없음 — 이 파일 옆에 두세요")
        _key_warned = True
    return ""


def lp_get(server, q, timeout=None):
    import requests
    s = SERVERS[server]
    key = api_key(server)
    url = (f"http://{s['host']}:{s['port']}/logpresso/httpexport/query.csv"
           f"?_apikey={key}&_q={urllib.parse.quote(q, safe='')}")
    r = requests.get(url, verify=False, timeout=(CONNECT_TIMEOUT, timeout or HTTP_TIMEOUT))
    if r.status_code != 200 or r.text.lstrip().startswith("<"):
        ks = f"키 끝 4자 …{key[-4:]}" if key else f"키 없음 ({KEY_FILE})"
        what = "데이터 대신 HTML 화면 (키 · 쿼리 확인)" if r.text.lstrip().startswith("<") else r.text[:300]
        raise RuntimeError(f"HTTP {r.status_code} from {s['host']}:{s['port']} ({ks}) — {what}")
    return r.content


# ==========================================================
# 설정 파일 HID_CONFIG.json — FAB 별 경계 · 위험 · 초위험
# ==========================================================
def default_config():
    import copy
    out = {"_설명": CONFIG_HELP, "RUN_FABS": list(DEFAULT_RUN_FABS)}
    for fab in FABS:
        out[fab] = copy.deepcopy(DEFAULT_LEVELS)
    return out


def dump_config(cfg):
    """사람이 고치기 쉽게 — 단계마다 한 줄"""
    J = lambda o: json.dumps(o, ensure_ascii=False)
    lines = ["{", '  "_설명": [']
    lines += [f"    {J(t)}," for t in cfg.get("_설명", [])]
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("  ],")
    if "RUN_FABS" in cfg:
        lines.append(f'  "RUN_FABS": {J(cfg["RUN_FABS"])},')
    fabs = [k for k in cfg if k not in ("_설명", "RUN_FABS")]
    for n, fab in enumerate(fabs):
        c = cfg[fab]
        lines.append(f'  "{fab}": {{')
        lvs = [lv for lv in DEFAULT_LEVELS if lv in c]
        for k, lv in enumerate(lvs):
            pad = " " * (6 - len(lv) * 2)
            lines.append(f'    "{lv}": {pad}{J(c[lv])}{"," if k < len(lvs) - 1 else ""}')
        lines.append("  }" + ("," if n < len(fabs) - 1 else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


def _num(v, d):
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return d


def load_config(first=False):
    """HID_CONFIG.json 읽기 — 없으면 기본값으로 만들고, 빠진 값 · 잘못된 값은 기본값으로 채운다."""
    global CFG, _cfg_mtime, RUN_FABS
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(dump_config(default_config()), encoding="utf-8")
        log.info(f"  {CONFIG_FILE.name} 없음 — 기본값으로 만들었습니다 (FAB 마다 고쳐 쓰세요)")
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
    except Exception as e:
        log.warning(f"  {CONFIG_FILE.name} 읽기 실패 — {'기본값' if first else '이전 설정'}으로 계속: {e}")
        if first or not CFG:
            raw = default_config()
        else:
            _cfg_mtime = CONFIG_FILE.stat().st_mtime
            return
    new = {}
    for fab in FABS:
        f = raw.get(fab) or {}
        c = {}
        for lv, dv in DEFAULT_LEVELS.items():
            got = f.get(lv) or {}
            co = got.get("미보고_JAM_동시")              # 잠깐 썼던 묶음 형식도 받아 준다
            if isinstance(co, dict):
                got = {**co, "HT_STOP": got.get("HT_STOP")}
            if "missing_ONLY" not in got and got:    # 예전 형식(숫자 3개) → missing_ONLY 는 꺼진 채로 둔다
                got = {**got, "missing_ONLY": 0}
            if "ZONE_STOP" not in got and got:       # 예전 형식(숫자 3 · 4개) → ZONE_STOP 도 꺼진 채로 둔다
                got = {**got, "ZONE_STOP": 0}
            c[lv] = {k: _num(got.get(k), d) for k, d in dv.items()}
        new[fab] = c
    # ★예전 프로그램이 만든 파일에 남은 칸(전체차량 · 미보고_JAM_동시 · 구간수 등)은 지우고
    #   지금 형식으로 다시 쓴다. 숫자(missing · JAM · HT_STOP)는 고쳐 둔 값 그대로.
    rf = raw.get("RUN_FABS")                         # 돌릴 FAB — 모르는 이름은 빼고, 비면 기본값
    if isinstance(rf, str):
        rf = [x.strip() for x in rf.replace(" ", ",").split(",") if x.strip()]
    rf = [f.upper() for f in rf] if isinstance(rf, list) else []
    bad = [f for f in rf if f not in FABS]
    rf = [f for f in FABS if f in rf] or list(DEFAULT_RUN_FABS)
    if bad:
        log.warning(f"  {CONFIG_FILE.name} RUN_FABS 에 모르는 FAB {bad} — 뺍니다 (가능: {' '.join(FABS)})")
    if first:
        RUN_FABS = rf
    elif rf != RUN_FABS:
        log.warning(f"  RUN_FABS 바뀜 {RUN_FABS} → {rf} — 프로그램을 다시 켜야 적용됩니다")
    clean = {"_설명": CONFIG_HELP, "RUN_FABS": rf, **new}
    if raw != clean:
        try:
            CONFIG_FILE.write_text(dump_config(clean), encoding="utf-8")
            log.info(f"  {CONFIG_FILE.name} 정리 — 빠진 칸은 0(끔)으로 채우고 안 쓰는 칸(전체차량 등)은 지웠습니다 (숫자는 그대로)")
        except Exception as e:
            log.warning(f"  {CONFIG_FILE.name} 정리 실패 (읽기는 정상): {e}")
    CFG = new
    _cfg_mtime = CONFIG_FILE.stat().st_mtime
    log.info(f"  돌릴 FAB (RUN_FABS): {' '.join(RUN_FABS)}")
    for fab, c in CFG.items():
        if fab not in RUN_FABS:
            continue
        def one(lv):
            p = c[lv]
            a = f"missing{p['missing']}+JAM{p['JAM']}" if (p["missing"] and p["JAM"]) else "missing+JAM 끔"
            return (f"{lv} {a} / missing단독{p['missing_ONLY'] or ' 끔'}"
                    f" / ZONE_STOP{p['ZONE_STOP'] or ' 끔'}")
        log.info(f"  기준 {fab:6} {one('경계')} · {one('위험')} · {one('초위험')}")


def maybe_reload_config():
    try:
        if CONFIG_FILE.exists() and CONFIG_FILE.stat().st_mtime != _cfg_mtime:
            log.info(f"  {CONFIG_FILE.name} 바뀜 — 다시 읽습니다")
            load_config()
    except Exception as e:
        log.warning(f"  {CONFIG_FILE.name} 확인 실패: {e}")


def policy(fab):
    return CFG.get(fab) or DEFAULT_LEVELS


# ==========================================================
# ① OHT MAP 가공 — 주소 → HID 구역
# ==========================================================
def _lanes(s):
    out = []
    for part in (s or "").split(";"):
        part = part.strip().replace("->", "→")
        if "→" in part:
            a, b = [x.strip() for x in part.split("→", 1)]
            if a and b:
                out.append((a, b))
    return out


def _find_map_files(fab_dir, prefix):
    for base in MAP_DIRS:
        master = base / "MAP" / fab_dir / f"HID_Zone_Master_{fab_dir}_{prefix}.csv"
        layout = base / "cache" / f"{fab_dir}_{prefix}_layout_cache.json"
        if master.exists() and layout.exists():
            return master, layout
    return None, None


def build_zone_map(fab_dir, prefix, cap=400):
    """
    HID_Zone_Master 의 IN/OUT 레인 + 레이아웃 → {주소: ZONE_ID}, {ZONE_ID: (Full_Name, Bay_Zone)}
      ZONE_ID · ZONE_ID2 는 마스터에 적힌 값 그대로
      구역마다 IN 레인 도착 주소에서 출발해 레이아웃을 따라가되 그 구역 OUT 레인은 넘지 않는다.
      여러 구역에 걸리는 주소는 IN 에서 가장 가까운 구역 = 차가 마지막으로 들어간 구역.
    """
    master, layout = _find_map_files(fab_dir, prefix)
    if not master:
        return None, None, f"맵 파일 없음 (OHT_MAP/MAP/{fab_dir}/HID_Zone_Master_{fab_dir}_{prefix}.csv · " \
                           f"OHT_MAP/cache/{fab_dir}_{prefix}_layout_cache.json)"
    L = json.loads(layout.read_text(encoding="utf-8"))
    adj = {str(a): [str(b) for b in bs] for a, bs in L.get("adj", {}).items()}

    try:                                            # 엑셀로 저장한 파일은 cp949 일 수 있다
        text = master.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = master.read_text(encoding="cp949")
    rd = csv.DictReader(io.StringIO(text))
    cols = {c.strip().lower().replace(" ", "_"): c for c in (rd.fieldnames or [])}

    def col(r, *names):                             # 칸 이름 대소문자 · 공백 상관없이
        for n in names:
            c = cols.get(n)
            if c is not None:
                return (r.get(c) or "").strip()
        return ""

    if "zone_id" not in cols:
        return None, None, f"{master.name}: ZONE_ID 칸이 없음 (칸: {', '.join(rd.fieldnames or [])})"
    zones = {}
    for r in rd:
        z = col(r, "zone_id")                       # 구역 묶음 (IN/OUT 레인) 기준
        if not z or z == "0":
            continue
        d = zones.setdefault(z, {"name": col(r, "full_name") or f"Zone-{z}",
                                 "bay": col(r, "bay_zone"), "z2": "", "vmax": 0,
                                 "in": set(), "out": set()})
        if not d["vmax"]:
            vm = col(r, "vehicle_max")
            d["vmax"] = int(vm) if vm.isdigit() else 0     # 구역 정원 (점유율 분모)
        if not d["z2"]:
            d["z2"] = col(r, "zone_id2")            # ★HID_ZONE 에 쓰는 값 (구역의 첫 번째 ZONE_ID2)
        d["in"].update(_lanes(col(r, "in_lanes")))
        d["out"].update(_lanes(col(r, "out_lanes")))

    def zkey(z):                                    # 같은 거리면 번호 작은 구역
        return (0, int(z), z) if z.isdigit() else (1, 0, z)

    best = {}                                       # 주소 → (거리, 구역)
    for z, d in zones.items():
        seen = {}
        q = deque()
        for _, b in d["in"]:
            if b not in seen:
                seen[b] = 0
                q.append(b)
        while q and len(seen) <= cap:
            n = q.popleft()
            for m in adj.get(n, []):
                if m in seen or (n, m) in d["out"]:
                    continue
                seen[m] = seen[n] + 1
                q.append(m)
        for n, dist in seen.items():
            if n not in best or dist < best[n][0] or (dist == best[n][0] and zkey(z) < zkey(best[n][1])):
                best[n] = (dist, z)

    zone_of = {n: z for n, (_, z) in best.items()}
    n_all = len(zones)
    drop = []
    if HID_ONLY_VMAX:                               # ★HID 구역(정원 > 0)만 — 주소 배정은 그대로 두고 HID 아닌 구역 주소는 뺀다
        hid = {z for z, d in zones.items() if d["vmax"] > 0}
        if hid:
            drop = sorted((z for z in zones if z not in hid), key=zkey)
            zones = {z: d for z, d in zones.items() if z in hid}
            zone_of = {n: z for n, z in zone_of.items() if z in hid}
    info = {z: (d["name"], d["bay"], d["z2"], d["vmax"]) for z, d in zones.items()}
    ex = ", ".join(f"{z}→{zones[z]['z2'] or 0}" for z in sorted(zones, key=zkey)[:3])
    n2 = sum(1 for d in zones.values() if d["z2"])
    msg = (f"{master.name}: 구역 {len(zones)}"
           + (f" (HID 만 · Vehicle_Max 0 {len(drop)}개 제외: {', '.join(drop[:3])} …)" if drop else
              (" (★Vehicle_Max 없음 — 전체 구역)" if HID_ONLY_VMAX and n_all else ""))
           + f" · 주소 {len(zone_of)}/{len(L.get('nodes', {}))} · "
           f"ZONE_ID2 있는 구역 {n2} · 예 ZONE_ID→ZONE_ID2 {ex} …")
    if "zone_id2" not in cols:
        msg += " · ★ZONE_ID2 칸 없음 (HID_ZONE 0)"
    return zone_of, info, msg


# ==========================================================
# ② FAB 상태 — 50초 구간마다 멈춘 차를 HID 구역별로
# ==========================================================
class FabState:
    def __init__(self, fab, cfg):
        self.fab, self.cfg = fab, cfg
        self.zone_of, self.info, self.map_msg = build_zone_map(*cfg["map"])
        self.buckets = OrderedDict()                # 구간시작 → dict
        self.last = {}                              # 차량 → (마지막 보고 구간, ADDRESS)
        self.first = {}                             # 차량 → 처음 보고 구간 (자동 전체 대수)
        self.nseen = Counter()                      # 차량 → 보고한 50초 구간 수 (한 번만 보인 차는 전체에서 뺀다)
        self.seen = defaultdict(dict)               # 구간 → {차량: (상태들, 주소)} — 문제맵 (차량 위치 · 미보고 마지막 위치 · 속도)
        self.pm = None                              # 문제맵 재료 (레이아웃 · 지도 그림) — 처음 쓸 때 만든다
        self.done_to = None
        self.use_remote = None                      # None 모름 / False 바로 / True remote
        self.minute_done = None                     # 여기까지 1분 행을 썼다

    @property
    def ok(self):
        return self.zone_of is not None

    # ---------- 로그프레소 ----------
    def _query(self, frm, to, use_remote):
        inner = (f"table from={frm:%Y%m%d%H%M%S} to={to:%Y%m%d%H%M%S} {self.cfg['table']}"
                 ' | search MSG_ID == "2"'
                 " | fields _time, VEHICLE, STATUS, ADDRESS"
                 f' | eval _time = datetrunc(_time, "{BUCKET_SEC}s")'
                 " | stats first(STATUS) as STATUS, last(STATUS) as STATUS_LAST,"
                 " last(ADDRESS) as ADDRESS by VEHICLE, _time")
        remote = SERVERS[self.cfg["server"]].get("remote")
        return f"remote {remote} [ {inner} ]" if (use_remote and remote) else inner

    def fetch(self, frm, to, timeout=None):
        """remote 없이 먼저, 안 되면 remote — 되는 쪽을 기억한다."""
        import requests
        tries = [self.use_remote] if self.use_remote is not None else [False, True]
        last = None
        for m in tries:
            try:
                body = lp_get(self.cfg["server"], self._query(frm, to, m), timeout)
            except requests.exceptions.ConnectionError:
                raise
            except Exception as e:
                last = e
                continue
            if self.use_remote is None:
                self.use_remote = m
                log.info(f"  {self.fab}: {'remote' if m else 'remote 없이'} 조회로 확정")
            return body
        self.use_remote = None
        raise last

    # ---------- 쌓기 ----------
    def ingest(self, body, start, end):
        """조회 결과 → 50초 구간별 집계. 새로 생긴 (보고 있는) 구간 시각 목록을 돌려준다."""
        step = timedelta(seconds=BUCKET_SEC)
        rep, jam, ht, addr = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(dict)
        for r in csv.DictReader(io.StringIO(body.decode("utf-8-sig", "replace"))):
            v = (r.get("VEHICLE") or "").strip()
            t = parse_time(r.get("_time"))
            if not v or not t:
                continue
            b = bucket_floor(t)
            sts = {(r.get("STATUS") or "").strip(), (r.get("STATUS_LAST") or "").strip()}
            rep[b].add(v)
            addr[b][v] = (r.get("ADDRESS") or "").strip()
            self.seen[b][v] = (sts, addr[b][v])
            if "7" in sts:
                jam[b].add(v)
            if "8" in sts:
                ht[b].add(v)

        new = []
        b = start
        while b < end:
            n = len(rep[b])
            if n == 0:                               # 수집 누락 — 판정에서 뺀다
                self.buckets[b] = {"gap": True}
                b += step
                continue
            for v in rep[b]:
                if self.last.get(v, (None,))[0] != b:
                    self.nseen[v] += 1
                self.last[v] = (b, addr[b][v])
                self.first.setdefault(v, b)
            fleet = sum(                             # 최근 100분 안에 보고한 차량 수 (구간 2개 이상에서 보인 차만)
                1 for v, (lb, _) in self.last.items()
                if lb >= b - timedelta(minutes=FLEET_AUTO_MIN) and self.first[v] <= b
                and (v in rep[b] or self.nseen[v] >= FLEET_MIN_SEEN))
            miss = max(0, fleet - n)
            gap = miss >= fleet * 0.9 and not jam[b] and not ht[b]
            zm, zj, zh, zs, zv = Counter(), Counter(), Counter(), Counter(), Counter()
            if not gap:
                for v in rep[b]:                     # 구역 안 전체 차량 — 보고한 차
                    z = self.zone_of.get(addr[b][v])
                    if z:
                        zv[z] += 1
                        if v in jam[b] or v in ht[b]:
                            zs[z] += 1               # 멈춘 차 (JAM · HT, 한 대 한 번)
                idle = b - timedelta(minutes=IDLE_MIN)
                for v, (lb, a) in self.last.items():
                    if v in rep[b] or lb < idle or self.nseen[v] < FLEET_MIN_SEEN:
                        continue
                    z = self.zone_of.get(a)
                    if z:
                        zm[z] += 1
                        zs[z] += 1                   # 미보고도 멈춘 차
                        zv[z] += 1                   # 미보고 차는 끊기기 직전 위치
                for v in jam[b]:
                    z = self.zone_of.get(addr[b][v])
                    if z:
                        zj[z] += 1
                for v in ht[b]:
                    z = self.zone_of.get(addr[b][v])
                    if z:
                        zh[z] += 1
            self.buckets[b] = {"gap": gap, "n": n, "miss": miss, "jam": len(jam[b]), "ht": len(ht[b]),
                               "zm": zm, "zj": zj, "zh": zh, "zs": zs, "zv": zv}
            if not gap:
                new.append(b)
            b += step
        keep = start - timedelta(minutes=WINDOW_MIN + 5)   # 이번에 받은 구간들의 과거 10분은 남겨 둔다
        for k in [k for k in self.buckets if k < keep]:
            del self.buckets[k]
        old = start - timedelta(minutes=IDLE_MIN + 10)     # 문제맵 — 미보고 차의 마지막 위치를 찾을 만큼(30분+)만
        for k in [k for k in self.seen if k < old]:
            del self.seen[k]
        self.done_to = end
        return new

    # ---------- ③ 알람 ----------
    def window(self, b):
        lo = b - timedelta(minutes=WINDOW_MIN) + timedelta(seconds=BUCKET_SEC)
        return [(k, v) for k, v in self.buckets.items() if lo <= k <= b and not v.get("gap")]

    def alarm(self, b):
        """
        50초 구간 b 하나로 판정 → (ALARM, 이유) · 알람 없으면 (None, '')
          그 구간 값만 본다 (과거 구간을 끌고 오지 않는다) — 줄에 찍히는 숫자가 곧 판정 근거.
          1. missing + JAM : 같은 구간에 2개가 같이 기준 이상
          2. missing_ONLY  : missing 1개만 기준 이상
          3. HT_STOP       : 1개만 기준 이상
          4. ZONE_STOP     : 멈춘 차가 가장 많은 HID 구역 하나의 멈춘 차 수가 기준 이상
          높은 단계부터 (초위험 → 위험 → 경계), 기준은 HID_CONFIG.json (0 = 그 조건 끔)
        """
        v = self.buckets.get(b)
        if not v or v.get("gap"):
            return None, ""
        P = policy(self.fab)
        zmax = max(v["zs"].values(), default=0)     # 멈춘 차가 가장 많은 구역의 멈춘 차 (= 줄의 ZONE_STOP)
        for lv in ("초위험", "위험", "경계"):
            p = P[lv]
            hit_co = bool(p["missing"] and p["JAM"] and v["miss"] >= p["missing"] and v["jam"] >= p["JAM"])
            hit_mo = bool(p["missing_ONLY"] and v["miss"] >= p["missing_ONLY"])
            hit_ht = bool(p["HT_STOP"] and v["ht"] >= p["HT_STOP"])
            hit_zs = bool(p["ZONE_STOP"] and zmax >= p["ZONE_STOP"])
            if hit_co or hit_mo or hit_ht or hit_zs:
                why = []
                if hit_co:
                    why.append(f"missing {v['miss']} + JAM {v['jam']} 같이 (≥{p['missing']}+{p['JAM']})")
                if hit_mo:
                    why.append(f"missing {v['miss']} 단독 (≥{p['missing_ONLY']})")
                if hit_zs:
                    why.append(f"구역 멈춘 차 {zmax} (≥{p['ZONE_STOP']})")
                if not why:                          # HT_STOP 으로만 걸림 — HT_STOP 은 숨김이라 글자에 안 적는다
                    why.append("기준 충족")
                return lv, " · ".join(why)
        return None, ""

    # ---------- ④ 병목 HID 구역 ----------
    def top_zones(self, b, n=TOP_N):
        """그 50초 구간에 멈춘 차가 많은 구역 → [(Zone_ID, 미보고, JAM, HT, 멈춘 차)]"""
        v = self.buckets.get(b)
        if not v or v.get("gap"):
            return []
        return [(z, v["zm"].get(z, 0), v["zj"].get(z, 0), v["zh"].get(z, 0), c)
                for z, c in v["zs"].most_common(n) if c > 0]

    def bucket_of_minute(self, m):
        """1분 행 m(HH:MM) 의 기본 50초 구간 = 그 분이 끝날 때까지 끝난 마지막 구간"""
        return bucket_floor(m + timedelta(seconds=60)) - timedelta(seconds=BUCKET_SEC)

    def buckets_of_minute(self, m):
        """그 분 안에 끝난 50초 구간들 (1~2개)"""
        lo, hi = m, m + timedelta(seconds=60)
        out = [k for k, v in self.buckets.items()
               if not v.get("gap") and lo < k + timedelta(seconds=BUCKET_SEC) <= hi]
        return sorted(out)

    def minute_row(self, m):
        """
        1분 한 줄: 날짜, 시간, HID_ZONE, {FAB}_OHT_report, _missing, _JAM, _HT_STOP, ALARM, HID_section, …
          그 분 안에 끝난 50초 구간(1~2개) 중 가장 높은 단계 구간을 쓴다 (같으면 나중 구간).
          줄의 report · missing · JAM · HT_STOP · 구역 칸은 모두 **그 구간 값** — ALARM 과 숫자가 항상 맞는다.
          ALARM = 경계 / 위험 / 초위험 / 정상 · HID_ZONE · HID_section 등 구역 칸은 알람일 때만, 아니면 0
          데이터가 없으면 숫자 0 · ALARM 정상
        """
        rank = {None: 0, "경계": 1, "위험": 2, "초위험": 3}
        cands = self.buckets_of_minute(m) or [self.bucket_of_minute(m)]
        best, level = None, None
        for k in cands:
            lv, _ = self.alarm(k)
            if best is None or rank[lv] >= rank[level]:
                best, level = k, lv
        v = self.buckets.get(best)
        head = [f"{m:%Y-%m-%d}", f"{m:%H:%M}"]
        if not v or v.get("gap"):
            return head + [0, 0, 0, 0, 0, "정상", 0] + [0] * 6, None
        zone, bay, extra = 0, 0, [0] * 6
        if level:
            top = self.top_zones(best, 3)
            if top:
                z1 = top[0][0]
                zone = hid_zone(self, z1)
                _, bay, _, vmax = self.info.get(z1, ("", "", "", 0))
                bay = bay or 0
                zstop = v["zs"].get(z1, 0)
                zvhl = v["zv"].get(z1, 0)
                occ = round(zvhl / vmax * 100) if vmax else 0
                z2 = hid_zone(self, top[1][0]) if len(top) > 1 else 0
                z3 = hid_zone(self, top[2][0]) if len(top) > 2 else 0
                extra = [zstop, zvhl, vmax, occ, z2, z3]
        return head + [zone, v["n"], v["miss"], v["jam"], v["ht"], level or "정상", bay] + extra, level

    def minute_rows(self):
        """받은 데까지 아직 안 쓴 분을 1분씩 꺼낸다."""
        if self.done_to is None:
            return []
        if self.minute_done is None:
            first = min((k for k, v in self.buckets.items() if not v.get("gap")), default=None)
            if first is None:
                return []
            self.minute_done = first.replace(second=0) - timedelta(minutes=1)
        out = []
        m = self.minute_done + timedelta(minutes=1)
        while bucket_floor(m + timedelta(seconds=60)) <= self.done_to:    # 그 분의 마지막 구간까지 받았나
            out.append(self.minute_row(m)[0])
            self.minute_done = m
            m += timedelta(minutes=1)
        return out


# ==========================================================
# ⑤ 저장
# ==========================================================
def header(fab):
    return ["날짜", "시간", "HID_ZONE", f"{fab}_OHT_report", f"{fab}_OHT_missing", f"{fab}_OHT_JAM",
            "ALARM_KR", "ALARM_EN", "HID_section",
            "ZONE_STOP", "ZONE_VHL", "VHL_MAX", "ZONE_OCC", "HID_ZONE_2", "HID_ZONE_3"]


ALARM_EN_OF = {"정상": "NORMAL", "경계": "WARNING", "위험": "DANGER", "초위험": "CRITICAL"}


def csv_row(r):
    """안쪽 1분 줄(15칸) → CSV 줄(15칸): HT_STOP 은 빼고(숨김), ALARM 은 ALARM_KR(한글) · ALARM_EN(영문) 두 칸.
       ★HT_STOP 은 판정에만 쓰고 CSV · 화면에는 안 보인다."""
    return list(r[:6]) + [r[7], ALARM_EN_OF.get(r[7], "NORMAL")] + list(r[8:])


def fab_file(fab, day, out_dir=None):
    """FAB 마다 폴더 따로 — HID_BOTTLENECK/{FAB}/HID_BOTTLENECK_{FAB}_YYYYMMDD.csv"""
    d = (out_dir or OUT_DIR) / fab
    d.mkdir(parents=True, exist_ok=True)
    return d / f"HID_BOTTLENECK_{fab}_{day}.csv"


def ensure_file(fab, day, out_dir=None):
    """알람이 없어도 그 날 파일은 헤더만 있는 채로 만들어 둔다 (FAB 5개 다 보이게)."""
    p = fab_file(fab, day, out_dir)
    if not p.exists():
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(header(fab))
    return p


# ★실시간 1분 줄을 다른 곳(로그프레소 적재 — Rule_hid.py)에도 넘길 때 — run_oht.py 가 여기에 함수를 단다.
#   hook(fab, 헤더, CSV 줄들) — CSV 에 새로 쓴 줄만 넘긴다. 훅이 실패해도 CSV 저장 · 판정은 계속.
SAVE_HOOKS = []
_RULE_HID_ERR = ""
try:                                                # ★Rule_hid.py 가 옆에 있으면 CSV 에 쓰자마자 로그프레소 AMHS_VHL_OHT 에도 넣는다
    sys.path.insert(0, str(HERE))
    import Rule_hid as _RULE_HID
except ModuleNotFoundError:                         # 없으면 CSV · 문제맵만
    _RULE_HID = None
except Exception as _e:                             # 있는데 못 읽으면 — 이유를 로그에 남긴다 (판정은 계속)
    _RULE_HID, _RULE_HID_ERR = None, f"{type(_e).__name__}: {_e}"
_last_written = {}                                  # 파일 → 마지막으로 쓴 '날짜 시간' (다시 켰을 때 같은 분 중복 방지)


def _last_in_file(p):
    """파일 맨 끝 줄의 '날짜 시간' (없으면 '')"""
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode("utf-8-sig", "replace").strip().splitlines()
        for line in reversed(tail):
            c = line.split(",")
            if len(c) > 2 and c[0][:2] == "20":
                return f"{c[0]} {c[1]}"
    except OSError:
        pass
    return ""


def save(fab, rows, out_dir=None):
    if not rows:
        return
    by_day = defaultdict(list)
    for r in rows:
        by_day[r[0].replace("-", "")].append(r)
    for day, rs in by_day.items():
        p = ensure_file(fab, day, out_dir)
        if p not in _last_written:
            _last_written[p] = _last_in_file(p)
        rs = [r for r in rs if f"{r[0]} {r[1]}" > _last_written[p]]       # 이미 쓴 분은 건너뜀 (다시 켰을 때)
        if not rs:
            continue
        out = [csv_row(r) for r in rs]
        with open(p, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(out)
        _last_written[p] = f"{rs[-1][0]} {rs[-1][1]}"
        for hook in SAVE_HOOKS:
            try:
                hook(fab, header(fab), out)
            except Exception as e:
                log.warning(f"  저장 훅 실패 ({getattr(hook, '__module__', '')}) — CSV 는 정상: {e}")


# ==========================================================
# 실행
# ==========================================================
def hid_zone(st, z):
    """구역(ZONE_ID) → HID_ZONE 값 (정해진 칸, 비어 있으면 0)"""
    if HID_ZONE_USE == "ZONE_ID2":
        return st.info.get(z, ("", "", "", 0))[2] or 0
    return z or 0


def resolve_hid_zone(states):
    """HID_ZONE 에 ZONE_ID · ZONE_ID2 중 무엇을 쓸지 — 문서 번호(참조표)와 더 맞는 칸."""
    global HID_ZONE_USE
    if HID_ZONE_COL.upper() in ("ZONE_ID", "ZONE_ID2"):
        HID_ZONE_USE = HID_ZONE_COL.upper()
        log.info(f"  HID_ZONE = {HID_ZONE_USE} (설정)")
        return
    st = states.get("M16HUB")
    if not REF_FILE.exists() or not st or not st.ok:
        HID_ZONE_USE = "ZONE_ID"
        log.info(f"  HID_ZONE = ZONE_ID (기본 — 참조표 {REF_FILE.name} 없음)")
        return
    ref = {}
    with open(REF_FILE, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            a, h = (r.get("ADDRESS") or "").strip(), (r.get("HID_ID") or "").strip()
            if a and h:
                ref[a] = h
    both = [a for a in ref if a in st.zone_of]
    n1 = sum(1 for a in both if str(st.zone_of[a]) == ref[a])
    n2 = sum(1 for a in both if str(st.info.get(st.zone_of[a], ("", "", "", 0))[2]) == ref[a])
    HID_ZONE_USE = "ZONE_ID2" if n2 > n1 else "ZONE_ID"
    tot = max(len(both), 1)
    log.info(f"  HID_ZONE = {HID_ZONE_USE} — 문서 HID 구간과 일치: ZONE_ID {n1/tot:.0%} · ZONE_ID2 {n2/tot:.0%} "
             f"(M16HUB 주소 {len(both)}개)")


def load_states(fabs=None):
    """fabs 의 FAB 만 맵을 읽는다 (없으면 RUN_FABS). HID_ZONE 번호 칸은 M16HUB 로 정하므로 M16HUB 는 늘 읽는다."""
    want = [f for f in FABS if f in (fabs or RUN_FABS)]
    states, hub = OrderedDict(), None
    for fab in want:
        st = FabState(fab, FABS[fab])
        log.info(f"  {fab:6} {st.map_msg}")
        states[fab] = st
    if "M16HUB" not in states:
        hub = FabState("M16HUB", FABS["M16HUB"])
    resolve_hid_zone({**states, **({"M16HUB": hub} if hub else {})})
    return states


def process(st, frm, to, timeout=None, write=True, verbose=False, pmap=False):
    """frm~to 를 받아 쌓고, 끝난 분마다 1분 한 줄씩 저장. 알람 난 50초 구간 수를 돌려준다.
       pmap = True 면 ALARM 이 MAP_MIN_LEVEL(경계) 이상인 1분마다 문제맵(HTML)도 만든다."""
    new = st.ingest(st.fetch(frm, to, timeout), frm, to)
    n_alarm = 0
    for b in new:                                    # 로그 — 알람 난 50초 구간
        level, why = st.alarm(b)
        if level:
            n_alarm += 1
            top = " · ".join(f"HID_ZONE {hid_zone(st, z)} {st.info.get(z, ('', '', '', 0))[0]}({m}/{j})"
                             for z, m, j, h, _ in st.top_zones(b))
            log.info(f"  ▲ {st.fab} {b:%H:%M:%S} {level} — {why} → {top or '위치 없음'}")
        elif verbose:
            v = st.buckets[b]
            log.info(f"    {st.fab} {b:%H:%M:%S} 정상 — 미보고 {v['miss']} JAM {v['jam']}")
    rows = st.minute_rows()
    if write:
        save(st.fab, rows)
    if pmap:
        for row in rows:
            if MAP_LEVEL.get(row[7], 0) >= MAP_LEVEL[MAP_MIN_LEVEL]:
                problem_map(st, row)
    return n_alarm, rows


def run_cycle(states, now=None, write=True, verbose=False):
    import requests
    now = now or datetime.now()
    end = bucket_floor(now - timedelta(seconds=LAG_SEC))
    down = set()
    status = OrderedDict()                          # FAB → 이번 구간 상태 (한 줄 요약용)
    for fab, st in states.items():
        if not st.ok:
            status[fab] = "맵 없음"
            continue
        if write:
            ensure_file(fab, f"{now:%Y%m%d}")
        if st.cfg["server"] in down:
            status[fab] = "연결 안 됨"
            continue
        start = st.done_to or bucket_floor(now - timedelta(minutes=FIRST_MIN))
        start = max(start, bucket_floor(now - timedelta(minutes=FLEET_AUTO_MIN)))
        if start >= end:
            continue
        try:
            n_alarm, _ = process(st, start, end, write=write, verbose=verbose, pmap=write and MAP_REALTIME)
            last = next((v for k, v in reversed(st.buckets.items()) if not v.get("gap")), None)
            if last is None:
                status[fab] = "데이터 없음"
            else:
                level, _ = st.alarm(max(k for k, v in st.buckets.items() if not v.get("gap")))
                status[fab] = (f"{level or '정상'}(보고 {last['n']} 미보고 {last['miss']} "
                               f"JAM {last['jam']})")
        except requests.exceptions.ConnectionError:
            down.add(st.cfg["server"])
            s = SERVERS[st.cfg["server"]]
            status[fab] = "연결 안 됨"
            log.warning(f"  로그프레소 {s['host']}:{s['port']} 연결 안 됨 — 다음 구간에 이어 받는다")
        except Exception as e:
            status[fab] = "조회 실패"
            log.warning(f"  {fab} 조회 실패 — 다음 구간에 이어 받는다: {e}")
    if status:
        log.info(f"  [{end - timedelta(seconds=BUCKET_SEC):%H:%M:%S}] " +
                 " · ".join(f"{f} {s}" for f, s in status.items()))


def judge_range(st, frm, to):
    """한 FAB 의 frm~to 를 받아 1분 줄로 판정. (rows, 실패 조각 수)"""
    st.minute_done = frm.replace(second=0) - timedelta(minutes=1)
    cur = bucket_floor(frm - timedelta(minutes=WINDOW_MIN))   # 앞 10분부터 받아야 첫 판정이 맞다
    first_save = bucket_floor(frm)
    rows, fails = [], 0
    while cur < to:
        nxt = min(bucket_floor(cur + timedelta(minutes=RANGE_CHUNK_MIN)), bucket_floor(to) + timedelta(seconds=BUCKET_SEC))
        if nxt <= cur:
            nxt = cur + timedelta(seconds=BUCKET_SEC)
        try:
            if nxt <= first_save:                    # 앞 10분은 쌓기만
                st.ingest(st.fetch(cur, nxt, RANGE_TIMEOUT), cur, nxt)
            else:
                if cur < first_save:
                    st.ingest(st.fetch(cur, first_save, RANGE_TIMEOUT), cur, first_save)
                    cur = first_save
                rows += process(st, cur, nxt, RANGE_TIMEOUT, write=False, pmap=MAP_PAST)[1]
        except Exception as e:
            fails += 1
            log.warning(f"  {st.fab} {cur:%m/%d %H:%M}~{nxt:%H:%M} 실패 — 건너뜀: {e}")
        cur = nxt
    lo, hi = f"{frm:%Y-%m-%d %H:%M}", f"{to:%Y-%m-%d %H:%M}"
    rows = [r for r in rows if lo <= r[0] + " " + r[1] < hi]
    return rows, fails


def _save_past(fab, rows, name):
    d = PAST_DIR / fab
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"HID_BOTTLENECK_{fab}_{name}.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:        # 다시 돌리면 덮어씀
        w = csv.writer(f)
        w.writerow(header(fab))
        w.writerows(csv_row(r) for r in rows)
    return p


def _summary_row(fab, label, rows, fails):
    cnt = Counter(r[7] for r in rows)
    first = next((r for r in rows if r[7] != "정상"), None)
    return [fab, label, "실패 " + str(fails) if fails else "정상", len(rows),
            cnt.get("정상", 0), cnt.get("경계", 0), cnt.get("위험", 0), cnt.get("초위험", 0),
            f"{first[1]} {first[7]}" if first else "", first[2] if first else ""]


def _write_summary(summary, name):
    if not summary:
        return
    PAST_DIR.mkdir(parents=True, exist_ok=True)
    sp = PAST_DIR / f"SUMMARY_{name}.csv"
    with open(sp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FAB", "날짜/구간", "조회", "분", "정상", "경계", "위험", "초위험", "첫 알람", "첫 알람 HID_ZONE"])
        w.writerows(summary)
    log.info(f"  요약 → {sp}")


def run_days(states, d_start, d_end, fabs=None):
    """
    FAB · 날짜 ~ 날짜 — 하루씩 판정해 FAB 별 · 날짜별 CSV 로 저장
      HID_BOTTLENECK/PAST/{FAB}/HID_BOTTLENECK_{FAB}_YYYYMMDD.csv    (하루 1440줄, 다시 돌리면 덮어씀)
      HID_BOTTLENECK/PAST/SUMMARY_{FAB|ALL}_{시작}_{끝}.csv       (FAB · 날짜별 정상 · 경계 · 위험 · 초위험 분 수)
    """
    now_min = (datetime.now() - timedelta(seconds=LAG_SEC + BUCKET_SEC)).replace(second=0, microsecond=0)
    summary = []
    for fab, st in states.items():
        if fabs and fab not in fabs:
            continue
        if not st.ok:
            log.info(f"  {fab}: 맵 없음 — 건너뜀")
            summary.append([fab, "", "맵 없음", 0, 0, 0, 0, 0, "", ""])
            continue
        d = d_start
        while d <= d_end:
            frm, to = d, min(d + timedelta(days=1), now_min)
            if frm >= to:
                break
            rows, fails = judge_range(st, frm, to)
            p = _save_past(fab, rows, f"{d:%Y%m%d}")
            sr = _summary_row(fab, f"{d:%Y-%m-%d}", rows, fails)
            summary.append(sr)
            log.info(f"  {fab} {d:%Y-%m-%d}: {len(rows)}분 → {p.name}  "
                     f"(정상 {sr[4]} · 경계 {sr[5]} · 위험 {sr[6]} · 초위험 {sr[7]}"
                     f"{' · 첫 알람 ' + sr[8] + ' HID_ZONE ' + str(sr[9]) if sr[8] else ''})")
            d += timedelta(days=1)
    who = fabs[0] if fabs and len(fabs) == 1 else "ALL" if not fabs else "_".join(fabs)
    _write_summary(summary, f"{who}_{d_start:%Y%m%d}_{d_end:%Y%m%d}")


def run_range(states, frm, to, fabs=None):
    """분 단위 구간 (--range) — HID_BOTTLENECK/PAST/{FAB}/HID_BOTTLENECK_{FAB}_{시작}_{끝}.csv"""
    tag = f"{frm:%Y%m%d%H%M}_{to:%Y%m%d%H%M}"
    summary = []
    for fab, st in states.items():
        if fabs and fab not in fabs:
            continue
        if not st.ok:
            log.info(f"  {fab}: 맵 없음 — 건너뜀")
            summary.append([fab, tag, "맵 없음", 0, 0, 0, 0, 0, "", ""])
            continue
        rows, fails = judge_range(st, frm, to)
        p = _save_past(fab, rows, tag)
        sr = _summary_row(fab, tag, rows, fails)
        summary.append(sr)
        log.info(f"  {fab}: {len(rows)}분 저장 → {p}  (정상 {sr[4]} · 경계 {sr[5]} · 위험 {sr[6]} · 초위험 {sr[7]})")
    _write_summary(summary, tag)


def export_maps(states):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fab, st in states.items():
        if not st.ok:
            continue
        p = OUT_DIR / f"ZONE_TABLE_{fab}.csv"
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ADDRESS", "HID_ZONE", "ZONE_ID", "ZONE_ID2", "Full_Name", "HID_section"])
            for a in sorted(st.zone_of, key=lambda x: (len(x), x)):
                z = st.zone_of[a]
                name, bay, z2, _ = st.info.get(z, ("", "", "", 0))
                w.writerow([a, hid_zone(st, z), z, z2 or 0, name, bay])
        log.info(f"  {p}")



# ==========================================================
# ⑤ 문제맵 — 경계 이상인 1분 → 월드모델파생 맵 위 HTML 지도 (예전 HID_PROBLEM_MAP.py 를 합침)
#    1위 구역 ALARM 색 · 2·3위 파랑 · OHT 차량(월드모델파생 삼각형) · 2D / 유사 3D / 아이소메트리 · 히트맵
# ==========================================================
MAP_LEVEL = {"정상": 0, "경계": 1, "위험": 2, "초위험": 3}
LEVEL = {"경계": 1, "위험": 2, "초위험": 3}
ALARM_EN = {"정상": "NORMAL", "경계": "WARNING", "위험": "DANGER", "초위험": "CRITICAL"}   # 파일 이름은 영문
RANK = {None: 0, "경계": 1, "위험": 2, "초위험": 3}
STEP = timedelta(seconds=BUCKET_SEC)


def say(msg):
    log.info(msg)

def zone_number(st, z, use):
    if use == "ZONE_ID2":
        return str(st.info.get(z, ("", "", "", 0))[2] or 0)
    return str(z or 0)



def geometry(m, use):
    st, L = m["st"], m["layout"]
    nodes = L.get("nodes", {})
    num = {z: zone_number(st, z, use) for z in st.info}
    ids, xs, ys, index = [], [], [], {}

    def idx(a):
        a = str(a)
        if a not in index and a in nodes:
            index[a] = len(ids)
            ids.append(a)
            xs.append(round(float(nodes[a][0]), 1))
            ys.append(round(float(nodes[a][1]), 1))
        return index.get(a)

    zlist = sorted({n for n in num.values() if n != "0"}, key=lambda s: (len(s), s))
    zi = {n: i for i, n in enumerate(zlist)}
    edges = []
    for k in L.get("edges", {}):
        a, _, b = str(k).partition(",")
        ia, ib = idx(a), idx(b)
        if ia is None or ib is None:
            continue
        za, zb = st.zone_of.get(a), st.zone_of.get(b)
        z = num.get(za) if za is not None and za == zb else None
        edges += [ia, ib, zi.get(z, -1) if z else -1]
    # 구역 정보 · 가운데 점 (라벨 · 클릭)
    acc = {n: [0.0, 0.0, 0, 1e18, 1e18, -1e18, -1e18] for n in zlist}
    for a, z in st.zone_of.items():
        n = num.get(z)
        if n in acc and str(a) in nodes:
            x, y = float(nodes[str(a)][0]), float(nodes[str(a)][1])
            c = acc[n]
            c[0] += x
            c[1] += y
            c[2] += 1
            c[3], c[4], c[5], c[6] = min(c[3], x), min(c[4], y), max(c[5], x), max(c[6], y)
    meta = {}
    for z, (name, bay, _, vmax) in st.info.items():
        n = num[z]
        if n != "0" and n not in meta:
            meta[n] = (name, bay, vmax)
    zones = []
    for n in zlist:
        sx, sy, c, x0, y0, x1, y1 = acc[n]
        name, bay, vmax = meta.get(n, ("", "", 0))
        zones.append({"id": n, "name": name, "bay": bay, "vmax": vmax, "n": c,
                      "cx": round(sx / c, 1) if c else None, "cy": round(sy / c, 1) if c else None,
                      "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)] if c else None})
    lanes = []                                       # [from, to, 구역, OUT?] — 월드모델파생 2D 의 HID 존 진입(실선) · 진출(점선)
    master, _ = _find_map_files(*st.cfg["map"])
    try:
        try:
            text = Path(master).read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = Path(master).read_text(encoding="cp949")
        rd = csv.DictReader(io.StringIO(text))
        cols = {c.strip().lower().replace(" ", "_"): c for c in (rd.fieldnames or [])}
        for r in rd:
            z = (r.get(cols.get("zone_id", ""), "") or "").strip()
            if not z or z == "0" or z not in st.info:   # ★HID 아닌 구역(정원 0)의 레인은 안 그린다
                continue
            n = zone_number(st, z, use)
            for key, out in (("in_lanes", 0), ("out_lanes", 1)):
                for a, b in _lanes(r.get(cols.get(key, ""), "") or ""):
                    ia, ib = idx(a), idx(b)
                    if ia is not None and ib is not None:
                        lanes.append([ia, ib, zi.get(n, -1), out])
    except Exception:
        pass
    m["_ids"] = ids
    return {"x": xs, "y": ys, "e": edges, "zones": zones, "lanes": lanes, "ids": ids}




def _vehicles(seen, b):
    """구간 b 의 차량 → [[차량, 상태(0 운행 · 1 JAM · 2 HT · 3 미보고), 주소]]
       미보고 = 그 구간에 보고 없고, 최근 30분(IDLE_MIN) 안에 보였고, 구간 2개 이상에서 보인 차 (HID_VHL_OHT.py 와 같은 조건)"""
    rep = seen.get(b, {})
    last, cnt = {}, defaultdict(int)
    for k in sorted(seen):
        if k > b:
            break
        for v, (_, a) in seen[k].items():
            last[v] = (k, a)
            cnt[v] += 1
    out = [[v, 1 if "7" in sts else 2 if "8" in sts else 0, a] for v, (sts, a) in rep.items()]
    idle = b - timedelta(minutes=IDLE_MIN)
    out += [[v, 3, a] for v, (lb, a) in last.items() if v not in rep and lb >= idle and cnt[v] >= FLEET_MIN_SEEN]
    return out


SPEED_CAP_MM = 400_000          # 50초에 400 m 넘게는 안 찾는다 (그보다 멀면 속도 모름)


def _speed(m, a0, a1):
    """주소 a0 → a1 을 레일 방향대로 간 가장 짧은 거리(mm) ÷ 50초 → m/min. 같은 자리면 0, 못 찾으면 None"""
    if not a0 or not a1:
        return None
    if a0 == a1:
        return 0
    adj = m.setdefault("_adjw", None)
    if adj is None:
        adj = defaultdict(list)
        for k, mm in m["layout"].get("edges", {}).items():
            a, _, b = str(k).partition(",")
            adj[a].append((b, float(mm)))
        m["_adjw"] = adj
    dist, q = {a0: 0.0}, [(0.0, a0)]
    while q:
        d, n = heapq.heappop(q)
        if n == a1:
            return round(d / 1000 / BUCKET_SEC * 60)
        if d > dist.get(n, 1e18) or d > SPEED_CAP_MM:
            continue
        for b, w in adj.get(n, ()):
            nd = d + w
            if nd < dist.get(b, 1e18) and nd <= SPEED_CAP_MM:
                dist[b] = nd
                heapq.heappush(q, (nd, b))
    return None


def _best_bucket(st, mm):
    """HID_VHL_OHT.py minute_row 가 고르는 구간 (그 분 안에 끝난 구간 중 가장 높은 단계, 같으면 나중)"""
    best, level = None, None
    for k in st.buckets_of_minute(mm) or [st.bucket_of_minute(mm)]:
        lv, _ = st.alarm(k)
        if best is None or RANK[lv] >= RANK[level]:
            best, level = k, lv
    return best


def _row_dict(row):
    g = lambda i: str(row[i])
    return {"t": f"{row[0]} {row[1]}", "alarm": row[7], "z1": g(2), "z2": g(13), "z3": g(14), "sect": g(8),
            "n": row[3], "miss": row[4], "jam": row[5], "ht": row[6],
            "zstop": row[9], "zvhl": row[10], "vmax": row[11], "occ": row[12]}



DETAIL_COLS = ["ADDRESS", "NEXT_ADDRESS", "STATUS", "STOCK_INFO", "VEHICLE_EXECUTE_CYCLE", "DESTINATION"]


def fetch_detail(st, b):
    """그 50초 구간 차량의 상세 — 월드모델파생 쿼리(logpresso_query.py)와 같은 칸.
       진행 방향(ADDRESS → NEXT_ADDRESS) · 적재(STOCK_INFO) · 사이클(점) · 목적지. 판정에는 안 쓴다 (그림만)."""
    inner = (f"table from={b:%Y%m%d%H%M%S} to={b + STEP:%Y%m%d%H%M%S} {st.cfg['table']}"
             ' | search MSG_ID == "2" | sort _time'
             " | stats " + ", ".join(f"last({c}) as {c}" for c in DETAIL_COLS) + " by VEHICLE")
    remote = SERVERS[st.cfg["server"]].get("remote")
    q = f"remote {remote} [ {inner} ]" if (st.use_remote and remote) else inner
    body = lp_get(st.cfg["server"], q, RANGE_TIMEOUT)
    out = {}
    for x in csv.DictReader(io.StringIO(body.decode("utf-8-sig", "replace"))):
        v = (x.get("VEHICLE") or "").strip()
        if v:
            out[v] = {c: (x.get(c) or "").strip() for c in DETAIL_COLS}
    return out


def _i(v, d=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def place(m, vs, r, b, det=None):
    """차량 주소 → 지도 노드. → (차량 목록, 설명)
       차량 = [ID, 판정상태(0 운행 · 1 JAM · 2 HT · 3 미보고), 노드, 주소, STATUS, 다음노드, 적재, 사이클, 목적지, 속도(m/min · 모르면 null)]"""
    if b is None or not vs:
        return [], f"{r['t']} 구간 차량 보고 없음"
    idx = {a: i for i, a in enumerate(m["_ids"])}
    det = det or {}
    out, nopos = [], 0
    for v, c, a, spd in vs:
        i = idx.get(str(a))
        if i is None:
            nopos += 1
            continue
        d = det.get(v, {}) if c != 3 else {}
        nx = idx.get(d.get("NEXT_ADDRESS", ""), -1)
        out.append([v, c, i, a, _i(d.get("STATUS"), -1), nx, _i(d.get("STOCK_INFO")),
                    _i(d.get("VEHICLE_EXECUTE_CYCLE")), _i(d.get("DESTINATION")), spd])
    cnt = [sum(1 for x in vs if x[1] == c) for c in range(4)]
    return out, (f"{b:%H:%M:%S} 구간 · 운행 {cnt[0] + cnt[2]} · JAM {cnt[1]} · 미보고 {cnt[3]}"
                 + (f" · 맵에 없는 주소 {nopos}대" if nopos else ""))


# ==========================================================
# 판정 근거 — HID_CONFIG.json 기준 (판정에 쓴 그 기준)
# ==========================================================
def reasons(r, P):
    """그 줄 숫자가 어느 조건에 걸렸나 (높은 단계부터)"""
    out = []
    for lv in ("초위험", "위험", "경계"):
        p = P[lv]
        if p["missing"] and p["JAM"] and r["miss"] >= p["missing"] and r["jam"] >= p["JAM"]:
            out.append(f"{lv}: missing {r['miss']} + JAM {r['jam']} 같이 (기준 {p['missing']} + {p['JAM']})")
        if p["missing_ONLY"] and r["miss"] >= p["missing_ONLY"]:
            out.append(f"{lv}: missing {r['miss']} 단독 (기준 {p['missing_ONLY']})")
        # HT_STOP 은 판정에 쓰지만 화면에는 숨김 — 근거 목록에 안 적는다
        if p.get("ZONE_STOP") and r["zstop"] >= p["ZONE_STOP"]:
            out.append(f"{lv}: 구역 멈춘 차 {r['zstop']} (기준 {p['ZONE_STOP']})")
    return out


# ==========================================================
# 지도 표시 설정 — 월드모델파생 ⚙ 설정(DEFAULT_MAP_SETTINGS)과 같은 이름 · 같은 기본값
# ==========================================================
SETTINGS_FILE = HERE / "HID_MAP_SETTINGS.json"
DEFAULT_SETTINGS = OrderedDict([
    ("mapTheme", "hmi"),         # 'hmi' = 현장 HMI 처럼 밝게 | 'dark' = 어두운 맵 (월드모델파생과 같음)
    ("vehicleRadius", 3),        # ★OHT 크기 (월드모델파생 기본 3)
    ("railScale", 1.0),          # 레일 굵기 배수
    ("zoneScale", 1.0),          # 문제 구역(1~3위) 굵기 배수
    ("labelScale", 1.0),         # 글자 크기 배수
    ("colorEmpty", "#22c55e"),   # 공차 (초록)        ┐
    ("colorLoaded", "#22d3ee"),  # 적재 (하늘)        │ 월드모델파생 차량 색
    ("colorObs", "#f59e0b"),     # OBS               │ (STATUS 6 OBS · 7 JAM · 2/8/9 정지,
    ("colorStop", "#9ca3af"),    # 정지              │  나머지는 적재 여부)
    ("colorJam", "#ef4444"),     # JAM               ┘
    ("colorMiss", "#111827"),    # 미보고 ✕ (끊기기 직전 위치)
    ("carryDot", "on"),          # 삼각형 안 점 — 검은 점 들고 감 · 흰 점 가지러 감
    ("dotLoaded", "#000000"),
    ("dotAssign", "#ffffff"),
    ("dotSize", 0.55),
    ("heat", "off"),             # 히트맵 — 처음엔 꺼짐 (🔥 단추로 켠다) (정체 무리 — 대수대로 노랑 → 주황 → 빨강 → 짙은 적, 20대 이상 제일 짙게)
    ("jamMinJam", 1),            # ┐ 정체 판정 '몇 대 이상' — 월드모델파생 ⚙ 설정 기본값 그대로
    ("jamMinObs", 0),            # │  JAM · OBS · 멈춘 차(2·9) · 미보고, 0 = 안 봄
    ("jamMinStop", 0),           # │  한 무리(12 m 안) 안에서 어느 한 종류라도 그 수를 넘으면 정체
    ("jamMinMiss", 3),           # ┘ (HT_STOP 은 숨김 — 히트맵에 안 들어감)
])
SETTINGS_HELP = ("지도 표시 기본값 — 월드모델파생 ⚙ 설정과 같은 이름. 지도 오른쪽 위 ⚙ 에서 바꾸면 그 브라우저에만 저장되고, "
                 "여기를 고치면 앞으로 만드는 지도 전부의 기본값이 됩니다. vehicleRadius = OHT 크기.")


def load_settings():
    cfg = OrderedDict(DEFAULT_SETTINGS)
    try:
        if SETTINGS_FILE.exists():
            got = json.loads(SETTINGS_FILE.read_text(encoding="utf-8-sig"))
            for k, d in DEFAULT_SETTINGS.items():
                v = got.get(k)
                if v is None:
                    continue
                cfg[k] = float(v) if isinstance(d, float) else int(v) if isinstance(d, int) else str(v)
        else:
            SETTINGS_FILE.write_text(json.dumps({"_설명": SETTINGS_HELP, **DEFAULT_SETTINGS}, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    except Exception as ex:
        say(f"  {SETTINGS_FILE.name} 읽기 실패 — 기본값으로: {ex}")
    return cfg


# ==========================================================
# HTML — 판정된 1분 = 지도 하나 (월드모델파생 2D 맵과 같은 모양)
# ==========================================================
def build_html(fab, r, geo, info, why, oht=None, oht_desc="", settings=None):
    data = {"fab": fab, "info": info, "geo": geo, "why": why, "oht": oht, "ohtd": oht_desc,
            "set": settings or DEFAULT_SETTINGS, "r": {**r, "lv": LEVEL.get(r["alarm"], 0)}}
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = f"HID 문제맵 · {fab} · {r['t']} · {r['alarm']}"
    o3d, three = v3d_sources()
    return (HTML.replace("__TITLE__", title).replace("__DATA__", js)
            .replace("__OHT3D__", o3d).replace("__THREE__", three))


V3D_DIRS = [HERE / "oht3d", HERE / "static" / "js" / "oht3d"]   # 월드모델파생 static/js/oht3d 폴더 그대로
_v3d_cache = None


def v3d_sources():
    """월드모델파생의 3D 뷰어(oht3d.js + three.module.min.js) — HTML 안에 넣어 인터넷 · 서버 없이 아이소메트리를 연다."""
    global _v3d_cache
    if _v3d_cache is None:
        _v3d_cache = ("", "")
        for d in V3D_DIRS:
            a, b = d / "oht3d.js", d / "three.module.min.js"
            if a.exists() and b.exists():
                o3d, three = a.read_text(encoding="utf-8"), b.read_text(encoding="utf-8")
                _v3d_cache = (o3d.replace("</script", "<\\/script"), three.replace("</script", "<\\/script"))
                break
        else:
            say(f"  ★3D 뷰어 없음 — {V3D_DIRS[0]} 에 oht3d.js · three.module.min.js 를 두면 아이소메트리가 켜집니다 (2D · 유사 3D 는 됨)")
    return _v3d_cache


HTML = r"""<!doctype html>
<html lang="ko" data-theme="hmi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root,[data-theme="hmi"]{--bg:#eceef2;--panel:#fff;--fg:#111827;--fg2:#4b5563;--line:#d5dae2;--chip:#eef1f5;
--l0:#3f9b5f;--l1:#e0a12e;--l2:#e2533a;--l3:#8f1d4f;--sub:#2f6fb5}
[data-theme="dark"]{--bg:#0a0e17;--panel:#111827;--fg:#e5e7eb;--fg2:#9ca3af;--line:#273244;--chip:#1f2937;
--l0:#5cc283;--l1:#f0b545;--l2:#ff6b52;--l3:#e0508f;--sub:#5aa7ef}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--fg);font:13px/1.5 "Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;display:flex;flex-direction:column}
header{padding:9px 16px;border-bottom:1px solid var(--line);background:var(--panel);display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center}
header h1{font-size:17px;margin:0}header .m{color:var(--fg2);font-size:12px}
.b{display:inline-block;border-radius:5px;padding:1px 10px;color:#fff;font-weight:700;font-size:14px}
.b0{background:var(--l0)}.b1{background:var(--l1);color:#222}.b2{background:var(--l2)}.b3{background:var(--l3)}
main{flex:1;display:flex;min-height:0}
#mapbox{flex:1;position:relative;min-width:0;min-height:320px}
canvas{display:block;width:100%;height:100%;cursor:grab}canvas.drag{cursor:grabbing}
#side{width:360px;max-width:100%;border-left:1px solid var(--line);background:var(--panel);overflow:auto;padding:12px 14px}
h2{font-size:13px;margin:14px 0 6px;color:var(--fg2);font-weight:600}h2:first-child{margin-top:0}
.zone{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin-bottom:8px;cursor:pointer}
.zone .t{font-weight:700;font-size:15px}.zone .s{color:var(--fg2);font-size:12px}
.zone.r1{border-left:6px solid var(--lv)}.zone.r2,.zone.r3{border-left:6px solid var(--sub)}
table{border-collapse:collapse;width:100%;font-size:12.5px}td{padding:4px 6px;border-bottom:1px solid var(--line)}td:first-child{color:var(--fg2)}td:last-child{text-align:right;font-weight:600}
ul{margin:4px 0;padding-left:18px}li{margin:2px 0}
.ov{position:absolute;background:var(--panel);border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 10px #0002}
#abox{left:10px;top:10px;padding:8px 12px}
#abox .lv{display:inline-block;font-size:22px;font-weight:800;border-radius:8px;padding:2px 14px;color:#fff}
#abox .lv.l0{background:var(--l0)}#abox .lv.l1{background:var(--l1);color:#222}#abox .lv.l2{background:var(--l2)}#abox .lv.l3{background:var(--l3)}
#abox .z{font-size:13px;font-weight:700;margin-top:4px}#abox .s{font-size:11.5px;color:var(--fg2)}
#abox .steps{display:flex;gap:3px;margin-top:6px}#abox .steps span{flex:1;font-size:10.5px;text-align:center;border-radius:4px;padding:1px 4px;color:#fff;opacity:.3}
#abox .steps span.on{opacity:1;outline:2px solid var(--fg);outline-offset:1px}
#lg{left:10px;bottom:10px;padding:6px 9px;font-size:11.5px;color:var(--fg2);border-radius:6px;max-width:calc(100% - 20px)}
#lg i{display:inline-block;width:18px;height:5px;border-radius:2px;margin:0 4px 2px 8px;vertical-align:middle}
#lg svg{vertical-align:-2px;margin:0 3px 0 8px}
.tools{position:absolute;right:10px;top:10px;display:flex;flex-wrap:wrap;justify-content:flex-end;gap:6px;z-index:5;max-width:calc(100% - 300px)}
.tools .seg{display:inline-flex}.tools .seg button{border-radius:0;margin-left:-1px}.tools .seg button:first-child{border-radius:6px 0 0 6px}.tools .seg button:last-child{border-radius:0 6px 6px 0}
.tools button.on{background:var(--fg);color:var(--panel);border-color:var(--fg)}
#map-3d{position:absolute;inset:0;display:none;z-index:1}#map-3d .err{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);max-width:520px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
body.m3d #abox,body.m3d #lg{z-index:4}body.m3d #abox{top:auto;bottom:64px}body.m3d .tools{top:auto;bottom:10px;max-width:calc(100% - 20px)}body.m3d #lg{bottom:auto;top:auto;display:none}
body.m3d #set{top:auto;bottom:50px}.tools button{border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;padding:4px 9px;font:inherit;font-size:12px;cursor:pointer}
#set{right:10px;top:46px;z-index:6;width:290px;padding:10px 12px;display:none;font-size:12px;max-height:calc(100% - 60px);overflow:auto}#set.on{display:block}
#set label{display:grid;grid-template-columns:96px 1fr 38px;gap:6px;align-items:center;margin:5px 0}#set b{font-size:12.5px}
#set input[type=range]{width:100%}#set .c{display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;margin-top:6px}
#set .c label{grid-template-columns:1fr 34px;margin:2px 0}#set input[type=color]{width:34px;height:22px;padding:0;border:1px solid var(--line);background:none}
#set select{font:inherit;background:var(--chip);color:var(--fg);border:1px solid var(--line);border-radius:4px}
#set .btns{display:flex;gap:6px;margin-top:8px}#set .btns button{flex:1;border:1px solid var(--line);background:var(--chip);color:var(--fg);border-radius:6px;padding:4px;font:inherit;cursor:pointer}
#tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 7px;font-size:12px;display:none;box-shadow:0 2px 8px #0003}
.warn{color:var(--l2);font-size:12px}.note{color:var(--fg2);font-size:11.5px}
@media (max-width:820px){main{flex-direction:column}#side{width:100%;border-left:0;border-top:1px solid var(--line);flex:none}#mapbox{height:62vh;flex:none}}
</style></head><body>
<header><h1 id="h1"></h1><span id="hb"></span><span class="m" id="hm"></span></header>
<main>
 <div id="mapbox"><canvas id="cv"></canvas>
  <div class="tools"><span class="seg"><button data-m="2d" class="on">▭ 2D</button><button data-m="iso">◈ 유사 3D</button><button data-m="3d">⬢ 아이소메트리</button></span>
   <button id="bt-heat">🔥 히트맵</button><button id="bt-z">◎ 문제 구역</button><button id="bt-all">⤢ 전체</button><button id="bt-set">⚙ 설정</button></div>
  <div id="map-3d"></div>
  <div class="ov" id="abox"></div><div class="ov" id="lg"></div><div class="ov" id="set"></div><div id="tip"></div></div>
 <aside id="side"></aside>
</main>
<script>
const D=__DATA__, R=D.r, G=D.geo, Z=G.zones, ZI={}; Z.forEach((z,i)=>ZI[z.id]=i);
const LV=['정상','경계','위험','초위험'];
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const TOP=[[R.z1,1],[R.z2,2],[R.z3,3]].filter(([z])=>z&&z!=='0');
const ZOF={};for(let i=0;i<G.e.length;i+=3){if(G.e[i+2]>=0){ZOF[G.e[i]]=G.e[i+2];ZOF[G.e[i+1]]=G.e[i+2];}}   // 노드 → HID 구역
// ---------- 설정 (월드모델파생 ⚙ 설정과 같은 이름) — 파일 기본값 + 이 브라우저에서 바꾼 값 ----------
const SKEY='hid_problem_map_settings_v1';
let S={...D.set};try{const raw=localStorage.getItem(SKEY);if(raw)S={...D.set,...JSON.parse(raw)};}catch(e){}
S.heat=D.set.heat;                                   // 히트맵은 열 때마다 기본값(꺼짐)으로 — 켜 둔 채 저장돼 있어도
const save=()=>{try{localStorage.setItem(SKEY,JSON.stringify(S));}catch(e){}};
// 월드모델파생 MAP_THEMES 그대로 (hmi = 현장 HMI 처럼 밝은 바탕 · 검은 레일 · 초록 HID)
const THEMES={hmi:{bg:'#eceef2',rail:'#23272e',railW:1.15,textBg:'rgba(236,238,242,0.88)',hid:'#3ddc5a',hidText:'#062b10',zone:'#059669',vehicleStroke:'#0b0f14',text2:'#374151'},
 dark:{bg:'#0a0e17',rail:'rgba(255,255,255,0.75)',railW:1.0,textBg:'rgba(10,14,23,0.82)',hid:'#16a34a',hidText:'#ecfdf5',zone:'#00ff88',vehicleStroke:'#0a0e17',text2:'#cbd5e1'}};
const P=()=>THEMES[S.mapTheme]||THEMES.hmi;
const LVC=l=>css('--l'+(l||0));
$('h1').textContent=`HID 문제맵 · ${D.fab} · ${R.t}`;
$('hb').innerHTML=`<span class="b b${R.lv}">${R.alarm}</span>`;
$('hm').textContent=`맵 ${D.info.layout} · ${D.info.src}`;
// ---------- 좌표 — 월드모델파생 mapProjPoint 그대로: 유사 3D(등각) u=(x−y)·cos30°, v=(x+y)·sin30° ----------
const ISO_C=Math.cos(Math.PI/6),ISO_S=Math.sin(Math.PI/6);let mode='2d';   // '2d' | 'iso'(유사 3D) | '3d'(아이소메트리)
const proj=(x,y)=>mode==='iso'?[(x-y)*ISO_C,(x+y)*ISO_S]:[x,y];
let U=[],V=[],minx=0,miny=0,maxx=1,maxy=1;
function setProj(){U=new Array(G.x.length);V=new Array(G.x.length);minx=miny=Infinity;maxx=maxy=-Infinity;
 for(let i=0;i<G.x.length;i++){const [u,v]=proj(G.x[i],G.y[i]);U[i]=u;V[i]=v;if(u<minx)minx=u;if(u>maxx)maxx=u;if(v<miny)miny=v;if(v>maxy)maxy=v;}}
const cv=$('cv'),ctx=cv.getContext('2d');let W=0,H=0,zoom=1,px=0,py=0,base=1;
function fit(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1;W=r.width;H=r.height;cv.width=W*d;cv.height=H*d;ctx.setTransform(d,0,0,d,0,0);
 base=Math.min((W-40)/Math.max(1,maxx-minx),(H-40)/Math.max(1,maxy-miny));}
const S2=(u,v)=>{const s=base*zoom;return[20+(u-minx)*s+px,20+(v-miny)*s+py];};     // 투영 좌표 → 화면
const I2=(sx,sy)=>{const s=base*zoom;return[(sx-20-px)/s+minx,(sy-20-py)/s+miny];};
const SN=i=>S2(U[i],V[i]);                                                          // 노드 → 화면
const SP=(x,y)=>{const q=proj(x,y);return S2(q[0],q[1]);};                           // 도면 좌표 → 화면
function view(x0,y0,x1,y1){const w=Math.max(1,x1-x0),h=Math.max(1,y1-y0);zoom=Math.max(.5,Math.min(80,Math.min((W-80)/w,(H-80)/h)/base));
 const s=base*zoom;px=W/2-20-((x0+x1)/2-minx)*s;py=H/2-20-((y0+y1)/2-miny)*s;draw();}
const home=()=>view(minx,miny,maxx,maxy);
function pbox(b){const cs=[[b[0],b[1]],[b[2],b[1]],[b[0],b[3]],[b[2],b[3]]].map(c=>proj(c[0],c[1]));
 return[Math.min(...cs.map(c=>c[0])),Math.min(...cs.map(c=>c[1])),Math.max(...cs.map(c=>c[0])),Math.max(...cs.map(c=>c[1]))];}
function focusTop(){if(mode==='3d')return v3dFocus();const bs=TOP.map(([z])=>Z[ZI[z]]).filter(z=>z&&z.box).map(z=>pbox(z.box));if(!bs.length)return home();
 const x0=Math.min(...bs.map(b=>b[0])),y0=Math.min(...bs.map(b=>b[1])),x1=Math.max(...bs.map(b=>b[2])),y1=Math.max(...bs.map(b=>b[3]));
 const pad=Math.max(x1-x0,y1-y0)*.35+50;view(x0-pad,y0-pad,x1+pad,y1+pad);}
function focusZone(id){if(mode==='3d')return v3dFocus(id);const z=Z[ZI[id]];if(!z||!z.box)return;const b=pbox(z.box),pad=Math.max(b[2]-b[0],b[3]-b[1])*.6+50;view(b[0]-pad,b[1]-pad,b[2]+pad,b[3]+pad);}
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
// ---------- 차량 (월드모델파생 drawVehicleShape · drawCarryDot 그대로) ----------
const vehR=()=>Math.max(2.5,Math.min(10,(+S.vehicleRadius||3)*(1.2+0.18*Math.max(0,zoom))));
function vcolor(o){ if(o[1]===3)return S.colorMiss;
 let st=o[4];if(st<0)st=o[1]===1?7:o[1]===2?2:1;
 if(st===6)return S.colorObs;if(st===7||o[1]===1)return S.colorJam;if(st===2||st===8||st===9||o[1]===2)return S.colorStop;return o[6]?S.colorLoaded:S.colorEmpty;}
function vkind(o){if(o[7]===4)return'loaded';if(o[7]===2)return'assign';if(o[6])return'loaded';if(o[8]>0)return'assign';return'';}
function tri(sx,sy,rr,ang,color,stroke){ctx.save();ctx.translate(sx,sy);ctx.rotate(ang);const tr=rr*1.56;ctx.beginPath();ctx.moveTo(tr,0);ctx.lineTo(-tr*.7,tr*.78);ctx.lineTo(-tr*.7,-tr*.78);ctx.closePath();
 ctx.fillStyle=color;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=Math.max(.6,Math.min(1.2,rr*.18));ctx.stroke();ctx.restore();}
function dot(sx,sy,rr,kind,ang){if(!kind||S.carryDot==='off')return;const c=kind==='loaded'?S.dotLoaded:S.dotAssign;const dr=Math.max(.9,rr*(+S.dotSize||.55));
 ctx.save();ctx.translate(sx,sy);ctx.rotate(ang||0);ctx.beginPath();ctx.arc(-.133*rr*1.56,0,dr,0,6.283);ctx.fillStyle=c;ctx.fill();ctx.restore();}
// ---------- 히트맵 — 월드모델파생 JAM_RAMP · jamClusters · drawJamBlobs 그대로 ----------
const JAM_R=1200,JAM_HOT_N=20;                       // 12 m 안을 한 무리로 · 20대 이상이 제일 짙다
const JAM_RAMP=[[0,250,204,21,.30],[.45,249,115,22,.46],[.75,239,68,68,.62],[1,153,27,27,.78]];
function jamHeat(cnt){const n=Math.max(1,+cnt||1),t=Math.min(1,Math.max(0,(n-1)/(JAM_HOT_N-1)));let i=1;while(i<JAM_RAMP.length-1&&JAM_RAMP[i][0]<t)i++;
 const p=JAM_RAMP[i-1],q=JAM_RAMP[i],f=(t-p[0])/((q[0]-p[0])||1e-9),v=j=>p[j]+(q[j]-p[j])*f;return{r:Math.round(v(1)),g:Math.round(v(2)),b:Math.round(v(3)),a:v(4),t};}
const JAM_KIND=[['jamMinJam','JAM'],['jamMinObs','OBS'],['jamMinStop','멈춘 차'],['jamMinMiss','미보고']];
// ★HT_STOP 은 숨김 — 히트맵 · 3D 정체 판정에 넣지 않는다 (3D 다섯째 칸 = 0)
const jamMins=()=>[...JAM_KIND.map(([k])=>{const n=parseInt(S[k],10);return(isNaN(n)||n<0)?0:n;}),0];
function vstate(o){let st=o[4];if(st<0)st=o[1]===1?7:1;return st;}
function jamKindOf(o,mins){const live=o[1]!==3,st=vstate(o);
 const ht=o[1]===2||st===8;                         // HT_STOP 차는 히트맵에서 뺀다 (숨김)
 const has=[!ht&&live&&(st===7||o[1]===1),!ht&&live&&st===6,!ht&&live&&(st===2||st===9),o[1]===3];
 for(let i=0;i<4;i++)if(mins[i]>0&&has[i])return i;return-1;}
let _cl=null;
function jamClusters(){if(_cl)return _cl;const mins=jamMins();_cl=[];if(!D.oht||!mins.some(n=>n>0))return _cl;
 const js=[];for(const o of D.oht){const k=jamKindOf(o,mins);if(k>=0)js.push({x:G.x[o[2]],y:G.y[o[2]],k,z:ZOF[o[2]]});}
 const used=js.map(()=>false),counted=js.map(()=>false);
 for(let guard=0;guard<js.length;guard++){let bi=-1,bc=0,bx=0,by=0;
  for(let i=0;i<js.length;i++){if(used[i])continue;let c=0,sx=0,sy=0;for(let k=0;k<js.length;k++){if(used[k])continue;if(Math.hypot(js[i].x-js[k].x,js[i].y-js[k].y)<=JAM_R){c++;sx+=js[k].x;sy+=js[k].y;}}
   if(c>bc){bc=c;bi=i;bx=sx/c;by=sy/c;}}
  if(bi<0)break;for(let k=0;k<js.length;k++)if(!used[k]&&Math.hypot(js[bi].x-js[k].x,js[bi].y-js[k].y)<=JAM_R)used[k]=true;
  const cnt=[0,0,0,0],zc={};for(let k=0;k<js.length;k++)if(used[k]&&!counted[k]&&Math.hypot(js[bi].x-js[k].x,js[bi].y-js[k].y)<=JAM_R){cnt[js[k].k]++;counted[k]=true;if(js[k].z!=null)zc[js[k].z]=(zc[js[k].z]||0)+1;}
  if(!cnt.some((c,i)=>mins[i]>0&&c>=mins[i]))continue;
  let zb=null,zn=0;for(const z in zc)if(zc[z]>zn){zn=zc[z];zb=+z;}       // 무리에 제일 많이 든 HID 구역
  _cl.push({x:bx,y:by,n:cnt.reduce((a,b)=>a+b,0),kinds:cnt,zone:zb});}
 return _cl;}
function drawJamBlobs(){const cl=jamClusters();if(!cl.length)return;const base=JAM_R*base_sc();ctx.save();
 for(const c of cl){const [sx,sy]=SP(c.x,c.y),R0=Math.max(18,base*(.9+.22*Math.log2(1+c.n)));if(sx<-R0||sy<-R0||sx>W+R0||sy>H+R0)continue;
  const hc=jamHeat(c.n),rgb=`${hc.r},${hc.g},${hc.b}`,g=ctx.createRadialGradient(sx,sy,0,sx,sy,R0);
  g.addColorStop(0,`rgba(${rgb},${hc.a.toFixed(3)})`);g.addColorStop(.35,`rgba(${rgb},${(hc.a*.52).toFixed(3)})`);g.addColorStop(.7,`rgba(${rgb},${(hc.a*.18).toFixed(3)})`);g.addColorStop(1,`rgba(${rgb},0)`);
  ctx.fillStyle=g;ctx.beginPath();ctx.arc(sx,sy,R0,0,6.283);ctx.fill();}
 ctx.restore();}
function drawJamText(){const cl=jamClusters();if(!cl.length)return;const base=JAM_R*base_sc();ctx.save();ctx.textAlign='center';ctx.textBaseline='middle';
 for(const c of cl){const [sx,sy]=SP(c.x,c.y),R0=Math.max(18,base*(.9+.22*Math.log2(1+c.n)));if(R0<26||sx<-R0||sy<-R0||sx>W+R0||sy>H+R0)continue;
  const fs=Math.round(Math.min(18,Math.max(11,R0*.22))*(+S.labelScale||1));ctx.lineWidth=3;ctx.strokeStyle='rgba(255,255,255,0.9)';ctx.font=`700 ${fs}px sans-serif`;
  const kk=c.kinds.map((n,i)=>n?`${JAM_KIND[i][1]} ${n}`:'').filter(Boolean),num=kk.length>1?kk.join(' · '):`${c.n}대`;
  ctx.strokeText(num,sx,sy);ctx.fillStyle='#b91c1c';ctx.fillText(num,sx,sy);
  const z=c.zone!=null?Z[c.zone]:null,nm=z?(z.name||'HID '+z.id):'구역 밖';ctx.font=`700 ${Math.round(fs*.92)}px sans-serif`;
  ctx.strokeText(nm,sx,sy-fs-3);ctx.fillStyle=z?'#7f1d1d':'#6b7280';ctx.fillText(nm,sx,sy-fs-3);}
 ctx.restore();}
const base_sc=()=>base*zoom;
// ---------- 그리기 ----------
function lines(test,col,w,dash){ctx.strokeStyle=col;ctx.lineWidth=w;ctx.setLineDash(dash||[]);ctx.beginPath();
 for(let i=0;i<G.e.length;i+=3){if(!test(G.e[i+2]))continue;const a=SN(G.e[i]),b=SN(G.e[i+1]);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);}ctx.stroke();ctx.setLineDash([]);}
function draw(){const p=P(),sc=base*zoom;ctx.setTransform(devicePixelRatio||1,0,0,devicePixelRatio||1,0,0);
 ctx.fillStyle=p.bg;ctx.fillRect(0,0,W,H);ctx.lineCap='round';ctx.lineJoin='round';
 if(mode==='iso'){const g=[[GX0,GY0],[GX1,GY0],[GX1,GY1],[GX0,GY1]].map(c=>SP(c[0],c[1]));ctx.fillStyle=S.mapTheme==='dark'?'rgba(255,255,255,0.05)':'#dfe3ea';
  ctx.strokeStyle='#b8c0cc';ctx.lineWidth=1;ctx.beginPath();g.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.closePath();ctx.fill();ctx.stroke();}
 const hot={};TOP.forEach(([z,r])=>{if(ZI[z]!=null&&!(ZI[z] in hot))hot[ZI[z]]=r;});
 // ① HID 존 진입(실선)·진출(점선) — 레일보다 먼저, 초록 형광펜 (월드모델파생 2D 그대로)
 if(sc>=0.28&&G.lanes){const zw=clamp(2+sc*2.2,2,9),dash=clamp(2+sc*2.5,3,10);ctx.strokeStyle=p.zone;ctx.lineWidth=zw;ctx.globalAlpha=.30;
  for(const out of [0,1]){ctx.setLineDash(out?[dash,dash*.7]:[]);ctx.beginPath();for(const [a,b,,o] of G.lanes){if(o!==out)continue;const A=SN(a),B=SN(b);ctx.moveTo(A[0],A[1]);ctx.lineTo(B[0],B[1]);}ctx.stroke();}
  ctx.setLineDash([]);ctx.globalAlpha=1;}
 // ② 문제 구역 — 레일 밑에 굵은 형광펜 (2·3위 파랑, 1위 ALARM 색)
 const zs=+S.zoneScale||1,rw=clamp(.9+sc*.8,.8,2.4)*(p.railW||1)*(+S.railScale||1);
 ctx.globalAlpha=.7;[3,2].forEach(r=>lines(z=>hot[z]===r,css('--sub'),Math.max(6,rw*5)*zs));
 ctx.globalAlpha=.88;lines(z=>hot[z]===1,LVC(R.lv),Math.max(9,rw*7)*zs);ctx.globalAlpha=1;
 // ③ 레일 (검은 선, 월드모델파생 굵기 규칙)
 lines(()=>true,p.rail,rw);
 // ③' 히트맵 — 차량 **밑에** (위에 덮으면 삼각형이 묻힌다, 월드모델파생과 같음)
 if(S.heat!=='off')drawJamBlobs();
 // ④ OHT — 미보고 ✕ 먼저, 운행 · 멈춘 차는 위에
 if(D.oht){const rr=vehR();
  for(const o of D.oht){if(o[1]!==3)continue;const [x,y]=SN(o[2]);if(x<-9||y<-9||x>W+9||y>H+9)continue;const q=rr*1.05;
   ctx.strokeStyle=p.bg;ctx.lineWidth=Math.max(2.5,rr*.75);ctx.beginPath();ctx.moveTo(x-q,y-q);ctx.lineTo(x+q,y+q);ctx.moveTo(x+q,y-q);ctx.lineTo(x-q,y+q);ctx.stroke();
   ctx.strokeStyle=S.mapTheme==='dark'&&S.colorMiss==='#111827'?'#f3f4f6':S.colorMiss;ctx.lineWidth=Math.max(1.4,rr*.42);ctx.stroke();}
  for(const pass of [0,1])for(const o of D.oht){if(o[1]===3)continue;const stop=o[1]>0||[6,7,2,8,9].includes(o[4]);if(stop!==!!pass)continue;
   const [x,y]=SN(o[2]);if(x<-9||y<-9||x>W+9||y>H+9)continue;
   let ang=-Math.PI/2;if(o[5]>=0&&o[5]!==o[2]){const B=SN(o[5]);if(B[0]!==x||B[1]!==y)ang=Math.atan2(B[1]-y,B[0]-x);}
   tri(x,y,rr,ang,vcolor(o),p.vehicleStroke);dot(x,y,rr,vkind(o),ang);}}
 if(S.heat!=='off')drawJamText();
 // ⑤ 라벨 — 1·2·3위 먼저 (비켜서라도), 나머지 HID 는 확대하면 초록 상자
 const ls=+S.labelScale||1;ctx.textAlign='center';ctx.textBaseline='middle';const placed=[];
 const L=[...TOP,...(zoom>=2.5?Z.map(z=>[z.id,9]):[])];const done={};
 for(const [id,r] of L){if(done[id])continue;done[id]=1;const z=Z[ZI[id]];if(!z||z.cx==null)continue;let [x,y]=SP(z.cx,z.cy);if(x<-30||y<-30||x>W+30||y>H+30)continue;
  const big=r<=3,t=big?`${r}위 HID ${id}${z.bay?' · '+z.bay:''}${r===1&&R.lv?' · '+R.alarm:''}`:`HID ${id}`;
  ctx.font=big?`700 ${12.5*ls}px system-ui,sans-serif`:`700 ${10*ls}px system-ui,sans-serif`;
  const w=ctx.measureText(t).width+10,h=(big?20:15)*ls;x=Math.max(w/2+2,Math.min(W-w/2-2,x));if(big)y-=16*ls;
  const hit=yy=>placed.some(q=>Math.abs(q[0]-x)<(q[2]+w)/2&&Math.abs(q[1]-yy)<(q[3]+h)/2+1);
  if(hit(y)){if(!big)continue;const alt=[y+h+4,y-h-4,y+2*h+8,y-2*h-8].find(yy=>!hit(yy));if(alt==null)continue;y=alt;}
  placed.push([x,y,w,h]);
  ctx.fillStyle=big?(r===1?LVC(R.lv):css('--sub')):p.hid;ctx.fillRect(x-w/2,y-h/2,w,h);
  ctx.fillStyle=big?((r===1&&R.lv===1)?'#222':'#fff'):p.hidText;ctx.fillText(t,x,y+.5);}
 legend();}
const triSvg=c=>`<svg width="12" height="12" viewBox="-6 -6 12 12"><path d="M0,-5.5 L4.7,4 L-4.7,4 Z" fill="${c}" stroke="#0b0f14" stroke-width=".8"/></svg>`;
function legend(){const oc=D.oht?[0,1,2,3].map(c=>D.oht.filter(x=>x[1]===c).length):null;
 $('lg').innerHTML=`<b>${R.t}</b> 1위 구역:<i style="background:${LVC(1)}"></i>경계<i style="background:${LVC(2)}"></i>위험<i style="background:${LVC(3)}"></i>초위험<i style="background:${css('--sub')}"></i>2·3위<i style="background:${P().zone};opacity:.5"></i>HID 존`+
 (oc?`<br>OHT:${triSvg(S.colorEmpty)}공차${triSvg(S.colorLoaded)}적재${triSvg(S.colorObs)}OBS${triSvg(S.colorStop)}정지${triSvg(S.colorJam)}JAM<b style="margin:0 3px 0 8px">✕</b>미보고`+
  ` — 보고 ${oc[0]+oc[1]+oc[2]} (JAM ${oc[1]}) · 미보고 ${oc[3]}`:`<br>OHT 차량 없음 — ${D.ohtd}`)+
 (S.heat!=='off'?`<br>히트맵 (정체 무리 대수):<i style="width:90px;height:8px;background:linear-gradient(90deg,rgb(250,204,21),rgb(249,115,22),rgb(239,68,68),rgb(153,27,27))"></i>1대 → 20대 이상 · 정체 ${jamClusters().length}곳`:'');}
// ---------- ⚙ 설정 ----------
const RANGES=[['vehicleRadius','OHT 크기',1,8,.5],['railScale','레일 굵기',.4,3,.1],['zoneScale','문제 구역 굵기',.4,3,.1],['labelScale','글자 크기',.6,1.8,.1],['dotSize','점 크기',.2,1,.05]];
const COLORS=[['colorEmpty','공차'],['colorLoaded','적재'],['colorObs','OBS'],['colorStop','정지'],['colorJam','JAM'],['colorMiss','미보고 ✕']];
function setPanel(){let h=`<b>⚙ 지도 설정</b> <span class="note">(월드모델파생 ⚙ 설정과 같은 값)</span>`;
 h+=`<label>테마<select id="s-mapTheme"><option value="hmi">HMI (밝게)</option><option value="dark">다크</option></select><span></span></label>`;
 for(const [k,n,a,b,st] of RANGES)h+=`<label>${n}<input type="range" id="s-${k}" min="${a}" max="${b}" step="${st}" value="${S[k]}"><span id="v-${k}">${S[k]}</span></label>`;
 h+=`<label>삼각형 안 점<select id="s-carryDot"><option value="on">켜기</option><option value="off">끄기</option></select><span></span></label>`;
 h+=`<label>히트맵<select id="s-heat"><option value="on">켜기</option><option value="off">끄기</option></select><span></span></label>`;
 h+=`<div class="note" style="margin-top:4px">정체 판정 — 12 m 안 한 무리에서 몇 대 이상이면 (0 = 안 봄, 월드모델파생 ⚙ 와 같음)</div><div class="c">`;
 for(const [k,n] of JAM_KIND)h+=`<label>${n}<input type="number" id="s-${k}" min="0" max="99" value="${S[k]}" style="width:44px"></label>`;
 h+=`</div><div class="c">`;
 for(const [k,n] of COLORS)h+=`<label>${n}<input type="color" id="s-${k}" value="${S[k]}"></label>`;
 h+=`</div><div class="btns"><button id="s-def">기본값</button><button id="s-close">닫기</button></div><div class="note" style="margin-top:6px">여기서 바꾼 값은 이 브라우저에 저장됩니다. 모든 지도의 기본값은 HID_MAP_SETTINGS.json 에서.</div>`;
 $('set').innerHTML=h;$('s-mapTheme').value=S.mapTheme;$('s-carryDot').value=S.carryDot;$('s-heat').value=S.heat;
 $('set').querySelectorAll('input,select').forEach(el=>el.oninput=el.onchange=()=>{const k=el.id.slice(2);S[k]=el.type==='range'?+el.value:el.value;
  const v=$('v-'+k);if(v)v.textContent=el.value;if(el.type==='number')S[k]=+el.value||0;_cl=null;save();applyTheme();syncHeat();draw();});
 $('s-def').onclick=()=>{S={...D.set};_cl=null;try{localStorage.removeItem(SKEY);}catch(e){}setPanel();applyTheme();syncHeat();draw();};
 $('s-close').onclick=()=>$('set').classList.remove('on');}
let applyTheme=function(){document.documentElement.dataset.theme=S.mapTheme==='dark'?'dark':'hmi';document.documentElement.style.setProperty('--lv',LVC(R.lv));
 const z1=Z[ZI[R.z1]];$('abox').innerHTML=`<span class="lv l${R.lv}">${R.alarm}</span>`+
 (R.z1&&R.z1!=='0'?`<div class="z">1위 HID ${R.z1}${z1&&z1.bay?' · '+z1.bay:''}</div><div class="s">구역 멈춘 차 ${R.zstop} · missing ${R.miss} · JAM ${R.jam}</div>`:'')+
 `<div class="steps">${[1,2,3].map(l=>`<span class="${l===R.lv?'on':''}" style="background:${LVC(l)};${l===1?'color:#222':''}">${LV[l]}</span>`).join('')}</div>`;};
$('bt-set').onclick=()=>$('set').classList.toggle('on');
function syncHeat(){$('bt-heat').classList.toggle('on',S.heat!=='off');if(v3d){v3d.setOptions({heat:S.heat!=='off',jamMin:jamMins()});
  const bt=$('map-3d').querySelector(S.heat!=='off'?'.o3d-btn[data-a=hot]':'.o3d-btn[data-a=all]');if(mode==='3d'&&bt)bt.click();}}
$('bt-heat').onclick=()=>{S.heat=S.heat==='off'?'on':'off';save();const el=$('s-heat');if(el)el.value=S.heat;syncHeat();if(mode!=='3d')draw();};
// ---------- 조작 ----------
let drag=null;
cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,px,py};cv.setPointerCapture(e.pointerId);cv.classList.add('drag');});
cv.addEventListener('pointermove',e=>{if(drag){px=drag.px+e.clientX-drag.x;py=drag.py+e.clientY-drag.y;draw();}else hover(e);});
cv.addEventListener('pointerup',()=>{drag=null;cv.classList.remove('drag');});
cv.addEventListener('wheel',e=>{e.preventDefault();const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,[wx,wy]=I2(mx,my);
 zoom=Math.max(.5,Math.min(80,zoom*(e.deltaY<0?1.2:1/1.2)));const [sx,sy]=S2(wx,wy);px+=mx-sx;py+=my-sy;draw();},{passive:false});
cv.addEventListener('dblclick',home);
function seg(x,y,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],l=dx*dx+dy*dy;let t=l?((x-a[0])*dx+(y-a[1])*dy)/l:0;t=Math.max(0,Math.min(1,t));const qx=a[0]+t*dx-x,qy=a[1]+t*dy-y;return qx*qx+qy*qy;}
const SNAME={1:'운행',2:'정지',6:'OBS',7:'JAM',8:'정지',9:'정지'};      // HT_STOP 은 '정지' 로만 보인다 (숨김)
function hover(e){const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,tp=$('tip');
 const show=h=>{tp.innerHTML=h;tp.style.display='block';tp.style.left=Math.min(mx+12,W-250)+'px';tp.style.top=(my+12)+'px';};
 if(D.oht){let vb=null,vd=Math.max(8,vehR()*1.6)**2;for(const o of D.oht){const [x,y]=SN(o[2]);const d=(x-mx)**2+(y-my)**2;if(d<vd){vd=d;vb=o;}}
  if(vb){const z=ZOF[vb[2]];const st=vb[1]===3?'미보고 (끊기기 직전 위치)':vb[1]===1?'JAM':vb[1]===2?'정지':(SNAME[vb[4]]||'운행')+(vb[6]?' · 적재':' · 공차');
   return show(`<b>OHT ${vb[0]}</b> · ${st}<br>주소 ${vb[3]}${z!=null?` · HID ${Z[z].id}${Z[z].bay?' · '+Z[z].bay:''}`:''}${vb[8]>0?`<br>목적지 ${vb[8]}`:''}${vb[9]!=null&&vb[1]!==3?`<br>속도 ${vb[9]} m/min (50초 평균)`:''}`);}}
 let best=-1,bd=12*12;
 for(let i=0;i<G.e.length;i+=3){const z=G.e[i+2];if(z<0)continue;const d=seg(mx,my,SN(G.e[i]),SN(G.e[i+1]));if(d<bd){bd=d;best=z;}}
 if(best<0){tp.style.display='none';return;}const z=Z[best],rk=TOP.find(([id])=>id===z.id);
 show(`<b>HID ${z.id}</b> ${z.name||''}${z.bay?' · '+z.bay:''}${rk?` — <b>${rk[1]}위</b>`:''}${z.vmax?`<br>정원(Vehicle_Max) ${z.vmax}`:''}`);}
$('bt-all').onclick=()=>{if(mode==='3d'){if(v3d)v3d.viewAll();}else home();};$('bt-z').onclick=focusTop;
// ---------- 오른쪽 ----------
const zname=id=>{const z=Z[ZI[id]];return z?`${z.name||''}${z.bay?' · '+z.bay:''}`:'⚠ 지도에 없는 구역';};
let h=`<h2>문제 구역 (그 1분)</h2>`;
if(!TOP.length)h+=`<div class="note">${R.lv?'구역 정보 없음':'정상 — 문제 구역 없음'}</div>`;
TOP.forEach(([id,r])=>{h+=`<div class="zone r${r}" data-z="${id}"><div class="t">${r}위 · HID ${id}</div><div class="s">${zname(id)}</div>`+
 (r===1?`<div class="s">구역 안 멈춘 차 <b>${R.zstop}</b> · 구역 안 차량 <b>${R.zvhl}</b> / 정원 ${R.vmax} · 점유율 <b>${R.occ}%</b></div>`:'')+`</div>`;});
h+=`<h2>그 1분 숫자</h2><table>
<tr><td>날짜 · 시간</td><td>${R.t}</td></tr><tr><td>ALARM</td><td>${R.alarm}</td></tr>
<tr><td>HID_ZONE · section</td><td>${R.z1} · ${R.sect}</td></tr>
<tr><td>OHT_report (보고 차량)</td><td>${R.n}</td></tr><tr><td>OHT_missing (미보고)</td><td>${R.miss}</td></tr>
<tr><td>OHT_JAM</td><td>${R.jam}</td></tr>
<tr><td>ZONE_STOP (구역 멈춘 차)</td><td>${R.zstop}</td></tr><tr><td>ZONE_VHL / VHL_MAX</td><td>${R.zvhl} / ${R.vmax}</td></tr>
<tr><td>ZONE_OCC (점유율)</td><td>${R.occ}%</td></tr><tr><td>HID_ZONE_2 · 3</td><td>${R.z2} · ${R.z3}</td></tr></table>`;
h+=`<h2>판정 근거 (HID_CONFIG.json 기준)</h2>`+(D.why.length?`<ul>${D.why.map(w=>`<li>${w}</li>`).join('')}</ul>`:`<div class="note">${R.lv?'HID_CONFIG.json 기준 충족':'걸린 조건 없음 — 정상'}</div>`);
if(D.info.diff)h+=`<div class="warn">⚠ ${D.info.diff} — HID_Zone_Master 를 확인하세요.</div>`;
if(D.info.miss.length)h+=`<div class="warn">⚠ 지도에 없는 구역: ${D.info.miss.join(', ')} — HID_Zone_Master 를 확인하세요.</div>`;
h+=`<h2>OHT 차량</h2><div class="note">${D.ohtd}<br>모양 · 색은 월드모델파생 맵과 같습니다 — 삼각형 꼭짓점 = 진행 방향, 안의 검은 점 = 들고 감 · 흰 점 = 가지러 감.</div>`;
h+=`<h2>맵</h2><div class="note">${D.info.layout} · 구역 번호 ${D.info.use}<br>${D.info.msg}<br>휠 확대 · 드래그 이동 · 더블클릭 전체 · 차량 · 구역에 마우스를 올리면 정보 · ⚙ 설정에서 OHT 크기 등</div>`;
$('side').innerHTML=h;
document.querySelectorAll('.zone[data-z]').forEach(el=>el.onclick=()=>focusZone(el.dataset.z));
setPanel();applyTheme();
// 도면 범위 (유사 3D 바닥판)
let GX0=Infinity,GY0=Infinity,GX1=-Infinity,GY1=-Infinity;for(let i=0;i<G.x.length;i++){GX0=Math.min(GX0,G.x[i]);GX1=Math.max(GX1,G.x[i]);GY0=Math.min(GY0,G.y[i]);GY1=Math.max(GY1,G.y[i]);}
// ---------- ⬢ 아이소메트리 — 월드모델파생 oht3d.js (three.js r169) 그대로, 이 파일 안에 들어 있다 ----------
let v3d=null,v3dLoading=null;
const V3D_SCALE=0.01;                                          // 도면 1 단위 = 10 mm (월드모델파생과 같음)
const zname3=z=>z.name||('HID '+z.id);
function layout3D(){const ids=G.ids,nodes=ids.map((id,i)=>({id,x:G.x[i],y:G.y[i]})),seen=new Set(),edges=[];
 for(let i=0;i<G.e.length;i+=3){const id=ids[G.e[i]]+'-'+ids[G.e[i+1]];if(seen.has(id)||G.e[i]===G.e[i+1])continue;seen.add(id);edges.push({id,from:ids[G.e[i]],to:ids[G.e[i+1]]});}
 const ze={};for(const [a,b,zi] of (G.lanes||[])){if(zi<0)continue;const id=ids[a]+'-'+ids[b];if(!seen.has(id)){seen.add(id);edges.push({id,from:ids[a],to:ids[b]});}(ze[zi]=ze[zi]||[]).push(id);}
 const zones=Object.entries(ze).map(([zi,es])=>({id:zname3(Z[zi]),edges:es}));return{nodes,edges,zones,ports:[]};}
function rows3D(){const ids=G.ids;return (D.oht||[]).map(o=>{let st=o[4];if(o[1]===3)st=2;else if(st<0)st=o[1]===1?7:o[1]===2?2:1;
 const s3=st===7||o[1]===1?3:st===6?4:(st===2||st===8||st===9||o[1]===2)?2:(o[6]?1:0);
 const stop=o[1]>0||[2,6,7,8,9].includes(st);
 /* 속도 = 앞 50초 구간 위치에서 지금 위치까지 레일을 따라 간 거리 ÷ 50초 (m/min). 멈춘 차 · 모르면 0.
    3D 히트맵(레일 원활 ↔ 정체)은 월드모델파생처럼 이 속도로 칠한다. */
 return{id:o[0],from:ids[o[2]],to:o[5]>=0?ids[o[5]]:null,ratio:0,x:G.x[o[2]],y:G.y[o[2]],speed_mpm:stop?0:(o[9]||0),state:s3,loaded:!!o[6],miss:o[1]===3,ht:false};   /* HT_STOP 숨김 */});}
const v3dColors=()=>[S.colorEmpty,S.colorLoaded,S.colorStop,S.colorJam,S.colorObs];
function v3dFocus(id){if(!v3d)return;const zid=id||R.z1;const z=Z[ZI[zid]];if(z)try{v3d.focusZone(zname3(z));}catch(e){}}
async function open3D(){const box=$('map-3d');box.style.display='block';cv.style.display='none';
 if(!v3d){if(!v3dLoading)v3dLoading=(async()=>{
   const blob=(id)=>URL.createObjectURL(new Blob([document.getElementById(id).textContent],{type:'text/javascript'}));
   const threeUrl=blob('src-three'),mod=await import(blob('src-oht3d'));
   const inst=await mod.createOHT3D(box,{threeUrl,coordScale:V3D_SCALE,flipY:false,projection:'iso',walls:false,dark:S.mapTheme==='dark',
     colors:{state:v3dColors(),background:P().bg,accent:'#1f6feb'},sizeUI:true,panel:false,heat:S.heat!=='off',jamMin:jamMins()});
   inst.setLayout(layout3D());inst.setVehicles(rows3D(),{ts:R.t.replace(/[-: ]/g,'')+'00'});return inst;})();
  try{v3d=await v3dLoading;}catch(e){v3dLoading=null;box.innerHTML=`<div class="err"><b>아이소메트리를 못 불러왔습니다.</b><br>${String(e&&e.message||e).replace(/[<>&]/g,'')}</div>`;return;}}
 v3d.setActive(true);v3d.resize&&v3d.resize();v3d.setOptions({dark:S.mapTheme==='dark',background:P().bg,stateColors:v3dColors(),heat:S.heat!=='off',jamMin:jamMins()});
 setTimeout(()=>v3dFocus(),300);}                    // 정체 지점은 처음엔 꺼짐 — 🔥 히트맵 단추나 뷰어의 '정체 지점' 으로 켠다
function close3D(){$('map-3d').style.display='none';cv.style.display='';if(v3d)v3d.setActive(false);}
function setMode(m){if(m===mode)return;const was=mode;mode=m;document.querySelectorAll('.seg button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
 document.body.classList.toggle('m3d',m==='3d');
 if(m==='3d'){open3D();return;}if(was==='3d')close3D();setProj();fit();home();}
document.querySelectorAll('.seg button').forEach(b=>b.onclick=()=>setMode(b.dataset.m));
const _apply=applyTheme;applyTheme=function(){_apply();if(v3d)v3d.setOptions({dark:S.mapTheme==='dark',background:P().bg,stateColors:v3dColors()});};
setProj();syncHeat();
addEventListener('resize',()=>{if(mode==='3d')return;fit();home();});fit();home();
</script>
<script type="text/plain" id="src-oht3d">__OHT3D__</script>
<script type="text/plain" id="src-three">__THREE__</script>
</body></html>
"""


# ==========================================================

def _pm(st):
    """FAB 의 문제맵 재료 — 레이아웃 · 지도 그림. 처음 한 번만 만든다."""
    if st.pm is None:
        _, layout = _find_map_files(*st.cfg["map"])
        L = json.loads(Path(layout).read_text(encoding="utf-8"))
        m = {"st": st, "layout": L, "layout_file": Path(layout).name, "msg": st.map_msg}
        m["geo"] = geometry(m, HID_ZONE_USE)
        m["have"] = {z["id"] for z in m["geo"]["zones"] if z["n"]}
        m["zbay"] = {z["id"]: z["bay"] for z in m["geo"]["zones"]}
        st.pm = m
    return st.pm


def problem_map(st, row, out_dir=None):
    """1분 한 줄(row) → 문제맵 HTML. 판정 · 숫자는 그 줄 그대로, 차량은 그 줄과 같은 50초 구간 (메모리에 있는 것)."""
    try:
        m = _pm(st)
        r = _row_dict(row)
        mm = datetime.strptime(r["t"], "%Y-%m-%d %H:%M")
        b = _best_bucket(st, mm)
        vs = _vehicles(st.seen, b) if b in st.seen else []
        prev = st.seen.get(b - STEP, {}) if b is not None else {}
        for x in vs:                                 # 50초 평균 속도 (m/min) — 3D 히트맵 · 차량 정보
            x.append(_speed(m, prev[x[0]][1], x[2]) if (x[1] != 3 and x[0] in prev) else None)
        det = None
        if b is not None and vs:
            try:
                det = fetch_detail(st, b)
            except Exception as ex:
                log.warning(f"    문제맵 {st.fab} {r['t']} — 차량 방향 · 적재 못 받음 (상태만 그림): {str(ex)[:120]}")
        oht, od = place(m, vs, r, b, det)
        r = {k: v for k, v in r.items() if k != "ht"}       # ★HT_STOP 숨김 — 지도 파일에 숫자도 안 넣는다
        geo, have, zbay = m["geo"], m["have"], m["zbay"]
        miss = [z for z in (r["z1"], r["z2"], r["z3"]) if z not in ("", "0") and z not in have]
        bay = zbay.get(r["z1"], "")
        diff = (f"판정 HID_section {r['sect']} ≠ 지도의 {bay} (HID {r['z1']})"
                if r["z1"] in zbay and r["sect"] not in ("", "0") and bay and r["sect"] != bay else "")
        info = {"layout": m["layout_file"], "msg": m["msg"], "use": HID_ZONE_USE, "miss": miss, "diff": diff,
                "src": f"로그프레소 {st.cfg['table']} ({SERVERS[st.cfg['server']]['host']})"}
        day, hm = r["t"][:10].replace("-", ""), r["t"][11:].replace(":", "")
        d = (out_dir or MAP_OUT) / f"{st.fab}_{day}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"PROBLEM_MAP_{st.fab}_{day}_{hm}_{ALARM_EN.get(r['alarm'], 'NORMAL')}.html"
        p.write_text(build_html(st.fab, r, geo, info, reasons(r, policy(st.fab)), oht, od, load_settings()),
                     encoding="utf-8")
        log.info(f"  🗺 문제맵 {st.fab} {r['t']} {r['alarm']} · 1위 HID {r['z1']} → {p.name}")
        return p
    except Exception as ex:                          # 지도가 실패해도 판정 · CSV 는 계속
        log.warning(f"  문제맵 {st.fab} 실패 (판정 · CSV 는 정상): {ex}")
        return None


def map_at(st, t):
    """그 1분만 로그프레소에서 판정해 문제맵 하나 (정상이어도 만든다) — 앞 100분부터 받는다."""
    st.buckets.clear(); st.last.clear(); st.first.clear(); st.nseen.clear(); st.seen.clear()
    st.done_to = None
    st.minute_done = t - timedelta(minutes=1)
    cur, end = bucket_floor(t - timedelta(minutes=FLEET_AUTO_MIN)), bucket_floor(t + timedelta(minutes=1)) + STEP
    made = None
    while cur < end:
        nxt = min(bucket_floor(cur + timedelta(minutes=RANGE_CHUNK_MIN)), end)
        if nxt <= cur:
            nxt = cur + STEP
        _, rows = process(st, cur, nxt, RANGE_TIMEOUT, write=False)
        for row in rows:
            if f"{row[0]} {row[1]}" == f"{t:%Y-%m-%d %H:%M}":
                if not row[3]:                       # 그 1분 보고 차량 0 = 데이터 없음 — 빈 지도는 안 만든다
                    log.warning(f"  {st.fab} {t:%Y-%m-%d %H:%M} 그 시각 차량 보고 없음 (데이터 없음)")
                else:
                    made = problem_map(st, row)
        cur = nxt
    return made


def main():
    ap = argparse.ArgumentParser(description="5 FAB OHT 병목 HID 구간 찾기")
    ap.add_argument("--test", action="store_true", help="구간표 확인 + 최근 10분 한 번 판정 (저장 안 함)")
    ap.add_argument("--map", action="store_true", help="주소 → HID 구역표 CSV 내보내기")
    ap.add_argument("--range", nargs=2, metavar=("YYYYMMDDHHMM", "YYYYMMDDHHMM"),
                    help="과거 구간 다시 판정 — FAB 별로 HID_BOTTLENECK/PAST/ 에 저장")
    ap.add_argument("--fab", nargs="+", metavar="FAB", help=f"--range 할 FAB (기본 전부: {' '.join(FABS)})")
    ap.add_argument("--no-problem-map", action="store_true", help="실시간에서 문제맵(HTML)을 만들지 않음")
    ap.add_argument("--problem-map", action="store_true", help="과거 판정(FAB 날짜 · --range)에서도 경계 이상 분마다 문제맵 만들기")
    ap.add_argument("--map-at", nargs="+", metavar=("FAB", "YYYYMMDDHHMM"),
                    help="그 1분만 로그프레소에서 판정해 문제맵 하나 (예: --map-at M16HUB 202609291348 [202609291355 …])")
    ap.add_argument("args", nargs="*", metavar="FAB 날짜 [날짜]",
                    help="과거 판정: FAB(M14 M14B M16A M16B M16HUB 또는 ALL) 시작날짜 [끝날짜] — YYYYMMDD, 하루씩 CSV")
    a = ap.parse_args()

    day_job = None                                   # (fabs, 시작일, 끝일)
    if a.args:
        toks = list(a.args)
        fabs = None
        if toks and not toks[0].replace("-", "").isdigit():
            f0 = toks.pop(0).upper()
            if f0 not in ("ALL", "전체"):
                fabs = [x.strip() for x in f0.split(",") if x.strip()]
                bad = [x for x in fabs if x not in FABS]
                if bad:
                    sys.exit(f"모르는 FAB {bad} — 가능: {' '.join(FABS)} 또는 ALL")
        if not toks or len(toks) > 2:
            sys.exit("사용법: python HID_VHL_OHT.py <FAB|ALL> <시작날짜 YYYYMMDD> [끝날짜 YYYYMMDD]")
        try:
            ds = [datetime.strptime(t.replace("-", ""), "%Y%m%d") for t in toks]
        except ValueError:
            sys.exit(f"날짜 형식 오류: {toks} (YYYYMMDD)")
        d0, d1 = ds[0], ds[-1]
        if d1 < d0:
            d0, d1 = d1, d0
        day_job = (fabs, d0, d1)

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

    log.info("=" * 60)
    log.info("HID 병목 구간 — 맵 읽는 중")
    load_config(first=True)                          # ★HID_CONFIG.json — RUN_FABS · FAB 별 경계 · 위험 · 초위험
    need = None                                      # 이번에 돌릴 FAB — 기본 RUN_FABS
    if day_job:
        need = day_job[0] or list(FABS)              # 과거 판정: 적은 FAB, ALL 이면 5개 다
    elif a.range:
        need = [f.upper() for f in a.fab] if a.fab else list(RUN_FABS)
    elif a.map_at:
        need = [a.map_at[0].upper()] if a.map_at[0].upper() in FABS else None
    elif a.map:
        need = list(FABS)
    states = load_states(need)
    log.info(f"  돌리는 FAB: {' '.join(states)}")
    if not any(st.ok for st in states.values()):
        sys.exit("OHT_MAP 을 못 찾았습니다 — 월드모델파생의 OHT_MAP 폴더를 이 파일 옆에 두세요")
    log.info(f"  저장: {OUT_DIR}")
    log.info("=" * 60)

    global MAP_REALTIME, MAP_PAST
    if a.no_problem_map:
        MAP_REALTIME = False
    if a.problem_map:
        MAP_PAST = True
    log.info(f"  문제맵: ALARM {MAP_MIN_LEVEL} 이상인 1분마다 → {MAP_OUT} "
             f"(실시간 {'켬' if MAP_REALTIME else '끔'} · 과거 판정 {'켬' if MAP_PAST else '끔'})")
    if a.map:
        export_maps(states)
        return
    if a.map_at:
        fab = a.map_at[0].upper()
        if fab not in FABS or len(a.map_at) < 2:
            sys.exit(f"사용법: --map-at FAB YYYYMMDDHHMM [...]  (FAB: {' '.join(FABS)})")
        st = states[fab]
        if not st.ok:
            sys.exit(f"{fab} 맵 없음 — {st.map_msg}")
        for w in a.map_at[1:]:
            try:
                t = datetime.strptime(w.replace("-", "").replace(":", "").replace(" ", ""), "%Y%m%d%H%M")
            except ValueError:
                sys.exit(f"시간 형식 오류: {w} (YYYYMMDDHHMM)")
            log.info(f"  {fab} {t:%Y-%m-%d %H:%M} 문제맵 — 앞 {FLEET_AUTO_MIN}분부터 받아 판정")
            if not map_at(st, t):
                log.warning(f"  {fab} {t:%Y-%m-%d %H:%M} 문제맵 못 만듦 (그 시각 데이터 없음 · 조회 실패)")
        return
    if day_job:
        fabs, d0, d1 = day_job
        log.info(f"  과거 판정: FAB {' '.join(fabs or FABS)} · {d0:%Y-%m-%d} ~ {d1:%Y-%m-%d} "
                 f"({(d1 - d0).days + 1}일) → {PAST_DIR}")
        run_days(states, d0, d1, fabs)
        return
    if a.range:
        frm = datetime.strptime(a.range[0], "%Y%m%d%H%M")
        to = datetime.strptime(a.range[1], "%Y%m%d%H%M")
        fabs = None
        if a.fab:
            fabs = [f.upper() for f in a.fab]
            bad = [f for f in fabs if f not in FABS]
            if bad:
                sys.exit(f"모르는 FAB {bad} — 가능: {' '.join(FABS)}")
        log.info(f"  과거 판정: {frm:%Y-%m-%d %H:%M} ~ {to:%Y-%m-%d %H:%M} · FAB {' '.join(fabs or FABS)}")
        run_range(states, frm, to, fabs)
        return
    if a.test:
        run_cycle(states, write=False, verbose=True)
        return

    if _RULE_HID is not None:                        # 실시간 — CSV 쓰고 바로 로그프레소 적재 (config.json)
        _RULE_HID.start()
        if _RULE_HID.upload_rows not in SAVE_HOOKS:
            SAVE_HOOKS.append(_RULE_HID.upload_rows)
    elif _RULE_HID_ERR:
        log.warning(f"  Rule_hid.py 를 못 읽음 — 로그프레소 적재 없이 CSV · 문제맵만: {_RULE_HID_ERR}")
    else:
        log.info("  Rule_hid.py 없음 — 로그프레소 적재 없이 CSV · 문제맵만")
    try:
        _loop(states)
    finally:
        if _RULE_HID is not None:
            _RULE_HID.stop()


def _loop(states):
    while True:
        try:
            maybe_reload_config()
            run_cycle(states)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.warning(f"  사이클 실패 — 다음 구간에 다시: {e}")
        now = datetime.now()
        nxt = bucket_floor(now - timedelta(seconds=LAG_SEC)) + timedelta(seconds=BUCKET_SEC + LAG_SEC)
        time.sleep(max(5.0, (nxt - now).total_seconds() + 1))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("멈춤 (Ctrl+C)")
