#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HID_병목구간.py — 5 FAB OHT 병목 HID 구간 찾기 (따로 실행)

  50초마다 로그프레소에서 차량별 상태 · 위치(ADDRESS)를 받아
    ① 멈춘 차 = 미보고(끊기기 직전 위치) · JAM(STATUS 7) · HT_STOP(STATUS 8)
    ② 주소 → HID 구역 (OHT_MAP 의 HID_Zone_Master + 레이아웃)
    ③ 그 50초 구간 값으로 FAB 알람 판정 — 경계(전조) · 위험(사건) · 초위험(사건 시작)
    ④ 알람이면 멈춘 차가 많은 HID 구역 상위 TOP_N 개를 저장

  저장  HID_병목/{FAB}/HID병목_{FAB}_YYYYMMDD.csv  (FAB 마다 폴더 · 파일 따로, 실시간 1분마다 한 줄)
        날짜, 시간, HID_ZONE, {FAB}_OHT_report, {FAB}_OHT_missing, {FAB}_OHT_JAM, {FAB}_OHT_HT_STOP, ALARM, HID_section, …
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
    단계마다 숫자 3개, 둘 중 하나라도 맞으면:
      1. missing + JAM : 같은 50초 구간에 2개가 같이 기준 이상
      2. HT_STOP       : 1개만 기준 이상
    경계   전조       missing 7↑ + JAM 10↑   또는  HT_STOP 1↑
    위험   사건       missing 30↑ + JAM 20↑  또는  HT_STOP 10↑
    초위험 사건 시작  missing 50↑ + JAM 30↑  또는  HT_STOP 30↑

  ★기존 수집 데이터(M16A_HUBROOM_PR.csv 등)와 별개 — 로그프레소에서 차량 보고를 직접 받아
    보고 · 미보고 · JAM · HT_STOP 을 여기서 새로 계산한다. 다른 .py 파일 필요 없음.

  필요한 것 (같은 폴더)
    hdi_api_key.txt                     로그프레소 키 (첫 줄, 두 서버 같은 키)
    HID_CONFIG.json                     FAB 별 경계 · 위험 · 초위험 기준 (없으면 기본값으로 만든다)
    OHT_MAP/                            월드모델파생의 OHT_MAP 폴더 그대로
        MAP/{M14A,M14B,M16A,M16B}/HID_Zone_Master_*.csv
        cache/*_layout_cache.json

  실행
    python HID_병목구간.py                       계속 돈다 (Ctrl+C 로 멈춤)
    python HID_병목구간.py --test                구간표 확인 + 최근 10분 한 번 판정 (저장 안 함)
    python HID_병목구간.py --map                 주소 → HID 구역표를 CSV 로 내보냄 (확인용)
    python HID_병목구간.py M16HUB 20260929                  과거 판정 — FAB 하루
    python HID_병목구간.py M16HUB 20260912 20260929         과거 판정 — FAB 날짜 ~ 날짜 (하루씩 파일)
    python HID_병목구간.py ALL 20260912 20260929            FAB 5개 다
        → HID_병목/과거판정/{FAB}/HID병목_{FAB}_YYYYMMDD.csv   (FAB · 날짜별, 다시 돌리면 덮어씀)
        → HID_병목/과거판정/요약_{FAB|ALL}_{시작}_{끝}.csv     (FAB · 날짜별 정상 · 경계 · 위험 · 초위험 분 수)
    python HID_병목구간.py --range 202609290750 202609290830 --fab M16HUB   분 단위 구간 (검증용)
"""
import argparse
import csv
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

# ★알람 기준은 FAB 마다 따로 HID_CONFIG.json 에서 정한다 (없으면 아래 기본값으로 파일을 만든다).
#   기본값은 M16HUB 37일 데이터로 정한 값이라, 다른 FAB 는 운영하며 따로 맞춰야 한다.
#   고치면 다시 켜지 않아도 다음 50초 구간부터 적용된다. 숫자를 0 으로 두면 그 조건은 끈다.
CONFIG_FILE = HERE / "HID_CONFIG.json"
DEFAULT_LEVELS = {              # 단계마다 숫자 3개 — missing + JAM 은 같이, HT_STOP 은 혼자
    "경계":   {"missing": 7,  "JAM": 10, "HT_STOP": 1},
    "위험":   {"missing": 30, "JAM": 20, "HT_STOP": 10},
    "초위험": {"missing": 50, "JAM": 30, "HT_STOP": 30},
}
CONFIG_HELP = [
    "★ FAB 마다 따로 설정해야 합니다 — M14 · M14B · M16A · M16B · M16HUB 는 차량 수와 평소 수준이 달라서 같은 숫자를 쓰면 안 맞습니다.",
    "FAB 마다 경계 / 위험 / 초위험 기준 (그 50초 구간의 FAB 전체 값). 둘 중 하나라도 맞으면 그 단계:",
    "  1. missing + JAM : 같은 50초 구간에 missing 이상 AND JAM 이상 — 2개가 같이 발생해야 함",
    "  2. HT_STOP       : HT_STOP 이상 — 1개만 발생해도 됨",
    "초위험 -> 위험 -> 경계 순으로 본다. 숫자 0 = 그 조건 끔.",
    "고치면 프로그램을 다시 켜지 않아도 다음 50초 구간부터 적용됩니다.",
]
CFG = {}                        # FAB → {"경계", "위험", "초위험"}
_cfg_mtime = None
# HID_ZONE 에 쓸 마스터 칸 — 문서의 "HID 4 · HID 33" 번호 (HID IN/OUT 로그의 HID 번호) 와 같은 칸
#   "auto"     참조표(HID구간_참조_M16HUB.csv)와 ZONE_ID · ZONE_ID2 를 맞춰 보고 더 맞는 칸 (참조표 없으면 ZONE_ID)
#   "ZONE_ID" / "ZONE_ID2"   고정
HID_ZONE_COL = "auto"
REF_FILE     = HERE / "HID구간_참조_M16HUB.csv"     # 문서 만들 때 쓴 주소 → HID 구간 (4/21 HID IN/OUT 로그)
HID_ZONE_USE = "ZONE_ID"                            # 시작할 때 정해진다

WINDOW_MIN = 10                 # 메모리에 남겨 둘 과거 분 (판정은 그 50초 구간 하나로 한다)
TOP_N      = 3                  # 알람 때 저장할 HID 구역 수 (멈춘 차 많은 순)
IDLE_MIN   = 30                 # 이보다 오래 안 보인 차는 위치에서 뺀다 (정비 · 이탈)
FIRST_MIN  = 12                 # 처음 켤 때 받을 분
RANGE_CHUNK_MIN = 60            # --range 때 한 번에 받을 분
RANGE_TIMEOUT   = 180

MAP_DIRS = [HERE / "OHT_MAP", Path.cwd() / "OHT_MAP"]     # 앞에서부터 찾는다
OUT_DIR  = HERE / "HID_병목"
PAST_DIR = OUT_DIR / "과거판정"     # --range 결과 — 실시간 파일과 섞이지 않게 따로
LOG_FILE = HERE / "HID_병목구간.log"

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
    out = {"_설명": CONFIG_HELP}
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
    fabs = [k for k in cfg if k != "_설명"]
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
    global CFG, _cfg_mtime
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
            c[lv] = {k: _num(got.get(k), d) for k, d in dv.items()}
        new[fab] = c
    # ★예전 프로그램이 만든 파일에 남은 칸(전체차량 · 미보고_JAM_동시 · 구간수 등)은 지우고
    #   지금 형식으로 다시 쓴다. 숫자(missing · JAM · HT_STOP)는 고쳐 둔 값 그대로.
    clean = {"_설명": CONFIG_HELP, **new}
    if raw != clean:
        try:
            CONFIG_FILE.write_text(dump_config(clean), encoding="utf-8")
            log.info(f"  {CONFIG_FILE.name} 정리 — 안 쓰는 칸(전체차량 등)을 지우고 다시 썼습니다 (숫자는 그대로)")
        except Exception as e:
            log.warning(f"  {CONFIG_FILE.name} 정리 실패 (읽기는 정상): {e}")
    CFG = new
    _cfg_mtime = CONFIG_FILE.stat().st_mtime
    for fab, c in CFG.items():
        def one(lv):
            p = c[lv]
            a = f"missing{p['missing']}+JAM{p['JAM']}" if (p["missing"] and p["JAM"]) else "missing+JAM 끔"
            return f"{lv} {a} / HT_STOP{p['HT_STOP'] or ' 끔'}"
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
    info = {z: (d["name"], d["bay"], d["z2"], d["vmax"]) for z, d in zones.items()}
    ex = ", ".join(f"{z}→{zones[z]['z2'] or 0}" for z in sorted(zones, key=zkey)[:3])
    n2 = sum(1 for d in zones.values() if d["z2"])
    msg = (f"{master.name}: 구역 {len(zones)} · 주소 {len(zone_of)}/{len(L.get('nodes', {}))} · "
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
                self.last[v] = (b, addr[b][v])
                self.first.setdefault(v, b)
            fleet = sum(                             # 최근 100분 안에 보고한 차량 수
                1 for v, (lb, _) in self.last.items()
                if lb >= b - timedelta(minutes=FLEET_AUTO_MIN) and self.first[v] <= b)
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
                    if v in rep[b] or lb < idle:
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
          2. HT_STOP       : 1개만 기준 이상
          높은 단계부터 (초위험 → 위험 → 경계), 기준은 HID_CONFIG.json (0 = 그 조건 끔)
        """
        v = self.buckets.get(b)
        if not v or v.get("gap"):
            return None, ""
        P = policy(self.fab)
        for lv in ("초위험", "위험", "경계"):
            p = P[lv]
            hit_co = bool(p["missing"] and p["JAM"] and v["miss"] >= p["missing"] and v["jam"] >= p["JAM"])
            hit_ht = bool(p["HT_STOP"] and v["ht"] >= p["HT_STOP"])
            if hit_co or hit_ht:
                why = []
                if hit_co:
                    why.append(f"missing {v['miss']} + JAM {v['jam']} 같이 (≥{p['missing']}+{p['JAM']})")
                if hit_ht:
                    why.append(f"HT_STOP {v['ht']} (≥{p['HT_STOP']})")
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
    return ["날짜", "시간", "HID_ZONE", f"{fab}_OHT_report", f"{fab}_OHT_missing", f"{fab}_OHT_JAM", f"{fab}_OHT_HT_STOP",
            "ALARM", "HID_section",
            "ZONE_STOP", "ZONE_VHL", "VHL_MAX", "ZONE_OCC", "HID_ZONE_2", "HID_ZONE_3"]


def fab_file(fab, day, out_dir=None):
    """FAB 마다 폴더 따로 — HID_병목/{FAB}/HID병목_{FAB}_YYYYMMDD.csv"""
    d = (out_dir or OUT_DIR) / fab
    d.mkdir(parents=True, exist_ok=True)
    return d / f"HID병목_{fab}_{day}.csv"


def ensure_file(fab, day, out_dir=None):
    """알람이 없어도 그 날 파일은 헤더만 있는 채로 만들어 둔다 (FAB 5개 다 보이게)."""
    p = fab_file(fab, day, out_dir)
    if not p.exists():
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(header(fab))
    return p


def save(fab, rows, out_dir=None):
    if not rows:
        return
    by_day = defaultdict(list)
    for r in rows:
        by_day[r[0].replace("-", "")].append(r)
    for day, rs in by_day.items():
        p = ensure_file(fab, day, out_dir)
        with open(p, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rs)


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


def load_states():
    states = OrderedDict()
    for fab, cfg in FABS.items():
        st = FabState(fab, cfg)
        log.info(f"  {fab:6} {st.map_msg}")
        states[fab] = st
    resolve_hid_zone(states)
    return states


def process(st, frm, to, timeout=None, write=True, verbose=False):
    """frm~to 를 받아 쌓고, 끝난 분마다 1분 한 줄씩 저장. 알람 난 50초 구간 수를 돌려준다."""
    new = st.ingest(st.fetch(frm, to, timeout), frm, to)
    n_alarm = 0
    for b in new:                                    # 로그 — 알람 난 50초 구간
        level, why = st.alarm(b)
        if level:
            n_alarm += 1
            top = " · ".join(f"HID_ZONE {hid_zone(st, z)} {st.info.get(z, ('', '', '', 0))[0]}({m}/{j}/{h})"
                             for z, m, j, h, _ in st.top_zones(b))
            log.info(f"  ▲ {st.fab} {b:%H:%M:%S} {level} — {why} → {top or '위치 없음'}")
        elif verbose:
            v = st.buckets[b]
            log.info(f"    {st.fab} {b:%H:%M:%S} 정상 — 미보고 {v['miss']} JAM {v['jam']} HT {v['ht']}")
    rows = st.minute_rows()
    if write:
        save(st.fab, rows)
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
            n_alarm, _ = process(st, start, end, write=write, verbose=verbose)
            last = next((v for k, v in reversed(st.buckets.items()) if not v.get("gap")), None)
            if last is None:
                status[fab] = "데이터 없음"
            else:
                level, _ = st.alarm(max(k for k, v in st.buckets.items() if not v.get("gap")))
                status[fab] = (f"{level or '정상'}(보고 {last['n']} 미보고 {last['miss']} "
                               f"JAM {last['jam']} HT {last['ht']})")
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
                rows += process(st, cur, nxt, RANGE_TIMEOUT, write=False)[1]
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
    p = d / f"HID병목_{fab}_{name}.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:        # 다시 돌리면 덮어씀
        w = csv.writer(f)
        w.writerow(header(fab))
        w.writerows(rows)
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
    sp = PAST_DIR / f"요약_{name}.csv"
    with open(sp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FAB", "날짜/구간", "조회", "분", "정상", "경계", "위험", "초위험", "첫 알람", "첫 알람 HID_ZONE"])
        w.writerows(summary)
    log.info(f"  요약 → {sp}")


def run_days(states, d_start, d_end, fabs=None):
    """
    FAB · 날짜 ~ 날짜 — 하루씩 판정해 FAB 별 · 날짜별 CSV 로 저장
      HID_병목/과거판정/{FAB}/HID병목_{FAB}_YYYYMMDD.csv    (하루 1440줄, 다시 돌리면 덮어씀)
      HID_병목/과거판정/요약_{FAB|ALL}_{시작}_{끝}.csv       (FAB · 날짜별 정상 · 경계 · 위험 · 초위험 분 수)
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
    """분 단위 구간 (--range) — HID_병목/과거판정/{FAB}/HID병목_{FAB}_{시작}_{끝}.csv"""
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
        p = OUT_DIR / f"구간표_{fab}.csv"
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ADDRESS", "HID_ZONE", "ZONE_ID", "ZONE_ID2", "Full_Name", "HID_section"])
            for a in sorted(st.zone_of, key=lambda x: (len(x), x)):
                z = st.zone_of[a]
                name, bay, z2, _ = st.info.get(z, ("", "", "", 0))
                w.writerow([a, hid_zone(st, z), z, z2 or 0, name, bay])
        log.info(f"  {p}")


def main():
    ap = argparse.ArgumentParser(description="5 FAB OHT 병목 HID 구간 찾기")
    ap.add_argument("--test", action="store_true", help="구간표 확인 + 최근 10분 한 번 판정 (저장 안 함)")
    ap.add_argument("--map", action="store_true", help="주소 → HID 구역표 CSV 내보내기")
    ap.add_argument("--range", nargs=2, metavar=("YYYYMMDDHHMM", "YYYYMMDDHHMM"),
                    help="과거 구간 다시 판정 — FAB 별로 HID_병목/과거판정/ 에 저장")
    ap.add_argument("--fab", nargs="+", metavar="FAB", help=f"--range 할 FAB (기본 전부: {' '.join(FABS)})")
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
            sys.exit("사용법: python HID_병목구간.py <FAB|ALL> <시작날짜 YYYYMMDD> [끝날짜 YYYYMMDD]")
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
    states = load_states()
    if not any(st.ok for st in states.values()):
        sys.exit("OHT_MAP 을 못 찾았습니다 — 월드모델파생의 OHT_MAP 폴더를 이 파일 옆에 두세요")
    load_config(first=True)                         # ★HID_CONFIG.json — FAB 별 경계 · 위험 · 초위험
    log.info(f"  저장: {OUT_DIR}")
    log.info("=" * 60)

    if a.map:
        export_maps(states)
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
