# -*- coding: utf-8 -*-
"""
hid_aws_idc_realtime_collector.py — FAB별 OHT 50초 집계 (aws_idc_realtime_collector.py 에서 import)

  서버                  테이블              FAB      컬럼
  10.40.42.167:8888    oht_data_m16br  →  M16HUB   M16HUB_OHT_missing, M16HUB_OHT_JAM, M16HUB_OHT_HT_STOP
                       oht_data_m16a   →  M16A     M16A_OHT_missing,   M16A_OHT_JAM,   M16A_OHT_HT_STOP
                       oht_data_m16b   →  M16B     M16B_OHT_missing,   M16B_OHT_JAM,   M16B_OHT_HT_STOP
  10.40.42.27:8888     oht_data_m14a   →  M14      M14_OHT_missing,    M14_OHT_JAM,    M14_OHT_HT_STOP
                       oht_data_m14b   →  M14B     M14B_OHT_missing,   M14B_OHT_JAM,   M14B_OHT_HT_STOP

  {FAB}_OHT_missing = 미보고 = 전체 차량 − 그 50초 안에 보고한 차량
  {FAB}_OHT_JAM     = 그 50초 안에 STATUS 7 이 있었던 차량
  {FAB}_OHT_HT_STOP = 그 50초 안에 STATUS 8 이 있었던 차량

  · 50초 구간으로 집계한다 (1분으로 합치지 않는다).
    1분 행에는 그 분이 끝날 때까지 끝난 마지막 50초 구간 값을 넣는다.
  · 처음엔 최근 12분만 받고, 그다음부터는 새로 끝난 50초 구간만 받는다 (그 전 행은 빈칸).
  · 조회가 안 되거나 그 구간 보고가 0 이면 빈칸. 절대 예외를 밖으로 던지지 않는다.
  · 서버 하나에 못 붙으면 3초 만에 포기하고 그 서버 FAB 만 빈칸 — 다른 서버는 그대로 받는다.

  키:    같은 폴더 hdi_api_key.txt  (한 줄 = 두 서버 공통, 또는 M16=… / M14=… 서버별)
  확인:  python hid_aws_idc_realtime_collector.py      (Oracle 없이 최근 값만 출력)
"""
import csv
import io
import logging
import urllib.parse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("idc_collector_v42")

# ==========================================================
# 설정
# ==========================================================
# 서버별 접속. key 를 비우면 같은 폴더 hdi_api_key.txt 에서 읽는다.
#   hdi_api_key.txt — 한 줄이면 두 서버 같은 키,  서버별로 다르면
#       M16=167서버키
#       M14=27서버키
#   ★저장소에 올릴 때는 key 를 비우세요.
SERVERS = {
    "M16": {"host": "10.40.42.167", "port": 8888, "key": "", "remote": "icamcslogdt01"},
    "M14": {"host": "10.40.42.27",  "port": 8888, "key": "", "remote": "icamcslogdt01"},
}

TABLES = {                        # 테이블 → (FAB, 서버)
    "oht_data_m14a":  ("M14",    "M14"),
    "oht_data_m14b":  ("M14B",   "M14"),
    "oht_data_m16a":  ("M16A",   "M16"),
    "oht_data_m16b":  ("M16B",   "M16"),
    "oht_data_m16br": ("M16HUB", "M16"),
}
# 전체 차량 수. 0 이면 최근에 한 번이라도 보고한 차량 수로 자동 계산
FLEET = {"M14": 0, "M14B": 0, "M16A": 0, "M16B": 0, "M16HUB": 270}

CONNECT_TIMEOUT = 3               # 서버에 못 붙으면 3초 만에 포기 (매분 저장을 붙잡지 않게)
HTTP_TIMEOUT    = 15              # 붙은 뒤 응답 기다리는 시간
BUCKET_SEC = 50                   # ★50초
LAG_SEC    = 30                   # 구간 끝나고 이만큼 지난 뒤에 받는다 (로그프레소 적재 지연)
KEEP_MIN   = 100                  # 메모리에 들고 있을 분 (수집기 WINDOW 90분 + 여유)
FIRST_MIN  = 12                   # 처음 켤 때 받을 분 (길게 받으면 첫 저장이 늦어진다)

COLUMNS = []
for _fab, _ in TABLES.values():
    COLUMNS += [f"{_fab}_OHT_missing", f"{_fab}_OHT_JAM", f"{_fab}_OHT_HT_STOP"]

EPOCH_KST = datetime(1970, 1, 1, 9, 0, 0)       # datetrunc(_time, "50s") 와 같은 기준
FMT = "%Y%m%d%H%M%S"

_cache = {t: {} for t in TABLES}                 # 테이블 → {구간시작: (보고, JAM, HT)}
_seen = {t: {} for t in TABLES}                  # 테이블 → {차량: 마지막 보고 구간}  (FLEET 0 일 때)
_done = {t: None for t in TABLES}                # 테이블 → 여기 전까지 받았다


def bucket_floor(t):
    sec = int((t - EPOCH_KST).total_seconds()) // BUCKET_SEC * BUCKET_SEC
    return EPOCH_KST + timedelta(seconds=sec)


KEY_FILES = ("hdi_api_key.txt", "hid_api_key.txt")
_key_warned = False


def _key(server):
    """파일 위 SERVERS 의 key → hdi_api_key.txt (서버별 'M16=…' 줄, 없으면 첫 줄)"""
    global _key_warned
    if SERVERS[server].get("key"):
        return SERVERS[server]["key"]
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        for name in KEY_FILES:
            p = base / name
            if not p.exists():
                continue
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
            plain = [ln for ln in lines if "=" not in ln]
            for ln in lines:
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    if k.strip().upper() == server:
                        return v.strip()
            return plain[0] if plain else ""
    if not _key_warned:
        log.warning(f"  OHT: {KEY_FILES[0]} 없음 — 수집기와 같은 폴더에 두세요 (인증 실패로 빈칸)")
        _key_warned = True
    return ""


def _query(server, table, frm, to):
    inner = (f"table from={frm.strftime(FMT)} to={to.strftime(FMT)} {table}"
             ' | search MSG_ID == "2"'
             " | fields _time, VEHICLE, STATUS"
             f' | eval _time = datetrunc(_time, "{BUCKET_SEC}s")'
             " | stats first(STATUS) as STATUS, last(STATUS) as STATUS_LAST by VEHICLE, _time")
    remote = SERVERS[server].get("remote")
    return f"remote {remote} [ {inner} ]" if remote else inner


def _get(server, q):
    import requests
    s = SERVERS[server]
    url = (f"http://{s['host']}:{s['port']}/logpresso/httpexport/query.csv"
           f"?_apikey={_key(server)}&_q={urllib.parse.quote(q, safe='')}")
    r = requests.get(url, verify=False, timeout=(CONNECT_TIMEOUT, HTTP_TIMEOUT))
    if r.status_code != 200 or r.text.lstrip().startswith("<"):
        raise RuntimeError(f"HTTP {r.status_code} from {s['host']}:{s['port']}: {r.text[:200]}")
    return r.content


def _parse_time(s):
    s = (s or "").strip().strip('"')[:19]
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    return None


def _fetch_table(table, server, now):
    end = bucket_floor(now - timedelta(seconds=LAG_SEC))
    oldest = bucket_floor(now - timedelta(minutes=KEEP_MIN))
    start = max(_done[table] or bucket_floor(now - timedelta(minutes=FIRST_MIN)), oldest)
    if start >= end:
        return
    body = _get(server, _query(server, table, start, end))

    rep, jam, ht = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in csv.DictReader(io.StringIO(body.decode("utf-8-sig", "replace"))):
        v = (r.get("VEHICLE") or "").strip()
        t = _parse_time(r.get("_time"))
        if not v or not t:
            continue
        b = bucket_floor(t)
        sts = {(r.get("STATUS") or "").strip(), (r.get("STATUS_LAST") or "").strip()}
        rep[b].add(v)
        if "7" in sts:
            jam[b].add(v)
        if "8" in sts:
            ht[b].add(v)

    c, seen = _cache[table], _seen[table]
    b = start
    while b < end:
        for v in rep[b]:
            seen[v] = b
        c[b] = (len(rep[b]), len(jam[b]), len(ht[b]))
        b += timedelta(seconds=BUCKET_SEC)
    for k in [k for k in c if k < oldest]:
        del c[k]
    for v in [v for v, tb in seen.items() if tb < oldest]:
        del seen[v]
    _done[table] = end


def fetch(minute_keys=None, now=None):
    """새로 끝난 50초 구간을 테이블마다 받는다. 실패해도 예외 없음 (그 FAB 만 빈칸)."""
    import requests
    now = now or datetime.now()
    down = set()                                  # 이번 분에 못 붙은 서버
    for table, (fab, server) in TABLES.items():
        if server in down:
            continue
        try:
            _fetch_table(table, server, now)
        except requests.exceptions.ConnectionError:
            # 서버에 못 붙음 — 같은 서버의 나머지 테이블은 이번 분 건너뛴다 (3초만 쓰고 빠진다)
            down.add(server)
            s = SERVERS[server]
            fabs = ", ".join(f for f, sv in TABLES.values() if sv == server)
            log.warning(f"  OHT: 로그프레소 {s['host']}:{s['port']} 연결 안 됨 — "
                        f"이번 분 {fabs} 빈칸, 다음 분에 이어 받는다")
        except Exception as e:
            log.warning(f"  OHT {fab}({table}) 조회 실패 — 다음 분에 이어 받는다: {e}")


def values(minute_key):
    """minute_key('YYYY-MM-DD HH:MM[:SS]') 행에 붙일 값 (COLUMNS 순서). 없으면 빈칸."""
    try:
        asof = datetime.strptime(str(minute_key)[:16], "%Y-%m-%d %H:%M") + timedelta(seconds=60)
    except ValueError:
        return [""] * len(COLUMNS)
    b = bucket_floor(asof) - timedelta(seconds=BUCKET_SEC)   # 그 분 끝까지 끝난 마지막 50초 구간
    out = []
    for table, (fab, _) in TABLES.items():
        v = _cache[table].get(b)
        if not v or v[0] == 0:                                # 조회 안 됨 / 보고 0 = 수집 누락
            out += ["", "", ""]
            continue
        n, j, h = v
        fleet = FLEET.get(fab) or len(_seen[table])
        out += [max(0, fleet - n), j, h]
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    for name, s in SERVERS.items():
        k = _key(name)
        fabs = ", ".join(f for f, sv in TABLES.values() if sv == name)
        print(f"[접속] {s['host']}:{s['port']} → {fabs} · 키 {'있음 …' + k[-4:] if k else '없음 (hdi_api_key.txt)'}")
    now = datetime.now()
    fetch(now=now)
    for m in range(5, 0, -1):
        k = (now - timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M")
        print(k, dict(zip(COLUMNS, values(k))))
