#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HID_병목구간.py — 5 FAB OHT 병목 HID 구간 찾기 (따로 실행)

  50초마다 로그프레소에서 차량별 상태 · 위치(ADDRESS)를 받아
    ① 멈춘 차 = 미보고(끊기기 직전 위치) · JAM(STATUS 7) · HT_STOP(STATUS 8)
    ② 주소 → HID 구역 (OHT_MAP 의 HID_Zone_Master + 레이아웃)
    ③ 과거 10분(50초 12구간)으로 FAB 알람 판정 — 경계(전조) · 위험(사건) · 초위험(사건 시작)
    ④ 알람이면 멈춘 차가 많은 HID 구역 상위 TOP_N 개를 저장

  저장  HID_병목/{FAB}/HID병목_{FAB}_YYYYMMDD.csv  (FAB 마다 폴더 · 파일 따로, 실시간 1분마다 한 줄)
        날짜, 시간, HID_ZONE, {FAB}_OHT_missing, {FAB}_OHT_JAM, {FAB}_OHT_HT_STOP, ALARM, HID_section
          missing/JAM/HT_STOP = FAB 전체 (그 분이 끝날 때까지 끝난 마지막 50초 구간)
          ALARM        = 경계 / 위험 / 초위험, 조건에 안 맞으면 0
          HID_ZONE     = 병목 HID 구역 번호 (HID_Zone_Master 의 Zone_ID, 과거 10분 멈춘 차 최다), 알람 아니면 0
          HID_section  = 그 구역의 Bay_Zone, 알람 아니면 0
          데이터가 없는 분은 전부 0

  알람 (과거 10분, FAB 전체 50초 구간 값)  — POLICY 에서 바꾼다
    경계   전조       같은 50초 구간에 미보고 7↑ AND JAM 10↑, 또는 HT_STOP 1↑
    위험   사건       미보고 30↑ 2구간 연속, 또는 JAM 20↑ 구간 2개↑, 또는 HT_STOP 10↑
    초위험 사건 시작  미보고 100↑ 2구간 연속, 또는 JAM 40↑, 또는 HT_STOP 30↑

  ★기존 수집 데이터(M16A_HUBROOM_PR.csv 등)와 별개 — 로그프레소에서 차량 보고를 직접 받아
    보고 · 미보고 · JAM · HT_STOP 을 여기서 새로 계산한다. 다른 .py 파일 필요 없음.

  필요한 것 (같은 폴더)
    hdi_api_key.txt                     로그프레소 키 (첫 줄, 두 서버 같은 키)
    OHT_MAP/                            월드모델파생의 OHT_MAP 폴더 그대로
        MAP/{M14A,M14B,M16A,M16B}/HID_Zone_Master_*.csv
        cache/*_layout_cache.json

  실행
    python HID_병목구간.py                       계속 돈다 (Ctrl+C 로 멈춤)
    python HID_병목구간.py --test                구간표 확인 + 최근 10분 한 번 판정 (저장 안 함)
    python HID_병목구간.py --map                 주소 → HID 구역표를 CSV 로 내보냄 (확인용)
    python HID_병목구간.py --range 202609290750 202609290830     과거 구간 다시 판정해 저장
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
FLEET = {"M14": 0, "M14B": 0, "M16A": 0, "M16B": 0, "M16HUB": 270}   # 0 = 최근 100분 보고 차량 수로 자동
FLEET_AUTO_MIN  = 100

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

POLICY = {                      # ★스코어 정책 — 기본 미보고 7 · JAM 10
    "경계":   {"co_miss": 7, "co_jam": 10, "ht": 1},
    "위험":   {"miss": 30, "miss_consec": 2, "jam": 20, "jam_count": 2, "ht": 10},
    "초위험": {"miss": 100, "miss_consec": 2, "jam": 40, "ht": 30},
}
WINDOW_MIN = 10                 # 과거 10분 = 50초 12구간
TOP_N      = 3                  # 알람 때 저장할 HID 구역 수 (멈춘 차 많은 순)
IDLE_MIN   = 30                 # 이보다 오래 안 보인 차는 위치에서 뺀다 (정비 · 이탈)
FIRST_MIN  = 12                 # 처음 켤 때 받을 분
RANGE_CHUNK_MIN = 60            # --range 때 한 번에 받을 분
RANGE_TIMEOUT   = 180

MAP_DIRS = [HERE / "OHT_MAP", Path.cwd() / "OHT_MAP"]     # 앞에서부터 찾는다
OUT_DIR  = HERE / "HID_병목"
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
    HID_Zone_Master 의 IN/OUT 레인 + 레이아웃 → {주소: Zone_ID}, {Zone_ID: (Full_Name, Bay_Zone)}
      구역마다 IN 레인 도착 주소에서 출발해 레이아웃을 따라가되 그 구역 OUT 레인은 넘지 않는다.
      여러 구역에 걸리는 주소는 IN 에서 가장 가까운 구역 = 차가 마지막으로 들어간 구역.
    """
    master, layout = _find_map_files(fab_dir, prefix)
    if not master:
        return None, None, f"맵 파일 없음 (OHT_MAP/MAP/{fab_dir}/HID_Zone_Master_{fab_dir}_{prefix}.csv · " \
                           f"OHT_MAP/cache/{fab_dir}_{prefix}_layout_cache.json)"
    L = json.loads(layout.read_text(encoding="utf-8"))
    adj = {str(a): [str(b) for b in bs] for a, bs in L.get("adj", {}).items()}

    zones = {}
    with open(master, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            z = (r.get("Zone_ID") or "").strip()
            if not z.isdigit() or int(z) <= 0:
                continue
            d = zones.setdefault(int(z), {"name": (r.get("Full_Name") or "").strip() or f"Zone-{z}",
                                          "bay": (r.get("Bay_Zone") or "").strip(),
                                          "in": set(), "out": set()})
            d["in"].update(_lanes(r.get("IN_Lanes")))
            d["out"].update(_lanes(r.get("OUT_Lanes")))

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
            if n not in best or dist < best[n][0] or (dist == best[n][0] and z < best[n][1]):
                best[n] = (dist, z)

    zone_of = {n: z for n, (_, z) in best.items()}
    info = {z: (d["name"], d["bay"]) for z, d in zones.items()}
    msg = f"{master.name}: 구역 {len(zones)} · 주소 {len(zone_of)}/{len(L.get('nodes', {}))}"
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
            fleet = FLEET.get(self.fab) or sum(
                1 for v, (lb, _) in self.last.items()
                if lb >= b - timedelta(minutes=FLEET_AUTO_MIN) and self.first[v] <= b)
            miss = max(0, fleet - n)
            gap = miss >= fleet * 0.9 and not jam[b] and not ht[b]
            zm, zj, zh = Counter(), Counter(), Counter()
            if not gap:
                idle = b - timedelta(minutes=IDLE_MIN)
                for v, (lb, a) in self.last.items():
                    if v in rep[b] or lb < idle:
                        continue
                    z = self.zone_of.get(a)
                    if z:
                        zm[z] += 1
                for v in jam[b]:
                    z = self.zone_of.get(addr[b][v])
                    if z:
                        zj[z] += 1
                for v in ht[b]:
                    z = self.zone_of.get(addr[b][v])
                    if z:
                        zh[z] += 1
            self.buckets[b] = {"gap": gap, "n": n, "miss": miss, "jam": len(jam[b]), "ht": len(ht[b]),
                               "zm": zm, "zj": zj, "zh": zh}
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
        """과거 10분 → (ALARM, 이유) · 알람 없으면 (None, '')"""
        win = self.window(b)
        if not win:
            return None, ""
        step = timedelta(seconds=BUCKET_SEC)
        byk = dict(win)

        def consec(th, k):                           # th 이상이 k 구간 연속
            for t, v in win:
                if all((t + step * i) in byk and byk[t + step * i]["miss"] >= th for i in range(k)):
                    return True
            return False

        mx_jam = max(v["jam"] for _, v in win)
        mx_ht = max(v["ht"] for _, v in win)
        mx_miss = max(v["miss"] for _, v in win)
        p = POLICY["초위험"]
        if consec(p["miss"], p["miss_consec"]) or mx_jam >= p["jam"] or mx_ht >= p["ht"]:
            return "초위험", f"미보고 {mx_miss} · JAM {mx_jam} · HT {mx_ht}"
        p = POLICY["위험"]
        if (consec(p["miss"], p["miss_consec"]) or sum(1 for _, v in win if v["jam"] >= p["jam"]) >= p["jam_count"]
                or mx_ht >= p["ht"]):
            return "위험", f"미보고 {mx_miss} · JAM {mx_jam} · HT {mx_ht}"
        p = POLICY["경계"]
        co = [v for _, v in win if v["miss"] >= p["co_miss"] and v["jam"] >= p["co_jam"]]
        if co or mx_ht >= p["ht"]:
            return "경계", f"동시 구간 {len(co)} · HT {mx_ht}"
        return None, ""

    # ---------- ④ 병목 HID 구역 ----------
    def top_zones(self, b, n=TOP_N):
        """과거 10분 멈춘 차 합이 많은 구역 → [(Zone_ID, 미보고 최대, JAM 최대, HT 최대, 합)]"""
        tot, mm, mj, mh = Counter(), Counter(), Counter(), Counter()
        for _, v in self.window(b):
            for src, mx in ((v["zm"], mm), (v["zj"], mj), (v["zh"], mh)):
                for z, c in src.items():
                    tot[z] += c
                    mx[z] = max(mx[z], c)
        return [(z, mm[z], mj[z], mh[z], s) for z, s in tot.most_common(n)]

    def bucket_of_minute(self, m):
        """1분 행 m(HH:MM) 에 쓸 50초 구간 = 그 분이 끝날 때까지 끝난 마지막 구간"""
        return bucket_floor(m + timedelta(seconds=60)) - timedelta(seconds=BUCKET_SEC)

    def minute_row(self, m):
        """
        1분 한 줄: 날짜, 시간, HID_ZONE, {FAB}_OHT_missing, _JAM, _HT_STOP, ALARM, HID_section
          missing · JAM · HT_STOP = FAB 전체 (그 분의 마지막 50초 구간)
          ALARM · HID_ZONE · HID_section = 알람일 때만, 아니면 0
          데이터가 없으면 전부 0
        """
        b = self.bucket_of_minute(m)
        v = self.buckets.get(b)
        head = [f"{m:%Y-%m-%d}", f"{m:%H:%M}"]
        if not v or v.get("gap"):
            return head + [0, 0, 0, 0, 0, 0], None
        level, _ = self.alarm(b)
        zone, bay = 0, 0
        if level:
            top = self.top_zones(b, 1)
            if top:
                zone = top[0][0]
                bay = self.info.get(zone, ("", ""))[1] or 0
        return head + [zone, v["miss"], v["jam"], v["ht"], level or 0, bay], level

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
    return ["날짜", "시간", "HID_ZONE", f"{fab}_OHT_missing", f"{fab}_OHT_JAM", f"{fab}_OHT_HT_STOP",
            "ALARM", "HID_section"]


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
def load_states():
    states = OrderedDict()
    for fab, cfg in FABS.items():
        st = FabState(fab, cfg)
        log.info(f"  {fab:6} {st.map_msg}")
        states[fab] = st
    return states


def process(st, frm, to, timeout=None, write=True, verbose=False):
    """frm~to 를 받아 쌓고, 끝난 분마다 1분 한 줄씩 저장. 알람 난 50초 구간 수를 돌려준다."""
    new = st.ingest(st.fetch(frm, to, timeout), frm, to)
    n_alarm = 0
    for b in new:                                    # 로그 — 알람 난 50초 구간
        level, why = st.alarm(b)
        if level:
            n_alarm += 1
            top = " · ".join(f"{z}번 {st.info.get(z, ('', ''))[0]}({m}/{j}/{h})"
                             for z, m, j, h, _ in st.top_zones(b))
            log.info(f"  ▲ {st.fab} {b:%H:%M:%S} {level} — {why} → {top or '위치 없음'}")
        elif verbose:
            v = st.buckets[b]
            log.info(f"    {st.fab} {b:%H:%M:%S} 정상 — 미보고 {v['miss']} JAM {v['jam']} HT {v['ht']}")
    rows = st.minute_rows()
    if write:
        save(st.fab, rows)
    return n_alarm


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
            n_alarm = process(st, start, end, write=write, verbose=verbose)
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


def run_range(states, frm, to):
    for fab, st in states.items():
        if not st.ok:
            log.info(f"  {fab}: 맵 없음 — 건너뜀")
            continue
        ensure_file(fab, f"{frm:%Y%m%d}")
        st.minute_done = frm.replace(second=0) - timedelta(minutes=1)
        cur = bucket_floor(frm - timedelta(minutes=WINDOW_MIN))   # 앞 10분부터 받아야 첫 판정이 맞다
        first_save = bucket_floor(frm)
        n = 0
        while cur < to:
            nxt = min(bucket_floor(cur + timedelta(minutes=RANGE_CHUNK_MIN)), bucket_floor(to))
            if nxt <= cur:
                nxt = cur + timedelta(seconds=BUCKET_SEC)
            try:
                if nxt <= first_save:                # 앞 10분은 쌓기만
                    st.ingest(st.fetch(cur, nxt, RANGE_TIMEOUT), cur, nxt)
                else:
                    if cur < first_save:
                        st.ingest(st.fetch(cur, first_save, RANGE_TIMEOUT), cur, first_save)
                        cur = first_save
                    n += process(st, cur, nxt, RANGE_TIMEOUT)
            except Exception as e:
                log.warning(f"  {fab} {cur:%m/%d %H:%M}~{nxt:%H:%M} 실패 — 건너뜀: {e}")
            cur = nxt
        log.info(f"  {fab}: 알람 구간 {n}개 저장")


def export_maps(states):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fab, st in states.items():
        if not st.ok:
            continue
        p = OUT_DIR / f"구간표_{fab}.csv"
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ADDRESS", "Zone_ID", "HID구역", "HID_section"])
            for a in sorted(st.zone_of, key=lambda x: (len(x), x)):
                z = st.zone_of[a]
                w.writerow([a, z, *st.info.get(z, ("", ""))])
        log.info(f"  {p}")


def main():
    ap = argparse.ArgumentParser(description="5 FAB OHT 병목 HID 구간 찾기")
    ap.add_argument("--test", action="store_true", help="구간표 확인 + 최근 10분 한 번 판정 (저장 안 함)")
    ap.add_argument("--map", action="store_true", help="주소 → HID 구역표 CSV 내보내기")
    ap.add_argument("--range", nargs=2, metavar=("YYYYMMDDHHMM", "YYYYMMDDHHMM"), help="과거 구간 다시 판정")
    a = ap.parse_args()

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
    log.info(f"  정책: 경계 미보고 {POLICY['경계']['co_miss']}+JAM {POLICY['경계']['co_jam']} · "
             f"위험 미보고 {POLICY['위험']['miss']}×{POLICY['위험']['miss_consec']} · "
             f"초위험 미보고 {POLICY['초위험']['miss']}×{POLICY['초위험']['miss_consec']} · 과거 {WINDOW_MIN}분")
    log.info(f"  저장: {OUT_DIR}")
    log.info("=" * 60)

    if a.map:
        export_maps(states)
        return
    if a.range:
        frm = datetime.strptime(a.range[0], "%Y%m%d%H%M")
        to = datetime.strptime(a.range[1], "%Y%m%d%H%M")
        run_range(states, frm, to)
        return
    if a.test:
        run_cycle(states, write=False, verbose=True)
        return

    while True:
        try:
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
