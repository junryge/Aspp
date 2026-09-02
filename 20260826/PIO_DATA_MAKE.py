#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PIO_DATA_MAKE — PIO 반송 실패(DEPOSITED_FAIL_CNT) → 발동이벤트.csv 에 12컬럼 직접 기입 (운영용)
# ====================================================================
# lo_mac_maxcapa / LO_LOW_AMOS 와 동일한 구조. 조회 대상이 ICASTAR Oracle
# (STA_TRANS_TIMEOUT_FAIL_HIS) 이라는 것만 다르다. 별도 병합파일 안 만들고
# 발동이벤트.csv 자체에 기입한다.
#
# 접속 — 환경변수 (USER / PASSWORD 는 운영에서 직접 기입, 여기엔 넣지 않는다)
#   ORA_USER   접속 계정
#   ORA_PASS   비밀번호
#   ORA_DSN    기본 10.40.41.103:1521/ICASTARPP  (EZConnect, oracledb thin 모드)
#
# 조회 — 매분 1회, [now-lookback, now+1분) 구간을 분 단위로 집계:
#   GUBUN(경로) × 분 → FAIL_TYP='DEPOSIT' 건수 (DEPOSITED_FAIL_CNT)
#   GUBUN 판정은 고객이 준 CASE 식 그대로 (FAC_ID / FAB_ID / PORT_NM 앞글자)
#
# 추가 12컬럼 (같은 분끼리: 발동이벤트 datetime T == COMPLT_TM 의 분 T):
#   M16HUB->MLUD&DEPOSITED_FAIL_CNT&PIOERROR
#   M16HUB->M14B&DEPOSITED_FAIL_CNT&PIOERROR
#   M16HUB<-M14B&DEPOSITED_FAIL_CNT&PIOERROR
#   M16HUB->M14A&DEPOSITED_FAIL_CNT&PIOERROR
#   M16HUB<-M14A&DEPOSITED_FAIL_CNT&PIOERROR
#   M16HUB->M16A&DEPOSITED_FAIL_CNT&PIOERROR
#   M16HUB<-M16A&DEPOSITED_FAIL_CNT&PIOERROR
#   M16A->M16B&DEPOSITED_FAIL_CNT&PIOERROR
#   M16B->M16A&DEPOSITED_FAIL_CNT&PIOERROR
#   M14A->M14B&DEPOSITED_FAIL_CNT&PIOERROR
#   M14A<-M14B&DEPOSITED_FAIL_CNT&PIOERROR
#   M14A->M10A&DEPOSITED_FAIL_CNT&PIOERROR
#   · 조회가 닿은 분인데 실패가 없으면 0, 아직 조회 안 된 분(조회 실패 등)은 공란
#
# 실행:
#   운영(1분 루프):  python PIO_DATA_MAKE.py --event .\predict_tobe --loop
#                    (--event 폴더를 주면 최신 *발동이벤트*.csv 자동 선택, 자정 전환 대응)
#   1회만:           python PIO_DATA_MAKE.py --event .\predict_tobe\20260826_발동이벤트.csv
#   과거 일괄백필:   python PIO_DATA_MAKE.py --event .\predict_tobe --alldays
#   접속 점검:       python PIO_DATA_MAKE.py --test
#   조회본으로 테스트: python PIO_DATA_MAKE.py --event ... --source csv --pio 조회결과.csv --out 테스트.csv
#                    (조회결과.csv = GUBUN,EQP,GROUP1,DEPOSITED_FAIL_CNT 형식)
#   옵션: --interval 60 · --lookback 15 · --offset 35 · --dsn ... · --force
#
# 동작 원리 (lo_mac_maxcapa 와 동일):
#   · 시작 직후 1회 그날 파일 전체 재기입 + 그날 00:00~현재 전체 조회 → 자가복구
#   · 이후 사이클은 최근 --lookback 분(기본 15)만 조회해 캐시 갱신 (DB 부담 최소)
#     COMPLT_TM 은 완료 시각이라 늦게 적재될 수 있어, 최근 RECHECK_MIN 분은 매번 다시 쓴다
#   · 파일에 12컬럼 없으면 헤더에 추가, 있으면 이어서 기입
#   · 조회 실패(DB 불안정)면 그 사이클은 파일 안 건드리고 다음 분에 재시도
#   · 저장은 임시파일 → 원자 교체, 기입 중 파일 변경/잠김 감지 시 스킵 후 재시도
#   · 자정 전환 시 전날 파일 6사이클 더 마무리
#
# run_ml 통합 (스레드) — 다른 기입기 다음, 영역분리(area_split) 앞에 넣는다:
#   import PIO_DATA_MAKE
#   threading.Thread(target=PIO_DATA_MAKE.run_watch,
#                    kwargs={'event': str(predictor.DEFAULT_OUTPUT_DIR)}, daemon=True).start()
import argparse, csv, os, re, sys, time
from datetime import datetime, timedelta

# ────────────────────────────────────────────────────────────
# 접속 정보 — USER / PASSWORD 는 운영에서 기입 (여기 비워 둔다)
# ────────────────────────────────────────────────────────────
ORACLE_USER     = os.getenv("ORA_USER", "")
ORACLE_PASSWORD = os.getenv("ORA_PASS", "")
ORACLE_DSN      = os.getenv("ORA_DSN",  "10.40.41.103:1521/ICASTARPP")

# 컬럼 순서 = 고객 요청 순서 그대로
GUBUNS = [
    'M16HUB->MLUD', 'M16HUB->M14B', 'M16HUB<-M14B',
    'M16HUB->M14A', 'M16HUB<-M14A',
    'M16HUB->M16A', 'M16HUB<-M16A',
    'M16A->M16B', 'M16B->M16A',
    'M14A->M14B', 'M14A<-M14B',
    'M14A->M10A',
]
SUFFIX = '&DEPOSITED_FAIL_CNT&PIOERROR'      # 컬럼명 = {GUBUN}&DEPOSITED_FAIL_CNT&PIOERROR
NEW_COLS = [g + SUFFIX for g in GUBUNS]

RECHECK_MIN = 10      # 최근 N분은 매 사이클 재기입 (COMPLT_TM 지연 적재 보정)
FINISH_CYCLES = 6     # 자정 전환 후 전날 파일 마무리 사이클 수
CONN_TIMEOUT = 5      # 접속 타임아웃(초) — 죽은 노드에서 수십초 매달리지 않게

# 고객 쿼리를 분 구간 바인드로 바꾼 것. GUBUN 판정 CASE 는 그대로.
#   · EQP 컬럼은 기입에 안 쓰므로 뺐다 (GUBUN × 분 만 집계)
#   · 구간은 COMPLT_TM 문자열(YYYYMMDDHH24MISS) 비교 — 인덱스 그대로 탄다
SQL_FAIL = """
SELECT GUBUN,
       TO_CHAR(GROUP1, 'YYYY-MM-DD HH24:MI') AS GROUP1,
       SUM(CASE WHEN FT = 'DEPOSIT' THEN 1 ELSE 0 END) AS DEPOSITED_FAIL_CNT
FROM (
    SELECT TO_DATE(SUBSTR(A.COMPLT_TM, 1, 12), 'YYYYMMDDHH24MI') AS GROUP1,
           UPPER(TRIM(A.FAIL_TYP)) AS FT,
           CASE
               WHEN A.FAC_ID='M16' AND A.FAB_ID='M16HUB' AND UPPER(TRIM(A.PORT_NM)) LIKE '6FIOB%' THEN 'M16HUB->MLUD'
               WHEN A.FAC_ID='M16' AND A.FAB_ID='M16HUB' AND UPPER(TRIM(A.PORT_NM)) LIKE '4ABLD%' THEN 'M16HUB->M14B'
               WHEN A.FAC_ID='M14' AND A.FAB_ID='M14B'   AND UPPER(TRIM(A.PORT_NM)) LIKE '4ABLD%' THEN 'M16HUB<-M14B'
               WHEN A.FAC_ID='M16' AND A.FAB_ID='M16HUB' AND UPPER(TRIM(A.PORT_NM)) LIKE '4AFC%'  THEN 'M16HUB->M14A'
               WHEN A.FAC_ID='M14' AND A.FAB_ID='M14A'   AND UPPER(TRIM(A.PORT_NM)) LIKE '4AFC%'  THEN 'M16HUB<-M14A'
               WHEN A.FAC_ID='M16' AND A.FAB_ID='M14B'   AND UPPER(TRIM(A.PORT_NM)) LIKE '6ABL%'  THEN 'M16HUB->M16A'
               WHEN A.FAC_ID='M16' AND A.FAB_ID='M16A'   AND UPPER(TRIM(A.PORT_NM)) LIKE '6ABL%'  THEN 'M16HUB<-M16A'
               WHEN A.FAC_ID='M16' AND A.FAB_ID='M16A'   AND UPPER(TRIM(A.PORT_NM)) LIKE '6ALF%'  THEN 'M16A->M16B'
               WHEN A.FAC_ID='M16' AND A.FAB_ID='M16B'   AND UPPER(TRIM(A.PORT_NM)) LIKE '6ALF%'  THEN 'M16B->M16A'
               WHEN A.FAC_ID='M14' AND A.FAB_ID='M14A'   AND UPPER(TRIM(A.PORT_NM)) LIKE '4ALF%'  THEN 'M14A->M14B'
               WHEN A.FAC_ID='M14' AND A.FAB_ID='M14B'   AND UPPER(TRIM(A.PORT_NM)) LIKE '4ALF%'  THEN 'M14A<-M14B'
               WHEN A.FAC_ID='M14' AND A.FAB_ID='M10A'   AND UPPER(TRIM(A.PORT_NM)) LIKE '4ABL%'  THEN 'M14A->M10A'
           END AS GUBUN
    FROM STA_TRANS_TIMEOUT_FAIL_HIS A
    WHERE A.COMPLT_TM >= :t_from
      AND A.COMPLT_TM <  :t_to
      AND A.FAC_ID IN ('M14', 'M16')
)
WHERE GUBUN IS NOT NULL
GROUP BY GUBUN, GROUP1
ORDER BY GROUP1, GUBUN
"""


# ────────────────────────────────────────────────────────────
# Oracle 접속 (oracledb thin — Oracle Client 설치 불필요)
# ────────────────────────────────────────────────────────────
def _driver():
    try:
        import oracledb as drv
    except ImportError:
        try:
            import cx_Oracle as drv
        except ImportError:
            raise SystemExit('❌ oracledb 필요:  pip install oracledb')
    return drv


def connect(a):
    if not a.user or not a.password:
        raise SystemExit('❌ ORA_USER / ORA_PASS 환경변수(또는 --user/--password)가 비어 있습니다')
    drv = _driver()
    kw = dict(user=a.user, password=a.password, dsn=a.dsn)
    try:
        # oracledb ≥ 1.x : 접속 타임아웃 지원
        return drv.connect(tcp_connect_timeout=CONN_TIMEOUT, **kw)
    except TypeError:
        return drv.connect(**kw)


# ★ 커넥션 재사용 — 매분 새로 접속하면 수집기와 경합해 타임아웃난다
_CONN = {'h': None}


def get_conn(a, reset=False):
    if reset and _CONN['h'] is not None:
        try:
            _CONN['h'].close()
        except Exception:
            pass
        _CONN['h'] = None
    if _CONN['h'] is not None:
        try:
            _CONN['h'].ping()
            return _CONN['h']
        except Exception:
            try:
                _CONN['h'].close()
            except Exception:
                pass
            _CONN['h'] = None
    _CONN['h'] = connect(a)
    print(f'  🔌 Oracle 접속 {a.dsn} (커넥션 재사용 시작)')
    return _CONN['h']


def close_conn():
    if _CONN['h'] is not None:
        try:
            _CONN['h'].close()
        except Exception:
            pass
        _CONN['h'] = None


def _s(v):
    if v is None:
        return ''
    if isinstance(v, str):
        return v
    if isinstance(v, datetime):
        return v.strftime('%Y-%m-%d %H:%M:%S')
    if hasattr(v, 'read'):
        try:
            return v.read() or ''
        except Exception:
            return ''
    return str(v)


# ────────────────────────────────────────────────────────────
# 캐시 구조
#   cache['hits'][분키][GUBUN] = 건수
#   cache['covered'] = [(from, to), ...]  조회가 성공한 구간 — 이 안의 분은 없으면 0
# ────────────────────────────────────────────────────────────
def new_cache():
    return {'hits': {}, 'covered': []}


def _mark_covered(cache, dt_from, dt_to):
    cache['covered'].append((dt_from, dt_to))
    # 구간이 계속 쌓이지 않게 병합
    iv = sorted(cache['covered'])
    merged = [iv[0]]
    for f, t in iv[1:]:
        if f <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t))
        else:
            merged.append((f, t))
    cache['covered'] = merged


def _is_covered(cache, t):
    return any(f <= t < to for f, to in cache['covered'])


def _clear_window(cache, dt_from, dt_to):
    """재조회 구간의 옛 값을 비운다 (지연 적재로 건수가 늘어난 분을 새 값으로)."""
    for k in list(cache['hits']):
        t = parse_dt(k)
        if t and dt_from <= t < dt_to:
            del cache['hits'][k]


def fetch_db(a, dt_from, dt_to, cache):
    """[dt_from, dt_to) 구간 실패건 조회 → cache 갱신. 실패 시 False.
    커넥션은 재사용하고, 실패하면 한 번 재접속해서 재시도한다."""
    for attempt in (0, 1):
        try:
            return _fetch_once(a, dt_from, dt_to, cache, reset=(attempt == 1))
        except SystemExit:
            raise
        except Exception as e:
            msg = str(e).splitlines()[0]
            if attempt == 0:
                print(f'  ⚠️ 조회 실패({msg}) → 재접속 후 재시도')
                time.sleep(2)
            else:
                print(f'  ⚠️ Oracle 실패(재시도 소진): {msg}')
                close_conn()
    return False


def _fetch_once(a, dt_from, dt_to, cache, reset=False):
    t0 = time.time()
    conn = get_conn(a, reset=reset)
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(SQL_FAIL, {'t_from': dt_from.strftime('%Y%m%d%H%M%S'),
                               't_to': dt_to.strftime('%Y%m%d%H%M%S')})
        rows = cur.fetchall()
        _clear_window(cache, dt_from, dt_to)
        n = _ingest(cache, ((_s(r[0]), _s(r[1]), r[2]) for r in rows))
        _mark_covered(cache, dt_from, dt_to)
        print(f'  [DB] {dt_from:%m/%d %H:%M}~{dt_to:%H:%M} → 실패 있는 (분×경로) {n}건 · '
              f'{time.time()-t0:.1f}s')
        return True
    finally:
        try:
            if cur is not None:
                cur.close()
        except Exception:
            pass


def _ingest(cache, triples):
    """(GUBUN, 'YYYY-MM-DD HH:MI', cnt) 를 cache['hits'] 에 넣는다. 반환 = 건수>0 인 항목 수."""
    n = 0
    for gubun, tm, cnt in triples:
        gubun = (gubun or '').strip()
        if gubun not in GUBUNS:
            continue
        t = parse_dt(tm)
        if not t:
            continue
        try:
            c = int(float(cnt or 0))
        except (TypeError, ValueError):
            c = 0
        if c <= 0:
            continue
        k = t.strftime('%Y-%m-%d %H:%M')
        cache['hits'].setdefault(k, {})[gubun] = cache['hits'].get(k, {}).get(gubun, 0) + c
        n += 1
    return n


# ────────────────────────────────────────────────────────────
# CSV 원본 (--source csv) — 조회 결과 파일(GUBUN,EQP,GROUP1,DEPOSITED_FAIL_CNT)
# ────────────────────────────────────────────────────────────
def load_csv(path, cache):
    files = ([os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith('.csv')]
             if os.path.isdir(path) else ([path] if os.path.exists(path) else []))
    if not files:
        print(f'  ⚠️ PIO 조회 CSV 없음: {os.path.abspath(path)}')
        return False
    n, tmin, tmax = 0, None, None
    for fp in files:
        with open(fp, encoding='utf-8-sig') as f:
            trip = []
            for r in csv.DictReader(f):
                g, tm, c = r.get('GUBUN'), r.get('GROUP1'), r.get('DEPOSITED_FAIL_CNT')
                t = parse_dt(tm)
                if t:
                    tmin = t if tmin is None or t < tmin else tmin
                    tmax = t if tmax is None or t > tmax else tmax
                trip.append((g, tm, c))
            n += _ingest(cache, trip)
    if tmin and tmax:
        # 파일이 덮는 구간(그 날 하루)은 조회된 것으로 본다 → 없는 분은 0
        d0 = tmin.replace(hour=0, minute=0)
        d1 = tmax.replace(hour=0, minute=0) + timedelta(days=1)
        _mark_covered(cache, d0, d1)
    print(f'  [CSV] {len(files)}개 파일 → 실패 있는 (분×경로) {n}건 · 분 {len(cache["hits"])}개')
    return True


# ────────────────────────────────────────────────────────────
# 발동이벤트 파일
# ────────────────────────────────────────────────────────────
def parse_dt(s):
    s = (s or '').strip()
    if not s:
        return None
    try:
        d, t = s.split(' ', 1)
        y, mo, dd = [int(x) for x in d.replace('/', '-').split('-')]
        hm = t.split(':')
        return datetime(y, mo, dd, int(hm[0]), int(hm[1]))
    except (ValueError, IndexError):
        return None


def resolve_event(path):
    """폴더면 파일명 날짜(YYYYMMDD)가 가장 큰 *발동이벤트*.csv 자동 선택."""
    if os.path.isdir(path):
        cands = [f for f in os.listdir(path)
                 if f.lower().endswith('.csv') and '발동이벤트' in f]
        if not cands:
            return None
        dated = [(m.group(1), f) for f in cands for m in [re.search(r'(\d{8})', f)] if m]
        if dated:
            return os.path.join(path, max(dated)[1])
        return max((os.path.join(path, f) for f in cands), key=os.path.getmtime)
    return path if os.path.exists(path) else None


def cycle(a, cache, fp=None, state={'seen': set()}):
    """1사이클: 파일 읽기 → 해당 분 실패건수 채우기 → 원자 교체. return 기입행수 or None(스킵)."""
    if fp is None:
        fp = resolve_event(a.event)
    if not fp:
        ab = os.path.abspath(a.event)
        print(f'  ⚠️ {"경로 없음" if not os.path.exists(ab) else "발동이벤트 CSV 없음"}: {ab} (대기)')
        return None
    if fp not in state['seen']:
        print(f'  📄 대상 파일: {fp}'); state['seen'].add(fp)

    stat0 = os.stat(fp)
    with open(fp, encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        header = list(rd.fieldnames or [])
        rows = list(rd)
    if 'datetime' not in header:
        print("  ❌ 'datetime' 컬럼 없음"); return None
    times = [parse_dt(r.get('datetime')) for r in rows]
    valid = [t for t in times if t]
    if not valid:
        return None
    tmax = max(valid)
    force = getattr(a, 'force', False)

    def unfilled(r):
        # 컬럼이 없거나(None) 아직 공란인 행
        return any((r.get(c) is None or r.get(c) == '') for c in NEW_COLS)
    targets = [i for i, (r, t) in enumerate(zip(rows, times))
               if t and (force or unfilled(r) or (tmax - t) <= timedelta(minutes=RECHECK_MIN))]
    if not targets:
        return 0

    out_header = header + [c for c in NEW_COLS if c not in header]
    hit = filled = 0
    for i in targets:
        t = times[i]
        if not _is_covered(cache, t):
            continue                                   # 아직 조회 안 된 분 → 공란 유지
        e = cache['hits'].get(t.strftime('%Y-%m-%d %H:%M'), {})
        for g, col in zip(GUBUNS, NEW_COLS):
            rows[i][col] = str(e.get(g, 0))
        filled += 1
        hit += bool(e)
    for r in rows:
        for c in NEW_COLS:
            if r.get(c) is None:
                r[c] = ''

    out = a.out or fp
    tmp = out + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=out_header)
        w.writeheader(); w.writerows(rows)
    if out == fp:
        s1 = os.stat(fp)
        if (s1.st_mtime_ns, s1.st_size) != (stat0.st_mtime_ns, stat0.st_size):
            os.remove(tmp)
            print('  ⚠️ 기입 중 파일 변경 감지 → 스킵 (다음 사이클 재시도)')
            return None
    try:
        os.replace(tmp, out)
    except PermissionError:
        os.remove(tmp)
        print('  ⚠️ 파일 잠김(생성기 사용 중) → 스킵 (다음 사이클 재시도)')
        return None
    if hit:
        print(f'  ✏️ 실패 있는 분 {hit}개 기입 (0 포함 {filled}행)')
    return filled


def refresh(a, cache, dt_from, dt_to):
    if a.source == 'csv':
        return load_csv(a.pio, cache)
    return fetch_db(a, dt_from, dt_to, cache)


# ────────────────────────────────────────────────────────────
# 운영 루프
# ────────────────────────────────────────────────────────────
def _loop(a):
    print(f'[PIO_DATA_MAKE] {a.interval}초 간격 · 대상 {a.event} · 원본 '
          + ('CSV ' + str(a.pio) if a.source == 'csv' else f'Oracle {a.dsn}'))
    # ★ 수집기(매분 00초)·다른 기입기(+25초)와 같은 순간에 붙지 않게 시작을 어긋나게
    off = getattr(a, 'offset', 0)
    if off > 0:
        print(f'[PIO_DATA_MAKE] 다른 기입기와 겹치지 않게 {off}초 후 시작')
        time.sleep(off)
    cache, healed, finishing, cur = new_cache(), set(), {}, None
    user_force = getattr(a, 'force', False)
    while True:
        try:
            fp = resolve_event(a.event)
            now = datetime.now()
            heal = bool(fp) and fp not in healed
            if heal:
                print(f'  🩹 시작 복구: {os.path.basename(fp)} 전체 재기입')
                ok = refresh(a, cache, now.replace(hour=0, minute=0, second=0, microsecond=0),
                             now + timedelta(minutes=1))
            else:
                ok = refresh(a, cache, now - timedelta(minutes=a.lookback), now + timedelta(minutes=1))
            if not ok:
                print('  ⚠️ 원본 갱신 실패 → 이번 사이클 기입 생략')
                time.sleep(a.interval); continue

            a.force = user_force or heal
            if fp and cur and fp != cur and os.path.exists(cur):
                finishing[cur] = FINISH_CYCLES
                print(f'  🔄 날짜 전환 — 전날 파일 마무리: {os.path.basename(cur)}')
            if fp:
                cur = fp
            n = cycle(a, cache, fp=fp)
            if heal and n is not None:
                healed.add(fp)
            a.force = user_force
            for old in list(finishing):
                cycle(a, cache, fp=old)
                finishing[old] -= 1
                if finishing[old] <= 0:
                    del finishing[old]
                    print(f'  ✅ 전날 파일 마무리 완료: {os.path.basename(old)}')
            if n is not None:
                print(f'[PIO_DATA_MAKE {datetime.now():%H:%M:%S}] 기입 {n}행 '
                      f'(실패 있는 분 {len(cache["hits"])}개)')
            # 캐시가 하루치 넘게 쌓이지 않게 정리 (자정 전환 후)
            _prune(cache, now - timedelta(days=1))
            time.sleep(a.interval)
        except KeyboardInterrupt:
            print('\n[PIO_DATA_MAKE] 종료.'); break
        except Exception as e:
            print(f'  ⚠️ [PIO_DATA_MAKE] 오류(계속): {e}'); time.sleep(a.interval)


def _prune(cache, before):
    for k in list(cache['hits']):
        t = parse_dt(k)
        if t and t < before:
            del cache['hits'][k]
    cache['covered'] = [(f, t) for f, t in cache['covered'] if t >= before]


def run_watch(event='./predict_tobe', interval=60, lookback=15, offset=35,
              dsn=ORACLE_DSN, user=ORACLE_USER, password=ORACLE_PASSWORD):
    """run_ml 등에서 스레드로 돌리는 진입점. offset=다른 기입기와 겹침 방지(초)."""
    a = argparse.Namespace(event=event, out=None, interval=interval, lookback=lookback,
                           source='db', pio=None, force=False, offset=offset,
                           dsn=dsn, user=user, password=password)
    _loop(a)


def backfill_alldays(a):
    """폴더 내 모든 날짜 파일 일괄 기입 (항상 전체 덮어쓰기)."""
    if not os.path.isdir(a.event):
        print(f'❌ --alldays 는 폴더를 주세요: {a.event}'); sys.exit(2)
    files = sorted(f for f in os.listdir(a.event)
                   if f.lower().endswith('.csv') and '발동이벤트' in f)
    if not files:
        print(f'❌ {os.path.abspath(a.event)} 안에 *발동이벤트*.csv 없음'); sys.exit(2)
    a.force = True
    print(f'[백필] 대상 {len(files)}개 파일 — 전체 덮어쓰기')
    ok = fail = 0
    for f in files:
        fp = os.path.join(a.event, f)
        cache = new_cache()
        m = re.search(r'(\d{8})', f)
        if a.source == 'csv':
            refresh(a, cache, datetime.now(), datetime.now())
        elif m:
            d = datetime.strptime(m.group(1), '%Y%m%d')
            if not refresh(a, cache, d, d + timedelta(days=1)):
                fail += 1; print(f'  ❌ {f} — 조회 실패'); continue
        else:
            print(f'  ⚠️ {f} — 파일명에 날짜(YYYYMMDD)가 없어 건너뜀'); fail += 1; continue
        n = cycle(a, cache, fp=fp)
        if n is None:
            fail += 1; print(f'  ❌ {f} 실패')
        else:
            ok += 1; print(f'  ✅ {f} — {n}행 기입')
    print(f'🎉 백필 완료 — 성공 {ok} / 실패 {fail}')


def diagnose(a):
    """접속·조회 점검 — TCP 도달성 → 로그인 → 오늘/어제 조회."""
    import socket
    host, rest = a.dsn.split(':', 1) if ':' in a.dsn else (a.dsn, '1521')
    port = int(rest.split('/', 1)[0])
    print(f"DSN  : {a.dsn}   user: {a.user or '(비어 있음 — ORA_USER 를 넣으세요)'}")
    print()
    print(f"[1] TCP 도달성 ({CONN_TIMEOUT}초 제한)")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(float(CONN_TIMEOUT))
    t0 = time.time()
    try:
        s.connect((host, port))
        print(f"    ✅ {host}:{port}  ({time.time()-t0:.2f}s)")
    except socket.timeout:
        print(f"    ⛔ {host}:{port}  타임아웃 — 응답 없음(방화벽/노드 다운)")
    except Exception as e:
        print(f"    ❌ {host}:{port}  {e}")
    finally:
        s.close()
    print()
    print("[2] Oracle 로그인 + 조회")
    now = datetime.now()
    d0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for label, (f0, f1) in (('오늘 00:00~현재', (d0, now)),
                            ('어제 하루', (d0 - timedelta(days=1), d0))):
        cache = new_cache()
        ok = fetch_db(a, f0, f1, cache)
        print(f"  {'✅' if ok else '❌'} {label}: 실패 있는 분 {len(cache['hits'])}개")
        for k in sorted(cache['hits'])[:5]:
            e = cache['hits'][k]
            print(f"     {k}  " + '  '.join(f'{g}={c}' for g, c in e.items()))
    print()
    print('[해석]')
    print('  · TCP ⛔  → 이 서버에서 DB 로 나가는 망이 막힘 (코드 문제 아님, 방화벽 신청)')
    print('  · TCP ✅ 인데 로그인 실패 → 계정/서비스명 확인 (ORA-01017 / ORA-12514)')
    print('  · 전부 ✅ 인데 조회 0건 → 그 기간에 DEPOSIT 실패가 없던 것 (정상)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', default=None, help='발동이벤트.csv 또는 폴더(최신 날짜파일 자동)')
    ap.add_argument('--out', default=None, help='(테스트용) 원본 대신 여기에 저장')
    ap.add_argument('--loop', action='store_true', help='운영: interval초마다 반복')
    ap.add_argument('--alldays', action='store_true', help='폴더 내 모든 날짜 파일 일괄 기입(덮어쓰기)')
    ap.add_argument('--force', action='store_true', help='단일 파일도 전체 덮어쓰기')
    ap.add_argument('--test', action='store_true', help='접속·조회 점검 (--event 불필요)')
    ap.add_argument('--interval', type=int, default=60)
    ap.add_argument('--lookback', type=int, default=15, help='매 사이클 재조회할 최근 분 (기본 15)')
    ap.add_argument('--offset', type=int, default=35,
                    help='다른 기입기와 접속 시점 겹침 방지 지연(초, 기본 35). --loop 에만 적용')
    ap.add_argument('--source', choices=['db', 'csv'], default='db',
                    help='db=Oracle 직접(기본) · csv=조회결과 파일')
    ap.add_argument('--pio', default=None, help='--source csv 일 때 조회결과 CSV(또는 폴더)')
    ap.add_argument('--dsn', default=ORACLE_DSN, help=f'기본 {ORACLE_DSN} (환경변수 ORA_DSN)')
    ap.add_argument('--user', default=ORACLE_USER, help='기본 환경변수 ORA_USER')
    ap.add_argument('--password', default=ORACLE_PASSWORD, help='기본 환경변수 ORA_PASS')
    a = ap.parse_args()

    print('=' * 60)
    print('발동이벤트 ← PIO DEPOSITED_FAIL_CNT 12컬럼 기입'
          + (' (접속점검)' if a.test else ' (운영 루프)' if a.loop
             else ' (과거 일괄백필)' if a.alldays else ' (1회)'))
    print('=' * 60)

    if a.test:
        diagnose(a); return
    if not a.event:
        print('❌ --event 가 필요합니다 (점검은 --test)'); sys.exit(2)
    if a.source == 'csv' and not a.pio:
        print('❌ --source csv 는 --pio 조회결과.csv 가 필요합니다'); sys.exit(2)
    if a.loop:
        _loop(a)
    elif a.alldays:
        backfill_alldays(a)
    else:
        cache = new_cache()
        fp = resolve_event(a.event)
        day = None
        if fp:
            m = re.search(r'(\d{8})', os.path.basename(fp))
            if m:
                day = datetime.strptime(m.group(1), '%Y%m%d')
        day = day or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if not refresh(a, cache, day, day + timedelta(days=1)):
            sys.exit(2)
        a.force = True
        n = cycle(a, cache, fp=fp)
        if n is None:
            sys.exit(2)
        print(f'🎉 완료 — {n}행 기입 → {a.out or fp}')


if __name__ == '__main__':
    main()
