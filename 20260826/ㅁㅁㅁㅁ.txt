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
#   M16HUB->MLUD_PIOERROR_DEPOSITED
#   M16HUB->M14B_PIOERROR_DEPOSITED
#   M16HUB<-M14B_PIOERROR_DEPOSITED
#   M16HUB->M14A_PIOERROR_DEPOSITED
#   M16HUB<-M14A_PIOERROR_DEPOSITED
#   M16HUB->M16A_PIOERROR_DEPOSITED
#   M16HUB<-M16A_PIOERROR_DEPOSITED
#   M16A->M16B_PIOERROR_DEPOSITED
#   M16B->M16A_PIOERROR_DEPOSITED
#   M14A->M14B_PIOERROR_DEPOSITED
#   M14A<-M14B_PIOERROR_DEPOSITED
#   M14A->M10A_PIOERROR_DEPOSITED
#   · 조회가 닿은 분인데 실패가 없으면 0, 아직 조회 안 된 분(조회 실패 등)은 공란
#
# 실행:
#   운영(1분 루프):  python PIO_DATA_MAKE.py --event .\predict_tobe --loop
#                    (--event 폴더를 주면 최신 *발동이벤트*.csv 자동 선택, 자정 전환 대응)
#   1회만:           python PIO_DATA_MAKE.py --event .\predict_tobe\20260826_발동이벤트.csv
#   접속 점검:       python PIO_DATA_MAKE.py --test
#
#   ★ 빠진 데이터 다시 넣기 (수동 — 자동으로는 안 돈다)
#     빈 곳 보기:    python PIO_DATA_MAKE.py --event .\predict_tobe --gaps
#     빈 곳만 채움:  python PIO_DATA_MAKE.py --event .\predict_tobe --heal
#     어제부터:      python PIO_DATA_MAKE.py --event .\predict_tobe --heal --since 20260902
#     최근 3일:      python PIO_DATA_MAKE.py --event .\predict_tobe --heal --days 3
#     한 파일만:     python PIO_DATA_MAKE.py --event .\predict_tobe\20260902_발동이벤트.csv --heal
#     통째로 다시:   python PIO_DATA_MAKE.py --event .\predict_tobe --alldays --days 2
#                    (--heal 은 빈 칸만 / --alldays 는 전체 덮어쓰기)
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
SUFFIX = '_PIOERROR_DEPOSITED'      # 컬럼명 = {GUBUN}_PIOERROR_DEPOSITED  (고객 확정 이름)
NEW_COLS = [g + SUFFIX for g in GUBUNS]
# 예전 이름으로 이미 기입된 파일은 헤더만 새 이름으로 바꾸고 값은 그대로 둔다
OLD_SUFFIXES = ['&DEPOSITED_FAIL_CNT&PIOERROR', '&DEPOSITED_FAIL_CNT']
RENAME = {g + o: g + SUFFIX for g in GUBUNS for o in OLD_SUFFIXES}

RECHECK_MIN = 10      # 최근 N분은 매 사이클 재기입 (COMPLT_TM 지연 적재 보정)
FINISH_CYCLES = 6     # 자정 전환 후 전날 파일 마무리 사이클 수
CONN_TIMEOUT = 5      # 접속 타임아웃(초) — 죽은 노드에서 수십초 매달리지 않게
# ★ 커넥션 최대 수명(초). 밤새 유휴면 방화벽이 TCP 세션을 조용히 끊어 ping 으로도 못 잡는다.
#   이 시간이 지나면 조회 전에 무조건 새로 접속한다.
CONN_MAX_AGE = 600
# ★ 사이클 로그 — predict_tobe/pio_log.txt.  콘솔이 스크롤돼 사라져도 "언제부터 왜 안 됐나" 를 남긴다
#   (2026-09 암호 문제로 하루치가 통째로 안 들어간 뒤 추가)
LOG_NAME = 'pio_log.txt'
# ★ 빈 구간 메우기 — 자동으로 돌지 않는다. 필요할 때 --heal 로만 실행한다.
#   (암호·망 문제로 빠진 시간대를 그날 파일에서 찾아 그 구간만 다시 조회해 채움)
SELFHEAL_MAX_RANGES = 20     # 한 번에 메우는 구간 수 (DB 부담 제한)

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


def _fatal_reason(msg):
    """재시도해도 안 풀리는 실패인지 — 사람이 고쳐야 하는 것만 골라 이름을 붙인다."""
    m = (msg or '').upper()
    if 'ORA-01017' in m or 'DPY-4001' in m:  return '계정/암호 틀림 (ORA-01017) — ORA_USER / ORA_PASS 확인'
    if 'ORA-28000' in m:                     return '계정 잠김 (ORA-28000) — DBA 에게 잠금 해제 요청'
    if 'ORA-28001' in m:                     return '암호 만료 (ORA-28001) — 새 암호로 ORA_PASS 갱신'
    if 'ORA-28002' in m:                     return '암호 만료 임박 (ORA-28002) — 곧 갱신 필요'
    if 'ORA-01005' in m:                     return '암호가 비어 있음 (ORA-01005) — ORA_PASS 확인'
    if 'ORA-12514' in m or 'ORA-12541' in m: return '서비스명/리스너 문제 — ORA_DSN 확인'
    if 'ORA-00942' in m:                     return '테이블 권한 없음 (ORA-00942) — 계정 권한 확인'
    return None


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
_CONN = {'h': None, 'born': 0.0}


def get_conn(a, reset=False):
    # 오래된 커넥션은 살아 있어 보여도 버린다 (유휴 중 끊긴 세션은 ping 이 못 잡는다)
    if _CONN['h'] is not None and (time.time() - _CONN['born']) > CONN_MAX_AGE:
        reset = True
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
    _CONN['born'] = time.time()
    _log(a, f'🔌 Oracle 접속 {a.dsn} (user={a.user or "(비어 있음)"})')
    return _CONN['h']


def close_conn():
    if _CONN['h'] is not None:
        try:
            _CONN['h'].close()
        except Exception:
            pass
        _CONN['h'] = None


def _log(a, msg):
    """콘솔 + predict_tobe/pio_log.txt 양쪽에. 로그 실패로 기입을 멈추지는 않는다."""
    line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
    print('  ' + line)
    try:
        ev = getattr(a, 'event', None)
        if not ev:
            return
        d = ev if os.path.isdir(ev) else os.path.dirname(os.path.abspath(ev))
        with open(os.path.join(d, LOG_NAME), 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


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
            fatal = _fatal_reason(msg)
            if fatal:
                # 재시도해도 안 풀린다 — 사람이 고쳐야 하는 것이므로 즉시 크게 남긴다
                _log(a, f'❌❌ 접속 실패 — {fatal}   [원문: {msg}]')
                close_conn()
                return False
            if attempt == 0:
                _log(a, f'⚠️ 조회 실패({msg}) → 재접속 후 재시도')
                time.sleep(2)
            else:
                _log(a, f'⚠️ Oracle 실패(재시도 소진): {msg}')
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
    # CSV 원본은 항상 전체를 담는다 — 여러 번 불려도 건수가 누적되지 않게 비우고 시작
    cache['hits'].clear()
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
    # 예전 컬럼명 → 새 컬럼명 (값 보존). 새 이름이 이미 있으면 예전 것은 버린다.
    old_in = [c for c in header if c in RENAME]
    if old_in:
        for c in old_in:
            new = RENAME[c]
            if new in header:
                header.remove(c)
                for r in rows:
                    r.pop(c, None)
            else:
                header[header.index(c)] = new
                for r in rows:
                    r[new] = r.pop(c, None)
        print(f'  🔁 예전 PIO 컬럼명 {len(old_in)}개 → 새 이름으로 변경 (값 유지)')
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


def find_gaps(fp, gap_join_min=5):
    """PIO 12컬럼이 비어 있는(또는 컬럼 자체가 없는) 행들의 시간 구간 [(from, to), ...]."""
    miss = []
    try:
        with open(fp, encoding='utf-8-sig') as f:
            rd = csv.DictReader(f)
            hdr = rd.fieldnames or []
            if 'datetime' not in hdr:
                return []
            no_col = not all(c in hdr for c in NEW_COLS)
            for r in rd:
                t = parse_dt(r.get('datetime'))
                if not t:
                    continue
                if no_col or any((r.get(c) or '') == '' for c in NEW_COLS):
                    miss.append(t)
    except Exception:
        return []
    if not miss:
        return []
    miss.sort()
    out, s, e = [], miss[0], miss[0]
    for t in miss[1:]:
        if (t - e) <= timedelta(minutes=gap_join_min):
            e = t
        else:
            out.append((s, e)); s = e = t
    out.append((s, e))
    return out


def selfheal(a, cache, fp, now, quiet=False):
    """빈 PIO 구간만 다시 조회해 채운다 (--heal). 채운 구간 수 반환."""
    gaps = find_gaps(fp)
    # 최근 RECHECK_MIN 분은 아직 안 들어왔을 뿐이라 정상 — 제외
    gaps = [(s, e) for s, e in gaps if (now - e) > timedelta(minutes=RECHECK_MIN)]
    if not gaps:
        if not quiet:
            print(f'  ✅ {os.path.basename(fp)} — 빈 구간 없음')
        return 0
    total = sum(int((e - s).total_seconds() // 60) + 1 for s, e in gaps)
    _log(a, f'🩹 빈 PIO 구간 {len(gaps)}개 ({total}분) 발견 → 다시 조회해 채웁니다: '
            + ', '.join(f'{s:%H:%M}~{e:%H:%M}' for s, e in gaps[:SELFHEAL_MAX_RANGES]))
    done = 0
    for s, e in gaps[:SELFHEAL_MAX_RANGES]:
        if refresh(a, cache, s, e + timedelta(minutes=1)):
            done += 1
    if done:
        keep, a.force = getattr(a, 'force', False), True
        n = cycle(a, cache, fp=fp)
        a.force = keep
        _log(a, f'🩹 자가치유 기입 {n if n is not None else "스킵"}행 ({done}/{len(gaps)} 구간 조회 성공)')
    return done


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
    _log(a, f'▶ PIO 기입 시작 — {a.interval}초 간격 · '
            + (f'CSV {a.pio}' if a.source == 'csv' else f'Oracle {a.dsn} · user={a.user or "(비어 있음)"}'))
    cache, healed, finishing, cur = new_cache(), set(), {}, None
    user_force = getattr(a, 'force', False)
    fails = 0          # 연속 조회 실패 횟수
    while True:
        try:
            fp = resolve_event(a.event)
            now = datetime.now()
            heal = bool(fp) and fp not in healed
            if heal:
                _log(a, f'🩹 시작 복구: {os.path.basename(fp)} 전체 재기입')
                ok = refresh(a, cache, now.replace(hour=0, minute=0, second=0, microsecond=0),
                             now + timedelta(minutes=1))
            else:
                ok = refresh(a, cache, now - timedelta(minutes=a.lookback), now + timedelta(minutes=1))
            if not ok:
                fails += 1
                # 계속 실패하면 그냥 조용히 돌지 않는다 — 1·3·10·30·60회째에 크게 남긴다
                if fails in (1, 3, 10, 30, 60) or fails % 60 == 0:
                    _log(a, f'⚠️⚠️ 원본 갱신 {fails}회 연속 실패 — 이 시간 PIO 컬럼은 비어 있게 됩니다. '
                            '위의 ❌❌ 줄에 사유가 있으면 그것부터 고치세요.')
                time.sleep(a.interval); continue
            if fails:
                _log(a, f'✅ 조회 정상 복구 (연속 실패 {fails}회 후)')
                fails = 0

            a.force = user_force or heal
            if fp and cur and fp != cur and os.path.exists(cur):
                finishing[cur] = FINISH_CYCLES
                _log(a, f'🔄 날짜 전환 — 전날 파일 마무리: {os.path.basename(cur)}')
                close_conn()          # 밤새 유휴였을 커넥션은 날짜 넘어갈 때 새로 잡는다
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
                    _log(a, f'✅ 전날 파일 마무리 완료: {os.path.basename(old)}')
            if n is not None:
                _log(a, f'기입 {n}행 · {os.path.basename(fp) if fp else "-"} '
                        f'(실패 있는 분 {len(cache["hits"])}개)')
            else:
                _log(a, '⚠️ 기입 스킵 (파일 잠김/변경 감지) — 다음 사이클 재시도')

            # 캐시가 하루치 넘게 쌓이지 않게 정리 (자정 전환 후)
            _prune(cache, now - timedelta(days=1))
            time.sleep(a.interval)
        except KeyboardInterrupt:
            print('\n[PIO_DATA_MAKE] 종료.'); break
        except Exception as e:
            _log(a, f'⚠️ 오류(계속): {e}'); time.sleep(a.interval)


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
    if not a.user or not a.password:
        # 스레드 안에서 SystemExit 이 나면 소리 없이 죽는다 → 여기서 분명히 알리고 물러난다
        miss = ' / '.join(n for n, v in (('ORA_USER', a.user), ('ORA_PASS', a.password)) if not v)
        _log(a, f'❌❌ {miss} 가 비어 있어 PIO 기입을 시작하지 않습니다 — '
                'run_ml 을 띄우는 창(또는 서비스 계정)에 환경변수를 넣고 재시작하세요. '
                '다른 스레드에는 영향 없음.')
        return
    try:
        _loop(a)
    except SystemExit as e:
        _log(a, f'❌❌ 중단: {e}')


def backfill_alldays(a):
    """과거 발동이벤트에 PIO 12컬럼을 채운다 (기본: 전체 덮어쓰기).

    범위 좁히기 — 안 주면 폴더 안 전부:
        --since 20260901        그 날짜부터
        --until 20260903        그 날짜까지
        --days 3                최근 3일 (오늘 포함)
        --only-missing          PIO 12컬럼이 비어 있는 파일만 (이미 채워진 날은 건너뜀)
    운영이 돌고 있어도 안전하다 — 임시파일 원자 교체 + 잠김이면 재시도.
    """
    if not os.path.isdir(a.event):
        print(f'❌ 백필은 폴더를 주세요: {a.event}'); sys.exit(2)
    files = sorted(f for f in os.listdir(a.event)
                   if f.lower().endswith('.csv') and '발동이벤트' in f and '_M1' not in f)
    if not files:
        print(f'❌ {os.path.abspath(a.event)} 안에 *발동이벤트*.csv 없음'); sys.exit(2)

    # 날짜 범위 필터
    since = getattr(a, 'since', None)
    until = getattr(a, 'until', None)
    if getattr(a, 'days', None):
        since = max(since or '', (datetime.now() - timedelta(days=a.days - 1)).strftime('%Y%m%d'))
    picked = []
    for f in files:
        m = re.search(r'(\d{8})', f)
        if not m:
            print(f'  ⚠️ {f} — 파일명에 날짜(YYYYMMDD)가 없어 건너뜀'); continue
        d = m.group(1)
        if (since and d < since) or (until and d > until):
            continue
        picked.append((d, f))
    if not picked:
        print(f'❌ 범위에 맞는 파일 없음 (since={since} until={until})'); sys.exit(2)

    # 이미 채워진 날은 건너뛰기
    if getattr(a, 'only_missing', False):
        keep = []
        for d, f in picked:
            try:
                with open(os.path.join(a.event, f), encoding='utf-8-sig') as fh:
                    rd = csv.DictReader(fh)
                    hdr = rd.fieldnames or []
                    has = all(c in hdr for c in NEW_COLS)
                    filled = has and any((r.get(NEW_COLS[0]) or '') != '' for r in rd)
            except Exception:
                filled = False
            (print(f'  ⏭️  {f} — 이미 기입돼 있어 건너뜀') if filled else keep.append((d, f)))
        picked = keep
        if not picked:
            print('🎉 백필할 파일 없음 — 전부 이미 채워져 있습니다'); return

    a.force = True
    print(f'[백필] 대상 {len(picked)}개 파일 ({picked[0][0]} ~ {picked[-1][0]}) — 전체 덮어쓰기')
    ok = fail = 0
    for d, f in picked:
        fp = os.path.join(a.event, f)
        cache = new_cache()
        if a.source == 'csv':
            refresh(a, cache, datetime.now(), datetime.now())
        else:
            day = datetime.strptime(d, '%Y%m%d')
            if not refresh(a, cache, day, day + timedelta(days=1)):
                fail += 1; print(f'  ❌ {f} — 조회 실패 (위 사유 확인)'); continue
        # 운영이 같은 파일을 쓰는 중이면 잠김/변경으로 스킵될 수 있다 → 몇 번 다시 시도
        n = None
        for attempt in range(5):
            n = cycle(a, cache, fp=fp)
            if n is not None:
                break
            time.sleep(2)
        if n is None:
            fail += 1; print(f'  ❌ {f} 실패 (파일 사용 중 — 잠시 후 다시 시도하세요)')
        else:
            ok += 1; print(f'  ✅ {f} — {n}행 기입')
    print(f'🎉 백필 완료 — 성공 {ok} / 실패 {fail}')


def _pick_files(a, need_dir_msg=''):
    """--event 가 파일이면 그 하나, 폴더면 발동이벤트 전부 (+ --since/--until/--days 필터).

    파일을 못 찾으면 어디를 봤는지·비슷한 후보가 뭔지 찍는다 (경로 오타 잡기용).
    """
    ev = (a.event or '').strip().strip('"').strip("'")     # 따옴표·공백이 붙어 오는 경우
    if not os.path.isdir(ev):
        if os.path.exists(ev):
            return [ev]
        print(f'  ❌ 파일을 못 찾음: {os.path.abspath(ev)}')
        # 같은 폴더에서 비슷한 이름 찾아 보여 준다
        d = os.path.dirname(os.path.abspath(ev)) or '.'
        base = os.path.basename(ev)
        if os.path.isdir(d):
            cand = [f for f in sorted(os.listdir(d))
                    if f.lower().endswith('.csv') and '발동이벤트' in f and '_M1' not in f]
            if cand:
                print(f'     그 폴더({d})에 있는 발동이벤트 파일:')
                for f in cand[-10:]:
                    print(f'       {f}')
                print('     ↑ 이름을 그대로 복사해 쓰거나, 폴더만 주고 --days 1 로 지정하세요.')
            else:
                print(f'     그 폴더({d})에는 발동이벤트 CSV 가 없습니다 — 경로를 확인하세요.')
        else:
            print(f'     상위 폴더도 없습니다: {d}')
        if base and '발동이벤트' not in base:
            print('     ※ 파일명에 "발동이벤트" 가 없습니다 — 다른 파일을 지정한 것 아닌가요?')
        return []
    files = sorted(f for f in os.listdir(ev)
                   if f.lower().endswith('.csv') and '발동이벤트' in f and '_M1' not in f)
    if not files:
        print(f'  ❌ {os.path.abspath(ev)} 안에 *발동이벤트*.csv 가 없습니다')
        return []
    a.event = ev
    since, until = getattr(a, 'since', None), getattr(a, 'until', None)
    if getattr(a, 'days', None):
        since = max(since or '', (datetime.now() - timedelta(days=a.days - 1)).strftime('%Y%m%d'))
    out = []
    for f in files:
        m = re.search(r'(\d{8})', f)
        if not m:
            continue
        if (since and m.group(1) < since) or (until and m.group(1) > until):
            continue
        out.append(os.path.join(a.event, f))
    if not out:
        got = ', '.join(sorted({re.search(r'(\d{8})', f).group(1)
                                for f in files if re.search(r'(\d{8})', f)})) or '(날짜 없는 파일들)'
        print(f'  ❌ 범위에 맞는 파일 없음 — since={since or "-"} until={until or "-"} / 폴더에 있는 날짜: {got}')
    return out


def show_gaps(a):
    """--gaps : 빈 PIO 구간이 어디인지 보기만 한다 (기입·조회 없음)."""
    files = _pick_files(a)
    if not files:
        sys.exit(2)
    now = datetime.now()
    total_files = total_min = 0
    for fp in files:
        gaps = [(s, e) for s, e in find_gaps(fp) if (now - e) > timedelta(minutes=RECHECK_MIN)]
        if not gaps:
            print(f'  ✅ {os.path.basename(fp)} — 빈 구간 없음')
            continue
        mins = sum(int((e - s).total_seconds() // 60) + 1 for s, e in gaps)
        total_files += 1; total_min += mins
        print(f'  ⚠️  {os.path.basename(fp)} — 빈 구간 {len(gaps)}개 · {mins}분')
        for s, e in gaps[:20]:
            print(f'        {s:%m-%d %H:%M} ~ {e:%H:%M}')
        if len(gaps) > 20:
            print(f'        … 외 {len(gaps)-20}개')
    print(f'\n🎉 점검 완료 — 빈 구간 있는 파일 {total_files}개 / 총 {total_min}분')
    if total_files:
        print('   채우려면 같은 명령에서 --gaps 를 --heal 로 바꿔 실행하세요.')


def heal_files(a):
    """--heal : 빈 PIO 구간만 그 시간대를 다시 조회해 채운다."""
    files = _pick_files(a)
    if not files:
        sys.exit(2)
    now = datetime.now()
    print(f'[빈 구간 메우기] 대상 {len(files)}개 파일')
    ok = skip = 0
    for fp in files:
        cache = new_cache()
        n = selfheal(a, cache, fp, now)
        ok += bool(n); skip += (not n)
    print(f'🎉 완료 — 채운 파일 {ok}개 / 빈 구간 없던 파일 {skip}개')


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
    ap.add_argument('--alldays', action='store_true', help='과거 파일 일괄 기입(덮어쓰기). 범위는 --since/--until/--days')
    ap.add_argument('--since', default=None, metavar='YYYYMMDD', help='백필 시작 날짜 (이 날짜 포함)')
    ap.add_argument('--until', default=None, metavar='YYYYMMDD', help='백필 끝 날짜 (이 날짜 포함)')
    ap.add_argument('--days', type=int, default=None, help='백필: 최근 N일 (오늘 포함)')
    ap.add_argument('--only-missing', dest='only_missing', action='store_true',
                    help='백필: PIO 컬럼이 비어 있는 파일만 (이미 채워진 날은 건너뜀)')
    ap.add_argument('--force', action='store_true', help='단일 파일도 전체 덮어쓰기')
    ap.add_argument('--test', action='store_true', help='접속·조회 점검 (--event 불필요)')
    ap.add_argument('--heal', action='store_true',
                    help='빈 PIO 구간만 찾아 그 시간대를 다시 조회해 채움 (파일/폴더 모두 가능). '
                         '범위는 --since/--until/--days')
    ap.add_argument('--gaps', action='store_true',
                    help='빈 PIO 구간이 어디인지 보기만 함 (기입 안 함)')
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
          + (' (접속점검)' if a.test else ' (빈 구간 보기)' if a.gaps
             else ' (빈 구간 메우기)' if a.heal else ' (운영 루프)' if a.loop
             else ' (과거 일괄백필)' if a.alldays else ' (1회)'))
    print('=' * 60)

    if a.test:
        diagnose(a); return
    if not a.event:
        print('❌ --event 가 필요합니다 (점검은 --test)'); sys.exit(2)
    if a.source == 'csv' and not a.pio:
        print('❌ --source csv 는 --pio 조회결과.csv 가 필요합니다'); sys.exit(2)
    if a.gaps:
        show_gaps(a)
    elif a.heal:
        heal_files(a)
    elif a.loop:
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
