#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcs_maxcapa — MAXCAPA 조작내역 → 발동이벤트.csv 에 4컬럼 직접 기입 (운영용)
====================================================================
LO_LOW_AMOS 와 동일한 구조. 다른 점은 조회 대상이 로그프레소(HTTP) 가 아니라
MCS 운영 Oracle 이라는 것뿐이다. 별도 병합파일 안 만들고 발동이벤트.csv 자체에 기입.

조회 (maxcapa_v3.py 와 동일한 2단계):
  ① NT_L_LOGMESSAGE 에서 COMMUNICATIONMESSAGENAME='UI-UNIT-PORT-MAXCAPACITY-UPDATE'
     → TRANSACTIONID / TIME / PROCESSNAME / PARTITIONID
  ② 각 TRANSACTIONID 재조회 → TEXT 에서
     port{6ABL6031_AI612}.maxCapacity was changed to {1}  파싱

추가 4컬럼 (같은 분끼리: 발동이벤트 datetime T == 조작시각 T):
  MACHINE            그 분에 조작된 설비 (여러개면 쉼표)
  PORT:후(after)     'PORT:후값' 을 줄바꿈으로 나열 — 예)
                       6ABL6031_AI612:1
                       6ABL6031_AI622:1
                       6ABL6031_AO623:1
  PROCESS            TS15 등 (여러개면 쉼표)
  TRANSACTIONID      MCS... (여러개면 쉼표)

접속: 같은 폴더 mcs_config.ini  ([oracle] hosts/port/service_name/user/password)
      드라이버는 oracledb 또는 cx_Oracle 중 설치된 것을 자동 사용.
      RAC 4노드 FAILOVER DSN 을 config 값으로 조립한다.

실행:
  운영(1분 루프):  python mcs_maxcapa.py --event .\predict_tobe --loop
                   (--event 폴더를 주면 최신 *발동이벤트*.csv 자동 선택, 자정 전환 대응)
  1회만:           python mcs_maxcapa.py --event .\predict_tobe\20260728_발동이벤트.csv
  과거 일괄백필:   python mcs_maxcapa.py --event .\predict_tobe --alldays
  접속 점검:       python mcs_maxcapa.py --test
  CSV 사용(옵션):  --source csv --maxcapa .\maxcapa_v3.csv   (DB 대신 수집본 사용)
  테스트(원본보존): --out .\테스트.csv
  옵션: --interval 60 · --lookback 20 · --config mcs_config.ini · --force

동작 원리 (LO_LOW_AMOS 와 동일):
  · 시작 직후 1회 그날 파일 전체 재기입 + 그날 00:00~현재 전체 조회 → 자가복구
  · 이후 사이클은 최근 --lookback 분만 조회해서 캐시 갱신 (DB 부담 최소)
  · 파일에 4컬럼 없으면 헤더에 추가, 있으면 이어서 기입
  · 조회 실패(DB 불안정)면 그 사이클은 파일 안 건드리고 다음 분에 재시도
  · 저장은 임시파일 → 원자 교체, 기입 중 파일 변경/잠김 감지 시 스킵 후 재시도
  · 자정 전환 시 전날 파일 6사이클 더 마무리
  · 조작 0건인 분은 공란 (정상 — 대부분의 분에는 조작이 없다)

run_ml 통합 (스레드):
  import mcs_maxcapa
  threading.Thread(target=mcs_maxcapa.run_watch,
                   kwargs={'event': str(predictor.DEFAULT_OUTPUT_DIR)}, daemon=True).start()
"""
import argparse, configparser, csv, os, re, sys, time
from datetime import datetime, timedelta

NEW_COLS = ['MACHINE', 'PORT:후(after)', 'PROCESS', 'TRANSACTIONID']
RECHECK_MIN = 5       # 최근 N분은 매 사이클 재기입 (지연 적재 보정)
FINISH_CYCLES = 6     # 자정 전환 후 전날 파일 마무리 사이클 수
MSG = 'UI-UNIT-PORT-MAXCAPACITY-UPDATE'

SQL_TX = """
SELECT /*+ INDEX(NT_L_LOGMESSAGE NT_L_LOGMESSAGE_IX2) */
       TRANSACTIONID, TIME, PROCESSNAME, PARTITIONID
  FROM NT_L_LOGMESSAGE
 WHERE COMMUNICATIONMESSAGENAME = :msg
   AND TIME >= TO_DATE(:dt_from, 'YYYY-MM-DD HH24:MI:SS')
   AND TIME <  TO_DATE(:dt_to,   'YYYY-MM-DD HH24:MI:SS')
 ORDER BY TIME
"""
SQL_DETAIL = """
SELECT TIME, OPERATIONNAME, MACHINENAME, UNITNAME, TEXT
  FROM NT_L_LOGMESSAGE
 WHERE TRANSACTIONID = :txid
   AND PARTITIONID = :pid
 ORDER BY TIME
"""
RE_CHANGED = re.compile(r"port\{([^}]+)\}\.maxCapacity was changed to \{(-?\d+)\}")


# ────────────────────────────────────────────────────────────
# Oracle 접속
# ────────────────────────────────────────────────────────────
def load_cfg(path):
    """mcs_config.ini(기본) 탐색 — CWD 우선, 없으면 스크립트 폴더. config.ini 도 허용."""
    cands = []
    if os.path.isabs(path):
        cands = [path]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        for name in (path, 'mcs_config.ini', 'config.ini'):
            cands += [os.path.join(os.getcwd(), name), os.path.join(here, name)]
    found = next((p for p in cands if os.path.exists(p)), None)
    if not found:
        raise SystemExit(f'❌ 접속설정 없음: {path} (mcs_config.ini 를 같은 폴더에 두세요)')
    cp = configparser.ConfigParser()
    cp.read(found, encoding='utf-8')
    o = cp['oracle']
    print(f'  [CFG] {found}')
    return {
        'hosts': [h.strip() for h in o.get('hosts', '').split(',') if h.strip()],
        'port': o.get('port', '1521').strip(),
        'service': o.get('service_name', '').strip(),
        'user': o.get('user', '').strip(),
        'pw': o.get('password', '').strip(),
        'failover': o.get('failover', 'on').strip(),
        'failover_type': o.get('failover_type', 'SELECT').strip(),
        'method': o.get('method', 'BASIC').strip(),
        'retries': o.get('retries', '5').strip(),
        'delay': o.get('delay', '5').strip(),
    }


def build_dsn(c):
    """RAC FAILOVER DSN — mcs_query.py 와 동일한 형식 (검증된 것)."""
    addr = ''.join(f'(ADDRESS=(PROTOCOL=TCP)(HOST={h})(PORT={c["port"]}))' for h in c['hosts'])
    return ('(DESCRIPTION='
            f'(FAILOVER={c["failover"]})'
            f'{addr}'
            f'(CONNECT_DATA=(SERVICE_NAME={c["service"]})'
            f'(FAILOVER_MODE=(TYPE={c["failover_type"]})(METHOD={c["method"]})'
            f'(RETRIES={c["retries"]})(DELAY={c["delay"]}))))')


def connect(c):
    """oracledb thin 모드 (Oracle Client 설치 불필요) — mcs_query.py 와 동일."""
    try:
        import oracledb as drv
    except ImportError:
        try:
            import cx_Oracle as drv
        except ImportError:
            raise SystemExit('❌ oracledb 필요:  pip install oracledb')
    return drv.connect(user=c['user'], password=c['pw'], dsn=build_dsn(c))


def _s(v):
    """LOB(CLOB) 는 반드시 read() — TEXT 컬럼이 LOB 로 오면 str() 은 내용이 아니다."""
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


def fetch_db(a, dt_from, dt_to, cache):
    """[dt_from, dt_to) 구간 조작을 조회해 cache[분키] 갱신. 실패 시 False."""
    try:
        conn = connect(a.cfg)
    except SystemExit:
        raise
    except Exception as e:
        print(f'  ⚠️ Oracle 접속 실패: {e}')
        return False
    try:
        cur = conn.cursor()
        cur.execute(SQL_TX, {'msg': MSG,
                             'dt_from': dt_from.strftime('%Y-%m-%d %H:%M:%S'),
                             'dt_to': dt_to.strftime('%Y-%m-%d %H:%M:%S')})
        cols = [d[0] for d in cur.description]
        ix = {c: i for i, c in enumerate(cols)}
        txs = [{'txid': _s(r[ix['TRANSACTIONID']]), 'time': r[ix['TIME']],
                'proc': _s(r[ix['PROCESSNAME']]), 'pid': _s(r[ix['PARTITIONID']])}
               for r in cur.fetchall() if _s(r[ix['TRANSACTIONID']])]

        nport = 0
        for t in txs:
            cur.execute(SQL_DETAIL, {'txid': t['txid'], 'pid': t['pid']})
            c2 = [d[0] for d in cur.description]
            j = {c: i for i, c in enumerate(c2)}
            seen = set()
            k = t['time'].strftime('%Y-%m-%d %H:%M')
            e = cache.setdefault(k, {'machine': [], 'ports': [], 'proc': [], 'tx': []})
            for r in cur.fetchall():
                if 'compareAndUpdatePortMaxCapacity' not in _s(r[j['OPERATIONNAME']]):
                    continue
                m = RE_CHANGED.search(_s(r[j['TEXT']]))
                if not m:
                    continue                       # 값 동일 = 변경 아님
                port, after = m.group(1), m.group(2)
                if port in seen:
                    continue
                seen.add(port)
                mach = _s(r[j['MACHINENAME']]) or port.split('_')[0]
                if mach and mach not in e['machine']:
                    e['machine'].append(mach)
                pair = f'{port}:{after}'
                if pair not in e['ports']:
                    e['ports'].append(pair)
                nport += 1
            if t['proc'] and t['proc'] not in e['proc']:
                e['proc'].append(t['proc'])
            if t['txid'] not in e['tx']:
                e['tx'].append(t['txid'])
            if not e['ports']:                     # 실제 변경 없던 조작이면 비움
                cache.pop(k, None)
        cur.close()
        print(f'  [DB] {dt_from:%m/%d %H:%M}~{dt_to:%H:%M} → 조작 {len(txs)}회 · 포트변경 {nport}건')
        return True
    except Exception as e:
        print(f'  ⚠️ 조회 오류: {e}')
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────
# CSV 원본 (--source csv)
# ────────────────────────────────────────────────────────────
def load_csv(path, cache):
    files = ([os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith('.csv')]
             if os.path.isdir(path) else ([path] if os.path.exists(path) else []))
    if not files:
        print(f'  ⚠️ MAXCAPA CSV 없음: {os.path.abspath(path)}')
        return False
    n = 0
    for fp in files:
        with open(fp, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                dt = parse_dt(r.get('조작시각'))
                if not dt:
                    continue
                e = cache.setdefault(dt.strftime('%Y-%m-%d %H:%M'),
                                     {'machine': [], 'ports': [], 'proc': [], 'tx': []})
                for key, col in (('machine', 'MACHINE'), ('proc', 'PROCESS'), ('tx', 'TRANSACTIONID')):
                    v = (r.get(col) or '').strip()
                    if v and v not in e[key]:
                        e[key].append(v)
                port = (r.get('PORT') or '').strip()
                if port:
                    pair = f"{port}:{(r.get('후(after)') or '').strip()}"
                    if pair not in e['ports']:
                        e['ports'].append(pair)
                n += 1
    print(f'  [CSV] {len(files)}개 파일 · {n}행 → 조작 있는 분 {len(cache)}개')
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
    """1사이클: 파일 읽기 → 해당 분 조작 채우기 → 원자 교체. return 기입행수 or None(스킵)."""
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
        return any(r.get(c) is None for c in NEW_COLS)
    targets = [i for i, (r, t) in enumerate(zip(rows, times))
               if t and (force or unfilled(r) or (tmax - t) <= timedelta(minutes=RECHECK_MIN))]
    if not targets:
        return 0

    out_header = header + [c for c in NEW_COLS if c not in header]
    hit = 0
    for i in targets:
        e = cache.get(times[i].strftime('%Y-%m-%d %H:%M'))
        rows[i]['MACHINE'] = ','.join(e['machine']) if e else ''
        rows[i]['PORT:후(after)'] = '\n'.join(e['ports']) if e else ''
        rows[i]['PROCESS'] = ','.join(e['proc']) if e else ''
        rows[i]['TRANSACTIONID'] = ','.join(e['tx']) if e else ''
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
        print(f'  ✏️ 조작 있는 분 {hit}개 기입')
    return len(targets)


def refresh(a, cache, dt_from, dt_to):
    """원본 갱신 — DB 또는 CSV."""
    if a.source == 'csv':
        return load_csv(a.maxcapa, cache)
    return fetch_db(a, dt_from, dt_to, cache)


# ────────────────────────────────────────────────────────────
# 운영 루프
# ────────────────────────────────────────────────────────────
def _loop(a):
    print(f'[mcs_maxcapa] {a.interval}초 간격 · 대상 {a.event} · 원본 '
          + ('CSV ' + str(a.maxcapa) if a.source == 'csv' else f"Oracle {a.cfg['service']}"))
    cache, healed, finishing, cur = {}, set(), {}, None
    user_force = getattr(a, 'force', False)
    while True:
        try:
            fp = resolve_event(a.event)
            now = datetime.now()
            heal = bool(fp) and fp not in healed
            # 시작(또는 새 날짜 첫 대면)엔 그날 전체, 이후엔 최근 lookback 분만 조회
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
                print(f'[mcs_maxcapa {datetime.now():%H:%M:%S}] 기입 {n}행 (캐시 {len(cache)}분)')
            time.sleep(a.interval)
        except KeyboardInterrupt:
            print('\n[mcs_maxcapa] 종료.'); break
        except Exception as e:
            print(f'  ⚠️ [mcs_maxcapa] 오류(계속): {e}'); time.sleep(a.interval)


def run_watch(event='./predict_tobe', interval=60, lookback=20,
              config='mcs_config.ini', source='db', maxcapa='./maxcapa_v3.csv'):
    """run_ml 등에서 스레드로 돌리는 진입점."""
    a = argparse.Namespace(event=event, out=None, interval=interval, lookback=lookback,
                           source=source, maxcapa=maxcapa, force=False, config=config)
    a.cfg = load_cfg(config) if source == 'db' else None
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
        cache = {}
        m = re.search(r'(\d{8})', f)
        if m and a.source == 'db':
            d = datetime.strptime(m.group(1), '%Y%m%d')
            if not refresh(a, cache, d, d + timedelta(days=1)):
                fail += 1; print(f'  ❌ {f} — 조회 실패'); continue
        else:
            refresh(a, cache, datetime.now(), datetime.now())
        n = cycle(a, cache, fp=fp)
        if n is None:
            fail += 1; print(f'  ❌ {f} 실패')
        else:
            ok += 1; print(f'  ✅ {f} — {n}행 기입')
    print(f'🎉 백필 완료 — 성공 {ok} / 실패 {fail}')


def diagnose(a):
    """접속·조회 점검."""
    print(f"설정: {a.config}")
    print(f"  hosts   : {', '.join(a.cfg['hosts'])}:{a.cfg['port']}")
    print(f"  service : {a.cfg['service']}   user: {a.cfg['user']}")
    print(f"  DSN     : {build_dsn(a.cfg)[:110]}...")
    print()
    now = datetime.now()
    for label, (f0, f1) in (('오늘 00:00~현재', (now.replace(hour=0, minute=0, second=0, microsecond=0), now)),
                            ('어제 하루', (now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1),
                                        now.replace(hour=0, minute=0, second=0, microsecond=0)))):
        cache = {}
        ok = fetch_db(a, f0, f1, cache)
        print(f"  {'✅' if ok else '❌'} {label}: 조작 있는 분 {len(cache)}개")
        for k in sorted(cache)[:3]:
            e = cache[k]
            print(f"     {k}  {','.join(e['machine'])}  {' / '.join(e['ports'])}  {','.join(e['proc'])}")
    print('\n[해석]  접속 실패 → hosts/service/계정 확인 · 조회 0건 → 그 기간에 조작이 없던 것')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', default=None, help='발동이벤트.csv 또는 폴더(최신 날짜파일 자동)')
    ap.add_argument('--out', default=None, help='(테스트용) 원본 대신 여기에 저장')
    ap.add_argument('--loop', action='store_true', help='운영: interval초마다 반복')
    ap.add_argument('--alldays', action='store_true', help='폴더 내 모든 날짜 파일 일괄 기입(덮어쓰기)')
    ap.add_argument('--force', action='store_true', help='단일 파일도 전체 덮어쓰기')
    ap.add_argument('--test', action='store_true', help='접속·조회 점검 (--event 불필요)')
    ap.add_argument('--interval', type=int, default=60)
    ap.add_argument('--lookback', type=int, default=20, help='매 사이클 재조회할 최근 분 (기본 20)')
    ap.add_argument('--config', default='mcs_config.ini')
    ap.add_argument('--source', choices=['db', 'csv'], default='db')
    ap.add_argument('--maxcapa', default='./maxcapa_v3.csv', help='--source csv 일 때 원본')
    a = ap.parse_args()
    a.cfg = load_cfg(a.config) if a.source == 'db' else None

    print('=' * 60)
    print('발동이벤트 ← MAXCAPA 조작내역 4컬럼 기입'
          + (' (접속점검)' if a.test else ' (운영 루프)' if a.loop
             else ' (과거 일괄백필)' if a.alldays else ' (1회)'))
    print('=' * 60)

    if a.test:
        diagnose(a); return
    if not a.event:
        print('❌ --event 가 필요합니다 (점검은 --test)'); sys.exit(2)
    if a.loop:
        _loop(a)
    elif a.alldays:
        backfill_alldays(a)
    else:
        cache = {}
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
