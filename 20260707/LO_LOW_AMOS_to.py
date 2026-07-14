#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LO_LOW_AMOS — 로그프레소 조회 → 발동이벤트.csv 에 4컬럼 직접 기입 (운영용)
====================================================================
별도 병합파일 안 만들고 발동이벤트.csv 그 자체에 컬럼을 추가/기입한다.
매분 발동이벤트에 새 행이 붙으면 → 그 분만 로그프레소 조회해서 채움 (이미 채운 행은 유지).

조회 (MCP_NM=="BR"):
  table from=... to=... ATLAS_BOTTLENECK_ANOMALY | search MCP_NM == "BR" | sort _time
  table from=... to=... ATLAS_QUEUE_ANOMALY      | search MCP_NM == "BR" | sort _time

추가 4컬럼 (시간정렬: 발동이벤트 datetime T ← 로그프레소 EVENT_DT T-1분):
  BOTTLENECK_downward_anomaly_cols, BOTTLENECK_upward_anomaly_cols
  QUEUE_downward_anomaly_cols,      QUEUE_upward_anomaly_cols

실행 (pip: requests 만):
  운영(1분 루프):  python LO_LOW_AMOS.py --event .\predict_tobe\발동이벤트.csv --loop
  1회만:           python LO_LOW_AMOS.py --event .\predict_tobe\20260713_발동이벤트.csv
  테스트(원본보존): 위에 --out .\테스트.csv 추가
  옵션: --lag 1 · --interval 60 · --host/--port/--apikey

run_ml 통합 (스레드):
  import LO_LOW_AMOS
  threading.Thread(target=LO_LOW_AMOS.run_watch, daemon=True).start()
  # 경로 다르면: threading.Thread(target=LO_LOW_AMOS.run_watch,
  #                kwargs={'event': r'.\predict_tobe\발동이벤트.csv'}, daemon=True).start()

동작 원리:
  · 처음 실행: 파일 전체(안 채워진 행 전부) 범위를 한 번에 조회해서 백필
  · 루프 중: 새 행 + 최근 5분만 재조회(로그프레소 늦게 쓰인 분 자동 보정) → 쿼리 가볍다
  · 파일에 4컬럼 없으면 헤더에 추가, 있으면 그대로 이어서 기입
  · 조회 실패(서버 불안정)면 그 사이클은 파일 안 건드리고 다음 분에 재시도
  · 저장은 임시파일 → 원자 교체(os.replace), 교체 직전 파일 변경 감지되면 스킵 후 재시도
"""
import argparse, csv, os, sys, time, urllib.parse
from datetime import datetime, timedelta
from io import StringIO

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 로그프레소 접속 (logpresso_client.py 와 동일 — 검증된 직접 HTTP 방식)
HOST = '10.40.42.27'
PORT = 8888
API_KEY = 'db1d2335-49cf-e859-3519-1ca132922e38'

TABLES = {'BOTTLENECK': 'ATLAS_BOTTLENECK_ANOMALY', 'QUEUE': 'ATLAS_QUEUE_ANOMALY'}
NEW_COLS = ['BOTTLENECK_downward_anomaly_cols', 'BOTTLENECK_upward_anomaly_cols',
            'QUEUE_downward_anomaly_cols', 'QUEUE_upward_anomaly_cols']
RECHECK_MIN = 5  # 최근 N분은 매 사이클 재조회 (로그프레소 지연 기입 보정)


def query_logpresso(query, a, timeout=180):
    """LPQL 실행 → CSV 텍스트 (재시도 3회). 실패 시 None."""
    clean_q = ' '.join(query.split())
    url = (f'http://{a.host}:{a.port}/logpresso/httpexport/query.csv'
           f'?_apikey={a.apikey}&_q={urllib.parse.quote(clean_q, safe="")}')
    for attempt in range(3):
        try:
            resp = requests.get(url, verify=False, timeout=timeout)
            if resp.status_code == 200 and resp.text.strip() and not resp.text.strip().startswith('<!'):
                return resp.text
            if resp.status_code >= 500 and attempt < 2:
                time.sleep(2 * (attempt + 1)); continue
            print(f'  ⚠️ Logpresso HTTP {resp.status_code}: {resp.text[:150]!r}')
            return None
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError):
            if attempt < 2:
                time.sleep(2 * (attempt + 1)); continue
            print('  ⚠️ Logpresso 연결 실패')
            return None
        except Exception as e:
            print(f'  ⚠️ Logpresso 예외: {e}')
            return None
    return None


def fetch_range(dt_from, dt_to, a, cache):
    """두 테이블을 [dt_from, dt_to] 조회 → cache[pfx][분키]=(down,up) 갱신. 실패 시 False."""
    for pfx, tbl in TABLES.items():
        lpql = (f'table from={dt_from:%Y%m%d%H%M}00 to={dt_to:%Y%m%d%H%M}59 {tbl} '
                f'| search MCP_NM == "BR" | sort _time')
        text = query_logpresso(lpql, a)
        if text is None:
            return False
        n = 0
        for r in csv.DictReader(StringIO(text)):
            k = (r.get('EVENT_DT') or '').strip()[:16]  # 초 버리고 분단위
            if k:
                cache[pfx][k] = ((r.get('downward_anomaly_cols') or '').strip(),
                                 (r.get('upward_anomaly_cols') or '').strip())
                n += 1
        print(f'  [{tbl}] {dt_from:%H:%M}~{dt_to:%H:%M} → {n}분 수신')
    return True


def read_event(fp):
    with open(fp, encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        header = list(rd.fieldnames or [])
        rows = list(rd)
    return header, rows


def parse_dt(s):
    try:
        return datetime.strptime((s or '').strip()[:16], '%Y-%m-%d %H:%M')
    except ValueError:
        return None


def cycle(a, cache):
    """1사이클: 파일 읽기 → 필요한 분 조회 → 채워서 원자 교체. return 기입행수 or None(스킵)."""
    if not os.path.exists(a.event):
        print(f'  ⚠️ 파일 없음: {a.event} (대기)'); return None
    stat0 = os.stat(a.event)
    header, rows = read_event(a.event)
    if 'datetime' not in header:
        print("  ❌ 'datetime' 컬럼 없음"); return None

    times = [parse_dt(r.get('datetime')) for r in rows]
    valid = [t for t in times if t]
    if not valid:
        return None
    tmax = max(valid)

    # 채울 대상: 4컬럼이 물리적으로 없는 행(None) + 최근 RECHECK_MIN분(지연기입 보정)
    def unfilled(r):
        return any(r.get(c) is None for c in NEW_COLS)
    targets = [i for i, (r, t) in enumerate(zip(rows, times))
               if t and (unfilled(r) or (tmax - t) <= timedelta(minutes=RECHECK_MIN))]
    if not targets:
        return 0

    # 조회 필요한 키 = 대상 행의 (T-lag) 중 캐시에 없는 것 + 최근분(항상 갱신)
    keys = {times[i] - timedelta(minutes=a.lag) for i in targets}
    need = {k for k in keys
            if any(k.strftime('%Y-%m-%d %H:%M') not in cache[p] for p in TABLES)
            or (tmax - k) <= timedelta(minutes=RECHECK_MIN)}
    if need:
        if not fetch_range(min(need), max(need), a, cache):
            print('  ⚠️ 조회 실패 → 이번 사이클 기입 생략 (다음에 재시도)')
            return None

    # 기입
    out_header = header + [c for c in NEW_COLS if c not in header]
    for i in targets:
        k = (times[i] - timedelta(minutes=a.lag)).strftime('%Y-%m-%d %H:%M')
        for pfx in TABLES:
            v = cache[pfx].get(k)
            rows[i][f'{pfx}_downward_anomaly_cols'] = v[0] if v else ''
            rows[i][f'{pfx}_upward_anomaly_cols'] = v[1] if v else ''
    # 대상 아닌 행의 None(이론상 없음)도 '' 보정
    for r in rows:
        for c in NEW_COLS:
            if r.get(c) is None:
                r[c] = ''

    # 원자 저장 (교체 직전 생성기가 파일 바꿨으면 스킵 → 다음 사이클 재시도)
    out = a.out or a.event
    tmp = out + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=out_header)
        w.writeheader(); w.writerows(rows)
    if out == a.event:
        stat1 = os.stat(a.event)
        if (stat1.st_mtime_ns, stat1.st_size) != (stat0.st_mtime_ns, stat0.st_size):
            os.remove(tmp)
            print('  ⚠️ 기입 중 파일 변경 감지 → 스킵 (다음 사이클 재시도)')
            return None
    try:
        os.replace(tmp, out)
    except PermissionError:
        os.remove(tmp)
        print('  ⚠️ 파일 잠김(생성기 사용 중) → 스킵 (다음 사이클 재시도)')
        return None
    return len(targets)


def _loop(a):
    """운영 루프 본체 (main --loop 와 run_watch 공용)."""
    print(f'[LO_LOW_AMOS] {a.interval}초 간격 · 대상: {a.event}')
    cache = {p: {} for p in TABLES}  # 분키 → (down, up)
    while True:
        try:
            n = cycle(a, cache)
            if n is not None:
                print(f"[LO_LOW_AMOS {datetime.now():%H:%M:%S}] 기입 {n}행 (캐시 {len(cache['BOTTLENECK'])}분)")
            time.sleep(a.interval)
        except KeyboardInterrupt:
            print('\n[LO_LOW_AMOS] 종료.'); break
        except Exception as e:
            print(f'  ⚠️ [LO_LOW_AMOS] 오류(계속): {e}'); time.sleep(a.interval)


def run_watch(event='./predict_tobe/발동이벤트.csv', interval=60, lag=1,
              host=HOST, port=PORT, apikey=API_KEY):
    """run_ml 등에서 스레드로 돌리는 진입점:
        threading.Thread(target=LO_LOW_AMOS.run_watch, daemon=True).start()
    """
    a = argparse.Namespace(event=event, out=None, lag=lag, loop=True,
                           interval=interval, host=host, port=port, apikey=apikey)
    _loop(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', required=True, help='발동이벤트.csv (여기에 직접 기입)')
    ap.add_argument('--out', default=None, help='(테스트용) 지정하면 원본 대신 여기에 저장')
    ap.add_argument('--lag', type=int, default=1, help='몇 분 전 로그프레소를 기입할지 (기본 1)')
    ap.add_argument('--loop', action='store_true', help='운영: interval초마다 반복')
    ap.add_argument('--interval', type=int, default=60)
    ap.add_argument('--host', default=HOST)
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--apikey', default=API_KEY)
    a = ap.parse_args()

    print('=' * 60)
    print('발동이벤트 ← 로그프레소 이상감지 4컬럼 기입' + (' (운영 루프)' if a.loop else ' (1회)'))
    print('=' * 60)

    if a.loop:
        _loop(a)
    else:
        cache = {p: {} for p in TABLES}
        n = cycle(a, cache)
        if n is None:
            sys.exit(2)
        print(f'🎉 완료 — {n}행 기입 → {a.out or a.event}')


if __name__ == '__main__':
    main()
