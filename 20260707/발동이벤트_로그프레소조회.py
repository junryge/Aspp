#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
발동이벤트_로그프레소조회 — 로그프레소 직접 조회 → 발동이벤트.csv에 4컬럼 기입
====================================================================
한 파일로 끝: 로그프레소 HTTP API 조회 + 병합 (export CSV 필요 없음)

조회 (발동이벤트 날짜 범위 자동, MCP_NM=="BR"):
  table from=... to=... ATLAS_BOTTLENECK_ANOMALY | search MCP_NM == "BR" | sort _time
  table from=... to=... ATLAS_QUEUE_ANOMALY      | search MCP_NM == "BR" | sort _time

추가 4컬럼 (시간정렬: 발동이벤트 datetime T ← 로그프레소 EVENT_DT T-1분):
  BOTTLENECK_downward_anomaly_cols, BOTTLENECK_upward_anomaly_cols
  QUEUE_downward_anomaly_cols,      QUEUE_upward_anomaly_cols

실행 (pip: requests 만):
  python 발동이벤트_로그프레소조회.py --event .\predict_tobe\20260713_발동이벤트.csv
  → 같은 폴더에 20260713_발동이벤트_병합.csv 생성
  옵션: --out 출력경로 (원본경로 주면 덮어쓰기) · --lag 1 · --host/--port/--apikey
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


def query_logpresso(query, host, port, apikey, timeout=180):
    """LPQL 실행 → CSV 텍스트 (재시도 3회). 실패 시 None."""
    clean_q = ' '.join(query.split())
    url = (f'http://{host}:{port}/logpresso/httpexport/query.csv'
           f'?_apikey={apikey}&_q={urllib.parse.quote(clean_q, safe="")}')
    print(f'[Logpresso] ▶ {clean_q}')
    for attempt in range(3):
        try:
            resp = requests.get(url, verify=False, timeout=timeout)
            if resp.status_code == 200 and resp.text.strip() and not resp.text.strip().startswith('<!'):
                return resp.text
            if resp.status_code >= 500 and attempt < 2:
                wait = 2 * (attempt + 1)
                print(f'[Logpresso] ⚠️ HTTP {resp.status_code} → {wait}초 후 재시도...')
                time.sleep(wait); continue
            print(f'[Logpresso] ❌ HTTP {resp.status_code}: {resp.text[:200]!r}')
            return None
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < 2:
                wait = 2 * (attempt + 1)
                print(f'[Logpresso] ⚠️ 연결 실패 → {wait}초 후 재시도...')
                time.sleep(wait); continue
            print(f'[Logpresso] ❌ 연결 실패: {e}')
            return None
        except Exception as e:
            print(f'[Logpresso] ❌ 예외: {e}')
            return None
    return None


def fetch_anomaly(table, dt_from, dt_to, a):
    """테이블 조회 → {EVENT_DT 분단위 'YYYY-MM-DD HH:MM': (down, up)}"""
    lpql = (f'table from={dt_from:%Y%m%d%H%M%S} to={dt_to:%Y%m%d%H%M%S} {table} '
            f'| search MCP_NM == "BR" | sort _time')
    text = query_logpresso(lpql, a.host, a.port, a.apikey)
    if text is None:
        print(f'❌ {table} 조회 실패 — 접속/기간 확인'); sys.exit(2)
    m = {}
    for r in csv.DictReader(StringIO(text)):
        k = (r.get('EVENT_DT') or '').strip()[:16]  # 초 버리고 분단위
        if k:
            m[k] = ((r.get('downward_anomaly_cols') or '').strip(),
                    (r.get('upward_anomaly_cols') or '').strip())
    print(f'[{table}] {len(m)}분 수신 (이상감지 있는 분: {sum(1 for v in m.values() if v[0] or v[1])}개)')
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', required=True, help='발동이벤트.csv (datetime 컬럼 필요)')
    ap.add_argument('--out', default=None, help='출력 (기본: <입력>_병합.csv)')
    ap.add_argument('--lag', type=int, default=1, help='몇 분 전 로그프레소를 기입할지 (기본 1)')
    ap.add_argument('--host', default=HOST)
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--apikey', default=API_KEY)
    a = ap.parse_args()

    # ① 발동이벤트 로드 → 날짜 범위 산출
    with open(a.event, encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        header = list(rd.fieldnames)
        rows = list(rd)
    if 'datetime' not in header:
        print("❌ 'datetime' 컬럼 없음"); sys.exit(2)
    times = []
    for r in rows:
        try:
            times.append(datetime.strptime(r['datetime'].strip()[:16], '%Y-%m-%d %H:%M'))
        except ValueError:
            times.append(None)
    valid = [t for t in times if t]
    if not valid:
        print('❌ datetime 파싱 실패'); sys.exit(2)
    dt_from = min(valid) - timedelta(minutes=a.lag)   # 00:00행 채우게 lag분 앞부터
    dt_to = max(valid)
    print(f'[발동이벤트] {len(rows)}행 · {min(valid):%Y-%m-%d %H:%M} ~ {dt_to:%H:%M} '
          f'→ 로그프레소 조회구간 {dt_from:%Y-%m-%d %H:%M}~{dt_to:%H:%M}')

    # ② 로그프레소 2테이블 조회
    maps = {pfx: fetch_anomaly(tbl, dt_from, dt_to, a) for pfx, tbl in TABLES.items()}

    # ③ 병합 (T행 ← T-lag분)
    out_header = header + [c for c in NEW_COLS if c not in header]
    hit = {p: 0 for p in TABLES}
    for r, t in zip(rows, times):
        k = (t - timedelta(minutes=a.lag)).strftime('%Y-%m-%d %H:%M') if t else ''
        for pfx in TABLES:
            v = maps[pfx].get(k)
            r[f'{pfx}_downward_anomaly_cols'] = v[0] if v else ''
            r[f'{pfx}_upward_anomaly_cols'] = v[1] if v else ''
            hit[pfx] += v is not None

    out = a.out or (os.path.splitext(a.event)[0] + '_병합.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=out_header)
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    print(f'[병합] 컬럼 {len(header)}→{len(out_header)} · '
          f'BOTTLENECK 매칭 {hit["BOTTLENECK"]}/{n} · QUEUE 매칭 {hit["QUEUE"]}/{n} (미매칭=공란)')
    print(f'💾 → {out}')


if __name__ == '__main__':
    main()
