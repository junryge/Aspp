#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
logpresso_query.py — 로그프레소 OHT 조회 (시간 구간 → CSV DataFrame)

요청 시간 구간을 chunk_minutes 단위로 끊어 로그프레소 HTTP export API 호출.
응답 크기가 30MB 초과하면 해당 구간을 절반으로 재귀 분할.

쿼리 형식: 'remote icamcslogdt01' 로 감싸 원격 노드에서 조회.
  remote icamcslogdt01 [ table from=... to=... <table> | sort _time ]

출력 컬럼 (oht_data_m16br 기준, 25컬럼 parsed 포맷):
  _id, _table, _time, ADDRESS, CARRIER, DESTINATION, DEST_RETURN_PORT,
  DISTANCE, E/M, EDGE, ERROR_CODE, EXECUTE_CYCLE, FROM_RETURN_PORT,
  GROUP_ID, MCP, MSG_ID, NETWORK_CONDITION, NEXT_ADDRESS, OPERATION_STATUS,
  RETURN_PRIORITY, STATUS, STOCK_INFO, VEHICLE, VEHICLE_EXECUTE_CYCLE,
  VEHICLE_MILEAGE

→ 이 CSV가 derive_from_oht.py / main.py 입력으로 그대로 사용 가능.
"""

import requests
import urllib.parse
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ───────────────────────────────────────────────────────────
# 접속 설정 — 운영/개발 둘 중 하나만 활성화 (다른 건 주석 처리)
# ───────────────────────────────────────────────────────────

# [운영]
#HOST    = "10.40.42.27"
#PORT    = 8888
#API_KEY = "10f12ae0-5a80-55cd-7b15-e5554f0612f3"

# [개발]  http://10.125.173.63/
HOST    = "10.125.173.63"
PORT    = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"

# 원격 노드명 (쿼리에서 remote <NODE> [ ... ] 로 감쌀 때 사용)
# 비우면 remote 감싸지 않고 로컬 쿼리로 동작.
REMOTE_NODE = "icamcslogdt01"

FMT = "%Y%m%d%H%M%S"
MAX_BYTES = 30 * 1024 * 1024   # 30MB


def _build_query(from_dt: str, to_dt: str, table: str) -> str:
    """remote <NODE> [ ... ] 로 감싼 쿼리 문자열 생성.
       REMOTE_NODE 가 빈 문자열이면 감싸지 않고 원본 쿼리만 반환."""
    inner = f'table from={from_dt} to={to_dt} {table} | sort _time'
    if REMOTE_NODE:
        return f'remote {REMOTE_NODE} [ {inner} ]'
    return inner


def _fetch(from_dt: str, to_dt: str, table: str):
    """단일 구간 조회. (df, byte_size) 반환. 실패 시 예외."""
    q = _build_query(from_dt, to_dt, table)
    encoded = urllib.parse.quote(q, safe="")
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"

    # 디버그: 어떤 쿼리가 나가는지 콘솔에 표시 (오류 발생 시 진단용)
    print(f"  [Q] {q}")

    resp = requests.get(url, verify=False, timeout=300)
    if resp.status_code != 200:
        # Logpresso 가 HTML 페이지를 반환하는 경우(대시보드 응답) 가독성 위해 앞부분만 추림
        body = resp.text[:500]
        raise RuntimeError(
            f"HTTP {resp.status_code} from {HOST}:{PORT}\n"
            f"  실패 쿼리: {q}\n"
            f"  응답(앞 500자): {body}"
        )

    size = len(resp.content)
    df = (pd.read_csv(StringIO(resp.text), low_memory=False, dtype=str)
          if resp.text.strip() else pd.DataFrame())
    return df, size


def query_oht_chunked(from_dt: str, to_dt: str,
                      table: str = "oht_data_m16br",
                      chunk_minutes: int = 10) -> pd.DataFrame:
    """
    시간 구간을 chunk_minutes 단위로 끊어서 조회 → concat.
    조회 결과가 30MB 넘으면 해당 구간을 절반으로 재분할(재귀).

    Parameters
    ----------
    from_dt, to_dt : str   "yyyyMMddHHmmss"
    table          : str
    chunk_minutes  : int   기본 분할 단위(분)
    """
    start = datetime.strptime(from_dt, FMT)
    end   = datetime.strptime(to_dt, FMT)
    step  = timedelta(minutes=chunk_minutes)

    frames = []
    cur = start

    while cur < end:
        nxt = min(cur + step, end)
        f_s = cur.strftime(FMT)
        t_s = nxt.strftime(FMT)

        df, size = _fetch(f_s, t_s, table)

        # 30MB 초과 시 해당 구간만 절반으로 재분할
        if size > MAX_BYTES and (nxt - cur) > timedelta(seconds=1):
            mid = cur + (nxt - cur) / 2
            print(f"[SPLIT] {f_s}~{t_s} = {size/1024/1024:.1f}MB 초과 → 분할")
            sub = query_oht_chunked(f_s, mid.strftime(FMT), table, chunk_minutes)
            sub2 = query_oht_chunked(mid.strftime(FMT), t_s, table, chunk_minutes)
            frames.extend([sub, sub2])
        else:
            print(f"[OK] {f_s}~{t_s}  {len(df):>6}건  {size/1024/1024:5.1f}MB")
            if not df.empty:
                frames.append(df)

        cur = nxt

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    if "_time" in result.columns:
        result = result.sort_values("_time").reset_index(drop=True)
    print(f"[DONE] 총 {len(result)}건")
    return result


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[설정] HOST={HOST}:{PORT}  REMOTE={REMOTE_NODE}")
    df = query_oht_chunked(
        from_dt       = "20260621000000",
        to_dt         = "20260621010101",
        table         = "oht_data_m16br",
        chunk_minutes = 10,
    )
    print(df)
