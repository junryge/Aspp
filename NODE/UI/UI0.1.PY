"""
로그프레소 쿼리 - 절대시간 방식
(상대시간 1h, 1m 안됨 → from/to 사용)
"""

import requests
import urllib.parse
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta

requests.packages.urllib3.disable_warnings()

HOST = "10.40.42.27"
PORT = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"


def get_time_range(hours_ago=1):
    """현재 시간 기준 from/to 계산"""
    now = datetime.now()
    from_time = now - timedelta(hours=hours_ago)
    
    from_str = from_time.strftime("%Y%m%d%H%M")
    to_str = now.strftime("%Y%m%d%H%M")
    
    return from_str, to_str


def test_query(query, timeout=30):
    """쿼리 테스트"""
    query_clean = ' '.join(query.split())
    encoded = urllib.parse.quote(query_clean, safe='')
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"
    
    print(f"\n쿼리: {query_clean[:100]}...")
    
    try:
        resp = requests.get(url, verify=False, timeout=timeout)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            if resp.text.strip() and not resp.text.startswith('<!'):
                df = pd.read_csv(StringIO(resp.text))
                print(f"결과: {len(df)} rows, {len(df.columns)} cols")
                if len(df) > 0:
                    print(df.head(5))
                return df
            else:
                print("결과: 빈 데이터")
        else:
            print("에러: 500 (쿼리 문법 오류)")
        return None
        
    except Exception as e:
        print(f"에러: {e}")
        return None


if __name__ == "__main__":
    
    print("=" * 60)
    print("로그프레소 절대시간 테스트")
    print("=" * 60)
    
    # 시간 범위 계산
    from_1h, to_now = get_time_range(hours_ago=1)
    from_24h, _ = get_time_range(hours_ago=24)
    from_7d, _ = get_time_range(hours_ago=24*7)
    
    print(f"\n현재 시간: {datetime.now()}")
    print(f"1시간 전: {from_1h}")
    print(f"24시간 전: {from_24h}")
    print(f"7일 전: {from_7d}")
    
    # 테스트 1: ts_current_job 최근 1시간
    print("\n" + "=" * 60)
    print("[테스트 1] ts_current_job 최근 1시간")
    test_query(f"table from={from_1h} to={to_now} ts_current_job | limit 5")
    
    # 테스트 2: ts_current_job 최근 24시간
    print("\n" + "=" * 60)
    print("[테스트 2] ts_current_job 최근 24시간")
    test_query(f"table from={from_24h} to={to_now} ts_current_job | limit 5")
    
    # 테스트 3: ts_current_job 최근 7일
    print("\n" + "=" * 60)
    print("[테스트 3] ts_current_job 최근 7일")
    test_query(f"table from={from_7d} to={to_now} ts_current_job | limit 5")
    
    # 테스트 4: star_transport_view 최근 24시간
    print("\n" + "=" * 60)
    print("[테스트 4] star_transport_view 최근 24시간")
    test_query(f"table from={from_24h} to={to_now} star_transport_view | limit 5")
    
    # 테스트 5: 전체 테이블 목록 다시 확인
    print("\n" + "=" * 60)
    print("[테스트 5] 전체 테이블 목록")
    df = test_query("logdb tables | limit 20")
    if df is not None:
        print("\n테이블 이름들:")
        print(df['table'].tolist() if 'table' in df.columns else df)
    
    # 테스트 6: ts로 시작하는 테이블 (다른 검색 방식)
    print("\n" + "=" * 60)
    print("[테스트 6] 테이블명에 'current' 포함")
    test_query("logdb tables | search table == '*current*'")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")