"""
로그프레소 쿼리 디버깅 테스트
"""

import requests
import urllib.parse
import pandas as pd
from io import StringIO

requests.packages.urllib3.disable_warnings()

HOST = "10.40.42.27"
PORT = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"


def test_query(query, timeout=30):
    """쿼리 테스트"""
    query_clean = ' '.join(query.split())
    encoded = urllib.parse.quote(query_clean, safe='')
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"
    
    print(f"\n쿼리: {query_clean[:80]}...")
    
    try:
        resp = requests.get(url, verify=False, timeout=timeout)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            print(f"결과:\n{resp.text[:800]}")
            if resp.text.strip() and not resp.text.startswith('<!'):
                df = pd.read_csv(StringIO(resp.text))
                print(f"\nDataFrame: {len(df)} rows")
                return df
        else:
            # 에러 메시지에서 핵심만 추출
            if "Internal Server Error" in resp.text:
                print("에러: 500 Internal Server Error (쿼리 문법 오류 가능성)")
            else:
                print(f"에러: {resp.text[:200]}")
        return None
        
    except Exception as e:
        print(f"에러: {e}")
        return None


if __name__ == "__main__":
    
    print("=" * 60)
    print("로그프레소 쿼리 디버깅 테스트")
    print("=" * 60)
    
    # 테스트 1: 테이블 목록에서 ts_ 관련 찾기
    print("\n[테스트 1] ts_ 테이블 목록")
    test_query("logdb tables | search table == '*ts_*'")
    
    # 테스트 2: ts_current_job 테이블 기본 조회
    print("\n" + "=" * 60)
    print("[테스트 2] ts_current_job 최근 1분")
    test_query("table ts_current_job 1m | limit 3")
    
    # 테스트 3: ts_current_job 최근 1시간
    print("\n" + "=" * 60)
    print("[테스트 3] ts_current_job 최근 1시간")
    test_query("table ts_current_job 1h | limit 3")
    
    # 테스트 4: star_transport_view 테이블
    print("\n" + "=" * 60)
    print("[테스트 4] star_transport_view")
    test_query("table star_transport_view 1m | limit 3")
    
    # 테스트 5: 날짜 직접 지정 (변수 없이)
    print("\n" + "=" * 60)
    print("[테스트 5] 날짜 직접 지정")
    test_query("table from=202501130000 to=202501131000 ts_current_job | limit 3")
    
    # 테스트 6: FAB 필터
    print("\n" + "=" * 60)
    print("[테스트 6] FAB=M14 필터")
    test_query('table ts_current_job 1h | search FAB == "M14" | limit 3')
    
    # 테스트 7: 컬럼 확인
    print("\n" + "=" * 60)
    print("[테스트 7] ts_current_job 컬럼 확인")
    test_query("table ts_current_job 1h | limit 1")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")