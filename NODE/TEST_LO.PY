"""
로그프레소 API 간단 테스트
"""

import requests
import urllib.parse
import pandas as pd
from io import StringIO

requests.packages.urllib3.disable_warnings()

# 설정
HOST = "10.40.42.27"
PORT = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"

def test_query(query):
    """API Key로 쿼리 테스트"""
    encoded = urllib.parse.quote(query, safe='')
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"
    
    print(f"URL: {url[:80]}...")
    
    try:
        resp = requests.get(url, verify=False, timeout=30)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            print(f"\n결과:\n{resp.text[:500]}")
            
            # DataFrame 변환
            if resp.text.strip():
                df = pd.read_csv(StringIO(resp.text))
                print(f"\nDataFrame: {len(df)} rows")
                print(df.head(10))
                return df
        else:
            print(f"Error: {resp.text}")
            
    except Exception as e:
        print(f"에러: {e}")
    
    return None


if __name__ == "__main__":
    
    # 테스트 1: 테이블 목록 조회
    print("=" * 60)
    print("테스트 1: 테이블 목록")
    print("=" * 60)
    test_query("logdb tables")
    
    # 테스트 2: 시스템 정보
    print("\n" + "=" * 60)
    print("테스트 2: 시스템 테이블")
    print("=" * 60)
    test_query("system tables | limit 10")
    
    # 테스트 3: AWS_IDC_DATA_HIS 조회 (V7용)
    print("\n" + "=" * 60)
    print("테스트 3: IDC 데이터 (최근 10분)")
    print("=" * 60)
    test_query("""
        table AWS_IDC_DATA_HIS 10m
        | search IDC_COL == "CURRENT_M16A_3F_JOB_2"
        | limit 5
    """)