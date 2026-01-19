# -*- coding: utf-8 -*-
"""
로그프레소 응답 디버깅 - 실제 응답 내용 확인
"""

import requests
import urllib.parse
import warnings

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

HOST = "10.40.42.27"
PORT = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"

# 오늘 날짜로 테스트
from datetime import datetime
date_str = datetime.now().strftime("%Y%m%d")
from_time = f"{date_str}0000"
to_time = f"{date_str}2359"

query = f'''
table from={from_time} to={to_time} ts_current_job
| search FAB == "M14"
| eval A = case(trim(DESTMACHINENAME) == "4ABL_M10", 1, 0)
| eval B = case(substr(trim(SOURCEMACHINENAME), 0, 7) == "4ABL330", 1, 0)
| eval C = case(substr(trim(DESTMACHINENAME), 0, 4) == "4ALF", 1, 0)
| eval D = case(substr(trim(SOURCEMACHINENAME), 0, 4) == "4ALF", 1, 0)
| eval E = case(substr(trim(DESTMACHINENAME), 0, 4) == "4AFC", 1, 0)
| eval F = case(substr(trim(SOURCEMACHINENAME), 0, 4) == "4AFC", 1, 0)
| stats sum(A), sum(B), sum(C), sum(D), sum(E), sum(F), count by CURRTIME
| rename count as TOTALCNT, sum(A) as M14AM10A, sum(B) as M10AM14A, sum(C) as M14AM14B, sum(D) as M14BM14A, sum(E) as M14AM16, sum(F) as M16M14A
| sort CURRTIME
| limit 5
'''

print("=" * 60)
print(f"로그프레소 응답 디버깅 - {date_str}")
print("=" * 60)

query_clean = ' '.join(query.split())
encoded = urllib.parse.quote(query_clean, safe='')
url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"

print(f"\n[요청 중...]")

try:
    resp = requests.get(url, verify=False, timeout=60)
    
    print(f"\n[Status Code] {resp.status_code}")
    print(f"[Content-Length] {len(resp.text)} bytes")
    
    print(f"\n[응답 전체 (처음 1000자)]")
    print("-" * 40)
    print(repr(resp.text[:1000]))  # repr로 특수문자 확인
    print("-" * 40)
    
    # 조건 체크
    print(f"\n[조건 체크]")
    print(f"  1. resp.text 존재? {bool(resp.text)}")
    print(f"  2. resp.text.strip() 존재? {bool(resp.text.strip())}")
    print(f"  3. '<!' 로 시작? {resp.text.startswith('<!') if resp.text else 'N/A'}")
    print(f"  4. 첫 10글자: {repr(resp.text[:10]) if resp.text else 'N/A'}")
    
    # 현재 코드의 조건
    if resp.status_code == 200 and resp.text.strip() and not resp.text.startswith('<!'):
        print("\n✅ 조건 통과 - CSV 파싱 가능해야 함")
        
        # CSV 파싱 시도
        from io import StringIO
        import pandas as pd
        try:
            df = pd.read_csv(StringIO(resp.text))
            print(f"✅ CSV 파싱 성공: {len(df)}행")
            print(df.head())
        except Exception as e:
            print(f"❌ CSV 파싱 실패: {e}")
    else:
        print("\n❌ 조건 실패 - 이게 문제!")
        if not resp.text:
            print("   → 응답이 None")
        elif not resp.text.strip():
            print("   → 응답이 빈 문자열 또는 공백만")
        elif resp.text.startswith('<!'):
            print("   → HTML 에러 페이지")

except Exception as e:
    print(f"\n[예외 발생] {e}")
    import traceback
    traceback.print_exc()