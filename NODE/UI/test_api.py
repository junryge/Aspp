# -*- coding: utf-8 -*-
"""
로그프레소 응답 디버깅 - 완전 독립 스크립트
"""

import requests
import urllib.parse
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

print("=" * 60)
print("로그프레소 응답 디버깅")
print("=" * 60)

HOST = "10.40.42.27"
PORT = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"

# 오늘 날짜
date_str = datetime.now().strftime("%Y%m%d")
from_time = f"{date_str}0000"
to_time = f"{date_str}2359"

print(f"날짜: {date_str}")
print(f"조회: {from_time} ~ {to_time}")

query = f'''
table from={from_time} to={to_time} ts_current_job
| search FAB == "M14"
| stats count by CURRTIME
| sort CURRTIME
| limit 5
'''

query_clean = ' '.join(query.split())
encoded = urllib.parse.quote(query_clean, safe='')
url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"

print(f"\n요청 중...")

try:
    resp = requests.get(url, verify=False, timeout=60)
    
    print(f"\n[결과]")
    print(f"  Status: {resp.status_code}")
    print(f"  Length: {len(resp.text)} bytes")
    
    print(f"\n[응답 내용]")
    print("-" * 40)
    if resp.text:
        print(resp.text[:500])
    else:
        print("(응답 없음)")
    print("-" * 40)
    
    # 체크
    print(f"\n[체크]")
    print(f"  text 있음: {bool(resp.text)}")
    print(f"  strip 있음: {bool(resp.text.strip()) if resp.text else False}")
    print(f"  '<!'로 시작: {resp.text.startswith('<!') if resp.text else False}")
    
except Exception as e:
    print(f"에러: {e}")