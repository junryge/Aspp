# -*- coding: utf-8 -*-
"""
로그프레소 실제 알람 데이터 조회 테스트
"""

import requests
import urllib.parse
import pandas as pd
from io import StringIO

requests.packages.urllib3.disable_warnings()

# 로그프레소 설정
HOST = "10.40.42.27"
PORT = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"


def query_logpresso(query, timeout=180):
    """로그프레소 쿼리 실행"""
    query_clean = ' '.join(query.split())
    encoded = urllib.parse.quote(query_clean, safe='')
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"
    
    print(f"[쿼리 실행]")
    print(f"URL: {url[:100]}...")
    
    try:
        resp = requests.get(url, verify=False, timeout=timeout)
        
        print(f"[응답] Status: {resp.status_code}")
        
        if resp.status_code == 200 and resp.text.strip() and not resp.text.startswith('<!'):
            df = pd.read_csv(StringIO(resp.text))
            return df
        else:
            print(f"[에러] 응답 내용: {resp.text[:500]}")
            return None
            
    except Exception as e:
        print(f"[예외] {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("로그프레소 알람 데이터 조회 테스트")
    print("=" * 60)
    
    # 알람 쿼리
    query = '''
    table from=20260102070000 to=20260105070000 test_currentjob_predict
    | eval LSTM_FCAST_TM = string(dateadd(date(TIME,"yyyyMMddHHmm"),"min",1),"yyyyMMddHHmm")
    | rename TIME as MEAS_TM
    | eval ALARM_DESC = case(isnull(ALARM_DESC),"",ALARM_DESC),ALARM_YN = case(isnull(ALARM_YN),"N",ALARM_YN)
    | search ALARM_YN == "Y" and ALARM_DESC == "*반송 큐 개수 다량 증가*"
    | fields MEAS_TM, LSTM_FCAST_TM, ALARM_DESC, ALARM_YN
    '''
    
    print(f"\n[쿼리]\n{query}")
    print("\n" + "-" * 60)
    
    df = query_logpresso(query)
    
    if df is not None and len(df) > 0:
        print(f"\n✅ 성공! {len(df)}개 행 조회됨")
        print("\n[컬럼]")
        print(df.columns.tolist())
        print("\n[데이터 미리보기]")
        print(df.head(20).to_string())
        print("\n[전체 데이터]")
        print(df.to_string())
    elif df is not None:
        print(f"\n⚠️ 조회 성공, 하지만 데이터 0개")
    else:
        print(f"\n❌ 조회 실패")