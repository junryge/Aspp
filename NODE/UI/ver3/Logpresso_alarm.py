# -*- coding: utf-8 -*-
"""
================================================================================
로그프레소 알람 조회 모듈
- test_currentjob_predict 테이블에서 실제 알람 조회
- main.py에서 import해서 사용
================================================================================
"""

import requests
import urllib.parse
import pandas as pd
from io import StringIO
import warnings

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

# ============================================================================
# 로그프레소 설정
# ============================================================================
HOST = "10.40.42.27"
PORT = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"


def query_logpresso(query, timeout=180):
    """로그프레소 쿼리 실행"""
    query_clean = ' '.join(query.split())
    encoded = urllib.parse.quote(query_clean, safe='')
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"
    
    try:
        resp = requests.get(url, verify=False, timeout=timeout)
        
        if resp.status_code == 200 and resp.text.strip() and not resp.text.startswith('<!'):
            df = pd.read_csv(StringIO(resp.text))
            return df
        else:
            print(f"[로그프레소 알람] 쿼리 에러: Status {resp.status_code}")
            return None
            
    except Exception as e:
        print(f"[로그프레소 알람] 쿼리 예외: {e}")
        return None


def get_alarm_data(from_time, to_time):
    """
    로그프레소 알람 데이터 조회
    
    Args:
        from_time: 시작 시간 (YYYYMMDDHHMM00 형식)
        to_time: 종료 시간 (YYYYMMDDHHMM00 형식)
    
    Returns:
        list: 알람 데이터 리스트
            - MEAS_TM: 측정 시간
            - LSTM_FCAST_TM: 예측 시간 (+1분)
            - ALARM_DESC: 알람 설명
            - ALARM_YN: 알람 여부 (Y)
    """
    print(f"[로그프레소 알람] 조회: {from_time} ~ {to_time}")
    
    query = f'''
    table from={from_time} to={to_time} test_currentjob_predict
    | eval LSTM_FCAST_TM = string(dateadd(date(TIME,"yyyyMMddHHmm"),"min",1),"yyyyMMddHHmm")
    | rename TIME as MEAS_TM
    | eval ALARM_DESC = case(isnull(ALARM_DESC),"",ALARM_DESC),ALARM_YN = case(isnull(ALARM_YN),"N",ALARM_YN)
    | search ALARM_YN == "Y" and ALARM_DESC == "*반송 큐 개수 다량 증가*"
    | fields MEAS_TM, LSTM_FCAST_TM, ALARM_DESC, ALARM_YN
    '''
    
    df = query_logpresso(query)
    
    if df is not None and len(df) > 0:
        print(f"  → {len(df)}개 알람 조회됨")
        return df.to_dict('records')
    elif df is not None:
        print(f"  → 알람 없음 (0개)")
        return []
    else:
        print(f"  → 조회 실패")
        return []


def get_alarm_map(from_time, to_time):
    """
    알람 데이터를 MEAS_TM 기준 맵으로 반환
    
    Returns:
        dict: {MEAS_TM: {LSTM_FCAST_TM, ALARM_DESC, ALARM_YN}}
    """
    alarm_list = get_alarm_data(from_time, to_time)
    
    alarm_map = {}
    for a in alarm_list:
        meas_tm = str(a.get('MEAS_TM', ''))
        alarm_map[meas_tm] = {
            'LSTM_FCAST_TM': a.get('LSTM_FCAST_TM', ''),
            'ALARM_DESC': a.get('ALARM_DESC', ''),
            'ALARM_YN': a.get('ALARM_YN', 'Y')
        }
    
    return alarm_map


# 테스트
if __name__ == "__main__":
    print("=" * 60)
    print("로그프레소 알람 조회 테스트")
    print("=" * 60)
    
    # 테스트 조회
    alarms = get_alarm_data('20260102070000', '20260105070000')
    
    if alarms:
        print(f"\n총 {len(alarms)}개 알람")
        print("\n[샘플 데이터]")
        for a in alarms[:10]:
            print(f"  {a['MEAS_TM']} | {a['ALARM_DESC']}")