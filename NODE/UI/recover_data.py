# -*- coding: utf-8 -*-
# python recover_data.py 20260115 20260119
"""
================================================================================
M14 데이터 복구 스크립트
- evaluator.py의 get_logpresso_data_range 함수 활용
- predictor로 예측값 계산하여 data, pred, alert CSV 파일 재생성
- 사용법: python recover_data.py 20250115 20250119
================================================================================
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# evaluator.py에서 로그프레소 조회 함수 import
import evaluator

DATA_DIR = 'data'
ALARM_COOLDOWN = 60 * 60  # 쿨타임 1시간 (초)

# ============================================================================
# 예측 모듈 로드
# ============================================================================
try:
    import predictor_10min
    import predictor_30min
    PREDICTOR_AVAILABLE = True
    print("[복구] ✅ 예측 모듈 로드 완료")
except ImportError as e:
    PREDICTOR_AVAILABLE = False
    print(f"[복구] ⚠️ 예측 모듈 로드 실패: {e}")
    print("       → pred 파일은 예측값 0으로, alert 파일은 생성되지 않습니다")


def add_minutes_to_time(time_str, mins):
    """YYYYMMDDHHMM 형식에 분 더하기"""
    try:
        if len(time_str) >= 12:
            dt = datetime.strptime(time_str[:12], "%Y%m%d%H%M")
            dt = dt + timedelta(minutes=mins)
            return dt.strftime("%Y%m%d%H%M")
    except:
        pass
    return time_str


def currtime_to_datetime(curr_time):
    """YYYYMMDDHHMM → datetime"""
    try:
        return datetime.strptime(str(curr_time)[:12], "%Y%m%d%H%M")
    except:
        return None


def currtime_to_timestamp(curr_time):
    """YYYYMMDDHHMM → YYYY-MM-DD HH:MM:SS"""
    dt = currtime_to_datetime(curr_time)
    if dt:
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    return ''


def get_data_for_date(date_str):
    """특정 날짜의 전체 데이터 조회 (00:00 ~ 23:59)"""
    from_time = f"{date_str}0000"
    to_time = f"{date_str}2359"
    
    print(f"\n[조회] {date_str} ({from_time} ~ {to_time})")
    
    # evaluator.py의 함수 사용
    df = evaluator.get_logpresso_data_range(from_time, to_time)
    
    if df is None or len(df) == 0:
        print("  → 데이터 없음")
        return None
    
    print(f"  → {len(df)}개 row 조회 완료")
    return df


def calculate_predictions_and_alerts(df_all, target_date, last_alarm_time=None, alarm_count=0):
    """
    예측값 및 알람 기록 계산 (280분 시퀀스 필요)
    
    Returns:
        pred_data: 예측 데이터 리스트
        alert_data: 알람 기록 리스트
        last_alarm_time: 마지막 알람 시간 (다음 날짜로 전달)
        alarm_count: 현재 알람 카운트 (다음 날짜로 전달)
    """
    seq_len = 280
    
    # 해당 날짜 데이터만 필터
    df_target = df_all[df_all['CURRTIME'].str.startswith(target_date)].copy()
    
    if len(df_target) == 0:
        return [], [], last_alarm_time, alarm_count
    
    pred_data = []
    alert_data = []
    total = len(df_target)
    
    for count, (orig_idx, row) in enumerate(df_target.iterrows()):
        curr_time = str(row['CURRTIME'])
        total_cnt = int(row['TOTALCNT']) if pd.notna(row['TOTALCNT']) else 0
        
        # 예측값 계산
        pred_10 = 0
        pred_30 = 0
        
        if PREDICTOR_AVAILABLE:
            # df_all에서 현재 시간까지의 인덱스 찾기
            mask = df_all['CURRTIME'] <= curr_time
            idx = mask.sum()
            
            if idx >= seq_len:
                df_slice = df_all.iloc[:idx].copy()
                try:
                    p10 = predictor_10min.predict(df_slice)
                    p30 = predictor_30min.predict(df_slice)
                    pred_10 = p10['predict_value']
                    pred_30 = p30['predict_value']
                except Exception as e:
                    pass
        
        pred_time_10 = add_minutes_to_time(curr_time, 10)
        pred_time_30 = add_minutes_to_time(curr_time, 30)
        
        pred_data.append({
            'CURRTIME': curr_time,
            'TOTALCNT': total_cnt,
            'PRED_TIME_10': pred_time_10,
            'PREDICT_10': pred_10,
            'PRED_TIME_30': pred_time_30,
            'PREDICT_30': pred_30
        })
        
        # ========== 알람 기록 생성 ==========
        current_dt = currtime_to_datetime(curr_time)
        timestamp = currtime_to_timestamp(curr_time)
        
        # 10분 예측 알람
        if pred_10 >= 1700:
            in_cooldown = False
            cooldown_mins = 0
            
            if last_alarm_time:
                elapsed = (current_dt - last_alarm_time).total_seconds()
                if elapsed < ALARM_COOLDOWN:
                    in_cooldown = True
                    cooldown_mins = int((ALARM_COOLDOWN - elapsed) / 60)
            
            is_alarm = not in_cooldown
            if is_alarm:
                alarm_count += 1
                last_alarm_time = current_dt
            
            alert_data.append({
                'TYPE': '10',
                'CURRTIME': curr_time,
                'VALUE': pred_10,
                'TIMESTAMP': timestamp,
                'ALARM_NO': alarm_count,
                'IS_ALARM': is_alarm,
                'COOLDOWN_MINS': cooldown_mins if not is_alarm else 0
            })
        
        # 30분 예측 알람
        if pred_30 >= 1700:
            in_cooldown = False
            cooldown_mins = 0
            
            if last_alarm_time:
                elapsed = (current_dt - last_alarm_time).total_seconds()
                if elapsed < ALARM_COOLDOWN:
                    in_cooldown = True
                    cooldown_mins = int((ALARM_COOLDOWN - elapsed) / 60)
            
            is_alarm = not in_cooldown
            if is_alarm:
                alarm_count += 1
                last_alarm_time = current_dt
            
            alert_data.append({
                'TYPE': '30',
                'CURRTIME': curr_time,
                'VALUE': pred_30,
                'TIMESTAMP': timestamp,
                'ALARM_NO': alarm_count,
                'IS_ALARM': is_alarm,
                'COOLDOWN_MINS': cooldown_mins if not is_alarm else 0
            })
        
        # 진행률 표시 (100개마다)
        if (count + 1) % 100 == 0 or count + 1 == total:
            pct = (count + 1) / total * 100
            print(f"  → 예측 진행: {count + 1}/{total} ({pct:.1f}%)")
    
    return pred_data, alert_data, last_alarm_time, alarm_count


def recover_date(date_str, df_prev=None, last_alarm_time=None, alarm_count=0):
    """특정 날짜 데이터 복구"""
    
    # 데이터 조회
    df = get_data_for_date(date_str)
    
    if df is None or len(df) == 0:
        print(f"  [실패] {date_str} 데이터 조회 실패")
        return None, last_alarm_time, alarm_count
    
    # 폴더 생성
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # data 파일 저장
    data_file = os.path.join(DATA_DIR, f'm14_data_{date_str}.csv')
    df.to_csv(data_file, index=False)
    print(f"  [저장] {data_file} ({len(df)}개)")
    
    # 예측값 계산을 위한 전체 데이터 (이전 데이터 + 현재 데이터)
    if df_prev is not None and len(df_prev) > 0:
        df_combined = pd.concat([df_prev, df], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
    else:
        df_combined = df.copy()
    
    # pred 및 alert 파일 생성
    print(f"  → 예측값 및 알람 계산 중... (총 {len(df)}개)")
    pred_data, alert_data, last_alarm_time, alarm_count = calculate_predictions_and_alerts(
        df_combined, date_str, last_alarm_time, alarm_count
    )
    
    if pred_data:
        pred_file = os.path.join(DATA_DIR, f'm14_pred_{date_str}.csv')
        pd.DataFrame(pred_data).to_csv(pred_file, index=False)
        
        # 예측 통계
        pred_10_vals = [p['PREDICT_10'] for p in pred_data]
        pred_30_vals = [p['PREDICT_30'] for p in pred_data]
        over_1700_10 = sum(1 for v in pred_10_vals if v >= 1700)
        over_1700_30 = sum(1 for v in pred_30_vals if v >= 1700)
        
        print(f"  [저장] {pred_file}")
        print(f"         10분 예측 1700+ : {over_1700_10}개")
        print(f"         30분 예측 1700+ : {over_1700_30}개")
    
    # alert 파일 저장
    if alert_data:
        alert_file = os.path.join(DATA_DIR, f'm14_alert_{date_str}.csv')
        pd.DataFrame(alert_data).to_csv(alert_file, index=False)
        
        # 알람 통계
        is_alarm_count = sum(1 for a in alert_data if a['IS_ALARM'] == True)
        cooldown_count = sum(1 for a in alert_data if a['IS_ALARM'] == False)
        
        print(f"  [저장] {alert_file}")
        print(f"         실제 알람 발생: {is_alarm_count}개")
        print(f"         쿨타임 중 억제: {cooldown_count}개")
    else:
        print(f"  [알람] 1700+ 예측 없음 → alert 파일 미생성")
    
    return df_combined, last_alarm_time, alarm_count


def recover_range(start_date, end_date):
    """날짜 범위 복구"""
    print("=" * 60)
    print("M14 데이터 복구 시작 (evaluator 로그프레소 사용)")
    print("=" * 60)
    print(f"기간: {start_date} ~ {end_date}")
    print(f"저장 경로: {DATA_DIR}/")
    print(f"예측 모듈: {'✅ 사용 가능' if PREDICTOR_AVAILABLE else '❌ 사용 불가 (예측값 0)'}")
    print(f"알람 쿨타임: {ALARM_COOLDOWN // 60}분")
    print("=" * 60)
    
    # 날짜 리스트 생성
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    print(f"복구 대상: {len(dates)}일")
    
    # 시퀀스를 위해 시작일 하루 전 데이터도 로드
    prev_date = (start - timedelta(days=1)).strftime("%Y%m%d")
    print(f"\n[사전 로드] {prev_date} (시퀀스용)")
    df_all = get_data_for_date(prev_date)
    
    if df_all is None:
        print("  → 전날 데이터 없음, 첫 날은 시퀀스 부족할 수 있음")
        df_all = pd.DataFrame()
    
    # 알람 상태 초기화
    last_alarm_time = None
    alarm_count = 0
    
    # 각 날짜 복구
    success_count = 0
    for i, date_str in enumerate(dates):
        print(f"\n[{i+1}/{len(dates)}] {date_str} 복구 중...")
        result, last_alarm_time, alarm_count = recover_date(
            date_str, df_all, last_alarm_time, alarm_count
        )
        if result is not None:
            df_all = result
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ 복구 완료: {success_count}/{len(dates)}일")
    print(f"   총 알람 발생 횟수: {alarm_count}회")
    print("=" * 60)
    
    if success_count > 0:
        print(f"\n생성된 파일:")
        for date_str in dates:
            data_file = os.path.join(DATA_DIR, f'm14_data_{date_str}.csv')
            pred_file = os.path.join(DATA_DIR, f'm14_pred_{date_str}.csv')
            alert_file = os.path.join(DATA_DIR, f'm14_alert_{date_str}.csv')
            if os.path.exists(data_file):
                print(f"  ✅ {data_file}")
                print(f"  ✅ {pred_file}")
                if os.path.exists(alert_file):
                    print(f"  ✅ {alert_file}")
                else:
                    print(f"  ⚪ {alert_file} (1700+ 없음)")


def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print("M14 데이터 복구 스크립트")
        print("=" * 60)
        print("\n사용법:")
        print("  python recover_data.py 20250115           # 특정 날짜")
        print("  python recover_data.py 20250115 20250119  # 날짜 범위")
        print("")
        
        # 대화형 모드
        print("날짜를 입력하세요:")
        start_date = input("시작 날짜 (YYYYMMDD): ").strip()
        if not start_date:
            print("취소됨")
            return
        
        end_date = input("종료 날짜 (YYYYMMDD, 같으면 Enter): ").strip()
        if not end_date:
            end_date = start_date
        
        recover_range(start_date, end_date)
    
    elif len(sys.argv) == 2:
        # 단일 날짜
        date_str = sys.argv[1]
        recover_range(date_str, date_str)
    
    else:
        # 날짜 범위
        start_date = sys.argv[1]
        end_date = sys.argv[2]
        recover_range(start_date, end_date)


if __name__ == "__main__":
    main()