# -*- coding: utf-8 -*-
"""
V6 평가 전용 코드 - 이상감지 Feature FULL 버전
전체 데이터를 슬라이딩 윈도우로 평가
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta


def evaluate_all_predictions():
    """전체 데이터를 슬라이딩 윈도우로 평가 (이상감지 Feature FULL)"""
   
    # V6 Ultimate 필수 컬럼 정의
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
    }
     
    # 모델 로드
    try:
        with open('xgboost_model_30min_10min_이상감지FULL.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 완료: xgboost_model_30min_10min_이상감지FULL.pkl")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        return None
   
    # 데이터 로드
    df = pd.read_csv('HUB0905101512.CSV', on_bad_lines='skip')
    print(f"✅ 데이터 로드 완료: {len(df)}개 행")
   
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # 사용 가능한 컬럼 확인
    print(f"\n사용 가능한 컬럼 확인:")
    all_feature_cols = []
    for group_name, cols in FEATURE_COLS.items():
        available = [col for col in cols if col in df.columns]
        all_feature_cols.extend(available)
        print(f"  - {group_name}: {len(available)}/{len(cols)}개")
   
    # STAT_DT 처리
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            print("⚠️ STAT_DT 변환 실패, 가상 시간 생성")
            base_time = datetime(2024, 1, 1, 0, 0)
            df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
   
    results = []
   
    # 슬라이딩 윈도우: 30개 시퀀스 → 10분 후 예측
    for i in range(30, len(df)):
        # 과거 30개 데이터
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
       
        # 현재 시점 (시퀀스 마지막)
        current_time = seq_data['STAT_DT'].iloc[-1]
       
        # 예측 시점 (10분 후)
        prediction_time = current_time + timedelta(minutes=10)
       
        # 실제값 (i번째 행)
        actual_value = df.iloc[i][TARGET_COL]
        actual_time = df.iloc[i]['STAT_DT']
       
        # 🔥 타겟 컬럼 기본 통계 (재사용)
        seq_mean = np.mean(seq_target)
        seq_std = np.std(seq_target)
        seq_max = np.max(seq_target)
        seq_min = np.min(seq_target)
        last_5_mean = np.mean(seq_target[-5:])
        last_10_mean = np.mean(seq_target[-10:])
        first_10_mean = np.mean(seq_target[:10])
        slope = np.polyfit(np.arange(30), seq_target, 1)[0]
        
        features = {
            # 타겟 컬럼 기본 특성
            'target_mean': seq_mean,
            'target_std': seq_std,
            'target_last_5_mean': last_5_mean,
            'target_max': seq_max,
            'target_min': seq_min,
            'target_slope': slope,
            'target_last_10_mean': last_10_mean,
            'target_first_10_mean': first_10_mean,
        }
        
        # 🔥🔥🔥 타겟 컬럼 이상감지 Feature
        features['target_near_threshold'] = 1 if 290 <= seq_max < 300 else 0
        features['target_rising_trend'] = 1 if slope > 0.5 else 0
        features['target_recent_high'] = 1 if last_5_mean > seq_mean + 10 else 0
        features['target_risk_score'] = max(0, (seq_max - 290) / 10)
        features['target_near_300'] = 1 if seq_max >= 295 else 0
        features['target_last_5_rising'] = 1 if last_5_mean > last_10_mean else 0
        features['target_acceleration'] = np.mean(np.diff(seq_target[-5:])) if len(seq_target) >= 6 else 0
        
        # 각 컬럼 그룹별 특성 추가
        for group_name, cols in FEATURE_COLS.items():
            for col in cols:
                if col in df.columns:
                    seq_data_col = seq_data[col].values
                    
                    # 🔥 컬럼별 기본 통계 (재사용)
                    col_mean = np.mean(seq_data_col)
                    col_std = np.std(seq_data_col)
                    col_max = np.max(seq_data_col)
                    col_min = np.min(seq_data_col)
                    col_last_5_mean = np.mean(seq_data_col[-5:])
                    col_last_10_mean = np.mean(seq_data_col[-10:])
                    col_first_10_mean = np.mean(seq_data_col[:10])
                    col_slope = np.polyfit(np.arange(30), seq_data_col, 1)[0]
                    
                    # 기본 통계
                    features[f'{col}_mean'] = col_mean
                    features[f'{col}_std'] = col_std
                    features[f'{col}_max'] = col_max
                    features[f'{col}_min'] = col_min
                    features[f'{col}_last_5_mean'] = col_last_5_mean
                    features[f'{col}_last_10_mean'] = col_last_10_mean
                    features[f'{col}_slope'] = col_slope
                    features[f'{col}_first_10_mean'] = col_first_10_mean
                    features[f'{col}_mid_10_mean'] = np.mean(seq_data_col[10:20])
                    features[f'{col}_last_value'] = seq_data_col[-1]
                    
                    # 🔥🔥🔥 각 컬럼별 이상감지 Feature
                    if 'CMD' in col:
                        col_threshold_low = 180
                        col_threshold_high = 280
                        features[f'{col}_near_low_threshold'] = 1 if col_min <= col_threshold_low else 0
                        features[f'{col}_near_high_threshold'] = 1 if col_max >= col_threshold_high else 0
                        features[f'{col}_in_danger_zone'] = 1 if (col_last_5_mean < 220) else 0
                    elif 'STORAGE' in col:
                        features[f'{col}_near_threshold'] = 1 if 30 <= col_max <= 40 else 0
                        features[f'{col}_in_danger_zone'] = 1 if (30 <= col_last_5_mean <= 40) else 0
                    else:
                        features[f'{col}_near_max'] = 1 if col_max >= (col_mean + 2*col_std) else 0
                    
                    # 공통 이상감지 Feature
                    features[f'{col}_rising_trend'] = 1 if col_slope > 0.5 else 0
                    features[f'{col}_falling_trend'] = 1 if col_slope < -0.5 else 0
                    features[f'{col}_recent_high'] = 1 if col_last_5_mean > col_mean + (col_std * 0.5) else 0
                    features[f'{col}_recent_low'] = 1 if col_last_5_mean < col_mean - (col_std * 0.5) else 0
                    features[f'{col}_last_5_rising'] = 1 if col_last_5_mean > col_last_10_mean else 0
                    features[f'{col}_acceleration'] = np.mean(np.diff(seq_data_col[-5:])) if len(seq_data_col) >= 6 else 0
                    features[f'{col}_volatility'] = col_std / (col_mean + 1e-6)
        
        # 유입-유출 차이 (Net Flow)
        inflow_sum = 0
        outflow_sum = 0
        for col in FEATURE_COLS['inflow']:
            if col in df.columns:
                inflow_sum += df[col].iloc[i-1]
        for col in FEATURE_COLS['outflow']:
            if col in df.columns:
                outflow_sum += df[col].iloc[i-1]
        features['net_flow'] = inflow_sum - outflow_sum
        features['net_flow_positive'] = 1 if (inflow_sum - outflow_sum) > 0 else 0
        
        # CMD 총합
        cmd_sum = 0
        for col in FEATURE_COLS['cmd']:
            if col in df.columns:
                cmd_sum += df[col].iloc[i-1]
        features['total_cmd'] = cmd_sum
        features['total_cmd_low'] = 1 if cmd_sum < 400 else 0
       
        X_pred = pd.DataFrame([features])
       
        # 예측
        prediction = model.predict(X_pred)[0]
       
        # 300 이상 점프 감지 (시퀀스 내)
        jump_detected = np.any(seq_target >= 300)
       
        # 결과 저장
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '실제시점': actual_time.strftime('%Y-%m-%d %H:%M'),
            '실제값': actual_value,
            '예측값': round(prediction, 2),
            '오차': round(actual_value - prediction, 2),
            '오차율(%)': round(abs(actual_value - prediction) / max(actual_value, 1) * 100, 2),
            '시퀀스MAX': np.max(seq_target),
            '시퀀스MIN': np.min(seq_target),
            '시퀀스평균': round(np.mean(seq_target), 2),
            '300이상점프': '🔴' if jump_detected else '',
            '실제값상태': '🔴극단' if actual_value >= 300 else ('🟡주의' if actual_value >= 280 else '🟢정상'),
            '예측값상태': '🔴극단' if prediction >= 300 else ('🟡주의' if prediction >= 280 else '🟢정상')
        })
       
        # 진행상황 출력
        if (i - 30) % 100 == 0:
            print(f"진행중... {i-30}/{len(df)-30} ({(i-30)/(len(df)-30)*100:.1f}%)")
   
    # DataFrame 변환
    results_df = pd.DataFrame(results)
   
    # CSV 저장
    output_file = 'prediction_evaluation_이상감지FULL.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장 완료: {output_file}")
   
    # 통계 출력
    print("\n" + "="*80)
    print("📊 평가 통계 (이상감지 Feature FULL)")
    print("="*80)
    print(f"총 예측 수: {len(results_df)}")
    print(f"평균 오차: {results_df['오차'].abs().mean():.2f}")
    print(f"평균 오차율: {results_df['오차율(%)'].mean():.2f}%")
    print(f"최대 오차: {results_df['오차'].abs().max():.2f}")
    print(f"\n300이상 점프 구간: {results_df['300이상점프'].value_counts().get('🔴', 0)}개")
    print(f"실제값 극단(≥300): {(results_df['실제값'] >= 300).sum()}개")
    print(f"예측값 극단(≥300): {(results_df['예측값'] >= 300).sum()}개")
   
    # 상위 오차 구간
    print("\n" + "="*80)
    print("❌ 오차 상위 10개 구간")
    print("="*80)
    top_errors = results_df.nlargest(10, '오차율(%)')
    print(top_errors[['현재시간', '실제값', '예측값', '오차', '오차율(%)', '시퀀스MAX', '300이상점프']].to_string(index=False))
   
    # 극단값 구간
    extreme_cases = results_df[results_df['실제값'] >= 300]
    if len(extreme_cases) > 0:
        print("\n" + "="*80)
        print("🔴 실제 극단값(≥300) 구간")
        print("="*80)
        print(extreme_cases[['현재시간', '실제값', '예측값', '오차', '시퀀스MAX', '시퀀스MIN']].head(10).to_string(index=False))
   
    return results_df

if __name__ == '__main__':
    print("🚀 V6 실시간 예측 평가 시작 (이상감지 Feature FULL)...\n")
    results = evaluate_all_predictions()
   
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측 생성")
        print(f"📁 결과 파일: prediction_evaluation_이상감지FULL.csv")