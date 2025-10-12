# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 07:35:29 2025

@author: X0163954
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def create_features(seq_data, seq_target, FEATURE_COLS, df):
    """Feature 생성 함수"""
    features = {
        'target_mean': np.mean(seq_target),
        'target_std': np.std(seq_target),
        'target_last_5_mean': np.mean(seq_target[-5:]),
        'target_max': np.max(seq_target),
        'target_min': np.min(seq_target),
        'target_slope': np.polyfit(np.arange(len(seq_target)), seq_target, 1)[0],
        'target_last_10_mean': np.mean(seq_target[-10:]),
        'target_first_10_mean': np.mean(seq_target[:10])
    }
    
    # 각 컬럼 그룹별 특성 추가
    for group_name, cols in FEATURE_COLS.items():
        for col in cols:
            if col in df.columns:
                col_seq = seq_data[col].values
                
                # 기본 통계
                features[f'{col}_mean'] = np.mean(col_seq)
                features[f'{col}_std'] = np.std(col_seq)
                features[f'{col}_max'] = np.max(col_seq)
                features[f'{col}_min'] = np.min(col_seq)
                
                # 최근 특성
                features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                features[f'{col}_last_10_mean'] = np.mean(col_seq[-10:])
                
                # 추세
                features[f'{col}_slope'] = np.polyfit(np.arange(len(col_seq)), col_seq, 1)[0]
                
                # 구간별 평균
                features[f'{col}_first_10_mean'] = np.mean(col_seq[:10])
                features[f'{col}_mid_10_mean'] = np.mean(col_seq[10:20])
                features[f'{col}_last_value'] = col_seq[-1]
    
    # 유입-유출 차이 (Net Flow)
    inflow_sum = 0
    outflow_sum = 0
    for col in FEATURE_COLS['inflow']:
        if col in df.columns:
            inflow_sum += seq_data[col].iloc[-1]
    for col in FEATURE_COLS['outflow']:
        if col in df.columns:
            outflow_sum += seq_data[col].iloc[-1]
    features['net_flow'] = inflow_sum - outflow_sum
    
    # CMD 총합
    cmd_sum = 0
    for col in FEATURE_COLS['cmd']:
        if col in df.columns:
            cmd_sum += seq_data[col].iloc[-1]
    features['total_cmd'] = cmd_sum
    
    return features

def evaluate_all_predictions():
    """전체 데이터를 슬라이딩 윈도우로 평가 - 10분/20분/30분 예측"""
   
    # 핵심 12개 컬럼
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
    }
    
    # 모델 로드
    try:
        with open('xgboost_model_30min_10min_12컬럼.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        return None
   
    # 데이터 로드
    df = pd.read_csv('HUB0906_0929.CSV', on_bad_lines='skip')
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
   
    # 슬라이딩 윈도우: 30개 시퀀스 → 10/20/30분 후 예측
    for i in range(30, len(df) - 30):  # -30: 30분 후 실제값 확보
        # 과거 30개 데이터
        seq_data_original = df.iloc[i-30:i].copy()
        seq_target_original = seq_data_original[TARGET_COL].values.copy()
       
        # 현재 시점
        current_time = seq_data_original['STAT_DT'].iloc[-1]
       
        # ===== 10분 후 예측 =====
        seq_data_10 = seq_data_original.copy()
        seq_target_10 = seq_target_original.copy()
        
        features_10 = create_features(seq_data_10, seq_target_10, FEATURE_COLS, df)
        X_pred_10 = pd.DataFrame([features_10])
        prediction_10min = model.predict(X_pred_10)[0]
        
        # 실제값 10분 후
        actual_10min = df.iloc[i][TARGET_COL] if i < len(df) else None
        actual_time_10 = df.iloc[i]['STAT_DT'] if i < len(df) else None
        
        # ===== 20분 후 예측 (재귀적) =====
        # 시퀀스 업데이트: 한 칸 밀고 10분 예측값 추가
        seq_target_20 = np.append(seq_target_10[1:], prediction_10min)
        
        # 다른 컬럼들은 마지막 값 유지 (간단한 방법)
        new_row = seq_data_10.iloc[-1:].copy()
        new_row[TARGET_COL] = prediction_10min
        seq_data_20 = pd.concat([seq_data_10.iloc[1:], new_row], ignore_index=True)
        
        features_20 = create_features(seq_data_20, seq_target_20, FEATURE_COLS, df)
        X_pred_20 = pd.DataFrame([features_20])
        prediction_20min = model.predict(X_pred_20)[0]
        
        # 실제값 20분 후
        actual_20min = df.iloc[i+10][TARGET_COL] if i+10 < len(df) else None
        actual_time_20 = df.iloc[i+10]['STAT_DT'] if i+10 < len(df) else None
        
        # ===== 30분 후 예측 (재귀적) =====
        seq_target_30 = np.append(seq_target_20[1:], prediction_20min)
        
        new_row_30 = seq_data_20.iloc[-1:].copy()
        new_row_30[TARGET_COL] = prediction_20min
        seq_data_30 = pd.concat([seq_data_20.iloc[1:], new_row_30], ignore_index=True)
        
        features_30 = create_features(seq_data_30, seq_target_30, FEATURE_COLS, df)
        X_pred_30 = pd.DataFrame([features_30])
        prediction_30min = model.predict(X_pred_30)[0]
        
        # 실제값 30분 후
        actual_30min = df.iloc[i+20][TARGET_COL] if i+20 < len(df) else None
        actual_time_30 = df.iloc[i+20]['STAT_DT'] if i+20 < len(df) else None
        
        # 300 이상 점프 감지
        jump_detected = np.any(seq_target_original >= 300)
        
        # 결과 저장
        result = {
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            
            # 10분 후
            '예측시점_10min': (current_time + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M'),
            '실제시점_10min': actual_time_10.strftime('%Y-%m-%d %H:%M') if actual_time_10 else '',
            '실제값_10min': actual_10min if actual_10min else 0,
            '예측값_10min': round(prediction_10min, 2),
            '오차_10min': round(actual_10min - prediction_10min, 2) if actual_10min else 0,
            '오차율_10min(%)': round(abs(actual_10min - prediction_10min) / max(actual_10min, 1) * 100, 2) if actual_10min else 0,
            
            # 20분 후
            '예측시점_20min': (current_time + timedelta(minutes=20)).strftime('%Y-%m-%d %H:%M'),
            '실제시점_20min': actual_time_20.strftime('%Y-%m-%d %H:%M') if actual_time_20 else '',
            '실제값_20min': actual_20min if actual_20min else 0,
            '예측값_20min': round(prediction_20min, 2),
            '오차_20min': round(actual_20min - prediction_20min, 2) if actual_20min else 0,
            '오차율_20min(%)': round(abs(actual_20min - prediction_20min) / max(actual_20min, 1) * 100, 2) if actual_20min else 0,
            
            # 30분 후
            '예측시점_30min': (current_time + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M'),
            '실제시점_30min': actual_time_30.strftime('%Y-%m-%d %H:%M') if actual_time_30 else '',
            '실제값_30min': actual_30min if actual_30min else 0,
            '예측값_30min': round(prediction_30min, 2),
            '오차_30min': round(actual_30min - prediction_30min, 2) if actual_30min else 0,
            '오차율_30min(%)': round(abs(actual_30min - prediction_30min) / max(actual_30min, 1) * 100, 2) if actual_30min else 0,
            
            # 시퀀스 정보
            '시퀀스MAX': np.max(seq_target_original),
            '시퀀스MIN': np.min(seq_target_original),
            '시퀀스평균': round(np.mean(seq_target_original), 2),
            '300이상점프': '🔴' if jump_detected else '',
            
            # 상태 플래그
            '실제값상태_10min': '🔴극단' if (actual_10min and actual_10min >= 300) else ('🟡주의' if (actual_10min and actual_10min >= 280) else '🟢정상'),
            '예측값상태_10min': '🔴극단' if prediction_10min >= 300 else ('🟡주의' if prediction_10min >= 280 else '🟢정상'),
            '실제값상태_20min': '🔴극단' if (actual_20min and actual_20min >= 300) else ('🟡주의' if (actual_20min and actual_20min >= 280) else '🟢정상'),
            '예측값상태_20min': '🔴극단' if prediction_20min >= 300 else ('🟡주의' if prediction_20min >= 280 else '🟢정상'),
            '실제값상태_30min': '🔴극단' if (actual_30min and actual_30min >= 300) else ('🟡주의' if (actual_30min and actual_30min >= 280) else '🟢정상'),
            '예측값상태_30min': '🔴극단' if prediction_30min >= 300 else ('🟡주의' if prediction_30min >= 280 else '🟢정상')
        }
        
        results.append(result)
       
        # 진행상황 출력
        if (i - 30) % 100 == 0:
            print(f"진행중... {i-30}/{len(df)-60} ({(i-30)/(len(df)-60)*100:.1f}%)")
   
    # DataFrame 변환
    results_df = pd.DataFrame(results)
   
    # CSV 저장
    output_file = 'prediction_evaluation_10_20_30min.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장 완료: {output_file}")
   
    # 통계 출력
    print("\n" + "="*80)
    print("📊 평가 통계 - 10분/20분/30분 후 예측")
    print("="*80)
    print(f"총 예측 수: {len(results_df)}")
    
    print(f"\n[10분 후 예측]")
    print(f"  평균 오차: {results_df['오차_10min'].abs().mean():.2f}")
    print(f"  평균 오차율: {results_df['오차율_10min(%)'].mean():.2f}%")
    print(f"  최대 오차: {results_df['오차_10min'].abs().max():.2f}")
    print(f"  실제값 극단(≥300): {(results_df['실제값_10min'] >= 300).sum()}개")
    print(f"  예측값 극단(≥300): {(results_df['예측값_10min'] >= 300).sum()}개")
    
    print(f"\n[20분 후 예측]")
    print(f"  평균 오차: {results_df['오차_20min'].abs().mean():.2f}")
    print(f"  평균 오차율: {results_df['오차율_20min(%)'].mean():.2f}%")
    print(f"  최대 오차: {results_df['오차_20min'].abs().max():.2f}")
    print(f"  실제값 극단(≥300): {(results_df['실제값_20min'] >= 300).sum()}개")
    print(f"  예측값 극단(≥300): {(results_df['예측값_20min'] >= 300).sum()}개")
    
    print(f"\n[30분 후 예측]")
    print(f"  평균 오차: {results_df['오차_30min'].abs().mean():.2f}")
    print(f"  평균 오차율: {results_df['오차율_30min(%)'].mean():.2f}%")
    print(f"  최대 오차: {results_df['오차_30min'].abs().max():.2f}")
    print(f"  실제값 극단(≥300): {(results_df['실제값_30min'] >= 300).sum()}개")
    print(f"  예측값 극단(≥300): {(results_df['예측값_30min'] >= 300).sum()}개")
    
    print(f"\n300이상 점프 구간: {results_df['300이상점프'].value_counts().get('🔴', 0)}개")
   
    # 상위 오차 구간 (10분 기준)
    print("\n" + "="*80)
    print("❌ 10분 후 예측 오차 상위 10개 구간")
    print("="*80)
    top_errors = results_df.nlargest(10, '오차율_10min(%)')
    print(top_errors[['현재시간', '실제값_10min', '예측값_10min', '오차_10min', '오차율_10min(%)', '시퀀스MAX']].to_string(index=False))
   
    # 극단값 구간 (10분 후 기준)
    extreme_cases = results_df[results_df['실제값_10min'] >= 300]
    if len(extreme_cases) > 0:
        print("\n" + "="*80)
        print("🔴 실제 극단값(≥300) 구간 - 10분 후")
        print("="*80)
        print(extreme_cases[['현재시간', '실제값_10min', '예측값_10min', '오차_10min', '시퀀스MAX', '시퀀스MIN']].head(10).to_string(index=False))
   
    return results_df

if __name__ == '__main__':
    print("🚀 10분/20분/30분 후 예측 평가 시작...\n")
    results = evaluate_all_predictions()
   
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측 생성")
        print(f"📁 결과 파일: prediction_evaluation_10_20_30min.csv")