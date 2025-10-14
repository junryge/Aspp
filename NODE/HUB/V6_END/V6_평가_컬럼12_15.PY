# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 07:35:29 2025

@author: X0163954
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_all_predictions():
    """전체 데이터를 슬라이딩 윈도우로 평가"""
   
    # V4 Ultimate 필수 컬럼 정의
    # 핵심 12개 컬럼만 사용 (물리적으로 중요한 것만)
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],  # 극단값 핵심 지표
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],  # 주요 CMD
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],  # 주요 유입
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],  # 주요 유출
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']  # 용량 제한
    }
   
    # 모델 로드
    try:
        with open('xgboost_model_30min_15min.pkl', 'rb') as f:
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
   
    # 슬라이딩 윈도우: 30개 시퀀스 → 15분 후 예측
    for i in range(30, len(df) - 15):
        # 과거 30개 데이터
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
       
        # 현재 시점 (시퀀스 마지막)
        current_time = seq_data['STAT_DT'].iloc[-1]
       
        # 예측 시점 (15분 후)
        prediction_time = current_time + timedelta(minutes=15)
       
        # 실제값 (i+15번째 행)
        actual_value = df.iloc[i+15][TARGET_COL]
        actual_time = df.iloc[i+15]['STAT_DT']
       
        # Feature 생성 (타겟 컬럼)
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_last_5_mean': np.mean(seq_target[-5:]),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
            'target_last_10_mean': np.mean(seq_target[-10:]),
            'target_first_10_mean': np.mean(seq_target[:10]),
            'target_first_15_mean': np.mean(seq_target[:15]),  # 처음 15분
            'target_last_15_mean': np.mean(seq_target[15:])    # 마지막 15분
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
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                   
                    # 구간별 평균
                    features[f'{col}_first_10_mean'] = np.mean(col_seq[:10])
                    features[f'{col}_mid_10_mean'] = np.mean(col_seq[10:20])
                    features[f'{col}_last_value'] = col_seq[-1]
                    
                    # 처음 15분 / 마지막 15분
                    features[f'{col}_first_15_mean'] = np.mean(col_seq[:15])
                    features[f'{col}_last_15_mean'] = np.mean(col_seq[15:])
       
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
       
        # CMD 총합
        cmd_sum = 0
        for col in FEATURE_COLS['cmd']:
            if col in df.columns:
                cmd_sum += df[col].iloc[i-1]
        features['total_cmd'] = cmd_sum
       
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
            print(f"진행중... {i-30}/{len(df)-30-15} ({(i-30)/(len(df)-30-15)*100:.1f}%)")
   
    # DataFrame 변환
    results_df = pd.DataFrame(results)
   
    # CSV 저장
    output_file = 'prediction_evaluation_컬럼12.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장 완료: {output_file}")
   
    # 통계 출력
    print("\n" + "="*80)
    print("📊 평가 통계")
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
        print(extreme_cases[['현재시간', '실제값', '예측값', '오차', '시퀀스MAX', '시퀀스MIN']].to_string(index=False))
   
    return results_df

if __name__ == '__main__':
    print("🚀 실시간 예측 평가 시작 (Multi-Column Features + first15/last15)...\n")
    results = evaluate_all_predictions()
   
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측 생성")
        print(f"📁 결과 파일: prediction_evaluation_컬럼12.csv")