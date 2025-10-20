# -*- coding: utf-8 -*-
"""
V6 평가코드 - LightGBM 분위 회귀 (Quantile Regression)
전체 데이터 슬라이딩 윈도우 평가
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta


def evaluate_all_predictions_quantile():
    """전체 데이터를 슬라이딩 윈도우로 평가 - 분위 회귀 모델 사용"""
   
    # 필수 컬럼 정의
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
    }
     
    # 모델 로드
    try:
        with open('lightgbm_quantile_model_30min_10min_12컬럼.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 분위 회귀 모델 로드 완료")
        print("   - objective: quantile")
        print("   - alpha: 0.9 (보수적 예측)")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        print("   먼저 V6_학습코드_분위회귀.py를 실행하세요!")
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
    보수적예측_count = 0
    사전감지_성공 = 0
    실제_300_count = 0
   
    print("\n" + "="*80)
    print("🔮 LightGBM 분위 회귀 (alpha=0.9) 평가")
    print("="*80)
    print("특징: 보수적 예측 (실제보다 높게 예측)")
    print("목표: 사전감지 성능 극대화")
    print("="*80 + "\n")
   
    # 슬라이딩 윈도우
    for i in range(30, len(df)):
        # 과거 30개 데이터
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
       
        # 현재 시점
        current_time = seq_data['STAT_DT'].iloc[-1]
       
        # 예측 시점 (10분 후)
        prediction_time = current_time + timedelta(minutes=10)
       
        # 실제값
        actual_value = df.iloc[i][TARGET_COL]
        actual_time = df.iloc[i]['STAT_DT']
       
        # Feature 생성
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_last_5_mean': np.mean(seq_target[-5:]),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
            'target_last_10_mean': np.mean(seq_target[-10:]),
            'target_first_10_mean': np.mean(seq_target[:10])
        }
        
        # 각 컬럼 그룹별 특성 추가
        for group_name, cols in FEATURE_COLS.items():
            for col in cols:
                if col in df.columns:
                    col_seq = seq_data[col].values
                    
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_std'] = np.std(col_seq)
                    features[f'{col}_max'] = np.max(col_seq)
                    features[f'{col}_min'] = np.min(col_seq)
                    features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                    features[f'{col}_last_10_mean'] = np.mean(col_seq[-10:])
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                    features[f'{col}_first_10_mean'] = np.mean(col_seq[:10])
                    features[f'{col}_mid_10_mean'] = np.mean(col_seq[10:20])
                    features[f'{col}_last_value'] = col_seq[-1]
        
        # 유입-유출 차이
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
       
        # 분위 회귀 예측 (alpha=0.9)
        prediction = model.predict(X_pred)[0]
        
        # 통계
        seq_max = np.max(seq_target)
        seq_min = np.min(seq_target)
        increase_rate = seq_target[-1] - seq_target[0]
        
        # 보수적 예측 여부
        보수적예측 = prediction > actual_value
        if 보수적예측:
            보수적예측_count += 1
        
        # 사전감지 성공 여부
        if actual_value >= 300:
            실제_300_count += 1
            if prediction >= 300:
                사전감지_성공 += 1
        
        # 300 이상 점프 감지
        jump_detected = np.any(seq_target >= 300)
       
        # 결과 저장
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '실제시점': actual_time.strftime('%Y-%m-%d %H:%M'),
            '실제값': actual_value,
            '예측값': round(prediction, 2),
            '오차': round(prediction - actual_value, 2),  # 예측 - 실제
            '절대오차': round(abs(actual_value - prediction), 2),
            '오차율(%)': round(abs(actual_value - prediction) / max(actual_value, 1) * 100, 2),
            '시퀀스MAX': round(seq_max, 2),
            '시퀀스MIN': round(seq_min, 2),
            '시퀀스평균': round(np.mean(seq_target), 2),
            '시퀀스증가': round(increase_rate, 2),
            '보수적예측': 'O' if 보수적예측 else 'X',
            '300이상점프': '🔴' if jump_detected else '',
            '실제값상태': '🔴극단' if actual_value >= 300 else ('🟡주의' if actual_value >= 280 else '🟢정상'),
            '예측값상태': '🔴극단' if prediction >= 300 else ('🟡주의' if prediction >= 280 else '🟢정상'),
            '사전감지성공': 'O' if (actual_value >= 300 and prediction >= 300) else '-'
        })
       
        # 진행상황 출력
        if (i - 30) % 100 == 0:
            print(f"진행중... {i-30}/{len(df)-30} ({(i-30)/(len(df)-30)*100:.1f}%)")
   
    # DataFrame 변환
    results_df = pd.DataFrame(results)
   
    # CSV 저장
    output_file = 'prediction_evaluation_분위회귀.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장 완료: {output_file}")
   
    # 통계 출력
    print("\n" + "="*80)
    print("📊 평가 통계")
    print("="*80)
    print(f"총 예측 수: {len(results_df)}")
    print(f"평균 오차: {results_df['절대오차'].mean():.2f}")
    print(f"평균 오차율: {results_df['오차율(%)'].mean():.2f}%")
    print(f"최대 오차: {results_df['절대오차'].max():.2f}")
    
    print("\n" + "="*80)
    print("🎯 보수적 예측 특성")
    print("="*80)
    print(f"보수적 예측 (예측 > 실제): {보수적예측_count}/{len(results_df)} ({보수적예측_count/len(results_df)*100:.1f}%)")
    print(f"평균 (예측 - 실제): {results_df['오차'].mean():.2f}")
    
    print("\n" + "="*80)
    print("🚨 사전감지 성능")
    print("="*80)
    print(f"실제 300+ 발생: {실제_300_count}개")
    print(f"예측 300+: {(results_df['예측값'] >= 300).sum()}개")
    print(f"사전감지 성공: {사전감지_성공}/{실제_300_count}개", end="")
    if 실제_300_count > 0:
        print(f" ({사전감지_성공/실제_300_count*100:.1f}%)")
    else:
        print()
    
    # 극단값 케이스 상세
    extreme_cases = results_df[results_df['실제값'] >= 300]
    if len(extreme_cases) > 0:
        print("\n" + "="*80)
        print(f"🔴 실제 극단값(≥300) 케이스 상세 (전체 {len(extreme_cases)}개)")
        print("="*80)
        print(extreme_cases[['현재시간', '실제값', '예측값', '오차', '시퀀스MAX', '사전감지성공']].head(20).to_string(index=False))
   
    # 사전감지 실패 케이스 (중요!)
    실패_cases = results_df[(results_df['실제값'] >= 300) & (results_df['예측값'] < 300)]
    if len(실패_cases) > 0:
        print("\n" + "="*80)
        print(f"❌ 사전감지 실패 케이스 (전체 {len(실패_cases)}개)")
        print("="*80)
        print(실패_cases[['현재시간', '실제값', '예측값', '오차', '시퀀스MAX', '시퀀스증가']].to_string(index=False))
    
    # 오탐 케이스
    오탐_cases = results_df[(results_df['실제값'] < 300) & (results_df['예측값'] >= 300)]
    if len(오탐_cases) > 0:
        print("\n" + "="*80)
        print(f"⚠️ 오탐 케이스 (예측 300+ but 실제 < 300) (전체 {len(오탐_cases)}개)")
        print("="*80)
        print(오탐_cases[['현재시간', '실제값', '예측값', '오차', '시퀀스MAX']].head(10).to_string(index=False))
   
    return results_df

if __name__ == '__main__':
    print("🚀 분위 회귀 모델 평가 시작...\n")
    results = evaluate_all_predictions_quantile()
   
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측 생성")
        print(f"📁 결과 파일: prediction_evaluation_분위회귀.csv")