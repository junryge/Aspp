# -*- coding: utf-8 -*-
"""
V6 평가코드 - 사전감지 조건 및 예측값 보정 적용
조건: 30 시퀀스에 283 이상 + 증가률 15 이상
보정: 조건 만족 시 예측값 +15
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta


def evaluate_all_predictions():
    """전체 데이터를 슬라이딩 윈도우로 평가 (사전감지 보정 적용)"""
   
    # V4 Ultimate 필수 컬럼 정의
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
    사전감지_count = 0
    보정적용_count = 0
   
    print("\n" + "="*80)
    print("🚨 사전감지 조건 적용")
    print("="*80)
    print("조건 1: 30 시퀀스 MAX < 300")
    print("조건 2: 30 시퀀스에 283 이상 값 존재")
    print("조건 3: 증가률(끝-처음) >= 15")
    print("보정값: +15")
    print("="*80 + "\n")
   
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
       
        # Feature 생성 (타겟 컬럼)
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
       
        # 기본 예측
        prediction = model.predict(X_pred)[0]
        
        # ========================================
        # 🚨 사전감지 조건 체크 및 보정 적용
        # ========================================
        seq_max = np.max(seq_target)
        seq_min = np.min(seq_target)
        
        # 조건 1: 시퀀스 MAX < 300
        condition1 = seq_max < 300
        
        # 조건 2: 시퀀스에 283 이상 존재
        condition2 = np.any(seq_target >= 283)
        
        # 조건 3: 증가률 >= 15
        increase_rate = seq_target[-1] - seq_target[0]
        condition3 = increase_rate >= 15
        
        # 사전감지 조건 만족 여부
        사전감지_조건 = condition1 and condition2 and condition3
        
        # 예측값 보정
        if 사전감지_조건:
            사전감지_count += 1
            # 예측값이 300 미만일 때만 +15 보정
            if prediction < 300:
                예측값내리기 = prediction + 15  # +15 보정
            else:
                예측값내리기 = prediction  # 이미 300 이상이면 보정 안 함
            
            if actual_value >= 300:
                보정적용_count += 1
        else:
            예측값내리기 = prediction  # 보정 없음
        
        # 300 이상 점프 감지
        jump_detected = np.any(seq_target >= 300)
       
        # 결과 저장
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '실제시점': actual_time.strftime('%Y-%m-%d %H:%M'),
            '실제값': actual_value,
            '예측값': round(prediction, 2),
            '예측값내리기': round(예측값내리기, 2),
            '오차': round(actual_value - 예측값내리기, 2),
            '오차율(%)': round(abs(actual_value - 예측값내리기) / max(actual_value, 1) * 100, 2),
            '시퀀스MAX': seq_max,
            '시퀀스MIN': seq_min,
            '시퀀스평균': round(np.mean(seq_target), 2),
            '시퀀스증가': round(increase_rate, 2),
            '사전감지': '사전감지' if 사전감지_조건 else '이상없음',
            '보정적용': '✅' if 사전감지_조건 else '',
            '300이상점프': '🔴' if jump_detected else '',
            '실제값상태': '🔴극단' if actual_value >= 300 else ('🟡주의' if actual_value >= 280 else '🟢정상'),
            '예측값상태': '🔴극단' if 예측값내리기 >= 300 else ('🟡주의' if 예측값내리기 >= 280 else '🟢정상')
        })
       
        # 진행상황 출력
        if (i - 30) % 100 == 0:
            print(f"진행중... {i-30}/{len(df)-30} ({(i-30)/(len(df)-30)*100:.1f}%)")
   
    # DataFrame 변환
    results_df = pd.DataFrame(results)
   
    # CSV 저장
    output_file = 'prediction_evaluation_사전감지보정_적용.csv'
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
    
    print("\n" + "="*80)
    print("🚨 사전감지 통계")
    print("="*80)
    print(f"사전감지 조건 만족: {사전감지_count}개")
    print(f"실제 300+ 점프: {(results_df['실제값'] >= 300).sum()}개")
    print(f"예측값내리기 300+: {(results_df['예측값내리기'] >= 300).sum()}개")
    print(f"사전감지 중 실제 300+ 점프: {보정적용_count}개")
    
    if 사전감지_count > 0:
        print(f"사전감지 정확도: {보정적용_count/사전감지_count*100:.1f}%")
    
    # 사전감지 케이스 상세
    사전감지_cases = results_df[results_df['사전감지'] == '사전감지']
    if len(사전감지_cases) > 0:
        print("\n" + "="*80)
        print(f"🚨 사전감지 케이스 상세 (상위 20개)")
        print("="*80)
        print(사전감지_cases[['현재시간', '실제값', '예측값', '예측값내리기', '오차', '실제값상태']].head(20).to_string(index=False))
    
    # 극단값 구간
    extreme_cases = results_df[results_df['실제값'] >= 300]
    if len(extreme_cases) > 0:
        print("\n" + "="*80)
        print("🔴 실제 극단값(≥300) 구간")
        print("="*80)
        print(extreme_cases[['현재시간', '실제값', '예측값내리기', '오차', '시퀀스MAX', '사전감지']].to_string(index=False))
   
    return results_df

if __name__ == '__main__':
    print("🚀 실시간 예측 평가 시작 (사전감지 보정 적용)...\n")
    results = evaluate_all_predictions()
   
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측 생성")
        print(f"📁 결과 파일: prediction_evaluation_사전감지보정_적용.csv")