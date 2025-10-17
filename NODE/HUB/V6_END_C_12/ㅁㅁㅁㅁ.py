# -*- coding: utf-8 -*-
"""
V6 모델로 예측 + 사전감지 분석
- 모델(pkl) 사용해서 예측
- 예측타겟시점(+10분)의 정확한 실제값 사용
- 사전감지 조건: 과거30개<300 AND 예측타겟시점 실제값≥300
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def predict_and_detect_early():
    """
    모델로 예측하고 사전감지 케이스 분석
    """
    
    # 핵심 12개 컬럼 정의
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
    }
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    print("="*80)
    print("🔍 V6 모델 예측 + 사전감지 분석")
    print("="*80)
    
    # 1. 모델 로드
    print("\n[STEP 1] 모델 로드")
    try:
        with open('xgboost_model_30min_10min_12컬럼.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 완료: xgboost_model_30min_10min_12컬럼.pkl")
    except:
        try:
            with open('xgboost_model_30min_10min.pkl', 'rb') as f:
                model = pickle.load(f)
            print("✅ 모델 로드 완료: xgboost_model_30min_10min.pkl")
        except Exception as e:
            print(f"❌ 모델 파일 없음: {e}")
            return None
    
    # 2. 원본 데이터 로드
    print("\n[STEP 2] 원본 데이터 로드")
    try:
        df = pd.read_csv('HUB0905101512.CSV', on_bad_lines='skip')
        print(f"✅ 데이터 로드 완료: {len(df)}개 행")
    except FileNotFoundError:
        print("❌ HUB0905101512.CSV 파일이 없습니다.")
        return None
    
    # 컬럼 확인
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
            print(f"✅ STAT_DT 변환 완료")
        except:
            print("⚠️ STAT_DT 변환 실패, 가상 시간 생성")
            base_time = datetime(2025, 9, 5, 0, 0)
            df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    else:
        print("⚠️ STAT_DT 컬럼 없음, 가상 시간 생성")
        base_time = datetime(2025, 9, 5, 0, 0)
        df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    
    # 3. 슬라이딩 윈도우로 예측 + 사전감지 분석
    print("\n[STEP 3] 슬라이딩 윈도우 예측 + 사전감지 분석")
    print("조건: 과거30개<300 AND 예측타겟시점(+10분) 실제값≥300")
    
    results = []
    early_detection_cases = []
    
    # i: 현재 시점 인덱스
    # i+10: 예측타겟시점 인덱스 (10분 후)
    for i in range(30, len(df) - 10):  # -10: 10분 후 데이터 필요
        # 과거 30개 데이터 (i-30 ~ i-1)
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
        
        # 현재 시점
        current_time = df['STAT_DT'].iloc[i]
        
        # 예측타겟시점 (10분 후)
        target_time = df['STAT_DT'].iloc[i+10]
        target_actual_value = df[TARGET_COL].iloc[i+10]
        
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
        
        # Net Flow
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
        
        # 모델 예측
        prediction = model.predict(X_pred)[0]
        
        # 시퀀스 통계
        seq_max = np.max(seq_target)
        seq_min = np.min(seq_target)
        seq_mean = np.mean(seq_target)
        seq_std = np.std(seq_target)
        
        # ★★★ 사전감지 조건 체크 ★★★
        # 과거 30개 모두 < 300 AND 예측타겟시점 실제값 >= 300
        is_early_detection = (seq_max < 300) and (target_actual_value >= 300)
        
        # 결과 저장
        result_row = {
            '인덱스': i,
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측타겟시점': target_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스MAX': round(seq_max, 2),
            '시퀀스MIN': round(seq_min, 2),
            '시퀀스평균': round(seq_mean, 2),
            '시퀀스STD': round(seq_std, 2),
            '예측타겟시점_실제값': round(target_actual_value, 2),
            '예측값': round(prediction, 2),
            '오차': round(target_actual_value - prediction, 2),
            '오차율(%)': round(abs(target_actual_value - prediction) / max(target_actual_value, 1) * 100, 2),
            '사전감지': '✅' if is_early_detection else ''
        }
        
        results.append(result_row)
        
        # 사전감지 케이스 별도 저장
        if is_early_detection:
            early_detection_cases.append({
                **result_row,
                '사전감지_성공여부': '✅ 성공' if prediction >= 290 else '❌ 실패',
                '사전감지_점수(%)': round(prediction / target_actual_value * 100, 2)
            })
        
        # 진행상황
        if (i - 30) % 100 == 0:
            print(f"  진행중... {i-30}/{len(df)-40} ({(i-30)/(len(df)-40)*100:.1f}%)")
    
    # 4. DataFrame 생성
    df_results = pd.DataFrame(results)
    
    print(f"\n✅ 예측 완료: {len(df_results)}개")
    
    # 5. 사전감지 케이스 분석
    if len(early_detection_cases) == 0:
        print("\n⚠️ 사전감지 케이스가 없습니다.")
        print("   (과거30개<300 AND 예측타겟시점≥300 조건)")
        
        # 전체 결과 저장
        output_file = '전체_예측결과.csv'
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 전체 결과 저장: {output_file}")
        return df_results
    
    df_early = pd.DataFrame(early_detection_cases)
    
    # 통계 출력
    print("\n" + "="*80)
    print("📊 사전감지 케이스 통계")
    print("="*80)
    
    success_count = (df_early['예측값'] >= 290).sum()
    success_rate = success_count / len(df_early) * 100
    
    print(f"전체 예측: {len(df_results)}개")
    print(f"사전감지 케이스: {len(df_early)}개 ({len(df_early)/len(df_results)*100:.2f}%)")
    print(f"\n사전감지 성공 (예측값≥290): {success_count}개 ({success_rate:.1f}%)")
    print(f"사전감지 실패 (예측값<290): {len(df_early) - success_count}개 ({100-success_rate:.1f}%)")
    print(f"\n평균 오차: {df_early['오차'].abs().mean():.2f}")
    print(f"평균 오차율: {df_early['오차율(%)'].mean():.2f}%")
    print(f"평균 사전감지 점수: {df_early['사전감지_점수(%)'].mean():.1f}%")
    
    # 6. 상세 출력
    print("\n" + "="*80)
    print("🔥 사전감지 케이스 상세 (상위 10개)")
    print("="*80)
    display_cols = ['현재시간', '예측타겟시점', '시퀀스MAX', '시퀀스평균', 
                    '예측타겟시점_실제값', '예측값', '오차', '사전감지_성공여부']
    print(df_early[display_cols].head(10).to_string(index=False))
    
    # 7. 성공/실패 케이스 분석
    success_df = df_early[df_early['예측값'] >= 290]
    failed_df = df_early[df_early['예측값'] < 290]
    
    if len(success_df) > 0:
        print("\n" + "="*80)
        print("✅ 사전감지 성공 케이스")
        print("="*80)
        print(success_df[display_cols].head(5).to_string(index=False))
        print(f"\n평균 시퀀스MAX: {success_df['시퀀스MAX'].mean():.2f}")
        print(f"평균 예측값: {success_df['예측값'].mean():.2f}")
        print(f"평균 실제값: {success_df['예측타겟시점_실제값'].mean():.2f}")
    
    if len(failed_df) > 0:
        print("\n" + "="*80)
        print("❌ 사전감지 실패 케이스")
        print("="*80)
        print(failed_df[display_cols].head(5).to_string(index=False))
        print(f"\n평균 시퀀스MAX: {failed_df['시퀀스MAX'].mean():.2f}")
        print(f"평균 예측값: {failed_df['예측값'].mean():.2f}")
        print(f"평균 실제값: {failed_df['예측타겟시점_실제값'].mean():.2f}")
    
    # 8. CSV 저장
    output_file1 = '사전감지_분석결과.csv'
    df_early.to_csv(output_file1, index=False, encoding='utf-8-sig')
    print(f"\n✅ 사전감지 결과 저장: {output_file1}")
    
    output_file2 = '전체_예측결과.csv'
    df_results.to_csv(output_file2, index=False, encoding='utf-8-sig')
    print(f"✅ 전체 결과 저장: {output_file2}")
    
    # 9. 최종 요약
    print("\n" + "="*80)
    print("📋 최종 요약")
    print("="*80)
    print(f"1. 사전감지 정의:")
    print(f"   - 과거 30개 데이터 모두 < 300")
    print(f"   - 예측타겟시점(+10분) 실제값 >= 300")
    print(f"\n2. 발견된 사전감지 케이스: {len(df_early)}개")
    print(f"3. 사전감지 성공률: {success_rate:.1f}% (예측값≥290 기준)")
    print(f"4. 평균 오차: {df_early['오차'].abs().mean():.2f}")
    print(f"5. 저장 파일:")
    print(f"   - 사전감지만: {output_file1}")
    print(f"   - 전체 결과: {output_file2}")
    
    print(f"\n💡 인사이트:")
    if success_rate >= 70:
        print(f"   ✅ 사전감지 성공률 {success_rate:.1f}%로 우수!")
    elif success_rate >= 50:
        print(f"   🟡 사전감지 성공률 {success_rate:.1f}%로 양호")
    else:
        print(f"   ❌ 사전감지 성공률 {success_rate:.1f}%로 개선 필요")
    
    return df_early

if __name__ == '__main__':
    print("🚀 V6 모델 예측 + 사전감지 분석 시작\n")
    results = predict_and_detect_early()
    
    if results is not None:
        print(f"\n✅ 분석 완료! 사전감지 케이스 {len(results)}개 발견")