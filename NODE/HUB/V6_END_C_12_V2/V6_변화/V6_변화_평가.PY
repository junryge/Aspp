import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_multi_change_prediction():
    """
    🎯 MIN/MAX/AVG 변화량 예측 모델 평가
    - 10분 구간의 MIN, MAX, AVG 각각 평가
    - 큰 변화 기준: ±50
    """
    print("="*80)
    print("🎯 MIN/MAX/AVG 변화량 예측 모델 평가")
    print("="*80)
   
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
        'hub': ['HUBROOMTOTAL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
    }
   
    # 모델 로드
    try:
        with open('xgboost_변화량예측_MIN_MAX_AVG.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 완료: xgboost_변화량예측_MIN_MAX_AVG.pkl")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        print("   먼저 'V7_변화량예측_MIN_MAX_AVG_학습코드.py'를 실행하세요.")
        return None
   
    # 데이터 로드
    try:
        df = pd.read_csv('HUB0905101512.csv', on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df = pd.read_csv('HUB0905101512.csv', on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df = pd.read_csv('HUB0905101512.csv', on_bad_lines='skip', encoding='euc-kr', low_memory=False)
   
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
   
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
   
    print(f"✅ 데이터 로드: {len(df)}개 행")
   
    # STAT_DT 처리
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime(2024, 1, 1, 0, 0)
            df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
   
    results = []
   
    print("\n슬라이딩 윈도우 평가 시작 (MIN/MAX/AVG 변화량)...")
    
    # ✅ 10분 후 데이터 필요
    for i in range(30, len(df) - 10):
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
       
        current_time = seq_data['STAT_DT'].iloc[-1]
        current_value = seq_target[-1]
        prediction_time = current_time + timedelta(minutes=10)
       
        # 🎯 10분 구간의 MIN, MAX, AVG
        future_10min = df[TARGET_COL].iloc[i:i+10].values
        actual_min = np.min(future_10min)
        actual_max = np.max(future_10min)
        actual_avg = np.mean(future_10min)
        
        # 🎯 실제 변화량
        actual_change_min = actual_min - current_value
        actual_change_max = actual_max - current_value
        actual_change_avg = actual_avg - current_value
       
        # Feature 생성 (학습 코드와 동일!)
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': seq_target[-1],
            'target_last_5_mean': np.mean(seq_target[-5:]),
            'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
        }
       
        # Momentum Feature
        features['target_acceleration'] = (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5
        features['target_is_rising'] = 1 if seq_target[-1] > seq_target[-5] else 0
        features['target_rapid_rise'] = 1 if (seq_target[-1] - seq_target[-5] > 10) else 0
        features['target_last_10_mean'] = np.mean(seq_target[-10:])
       
        # 각 컬럼 Feature
        for group_name, cols in FEATURE_COLS.items():
            for col in cols:
                if col not in df.columns:
                    continue
               
                col_seq = seq_data[col].values
               
                if group_name == 'maxcapa':
                    features[f'{col}_last_value'] = col_seq[-1]
               
                elif group_name in ['cmd', 'storage', 'fs_storage', 'hub']:
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_std'] = np.std(col_seq)
                    features[f'{col}_max'] = np.max(col_seq)
                    features[f'{col}_min'] = np.min(col_seq)
                    features[f'{col}_last_value'] = col_seq[-1]
                    features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
               
                else:
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_last_value'] = col_seq[-1]
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
       
        # FS Storage Feature
        if 'CD_M163FSTORAGEUSE' in df.columns and 'CD_M163FSTORAGETOTAL' in df.columns and 'CD_M163FSTORAGEUTIL' in df.columns:
            use_seq = seq_data['CD_M163FSTORAGEUSE'].values
            total_seq = seq_data['CD_M163FSTORAGETOTAL'].values
            util_seq = seq_data['CD_M163FSTORAGEUTIL'].values
           
            # CD_M163FSTORAGEUSE
            features['CD_M163FSTORAGEUSE_mean'] = np.mean(use_seq)
            features['CD_M163FSTORAGEUSE_std'] = np.std(use_seq)
            features['CD_M163FSTORAGEUSE_max'] = np.max(use_seq)
            features['CD_M163FSTORAGEUSE_min'] = np.min(use_seq)
            features['CD_M163FSTORAGEUSE_last_value'] = use_seq[-1]
            features['CD_M163FSTORAGEUSE_last_5_mean'] = np.mean(use_seq[-5:])
            features['CD_M163FSTORAGEUSE_slope'] = np.polyfit(np.arange(30), use_seq, 1)[0]
           
            # CD_M163FSTORAGETOTAL
            features['CD_M163FSTORAGETOTAL_mean'] = np.mean(total_seq)
            features['CD_M163FSTORAGETOTAL_std'] = np.std(total_seq)
            features['CD_M163FSTORAGETOTAL_max'] = np.max(total_seq)
            features['CD_M163FSTORAGETOTAL_min'] = np.min(total_seq)
            features['CD_M163FSTORAGETOTAL_last_value'] = total_seq[-1]
            features['CD_M163FSTORAGETOTAL_last_5_mean'] = np.mean(total_seq[-5:])
            features['CD_M163FSTORAGETOTAL_slope'] = np.polyfit(np.arange(30), total_seq, 1)[0]
           
            # CD_M163FSTORAGEUTIL
            features['CD_M163FSTORAGEUTIL_mean'] = np.mean(util_seq)
            features['CD_M163FSTORAGEUTIL_std'] = np.std(util_seq)
            features['CD_M163FSTORAGEUTIL_max'] = np.max(util_seq)
            features['CD_M163FSTORAGEUTIL_min'] = np.min(util_seq)
            features['CD_M163FSTORAGEUTIL_last_value'] = util_seq[-1]
            features['CD_M163FSTORAGEUTIL_last_5_mean'] = np.mean(util_seq[-5:])
            features['CD_M163FSTORAGEUTIL_slope'] = np.polyfit(np.arange(30), util_seq, 1)[0]
           
            # 특수 Feature
            features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
            features['storage_remaining'] = total_seq[-1] - use_seq[-1]
            features['storage_util_last'] = util_seq[-1]
            features['storage_util_high'] = 1 if util_seq[-1] >= 7 else 0
            features['storage_util_critical'] = 1 if util_seq[-1] >= 10 else 0
       
        # HUBROOMTOTAL Feature
        if 'HUBROOMTOTAL' in df.columns:
            hub_seq = seq_data['HUBROOMTOTAL'].values
            hub_last = hub_seq[-1]
           
            features['hub_critical'] = 1 if hub_last < 590 else 0
            features['hub_high'] = 1 if hub_last < 610 else 0
            features['hub_warning'] = 1 if hub_last < 620 else 0
            features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
           
            if 'CD_M163FSTORAGEUTIL' in df.columns:
                storage_util_last = df['CD_M163FSTORAGEUTIL'].iloc[i-1]
                features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
       
        # 유입-유출, CMD
        inflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['inflow'] if col in df.columns)
        outflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['outflow'] if col in df.columns)
        features['net_flow'] = inflow_sum - outflow_sum
       
        cmd_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['cmd'] if col in df.columns)
        features['total_cmd'] = cmd_sum
        features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
        features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
       
        # HUB × CMD 복합
        if 'HUBROOMTOTAL' in df.columns:
            hub_last = df['HUBROOMTOTAL'].iloc[i-1]
            features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
       
        # Storage Util 위험
        if 'M16A_3F_STORAGE_UTIL' in df.columns:
            storage_util = df['M16A_3F_STORAGE_UTIL'].iloc[i-1]
            features['storage_util_critical'] = 1 if storage_util >= 205 else 0
            features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
       
        # 급증 위험도 점수
        features['surge_risk_score'] = (
            features.get('hub_high', 0) * 3 +
            features.get('storage_util_critical', 0) * 2 +
            features.get('total_cmd_low', 0) * 1 +
            features.get('storage_util_high', 0) * 1
        )
       
        # 급증 임박 신호
        features['surge_imminent'] = 1 if (
            seq_target[-1] > 280 and
            features.get('target_acceleration', 0) > 0.5 and
            features.get('hub_high', 0) == 1
        ) else 0
       
        X_pred = pd.DataFrame([features])
       
        # 🎯 예측 (변화량 3개!)
        predictions = model.predict(X_pred)[0]
        pred_change_min = predictions[0]
        pred_change_max = predictions[1]
        pred_change_avg = predictions[2]
        
        # 🎯 예측값 = 현재값 + 예측 변화량
        pred_min = current_value + pred_change_min
        pred_max = current_value + pred_change_max
        pred_avg = current_value + pred_change_avg
        
        # 방향 일치 여부
        direction_match_min = (
            (actual_change_min > 0 and pred_change_min > 0) or
            (actual_change_min < 0 and pred_change_min < 0) or
            (abs(actual_change_min) < 5 and abs(pred_change_min) < 5)
        )
        direction_match_max = (
            (actual_change_max > 0 and pred_change_max > 0) or
            (actual_change_max < 0 and pred_change_max < 0) or
            (abs(actual_change_max) < 5 and abs(pred_change_max) < 5)
        )
        direction_match_avg = (
            (actual_change_avg > 0 and pred_change_avg > 0) or
            (actual_change_avg < 0 and pred_change_avg < 0) or
            (abs(actual_change_avg) < 5 and abs(pred_change_avg) < 5)
        )
       
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            # MIN
            '실제_MIN': round(actual_min, 2),
            '실제_변화량_MIN': round(actual_change_min, 2),
            '예측_변화량_MIN': round(pred_change_min, 2),
            '예측값_MIN': round(pred_min, 2),
            '오차_MIN': round(abs(actual_change_min - pred_change_min), 2),
            '방향일치_MIN': '✅' if direction_match_min else '❌',
            # MAX
            '실제_MAX': round(actual_max, 2),
            '실제_변화량_MAX': round(actual_change_max, 2),
            '예측_변화량_MAX': round(pred_change_max, 2),
            '예측값_MAX': round(pred_max, 2),
            '오차_MAX': round(abs(actual_change_max - pred_change_max), 2),
            '방향일치_MAX': '✅' if direction_match_max else '❌',
            # AVG
            '실제_AVG': round(actual_avg, 2),
            '실제_변화량_AVG': round(actual_change_avg, 2),
            '예측_변화량_AVG': round(pred_change_avg, 2),
            '예측값_AVG': round(pred_avg, 2),
            '오차_AVG': round(abs(actual_change_avg - pred_change_avg), 2),
            '방향일치_AVG': '✅' if direction_match_avg else '❌',
            # 기타
            '시퀀스MAX': round(np.max(seq_target), 2),
            'HUBROOMTOTAL': round(features.get('HUBROOMTOTAL_last_value', 0), 0),
            'FS_UTIL': round(features.get('CD_M163FSTORAGEUTIL_last_value', 0), 0),
            'CMD': round(features.get('total_cmd', 0), 0),
            'Momentum가속': round(features.get('target_acceleration', 0), 2),
        })
       
        if (i - 30) % 1000 == 0:
            print(f"진행: {i-30}/{len(df)-40} ({(i-30)/(len(df)-40)*100:.1f}%)")
   
    results_df = pd.DataFrame(results)
   
    output_file = '변화량예측_MIN_MAX_AVG_평가결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
   
    # 통계
    print("\n" + "="*80)
    print("📊 평가 통계 (MIN/MAX/AVG 변화량)")
    print("="*80)
   
    print(f"총 예측: {len(results_df)}개")
   
    # 각 타입별 통계
    for target_type in ['MIN', 'MAX', 'AVG']:
        print(f"\n🎯 {target_type} 변화량:")
        
        error_col = f'오차_{target_type}'
        direction_col = f'방향일치_{target_type}'
        actual_col = f'실제_변화량_{target_type}'
        pred_col = f'예측_변화량_{target_type}'
        
        avg_error = results_df[error_col].mean()
        direction_correct = (results_df[direction_col] == '✅').sum()
        direction_accuracy = direction_correct / len(results_df) * 100
        
        print(f"  - 평균 오차: {avg_error:.2f}")
        print(f"  - 방향 정확도: {direction_correct}/{len(results_df)}개 ({direction_accuracy:.1f}%)")
        
        # 큰 변화 감지 (기준: ±50)
        large_increase = results_df[results_df[actual_col] >= 50]
        if len(large_increase) > 0:
            detected_increase = large_increase[large_increase[pred_col] >= 40]
            print(f"  - 큰 상승 (+50 이상):")
            print(f"    발생: {len(large_increase)}개")
            print(f"    감지 (+40 이상): {len(detected_increase)}개 ({len(detected_increase)/len(large_increase)*100:.1f}%)")
        
        large_decrease = results_df[results_df[actual_col] <= -50]
        if len(large_decrease) > 0:
            detected_decrease = large_decrease[large_decrease[pred_col] <= -40]
            print(f"  - 큰 하락 (-50 이하):")
            print(f"    발생: {len(large_decrease)}개")
            print(f"    감지 (-40 이하): {len(detected_decrease)}개 ({len(detected_decrease)/len(large_decrease)*100:.1f}%)")
   
    # 오차 상위 10개 (MAX 기준)
    print(f"\n❌ MAX 변화량 오차 상위 10개:")
    top_errors = results_df.nlargest(10, '오차_MAX')
    print(top_errors[['현재시간', '현재값', '실제_변화량_MAX', '예측_변화량_MAX', '오차_MAX', 'HUBROOMTOTAL']].to_string(index=False))
   
    return results_df

if __name__ == '__main__':
    print("🚀 MIN/MAX/AVG 변화량 예측 모델 평가 시작...\n")
    results = evaluate_multi_change_prediction()
   
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측")
        print(f"📁 결과: 변화량예측_MIN_MAX_AVG_평가결과.csv")