'''
# 모델 로드
try:
    with open('xgboost_수치형_V8.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ 모델 로드: xgboost_수치형_V8.pkl")
except Exception as e:
    print(f"❌ 모델 없음: {e}")
    return None


try:
    with open('xgboost_수치형_V8.pkl', 'rb') as f:
        model_dict = pickle.load(f)
    models = model_dict['models']  # 3개 모델 리스트 [MIN, MAX, AVG]
    print("✅ 모델 로드: xgboost_수치형_V8.pkl (3개 모델)")
except Exception as e:
    print(f"❌ 모델 없음: {e}")
    return None
'''



import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_v8_regression():
    """
    🎯 V8-수치형 변화량 예측 모델 평가
    """
    print("="*80)
    print("🎯 V8-수치형 변화량 예측 모델 평가")
    print("="*80)
   
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
        'hub': ['HUBROOMTOTAL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA'],
        'que': ['M16HUB.QUE.ALL.CURRENTQCNT', 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN'],
        'target_alt': ['CURRENT_M16A_3F_JOB'],
        'm16hub_que': [
            'M16HUB.QUE.ALL.CURRENTQCOMPLETED',
            'M16HUB.QUE.ALL.FABTRANSJOBCNT',
            'M16HUB.QUE.TIME.AVGTOTALTIME',
            'M16HUB.QUE.OHT.CURRENTOHTQCNT',
            'M16HUB.QUE.OHT.OHTUTIL'
        ],
        'm16a_que': [
            'M16A.QUE.ALL.CURRENTQCOMPLETED',
            'M16A.QUE.ALL.CURRENTQCREATED',
            'M16A.QUE.OHT.CURRENTOHTQCNT',
            'M16A.QUE.OHT.OHTUTIL',
            'M16A.QUE.LOAD.AVGLOADTIME1MIN',
            'M16A.QUE.ALL.TRANSPORT4MINOVERCNT',
            'M16A.QUE.ABN.QUETIMEDELAY'
        ]
    }
   
    # 모델 로드
    try:
        with open('xgboost_수치형_V8.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드: xgboost_수치형_V8.pkl")
    except Exception as e:
        print(f"❌ 모델 없음: {e}")
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
   
    available_cols = set(df.columns)
   
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime(2024, 1, 1, 0, 0)
            df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
   
    results = []
   
    print("\n평가 시작...")
    
    total_samples = len(df) - 40
    
    for idx, i in enumerate(range(30, len(df) - 10)):
        if idx % 500 == 0:
            print(f"  진행: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
        
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
       
        current_time = seq_data['STAT_DT'].iloc[-1]
        current_value = seq_target[-1]
        prediction_time = current_time + timedelta(minutes=10)
       
        actual_value = df[TARGET_COL].iloc[i+9]
        actual_change = actual_value - current_value
        
        future_10min = df[TARGET_COL].iloc[i:i+10].values
        actual_min = np.min(future_10min)
        actual_max = np.max(future_10min)
        actual_avg = np.mean(future_10min)
       
        # Feature 생성 (학습과 동일)
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': seq_target[-1],
            'target_last_5_mean': np.mean(seq_target[-5:]),
            'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
            'target_acceleration': (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5,
            'target_is_rising': 1 if seq_target[-1] > seq_target[-5] else 0,
            'target_rapid_rise': 1 if (seq_target[-1] - seq_target[-5] > 10) else 0,
            'target_last_10_mean': np.mean(seq_target[-10:])
        }
       
        for group_name, cols in FEATURE_COLS.items():
            for col in cols:
                if col not in available_cols:
                    continue
               
                col_seq = seq_data[col].values
               
                if group_name == 'maxcapa':
                    features[f'{col}_last_value'] = col_seq[-1]
                elif group_name in ['cmd', 'storage', 'fs_storage', 'hub', 'que', 
                                   'target_alt', 'm16hub_que', 'm16a_que']:
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
       
        if all(col in available_cols for col in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
            use_seq = seq_data['CD_M163FSTORAGEUSE'].values
            total_seq = seq_data['CD_M163FSTORAGETOTAL'].values
            util_seq = seq_data['CD_M163FSTORAGEUTIL'].values
            features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
            features['storage_remaining'] = total_seq[-1] - use_seq[-1]
            features['storage_util_last'] = util_seq[-1]
            features['storage_util_high'] = 1 if util_seq[-1] >= 7 else 0
            features['storage_util_critical'] = 1 if util_seq[-1] >= 10 else 0
       
        if 'HUBROOMTOTAL' in available_cols:
            hub_seq = seq_data['HUBROOMTOTAL'].values
            hub_last = hub_seq[-1]
            features['hub_critical'] = 1 if hub_last < 590 else 0
            features['hub_high'] = 1 if hub_last < 610 else 0
            features['hub_warning'] = 1 if hub_last < 620 else 0
            features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
            if 'CD_M163FSTORAGEUTIL' in available_cols:
                storage_util_last = df['CD_M163FSTORAGEUTIL'].iloc[i-1]
                features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
       
        inflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['inflow'] if col in available_cols)
        outflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['outflow'] if col in available_cols)
        features['net_flow'] = inflow_sum - outflow_sum
       
        cmd_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['cmd'] if col in available_cols)
        features['total_cmd'] = cmd_sum
        features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
        features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
       
        if 'HUBROOMTOTAL' in available_cols:
            hub_last = df['HUBROOMTOTAL'].iloc[i-1]
            features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
       
        if 'M16A_3F_STORAGE_UTIL' in available_cols:
            storage_util = df['M16A_3F_STORAGE_UTIL'].iloc[i-1]
            features['storage_util_critical'] = 1 if storage_util >= 205 else 0
            features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
       
        features['surge_risk_score'] = (
            features.get('hub_high', 0) * 3 +
            features.get('storage_util_critical', 0) * 2 +
            features.get('total_cmd_low', 0) * 1 +
            features.get('storage_util_high', 0) * 1
        )
       
        features['surge_imminent'] = 1 if (
            seq_target[-1] > 280 and
            features.get('target_acceleration', 0) > 0.5 and
            features.get('hub_high', 0) == 1
        ) else 0
       
        # Boolean features (생략 - 위와 동일)
        if 'M16HUB.QUE.ALL.CURRENTQCNT' in available_cols:
            currentq = df['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[i-1]
            features['currentq_high'] = 1 if currentq >= 1200 else 0
            features['currentq_critical'] = 1 if currentq >= 1400 else 0
        else:
            features['currentq_high'] = 0
            features['currentq_critical'] = 0
       
        if 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in available_cols:
            avgtime = df['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[i-1]
            features['avgtime1min_high'] = 1 if avgtime >= 4.0 else 0
            features['avgtime1min_critical'] = 1 if avgtime >= 4.5 else 0
        else:
            features['avgtime1min_high'] = 0
            features['avgtime1min_critical'] = 0
       
        if 'M16HUB.QUE.ALL.CURRENTQCNT' in available_cols and 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in available_cols:
            currentq = df['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[i-1]
            avgtime = df['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[i-1]
            features['que_severe_bottleneck'] = 1 if (currentq >= 1200 and avgtime >= 4.0) else 0
        else:
            features['que_severe_bottleneck'] = 0
       
        if 'M16HUB.QUE.OHT.OHTUTIL' in available_cols:
            ohtutil = df['M16HUB.QUE.OHT.OHTUTIL'].iloc[i-1]
            features['m16hub_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
            features['m16hub_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
        else:
            features['m16hub_ohtutil_high'] = 0
            features['m16hub_ohtutil_critical'] = 0
       
        if 'M16HUB.QUE.TIME.AVGTOTALTIME' in available_cols:
            avgtime = df['M16HUB.QUE.TIME.AVGTOTALTIME'].iloc[i-1]
            features['m16hub_avgtime_high'] = 1 if avgtime >= 5.0 else 0
            features['m16hub_avgtime_critical'] = 1 if avgtime >= 6.0 else 0
        else:
            features['m16hub_avgtime_high'] = 0
            features['m16hub_avgtime_critical'] = 0
       
        if 'M16HUB.QUE.OHT.OHTUTIL' in available_cols and 'M16HUB.QUE.TIME.AVGTOTALTIME' in available_cols:
            ohtutil = df['M16HUB.QUE.OHT.OHTUTIL'].iloc[i-1]
            avgtime = df['M16HUB.QUE.TIME.AVGTOTALTIME'].iloc[i-1]
            features['m16hub_severe_bottleneck'] = 1 if (ohtutil >= 85.0 and avgtime >= 5.0) else 0
        else:
            features['m16hub_severe_bottleneck'] = 0
       
        if 'M16A.QUE.OHT.OHTUTIL' in available_cols:
            ohtutil = df['M16A.QUE.OHT.OHTUTIL'].iloc[i-1]
            features['m16a_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
            features['m16a_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
        else:
            features['m16a_ohtutil_high'] = 0
            features['m16a_ohtutil_critical'] = 0
       
        if 'M16A.QUE.LOAD.AVGLOADTIME1MIN' in available_cols:
            loadtime = df['M16A.QUE.LOAD.AVGLOADTIME1MIN'].iloc[i-1]
            features['m16a_loadtime_high'] = 1 if loadtime >= 2.5 else 0
            features['m16a_loadtime_critical'] = 1 if loadtime >= 2.8 else 0
        else:
            features['m16a_loadtime_high'] = 0
            features['m16a_loadtime_critical'] = 0
       
        if 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in available_cols:
            transport4min = df['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-1]
            features['m16a_transport4min_high'] = 1 if transport4min >= 40 else 0
            features['m16a_transport4min_critical'] = 1 if transport4min >= 50 else 0
        else:
            features['m16a_transport4min_high'] = 0
            features['m16a_transport4min_critical'] = 0
       
        if 'M16A.QUE.ABN.QUETIMEDELAY' in available_cols:
            delay = df['M16A.QUE.ABN.QUETIMEDELAY'].iloc[i-1]
            features['m16a_delay_warning'] = 1 if delay >= 1 else 0
            features['m16a_delay_critical'] = 1 if delay >= 3 else 0
        else:
            features['m16a_delay_warning'] = 0
            features['m16a_delay_critical'] = 0
       
        if 'M16A.QUE.OHT.OHTUTIL' in available_cols and 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in available_cols:
            ohtutil = df['M16A.QUE.OHT.OHTUTIL'].iloc[i-1]
            transport4min = df['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-1]
            features['m16a_severe_bottleneck'] = 1 if (ohtutil >= 85.0 and transport4min >= 40) else 0
        else:
            features['m16a_severe_bottleneck'] = 0
       
        X_pred = pd.DataFrame([features])
       
        # 예측
        predictions = model.predict(X_pred)[0]
        pred_change_min = predictions[0]
        pred_change_max = predictions[1]
        pred_change_avg = predictions[2]
        
        pred_min = current_value + pred_change_min
        pred_max = current_value + pred_change_max
        pred_avg = current_value + pred_change_avg
        
        direction_match_min = (
            (actual_change > 0 and pred_change_min > 0) or
            (actual_change < 0 and pred_change_min < 0) or
            (abs(actual_change) < 5 and abs(pred_change_min) < 5)
        )
        direction_match_max = (
            (actual_change > 0 and pred_change_max > 0) or
            (actual_change < 0 and pred_change_max < 0) or
            (abs(actual_change) < 5 and abs(pred_change_max) < 5)
        )
        direction_match_avg = (
            (actual_change > 0 and pred_change_avg > 0) or
            (actual_change < 0 and pred_change_avg < 0) or
            (abs(actual_change) < 5 and abs(pred_change_avg) < 5)
        )
       
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            '실제값': round(actual_value, 2),
            '실제_변화량': round(actual_change, 2),
            '구간_MIN': round(actual_min, 2),
            '구간_MAX': round(actual_max, 2),
            '구간_AVG': round(actual_avg, 2),
            '예측_변화량_MIN': round(pred_change_min, 2),
            '예측값_MIN': round(pred_min, 2),
            '오차_MIN': round(abs(actual_change - pred_change_min), 2),
            '방향일치_MIN': '✅' if direction_match_min else '❌',
            '예측_변화량_MAX': round(pred_change_max, 2),
            '예측값_MAX': round(pred_max, 2),
            '오차_MAX': round(abs(actual_change - pred_change_max), 2),
            '방향일치_MAX': '✅' if direction_match_max else '❌',
            '예측_변화량_AVG': round(pred_change_avg, 2),
            '예측값_AVG': round(pred_avg, 2),
            '오차_AVG': round(abs(actual_change - pred_change_avg), 2),
            '방향일치_AVG': '✅' if direction_match_avg else '❌',
            'HUBROOMTOTAL': round(features.get('HUBROOMTOTAL_last_value', 0), 0),
        })
   
    results_df = pd.DataFrame(results)
   
    output_file = '수치형_V8_평가결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
   
    # 통계
    print("\n" + "="*80)
    print("📊 평가 통계 (V8-수치형)")
    print("="*80)
   
    print(f"총 예측: {len(results_df)}개")
   
    for target_type in ['MIN', 'MAX', 'AVG']:
        print(f"\n🎯 {target_type} 예측:")
        
        error_col = f'오차_{target_type}'
        direction_col = f'방향일치_{target_type}'
        pred_col = f'예측_변화량_{target_type}'
        
        avg_error = results_df[error_col].mean()
        direction_correct = (results_df[direction_col] == '✅').sum()
        direction_accuracy = direction_correct / len(results_df) * 100
        
        print(f"  - 평균 오차: {avg_error:.2f}")
        print(f"  - 방향 정확도: {direction_accuracy:.1f}%")
        
        # 큰 변화 감지
        large_increase = results_df[results_df['실제_변화량'] >= 50]
        if len(large_increase) > 0:
            detected = large_increase[large_increase[pred_col] >= 40]
            print(f"  - 큰 상승(+50) 감지: {len(detected)}/{len(large_increase)}개 ({len(detected)/len(large_increase)*100:.1f}%)")
        
        large_decrease = results_df[results_df['실제_변화량'] <= -50]
        if len(large_decrease) > 0:
            detected = large_decrease[large_decrease[pred_col] <= -40]
            print(f"  - 큰 하락(-50) 감지: {len(detected)}/{len(large_decrease)}개 ({len(detected)/len(large_decrease)*100:.1f}%)")
   
    return results_df

if __name__ == '__main__':
    print("🚀 V8-수치형 평가 시작...\n")
    results = evaluate_v8_regression()
   
    if results is not None:
        print(f"\n✅ 평가 완료!")
        print(f"📁 결과: 수치형_V8_평가결과.csv")