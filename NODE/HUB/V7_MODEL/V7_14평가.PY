import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_v8_final_change_prediction():
    """
    🎯 V8-Final 변화량 예측 모델 평가
    - 10분 구간의 MIN, MAX, AVG 각각 평가
    - 큰 변화 기준: ±50
    - V8-Final: 32개 컬럼 (V7 19개 + V8 신규 13개, velocity 제외)
    """
    print("="*80)
    print("🎯 V8-Final 변화량 예측 모델 평가")
    print("="*80)
   
    FEATURE_COLS = {
        # ========== V7 기존 (17개) ==========
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
        'hub': ['HUBROOMTOTAL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA'],
        
        # V7 QUE (2개)
        'que': ['M16HUB.QUE.ALL.CURRENTQCNT', 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN'],
        
        # ========== V8 신규 (13개) ==========
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
        with open('xgboost_변화량예측_V8_Final.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 완료: xgboost_변화량예측_V8_Final.pkl")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        print("   먼저 'V8_Final_학습코드.py'를 실행하세요.")
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
   
    # 컬럼 존재 확인
    print("\n📋 컬럼 존재 확인:")
    available_cols = set(df.columns)
    total_cols = 0
    available_count = 0
    
    for group_name, cols in FEATURE_COLS.items():
        print(f"\n  [{group_name}] ({len(cols)}개)")
        for col in cols:
            total_cols += 1
            if col in available_cols:
                available_count += 1
                print(f"    ✅ {col}")
            else:
                print(f"    ❌ {col} - 누락 (Feature는 0으로 생성)")
    
    print(f"\n  총 컬럼: {available_count}/{total_cols}개 사용 가능")
   
    # STAT_DT 처리
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime(2024, 1, 1, 0, 0)
            df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
   
    results = []
   
    print("\n슬라이딩 윈도우 평가 시작 (V8-Final)...")
    
    total_samples = len(df) - 40
    
    for idx, i in enumerate(range(30, len(df) - 10)):
        if idx % 500 == 0:
            print(f"  평가 진행: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
        
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
       
        current_time = seq_data['STAT_DT'].iloc[-1]
        current_value = seq_target[-1]
        prediction_time = current_time + timedelta(minutes=10)
       
        # 🎯 정확히 10분 후 실제값 (i-1이 현재, i-1+10=i+9가 10분 후)
        actual_value = df[TARGET_COL].iloc[i+9]
        actual_change = actual_value - current_value
        
        # 🎯 10분 구간의 MIN, MAX, AVG (i부터 i+9까지 10개)
        future_10min = df[TARGET_COL].iloc[i:i+10].values
        actual_min = np.min(future_10min)
        actual_max = np.max(future_10min)
        actual_avg = np.mean(future_10min)
       
        # ===== Feature 생성 (학습과 동일!) =====
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
               
                else:  # inflow, outflow
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_last_value'] = col_seq[-1]
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
       
        # FS Storage Feature
        if all(col in available_cols for col in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
            use_seq = seq_data['CD_M163FSTORAGEUSE'].values
            total_seq = seq_data['CD_M163FSTORAGETOTAL'].values
            util_seq = seq_data['CD_M163FSTORAGEUTIL'].values
           
            features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
            features['storage_remaining'] = total_seq[-1] - use_seq[-1]
            features['storage_util_last'] = util_seq[-1]
            features['storage_util_high'] = 1 if util_seq[-1] >= 7 else 0
            features['storage_util_critical'] = 1 if util_seq[-1] >= 10 else 0
       
        # HUBROOMTOTAL Feature
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
       
        # 유입-유출, CMD
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
       
        # ===== V7 QUE Boolean =====
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
       
        # ===== 🆕 V8 M16HUB QUE Boolean =====
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
       
        # ===== 🆕 V8 M16A QUE Boolean =====
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
        
        # ===== 🚨 병목 예측 =====
        # 10분 후 병목 발생 예측 (현재 지표 기반)
        current_hub = df['HUBROOMTOTAL'].iloc[i-1] if 'HUBROOMTOTAL' in available_cols else 1000
        current_trans4min = df['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-1] if 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in available_cols else 0
        current_currentq = df['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[i-1] if 'M16HUB.QUE.ALL.CURRENTQCNT' in available_cols else 0
        
        # 병목 예측 조건
        pred_bottleneck_severe = 0  # 심각 병목 예측 (HUB < 550)
        pred_bottleneck_warning = 0  # 병목 경고 예측 (HUB < 590)
        pred_transport_high = 0      # 반송 지연 예측 (TRANS > 40)
        
        # 현재 상태 기반 예측
        if current_hub < 550:
            pred_bottleneck_severe = 1
            pred_bottleneck_warning = 1
        elif current_hub < 590:
            pred_bottleneck_warning = 1
        elif current_hub < 610 and current_trans4min > 35:
            pred_bottleneck_warning = 1  # 복합 위험
            
        if current_trans4min > 40:
            pred_transport_high = 1
        elif current_trans4min > 35 and current_hub < 610:
            pred_transport_high = 1  # 복합 위험
            
        # 10분 후 실제 병목 상태 (정확히 10분 후는 i+9)
        future_hub = df['HUBROOMTOTAL'].iloc[i+9] if i+9 < len(df) and 'HUBROOMTOTAL' in available_cols else 1000
        future_trans4min = df['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i+9] if i+9 < len(df) and 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in available_cols else 0
        
        actual_bottleneck_severe = 1 if future_hub < 550 else 0
        actual_bottleneck_warning = 1 if future_hub < 590 else 0
        actual_transport_high = 1 if future_trans4min > 40 else 0
        
        # 병목 예측 정확도
        bottleneck_severe_correct = (pred_bottleneck_severe == actual_bottleneck_severe)
        bottleneck_warning_correct = (pred_bottleneck_warning == actual_bottleneck_warning)
        transport_high_correct = (pred_transport_high == actual_transport_high)
       
        # V8 데이터 추가
        target_alt_val = df['CURRENT_M16A_3F_JOB'].iloc[i-1] if 'CURRENT_M16A_3F_JOB' in available_cols else 0
        m16hub_ohtutil_val = df['M16HUB.QUE.OHT.OHTUTIL'].iloc[i-1] if 'M16HUB.QUE.OHT.OHTUTIL' in available_cols else 0
        m16a_completed_val = df['M16A.QUE.ALL.CURRENTQCOMPLETED'].iloc[i-1] if 'M16A.QUE.ALL.CURRENTQCOMPLETED' in available_cols else 0
        m16a_created_val = df['M16A.QUE.ALL.CURRENTQCREATED'].iloc[i-1] if 'M16A.QUE.ALL.CURRENTQCREATED' in available_cols else 0
        m16a_loadtime_val = df['M16A.QUE.LOAD.AVGLOADTIME1MIN'].iloc[i-1] if 'M16A.QUE.LOAD.AVGLOADTIME1MIN' in available_cols else 0
        m16a_transport4min_val = df['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-1] if 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in available_cols else 0
        m16a_delay_val = df['M16A.QUE.ABN.QUETIMEDELAY'].iloc[i-1] if 'M16A.QUE.ABN.QUETIMEDELAY' in available_cols else 0
       
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            '실제값': round(actual_value, 2),
            '실제_변화량': round(actual_change, 2),
            # 구간 참고 정보
            '구간_MIN': round(actual_min, 2),
            '구간_MAX': round(actual_max, 2),
            '구간_AVG': round(actual_avg, 2),
            # MIN 예측
            '예측_변화량_MIN': round(pred_change_min, 2),
            '예측값_MIN': round(pred_min, 2),
            '오차_MIN': round(abs(actual_change - pred_change_min), 2),
            '방향일치_MIN': '✅' if direction_match_min else '❌',
            # MAX 예측
            '예측_변화량_MAX': round(pred_change_max, 2),
            '예측값_MAX': round(pred_max, 2),
            '오차_MAX': round(abs(actual_change - pred_change_max), 2),
            '방향일치_MAX': '✅' if direction_match_max else '❌',
            # AVG 예측
            '예측_변화량_AVG': round(pred_change_avg, 2),
            '예측값_AVG': round(pred_avg, 2),
            '오차_AVG': round(abs(actual_change - pred_change_avg), 2),
            '방향일치_AVG': '✅' if direction_match_avg else '❌',
            
            # ===== 🚨 병목 예측 컬럼 =====
            '현재_HUB': round(current_hub, 0),
            '10분후_실제_HUB': round(future_hub, 0),
            '예측_심각병목': '🔴' if pred_bottleneck_severe else '',
            '실제_심각병목': '🔴' if actual_bottleneck_severe else '',
            '심각병목_적중': '✅' if bottleneck_severe_correct else '❌',
            '예측_병목경고': '🟠' if pred_bottleneck_warning else '',
            '실제_병목경고': '🟠' if actual_bottleneck_warning else '',
            '병목경고_적중': '✅' if bottleneck_warning_correct else '❌',
            
            '현재_TRANS4MIN': round(current_trans4min, 0),
            '10분후_실제_TRANS4MIN': round(future_trans4min, 0),
            '예측_반송지연': '⚠️' if pred_transport_high else '',
            '실제_반송지연': '⚠️' if actual_transport_high else '',
            '반송지연_적중': '✅' if transport_high_correct else '❌',
            
            # V7 기타
            '시퀀스MAX': round(np.max(seq_target), 2),
            'HUBROOMTOTAL': round(features.get('HUBROOMTOTAL_last_value', 0), 0),
            'FS_UTIL': round(features.get('CD_M163FSTORAGEUTIL_last_value', 0), 0),
            'CMD': round(features.get('total_cmd', 0), 0),
            'Momentum가속': round(features.get('target_acceleration', 0), 2),
            'CURRENTQ': round(df['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[i-1] if 'M16HUB.QUE.ALL.CURRENTQCNT' in available_cols else 0, 0),
            'AVGTIME1MIN': round(df['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[i-1] if 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in available_cols else 0, 2),
            'QUE병목': '✅' if features.get('que_severe_bottleneck', 0) == 1 else '',
            # V8
            'TARGET_ALT': round(target_alt_val, 2),
            'M16HUB_OHTUTIL': round(m16hub_ohtutil_val, 1),
            'M16HUB병목': '✅' if features.get('m16hub_severe_bottleneck', 0) == 1 else '',
            'M16A_COMPLETED': round(m16a_completed_val, 0),
            'M16A_CREATED': round(m16a_created_val, 0),
            'M16A_LOADTIME1MIN': round(m16a_loadtime_val, 2),
            'M16A_TRANSPORT4MIN': round(m16a_transport4min_val, 0),
            'M16A_DELAY': round(m16a_delay_val, 0),
            'M16A병목': '✅' if features.get('m16a_severe_bottleneck', 0) == 1 else '',
        })
   
    results_df = pd.DataFrame(results)
   
    output_file = '변화량예측_V8_Final_평가결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
   
    # 통계
    print("\n" + "="*80)
    print("📊 평가 통계 (V8-Final)")
    print("="*80)
   
    print(f"총 예측: {len(results_df)}개")
   
    # 각 타입별 통계
    for target_type in ['MIN', 'MAX', 'AVG']:
        print(f"\n🎯 {target_type} 예측 성능:")
        
        error_col = f'오차_{target_type}'
        direction_col = f'방향일치_{target_type}'
        pred_col = f'예측_변화량_{target_type}'
        
        avg_error = results_df[error_col].mean()
        direction_correct = (results_df[direction_col] == '✅').sum()
        direction_accuracy = direction_correct / len(results_df) * 100
        
        print(f"  - 평균 오차: {avg_error:.2f}")
        print(f"  - 방향 정확도: {direction_correct}/{len(results_df)}개 ({direction_accuracy:.1f}%)")
        
        # 큰 변화 감지
        large_increase = results_df[results_df['실제_변화량'] >= 50]
        if len(large_increase) > 0:
            detected_increase = large_increase[large_increase[pred_col] >= 40]
            print(f"  - 큰 상승 (+50 이상) 감지:")
            print(f"    발생: {len(large_increase)}개")
            print(f"    감지 (+40 이상): {len(detected_increase)}개 ({len(detected_increase)/len(large_increase)*100:.1f}%)")
        
        large_decrease = results_df[results_df['실제_변화량'] <= -50]
        if len(large_decrease) > 0:
            detected_decrease = large_decrease[large_decrease[pred_col] <= -40]
            print(f"  - 큰 하락 (-50 이하) 감지:")
            print(f"    발생: {len(large_decrease)}개")
            print(f"    감지 (-40 이하): {len(detected_decrease)}개 ({len(detected_decrease)/len(large_decrease)*100:.1f}%)")
   
    # V8 Feature 분석
    print(f"\n🆕 V8 Feature 분석:")
    
    if 'TARGET_ALT' in results_df.columns and results_df['TARGET_ALT'].sum() > 0:
        print(f"  평균 TARGET_ALT: {results_df['TARGET_ALT'].mean():.1f}")
    
    if 'M16HUB_OHTUTIL' in results_df.columns and results_df['M16HUB_OHTUTIL'].sum() > 0:
        print(f"  평균 M16HUB_OHTUTIL: {results_df['M16HUB_OHTUTIL'].mean():.1f}%")
        m16hub_bottleneck_count = (results_df['M16HUB병목'] == '✅').sum()
        print(f"  M16HUB병목 발생: {m16hub_bottleneck_count}개 ({m16hub_bottleneck_count/len(results_df)*100:.1f}%)")
    
    if 'M16A_TRANSPORT4MIN' in results_df.columns and results_df['M16A_TRANSPORT4MIN'].sum() > 0:
        print(f"  평균 M16A_TRANSPORT4MIN: {results_df['M16A_TRANSPORT4MIN'].mean():.1f}개")
        
        # 큰 상승 시 TRANSPORT4MIN 관계
        large_increase = results_df[results_df['실제_변화량'] >= 50]
        if len(large_increase) > 0:
            avg_transport4min = large_increase['M16A_TRANSPORT4MIN'].mean()
            print(f"  큰 상승 시 평균 TRANSPORT4MIN: {avg_transport4min:.1f}개")
    
    if 'M16A_DELAY' in results_df.columns and results_df['M16A_DELAY'].sum() > 0:
        delay_count = (results_df['M16A_DELAY'] >= 1).sum()
        print(f"  M16A_DELAY 발생: {delay_count}개 ({delay_count/len(results_df)*100:.1f}%)")
    
    if 'M16A병목' in results_df.columns:
        m16a_bottleneck_count = (results_df['M16A병목'] == '✅').sum()
        print(f"  M16A병목 발생: {m16a_bottleneck_count}개 ({m16a_bottleneck_count/len(results_df)*100:.1f}%)")
        
        # 큰 상승 시 M16A 병목
        large_increase = results_df[results_df['실제_변화량'] >= 50]
        if len(large_increase) > 0:
            m16a_in_surge = (large_increase['M16A병목'] == '✅').sum()
            print(f"  큰 상승 시 M16A병목: {m16a_in_surge}/{len(large_increase)}개 ({m16a_in_surge/len(large_increase)*100:.1f}%)")
    
    # ===== 🚨 병목 예측 성능 분석 =====
    print(f"\n" + "="*80)
    print("🚨 병목 예측 성능")
    print("="*80)
    
    # 심각 병목 예측 (HUB < 550)
    severe_total = len(results_df)
    severe_correct = (results_df['심각병목_적중'] == '✅').sum()
    severe_accuracy = severe_correct / severe_total * 100
    
    severe_predicted = (results_df['예측_심각병목'] == '🔴').sum()
    severe_actual = (results_df['실제_심각병목'] == '🔴').sum()
    
    if severe_actual > 0:
        severe_detected = results_df[(results_df['예측_심각병목'] == '🔴') & (results_df['실제_심각병목'] == '🔴')].shape[0]
        severe_detection_rate = severe_detected / severe_actual * 100
    else:
        severe_detection_rate = 0
    
    print(f"\n🔴 심각 병목 예측 (HUB < 550):")
    print(f"  - 전체 정확도: {severe_correct}/{severe_total}개 ({severe_accuracy:.1f}%)")
    print(f"  - 예측 발생: {severe_predicted}개")
    print(f"  - 실제 발생: {severe_actual}개")
    if severe_actual > 0:
        print(f"  - 감지율: {severe_detected}/{severe_actual}개 ({severe_detection_rate:.1f}%)")
    
    # 병목 경고 예측 (HUB < 590)
    warning_total = len(results_df)
    warning_correct = (results_df['병목경고_적중'] == '✅').sum()
    warning_accuracy = warning_correct / warning_total * 100
    
    warning_predicted = (results_df['예측_병목경고'] == '🟠').sum()
    warning_actual = (results_df['실제_병목경고'] == '🟠').sum()
    
    if warning_actual > 0:
        warning_detected = results_df[(results_df['예측_병목경고'] == '🟠') & (results_df['실제_병목경고'] == '🟠')].shape[0]
        warning_detection_rate = warning_detected / warning_actual * 100
    else:
        warning_detection_rate = 0
    
    print(f"\n🟠 병목 경고 예측 (HUB < 590):")
    print(f"  - 전체 정확도: {warning_correct}/{warning_total}개 ({warning_accuracy:.1f}%)")
    print(f"  - 예측 발생: {warning_predicted}개")
    print(f"  - 실제 발생: {warning_actual}개")
    if warning_actual > 0:
        print(f"  - 감지율: {warning_detected}/{warning_actual}개 ({warning_detection_rate:.1f}%)")
    
    # 반송 지연 예측 (TRANS > 40)
    transport_total = len(results_df)
    transport_correct = (results_df['반송지연_적중'] == '✅').sum()
    transport_accuracy = transport_correct / transport_total * 100
    
    transport_predicted = (results_df['예측_반송지연'] == '⚠️').sum()
    transport_actual = (results_df['실제_반송지연'] == '⚠️').sum()
    
    if transport_actual > 0:
        transport_detected = results_df[(results_df['예측_반송지연'] == '⚠️') & (results_df['실제_반송지연'] == '⚠️')].shape[0]
        transport_detection_rate = transport_detected / transport_actual * 100
    else:
        transport_detection_rate = 0
    
    print(f"\n⚠️ 반송 지연 예측 (TRANSPORT4MIN > 40):")
    print(f"  - 전체 정확도: {transport_correct}/{transport_total}개 ({transport_accuracy:.1f}%)")
    print(f"  - 예측 발생: {transport_predicted}개")
    print(f"  - 실제 발생: {transport_actual}개")
    if transport_actual > 0:
        print(f"  - 감지율: {transport_detected}/{transport_actual}개 ({transport_detection_rate:.1f}%)")
    
    # 병목 예측 샘플
    print(f"\n📋 병목 예측 샘플 (최근 10개):")
    bottleneck_samples = results_df[
        (results_df['예측_심각병목'] == '🔴') | 
        (results_df['실제_심각병목'] == '🔴')
    ].tail(10)
    
    if len(bottleneck_samples) > 0:
        display_cols = ['현재시간', '현재_HUB', '10분후_실제_HUB', 
                       '예측_심각병목', '실제_심각병목', '심각병목_적중',
                       '현재_TRANS4MIN', '실제_변화량']
        available_cols = [col for col in display_cols if col in bottleneck_samples.columns]
        print(bottleneck_samples[available_cols].to_string(index=False))
   
    # 오차 상위 10개 (MAX 기준)
    print(f"\n❌ MAX 변화량 오차 상위 10개:")
    top_errors = results_df.nlargest(10, '오차_MAX')
    display_cols = ['현재시간', '현재값', '실제값', '실제_변화량', '예측_변화량_MAX', '오차_MAX', 
                    'HUBROOMTOTAL', '현재_HUB', '10분후_실제_HUB', '예측_심각병목', '실제_심각병목',
                    'M16A_TRANSPORT4MIN', '예측_반송지연', '실제_반송지연']
    available_cols = [col for col in display_cols if col in top_errors.columns]
    print(top_errors[available_cols].to_string(index=False))
   
    return results_df

if __name__ == '__main__':
    print("🚀 V8-Final 변화량 예측 모델 평가 시작...\n")
    results = evaluate_v8_final_change_prediction()
   
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측")
        print(f"📁 결과: 변화량예측_V8_Final_평가결과.csv")
        print(f"\n🎉 V8-Final: 32개 컬럼 (velocity 제외)!")
        print(f"🚨 병목 예측 기능 포함!")