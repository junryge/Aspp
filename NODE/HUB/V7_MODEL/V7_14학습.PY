import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def train_v8_final_change_prediction():
    """
    🎯 V8-Final 변화량 예측: 30분 시퀀스 → 10분 후 MIN/MAX/AVG 변화량 예측
    - 타겟: [MIN 변화량, MAX 변화량, AVG 변화량]
    - 큰 변화 기준: ±50
    - 멀티아웃풋 예측
    - V8-Final: 32개 컬럼 (V7 19개 + V8 신규 13개, velocity 제외)
    """
    print("="*80)
    print("🎯 V8-Final 변화량 예측: MIN/MAX/AVG 동시 예측 (32개 컬럼)")
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
   
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
   
    def create_features_v8_final(df, available_cols, start_idx=30):
        """V8-Final Feature 생성 (컬럼 존재 여부 체크)"""
        features_list = []
        labels = []
        current_values = []
        future_actual_values = []
        future_min_values = []
        future_max_values = []
        future_avg_values = []
        indices = []
       
        total_samples = len(df) - 10 - start_idx
        
        for idx, i in enumerate(range(start_idx, len(df) - 10)):
            if idx % 1000 == 0:
                print(f"  Feature 생성 진행: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
            
            seq_target = df[TARGET_COL].iloc[i-30:i].values
            current_value = seq_target[-1]
           
            # 🎯 정확히 10분 후 실제값 (i-1이 현재, i-1+10=i+9가 10분 후)
            future_actual = df[TARGET_COL].iloc[i+9]
            
            # 🎯 10분 구간의 MIN, MAX, AVG (i부터 i+9까지 10개)
            future_10min = df[TARGET_COL].iloc[i:i+10].values
            future_min = np.min(future_10min)
            future_max = np.max(future_10min)
            future_avg = np.mean(future_10min)
           
            actual_change = future_actual - current_value
            change_min = future_min - current_value
            change_max = future_max - current_value
            change_avg = future_avg - current_value
           
            # ===== 타겟 기본 통계 (11개) =====
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
           
            # ===== 각 컬럼 그룹 Feature =====
            for group_name, cols in FEATURE_COLS.items():
                for col in cols:
                    # 컬럼 존재 여부 체크
                    if col not in available_cols:
                        continue
                   
                    col_seq = df[col].iloc[i-30:i].values
                   
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
           
            # ===== FS Storage 특수 Feature (5개) =====
            if all(col in available_cols for col in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
                storage_use = df['CD_M163FSTORAGEUSE'].iloc[i-30:i].values
                storage_total = df['CD_M163FSTORAGETOTAL'].iloc[i-30:i].values
                storage_util = df['CD_M163FSTORAGEUTIL'].iloc[i-30:i].values
               
                features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
                features['storage_remaining'] = storage_total[-1] - storage_use[-1]
                features['storage_util_last'] = storage_util[-1]
                features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
                features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
           
            # ===== HUBROOMTOTAL 특수 Feature (5개) =====
            if 'HUBROOMTOTAL' in available_cols:
                hub_seq = df['HUBROOMTOTAL'].iloc[i-30:i].values
                hub_last = hub_seq[-1]
               
                features['hub_critical'] = 1 if hub_last < 590 else 0
                features['hub_high'] = 1 if hub_last < 610 else 0
                features['hub_warning'] = 1 if hub_last < 620 else 0
                features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
               
                if 'CD_M163FSTORAGEUTIL' in available_cols:
                    storage_util_last = df['CD_M163FSTORAGEUTIL'].iloc[i-1]
                    features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
           
            # ===== 복합 Feature (7개) =====
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
           
            # ===== 급증 위험도 점수 (1개) =====
            features['surge_risk_score'] = (
                features.get('hub_high', 0) * 3 +
                features.get('storage_util_critical', 0) * 2 +
                features.get('total_cmd_low', 0) * 1 +
                features.get('storage_util_high', 0) * 1
            )
           
            # ===== 급증 임박 신호 (1개) =====
            features['surge_imminent'] = 1 if (
                seq_target[-1] > 280 and
                features.get('target_acceleration', 0) > 0.5 and
                features.get('hub_high', 0) == 1
            ) else 0
           
            # ===== V7 QUE Boolean (5개) =====
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
           
            # ===== 🆕 V8 M16HUB QUE Boolean (5개) =====
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
           
            # ===== 🆕 V8 M16A QUE Boolean (9개) =====
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
           
            features_list.append(features)
            labels.append([change_min, change_max, change_avg])
            current_values.append(current_value)
            future_actual_values.append(future_actual)
            future_min_values.append(future_min)
            future_max_values.append(future_max)
            future_avg_values.append(future_avg)
            indices.append(i)
       
        return (pd.DataFrame(features_list), np.array(labels), 
                current_values, future_actual_values, future_min_values, future_max_values, future_avg_values, indices)
   
    # ===== 1. 학습 =====
    print("\n[STEP 1] 데이터 학습 (V8-Final: 32개 컬럼)")
    print("-"*40)
   
    try:
        df_train = pd.read_csv('20250904_TO_20251020.csv', on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df_train = pd.read_csv('20250904_TO_20251020.csv', on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df_train = pd.read_csv('20250904_TO_20251020.csv', on_bad_lines='skip', encoding='euc-kr', low_memory=False)
   
    df_train[TARGET_COL] = pd.to_numeric(df_train[TARGET_COL], errors='coerce')
    df_train = df_train.dropna(subset=[TARGET_COL])
   
    print(f"학습 데이터: {len(df_train)}개 행")
   
    # 컬럼 존재 확인
    print("\n📋 컬럼 존재 확인:")
    available_cols = set(df_train.columns)
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
                print(f"    ❌ {col} - 누락")
    
    print(f"\n  총 컬럼: {available_count}/{total_cols}개 사용 가능")
   
    # Feature 생성
    X_train, y_train, current_vals, actual_vals, min_vals, max_vals, avg_vals, _ = create_features_v8_final(df_train, available_cols)
   
    print(f"\n✅ Feature 생성 완료:")
    print(f"  - 총 Feature: {len(X_train.columns)}개")
    print(f"  - 학습 샘플: {len(X_train)}개")
    print(f"  - 타겟: [MIN 변화량, MAX 변화량, AVG 변화량]")
   
    # 변화량 통계
    print(f"\n📊 변화량 통계:")
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        changes = y_train[:, idx]
        print(f"\n  {name} 변화량:")
        print(f"    - 평균: {np.mean(changes):.2f}")
        print(f"    - 표준편차: {np.std(changes):.2f}")
        print(f"    - 최소: {np.min(changes):.2f}")
        print(f"    - 최대: {np.max(changes):.2f}")
       
        positive = sum(1 for c in changes if c > 0)
        negative = sum(1 for c in changes if c < 0)
        large_inc = sum(1 for c in changes if c >= 50)
        large_dec = sum(1 for c in changes if c <= -50)
       
        print(f"    - 상승 (+): {positive}개 ({positive/len(changes)*100:.1f}%)")
        print(f"    - 하락 (-): {negative}개 ({negative/len(changes)*100:.1f}%)")
        print(f"    - 큰 상승 (+50 이상): {large_inc}개 ({large_inc/len(changes)*100:.1f}%)")
        print(f"    - 큰 하락 (-50 이하): {large_dec}개 ({large_dec/len(changes)*100:.1f}%)")
   
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
   
    # GPU/CPU 감지
    print("\n🔍 학습 환경 감지...")
    use_gpu = False
   
    try:
        test_model = xgb.XGBRegressor(
            n_estimators=5, max_depth=3, random_state=42,
            tree_method='gpu_hist', gpu_id=0
        )
        test_model.fit(X_tr[:100], y_tr[:100, 0], verbose=False)
        use_gpu = True
        print("  ✅ GPU 모드\n")
    except:
        print("  ⚠️ CPU 모드\n")
        use_gpu = False
   
    # 멀티아웃풋 모델 생성
    if use_gpu:
        base_model = xgb.XGBRegressor(
            n_estimators=250,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=0.8,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor'
        )
    else:
        base_model = xgb.XGBRegressor(
            n_estimators=250,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=0.8,
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
   
    model = MultiOutputRegressor(base_model)
   
    print("모델 학습 중 (V8-Final: MIN/MAX/AVG 변화량 동시 예측)...")
    print("="*60)
    model.fit(X_tr, y_tr)
    print("="*60)
    print("✅ 학습 완료!")
   
    # 평가
    y_val_pred = model.predict(X_val)
    
    print(f"\n학습 성능:")
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        mae = mean_absolute_error(y_val[:, idx], y_val_pred[:, idx])
        rmse = np.sqrt(mean_squared_error(y_val[:, idx], y_val_pred[:, idx]))
        r2 = r2_score(y_val[:, idx], y_val_pred[:, idx])
        print(f"  {name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
   
    # 모델 저장
    model_filename = 'xgboost_변화량예측_V8_Final.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ 모델 저장: {model_filename}")
   
    # Feature 중요도
    print("\n🔥 Feature 중요도 Top 30:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.estimators_[0].feature_importances_
    }).sort_values('importance', ascending=False).head(30)
   
    for idx, row in feature_importance.iterrows():
        # V8 Feature 표시
        if any(v8_col in row['feature'] for v8_col in 
               ['CURRENT_M16A_3F_JOB', 'M16HUB.QUE.ALL.CURRENTQCOMPLETED', 'M16HUB.QUE.ALL.FABTRANSJOBCNT',
                'M16HUB.QUE.OHT', 'M16A.QUE']):
            marker = "🆕"
        elif any(keyword in row['feature'] for keyword in ['acceleration', 'rising', 'rapid_rise', 'imminent', 'last_10_mean']):
            marker = "💨"
        elif 'surge_risk_score' in row['feature']:
            marker = "🔥"
        elif any(keyword in row['feature'] for keyword in ['hub_high', 'hub_critical', 'storage_util_high',
                                                          'total_cmd_low', 'hub_storage_risk', 'hub_cmd_bottleneck',
                                                          'storage_util_critical', 'currentq', 'avgtime', 'ohtutil',
                                                          'loadtime', 'transport4min', 'delay', 'bottleneck']):
            marker = "🎯"
        elif 'HUBROOM' in row['feature'] or 'storage_util' in row['feature'] or 'CMD' in row['feature']:
            marker = "⚡"
        else:
            marker = "  "
        print(f"{marker} {row['feature']}: {row['importance']:.4f}")
   
    print("\n✅ V8-Final (32개 컬럼) 학습 완료!")
    print(f"   - V7 기존: 19개")
    print(f"   - V8 신규: 13개")
    print(f"   - Feature: {len(X_train.columns)}개")

if __name__ == '__main__':
    print("🚀 V8-Final 변화량 예측 모델 학습 시작...\n")
    train_v8_final_change_prediction()
    print(f"\n🎉 완료!")