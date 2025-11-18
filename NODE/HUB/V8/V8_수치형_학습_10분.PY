import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def train_v8_regression_with_weights():
    """
    🎯 V8-수치형 변화량 예측 (50+ 가중치 버전)
    - 타겟: [MIN 변화량, MAX 변화량, AVG 변화량]
    - 큰 변화(±50) 샘플에 3배 가중치
    - 멀티아웃풋 예측
    - V8: 32개 컬럼
    """
    print("="*80)
    print("🎯 V8-수치형 변화량 예측: 50+ 가중치 적용")
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
   
    def create_features_v8(df, available_cols, start_idx=30):
        """V8 Feature 생성"""
        features_list = []
        labels = []
        indices = []
       
        total_samples = len(df) - 10 - start_idx
        
        for idx, i in enumerate(range(start_idx, len(df) - 10)):
            if idx % 1000 == 0:
                print(f"  Feature 생성: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
            
            seq_target = df[TARGET_COL].iloc[i-30:i].values
            current_value = seq_target[-1]
           
            future_actual = df[TARGET_COL].iloc[i+9]
            future_10min = df[TARGET_COL].iloc[i:i+10].values
            future_min = np.min(future_10min)
            future_max = np.max(future_10min)
            future_avg = np.mean(future_10min)
           
            change_min = future_min - current_value
            change_max = future_max - current_value
            change_avg = future_avg - current_value
           
            # 타겟 기본 통계
            features = {
                'target_mean': np.mean(seq_target),
                'target_std': np.std(seq_target),
                'target_max': np.max(seq_target),
                'target_min': np.min(seq_target),
                'target_last_value': seq_target[-1],
                'target_last_5_mean': np.mean(seq_target[-5:]),
                'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
            }
           
            features['target_acceleration'] = (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5
            features['target_is_rising'] = 1 if seq_target[-1] > seq_target[-5] else 0
            features['target_rapid_rise'] = 1 if (seq_target[-1] - seq_target[-5] > 10) else 0
            features['target_last_10_mean'] = np.mean(seq_target[-10:])
           
            # 각 컬럼 그룹 Feature
            for group_name, cols in FEATURE_COLS.items():
                for col in cols:
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
           
            # FS Storage 특수 Feature
            if all(col in available_cols for col in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
                storage_use = df['CD_M163FSTORAGEUSE'].iloc[i-30:i].values
                storage_total = df['CD_M163FSTORAGETOTAL'].iloc[i-30:i].values
                storage_util = df['CD_M163FSTORAGEUTIL'].iloc[i-30:i].values
               
                features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
                features['storage_remaining'] = storage_total[-1] - storage_use[-1]
                features['storage_util_last'] = storage_util[-1]
                features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
                features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
           
            # HUBROOMTOTAL 특수 Feature
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
           
            # 복합 Feature
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
           
            # 급증 위험도
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
           
            # V7 QUE Boolean
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
           
            # V8 M16HUB QUE Boolean
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
           
            # V8 M16A QUE Boolean
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
            indices.append(i)
       
        return pd.DataFrame(features_list), np.array(labels), indices
   
    # 학습
    print("\n[STEP 1] 데이터 학습 (V8-수치형: 50+ 가중치)")
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
    print("\n📋 컬럼 확인:")
    available_cols = set(df_train.columns)
    available_count = sum(1 for cols in FEATURE_COLS.values() for col in cols if col in available_cols)
    total_cols = sum(len(cols) for cols in FEATURE_COLS.values())
    print(f"  사용 가능: {available_count}/{total_cols}개")
   
    # Feature 생성
    X_train, y_train, _ = create_features_v8(df_train, available_cols)
   
    print(f"\n✅ Feature 생성 완료:")
    print(f"  - Feature: {len(X_train.columns)}개")
    print(f"  - 샘플: {len(X_train)}개")
   
    # 🔥 가중치 계산 (50+ 샘플에 3배)
    print("\n🔥 가중치 계산 (50+ 샘플 강조):")
    
    # MAX 변화량 기준으로 가중치 (가장 중요)
    large_changes = np.abs(y_train[:, 1]) >= 50
    sample_weights = np.where(large_changes, 3.0, 1.0)
    
    large_count = large_changes.sum()
    normal_count = len(large_changes) - large_count
    
    print(f"  - 일반 샘플 (가중치 1.0): {normal_count}개")
    print(f"  - 큰 변화 (가중치 3.0): {large_count}개 ({large_count/len(large_changes)*100:.1f}%)")
   
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, sample_weights, test_size=0.2, random_state=42
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
   
    print("모델 학습 중 (가중치 적용 - 3개 모델 개별 학습)...")
    print("="*60)
    
    # 🔥 각 출력(MIN/MAX/AVG)에 대해 개별 모델 학습
    # MultiOutputRegressor 대신 개별 모델로 가중치 적용
    models = []
    
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        print(f"\n[{name}] 학습 중...")
        
        # 개별 모델 생성
        if use_gpu:
            individual_model = xgb.XGBRegressor(
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
            individual_model = xgb.XGBRegressor(
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
        
        # 가중치 적용 학습
        individual_model.fit(X_tr, y_tr[:, idx], sample_weight=w_tr, verbose=False)
        models.append(individual_model)
        print(f"✓ {name} 학습 완료")
    
    print("="*60)
    print("✅ 학습 완료!")
   
    # 평가 (3개 모델 개별 예측)
    y_val_pred = np.column_stack([
        models[0].predict(X_val),
        models[1].predict(X_val),
        models[2].predict(X_val)
    ])
    
    print(f"\n📊 학습 성능 (50+ 가중치 적용):")
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        mae = mean_absolute_error(y_val[:, idx], y_val_pred[:, idx])
        rmse = np.sqrt(mean_squared_error(y_val[:, idx], y_val_pred[:, idx]))
        r2 = r2_score(y_val[:, idx], y_val_pred[:, idx])
        print(f"  {name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # 큰 변화 성능
    print(f"\n🎯 큰 변화(±50) 예측 성능:")
    large_idx = np.abs(y_val[:, 1]) >= 50
    if large_idx.sum() > 0:
        mae_large = mean_absolute_error(y_val[large_idx, 1], y_val_pred[large_idx, 1])
        print(f"  - 큰 변화 샘플: {large_idx.sum()}개")
        print(f"  - MAX MAE (큰 변화만): {mae_large:.4f}")
   
    # 모델 저장 (3개 모델을 딕셔너리로)
    model_dict = {
        'models': models,  # 3개 모델 리스트
        'feature_names': X_train.columns.tolist()
    }
    
    model_filename = 'xgboost_수치형_V8.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"\n✅ 모델 저장: {model_filename}")
   
    # Feature 중요도 (MAX 모델 기준)
    print("\n🔥 Feature 중요도 Top 20 (MAX 모델 기준):")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': models[1].feature_importances_
    }).sort_values('importance', ascending=False).head(20)
   
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
   
    print("\n✅ V8-수치형 (50+ 가중치) 학습 완료!")

if __name__ == '__main__':
    print("🚀 V8-수치형 변화량 예측 모델 학습 (50+ 가중치)\n")
    train_v8_regression_with_weights()
    print(f"\n🎉 완료!")