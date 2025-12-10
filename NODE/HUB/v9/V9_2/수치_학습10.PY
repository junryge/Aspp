import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

def train_v9_regression():
    """
    🎯 V9-수치형: MIN/MAX/AVG 구간 예측 (Quantile Regression)
    - MIN: 25% 분위수 (하한)
    - MAX: 75% 분위수 (상한)
    - AVG: 50% 분위수 (중앙)
    - 핵심 Feature: mean, std, max, min, last_value, slope
    - V8 14개 + V9 20개 = 34개 컬럼
    """
    print("="*80)
    print("🎯 V9-수치형: Quantile Regression (0.25 ~ 0.75)")
    print("="*80)
    
    # ========== V8 기존 14개 ==========
    FEATURE_COLS_V8 = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
        'hub': ['HUBROOMTOTAL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA'],
    }
    
    # ========== V9 신규 20개 ==========
    V9_NEW_COLS = [
        # 필수 8개
        'M16A_LFTTOALL',
        'M16HUB_M16TOM14_CREATED',
        'M16A_ALLTONORTHCNV',
        'M16A_NORTHCNVTOALL',
        'M16A_M16ATOM14A',
        'M14_TOTALCNVCURRENTQCNT',
        'M16HUB_M16TOM14B_CREATED',
        'M16HUB_M14TOM16_CREATED',
        # 권장 12개
        'M16A_CURRENTQCNT',
        'M16A_ALLTOSOUTHCNV',
        'M14_ALLTOSOUTHCNV',
        'M14_ALLTONORTHCNV',
        'M14_CURRENTQCREATED',
        'M14_RETURNTOM16',
        'M14B_LFTTOALL',
        'M14B_M14BTOM16A',
        'M16B_CURRENTQCREATED',
        'M16_SENDTOM14',
        'M16HUB_M14TOM16_MESCNT',
        'M16HUB_MESCURRENTQCNT',
    ]
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    def create_features_v9(df, available_cols, start_idx=30):
        """V9 Feature 생성 - 핵심만 (mean, std, max, min, last_value, slope)"""
        features_list = []
        labels = []
        indices = []
        
        total_samples = len(df) - 10 - start_idx
        
        for idx, i in enumerate(range(start_idx, len(df) - 10)):
            if idx % 1000 == 0:
                print(f"  Feature 생성: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
            
            seq_target = df[TARGET_COL].iloc[i-30:i].values
            current_value = seq_target[-1]
            
            # 10분 후 구간 타겟
            future_10min = df[TARGET_COL].iloc[i:i+10].values
            future_min = np.min(future_10min)
            future_max = np.max(future_10min)
            future_avg = np.mean(future_10min)
            
            change_min = future_min - current_value
            change_max = future_max - current_value
            change_avg = future_avg - current_value
            
            # ========== 타겟 핵심 Feature (6개 + 급증/급락 6개) ==========
            features = {
                'target_mean': np.mean(seq_target),
                'target_std': np.std(seq_target),
                'target_max': np.max(seq_target),
                'target_min': np.min(seq_target),
                'target_last_value': seq_target[-1],
                'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
                # 🔥 급증/급락 감지용 Feature
                'target_is_rising': 1 if seq_target[-1] > seq_target[-10] else 0,
                'target_is_falling': 1 if seq_target[-1] < seq_target[-10] else 0,
                'target_rapid_rise': 1 if (seq_target[-1] - seq_target[-10] > 20) else 0,
                'target_rapid_fall': 1 if (seq_target[-1] - seq_target[-10] < -20) else 0,
                'target_change_10': seq_target[-1] - seq_target[-10],
                'target_acceleration': (np.mean(seq_target[-10:]) - np.mean(seq_target[-20:-10])),
            }
            
            # ========== V8 기존 컬럼 Feature ==========
            for group_name, cols in FEATURE_COLS_V8.items():
                for col in cols:
                    if col not in available_cols:
                        continue
                    
                    col_seq = df[col].iloc[i-30:i].values
                    
                    if group_name == 'maxcapa':
                        features[f'{col}_last_value'] = col_seq[-1]
                    elif group_name in ['cmd', 'storage', 'fs_storage', 'hub']:
                        features[f'{col}_mean'] = np.mean(col_seq)
                        features[f'{col}_std'] = np.std(col_seq)
                        features[f'{col}_max'] = np.max(col_seq)
                        features[f'{col}_min'] = np.min(col_seq)
                        features[f'{col}_last_value'] = col_seq[-1]
                        features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                    else:  # inflow, outflow
                        features[f'{col}_mean'] = np.mean(col_seq)
                        features[f'{col}_last_value'] = col_seq[-1]
                        features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
            
            # ========== V9 신규 컬럼 Feature ==========
            for col in V9_NEW_COLS:
                if col not in available_cols:
                    continue
                
                col_seq = df[col].iloc[i-30:i].values
                features[f'{col}_mean'] = np.mean(col_seq)
                features[f'{col}_std'] = np.std(col_seq)
                features[f'{col}_max'] = np.max(col_seq)
                features[f'{col}_min'] = np.min(col_seq)
                features[f'{col}_last_value'] = col_seq[-1]
                features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
            
            # ========== 특수 Feature ==========
            # FS Storage
            if all(col in available_cols for col in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
                storage_use = df['CD_M163FSTORAGEUSE'].iloc[i-30:i].values
                storage_total = df['CD_M163FSTORAGETOTAL'].iloc[i-30:i].values
                storage_util = df['CD_M163FSTORAGEUTIL'].iloc[i-30:i].values
                
                features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
                features['storage_remaining'] = storage_total[-1] - storage_use[-1]
                features['storage_util_last'] = storage_util[-1]
                features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
                features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
            
            # HUBROOMTOTAL
            if 'HUBROOMTOTAL' in available_cols:
                hub_seq = df['HUBROOMTOTAL'].iloc[i-30:i].values
                hub_last = hub_seq[-1]
                
                features['hub_critical'] = 1 if hub_last < 590 else 0
                features['hub_high'] = 1 if hub_last < 610 else 0
                features['hub_warning'] = 1 if hub_last < 620 else 0
                features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
            
            # 복합 Feature
            inflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS_V8['inflow'] if col in available_cols)
            outflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS_V8['outflow'] if col in available_cols)
            features['net_flow'] = inflow_sum - outflow_sum
            
            cmd_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS_V8['cmd'] if col in available_cols)
            features['total_cmd'] = cmd_sum
            features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
            features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
            
            features['surge_risk_score'] = (
                features.get('hub_high', 0) * 3 +
                features.get('storage_util_critical', 0) * 2 +
                features.get('total_cmd_low', 0) * 1 +
                features.get('storage_util_high', 0) * 1
            )
            
            # V9 Boolean
            if 'M16A_LFTTOALL' in available_cols:
                lft = df['M16A_LFTTOALL'].iloc[i-1]
                features['lfttoall_high'] = 1 if lft >= 100 else 0
                features['lfttoall_critical'] = 1 if lft >= 150 else 0
            
            if 'M16HUB_M16TOM14_CREATED' in available_cols:
                m16tom14 = df['M16HUB_M16TOM14_CREATED'].iloc[i-1]
                features['m16tom14_high'] = 1 if m16tom14 >= 50 else 0
                features['m16tom14_critical'] = 1 if m16tom14 >= 80 else 0
            
            if 'M16HUB_M14TOM16_CREATED' in available_cols:
                m14tom16 = df['M16HUB_M14TOM16_CREATED'].iloc[i-1]
                features['m14tom16_high'] = 1 if m14tom16 >= 50 else 0
                features['m14tom16_critical'] = 1 if m14tom16 >= 80 else 0
            
            features_list.append(features)
            labels.append([change_min, change_max, change_avg])
            indices.append(i)
        
        return pd.DataFrame(features_list), np.array(labels), indices
    
    # ========== 데이터 로드 ==========
    print("\n[STEP 1] 데이터 로드")
    print("-"*40)
    
    try:
        df_train = pd.read_csv('train_data.csv', on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df_train = pd.read_csv('train_data.csv', on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df_train = pd.read_csv('train_data.csv', on_bad_lines='skip', encoding='euc-kr', low_memory=False)
    
    df_train[TARGET_COL] = pd.to_numeric(df_train[TARGET_COL], errors='coerce')
    df_train = df_train.dropna(subset=[TARGET_COL])
    
    print(f"학습 데이터: {len(df_train)}개 행")
    
    # 컬럼 확인
    available_cols = set(df_train.columns)
    
    v8_count = sum(1 for cols in FEATURE_COLS_V8.values() for col in cols if col in available_cols)
    v9_count = sum(1 for col in V9_NEW_COLS if col in available_cols)
    
    print(f"\n📋 컬럼 확인:")
    print(f"  V8 기존: {v8_count}/14개")
    print(f"  V9 신규: {v9_count}/20개")
    print(f"  총: {v8_count + v9_count}/34개")
    
    # V9 컬럼 상세
    print(f"\n📋 V9 컬럼 상세:")
    for col in V9_NEW_COLS:
        status = "✅" if col in available_cols else "❌"
        print(f"  {status} {col}")
    
    # ========== Feature 생성 ==========
    print("\n[STEP 2] Feature 생성")
    print("-"*40)
    
    X_train, y_train, _ = create_features_v9(df_train, available_cols)
    
    print(f"\n✅ Feature 생성 완료:")
    print(f"  - Feature: {len(X_train.columns)}개")
    print(f"  - 샘플: {len(X_train)}개")
    
    # ========== 가중치 계산 ==========
    print("\n[STEP 3] 가중치 계산")
    print("-"*40)
    
    large_changes = (np.abs(y_train[:, 0]) >= 50) | (np.abs(y_train[:, 1]) >= 50) | (np.abs(y_train[:, 2]) >= 50)
    sample_weights = np.where(large_changes, 5.0, 1.0)
    
    large_count = large_changes.sum()
    print(f"  일반 샘플 (1.0배): {len(large_changes) - large_count}개")
    print(f"  큰 변화 (5.0배): {large_count}개 ({large_count/len(large_changes)*100:.1f}%)")
    
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train, y_train, sample_weights, test_size=0.2, random_state=42
    )
    
    # ========== GPU/CPU 감지 ==========
    print("\n[STEP 4] 학습 환경")
    print("-"*40)
    
    use_gpu = False
    try:
        test_model = xgb.XGBRegressor(
            n_estimators=5, max_depth=3, random_state=42,
            tree_method='gpu_hist', gpu_id=0
        )
        test_model.fit(X_tr[:100], y_tr[:100, 0], verbose=False)
        use_gpu = True
        print("  ✅ GPU 모드")
    except:
        print("  ⚠️ CPU 모드")
        use_gpu = False
    
    # ========== 모델 학습 ==========
    print("\n[STEP 5] 모델 학습 (Quantile 0.25 / 0.75 / 0.50)")
    print("-"*40)
    
    models = []
    
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        print(f"\n[{name}] 학습 중...")
        
        if name == 'MIN':
            quantile_alpha = 0.25
            print(f"  → 25% 분위수 (하한)")
        elif name == 'MAX':
            quantile_alpha = 0.75
            print(f"  → 75% 분위수 (상한)")
        else:
            quantile_alpha = 0.5
            print(f"  → 50% 분위수 (중앙)")
        
        if use_gpu:
            model = xgb.XGBRegressor(
                n_estimators=300,
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
                predictor='gpu_predictor',
                objective='reg:quantileerror',
                quantile_alpha=quantile_alpha
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=300,
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
                n_jobs=-1,
                objective='reg:quantileerror',
                quantile_alpha=quantile_alpha
            )
        
        model.fit(X_tr, y_tr[:, idx], sample_weight=w_tr, verbose=False)
        models.append(model)
        print(f"  ✓ {name} 완료")
    
    print("\n✅ 학습 완료!")
    
    # ========== 검증 ==========
    print("\n[STEP 6] 검증")
    print("-"*40)
    
    y_val_pred = np.column_stack([
        models[0].predict(X_val),
        models[1].predict(X_val),
        models[2].predict(X_val)
    ])
    
    print(f"\n📊 검증 성능:")
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        mae = mean_absolute_error(y_val[:, idx], y_val_pred[:, idx])
        print(f"  {name} - MAE: {mae:.4f}")
    
    # 범위폭
    pred_widths = y_val_pred[:, 1] - y_val_pred[:, 0]
    print(f"\n📏 예측 범위폭:")
    print(f"  - 평균: {pred_widths.mean():.1f}")
    print(f"  - 중앙값: {np.median(pred_widths):.1f}")
    
    # 구간 겹침률
    in_range = 0
    for i in range(len(y_val)):
        pred_min, pred_max = y_val_pred[i, 0], y_val_pred[i, 1]
        actual_min, actual_max = y_val[i, 0], y_val[i, 1]
        if pred_min <= actual_max and pred_max >= actual_min:
            in_range += 1
    print(f"\n🎯 구간 겹침률: {in_range}/{len(y_val)}개 ({in_range/len(y_val)*100:.1f}%)")
    
    # ========== 모델 저장 ==========
    model_dict = {
        'models': models,
        'feature_names': X_train.columns.tolist()
    }
    
    model_filename = 'xgboost_수치형_V9.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"\n✅ 모델 저장: {model_filename}")
    
    # ========== Feature 중요도 ==========
    print("\n🔥 Feature 중요도 Top 20 (MAX 모델):")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': models[1].feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    for i, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "="*80)
    print("✅ V9-수치형 학습 완료!")
    print("="*80)

if __name__ == '__main__':
    print("🚀 V9-수치형 Quantile Regression 학습\n")
    train_v9_regression()
    print(f"\n🎉 완료!")