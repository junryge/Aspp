import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

warnings.filterwarnings('ignore')

# V9 3단계 학습 - 34개 컬럼 (V8 14개 + V9 20개)

def train_v9_3stage():
    """
    🎯 V9 3단계 학습 (GPU)
    
    1단계: 분류 모델 (급등/급락/유지)
    2단계: 전용 수치 모델 3개
      - 급등 모델: 급등 데이터만 학습
      - 급락 모델: 급락 데이터만 학습
      - 유지 모델: 유지 데이터만 학습
    
    저장: xgboost_3stage_V9.pkl
    """
    print("="*80)
    print("🎯 V9 3단계 학습: 분류 + 전용 수치모델 (GPU)")
    print("="*80)
    
    # ========== 컬럼 정의 ==========
    FEATURE_COLS_V8 = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
        'hub': ['HUBROOMTOTAL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA'],
    }
    
    V9_NEW_COLS = [
        'M16A_LFTTOALL', 'M16HUB_M16TOM14_CREATED', 'M16A_ALLTONORTHCNV',
        'M16A_NORTHCNVTOALL', 'M16A_M16ATOM14A', 'M14_TOTALCNVCURRENTQCNT',
        'M16HUB_M16TOM14B_CREATED', 'M16HUB_M14TOM16_CREATED', 'M16A_CURRENTQCNT',
        'M16A_ALLTOSOUTHCNV', 'M14_ALLTOSOUTHCNV', 'M14_ALLTONORTHCNV',
        'M14_CURRENTQCREATED', 'M14_RETURNTOM16', 'M14B_LFTTOALL',
        'M14B_M14BTOM16A', 'M16B_CURRENTQCREATED', 'M16_SENDTOM14',
        'M16HUB_M14TOM16_MESCNT', 'M16HUB_MESCURRENTQCNT',
    ]
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    class_names = {0: '급락', 1: '유지', 2: '급등'}
    
    # ========== 데이터 로드 ==========
    print("\n[STEP 1] 데이터 로드")
    print("-"*40)
    
    try:
        df = pd.read_csv('train_data.csv', on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df = pd.read_csv('train_data.csv', on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df = pd.read_csv('train_data.csv', on_bad_lines='skip', encoding='euc-kr', low_memory=False)
    
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
    print(f"학습 데이터: {len(df)}개 행")
    
    available_cols = set(df.columns)
    
    # ========== numpy 배열 변환 ==========
    print("\n[STEP 2] 데이터 변환")
    print("-"*40)
    
    target_arr = df[TARGET_COL].values.astype(np.float32)
    
    col_arrays = {}
    all_cols = []
    for cols in FEATURE_COLS_V8.values():
        all_cols.extend(cols)
    all_cols.extend(V9_NEW_COLS)
    
    for col in all_cols:
        if col in available_cols:
            col_arrays[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).values.astype(np.float32)
    
    print(f"  변환 완료: {len(col_arrays)}개 컬럼")
    del df
    gc.collect()
    
    # ========== Feature 생성 ==========
    print("\n[STEP 3] Feature 생성")
    print("-"*40)
    
    n_samples = len(target_arr) - 40
    x_range = np.arange(30, dtype=np.float32)
    
    features_list = []
    labels_class = []
    labels_change = []
    current_values = []
    
    for idx, i in enumerate(range(30, len(target_arr) - 10)):
        if idx % 10000 == 0:
            print(f"  진행: {idx}/{n_samples} ({idx/n_samples*100:.1f}%)")
            gc.collect()
        
        seq_target = target_arr[i-30:i]
        current_value = seq_target[-1]
        future_value = target_arr[i+9]
        actual_change = future_value - current_value
        
        # 클래스
        if actual_change >= 50:
            class_label = 2
        elif actual_change <= -50:
            class_label = 0
        else:
            class_label = 1
        
        # Feature
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': current_value,
            'target_slope': np.polyfit(x_range, seq_target, 1)[0],
        }
        
        for group_name, cols in FEATURE_COLS_V8.items():
            for col in cols:
                if col not in col_arrays:
                    continue
                col_seq = col_arrays[col][i-30:i]
                if group_name == 'maxcapa':
                    features[f'{col}_last_value'] = col_seq[-1]
                elif group_name in ['cmd', 'storage', 'fs_storage', 'hub']:
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_std'] = np.std(col_seq)
                    features[f'{col}_max'] = np.max(col_seq)
                    features[f'{col}_min'] = np.min(col_seq)
                    features[f'{col}_last_value'] = col_seq[-1]
                    features[f'{col}_slope'] = np.polyfit(x_range, col_seq, 1)[0]
                else:
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_last_value'] = col_seq[-1]
                    features[f'{col}_slope'] = np.polyfit(x_range, col_seq, 1)[0]
        
        for col in V9_NEW_COLS:
            if col not in col_arrays:
                continue
            col_seq = col_arrays[col][i-30:i]
            features[f'{col}_mean'] = np.mean(col_seq)
            features[f'{col}_std'] = np.std(col_seq)
            features[f'{col}_max'] = np.max(col_seq)
            features[f'{col}_min'] = np.min(col_seq)
            features[f'{col}_last_value'] = col_seq[-1]
            features[f'{col}_slope'] = np.polyfit(x_range, col_seq, 1)[0]
        
        # 특수 Feature
        if 'CD_M163FSTORAGEUTIL' in col_arrays:
            util_last = col_arrays['CD_M163FSTORAGEUTIL'][i-1]
            features['storage_util_high'] = 1 if util_last >= 7 else 0
            features['storage_util_critical'] = 1 if util_last >= 10 else 0
            features['storage_util_last'] = util_last
        
        if 'CD_M163FSTORAGEUSE' in col_arrays and 'CD_M163FSTORAGETOTAL' in col_arrays:
            use_seq = col_arrays['CD_M163FSTORAGEUSE'][i-30:i]
            total_seq = col_arrays['CD_M163FSTORAGETOTAL'][i-30:i]
            features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
            features['storage_remaining'] = total_seq[-1] - use_seq[-1]
        
        if 'HUBROOMTOTAL' in col_arrays:
            hub_seq = col_arrays['HUBROOMTOTAL'][i-30:i]
            hub_last = hub_seq[-1]
            features['hub_critical'] = 1 if hub_last < 590 else 0
            features['hub_high'] = 1 if hub_last < 610 else 0
            features['hub_warning'] = 1 if hub_last < 620 else 0
            features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
        
        # 복합 Feature
        inflow_sum = 0
        outflow_sum = 0
        for col in FEATURE_COLS_V8['inflow']:
            if col in col_arrays:
                inflow_sum += col_arrays[col][i-1]
        for col in FEATURE_COLS_V8['outflow']:
            if col in col_arrays:
                outflow_sum += col_arrays[col][i-1]
        features['net_flow'] = inflow_sum - outflow_sum
        
        cmd_sum = 0
        for col in FEATURE_COLS_V8['cmd']:
            if col in col_arrays:
                cmd_sum += col_arrays[col][i-1]
        features['total_cmd'] = cmd_sum
        features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
        features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
        
        features['surge_risk_score'] = (
            features.get('hub_high', 0) * 3 +
            features.get('storage_util_critical', 0) * 2 +
            features.get('total_cmd_low', 0) * 1 +
            features.get('storage_util_high', 0) * 1
        )
        
        features_list.append(features)
        labels_class.append(class_label)
        labels_change.append(actual_change)
        current_values.append(current_value)
    
    X_all = pd.DataFrame(features_list)
    y_class = np.array(labels_class)
    y_change = np.array(labels_change)
    current_vals = np.array(current_values)
    
    del features_list, col_arrays
    gc.collect()
    
    print(f"\n✅ Feature 생성 완료: {len(X_all)}개 샘플, {len(X_all.columns)}개 Feature")
    
    # 클래스 분포
    print(f"\n📊 클래스 분포:")
    for cls in [0, 1, 2]:
        count = (y_class == cls).sum()
        print(f"  {class_names[cls]}: {count}개 ({count/len(y_class)*100:.1f}%)")
    
    # ========== 1단계: 분류 모델 ==========
    print("\n[STEP 4] 분류 모델 학습 (GPU)")
    print("-"*40)
    
    # 가중치
    sample_weights = np.ones(len(y_class))
    sample_weights[y_class == 0] = 10.0  # 급락
    sample_weights[y_class == 2] = 10.0  # 급등
    sample_weights[current_vals >= 400] *= 3  # 400대 추가
    
    print(f"  가중치: 급락/급등 10배, 400대 추가 3배")
    
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_all, y_class, sample_weights, test_size=0.2, random_state=42, stratify=y_class
    )
    
    clf_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=2, gamma=0.05,
        reg_alpha=0.05, reg_lambda=0.8,
        random_state=42,
        tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor',
        objective='multi:softprob', num_class=3
    )
    
    clf_model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
    
    y_pred = clf_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"  ✅ 분류 정확도: {acc:.1%}")
    
    for cls in [0, 1, 2]:
        mask = y_val == cls
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == cls).mean()
            print(f"    {class_names[cls]}: {cls_acc:.1%}")
    
    # ========== 2단계: 전용 수치 모델 ==========
    print("\n[STEP 5] 전용 수치 모델 학습 (GPU)")
    print("-"*40)
    
    regression_models = {}
    
    for cls in [0, 1, 2]:
        cls_name = class_names[cls]
        mask = y_class == cls
        X_cls = X_all[mask]
        y_cls = y_change[mask]
        
        print(f"\n  [{cls_name}] {len(X_cls)}개 데이터")
        
        if len(X_cls) < 100:
            print(f"    ⚠️ 데이터 부족")
            regression_models[cls] = None
            continue
        
        # 가중치
        weights = np.ones(len(y_cls))
        if cls == 2:  # 급등
            weights[y_cls >= 80] = 3.0
            weights[y_cls >= 100] = 5.0
        elif cls == 0:  # 급락
            weights[y_cls <= -80] = 3.0
            weights[y_cls <= -100] = 5.0
        
        X_tr_cls, X_val_cls, y_tr_cls, y_val_cls, w_tr_cls, _ = train_test_split(
            X_cls, y_cls, weights, test_size=0.2, random_state=42
        )
        
        reg_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=2, gamma=0.05,
            reg_alpha=0.05, reg_lambda=0.8,
            random_state=42,
            tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor'
        )
        
        reg_model.fit(X_tr_cls, y_tr_cls, sample_weight=w_tr_cls, verbose=False)
        
        y_pred_cls = reg_model.predict(X_val_cls)
        mae = mean_absolute_error(y_val_cls, y_pred_cls)
        print(f"    ✅ MAE: {mae:.2f}")
        
        regression_models[cls] = reg_model
    
    # ========== 모델 저장 ==========
    print("\n[STEP 6] 모델 저장")
    print("-"*40)
    
    model_dict = {
        'classifier': clf_model,
        'regressor_surge': regression_models[2],  # 급등
        'regressor_drop': regression_models[0],   # 급락
        'regressor_hold': regression_models[1],   # 유지
        'feature_names': X_all.columns.tolist(),
        'class_names': class_names
    }
    
    with open('xgboost_3stage_V9.pkl', 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"✅ 저장: xgboost_3stage_V9.pkl")
    
    print("\n" + "="*80)
    print("🎉 V9 3단계 학습 완료!")
    print("="*80)

if __name__ == '__main__':
    train_v9_3stage()