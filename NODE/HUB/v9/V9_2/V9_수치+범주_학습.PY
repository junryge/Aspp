import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

warnings.filterwarnings('ignore')

def train_v9_5stage():
    """
    🎯 V9 5단계 학습 - 34개 컬럼 (V8 14개 + V9 20개)
    
    5단계 분류:
      - Class 0: 급락 (< -50)
      - Class 1: 소폭하락 (-50 ~ -20)
      - Class 2: 정체 (-20 ~ +20)
      - Class 3: 소폭상승 (+20 ~ +50)
      - Class 4: 급등 (> +50)
    
    저장: xgboost_5stage_V9.pkl
    """
    print("="*80)
    print("🎯 V9 5단계 학습: 급락/소폭하락/정체/소폭상승/급등 (34개 컬럼)")
    print("="*80)
    
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
    
    class_names = {0: '급락', 1: '소폭하락', 2: '정체', 3: '소폭상승', 4: '급등'}
    
    def get_class(change):
        if change <= -50: return 0
        elif change <= -20: return 1
        elif change <= 20: return 2
        elif change <= 50: return 3
        else: return 4
    
    # 데이터 로드
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
    v8_cols = [col for cols in FEATURE_COLS_V8.values() for col in cols]
    all_cols = v8_cols + V9_NEW_COLS
    
    v8_cnt = sum(1 for c in v8_cols if c in available_cols)
    v9_cnt = sum(1 for c in V9_NEW_COLS if c in available_cols)
    print(f"📋 V8: {v8_cnt}/14개, V9: {v9_cnt}/20개")
    
    # 배열 변환
    print("\n[STEP 2] 데이터 변환")
    print("-"*40)
    
    target_arr = df[TARGET_COL].values.astype(np.float32)
    col_arrays = {col: pd.to_numeric(df[col], errors='coerce').fillna(0).values.astype(np.float32) 
                  for col in all_cols if col in available_cols}
    print(f"  변환 완료: {len(col_arrays)}개 컬럼")
    del df
    gc.collect()
    
    # Feature 생성
    print("\n[STEP 3] Feature 생성")
    print("-"*40)
    
    n_samples = len(target_arr) - 40
    x_range = np.arange(30, dtype=np.float32)
    
    features_list, labels_class, labels_change, current_values = [], [], [], []
    
    for idx, i in enumerate(range(30, len(target_arr) - 10)):
        if idx % 10000 == 0:
            print(f"  진행: {idx}/{n_samples} ({idx/n_samples*100:.1f}%)")
            gc.collect()
        
        seq_target = target_arr[i-30:i]
        current_value = seq_target[-1]
        actual_change = target_arr[i+9] - current_value
        
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': current_value,
            'target_slope': np.polyfit(x_range, seq_target, 1)[0],
        }
        
        # V8 컬럼
        for group_name, cols in FEATURE_COLS_V8.items():
            for col in cols:
                if col not in col_arrays: continue
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
        
        # V9 컬럼
        for col in V9_NEW_COLS:
            if col not in col_arrays: continue
            col_seq = col_arrays[col][i-30:i]
            features[f'{col}_mean'] = np.mean(col_seq)
            features[f'{col}_std'] = np.std(col_seq)
            features[f'{col}_max'] = np.max(col_seq)
            features[f'{col}_min'] = np.min(col_seq)
            features[f'{col}_last_value'] = col_seq[-1]
            features[f'{col}_slope'] = np.polyfit(x_range, col_seq, 1)[0]
        
        if 'CD_M163FSTORAGEUTIL' in col_arrays:
            util_last = col_arrays['CD_M163FSTORAGEUTIL'][i-1]
            features['storage_util_high'] = 1 if util_last >= 7 else 0
            features['storage_util_critical'] = 1 if util_last >= 10 else 0
        
        if 'HUBROOMTOTAL' in col_arrays:
            hub_last = col_arrays['HUBROOMTOTAL'][i-1]
            features['hub_critical'] = 1 if hub_last < 590 else 0
            features['hub_high'] = 1 if hub_last < 610 else 0
        
        inflow = sum(col_arrays[c][i-1] for c in FEATURE_COLS_V8['inflow'] if c in col_arrays)
        outflow = sum(col_arrays[c][i-1] for c in FEATURE_COLS_V8['outflow'] if c in col_arrays)
        features['net_flow'] = inflow - outflow
        
        features_list.append(features)
        labels_class.append(get_class(actual_change))
        labels_change.append(actual_change)
        current_values.append(current_value)
    
    X_all = pd.DataFrame(features_list)
    y_class = np.array(labels_class)
    y_change = np.array(labels_change)
    current_vals = np.array(current_values)
    
    del features_list, col_arrays
    gc.collect()
    
    print(f"\n✅ Feature: {len(X_all)}개 샘플, {len(X_all.columns)}개 Feature")
    
    print(f"\n📊 5단계 클래스 분포:")
    for cls in range(5):
        count = (y_class == cls).sum()
        print(f"  {class_names[cls]}: {count}개 ({count/len(y_class)*100:.1f}%)")
    
    # 분류 모델
    print("\n[STEP 4] 분류 모델 학습")
    print("-"*40)
    
    sample_weights = np.ones(len(y_class))
    sample_weights[y_class == 0] = 15.0
    sample_weights[y_class == 1] = 5.0
    sample_weights[y_class == 3] = 5.0
    sample_weights[y_class == 4] = 15.0
    sample_weights[current_vals >= 400] *= 2
    
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_all, y_class, sample_weights, test_size=0.2, random_state=42, stratify=y_class
    )
    
    clf_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, tree_method='hist', n_jobs=-1,
        objective='multi:softprob', num_class=5
    )
    clf_model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
    
    acc = accuracy_score(y_val, clf_model.predict(X_val))
    print(f"  ✅ 분류 정확도: {acc:.1%}")
    
    # 수치 모델 5개
    print("\n[STEP 5] 전용 수치 모델 학습 (5개)")
    print("-"*40)
    
    regression_models = {}
    for cls in range(5):
        mask = y_class == cls
        X_cls, y_cls = X_all[mask], y_change[mask]
        print(f"\n  [{class_names[cls]}] {len(X_cls)}개")
        
        if len(X_cls) < 50:
            regression_models[cls] = None
            continue
        
        X_tr_cls, X_val_cls, y_tr_cls, y_val_cls = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42
        )
        
        reg_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, tree_method='hist', n_jobs=-1
        )
        reg_model.fit(X_tr_cls, y_tr_cls, verbose=False)
        
        mae = mean_absolute_error(y_val_cls, reg_model.predict(X_val_cls))
        print(f"    ✅ MAE: {mae:.2f}")
        regression_models[cls] = reg_model
    
    # 저장
    model_dict = {
        'classifier': clf_model,
        'regressors': regression_models,
        'feature_names': X_all.columns.tolist(),
        'class_names': class_names,
        'version': 'V9_5stage'
    }
    
    with open('xgboost_5stage_V9.pkl', 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"\n✅ 저장: xgboost_5stage_V9.pkl")
    print("🎉 V9 5단계 학습 완료!")

if __name__ == '__main__':
    train_v9_5stage()