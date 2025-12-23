import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

warnings.filterwarnings('ignore')

def train_v8_2stage():
    """
    🎯 V8 2단계 분리 모델 학습
    
    [1단계] 급변 탐지 모델 (이진 분류)
            → 급변(±50 이상) vs 일반
            → 급변에 50배 가중치
    
    [2단계] 
        급변 Yes → 급등/급락 전용 모델
        급변 No  → 일반 3단계 모델 (소폭하락/정체/소폭상승)
    
    저장: xgboost_2stage_V8.pkl
    """
    print("="*80)
    print("🎯 V8 2단계 분리 모델 학습")
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
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
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
    print(f"📋 V8 컬럼: {sum(1 for c in v8_cols if c in available_cols)}/14개")
    
    # 배열 변환
    print("\n[STEP 2] 데이터 변환")
    print("-"*40)
    
    target_arr = df[TARGET_COL].values.astype(np.float32)
    col_arrays = {col: pd.to_numeric(df[col], errors='coerce').fillna(0).values.astype(np.float32) 
                  for col in v8_cols if col in available_cols}
    print(f"  변환 완료: {len(col_arrays)}개 컬럼")
    del df
    gc.collect()
    
    # Feature 생성
    print("\n[STEP 3] Feature 생성")
    print("-"*40)
    
    n_samples = len(target_arr) - 40
    x_range = np.arange(30, dtype=np.float32)
    
    features_list = []
    labels_sudden = []      # 급변 여부 (0: 일반, 1: 급변)
    labels_direction = []   # 방향 (0: 급락, 1: 급등)
    labels_normal = []      # 일반 분류 (0: 소폭하락, 1: 정체, 2: 소폭상승)
    labels_change = []      # 실제 변화량
    current_values = []
    slopes = []
    
    for idx, i in enumerate(range(30, len(target_arr) - 10)):
        if idx % 10000 == 0:
            print(f"  진행: {idx}/{n_samples} ({idx/n_samples*100:.1f}%)")
            gc.collect()
        
        seq_target = target_arr[i-30:i]
        current_value = seq_target[-1]
        actual_change = target_arr[i+9] - current_value
        slope = np.polyfit(x_range, seq_target, 1)[0]
        
        # 급변 여부 (±50 이상)
        is_sudden = 1 if abs(actual_change) >= 50 else 0
        
        # 방향 (급변일 때만 의미 있음)
        direction = 1 if actual_change >= 0 else 0  # 1: 급등, 0: 급락
        
        # 일반 분류 (급변 아닐 때)
        if actual_change <= -20:
            normal_class = 0  # 소폭하락
        elif actual_change <= 20:
            normal_class = 1  # 정체
        else:
            normal_class = 2  # 소폭상승
        
        # Feature
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': current_value,
            'target_slope': slope,
            # 급변 감지용 추가 Feature
            'target_range': np.max(seq_target) - np.min(seq_target),
            'target_momentum': seq_target[-1] - seq_target[-5] if len(seq_target) >= 5 else 0,
            'target_accel': (seq_target[-1] - seq_target[-5]) - (seq_target[-5] - seq_target[-10]) if len(seq_target) >= 10 else 0,
            'target_volatility': np.std(np.diff(seq_target)),
        }
        
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
        labels_sudden.append(is_sudden)
        labels_direction.append(direction)
        labels_normal.append(normal_class)
        labels_change.append(actual_change)
        current_values.append(current_value)
        slopes.append(slope)
    
    X_all = pd.DataFrame(features_list)
    y_sudden = np.array(labels_sudden)
    y_direction = np.array(labels_direction)
    y_normal = np.array(labels_normal)
    y_change = np.array(labels_change)
    current_vals = np.array(current_values)
    slope_arr = np.array(slopes)
    
    del features_list, col_arrays
    gc.collect()
    
    print(f"\n✅ Feature: {len(X_all)}개 샘플, {len(X_all.columns)}개 Feature")
    
    # 분포 확인
    sudden_count = y_sudden.sum()
    print(f"\n📊 급변 분포:")
    print(f"  급변: {sudden_count}개 ({sudden_count/len(y_sudden)*100:.1f}%)")
    print(f"  일반: {len(y_sudden) - sudden_count}개 ({(len(y_sudden)-sudden_count)/len(y_sudden)*100:.1f}%)")
    
    # ========== [1단계] 급변 탐지 모델 ==========
    print("\n" + "="*60)
    print("[1단계] 급변 탐지 모델 학습 (이진 분류)")
    print("="*60)
    
    # 가중치: 급변 50배
    sudden_weights = np.ones(len(y_sudden))
    sudden_weights[y_sudden == 1] = 50.0
    
    print(f"  가중치: 급변 50배")
    
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X_all, y_sudden, sudden_weights, test_size=0.2, random_state=42, stratify=y_sudden
    )
    
    sudden_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, tree_method='hist', n_jobs=-1,
        scale_pos_weight=1  # sample_weight로 처리
    )
    sudden_model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
    
    y_pred_sudden = sudden_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred_sudden)
    print(f"\n  ✅ 급변 탐지 정확도: {acc:.1%}")
    
    # 급변 Recall 확인
    sudden_mask = y_val == 1
    if sudden_mask.sum() > 0:
        recall = (y_pred_sudden[sudden_mask] == 1).mean()
        print(f"  ✅ 급변 Recall: {recall:.1%} ({(y_pred_sudden[sudden_mask] == 1).sum()}/{sudden_mask.sum()}개)")
    
    # 일반 중 오탐
    normal_mask = y_val == 0
    if normal_mask.sum() > 0:
        false_alarm = (y_pred_sudden[normal_mask] == 1).mean()
        print(f"  ⚠️ 오탐률 (일반→급변): {false_alarm:.1%}")
    
    # ========== [2단계-A] 급등 전용 모델 ==========
    print("\n" + "="*60)
    print("[2단계-A] 급등 전용 모델 학습")
    print("="*60)
    
    # 급등 데이터만 (change >= 50)
    surge_mask = y_change >= 50
    X_surge = X_all[surge_mask]
    y_surge = y_change[surge_mask]
    
    print(f"  급등 데이터: {len(X_surge)}개")
    
    if len(X_surge) >= 30:
        X_tr_s, X_val_s, y_tr_s, y_val_s = train_test_split(
            X_surge, y_surge, test_size=0.2, random_state=42
        )
        
        surge_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, tree_method='hist', n_jobs=-1
        )
        surge_model.fit(X_tr_s, y_tr_s, verbose=False)
        
        mae = mean_absolute_error(y_val_s, surge_model.predict(X_val_s))
        print(f"  ✅ 급등모델 MAE: {mae:.2f}")
    else:
        surge_model = None
        print(f"  ⚠️ 데이터 부족, 기본값 사용")
    
    # ========== [2단계-B] 급락 전용 모델 ==========
    print("\n" + "="*60)
    print("[2단계-B] 급락 전용 모델 학습")
    print("="*60)
    
    # 급락 데이터만 (change <= -50)
    drop_mask = y_change <= -50
    X_drop = X_all[drop_mask]
    y_drop = y_change[drop_mask]
    
    print(f"  급락 데이터: {len(X_drop)}개")
    
    if len(X_drop) >= 30:
        X_tr_d, X_val_d, y_tr_d, y_val_d = train_test_split(
            X_drop, y_drop, test_size=0.2, random_state=42
        )
        
        drop_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, tree_method='hist', n_jobs=-1
        )
        drop_model.fit(X_tr_d, y_tr_d, verbose=False)
        
        mae = mean_absolute_error(y_val_d, drop_model.predict(X_val_d))
        print(f"  ✅ 급락모델 MAE: {mae:.2f}")
    else:
        drop_model = None
        print(f"  ⚠️ 데이터 부족, 기본값 사용")
    
    # ========== [2단계-C] 일반 3단계 모델 ==========
    print("\n" + "="*60)
    print("[2단계-C] 일반 3단계 모델 학습 (소폭하락/정체/소폭상승)")
    print("="*60)
    
    # 급변 아닌 데이터만
    normal_data_mask = y_sudden == 0
    X_normal = X_all[normal_data_mask]
    y_normal_cls = y_normal[normal_data_mask]
    y_normal_change = y_change[normal_data_mask]
    
    print(f"  일반 데이터: {len(X_normal)}개")
    
    normal_class_names = {0: '소폭하락', 1: '정체', 2: '소폭상승'}
    
    print(f"\n  📊 일반 클래스 분포:")
    for cls in range(3):
        count = (y_normal_cls == cls).sum()
        print(f"    {normal_class_names[cls]}: {count}개 ({count/len(y_normal_cls)*100:.1f}%)")
    
    # 일반 분류기
    normal_weights = np.ones(len(y_normal_cls))
    normal_weights[y_normal_cls == 0] = 3.0  # 소폭하락
    normal_weights[y_normal_cls == 2] = 3.0  # 소폭상승
    
    X_tr_n, X_val_n, y_tr_n, y_val_n, w_tr_n, _ = train_test_split(
        X_normal, y_normal_cls, normal_weights, test_size=0.2, random_state=42, stratify=y_normal_cls
    )
    
    normal_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, tree_method='hist', n_jobs=-1,
        objective='multi:softprob', num_class=3
    )
    normal_clf.fit(X_tr_n, y_tr_n, sample_weight=w_tr_n, verbose=False)
    
    acc = accuracy_score(y_val_n, normal_clf.predict(X_val_n))
    print(f"\n  ✅ 일반 분류 정확도: {acc:.1%}")
    
    # 일반 수치 모델 3개
    normal_regressors = {}
    for cls in range(3):
        cls_mask = y_normal_cls == cls
        X_cls = X_normal[cls_mask]
        y_cls = y_normal_change[cls_mask]
        
        print(f"\n  [{normal_class_names[cls]}] {len(X_cls)}개")
        
        if len(X_cls) >= 50:
            X_tr_c, X_val_c, y_tr_c, y_val_c = train_test_split(
                X_cls, y_cls, test_size=0.2, random_state=42
            )
            
            reg_model = xgb.XGBRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.03,
                subsample=0.85, colsample_bytree=0.85,
                random_state=42, tree_method='hist', n_jobs=-1
            )
            reg_model.fit(X_tr_c, y_tr_c, verbose=False)
            
            mae = mean_absolute_error(y_val_c, reg_model.predict(X_val_c))
            print(f"    ✅ MAE: {mae:.2f}")
            normal_regressors[cls] = reg_model
        else:
            normal_regressors[cls] = None
    
    # ========== 저장 ==========
    print("\n" + "="*60)
    print("모델 저장")
    print("="*60)
    
    model_dict = {
        # 1단계: 급변 탐지
        'sudden_detector': sudden_model,
        # 2단계-A: 급등 전용
        'surge_model': surge_model,
        # 2단계-B: 급락 전용
        'drop_model': drop_model,
        # 2단계-C: 일반 분류 + 수치
        'normal_classifier': normal_clf,
        'normal_regressors': normal_regressors,
        # 메타
        'feature_names': X_all.columns.tolist(),
        'normal_class_names': normal_class_names,
        'version': 'V8_2stage_separate'
    }
    
    with open('xgboost_2stage_V8.pkl', 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"✅ 저장: xgboost_2stage_V8.pkl")
    print("\n🎉 V8 2단계 분리 모델 학습 완료!")

if __name__ == '__main__':
    train_v8_2stage()