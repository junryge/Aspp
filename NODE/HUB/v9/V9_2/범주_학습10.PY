import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

def train_v9_classification():
    """
    🎯 V9-범주형: 3-Class 분류
    - Class 0: < 0 (하락/정체)
    - Class 1: 0~50 (소폭 증가)
    - Class 2: 50+ (급증, 5배 가중치)
    - 핵심 Feature: mean, std, max, min, last_value, slope
    - V8 14개 + V9 20개 = 34개 컬럼
    """
    print("="*80)
    print("🎯 V9-범주형: 3-Class Classification")
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
        'M16A_LFTTOALL',
        'M16HUB_M16TOM14_CREATED',
        'M16A_ALLTONORTHCNV',
        'M16A_NORTHCNVTOALL',
        'M16A_M16ATOM14A',
        'M14_TOTALCNVCURRENTQCNT',
        'M16HUB_M16TOM14B_CREATED',
        'M16HUB_M14TOM16_CREATED',
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
    
    def create_class_label(change):
        """변화량을 클래스로 변환"""
        if change < 0:
            return 0  # 하락/정체
        elif change < 50:
            return 1  # 소폭 증가
        else:
            return 2  # 급증
    
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
            
            # 10분 후 실제값
            future_actual = df[TARGET_COL].iloc[i+9]
            actual_change = future_actual - current_value
            
            # 클래스 라벨
            class_label = create_class_label(actual_change)
            
            # ========== 타겟 핵심 Feature (6개) ==========
            features = {
                'target_mean': np.mean(seq_target),
                'target_std': np.std(seq_target),
                'target_max': np.max(seq_target),
                'target_min': np.min(seq_target),
                'target_last_value': seq_target[-1],
                'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
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
            labels.append(class_label)
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
    
    # 클래스 분포
    print(f"\n📊 클래스 분포:")
    class_counts = pd.Series(y_train).value_counts().sort_index()
    for cls, count in class_counts.items():
        class_name = ['Class 0 (<0)', 'Class 1 (0~50)', 'Class 2 (50+)'][cls]
        print(f"  {class_name}: {count}개 ({count/len(y_train)*100:.1f}%)")
    
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # ========== GPU/CPU 감지 ==========
    print("\n[STEP 3] 학습 환경")
    print("-"*40)
    
    use_gpu = False
    try:
        test_model = xgb.XGBClassifier(
            n_estimators=5, max_depth=3, random_state=42,
            tree_method='gpu_hist', gpu_id=0
        )
        test_model.fit(X_tr[:100], y_tr[:100], verbose=False)
        use_gpu = True
        print("  ✅ GPU 모드")
    except:
        print("  ⚠️ CPU 모드")
        use_gpu = False
    
    # ========== 가중치 ==========
    print("\n[STEP 4] 클래스 가중치")
    print("-"*40)
    
    class_weights = {0: 1.0, 1: 1.0, 2: 5.0}
    sample_weights = np.array([class_weights[label] for label in y_tr])
    
    print(f"  Class 0 (<0): 1.0배")
    print(f"  Class 1 (0~50): 1.0배")
    print(f"  Class 2 (50+): 5.0배 (급증 중점)")
    
    # ========== 모델 학습 ==========
    print("\n[STEP 5] 모델 학습")
    print("-"*40)
    
    if use_gpu:
        model = xgb.XGBClassifier(
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
            objective='multi:softprob',
            num_class=3
        )
    else:
        model = xgb.XGBClassifier(
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
            objective='multi:softprob',
            num_class=3
        )
    
    model.fit(X_tr, y_tr, sample_weight=sample_weights, verbose=False)
    
    print("✅ 학습 완료!")
    
    # ========== 검증 ==========
    print("\n[STEP 6] 검증")
    print("-"*40)
    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    
    accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"\n📊 검증 성능:")
    print(f"  전체 정확도: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    class_names = ['Class 0 (<0)', 'Class 1 (0~50)', 'Class 2 (50+)']
    print(classification_report(y_val, y_val_pred, target_names=class_names))
    
    # Confusion Matrix
    print(f"\n🎯 Confusion Matrix:")
    cm = confusion_matrix(y_val, y_val_pred)
    print(f"              예측")
    print(f"        Class 0  Class 1  Class 2")
    for i, row in enumerate(cm):
        print(f"실제 {i}  {row[0]:7d}  {row[1]:7d}  {row[2]:7d}")
    
    # Class 2 성능
    if cm[:, 2].sum() > 0 and cm[2, :].sum() > 0:
        class2_precision = cm[2, 2] / cm[:, 2].sum()
        class2_recall = cm[2, 2] / cm[2, :].sum()
        
        print(f"\n🔥 급증(Class 2) 감지 성능:")
        print(f"  - Precision: {class2_precision:.4f} ({class2_precision*100:.1f}%)")
        print(f"  - Recall: {class2_recall:.4f} ({class2_recall*100:.1f}%)")
    
    # ========== 모델 저장 ==========
    model_filename = 'xgboost_범주형_V9.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ 모델 저장: {model_filename}")
    
    # ========== Feature 중요도 ==========
    print("\n🔥 Feature 중요도 Top 20:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    for i, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "="*80)
    print("✅ V9-범주형 학습 완료!")
    print("="*80)

if __name__ == '__main__':
    print("🚀 V9-범주형 Classification 학습\n")
    train_v9_classification()
    print(f"\n🎉 완료!")