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

def train_change_prediction():
    """
    🎯 변화량 예측 모델 (단일 모델로 급증/급감 해결)
    
    핵심 아이디어:
    - 타겟: 10분 후 변화량 (delta = future - current)
    - 급증: 변화량 +30 이상 (양수)
    - 급감: 변화량 -30 이하 (음수)
    - 정상: -30 ~ +30 범위
    
    장점:
    - 단일 모델로 급증/급감 자동 구분
    - 기존 Feature 100% 활용
    - 부호로 명확히 구분됨
    """
    print("="*80)
    print("🎯 변화량 예측 모델 학습 (급증/급감 통합)")
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
   
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
   
    def create_features_for_change(df, start_idx=30):
        """변화량 예측을 위한 Feature 생성"""
        features_list = []
        labels = []  # 변화량
        current_values = []
        future_values = []
        indices = []
        seq_target_list = []
       
        for i in range(start_idx, len(df) - 10):
            seq_target = df[TARGET_COL].iloc[i-30:i].values
            current_value = seq_target[-1]  # 현재값
            future_value = df[TARGET_COL].iloc[i+10]  # 10분 후
            change = future_value - current_value  # 변화량 (타겟!)
           
            # 기본 타겟 Feature
            features = {
                'target_mean': np.mean(seq_target),
                'target_std': np.std(seq_target),
                'target_max': np.max(seq_target),
                'target_min': np.min(seq_target),
                'target_last_value': seq_target[-1],
                'target_last_5_mean': np.mean(seq_target[-5:]),
                'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
            }
           
            # 변화량 예측에 유용한 추가 Feature
            features['target_from_max'] = np.max(seq_target) - seq_target[-1]
            features['target_decreasing'] = 1 if features['target_slope'] < -1 else 0
            features['target_increasing'] = 1 if features['target_slope'] > 1 else 0
            features['target_volatility'] = np.std(np.diff(seq_target))
            features['target_momentum'] = np.mean(seq_target[-5:]) - np.mean(seq_target[:5])
           
            # 각 컬럼 그룹 Feature
            for group_name, cols in FEATURE_COLS.items():
                for col in cols:
                    if col not in df.columns:
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
                        features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                        features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                   
                    else:
                        features[f'{col}_mean'] = np.mean(col_seq)
                        features[f'{col}_last_value'] = col_seq[-1]
                        features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
           
            # FS Storage Feature
            if 'CD_M163FSTORAGEUSE' in df.columns and 'CD_M163FSTORAGETOTAL' in df.columns:
                storage_use = df['CD_M163FSTORAGEUSE'].iloc[i-30:i].values
                storage_total = df['CD_M163FSTORAGETOTAL'].iloc[i-30:i].values
                storage_util = df['CD_M163FSTORAGEUTIL'].iloc[i-30:i].values
               
                features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
                features['storage_remaining'] = storage_total[-1] - storage_use[-1]
                features['storage_util_last'] = storage_util[-1]
                features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
                features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
           
            # HUBROOMTOTAL Feature
            if 'HUBROOMTOTAL' in df.columns:
                hub_seq = df['HUBROOMTOTAL'].iloc[i-30:i].values
                hub_last = hub_seq[-1]
               
                features['hub_critical'] = 1 if hub_last < 590 else 0
                features['hub_high'] = 1 if hub_last < 610 else 0
                features['hub_warning'] = 1 if hub_last < 620 else 0
                features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
               
                if 'CD_M163FSTORAGEUTIL' in df.columns:
                    storage_util_last = df['CD_M163FSTORAGEUTIL'].iloc[i-1]
                    features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
           
            # CMD Feature
            inflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['inflow'] if col in df.columns)
            outflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['outflow'] if col in df.columns)
            features['net_flow'] = inflow_sum - outflow_sum
           
            cmd_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['cmd'] if col in df.columns)
            features['total_cmd'] = cmd_sum
            features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
            features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
           
            if 'HUBROOMTOTAL' in df.columns:
                hub_last = df['HUBROOMTOTAL'].iloc[i-1]
                features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
           
            # Storage Util
            if 'M16A_3F_STORAGE_UTIL' in df.columns:
                storage_util = df['M16A_3F_STORAGE_UTIL'].iloc[i-1]
                features['storage_util_critical'] = 1 if storage_util >= 205 else 0
                features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
           
            features_list.append(features)
            labels.append(change)  # 🔥 타겟: 변화량!
            current_values.append(current_value)
            future_values.append(future_value)
            indices.append(i)
            seq_target_list.append(seq_target)
       
        return (pd.DataFrame(features_list), np.array(labels), 
                current_values, future_values, indices, seq_target_list)
   
    # ===== 1. 학습 =====
    print("\n[STEP 1] 데이터 로드 및 Feature 생성")
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
   
    X_train, y_train, current_vals, future_vals, _, seq_list = create_features_for_change(df_train)
   
    print(f"\n✅ Feature 생성 완료:")
    print(f"  - 총 Feature: {len(X_train.columns)}개")
    print(f"  - 학습 샘플: {len(X_train)}개")
   
    # 변화량 분포 확인
    print(f"\n변화량 분포:")
    print(f"  평균: {np.mean(y_train):.2f}")
    print(f"  표준편차: {np.std(y_train):.2f}")
    print(f"  최소: {np.min(y_train):.2f}")
    print(f"  최대: {np.max(y_train):.2f}")
   
    # 급증/급감 케이스 확인
    surge_cases = sum(1 for i in range(len(y_train))
                      if current_vals[i] < 300 and future_vals[i] >= 300)
    drop_cases = sum(1 for i in range(len(y_train))
                     if current_vals[i] >= 300 and future_vals[i] < 300)
   
    print(f"\n케이스 분포:")
    print(f"  🔺 급증 (current<300, future≥300): {surge_cases}개 ({surge_cases/len(y_train)*100:.2f}%)")
    print(f"  🔻 급감 (current≥300, future<300): {drop_cases}개 ({drop_cases/len(y_train)*100:.2f}%)")
    print(f"  ⚪ 정상: {len(y_train)-surge_cases-drop_cases}개")
   
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
        test_model.fit(X_tr[:100], y_tr[:100], verbose=False)
        use_gpu = True
        print("  ✅ GPU 모드\n")
    except:
        print("  ⚠️ CPU 모드\n")
        use_gpu = False
   
    # 모델 생성
    if use_gpu:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.04,
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
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.04,
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
   
    print("모델 학습 중 (변화량 예측)...")
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
   
    # 평가
    y_val_pred = model.predict(X_val)
    train_mae = mean_absolute_error(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_val, y_val_pred)
   
    print(f"\n학습 성능:")
    print(f"  MAE (변화량):  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²:   {train_r2:.4f}")
   
    # 모델 저장
    with open('xgboost_변화량예측.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 모델 저장: xgboost_변화량예측.pkl")
   
    # Feature 중요도
    print("\n🔥 Feature 중요도 Top 20:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
   
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
   
    # ===== 2. 평가 =====
    print("\n[STEP 2] 테스트 데이터 평가")
    print("-"*40)
   
    split_idx = int(len(df_train) * 0.8)
    df_test = df_train.iloc[split_idx:].copy()
   
    X_test, y_test_change, current_test, future_test, indices, seq_test = create_features_for_change(df_test, start_idx=30)
   
    y_pred_change = model.predict(X_test)
   
    # 변화량 예측 성능
    test_mae = mean_absolute_error(y_test_change, y_pred_change)
    test_rmse = np.sqrt(mean_squared_error(y_test_change, y_pred_change))
    test_r2 = r2_score(y_test_change, y_pred_change)
   
    print(f"\n변화량 예측 성능:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
   
    # 급증/급감 감지 성능
    print(f"\n[STEP 3] 급증/급감 감지 성능")
    print("-"*40)
   
    surge_count = 0
    surge_detected = 0
    drop_count = 0
    drop_detected = 0
   
    for i in range(len(X_test)):
        current = current_test[i]
        future_actual = future_test[i]
        predicted_change = y_pred_change[i]
        predicted_future = current + predicted_change
       
        # 급증
        if current < 300 and future_actual >= 300:
            surge_count += 1
            if predicted_future >= 290:  # 여유있게 290
                surge_detected += 1
       
        # 급감
        elif current >= 300 and future_actual < 300:
            drop_count += 1
            if predicted_future < 310:  # 여유있게 310
                drop_detected += 1
   
    print(f"🔺 급증 감지:")
    print(f"  발생: {surge_count}개")
    print(f"  감지: {surge_detected}/{surge_count}개 ({surge_detected/surge_count*100 if surge_count > 0 else 0:.1f}%)")
   
    print(f"\n🔻 급감 감지:")
    print(f"  발생: {drop_count}개")
    print(f"  감지: {drop_detected}/{drop_count}개 ({drop_detected/drop_count*100 if drop_count > 0 else 0:.1f}%)")
   
    # 극단값 성능
    extreme_count = sum(1 for i in range(len(future_test)) if future_test[i] >= 300)
    extreme_detected = sum(1 for i in range(len(future_test)) 
                          if future_test[i] >= 300 and (current_test[i] + y_pred_change[i]) >= 290)
   
    print(f"\n극단값 (300+) 감지:")
    print(f"  발생: {extreme_count}개")
    print(f"  감지: {extreme_detected}/{extreme_count}개 ({extreme_detected/extreme_count*100 if extreme_count > 0 else 0:.1f}%)")
   
    # 결과 저장
    results = []
    for i in range(len(X_test)):
        current = current_test[i]
        future_actual = future_test[i]
        change_actual = y_test_change[i]
        change_pred = y_pred_change[i]
        predicted_future = current + change_pred
       
        is_surge = (current < 300 and future_actual >= 300)
        is_drop = (current >= 300 and future_actual < 300)
        is_extreme = (future_actual >= 300)
       
        results.append({
            '현재값': round(current, 2),
            '실제_10분후': round(future_actual, 2),
            '예측_10분후': round(predicted_future, 2),
            '실제_변화량': round(change_actual, 2),
            '예측_변화량': round(change_pred, 2),
            '변화량_오차': round(abs(change_actual - change_pred), 2),
            '급증': 'O' if is_surge else '-',
            '급증감지': 'O' if (is_surge and predicted_future >= 290) else '-',
            '급감': 'O' if is_drop else '-',
            '급감감지': 'O' if (is_drop and predicted_future < 310) else '-',
            '극단값': 'O' if is_extreme else '-',
            '극단값감지': 'O' if (is_extreme and predicted_future >= 290) else '-'
        })
   
    df_results = pd.DataFrame(results)
    df_results.to_csv('변화량예측_결과.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: 변화량예측_결과.csv")
   
    # 최종 요약
    print("\n" + "="*80)
    print("📊 최종 요약 (변화량 예측 방식)")
    print("="*80)
   
    print(f"\n1. 접근 방식:")
    print(f"   - 타겟: 10분 후 변화량")
    print(f"   - 급증/급감 자동 구분 (부호)")
    print(f"   - 단일 모델")
   
    print(f"\n2. 변화량 예측 성능:")
    print(f"   - MAE: {test_mae:.2f}")
    print(f"   - R²: {test_r2:.3f}")
   
    print(f"\n3. 급증 감지:")
    print(f"   - 성공률: {surge_detected/surge_count*100 if surge_count > 0 else 0:.1f}%")
   
    print(f"\n4. 급감 감지:")
    print(f"   - 성공률: {drop_detected/drop_count*100 if drop_count > 0 else 0:.1f}%")
   
    print(f"\n5. 극단값 (300+) 감지:")
    print(f"   - 발생: {extreme_count}개")
    print(f"   - 감지율: {extreme_detected/extreme_count*100 if extreme_count > 0 else 0:.1f}%")
   
    if surge_detected/surge_count >= 0.78 if surge_count > 0 else False:
        print(f"\n✅ 급증 감지율 78% 이상 달성!")
    else:
        print(f"\n⚠️ 급증 감지율 개선 필요")
   
    return model, df_results, feature_importance

if __name__ == '__main__':
    model, results, importance = train_change_prediction()