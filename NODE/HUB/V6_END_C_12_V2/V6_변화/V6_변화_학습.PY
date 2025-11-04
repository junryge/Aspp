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

def train_multi_change_prediction():
    """
    🎯 변화량 예측 버전: 30분 시퀀스 → 10분 후 MIN/MAX/AVG 변화량 예측
    - 타겟: [MIN 변화량, MAX 변화량, AVG 변화량]
    - 큰 변화 기준: ±50
    - 멀티아웃풋 예측
    """
    print("="*80)
    print("🎯 변화량 예측 모델: MIN / MAX / AVG 동시 예측")
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
   
    def create_features_multi_change(df, start_idx=30):
        """30분 시퀀스 → 10분 후 MIN/MAX/AVG 변화량 Feature 생성"""
        features_list = []
        labels = []  # [MIN 변화량, MAX 변화량, AVG 변화량]
        current_values = []
        future_actual_values = []  # 10분 후 실제값 (1개!)
        future_min_values = []
        future_max_values = []
        future_avg_values = []
        indices = []
       
        for i in range(start_idx, len(df) - 10):
            seq_target = df[TARGET_COL].iloc[i-30:i].values
           
            # 현재값 (시퀀스 마지막)
            current_value = seq_target[-1]
           
            # 🎯 정확히 10분 후 실제값 (1개!)
            future_actual = df[TARGET_COL].iloc[i+10]
            
            # 🎯 10분 구간의 MIN, MAX, AVG (예측용)
            future_10min = df[TARGET_COL].iloc[i:i+10].values
            future_min = np.min(future_10min)
            future_max = np.max(future_10min)
            future_avg = np.mean(future_10min)
           
            # 🎯 실제 변화량 (1개!)
            actual_change = future_actual - current_value
            
            # 🎯 각각의 변화량 (예측 타겟)
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
           
            # Momentum Feature
            features['target_acceleration'] = (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5
            features['target_is_rising'] = 1 if seq_target[-1] > seq_target[-5] else 0
            features['target_rapid_rise'] = 1 if (seq_target[-1] - seq_target[-5] > 10) else 0
            features['target_last_10_mean'] = np.mean(seq_target[-10:])
           
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
           
            # 유입-유출, CMD
            inflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['inflow'] if col in df.columns)
            outflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['outflow'] if col in df.columns)
            features['net_flow'] = inflow_sum - outflow_sum
           
            cmd_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['cmd'] if col in df.columns)
            features['total_cmd'] = cmd_sum
            features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
            features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
           
            # HUB × CMD 복합
            if 'HUBROOMTOTAL' in df.columns:
                hub_last = df['HUBROOMTOTAL'].iloc[i-1]
                features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
           
            # Storage Util 위험
            if 'M16A_3F_STORAGE_UTIL' in df.columns:
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
           
            features_list.append(features)
            labels.append([change_min, change_max, change_avg])  # 🎯 3개 타겟!
            current_values.append(current_value)
            future_actual_values.append(future_actual)  # 🎯 실제값 1개!
            future_min_values.append(future_min)
            future_max_values.append(future_max)
            future_avg_values.append(future_avg)
            indices.append(i)
       
        return (pd.DataFrame(features_list), np.array(labels), 
                current_values, future_actual_values, future_min_values, future_max_values, future_avg_values, indices)
   
    # ===== 1. 학습 =====
    print("\n[STEP 1] 데이터 학습 (30시퀀스 → 10분 후 MIN/MAX/AVG 변화량)")
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
   
    # Feature 생성
    X_train, y_train, current_vals, actual_vals, min_vals, max_vals, avg_vals, _ = create_features_multi_change(df_train)
   
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
   
    print("모델 학습 중 (MIN/MAX/AVG 변화량 동시 예측)...")
    model.fit(X_tr, y_tr)
   
    # 평가
    y_val_pred = model.predict(X_val)
    
    print(f"\n학습 성능:")
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        mae = mean_absolute_error(y_val[:, idx], y_val_pred[:, idx])
        rmse = np.sqrt(mean_squared_error(y_val[:, idx], y_val_pred[:, idx]))
        r2 = r2_score(y_val[:, idx], y_val_pred[:, idx])
        print(f"  {name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
   
    # 모델 저장
    model_filename = 'xgboost_변화량예측_MIN_MAX_AVG.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ 모델 저장: {model_filename}")
   
    # Feature 중요도 (첫 번째 모델 기준)
    print("\n🔥 Feature 중요도 Top 35:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.estimators_[0].feature_importances_
    }).sort_values('importance', ascending=False).head(35)
   
    for idx, row in feature_importance.iterrows():
        if any(keyword in row['feature'] for keyword in ['acceleration', 'rising', 'rapid_rise', 'imminent', 'last_10_mean']):
            marker = "🆕"
        elif 'surge_risk_score' in row['feature']:
            marker = "🔥"
        elif any(keyword in row['feature'] for keyword in ['hub_high', 'hub_critical', 'storage_util_high',
                                                          'total_cmd_low', 'hub_storage_risk', 'hub_cmd_bottleneck',
                                                          'storage_util_critical']):
            marker = "🎯"
        elif 'HUBROOM' in row['feature'] or 'storage_util' in row['feature'] or 'CMD' in row['feature']:
            marker = "⚡"
        else:
            marker = "  "
        print(f"{marker} {row['feature']}: {row['importance']:.4f}")
   
    # ===== 2. 평가 =====
    print("\n[STEP 2] 평가 (MIN/MAX/AVG 변화량)")
    print("-"*40)
   
    split_idx = int(len(df_train) * 0.8)
    df_test = df_train.iloc[split_idx:].copy()
   
    X_test, y_test, current_vals_test, actual_vals_test, min_vals_test, max_vals_test, avg_vals_test, _ = create_features_multi_change(df_test, start_idx=30)
   
    y_pred = model.predict(X_test)
   
    print(f"\n평가 성능:")
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        mae = mean_absolute_error(y_test[:, idx], y_pred[:, idx])
        rmse = np.sqrt(mean_squared_error(y_test[:, idx], y_pred[:, idx]))
        r2 = r2_score(y_test[:, idx], y_pred[:, idx])
        print(f"  {name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
   
    # 결과 저장
    results = []
    for i in range(len(y_test)):
        current_val = current_vals_test[i]
        
        # 🎯 실제값 (1개!)
        actual_value = actual_vals_test[i]
        actual_change = actual_value - current_val
        
        # 예측용 MIN/MAX/AVG 값
        pred_min_val = min_vals_test[i]
        pred_max_val = max_vals_test[i]
        pred_avg_val = avg_vals_test[i]
        
        # 예측 변화량 (3개)
        pred_change_min = y_pred[i, 0]
        pred_change_max = y_pred[i, 1]
        pred_change_avg = y_pred[i, 2]
        
        # 예측값 (3개)
        pred_min = current_val + pred_change_min
        pred_max = current_val + pred_change_max
        pred_avg = current_val + pred_change_avg
        
        results.append({
            '현재값': round(current_val, 2),
            '실제값': round(actual_value, 2),
            '실제_변화량': round(actual_change, 2),
            '구간_MIN': round(pred_min_val, 2),
            '구간_MAX': round(pred_max_val, 2),
            '구간_AVG': round(pred_avg_val, 2),
            '예측_변화량_MIN': round(pred_change_min, 2),
            '예측_변화량_MAX': round(pred_change_max, 2),
            '예측_변화량_AVG': round(pred_change_avg, 2),
            '예측값_MIN': round(pred_min, 2),
            '예측값_MAX': round(pred_max, 2),
            '예측값_AVG': round(pred_avg, 2),
            '오차_MIN': round(abs(y_test[i, 0] - pred_change_min), 2),
            '오차_MAX': round(abs(y_test[i, 1] - pred_change_max), 2),
            '오차_AVG': round(abs(y_test[i, 2] - pred_change_avg), 2),
        })
   
    df_results = pd.DataFrame(results)
    output_filename = '변화량예측_MIN_MAX_AVG_결과.csv'
    df_results.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_filename}")
   
    # 최종 요약
    print("\n" + "="*80)
    print("📊 최종 요약 (MIN/MAX/AVG 변화량 예측)")
    print("="*80)
   
    print(f"\n1. 모델 설정:")
    print(f"   - 시퀀스: 30분")
    print(f"   - 예측: 10분 구간의 MIN/MAX/AVG 변화량")
    print(f"   - 큰 변화 기준: ±50")
    print(f"   - Momentum Feature 포함")
   
    print(f"\n2. 성능:")
    for idx, name in enumerate(['MIN', 'MAX', 'AVG']):
        mae = mean_absolute_error(y_test[:, idx], y_pred[:, idx])
        print(f"   {name} - MAE: {mae:.2f}")
   
    return model, df_results, feature_importance

if __name__ == '__main__':
    model, results, importance = train_multi_change_prediction()