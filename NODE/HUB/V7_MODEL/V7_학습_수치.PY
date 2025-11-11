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

def train_v8_version():
    """
    🎯 V8 버전: V7 + QUE 2개 컬럼 추가
    - 30분 시퀀스 → 10분 후 예측
    - 급증 케이스 가중치 10배
    - 예측 임계값 280
    - Momentum Feature 4개
    - surge_imminent
    - 🆕 QUE 컬럼 2개 추가!
    """
    print("="*80)
    print("🎯 V8 버전: V7 + QUE 2개 컬럼 추가")
    print("="*80)
   
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
        'hub': ['HUBROOMTOTAL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA'],
        # 🆕 V8: QUE 컬럼 2개 추가
        'que': ['M16HUB.QUE.ALL.CURRENTQCNT', 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN']
    }
   
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
   
    def create_features_v8(df, start_idx=30):
        """V8: 30분 시퀀스 → 10분 후 예측 Feature 생성 (QUE 2개 추가)"""
        features_list = []
        labels = []
        seq_max_list = []
        seq_min_list = []
        indices = []
        seq_target_list = []
       
        for i in range(start_idx, len(df) - 10):  # ✅ 10분 후 예측
            seq_target = df[TARGET_COL].iloc[i-30:i].values
           
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
           
            # 🆕 Momentum Feature 추가
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
                   
                    elif group_name in ['cmd', 'storage', 'fs_storage', 'hub', 'que']:  # 🆕 que 추가!
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
           
            # 🎯 FS Storage Feature
            if 'CD_M163FSTORAGEUSE' in df.columns and 'CD_M163FSTORAGETOTAL' in df.columns:
                storage_use = df['CD_M163FSTORAGEUSE'].iloc[i-30:i].values
                storage_total = df['CD_M163FSTORAGETOTAL'].iloc[i-30:i].values
                storage_util = df['CD_M163FSTORAGEUTIL'].iloc[i-30:i].values
               
                features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
                features['storage_remaining'] = storage_total[-1] - storage_use[-1]
                features['storage_util_last'] = storage_util[-1]
                features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
                features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
           
            # 🎯 HUBROOMTOTAL Feature
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
           
            # 🎯 HUB × CMD 복합
            if 'HUBROOMTOTAL' in df.columns:
                hub_last = df['HUBROOMTOTAL'].iloc[i-1]
                features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
           
            # 🎯 Storage Util 위험
            if 'M16A_3F_STORAGE_UTIL' in df.columns:
                storage_util = df['M16A_3F_STORAGE_UTIL'].iloc[i-1]
                features['storage_util_critical'] = 1 if storage_util >= 205 else 0
                features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
           
            # 🔥 급증 위험도 점수
            features['surge_risk_score'] = (
                features.get('hub_high', 0) * 3 +
                features.get('storage_util_critical', 0) * 2 +
                features.get('total_cmd_low', 0) * 1 +
                features.get('storage_util_high', 0) * 1
            )
           
            # 🆕 급증 임박 신호
            features['surge_imminent'] = 1 if (
                seq_target[-1] > 280 and
                features.get('target_acceleration', 0) > 0.5 and
                features.get('hub_high', 0) == 1
            ) else 0
           
            # 🆕 V8: QUE Boolean Feature
            if 'M16HUB.QUE.ALL.CURRENTQCNT' in df.columns:
                currentq = df['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[i-1]
                features['currentq_high'] = 1 if currentq >= 1200 else 0
                features['currentq_critical'] = 1 if currentq >= 1400 else 0
           
            if 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in df.columns:
                avgtime = df['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[i-1]
                features['avgtime1min_high'] = 1 if avgtime >= 4.0 else 0
                features['avgtime1min_critical'] = 1 if avgtime >= 4.5 else 0
           
            # 🆕 V8: QUE 복합 Boolean
            if 'M16HUB.QUE.ALL.CURRENTQCNT' in df.columns and 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in df.columns:
                currentq = df['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[i-1]
                avgtime = df['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[i-1]
                features['que_severe_bottleneck'] = 1 if (currentq >= 1200 and avgtime >= 4.0) else 0
           
            features_list.append(features)
            labels.append(df[TARGET_COL].iloc[i:i+10].max())  # ✅ 10분 후 최대값
            seq_max_list.append(np.max(seq_target))
            seq_min_list.append(np.min(seq_target))
            indices.append(i)
            seq_target_list.append(seq_target)
       
        return pd.DataFrame(features_list), np.array(labels), seq_max_list, seq_min_list, indices, seq_target_list
   
    # ===== 1. 학습 =====
    print("\n[STEP 1] 데이터 학습 (V8: QUE 2개 추가)")
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
   
    # 🆕 V8: QUE 컬럼 확인
    que_cols_available = []
    for col in FEATURE_COLS['que']:
        if col in df_train.columns:
            que_cols_available.append(col)
            print(f"  ✅ {col} 컬럼 존재")
        else:
            print(f"  ❌ {col} 컬럼 없음")
   
    if len(que_cols_available) == 0:
        print("\n⚠️ 경고: QUE 컬럼이 없습니다. V7과 동일하게 학습됩니다.")
   
    # Feature 생성
    X_train, y_train, seq_max_list, _, _, seq_target_list = create_features_v8(df_train)
   
    print(f"\n✅ Feature 생성 완료:")
    print(f"  - 총 Feature: {len(X_train.columns)}개")
    print(f"  - 학습 샘플: {len(X_train)}개")
    print(f"  - 기존 V7 Feature: 약 100개")
    print(f"  - 🆕 V8 QUE 수치형: {len(que_cols_available) * 7}개")
    print(f"  - 🆕 V8 QUE Boolean: ~5개")
   
    # 급증 케이스 확인
    surge_count = sum(1 for i in range(len(y_train))
                      if seq_max_list[i] < 300 and y_train[i] >= 300)
    print(f"  - 급증 케이스: {surge_count}개 ({surge_count/len(y_train)*100:.2f}%)")
   
    # 🔥 가중치 10배 적용
    sample_weights = np.ones(len(y_train))
    for i in range(len(y_train)):
        if seq_max_list[i] < 300 and y_train[i] >= 300:
            sample_weights[i] = 10.0  # ✅ 10배 가중치!
   
    print(f"  - 가중치 적용: 급증 케이스 10배")
   
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
        test_model.fit(X_tr[:100], y_tr[:100], verbose=False)
        use_gpu = True
        print("  ✅ GPU 모드\n")
    except:
        print("  ⚠️ CPU 모드\n")
        use_gpu = False
   
    # 모델 생성
    if use_gpu:
        model = xgb.XGBRegressor(
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
        model = xgb.XGBRegressor(
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
   
    print("모델 학습 중 (V8: 가중치 10배 + Momentum + QUE 2개)...")
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
   
    # 평가
    y_val_pred = model.predict(X_val)
    train_mae = mean_absolute_error(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_val, y_val_pred)
   
    print(f"\n학습 성능:")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²:   {train_r2:.4f}")
   
    # 모델 저장
    with open('xgboost_V8_que2.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 모델 저장: xgboost_V8_que2.pkl")
   
    # Feature 중요도
    print("\n🔥 Feature 중요도 Top 40:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(40)
   
    for idx, row in feature_importance.iterrows():
        # 🆕 V8 QUE Feature 표시
        if 'M16HUB.QUE' in row['feature']:
            marker = "🆕"
        elif any(keyword in row['feature'] for keyword in ['acceleration', 'rising', 'rapid_rise', 'imminent', 'last_10_mean']):
            marker = "💨"
        elif 'surge_risk_score' in row['feature']:
            marker = "🔥"
        elif any(keyword in row['feature'] for keyword in ['hub_high', 'hub_critical', 'storage_util_high',
                                                          'total_cmd_low', 'hub_storage_risk', 'hub_cmd_bottleneck',
                                                          'storage_util_critical', 'currentq', 'avgtime']):
            marker = "🎯"
        elif 'HUBROOM' in row['feature'] or 'storage_util' in row['feature'] or 'CMD' in row['feature']:
            marker = "⚡"
        else:
            marker = "  "
        print(f"{marker} {row['feature']}: {row['importance']:.4f}")
   
    # ===== 2. 평가 =====
    print("\n[STEP 2] 평가 (임계값 280)")
    print("-"*40)
   
    split_idx = int(len(df_train) * 0.8)
    df_test = df_train.iloc[split_idx:].copy()
   
    X_test, y_test, seq_max_list_test, _, _, seq_target_list_test = create_features_v8(df_test, start_idx=30)
   
    y_pred = model.predict(X_test)
   
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
   
    print(f"\n평가 성능:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
   
    # 🔥 급증 예측 성능
    surge_count = sum(1 for i in range(len(y_test))
                      if seq_max_list_test[i] < 300 and y_test[i] >= 300)
    
    # 290, 285, 280, 275 각각 테스트
    thresholds = [290, 285, 280, 275]
    print(f"\n🎯 급증 예측 성능 (발생: {surge_count}개):")
    
    for threshold in thresholds:
        surge_detected = sum(1 for i in range(len(y_test))
                            if seq_max_list_test[i] < 300 and y_test[i] >= 300 and y_pred[i] >= threshold)
        marker = "⭐" if threshold == 280 else "  "
        print(f"{marker} 임계값 {threshold}: {surge_detected}/{surge_count}개 ({surge_detected/surge_count*100 if surge_count > 0 else 0:.1f}%)")
   
    # 극단값 성능
    extreme_mask = y_test >= 300
    extreme_count = extreme_mask.sum()
    extreme_detected_280 = ((y_pred >= 280) & extreme_mask).sum()
   
    print(f"\n극단값 (300+) 감지 (임계값 280):")
    print(f"  발생: {extreme_count}개")
    print(f"  감지: {extreme_detected_280}/{extreme_count}개 ({extreme_detected_280/extreme_count*100 if extreme_count > 0 else 0:.1f}%)")
   
    # 결과 저장
    results = []
    for i in range(len(y_test)):
        is_surge = (seq_max_list_test[i] < 300 and y_test[i] >= 300)
        results.append({
            '실제값': y_test[i],
            '예측값': round(y_pred[i], 2),
            '오차': round(abs(y_test[i] - y_pred[i]), 2),
            '시퀀스MAX': seq_max_list_test[i],
            '급증': 'O' if is_surge else '-',
            '감지_290': 'O' if (is_surge and y_pred[i] >= 290) else '-',
            '감지_285': 'O' if (is_surge and y_pred[i] >= 285) else '-',
            '감지_280': 'O' if (is_surge and y_pred[i] >= 280) else '-',
            '감지_275': 'O' if (is_surge and y_pred[i] >= 275) else '-',
            '극단값': 'O' if y_test[i] >= 300 else '-',
            '감지': 'O' if (y_test[i] >= 300 and y_pred[i] >= 280) else '-'
        })
   
    df_results = pd.DataFrame(results)
    df_results.to_csv('V8_que2_결과.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: V8_que2_결과.csv")
   
    # 최종 요약
    print("\n" + "="*80)
    print("📊 최종 요약 (V8: V7 + QUE 2개)")
    print("="*80)
   
    print(f"\n1. 모델 설정:")
    print(f"   - 시퀀스: 30분")
    print(f"   - 예측: 10분 후")
    print(f"   - 급증 가중치: 10배")
    print(f"   - 감지 임계값: 280")
    print(f"   - V7 Feature: Momentum 4개 + surge_imminent")
    print(f"   - 🆕 V8 Feature: QUE 2개 컬럼 ({len(que_cols_available) * 7 + 5}개 Feature)")
   
    print(f"\n2. 성능:")
    print(f"   - 학습 MAE: {train_mae:.2f}")
    print(f"   - 평가 MAE: {test_mae:.2f}")
    print(f"   - R²: {test_r2:.3f}")
   
    surge_detected_280 = sum(1 for i in range(len(y_test))
                            if seq_max_list_test[i] < 300 and y_test[i] >= 300 and y_pred[i] >= 280)
    print(f"\n3. 급증 예측 (임계값 280):")
    print(f"   - 발생: {surge_count}개")
    print(f"   - 성공률: {surge_detected_280/surge_count*100 if surge_count > 0 else 0:.1f}%")
   
    if len(que_cols_available) == 2:
        print(f"\n4. 🆕 V8 추가 컬럼:")
        print(f"   - M16HUB.QUE.ALL.CURRENTQCNT (큐 수)")
        print(f"   - M16HUB.QUE.TIME.AVGTOTALTIME1MIN (반송시간)")
        print(f"   → V7 대비 성능 향상 예상: 60% → 63~67%")
   
    return model, df_results, feature_importance

if __name__ == '__main__':
    model, results, importance = train_v8_version()