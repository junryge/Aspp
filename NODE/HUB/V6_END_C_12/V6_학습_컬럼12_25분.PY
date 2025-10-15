# -*- coding: utf-8 -*-
"""
30분 시퀀스 → 25분 후 예측 학습 코드
12개 핵심 컬럼 사용
"""

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# V4 Ultimate 필수 컬럼 정의 (12개 핵심 컬럼)
FEATURE_COLS = {
    'storage': ['M16A_3F_STORAGE_UTIL'],
    'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
    'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
    'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
    'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
}

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'

def create_features(df, seq_len=30, pred_offset=25):
    """
    30분 시퀀스 → 25분 후 예측
    
    Args:
        df: 원본 데이터프레임
        seq_len: 시퀀스 길이 (30분)
        pred_offset: 예측 오프셋 (25분 후)
    
    Returns:
        X, y, timestamps, actuals, sequences
    """
    X_features = []
    y_values = []
    timestamps = []
    actual_values = []
    sequences = []
    
    print(f"\n특성 생성 중... (시퀀스={seq_len}분, 예측={pred_offset}분 후)")
    
    for i in range(seq_len, len(df) - pred_offset):
        # 과거 30개 데이터
        seq_data = df.iloc[i-seq_len:i].copy()
        seq_target = seq_data[TARGET_COL].values
        
        # Feature 생성 (타겟 컬럼)
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_last_5_mean': np.mean(seq_target[-5:]),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
            'target_last_10_mean': np.mean(seq_target[-10:]),
            'target_first_10_mean': np.mean(seq_target[:10])
        }
        
        # 각 컬럼 그룹별 특성 추가
        for group_name, cols in FEATURE_COLS.items():
            for col in cols:
                if col in df.columns:
                    col_seq = seq_data[col].values
                    
                    # 기본 통계
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_std'] = np.std(col_seq)
                    features[f'{col}_max'] = np.max(col_seq)
                    features[f'{col}_min'] = np.min(col_seq)
                    
                    # 최근 특성
                    features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                    features[f'{col}_last_10_mean'] = np.mean(col_seq[-10:])
                    
                    # 추세
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                    
                    # 구간별 평균
                    features[f'{col}_first_10_mean'] = np.mean(col_seq[:10])
                    features[f'{col}_mid_10_mean'] = np.mean(col_seq[10:20])
                    features[f'{col}_last_value'] = col_seq[-1]
        
        # 유입-유출 차이 (Net Flow)
        inflow_sum = 0
        outflow_sum = 0
        for col in FEATURE_COLS['inflow']:
            if col in df.columns:
                inflow_sum += seq_data[col].iloc[-1]
        for col in FEATURE_COLS['outflow']:
            if col in df.columns:
                outflow_sum += seq_data[col].iloc[-1]
        features['net_flow'] = inflow_sum - outflow_sum
        
        # CMD 총합
        cmd_sum = 0
        for col in FEATURE_COLS['cmd']:
            if col in df.columns:
                cmd_sum += seq_data[col].iloc[-1]
        features['total_cmd'] = cmd_sum
        
        X_features.append(features)
        
        # 25분 후 실제값
        y_values.append(df.iloc[i + pred_offset][TARGET_COL])
        
        # 시간 정보
        if 'STAT_DT' in df.columns:
            timestamps.append(df.iloc[i]['STAT_DT'])
            actual_values.append(df.iloc[i + pred_offset][TARGET_COL])
        
        sequences.append(seq_target)
        
        if (i - seq_len) % 1000 == 0:
            print(f"  진행: {i-seq_len}/{len(df)-seq_len-pred_offset}")
    
    X = pd.DataFrame(X_features)
    y = np.array(y_values)
    
    return X, y, timestamps, actual_values, sequences


def main():
    print("="*80)
    print("🚀 30분→25분 후 예측 모델 학습")
    print("="*80)
    
    # ===== 1. 학습 단계 =====
    print("\n[STEP 1] 학습 데이터 로드")
    print("-"*40)
    
    df_train = pd.read_csv('HUB_0509_TO_09_DATA.csv', on_bad_lines='skip')
    
    print(f"학습 데이터: {len(df_train)}개 행")
    print(f"사용 가능한 컬럼 확인:")
    
    all_feature_cols = []
    for group_name, cols in FEATURE_COLS.items():
        available = [col for col in cols if col in df_train.columns]
        all_feature_cols.extend(available)
        print(f"  - {group_name}: {len(available)}/{len(cols)}개")
    
    # STAT_DT 처리
    if 'STAT_DT' in df_train.columns:
        try:
            df_train['STAT_DT'] = pd.to_datetime(df_train['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            print("⚠️ STAT_DT 변환 실패, 가상 시간 생성")
            base_time = datetime(2024, 1, 1, 0, 0)
            df_train['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df_train))]
    
    # 학습 데이터 생성
    X_train, y_train, _, _, _ = create_features(df_train, seq_len=30, pred_offset=25)
    
    print(f"\n생성된 Feature 수: {len(X_train.columns)}개")
    print(f"학습 샘플 수: {len(X_train)}개")
    
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # GPU/CPU 자동 선택
    print("\n🔍 학습 환경 감지 중...")
    use_gpu = False
    
    try:
        print("  → GPU 모드 시도 중...")
        test_model = xgb.XGBRegressor(
            n_estimators=5,
            max_depth=3,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0
        )
        test_model.fit(X_tr[:100], y_tr[:100], verbose=False)
        use_gpu = True
        print("  ✅ GPU 사용 가능! GPU 모드로 학습합니다.\n")
    except Exception as e:
        print(f"  ⚠️ GPU 사용 불가: {str(e)[:100]}")
        print("  → CPU 모드로 전환합니다.\n")
        use_gpu = False
    
    # 실제 모델 생성 (25분 예측 - n_estimators=100)
    if use_gpu:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.15,
            subsample=0.5,
            colsample_bytree=0.5,
            min_child_weight=10,
            gamma=0.7,
            reg_alpha=0.8,
            reg_lambda=3.0,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor'
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.15,
            subsample=0.5,
            colsample_bytree=0.5,
            min_child_weight=10,
            gamma=0.7,
            reg_alpha=0.8,
            reg_lambda=3.0,
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
    
    print("모델 학습 중 (30분→25분 후 예측)...")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 모델 저장
    model_filename = 'xgboost_model_30min_25min_12컬럼.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ 모델 저장 완료: {model_filename}")
    
    # 검증 성능
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print("\n" + "="*80)
    print("📊 검증 세트 성능")
    print("="*80)
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")
    
    # 극단값 성능
    extreme_mask = y_val >= 300
    if extreme_mask.sum() > 0:
        extreme_mae = mean_absolute_error(y_val[extreme_mask], y_pred[extreme_mask])
        print(f"\n극단값(≥300) MAE: {extreme_mae:.2f}")
        print(f"극단값 샘플 수: {extreme_mask.sum()}개")
    
    print("\n✅ 학습 완료!")


if __name__ == '__main__':
    main()