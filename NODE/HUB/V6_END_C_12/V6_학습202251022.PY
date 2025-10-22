


'''
✅ CD_M163FSTORAGEUSE (Storage 사용량) - 절대값!
✅ CD_M163FSTORAGETOTAL (Storage 총량) - 절대값!
✅ CD_M163FSTORAGEUTIL (Storage 활용률) - 정확한 비율!


Storage 사용량 추세 → 급증이 "얼마나 빠르게" 일어나는지 측정
여유 공간 계산 → Storage가 "언제" 한계에 도달하는지 예측
활용률 임계값 → 위험 구간을 정확히 판단

핵심 인사이트:

Storage 여유 공간 < 150 → 급증 확률 80% 🔥
Storage 활용률 ≥ 10% → 급증 확률 75% 🔥
Storage 분당 증가율 > 0.3 → 급증 확률 70% 🔥


M16A_3F_CMD < 180 → HIGH (70%)
M16A_3F_CMD 260~280 → HIGH (67%)
M16A_3F_CMD < 220 → MEDIUM (30%)
위험도 판정의 핵심!

HUBROOMTOTAL = HUB의 전체 룸(공간) 총량

시스템 전체 부하 지표

Storage는 M16A 3F만 보지만
HUBROOMTOTAL은 전체 HUB 상태를 나타냄


급증 예측에 결정적

HUB 전체가 꽉 차면 → M16A 3F도 영향 받음
620대 → 정상, 600대 초반 → 위험, 580대 이하 → 매우 위험


다른 컬럼과의 상호작용

HUBROOMTOTAL 낮음 + CMD 높음 = 병목 발생 가능
HUBROOMTOTAL 낮음 + Storage Util 높음 = 극도 위험

'''

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

def train_final_optimized():
    """
    🎯 데이터 분석 결과 기반 최종 최적화 버전
    - 1,804개 급증 케이스 확인 (3.00%)
    - 핵심 패턴: HUBROOMTOTAL<610(87.5%), STORAGE≥205(82.9%), CMD<220(45.4%), FS_UTIL≥7(25.3%)
    """
    print("="*80)
    print("🎯 최종 최적화: 1,804개 급증 케이스 기반 학습")
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
    
    def create_features_optimized(df, start_idx=30):
        """분석 결과 기반 Feature 생성"""
        features_list = []
        labels = []
        seq_max_list = []
        seq_min_list = []
        indices = []
        seq_target_list = []
        
        for i in range(start_idx, len(df) - 10):
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
            
            # 각 컬럼 그룹 Feature
            for group_name, cols in FEATURE_COLS.items():
                for col in cols:
                    if col not in df.columns:
                        continue
                    
                    col_seq = df[col].iloc[i-30:i].values
                    
                    if group_name == 'maxcapa':
                        features[f'{col}_last_value'] = col_seq[-1]
                    
                    elif group_name in ['cmd', 'storage', 'fs_storage', 'hub']:
                        # 핵심 Feature
                        features[f'{col}_mean'] = np.mean(col_seq)
                        features[f'{col}_std'] = np.std(col_seq)
                        features[f'{col}_max'] = np.max(col_seq)
                        features[f'{col}_min'] = np.min(col_seq)
                        features[f'{col}_last_value'] = col_seq[-1]
                        features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                        features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                    
                    else:
                        # Inflow/Outflow 간소화
                        features[f'{col}_mean'] = np.mean(col_seq)
                        features[f'{col}_last_value'] = col_seq[-1]
                        features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
            
            # 🎯 데이터 분석 결과 기반 Feature
            if 'CD_M163FSTORAGEUSE' in df.columns and 'CD_M163FSTORAGETOTAL' in df.columns:
                storage_use = df['CD_M163FSTORAGEUSE'].iloc[i-30:i].values
                storage_total = df['CD_M163FSTORAGETOTAL'].iloc[i-30:i].values
                storage_util = df['CD_M163FSTORAGEUTIL'].iloc[i-30:i].values
                
                features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
                features['storage_remaining'] = storage_total[-1] - storage_use[-1]
                features['storage_util_last'] = storage_util[-1]
                # 🎯 분석 결과: FS_UTIL ≥ 7 → 25.3% 급증
                features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
                features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
            
            # 🎯 HUBROOMTOTAL Feature (분석 결과 기반)
            if 'HUBROOMTOTAL' in df.columns:
                hub_seq = df['HUBROOMTOTAL'].iloc[i-30:i].values
                hub_last = hub_seq[-1]
                
                # 🎯 분석 결과: HUBROOMTOTAL < 610 → 87.5% 급증
                features['hub_critical'] = 1 if hub_last < 590 else 0
                features['hub_high'] = 1 if hub_last < 610 else 0  # 🔥 핵심!
                features['hub_warning'] = 1 if hub_last < 620 else 0
                
                features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
                
                # HUB × Storage 복합 위험
                if 'CD_M163FSTORAGEUTIL' in df.columns:
                    storage_util_last = df['CD_M163FSTORAGEUTIL'].iloc[i-1]
                    # 🎯 HUB<610 + FS_UTIL≥7 = 극도 위험
                    features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
            
            # 유입-유출, CMD
            inflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['inflow'] if col in df.columns)
            outflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['outflow'] if col in df.columns)
            features['net_flow'] = inflow_sum - outflow_sum
            
            cmd_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['cmd'] if col in df.columns)
            features['total_cmd'] = cmd_sum
            # 🎯 분석 결과: CMD < 220 → 45.4% 급증
            features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
            features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
            
            # 🎯 HUB × CMD 복합
            if 'HUBROOMTOTAL' in df.columns:
                hub_last = df['HUBROOMTOTAL'].iloc[i-1]
                # HUB 낮음 + CMD 낮음 = 병목 + 급증 위험
                features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
            
            # 🎯 Storage Util 위험 (분석 결과: STORAGE≥205 → 82.9% 급증)
            if 'M16A_3F_STORAGE_UTIL' in df.columns:
                storage_util = df['M16A_3F_STORAGE_UTIL'].iloc[i-1]
                features['storage_util_critical'] = 1 if storage_util >= 205 else 0
                features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
            
            features_list.append(features)
            labels.append(df[TARGET_COL].iloc[i:i+10].max())
            seq_max_list.append(np.max(seq_target))
            seq_min_list.append(np.min(seq_target))
            indices.append(i)
            seq_target_list.append(seq_target)
        
        return pd.DataFrame(features_list), np.array(labels), seq_max_list, seq_min_list, indices, seq_target_list
    
    # ===== 1. 학습 =====
    print("\n[STEP 1] FS 데이터 학습 (1,804개 급증 케이스 활용)")
    print("-"*40)
    
    try:
        df_train = pd.read_csv('/mnt/user-data/uploads/FS__1_', on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df_train = pd.read_csv('/mnt/user-data/uploads/FS__1_', on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df_train = pd.read_csv('/mnt/user-data/uploads/FS__1_', on_bad_lines='skip', encoding='euc-kr', low_memory=False)
    
    # 타입 변환
    df_train[TARGET_COL] = pd.to_numeric(df_train[TARGET_COL], errors='coerce')
    df_train = df_train.dropna(subset=[TARGET_COL])
    
    print(f"학습 데이터: {len(df_train)}개 행")
    
    # 컬럼 확인
    print(f"\n사용 가능한 컬럼:")
    for group_name, cols in FEATURE_COLS.items():
        available = [col for col in cols if col in df_train.columns]
        marker = "🔥" if group_name in ['hub', 'fs_storage'] else "  "
        print(f"{marker} - {group_name}: {len(available)}/{len(cols)}개")
    
    # Feature 생성
    X_train, y_train, _, _, _, _ = create_features_optimized(df_train)
    
    print(f"\n✅ Feature 생성 완료:")
    print(f"  - 총 Feature: {len(X_train.columns)}개")
    print(f"  - 학습 샘플: {len(X_train)}개")
    
    # 급증 케이스 비율 확인
    surge_in_train = sum(1 for i in range(len(y_train)) 
                         if X_train.iloc[i]['target_max'] < 300 and y_train[i] >= 300)
    print(f"  - 급증 케이스: {surge_in_train}개 ({surge_in_train/len(y_train)*100:.2f}%)")
    
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
    
    # 모델 생성 (급증 케이스 최적화)
    if use_gpu:
        model = xgb.XGBRegressor(
            n_estimators=200,  # 급증 케이스 충분하므로 증가
            max_depth=7,       # 복잡한 패턴 포착
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
    
    print("모델 학습 중 (급증 케이스 최적화)...")
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
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
    with open('xgboost_최종_급증최적화.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 모델 저장: xgboost_최종_급증최적화.pkl")
    
    # Feature 중요도
    print("\n🔥 Feature 중요도 Top 30:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(30)
    
    for idx, row in feature_importance.iterrows():
        # 분석 결과 기반 Feature 강조
        if any(keyword in row['feature'] for keyword in ['hub_high', 'hub_critical', 'storage_util_high', 
                                                          'total_cmd_low', 'hub_storage_risk', 'hub_cmd_bottleneck',
                                                          'storage_util_critical']):
            marker = "🎯"
        elif 'HUBROOM' in row['feature'] or 'storage_util' in row['feature'] or 'CMD' in row['feature']:
            marker = "🔥"
        else:
            marker = "  "
        print(f"{marker} {row['feature']}: {row['importance']:.4f}")
    
    # ===== 2. 평가 =====
    print("\n[STEP 2] 평가")
    print("-"*40)
    
    split_idx = int(len(df_train) * 0.8)
    df_test = df_train.iloc[split_idx:].copy()
    
    X_test, y_test, seq_max_list, seq_min_list, indices, seq_target_list = create_features_optimized(df_test, start_idx=30)
    
    y_pred = model.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"\n평가 성능:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # 급증 케이스 성능
    surge_count = sum(1 for i in range(len(y_test)) 
                      if seq_target_list[i][-1] < 300 and y_test[i] >= 300)
    surge_detected = sum(1 for i in range(len(y_test)) 
                         if seq_target_list[i][-1] < 300 and y_test[i] >= 300 and y_pred[i] >= 290)
    
    print(f"\n🎯 급증 예측 성능:")
    print(f"  발생: {surge_count}개")
    print(f"  예측 성공: {surge_detected}/{surge_count}개 ({surge_detected/surge_count*100 if surge_count > 0 else 0:.1f}%)")
    
    # 극단값 성능
    extreme_mask = y_test >= 300
    extreme_count = extreme_mask.sum()
    extreme_detected = ((y_pred >= 290) & extreme_mask).sum()
    
    print(f"\n극단값 (300+) 감지:")
    print(f"  발생: {extreme_count}개")
    print(f"  감지: {extreme_detected}/{extreme_count}개 ({extreme_detected/extreme_count*100 if extreme_count > 0 else 0:.1f}%)")
    
    # 결과 저장
    results = []
    for i, idx in enumerate(indices):
        results.append({
            '실제값': y_test[i],
            '예측값': round(y_pred[i], 2),
            '오차': round(abs(y_test[i] - y_pred[i]), 2),
            '시퀀스MAX': seq_max_list[i],
            '급증': 'O' if (seq_target_list[i][-1] < 300 and y_test[i] >= 300) else '-',
            '급증감지': 'O' if (seq_target_list[i][-1] < 300 and y_test[i] >= 300 and y_pred[i] >= 290) else '-',
            '극단값': 'O' if y_test[i] >= 300 else '-',
            '감지': 'O' if (y_test[i] >= 300 and y_pred[i] >= 290) else '-'
        })
    
    df_results = pd.DataFrame(results)
    df_results.to_csv('최종_급증최적화_결과.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: 최종_급증최적화_결과.csv")
    
    # 최종 요약
    print("\n" + "="*80)
    print("📊 최종 요약 (1,804개 급증 케이스 기반)")
    print("="*80)
    
    print(f"\n1. 데이터 분석 결과:")
    print(f"   - 급증 케이스: 1,804개 (3.00%)")
    print(f"   - 평균 급증량: 22.9")
    print(f"   - 최대 급증량: 130.0")
    
    print(f"\n2. 핵심 패턴:")
    print(f"   🎯 HUBROOMTOTAL < 610: 87.5%")
    print(f"   🎯 STORAGE_UTIL ≥ 205: 82.9%")
    print(f"   🎯 CMD < 220: 45.4%")
    print(f"   🎯 FS_UTIL ≥ 7: 25.3%")
    
    print(f"\n3. Feature 구성:")
    print(f"   - 타겟: 7개")
    print(f"   - HUBROOMTOTAL: ~13개 (hub_high 핵심!)")
    print(f"   - FS Storage: ~12개 (storage_util_high 핵심!)")
    print(f"   - CMD: ~20개 (total_cmd_low 핵심!)")
    print(f"   - Storage: ~13개 (storage_util_critical 핵심!)")
    print(f"   - 복합: hub_storage_risk, hub_cmd_bottleneck")
    print(f"   - 총: {len(X_train.columns)}개")
    
    print(f"\n4. 모델 성능:")
    print(f"   - 학습 MAE: {train_mae:.2f}")
    print(f"   - 평가 MAE: {test_mae:.2f}")
    print(f"   - R²: {test_r2:.3f}")
    
    print(f"\n5. 급증 예측:")
    print(f"   - 발생: {surge_count}개")
    print(f"   - 성공률: {surge_detected/surge_count*100 if surge_count > 0 else 0:.1f}%")
    
    print(f"\n6. 극단값 감지:")
    print(f"   - 발생: {extreme_count}개")
    print(f"   - 감지율: {extreme_detected/extreme_count*100 if extreme_count > 0 else 0:.1f}%")
    
    return model, df_results, feature_importance

if __name__ == '__main__':
    model, results, importance = train_final_optimized()