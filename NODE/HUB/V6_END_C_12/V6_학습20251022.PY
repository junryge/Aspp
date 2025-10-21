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

def calculate_surge_risk(features):
    """
    급증 위험도 및 예상 급증량 계산
    
    Returns:
        risk_level: 위험도 (HIGH/MEDIUM/LOW)
        surge_prob: 급증 확률 (0~1)
        expected_surge: 예상 급증량
    """
    # 주요 지표 추출
    cmd_last = features.get('M16A_3F_CMD_last_value', 999)
    cmd_mean = features.get('M16A_3F_CMD_mean', 999)
    storage_last = features.get('M16A_3F_STORAGE_UTIL_last_value', 0)
    storage_mean = features.get('M16A_3F_STORAGE_UTIL_mean', 0)
    
    # CMD 변화량 계산
    cmd_slope = features.get('M16A_3F_CMD_slope', 0)
    
    # 위험도 판정 (분석 결과 기반)
    if cmd_last < 180:
        # CMD가 매우 낮음 → 큰 급증 가능성
        risk_level = "HIGH"
        surge_prob = 0.70
        expected_surge = 60
    elif storage_last >= 30 and storage_last <= 40:
        # Storage Util 위험 구간
        risk_level = "HIGH"
        surge_prob = 0.45
        expected_surge = 50
    elif cmd_last >= 260 and cmd_last <= 280:
        # CMD 높은 구간에서도 급증 가능
        risk_level = "HIGH"
        surge_prob = 0.67
        expected_surge = 60
    elif cmd_last < 220:
        # CMD 중간 구간
        risk_level = "MEDIUM"
        surge_prob = 0.30
        expected_surge = 35
    else:
        # 일반 상황
        risk_level = "LOW"
        surge_prob = 0.05
        expected_surge = 20
    
    # 추가 보정 요소
    # 1. CMD 급감 패턴 (slope가 음수이고 큼)
    if cmd_slope < -5:
        surge_prob *= 1.2  # 20% 증가
        expected_surge *= 1.1
    
    # 2. Storage Util이 증가 추세
    storage_slope = features.get('M16A_3F_STORAGE_UTIL_slope', 0)
    if storage_slope > 2:
        surge_prob *= 1.1  # 10% 증가
    
    # 확률 상한 조정
    surge_prob = min(surge_prob, 0.85)
    
    return risk_level, surge_prob, expected_surge

def apply_adaptive_correction(prediction, features, seq_target):
    """
    적응형 예측값 보정
    
    Args:
        prediction: 원래 모델 예측값
        features: 특성 딕셔너리
        seq_target: 30개 시퀀스 타겟값
    
    Returns:
        보정된 예측값, 위험도, 급증 확률, 보정량
    """
    # 위험도 계산
    risk_level, surge_prob, expected_surge = calculate_surge_risk(features)
    
    # 현재 값 (시퀀스 마지막)
    current_value = seq_target[-1]
    
    # 시퀀스 통계
    seq_mean = np.mean(seq_target)
    seq_std = np.std(seq_target)
    seq_max = np.max(seq_target)
    recent_mean = np.mean(seq_target[-5:])
    
    # 보정 전략
    correction = 0
    
    # 1. 기본 보정: 위험도에 따른 급증 반영
    if current_value < 300:  # 현재 300 미만일 때만
        if risk_level == "HIGH":
            # 높은 위험도: 급증 가능성 적극 반영
            if prediction < 300:
                # 예측값이 300 미만이면 급증 가능성 반영
                potential_value = current_value + expected_surge * surge_prob
                
                # 300 도달 가능성 체크
                if potential_value >= 300:
                    # 300 넘을 가능성이 높으면 보정
                    correction = (potential_value - prediction) * 0.7  # 70% 반영
                else:
                    # 300 미달이어도 상향 보정
                    correction = expected_surge * surge_prob * 0.5
                    
        elif risk_level == "MEDIUM":
            # 중간 위험도: 보수적 보정
            if prediction < 300 and surge_prob > 0.25:
                correction = expected_surge * surge_prob * 0.3
                
        else:  # LOW
            # 낮은 위험도: 최소 보정
            correction = expected_surge * surge_prob * 0.1
    
    # 2. 추가 보정: 시퀀스 패턴 반영
    # 최근 급증 추세
    recent_increase = recent_mean - seq_mean
    if recent_increase > 10:  # 최근 급증 중
        correction += recent_increase * 0.2
    
    # 시퀀스 내 300 이상 있었던 경우
    if seq_max >= 300:
        correction += 5  # 추가 상향
    
    # 3. 보정값 제한
    corrected_prediction = prediction + correction
    
    # 상한/하한 제한
    corrected_prediction = max(corrected_prediction, current_value - 50)  # 급락 제한
    corrected_prediction = min(corrected_prediction, current_value + 100)  # 급등 제한
    
    # 현실적 범위 제한
    if current_value < 250 and corrected_prediction > 350:
        corrected_prediction = 320  # 과도한 급증 제한
    
    return corrected_prediction, risk_level, surge_prob, correction

def train_and_evaluate_complete():
    """
    학습(aas.csv) → 평가(BBB.CSV) 전체 프로세스 + 급증 위험도 기반 보정
    V6 Ultimate + 보정 로직 통합
    """
    print("="*80)
    print("XGBoost 30분→10분 예측 모델 학습 및 평가 + 급증 위험도 보정")
    print("="*80)
    
    # 핵심 12개 컬럼만 사용 (물리적으로 중요한 것만)
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],  # 극단값 핵심 지표
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],  # 주요 CMD
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],  # 주요 유입
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],  # 주요 유출
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']  # 용량 제한
    }
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # Feature 생성 함수
    def create_features(df, start_idx=30):
        features_list = []
        labels = []
        seq_max_list = []
        seq_min_list = []
        indices = []
        seq_target_list = []
        
        for i in range(start_idx, len(df)):
            seq_target = df[TARGET_COL].iloc[i-30:i].values
            
            features = {
                # 타겟 컬럼 특성
                'target_mean': np.mean(seq_target),
                'target_std': np.std(seq_target),
                'target_last_5_mean': np.mean(seq_target[-5:]),
                'target_max': np.max(seq_target),
                'target_min': np.min(seq_target),
                'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
                'target_last_10_mean': np.mean(seq_target[-10:]),
                'target_first_10_mean': np.mean(seq_target[:10]),
            }
            
            # 각 컬럼 그룹별 특성 추가
            for group_name, cols in FEATURE_COLS.items():
                for col in cols:
                    if col in df.columns:
                        seq_data = df[col].iloc[i-30:i].values
                        
                        # 기본 통계
                        features[f'{col}_mean'] = np.mean(seq_data)
                        features[f'{col}_std'] = np.std(seq_data)
                        features[f'{col}_max'] = np.max(seq_data)
                        features[f'{col}_min'] = np.min(seq_data)
                        
                        # 최근 특성
                        features[f'{col}_last_5_mean'] = np.mean(seq_data[-5:])
                        features[f'{col}_last_10_mean'] = np.mean(seq_data[-10:])
                        
                        # 추세
                        features[f'{col}_slope'] = np.polyfit(np.arange(30), seq_data, 1)[0]
                        
                        # 구간별 평균 (초기/중간/최근)
                        features[f'{col}_first_10_mean'] = np.mean(seq_data[:10])
                        features[f'{col}_mid_10_mean'] = np.mean(seq_data[10:20])
                        features[f'{col}_last_value'] = seq_data[-1]
            
            # 유입-유출 차이 (Net Flow)
            inflow_sum = 0
            outflow_sum = 0
            for col in FEATURE_COLS['inflow']:
                if col in df.columns:
                    inflow_sum += df[col].iloc[i-1]
            for col in FEATURE_COLS['outflow']:
                if col in df.columns:
                    outflow_sum += df[col].iloc[i-1]
            features['net_flow'] = inflow_sum - outflow_sum
            
            # CMD 총합
            cmd_sum = 0
            for col in FEATURE_COLS['cmd']:
                if col in df.columns:
                    cmd_sum += df[col].iloc[i-1]
            features['total_cmd'] = cmd_sum
            
            features_list.append(features)
            labels.append(df[TARGET_COL].iloc[i])
            # NaN/Inf 체크
            if np.isnan(labels[-1]) or np.isinf(labels[-1]):
                features_list.pop()
                continue
            seq_max_list.append(np.max(seq_target))
            seq_min_list.append(np.min(seq_target))
            indices.append(i)
            seq_target_list.append(seq_target)
        
        return pd.DataFrame(features_list), np.array(labels), seq_max_list, seq_min_list, indices, seq_target_list
    
    # ===== 1. 학습 단계 (aas.csv) =====
    print("\n[STEP 1] 학습 데이터로 모델 학습")
    print("-"*40)
    
    # 인코딩 자동 감지 및 로드
    try:
        df_train = pd.read_csv('81.csv', on_bad_lines='skip', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df_train = pd.read_csv('81.csv', on_bad_lines='skip', encoding='cp949')
            print("⚠️ 인코딩: CP949로 파일 로드")
        except:
            df_train = pd.read_csv('81.csv', on_bad_lines='skip', encoding='euc-kr')
            print("⚠️ 인코딩: EUC-KR로 파일 로드")
    
    print(f"학습 데이터: {len(df_train)}개 행")
    print(f"사용 가능한 컬럼 확인:")
    
    all_feature_cols = []
    for group_name, cols in FEATURE_COLS.items():
        available = [col for col in cols if col in df_train.columns]
        all_feature_cols.extend(available)
        print(f"  - {group_name}: {len(available)}/{len(cols)}개")
    
    # 학습 데이터 생성
    X_train, y_train, _, _, _, _ = create_features(df_train)
    
    print(f"\n생성된 Feature 수: {len(X_train.columns)}개")
    print(f"학습 샘플 수: {len(X_train)}개")
    
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # GPU/CPU 자동 선택 (실제 학습으로 테스트)
    print("\n🔍 학습 환경 감지 중...")
    use_gpu = False
    
    # 1차 시도: GPU
    try:
        print("  → GPU 모드 시도 중...")
        test_model = xgb.XGBRegressor(
            n_estimators=5,
            max_depth=3,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0
        )
        # 실제로 학습 시도 (작은 샘플로 테스트)
        test_model.fit(X_tr[:100], y_tr[:100], verbose=False)
        use_gpu = True
        print("  ✅ GPU 사용 가능! GPU 모드로 학습합니다.\n")
    except Exception as e:
        print(f"  ⚠️ GPU 사용 불가: {str(e)[:100]}")
        print("  → CPU 모드로 전환합니다.\n")
        use_gpu = False
    
    # 실제 모델 생성
    if use_gpu:
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor'
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',  # CPU 최적화
            n_jobs=-1
        )
    
    print("모델 학습 중...")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 학습 데이터 평가
    y_val_pred = model.predict(X_val)
    train_mae = mean_absolute_error(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n학습 데이터 성능:")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    # 모델 저장
    with open('xgboost_model_30min_10min.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 모델 저장 완료: xgboost_model_30min_10min.pkl")
    
    # ===== 2. 평가 단계 (BBB.CSV) + 보정 로직 =====
    print("\n[STEP 2] BBB.CSV로 모델 평가 + 급증 위험도 보정")
    print("-"*40)
    
    # 인코딩 자동 감지 및 로드
    try:
        df_test = pd.read_csv('91.CSV', on_bad_lines='skip', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df_test = pd.read_csv('91.CSV', on_bad_lines='skip', encoding='cp949')
            print("⚠️ 인코딩: CP949로 파일 로드")
        except:
            df_test = pd.read_csv('91.CSV', on_bad_lines='skip', encoding='euc-kr')
            print("⚠️ 인코딩: EUC-KR로 파일 로드")
    
    print(f"평가 데이터: {len(df_test)}개 행")
    
    # BBB.CSV Feature 생성
    X_test, y_test, seq_max_list, seq_min_list, indices, seq_target_list = create_features(df_test)
    
    # 기본 예측
    y_pred_original = model.predict(X_test)
    
    # 보정된 예측 생성
    y_pred_corrected = []
    risk_levels = []
    surge_probs = []
    corrections = []
    
    print("보정 로직 적용 중...")
    for i in range(len(X_test)):
        features_dict = X_test.iloc[i].to_dict()
        seq_target = seq_target_list[i]
        
        pred_corrected, risk_level, surge_prob, correction = \
            apply_adaptive_correction(y_pred_original[i], features_dict, seq_target)
        
        y_pred_corrected.append(pred_corrected)
        risk_levels.append(risk_level)
        surge_probs.append(surge_prob)
        corrections.append(correction)
    
    y_pred_corrected = np.array(y_pred_corrected)
    
    # 평가 지표 (보정 전)
    test_mae_original = mean_absolute_error(y_test, y_pred_original)
    test_rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))
    test_r2_original = r2_score(y_test, y_pred_original)
    
    # 평가 지표 (보정 후)
    test_mae_corrected = mean_absolute_error(y_test, y_pred_corrected)
    test_rmse_corrected = np.sqrt(mean_squared_error(y_test, y_pred_corrected))
    test_r2_corrected = r2_score(y_test, y_pred_corrected)
    
    print(f"\nBBB.CSV 평가 결과 (보정 전):")
    print(f"  MAE:  {test_mae_original:.4f}")
    print(f"  RMSE: {test_rmse_original:.4f}")
    print(f"  R²:   {test_r2_original:.4f}")
    
    print(f"\nBBB.CSV 평가 결과 (보정 후):")
    print(f"  MAE:  {test_mae_corrected:.4f}")
    print(f"  RMSE: {test_rmse_corrected:.4f}")
    print(f"  R²:   {test_r2_corrected:.4f}")
    print(f"  개선도: {test_mae_original - test_mae_corrected:.4f}")
    
    # ===== 3. 상세 분석 결과 생성 =====
    print("\n[STEP 3] 상세 분석 결과 생성")
    print("-"*40)
    
    # STAT_DT 처리
    if 'STAT_DT' in df_test.columns:
        try:
            df_test['STAT_DT'] = pd.to_datetime(df_test['STAT_DT'], format='%Y%m%d%H%M')
        except:
            try:
                df_test['STAT_DT'] = pd.to_datetime(df_test['STAT_DT'])
            except:
                base_date = datetime(2024, 1, 1)
                df_test['STAT_DT'] = [base_date + timedelta(minutes=i) for i in range(len(df_test))]
    else:
        base_date = datetime(2024, 1, 1)
        df_test['STAT_DT'] = [base_date + timedelta(minutes=i) for i in range(len(df_test))]
    
    # 결과 DataFrame 생성
    results = []
    jump_count = 0
    extreme_count = 0
    extreme_detected_original = 0
    extreme_detected_corrected = 0
    surge_count = 0
    surge_detected_original = 0
    surge_detected_corrected = 0
    
    for i, idx in enumerate(indices):
        current_time = df_test['STAT_DT'].iloc[idx]
        seq_start_time = df_test['STAT_DT'].iloc[idx-30]
        prediction_time = current_time + timedelta(minutes=10)
        current_value = seq_target_list[i][-1]
        
        is_jump = (seq_max_list[i] < 280) and (y_test[i] >= 300)
        is_surge = (current_value < 300) and (y_test[i] >= 300)
        
        if is_jump:
            jump_count += 1
        
        if is_surge:
            surge_count += 1
            if y_pred_original[i] >= 300:
                surge_detected_original += 1
            if y_pred_corrected[i] >= 300:
                surge_detected_corrected += 1
        
        if y_test[i] >= 300:
            extreme_count += 1
            if y_pred_original[i] >= 290:
                extreme_detected_original += 1
            if y_pred_corrected[i] >= 290:
                extreme_detected_corrected += 1
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시간(+10분)': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스완료': current_time.strftime('%Y-%m-%d %H:%M'),
            '실제값': round(y_test[i], 2),
            '예측값': round(y_pred_original[i], 2),
            '예측값_보정후': round(y_pred_corrected[i], 2),
            '위험도': risk_levels[i],
            '급증확률': f"{surge_probs[i]*100:.0f}%",
            '보정량': round(corrections[i], 2),
            '오차': round(abs(y_test[i] - y_pred_original[i]), 2),
            '오차_보정후': round(abs(y_test[i] - y_pred_corrected[i]), 2),
            '오차율(%)': round(abs(y_test[i] - y_pred_original[i]) / y_test[i] * 100, 2) if y_test[i] > 0 else 0,
            '오차율_보정후(%)': round(abs(y_test[i] - y_pred_corrected[i]) / y_test[i] * 100, 2) if y_test[i] > 0 else 0,
            '시퀀스MAX': round(seq_max_list[i], 2),
            '시퀀스MIN': round(seq_min_list[i], 2),
            '시퀀스범위': round(seq_max_list[i] - seq_min_list[i], 2),
            '점프케이스': 'O' if is_jump else '-',
            '300급증': '✅' if is_surge else '-',
            '급증예측성공_보정전': 'O' if (is_surge and y_pred_original[i] >= 300) else '-',
            '급증예측성공_보정후': '✅' if (is_surge and y_pred_corrected[i] >= 300) else '-',
            '극단값(300+)': 'O' if y_test[i] >= 300 else '-',
            '극단값감지_보정전': 'O' if (y_test[i] >= 300 and y_pred_original[i] >= 290) else '-',
            '극단값감지_보정후': 'O' if (y_test[i] >= 300 and y_pred_corrected[i] >= 290) else '-'
        })
    
    df_results = pd.DataFrame(results)
    
    # 통계 출력
    print(f"\n📊 특수 케이스 분석:")
    print(f"  - 전체 예측: {len(df_results)}개")
    print(f"\n[극단값 (300+) 분석]")
    print(f"  - 극단값 발생: {extreme_count}개 ({extreme_count/len(df_results)*100:.1f}%)")
    print(f"  - 감지(보정전): {extreme_detected_original}/{extreme_count}개 ({extreme_detected_original/extreme_count*100 if extreme_count > 0 else 0:.1f}%)")
    print(f"  - 감지(보정후): {extreme_detected_corrected}/{extreme_count}개 ({extreme_detected_corrected/extreme_count*100 if extreme_count > 0 else 0:.1f}%)")
    
    print(f"\n[300 급증 (300미만→300이상) 분석]")
    print(f"  - 급증 발생: {surge_count}개 ({surge_count/len(df_results)*100:.1f}%)")
    print(f"  - 예측성공(보정전): {surge_detected_original}/{surge_count}개 ({surge_detected_original/surge_count*100 if surge_count > 0 else 0:.1f}%)")
    print(f"  - 예측성공(보정후): {surge_detected_corrected}/{surge_count}개 ({surge_detected_corrected/surge_count*100 if surge_count > 0 else 0:.1f}%)")
    print(f"  - 개선: +{surge_detected_corrected - surge_detected_original}개")
    
    print(f"\n[점프 케이스 (시퀀스MAX<280 → 실제값≥300)]")
    print(f"  - 점프 케이스: {jump_count}개 ({jump_count/len(df_results)*100:.1f}%)")
    
    # 위험도별 통계
    print(f"\n[위험도별 분석]")
    for risk in ['HIGH', 'MEDIUM', 'LOW']:
        risk_df = df_results[df_results['위험도'] == risk]
        if len(risk_df) > 0:
            surge_in_risk = (risk_df['300급증'] == '✅').sum()
            print(f"  {risk}: {len(risk_df)}개 - 급증 발생 {surge_in_risk}개 ({surge_in_risk/len(risk_df)*100:.1f}%)")
    
    # CSV 저장
    df_results.to_csv('BBB_evaluation_results_with_correction.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 상세 결과 저장: BBB_evaluation_results_with_correction.csv")
    
    # ===== 4. 그래프 생성 =====
    print("\n[STEP 4] 평가 그래프 생성")
    print("-"*40)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 1. 예측 vs 실제 산점도 (보정 전)
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred_original, alpha=0.5, s=10, color='blue')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'[Before Correction] Actual vs Predicted\nMAE={test_mae_original:.2f}, R²={test_r2_original:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # 2. 예측 vs 실제 산점도 (보정 후)
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_pred_corrected, alpha=0.5, s=10, color='green')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted (Corrected)')
    ax2.set_title(f'[After Correction] Actual vs Predicted\nMAE={test_mae_corrected:.2f}, R²={test_r2_corrected:.3f}')
    ax2.grid(True, alpha=0.3)
    
    # 3. 시계열 비교 (보정 전)
    ax3 = axes[0, 2]
    plot_size = min(300, len(y_test))
    ax3.plot(range(plot_size), y_test[:plot_size], 'b-', label='Actual', alpha=0.7, linewidth=1)
    ax3.plot(range(plot_size), y_pred_original[:plot_size], 'r--', label='Predicted', alpha=0.7, linewidth=1)
    ax3.axhline(y=300, color='orange', linestyle='--', label='Extreme(300)', alpha=0.5)
    ax3.set_xlabel('Time Index')
    ax3.set_ylabel('Value')
    ax3.set_title('Time Series Comparison (Before Correction)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 시계열 비교 (보정 후)
    ax4 = axes[1, 0]
    ax4.plot(range(plot_size), y_test[:plot_size], 'b-', label='Actual', alpha=0.7, linewidth=1)
    ax4.plot(range(plot_size), y_pred_corrected[:plot_size], 'g--', label='Predicted (Corrected)', alpha=0.7, linewidth=1)
    ax4.axhline(y=300, color='orange', linestyle='--', label='Extreme(300)', alpha=0.5)
    ax4.set_xlabel('Time Index')
    ax4.set_ylabel('Value')
    ax4.set_title('Time Series Comparison (After Correction)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 오차 분포 비교
    ax5 = axes[1, 1]
    errors_original = y_pred_original - y_test
    errors_corrected = y_pred_corrected - y_test
    ax5.hist(errors_original, bins=50, alpha=0.5, label='Before', color='blue', edgecolor='black')
    ax5.hist(errors_corrected, bins=50, alpha=0.5, label='After', color='green', edgecolor='black')
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Prediction Error')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'Error Distribution Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 극단값 성능 (보정 후)
    ax6 = axes[1, 2]
    extreme_mask = y_test >= 300
    if extreme_mask.any():
        ax6.scatter(y_test[~extreme_mask], y_pred_corrected[~extreme_mask], 
                   alpha=0.3, s=5, label='Normal', color='blue')
        ax6.scatter(y_test[extreme_mask], y_pred_corrected[extreme_mask], 
                   alpha=0.8, s=20, label='Extreme(300+)', color='red')
        ax6.plot([200, 500], [200, 500], 'k--', lw=1)
        ax6.axhline(y=300, color='orange', linestyle='--', alpha=0.5)
        ax6.axvline(x=300, color='orange', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Actual')
        ax6.set_ylabel('Predicted (Corrected)')
        ax6.set_title(f'Extreme Value Performance (After Correction)\nDetected: {extreme_detected_corrected}/{extreme_count}')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. 점프 케이스 분석
    ax7 = axes[2, 0]
    if jump_count > 0:
        jump_mask = df_results['점프케이스'] == 'O'
        jump_actual = df_results[jump_mask]['실제값'].values
        jump_pred_corrected = df_results[jump_mask]['예측값_보정후'].values
        jump_seq_max = df_results[jump_mask]['시퀀스MAX'].values
        
        ax7.scatter(jump_seq_max, jump_actual, label='Actual Jump', s=50, alpha=0.8, color='red')
        ax7.scatter(jump_seq_max, jump_pred_corrected, label='Predicted (Corrected)', s=30, alpha=0.6, color='blue')
        
        for i in range(len(jump_seq_max)):
            ax7.plot([jump_seq_max[i], jump_seq_max[i]], 
                    [jump_pred_corrected[i], jump_actual[i]], 
                    'gray', alpha=0.3, linewidth=0.5)
        
        ax7.axhline(y=300, color='orange', linestyle='--', label='Threshold')
        ax7.axvline(x=280, color='green', linestyle='--', alpha=0.5, label='SeqMax=280')
        ax7.set_xlabel('Sequence MAX')
        ax7.set_ylabel('Value')
        ax7.set_title(f'Jump Cases Analysis (After Correction)\n(SeqMax<280 → Actual≥300: {jump_count} cases)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No Jump Cases', ha='center', va='center', fontsize=14)
        ax7.set_title('Jump Cases Analysis')
    
    # 8. 위험도별 분포
    ax8 = axes[2, 1]
    risk_counts = df_results['위험도'].value_counts()
    colors_risk = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
    bars = ax8.bar(risk_counts.index, risk_counts.values, 
                   color=[colors_risk.get(x, 'gray') for x in risk_counts.index])
    ax8.set_xlabel('Risk Level')
    ax8.set_ylabel('Count')
    ax8.set_title('Risk Level Distribution')
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    ax8.grid(True, alpha=0.3)
    
    # 9. Feature 중요도 (Top 20)
    ax9 = axes[2, 2]
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
    ax9.barh(range(len(feature_importance)), feature_importance['importance'].values[::-1], color=colors[::-1])
    ax9.set_yticks(range(len(feature_importance)))
    ax9.set_yticklabels(feature_importance['feature'].values[::-1], fontsize=7)
    ax9.set_xlabel('Importance')
    ax9.set_title('Top 20 Feature Importance')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('BBB.CSV Evaluation Results with Surge Risk Correction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('BBB_evaluation_graphs_with_correction.png', dpi=150, bbox_inches='tight')
    print("✅ 그래프 저장: BBB_evaluation_graphs_with_correction.png")
    
    # ===== 5. 최종 요약 =====
    print("\n" + "="*80)
    print("📊 최종 평가 요약 (급증 위험도 보정 포함)")
    print("="*80)
    
    print(f"\n1. 모델 성능:")
    print(f"   [학습 데이터]")
    print(f"   - MAE: {train_mae:.2f}")
    print(f"   - R²:  {train_r2:.3f}")
    
    print(f"\n   [평가 데이터 - 보정 전]")
    print(f"   - MAE:  {test_mae_original:.2f}")
    print(f"   - RMSE: {test_rmse_original:.2f}")
    print(f"   - R²:   {test_r2_original:.3f}")
    
    print(f"\n   [평가 데이터 - 보정 후]")
    print(f"   - MAE:  {test_mae_corrected:.2f} (개선: {test_mae_original - test_mae_corrected:.2f})")
    print(f"   - RMSE: {test_rmse_corrected:.2f}")
    print(f"   - R²:   {test_r2_corrected:.3f}")
    
    print(f"\n2. 특수 케이스 성능:")
    print(f"   [극단값 (300+)]")
    print(f"   - 발생: {extreme_count}개")
    print(f"   - 감지(보정전): {extreme_detected_original}개 ({extreme_detected_original/extreme_count*100 if extreme_count > 0 else 0:.1f}%)")
    print(f"   - 감지(보정후): {extreme_detected_corrected}개 ({extreme_detected_corrected/extreme_count*100 if extreme_count > 0 else 0:.1f}%)")
    
    print(f"\n   [300 급증 (300미만→300이상)]")
    print(f"   - 발생: {surge_count}개")
    print(f"   - 예측성공(보정전): {surge_detected_original}개 ({surge_detected_original/surge_count*100 if surge_count > 0 else 0:.1f}%)")
    print(f"   - 예측성공(보정후): {surge_detected_corrected}개 ({surge_detected_corrected/surge_count*100 if surge_count > 0 else 0:.1f}%)")
    print(f"   - 개선: +{surge_detected_corrected - surge_detected_original}개")
    
    print(f"\n   [점프 케이스]")
    print(f"   - 발생: {jump_count}개")
    
    print(f"\n3. Feature 정보:")
    print(f"   - 총 Feature 수: {len(X_train.columns)}개")
    print(f"   - 사용된 컬럼 그룹: inflow(3), outflow(3), cmd(2), maxcapa(2), storage(1)")
    
    print(f"\n4. Top 10 중요 Features:")
    top_features = feature_importance.head(10)
    for idx, row in top_features.iterrows():
        print(f"   - {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n5. 저장 파일:")
    print(f"   - 모델: xgboost_model_30min_10min.pkl")
    print(f"   - 결과: BBB_evaluation_results_with_correction.csv")
    print(f"   - 그래프: BBB_evaluation_graphs_with_correction.png")
    
    print(f"\n6. 보정 로직 요약:")
    print(f"   - HIGH 위험도: CMD<180, STORAGE 30~40, CMD 260~280")
    print(f"   - MEDIUM 위험도: CMD<220")
    print(f"   - LOW 위험도: 기타")
    
    return model, df_results

# 실행
if __name__ == '__main__':
    model, results = train_and_evaluate_complete()