import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_v8_3stage():
    """
    🎯 V8 3단계 평가 (14개 컬럼만 사용)
    
    1단계: 분류 모델로 급등/급락/유지 판단
    2단계: 3개 전용 모델로 각각 예측 (전부 표시!)
    3단계: 분류 결과에 따라 최종 예측 선택
    """
    print("="*80)
    print("🎯 V8 3단계 평가: 분류 → 전용 수치모델 (14개 컬럼)")
    print("="*80)
    
    # ========== V8 기존 14개만 사용 ==========
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
    
    def create_class_label(change):
        if change >= 50:
            return 2  # 급등
        elif change <= -50:
            return 0  # 급락
        else:
            return 1  # 유지
    
    class_names = {0: '급락', 1: '유지', 2: '급등'}
    
    # ========== 모델 로드 ==========
    try:
        with open('xgboost_3stage_V8.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        
        clf_model = model_dict['classifier']
        reg_surge = model_dict['regressor_surge']
        reg_drop = model_dict['regressor_drop']
        reg_hold = model_dict['regressor_hold']
        
        print("✅ 모델 로드: xgboost_3stage_V8.pkl")
        print(f"  - 분류 모델: ✅")
        print(f"  - 급등 모델: {'✅' if reg_surge else '❌'}")
        print(f"  - 급락 모델: {'✅' if reg_drop else '❌'}")
        print(f"  - 유지 모델: {'✅' if reg_hold else '❌'}")
    except Exception as e:
        print(f"❌ 모델 없음: {e}")
        return None
    
    # ========== 데이터 로드 ==========
    try:
        df = pd.read_csv('test_data.csv', on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df = pd.read_csv('test_data.csv', on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df = pd.read_csv('test_data.csv', on_bad_lines='skip', encoding='euc-kr', low_memory=False)
    
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
    
    print(f"✅ 데이터 로드: {len(df)}개 행")
    
    available_cols = set(df.columns)
    
    # 시간 처리
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime(2024, 1, 1, 0, 0)
            df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    
    results = []
    all_true_class = []
    all_pred_class = []
    
    print("\n평가 시작...")
    
    total_samples = len(df) - 40
    x_range = np.arange(30)
    
    for idx, i in enumerate(range(30, len(df) - 10)):
        if idx % 500 == 0:
            print(f"  진행: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
        
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
        
        current_time = seq_data['STAT_DT'].iloc[-1]
        current_value = seq_target[-1]
        prediction_time = current_time + timedelta(minutes=10)
        
        # 실제 10분 후
        actual_value = df[TARGET_COL].iloc[i+9]
        actual_change = actual_value - current_value
        actual_class = create_class_label(actual_change)
        
        # 추세
        target_slope = np.polyfit(x_range, seq_target, 1)[0]
        
        # ========== Feature 생성 ==========
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': seq_target[-1],
            'target_slope': target_slope,
        }
        
        # V8 기존 컬럼
        for group_name, cols in FEATURE_COLS_V8.items():
            for col in cols:
                if col not in available_cols:
                    continue
                col_seq = seq_data[col].values
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
        
        # 특수 Feature
        if all(col in available_cols for col in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
            use_seq = seq_data['CD_M163FSTORAGEUSE'].values
            total_seq = seq_data['CD_M163FSTORAGETOTAL'].values
            util_seq = seq_data['CD_M163FSTORAGEUTIL'].values
            features['storage_util_high'] = 1 if util_seq[-1] >= 7 else 0
            features['storage_util_critical'] = 1 if util_seq[-1] >= 10 else 0
            features['storage_util_last'] = util_seq[-1]
            features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
            features['storage_remaining'] = total_seq[-1] - use_seq[-1]
        
        if 'HUBROOMTOTAL' in available_cols:
            hub_seq = seq_data['HUBROOMTOTAL'].values
            hub_last = hub_seq[-1]
            features['hub_critical'] = 1 if hub_last < 590 else 0
            features['hub_high'] = 1 if hub_last < 610 else 0
            features['hub_warning'] = 1 if hub_last < 620 else 0
            features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
        
        # 복합 Feature
        inflow_sum = sum(seq_data[col].iloc[-1] for col in FEATURE_COLS_V8['inflow'] if col in available_cols)
        outflow_sum = sum(seq_data[col].iloc[-1] for col in FEATURE_COLS_V8['outflow'] if col in available_cols)
        features['net_flow'] = inflow_sum - outflow_sum
        
        cmd_sum = sum(seq_data[col].iloc[-1] for col in FEATURE_COLS_V8['cmd'] if col in available_cols)
        features['total_cmd'] = cmd_sum
        features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
        features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
        
        features['surge_risk_score'] = (
            features.get('hub_high', 0) * 3 +
            features.get('storage_util_critical', 0) * 2 +
            features.get('total_cmd_low', 0) * 1 +
            features.get('storage_util_high', 0) * 1
        )
        
        X_pred = pd.DataFrame([features])
        
        # ========== 1단계: 분류 ==========
        pred_class = clf_model.predict(X_pred)[0]
        pred_proba = clf_model.predict_proba(X_pred)[0]
        
        all_true_class.append(actual_class)
        all_pred_class.append(pred_class)
        
        # ========== 2단계: 3개 모델 전부 예측 ==========
        if reg_surge is not None:
            surge_change = reg_surge.predict(X_pred)[0]
        else:
            surge_change = 50
        
        if reg_drop is not None:
            drop_change = reg_drop.predict(X_pred)[0]
        else:
            drop_change = -50
        
        if reg_hold is not None:
            hold_change = reg_hold.predict(X_pred)[0]
        else:
            hold_change = 0
        
        surge_pred = current_value + surge_change
        drop_pred = current_value + drop_change
        hold_pred = current_value + hold_change
        
        # ========== 3단계: 분류 결과에 따라 최종 선택 ==========
        if pred_class == 2:
            final_change = surge_change
            final_pred = surge_pred
            used_model = "급등모델"
        elif pred_class == 0:
            final_change = drop_change
            final_pred = drop_pred
            used_model = "급락모델"
        else:
            final_change = hold_change
            final_pred = hold_pred
            used_model = "유지모델"
        
        final_error = abs(actual_value - final_pred)
        class_correct = (pred_class == actual_class)
        
        is_surge = actual_change >= 50
        is_drop = actual_change <= -50
        
        hub_value = 0
        if 'HUBROOMTOTAL' in available_cols:
            hub_value = seq_data['HUBROOMTOTAL'].iloc[-1]
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            '실제값': round(actual_value, 2),
            '실제_변화량': round(actual_change, 2),
            '실제_클래스': class_names[actual_class],
            '예측_클래스': class_names[pred_class],
            '분류정확': '✅' if class_correct else '❌',
            '급등예측': round(surge_pred, 2),
            '급락예측': round(drop_pred, 2),
            '유지예측': round(hold_pred, 2),
            '급등변화량': round(surge_change, 2),
            '급락변화량': round(drop_change, 2),
            '유지변화량': round(hold_change, 2),
            '최종_예측': round(final_pred, 2),
            '최종_변화량': round(final_change, 2),
            '사용모델': used_model,
            '최종_오차': round(final_error, 2),
            '오차≤10': '✅' if final_error <= 10 else '',
            '오차≤20': '✅' if final_error <= 20 else '',
            '오차≤30': '✅' if final_error <= 30 else '',
            '급등여부': '📈' if is_surge else '',
            '급락여부': '📉' if is_drop else '',
            '급락확률': round(pred_proba[0] * 100, 1),
            '유지확률': round(pred_proba[1] * 100, 1),
            '급등확률': round(pred_proba[2] * 100, 1),
            '추세': round(target_slope, 4),
            'HUBROOMTOTAL': round(hub_value, 0),
        })
    
    results_df = pd.DataFrame(results)
    
    output_file = '3단계_V8_평가결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
    
    # ========== 통계 ==========
    print("\n" + "="*80)
    print("📊 평가 통계 (V8 3단계)")
    print("="*80)
    
    print(f"총 예측: {len(results_df)}개")
    
    clf_accuracy = accuracy_score(all_true_class, all_pred_class)
    print(f"\n🎯 분류 정확도: {clf_accuracy:.4f} ({clf_accuracy*100:.1f}%)")
    
    print(f"\n📊 클래스별 분류:")
    for cls in [0, 1, 2]:
        mask = np.array(all_true_class) == cls
        if mask.sum() > 0:
            acc = (np.array(all_pred_class)[mask] == cls).mean()
            print(f"  {class_names[cls]}: {acc:.1%} ({mask.sum()}개)")
    
    acc_10 = (results_df['오차≤10'] == '✅').sum()
    acc_20 = (results_df['오차≤20'] == '✅').sum()
    acc_30 = (results_df['오차≤30'] == '✅').sum()
    avg_error = results_df['최종_오차'].mean()
    
    print(f"\n🎯 최종_예측 정확도:")
    print(f"  오차 ≤ 10: {acc_10}/{len(results_df)}개 ({acc_10/len(results_df)*100:.1f}%)")
    print(f"  오차 ≤ 20: {acc_20}/{len(results_df)}개 ({acc_20/len(results_df)*100:.1f}%)")
    print(f"  오차 ≤ 30: {acc_30}/{len(results_df)}개 ({acc_30/len(results_df)*100:.1f}%)")
    print(f"  평균 오차: {avg_error:.2f}")
    
    print(f"\n📊 모델별 오차:")
    for model in ['급등모델', '급락모델', '유지모델']:
        subset = results_df[results_df['사용모델'] == model]
        if len(subset) > 0:
            avg_err = subset['최종_오차'].mean()
            print(f"  {model}: 평균오차 {avg_err:.1f} ({len(subset)}개)")
    
    print(f"\n📈📉 급등/급락 성능:")
    
    surge_df = results_df[results_df['급등여부'] == '📈']
    drop_df = results_df[results_df['급락여부'] == '📉']
    
    if len(surge_df) > 0:
        surge_error = surge_df['최종_오차'].mean()
        surge_detected = (surge_df['예측_클래스'] == '급등').sum()
        print(f"  📈 급등 ({len(surge_df)}개):")
        print(f"     평균오차: {surge_error:.1f}")
        print(f"     감지 성공: {surge_detected}/{len(surge_df)}개 ({surge_detected/len(surge_df)*100:.1f}%)")
    
    if len(drop_df) > 0:
        drop_error = drop_df['최종_오차'].mean()
        drop_detected = (drop_df['예측_클래스'] == '급락').sum()
        print(f"  📉 급락 ({len(drop_df)}개):")
        print(f"     평균오차: {drop_error:.1f}")
        print(f"     감지 성공: {drop_detected}/{len(drop_df)}개 ({drop_detected/len(drop_df)*100:.1f}%)")
    
    print(f"\n📍 400대 케이스 분석:")
    high_value = results_df[(results_df['현재값'] >= 400) & (results_df['현재값'] <= 500)]
    if len(high_value) > 0:
        print(f"  케이스: {len(high_value)}개")
        print(f"  평균 오차: {high_value['최종_오차'].mean():.1f}")
    else:
        print(f"  케이스 없음")
    
    return results_df

if __name__ == '__main__':
    print("🚀 V8 3단계 평가 시작...\n")
    results = evaluate_v8_3stage()
    
    if results is not None:
        print(f"\n✅ 평가 완료!")
        print(f"📁 결과: 3단계_V8_평가결과.csv")