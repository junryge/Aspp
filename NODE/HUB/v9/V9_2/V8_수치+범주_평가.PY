import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

def evaluate_v8_2stage():
    """
    🎯 V8 2단계 분리 모델 평가
    
    [1단계] 급변 탐지 → Yes/No
    [2단계] 
        급변 Yes → slope로 급등/급락 선택
        급변 No  → 일반 3단계 모델
    """
    print("="*80)
    print("🎯 V8 2단계 분리 모델 평가")
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
    
    normal_class_names = {0: '소폭하락', 1: '정체', 2: '소폭상승'}
    
    def get_actual_class(change):
        """실제 5단계 클래스"""
        if change <= -50: return '급락'
        elif change <= -20: return '소폭하락'
        elif change <= 20: return '정체'
        elif change <= 50: return '소폭상승'
        else: return '급등'
    
    # 모델 로드
    try:
        with open('xgboost_2stage_V8.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        sudden_detector = model_dict['sudden_detector']
        surge_model = model_dict['surge_model']
        drop_model = model_dict['drop_model']
        normal_clf = model_dict['normal_classifier']
        normal_regressors = model_dict['normal_regressors']
        print("✅ 모델 로드: xgboost_2stage_V8.pkl")
    except Exception as e:
        print(f"❌ 모델 없음: {e}")
        return None
    
    # 데이터 로드
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
    
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            df['STAT_DT'] = [datetime(2024,1,1) + timedelta(minutes=i) for i in range(len(df))]
    
    results = []
    x_range = np.arange(30)
    
    # 통계용
    sudden_true_list = []
    sudden_pred_list = []
    
    print("\n평가 시작...")
    total = len(df) - 40
    
    for idx, i in enumerate(range(30, len(df) - 10)):
        if idx % 500 == 0:
            print(f"  진행: {idx}/{total} ({idx/total*100:.1f}%)")
        
        seq_data = df.iloc[i-30:i]
        seq_target = seq_data[TARGET_COL].values
        
        current_time = seq_data['STAT_DT'].iloc[-1]
        current_value = seq_target[-1]
        actual_value = df[TARGET_COL].iloc[i+9]
        actual_change = actual_value - current_value
        actual_class = get_actual_class(actual_change)
        actual_is_sudden = 1 if abs(actual_change) >= 50 else 0
        
        slope = np.polyfit(x_range, seq_target, 1)[0]
        
        # Feature 생성
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': seq_target[-1],
            'target_slope': slope,
            'target_range': np.max(seq_target) - np.min(seq_target),
            'target_momentum': seq_target[-1] - seq_target[-5] if len(seq_target) >= 5 else 0,
            'target_accel': (seq_target[-1] - seq_target[-5]) - (seq_target[-5] - seq_target[-10]) if len(seq_target) >= 10 else 0,
            'target_volatility': np.std(np.diff(seq_target)),
        }
        
        for group_name, cols in FEATURE_COLS_V8.items():
            for col in cols:
                if col not in available_cols: continue
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
        
        if 'CD_M163FSTORAGEUTIL' in available_cols:
            util_last = seq_data['CD_M163FSTORAGEUTIL'].iloc[-1]
            features['storage_util_high'] = 1 if util_last >= 7 else 0
            features['storage_util_critical'] = 1 if util_last >= 10 else 0
        
        if 'HUBROOMTOTAL' in available_cols:
            hub_last = seq_data['HUBROOMTOTAL'].iloc[-1]
            features['hub_critical'] = 1 if hub_last < 590 else 0
            features['hub_high'] = 1 if hub_last < 610 else 0
        
        inflow = sum(seq_data[c].iloc[-1] for c in FEATURE_COLS_V8['inflow'] if c in available_cols)
        outflow = sum(seq_data[c].iloc[-1] for c in FEATURE_COLS_V8['outflow'] if c in available_cols)
        features['net_flow'] = inflow - outflow
        
        X_pred = pd.DataFrame([features])
        
        # ========== [1단계] 급변 탐지 ==========
        pred_sudden = sudden_detector.predict(X_pred)[0]
        pred_sudden_proba = sudden_detector.predict_proba(X_pred)[0]
        sudden_prob = pred_sudden_proba[1] * 100  # 급변 확률
        
        sudden_true_list.append(actual_is_sudden)
        sudden_pred_list.append(pred_sudden)
        
        # ========== [2단계] 분기 ==========
        if pred_sudden == 1:  # 급변 예측
            # slope로 방향 결정
            if slope >= 0:  # 상승 추세 → 급등
                if surge_model is not None:
                    final_change = surge_model.predict(X_pred)[0]
                else:
                    final_change = 60  # 기본값
                used_model = "급등모델"
                pred_class = "급등"
            else:  # 하락 추세 → 급락
                if drop_model is not None:
                    final_change = drop_model.predict(X_pred)[0]
                else:
                    final_change = -60  # 기본값
                used_model = "급락모델"
                pred_class = "급락"
        else:  # 일반 예측
            # 일반 분류
            normal_pred = normal_clf.predict(X_pred)[0]
            normal_proba = normal_clf.predict_proba(X_pred)[0]
            
            pred_class = normal_class_names[normal_pred]
            
            # 일반 수치 예측
            if normal_regressors[normal_pred] is not None:
                final_change = normal_regressors[normal_pred].predict(X_pred)[0]
            else:
                default = {0: -35, 1: 0, 2: 35}
                final_change = default[normal_pred]
            
            used_model = f"{pred_class}모델"
        
        final_pred = current_value + final_change
        final_error = abs(actual_value - final_pred)
        
        # 분류 정확 여부
        class_correct = (pred_class == actual_class)
        
        hub_value = seq_data['HUBROOMTOTAL'].iloc[-1] if 'HUBROOMTOTAL' in available_cols else 0
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': (current_time + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            '실제값': round(actual_value, 2),
            '실제_변화량': round(actual_change, 2),
            '실제_클래스': actual_class,
            '실제_급변': '급변' if actual_is_sudden else '일반',
            # 1단계 결과
            '급변탐지': '급변' if pred_sudden else '일반',
            '급변확률': round(sudden_prob, 1),
            '급변탐지정확': '✅' if pred_sudden == actual_is_sudden else '❌',
            # 2단계 결과
            '예측_클래스': pred_class,
            '분류정확': '✅' if class_correct else '❌',
            '사용모델': used_model,
            '최종_변화량': round(final_change, 2),
            '최종_예측': round(final_pred, 2),
            '최종_오차': round(final_error, 2),
            '오차≤10': '✅' if final_error <= 10 else '',
            '오차≤20': '✅' if final_error <= 20 else '',
            '오차≤30': '✅' if final_error <= 30 else '',
            'slope': round(slope, 4),
            'HUBROOMTOTAL': round(hub_value, 0),
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('2단계_V8_평가결과.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: 2단계_V8_평가결과.csv")
    
    # ========== 통계 ==========
    print("\n" + "="*80)
    print("📊 평가 통계 (V8 2단계 분리)")
    print("="*80)
    
    print(f"총 예측: {len(results_df)}개")
    
    # 1단계: 급변 탐지 성능
    print(f"\n🔍 [1단계] 급변 탐지 성능:")
    sudden_true = np.array(sudden_true_list)
    sudden_pred = np.array(sudden_pred_list)
    
    sudden_acc = accuracy_score(sudden_true, sudden_pred)
    print(f"  전체 정확도: {sudden_acc:.1%}")
    
    # 급변 Recall (실제 급변 중 맞춘 비율)
    actual_sudden_mask = sudden_true == 1
    if actual_sudden_mask.sum() > 0:
        recall = (sudden_pred[actual_sudden_mask] == 1).mean()
        print(f"  급변 Recall: {recall:.1%} ({(sudden_pred[actual_sudden_mask] == 1).sum()}/{actual_sudden_mask.sum()}개)")
    
    # 급변 Precision (급변 예측 중 실제 급변 비율)
    pred_sudden_mask = sudden_pred == 1
    if pred_sudden_mask.sum() > 0:
        precision = (sudden_true[pred_sudden_mask] == 1).mean()
        print(f"  급변 Precision: {precision:.1%} ({(sudden_true[pred_sudden_mask] == 1).sum()}/{pred_sudden_mask.sum()}개)")
    
    # 오탐 (일반인데 급변으로 예측)
    normal_mask = sudden_true == 0
    if normal_mask.sum() > 0:
        false_alarm = (sudden_pred[normal_mask] == 1).sum()
        print(f"  오탐 (일반→급변): {false_alarm}개 ({false_alarm/normal_mask.sum()*100:.2f}%)")
    
    # 2단계: 최종 정확도
    print(f"\n🎯 [2단계] 최종 예측 정확도:")
    
    acc_10 = (results_df['오차≤10'] == '✅').sum() / len(results_df) * 100
    acc_20 = (results_df['오차≤20'] == '✅').sum() / len(results_df) * 100
    acc_30 = (results_df['오차≤30'] == '✅').sum() / len(results_df) * 100
    avg_err = results_df['최종_오차'].mean()
    
    print(f"  오차 ≤ 10: {acc_10:.1f}%")
    print(f"  오차 ≤ 20: {acc_20:.1f}%")
    print(f"  오차 ≤ 30: {acc_30:.1f}%")
    print(f"  평균 오차: {avg_err:.2f}")
    
    # 클래스별 성능
    print(f"\n📊 실제 클래스별 성능:")
    for cls in ['급락', '소폭하락', '정체', '소폭상승', '급등']:
        subset = results_df[results_df['실제_클래스'] == cls]
        if len(subset) > 0:
            avg_err = subset['최종_오차'].mean()
            correct = (subset['분류정확'] == '✅').sum()
            print(f"  {cls}: 평균오차 {avg_err:.1f}, 분류정확 {correct}/{len(subset)}개 ({correct/len(subset)*100:.1f}%)")
    
    # 급등/급락 감지율 (핵심!)
    print(f"\n🔥 급등/급락 감지 성능 (핵심!):")
    
    surge_df = results_df[results_df['실제_클래스'] == '급등']
    if len(surge_df) > 0:
        detected = (surge_df['예측_클래스'] == '급등').sum()
        print(f"  급등 감지: {detected}/{len(surge_df)}개 ({detected/len(surge_df)*100:.1f}%)")
    
    drop_df = results_df[results_df['실제_클래스'] == '급락']
    if len(drop_df) > 0:
        detected = (drop_df['예측_클래스'] == '급락').sum()
        print(f"  급락 감지: {detected}/{len(drop_df)}개 ({detected/len(drop_df)*100:.1f}%)")
    
    # 모델 사용 분포
    print(f"\n📊 모델 사용 분포:")
    for model in results_df['사용모델'].value_counts().items():
        print(f"  {model[0]}: {model[1]}개 ({model[1]/len(results_df)*100:.1f}%)")
    
    return results_df

if __name__ == '__main__':
    print("🚀 V8 2단계 분리 모델 평가 시작...\n")
    evaluate_v8_2stage()