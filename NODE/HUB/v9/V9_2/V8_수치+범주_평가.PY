import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

def evaluate_v8_5stage_ensemble():
    """
    🎯 V8 5단계 평가 - 확률 가중 앙상블
    
    기존: 분류 결과 → 해당 모델만 사용
    변경: 5개 모델 예측을 확률로 가중 평균
    
    예) 정체 60%, 소폭상승 30%, 급등 10%
        → 정체예측×0.6 + 소폭상승예측×0.3 + 급등예측×0.1
    """
    print("="*80)
    print("🎯 V8 5단계 평가: 확률 가중 앙상블")
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
    
    class_names = {0: '급락', 1: '소폭하락', 2: '정체', 3: '소폭상승', 4: '급등'}
    
    def get_class(change):
        if change <= -50: return 0
        elif change <= -20: return 1
        elif change <= 20: return 2
        elif change <= 50: return 3
        else: return 4
    
    # 모델 로드
    try:
        with open('xgboost_5stage_V8.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        clf_model = model_dict['classifier']
        regressors = model_dict['regressors']
        print("✅ 모델 로드: xgboost_5stage_V8.pkl")
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
    all_true, all_pred = [], []
    x_range = np.arange(30)
    
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
        actual_class = get_class(actual_change)
        
        # Feature 생성
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': seq_target[-1],
            'target_slope': np.polyfit(x_range, seq_target, 1)[0],
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
        
        # 분류
        pred_class = clf_model.predict(X_pred)[0]
        pred_proba = clf_model.predict_proba(X_pred)[0]
        
        all_true.append(actual_class)
        all_pred.append(pred_class)
        
        # 5개 모델 전부 예측
        preds = {}
        for cls in range(5):
            if regressors[cls] is not None:
                preds[cls] = regressors[cls].predict(X_pred)[0]
            else:
                default = {0: -60, 1: -35, 2: 0, 3: 35, 4: 60}
                preds[cls] = default[cls]
        
        # ========== 핵심: 확률 가중 앙상블 ==========
        # 기존: final_change = preds[pred_class]
        # 변경: 5개 모델 예측을 확률로 가중 평균
        
        ensemble_change = 0
        for cls in range(5):
            ensemble_change += preds[cls] * pred_proba[cls]
        
        ensemble_pred = current_value + ensemble_change
        ensemble_error = abs(actual_value - ensemble_pred)
        
        # 기존 방식도 계산 (비교용)
        old_change = preds[pred_class]
        old_pred = current_value + old_change
        old_error = abs(actual_value - old_pred)
        
        hub_value = seq_data['HUBROOMTOTAL'].iloc[-1] if 'HUBROOMTOTAL' in available_cols else 0
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': (current_time + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            '실제값': round(actual_value, 2),
            '실제_변화량': round(actual_change, 2),
            '실제_클래스': class_names[actual_class],
            '예측_클래스': class_names[pred_class],
            '분류정확': '✅' if pred_class == actual_class else '❌',
            '급락예측': round(preds[0], 2),
            '소폭하락예측': round(preds[1], 2),
            '정체예측': round(preds[2], 2),
            '소폭상승예측': round(preds[3], 2),
            '급등예측': round(preds[4], 2),
            # 기존 방식
            '기존_변화량': round(old_change, 2),
            '기존_예측': round(old_pred, 2),
            '기존_오차': round(old_error, 2),
            # 앙상블 방식
            '앙상블_변화량': round(ensemble_change, 2),
            '앙상블_예측': round(ensemble_pred, 2),
            '앙상블_오차': round(ensemble_error, 2),
            '앙상블_오차≤10': '✅' if ensemble_error <= 10 else '',
            '앙상블_오차≤20': '✅' if ensemble_error <= 20 else '',
            '앙상블_오차≤30': '✅' if ensemble_error <= 30 else '',
            # 확률
            '급락확률': round(pred_proba[0]*100, 1),
            '소폭하락확률': round(pred_proba[1]*100, 1),
            '정체확률': round(pred_proba[2]*100, 1),
            '소폭상승확률': round(pred_proba[3]*100, 1),
            '급등확률': round(pred_proba[4]*100, 1),
            'HUBROOMTOTAL': round(hub_value, 0),
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('5단계_V8_앙상블_평가결과.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: 5단계_V8_앙상블_평가결과.csv")
    
    # 통계
    print("\n" + "="*80)
    print("📊 평가 통계 (V8 5단계 - 확률 가중 앙상블)")
    print("="*80)
    
    print(f"총 예측: {len(results_df)}개")
    
    # 기존 vs 앙상블 비교
    print(f"\n🔄 기존 vs 앙상블 비교:")
    
    old_avg = results_df['기존_오차'].mean()
    new_avg = results_df['앙상블_오차'].mean()
    print(f"\n  평균 오차:")
    print(f"    기존: {old_avg:.2f}")
    print(f"    앙상블: {new_avg:.2f} ({'개선 ✅' if new_avg < old_avg else '악화 ❌'} {abs(old_avg-new_avg):.2f})")
    
    old_10 = (results_df['기존_오차'] <= 10).sum() / len(results_df) * 100
    new_10 = (results_df['앙상블_오차≤10'] == '✅').sum() / len(results_df) * 100
    print(f"\n  오차 ≤ 10:")
    print(f"    기존: {old_10:.1f}%")
    print(f"    앙상블: {new_10:.1f}% ({'개선 ✅' if new_10 > old_10 else '악화 ❌'} {abs(new_10-old_10):.1f}%p)")
    
    old_20 = (results_df['기존_오차'] <= 20).sum() / len(results_df) * 100
    new_20 = (results_df['앙상블_오차≤20'] == '✅').sum() / len(results_df) * 100
    print(f"\n  오차 ≤ 20:")
    print(f"    기존: {old_20:.1f}%")
    print(f"    앙상블: {new_20:.1f}% ({'개선 ✅' if new_20 > old_20 else '악화 ❌'} {abs(new_20-old_20):.1f}%p)")
    
    old_30 = (results_df['기존_오차'] <= 30).sum() / len(results_df) * 100
    new_30 = (results_df['앙상블_오차≤30'] == '✅').sum() / len(results_df) * 100
    print(f"\n  오차 ≤ 30:")
    print(f"    기존: {old_30:.1f}%")
    print(f"    앙상블: {new_30:.1f}% ({'개선 ✅' if new_30 > old_30 else '악화 ❌'} {abs(new_30-old_30):.1f}%p)")
    
    # 클래스별 개선
    print(f"\n📊 실제 클래스별 앙상블 효과:")
    for cls in range(5):
        cls_name = class_names[cls]
        subset = results_df[results_df['실제_클래스'] == cls_name]
        if len(subset) > 0:
            old_err = subset['기존_오차'].mean()
            new_err = subset['앙상블_오차'].mean()
            diff = old_err - new_err
            print(f"  {cls_name}: {old_err:.1f} → {new_err:.1f} ({'↓' if diff > 0 else '↑'}{abs(diff):.1f})")
    
    return results_df

if __name__ == '__main__':
    print("🚀 V8 5단계 앙상블 평가 시작...\n")
    evaluate_v8_5stage_ensemble()