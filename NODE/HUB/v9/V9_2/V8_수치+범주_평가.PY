import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

def evaluate_v8_5stage():
    """
    🎯 V8 5단계 평가 (14개 컬럼)
    
    5단계: 급락/소폭하락/정체/소폭상승/급등
    """
    print("="*80)
    print("🎯 V8 5단계 평가: 급락/소폭하락/정체/소폭상승/급등")
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
        
        # 최종 선택
        final_change = preds[pred_class]
        final_pred = current_value + final_change
        final_error = abs(actual_value - final_pred)
        
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
            '최종_변화량': round(final_change, 2),
            '최종_예측': round(final_pred, 2),
            '사용모델': class_names[pred_class],
            '최종_오차': round(final_error, 2),
            '오차≤10': '✅' if final_error <= 10 else '',
            '오차≤20': '✅' if final_error <= 20 else '',
            '오차≤30': '✅' if final_error <= 30 else '',
            '급락확률': round(pred_proba[0]*100, 1),
            '소폭하락확률': round(pred_proba[1]*100, 1),
            '정체확률': round(pred_proba[2]*100, 1),
            '소폭상승확률': round(pred_proba[3]*100, 1),
            '급등확률': round(pred_proba[4]*100, 1),
            'HUBROOMTOTAL': round(hub_value, 0),
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('5단계_V8_평가결과.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: 5단계_V8_평가결과.csv")
    
    # 통계
    print("\n" + "="*80)
    print("📊 평가 통계 (V8 5단계)")
    print("="*80)
    
    print(f"총 예측: {len(results_df)}개")
    
    clf_acc = accuracy_score(all_true, all_pred)
    print(f"\n🎯 분류 정확도: {clf_acc:.1%}")
    
    print(f"\n📊 클래스별 분류:")
    for cls in range(5):
        mask = np.array(all_true) == cls
        if mask.sum() > 0:
            acc = (np.array(all_pred)[mask] == cls).mean()
            print(f"  {class_names[cls]}: {acc:.1%} ({mask.sum()}개)")
    
    acc_10 = (results_df['오차≤10'] == '✅').sum() / len(results_df) * 100
    acc_20 = (results_df['오차≤20'] == '✅').sum() / len(results_df) * 100
    acc_30 = (results_df['오차≤30'] == '✅').sum() / len(results_df) * 100
    avg_err = results_df['최종_오차'].mean()
    
    print(f"\n🎯 최종 예측 정확도:")
    print(f"  오차 ≤ 10: {acc_10:.1f}%")
    print(f"  오차 ≤ 20: {acc_20:.1f}%")
    print(f"  오차 ≤ 30: {acc_30:.1f}%")
    print(f"  평균 오차: {avg_err:.2f}")
    
    print(f"\n📊 모델별 오차:")
    for cls in range(5):
        subset = results_df[results_df['사용모델'] == class_names[cls]]
        if len(subset) > 0:
            print(f"  {class_names[cls]}: 평균오차 {subset['최종_오차'].mean():.1f} ({len(subset)}개)")
    
    return results_df

if __name__ == '__main__':
    print("🚀 V8 5단계 평가 시작...\n")
    evaluate_v8_5stage()