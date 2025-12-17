#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V11.5 최종 평가 - AI + 하이브리드 강제 보정
기능:
  1. AI 모델로 1차 예측 (기존 방식)
  2. '입고량 폭발' 등 특정 조건에서 예측값을 강제로 1700 이상으로 보정
  3. 최종 결과 저장
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V11.5 평가 - 하이브리드 보정 (AI + Rule)")
print("="*70)

PRED_OFFSET = 10
EVAL_FILE = '20251211_16평가.CSV' # 🔴 평가할 파일명 확인!

# 1. 모델 및 피처 정보 로드
print("📂 모델 로딩 (model_v11_delta_focused.pkl 등)...")
try:
    # 모델 파일명이 다르면 여기서 수정하세요 (예: model_v11_280seq.pkl)
    with open('model_v11_280seq.pkl', 'rb') as f: 
        model_base = pickle.load(f)
    with open('features_v11_280seq.pkl', 'rb') as f: 
        feature_cols = pickle.load(f)
except FileNotFoundError:
    print("⚠ 모델 파일이 없어서 기본 모델(model_v10_base.pkl)을 찾습니다.")
    try:
        with open('model_v10_base.pkl', 'rb') as f: model_base = pickle.load(f)
        with open('features_v10.pkl', 'rb') as f: feature_cols = pickle.load(f)
    except:
        print("❌ 실행 실패: 모델 파일이 아예 없습니다.")
        exit()

# 2. 데이터 로드
print(f"📂 평가 데이터: {EVAL_FILE}")
try:
    df_raw = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='utf-8')
except:
    df_raw = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='cp949')

df_raw['CURRTIME_OBJ'] = pd.to_datetime(df_raw['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

# 3. 피처 생성 (학습 때와 동일하게)
def create_vectorized_features(df):
    df = df.copy()
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # Lag & Rolling (280분까지)
    lags = [1, 3, 5, 10, 20, 50, 100, 280]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    windows = [10, 30, 60, 100, 280]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    # 변화량
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)
    df['Total_Delta_10'] = df['TOTALCNT'] - df['Total_Lag_10']

    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    return df

print("🔄 피처 생성 중...")
df_feat = create_vectorized_features(df_raw)

# 유효 데이터 필터링
df_valid = df_feat.dropna(subset=feature_cols + ['Target_Total_Future']).copy()
X_eval = df_valid[feature_cols]
current_total = df_valid['TOTALCNT']
y_true = df_valid['Target_Total_Future']

# 4. AI 예측 (1차)
print("🚀 AI 예측 수행 (1차)...")
pred_delta = model_base.predict(X_eval)
pred_ai = current_total + pred_delta

# =========================================================================
# 🔥 [핵심] 하이브리드 보정 로직 (Rule Injection)
# AI가 놓친 1700 패턴을 강제로 잡아내는 구간
# =========================================================================
print("🔧 하이브리드 보정 적용 중 (Rule-Based)...")

final_preds = []
for i in range(len(pred_ai)):
    ai_val = pred_ai.iloc[i]
    cur_val = current_total.iloc[i]
    
    # 힌트 데이터 가져오기
    input_load = df_valid['M14AM14B'].iloc[i]      # 입고량
    input_sum = df_valid['M14AM14BSUM'].iloc[i]    # 입고량 합계
    gap_val = df_valid['Gap'].iloc[i]              # 대기열 차이
    
    boost = 0
    
    # -------------------------------------------------------------
    # [규칙 1] "입고량 폭발" 패턴 (Gap이 낮아도 무조건 1700 감)
    # -------------------------------------------------------------
    if input_load >= 540 and input_sum >= 610:
        # AI가 1680 밑으로 겁먹었으면 강제로 1710까지 올림
        if ai_val < 1680:
            boost = (1710 - ai_val) 
        else:
            boost = 30 # 이미 높게 불렀으면 +30 보너스

    # -------------------------------------------------------------
    # [규칙 2] "대기열 적체" 패턴 (이미 1600 넘었는데 Gap 큼)
    # -------------------------------------------------------------
    elif cur_val >= 1600 and gap_val > 200:
        if ai_val < cur_val: # AI가 떨어질거라고 하면 무시하고 현상유지+알파
            boost = (cur_val - ai_val) + 20
            
    # 최종값 적용
    final_preds.append(ai_val + boost)

# 리스트를 시리즈로 변환
pred_final = pd.Series(final_preds, index=pred_ai.index)

# 5. 결과 정리
results = pd.DataFrame()
results['현재시간'] = df_valid['CURRTIME_OBJ'].dt.strftime('%Y-%m-%d %H:%M')
results['현재TOTALCNT'] = current_total.round(2)
results['예측시점'] = (df_valid['CURRTIME_OBJ'] + timedelta(minutes=PRED_OFFSET)).dt.strftime('%Y-%m-%d %H:%M')
results['실제값'] = y_true.round(2)
results['AI예측'] = pred_ai.round(2)      # 보정 전
results['최종예측'] = pred_final.round(2) # 보정 후
results['오차'] = (results['실제값'] - results['최종예측']).round(2)
results['절대오차'] = results['오차'].abs()

# 보정 효과 확인용 컬럼
results['보정량'] = (results['최종예측'] - results['AI예측']).round(2)
results['M14AM14B'] = df_valid['M14AM14B']
results['queue_gap'] = df_valid['Gap']

# 위험 감지
results['실제위험'] = np.where(results['실제값'] >= 1700, 'O', '')
results['예측위험'] = np.where(results['최종예측'] >= 1650, 'O', '')

# 6. 통계 출력
mae = results['절대오차'].mean()
print(f"\n📊 전체 MAE: {mae:.2f}")

high_val_mask = results['실제값'] >= 1650
if high_val_mask.sum() > 0:
    mae_high_ai = (results.loc[high_val_mask, '실제값'] - results.loc[high_val_mask, 'AI예측']).abs().mean()
    mae_high_final = results.loc[high_val_mask, '절대오차'].mean()
    
    print(f"\n🔥 고부하(>=1650) 구간 개선 효과")
    print(f"   - AI 단독 MAE: {mae_high_ai:.2f}")
    print(f"   - 하이브리드 MAE: {mae_high_final:.2f} (📉 대폭 감소 예상)")
    
    # 1700 넘는 구간 샘플 출력
    print("\n[보정 적용된 1700+ 예측 샘플]")
    cols = ['현재시간', '실제값', 'AI예측', '최종예측', '보정량', 'M14AM14B']
    print(results.loc[high_val_mask, cols].head(10).to_string(index=False))

# CSV 저장
outfile = 'Hybrid_Eval_Result.CSV'
results.to_csv(outfile, index=False, encoding='utf-8-sig')
print(f"\n💾 최종 결과 저장 완료: {outfile}")