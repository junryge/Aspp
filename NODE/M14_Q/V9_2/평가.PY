#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V13 평가: 값 예측(Regression) 결과 출력
기능: 모든 데이터에 대해 10분 뒤 예상 최대값을 숫자로 표기
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

# ============================================
# ⚙️ 설정
# ============================================
EVAL_FILE = "UF09.CSV"    # 평가 파일
PRED_MINUTES = 10         # 10분 예지
DANGER_LEVEL = 1700       # 이 값을 넘으면 '위험' 표시 (참고용)

# 파일 컬럼명 (학습과 동일)
COL_TIME  = 'CURRTIME'
COL_TOTAL = 'TOTALCNT'
COL_M14B  = 'M14AM14B'
COL_SUM   = 'M14AM14BSUM'
COL_Q_CR  = 'M14.QUE.ALL.CURRENTQCREATED'
COL_Q_CP  = 'M14.QUE.ALL.CURRENTQCOMPLETED'

print("="*70)
print(f"🚀 V13 숫자 예측 평가: {EVAL_FILE}")
print("="*70)

def create_features(df):
    df = df.copy()
    if COL_TIME in df.columns:
        df[COL_TIME] = pd.to_datetime(df[COL_TIME].astype(str), format='%Y%m%d%H%M', errors='coerce')

    df['Queue_Gap'] = df[COL_Q_CR] - df[COL_Q_CP]
    
    for gap in [1, 3, 5, 10]:
        df[f'Vel_Total_{gap}m'] = df[COL_TOTAL].diff(gap)
        df[f'Vel_Gap_{gap}m'] = df['Queue_Gap'].diff(gap)
        df[f'Vel_M14B_{gap}m'] = df[COL_M14B].diff(gap)
    
    df['Acc_Total'] = df['Vel_Total_3m'].diff(1)
    df['Vol_Total_10m'] = df[COL_TOTAL].rolling(10).std()
    df['Triple_Score'] = (df[COL_M14B]/500) + (df[COL_SUM]/600) + (df['Queue_Gap']/200)
    
    return df

def evaluate():
    try:
        with open('model_v13_reg.pkl', 'rb') as f: model = pickle.load(f)
        with open('features_v13_reg.pkl', 'rb') as f: features = pickle.load(f)
        print("✅ 모델 로드 성공")
    except FileNotFoundError:
        print("❌ 모델 파일이 없습니다. train_v13_reg.py를 먼저 실행하세요.")
        return

    print("🔄 평가 데이터 처리 중...")
    df = pd.read_csv(EVAL_FILE, on_bad_lines='skip')
    df = create_features(df)
    
    # 정답지 (검증용 실제 미래값)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=PRED_MINUTES)
    df['Real_Future_Max'] = df[COL_TOTAL].rolling(window=indexer).max().shift(-1)
    
    # 평가 데이터셋
    df_eval = df.dropna(subset=features + ['Real_Future_Max']).reset_index(drop=True)
    
    # 🎯 예측 실행 (상승폭 예측)
    pred_delta = model.predict(df_eval[features])
    
    # 최종 예측값 = 현재값 + 예측된 상승폭
    df_eval['Pred_Future_Max'] = df_eval[COL_TOTAL] + pred_delta
    
    # 결과 정리
    results = pd.DataFrame()
    results['시간'] = df_eval[COL_TIME]
    results['현재값'] = df_eval[COL_TOTAL]
    
    # 🔥 사용자 요청: 예측값과 실제값 비교
    results['실제_MAX'] = df_eval['Real_Future_Max']      # 실제 10분내 최대값
    results['예측_MAX'] = df_eval['Pred_Future_Max'].round(1) # 모델이 예측한 값
    
    # 오차 (실제 - 예측)
    results['오차'] = (results['실제_MAX'] - results['예측_MAX']).round(1)
    
    # 1700 넘는지 여부 (O/X)
    results['실제_1700+'] = np.where(results['실제_MAX'] >= DANGER_LEVEL, 'O', '')
    results['예측_1700+'] = np.where(results['예측_MAX'] >= DANGER_LEVEL, 'O', '')

    # 주요 Feature 확인용
    results['M14B'] = df_eval[COL_M14B]
    results['Gap'] = df_eval['Queue_Gap']
    
    # 파일 저장
    results.to_csv("eval_result_v13.csv", index=False, encoding='utf-8-sig')
    print("\n💾 결과 저장 완료: eval_result_v13.csv")
    print("   (모든 행에 예측값이 기입되었습니다)")
    
    # 간단 통계
    mae = np.mean(np.abs(results['오차']))
    print("\n" + "="*50)
    print(f"📊 예측 성능 요약")
    print(f"   평균 오차(MAE): {mae:.1f}")
    print(f"   (예: 실제가 1700일 때, 예측은 {1700-mae:.0f} ~ {1700+mae:.0f} 사이)")

if __name__ == "__main__":
    evaluate()
