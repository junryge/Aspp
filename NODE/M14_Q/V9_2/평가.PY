#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 2단계: 평가 전용 (eval_v12.py) - 최종 수정
기능: 현재값, 실제값, 예측값 출력 (실제 컬럼명 사용)
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
EVAL_FILE = "UF09.CSV"      # 평가 파일명
PRED_MINUTES = 10           # 10분 예지
THRESHOLD = 1700            # 위험 기준
THRESHOLD_PROB = 0.45       # 45% 이상이면 위험(O)

print("="*70)
print(f"🚀 V12 평가 시작: {EVAL_FILE}")
print("="*70)

def create_features(df):
    df = df.copy()
    if 'CURRTIME' in df.columns:
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

    # 실제 컬럼명 사용
    df['Queue_Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    df['Dist_to_Limit'] = THRESHOLD - df['TOTALCNT']
    
    for gap in [1, 3, 5, 10]:
        df[f'Vel_Total_{gap}m'] = df['TOTALCNT'].diff(gap)
        df[f'Vel_Gap_{gap}m'] = df['Queue_Gap'].diff(gap)
        df[f'Vel_M14B_{gap}m'] = df['M14AM14B'].diff(gap)
    
    df['Acc_Total'] = df['Vel_Total_3m'].diff(1)
    df['Vol_Total_10m'] = df['TOTALCNT'].rolling(10).std()
    
    df['Is_Triple_Danger'] = ((df['M14AM14B'] > 500) & (df['M14AM14BSUM'] > 600) & (df['Queue_Gap'] > 200)).astype(int)
    df['Is_Gold_Pattern'] = ((df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600)).astype(int)
    
    return df

def evaluate():
    try:
        with open('model_v12.pkl', 'rb') as f: model = pickle.load(f)
        with open('features_v12.pkl', 'rb') as f: features = pickle.load(f)
        print("✅ 모델 로드 성공")
    except FileNotFoundError:
        print("❌ 모델 파일이 없습니다. train_v12.py를 먼저 실행하세요.")
        return

    print(f"🔄 데이터 로딩 중...")
    df = pd.read_csv(EVAL_FILE, on_bad_lines='skip')
    
    try:
        df = create_features(df)
    except KeyError as e:
        print(f"❌ 컬럼 에러: {e} 가 파일에 없습니다.")
        return

    # 실제값(Future Max)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=PRED_MINUTES)
    df['Future_Max'] = df['TOTALCNT'].rolling(window=indexer).max().shift(-1)
    
    # 데이터 준비
    df_eval = df.dropna(subset=features + ['Future_Max']).reset_index(drop=True)
    print(f"📊 평가 데이터: {len(df_eval):,}개")
    
    # 예측
    y_prob = model.predict_proba(df_eval[features])[:, 1]
    y_pred = (y_prob >= THRESHOLD_PROB).astype(int)
    
    # 결과 저장
    results = pd.DataFrame()
    results['현재시간'] = df_eval['CURRTIME']
    results['예측시점'] = df_eval['CURRTIME'] + timedelta(minutes=PRED_MINUTES)
    
    # 요청하신 3대장
    results['현재값'] = df_eval['TOTALCNT']
    results['실제값'] = df_eval['Future_Max']
    results['예측값'] = np.where(y_pred == 1, 'O', '')
    
    # 보조 정보
    results['예측확률'] = (y_prob * 100).round(1).astype(str) + '%'
    results['M14B'] = df_eval['M14AM14B']
    results['Gap'] = df_eval['Queue_Gap']
    
    # 저장
    results.to_csv("eval_result.csv", index=False, encoding='utf-8-sig')
    print("\n💾 결과 저장 완료: eval_result.csv")

    # 요약
    actual_danger = (df_eval['Future_Max'] >= THRESHOLD)
    detected = (actual_danger & (y_pred == 1)).sum()
    total = actual_danger.sum()
    
    print("\n" + "="*50)
    print(f"📊 요약: 실제 위험 {total}건 중 {detected}건 감지 성공")
    if total > 0:
        print(f"✅ 감지율: {detected/total*100:.1f}%")

if __name__ == "__main__":
    evaluate()
