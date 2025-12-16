#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V11.0 평가 - 10분 예측 (시퀀스 280분 + 12월 패턴 보정)
시퀀스: 280분, 예측: 10분 후
특징: 11월형(급상승) + 12월형(점진상승) 이원화 패턴 감지
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V11.0 평가 - 10분 예측 (280분 + 12월 패턴)")
print("="*70)

PRED_OFFSET = 10
SEQ_LEN = 280

def create_vectorized_features(df):
    """학습과 동일한 피처 생성 로직 (280분 시퀀스 + 12월 패턴)"""
    df = df.copy()
    
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    df['Transport'] = df['M14.QUE.ALL.TRANSPORT4MINOVERCNT']
    df['OHT'] = df['M14.QUE.OHT.OHTUTIL']
    
    # 1. 시차 변수 (Lags) - 280분까지 확장
    lags = [1, 3, 5, 10, 20, 50, 100, 200, 280]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
        df[f'Sum_Lag_{lag}'] = df['M14AM14BSUM'].shift(lag)
        df[f'Gap_Lag_{lag}'] = df['Gap'].shift(lag)
    
    # 2. 이동 평균/표준편차 (Rolling Stats) - 280분까지 확장
    windows = [10, 30, 60, 100, 200, 280]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()
        df[f'Total_Max_{w}'] = df['TOTALCNT'].rolling(w).max()
        df[f'Total_Min_{w}'] = df['TOTALCNT'].rolling(w).min()
        df[f'M14B_Mean_{w}'] = df['M14AM14B'].rolling(w).mean()
        df[f'Gap_Mean_{w}'] = df['Gap'].rolling(w).mean()
    
    # 3. 변화량 (Delta)
    df['Total_Delta_1'] = df['TOTALCNT'] - df['Total_Lag_1']
    df['Total_Delta_5'] = df['TOTALCNT'] - df['Total_Lag_5']
    df['Total_Delta_10'] = df['TOTALCNT'] - df['Total_Lag_10']
    df['Total_Delta_50'] = df['TOTALCNT'] - df['Total_Lag_50']
    df['Total_Delta_100'] = df['TOTALCNT'] - df['Total_Lag_100']
    df['Total_Delta_280'] = df['TOTALCNT'] - df['Total_Lag_280']
    
    df['M14B_Delta_10'] = df['M14AM14B'] - df['M14B_Lag_10']
    df['M14B_Delta_100'] = df['M14AM14B'] - df['M14B_Lag_100']
    df['Sum_Delta_10'] = df['M14AM14BSUM'] - df['Sum_Lag_10']
    df['Sum_Delta_100'] = df['M14AM14BSUM'] - df['Sum_Lag_100']
    
    # 4. 상호작용 (Interaction)
    df['Load_Ratio'] = df['M14AM14B'] / (df['TOTALCNT'] + 1)
    df['Gap_Load'] = df['Gap'] * df['M14AM14B']
    
    # 5. 패턴 (11월형 + 12월형 통합!)
    df['Gold_Pattern'] = ((df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600)).astype(int)
    df['Danger_Gap'] = (df['Gap'] > 250).astype(int)
    df['Danger_Total'] = (df['TOTALCNT'] >= 1600).astype(int)
    
    # 11월형: Triple Check (급상승)
    df['Triple_Check'] = ((df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600) & (df['Gap'] > 250)).astype(int)
    
    # 🔥 12월형: 높은 TOTALCNT + 황금패턴 (점진상승)
    df['Pattern_Dec'] = ((df['TOTALCNT'] >= 1680) & (df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600)).astype(int)
    
    # 🔥 통합 위험 패턴 (11월 OR 12월)
    df['Danger_Pattern_Combined'] = (df['Triple_Check'] | df['Pattern_Dec']).astype(int)
    
    # 6. 가속도 (Acceleration) 및 단기 변화
    df['Accel_1'] = df['Total_Delta_1'] - df['Total_Delta_1'].shift(1)
    df['Accel_3'] = df['Total_Delta_1'].diff(3)
    
    # 기울기 (Slope)
    df['Slope_3'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(3)) / 3
    df['Slope_5'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(5)) / 5
    df['Slope_30'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(30)) / 30
    df['Slope_100'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(100)) / 100
    df['Gap_Slope_3'] = (df['Gap'] - df['Gap'].shift(3)) / 3
    df['Gap_Slope_30'] = (df['Gap'] - df['Gap'].shift(30)) / 30
    
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    df['Target_Delta'] = df['Target_Total_Future'] - df['TOTALCNT']
    return df

def adjust_prediction(pred, current_total, m14b, m14bsum, gap):
    """
    🔥 보정 로직 - 11월형 + 12월형 이원화 패턴
    """
    # 1. 하한선 (10분 = -80)
    floor = current_total - 80
    if pred < floor:
        pred = floor
    
    # 2. 1650-1699 구간 Boost
    if 1650 <= pred < 1700:
        boost = 0
        
        # 11월형: Triple Check (급상승 - gap 기반)
        pattern_nov = (m14b > 520) and (m14bsum > 600) and (gap > 250)
        
        # 🔥 12월형: 높은 TOTALCNT + 황금패턴 (점진상승)
        pattern_dec = (current_total >= 1680) and (m14b > 520) and (m14bsum > 600)
        
        # 11월형 boost
        if pattern_nov:
            boost += 50
            if gap > 300:
                boost += 20
        
        # 🔥 12월형 boost (새로 추가!)
        elif pattern_dec:
            boost += 40
            if current_total >= 1700:
                boost += 20
        
        # 일반 황금패턴
        elif (m14b > 520) and (m14bsum > 600):
            boost += 30
        
        pred = pred + boost
    
    # 3. 현재 1700+ 보정
    if current_total >= 1700 and pred < 1680:
        pred = max(pred, current_total - 50)
    
    return pred

# 1. 모델 로드
print("📂 모델 로딩...")
with open('model_v11_280min.pkl', 'rb') as f:
    model_base = pickle.load(f)
with open('features_v11_280min.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
print(f"📊 피처 수: {len(feature_cols)}개")

# 2. 데이터 로드
EVAL_FILE = '20251211_16평가.CSV'
print(f"📂 데이터 로딩: {EVAL_FILE}")
try:
    df = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='cp949')

# 시간 파싱 (CURRTIME: YYYYMMDDHHMM)
df['CURRTIME_OBJ'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

# 3. 피처 생성
print(f"🔄 피처 생성 중 (280분 시퀀스 + 12월 패턴)...")
df_feat = create_vectorized_features(df)

# 유효 데이터 필터링
df_valid = df_feat.dropna(subset=feature_cols).copy()
df_valid = df_valid.dropna(subset=['Target_Total_Future'])
print(f"✅ 유효 샘플: {len(df_valid):,}개")

X_eval = df_valid[feature_cols]
y_true = df_valid['Target_Total_Future'].values
current_time = df_valid['CURRTIME_OBJ']
current_total = df_valid['TOTALCNT'].values
m14b = df_valid['M14AM14B'].values
m14bsum = df_valid['M14AM14BSUM'].values
gap = df_valid['Gap'].values
trans = df_valid['Transport'].values

# 4. 예측
print("🚀 예측 수행...")
pred_delta = model_base.predict(X_eval)
pred_raw = current_total + pred_delta

# 🔥 보정 적용 (12월 패턴 포함)
pred_adjusted = np.array([
    adjust_prediction(pred_raw[i], current_total[i], m14b[i], m14bsum[i], gap[i])
    for i in range(len(pred_raw))
])

# 5. 결과 저장 (한글 컬럼)
results = pd.DataFrame()
results['현재시간'] = current_time.dt.strftime('%Y-%m-%d %H:%M')
results['현재TOTALCNT'] = np.round(current_total, 2)
results['예측시점'] = (current_time + timedelta(minutes=PRED_OFFSET)).dt.strftime('%Y-%m-%d %H:%M')
results['실제값'] = np.round(y_true, 2)
results['원본예측'] = np.round(pred_raw, 2)
results['보정예측'] = np.round(pred_adjusted, 2)
results['오차'] = np.round(y_true - pred_adjusted, 2)
results['절대오차'] = np.abs(results['오차'])
results['M14AM14B'] = m14b
results['M14AM14BSUM'] = m14bsum
results['queue_gap'] = gap
results['TRANSPORT'] = trans

# 패턴 감지
results['11월패턴(Triple)'] = np.where((m14b > 520) & (m14bsum > 600) & (gap > 250), 'O', '')
results['12월패턴(Total>=1680)'] = np.where((current_total >= 1680) & (m14b > 520) & (m14bsum > 600), 'O', '')

# 위험 감지 여부
results['실제위험'] = np.where(y_true >= 1700, 'O', '')
results['예측위험(원본)'] = np.where(pred_raw >= 1700, 'O', '')
results['예측위험(보정)'] = np.where(pred_adjusted >= 1700, 'O', '')

# 6. 결과 출력
print("\n" + "="*70)
print("📊 평가 결과")
print("="*70)

mae = results['절대오차'].mean()
print(f"\n전체 MAE: {mae:.2f}")

# 고부하 구간
high_val_mask = results['실제값'] >= 1650
if high_val_mask.sum() > 0:
    mae_high = results.loc[high_val_mask, '절대오차'].mean()
    print(f"고부하 구간(>=1650) MAE: {mae_high:.2f}")

# 1700+ 감지율
actual_danger = (y_true >= 1700).sum()
if actual_danger > 0:
    detected_raw = ((y_true >= 1700) & (pred_raw >= 1700)).sum()
    detected_adj = ((y_true >= 1700) & (pred_adjusted >= 1700)).sum()
    
    print(f"\n🔥 1700+ 감지:")
    print(f"  실제 1700+ 건수: {actual_danger}개")
    print(f"  원본 감지: {detected_raw}개 ({detected_raw/actual_danger*100:.1f}%)")
    print(f"  보정 감지: {detected_adj}개 ({detected_adj/actual_danger*100:.1f}%)")

# 패턴별 감지율
print(f"\n📊 패턴별 분석:")
nov_pattern = (results['11월패턴(Triple)'] == 'O')
dec_pattern = (results['12월패턴(Total>=1680)'] == 'O')
actual_1700 = (results['실제위험'] == 'O')

print(f"  11월 패턴 발동: {nov_pattern.sum()}건")
print(f"  12월 패턴 발동: {dec_pattern.sum()}건")

if actual_1700.sum() > 0:
    nov_detected = (nov_pattern & actual_1700).sum()
    dec_detected = (dec_pattern & actual_1700).sum()
    combined_detected = ((nov_pattern | dec_pattern) & actual_1700).sum()
    
    print(f"\n  1700+ 중 11월 패턴 감지: {nov_detected}건 ({nov_detected/actual_1700.sum()*100:.1f}%)")
    print(f"  1700+ 중 12월 패턴 감지: {dec_detected}건 ({dec_detected/actual_1700.sum()*100:.1f}%)")
    print(f"  1700+ 중 통합 패턴 감지: {combined_detected}건 ({combined_detected/actual_1700.sum()*100:.1f}%)")

# CSV 저장
outfile = '평가결과_V11_280분_12월패턴.CSV'
results.to_csv(outfile, index=False, encoding='utf-8-sig')
print(f"\n💾 저장 완료: {outfile}")
print(f"📊 시퀀스: {SEQ_LEN}분")
print(f"🔥 12월 패턴 보정 적용됨!")