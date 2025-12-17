#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V8.3.1 평가 - 10분 예측
변화량 예측 모델 평가

출력 컬럼 (기존 형식 유지):
- 현재시간, 현재TOTALCNT, 예측시점, 실제값
- 원본예측, 보정예측, 오차, 절대오차
- 실제위험(1700+), 예측위험(1700+)
"""

import numpy as np
import pandas as pd
import pickle
from datetime import timedelta
import os
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V8.3.1 평가 - 10분 예측")
print("   변화량 예측 방식")
print("="*70)

# ========================================
# 설정
# ========================================
SEQ_LEN = 60
PRED_OFFSET = 10

# ========================================
# Feature 생성 함수 (학습과 동일)
# ========================================
def create_features_v831(row_dict):
    """핵심 Feature만 (33개)"""
    features = {}
    
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14bsum = np.array(row_dict['M14AM14BSUM'])
    seq_m10arev = np.array(row_dict['M10AM14A'])
    seq_m16m14a = np.array(row_dict['M16M14A'])
    seq_transport = np.array(row_dict['TRANSPORT'])
    seq_oht = np.array(row_dict['OHT'])
    seq_gap = np.array(row_dict['Q_CREATED']) - np.array(row_dict['Q_COMPLETED'])
    
    seq_len = len(seq_totalcnt)
    
    # 1. TOTALCNT (7개)
    features['total_current'] = seq_totalcnt[-1]
    features['total_last5'] = np.mean(seq_totalcnt[-5:])
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_trend5'] = seq_totalcnt[-1] - seq_totalcnt[-5]
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    features['total_slope'] = np.polyfit(np.arange(min(30, seq_len)), seq_totalcnt[-30:], 1)[0]
    features['total_std10'] = np.std(seq_totalcnt[-10:])
    
    # 2. M14AM14B (4개)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    features['m14b_slope'] = np.polyfit(np.arange(min(30, seq_len)), seq_m14b[-30:], 1)[0]
    
    # 3. M14AM14BSUM (4개)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    features['m14bsum_slope'] = np.polyfit(np.arange(min(30, seq_len)), seq_m14bsum[-30:], 1)[0]
    
    # 4. M16M14A (3개)
    features['m16_current'] = seq_m16m14a[-1]
    features['m16_last10'] = np.mean(seq_m16m14a[-10:])
    features['m16_trend10'] = seq_m16m14a[-1] - seq_m16m14a[-10]
    
    # 5. M10AM14A (2개)
    features['m10arev_current'] = seq_m10arev[-1]
    features['m10arev_trend10'] = seq_m10arev[-1] - seq_m10arev[-10]
    
    # 6. TRANSPORT (2개)
    features['trans_current'] = seq_transport[-1]
    features['trans_trend10'] = seq_transport[-1] - seq_transport[-10]
    
    # 7. OHT (2개)
    features['oht_current'] = seq_oht[-1]
    features['oht_last10'] = np.mean(seq_oht[-10:])
    
    # 8. Queue Gap (3개)
    features['gap_current'] = seq_gap[-1]
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_trend10'] = seq_gap[-1] - seq_gap[-10]
    
    # 9. 핵심 임계값 (4개)
    features['is_over_1550'] = 1 if seq_totalcnt[-1] >= 1550 else 0
    features['is_over_1580'] = 1 if seq_totalcnt[-1] >= 1580 else 0
    features['is_over_1600'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    features['is_over_1650'] = 1 if seq_totalcnt[-1] >= 1650 else 0
    
    # 10. 복합 지표 (2개)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['total_momentum'] = features['total_trend5'] + features['total_trend10']
    
    return features

# ========================================
# 보정 함수
# ========================================
def adjust_prediction(pred, current_total):
    """
    보정: 하한선만 적용 (단순화)
    - 10분 내 급락 제한: 현재값 - 100
    """
    floor = current_total - 100
    if pred < floor:
        pred = floor
    return pred

# ========================================
# 모델 로드
# ========================================
model = None
for mf in ['model_V8.3.1_10min.pkl', 'model_v831_10min.pkl']:
    if os.path.exists(mf):
        with open(mf, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ 모델: {mf}")
        break

if not model:
    print("❌ 모델 없음!")
    exit(1)

# ========================================
# 데이터 로드
# ========================================
DATA_FILE = '1211평가.CSV'  # 수정 필요!
if not os.path.exists(DATA_FILE):
    print(f"❌ 데이터 없음: {DATA_FILE}")
    exit(1)

df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
print(f"✅ 데이터: {len(df):,}행")

# 시간 파싱
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')

# queue_gap
df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']

# ========================================
# 평가 실행
# ========================================
results = []
total = len(df) - SEQ_LEN - PRED_OFFSET
print(f"\n🔄 평가 중... {total:,}개")

for i in range(SEQ_LEN, len(df) - PRED_OFFSET):
    row_dict = {
        'M14AM14B': df['M14AM14B'].iloc[i-SEQ_LEN:i].values,
        'M14AM14BSUM': df['M14AM14BSUM'].iloc[i-SEQ_LEN:i].values,
        'M10AM14A': df['M10AM14A'].iloc[i-SEQ_LEN:i].values,
        'TOTALCNT': df['TOTALCNT'].iloc[i-SEQ_LEN:i].values,
        'M16M14A': df['M16M14A'].iloc[i-SEQ_LEN:i].values,
        'TRANSPORT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-SEQ_LEN:i].values,
        'OHT': df['M14.QUE.OHT.OHTUTIL'].iloc[i-SEQ_LEN:i].values,
        'Q_CREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[i-SEQ_LEN:i].values,
        'Q_COMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[i-SEQ_LEN:i].values,
    }
    
    # 현재 상태
    current_time = df['CURRTIME'].iloc[i-1]
    current_total = df['TOTALCNT'].iloc[i-1]
    actual = df['TOTALCNT'].iloc[i + PRED_OFFSET - 1]
    prediction_time = current_time + timedelta(minutes=PRED_OFFSET)
    
    # 추가 정보
    m14b = row_dict['M14AM14B'][-1]
    m14bsum = row_dict['M14AM14BSUM'][-1]
    m16 = row_dict['M16M14A'][-1]
    gap = row_dict['Q_CREATED'][-1] - row_dict['Q_COMPLETED'][-1]
    transport = row_dict['TRANSPORT'][-1]
    
    # Feature 생성
    features = create_features_v831(row_dict)
    X = pd.DataFrame([features])
    
    # 예측 (변화량)
    pred_change = model.predict(X)[0]
    
    # 원본 예측 = 현재값 + 변화량
    pred_raw = current_total + pred_change
    
    # 보정 예측
    pred = adjust_prediction(pred_raw, current_total)
    
    # 결과 저장 (기존 형식!)
    results.append({
        '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
        '현재TOTALCNT': round(current_total, 2),
        '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
        '실제값': round(actual, 2),
        '원본예측': round(pred_raw, 2),
        '보정예측': round(pred, 2),
        '오차': round(actual - pred, 2),
        '절대오차': round(abs(actual - pred), 2),
        'M14AM14B': round(m14b, 2),
        'M14AM14BSUM': round(m14bsum, 2),
        'M16M14A': round(m16, 2),
        'queue_gap': round(gap, 2),
        'TRANSPORT': round(transport, 2),
        '>=1580': 'O' if current_total >= 1580 else '',
        '실제위험(1700+)': 'O' if actual >= 1700 else '',
        '예측위험(1700+)': 'O' if pred >= 1700 else ''
    })
    
    if (i - SEQ_LEN) % 500 == 0:
        print(f"  {i-SEQ_LEN:,}/{total:,}")
        gc.collect()

print(f"✅ 완료!")

# ========================================
# 결과 저장
# ========================================
df_result = pd.DataFrame(results)
output = 'evaluation_V8.3.1_10min.csv'
df_result.to_csv(output, index=False, encoding='utf-8-sig')
print(f"✅ 저장: {output}")

# ========================================
# 통계
# ========================================
print("\n" + "="*70)
print("📊 V8.3.1 10분 평가 통계")
print("="*70)
print(f"총 예측: {len(df_result):,}개")
print(f"MAE: {df_result['절대오차'].mean():.2f}")

actual_danger = df_result['실제위험(1700+)'] == 'O'
pred_danger = df_result['예측위험(1700+)'] == 'O'
actual_count = actual_danger.sum()
detected = (actual_danger & pred_danger).sum()

print(f"\n🔥 1700+ 감지:")
print(f"  실제: {actual_count}개")
if actual_count > 0:
    print(f"  감지: {detected}개 ({detected/actual_count*100:.1f}%)")
    
    # 오탐
    fp = (pred_danger & ~actual_danger).sum()
    print(f"  오탐(FP): {fp}개")
    
    # 정밀도
    if pred_danger.sum() > 0:
        precision = detected / pred_danger.sum() * 100
        print(f"  정밀도: {precision:.1f}%")
    
    # 원본 vs 보정 비교
    raw_detected = (df_result[actual_danger]['원본예측'] >= 1700).sum()
    print(f"\n📈 보정 효과:")
    print(f"  원본 감지율: {raw_detected/actual_count*100:.1f}%")
    print(f"  보정 감지율: {detected/actual_count*100:.1f}%")

# 단순 임계값 비교
simple_pred = df_result['>=1580'] == 'O'
simple_tp = (actual_danger & simple_pred).sum()
print(f"\n📌 단순 임계값(>=1580) 비교:")
print(f"  감지율: {simple_tp/actual_count*100:.1f}%" if actual_count > 0 else "  N/A")

print(f"\n✅ V8.3.1 평가 완료! → {output}")