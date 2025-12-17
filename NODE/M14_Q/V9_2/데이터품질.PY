#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ 학습 데이터 전처리 및 증폭 (1:1 비율)
목표: 학습에 필요한 컬럼(Feature)만 딱 맞춰서 생성하고, 
      12월 11일은 버리고, 1700+ 데이터는 증폭해서 저장함.
"""
import numpy as np
import pandas as pd
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🛡️ 학습 데이터 생성 (컬럼 통일 + 1:1 밸런싱)")
print("="*70)

PRED_OFFSET = 10
EXCLUDE_DATE = '2025-12-11' # 🔴 평가용 날짜 (학습 절대 제외)

def load_data(path_pattern):
    print(f"\n📂 원본 데이터 로딩 중...")
    files = glob.glob(path_pattern)
    if not files: files = glob.glob('M14_Q_*.CSV')
    dfs = []
    for f in sorted(files):
        try: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='utf-8'))
        except: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='cp949'))
    if not dfs: raise ValueError("파일 없음")
    return pd.concat(dfs, ignore_index=True)

def create_training_features(df):
    """
    🔥 학습 모델(V11~V14)이 사용하는 컬럼만 정확하게 생성
    """
    df = df.copy()
    
    # 1. 기본 컬럼 (숫자 변환)
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # 파생 변수 1: Gap
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 파생 변수 2: Lag (과거 데이터) - 280분까지
    lags = [1, 3, 5, 10, 20, 50, 100, 280]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    # 파생 변수 3: Rolling (이동평균) - 280분까지
    windows = [10, 30, 60, 100, 280]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    # 파생 변수 4: 변화량 (Slope, Accel)
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)

    # 🔥 타겟 (Target) - 학습의 정답지
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    df['Target_Delta'] = df['Target_Total_Future'] - df['TOTALCNT']
    
    # 결측치 제거 (앞부분 280분 + 뒷부분 10분은 계산 불가하므로 삭제)
    return df.dropna().reset_index(drop=True)

# --- 메인 실행 ---
if __name__ == '__main__':
    # 1. 데이터 로드
    df_raw = load_data('M14_Q_*.CSV')
    
    # 2. 날짜 필터링 (12월 11일 제거)
    # 날짜 파싱
    df_raw['datetime'] = pd.to_datetime(df_raw['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    # 12월 11일이 아닌 것만 남김
    mask_train = df_raw['datetime'].dt.date.astype(str) != EXCLUDE_DATE
    df_train_src = df_raw[mask_train].copy()
    
    print(f"📉 평가일({EXCLUDE_DATE}) 데이터 삭제 완료.")
    
    # 3. 피처 생성 (학습에 필요한 컬럼만 딱!)
    print("🔄 학습용 피처(280분 시퀀스) 생성 중...")
    df_features = create_training_features(df_train_src)
    
    # 4. 1:1 비율 맞추기 (데이터 증폭)
    print("\n⚖️ 1:1 비율 밸런싱 시작...")
    
    # 타겟(미래값) 기준으로 고부하/정상 분리
    df_high = df_features[df_features['Target_Total_Future'] >= 1700].copy()
    df_normal = df_features[df_features['Target_Total_Future'] < 1700].copy()
    
    cnt_high = len(df_high)
    cnt_normal = len(df_normal)
    
    print(f"   - 정상 데이터: {cnt_normal:,}개")
    print(f"   - 1700+ 데이터: {cnt_high:,}개")
    
    if cnt_high > 0:
        # 비율 계산 (정상 개수 / 고부하 개수)
        multiplier = int(cnt_normal / cnt_high)
        print(f"   🔥 1700+ 데이터를 {multiplier}배 복사합니다!")
        
        # 증폭 수행
        df_augmented_high = pd.concat([df_high] * multiplier, ignore_index=True)
        
        # 합치기
        df_balanced = pd.concat([df_normal, df_augmented_high], ignore_index=True)
        
        # 셔플 (랜덤 섞기 - 학습 시 필수)
        df_final = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 결과 확인
        final_high_cnt = (df_final['Target_Total_Future'] >= 1700).sum()
        ratio = final_high_cnt / len(df_final) * 100
        
        print(f"\n📊 최종 학습 파일 정보")
        print(f"   - 파일명: Final_Train_Data_1to1.csv")
        print(f"   - 총 데이터: {len(df_final):,}개")
        print(f"   - 1700+ 비율: {ratio:.1f}% (약 50%)")
        
        # 5. 저장
        df_final.to_csv('Final_Train_Data_1to1.csv', index=False, encoding='utf-8-sig')
        print(f"\n✅ 저장 완료! 이제 학습 코드에서 이 파일만 로드하면 됩니다.")
        
    else:
        print("⚠ 1700 이상 데이터가 없어서 증폭할 수 없습니다.")