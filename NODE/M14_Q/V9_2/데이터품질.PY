#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ 학습 데이터 정제기 (원본 컬럼 유지 + 1:1 비율)
기능: 잡다한 파생변수 생성 없이, 원본 컬럼(TOTALCNT, M14... 등) 그대로 유지하되
      데이터 비율만 50:50으로 맞춰서 저장함.
"""
import pandas as pd
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🛡️ 학습 데이터 생성 (원본 포맷 유지 + 1:1 비율)")
print("="*70)

PRED_OFFSET = 10
EXCLUDE_DATE = '2025-12-11' # 🔴 평가일 제외

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

# --- 메인 실행 ---
if __name__ == '__main__':
    # 1. 로드
    df = load_data('M14_Q_*.CSV')
    
    # 2. 날짜 필터링을 위한 임시 변수 생성
    df['datetime'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    
    # 12월 11일 제거
    mask_train = df['datetime'].dt.date.astype(str) != EXCLUDE_DATE
    df_train = df[mask_train].copy()
    
    print(f"📉 평가일({EXCLUDE_DATE}) 제외 완료.")
    
    # 3. 고부하 판단을 위한 임시 타겟 생성 (저장할 땐 지움)
    # 숫자로 변환 (에러 방지)
    df_train['TOTALCNT_NUM'] = pd.to_numeric(df_train['TOTALCNT'], errors='coerce').fillna(0)
    df_train['Target_Temp'] = df_train['TOTALCNT_NUM'].shift(-PRED_OFFSET)
    
    # NaN 제거 (Shift로 인한 끝부분)
    df_train = df_train.dropna(subset=['Target_Temp']).reset_index(drop=True)
    
    # 4. 비율 맞추기 (Downsampling)
    print("\n⚖️ 비율 1:1 맞춤 (정상 데이터 줄이기)...")
    
    df_high = df_train[df_train['Target_Temp'] >= 1700].copy()
    df_normal = df_train[df_train['Target_Temp'] < 1700].copy()
    
    cnt_high = len(df_high)
    print(f"   - 1700+ 데이터: {cnt_high:,}개 (기준)")
    
    if cnt_high > 0:
        # 정상 데이터를 고부하 개수만큼만 랜덤 추출
        df_normal_balanced = df_normal.sample(n=cnt_high, random_state=42)
        
        # 합치기
        df_final = pd.concat([df_high, df_normal_balanced], ignore_index=True)
        
        # 셔플
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 5. [중요] 임시 컬럼 삭제 (원본 컬럼만 남기기)
        # datetime, Target_Temp, TOTALCNT_NUM 등은 계산용이었으니 삭제
        original_cols = df.columns.tolist() 
        # datetime은 원본에 없었으므로 제외하고 저장
        cols_to_save = [c for c in df_final.columns if c in original_cols and c != 'datetime']
        
        df_save = df_final[cols_to_save]
        
        # 결과 확인
        print(f"\n📊 최종 데이터셋 정보")
        print(f"   - 파일명: Clean_Balanced_Data.csv")
        print(f"   - 컬럼: {list(df_save.columns)}") # 원본 컬럼인지 확인
        print(f"   - 총 데이터 수: {len(df_save):,}개")
        print(f"   - 구성: 고부하 {cnt_high}개 vs 정상 {cnt_high}개")
        
        # 저장
        df_save.to_csv('Clean_Balanced_Data.csv', index=False, encoding='utf-8-sig')
        print(f"\n✅ 저장 완료! 원본 CSV와 똑같은 모양입니다.")
        
    else:
        print("⚠ 1700 이상 데이터가 없습니다.")
