import pandas as pd
import numpy as np
from datetime import datetime, timedelta

"""
================================================================================
🔍 새 데이터 검증 - V6 학습코드 호환성 체크
================================================================================
사용법:
1. 이 파일을 데이터 파일과 같은 폴더에 저장
2. 아래 FILE_PATH에 실제 파일명 입력
3. 실행: python 데이터_검증.py
================================================================================
"""

# ⚙️ 여기에 실제 파일명 입력하세요!
FILE_PATH = 'HUB0905101512.csv'  # 또는 실제 파일명

print("="*80)
print("🔍 새 데이터 검증 시작")
print("="*80)

# V6 학습코드에서 요구하는 필수 컬럼
REQUIRED_COLS = {
    'target': 'CURRENT_M16A_3F_JOB_2',
    'storage': ['M16A_3F_STORAGE_UTIL'],
    'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
    'hub': ['HUBROOMTOTAL'],
    'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
    'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
    'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
    'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
}

# 1. 데이터 로드
print(f"\n📂 데이터 로드 시도: {FILE_PATH}")
try:
    df = pd.read_csv(FILE_PATH, on_bad_lines='skip', encoding='utf-8', low_memory=False)
    print(f"✅ 성공! {len(df)}개 행, {len(df.columns)}개 컬럼\n")
except:
    try:
        df = pd.read_csv(FILE_PATH, on_bad_lines='skip', encoding='cp949', low_memory=False)
        print(f"✅ 성공! (cp949 인코딩) {len(df)}개 행, {len(df.columns)}개 컬럼\n")
    except Exception as e:
        print(f"❌ 실패: {e}")
        exit()

# 2. 컬럼 존재 확인
print("="*80)
print("1️⃣ 필수 컬럼 존재 확인")
print("="*80)

all_required = [REQUIRED_COLS['target']]
for group_name, cols in REQUIRED_COLS.items():
    if group_name != 'target':
        all_required.extend(cols)

missing_cols = []
present_cols = []

for col in all_required:
    if col in df.columns:
        print(f"✅ {col}")
        present_cols.append(col)
    else:
        print(f"❌ {col} - 없음!")
        missing_cols.append(col)

print(f"\n📊 결과: {len(present_cols)}/{len(all_required)}개 존재")

if missing_cols:
    print(f"\n⚠️ 경고: {len(missing_cols)}개 컬럼 누락!")
    print("누락 컬럼:", missing_cols)
else:
    print("\n🎉 모든 필수 컬럼 존재!")

# 3. STAT_DT 확인
print("\n" + "="*80)
print("2️⃣ STAT_DT 날짜 확인")
print("="*80)

if 'STAT_DT' in df.columns:
    print(f"✅ STAT_DT 컬럼 존재")
    print(f"   샘플 값: {df['STAT_DT'].iloc[0]}")
    
    try:
        df['STAT_DT_TEST'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        print(f"✅ 날짜 변환 성공")
        print(f"   시작: {df['STAT_DT_TEST'].iloc[0]}")
        print(f"   종료: {df['STAT_DT_TEST'].iloc[-1]}")
        
        duration = (df['STAT_DT_TEST'].iloc[-1] - df['STAT_DT_TEST'].iloc[0])
        print(f"   총 기간: {duration.days}일 {duration.seconds//3600}시간")
    except Exception as e:
        print(f"⚠️ 날짜 변환 실패: {e}")
        print("   (학습 시 자동 생성되므로 문제없음)")
else:
    print(f"⚠️ STAT_DT 없음 (학습 시 자동 생성)")

# 4. 타겟 컬럼 분석
print("\n" + "="*80)
print("3️⃣ 타겟 컬럼 분석 (CURRENT_M16A_3F_JOB_2)")
print("="*80)

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'

if TARGET_COL not in df.columns:
    print(f"❌ 타겟 컬럼 없음!")
else:
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    
    print(f"데이터 타입: {df[TARGET_COL].dtype}")
    print(f"총 데이터: {len(df)}개")
    print(f"결측치: {df[TARGET_COL].isna().sum()}개 ({df[TARGET_COL].isna().sum()/len(df)*100:.2f}%)")
    print(f"\n기본 통계:")
    print(f"  최소값: {df[TARGET_COL].min():.1f}")
    print(f"  최대값: {df[TARGET_COL].max():.1f}")
    print(f"  평균: {df[TARGET_COL].mean():.1f}")
    print(f"  중간값: {df[TARGET_COL].median():.1f}")
    print(f"  표준편차: {df[TARGET_COL].std():.1f}")

# 5. 급증 케이스 분석
print("\n" + "="*80)
print("4️⃣ 급증 케이스 분석 (300 기준)")
print("="*80)

if TARGET_COL in df.columns:
    # 300 이상 케이스
    extreme_cases = (df[TARGET_COL] >= 300).sum()
    print(f"300 이상: {extreme_cases}개 ({extreme_cases/len(df)*100:.2f}%)")
    
    # 구간별 분포
    print(f"\n구간별 분포:")
    print(f"  350 이상: {(df[TARGET_COL] >= 350).sum()}개")
    print(f"  300-350: {((df[TARGET_COL] >= 300) & (df[TARGET_COL] < 350)).sum()}개")
    print(f"  290-300: {((df[TARGET_COL] >= 290) & (df[TARGET_COL] < 300)).sum()}개")
    print(f"  280-290: {((df[TARGET_COL] >= 280) & (df[TARGET_COL] < 290)).sum()}개")
    print(f"  270-280: {((df[TARGET_COL] >= 270) & (df[TARGET_COL] < 280)).sum()}개")
    
    # 급증 패턴 시뮬레이션 (30분 윈도우)
    print(f"\n🎯 급증 패턴 시뮬레이션 (30분 시퀀스 기반):")
    
    surge_count = 0
    surge_indices = []
    
    for i in range(30, len(df)):
        seq_max = df[TARGET_COL].iloc[i-30:i].max()
        current_val = df[TARGET_COL].iloc[i]
        
        # 시퀀스 최대값 < 300 이지만 현재값 >= 300 (급증!)
        if seq_max < 300 and current_val >= 300:
            surge_count += 1
            surge_indices.append(i)
    
    print(f"   발견된 급증: {surge_count}개")
    print(f"   비율: {surge_count/(len(df)-30)*100:.3f}%")
    
    if surge_count > 0:
        print(f"\n   처음 5개 급증 케이스:")
        for idx in surge_indices[:5]:
            seq_max = df[TARGET_COL].iloc[idx-30:idx].max()
            current = df[TARGET_COL].iloc[idx]
            print(f"   - 인덱스 {idx}: 시퀀스MAX={seq_max:.1f} → 현재={current:.1f}")

# 6. 주요 Feature 컬럼 통계
print("\n" + "="*80)
print("5️⃣ 주요 Feature 컬럼 통계")
print("="*80)

key_cols = {
    'HUBROOMTOTAL': '허브룸 총량',
    'CD_M163FSTORAGEUTIL': 'FS 스토리지 사용률',
    'M16A_3F_CMD': '3F CMD',
    'M16A_3F_STORAGE_UTIL': '3F 스토리지 사용률'
}

for col, desc in key_cols.items():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"\n{desc} ({col}):")
        print(f"  범위: {df[col].min():.1f} ~ {df[col].max():.1f}")
        print(f"  평균: {df[col].mean():.1f}")
        print(f"  표준편차: {df[col].std():.1f}")
        print(f"  결측치: {df[col].isna().sum()}개")
    else:
        print(f"\n{desc} ({col}): ❌ 없음")

# 7. 데이터 품질 체크
print("\n" + "="*80)
print("6️⃣ 데이터 품질 체크")
print("="*80)

quality_issues = []

# 데이터 양
if len(df) < 1000:
    quality_issues.append(f"데이터 부족: {len(df)}개 (최소 1000개 권장)")
    print(f"⚠️ 데이터 양 부족: {len(df)}개")
else:
    print(f"✅ 데이터 양 충분: {len(df)}개")

# 타겟 결측치
if TARGET_COL in df.columns:
    missing_rate = df[TARGET_COL].isna().sum() / len(df) * 100
    if missing_rate > 10:
        quality_issues.append(f"타겟 결측치 과다: {missing_rate:.1f}%")
        print(f"⚠️ 타겟 결측치 많음: {missing_rate:.1f}%")
    else:
        print(f"✅ 타겟 결측치 양호: {missing_rate:.2f}%")

# 급증 케이스
if surge_count < 10:
    quality_issues.append(f"급증 케이스 부족: {surge_count}개")
    print(f"⚠️ 급증 케이스 부족: {surge_count}개 (최소 10개 권장)")
elif surge_count < 50:
    print(f"⚠️ 급증 케이스 적음: {surge_count}개 (50개 이상 권장)")
else:
    print(f"✅ 급증 케이스 충분: {surge_count}개")

# 8. 최종 판정
print("\n" + "="*80)
print("✨ 최종 판정")
print("="*80)

if missing_cols:
    print("❌ 사용 불가: 필수 컬럼 누락")
    print(f"\n누락된 컬럼 ({len(missing_cols)}개):")
    for col in missing_cols:
        print(f"  - {col}")
elif len(df) < 500:
    print("❌ 사용 불가: 데이터 양 부족")
else:
    print("🎉 V6 학습코드 사용 가능!")
    
    if quality_issues:
        print("\n⚠️ 주의사항:")
        for issue in quality_issues:
            print(f"  - {issue}")
    
    print("\n" + "="*80)
    print("🚀 다음 단계")
    print("="*80)
    print("\n1. V6_학습코드.PY 파일 열기")
    print("\n2. 파일명 변경:")
    print(f"   df_train = pd.read_csv('{FILE_PATH}', ...)")
    print("\n3. 실행:")
    print("   python V6_학습코드.py")
    print("\n4. 학습 완료 후 평가:")
    print("   python V6_평가.py")

# 9. 추가 발견 컬럼
print("\n" + "="*80)
print("🆕 V7 업그레이드용 추가 컬럼 발견")
print("="*80)

v7_candidates = [
    'M16HUB.QUE.OHT.CURRENTOHTQCNT',
    'M16HUB.QUE.OHT.OHTUTIL',
    'M16HUB.QUE.ALL.CURRENTQCNT',
]

found_new = []
for col in v7_candidates:
    if col in df.columns:
        found_new.append(col)
        print(f"✅ {col}")

if found_new:
    print(f"\n💡 V7 모델에 추가 고려 가능: {len(found_new)}개")
else:
    print("\n현재 데이터로는 V6 Feature만 사용 가능")

print("\n" + "="*80)
print("검증 완료!")
print("="*80)