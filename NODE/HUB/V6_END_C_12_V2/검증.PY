"""
V7 신규 컬럼 검증 코드
- 8개 신규 컬럼의 실제 예측력 검증
- V6 vs V7 성능 비교
- 컬럼별 기여도 분석
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ===================================
# V7 신규 컬럼 정의
# ===================================
NEW_COLS_V7 = [
    'M16HUB.QUE.M16TOM14B.MESCURRENTQCNT',    # 1순위 (0.59)
    'M16HUB.QUE.ALL.CURRENTQCNT',             # 2순위 (0.57)
    'M16HUB.QUE.M16TOM14.MESCURRENTQCNT',     # 3순위 (0.48)
    'M16HUB.QUE.OHT.CURRENTOHTQCNT',          # 4순위 (0.46)
    'M16HUB.QUE.M16TOM14A.MESCURRENTQCNT',    # 5순위 (0.42)
    'M14.QUE.CNV.M14ATOM16ACURRNETQCNT',      # 6순위 (0.36)
    'M16HUB.QUE.OHT.OHTUTIL',                 # 7순위 (0.34)
    'M16HUB.QUE.M14TOM16.MESCURRENTQCNT',     # 8순위 (0.32)
]

# V6 기존 컬럼
V6_FEATURE_COLS = {
    'storage': ['M16A_3F_STORAGE_UTIL'],
    'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
    'hub': ['HUBROOMTOTAL'],
    'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
    'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
    'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
    'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
}

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'


def validate_new_columns(csv_file):
    """
    V7 신규 컬럼 검증 메인 함수
    """
    print("="*80)
    print("🔍 V7 신규 컬럼 검증 시작")
    print("="*80)
    
    # 데이터 로드
    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='euc-kr', low_memory=False)
    
    print(f"\n✅ 데이터 로드: {len(df)}개 행")
    
    # 타겟 컬럼 확인
    if TARGET_COL not in df.columns:
        print(f"❌ 타겟 컬럼 '{TARGET_COL}' 없음!")
        return None
    
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
    
    print(f"✅ 타겟 컬럼 전처리 완료: {len(df)}개 행")
    
    # ========================================
    # 1단계: 신규 컬럼 존재 여부 확인
    # ========================================
    print("\n" + "="*80)
    print("📊 1단계: 신규 컬럼 존재 여부 확인")
    print("="*80)
    
    available_cols = []
    missing_cols = []
    
    for i, col in enumerate(NEW_COLS_V7, 1):
        if col in df.columns:
            available_cols.append(col)
            print(f"✅ {i}. {col}")
        else:
            missing_cols.append(col)
            print(f"❌ {i}. {col} - 없음!")
    
    print(f"\n📊 결과: {len(available_cols)}/{len(NEW_COLS_V7)}개 존재")
    
    if len(available_cols) == 0:
        print("\n❌ 신규 컬럼이 하나도 없습니다!")
        print("💡 파일에 다음 컬럼들이 있는지 확인하세요:")
        for col in NEW_COLS_V7:
            print(f"   - {col}")
        return None
    
    # ========================================
    # 2단계: 신규 컬럼 기본 통계
    # ========================================
    print("\n" + "="*80)
    print("📊 2단계: 신규 컬럼 기본 통계")
    print("="*80)
    
    stats_list = []
    
    for col in available_cols:
        data = df[col]
        
        stats = {
            '컬럼': col.split('.')[-1],  # 짧게 표시
            '평균': round(data.mean(), 1),
            '표준편차': round(data.std(), 1),
            '최소': round(data.min(), 1),
            '최대': round(data.max(), 1),
            '범위': round(data.max() - data.min(), 1),
            '변동계수(%)': round(data.std() / data.mean() * 100, 1) if data.mean() != 0 else 0,
        }
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    print(stats_df.to_string(index=False))
    
    # ========================================
    # 3단계: 타겟과의 상관관계
    # ========================================
    print("\n" + "="*80)
    print("📊 3단계: 타겟과의 상관관계")
    print("="*80)
    
    corr_list = []
    
    for col in available_cols:
        corr = df[TARGET_COL].corr(df[col])
        
        # 급증 시 vs 정상 시 평균 차이
        high = df[df[TARGET_COL] >= 300][col].mean()
        low = df[df[TARGET_COL] < 300][col].mean()
        diff_pct = abs(high - low) / low * 100 if low != 0 else 0
        
        corr_list.append({
            '컬럼': col.split('.')[-1],
            '상관계수': round(corr, 4),
            '300+ 평균': round(high, 1),
            '300- 평균': round(low, 1),
            '차이(%)': round(diff_pct, 1),
        })
    
    corr_df = pd.DataFrame(corr_list).sort_values('상관계수', ascending=False, key=abs)
    print(corr_df.to_string(index=False))
    
    # ========================================
    # 4단계: 급증 예측력 개별 테스트
    # ========================================
    print("\n" + "="*80)
    print("📊 4단계: 급증 예측력 개별 테스트 (단일 Feature)")
    print("="*80)
    
    surge_results = []
    
    # 급증 케이스 찾기
    surge_indices = []
    for i in range(30, len(df) - 10):
        seq_max = df[TARGET_COL].iloc[i-30:i].max()
        actual = df[TARGET_COL].iloc[i:i+10].max()
        
        if seq_max < 300 and actual >= 300:
            surge_indices.append(i)
    
    print(f"✅ 급증 케이스: {len(surge_indices)}개 발견")
    
    if len(surge_indices) > 0:
        for col in available_cols:
            # 각 컬럼의 마지막 값으로 간단한 임계값 테스트
            correct = 0
            
            for idx in surge_indices:
                last_value = df[col].iloc[idx-1]
                
                # 컬럼별 임계값 (임시)
                if 'M16TOM14B' in col:
                    threshold = 120
                    detected = (last_value < threshold)
                elif 'ALL.CURRENT' in col:
                    threshold = 850
                    detected = (last_value > threshold)
                elif 'M16TOM14.MES' in col:
                    threshold = 300
                    detected = (last_value < threshold)
                elif 'OHT.CURRENT' in col:
                    threshold = 230
                    detected = (last_value > threshold)
                elif 'M16TOM14A' in col:
                    threshold = 160
                    detected = (last_value < threshold)
                elif 'M14ATOM16A' in col:
                    threshold = 220
                    detected = (last_value > threshold)
                elif 'OHTUTIL' in col:
                    threshold = 85
                    detected = (last_value > threshold)
                elif 'M14TOM16' in col:
                    threshold = 200
                    detected = (last_value < threshold)
                else:
                    detected = False
                
                if detected:
                    correct += 1
            
            detection_rate = correct / len(surge_indices) * 100
            
            surge_results.append({
                '컬럼': col.split('.')[-1],
                '감지': correct,
                '전체': len(surge_indices),
                '감지율(%)': round(detection_rate, 1),
            })
        
        surge_df = pd.DataFrame(surge_results).sort_values('감지율(%)', ascending=False)
        print(surge_df.to_string(index=False))
    
    # ========================================
    # 5단계: V6 vs V7 비교 (모델 필요 없는 간단 비교)
    # ========================================
    print("\n" + "="*80)
    print("📊 5단계: 복합 신호 테스트")
    print("="*80)
    
    if len(surge_indices) > 0:
        # V6 방식 (기존 지표만)
        v6_correct = 0
        # V7 방식 (신규 지표 포함)
        v7_correct = 0
        
        for idx in surge_indices:
            # V6 신호
            v6_signals = 0
            if 'HUBROOMTOTAL' in df.columns:
                if df['HUBROOMTOTAL'].iloc[idx-1] < 610:
                    v6_signals += 1
            if 'CD_M163FSTORAGEUTIL' in df.columns:
                if df['CD_M163FSTORAGEUTIL'].iloc[idx-1] >= 7:
                    v6_signals += 1
            if 'M16A_3F_CMD' in df.columns:
                if df['M16A_3F_CMD'].iloc[idx-1] < 220:
                    v6_signals += 1
            
            if v6_signals >= 2:
                v6_correct += 1
            
            # V7 신호 (V6 + 신규)
            v7_signals = v6_signals
            
            if 'M16HUB.QUE.M16TOM14B.MESCURRENTQCNT' in df.columns:
                if df['M16HUB.QUE.M16TOM14B.MESCURRENTQCNT'].iloc[idx-1] < 120:
                    v7_signals += 2  # 가중치
            if 'M16HUB.QUE.ALL.CURRENTQCNT' in df.columns:
                if df['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[idx-1] > 850:
                    v7_signals += 2  # 가중치
            if 'M16HUB.QUE.OHT.OHTUTIL' in df.columns:
                if df['M16HUB.QUE.OHT.OHTUTIL'].iloc[idx-1] > 85:
                    v7_signals += 1
            
            if v7_signals >= 3:
                v7_correct += 1
        
        print(f"\n📊 복합 신호 기반 급증 감지:")
        print(f"   V6 방식 (기존 지표): {v6_correct}/{len(surge_indices)} = {v6_correct/len(surge_indices)*100:.1f}%")
        print(f"   V7 방식 (신규 추가): {v7_correct}/{len(surge_indices)} = {v7_correct/len(surge_indices)*100:.1f}%")
        print(f"   개선: {(v7_correct-v6_correct)/len(surge_indices)*100:+.1f}%p")
    
    # ========================================
    # 6단계: 결과 저장
    # ========================================
    print("\n" + "="*80)
    print("📊 6단계: 결과 저장")
    print("="*80)
    
    # 통계 저장
    stats_df.to_csv('V7_신규컬럼_기본통계.csv', index=False, encoding='utf-8-sig')
    print("✅ V7_신규컬럼_기본통계.csv")
    
    # 상관관계 저장
    corr_df.to_csv('V7_신규컬럼_상관관계.csv', index=False, encoding='utf-8-sig')
    print("✅ V7_신규컬럼_상관관계.csv")
    
    # 급증 예측력 저장
    if len(surge_indices) > 0:
        surge_df.to_csv('V7_신규컬럼_급증예측력.csv', index=False, encoding='utf-8-sig')
        print("✅ V7_신규컬럼_급증예측력.csv")
    
    # ========================================
    # 최종 요약
    # ========================================
    print("\n" + "="*80)
    print("🎯 최종 요약")
    print("="*80)
    
    print(f"\n1️⃣ 컬럼 존재 여부:")
    print(f"   존재: {len(available_cols)}/{len(NEW_COLS_V7)}개")
    if len(missing_cols) > 0:
        print(f"   부족: {missing_cols}")
    
    print(f"\n2️⃣ 상관관계 우수:")
    top_3 = corr_df.head(3)
    for _, row in top_3.iterrows():
        print(f"   • {row['컬럼']}: {row['상관계수']:+.4f}")
    
    print(f"\n3️⃣ 급증 예측력 우수:")
    if len(surge_indices) > 0:
        top_3_surge = surge_df.head(3)
        for _, row in top_3_surge.iterrows():
            print(f"   • {row['컬럼']}: {row['감지율(%)']}%")
    
    print(f"\n4️⃣ 권장사항:")
    if len(available_cols) == len(NEW_COLS_V7):
        print(f"   ✅ 모든 컬럼 사용 가능!")
        print(f"   ✅ V7 모델 학습 진행 가능!")
    elif len(available_cols) >= 6:
        print(f"   ⚠️ {len(available_cols)}개 컬럼만 사용 가능")
        print(f"   ⚠️ V7 모델 학습은 가능하나 성능 제한적")
    else:
        print(f"   ❌ 신규 컬럼 부족 ({len(available_cols)}개)")
        print(f"   ❌ V6 모델 사용 권장")
    
    return {
        'available_cols': available_cols,
        'missing_cols': missing_cols,
        'stats_df': stats_df,
        'corr_df': corr_df,
        'surge_df': surge_df if len(surge_indices) > 0 else None,
    }


if __name__ == '__main__':
    import sys
    
    print("\n" + "="*80)
    print("🚀 V7 신규 컬럼 검증 도구")
    print("="*80)
    
    # 파일명 입력
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print("\n📁 검증할 CSV 파일명을 입력하세요:")
        print("   예: HUB_0509_TO_0929_DATA.csv")
        csv_file = input("\n파일명: ").strip()
    
    if not csv_file:
        print("\n❌ 파일명이 없습니다!")
        print("\n사용법:")
        print("   python V7_컬럼검증.py [파일명.csv]")
        print("   또는")
        print("   python V7_컬럼검증.py")
        print("   (파일명 입력)")
    else:
        result = validate_new_columns(csv_file)
        
        if result:
            print("\n" + "="*80)
            print("✅ 검증 완료!")
            print("="*80)
            print(f"\n📁 결과 파일:")
            print(f"   • V7_신규컬럼_기본통계.csv")
            print(f"   • V7_신규컬럼_상관관계.csv")
            print(f"   • V7_신규컬럼_급증예측력.csv")
        else:
            print("\n" + "="*80)
            print("❌ 검증 실패!")
            print("="*80)