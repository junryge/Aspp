"""
V7 신규 8개 컬럼 검증 코드
- NULL 값 자동 제거
- 기본 통계 분석
- 상관관계 분석
- 급증 예측력 테스트
"""

import pandas as pd
import numpy as np
import sys

# 신규 8개 컬럼
NEW_COLS_V7 = [
    'M16HUB.QUE.M16TOM14B.MESCURRENTQCNT',
    'M16HUB.QUE.ALL.CURRENTQCNT',
    'M16HUB.QUE.M16TOM14.MESCURRENTQCNT',
    'M16HUB.QUE.OHT.CURRENTOHTQCNT',
    'M16HUB.QUE.M16TOM14A.MESCURRENTQCNT',
    'M14.QUE.CNV.M14ATOM16ACURRNETQCNT',
    'M16HUB.QUE.OHT.OHTUTIL',
    'M16HUB.QUE.M14TOM16.MESCURRENTQCNT',
]

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'


def validate_v7_columns(csv_file):
    """V7 신규 컬럼 검증"""
    
    print("="*80)
    print("🔍 V7 신규 8개 컬럼 검증 (NULL 자동 제거)")
    print("="*80)
    
    # 데이터 로드
    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            try:
                df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='euc-kr', low_memory=False)
            except Exception as e:
                print(f"❌ 파일 로드 실패: {e}")
                return None
    
    print(f"\n✅ 데이터 로드: {len(df)}개 행, {len(df.columns)}개 컬럼")
    
    # ========================================
    # NULL 값 처리
    # ========================================
    print(f"\n🔍 NULL 값 체크 및 제거 중...")
    initial_len = len(df)
    
    # 타겟 컬럼 NULL 제거
    if TARGET_COL not in df.columns:
        print(f"❌ 타겟 컬럼 '{TARGET_COL}' 없음!")
        return None
    
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    if before != len(df):
        print(f"   타겟 컬럼 NULL 제거: {before} → {len(df)}개 ({before - len(df)}개 제거)")
    
    # 신규 컬럼들 NULL 제거
    for col in NEW_COLS_V7:
        if col in df.columns:
            before = len(df)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            null_count = df[col].isna().sum()
            
            if null_count > 0:
                print(f"   {col.split('.')[-1]}: NULL {null_count}개 발견")
                df = df.dropna(subset=[col])
                print(f"      → {before} → {len(df)}개 ({before - len(df)}개 제거)")
    
    print(f"\n✅ NULL 제거 완료: {len(df)}개 행 남음 (원본의 {len(df)/initial_len*100:.1f}%)")
    
    if len(df) < 100:
        print(f"⚠️ 경고: 데이터가 너무 적습니다! ({len(df)}개)")
        print(f"   신뢰할 수 있는 검증을 위해 최소 1000개 이상 권장")
    
    # ========================================
    # 1단계: 컬럼 존재 여부
    # ========================================
    print("\n" + "="*80)
    print("📊 1단계: 신규 컬럼 존재 여부")
    print("="*80)
    
    available = []
    missing = []
    
    for i, col in enumerate(NEW_COLS_V7, 1):
        if col in df.columns:
            available.append(col)
            valid_count = df[col].notna().sum()
            print(f"✅ {i}. {col}")
            print(f"      유효 데이터: {valid_count}개")
        else:
            missing.append(col)
            print(f"❌ {i}. {col} - 없음!")
    
    print(f"\n📊 결과: {len(available)}/{len(NEW_COLS_V7)}개 존재")
    
    if len(available) == 0:
        print("\n❌ 신규 컬럼이 하나도 없습니다!")
        print("\n💡 데이터 파일에 다음 컬럼이 필요합니다:")
        for col in NEW_COLS_V7:
            print(f"   - {col}")
        return None
    
    # ========================================
    # 2단계: 기본 통계
    # ========================================
    print("\n" + "="*80)
    print("📊 2단계: 기본 통계 (NULL 제외)")
    print("="*80)
    
    stats_list = []
    
    for col in available:
        data = df[col]
        mean = data.mean()
        std = data.std()
        cv = (std / mean * 100) if mean != 0 else 0
        
        stats_list.append({
            '컬럼': col.split('.')[-1],
            '평균': round(mean, 1),
            '표준편차': round(std, 1),
            '최소': round(data.min(), 1),
            '최대': round(data.max(), 1),
            '범위': round(data.max() - data.min(), 1),
            '변동계수(%)': round(cv, 1),
        })
        
        # 변동계수에 따른 마커
        if cv >= 20:
            marker = "🟢"
            comment = "(충분한 변동)"
        elif cv >= 10:
            marker = "🟡"
            comment = "(보통 변동)"
        elif cv >= 5:
            marker = "🟠"
            comment = "(낮은 변동)"
        else:
            marker = "🔴"
            comment = "(거의 고정)"
        
        print(f"\n{marker} {col.split('.')[-1]}:")
        print(f"   평균: {mean:.1f}, 표준편차: {std:.1f}")
        print(f"   범위: {data.min():.0f} ~ {data.max():.0f}")
        print(f"   변동계수: {cv:.1f}% {comment}")
    
    # ========================================
    # 3단계: 타겟과의 상관관계
    # ========================================
    print("\n" + "="*80)
    print("📊 3단계: 타겟과의 상관관계")
    print("="*80)
    
    corr_list = []
    
    for col in available:
        corr = df[TARGET_COL].corr(df[col])
        
        # 300+ vs 300- 비교
        high_df = df[df[TARGET_COL] >= 300]
        low_df = df[df[TARGET_COL] < 300]
        
        if len(high_df) > 0 and len(low_df) > 0:
            high = high_df[col].mean()
            low = low_df[col].mean()
            diff_pct = abs(high - low) / low * 100 if low != 0 else 0
        else:
            high = 0
            low = 0
            diff_pct = 0
        
        corr_list.append({
            '컬럼': col.split('.')[-1],
            '상관계수': round(corr, 4),
            '300+평균': round(high, 1),
            '300-평균': round(low, 1),
            '차이(%)': round(diff_pct, 1),
        })
        
        print(f"\n{col.split('.')[-1]}:")
        print(f"   상관계수: {corr:+.4f}")
        print(f"   300+ 평균: {high:.1f}")
        print(f"   300- 평균: {low:.1f}")
        print(f"   차이: {diff_pct:.1f}%", end="")
        
        if diff_pct >= 15:
            print(" 🔥 (매우 유용!)")
        elif diff_pct >= 10:
            print(" ⚡ (유용)")
        elif diff_pct >= 5:
            print(" 📊 (보통)")
        else:
            print(" ⚠️ (제한적)")
    
    # ========================================
    # 4단계: 급증 예측력 테스트
    # ========================================
    print("\n" + "="*80)
    print("📊 4단계: 급증 예측력 테스트 (임계값 기반)")
    print("="*80)
    
    # 급증 케이스 찾기
    surge_indices = []
    for i in range(30, len(df) - 10):
        seq_max = df[TARGET_COL].iloc[i-30:i].max()
        actual = df[TARGET_COL].iloc[i:i+10].max()
        if seq_max < 300 and actual >= 300:
            surge_indices.append(i)
    
    print(f"✅ 급증 케이스: {len(surge_indices)}개 발견")
    
    if len(surge_indices) == 0:
        print("⚠️ 급증 케이스가 없어서 예측력 테스트 불가")
        print("   (시퀀스MAX<300 → 실제값≥300 케이스)")
    else:
        print("\n임계값 기반 감지율:")
        
        surge_results = []
        
        for col in available:
            correct = 0
            
            for idx in surge_indices:
                last_val = df[col].iloc[idx-1]
                detected = False
                
                # 컬럼별 임계값
                if 'M16TOM14B' in col and last_val < 120:
                    detected = True
                elif 'ALL.CURRENT' in col and last_val > 850:
                    detected = True
                elif 'M16TOM14.MES' in col and last_val < 300:
                    detected = True
                elif 'OHT.CURRENT' in col and last_val > 230:
                    detected = True
                elif 'M16TOM14A' in col and last_val < 160:
                    detected = True
                elif 'M14ATOM16A' in col and last_val > 220:
                    detected = True
                elif 'OHTUTIL' in col and last_val > 85:
                    detected = True
                elif 'M14TOM16' in col and last_val < 200:
                    detected = True
                
                if detected:
                    correct += 1
            
            rate = correct / len(surge_indices) * 100
            
            surge_results.append({
                '컬럼': col.split('.')[-1],
                '감지': correct,
                '전체': len(surge_indices),
                '감지율(%)': round(rate, 1),
            })
            
            # 마커
            if rate >= 80:
                marker = "🔥"
            elif rate >= 60:
                marker = "⚡"
            elif rate >= 40:
                marker = "📊"
            else:
                marker = "  "
            
            print(f"   {marker} {col.split('.')[-1]}: {correct}/{len(surge_indices)} = {rate:.1f}%")
    
    # ========================================
    # CSV 저장
    # ========================================
    print("\n" + "="*80)
    print("📁 결과 저장")
    print("="*80)
    
    try:
        stats_df = pd.DataFrame(stats_list)
        corr_df = pd.DataFrame(corr_list).sort_values('상관계수', ascending=False, key=abs)
        
        stats_df.to_csv('V7_검증_기본통계.csv', index=False, encoding='utf-8-sig')
        print("✅ V7_검증_기본통계.csv")
        
        corr_df.to_csv('V7_검증_상관관계.csv', index=False, encoding='utf-8-sig')
        print("✅ V7_검증_상관관계.csv")
        
        if len(surge_indices) > 0:
            surge_df = pd.DataFrame(surge_results).sort_values('감지율(%)', ascending=False)
            surge_df.to_csv('V7_검증_급증예측력.csv', index=False, encoding='utf-8-sig')
            print("✅ V7_검증_급증예측력.csv")
    except Exception as e:
        print(f"⚠️ CSV 저장 실패: {e}")
        print("   (결과는 화면으로 확인 가능)")
    
    # ========================================
    # 최종 판정
    # ========================================
    print("\n" + "="*80)
    print("🎯 최종 판정")
    print("="*80)
    
    print(f"\n데이터 품질:")
    print(f"   원본: {initial_len}개 행")
    print(f"   NULL 제거 후: {len(df)}개 행")
    print(f"   사용 가능율: {len(df)/initial_len*100:.1f}%")
    
    if len(df) < initial_len * 0.5:
        print(f"   ⚠️ 경고: 50% 이상 데이터 손실!")
        print(f"   ⚠️ NULL이 너무 많습니다. 데이터 품질 점검 필요!")
    
    print(f"\n컬럼 상태:")
    print(f"   존재: {len(available)}/{len(NEW_COLS_V7)}개")
    
    if len(missing) > 0:
        print(f"   부족: {len(missing)}개")
        for col in missing:
            print(f"      ❌ {col}")
    
    print(f"\n최종 결론:")
    
    if len(available) == len(NEW_COLS_V7):
        print(f"   ✅ 8개 컬럼 모두 사용 가능!")
        print(f"   ✅ V7 모델 학습 진행 가능!")
        
        # 추가 권장사항
        high_corr = sum(1 for item in corr_list if abs(item['상관계수']) >= 0.4)
        print(f"\n   📊 고상관(≥0.4) 컬럼: {high_corr}개")
        
        if len(surge_indices) > 0:
            high_detect = sum(1 for item in surge_results if item['감지율(%)'] >= 60)
            print(f"   🎯 고감지(≥60%) 컬럼: {high_detect}개")
        
        return True
        
    elif len(available) >= 6:
        print(f"   ⚠️ {len(available)}개 컬럼만 사용 가능")
        print(f"   ⚠️ V7 학습 가능하나 성능이 제한적일 수 있음")
        return True
        
    elif len(available) >= 4:
        print(f"   ⚠️ {len(available)}개 컬럼만 사용 가능")
        print(f"   ⚠️ V7 학습 가능하나 성능이 매우 제한적")
        print(f"   💡 V6 모델 사용 권장")
        return False
        
    else:
        print(f"   ❌ 신규 컬럼 부족 ({len(available)}개)")
        print(f"   ❌ V6 모델 사용 권장")
        return False


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 V7 신규 컬럼 검증 도구 (NULL 자동 처리)")
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
        sys.exit(1)
    
    # 검증 실행
    result = validate_v7_columns(csv_file)
    
    if result is None:
        print("\n" + "="*80)
        print("❌ 검증 실패!")
        print("="*80)
        sys.exit(1)
    elif result:
        print("\n" + "="*80)
        print("✅ 검증 완료! V7 진행 가능!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("⚠️ 검증 완료! V6 사용 권장!")
        print("="*80)
        sys.exit(0)