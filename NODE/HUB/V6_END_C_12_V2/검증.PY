import pandas as pd
import numpy as np

def analyze_all_correlations():
    """
    🔍 전체 컬럼 상관계수 분석 - TOP 30 찾기
    """
    print("="*80)
    print("🔍 전체 컬럼 상관계수 분석")
    print("="*80)
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # 데이터 로드
    print("\n[1] 데이터 로드...")
    
    # ✅ 파일명을 여기에 입력하세요!
    data_file = '20250904_TO_20251020.csv'
    
    print(f"📂 파일: {data_file}")
    
    try:
        df = pd.read_csv(data_file, on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df = pd.read_csv(data_file, on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df = pd.read_csv(data_file, on_bad_lines='skip', encoding='euc-kr', low_memory=False)
    
    print(f"✅ 데이터 로드 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
    
    # 타겟 컬럼 확인
    if TARGET_COL not in df.columns:
        print(f"\n❌ 타겟 컬럼 없음: {TARGET_COL}")
        return
    
    # 🆕 NULL 값 처리 - 바로 위 값으로 채우기 (forward fill)
    print(f"\n🔧 NULL 값 처리 중...")
    null_before = df.isnull().sum().sum()
    print(f"  처리 전 NULL: {null_before}개")
    
    # 모든 컬럼에 대해 forward fill 적용
    df = df.ffill()
    
    # 만약 첫 행에 NULL이 있으면 backward fill로 보완
    df = df.bfill()
    
    null_after = df.isnull().sum().sum()
    print(f"  처리 후 NULL: {null_after}개")
    print(f"  ✅ {null_before - null_after}개 NULL 값 처리 완료!")
    
    # 타겟 컬럼 숫자 변환
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    
    # 여전히 NULL이 남아있다면 제거
    before_drop = len(df)
    df = df.dropna(subset=[TARGET_COL])
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"  타겟 컬럼 NULL 제거: {dropped}개 행")
    
    print(f"✅ 최종 데이터: {len(df)}개 행")
    
    # [2] 전체 컬럼 상관계수 계산
    print("\n[2] 전체 컬럼 상관계수 계산 중...")
    print("-"*80)
    
    results = []
    
    for col in df.columns:
        if col == TARGET_COL:
            continue  # 타겟 컬럼 자체는 제외
        
        if col == 'STAT_DT':
            continue  # 시간 컬럼 제외
        
        # 숫자 변환
        col_data = pd.to_numeric(df[col], errors='coerce')
        
        # NULL 처리
        col_data = col_data.ffill().bfill()
        
        # 변동성 확인 (모두 같은 값이면 제외)
        if col_data.nunique() <= 1:
            continue
        
        # 상관계수 계산
        valid_mask = col_data.notna() & df[TARGET_COL].notna()
        
        if valid_mask.sum() < 10:  # 유효 데이터 10개 미만이면 제외
            continue
        
        try:
            corr = col_data[valid_mask].corr(df[TARGET_COL][valid_mask])
            
            if pd.notna(corr):
                results.append({
                    '컬럼명': col,
                    '상관계수': round(corr, 4),
                    '절대값': round(abs(corr), 4),
                    '결측률(%)': round((col_data.isna().sum() / len(col_data)) * 100, 2),
                    '고유값개수': col_data.nunique(),
                    '평균': round(col_data.mean(), 2),
                    '표준편차': round(col_data.std(), 2)
                })
        except:
            continue
    
    # DataFrame 변환
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\n❌ 계산 가능한 컬럼이 없습니다!")
        return
    
    # 절대값 기준 정렬
    results_df = results_df.sort_values('절대값', ascending=False)
    
    # 결과 저장
    output_file = '전체컬럼_상관계수_분석결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 전체 결과 저장: {output_file}")
    
    # [3] TOP 30 출력
    print("\n" + "="*80)
    print("🔥 상관계수 TOP 30 (절대값 기준)")
    print("="*80)
    
    top30 = results_df.head(30)
    
    for idx, row in top30.iterrows():
        # 상관계수 양수/음수 표시
        if row['상관계수'] > 0:
            sign = "+"
            marker = "📈" if row['절대값'] >= 0.3 else "  "
        else:
            sign = ""
            marker = "📉" if row['절대값'] >= 0.3 else "  "
        
        print(f"{marker} {row['컬럼명'][:55]:<55} : {sign}{row['상관계수']:>7.4f}")
    
    # [4] 통계 요약
    print("\n" + "="*80)
    print("📊 분석 요약")
    print("="*80)
    
    print(f"\n총 분석 컬럼: {len(results_df)}개")
    
    high_corr = results_df[results_df['절대값'] >= 0.3]
    medium_corr = results_df[(results_df['절대값'] >= 0.2) & (results_df['절대값'] < 0.3)]
    low_corr = results_df[results_df['절대값'] < 0.2]
    
    print(f"\n상관계수 분포:")
    print(f"  - 높음 (≥0.30): {len(high_corr)}개")
    print(f"  - 중간 (0.20~0.29): {len(medium_corr)}개")
    print(f"  - 낮음 (<0.20): {len(low_corr)}개")
    
    # 상관계수 높은 컬럼 그룹화
    print(f"\n🎯 상관계수 ≥0.30 컬럼 ({len(high_corr)}개):")
    for idx, row in high_corr.iterrows():
        print(f"  • {row['컬럼명'][:50]:<50} : {row['상관계수']:>7.4f}")
    
    # TOP 10 상세 정보
    print(f"\n📌 TOP 10 상세 정보:")
    print("-"*80)
    top10 = results_df.head(10)
    print(top10[['컬럼명', '상관계수', '고유값개수', '평균', '표준편차']].to_string(index=False))
    
    return results_df

if __name__ == '__main__':
    print("🚀 전체 컬럼 상관계수 분석 시작...\n")
    results = analyze_all_correlations()
    
    if results is not None:
        print(f"\n✅ 분석 완료!")
        print(f"📁 결과: 전체컬럼_상관계수_분석결과.csv")
        print(f"\n💡 이제 이 결과를 보고 V7 feature를 선택하세요!")