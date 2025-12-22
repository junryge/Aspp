import pandas as pd
import numpy as np

def analyze_columns_for_model_selection():
    """
    📊 V8+V9 전체 컬럼 분석
    목적: MIN/MAX/AVG 선택에 도움되는 신호 찾기
    """
    print("="*80)
    print("📊 V8+V9 전체 컬럼 분석: MIN/MAX/AVG 선택 신호 찾기")
    print("="*80)
    
    # 데이터 로드
    try:
        df = pd.read_csv('test_data.csv', encoding='utf-8', on_bad_lines='skip', low_memory=False)
    except:
        try:
            df = pd.read_csv('test_data.csv', encoding='cp949', on_bad_lines='skip', low_memory=False)
        except:
            df = pd.read_csv('test_data.csv', encoding='euc-kr', on_bad_lines='skip', low_memory=False)
    
    print(f"✅ 데이터 로드: {len(df)}행")
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # V8 기존 14개
    V8_COLS = [
        'M16A_3F_STORAGE_UTIL',
        'CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL',
        'HUBROOMTOTAL',
        'M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD',
        'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2',
        'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB',
        'M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA',
    ]
    
    # V9 신규 20개
    V9_COLS = [
        'M16A_LFTTOALL',
        'M16HUB_M16TOM14_CREATED',
        'M16A_ALLTONORTHCNV',
        'M16A_NORTHCNVTOALL',
        'M16A_M16ATOM14A',
        'M14_TOTALCNVCURRENTQCNT',
        'M16HUB_M16TOM14B_CREATED',
        'M16HUB_M14TOM16_CREATED',
        'M16A_CURRENTQCNT',
        'M16A_ALLTOSOUTHCNV',
        'M14_ALLTOSOUTHCNV',
        'M14_ALLTONORTHCNV',
        'M14_CURRENTQCREATED',
        'M14_RETURNTOM16',
        'M14B_LFTTOALL',
        'M14B_M14BTOM16A',
        'M16B_CURRENTQCREATED',
        'M16_SENDTOM14',
        'M16HUB_M14TOM16_MESCNT',
        'M16HUB_MESCURRENTQCNT',
    ]
    
    ALL_COLS = V8_COLS + V9_COLS
    
    # 존재하는 컬럼만
    available = [c for c in ALL_COLS if c in df.columns]
    missing = [c for c in ALL_COLS if c not in df.columns]
    
    print(f"\n📋 컬럼 현황:")
    print(f"  사용 가능: {len(available)}/{len(ALL_COLS)}개")
    if missing:
        print(f"  누락: {missing}")
    
    # 타겟 컬럼 처리
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    
    # 10분 후 변화량 계산
    df['future_change'] = df[TARGET_COL].shift(-10) - df[TARGET_COL]
    df = df.dropna(subset=['future_change'])
    
    # 분류
    df['category'] = 'normal'
    df.loc[df['future_change'] >= 50, 'category'] = 'surge'   # 급등
    df.loc[df['future_change'] <= -50, 'category'] = 'drop'   # 급락
    
    surge_df = df[df['category'] == 'surge']
    drop_df = df[df['category'] == 'drop']
    normal_df = df[df['category'] == 'normal']
    
    print(f"\n📊 데이터 분류:")
    print(f"  📈 급등 (+50이상): {len(surge_df)}개 ({len(surge_df)/len(df)*100:.2f}%)")
    print(f"  📉 급락 (-50이하): {len(drop_df)}개 ({len(drop_df)/len(df)*100:.2f}%)")
    print(f"  ⚖️ 일반: {len(normal_df)}개 ({len(normal_df)/len(df)*100:.2f}%)")
    
    # ========== 분석 1: 평균값 비교 ==========
    print(f"\n" + "="*80)
    print("📊 [분석 1] 컬럼별 급등/급락/일반 평균값 비교")
    print("="*80)
    
    results = []
    
    for col in available:
        try:
            col_data = pd.to_numeric(df[col], errors='coerce')
            
            surge_mean = col_data[df['category']=='surge'].mean()
            drop_mean = col_data[df['category']=='drop'].mean()
            normal_mean = col_data[df['category']=='normal'].mean()
            
            # 급등 vs 일반 차이 (%)
            if abs(normal_mean) > 0.001:
                surge_diff = (surge_mean - normal_mean) / abs(normal_mean) * 100
                drop_diff = (drop_mean - normal_mean) / abs(normal_mean) * 100
            else:
                surge_diff = 0
                drop_diff = 0
            
            results.append({
                'col': col,
                'surge_mean': surge_mean,
                'drop_mean': drop_mean,
                'normal_mean': normal_mean,
                'surge_diff%': surge_diff,
                'drop_diff%': drop_diff,
                'abs_diff': abs(surge_diff) + abs(drop_diff)
            })
        except Exception as e:
            print(f"  ⚠️ {col} 처리 오류: {e}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('abs_diff', ascending=False)
    
    print(f"\n🔥 급등/급락 구분에 유용한 컬럼 (차이 큰 순 Top 20):")
    print("-"*90)
    print(f"{'컬럼':<30} {'급등평균':>10} {'급락평균':>10} {'일반평균':>10} {'급등차이%':>10} {'급락차이%':>10}")
    print("-"*90)
    
    for _, row in results_df.head(20).iterrows():
        print(f"{row['col']:<30} {row['surge_mean']:>10.1f} {row['drop_mean']:>10.1f} {row['normal_mean']:>10.1f} {row['surge_diff%']:>+10.1f} {row['drop_diff%']:>+10.1f}")
    
    # ========== 분석 2: 변곡점 감지 ==========
    print(f"\n" + "="*80)
    print("📊 [분석 2] 변곡점 감지 (하락추세→급등, 상승추세→급락)")
    print("="*80)
    
    # 30분 추세 계산
    slopes = []
    for i in range(30, len(df)):
        seq = df[TARGET_COL].iloc[i-30:i].values
        slope = np.polyfit(np.arange(30), seq, 1)[0]
        slopes.append(slope)
    
    df_with_slope = df.iloc[30:].copy()
    df_with_slope['slope'] = slopes
    
    # 변곡점 분류
    # 하락추세(slope<0) + 급등 = 변곡점 급등
    # 상승추세(slope>0) + 급락 = 변곡점 급락
    변곡점_급등 = df_with_slope[(df_with_slope['slope'] < 0) & (df_with_slope['category'] == 'surge')]
    변곡점_급락 = df_with_slope[(df_with_slope['slope'] > 0) & (df_with_slope['category'] == 'drop')]
    추세유지_급등 = df_with_slope[(df_with_slope['slope'] >= 0) & (df_with_slope['category'] == 'surge')]
    추세유지_급락 = df_with_slope[(df_with_slope['slope'] <= 0) & (df_with_slope['category'] == 'drop')]
    
    print(f"\n📈 급등 분석:")
    print(f"  변곡점 급등 (하락중→급등): {len(변곡점_급등)}개")
    print(f"  추세유지 급등 (상승중→급등): {len(추세유지_급등)}개")
    
    print(f"\n📉 급락 분석:")
    print(f"  변곡점 급락 (상승중→급락): {len(변곡점_급락)}개")
    print(f"  추세유지 급락 (하락중→급락): {len(추세유지_급락)}개")
    
    # 변곡점 vs 추세유지 컬럼 비교
    if len(변곡점_급등) > 0 and len(추세유지_급등) > 0:
        print(f"\n🔥 변곡점 급등 vs 추세유지 급등 비교:")
        print("-"*70)
        print(f"{'컬럼':<30} {'변곡점급등':>12} {'추세유지급등':>12} {'차이':>10}")
        print("-"*70)
        
        diff_results = []
        for col in available:
            try:
                col_data = pd.to_numeric(df_with_slope[col], errors='coerce')
                변곡_mean = col_data[df_with_slope.index.isin(변곡점_급등.index)].mean()
                추세_mean = col_data[df_with_slope.index.isin(추세유지_급등.index)].mean()
                diff = 변곡_mean - 추세_mean
                diff_results.append({'col': col, '변곡점': 변곡_mean, '추세유지': 추세_mean, 'diff': diff, 'abs_diff': abs(diff)})
            except:
                pass
        
        diff_df = pd.DataFrame(diff_results).sort_values('abs_diff', ascending=False)
        for _, row in diff_df.head(10).iterrows():
            print(f"{row['col']:<30} {row['변곡점']:>12.1f} {row['추세유지']:>12.1f} {row['diff']:>+10.1f}")
    
    # ========== 분석 3: 임계값 찾기 ==========
    print(f"\n" + "="*80)
    print("📊 [분석 3] MIN/MAX/AVG 선택 임계값 후보")
    print("="*80)
    
    print(f"\n각 컬럼별 급등/급락 시 분포:")
    
    for col in available[:10]:  # 상위 10개만
        try:
            col_data = pd.to_numeric(df[col], errors='coerce')
            
            print(f"\n{col}:")
            print(f"  급등 시: min={col_data[df['category']=='surge'].min():.1f}, "
                  f"max={col_data[df['category']=='surge'].max():.1f}, "
                  f"median={col_data[df['category']=='surge'].median():.1f}")
            print(f"  급락 시: min={col_data[df['category']=='drop'].min():.1f}, "
                  f"max={col_data[df['category']=='drop'].max():.1f}, "
                  f"median={col_data[df['category']=='drop'].median():.1f}")
            print(f"  일반 시: min={col_data[df['category']=='normal'].min():.1f}, "
                  f"max={col_data[df['category']=='normal'].max():.1f}, "
                  f"median={col_data[df['category']=='normal'].median():.1f}")
        except:
            pass
    
    # ========== 결과 저장 ==========
    output_file = '컬럼분석_결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 분석 결과 저장: {output_file}")
    
    print(f"\n" + "="*80)
    print("✅ 분석 완료!")
    print("="*80)
    print(f"\n💡 다음 단계:")
    print(f"  1. 차이가 큰 컬럼을 MIN/MAX/AVG 선택 조건에 추가")
    print(f"  2. 변곡점 감지용 특수 조건 설계")
    print(f"  3. 임계값 테스트")
    
    return results_df

if __name__ == '__main__':
    print("🚀 V9 컬럼 분석 시작...\n")
    results = analyze_columns_for_model_selection()