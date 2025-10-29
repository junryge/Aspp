import pandas as pd
import numpy as np

def verify_v7_features():
    """
    🔍 V7 신규 Feature 후보 8개 검증
    """
    print("="*80)
    print("🔍 V7 신규 Feature 검증 시작")
    print("="*80)
    
    # 검증할 컬럼 리스트
    new_features = [
        'M16HUB.QUE.M16TOM14B.MESCURRENTQCNT',  # +0.59
        'M16HUB.QUE.ALL.CURRENTQCNT',           # +0.57
        'M16HUB.QUE.M16TOM14.MESCURRENTQCNT',   # +0.48
        'M16HUB.QUE.OHT.CURRENTOHTQCNT',        # +0.46
        'M16HUB.QUE.M16TOM14A.MESCURRENTQCNT',  # +0.42
        'M14.QUE.CNV.M14ATOM16ACURRNETQCNT',    # +0.36
        'M16HUB.QUE.OHT.OHTUTIL',               # +0.34
        'M16HUB.QUE.M14TOM16.MESCURRENTQCNT'    # -0.32
    ]
    
    expected_corr = [0.59, 0.57, 0.48, 0.46, 0.42, 0.36, 0.34, -0.32]
    
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
    
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
    print(f"✅ 타겟 컬럼 전처리 완료: {len(df)}개 행")
    
    # 검증 시작
    print("\n[2] 컬럼 존재 여부 & 상관계수 검증")
    print("-"*80)
    
    results = []
    
    for i, col in enumerate(new_features):
        result = {
            '순위': i+1,
            '컬럼명': col,
            '존재여부': '',
            '결측률(%)': 0,
            '계산상관계수': 0,
            '예상상관계수': expected_corr[i],
            '오차': 0,
            '검증': ''
        }
        
        # 존재 여부 확인
        if col in df.columns:
            result['존재여부'] = '✅'
            
            # 숫자 변환
            col_data = pd.to_numeric(df[col], errors='coerce')
            
            # 결측률 계산
            missing_rate = col_data.isna().sum() / len(col_data) * 100
            result['결측률(%)'] = round(missing_rate, 2)
            
            # 상관계수 계산
            valid_mask = col_data.notna() & df[TARGET_COL].notna()
            if valid_mask.sum() > 0:
                corr = col_data[valid_mask].corr(df[TARGET_COL][valid_mask])
                result['계산상관계수'] = round(corr, 4)
                result['오차'] = round(abs(corr - expected_corr[i]), 4)
                
                # 검증 (오차 0.05 이내면 OK)
                if abs(corr - expected_corr[i]) <= 0.05:
                    result['검증'] = '✅'
                else:
                    result['검증'] = f'⚠️ 차이큼'
            else:
                result['검증'] = '❌ 계산불가'
        else:
            result['존재여부'] = '❌'
            result['검증'] = '❌ 컬럼없음'
        
        results.append(result)
        
        # 실시간 출력
        status = result['검증']
        print(f"{i+1}. {col[:40]:<40} | 존재:{result['존재여부']} | 상관:{result['계산상관계수']:>7.4f} | 예상:{expected_corr[i]:>6.2f} | {status}")
    
    # DataFrame 변환
    results_df = pd.DataFrame(results)
    
    # 결과 저장
    output_file = 'V7_feature_검증결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
    
    # 통계 요약
    print("\n" + "="*80)
    print("📊 검증 요약")
    print("="*80)
    
    total = len(results_df)
    exists = (results_df['존재여부'] == '✅').sum()
    verified = (results_df['검증'] == '✅').sum()
    
    print(f"\n총 검증 대상: {total}개")
    print(f"컬럼 존재: {exists}/{total}개 ({exists/total*100:.1f}%)")
    print(f"상관계수 일치: {verified}/{total}개 ({verified/total*100:.1f}%)")
    
    # 문제 컬럼
    problems = results_df[results_df['검증'] != '✅']
    if len(problems) > 0:
        print(f"\n⚠️ 문제 컬럼 ({len(problems)}개):")
        for idx, row in problems.iterrows():
            print(f"  - {row['컬럼명']}: {row['검증']}")
    else:
        print(f"\n🎉 모든 컬럼 검증 완료!")
    
    # 상관계수 TOP 8 재확인
    print(f"\n🔥 상관계수 TOP 8 (재계산):")
    verified_cols = results_df[results_df['존재여부'] == '✅'].copy()
    verified_cols = verified_cols.sort_values('계산상관계수', key=abs, ascending=False)
    
    for idx, row in verified_cols.iterrows():
        marker = "⭐" if abs(row['계산상관계수']) >= 0.40 else "  "
        print(f"{marker} {row['순위']}. {row['컬럼명'][:50]:<50} : {row['계산상관계수']:>7.4f}")
    
    return results_df

if __name__ == '__main__':
    print("🚀 V7 신규 Feature 검증 시작...\n")
    results = verify_v7_features()
    
    if results is not None:
        print(f"\n✅ 검증 완료!")
        print(f"📁 결과: V7_feature_검증결과.csv")