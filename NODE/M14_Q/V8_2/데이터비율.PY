#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 1700+ 구간 추출 - 초간단 버전
"""

import pandas as pd

# ============================================================
# 핵심 코드: 1700+ 구간 추출
# ============================================================

def extract_1700plus(input_csv, output_csv='extracted_1700plus.csv'):
    """
    1700+ 발생 구간을 자동 추출
    과거 280분 + 1700+ 시점 + 이후 30분 포함
    """
    
    print("="*60)
    print("🎯 1700+ 구간 추출")
    print("="*60)
    
    # 1. 데이터 로드
    print(f"\n📂 로딩: {input_csv}")
    df = pd.read_csv(input_csv)
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
    print(f"✅ 전체: {len(df):,}행")
    
    # 2. 1700+ 시점 찾기
    print(f"\n🔍 1700+ 탐색...")
    high_indices = df[df['TOTALCNT'] >= 1700].index.tolist()
    print(f"✅ 1700+: {len(high_indices)}개")
    
    if len(high_indices) == 0:
        print("❌ 1700+ 없음!")
        return None
    
    # 3. 필요한 인덱스 수집
    print(f"\n📊 필요 구간 계산...")
    needed_indices = set()
    
    for idx in high_indices:
        if idx >= 280:  # 과거 280분이 있어야 함
            # 과거 280분 ~ 이후 30분
            start = idx - 280
            end = min(idx + 30, len(df) - 1)
            needed_indices.update(range(start, end + 1))
    
    # 4. 데이터 추출
    print(f"\n💾 추출 중...")
    extracted_indices = sorted(needed_indices)
    extracted_df = df.iloc[extracted_indices].copy()
    extracted_df = extracted_df.sort_values('CURRTIME').reset_index(drop=True)
    
    # 5. 저장
    extracted_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"✅ 완료: {output_csv}")
    print(f"   추출: {len(extracted_df):,}행 (원본의 {len(extracted_df)/len(df)*100:.1f}%)")
    print(f"   1700+: {(extracted_df['TOTALCNT'] >= 1700).sum()}개")
    print("="*60)
    
    return extracted_df


# ============================================================
# 사용 예시
# ============================================================

if __name__ == '__main__':
    
    # 예시 1: Ois.csv에서 1700+ 구간 추출
    extract_1700plus('Ois.csv', 'extracted_1700plus.csv')
    
    # 예시 2: 다른 파일에서 추출
    # extract_1700plus('다른데이터.csv', 'extracted_other.csv')
    
    # 예시 3: 여러 파일 추출 후 합치기
    # df1 = extract_1700plus('8월데이터.csv', 'extracted_8월.csv')
    # df2 = extract_1700plus('9월데이터.csv', 'extracted_9월.csv')
    # df3 = extract_1700plus('10월데이터.csv', 'extracted_10월.csv')
    # 
    # # 합치기
    # combined = pd.concat([df1, df2, df3], ignore_index=True)
    # combined = combined.sort_values('CURRTIME').reset_index(drop=True)
    # combined.to_csv('combined_1700plus.csv', index=False, encoding='utf-8-sig')
    # print(f"✅ 합침 완료: {len(combined):,}행, 1700+: {(combined['TOTALCNT']>=1700).sum()}개")