#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 1700+ 데이터 생성기
실제 1700+ 패턴을 분석하여 새로운 데이터 생성
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_1700_patterns(df):
    """1700+ 패턴 분석"""
    print("\n📊 1700+ 패턴 분석")
    print("="*80)
    
    high_risk = df[df['TOTALCNT'] >= 1700].copy()
    
    if len(high_risk) == 0:
        print("❌ 1700+ 샘플 없음!")
        return None
    
    print(f"1700+ 샘플: {len(high_risk)}개")
    
    # 패턴 통계
    patterns = {
        'TOTALCNT': {
            'mean': high_risk['TOTALCNT'].mean(),
            'std': high_risk['TOTALCNT'].std(),
            'min': high_risk['TOTALCNT'].min(),
            'max': high_risk['TOTALCNT'].max()
        },
        'M14AM14B': {
            'mean': high_risk['M14AM14B'].mean(),
            'std': high_risk['M14AM14B'].std(),
            'min': high_risk['M14AM14B'].min(),
            'max': high_risk['M14AM14B'].max()
        },
        'M14AM10A': {
            'mean': high_risk['M14AM10A'].mean(),
            'std': high_risk['M14AM10A'].std(),
            'min': high_risk['M14AM10A'].min(),
            'max': high_risk['M14AM10A'].max()
        },
        'M14AM16': {
            'mean': high_risk['M14AM16'].mean(),
            'std': high_risk['M14AM16'].std(),
            'min': high_risk['M14AM16'].min(),
            'max': high_risk['M14AM16'].max()
        },
        'M14.QUE.OHT.OHTUTIL': {
            'mean': high_risk['M14.QUE.OHT.OHTUTIL'].mean(),
            'std': high_risk['M14.QUE.OHT.OHTUTIL'].std(),
            'min': high_risk['M14.QUE.OHT.OHTUTIL'].min(),
            'max': high_risk['M14.QUE.OHT.OHTUTIL'].max()
        },
        'M14.QUE.ALL.CURRENTQCREATED': {
            'mean': high_risk['M14.QUE.ALL.CURRENTQCREATED'].mean(),
            'std': high_risk['M14.QUE.ALL.CURRENTQCREATED'].std(),
            'min': high_risk['M14.QUE.ALL.CURRENTQCREATED'].min(),
            'max': high_risk['M14.QUE.ALL.CURRENTQCREATED'].max()
        },
        'M14.QUE.ALL.CURRENTQCOMPLETED': {
            'mean': high_risk['M14.QUE.ALL.CURRENTQCOMPLETED'].mean(),
            'std': high_risk['M14.QUE.ALL.CURRENTQCOMPLETED'].std(),
            'min': high_risk['M14.QUE.ALL.CURRENTQCOMPLETED'].min(),
            'max': high_risk['M14.QUE.ALL.CURRENTQCOMPLETED'].max()
        }
    }
    
    print(f"\nTOTALCNT: {patterns['TOTALCNT']['mean']:.1f} ± {patterns['TOTALCNT']['std']:.1f}")
    print(f"M14AM14B: {patterns['M14AM14B']['mean']:.1f} ± {patterns['M14AM14B']['std']:.1f}")
    print(f"OHT_UTIL: {patterns['M14.QUE.OHT.OHTUTIL']['mean']:.1f} ± {patterns['M14.QUE.OHT.OHTUTIL']['std']:.1f}")
    
    return patterns, high_risk

def generate_new_sample(base_sample, patterns, variation=0.3):
    """기존 샘플 기반으로 새 샘플 생성 (큰 변형!)"""
    new_sample = base_sample.copy()
    
    # TOTALCNT: 1700~1900 범위에서 생성 (정수!)
    new_sample['TOTALCNT'] = int(np.random.uniform(1700, 1900))
    
    # M14AM14B: 큰 변형 (200~600)
    base_val = base_sample['M14AM14B']
    noise = np.random.normal(0, patterns['M14AM14B']['std'] * 0.8)
    new_sample['M14AM14B'] = int(np.clip(base_val + noise, 200, 600))
    
    # M14AM10A: 큰 변형 (40~130)
    base_val = base_sample['M14AM10A']
    noise = np.random.normal(0, patterns['M14AM10A']['std'] * 0.8)
    new_sample['M14AM10A'] = int(np.clip(base_val + noise, 40, 130))
    
    # M14AM16: 큰 변형 (60~120)
    base_val = base_sample['M14AM16']
    noise = np.random.normal(0, patterns['M14AM16']['std'] * 0.8)
    new_sample['M14AM16'] = int(np.clip(base_val + noise, 60, 120))
    
    # OHT_UTIL: 80~98 범위 (고부하, 큰 변형)
    base_val = base_sample['M14.QUE.OHT.OHTUTIL']
    noise = np.random.normal(0, patterns['M14.QUE.OHT.OHTUTIL']['std'] * 0.8)
    new_sample['M14.QUE.OHT.OHTUTIL'] = round(np.clip(base_val + noise, 80, 98), 2)
    
    # CURRENTQCREATED: 큰 변형
    base_val = base_sample['M14.QUE.ALL.CURRENTQCREATED']
    noise = np.random.normal(0, patterns['M14.QUE.ALL.CURRENTQCREATED']['std'] * 0.8)
    new_sample['M14.QUE.ALL.CURRENTQCREATED'] = int(np.clip(base_val + noise, 2700, 3300))
    
    # CURRENTQCOMPLETED: 큰 변형
    base_val = base_sample['M14.QUE.ALL.CURRENTQCOMPLETED']
    noise = np.random.normal(0, patterns['M14.QUE.ALL.CURRENTQCOMPLETED']['std'] * 0.8)
    new_sample['M14.QUE.ALL.CURRENTQCOMPLETED'] = int(np.clip(base_val + noise, 2600, 3100))
    
    # M14.QUE.ALL.TRANSPORT4MINOVERCNT: 큰 변형
    base_val = base_sample['M14.QUE.ALL.TRANSPORT4MINOVERCNT']
    noise = np.random.uniform(-50, 50)
    new_sample['M14.QUE.ALL.TRANSPORT4MINOVERCNT'] = int(np.clip(base_val + noise, 60, 200))
    
    # M14B.QUE.SENDFAB.VERTICALQUEUECOUNT: 큰 변형
    base_val = base_sample['M14B.QUE.SENDFAB.VERTICALQUEUECOUNT']
    noise = np.random.uniform(-60, 60)
    new_sample['M14B.QUE.SENDFAB.VERTICALQUEUECOUNT'] = int(np.clip(base_val + noise, 130, 320))
    
    # M10A.QUE.ALL.CURRENTQCNT: 큰 변형
    base_val = base_sample['M10A.QUE.ALL.CURRENTQCNT']
    noise = np.random.uniform(-50, 50)
    new_sample['M10A.QUE.ALL.CURRENTQCNT'] = int(np.clip(base_val + noise, 280, 480))
    
    # M10A.QUE.OHT_OHS.OHTUTIL: 큰 변형
    base_val = base_sample['M10A.QUE.OHT_OHS.OHTUTIL']
    noise = np.random.uniform(-15, 15)
    new_sample['M10A.QUE.OHT_OHS.OHTUTIL'] = round(np.clip(base_val + noise, 25, 95), 2)
    
    # 기타 컬럼들 (큰 범위)
    new_sample['M10AM14A'] = int(np.random.uniform(35, 65))
    new_sample['M14AM10ASUM'] = new_sample['M14AM10A'] + new_sample['M10AM14A']
    new_sample['M14BM14A'] = int(np.random.uniform(50, 120))
    new_sample['M14AM14BSUM'] = new_sample['M14AM14B'] + new_sample['M14BM14A']
    new_sample['M16M14A'] = int(np.random.uniform(110, 170))
    new_sample['M14AM16SUM'] = new_sample['M14AM16'] + new_sample['M16M14A']
    
    return new_sample

def generate_synthetic_data(df, num_samples=2000):
    """합성 데이터 생성"""
    print(f"\n🔥 합성 데이터 생성 시작")
    print(f"목표: {num_samples}개 생성")
    print("="*80)
    
    # 1700+ 패턴 분석
    result = analyze_1700_patterns(df)
    if result is None:
        return None
    
    patterns, high_risk_samples = result
    
    # 새 데이터 생성
    synthetic_data = []
    
    # 마지막 시간 가져오기
    last_time = pd.to_datetime(df['CURRTIME'].iloc[-1], format='%Y%m%d%H%M')
    
    print(f"\n⚙️ 생성 중...")
    for i in range(num_samples):
        # 랜덤하게 기존 1700+ 샘플 선택
        base_sample = high_risk_samples.sample(1).iloc[0]
        
        # 새 샘플 생성 (큰 변형!)
        new_sample = generate_new_sample(base_sample, patterns, variation=0.5)
        
        # 시간 설정 (1분씩 증가)
        new_time = last_time + timedelta(minutes=i+1)
        new_sample['CURRTIME'] = new_time.strftime('%Y%m%d%H%M')
        new_sample['TIME'] = new_time.strftime('%Y%m%d%H%M')
        
        synthetic_data.append(new_sample)
        
        if (i+1) % 500 == 0:
            print(f"  진행: {i+1}/{num_samples} ({(i+1)/num_samples*100:.1f}%)")
    
    print(f"✅ 생성 완료: {len(synthetic_data)}개")
    
    # DataFrame 변환
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # 통계
    print(f"\n📊 생성된 데이터 통계:")
    print(f"  TOTALCNT 범위: {synthetic_df['TOTALCNT'].min():.0f} ~ {synthetic_df['TOTALCNT'].max():.0f}")
    print(f"  TOTALCNT 평균: {synthetic_df['TOTALCNT'].mean():.1f}")
    print(f"  1700+ 샘플: {(synthetic_df['TOTALCNT'] >= 1700).sum()}개 (100%)")
    
    return synthetic_df

def merge_data(original_df, synthetic_df):
    """원본 + 합성 데이터 병합"""
    print(f"\n🔗 데이터 병합")
    print("="*80)
    
    print(f"원본: {len(original_df):,}개")
    print(f"합성: {len(synthetic_df):,}개")
    
    # 병합
    merged_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    
    print(f"병합 후: {len(merged_df):,}개")
    print(f"1700+ 샘플: {(merged_df['TOTALCNT'] >= 1700).sum():,}개")
    
    return merged_df

def main():
    """메인 실행"""
    print("="*80)
    print("🚀 1700+ 합성 데이터 생성기")
    print("="*80)
    
    # 원본 데이터 로드
    csv_file = 'Ois.csv'
    print(f"\n📂 데이터 로딩: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"✅ 원본: {len(df):,}행")
    print(f"1700+ 샘플: {(df['TOTALCNT'] >= 1700).sum()}개")
    
    # 합성 데이터 생성 (더 많이!)
    num_synthetic = 3000  # 생성할 개수
    synthetic_df = generate_synthetic_data(df, num_samples=num_synthetic)
    
    if synthetic_df is None:
        print("❌ 합성 데이터 생성 실패")
        return
    
    # 병합
    merged_df = merge_data(df, synthetic_df)
    
    # 저장
    output_file = 'Ois_augmented.csv'
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장: {output_file}")
    
    print("\n" + "="*80)
    print("✅ 완료!")
    print("="*80)
    print(f"원본 파일: {csv_file}")
    print(f"증강 파일: {output_file}")
    print(f"\n원본: {len(df):,}개 (1700+: {(df['TOTALCNT'] >= 1700).sum()}개)")
    print(f"증강: {len(merged_df):,}개 (1700+: {(merged_df['TOTALCNT'] >= 1700).sum()}개)")
    print(f"\n1700+ 증가: {(df['TOTALCNT'] >= 1700).sum()}개 → {(merged_df['TOTALCNT'] >= 1700).sum()}개")
    print("="*80)

if __name__ == '__main__':
    main()