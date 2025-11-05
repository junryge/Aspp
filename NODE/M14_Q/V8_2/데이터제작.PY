#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 실제 1700+ 패턴 기반 데이터 생성기
실제로 발생한 1700+ 패턴만 정확히 복제!
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_realistic_1700_sample(base_sample):
    """실제 1700+ 패턴과 동일하게 생성"""
    new_sample = base_sample.copy()
    
    # 🎯 실제 1700+ 패턴 범위 (정확히!)
    
    # TOTALCNT: 1700~1750 집중!
    new_sample['TOTALCNT'] = int(np.random.uniform(1700, 1750))
    
    # M14AM14B: 400~530 (실제 패턴)
    base_val = base_sample['M14AM14B']
    new_sample['M14AM14B'] = int(np.clip(
        base_val + np.random.uniform(-30, 30), 400, 530
    ))
    
    # M14AM10A: 75~110 (실제 패턴)
    base_val = base_sample['M14AM10A']
    new_sample['M14AM10A'] = int(np.clip(
        base_val + np.random.uniform(-15, 15), 75, 110
    ))
    
    # M14AM16: 80~105 (실제 패턴)
    base_val = base_sample['M14AM16']
    new_sample['M14AM16'] = int(np.clip(
        base_val + np.random.uniform(-10, 10), 80, 105
    ))
    
    # OHT_UTIL: 88~96 (실제 패턴, 좁은 범위!)
    base_val = base_sample['M14.QUE.OHT.OHTUTIL']
    new_sample['M14.QUE.OHT.OHTUTIL'] = round(np.clip(
        base_val + np.random.uniform(-2, 2), 88, 96
    ), 2)
    
    # CURRENTQCREATED: 2830~2950 (실제 패턴)
    base_val = base_sample['M14.QUE.ALL.CURRENTQCREATED']
    new_sample['M14.QUE.ALL.CURRENTQCREATED'] = int(np.clip(
        base_val + np.random.uniform(-50, 50), 2830, 2950
    ))
    
    # CURRENTQCOMPLETED: 2680~2920 (실제 패턴)
    base_val = base_sample['M14.QUE.ALL.CURRENTQCOMPLETED']
    new_sample['M14.QUE.ALL.CURRENTQCOMPLETED'] = int(np.clip(
        base_val + np.random.uniform(-50, 50), 2680, 2920
    ))
    
    # TRANSPORT4MINOVERCNT: 180~280 (실제 패턴)
    base_val = base_sample['M14.QUE.ALL.TRANSPORT4MINOVERCNT']
    new_sample['M14.QUE.ALL.TRANSPORT4MINOVERCNT'] = int(np.clip(
        base_val + np.random.uniform(-30, 30), 180, 280
    ))
    
    # VERTICALQUEUECOUNT: 195~265 (실제 패턴)
    base_val = base_sample['M14B.QUE.SENDFAB.VERTICALQUEUECOUNT']
    new_sample['M14B.QUE.SENDFAB.VERTICALQUEUECOUNT'] = int(np.clip(
        base_val + np.random.uniform(-30, 30), 195, 265
    ))
    
    # M10A.QUE.ALL.CURRENTQCNT: 330~390 (실제 패턴)
    base_val = base_sample['M10A.QUE.ALL.CURRENTQCNT']
    new_sample['M10A.QUE.ALL.CURRENTQCNT'] = int(np.clip(
        base_val + np.random.uniform(-20, 20), 330, 390
    ))
    
    # M10A.QUE.OHT_OHS.OHTUTIL: 50~85 (실제 패턴)
    base_val = base_sample['M10A.QUE.OHT_OHS.OHTUTIL']
    new_sample['M10A.QUE.OHT_OHS.OHTUTIL'] = round(np.clip(
        base_val + np.random.uniform(-10, 10), 50, 85
    ), 2)
    
    # 기타 컬럼들 (실제 패턴 기반)
    new_sample['M10AM14A'] = int(np.random.uniform(43, 57))
    new_sample['M14AM10ASUM'] = new_sample['M14AM10A'] + new_sample['M10AM14A']
    new_sample['M14BM14A'] = int(np.random.uniform(107, 133))
    new_sample['M14AM14BSUM'] = new_sample['M14AM14B'] + new_sample['M14BM14A']
    new_sample['M16M14A'] = int(np.random.uniform(133, 158))
    new_sample['M14AM16SUM'] = new_sample['M14AM16'] + new_sample['M16M14A']
    
    return new_sample

def main():
    """메인 실행"""
    print("="*80)
    print("🎯 실제 1700+ 패턴 기반 데이터 생성기")
    print("="*80)
    
    # 원본 데이터 로드
    csv_file = 'Ois.csv'
    print(f"\n📂 데이터 로딩: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"✅ 원본: {len(df):,}행")
    
    # 1700+ 샘플만 추출
    high_risk = df[df['TOTALCNT'] >= 1700].copy()
    print(f"1700+ 샘플: {len(high_risk)}개")
    
    if len(high_risk) == 0:
        print("❌ 1700+ 샘플 없음!")
        return
    
    # 패턴 출력
    print(f"\n📊 실제 1700+ 패턴:")
    print(f"  TOTALCNT: {high_risk['TOTALCNT'].min():.0f} ~ {high_risk['TOTALCNT'].max():.0f}")
    print(f"  M14AM14B: {high_risk['M14AM14B'].min():.0f} ~ {high_risk['M14AM14B'].max():.0f}")
    print(f"  OHT_UTIL: {high_risk['M14.QUE.OHT.OHTUTIL'].min():.1f} ~ {high_risk['M14.QUE.OHT.OHTUTIL'].max():.1f}")
    
    # 합성 데이터 생성
    num_synthetic = 5000  # 대량 생성!
    print(f"\n⚙️ 생성 중: {num_synthetic}개")
    
    synthetic_data = []
    last_time = pd.to_datetime(df['CURRTIME'].iloc[-1], format='%Y%m%d%H%M')
    
    for i in range(num_synthetic):
        # 랜덤하게 실제 1700+ 샘플 선택
        base_sample = high_risk.sample(1).iloc[0]
        
        # 실제 패턴과 동일하게 생성!
        new_sample = generate_realistic_1700_sample(base_sample)
        
        # 시간 설정
        new_time = last_time + timedelta(minutes=i+1)
        new_sample['CURRTIME'] = new_time.strftime('%Y%m%d%H%M')
        new_sample['TIME'] = new_time.strftime('%Y%m%d%H%M')
        
        synthetic_data.append(new_sample)
        
        if (i+1) % 1000 == 0:
            print(f"  진행: {i+1}/{num_synthetic} ({(i+1)/num_synthetic*100:.0f}%)")
    
    print(f"✅ 생성 완료: {len(synthetic_data)}개")
    
    # DataFrame 변환
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # 병합
    print(f"\n🔗 데이터 병합")
    print(f"원본: {len(df):,}개")
    print(f"합성: {len(synthetic_df):,}개")
    
    merged_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    print(f"병합 후: {len(merged_df):,}개")
    print(f"1700+ 샘플: {(merged_df['TOTALCNT'] >= 1700).sum():,}개")
    print(f"1700+ 비율: {(merged_df['TOTALCNT'] >= 1700).sum() / len(merged_df) * 100:.1f}%")
    
    # 저장
    output_file = 'Ois_augmented_v2.csv'
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장: {output_file}")
    
    # 생성된 데이터 통계
    print(f"\n📊 생성된 데이터 통계:")
    print(f"  TOTALCNT: {synthetic_df['TOTALCNT'].min():.0f} ~ {synthetic_df['TOTALCNT'].max():.0f}")
    print(f"  M14AM14B: {synthetic_df['M14AM14B'].min():.0f} ~ {synthetic_df['M14AM14B'].max():.0f}")
    print(f"  OHT_UTIL: {synthetic_df['M14.QUE.OHT.OHTUTIL'].min():.1f} ~ {synthetic_df['M14.QUE.OHT.OHTUTIL'].max():.1f}")
    
    print("\n" + "="*80)
    print("✅ 완료!")
    print("="*80)
    print(f"원본 파일: {csv_file}")
    print(f"증강 파일: {output_file}")
    print(f"\n원본 1700+: {len(high_risk)}개")
    print(f"증강 1700+: {(merged_df['TOTALCNT'] >= 1700).sum():,}개")
    print(f"증가: {len(high_risk)}개 → {(merged_df['TOTALCNT'] >= 1700).sum():,}개")
    print("="*80)

if __name__ == '__main__':
    main()