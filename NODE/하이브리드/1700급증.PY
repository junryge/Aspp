"""
TOTALCNT 급증 케이스 분석
과거 20분 동안 1700 이하 → 10분 후 1700개 이상 급증 케이스 추출
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 1. 데이터 로드
# ============================================
print("="*60)
print("🔍 TOTALCNT 급증 케이스 분석 (1700개 기준)")
print("="*60)

# CSV 파일 읽기
print("\n📂 데이터 로딩 중...")
df = pd.read_csv('20250807_DATA.CSV')
print(f"✅ 데이터 로드 완료: {len(df):,}행")

# 시간 변환
df['datetime'] = pd.to_datetime(df['TIME'], format='%Y%m%d%H%M')
df = df.sort_values('datetime').reset_index(drop=True)

print(f"\n📅 데이터 정보:")
print(f"  기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"  TOTALCNT 범위: {df['TOTALCNT'].min()} ~ {df['TOTALCNT'].max()}")

# ============================================
# 2. 급증 케이스 찾기
# ============================================
print("\n🔍 급증 케이스 검색 중...")
print("  조건: 과거 20분 모두 < 1700 AND 10분 후 >= 1700")

spike_cases = []

# 데이터가 1분 단위인지 확인
time_diff = (df['datetime'].iloc[1] - df['datetime'].iloc[0]).total_seconds() / 60
print(f"  데이터 간격: {time_diff}분")

# 각 시점 검사
for i in range(20, len(df) - 10):
    # 과거 20분 (현재 포함 21개 데이터)
    past_20min_values = df.iloc[i-20:i+1]['TOTALCNT'].values
    
    # 10분 후
    future_value = df.iloc[i+10]['TOTALCNT']
    
    # 조건 체크
    if all(past_20min_values < 1700) and future_value >= 1700:
        spike_cases.append({
            'index': i,
            'time': df.iloc[i]['datetime'],
            'current_TOTALCNT': df.iloc[i]['TOTALCNT'],
            'past_20min_max': max(past_20min_values),
            'past_20min_min': min(past_20min_values),
            'past_20min_avg': np.mean(past_20min_values),
            'future_10min_TOTALCNT': future_value,
            'spike_amount': future_value - df.iloc[i]['TOTALCNT'],
            'spike_rate': ((future_value - df.iloc[i]['TOTALCNT']) / df.iloc[i]['TOTALCNT'] * 100),
            'M14AM14B': df.iloc[i]['M14AM14B'],
            'M14AM10A': df.iloc[i]['M14AM10A'],
            'M14AM16': df.iloc[i]['M14AM16'],
            'M14BM14A': df.iloc[i]['M14BM14A'],
            'M16M14A': df.iloc[i]['M16M14A']
        })

# ============================================
# 3. 결과 출력
# ============================================
print(f"\n✅ 검색 완료!")
print(f"🎯 총 급증 케이스: {len(spike_cases)}건")

if spike_cases:
    spike_df = pd.DataFrame(spike_cases)
    
    print("\n📊 급증 통계:")
    print(f"  평균 급증량: {spike_df['spike_amount'].mean():.0f}개")
    print(f"  최대 급증량: {spike_df['spike_amount'].max():.0f}개")
    print(f"  평균 급증률: {spike_df['spike_rate'].mean():.1f}%")
    print(f"  최대 급증률: {spike_df['spike_rate'].max():.1f}%")
    
    # ============================================
    # 4. 상위 10개 케이스 상세
    # ============================================
    print("\n🔝 급증량 상위 10개 케이스:")
    print("-"*120)
    print(f"{'No':>3} | {'시간':^20} | {'현재':>6} | {'20분최대':>8} | {'10분후':>7} | {'급증량':>7} | {'급증률':>7} | {'M14AM14B':>8} | {'M14AM10A':>8}")
    print("-"*120)
    
    top_10 = spike_df.nlargest(10, 'spike_amount')
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{idx:3d} | {row['time'].strftime('%Y-%m-%d %H:%M')} | "
              f"{row['current_TOTALCNT']:6,d} | {row['past_20min_max']:8,d} | "
              f"{row['future_10min_TOTALCNT']:7,d} | {row['spike_amount']:+7,d} | "
              f"{row['spike_rate']:6.1f}% | {row['M14AM14B']:8.0f} | {row['M14AM10A']:8.0f}")
    
    # ============================================
    # 5. M14 컬럼 분석
    # ============================================
    print("\n📊 급증 시점의 M14 컬럼 특성:")
    print(f"  M14AM14B 평균: {spike_df['M14AM14B'].mean():.1f} (전체 평균: {df['M14AM14B'].mean():.1f})")
    print(f"  M14AM10A 평균: {spike_df['M14AM10A'].mean():.1f} (전체 평균: {df['M14AM10A'].mean():.1f})")
    
    # M14AM14B 구간별 분포
    print("\n  M14AM14B 구간별 급증 케이스:")
    for low, high in [(0, 300), (300, 350), (350, 400), (400, 450), (450, 1000)]:
        count = ((spike_df['M14AM14B'] >= low) & (spike_df['M14AM14B'] < high)).sum()
        if count > 0:
            print(f"    {low:3d}~{high:3d}: {count:3d}건 ({count/len(spike_df)*100:5.1f}%)")
    
    # M14B/M14A 비율 분석
    spike_df['ratio_M14B_M14A'] = spike_df['M14AM14B'] / spike_df['M14AM10A'].clip(lower=1)
    print(f"\n  M14B/M14A 비율 평균: {spike_df['ratio_M14B_M14A'].mean():.2f}")
    
    # ============================================
    # 6. 시각화
    # ============================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('TOTALCNT 급증 패턴 분석 (과거 20분 <1700 → 10분 후 ≥1700)', fontsize=16)
    
    # 1. 급증량 히스토그램
    ax = axes[0, 0]
    ax.hist(spike_df['spike_amount'], bins=15, edgecolor='black', color='skyblue')
    ax.axvline(spike_df['spike_amount'].mean(), color='red', linestyle='--', 
               label=f'평균: {spike_df["spike_amount"].mean():.0f}')
    ax.set_title('급증량 분포')
    ax.set_xlabel('급증량 (개)')
    ax.set_ylabel('빈도')
    ax.legend()
    
    # 2. M14AM14B vs 급증량 산점도
    ax = axes[0, 1]
    scatter = ax.scatter(spike_df['M14AM14B'], spike_df['spike_amount'], 
                        c=spike_df['spike_rate'], cmap='Reds', alpha=0.7)
    ax.set_title('M14AM14B vs 급증량')
    ax.set_xlabel('M14AM14B')
    ax.set_ylabel('급증량 (개)')
    plt.colorbar(scatter, ax=ax, label='급증률(%)')
    
    # 3. 시간대별 발생 빈도
    ax = axes[0, 2]
    spike_df['hour'] = spike_df['time'].dt.hour
    hour_counts = spike_df['hour'].value_counts().sort_index()
    ax.bar(hour_counts.index, hour_counts.values, color='orange', edgecolor='black')
    ax.set_title('시간대별 급증 발생')
    ax.set_xlabel('시간대')
    ax.set_ylabel('발생 횟수')
    ax.set_xticks(range(0, 24, 2))
    
    # 4. M14B/M14A 비율 vs 급증률
    ax = axes[1, 0]
    ax.scatter(spike_df['ratio_M14B_M14A'], spike_df['spike_rate'], alpha=0.7, color='green')
    ax.axvline(4, color='red', linestyle='--', label='비율=4')
    ax.set_title('M14B/M14A 비율 vs 급증률')
    ax.set_xlabel('M14AM14B / M14AM10A')
    ax.set_ylabel('급증률 (%)')
    ax.legend()
    
    # 5. 현재값 vs 10분후값
    ax = axes[1, 1]
    ax.scatter(spike_df['current_TOTALCNT'], spike_df['future_10min_TOTALCNT'], alpha=0.7)
    ax.plot([1000, 1700], [1700, 1700], 'r--', label='y=1700')
    ax.plot([1000, 1700], [1000, 1700], 'k--', alpha=0.3, label='y=x')
    ax.set_title('현재 TOTALCNT vs 10분 후 TOTALCNT')
    ax.set_xlabel('현재 TOTALCNT')
    ax.set_ylabel('10분 후 TOTALCNT')
    ax.legend()
    
    # 6. 급증 케이스 시계열
    ax = axes[1, 2]
    ax.plot(spike_df['time'], spike_df['spike_amount'], 'o-', markersize=4)
    ax.set_title('급증 케이스 시계열')
    ax.set_xlabel('시간')
    ax.set_ylabel('급증량')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # ============================================
    # 7. 결과 저장
    # ============================================
    output_file = 'TOTALCNT_spike_1700_cases.csv'
    spike_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장: {output_file}")
    
    # ============================================
    # 8. 예측 가능성 분석
    # ============================================
    print("\n🎯 예측 가능성 분석:")
    
    # 황금 패턴 분석
    golden_pattern = spike_df[(spike_df['M14AM14B'] >= 300) & (spike_df['M14AM10A'] < 80)]
    print(f"\n  황금 패턴 (M14AM14B≥300 & M14AM10A<80): {len(golden_pattern)}건 ({len(golden_pattern)/len(spike_df)*100:.1f}%)")
    
    # M14AM14B 임계값별 포착률
    print("\n  M14AM14B 임계값별 포착률:")
    for threshold in [250, 300, 350, 400]:
        caught = (spike_df['M14AM14B'] >= threshold).sum()
        print(f"    M14AM14B >= {threshold}: {caught}건 ({caught/len(spike_df)*100:.1f}%)")
    
    # 비율 기반 포착률
    print("\n  M14B/M14A 비율별 포착률:")
    for ratio in [3, 4, 5, 6]:
        caught = (spike_df['ratio_M14B_M14A'] >= ratio).sum()
        print(f"    비율 >= {ratio}: {caught}건 ({caught/len(spike_df)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ 분석 완료!")
    print(f"   총 {len(spike_cases)}건의 급증 케이스 발견")
    print(f"   결과 파일: {output_file}")
    print("="*60)
    
else:
    print("\n⚠️ 조건에 맞는 급증 케이스가 없습니다.")
    print("   (과거 20분 모두 < 1700 → 10분 후 >= 1700)")