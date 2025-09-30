import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 읽기
df = pd.read_csv('RB2.CSV', names=['timestamp', 'description', 'metric', 'latency', 'save_time'])

# 타임스탬프를 datetime으로 변환
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['latency'] = df['latency'].astype(float)

# 시간순 정렬
df = df.sort_values('timestamp')

# 통계 계산
avg_latency = df['latency'].mean()
max_latency = df['latency'].max()
max_idx = df['latency'].idxmax()
max_time = df.loc[max_idx, 'timestamp']

print(f"=== 9월 16일 M16HUB 지연시간 분석 ===")
print(f"평균 지연시간: {avg_latency:.3f}ms")
print(f"최소 지연시간: {df['latency'].min():.3f}ms")
print(f"최대 지연시간: {max_latency:.3f}ms ({max_time.strftime('%H:%M')})")
print(f"표준편차: {df['latency'].std():.3f}ms")
print()

# 고지연 구간 분석
spike_threshold = 8.0
spike_df = df[df['latency'] > spike_threshold]
print(f"고지연 구간 (>{spike_threshold}ms): {len(spike_df)}개 샘플")
if len(spike_df) > 0:
    print(f"시작 시간: {spike_df['timestamp'].min().strftime('%H:%M')}")
    print(f"종료 시간: {spike_df['timestamp'].max().strftime('%H:%M')}")
    print(f"지속 시간: {(spike_df['timestamp'].max() - spike_df['timestamp'].min()).total_seconds() / 60:.1f}분")

# 그래프 생성
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 메인 그래프
ax1.plot(df['timestamp'], df['latency'], linewidth=1.5, color='#3b82f6', label='응답시간')
ax1.fill_between(df['timestamp'], df['latency'], alpha=0.2, color='#3b82f6')

# 평균선
ax1.axhline(y=avg_latency, color='#2563eb', linestyle='--', linewidth=1, 
            label=f'평균 {avg_latency:.3f}ms')

# 임계값선
ax1.axhline(y=spike_threshold, color='#dc2626', linestyle='--', linewidth=1, 
            label=f'임계값 {spike_threshold}ms')

# 최대값 표시
ax1.scatter([max_time], [max_latency], color='red', s=100, zorder=5, 
            label=f'최대값 {max_latency:.3f}ms')

# 고지연 구간 강조
if len(spike_df) > 0:
    ax1.fill_between(spike_df['timestamp'], 0, spike_df['latency'], 
                     alpha=0.3, color='red', label='고지연 구간')

ax1.set_xlabel('시간', fontsize=12)
ax1.set_ylabel('응답시간 (ms)', fontsize=12)
ax1.set_title('M16HUB 평균 응답시간 - 2025년 9월 16일', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 시간대별 평균 (서브 그래프)
df['hour'] = df['timestamp'].dt.hour
hourly_avg = df.groupby('hour')['latency'].agg(['mean', 'min', 'max']).reset_index()

bars = ax2.bar(hourly_avg['hour'], hourly_avg['mean'], color='#3b82f6', alpha=0.7, 
               label='시간대별 평균')
ax2.errorbar(hourly_avg['hour'], hourly_avg['mean'], 
             yerr=[hourly_avg['mean'] - hourly_avg['min'], 
                   hourly_avg['max'] - hourly_avg['mean']], 
             fmt='none', color='gray', alpha=0.5, capsize=3)

# 고지연 시간대 강조
for idx, row in hourly_avg.iterrows():
    if row['mean'] > spike_threshold:
        bars[idx].set_color('#dc2626')

ax2.axhline(y=avg_latency, color='#2563eb', linestyle='--', linewidth=1, 
            label=f'전체 평균')
ax2.set_xlabel('시간', fontsize=12)
ax2.set_ylabel('평균 응답시간 (ms)', fontsize=12)
ax2.set_title('시간대별 평균 응답시간', fontsize=12, fontweight='bold')
ax2.set_xticks(range(0, 24))
ax2.set_xticklabels([f'{h:02d}:00' for h in range(0, 24)], rotation=45, ha='right')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('m16hub_latency_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n그래프가 'm16hub_latency_analysis.png'로 저장되었습니다.")
plt.show()

# 추가 분석: 문제 구간 상세
print("\n=== 주요 이슈 구간 상세 분석 ===")
critical_start = pd.to_datetime('2025-09-16 20:47:00')
critical_end = pd.to_datetime('2025-09-16 21:12:00')
critical_df = df[(df['timestamp'] >= critical_start) & (df['timestamp'] <= critical_end)]

if len(critical_df) > 0:
    print(f"이슈 구간: {critical_start.strftime('%H:%M')} - {critical_end.strftime('%H:%M')}")
    print(f"평균 지연시간: {critical_df['latency'].mean():.3f}ms")
    print(f"최대 지연시간: {critical_df['latency'].max():.3f}ms")
    print(f"정상 대비 증가율: {(critical_df['latency'].mean() / avg_latency - 1) * 100:.1f}%")