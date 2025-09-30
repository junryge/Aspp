import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 로드 (여기에 본인의 파일 경로 입력)
# df = pd.read_csv('your_file.csv')

# 응답시간 추출
df['response_time_ms'] = df['_time'].str.extract(r'\.(\d+)\+')[0].astype(int)
df['datetime'] = pd.to_datetime(df['_time'].str.replace(r'\+\d+$', '', regex=True))
df['time_only'] = df['_time'].str[11:19]
df['hour'] = df['datetime'].dt.hour

# M14A와 M16A 분리
m14a = df[df['fabId'] == 'M14A'].copy()
m16a = df[df['fabId'] == 'M16A'].copy()

# 통계 계산
print("="*60)
print("M14A 응답시간 통계")
print("="*60)
print(f"전체 평균: {m14a['response_time_ms'].mean():.2f} ms")
print(f"전체 중앙값: {m14a['response_time_ms'].median():.2f} ms")
print(f"표준편차: {m14a['response_time_ms'].std():.2f} ms")
print(f"최소: {m14a['response_time_ms'].min()} ms")
print(f"최대: {m14a['response_time_ms'].max()} ms")

# 22시대만 필터링
hour_22 = m14a[m14a['hour'] == 22].copy()
print("\n" + "="*60)
print("22시대 M14A 응답시간")
print("="*60)
print(f"22시 평균: {hour_22['response_time_ms'].mean():.2f} ms")
print(f"22시 중앙값: {hour_22['response_time_ms'].median():.2f} ms")

# 이상치 탐지 (평균 대비)
baseline_mean = m14a['response_time_ms'].mean()
baseline_median = m14a['response_time_ms'].median()

# 22시 이상치 찾기
anomalies = hour_22[hour_22['response_time_ms'] > baseline_mean * 2].sort_values('response_time_ms', ascending=False)

print("\n" + "="*60)
print("이상 응답시간 탐지 (평균 대비 2배 이상)")
print("="*60)
for idx, row in anomalies.iterrows():
    ratio_mean = row['response_time_ms'] / baseline_mean
    ratio_median = row['response_time_ms'] / baseline_median
    print(f"{row['time_only']} - {row['response_time_ms']}ms "
          f"(평균 대비 {ratio_mean:.2f}배, 중앙값 대비 {ratio_median:.2f}배)")

# 22:04와 22:32 상세 분석
print("\n" + "="*60)
print("22:04와 22:32 상세 비교")
print("="*60)
target_times = ['22:04:00', '22:32:00']
for time in target_times:
    row = m14a[m14a['time_only'] == time]
    if not row.empty:
        rt = row['response_time_ms'].values[0]
        ratio_mean = rt / baseline_mean
        ratio_median = rt / baseline_median
        print(f"\n{time}:")
        print(f"  응답시간: {rt} ms")
        print(f"  평균({baseline_mean:.2f}ms) 대비: {ratio_mean:.2f}배")
        print(f"  중앙값({baseline_median:.2f}ms) 대비: {ratio_median:.2f}배")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. 전체 응답시간 추이
ax1 = axes[0, 0]
m14a_sorted = m14a.sort_values('datetime')
m16a_sorted = m16a.sort_values('datetime')
ax1.plot(m14a_sorted['datetime'], m14a_sorted['response_time_ms'], 
         label='M14A', alpha=0.7)
ax1.plot(m16a_sorted['datetime'], m16a_sorted['response_time_ms'], 
         label='M16A', alpha=0.7)
ax1.axhline(y=baseline_mean, color='r', linestyle='--', 
            label=f'M14A 평균 ({baseline_mean:.1f}ms)')
ax1.set_title('전체 응답시간 추이')
ax1.set_xlabel('시간')
ax1.set_ylabel('응답시간 (ms)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 22시대만 상세
ax2 = axes[0, 1]
hour_22_sorted = hour_22.sort_values('datetime')
ax2.plot(hour_22_sorted['datetime'], hour_22_sorted['response_time_ms'], 
         'o-', color='red', linewidth=2)
ax2.axhline(y=baseline_mean, color='blue', linestyle='--', 
            label=f'전체 평균 ({baseline_mean:.1f}ms)')
# 이상치 표시
for idx, row in anomalies.iterrows():
    ax2.scatter(row['datetime'], row['response_time_ms'], 
                s=200, color='red', zorder=5)
    ax2.annotate(f"{row['response_time_ms']}ms", 
                xy=(row['datetime'], row['response_time_ms']),
                xytext=(5, 5), textcoords='offset points')
ax2.set_title('22시대 M14A 응답시간 (이상치 강조)')
ax2.set_xlabel('시간')
ax2.set_ylabel('응답시간 (ms)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 히스토그램
ax3 = axes[1, 0]
ax3.hist(m14a['response_time_ms'], bins=30, alpha=0.7, edgecolor='black')
ax3.axvline(x=baseline_mean, color='r', linestyle='--', 
            label=f'평균 ({baseline_mean:.1f}ms)')
ax3.axvline(x=baseline_median, color='g', linestyle='--', 
            label=f'중앙값 ({baseline_median:.1f}ms)')
ax3.set_title('M14A 응답시간 분포')
ax3.set_xlabel('응답시간 (ms)')
ax3.set_ylabel('빈도')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 시간대별 평균
ax4 = axes[1, 1]
hourly_stats = m14a.groupby('hour')['response_time_ms'].agg(['mean', 'max', 'min'])
ax4.plot(hourly_stats.index, hourly_stats['mean'], 'o-', label='평균', linewidth=2)
ax4.fill_between(hourly_stats.index, hourly_stats['min'], hourly_stats['max'], 
                  alpha=0.3, label='최소-최대 범위')
ax4.axhline(y=baseline_mean, color='r', linestyle='--', 
            label=f'전체 평균 ({baseline_mean:.1f}ms)')
ax4.set_title('시간대별 M14A 응답시간')
ax4.set_xlabel('시간(hour)')
ax4.set_ylabel('응답시간 (ms)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 이상치 비율
print("\n" + "="*60)
print("이상치 요약")
print("="*60)
total_count = len(m14a)
anomaly_count = len(m14a[m14a['response_time_ms'] > baseline_mean * 2])
print(f"전체 데이터: {total_count}개")
print(f"이상치(평균의 2배 초과): {anomaly_count}개 ({anomaly_count/total_count*100:.2f}%)")
print(f"22시대 이상치: {len(anomalies)}개")