import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (선택사항)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 데이터 로드
# ============================================================
df = pd.read_csv('/mnt/user-data/uploads/OOOO.CSV')

# 시간 컬럼 변환
df['time'] = pd.to_datetime(df['STAT_DT'], format='%Y%m%d%H%M')
df['hour'] = df['time'].dt.hour

# 모든 수치형 컬럼을 숫자로 변환
for col in df.columns:
    if col not in ['STAT_DT', 'time', 'hour']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================================
# Figure 생성 - 대형 캔버스
# ============================================================
fig = plt.figure(figsize=(24, 20))

# ============================================================
# 그래프 1: 극단적 증가 컬럼들의 전체 추이
# ============================================================
ax1 = plt.subplot(4, 3, 1)
target_cols = ['CURRENT_M16A_3F_JOB_2', 'M16A_6F_TO_HUB_JOB', 'M16A_6F_TO_HUB_JOB_ALT']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for col, color in zip(target_cols, colors):
    plt.plot(df.index, df[col], linewidth=2, label=col, alpha=0.8, color=color)

plt.axhline(y=300, color='red', linestyle='--', linewidth=2, alpha=0.5, label='300 threshold')
plt.axhline(y=611, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Max 611')
plt.axhline(y=1670, color='purple', linestyle=':', linewidth=1.5, alpha=0.7, label='Max 1670')
plt.title('Extreme Spike: Target Columns Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Time Index', fontsize=11)
plt.ylabel('Value', fontsize=11)
plt.legend(fontsize=8, loc='upper left')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1800])

# ============================================================
# 그래프 2: 산점도 - HUB_JOB vs HUB_JOB_ALT
# ============================================================
ax2 = plt.subplot(4, 3, 2)
scatter = plt.scatter(df['M16A_6F_TO_HUB_JOB'], 
                     df['M16A_6F_TO_HUB_JOB_ALT'], 
                     c=df['M16A_3F_STORAGE_UTIL'], 
                     cmap='RdYlGn', 
                     alpha=0.6, 
                     s=40, 
                     edgecolors='black', 
                     linewidth=0.5)
cbar = plt.colorbar(scatter, label='Storage Utilization')
plt.axhline(y=300, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.axvline(x=300, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.title('M16A_6F_TO_HUB_JOB vs ALT\n(r=0.992, color: Storage)', 
         fontsize=12, fontweight='bold')
plt.xlabel('M16A_6F_TO_HUB_JOB', fontsize=11)
plt.ylabel('M16A_6F_TO_HUB_JOB_ALT', fontsize=11)
plt.grid(True, alpha=0.3)

# ============================================================
# 그래프 3: 이중 Y축 - Storage Utilization vs HUB_JOB
# ============================================================
ax3 = plt.subplot(4, 3, 3)
ax3_twin = ax3.twinx()

line1 = ax3.plot(df.index, df['M16A_6F_TO_HUB_JOB'], 
                 color='#4ECDC4', linewidth=2, label='HUB_JOB', alpha=0.8)
line2 = ax3_twin.plot(df.index, df['M16A_3F_STORAGE_UTIL'], 
                      color='#FF6B6B', linewidth=2, 
                      label='Storage Util', alpha=0.8, linestyle='--')

ax3.set_xlabel('Time Index', fontsize=11)
ax3.set_ylabel('HUB_JOB', fontsize=11, color='#4ECDC4')
ax3_twin.set_ylabel('Storage Utilization', fontsize=11, color='#FF6B6B')
ax3.tick_params(axis='y', labelcolor='#4ECDC4')
ax3_twin.tick_params(axis='y', labelcolor='#FF6B6B')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, fontsize=9, loc='upper left')
ax3.set_title('Storage Util vs HUB_JOB\n(Inverse: r=-0.890)', 
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# ============================================================
# 그래프 4: 시간대별 평균 (Area Plot)
# ============================================================
ax4 = plt.subplot(4, 3, 4)
hourly = df.groupby('hour')['CURRENT_M16A_3F_JOB_2'].agg(['mean', 'max', 'min'])
plt.fill_between(hourly.index, hourly['min'], hourly['max'], 
                alpha=0.3, color='#FF6B6B')
plt.plot(hourly.index, hourly['mean'], 'o-', linewidth=3, markersize=8, 
         color='#FF6B6B', label='Hourly Mean', alpha=0.8)
plt.axhline(y=300, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.title('CURRENT_M16A_3F_JOB_2\nHourly Pattern', 
         fontsize=12, fontweight='bold')
plt.xlabel('Hour', fontsize=11)
plt.ylabel('Value', fontsize=11)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))

# ============================================================
# 그래프 5: 시간대별 평균 - HUB_JOB
# ============================================================
ax5 = plt.subplot(4, 3, 5)
hourly = df.groupby('hour')['M16A_6F_TO_HUB_JOB'].agg(['mean', 'max', 'min'])
plt.fill_between(hourly.index, hourly['min'], hourly['max'], 
                alpha=0.3, color='#4ECDC4')
plt.plot(hourly.index, hourly['mean'], 'o-', linewidth=3, markersize=8, 
         color='#4ECDC4', label='Hourly Mean', alpha=0.8)
plt.axhline(y=300, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.title('M16A_6F_TO_HUB_JOB\nHourly Pattern (Max 1670!)', 
         fontsize=12, fontweight='bold')
plt.xlabel('Hour', fontsize=11)
plt.ylabel('Value', fontsize=11)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))

# ============================================================
# 그래프 6: 정규화된 비교 - 용량 vs 작업량
# ============================================================
ax6 = plt.subplot(4, 3, 6)
# 정규화 (0~150 범위)
normalized_hub = (df['M16A_6F_TO_HUB_JOB'] - df['M16A_6F_TO_HUB_JOB'].min()) / \
                 (df['M16A_6F_TO_HUB_JOB'].max() - df['M16A_6F_TO_HUB_JOB'].min()) * 150
normalized_cap = (df['M16A_6F_LFT_MAXCAPA'] - df['M16A_6F_LFT_MAXCAPA'].min()) / \
                 (df['M16A_6F_LFT_MAXCAPA'].max() - df['M16A_6F_LFT_MAXCAPA'].min()) * 150

plt.plot(df.index, normalized_hub, linewidth=2, 
        label='HUB_JOB (norm)', alpha=0.8, color='blue')
plt.plot(df.index, normalized_cap, linewidth=2, 
        label='LFT_MAXCAPA (norm)', alpha=0.8, color='green')
plt.title('Capacity Depletion Pattern\n(r=-0.589)', 
         fontsize=12, fontweight='bold')
plt.xlabel('Time Index', fontsize=11)
plt.ylabel('Normalized Value', fontsize=11)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

# ============================================================
# 그래프 7: 작업 분산 패턴 (산점도)
# ============================================================
ax7 = plt.subplot(4, 3, 7)
plt.scatter(df['M16A_3F_TO_M16A_6F_JOB'], 
           df['M16A_6F_TO_2F_JOB'], 
           c=df['CURRENT_M16A_3F_JOB_2'], 
           cmap='YlOrRd', 
           alpha=0.6, 
           s=50)
cbar = plt.colorbar(label='CURRENT_M16A_3F_JOB_2')
plt.title('Work Distribution: 3F->6F vs 6F->2F\n(color: Current Job)', 
         fontsize=12, fontweight='bold')
plt.xlabel('M16A_3F_TO_M16A_6F_JOB', fontsize=11)
plt.ylabel('M16A_6F_TO_2F_JOB', fontsize=11)
plt.grid(True, alpha=0.3)

# ============================================================
# 그래프 8: M14A 대체 경로 폭발 (이중 Y축)
# ============================================================
ax8 = plt.subplot(4, 3, 8)
ax8_twin = ax8.twinx()

line1 = ax8.plot(df.index, df['M14A_3F_TO_HUB_JOB2'], 
                 color='purple', linewidth=2, 
                 label='M14A_TO_HUB_JOB2', alpha=0.8)
line2 = ax8_twin.plot(df.index, df['M14A_3F_TO_HUB_JOB_ALT'], 
                      color='orange', linewidth=2, 
                      label='M14A_TO_HUB_ALT', alpha=0.8, linestyle='--')

ax8.set_xlabel('Time Index', fontsize=11)
ax8.set_ylabel('JOB2', fontsize=11, color='purple')
ax8_twin.set_ylabel('JOB_ALT', fontsize=11, color='orange')
ax8.tick_params(axis='y', labelcolor='purple')
ax8_twin.tick_params(axis='y', labelcolor='orange')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax8.legend(lines, labels, fontsize=9, loc='upper left')
ax8.set_title('M14A Alternate Route Explosion\n(ALT: +24,817%!)', 
              fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# ============================================================
# 그래프 9: 10F 허브 작업 패턴
# ============================================================
ax9 = plt.subplot(4, 3, 9)
plt.plot(df.index, df['M16B_10F_TO_HUB_JOB'], 
        linewidth=2, alpha=0.8, color='teal')
plt.title('M16B_10F_TO_HUB_JOB\nPattern Over Time', 
         fontsize=12, fontweight='bold')
plt.xlabel('Time Index', fontsize=11)
plt.ylabel('Value', fontsize=11)
plt.grid(True, alpha=0.3)

# ============================================================
# 그래프 10: 상관관계 히트맵
# ============================================================
ax10 = plt.subplot(4, 3, 10)
key_cols = ['CURRENT_M16A_3F_JOB_2', 'M16A_6F_TO_HUB_JOB', 
            'M16A_6F_TO_HUB_JOB_ALT', 'M16A_3F_STORAGE_UTIL', 
            'M16A_6F_LFT_MAXCAPA', 'M14A_3F_TO_HUB_JOB2']
corr_matrix = df[key_cols].corr()

im = ax10.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax10.set_xticks(range(len(key_cols)))
ax10.set_yticks(range(len(key_cols)))
ax10.set_xticklabels([c[:20] for c in key_cols], rotation=45, ha='right', fontsize=8)
ax10.set_yticklabels([c[:20] for c in key_cols], fontsize=8)

# 숫자 표시
for i in range(len(key_cols)):
    for j in range(len(key_cols)):
        text = ax10.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=9)
                        
plt.colorbar(im, ax=ax10)
ax10.set_title('Key Columns Correlation Heatmap', 
              fontsize=12, fontweight='bold')

# ============================================================
# 그래프 11: 300+ 이벤트 분포 (막대 그래프)
# ============================================================
ax11 = plt.subplot(4, 3, 11)
spike_times = []
for col in target_cols:
    spikes = df[df[col] >= 300].index.tolist()
    spike_times.append(len(spikes))

bars = plt.bar(range(len(target_cols)), spike_times, 
              alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.xticks(range(len(target_cols)), 
          [c[:20] + '...' for c in target_cols], 
          rotation=15, ha='right', fontsize=9)
plt.ylabel('Count of 300+ Events', fontsize=11)
plt.title('300+ Events Distribution', fontsize=12, fontweight='bold')

# 값 표시
for i, (bar, val) in enumerate(zip(bars, spike_times)):
    plt.text(bar.get_x() + bar.get_width()/2, 
            bar.get_height() + 10, 
            f'{val}\n({val/len(df)*100:.1f}%)', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')
            
plt.grid(True, alpha=0.3, axis='y')

# ============================================================
# 그래프 12: 전체 합계 추이
# ============================================================
ax12 = plt.subplot(4, 3, 12)
df['TOTAL_TARGET'] = df['CURRENT_M16A_3F_JOB_2'] + \
                     df['M16A_6F_TO_HUB_JOB'] + \
                     df['M16A_6F_TO_HUB_JOB_ALT']

plt.plot(df.index, df['TOTAL_TARGET'], 
        linewidth=2, alpha=0.8, color='#E74C3C')
plt.axhline(y=df['TOTAL_TARGET'].mean(), 
           color='green', linestyle='--', linewidth=2, 
           alpha=0.7, label=f'Mean: {df["TOTAL_TARGET"].mean():.0f}')
plt.axhline(y=df['TOTAL_TARGET'].max(), 
           color='red', linestyle=':', linewidth=2, 
           alpha=0.7, label=f'Max: {df["TOTAL_TARGET"].max():.0f}')
plt.title('Total Workload of 3 Key Columns', 
         fontsize=12, fontweight='bold')
plt.xlabel('Time Index', fontsize=11)
plt.ylabel('Total Value', fontsize=11)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

# ============================================================
# 저장
# ============================================================
plt.tight_layout()
plt.savefig('output_visualization.png', dpi=150, bbox_inches='tight')
print("✅ 시각화 완료: output_visualization.png")
print(f"   총 {len(df)} 데이터 포인트 분석")
print(f"   12개 그래프 생성")