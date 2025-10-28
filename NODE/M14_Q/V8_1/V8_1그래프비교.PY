# -*- coding: utf-8 -*-
"""
280분 → 10분/15분/25분 후 예측 결과 비교 그래프
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_evaluation_results():
    """평가 결과 CSV 로드"""
    results = {}
    
    files = {
        '10min': 'evaluation_280to10_1year.csv',
        '15min': 'evaluation_280to15_1year.csv',
        '25min': 'evaluation_280to25_1year.csv'
    }
    
    print("="*80)
    print("📊 평가 결과 파일 로딩")
    print("="*80)
    
    for key, filename in files.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            results[key] = df
            print(f"✅ {key} 로드: {len(df):,}개 행")
        else:
            print(f"⚠️ {filename} 파일 없음")
    
    if len(results) == 0:
        print("\n❌ 평가 결과 파일이 없습니다!")
        print("먼저 평가 코드를 실행하세요:")
        print("  - python eval_280to10_1YEAR.py")
        print("  - python eval_280to15_1YEAR.py")
        print("  - python eval_280to25_1YEAR.py")
        return None
    
    return results

def create_comparison_graphs(results):
    """비교 그래프 6개 생성"""
    
    fig = plt.figure(figsize=(20, 14))
    
    colors = {
        '10min': '#2E86AB',  # 파랑
        '15min': '#A23B72',  # 보라
        '25min': '#F18F01'   # 주황
    }
    
    labels_kr = {
        '10min': '10분 후 예측',
        '15min': '15분 후 예측',
        '25min': '25분 후 예측'
    }
    
    # ===== 1. MAE 분포 히스토그램 =====
    ax1 = plt.subplot(3, 3, 1)
    for key in ['10min', '15min', '25min']:
        if key in results:
            df = results[key]
            ax1.hist(df['절대오차'], bins=50, alpha=0.6, 
                   label=f"{labels_kr[key]}\n(MAE: {df['절대오차'].mean():.2f})",
                   color=colors[key], edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('1) MAE Distribution', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 200)
    
    # ===== 2. MAE 박스플롯 =====
    ax2 = plt.subplot(3, 3, 2)
    mae_data = []
    mae_labels = []
    mae_colors = []
    
    for key in ['10min', '15min', '25min']:
        if key in results:
            mae_data.append(results[key]['절대오차'])
            mae_labels.append(labels_kr[key])
            mae_colors.append(colors[key])
    
    bp = ax2.boxplot(mae_data, labels=mae_labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], mae_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    ax2.set_title('2) MAE Box Plot', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 평균값 표시
    for i, key in enumerate(['10min', '15min', '25min']):
        if key in results:
            mean_val = results[key]['절대오차'].mean()
            ax2.text(i+1, mean_val, f'{mean_val:.1f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # ===== 3. 평균 MAE 막대 그래프 =====
    ax3 = plt.subplot(3, 3, 3)
    mae_means = []
    bar_labels = []
    bar_colors = []
    
    for key in ['10min', '15min', '25min']:
        if key in results:
            mae_means.append(results[key]['절대오차'].mean())
            bar_labels.append(labels_kr[key])
            bar_colors.append(colors[key])
    
    bars = ax3.bar(range(len(mae_means)), mae_means, color=bar_colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(bar_labels)))
    ax3.set_xticklabels(bar_labels, fontsize=10)
    ax3.set_ylabel('Average MAE', fontsize=11, fontweight='bold')
    ax3.set_title('3) Average MAE Comparison', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # ===== 4. 실제값 vs 예측값 산점도 =====
    ax4 = plt.subplot(3, 3, 4)
    
    for key in ['10min', '15min', '25min']:
        if key in results:
            df = results[key].sample(min(3000, len(results[key])), random_state=42)
            ax4.scatter(df['실제값'], df['예측값'], 
                      alpha=0.4, s=5, 
                      label=labels_kr[key],
                      color=colors[key])
    
    # 완벽한 예측선
    min_val = min([results[k]['실제값'].min() for k in results])
    max_val = max([results[k]['실제값'].max() for k in results])
    ax4.plot([min_val, max_val], [min_val, max_val], 
           'r--', linewidth=2, label='Perfect', alpha=0.8)
    
    ax4.set_xlabel('Actual TOTALCNT', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Predicted TOTALCNT', fontsize=11, fontweight='bold')
    ax4.set_title('4) Actual vs Predicted (3K samples)', fontsize=12, fontweight='bold', pad=10)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ===== 5. 위험 구간(1700+) 감지율 =====
    ax5 = plt.subplot(3, 3, 5)
    
    detection_data = {}
    for key in ['10min', '15min', '25min']:
        if key in results:
            df = results[key]
            actual_danger = (df['실제위험(1700+)'] == 'O').sum()
            detected = ((df['실제위험(1700+)'] == 'O') & (df['예측위험(1650+)'] == 'O')).sum()
            
            if actual_danger > 0:
                rate = detected / actual_danger * 100
                detection_data[key] = {
                    'rate': rate,
                    'actual': actual_danger,
                    'detected': detected
                }
    
    if detection_data:
        x_pos = range(len(detection_data))
        rates = [detection_data[k]['rate'] for k in ['10min', '15min', '25min'] if k in detection_data]
        bar_labels = [labels_kr[k] for k in ['10min', '15min', '25min'] if k in detection_data]
        bar_colors = [colors[k] for k in ['10min', '15min', '25min'] if k in detection_data]
        
        bars = ax5.bar(x_pos, rates, color=bar_colors, alpha=0.7, edgecolor='black')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(bar_labels, fontsize=10)
        ax5.set_ylabel('Detection Rate (%)', fontsize=11, fontweight='bold')
        ax5.set_title('5) Danger Zone (1700+) Detection', fontsize=12, fontweight='bold', pad=10)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim(0, 100)
        
        # 값 표시
        for bar, key in zip(bars, detection_data.keys()):
            height = bar.get_height()
            det = detection_data[key]['detected']
            act = detection_data[key]['actual']
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%\n({det}/{act})',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # ===== 6. 오차율(%) 분포 =====
    ax6 = plt.subplot(3, 3, 6)
    for key in ['10min', '15min', '25min']:
        if key in results:
            df = results[key]
            error_rate = df['오차율(%)']
            error_rate_filtered = error_rate[error_rate <= 20]  # 20% 이하만
            ax6.hist(error_rate_filtered, bins=50, alpha=0.6, 
                   label=f"{labels_kr[key]}\n(평균: {error_rate.mean():.2f}%)",
                   color=colors[key], edgecolor='black', linewidth=0.5)
    
    ax6.set_xlabel('Error Rate (%)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('6) Error Rate Distribution (<20%)', fontsize=12, fontweight='bold', pad=10)
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # ===== 7. M14AM14B 구간별 MAE =====
    ax7 = plt.subplot(3, 3, 7)
    
    m14b_ranges = [
        ('200-250', 200, 250),
        ('250-300', 250, 300),
        ('300-350', 300, 350),
        ('350-400', 350, 400),
        ('400-450', 400, 450),
        ('450+', 450, 9999)
    ]
    
    width = 0.25
    x = np.arange(len(m14b_ranges))
    
    for idx, key in enumerate(['10min', '15min', '25min']):
        if key in results:
            df = results[key]
            mae_by_range = []
            
            for _, low, high in m14b_ranges:
                mask = (df['M14AM14B'] >= low) & (df['M14AM14B'] < high)
                if mask.sum() > 0:
                    mae_by_range.append(df[mask]['절대오차'].mean())
                else:
                    mae_by_range.append(0)
            
            offset = (idx - 1) * width
            ax7.bar(x + offset, mae_by_range, width, 
                   label=labels_kr[key], 
                   color=colors[key], alpha=0.7, edgecolor='black')
    
    ax7.set_xlabel('M14AM14B Range', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Average MAE', fontsize=11, fontweight='bold')
    ax7.set_title('7) MAE by M14AM14B Range', fontsize=12, fontweight='bold', pad=10)
    ax7.set_xticks(x)
    ax7.set_xticklabels([r[0] for r in m14b_ranges], rotation=45, ha='right', fontsize=9)
    ax7.legend(loc='upper left', fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # ===== 8. 시간대별 예측 추이 (샘플 200개) =====
    ax8 = plt.subplot(3, 3, 8)
    
    sample_size = 200
    for key in ['10min', '15min', '25min']:
        if key in results:
            df = results[key].iloc[:sample_size]
            ax8.plot(range(len(df)), df['실제값'], 
                    label=f'{labels_kr[key]} (실제)', 
                    color=colors[key], alpha=0.4, linewidth=1)
            ax8.plot(range(len(df)), df['예측값'], 
                    label=f'{labels_kr[key]} (예측)', 
                    color=colors[key], alpha=0.8, linewidth=2, linestyle='--')
    
    ax8.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax8.set_ylabel('TOTALCNT', fontsize=11, fontweight='bold')
    ax8.set_title('8) Prediction Trend (First 200)', fontsize=12, fontweight='bold', pad=10)
    ax8.legend(loc='upper left', fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    # ===== 9. 종합 성능 비교 표 =====
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # 표 데이터 생성
    table_data = []
    table_data.append(['Metric', '10분', '15분', '25분'])
    
    # MAE
    mae_row = ['MAE']
    for key in ['10min', '15min', '25min']:
        if key in results:
            mae_row.append(f"{results[key]['절대오차'].mean():.2f}")
        else:
            mae_row.append('-')
    table_data.append(mae_row)
    
    # 오차율
    err_row = ['Error Rate (%)']
    for key in ['10min', '15min', '25min']:
        if key in results:
            err_row.append(f"{results[key]['오차율(%)'].mean():.2f}%")
        else:
            err_row.append('-')
    table_data.append(err_row)
    
    # 위험 감지율
    det_row = ['Detection Rate']
    for key in ['10min', '15min', '25min']:
        if key in results and key in detection_data:
            det_row.append(f"{detection_data[key]['rate']:.1f}%")
        else:
            det_row.append('-')
    table_data.append(det_row)
    
    # 샘플 수
    count_row = ['Sample Count']
    for key in ['10min', '15min', '25min']:
        if key in results:
            count_row.append(f"{len(results[key]):,}")
        else:
            count_row.append('-')
    table_data.append(count_row)
    
    # 표 생성
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 헤더 스타일
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 데이터 행 색상
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#E8F5E9')
        for j in range(1, 4):
            if j == 1:
                table[(i, j)].set_facecolor('#E3F2FD')
            elif j == 2:
                table[(i, j)].set_facecolor('#F3E5F5')
            else:
                table[(i, j)].set_facecolor('#FFF3E0')
    
    ax9.set_title('9) Performance Summary', fontsize=12, fontweight='bold', pad=10)
    
    # 전체 타이틀
    fig.suptitle('280min Sequence Prediction Comparison: 10min vs 15min vs 25min', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 저장
    output_file = 'prediction_comparison_10_15_25min.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 그래프 저장: {output_file}")
    
    plt.show()
    
    return fig

def print_summary_statistics(results):
    """통계 요약 출력"""
    print("\n" + "="*80)
    print("📊 최종 성능 비교 요약")
    print("="*80)
    
    for key in ['10min', '15min', '25min']:
        if key in results:
            df = results[key]
            print(f"\n【{key} 예측】")
            print(f"  샘플 수: {len(df):,}개")
            print(f"  평균 MAE: {df['절대오차'].mean():.2f}")
            print(f"  중앙 MAE: {df['절대오차'].median():.2f}")
            print(f"  최대 오차: {df['절대오차'].max():.2f}")
            print(f"  평균 오차율: {df['오차율(%)'].mean():.2f}%")
            
            # 위험 구간
            actual_danger = (df['실제위험(1700+)'] == 'O').sum()
            detected = ((df['실제위험(1700+)'] == 'O') & (df['예측위험(1650+)'] == 'O')).sum()
            if actual_danger > 0:
                print(f"  위험 감지율: {detected/actual_danger*100:.1f}% ({detected}/{actual_danger})")

if __name__ == '__main__':
    print("\n🚀 10분/15분/25분 예측 결과 비교 그래프 생성\n")
    
    # 결과 로드
    results = load_evaluation_results()
    
    if results is None or len(results) == 0:
        print("\n❌ 평가 결과 파일이 없어서 그래프를 생성할 수 없습니다.")
        exit(1)
    
    # 통계 출력
    print_summary_statistics(results)
    
    # 그래프 생성
    print("\n" + "="*80)
    print("📊 그래프 생성 중...")
    print("="*80)
    
    fig = create_comparison_graphs(results)
    
    print("\n✅ 완료!")
    print("="*80)