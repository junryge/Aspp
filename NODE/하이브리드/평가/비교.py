"""
평가 결과 요약 및 비교표 생성
============================
실제값, 예측구형(LSTM), 예측신형(앙상블) 비교표를 생성합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def create_comparison_summary(times, actuals, pred_old, pred_new, save_path='comparison_summary.xlsx'):
    """
    실제값과 예측값 비교 요약표 생성
    
    Parameters:
    -----------
    times : array-like
        예측 시점
    actuals : array-like
        실제 물류량
    pred_old : array-like
        LSTM 구형 예측값
    pred_new : array-like
        앙상블 신형 예측값
    save_path : str
        저장 경로
    """
    
    # DataFrame 생성
    df = pd.DataFrame({
        '시간': times,
        '실제값': actuals.round().astype(int),
        '예측_구형(LSTM)': pred_old.round().astype(int),
        '예측_신형(앙상블)': pred_new.round().astype(int)
    })
    
    # 오차 계산
    df['오차_구형'] = np.abs(df['실제값'] - df['예측_구형(LSTM)'])
    df['오차_신형'] = np.abs(df['실제값'] - df['예측_신형(앙상블)'])
    
    # 오차율 계산 (%)
    df['오차율_구형(%)'] = (df['오차_구형'] / df['실제값'] * 100).round(2)
    df['오차율_신형(%)'] = (df['오차_신형'] / df['실제값'] * 100).round(2)
    
    # 더 정확한 모델 표시
    df['더_정확한_모델'] = df.apply(
        lambda x: '신형' if x['오차_신형'] < x['오차_구형'] else 
                  ('구형' if x['오차_신형'] > x['오차_구형'] else '동일'), 
        axis=1
    )
    
    # Excel Writer 생성
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        # 전체 데이터
        df.to_excel(writer, sheet_name='전체_비교', index=False)
        
        # 요약 통계
        summary_stats = pd.DataFrame({
            '지표': ['평균 오차', '최대 오차', '최소 오차', '평균 오차율(%)', 'RMSE'],
            'LSTM 구형': [
                df['오차_구형'].mean().round(2),
                df['오차_구형'].max(),
                df['오차_구형'].min(),
                df['오차율_구형(%)'].mean().round(2),
                np.sqrt(np.mean(df['오차_구형']**2)).round(2)
            ],
            '앙상블 신형': [
                df['오차_신형'].mean().round(2),
                df['오차_신형'].max(),
                df['오차_신형'].min(),
                df['오차율_신형(%)'].mean().round(2),
                np.sqrt(np.mean(df['오차_신형']**2)).round(2)
            ]
        })
        
        # 개선율 추가
        summary_stats['개선율(%)'] = ((summary_stats['LSTM 구형'] - summary_stats['앙상블 신형']) / 
                                    summary_stats['LSTM 구형'] * 100).round(2)
        
        summary_stats.to_excel(writer, sheet_name='요약_통계', index=False)
        
        # 시간대별 분석
        df['시간대'] = pd.to_datetime(df['시간']).dt.hour
        hourly_stats = df.groupby('시간대').agg({
            '오차_구형': 'mean',
            '오차_신형': 'mean',
            '실제값': 'mean'
        }).round(2)
        hourly_stats.to_excel(writer, sheet_name='시간대별_분석')
        
        # 상위 오차 케이스
        top_errors_old = df.nlargest(20, '오차_구형')[['시간', '실제값', '예측_구형(LSTM)', '오차_구형', '오차율_구형(%)']]
        top_errors_new = df.nlargest(20, '오차_신형')[['시간', '실제값', '예측_신형(앙상블)', '오차_신형', '오차율_신형(%)']]
        
        top_errors_old.to_excel(writer, sheet_name='구형_최대오차_TOP20', index=False)
        top_errors_new.to_excel(writer, sheet_name='신형_최대오차_TOP20', index=False)
    
    print(f"\n비교 요약표가 저장되었습니다: {save_path}")
    
    # 콘솔 출력
    print("\n" + "="*80)
    print("모델 성능 비교 요약")
    print("="*80)
    print(summary_stats.to_string(index=False))
    
    print("\n" + "="*80)
    print("최근 20개 예측 결과 비교")
    print("="*80)
    recent_df = df.tail(20)
    print(f"{'시간':^20} | {'실제':^8} | {'구형':^8} | {'신형':^8} | {'구형오차':^8} | {'신형오차':^8} | {'더정확':^8}")
    print("-" * 80)
    
    for _, row in recent_df.iterrows():
        time_str = pd.to_datetime(row['시간']).strftime('%m-%d %H:%M')
        print(f"{time_str:^20} | {row['실제값']:^8} | {row['예측_구형(LSTM)']:^8} | "
              f"{row['예측_신형(앙상블)']:^8} | {row['오차_구형']:^8} | {row['오차_신형']:^8} | {row['더_정확한_모델']:^8}")
    
    # 승률 계산
    win_counts = df['더_정확한_모델'].value_counts()
    total = len(df)
    
    print("\n" + "="*80)
    print("모델별 예측 정확도 승률")
    print("="*80)
    for model, count in win_counts.items():
        percentage = count / total * 100
        print(f"{model}: {count}건 ({percentage:.1f}%)")
    
    return df, summary_stats

def plot_detailed_comparison(df, save_path='detailed_comparison.png'):
    """상세 비교 차트 생성"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. 시계열 비교 (최근 100개)
    ax = axes[0, 0]
    recent = df.tail(100)
    ax.plot(range(len(recent)), recent['실제값'], 'k-', linewidth=2, label='실제값')
    ax.plot(range(len(recent)), recent['예측_구형(LSTM)'], 'r--', alpha=0.7, label='LSTM 구형')
    ax.plot(range(len(recent)), recent['예측_신형(앙상블)'], 'b-', alpha=0.7, label='앙상블 신형')
    ax.set_title('최근 100개 예측 비교', fontsize=14)
    ax.set_xlabel('시간 인덱스')
    ax.set_ylabel('물류량')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 오차 분포 비교
    ax = axes[0, 1]
    data_to_plot = [df['오차_구형'], df['오차_신형']]
    ax.boxplot(data_to_plot, labels=['LSTM 구형', '앙상블 신형'])
    ax.set_title('오차 분포 비교 (Box Plot)', fontsize=14)
    ax.set_ylabel('절대 오차')
    ax.grid(True, alpha=0.3)
    
    # 3. 오차율 히스토그램
    ax = axes[0, 2]
    ax.hist(df['오차율_구형(%)'], bins=30, alpha=0.5, label='LSTM 구형', color='red')
    ax.hist(df['오차율_신형(%)'], bins=30, alpha=0.5, label='앙상블 신형', color='blue')
    ax.set_title('오차율 분포', fontsize=14)
    ax.set_xlabel('오차율 (%)')
    ax.set_ylabel('빈도')
    ax.legend()
    
    # 4. 시간대별 평균 오차
    ax = axes[1, 0]
    hourly = df.groupby(pd.to_datetime(df['시간']).dt.hour).agg({
        '오차_구형': 'mean',
        '오차_신형': 'mean'
    })
    x = hourly.index
    width = 0.35
    ax.bar(x - width/2, hourly['오차_구형'], width, label='LSTM 구형', color='red', alpha=0.7)
    ax.bar(x + width/2, hourly['오차_신형'], width, label='앙상블 신형', color='blue', alpha=0.7)
    ax.set_title('시간대별 평균 오차', fontsize=14)
    ax.set_xlabel('시간대')
    ax.set_ylabel('평균 오차')
    ax.legend()
    ax.set_xticks(x)
    
    # 5. 산점도 - 실제 vs 예측 (겹쳐서)
    ax = axes[1, 1]
    ax.scatter(df['실제값'], df['예측_구형(LSTM)'], alpha=0.5, s=20, c='red', label='LSTM 구형')
    ax.scatter(df['실제값'], df['예측_신형(앙상블)'], alpha=0.5, s=20, c='blue', label='앙상블 신형')
    
    # 완벽한 예측선
    min_val = df['실제값'].min()
    max_val = df['실제값'].max()
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='완벽한 예측')
    
    ax.set_title('실제값 vs 예측값', fontsize=14)
    ax.set_xlabel('실제값')
    ax.set_ylabel('예측값')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 누적 오차 비교
    ax = axes[1, 2]
    cumsum_old = df['오차_구형'].cumsum()
    cumsum_new = df['오차_신형'].cumsum()
    ax.plot(cumsum_old, 'r-', label='LSTM 구형 누적오차')
    ax.plot(cumsum_new, 'b-', label='앙상블 신형 누적오차')
    ax.set_title('누적 오차 비교', fontsize=14)
    ax.set_xlabel('시간 인덱스')
    ax.set_ylabel('누적 오차')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n상세 비교 차트가 저장되었습니다: {save_path}")
    plt.show()

# 사용 예시
if __name__ == "__main__":
    # 예시 데이터 생성 (실제 사용시에는 모델 평가 결과 사용)
    n_samples = 100
    times = pd.date_range(start='2025-07-31', periods=n_samples, freq='1min')
    actuals = np.random.normal(1500, 100, n_samples)
    pred_old = actuals + np.random.normal(0, 50, n_samples)  # LSTM 구형
    pred_new = actuals + np.random.normal(0, 30, n_samples)  # 앙상블 신형 (더 정확)
    
    # 비교 요약표 생성
    df, summary = create_comparison_summary(times, actuals, pred_old, pred_new)
    
    # 상세 차트 생성
    plot_detailed_comparison(df)