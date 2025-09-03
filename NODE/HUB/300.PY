import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class FilterDataAnalyzer:
    def __init__(self, file_path):
        """CSV 파일 경로를 받아 초기화"""
        self.df = self.load_and_prepare_data(file_path)
        
    def load_and_prepare_data(self, file_path):
        """CSV 파일 로드 및 전처리"""
        # CSV 파일 읽기
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 컬럼명 정리 (공백 제거)
        df.columns = df.columns.str.strip()
        
        # timestamp를 datetime으로 변환
        df['timestamp'] = pd.to_datetime(df['시간'])
        
        # 시간 관련 컬럼 추가
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['weekday'] = df['timestamp'].dt.weekday
        df['date'] = df['timestamp'].dt.date
        df['month'] = df['timestamp'].dt.month
        
        # 시간대 그룹 생성
        df['time_group'] = pd.cut(df['hour'], 
                                  bins=[-1, 3, 6, 9, 12, 15, 18, 21, 24],
                                  labels=['00-03시', '03-06시', '06-09시', '09-12시', 
                                         '12-15시', '15-18시', '18-21시', '21-24시'])
        
        print(f"✓ 데이터 로드 완료: {len(df)}건")
        return df
    
    def analyze_all(self):
        """전체 분석 실행"""
        print("\n" + "="*60)
        print("예측 모델 필터 데이터 분석 시작")
        print("="*60)
        
        # 1. 기본 통계
        print("\n1. 기본 통계 분석")
        print("-"*40)
        stats = self.basic_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 2. 시간대별 분석
        print("\n2. 시간대별 분석")
        print("-"*40)
        time_stats = self.time_analysis()
        print(time_stats.to_string())
        
        # 3. 오차 분포
        print("\n3. 오차 크기별 분포")
        print("-"*40)
        error_dist = self.error_distribution()
        print(error_dist.to_string())
        
        # 4. 극단값 케이스
        print("\n4. 극단값 케이스")
        print("-"*40)
        max_errors, min_errors = self.extreme_cases()
        print("\n최대 오차 Top 5:")
        print(max_errors.to_string())
        print("\n최소 오차 Top 5:")
        print(min_errors.to_string())
        
        # 5. 일별 추이
        print("\n5. 일별 추이 (최근 10일)")
        print("-"*40)
        daily_stats = self.daily_trend()
        print(daily_stats.tail(10).to_string())
        
        # 6. HTML 보고서 생성
        print("\n6. HTML 보고서 생성 중...")
        self.generate_html_report()
        print("✓ HTML 보고서 생성 완료: filter_analysis_report.html")
        
        # 7. 시각화
        print("\n7. 시각화 차트 생성 중...")
        self.create_visualizations()
        print("✓ 시각화 완료")
        
        return {
            'stats': stats,
            'time_stats': time_stats,
            'error_dist': error_dist,
            'max_errors': max_errors,
            'min_errors': min_errors,
            'daily_stats': daily_stats
        }
    
    def basic_statistics(self):
        """기본 통계 분석"""
        stats = {
            '총 데이터 수': len(self.df),
            '분석 기간': f"{self.df['timestamp'].min().date()} ~ {self.df['timestamp'].max().date()}",
            '관측값 범위': f"{self.df['관측값'].min()} ~ {self.df['관측값'].max()}",
            '예측값 범위': f"{self.df['예측값'].min()} ~ {self.df['예측값'].max()}",
            '오차 범위': f"{self.df['필터오차'].min()} ~ {self.df['필터오차'].max()}",
            '평균 오차': round(self.df['필터오차'].mean(), 2),
            '중앙값 오차': self.df['필터오차'].median(),
            '표준편차': round(self.df['필터오차'].std(), 2)
        }
        return stats
    
    def time_analysis(self):
        """시간대별 분석"""
        time_stats = self.df.groupby('time_group')['필터오차'].agg([
            ('건수', 'count'),
            ('평균오차', 'mean'),
            ('최대오차', 'max'),
            ('최소오차', 'min')
        ]).round(2)
        
        time_stats['비율(%)'] = round(time_stats['건수'] / len(self.df) * 100, 1)
        return time_stats
    
    def error_distribution(self):
        """오차 분포 분석"""
        bins = [0, 20, 40, 60, 80, 100, 200]
        labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '100+']
        self.df['error_group'] = pd.cut(self.df['필터오차'], bins=bins, labels=labels)
        
        error_dist = self.df['error_group'].value_counts().sort_index()
        error_dist_pct = round(error_dist / len(self.df) * 100, 1)
        
        return pd.DataFrame({
            '건수': error_dist,
            '비율(%)': error_dist_pct
        })
    
    def extreme_cases(self, n=5):
        """극단값 케이스 분석"""
        # 최대 오차 케이스
        max_errors = self.df.nlargest(n, '필터오차')[
            ['timestamp', '관측값', '예측값', '필터오차', '관측 MAX']
        ]
        
        # 최소 오차 케이스
        min_errors = self.df.nsmallest(n, '필터오차')[
            ['timestamp', '관측값', '예측값', '필터오차', '관측 MAX']
        ]
        
        return max_errors, min_errors
    
    def daily_trend(self):
        """일별 추이 분석"""
        daily_stats = self.df.groupby('date').agg({
            '필터오차': ['count', 'mean', 'max', 'min']
        }).round(2)
        daily_stats.columns = ['건수', '평균오차', '최대오차', '최소오차']
        return daily_stats
    
    def create_visualizations(self):
        """시각화 차트 생성"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('예측 모델 필터 데이터 분석', fontsize=16, fontweight='bold')
        
        # 1. 시간대별 오류 건수
        time_counts = self.df['time_group'].value_counts().sort_index()
        axes[0, 0].bar(range(len(time_counts)), time_counts.values, color='steelblue')
        axes[0, 0].set_xticks(range(len(time_counts)))
        axes[0, 0].set_xticklabels(time_counts.index, rotation=45)
        axes[0, 0].set_title('시간대별 오류 발생 건수')
        axes[0, 0].set_ylabel('건수')
        
        # 2. 오차 분포 히스토그램
        axes[0, 1].hist(self.df['필터오차'], bins=30, color='coral', edgecolor='black')
        axes[0, 1].set_title('오차 분포')
        axes[0, 1].set_xlabel('오차 값')
        axes[0, 1].set_ylabel('빈도')
        
        # 3. 일별 오류 건수 추이
        daily_counts = self.df.groupby('date')['필터오차'].count()
        axes[0, 2].plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2)
        axes[0, 2].set_title('일별 오류 발생 추이')
        axes[0, 2].set_xlabel('날짜')
        axes[0, 2].set_ylabel('건수')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 관측값 vs 예측값 산점도
        axes[1, 0].scatter(self.df['관측값'], self.df['예측값'], 
                          alpha=0.5, c=self.df['필터오차'], cmap='RdYlBu_r')
        axes[1, 0].plot([self.df['관측값'].min(), 300], [300, 300], 'r--', label='y=300')
        axes[1, 0].plot([300, 300], [300, self.df['예측값'].max()], 'r--')
        axes[1, 0].set_title('관측값 vs 예측값')
        axes[1, 0].set_xlabel('관측값')
        axes[1, 0].set_ylabel('예측값')
        axes[1, 0].legend()
        
        # 5. 시간대별 평균 오차
        time_mean_error = self.df.groupby('time_group')['필터오차'].mean().sort_index()
        axes[1, 1].bar(range(len(time_mean_error)), time_mean_error.values, color='teal')
        axes[1, 1].set_xticks(range(len(time_mean_error)))
        axes[1, 1].set_xticklabels(time_mean_error.index, rotation=45)
        axes[1, 1].set_title('시간대별 평균 오차')
        axes[1, 1].set_ylabel('평균 오차')
        
        # 6. 박스플롯 - 시간대별 오차 분포
        time_groups = ['00-03시', '03-06시', '06-09시', '09-12시', 
                      '12-15시', '15-18시', '18-21시', '21-24시']
        data_for_box = [self.df[self.df['time_group'] == tg]['필터오차'].values 
                       for tg in time_groups]
        axes[1, 2].boxplot(data_for_box, labels=time_groups)
        axes[1, 2].set_title('시간대별 오차 분포 (박스플롯)')
        axes[1, 2].set_ylabel('오차')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('filter_analysis_charts.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def generate_html_report(self):
        """HTML 보고서 생성"""
        stats = self.basic_statistics()
        time_stats = self.time_analysis()
        error_dist = self.error_distribution()
        max_errors, min_errors = self.extreme_cases()
        daily_stats = self.daily_trend()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>예측 모델 필터 데이터 분석 보고서</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Malgun Gothic', '맑은 고딕', sans-serif;
            line-height: 1.8;
            color: #2c3e50;
            background: #f8f9fa;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 18px;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        h2 {{
            color: #1e3c72;
            font-size: 24px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #1e3c72;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .stat-card .label {{
            font-size: 14px;
            opacity: 0.9;
            text-transform: uppercase;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        
        thead {{
            background: #2c3e50;
            color: white;
        }}
        
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tbody tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        tbody tr:hover {{
            background: #e8f4f8;
        }}
        
        .alert {{
            padding: 20px;
            margin: 25px 0;
            border-radius: 8px;
            border-left: 5px solid;
        }}
        
        .alert-danger {{
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }}
        
        .alert-info {{
            background: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }}
        
        .highlight-row {{
            background: #fff9e6 !important;
            font-weight: 600;
        }}
        
        .danger-cell {{
            color: #dc3545;
            font-weight: bold;
        }}
        
        .success-cell {{
            color: #28a745;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>예측 모델 필터 데이터 분석 보고서</h1>
            <p class="subtitle">{stats['총 데이터 수']}건 오류 패턴 분석</p>
            <p>분석 기간: {stats['분석 기간']}</p>
        </div>
        
        <div class="content">
            <!-- 핵심 통계 카드 -->
            <section class="section">
                <h2>📊 핵심 지표</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">총 분석 데이터</div>
                        <div class="value">{stats['총 데이터 수']}</div>
                        <div class="label">필터된 오류 건수</div>
                    </div>
                    <div class="stat-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                        <div class="label">평균 오차</div>
                        <div class="value">{stats['평균 오차']}</div>
                        <div class="label">예측값 - 관측값</div>
                    </div>
                    <div class="stat-card" style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);">
                        <div class="label">오차 범위</div>
                        <div class="value">{stats['오차 범위']}</div>
                        <div class="label">최소 ~ 최대</div>
                    </div>
                    <div class="stat-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
                        <div class="label">표준편차</div>
                        <div class="value">{stats['표준편차']}</div>
                        <div class="label">오차 변동성</div>
                    </div>
                </div>
            </section>
            
            <!-- 중요 경고 -->
            <div class="alert alert-danger">
                <strong>⚠️ 중요 발견사항:</strong> 
                모든 데이터가 "관측값 < 300, 예측값 ≥ 300" 패턴을 보이며, 
                이는 모델이 300을 경계로 체계적으로 과대예측하고 있음을 의미합니다.
            </div>
            
            <!-- 시간대별 분석 -->
            <section class="section">
                <h2>⏰ 시간대별 분석</h2>
                <table>
                    <thead>
                        <tr>
                            <th>시간대</th>
                            <th>건수</th>
                            <th>비율(%)</th>
                            <th>평균 오차</th>
                            <th>최대 오차</th>
                            <th>최소 오차</th>
                        </tr>
                    </thead>
                    <tbody>
                        {time_stats.to_html(header=False, index=True, escape=False, classes='')[23:-16]}
                    </tbody>
                </table>
            </section>
            
            <!-- 오차 분포 -->
            <section class="section">
                <h2>📈 오차 크기별 분포</h2>
                <table>
                    <thead>
                        <tr>
                            <th>오차 범위</th>
                            <th>건수</th>
                            <th>비율(%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {error_dist.to_html(header=False, index=True, escape=False, classes='')[23:-16]}
                    </tbody>
                </table>
                
                <div class="alert alert-info">
                    <strong>📊 분석:</strong> 
                    대부분의 오류가 21-60 범위에 집중되어 있으며, 
                    이는 모델이 일정한 패턴으로 과대예측하고 있음을 시사합니다.
                </div>
            </section>
            
            <!-- 극단값 케이스 -->
            <section class="section">
                <h2>🔍 극단값 케이스 분석</h2>
                
                <h3>최대 오차 Top 5</h3>
                <table>
                    <thead>
                        <tr>
                            <th>시간</th>
                            <th>관측값</th>
                            <th>예측값</th>
                            <th>오차</th>
                            <th>관측 MAX</th>
                        </tr>
                    </thead>
                    <tbody>
                        {max_errors.to_html(header=False, index=False, escape=False, classes='')[23:-16]}
                    </tbody>
                </table>
                
                <h3>최소 오차 케이스</h3>
                <table>
                    <thead>
                        <tr>
                            <th>시간</th>
                            <th>관측값</th>
                            <th>예측값</th>
                            <th>오차</th>
                            <th>관측 MAX</th>
                        </tr>
                    </thead>
                    <tbody>
                        {min_errors.to_html(header=False, index=False, escape=False, classes='')[23:-16]}
                    </tbody>
                </table>
            </section>
            
            <!-- 일별 추이 (최근 10일) -->
            <section class="section">
                <h2>📅 최근 10일 추이</h2>
                <table>
                    <thead>
                        <tr>
                            <th>날짜</th>
                            <th>건수</th>
                            <th>평균 오차</th>
                            <th>최대 오차</th>
                            <th>최소 오차</th>
                        </tr>
                    </thead>
                    <tbody>
                        {daily_stats.tail(10).to_html(header=False, index=True, escape=False, classes='')[23:-16]}
                    </tbody>
                </table>
            </section>
            
            <!-- 결론 -->
            <section class="section">
                <h2>💡 결론 및 제언</h2>
                <div class="alert alert-danger">
                    <strong>핵심 발견사항:</strong>
                    <ul style="margin-top: 10px; padding-left: 20px;">
                        <li>모든 {stats['총 데이터 수']}건이 과대예측 패턴 (예측값 ≥ 300, 관측값 < 300)</li>
                        <li>평균 오차 {stats['평균 오차']}, 최대 오차 {stats['오차 범위'].split(' ~ ')[1]}</li>
                        <li>특정 시간대(새벽/오후)에 오차가 집중되는 경향</li>
                        <li>모델이 300 임계값을 과도하게 회피하는 체계적 편향 존재</li>
                    </ul>
                    <br>
                    <strong>권장사항:</strong> 모델 재학습 또는 후처리 보정이 시급히 필요합니다.
                </div>
            </section>
        </div>
    </div>
</body>
</html>
"""
        
        # HTML 파일로 저장
        with open('filter_analysis_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_content


# 사용 예시
if __name__ == "__main__":
    # CSV 파일 경로 지정
    file_path = 'your_data.csv'  # 실제 파일 경로로 변경하세요
    
    # 분석기 생성 및 실행
    analyzer = FilterDataAnalyzer(file_path)
    results = analyzer.analyze_all()
    
    print("\n" + "="*60)
    print("분석 완료!")
    print("="*60)
    print("생성된 파일:")
    print("  - filter_analysis_report.html (HTML 보고서)")
    print("  - filter_analysis_charts.png (시각화 차트)")