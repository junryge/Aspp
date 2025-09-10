import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class SequenceViewer:
    """
    CURRENT_M16A_3F_JOB_2의 시퀀스를 직접 확인하고 분석하는 클래스
    """
    
    def __init__(self, file_path):
        """데이터 로드 및 초기화"""
        self.df = pd.read_csv(file_path)
        self.target_column = 'CURRENT_M16A_3F_JOB_2'
        self.prepare_data()
        
    def prepare_data(self):
        """데이터 전처리"""
        # 시간 정보 추가
        self.df['datetime'] = pd.to_datetime(self.df['STAT_DT'], format='%Y%m%d%H%M')
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['minute'] = self.df['datetime'].dt.minute
        
        # 타켓 변수 숫자로 변환
        self.df[self.target_column] = pd.to_numeric(self.df[self.target_column], errors='coerce')
        
        # 주요 특징들
        self.key_features = [
            'M14A_3F_TO_HUB_CMD',
            'M14A_3F_TO_HUB_JOB2',
            'M16A_3F_CMD',
            'M14B_7F_TO_HUB_CMD',
            'M16A_3F_STORAGE_UTIL',
            'M16A_3F_TO_M14A_3F_JOB',
            'M16A_3F_TO_M14B_7F_JOB'
        ]
        
        # 숫자로 변환
        for col in self.key_features:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print("=" * 80)
        print("데이터 로드 완료")
        print("=" * 80)
        print(f"총 데이터: {len(self.df)}개 (24시간)")
        print(f"타켓 최대값: {self.df[self.target_column].max():.0f}")
        print(f"타켓 평균: {self.df[self.target_column].mean():.1f}")
        print()
    
    def find_high_value_sequences(self, sequence_length=15, top_n=10, threshold=None):
        """
        높은 타켓값을 가진 시퀀스 찾기
        
        Parameters:
        -----------
        sequence_length : int
            시퀀스 길이 (분)
        top_n : int
            상위 몇 개를 볼지
        threshold : float
            특정 임계값 이상만 찾기 (None이면 상위 n개)
        """
        print(f"\n{'='*80}")
        print(f"높은 타켓값 시퀀스 찾기 (시퀀스 길이: {sequence_length}분)")
        print("="*80)
        
        sequences = []
        
        # 모든 가능한 시퀀스 수집
        for i in range(sequence_length, len(self.df)):
            end_value = self.df.iloc[i][self.target_column]
            
            if threshold and end_value < threshold:
                continue
                
            sequence_data = {
                'end_index': i,
                'end_time': self.df.iloc[i]['datetime'],
                'end_value': end_value,
                'sequence_start': i - sequence_length,
                'sequence_end': i
            }
            
            # 시퀀스 통계 계산
            seq_df = self.df.iloc[i-sequence_length:i+1]
            
            # 타켓의 시퀀스 패턴
            sequence_data['target_start'] = seq_df.iloc[0][self.target_column]
            sequence_data['target_mean'] = seq_df[self.target_column].mean()
            sequence_data['target_std'] = seq_df[self.target_column].std()
            sequence_data['target_change'] = end_value - sequence_data['target_start']
            sequence_data['target_change_rate'] = sequence_data['target_change'] / sequence_length
            
            # 주요 특징들의 평균
            for col in self.key_features:
                if col in seq_df.columns:
                    sequence_data[f'{col}_mean'] = seq_df[col].mean()
                    sequence_data[f'{col}_change'] = seq_df[col].iloc[-1] - seq_df[col].iloc[0]
            
            sequences.append(sequence_data)
        
        # DataFrame으로 변환
        sequences_df = pd.DataFrame(sequences)
        
        # 상위 n개 선택
        if threshold:
            top_sequences = sequences_df[sequences_df['end_value'] >= threshold].sort_values('end_value', ascending=False).head(top_n)
            print(f"\n임계값 {threshold} 이상인 시퀀스 중 상위 {top_n}개:")
        else:
            top_sequences = sequences_df.nlargest(top_n, 'end_value')
            print(f"\n상위 {top_n}개 시퀀스:")
        
        print("\n순위 | 타켓값 | 시간 | 시작값→끝값 | 변화량 | 변화율(/분)")
        print("-" * 70)
        
        for idx, (_, row) in enumerate(top_sequences.iterrows(), 1):
            print(f"{idx:3d} | {row['end_value']:6.0f} | {row['end_time'].strftime('%H:%M')} | "
                  f"{row['target_start']:6.0f}→{row['end_value']:6.0f} | "
                  f"{row['target_change']:+7.1f} | {row['target_change_rate']:+6.2f}")
        
        return top_sequences
    
    def visualize_sequence(self, sequence_index, sequence_length=15, save=True):
        """
        특정 시퀀스를 시각화
        
        Parameters:
        -----------
        sequence_index : int
            시퀀스가 끝나는 인덱스
        sequence_length : int
            시퀀스 길이 (분)
        """
        if sequence_index < sequence_length:
            print(f"⚠️ 인덱스 {sequence_index}는 시퀀스 길이 {sequence_length}보다 작습니다.")
            return
        
        # 시퀀스 추출
        seq_start = sequence_index - sequence_length
        seq_df = self.df.iloc[seq_start:sequence_index+1].copy()
        seq_df['relative_time'] = range(len(seq_df))
        
        # 시각화
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. 타켓 변화
        axes[0, 0].plot(seq_df['relative_time'], seq_df[self.target_column], 
                       'o-', linewidth=2, markersize=8, color='darkblue')
        axes[0, 0].axhline(y=226, color='red', linestyle='--', alpha=0.5, label='Top 10% (226)')
        axes[0, 0].axhline(y=237, color='orange', linestyle='--', alpha=0.5, label='Top 5% (237)')
        axes[0, 0].scatter(len(seq_df)-1, seq_df[self.target_column].iloc[-1], 
                          color='red', s=150, zorder=5, marker='*')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('CURRENT_M16A_3F_JOB_2')
        axes[0, 0].set_title(f'Target Sequence (Final: {seq_df[self.target_column].iloc[-1]:.0f})', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 최종값과 시간 표시
        final_time = seq_df['datetime'].iloc[-1].strftime('%H:%M')
        axes[0, 0].text(0.02, 0.98, f'Time: {final_time}', 
                       transform=axes[0, 0].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. M14A_3F_TO_HUB_CMD
        if 'M14A_3F_TO_HUB_CMD' in seq_df.columns:
            axes[0, 1].plot(seq_df['relative_time'], seq_df['M14A_3F_TO_HUB_CMD'], 
                           'o-', linewidth=2, color='green')
            axes[0, 1].axhline(y=52, color='red', linestyle='--', alpha=0.5, label='Recommended (52)')
            axes[0, 1].set_xlabel('Time (minutes)')
            axes[0, 1].set_ylabel('M14A_3F_TO_HUB_CMD')
            axes[0, 1].set_title('M14A to HUB Command', fontsize=12)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. M16A_3F_CMD
        if 'M16A_3F_CMD' in seq_df.columns:
            axes[1, 0].plot(seq_df['relative_time'], seq_df['M16A_3F_CMD'], 
                           'o-', linewidth=2, color='purple')
            axes[1, 0].axhline(y=194, color='red', linestyle='--', alpha=0.5, label='Recommended (194)')
            axes[1, 0].set_xlabel('Time (minutes)')
            axes[1, 0].set_ylabel('M16A_3F_CMD')
            axes[1, 0].set_title('M16A 3F Command', fontsize=12)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. M14B_7F_TO_HUB_CMD
        if 'M14B_7F_TO_HUB_CMD' in seq_df.columns:
            axes[1, 1].plot(seq_df['relative_time'], seq_df['M14B_7F_TO_HUB_CMD'], 
                           'o-', linewidth=2, color='orange')
            axes[1, 1].axhline(y=39, color='red', linestyle='--', alpha=0.5, label='Recommended (39)')
            axes[1, 1].set_xlabel('Time (minutes)')
            axes[1, 1].set_ylabel('M14B_7F_TO_HUB_CMD')
            axes[1, 1].set_title('M14B to HUB Command', fontsize=12)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Storage Utilization
        if 'M16A_3F_STORAGE_UTIL' in seq_df.columns:
            axes[2, 0].plot(seq_df['relative_time'], seq_df['M16A_3F_STORAGE_UTIL'], 
                           'o-', linewidth=2, color='brown')
            axes[2, 0].axhline(y=8, color='red', linestyle='--', alpha=0.5, label='Recommended (8%)')
            axes[2, 0].set_xlabel('Time (minutes)')
            axes[2, 0].set_ylabel('Storage Util (%)')
            axes[2, 0].set_title('Storage Utilization', fontsize=12)
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 모든 특징 정규화 비교
        features_to_plot = ['M14A_3F_TO_HUB_CMD', 'M16A_3F_CMD', 'M14B_7F_TO_HUB_CMD', 'M16A_3F_STORAGE_UTIL']
        for feat in features_to_plot:
            if feat in seq_df.columns:
                normalized = (seq_df[feat] - seq_df[feat].min()) / (seq_df[feat].max() - seq_df[feat].min() + 1e-10)
                axes[2, 1].plot(seq_df['relative_time'], normalized, 'o-', label=feat[:15], alpha=0.7)
        
        # 타켓도 정규화해서 추가
        target_norm = (seq_df[self.target_column] - seq_df[self.target_column].min()) / \
                     (seq_df[self.target_column].max() - seq_df[self.target_column].min() + 1e-10)
        axes[2, 1].plot(seq_df['relative_time'], target_norm, 'o-', 
                       linewidth=3, label='TARGET', color='red', alpha=0.8)
        
        axes[2, 1].set_xlabel('Time (minutes)')
        axes[2, 1].set_ylabel('Normalized Value')
        axes[2, 1].set_title('All Features Normalized', fontsize=12)
        axes[2, 1].legend(loc='best', fontsize=8)
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Sequence Analysis: {sequence_length} minutes before {final_time} (Target: {seq_df[self.target_column].iloc[-1]:.0f})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filename = f'/mnt/user-data/outputs/sequence_view_{sequence_index}.png'
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"\n시각화 저장: {filename}")
        
        plt.show()
        
        return seq_df
    
    def analyze_multiple_sequences(self, sequence_lengths=None):
        """
        여러 시퀀스 길이 비교 분석
        
        Parameters:
        -----------
        sequence_lengths : list
            테스트할 시퀀스 길이들 (기본값: [5, 10, 15, 20, 30, 45, 60])
        """
        print("\n" + "="*80)
        print("다양한 시퀀스 길이 비교 (최적 길이 탐색)")
        print("="*80)
        
        if sequence_lengths is None:
            sequence_lengths = [5, 10, 15, 20, 30, 45, 60]
        
        results = []
        
        # 상위 10% 임계값
        threshold_90 = self.df[self.target_column].quantile(0.9)
        threshold_95 = self.df[self.target_column].quantile(0.95)
        
        print(f"\n타켓 임계값:")
        print(f"  - 상위 10%: {threshold_90:.1f}")
        print(f"  - 상위 5%: {threshold_95:.1f}")
        print(f"  - 최대값: {self.df[self.target_column].max():.1f}")
        
        print(f"\n시퀀스 길이 테스트: {sequence_lengths}")
        print("-" * 60)
        
        for length in sequence_lengths:
            # 높은 값 예측 성공률 계산
            success_count_90 = 0
            success_count_95 = 0
            total_high_90 = 0
            total_high_95 = 0
            
            # 패턴 일관성 계산을 위한 리스트
            high_patterns = []
            
            for i in range(length, len(self.df)):
                current_value = self.df.iloc[i][self.target_column]
                seq_mean = self.df.iloc[i-length:i][self.target_column].mean()
                seq_std = self.df.iloc[i-length:i][self.target_column].std()
                seq_trend = (self.df.iloc[i-1][self.target_column] - self.df.iloc[i-length][self.target_column]) / length
                
                # 상위 10% 체크
                if current_value >= threshold_90:
                    total_high_90 += 1
                    high_patterns.append(seq_mean)
                    # 예측 조건: 시퀀스 평균이 상위 30% 이상
                    if seq_mean >= self.df[self.target_column].quantile(0.7):
                        success_count_90 += 1
                
                # 상위 5% 체크
                if current_value >= threshold_95:
                    total_high_95 += 1
                    # 예측 조건: 시퀀스 평균이 상위 20% 이상
                    if seq_mean >= self.df[self.target_column].quantile(0.8):
                        success_count_95 += 1
            
            # 성공률 계산
            success_rate_90 = (success_count_90 / total_high_90 * 100) if total_high_90 > 0 else 0
            success_rate_95 = (success_count_95 / total_high_95 * 100) if total_high_95 > 0 else 0
            
            # 패턴 일관성 (표준편차가 낮을수록 좋음)
            pattern_consistency = 1 / (1 + np.std(high_patterns)) if high_patterns else 0
            
            results.append({
                'length': length,
                'success_rate': success_rate_90,  # 상위 10% 기준
                'success_rate_95': success_rate_95,  # 상위 5% 기준
                'total_high': total_high_90,
                'predicted': success_count_90,
                'total_high_95': total_high_95,
                'predicted_95': success_count_95,
                'consistency': pattern_consistency
            })
            
            print(f"{length:3d}분: 상위10% 예측 {success_rate_90:5.1f}% ({success_count_90:3d}/{total_high_90:3d}) | "
                  f"상위5% 예측 {success_rate_95:5.1f}% ({success_count_95:3d}/{total_high_95:3d})")
        
        results_df = pd.DataFrame(results)
        
        # 종합 점수 계산 (성공률과 일관성 고려)
        results_df['overall_score'] = (
            results_df['success_rate'] * 0.6 + 
            results_df['success_rate_95'] * 0.3 + 
            results_df['consistency'] * 100 * 0.1
        )
        
        print("\n" + "="*60)
        print("분석 결과 요약")
        print("="*60)
        
        # 최적 길이 찾기
        best_idx = results_df['overall_score'].idxmax()
        best_length = results_df.loc[best_idx, 'length']
        
        print(f"\n🎯 최적 시퀀스 길이: {best_length}분")
        print(f"  - 상위 10% 예측 성공률: {results_df.loc[best_idx, 'success_rate']:.1f}%")
        print(f"  - 상위 5% 예측 성공률: {results_df.loc[best_idx, 'success_rate_95']:.1f}%")
        print(f"  - 패턴 일관성: {results_df.loc[best_idx, 'consistency']:.3f}")
        print(f"  - 종합 점수: {results_df.loc[best_idx, 'overall_score']:.1f}")
        
        # Top 3 출력
        print("\n📊 상위 3개 시퀀스 길이:")
        top3 = results_df.nlargest(3, 'overall_score')
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            print(f"  {i}. {int(row['length']):2d}분 - 종합점수: {row['overall_score']:.1f} "
                  f"(상위10%: {row['success_rate']:.1f}%, 상위5%: {row['success_rate_95']:.1f}%)")
        
        return results_df
    
    def show_realtime_prediction(self, current_index, sequence_length=15):
        """
        특정 시점에서 다음 값 예측
        
        Parameters:
        -----------
        current_index : int
            현재 시점 인덱스
        sequence_length : int
            사용할 시퀀스 길이
        """
        if current_index < sequence_length:
            print("시퀀스 길이보다 인덱스가 작습니다.")
            return
        
        # 현재까지의 시퀀스
        seq_df = self.df.iloc[current_index-sequence_length:current_index]
        
        print("\n" + "="*80)
        print(f"실시간 예측 (현재 시간: {self.df.iloc[current_index-1]['datetime'].strftime('%H:%M')})")
        print("="*80)
        
        # 현재 시퀀스 통계
        print("\n현재 시퀀스 상태 (최근 15분):")
        print("-" * 40)
        
        stats = {
            'TARGET 평균': seq_df[self.target_column].mean(),
            'TARGET 트렌드': seq_df[self.target_column].iloc[-1] - seq_df[self.target_column].iloc[0],
            'M14A_3F_TO_HUB_CMD 평균': seq_df['M14A_3F_TO_HUB_CMD'].mean() if 'M14A_3F_TO_HUB_CMD' in seq_df.columns else 0,
            'M16A_3F_CMD 평균': seq_df['M16A_3F_CMD'].mean() if 'M16A_3F_CMD' in seq_df.columns else 0,
            'M14B_7F_TO_HUB_CMD 평균': seq_df['M14B_7F_TO_HUB_CMD'].mean() if 'M14B_7F_TO_HUB_CMD' in seq_df.columns else 0,
        }
        
        for key, value in stats.items():
            print(f"{key}: {value:.1f}")
        
        # 예측
        print("\n예측:")
        print("-" * 40)
        
        # 간단한 규칙 기반 예측
        prediction_score = 0
        reasons = []
        
        # M16A_3F_CMD 체크
        if 'M16A_3F_CMD' in seq_df.columns and seq_df['M16A_3F_CMD'].mean() >= 194:
            prediction_score += 40
            reasons.append(f"✓ M16A_3F_CMD 평균 {seq_df['M16A_3F_CMD'].mean():.1f} ≥ 194")
        
        # M14A_3F_TO_HUB_CMD 체크
        if 'M14A_3F_TO_HUB_CMD' in seq_df.columns and seq_df['M14A_3F_TO_HUB_CMD'].mean() >= 52:
            prediction_score += 30
            reasons.append(f"✓ M14A_3F_TO_HUB_CMD 평균 {seq_df['M14A_3F_TO_HUB_CMD'].mean():.1f} ≥ 52")
        
        # 타켓 트렌드 체크
        target_trend = seq_df[self.target_column].iloc[-1] - seq_df[self.target_column].iloc[0]
        if target_trend > 10:
            prediction_score += 20
            reasons.append(f"✓ 타켓 상승 트렌드 (+{target_trend:.1f})")
        
        # 시간대 체크
        hour = self.df.iloc[current_index-1]['hour']
        if hour in [7, 12, 18, 21]:
            prediction_score += 10
            reasons.append(f"✓ 좋은 시간대 ({hour}시)")
        
        # 예측 결과
        if prediction_score >= 70:
            print("🔴 높은 값 예상 (226 이상)")
        elif prediction_score >= 50:
            print("🟡 중간-높은 값 예상 (210-225)")
        else:
            print("🟢 일반 값 예상 (210 미만)")
        
        print(f"\n신뢰도: {prediction_score}%")
        print("\n근거:")
        for reason in reasons:
            print(f"  {reason}")
        
        # 실제 값 (다음 시점)
        if current_index < len(self.df):
            actual_value = self.df.iloc[current_index][self.target_column]
            print(f"\n실제 값: {actual_value:.0f}")
            if actual_value >= 226:
                print("→ 실제로 높은 값 달성! ✓")
            elif actual_value >= 210:
                print("→ 중간-높은 값")
            else:
                print("→ 일반 값")

# 메인 실행 함수
def main():
    """메인 실행"""
    print("=" * 80)
    print("시퀀스 직접 확인 도구")
    print("=" * 80)
    
    # 뷰어 초기화
    viewer = SequenceViewer('/mnt/user-data/uploads/Hub5월.CSV')
    
    # 1. 먼저 다양한 시퀀스 길이 테스트하여 최적값 찾기
    print("\n1️⃣ 시퀀스 길이별 성능 비교 (최적 길이 찾기)")
    comparison_df = viewer.analyze_multiple_sequences()
    
    # 최적 시퀀스 길이 결정
    best_length = comparison_df.loc[comparison_df['success_rate'].idxmax(), 'length']
    print(f"\n✅ 최적 시퀀스 길이: {best_length}분 (성공률: {comparison_df.loc[comparison_df['success_rate'].idxmax(), 'success_rate']:.1f}%)")
    
    # 2. 최적 길이로 높은 값 시퀀스 찾기
    print(f"\n2️⃣ 높은 타켓값 시퀀스 찾기 (시퀀스 길이: {best_length}분)")
    top_sequences = viewer.find_high_value_sequences(sequence_length=int(best_length), top_n=10)
    
    # 3. 최고값 시퀀스 시각화
    if len(top_sequences) > 0:
        best_seq = top_sequences.iloc[0]
        print(f"\n3️⃣ 최고값 시퀀스 시각화 (타켓: {best_seq['end_value']:.0f})")
        viewer.visualize_sequence(int(best_seq['end_index']), sequence_length=int(best_length))
    
    # 4. 실시간 예측 예시 (최적 길이 사용)
    print(f"\n4️⃣ 실시간 예측 시뮬레이션 (시퀀스: {best_length}분)")
    # 오후 6시 경 데이터로 테스트
    test_index = 18 * 60 + 30  # 18:30
    viewer.show_realtime_prediction(test_index, sequence_length=int(best_length))
    
    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)
    
    return viewer, top_sequences

if __name__ == "__main__":
    viewer, top_sequences = main()
    
    # 추가 분석을 위한 안내
    print("\n💡 추가 분석 방법:")
    print("1. 특정 시퀀스 보기: viewer.visualize_sequence(인덱스, 15)")
    print("2. 높은 값 찾기: viewer.find_high_value_sequences(15, top_n=20)")
    print("3. 실시간 예측: viewer.show_realtime_prediction(현재인덱스, 15)")
    print("4. 임계값 이상 찾기: viewer.find_high_value_sequences(15, threshold=250)")