import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import os

class SequenceAnalyzer:
    """
    타겟값 예측을 위한 시퀀스 분석 도구
    """
    
    def __init__(self, target_column='CURRENT_M16A_3F_JOB_2'):
        self.target_column = target_column
        self.results = []
        
    def load_data(self, file_path):
        """
        CSV 파일 로드 및 전처리
        """
        df = pd.read_csv(file_path)
        
        # 타겟 컬럼 숫자 변환
        if self.target_column in df.columns:
            df[self.target_column] = pd.to_numeric(df[self.target_column], errors='coerce')
        
        # 시간 정보 추가
        if 'STAT_DT' in df.columns:
            df['datetime'] = pd.to_datetime(df['STAT_DT'], format='%Y%m%d%H%M', errors='coerce')
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
        
        # 주요 특징 컬럼들 숫자 변환
        feature_cols = ['M14A_3F_TO_HUB_CMD', 'M14A_3F_TO_HUB_JOB2', 
                       'M16A_3F_CMD', 'M14B_7F_TO_HUB_CMD', 
                       'M16A_3F_STORAGE_UTIL', 'M16A_3F_TO_M14A_3F_JOB',
                       'M16A_3F_TO_M14B_7F_JOB']
        
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def analyze_sequence_performance(self, df, target_threshold, sequence_length, prediction_horizon=0):
        """
        특정 시퀀스 길이의 성능 분석
        
        Parameters:
        -----------
        df : DataFrame
            데이터
        target_threshold : int
            타겟 임계값 (예: 300)
        sequence_length : int
            시퀀스 길이 (분)
        prediction_horizon : int
            예측 시간 (0이면 현재, 10이면 10분 후)
        """
        
        patterns = []
        predictions = []
        actuals = []
        
        # 분석 범위
        start_idx = sequence_length
        end_idx = len(df) - prediction_horizon
        
        for i in range(start_idx, end_idx):
            # 현재 시퀀스
            seq = df.iloc[i-sequence_length:i]
            
            # 타겟 시점
            target_idx = i + prediction_horizon
            if target_idx < len(df):
                target_value = df.iloc[target_idx][self.target_column]
                
                # 타겟이 임계값 이상인지
                is_high = target_value >= target_threshold
                
                # 시퀀스 특징 추출
                features = {
                    'index': i,
                    'target_value': target_value,
                    'is_high': is_high,
                    'seq_mean': seq[self.target_column].mean(),
                    'seq_std': seq[self.target_column].std(),
                    'seq_min': seq[self.target_column].min(),
                    'seq_max': seq[self.target_column].max(),
                    'seq_last': seq[self.target_column].iloc[-1],
                    'seq_trend': (seq[self.target_column].iloc[-1] - seq[self.target_column].iloc[0]) / sequence_length
                }
                
                # 시간 정보
                if 'hour' in df.columns:
                    features['hour'] = df.iloc[target_idx]['hour']
                
                # 다른 특징들
                for col in ['M14A_3F_TO_HUB_CMD', 'M16A_3F_CMD', 'M14B_7F_TO_HUB_CMD']:
                    if col in seq.columns:
                        features[f'{col}_mean'] = seq[col].mean()
                        features[f'{col}_std'] = seq[col].std()
                
                # 예측 (간단한 규칙)
                prediction = self.predict(features, target_threshold)
                
                patterns.append(features)
                predictions.append(prediction)
                actuals.append(1 if is_high else 0)
        
        # 성능 계산
        correct = sum([1 for p, a in zip(predictions, actuals) if p == a])
        accuracy = correct / len(predictions) if predictions else 0
        
        # 고값 예측 성능
        true_high = sum([1 for p, a in zip(predictions, actuals) if p == 1 and a == 1])
        pred_high = sum(predictions)
        actual_high = sum(actuals)
        
        precision = true_high / pred_high if pred_high > 0 else 0
        recall = true_high / actual_high if actual_high > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'sequence_length': sequence_length,
            'target_threshold': target_threshold,
            'prediction_horizon': prediction_horizon,
            'total_samples': len(patterns),
            'actual_high': actual_high,
            'predicted_high': pred_high,
            'true_high': true_high,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'patterns': patterns
        }
    
    def predict(self, features, threshold):
        """
        간단한 예측 규칙
        """
        score = 0
        
        # 현재 값 체크
        if features['seq_last'] >= threshold * 0.9:
            score += 30
        
        # 평균 체크
        if features['seq_mean'] >= threshold * 0.85:
            score += 30
        
        # 트렌드 체크
        if features['seq_trend'] > 2:
            score += 30
        elif features['seq_trend'] > 0:
            score += 10
        
        # 최대값 체크
        if features['seq_max'] >= threshold:
            score += 20
        
        return 1 if score >= 50 else 0
    
    def find_optimal_sequence(self, df, target_thresholds, sequence_lengths, prediction_horizon=0):
        """
        최적 시퀀스 길이 찾기
        
        Parameters:
        -----------
        df : DataFrame
            데이터
        target_thresholds : list
            테스트할 임계값들 (예: [300, 400, 500])
        sequence_lengths : list
            테스트할 시퀀스 길이들 (예: [5, 10, 15, 20, 30])
        prediction_horizon : int
            예측 시간
        """
        
        results = []
        
        print(f"\n{'='*80}")
        print(f"시퀀스 성능 분석 (예측: {prediction_horizon}분 후)")
        print('='*80)
        
        for threshold in target_thresholds:
            print(f"\n타겟 임계값: {threshold}")
            print("-" * 60)
            print(f"{'시퀀스':>8} | {'샘플수':>8} | {'실제':>8} | {'정확도':>8} | {'정밀도':>8} | {'재현율':>8} | {'F1':>8}")
            print("-" * 60)
            
            for seq_len in sequence_lengths:
                result = self.analyze_sequence_performance(df, threshold, seq_len, prediction_horizon)
                results.append(result)
                
                if result['total_samples'] > 0:
                    print(f"{seq_len:8d} | {result['total_samples']:8d} | "
                          f"{result['actual_high']:8d} | {result['accuracy']:8.2%} | "
                          f"{result['precision']:8.2%} | {result['recall']:8.2%} | "
                          f"{result['f1_score']:8.2%}")
        
        return results
    
    def save_results(self, results, output_file='sequence_analysis_results.csv'):
        """
        결과를 CSV로 저장
        """
        # patterns 제외한 결과 저장
        save_data = []
        for r in results:
            save_data.append({
                'sequence_length': r['sequence_length'],
                'target_threshold': r['target_threshold'],
                'prediction_horizon': r['prediction_horizon'],
                'total_samples': r['total_samples'],
                'actual_high': r['actual_high'],
                'predicted_high': r['predicted_high'],
                'true_high': r['true_high'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1_score': r['f1_score']
            })
        
        df_results = pd.DataFrame(save_data)
        df_results.to_csv(output_file, index=False)
        print(f"\n✅ 결과 저장: {output_file}")
        
        return df_results
    
    def visualize_results(self, results_df):
        """
        결과 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 임계값별로 그룹화
        for threshold in results_df['target_threshold'].unique():
            threshold_data = results_df[results_df['target_threshold'] == threshold]
            
            # F1 Score
            axes[0, 0].plot(threshold_data['sequence_length'], 
                          threshold_data['f1_score'], 
                          marker='o', label=f'Threshold {threshold}')
        
        axes[0, 0].set_xlabel('Sequence Length (minutes)')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('F1 Score by Sequence Length')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 정밀도
        for threshold in results_df['target_threshold'].unique():
            threshold_data = results_df[results_df['target_threshold'] == threshold]
            axes[0, 1].plot(threshold_data['sequence_length'], 
                          threshold_data['precision'], 
                          marker='o', label=f'Threshold {threshold}')
        
        axes[0, 1].set_xlabel('Sequence Length (minutes)')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision by Sequence Length')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 재현율
        for threshold in results_df['target_threshold'].unique():
            threshold_data = results_df[results_df['target_threshold'] == threshold]
            axes[1, 0].plot(threshold_data['sequence_length'], 
                          threshold_data['recall'], 
                          marker='o', label=f'Threshold {threshold}')
        
        axes[1, 0].set_xlabel('Sequence Length (minutes)')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Recall by Sequence Length')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 정확도
        for threshold in results_df['target_threshold'].unique():
            threshold_data = results_df[results_df['target_threshold'] == threshold]
            axes[1, 1].plot(threshold_data['sequence_length'], 
                          threshold_data['accuracy'], 
                          marker='o', label=f'Threshold {threshold}')
        
        axes[1, 1].set_xlabel('Sequence Length (minutes)')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy by Sequence Length')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('sequence_analysis_visualization.png')
        print("✅ 시각화 저장: sequence_analysis_visualization.png")
        plt.show()

if __name__ == "__main__":
    # 분석기 초기화
    analyzer = SequenceAnalyzer(target_column='CURRENT_M16A_3F_JOB_2')
    
    # 데이터 로드
    df = analyzer.load_data('/mnt/user-data/uploads/Hub5월.CSV')
    
    # 데이터 확인
    print(f"데이터 크기: {len(df)}")
    print(f"타켓 최소값: {df[analyzer.target_column].min()}")
    print(f"타켓 최대값: {df[analyzer.target_column].max()}")
    print(f"타켓 평균: {df[analyzer.target_column].mean():.1f}")
    
    # 최적 시퀀스 찾기
    target_thresholds = [250, 260, 270, 300, 400, 500]  # 임계값
    sequence_lengths = [5, 10, 15, 20, 30, 45, 60, 90, 120]  # 시퀀스 길이
    prediction_horizon = 10  # 10분 후 예측
    
    results = analyzer.find_optimal_sequence(df, target_thresholds, sequence_lengths, prediction_horizon)
    
    # 결과 저장
    results_df = analyzer.save_results(results, '/mnt/user-data/outputs/sequence_analysis_results.csv')
    
    # 시각화
    analyzer.visualize_results(results_df)
    
    # 최적 시퀀스 출력
    print("\n" + "="*80)
    print("최적 시퀀스 요약")
    print("="*80)
    for threshold in target_thresholds:
        threshold_results = results_df[results_df['target_threshold'] == threshold]
        if len(threshold_results) > 0 and threshold_results['actual_high'].sum() > 0:
            best = threshold_results.loc[threshold_results['f1_score'].idxmax()]
            print(f"\n타겟 {threshold} 이상:")
            print(f"  최적 시퀀스: {best['sequence_length']}분")
            print(f"  F1 Score: {best['f1_score']:.3f}")
            print(f"  정밀도: {best['precision']:.3f}")
            print(f"  재현율: {best['recall']:.3f}")
            print(f"  실제 발생: {best['actual_high']}회")