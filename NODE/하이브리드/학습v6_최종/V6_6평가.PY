"""
📊 반도체 물류 예측 모델 평가
================================
실제값 vs 예측값 비교 및 시각화
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
import os
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ModelEvaluator:
    def __init__(self, model_path, scaler_path='scalers/'):
        """모델 평가기 초기화"""
        print("="*80)
        print("📊 모델 평가 시스템")
        print("="*80)
        
        # 모델 로드
        self.model = tf.keras.models.load_model(model_path)
        self.model_name = os.path.basename(model_path).split('.')[0]
        print(f"✅ 모델 로드: {self.model_name}")
        
        # 스케일러 로드
        with open(f'{scaler_path}feature_scaler.pkl', 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(f'{scaler_path}target_scaler.pkl', 'rb') as f:
            self.target_scaler = pickle.load(f)
        with open(f'{scaler_path}config.json', 'r') as f:
            config = json.load(f)
            self.seq_len = config['seq_len']
            self.pred_len = config['pred_len']
            self.feature_columns = config['feature_columns']
        print(f"✅ 스케일러 로드 완료")
        
    def load_test_data(self, filepath):
        """테스트 데이터 로드"""
        print(f"\n📂 평가 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        print(f"  원본: {df.shape[0]:,}행")
        
        # 0값 제거
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        # 시간 변환
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        print(f"  유효: {df.shape[0]:,}행")
        return df
    
    def create_features(self, df):
        """특성 생성"""
        # 기본 특성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        
        # 시간
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
        df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        
        # 이동평균
        for w in [10, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
        
        # 변화율
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        
        return df
    
    def evaluate(self, test_file):
        """평가 실행"""
        # 데이터 로드
        df = self.load_test_data(test_file)
        df = self.create_features(df)
        
        # 결과 저장용
        results = []
        
        print("\n🔮 예측 시작...")
        print(f"  시퀀스: {self.seq_len}분")
        print(f"  예측: {self.pred_len}분 후")
        
        # 예측 가능한 범위
        start_idx = self.seq_len
        end_idx = len(df) - self.pred_len
        
        total = end_idx - start_idx
        print(f"  예측 개수: {total:,}개")
        
        predictions = []
        actuals = []
        timestamps = []
        
        # 배치 예측
        batch_size = 1000
        for i in range(start_idx, end_idx, batch_size):
            batch_end = min(i + batch_size, end_idx)
            
            # 배치 데이터 준비
            X_batch = []
            for j in range(i, batch_end):
                # 100분 데이터
                seq_data = df.iloc[j-self.seq_len:j][self.feature_columns].values
                X_batch.append(seq_data)
            
            if len(X_batch) == 0:
                continue
                
            # 스케일링
            X_batch = np.array(X_batch)
            X_batch_scaled = []
            for seq in X_batch:
                seq_scaled = self.feature_scaler.transform(seq)
                X_batch_scaled.append(seq_scaled)
            X_batch_scaled = np.array(X_batch_scaled)
            
            # 예측
            preds = self.model.predict(X_batch_scaled, verbose=0)
            y_pred_scaled = preds[0].flatten()
            
            # 역변환
            y_pred = self.target_scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)).flatten()
            
            # 실제값
            for k, j in enumerate(range(i, batch_end)):
                actual_time = df.iloc[j]['CURRTIME']
                pred_time = actual_time + timedelta(minutes=self.pred_len)
                
                # 10분 후 실제값
                actual_idx = j + self.pred_len
                if actual_idx < len(df):
                    actual_value = df.iloc[actual_idx]['TOTALCNT']
                    pred_value = y_pred[k]
                    
                    predictions.append(pred_value)
                    actuals.append(actual_value)
                    timestamps.append(pred_time)
                    
                    results.append({
                        '예측시점': actual_time.strftime('%Y-%m-%d %H:%M'),
                        '예측대상시간': pred_time.strftime('%Y-%m-%d %H:%M'),
                        '실제값': actual_value,
                        '예측값': round(pred_value),
                        '오차': round(pred_value - actual_value),
                        '오차율(%)': round(abs(pred_value - actual_value) / actual_value * 100, 2)
                    })
            
            if len(predictions) % 10000 == 0:
                print(f"    처리: {len(predictions):,}/{total:,}")
        
        print(f"\n✅ 예측 완료: {len(predictions):,}개")
        
        # 성능 메트릭
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        print("\n📊 전체 성능:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  평균 정확도: {100-mape:.2f}%")
        
        # 결과 DataFrame
        results_df = pd.DataFrame(results)
        
        # 시간대별 성능
        results_df['시간대'] = pd.to_datetime(results_df['예측대상시간']).dt.hour
        hourly_stats = results_df.groupby('시간대').agg({
            '오차율(%)': ['mean', 'std'],
            '오차': ['mean', 'std']
        }).round(2)
        
        print("\n⏰ 시간대별 성능:")
        print(hourly_stats.head(5))
        
        # 구간별 성능
        results_df['구간'] = pd.cut(results_df['실제값'], 
                                   bins=[0, 1400, 1700, 999999],
                                   labels=['정상', '주의', '심각'])
        
        level_stats = results_df.groupby('구간').agg({
            '오차율(%)': ['mean', 'count'],
            '오차': ['mean', 'std']
        }).round(2)
        
        print("\n📊 구간별 성능:")
        print(level_stats)
        
        # CSV 저장
        output_file = f'evaluation_{self.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 결과 저장: {output_file}")
        
        # 샘플 출력
        print("\n📝 예측 결과 샘플 (최근 20개):")
        print("="*100)
        print(f"{'예측시점':^19} | {'대상시간':^19} | {'실제값':>8} | {'예측값':>8} | {'오차':>8} | {'오차율':>8}")
        print("-"*100)
        
        for _, row in results_df.tail(20).iterrows():
            print(f"{row['예측시점']:^19} | {row['예측대상시간']:^19} | "
                  f"{row['실제값']:8.0f} | {row['예측값']:8.0f} | "
                  f"{row['오차']:8.0f} | {row['오차율(%)']:7.1f}%")
        
        # 시각화
        self.visualize_results(results_df, timestamps, actuals, predictions)
        
        return results_df
    
    def visualize_results(self, results_df, timestamps, actuals, predictions):
        """결과 시각화"""
        print("\n📈 시각화 생성 중...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 시계열 비교 (최근 1000개)
        n_plot = min(1000, len(timestamps))
        axes[0, 0].plot(timestamps[-n_plot:], actuals[-n_plot:], 
                       label='Actual', alpha=0.7)
        axes[0, 0].plot(timestamps[-n_plot:], predictions[-n_plot:], 
                       label='Predicted', alpha=0.7)
        axes[0, 0].set_title(f'Actual vs Predicted (Last {n_plot} points)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('TOTALCNT')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 산점도
        axes[0, 1].scatter(actuals, predictions, alpha=0.5, s=1)
        axes[0, 1].plot([actuals.min(), actuals.max()], 
                       [actuals.min(), actuals.max()], 
                       'r--', lw=2)
        axes[0, 1].set_title(f'Scatter Plot (R² = {r2_score(actuals, predictions):.4f})')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 오차 분포
        errors = predictions - actuals
        axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_title(f'Error Distribution (MAE = {np.mean(np.abs(errors)):.2f})')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 시간대별 오차율
        hourly = results_df.groupby('시간대')['오차율(%)'].mean()
        axes[1, 1].bar(hourly.index, hourly.values, alpha=0.7)
        axes[1, 1].set_title('Error Rate by Hour')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Error Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f'evaluation_{self.model_name}_plot.png'
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        print(f"📊 그래프 저장: {plot_file}")
        plt.show()

def main():
    """메인 실행"""
    # 모델 선택
    model_files = [f for f in os.listdir('models') if f.endswith('.keras')]
    
    if not model_files:
        print("❌ 모델 파일이 없습니다!")
        return
    
    print("\n📁 사용 가능한 모델:")
    for i, model in enumerate(model_files):
        print(f"  {i+1}. {model}")
    
    # 최고 성능 모델 자동 선택 (PatchTST)
    if 'PatchTST.keras' in model_files:
        model_path = 'models/PatchTST.keras'
        print(f"\n✅ 최고 성능 모델 선택: PatchTST")
    else:
        model_path = f'models/{model_files[0]}'
        print(f"\n✅ 모델 선택: {model_files[0]}")
    
    # 평가기 생성
    evaluator = ModelEvaluator(model_path)
    
    # 테스트 파일
    test_files = [
        'data/20250731_to20250806.csv',
        'data/test_data.csv',
        '/mnt/user-data/uploads/test.csv'
    ]
    
    test_file = None
    for file in test_files:
        if os.path.exists(file):
            test_file = file
            break
    
    if not test_file:
        print("❌ 테스트 데이터를 찾을 수 없습니다!")
        print("  예상 경로: data/20250731_to20250806.csv")
        return
    
    # 평가 실행
    results = evaluator.evaluate(test_file)
    
    print("\n✅ 평가 완료!")
    print("="*80)

if __name__ == "__main__":
    main()