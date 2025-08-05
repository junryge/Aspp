"""
반도체 물류 하이브리드 예측 시스템
=====================================
H.PY의 하이브리드 모델과 예측모델.PY의 데이터 처리를 통합한
실시간 예측 시스템입니다.

주요 기능:
1. 학습된 하이브리드 모델 (LSTM, RNN, GRU, Bi-LSTM) 로드 및 앙상블 예측
2. 실시간 데이터 처리 및 예측
3. 병목 구간 예측 및 알림
4. CPU 기반 실행

버전: 2.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
import json
import logging

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 고정
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# 하이브리드 예측 시스템 클래스
# ===================================

class HybridPredictionSystem:
    """
    학습된 하이브리드 모델을 사용한 통합 예측 시스템
    """
    
    def __init__(self, config_path='results/training_config.json'):
        """
        시스템 초기화
        
        Parameters:
        - config_path: 학습 설정 파일 경로
        """
        self.models = {}
        self.scaler = None
        self.config = None
        self.arima_model = None
        
        # 설정 파일 로드
        self.load_config(config_path)
        
        # 모델 및 스케일러 로드
        self.load_models()
        
        logger.info("하이브리드 예측 시스템 초기화 완료")
    
    def load_config(self, config_path):
        """학습 설정 로드"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"설정 파일 로드 완료: {config_path}")
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {str(e)}")
            # 기본 설정 사용
            self.config = {
                'seq_length': 30,
                'future_minutes': 10,
                'model_weights': {
                    'lstm': 0.3,
                    'gru': 0.25,
                    'rnn': 0.15,
                    'bi_lstm': 0.3
                }
            }
    
    def load_models(self):
        """학습된 모델들 로드"""
        model_names = ['lstm', 'gru', 'rnn', 'bi_lstm']
        
        for model_name in model_names:
            model_path = f'model/{model_name}_final_hybrid.keras'
            try:
                self.models[model_name] = load_model(model_path, compile=False)
                # 컴파일 (예측만 수행하므로 간단히)
                self.models[model_name].compile(optimizer='adam', loss='mse')
                logger.info(f"{model_name.upper()} 모델 로드 완료")
            except Exception as e:
                logger.error(f"{model_name} 모델 로드 실패: {str(e)}")
        
        # ARIMA 모델 로드 (옵션)
        try:
            self.arima_model = joblib.load('model/arima_model.pkl')
            logger.info("ARIMA 모델 로드 완료")
        except:
            logger.warning("ARIMA 모델을 찾을 수 없습니다. 딥러닝 모델만 사용합니다.")
        
        # 스케일러 로드
        try:
            self.scaler = joblib.load('scaler/standard_scaler_hybrid.pkl')
            logger.info("스케일러 로드 완료")
        except:
            # 백업 스케일러 시도
            try:
                self.scaler = joblib.load('scaler/StdScaler_s30f10_0731_2079936.save')
                logger.info("백업 스케일러 로드 완료")
            except Exception as e:
                logger.error(f"스케일러 로드 실패: {str(e)}")
                raise
    
    def preprocess_data(self, data_path):
        """
        예측용 데이터 전처리
        
        Parameters:
        - data_path: CSV 파일 경로
        
        Returns:
        - processed_data: 전처리된 데이터
        """
        logger.info(f"데이터 로딩: {data_path}")
        
        # 데이터 로드
        data = pd.read_csv(data_path)
        
        # 시간 컬럼 datetime 변환
        data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
        data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
        
        # SUM 컬럼 제거
        columns_to_drop = [col for col in data.columns if 'SUM' in col]
        data = data.drop(columns=columns_to_drop)
        
        # 필요한 컬럼만 선택
        data = data[['CURRTIME', 'TOTALCNT', 'TIME']]
        
        # 인덱스 설정
        data.set_index('CURRTIME', inplace=True)
        
        # FUTURE 컬럼 생성 (10분 후 값)
        data['FUTURE'] = pd.NA
        future_minutes = self.config.get('future_minutes', 10)
        
        for i in data.index:
            future_time = i + pd.Timedelta(minutes=future_minutes)
            if (future_time <= data.index.max()) & (future_time in data.index):
                data.loc[i, 'FUTURE'] = data.loc[future_time, 'TOTALCNT']
        
        # NaN 제거
        data.dropna(subset=['FUTURE'], inplace=True)
        
        # 특징 엔지니어링
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        
        # 이동평균 특징
        data['MA_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
        data['MA_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
        data['MA_30'] = data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
        
        # 변동성 특징
        data['STD_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).std()
        data['STD_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).std()
        
        # 변화율
        data['change_rate'] = data['TOTALCNT'].pct_change()
        data['change_rate_5'] = data['TOTALCNT'].pct_change(5)
        
        # 결측값 처리
        data = data.fillna(method='ffill').fillna(0)
        
        logger.info(f"전처리 완료 - 데이터 shape: {data.shape}")
        
        return data
    
    def scale_data(self, data):
        """데이터 스케일링"""
        # 스케일링할 컬럼
        scale_cols = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
        scale_cols = [col for col in scale_cols if col in data.columns]
        
        # 스케일링
        scaled_values = self.scaler.transform(data[scale_cols])
        
        # 스케일링된 컬럼명
        scaled_columns = [f'scaled_{col}' for col in scale_cols]
        
        # DataFrame 생성
        scaled_df = pd.DataFrame(scaled_values, columns=scaled_columns, index=data.index)
        
        # 원본 데이터와 병합
        result = pd.concat([data, scaled_df], axis=1)
        
        return result
    
    def create_sequences(self, data):
        """시퀀스 데이터 생성"""
        # 연속성 체크하여 데이터 분할
        time_diff = data.index.to_series().diff()
        split_points = time_diff > pd.Timedelta(minutes=1)
        segment_ids = split_points.cumsum()
        
        # 입력 특징
        input_features = [col for col in data.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']
        
        all_X = []
        all_y = []
        all_y_original = []
        all_times = []
        
        seq_length = self.config.get('seq_length', 30)
        
        # 각 연속 구간별로 시퀀스 생성
        for segment_id in segment_ids.unique():
            segment = data[segment_ids == segment_id]
            
            if len(segment) > seq_length:
                X_features = segment[input_features].values
                y_scaled = segment['scaled_FUTURE'].values
                y_original = segment['FUTURE'].values
                times = segment.index.values
                
                for i in range(len(segment) - seq_length):
                    all_X.append(X_features[i:i+seq_length])
                    all_y.append(y_scaled[i+seq_length])
                    all_y_original.append(y_original[i+seq_length])
                    all_times.append(times[i+seq_length])
        
        return (np.array(all_X), np.array(all_y), 
                np.array(all_y_original), np.array(all_times))
    
    def ensemble_predict(self, X):
        """앙상블 예측 수행"""
        predictions = {}
        weights = self.config.get('model_weights', {})
        
        # 각 모델별 예측
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X, verbose=0)
                predictions[model_name] = pred.flatten()
                logger.info(f"{model_name.upper()} 예측 완료")
            except Exception as e:
                logger.error(f"{model_name} 예측 실패: {str(e)}")
        
        if not predictions:
            raise ValueError("모든 모델 예측 실패")
        
        # 가중 평균 계산
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0.25)
            ensemble_pred += weight * pred
            total_weight += weight
        
        # 정규화
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred, predictions
    
    def inverse_scale_predictions(self, predictions):
        """예측값 역스케일링"""
        # 더미 데이터 생성 (스케일러 차원 맞추기)
        n_features = self.scaler.n_features_in_
        dummy_data = np.zeros((len(predictions), n_features))
        dummy_data[:, 0] = predictions  # TOTALCNT 위치에 예측값
        
        # 역변환
        inversed = self.scaler.inverse_transform(dummy_data)
        
        return inversed[:, 0]
    
    def detect_bottlenecks(self, predictions, threshold_percentile=85):
        """병목 구간 감지"""
        # 임계값 계산
        threshold = np.percentile(predictions, threshold_percentile)
        
        # 병목 구간 찾기
        bottlenecks = predictions > threshold
        bottleneck_indices = np.where(bottlenecks)[0]
        
        return bottlenecks, threshold, bottleneck_indices
    
    def run_prediction(self, data_path, save_results=True):
        """
        전체 예측 프로세스 실행
        
        Parameters:
        - data_path: 입력 데이터 경로
        - save_results: 결과 저장 여부
        
        Returns:
        - results: 예측 결과 딕셔너리
        """
        logger.info("\n" + "="*60)
        logger.info("예측 프로세스 시작")
        logger.info("="*60)
        
        # 1. 데이터 전처리
        data = self.preprocess_data(data_path)
        
        # 2. 스케일링
        scaled_data = self.scale_data(data)
        
        # 3. 시퀀스 생성
        X, y_scaled, y_original, times = self.create_sequences(scaled_data)
        logger.info(f"시퀀스 생성 완료 - shape: {X.shape}")
        
        # 4. 앙상블 예측
        ensemble_pred_scaled, individual_preds = self.ensemble_predict(X)
        
        # 5. 역스케일링
        ensemble_pred = self.inverse_scale_predictions(ensemble_pred_scaled)
        
        # 6. 병목 구간 감지
        bottlenecks, threshold, bottleneck_indices = self.detect_bottlenecks(ensemble_pred)
        
        logger.info(f"예측 완료 - 병목 구간: {len(bottleneck_indices)}개")
        
        # 7. 결과 정리
        results = {
            'times': times,
            'actual': y_original,
            'predictions': ensemble_pred,
            'individual_predictions': {
                name: self.inverse_scale_predictions(pred) 
                for name, pred in individual_preds.items()
            },
            'bottlenecks': bottlenecks,
            'bottleneck_threshold': threshold,
            'bottleneck_indices': bottleneck_indices
        }
        
        # 8. 성능 평가
        mae = np.mean(np.abs(y_original - ensemble_pred))
        mse = np.mean((y_original - ensemble_pred) ** 2)
        rmse = np.sqrt(mse)
        
        results['metrics'] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
        
        logger.info(f"성능 지표 - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")
        
        # 9. 결과 저장
        if save_results:
            self.save_results(results)
        
        return results
    
    def save_results(self, results):
        """예측 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 결과 DataFrame 생성
        df = pd.DataFrame({
            'TIME': results['times'],
            'ACTUAL': results['actual'],
            'PREDICTED': results['predictions'],
            'IS_BOTTLENECK': results['bottlenecks'],
            'ERROR': np.abs(results['actual'] - results['predictions'])
        })
        
        # 개별 모델 예측값 추가
        for model_name, preds in results['individual_predictions'].items():
            df[f'PRED_{model_name.upper()}'] = preds
        
        # CSV 저장
        output_path = f'results/hybrid_predictions_{timestamp}.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"예측 결과 저장: {output_path}")
        
        # 메트릭 저장
        metrics_path = f'results/metrics_{timestamp}.txt'
        with open(metrics_path, 'w') as f:
            f.write("하이브리드 모델 예측 성능\n")
            f.write("="*50 + "\n")
            f.write(f"MAE: {results['metrics']['mae']:.4f}\n")
            f.write(f"MSE: {results['metrics']['mse']:.4f}\n")
            f.write(f"RMSE: {results['metrics']['rmse']:.4f}\n")
            f.write(f"\n병목 구간 감지\n")
            f.write(f"임계값: {results['bottleneck_threshold']:.2f}\n")
            f.write(f"병목 구간 수: {len(results['bottleneck_indices'])}\n")
            f.write(f"병목 비율: {np.mean(results['bottlenecks'])*100:.1f}%\n")
        
        logger.info(f"메트릭 저장: {metrics_path}")
    
    def visualize_results(self, results, sample_size=500):
        """예측 결과 시각화"""
        # 샘플 크기 조정
        n_samples = min(sample_size, len(results['actual']))
        
        # 1. 예측 결과 비교
        plt.figure(figsize=(20, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(results['actual'][:n_samples], label='실제값', color='black', linewidth=2)
        plt.plot(results['predictions'][:n_samples], label='앙상블 예측', 
                color='red', linewidth=2, alpha=0.8)
        
        # 개별 모델 예측값
        colors = ['blue', 'green', 'orange', 'purple']
        for idx, (name, preds) in enumerate(results['individual_predictions'].items()):
            plt.plot(preds[:n_samples], label=f'{name.upper()}', 
                    color=colors[idx], alpha=0.5, linewidth=1)
        
        plt.title('반도체 물류량 예측 결과 (하이브리드 모델)', fontsize=16)
        plt.xlabel('시간 인덱스', fontsize=12)
        plt.ylabel('물류량 (TOTALCNT)', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # 2. 병목 구간 표시
        plt.subplot(2, 1, 2)
        plt.plot(results['predictions'][:n_samples], label='예측값', color='green')
        plt.axhline(y=results['bottleneck_threshold'], color='red', 
                   linestyle='--', label=f'병목 임계값 ({results["bottleneck_threshold"]:.0f})')
        
        # 병목 구간 하이라이트
        bottleneck_mask = results['bottlenecks'][:n_samples]
        if np.any(bottleneck_mask):
            bottleneck_points = np.where(bottleneck_mask)[0]
            plt.scatter(bottleneck_points, 
                       results['predictions'][:n_samples][bottleneck_points],
                       color='red', s=50, label='병목 구간', zorder=5)
        
        plt.title('병목 구간 예측', fontsize=16)
        plt.xlabel('시간 인덱스', fontsize=12)
        plt.ylabel('물류량 (TOTALCNT)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hybrid_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 오차 분포
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        errors = results['actual'] - results['predictions']
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.title('예측 오차 분포', fontsize=14)
        plt.xlabel('오차', fontsize=12)
        plt.ylabel('빈도', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.scatter(results['actual'], results['predictions'], 
                   alpha=0.5, s=10)
        plt.plot([results['actual'].min(), results['actual'].max()],
                [results['actual'].min(), results['actual'].max()],
                'r--', linewidth=2)
        plt.title('실제값 vs 예측값', fontsize=14)
        plt.xlabel('실제값', fontsize=12)
        plt.ylabel('예측값', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        # 시간별 MAE
        window_size = 60  # 60분 단위
        rolling_mae = pd.Series(np.abs(errors)).rolling(window=window_size).mean()
        plt.plot(rolling_mae)
        plt.title(f'시간별 MAE (60분 이동평균)', fontsize=14)
        plt.xlabel('시간 인덱스', fontsize=12)
        plt.ylabel('MAE', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hybrid_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# ===================================
# 메인 실행 함수
# ===================================

def main():
    """메인 실행 함수"""
    # 예측 시스템 초기화
    predictor = HybridPredictionSystem()
    
    # 예측 실행
    data_path = 'data/0730to31.csv'  # 예측할 데이터 경로
    
    try:
        # 예측 수행
        results = predictor.run_prediction(data_path, save_results=True)
        
        # 결과 시각화
        predictor.visualize_results(results)
        
        # 병목 구간 알림
        if len(results['bottleneck_indices']) > 0:
            logger.warning("\n" + "="*60)
            logger.warning("⚠️  병목 구간 감지 알림")
            logger.warning("="*60)
            logger.warning(f"총 {len(results['bottleneck_indices'])}개의 병목 구간이 예측되었습니다.")
            logger.warning(f"병목 임계값: {results['bottleneck_threshold']:.0f}")
            logger.warning("즉각적인 대응이 필요할 수 있습니다.")
        
        logger.info("\n✅ 예측 프로세스가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()