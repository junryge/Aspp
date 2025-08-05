"""
반도체 물류 예측을 위한 하이브리드 딥러닝 모델 - 실시간 예측 시스템
=========================================================
본 시스템은 학습된 하이브리드 모델(LSTM, RNN, GRU, ARIMA)을 사용하여
실시간으로 반도체 팹 간 물류량을 예측하고 병목 구간을 감지합니다.

주요 기능:
1. 학습된 하이브리드 모델 로드 및 앙상블 예측
2. 실시간 데이터 처리 및 예측
3. 병목 구간 사전 감지 및 알림
4. 예측 결과 시각화 및 저장

개발일: 2024년
버전: 1.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta
import joblib
import logging
import warnings
import json

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# ===================================
# 1. 환경 설정 및 초기화
# ===================================

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

logger.info("="*60)
logger.info("하이브리드 모델 기반 실시간 예측 시스템 시작")
logger.info("="*60)

# ===================================
# 2. 학습된 모델 및 설정 로드
# ===================================

class HybridPredictor:
    """하이브리드 모델 기반 예측 시스템"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.config = None
        self.arima_model = None
        
    def load_models(self):
        """학습된 모델들 로드"""
        logger.info("학습된 모델 로딩 중...")
        
        # 딥러닝 모델들 로드
        model_names = ['lstm', 'gru', 'rnn', 'bi_lstm']
        for model_name in model_names:
            try:
                model_path = f'model/{model_name}_final_hybrid.keras'
                if os.path.exists(model_path):
                    self.models[model_name] = load_model(model_path, compile=False)
                    logger.info(f"✓ {model_name.upper()} 모델 로드 완료")
                else:
                    # 대체 경로 시도
                    alt_path = f'model/{model_name}_best.keras'
                    if os.path.exists(alt_path):
                        self.models[model_name] = load_model(alt_path, compile=False)
                        logger.info(f"✓ {model_name.upper()} 모델 로드 완료 (대체 경로)")
                    else:
                        logger.warning(f"⚠ {model_name.upper()} 모델 파일을 찾을 수 없음")
            except Exception as e:
                logger.error(f"❌ {model_name.upper()} 모델 로드 실패: {str(e)}")
        
        # ARIMA 모델 로드
        try:
            arima_path = 'model/arima_model.pkl'
            if os.path.exists(arima_path):
                self.arima_model = joblib.load(arima_path)
                logger.info("✓ ARIMA 모델 로드 완료")
            else:
                logger.warning("⚠ ARIMA 모델 파일을 찾을 수 없음")
        except Exception as e:
            logger.error(f"❌ ARIMA 모델 로드 실패: {str(e)}")
        
        # 스케일러 로드
        try:
            scaler_path = 'scaler/standard_scaler_hybrid.pkl'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("✓ 스케일러 로드 완료")
            else:
                # 대체 경로 시도
                alt_scaler_path = 'scaler/StdScaler_s30f10_0731_2079936.save'
                if os.path.exists(alt_scaler_path):
                    self.scaler = joblib.load(alt_scaler_path)
                    logger.info("✓ 스케일러 로드 완료 (대체 경로)")
                else:
                    raise FileNotFoundError("스케일러 파일을 찾을 수 없습니다")
        except Exception as e:
            logger.error(f"❌ 스케일러 로드 실패: {str(e)}")
            raise
        
        # 설정 파일 로드
        try:
            config_path = 'results/training_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("✓ 학습 설정 로드 완료")
            else:
                # 기본 설정 사용
                self.config = {
                    'seq_length': 30,
                    'future_minutes': 10,
                    'model_weights': {
                        'lstm': 0.3,
                        'gru': 0.25,
                        'rnn': 0.15,
                        'bi_lstm': 0.3
                    },
                    'bottleneck_threshold': 2000  # 기본값
                }
                logger.warning("⚠ 설정 파일을 찾을 수 없음 - 기본 설정 사용")
        except Exception as e:
            logger.error(f"❌ 설정 파일 로드 실패: {str(e)}")
    
    def preprocess_data(self, data_path):
        """데이터 전처리"""
        logger.info(f"데이터 전처리 시작: {data_path}")
        
        # 데이터 로드
        Full_Data = pd.read_csv(data_path)
        
        # 시간 컬럼 datetime 변환
        Full_Data['CURRTIME'] = pd.to_datetime(Full_Data['CURRTIME'], format='%Y%m%d%H%M')
        Full_Data['TIME'] = pd.to_datetime(Full_Data['TIME'], format='%Y%m%d%H%M')
        
        # SUM 컬럼 제거
        columns_to_drop = [col for col in Full_Data.columns if 'SUM' in col]
        Full_Data = Full_Data.drop(columns=columns_to_drop)
        
        # 필요한 컬럼만 선택
        Full_Data = Full_Data[['CURRTIME', 'TOTALCNT', 'TIME']]
        Full_Data.set_index('CURRTIME', inplace=True)
        
        # 타겟 변수 생성 (10분 후 예측)
        Full_Data['FUTURE'] = pd.NA
        future_minutes = self.config.get('future_minutes', 10)
        
        for i in Full_Data.index:
            future_time = i + pd.Timedelta(minutes=future_minutes)
            if (future_time <= Full_Data.index.max()) & (future_time in Full_Data.index):
                Full_Data.loc[i, 'FUTURE'] = Full_Data.loc[future_time, 'TOTALCNT']
        
        Full_Data.dropna(subset=['FUTURE'], inplace=True)
        
        # 특징 엔지니어링
        logger.info("특징 엔지니어링 수행 중...")
        Full_Data['hour'] = Full_Data.index.hour
        Full_Data['dayofweek'] = Full_Data.index.dayofweek
        Full_Data['is_weekend'] = (Full_Data.index.dayofweek >= 5).astype(int)
        
        # 이동 평균
        Full_Data['MA_5'] = Full_Data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
        Full_Data['MA_10'] = Full_Data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
        Full_Data['MA_30'] = Full_Data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
        
        # 변동성
        Full_Data['STD_5'] = Full_Data['TOTALCNT'].rolling(window=5, min_periods=1).std()
        Full_Data['STD_10'] = Full_Data['TOTALCNT'].rolling(window=10, min_periods=1).std()
        
        # 변화율
        Full_Data['change_rate'] = Full_Data['TOTALCNT'].pct_change()
        Full_Data['change_rate_5'] = Full_Data['TOTALCNT'].pct_change(5)
        
        # 결측값 처리
        Full_Data = Full_Data.fillna(method='ffill').fillna(0)
        
        logger.info(f"전처리 완료 - 데이터 shape: {Full_Data.shape}")
        
        return Full_Data
    
    def scale_data(self, data):
        """데이터 스케일링"""
        # 스케일링할 컬럼
        scale_columns = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
        scale_columns = [col for col in scale_columns if col in data.columns]
        
        # 스케일링 적용
        scaled_data = self.scaler.transform(data[scale_columns])
        
        # 스케일링된 데이터프레임 생성
        scaled_columns = [f'scaled_{col}' for col in scale_columns]
        scaled_df = pd.DataFrame(scaled_data, columns=scaled_columns, index=data.index)
        
        # 원본 데이터와 병합
        result = pd.merge(data, scaled_df, left_index=True, right_index=True, how='left')
        
        return result
    
    def create_sequences(self, data):
        """시퀀스 데이터 생성"""
        seq_length = self.config.get('seq_length', 30)
        
        # 연속성 확인하여 데이터 분할
        time_diff = data.index.to_series().diff()
        split_points = time_diff > pd.Timedelta(minutes=1)
        segment_ids = split_points.cumsum()
        
        # 입력 특징 선택
        input_features = [col for col in data.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']
        
        all_X = []
        all_y = []
        all_times = []
        all_future_vals = []
        
        # 각 세그먼트별로 시퀀스 생성
        for segment_id in segment_ids.unique():
            segment = data[segment_ids == segment_id]
            
            if len(segment) > seq_length:
                X_data = segment[input_features].values
                y_data = segment['scaled_FUTURE'].values
                time_data = segment['TIME'].values
                future_data = segment['FUTURE'].values
                
                for i in range(len(segment) - seq_length):
                    all_X.append(X_data[i:i+seq_length])
                    all_y.append(y_data[i+seq_length])
                    all_times.append(time_data[i+seq_length])
                    all_future_vals.append(future_data[i+seq_length])
        
        return (np.array(all_X), np.array(all_y), 
                np.array(all_times), np.array(all_future_vals))
    
    def ensemble_predict(self, X_data):
        """앙상블 예측 수행"""
        weights = self.config.get('model_weights', {
            'lstm': 0.3,
            'gru': 0.25,
            'rnn': 0.15,
            'bi_lstm': 0.3
        })
        
        predictions = {}
        ensemble_pred = np.zeros(len(X_data))
        
        # 각 모델별 예측
        for model_name, model in self.models.items():
            if model is not None:
                logger.info(f"{model_name.upper()} 예측 수행 중...")
                pred = model.predict(X_data, verbose=0).flatten()
                predictions[model_name] = pred
                
                # 가중 평균에 추가
                weight = weights.get(model_name, 0.25)
                ensemble_pred += weight * pred
        
        # 가중치 정규화 (사용 가능한 모델만)
        total_weight = sum(weights.get(name, 0.25) for name in predictions.keys())
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred, predictions
    
    def inverse_scale_predictions(self, predictions):
        """예측값 역스케일링"""
        # 더미 배열 생성 (스케일러가 여러 컬럼을 기대하는 경우)
        n_features = self.scaler.n_features_in_
        dummy_array = np.zeros((len(predictions), n_features))
        dummy_array[:, 0] = predictions  # 첫 번째 컬럼에 예측값 넣기
        
        # 역변환
        inverse_scaled = self.scaler.inverse_transform(dummy_array)
        
        return inverse_scaled[:, 0]
    
    def detect_bottlenecks(self, predictions):
        """병목 구간 감지"""
        threshold = self.config.get('bottleneck_threshold', 2000)
        
        bottlenecks = predictions > threshold
        bottleneck_indices = np.where(bottlenecks)[0]
        
        return bottlenecks, bottleneck_indices, threshold
    
    def run_prediction(self, data_path):
        """전체 예측 프로세스 실행"""
        # 모델 로드
        self.load_models()
        
        # 데이터 전처리
        processed_data = self.preprocess_data(data_path)
        
        # 스케일링
        scaled_data = self.scale_data(processed_data)
        
        # 시퀀스 생성
        X_seq, y_seq, time_seq, future_seq = self.create_sequences(scaled_data)
        
        logger.info(f"예측 데이터 준비 완료 - shape: {X_seq.shape}")
        
        # 앙상블 예측
        ensemble_pred, individual_preds = self.ensemble_predict(X_seq)
        
        # 역스케일링
        ensemble_pred_original = self.inverse_scale_predictions(ensemble_pred)
        
        # 병목 감지
        bottlenecks, bottleneck_indices, threshold = self.detect_bottlenecks(ensemble_pred_original)
        
        # 결과 정리
        results = {
            'predictions': ensemble_pred_original,
            'actual_values': future_seq,
            'times': time_seq,
            'bottlenecks': bottlenecks,
            'bottleneck_indices': bottleneck_indices,
            'threshold': threshold,
            'individual_predictions': individual_preds
        }
        
        return results
    
    def evaluate_predictions(self, predictions, actual_values):
        """예측 성능 평가"""
        mae = mean_absolute_error(actual_values, predictions)
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_values, predictions)
        
        # 오차 통계
        errors = np.abs(actual_values - predictions)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'median_error': np.median(errors),
            'mean_error': np.mean(errors)
        }
        
        return metrics
    
    def save_results(self, results, metrics):
        """예측 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 결과 디렉토리 생성
        os.makedirs('prediction_results', exist_ok=True)
        
        # 예측 결과 DataFrame 생성
        result_df = pd.DataFrame({
            'EVENT_DT': pd.Series(results['times']),
            'ACTUAL_VALUE': results['actual_values'],
            'PREDICT_DT': pd.Series(results['times']) + timedelta(minutes=10),
            'PREDICTED_VALUE': results['predictions'],
            'IS_BOTTLENECK': results['bottlenecks'],
            'ERROR': np.abs(results['actual_values'] - results['predictions'])
        })
        
        # CSV 저장
        csv_path = f'prediction_results/hybrid_predictions_{timestamp}.csv'
        result_df.to_csv(csv_path, index=False)
        logger.info(f"예측 결과 저장: {csv_path}")
        
        # 성능 지표 저장
        metrics_path = f'prediction_results/hybrid_metrics_{timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"성능 지표 저장: {metrics_path}")
        
        return result_df
    
    def visualize_results(self, results, metrics):
        """결과 시각화"""
        # 샘플 크기 설정
        sample_size = min(500, len(results['predictions']))
        
        # 1. 예측 결과 비교
        plt.figure(figsize=(20, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(results['actual_values'][:sample_size], label='실제값', color='blue', linewidth=2)
        plt.plot(results['predictions'][:sample_size], label='예측값', color='red', linewidth=1.5)
        plt.axhline(y=results['threshold'], color='orange', linestyle='--', 
                   label=f'병목 임계값 ({results["threshold"]:.0f})')
        
        # 병목 구간 표시
        bottleneck_mask = results['bottlenecks'][:sample_size]
        if np.any(bottleneck_mask):
            bottleneck_points = np.where(bottleneck_mask)[0]
            plt.scatter(bottleneck_points, results['predictions'][bottleneck_points], 
                       color='red', s=50, marker='o', label='병목 예측', zorder=5)
        
        plt.title('하이브리드 모델 예측 결과', fontsize=16)
        plt.xlabel('시간 인덱스', fontsize=12)
        plt.ylabel('물류량', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 오차 분포
        plt.subplot(2, 1, 2)
        errors = np.abs(results['actual_values'][:sample_size] - results['predictions'][:sample_size])
        plt.plot(errors, label='절대 오차', color='green')
        plt.axhline(y=metrics['mae'], color='red', linestyle='--', 
                   label=f'평균 절대 오차 (MAE: {metrics["mae"]:.2f})')
        
        plt.title('예측 오차 분포', fontsize=16)
        plt.xlabel('시간 인덱스', fontsize=12)
        plt.ylabel('절대 오차', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'prediction_results/hybrid_prediction_plot_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. 개별 모델 성능 비교 (가능한 경우)
        if results.get('individual_predictions'):
            plt.figure(figsize=(15, 8))
            
            # 각 모델의 MAE 계산
            model_metrics = {}
            for model_name, preds in results['individual_predictions'].items():
                # 역스케일링
                preds_original = self.inverse_scale_predictions(preds)
                mae = mean_absolute_error(results['actual_values'], preds_original)
                model_metrics[model_name] = mae
            
            # 앙상블 추가
            model_metrics['ensemble'] = metrics['mae']
            
            # 막대 그래프
            models = list(model_metrics.keys())
            mae_values = list(model_metrics.values())
            
            plt.bar(models, mae_values, color=['blue', 'green', 'red', 'orange', 'purple'])
            plt.title('모델별 MAE 비교', fontsize=16)
            plt.xlabel('모델', fontsize=12)
            plt.ylabel('MAE', fontsize=12)
            
            # 값 표시
            for i, (model, mae) in enumerate(model_metrics.items()):
                plt.text(i, mae + 1, f'{mae:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'prediction_results/model_comparison_{timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()

# ===================================
# 메인 실행 함수
# ===================================

def main():
    """메인 실행 함수"""
    # 예측 시스템 초기화
    predictor = HybridPredictor()
    
    # 데이터 경로 설정
    data_path = 'data/0730to31.csv'  # 실제 데이터 경로로 변경
    
    # 예측 실행
    logger.info("예측 프로세스 시작...")
    results = predictor.run_prediction(data_path)
    
    # 성능 평가
    logger.info("성능 평가 중...")
    metrics = predictor.evaluate_predictions(
        results['predictions'], 
        results['actual_values']
    )
    
    # 결과 출력
    logger.info("\n" + "="*60)
    logger.info("예측 성능 요약")
    logger.info("="*60)
    logger.info(f"평균 절대 오차 (MAE): {metrics['mae']:.2f}")
    logger.info(f"평균 제곱 오차 (MSE): {metrics['mse']:.2f}")
    logger.info(f"평균 제곱근 오차 (RMSE): {metrics['rmse']:.2f}")
    logger.info(f"결정 계수 (R²): {metrics['r2']:.4f}")
    logger.info(f"최대 오차: {metrics['max_error']:.2f}")
    logger.info(f"최소 오차: {metrics['min_error']:.2f}")
    logger.info(f"중간값 오차: {metrics['median_error']:.2f}")
    
    # 병목 구간 분석
    n_bottlenecks = np.sum(results['bottlenecks'])
    bottleneck_ratio = n_bottlenecks / len(results['predictions']) * 100
    
    logger.info("\n" + "="*60)
    logger.info("병목 구간 분석")
    logger.info("="*60)
    logger.info(f"병목 임계값: {results['threshold']:.0f}")
    logger.info(f"예측된 병목 구간: {n_bottlenecks}개 ({bottleneck_ratio:.1f}%)")
    
    # 결과 저장
    logger.info("\n결과 저장 중...")
    result_df = predictor.save_results(results, metrics)
    
    # 시각화
    logger.info("결과 시각화 중...")
    predictor.visualize_results(results, metrics)
    
    logger.info("\n" + "="*60)
    logger.info("예측 프로세스 완료!")
    logger.info("="*60)
    
    return results, metrics, result_df

# ===================================
# 실시간 예측을 위한 추가 함수
# ===================================

def predict_realtime(new_data_path):
    """실시간 데이터에 대한 예측 수행"""
    predictor = HybridPredictor()
    
    # 모델이 이미 로드되어 있지 않으면 로드
    if not predictor.models:
        predictor.load_models()
    
    # 새 데이터 처리 및 예측
    try:
        # 데이터 전처리
        processed_data = predictor.preprocess_data(new_data_path)
        
        # 마지막 30분 데이터만 사용 (실시간 예측)
        last_30_rows = processed_data.tail(30)
        
        if len(last_30_rows) < 30:
            logger.warning("실시간 예측을 위한 충분한 데이터가 없습니다 (최소 30개 필요)")
            return None
        
        # 스케일링
        scaled_data = predictor.scale_data(last_30_rows)
        
        # 입력 특징 선택
        input_features = [col for col in scaled_data.columns 
                         if col.startswith('scaled_') and col != 'scaled_FUTURE']
        
        # 시퀀스 생성 (마지막 30개 데이터)
        X_realtime = scaled_data[input_features].values.reshape(1, 30, -1)
        
        # 예측
        ensemble_pred, _ = predictor.ensemble_predict(X_realtime)
        
        # 역스케일링
        prediction = predictor.inverse_scale_predictions(ensemble_pred)[0]
        
        # 병목 여부 확인
        is_bottleneck = prediction > predictor.config.get('bottleneck_threshold', 2000)
        
        # 결과 반환
        current_time = processed_data.index[-1]
        predict_time = current_time + timedelta(minutes=10)
        
        result = {
            'current_time': current_time,
            'predict_time': predict_time,
            'current_value': processed_data['TOTALCNT'].iloc[-1],
            'predicted_value': prediction,
            'is_bottleneck': is_bottleneck,
            'confidence': 0.95 if len(predictor.models) >= 3 else 0.8  # 모델 수에 따른 신뢰도
        }
        
        logger.info(f"\n실시간 예측 결과:")
        logger.info(f"현재 시간: {current_time}")
        logger.info(f"예측 시간: {predict_time}")
        logger.info(f"현재 물류량: {result['current_value']:.0f}")
        logger.info(f"예측 물류량: {prediction:.0f}")
        logger.info(f"병목 예상: {'예' if is_bottleneck else '아니오'}")
        
        return result
        
    except Exception as e:
        logger.error(f"실시간 예측 실패: {str(e)}")
        return None

# ===================================
# 스크립트 실행
# ===================================

if __name__ == "__main__":
    # 전체 예측 실행
    results, metrics, result_df = main()
    
    # 실시간 예측 예시 (옵션)
    # realtime_result = predict_realtime('data/realtime_data.csv')