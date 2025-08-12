"""
반도체 물류 급증 예측을 위한 실시간 예측 시스템 v3.0
=========================================================
본 시스템은 학습된 이중 출력 하이브리드 모델을 사용하여
실시간으로 반도체 팹 간 물류량을 예측하고 특히 TOTALCNT > 1400
급증 구간을 사전에 감지합니다.

주요 기능:
1. 이중 출력 모델 로드 (수치 예측 + 급증 확률)
2. 개별 구간 데이터를 활용한 정밀 예측
3. 급증 구간 사전 감지 및 경고
4. 실시간 예측 결과 시각화

개발일: 2024년
버전: 3.0 (급증 예측 특화)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import sys
import os
import platform
from datetime import datetime, timedelta
import joblib
import logging
import warnings
import json

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# ===================================
# 환경 설정 및 초기화
# ===================================

# 한글 폰트 설정
def set_korean_font():
    """운영체제별 한글 폰트 자동 설정"""
    system = platform.system()
    
    if system == 'Windows':
        font_family = 'Malgun Gothic'
    elif system == 'Darwin':
        font_family = 'AppleGothic'
    else:
        font_family = 'NanumGothic'
    
    plt.rcParams['font.family'] = font_family
    plt.rcParams['axes.unicode_minus'] = False
    return True

USE_KOREAN = set_korean_font()

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
        logging.FileHandler('spike_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("급증 예측 실시간 시스템 v3.0 시작")
logger.info("="*60)

# ===================================
# 급증 예측 시스템 클래스
# ===================================

class SpikePredictor:
    """급증 예측 특화 실시간 예측 시스템"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.config = None
        self.spike_threshold = 1400  # 급증 임계값
        
    def load_models(self):
        """학습된 이중 출력 모델 로드"""
        logger.info("학습된 급증 예측 모델 로딩 중...")
        
        # v3 이중 출력 모델들 로드
        model_names = ['dual_lstm', 'dual_gru', 'dual_rnn', 'dual_bilstm']
        for model_name in model_names:
            try:
                # v3 모델 경로
                model_path = f'model_v3/{model_name}_final.keras'
                if os.path.exists(model_path):
                    self.models[model_name] = load_model(model_path, compile=False)
                    logger.info(f"✓ {model_name.upper()} 모델 로드 완료")
                else:
                    # 기존 모델 폴백
                    alt_path = f'model/{model_name.replace("dual_", "")}_final_hybrid.keras'
                    if os.path.exists(alt_path):
                        logger.warning(f"⚠ {model_name} v3 모델 없음, 기존 모델 사용")
            except Exception as e:
                logger.error(f"❌ {model_name.upper()} 모델 로드 실패: {str(e)}")
        
        # 스케일러 로드
        try:
            scaler_paths = [
                'scaler_v3/scaler_v3.pkl',
                'scaler/standard_scaler_hybrid.pkl'
            ]
            for scaler_path in scaler_paths:
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    logger.info("✓ 스케일러 로드 완료")
                    break
        except Exception as e:
            logger.error(f"❌ 스케일러 로드 실패: {str(e)}")
            raise
        
        # 설정 파일 로드
        try:
            config_paths = [
                'results_v3/training_config.json',
                'results/training_config.json'
            ]
            for config_path in config_paths:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    break
            
            if not self.config:
                self.config = {
                    'seq_length': 30,
                    'future_minutes': 10,
                    'spike_threshold': 1400,
                    'model_weights': {
                        'dual_lstm': 0.35,
                        'dual_gru': 0.25,
                        'dual_rnn': 0.15,
                        'dual_bilstm': 0.25
                    }
                }
            logger.info("✓ 설정 파일 로드 완료")
        except Exception as e:
            logger.error(f"❌ 설정 파일 로드 실패: {str(e)}")
    
    def preprocess_data(self, data_path):
        """급증 예측을 위한 데이터 전처리"""
        logger.info(f"데이터 전처리 시작: {data_path}")
        
        # 데이터 로드
        data = pd.read_csv(data_path)
        
        # 시간 컬럼 변환
        data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
        data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
        
        # 필요한 컬럼 선택 (개별 구간 포함)
        required_columns = ['CURRTIME', 'TOTALCNT', 'M14AM10A', 'M10AM14A', 
                          'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A', 'TIME']
        available_columns = [col for col in required_columns if col in data.columns]
        data = data[available_columns]
        data.set_index('CURRTIME', inplace=True)
        
        # FUTURE 컬럼 생성 (10분 후)
        data['FUTURE'] = pd.NA
        future_minutes = self.config.get('future_minutes', 10)
        
        for i in data.index:
            future_time = i + pd.Timedelta(minutes=future_minutes)
            if (future_time <= data.index.max()) & (future_time in data.index):
                data.loc[i, 'FUTURE'] = data.loc[future_time, 'TOTALCNT']
        
        data.dropna(subset=['FUTURE'], inplace=True)
        
        # 급증 라벨 생성
        data['future_spike'] = (data['FUTURE'] > self.spike_threshold).astype(int)
        
        logger.info(f"급증 비율: {data['future_spike'].mean():.2%}")
        
        # 개별 구간 특징 생성
        segment_columns = ['M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A']
        available_segments = [col for col in segment_columns if col in data.columns]
        
        for col in available_segments:
            # 비율
            data[f'{col}_ratio'] = data[col] / (data['TOTALCNT'] + 1e-6)
            # 변화율
            data[f'{col}_change_10'] = data[col].pct_change(10).fillna(0)
            # 이동평균
            data[f'{col}_MA5'] = data[col].rolling(window=5, min_periods=1).mean()
        
        # 급증 신호 특징
        if 'M14AM14B' in data.columns:
            data['M14AM14B_spike_signal'] = (data['M14AM14B_change_10'] > 0.5).astype(int)
        if 'M16M14A' in data.columns:
            data['M16M14A_spike_signal'] = (data['M16M14A_change_10'] > 0.5).astype(int)
        
        # 기본 특징
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        data['MA_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
        data['MA_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
        data['MA_30'] = data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
        data['STD_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).std()
        data['STD_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).std()
        data['change_rate'] = data['TOTALCNT'].pct_change()
        data['change_rate_5'] = data['TOTALCNT'].pct_change(5)
        
        # 결측값 처리
        data = data.ffill().fillna(0)
        
        logger.info(f"전처리 완료 - 데이터 shape: {data.shape}")
        
        return data
    
    def scale_data(self, data):
        """데이터 스케일링"""
        # 스케일링할 컬럼
        segment_columns = ['M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A']
        available_segments = [col for col in segment_columns if col in data.columns]
        
        scaling_columns = ['TOTALCNT', 'FUTURE'] + available_segments
        scaling_columns += [col for col in data.columns if 'MA' in col or 'STD' in col]
        scaling_columns += [f'{seg}_MA5' for seg in available_segments if f'{seg}_MA5' in data.columns]
        scaling_columns = list(set([col for col in scaling_columns if col in data.columns]))
        
        # 스케일러가 기대하는 컬럼 확인
        if hasattr(self.scaler, 'feature_names_in_'):
            expected_columns = list(self.scaler.feature_names_in_)
            scaling_columns = [col for col in expected_columns if col in data.columns]
        
        # 스케일링 적용
        scaled_data = self.scaler.transform(data[scaling_columns])
        scaled_df = pd.DataFrame(scaled_data, columns=[f'scaled_{col}' for col in scaling_columns], 
                               index=data.index)
        
        # 비스케일 특징
        non_scaled_features = [col for col in data.columns 
                             if ('ratio' in col or 'change' in col or 'signal' in col or 
                                 col in ['hour', 'dayofweek', 'is_weekend', 'future_spike'])]
        
        # 최종 데이터
        result = pd.concat([data[non_scaled_features], scaled_df], axis=1)
        
        # TIME과 FUTURE 원본값 보존
        result['TIME'] = data['TIME']
        result['FUTURE'] = data['FUTURE']
        
        return result
    
    def create_sequences(self, data):
        """시퀀스 데이터 생성"""
        seq_length = self.config.get('seq_length', 30)
        
        # 연속성 확인
        time_diff = data.index.to_series().diff()
        split_points = time_diff > pd.Timedelta(minutes=1)
        segment_ids = split_points.cumsum()
        
        # 입력 특징 선택
        input_features = [col for col in data.columns 
                         if col not in ['scaled_FUTURE', 'future_spike', 'TIME', 'FUTURE']]
        
        all_X = []
        all_y_reg = []
        all_y_cls = []
        all_times = []
        all_future_vals = []
        
        # 각 세그먼트별로 시퀀스 생성
        for segment_id in segment_ids.unique():
            segment = data[segment_ids == segment_id]
            
            if len(segment) > seq_length:
                X_data = segment[input_features].values
                y_reg_data = segment['scaled_FUTURE'].values if 'scaled_FUTURE' in segment.columns else segment['FUTURE'].values
                y_cls_data = segment['future_spike'].values
                time_data = segment['TIME'].values
                future_data = segment['FUTURE'].values
                
                for i in range(len(segment) - seq_length):
                    all_X.append(X_data[i:i+seq_length])
                    all_y_reg.append(y_reg_data[i+seq_length])
                    all_y_cls.append(y_cls_data[i+seq_length])
                    all_times.append(time_data[i+seq_length])
                    all_future_vals.append(future_data[i+seq_length])
        
        return (np.array(all_X), np.array(all_y_reg), np.array(all_y_cls),
                np.array(all_times), np.array(all_future_vals))
    
    def enhanced_ensemble_predict(self, X_data):
        """급증 예측 강화 앙상블"""
        weights = self.config.get('model_weights', {
            'dual_lstm': 0.35,
            'dual_gru': 0.25,
            'dual_rnn': 0.15,
            'dual_bilstm': 0.25
        })
        
        regression_preds = {}
        spike_preds = {}
        ensemble_reg = np.zeros(len(X_data))
        ensemble_spike = np.zeros(len(X_data))
        total_weight = 0
        
        # 각 모델별 예측
        for model_name, model in self.models.items():
            if model is not None:
                logger.info(f"{model_name.upper()} 예측 수행 중...")
                pred = model.predict(X_data, verbose=0)
                
                # 이중 출력 처리
                if isinstance(pred, list) and len(pred) == 2:
                    regression_preds[model_name] = pred[0].flatten()
                    spike_preds[model_name] = pred[1].flatten()
                else:
                    # 단일 출력 모델
                    regression_preds[model_name] = pred.flatten()
                    spike_preds[model_name] = np.zeros_like(pred.flatten())
                
                # 가중 평균
                weight = weights.get(model_name, 0.25)
                ensemble_reg += weight * regression_preds[model_name]
                ensemble_spike += weight * spike_preds[model_name]
                total_weight += weight
        
        # 가중치 정규화
        if total_weight > 0:
            ensemble_reg /= total_weight
            ensemble_spike /= total_weight
        
        # 급증 확률이 높으면 예측값 상향 조정
        spike_mask = ensemble_spike > 0.7
        ensemble_reg[spike_mask] *= 1.15
        
        return ensemble_reg, ensemble_spike, regression_preds, spike_preds
    
    def inverse_scale_predictions(self, predictions):
        """예측값 역스케일링"""
        if hasattr(self.scaler, 'feature_names_in_'):
            feature_names = list(self.scaler.feature_names_in_)
            n_features = len(feature_names)
            dummy = np.zeros((len(predictions), n_features))
            
            if 'FUTURE' in feature_names:
                future_idx = feature_names.index('FUTURE')
            else:
                future_idx = 0
            
            dummy[:, future_idx] = predictions
            return self.scaler.inverse_transform(dummy)[:, future_idx]
        else:
            # 기본 역스케일링
            n_features = self.scaler.n_features_in_
            dummy = np.zeros((len(predictions), n_features))
            dummy[:, 0] = predictions
            return self.scaler.inverse_transform(dummy)[:, 0]
    
    def print_spike_prediction_details(self, predictions, spike_probs, actual_values, times):
        """급증 예측 상세 출력"""
        print("\n" + "="*100)
        print("10분 후 급증 예측 상세 정보")
        print("="*100)
        
        # 최근 20개 예측 결과
        n_display = min(20, len(predictions))
        
        print(f"\n최근 {n_display}개 예측 결과:")
        print("-"*100)
        print(f"{'현재 시간':^20} | {'현재값':^10} | {'10분후 예측':^12} | {'실제값':^10} | {'급증확률':^10} | {'급증예측':^10} | {'실제급증':^10}")
        print("-"*100)
        
        for i in range(-n_display, 0):
            current_time = pd.Timestamp(times[i])
            predict_time = current_time + timedelta(minutes=10)
            current_val = actual_values[i]
            predicted_val = predictions[i]
            actual_val = actual_values[i]
            spike_prob = spike_probs[i]
            spike_pred = "★예★" if spike_prob > 0.5 else "아니오"
            actual_spike = "★예★" if actual_val > self.spike_threshold else "아니오"
            
            # 급증 예측 맞춘 경우 강조
            if (spike_prob > 0.5 and actual_val > self.spike_threshold):
                print(f"\033[92m{current_time.strftime('%Y-%m-%d %H:%M'):^20} | {current_val:^10.0f} | "
                      f"{predicted_val:^12.0f} | {actual_val:^10.0f} | {spike_prob:^10.1%} | "
                      f"{spike_pred:^10} | {actual_spike:^10}\033[0m")
            elif (spike_prob > 0.5 and actual_val <= self.spike_threshold):
                # 오탐
                print(f"\033[93m{current_time.strftime('%Y-%m-%d %H:%M'):^20} | {current_val:^10.0f} | "
                      f"{predicted_val:^12.0f} | {actual_val:^10.0f} | {spike_prob:^10.1%} | "
                      f"{spike_pred:^10} | {actual_spike:^10}\033[0m")
            elif (spike_prob <= 0.5 and actual_val > self.spike_threshold):
                # 미탐
                print(f"\033[91m{current_time.strftime('%Y-%m-%d %H:%M'):^20} | {current_val:^10.0f} | "
                      f"{predicted_val:^12.0f} | {actual_val:^10.0f} | {spike_prob:^10.1%} | "
                      f"{spike_pred:^10} | {actual_spike:^10}\033[0m")
            else:
                print(f"{current_time.strftime('%Y-%m-%d %H:%M'):^20} | {current_val:^10.0f} | "
                      f"{predicted_val:^12.0f} | {actual_val:^10.0f} | {spike_prob:^10.1%} | "
                      f"{spike_pred:^10} | {actual_spike:^10}")
        
        print("-"*100)
        
        # 급증 예측 성능 요약
        spike_pred_mask = spike_probs[-n_display:] > 0.5
        actual_spike_mask = actual_values[-n_display:] > self.spike_threshold
        
        tp = np.sum(spike_pred_mask & actual_spike_mask)
        fp = np.sum(spike_pred_mask & ~actual_spike_mask)
        fn = np.sum(~spike_pred_mask & actual_spike_mask)
        tn = np.sum(~spike_pred_mask & ~actual_spike_mask)
        
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        
        print(f"\n최근 {n_display}개 급증 예측 성능:")
        print(f"  • 정확히 예측한 급증: {tp}건")
        print(f"  • 오탐 (잘못된 급증 예측): {fp}건")
        print(f"  • 미탐 (놓친 급증): {fn}건")
        print(f"  • 정밀도: {precision:.1f}%")
        print(f"  • 재현율: {recall:.1f}%")
        
        # 가장 최근 예측 강조
        print("\n" + "="*100)
        print("가장 최근 10분 후 급증 예측")
        print("="*100)
        
        latest_time = pd.Timestamp(times[-1])
        latest_predict_time = latest_time + timedelta(minutes=10)
        
        print(f"현재 시간: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"예측 대상 시간: {latest_predict_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"현재 물류량: {actual_values[-1]:.0f}")
        print(f"10분 후 예측 물류량: {predictions[-1]:.0f}")
        print(f"급증 확률: {spike_probs[-1]:.1%}")
        print(f"급증 예측: {'★★★ 예 ★★★' if spike_probs[-1] > 0.5 else '아니오'}")
        
        if spike_probs[-1] > 0.7:
            print(f"\n🚨 경고: 10분 후 급증 발생 가능성 매우 높음! (확률: {spike_probs[-1]:.1%})")
            print(f"   예측값({predictions[-1]:.0f}) > 임계값({self.spike_threshold})")
        elif spike_probs[-1] > 0.5:
            print(f"\n⚠️  주의: 10분 후 급증 발생 가능성 있음 (확률: {spike_probs[-1]:.1%})")
        
        print("="*100 + "\n")
    
    def run_prediction(self, data_path):
        """전체 예측 프로세스 실행"""
        # 모델 로드
        self.load_models()
        
        # 데이터 전처리
        processed_data = self.preprocess_data(data_path)
        
        # 스케일링
        scaled_data = self.scale_data(processed_data)
        
        # 시퀀스 생성
        X_seq, y_reg_seq, y_cls_seq, time_seq, future_seq = self.create_sequences(scaled_data)
        
        logger.info(f"예측 데이터 준비 완료 - shape: {X_seq.shape}")
        
        # 앙상블 예측
        ensemble_reg, ensemble_spike, reg_preds, spike_preds = self.enhanced_ensemble_predict(X_seq)
        
        # 역스케일링
        ensemble_pred_original = self.inverse_scale_predictions(ensemble_reg)
        
        # 급증 예측 상세 출력
        self.print_spike_prediction_details(
            ensemble_pred_original,
            ensemble_spike,
            future_seq,
            time_seq
        )
        
        # 결과 정리
        results = {
            'predictions': ensemble_pred_original,
            'spike_probabilities': ensemble_spike,
            'actual_values': future_seq,
            'actual_spikes': y_cls_seq,
            'times': time_seq,
            'individual_regression': reg_preds,
            'individual_spike': spike_preds
        }
        
        return results
    
    def evaluate_spike_predictions(self, predictions, spike_probs, actual_values, actual_spikes):
        """급증 예측 성능 평가"""
        # 수치 예측 성능
        mae = mean_absolute_error(actual_values, predictions)
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_values, predictions)
        
        # 급증 예측 성능
        spike_pred_binary = (spike_probs > 0.5).astype(int)
        
        tp = np.sum((spike_pred_binary == 1) & (actual_spikes == 1))
        fp = np.sum((spike_pred_binary == 1) & (actual_spikes == 0))
        fn = np.sum((spike_pred_binary == 0) & (actual_spikes == 1))
        tn = np.sum((spike_pred_binary == 0) & (actual_spikes == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'spike_accuracy': accuracy,
            'spike_precision': precision,
            'spike_recall': recall,
            'spike_f1': f1,
            'spike_tp': tp,
            'spike_fp': fp,
            'spike_fn': fn,
            'spike_tn': tn
        }
        
        return metrics
    
    def save_results(self, results, metrics):
        """예측 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 결과 디렉토리 생성
        os.makedirs('spike_prediction_results', exist_ok=True)
        
        # 예측 결과 DataFrame 생성
        result_df = pd.DataFrame({
            'EVENT_DT': pd.Series(results['times']),
            'ACTUAL_VALUE': results['actual_values'],
            'PREDICT_DT': pd.Series(results['times']) + timedelta(minutes=10),
            'PREDICTED_VALUE': results['predictions'],
            'SPIKE_PROBABILITY': results['spike_probabilities'],
            'SPIKE_PREDICTED': (results['spike_probabilities'] > 0.5).astype(int),
            'SPIKE_ACTUAL': results['actual_spikes'],
            'ERROR': np.abs(results['actual_values'] - results['predictions'])
        })
        
        # CSV 저장
        csv_path = f'spike_prediction_results/spike_predictions_{timestamp}.csv'
        result_df.to_csv(csv_path, index=False)
        logger.info(f"예측 결과 저장: {csv_path}")
        
        # 성능 지표 저장
        metrics_path = f'spike_prediction_results/spike_metrics_{timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        logger.info(f"성능 지표 저장: {metrics_path}")
        
        return result_df
    
    def visualize_spike_results(self, results, metrics):
        """급증 예측 결과 시각화"""
        # 샘플 크기
        sample_size = min(500, len(results['predictions']))
        
        # 그림 크기 설정
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 예측값과 실제값 비교 + 급증 구간 표시
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(results['actual_values'][:sample_size], label='실제값', color='blue', linewidth=2)
        ax1.plot(results['predictions'][:sample_size], label='예측값', color='red', linewidth=1.5)
        ax1.axhline(y=self.spike_threshold, color='orange', linestyle='--', 
                   label=f'급증 임계값 ({self.spike_threshold})')
        
        # 급증 구간 하이라이트
        actual_spike_mask = results['actual_spikes'][:sample_size] == 1
        if np.any(actual_spike_mask):
            spike_indices = np.where(actual_spike_mask)[0]
            ax1.scatter(spike_indices, results['actual_values'][:sample_size][spike_indices],
                       color='darkred', s=50, marker='o', label='실제 급증', zorder=5, alpha=0.7)
        
        # 예측된 급증
        pred_spike_mask = results['spike_probabilities'][:sample_size] > 0.5
        if np.any(pred_spike_mask):
            pred_indices = np.where(pred_spike_mask)[0]
            ax1.scatter(pred_indices, results['predictions'][:sample_size][pred_indices],
                       color='orange', s=30, marker='^', label='예측 급증', zorder=4)
        
        ax1.set_title('급증 예측 결과 (10분 후)', fontsize=16)
        ax1.set_xlabel('시간 인덱스', fontsize=12)
        ax1.set_ylabel('물류량', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 급증 확률 그래프
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(results['spike_probabilities'][:sample_size], label='급증 확률', color='red', linewidth=1.5)
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='결정 임계값 (0.5)')
        ax2.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='높은 확신 (0.7)')
        
        # 실제 급증 구간 배경색
        for i in range(sample_size):
            if results['actual_spikes'][i] == 1:
                ax2.axvspan(i-0.5, i+0.5, alpha=0.2, color='red')
        
        ax2.set_title('급증 확률 예측', fontsize=16)
        ax2.set_xlabel('시간 인덱스', fontsize=12)
        ax2.set_ylabel('확률', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 성능 지표 요약
        ax3 = plt.subplot(3, 2, 5)
        
        # 급증 예측 성능
        metrics_text = f"""급증 예측 성능
        
재현율: {metrics['spike_recall']:.1f}%
정밀도: {metrics['spike_precision']:.1f}%
F1-Score: {metrics['spike_f1']:.1f}%
정확도: {metrics['spike_accuracy']:.1f}%

정확히 예측: {metrics['spike_tp']}건
오탐: {metrics['spike_fp']}건
미탐: {metrics['spike_fn']}건"""
        
        ax3.text(0.1, 0.5, metrics_text, fontsize=14, transform=ax3.transAxes,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.axis('off')
        
        # 4. 혼동 행렬
        ax4 = plt.subplot(3, 2, 6)
        confusion_matrix = np.array([[metrics['spike_tn'], metrics['spike_fp']],
                                    [metrics['spike_fn'], metrics['spike_tp']]])
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['정상', '급증'],
                   yticklabels=['정상', '급증'],
                   ax=ax4)
        ax4.set_title('급증 예측 혼동 행렬', fontsize=14)
        ax4.set_xlabel('예측', fontsize=12)
        ax4.set_ylabel('실제', fontsize=12)
        
        plt.tight_layout()
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'spike_prediction_results/spike_visualization_{timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

# ===================================
# 메인 실행 함수
# ===================================

def main(data_path=None):
    """메인 실행 함수"""
    # 예측 시스템 초기화
    predictor = SpikePredictor()
    
    # 데이터 경로 설정
    if data_path is None:
        # 기본 경로들
        possible_paths = [
            'data/20250731_to_20250806.csv',
            'data/0730to31.csv',
            'data/20240201_TO_202507281705.csv'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            logger.error("데이터 파일을 찾을 수 없습니다.")
            return None
    
    # 예측 실행
    logger.info("급증 예측 프로세스 시작...")
    results = predictor.run_prediction(data_path)
    
    # 성능 평가
    logger.info("성능 평가 중...")
    metrics = predictor.evaluate_spike_predictions(
        results['predictions'],
        results['spike_probabilities'],
        results['actual_values'],
        results['actual_spikes']
    )
    
    # 결과 출력
    logger.info("\n" + "="*60)
    logger.info("급증 예측 성능 요약")
    logger.info("="*60)
    logger.info(f"[수치 예측 성능]")
    logger.info(f"  MAE: {metrics['mae']:.2f}")
    logger.info(f"  RMSE: {metrics['rmse']:.2f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"\n[급증 예측 성능]")
    logger.info(f"  재현율: {metrics['spike_recall']:.1f}% {'✓ 목표 달성' if metrics['spike_recall'] >= 70 else ''}")
    logger.info(f"  정밀도: {metrics['spike_precision']:.1f}%")
    logger.info(f"  F1-Score: {metrics['spike_f1']:.1f}%")
    logger.info(f"  정확도: {metrics['spike_accuracy']:.1f}%")
    
    # 결과 저장
    logger.info("\n결과 저장 중...")
    result_df = predictor.save_results(results, metrics)
    
    # 시각화
    logger.info("결과 시각화 중...")
    predictor.visualize_spike_results(results, metrics)
    
    logger.info("\n" + "="*60)
    logger.info("급증 예측 프로세스 완료!")
    logger.info("="*60)
    
    return results, metrics, result_df

# ===================================
# 실시간 예측 함수
# ===================================

def predict_realtime_spike(new_data_path):
    """실시간 급증 예측"""
    predictor = SpikePredictor()
    
    # 모델 로드
    if not predictor.models:
        predictor.load_models()
    
    try:
        # 데이터 전처리
        processed_data = predictor.preprocess_data(new_data_path)
        
        # 마지막 30분 데이터
        last_30_rows = processed_data.tail(30)
        
        if len(last_30_rows) < 30:
            logger.warning("실시간 예측을 위한 충분한 데이터가 없습니다 (최소 30개 필요)")
            return None
        
        # 스케일링
        scaled_data = predictor.scale_data(last_30_rows)
        
        # 입력 특징 선택
        input_features = [col for col in scaled_data.columns
                         if col not in ['scaled_FUTURE', 'future_spike', 'TIME', 'FUTURE']]
        
        # 시퀀스 생성
        X_realtime = scaled_data[input_features].values.reshape(1, 30, -1)
        
        # 예측
        reg_pred, spike_pred, _, _ = predictor.enhanced_ensemble_predict(X_realtime)
        
        # 역스케일링
        prediction = predictor.inverse_scale_predictions(reg_pred)[0]
        spike_probability = spike_pred[0]
        
        # 급증 여부
        is_spike = spike_probability > 0.5
        
        # 결과
        current_time = processed_data.index[-1]
        predict_time = current_time + timedelta(minutes=10)
        
        result = {
            'current_time': current_time,
            'predict_time': predict_time,
            'current_value': processed_data['TOTALCNT'].iloc[-1],
            'predicted_value': prediction,
            'spike_probability': spike_probability,
            'is_spike': is_spike,
            'confidence': spike_probability if is_spike else 1 - spike_probability
        }
        
        # 실시간 예측 결과 출력
        print("\n" + "="*80)
        print("실시간 급증 예측 결과 (10분 후)")
        print("="*80)
        print(f"현재 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"예측 대상 시간: {predict_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"현재 물류량: {result['current_value']:.0f}")
        print(f"10분 후 예측 물류량: {prediction:.0f}")
        print(f"급증 확률: {spike_probability:.1%}")
        print(f"급증 예측: {'★★★ 예 ★★★' if is_spike else '아니오'}")
        print(f"예측 신뢰도: {result['confidence']:.1%}")
        
        if spike_probability > 0.7:
            print(f"\n🚨 경고: 10분 후 급증 발생 가능성 매우 높음!")
            print(f"   예측값({prediction:.0f}) > 임계값({predictor.spike_threshold})")
            print(f"   즉시 대응 조치 필요!")
        elif spike_probability > 0.5:
            print(f"\n⚠️  주의: 10분 후 급증 발생 가능성 있음")
            print(f"   모니터링 강화 필요")
        
        print("="*80 + "\n")
        
        return result
        
    except Exception as e:
        logger.error(f"실시간 예측 실패: {str(e)}")
        return None

# ===================================
# 스크립트 실행
# ===================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 명령줄 인자로 데이터 경로 받기
        data_path = sys.argv[1]
        print(f"데이터 경로: {data_path}")
        results, metrics, result_df = main(data_path)
    else:
        # 기본 실행
        results, metrics, result_df = main()
    
    # 실시간 예측 예시 (옵션)
    # realtime_result = predict_realtime_spike('data/realtime_data.csv')