"""
반도체 물류 예측을 위한 하이브리드 딥러닝 모델 - 실시간 예측
========================================================
학습된 하이브리드 모델(LSTM, RNN, GRU, Bi-LSTM)을 사용하여
실시간으로 10분 후 물류량을 예측합니다.

입력: 20250807_DATA.CSV (100분 데이터)
출력: 10분 후 예측값

개발일: 2025년 8월
버전: 1.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridRealTimePredictor:
    """하이브리드 모델 기반 실시간 예측 시스템"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.config = None
        self.seq_length = 30
        self.future_minutes = 10
        
    def load_models(self):
        """학습된 하이브리드 모델들 로드"""
        logger.info("="*60)
        logger.info("하이브리드 모델 로딩 중...")
        logger.info("="*60)
        
        # 스크립트 디렉토리
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 딥러닝 모델들 로드
        model_names = ['lstm', 'gru', 'rnn', 'bi_lstm']
        for model_name in model_names:
            try:
                # 첫 번째 경로 시도
                model_path = os.path.join(script_dir, 'model', f'{model_name}_final_hybrid.keras')
                if os.path.exists(model_path):
                    self.models[model_name] = load_model(model_path, compile=False)
                    logger.info(f"✓ {model_name.upper()} 모델 로드 완료")
                else:
                    # 대체 경로들 시도
                    alt_paths = [
                        os.path.join(script_dir, 'model', f'BM_s30f10_0731_2079936.keras'),
                        os.path.join(script_dir, 'model', f'Model_s30f10_0731_2079936.keras'),
                        os.path.join(script_dir, 'model', f'{model_name}_best.keras')
                    ]
                    model_loaded = False
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path) and not model_loaded:
                            self.models[model_name] = load_model(alt_path, compile=False)
                            logger.info(f"✓ {model_name.upper()} 모델 로드 완료 (대체 경로)")
                            model_loaded = True
                            break
                    
                    if not model_loaded:
                        logger.warning(f"⚠ {model_name.upper()} 모델을 찾을 수 없음")
            except Exception as e:
                logger.error(f"❌ {model_name.upper()} 모델 로드 실패: {str(e)}")
        
        # 스케일러 로드
        try:
            scaler_paths = [
                os.path.join(script_dir, 'scaler', 'standard_scaler_hybrid.pkl'),
                os.path.join(script_dir, 'scaler', 'StdScaler_s30f10_0731_2079936.save'),
                os.path.join(script_dir, 'scaler', 'StdScaler_s30f10_0724_2079936.save')
            ]
            
            scaler_loaded = False
            for scaler_path in scaler_paths:
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"✓ 스케일러 로드 완료: {os.path.basename(scaler_path)}")
                    scaler_loaded = True
                    break
            
            if not scaler_loaded:
                raise FileNotFoundError("스케일러 파일을 찾을 수 없습니다")
                
        except Exception as e:
            logger.error(f"❌ 스케일러 로드 실패: {str(e)}")
            raise
        
        # 설정 파일 로드
        try:
            config_path = os.path.join(script_dir, 'results', 'training_config.json')
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
                    'bottleneck_threshold': 1400  # 데이터에 맞게 조정
                }
                logger.warning("⚠ 설정 파일을 찾을 수 없음 - 기본 설정 사용")
        except Exception as e:
            logger.error(f"❌ 설정 파일 로드 실패: {str(e)}")
            self.config = {
                'seq_length': 30,
                'future_minutes': 10,
                'model_weights': {'lstm': 0.3, 'gru': 0.25, 'rnn': 0.15, 'bi_lstm': 0.3},
                'bottleneck_threshold': 1400
            }
    
    def preprocess_data(self, data_path):
        """데이터 전처리 및 특징 엔지니어링"""
        logger.info(f"\n데이터 전처리 시작: {data_path}")
        
        # 데이터 로드
        data = pd.read_csv(data_path)
        logger.info(f"원본 데이터 shape: {data.shape}")
        
        # 시간 컬럼 datetime 변환
        data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
        data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
        
        # SUM 컬럼 제거
        columns_to_drop = [col for col in data.columns if 'SUM' in col]
        data = data.drop(columns=columns_to_drop)
        
        # 필요한 컬럼만 선택
        data = data[['CURRTIME', 'TOTALCNT', 'TIME']]
        data.set_index('CURRTIME', inplace=True)
        
        # 특징 엔지니어링
        logger.info("특징 엔지니어링 수행 중...")
        
        # 시간 관련 특징
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        
        # 이동 평균 특징
        data['MA_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
        data['MA_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
        data['MA_30'] = data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
        
        # 표준편차 특징
        data['STD_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).std()
        data['STD_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).std()
        
        # 변화율 특징
        data['change_rate'] = data['TOTALCNT'].pct_change()
        data['change_rate_5'] = data['TOTALCNT'].pct_change(5)
        
        # 결측값 처리
        data = data.fillna(method='ffill').fillna(0)
        
        logger.info(f"전처리 완료 - 최종 shape: {data.shape}")
        logger.info(f"사용 가능한 특징: {list(data.columns)}")
        
        return data
    
    def scale_features(self, data):
        """특징 스케일링"""
        # 학습 시 사용된 스케일링 컬럼들
        scale_columns = ['TOTALCNT', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
        
        # FUTURE 컬럼은 예측용이므로 제외하고 사용 가능한 컬럼만 선택
        available_columns = [col for col in scale_columns if col in data.columns]
        
        # 더미 FUTURE 컬럼 추가 (스케일러가 기대하는 경우)
        if hasattr(self.scaler, 'feature_names_in_'):
            expected_features = list(self.scaler.feature_names_in_)
            if 'FUTURE' in expected_features:
                # FUTURE를 TOTALCNT로 대체
                data['FUTURE'] = data['TOTALCNT']
                available_columns = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
                available_columns = [col for col in available_columns if col in data.columns]
        
        logger.info(f"스케일링할 컬럼: {available_columns}")
        
        # 스케일링 적용
        scaled_data = self.scaler.transform(data[available_columns])
        
        # 스케일링된 컬럼명 생성
        scaled_columns = [f'scaled_{col}' for col in available_columns]
        scaled_df = pd.DataFrame(scaled_data, columns=scaled_columns, index=data.index)
        
        # 원본 데이터와 병합
        result = pd.merge(data, scaled_df, left_index=True, right_index=True, how='left')
        
        return result, available_columns
    
    def create_sequence_for_prediction(self, data):
        """예측을 위한 시퀀스 생성"""
        # 입력 특징 선택 (scaled_ 로 시작하는 컬럼들, scaled_FUTURE 제외)
        input_features = [col for col in data.columns 
                         if col.startswith('scaled_') and col != 'scaled_FUTURE']
        
        logger.info(f"입력 특징: {input_features}")
        
        # 마지막 30개 데이터로 시퀀스 생성
        if len(data) < self.seq_length:
            raise ValueError(f"데이터가 부족합니다. 최소 {self.seq_length}개 필요 (현재: {len(data)}개)")
        
        # 마지막 seq_length 개의 데이터 추출
        sequence_data = data[input_features].iloc[-self.seq_length:].values
        
        # 3차원으로 reshape (batch_size=1, seq_length, features)
        X_seq = sequence_data.reshape(1, self.seq_length, len(input_features))
        
        return X_seq, input_features
    
    def ensemble_predict(self, X_seq):
        """앙상블 예측 수행"""
        if not self.models:
            raise ValueError("로드된 모델이 없습니다")
        
        weights = self.config.get('model_weights', {
            'lstm': 0.3,
            'gru': 0.25,
            'rnn': 0.15,
            'bi_lstm': 0.3
        })
        
        predictions = {}
        weighted_sum = 0
        total_weight = 0
        
        logger.info("\n개별 모델 예측 수행 중...")
        
        # 각 모델별 예측
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_seq, verbose=0)[0][0]
                predictions[model_name] = pred
                
                # 가중 평균 계산
                weight = weights.get(model_name, 0.25)
                weighted_sum += pred * weight
                total_weight += weight
                
                logger.info(f"  {model_name.upper()}: {pred:.4f} (가중치: {weight})")
            except Exception as e:
                logger.warning(f"  {model_name.upper()} 예측 실패: {str(e)}")
        
        # 최종 앙상블 예측값
        if total_weight > 0:
            ensemble_pred = weighted_sum / total_weight
        else:
            ensemble_pred = np.mean(list(predictions.values()))
        
        logger.info(f"\n앙상블 예측값 (scaled): {ensemble_pred:.4f}")
        
        return ensemble_pred, predictions
    
    def inverse_scale_prediction(self, scaled_pred, scale_columns):
        """예측값 역스케일링"""
        # 더미 배열 생성
        n_features = len(scale_columns)
        dummy_array = np.zeros((1, n_features))
        
        # FUTURE 컬럼의 위치 찾기 (또는 TOTALCNT 위치 사용)
        if 'FUTURE' in scale_columns:
            pred_idx = scale_columns.index('FUTURE')
        elif 'TOTALCNT' in scale_columns:
            pred_idx = scale_columns.index('TOTALCNT')
        else:
            pred_idx = 0
        
        dummy_array[0, pred_idx] = scaled_pred
        
        # 역변환
        inverse_scaled = self.scaler.inverse_transform(dummy_array)
        
        return inverse_scaled[0, pred_idx]
    
    def predict_next_10min(self, data_path):
        """10분 후 예측 수행"""
        try:
            # 1. 모델 로드
            self.load_models()
            
            if not self.models:
                raise ValueError("로드된 모델이 없습니다. 학습을 먼저 수행하세요.")
            
            # 2. 데이터 전처리
            processed_data = self.preprocess_data(data_path)
            
            # 3. 특징 스케일링
            scaled_data, scale_columns = self.scale_features(processed_data)
            
            # 4. 시퀀스 생성
            X_seq, input_features = self.create_sequence_for_prediction(scaled_data)
            
            # 5. 앙상블 예측
            ensemble_pred_scaled, individual_preds = self.ensemble_predict(X_seq)
            
            # 6. 역스케일링
            ensemble_pred = self.inverse_scale_prediction(ensemble_pred_scaled, scale_columns)
            
            # 7. 현재 상태 정보
            current_time = processed_data.index[-1]
            current_value = processed_data['TOTALCNT'].iloc[-1]
            predict_time = current_time + timedelta(minutes=self.future_minutes)
            
            # 8. 병목 여부 확인
            threshold = self.config.get('bottleneck_threshold', 1400)
            is_bottleneck = ensemble_pred > threshold
            
            # 9. 결과 출력
            print("\n" + "="*70)
            print("하이브리드 모델 기반 10분 후 예측 결과")
            print("="*70)
            print(f"\n현재 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"예측 대상 시간: {predict_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\n현재 물류량: {current_value:.0f}")
            print(f"10분 후 예측 물류량: {ensemble_pred:.0f}")
            print(f"변화량: {ensemble_pred - current_value:+.0f}")
            print(f"변화율: {((ensemble_pred - current_value) / current_value * 100):+.1f}%")
            
            print("\n" + "-"*70)
            print("개별 모델 예측값 (원본 스케일):")
            for model_name, pred_scaled in individual_preds.items():
                pred_original = self.inverse_scale_prediction(pred_scaled, scale_columns)
                print(f"  {model_name.upper()}: {pred_original:.0f}")
            
            print("\n" + "-"*70)
            print(f"병목 임계값: {threshold}")
            print(f"병목 예상: {'★★★ 예 ★★★' if is_bottleneck else '아니오'}")
            
            if is_bottleneck:
                print(f"\n⚠️  경고: 10분 후 병목 발생 예상!")
                print(f"    예측값({ensemble_pred:.0f}) > 임계값({threshold})")
            
            print("\n" + "="*70)
            
            # 최근 추세 시각화
            self.visualize_prediction(processed_data, current_time, predict_time, 
                                    current_value, ensemble_pred, threshold)
            
            # 결과 딕셔너리 반환
            result = {
                'current_time': current_time,
                'predict_time': predict_time,
                'current_value': current_value,
                'predicted_value': ensemble_pred,
                'change_amount': ensemble_pred - current_value,
                'change_rate': ((ensemble_pred - current_value) / current_value * 100),
                'is_bottleneck': is_bottleneck,
                'individual_predictions': {
                    model_name: self.inverse_scale_prediction(pred, scale_columns)
                    for model_name, pred in individual_preds.items()
                },
                'confidence': len(self.models) / 4 * 100  # 모델 수에 따른 신뢰도
            }
            
            # 간단한 형식으로도 출력 (기존 형식 호환)
            predicted_dict = {
                "LSTM": int(round(result['individual_predictions'].get('lstm', ensemble_pred))),
                "GRU": int(round(result['individual_predictions'].get('gru', ensemble_pred))),
                "RNN": int(round(result['individual_predictions'].get('rnn', ensemble_pred))),
                "Bi-LSTM": int(round(result['individual_predictions'].get('bi_lstm', ensemble_pred))),
                "ENSEMBLE": int(round(ensemble_pred))
            }
            print(f"\n예측 결과 (딕셔너리): {[predicted_dict]}")
            
            return result
            
        except Exception as e:
            logger.error(f"예측 실행 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_prediction(self, data, current_time, predict_time, 
                           current_value, predicted_value, threshold):
        """예측 결과 시각화"""
        try:
            plt.figure(figsize=(12, 6))
            
            # 최근 100개 데이터만 표시
            recent_data = data['TOTALCNT'].tail(100)
            
            # 실제 데이터 플롯
            plt.plot(recent_data.index, recent_data.values, 
                    'b-', linewidth=2, label='실제 물류량')
            
            # 현재 시점 표시
            plt.scatter(current_time, current_value, 
                       color='blue', s=100, zorder=5, label='현재')
            
            # 예측값 표시
            plt.scatter(predict_time, predicted_value, 
                       color='red', s=100, marker='^', zorder=5, label='10분 후 예측')
            
            # 예측 연결선
            plt.plot([current_time, predict_time], 
                    [current_value, predicted_value], 
                    'r--', linewidth=1, alpha=0.7)
            
            # 병목 임계선
            plt.axhline(y=threshold, color='orange', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'병목 임계값 ({threshold})')
            
            # 레이블 및 제목
            plt.title('반도체 물류량 예측 (하이브리드 모델)', fontsize=14, fontweight='bold')
            plt.xlabel('시간', fontsize=12)
            plt.ylabel('물류량 (TOTALCNT)', fontsize=12)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # x축 날짜 포맷
            plt.gcf().autofmt_xdate()
            
            # 여백 조정
            plt.tight_layout()
            
            # 저장 및 표시
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(f'realtime_prediction_{timestamp}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.warning(f"시각화 중 오류 발생: {str(e)}")

# 메인 실행 함수
def main():
    """메인 실행 함수"""
    # 스크립트 디렉토리
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 데이터 파일 경로
    data_filename = '20250807_DATA.csv'
    data_path = os.path.join(script_dir, 'data', data_filename)
    
    # 데이터 파일 존재 확인
    if not os.path.exists(data_path):
        # 대체 경로 시도
        alt_path = os.path.join(script_dir, data_filename)
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
            return
    
    print(f"데이터 파일 경로: {data_path}")
    
    # 예측 시스템 초기화
    predictor = HybridRealTimePredictor()
    
    # 10분 후 예측 수행
    result = predictor.predict_next_10min(data_path)
    
    if result:
        print("\n✅ 예측이 성공적으로 완료되었습니다!")
        print(f"   모델 신뢰도: {result['confidence']:.0f}%")
    else:
        print("\n❌ 예측 실행 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main()