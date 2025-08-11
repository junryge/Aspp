"""
반도체 물류 예측을 위한 하이브리드 딥러닝 모델 (v3.1, TF 2.18.0 호환)
======================================================================
* Keras 콜백 기반 학습 및 재시작 기능 최적화 버전 *

본 시스템은 반도체 팹 간 물류 이동량을 예측하고 병목 구간을 사전에 감지하기 위한 통합 예측 모델입니다.

주요 기능:
1. LSTM, GRU, RNN, Bi-LSTM 모델을 통합한 앙상블 예측
2. 실시간 병목 구간 감지 및 알림
3. 10분 후 물류량 예측
4. CPU 기반 실행으로 범용성 확보
5. Keras 콜백 기반의 효율적인 학습 및 재시작 기능 지원

개발일: 2024년 (수정일: 2025-08-11)
버전: 3.1 (Keras 콜백 최적화 버전)
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import joblib
import logging
import warnings
import json
import pickle
import traceback
import shutil
import argparse

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# ===================================
# 1. 환경 설정 및 초기화
# ===================================
# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 랜덤 시드 고정
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', mode='a'), # 'a' 모드로 변경하여 로그 누적
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# 2. 체크포인트 관리 클래스
# ===================================
class CheckpointManager:
    """학습 상태 및 데이터를 저장하고 복원하는 클래스"""
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.state_file = os.path.join(checkpoint_dir, 'training_state.json')
        self.data_file = os.path.join(checkpoint_dir, 'preprocessed_data.pkl')

    def save_state(self, state_dict):
        with open(self.state_file, 'w') as f:
            json.dump(state_dict, f, indent=4, default=str)
        logger.info(f"학습 상태 저장됨: {self.state_file}")

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"학습 상태 로드됨: {self.state_file}")
            return state
        return None

    def save_data(self, data_dict):
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        logger.info(f"전처리된 데이터 저장됨: {self.data_file}")

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"전처리된 데이터 로드됨: {self.data_file}")
            return data
        return None

# ===================================
# 3. 모델 정의 클래스
# ===================================
class HybridModels:
    """딥러닝 모델 구조를 정의하는 클래스"""
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = {}
        self.histories = {}

    def build_lstm_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(units=100, return_sequences=True), Dropout(0.2), BatchNormalization(),
            LSTM(units=100, return_sequences=True), Dropout(0.2), BatchNormalization(),
            LSTM(units=100, return_sequences=False), Dropout(0.2),
            Dense(units=50, activation='relu'),
            Dense(units=1)
        ], name='lstm')
        return model

    def build_gru_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            GRU(units=100, return_sequences=True), Dropout(0.2),
            GRU(units=100, return_sequences=True), Dropout(0.2),
            GRU(units=50, return_sequences=False), Dropout(0.2),
            Dense(units=30, activation='relu'),
            Dense(units=1)
        ], name='gru')
        return model

    def build_simple_rnn_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            SimpleRNN(units=100, return_sequences=True), Dropout(0.2),
            SimpleRNN(units=50, return_sequences=False), Dropout(0.2),
            Dense(units=30, activation='relu'),
            Dense(units=1)
        ], name='rnn')
        return model

    def build_bidirectional_lstm_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            Bidirectional(LSTM(units=50, return_sequences=True)), Dropout(0.2),
            Bidirectional(LSTM(units=50, return_sequences=False)), Dropout(0.2),
            Dense(units=30, activation='relu'),
            Dense(units=1)
        ], name='bi_lstm')
        return model

# ===================================
# 4. 메인 학습 프로세스
# ===================================
def main(resume=False):
    """메인 학습 프로세스"""
    checkpoint_manager = CheckpointManager()
    state = {}

    # 재시작 모드 확인
    if resume:
        loaded_state = checkpoint_manager.load_state()
        if loaded_state:
            state = loaded_state
            logger.info("="*60)
            logger.info("이전 학습 상태에서 재시작합니다.")
            logger.info(f"완료된 모델: {state.get('completed_models', [])}")
            logger.info("="*60)

            saved_data = checkpoint_manager.load_data()
            if saved_data:
                # 데이터 로드
                X_train, y_train = saved_data['X_train'], saved_data['y_train']
                X_val, y_val = saved_data['X_val'], saved_data['y_val']
                X_test, y_test = saved_data['X_test'], saved_data['y_test']
                sscaler = saved_data['scaler']
                Modified_Data = saved_data['modified_data']
                input_shape = saved_data['input_shape']
                logger.info("저장된 전처리 데이터를 성공적으로 로드했습니다.")
            else:
                logger.error("재시작 실패: 저장된 데이터 파일(preprocessed_data.pkl)이 없습니다.")
                return
        else:
            logger.warning("저장된 학습 상태가 없습니다. 처음부터 학습을 시작합니다.")
            resume = False

    # 데이터 전처리 (재시작이 아닌 경우)
    if not resume:
        logger.info("="*60)
        logger.info("신규 학습 시작: 데이터 전처리부터 진행합니다.")
        logger.info("="*60)
        state = {'completed_models': [], 'model_histories': {}} # 새 상태 초기화

        # 데이터 로드
        data_path = 'data/20240201_TO_202507281705.csv'
        if not os.path.exists(data_path):
            logger.error(f"데이터 파일 없음: {data_path}")
            return
        Full_Data = pd.read_csv(data_path)
        Full_Data['CURRTIME'] = pd.to_datetime(Full_Data['CURRTIME'], format='%Y%m%d%H%M')
        Full_Data = Full_Data.drop(columns=[col for col in Full_Data.columns if 'SUM' in col], errors='ignore')
        Full_Data = Full_Data[['CURRTIME', 'TOTALCNT']].set_index('CURRTIME').sort_index()

        # 이상치 제거
        Modified_Data = Full_Data[
            (Full_Data['TOTALCNT'] >= 500) & (Full_Data['TOTALCNT'] <= 3000)
        ].copy()
        logger.info(f"원본 데이터 shape: {Full_Data.shape} -> 이상치 제거 후: {Modified_Data.shape}")

        # Target(FUTURE) 변수 생성
        future_minutes = 10
        Modified_Data['FUTURE'] = Modified_Data['TOTALCNT'].shift(-future_minutes)
        Modified_Data.dropna(inplace=True)
        logger.info(f"FUTURE 컬럼 생성 후: {Modified_Data.shape}")

        # 특징 엔지니어링
        Modified_Data['hour'] = Modified_Data.index.hour
        Modified_Data['dayofweek'] = Modified_Data.index.dayofweek
        Modified_Data['MA_5'] = Modified_Data['TOTALCNT'].rolling(5).mean()
        Modified_Data['STD_5'] = Modified_Data['TOTALCNT'].rolling(5).std()
        Modified_Data.fillna(method='bfill', inplace=True) # 역방향으로 채워 초기 NaN 방지

        # 데이터 스케일링
        sscaler = RobustScaler()
        scaled_features = ['TOTALCNT', 'FUTURE', 'MA_5', 'STD_5']
        Modified_Data[scaled_features] = sscaler.fit_transform(Modified_Data[scaled_features])

        # 시퀀스 데이터 생성
        def create_sequences(data, feature_cols, target_col, seq_length=30):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[feature_cols].iloc[i:i+seq_length].values)
                y.append(data[target_col].iloc[i+seq_length])
            return np.array(X), np.array(y)

        SEQ_LENGTH = 30
        input_features = ['TOTALCNT', 'MA_5', 'STD_5', 'hour', 'dayofweek']
        X_seq, y_seq = create_sequences(Modified_Data, input_features, 'FUTURE', SEQ_LENGTH)

        # ### 중요: 데이터 분할 방식에 대한 참고사항 ###
        # 아래 코드는 전체 시퀀스 데이터를 무작위로 섞어 훈련/검증/테스트 세트로 분할합니다.
        # 이 방식은 모델이 시간적 순서에 관계없이 일반적인 패턴을 학습하게 할 수 있으나,
        # 시계열 데이터의 시간적 종속성을 파괴하여 '데이터 누수(Data Leakage)'를 일으킬 수 있습니다.
        # 즉, 훈련 시점에 실제로는 알 수 없는 미래의 데이터 패턴 조각이 훈련 데이터에 포함될 수 있습니다.
        #
        # 일반적인 시계열 분석에서는 데이터를 시간 순으로 분할합니다 (예: 앞 70% 훈련, 중간 15% 검증, 뒤 15% 테스트).
        # 이 코드에서는 원본 로직을 유지하지만, 이러한 점을 인지하고 사용해야 합니다.
        # logger.info("데이터를 무작위로 섞어서 분할합니다. (시계열 순서 무시)")

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_seq, y_seq, test_size=0.15, random_state=RANDOM_SEED, shuffle=True
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=(0.15/0.85), random_state=RANDOM_SEED, shuffle=True
        )

        logger.info(f"데이터 분할 완료: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        input_shape = (X_train.shape[1], X_train.shape[2])

        # 전처리된 데이터 저장
        checkpoint_manager.save_data({
            'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test, 'scaler': sscaler,
            'modified_data': Modified_Data, 'input_shape': input_shape,
            'scaled_features': scaled_features
        })

    # 모델 학습
    hybrid_models = HybridModels(input_shape)
    model_builders = {
        'lstm': hybrid_models.build_lstm_model,
        'gru': hybrid_models.build_gru_model,
        'rnn': hybrid_models.build_simple_rnn_model,
        'bi_lstm': hybrid_models.build_bidirectional_lstm_model,
    }

    EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005

    completed_models = state.get('completed_models', [])

    for name, builder in model_builders.items():
        if name in completed_models:
            logger.info(f"\n>> 모델 '{name}'은(는) 이미 학습이 완료되었습니다. 건너뜁니다.")
            try:
                # 완료된 모델과 history 로드
                hybrid_models.models[name] = load_model(f'model/{name}_final_hybrid.keras')
                with open(f'results/{name}_history.json', 'r') as f:
                    hybrid_models.histories[name] = json.load(f)
            except Exception as e:
                logger.error(f"완료된 모델 '{name}' 로드 실패: {e}. 해당 모델을 다시 학습합니다.")
                completed_models.remove(name)
        
        if name not in completed_models:
            logger.info(f"\n{'='*60}\n>> 모델 '{name.upper()}' 학습 시작\n{'='*60}")
            model = builder()
            model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
            model.summary(print_fn=logger.info)

            # 재시작을 위한 가중치 로드
            initial_epoch = 0
            weights_path = os.path.join(checkpoint_manager.checkpoint_dir, f'{name}_latest_weights.weights.h5')
            if resume and os.path.exists(weights_path):
                try:
                    model.load_weights(weights_path)
                    initial_epoch = state.get('model_epochs', {}).get(name, 0)
                    logger.info(f"'{name}' 모델의 가중치를 로드했습니다. Epoch {initial_epoch + 1}부터 학습을 재개합니다.")
                except Exception as e:
                    logger.warning(f"가중치 로드 실패: {e}. 처음부터 학습합니다.")

            # Keras 콜백 설정
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1),
                ModelCheckpoint(
                    filepath=os.path.join(checkpoint_manager.checkpoint_dir, f'{name}_latest_weights.weights.h5'),
                    save_weights_only=True,
                    verbose=0 # 로그가 너무 많아지지 않도록 설정
                )
            ]

            try:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=callbacks,
                    initial_epoch=initial_epoch,
                    verbose=1
                )
                
                # 학습 완료 후 처리
                hybrid_models.models[name] = model
                hybrid_models.histories[name] = history.history
                completed_models.append(name)

                # 상태 업데이트
                state['completed_models'] = completed_models
                state['model_histories'][name] = history.history
                checkpoint_manager.save_state(state)

            except KeyboardInterrupt:
                logger.warning(f"\n! 학습 중단(KeyboardInterrupt) ! - 모델: {name}")
                # 현재 epoch 저장
                last_epoch = initial_epoch + len(history.history['loss']) if 'history' in locals() else initial_epoch
                if 'model_epochs' not in state: state['model_epochs'] = {}
                state['model_epochs'][name] = last_epoch
                checkpoint_manager.save_state(state)
                logger.info("현재까지의 진행 상황이 저장되었습니다. --resume 옵션으로 재시작하세요.")
                return # 프로그램 종료
            except Exception as e:
                logger.error(f"학습 중 오류 발생: {e}\n{traceback.format_exc()}")
                return

    logger.info("\n" + "="*60 + "\n모든 모델 학습 완료!\n" + "="*60)

    # 평가, 시각화, 저장
    evaluate_and_save(hybrid_models, X_test, y_test, checkpoint_manager)


def evaluate_and_save(hybrid_models, X_test, y_test, checkpoint_manager):
    """모델 평가, 시각화 및 결과 저장 함수"""
    logger.info("\n" + "="*60 + "\n모델 성능 평가\n" + "="*60)

    # 앙상블 예측
    weights = {'lstm': 0.3, 'gru': 0.25, 'rnn': 0.15, 'bi_lstm': 0.3}
    individual_preds = {name: model.predict(X_test, verbose=0) for name, model in hybrid_models.models.items()}
    ensemble_pred = sum(pred * weights[name] for name, pred in individual_preds.items())

    # 스케일 역변환을 위한 정보 로드
    saved_data = checkpoint_manager.load_data()
    scaler = saved_data['scaler']
    scaled_features = saved_data['scaled_features']
    future_idx = scaled_features.index('FUTURE')
    
    def inverse_transform(y_data):
        y_data_flat = y_data.flatten()
        dummy_array = np.zeros((len(y_data_flat), len(scaled_features)))
        dummy_array[:, future_idx] = y_data_flat
        return scaler.inverse_transform(dummy_array)[:, future_idx]

    # 성능 지표 계산 및 로깅
    results = {}
    for name, pred in individual_preds.items():
        results[name] = calculate_metrics(y_test, pred, name.upper(), inverse_transform)
    results['ensemble'] = calculate_metrics(y_test, ensemble_pred, "ENSEMBLE", inverse_transform)
    
    # 시각화 및 저장
    visualize_and_save_results(hybrid_models, results, y_test, individual_preds, ensemble_pred, inverse_transform, saved_data)
    
    # 최종 요약
    logger.info("\n" + "="*60 + "\n최종 요약\n" + "="*60)
    best_mae = float('inf')
    best_model_name = ''
    for name, res in results.items():
        if name != 'ensemble' and res['mae_original'] < best_mae:
            best_mae = res['mae_original']
            best_model_name = name
    
    logger.info(f"최고 성능 개별 모델: {best_model_name.upper()} (MAE: {best_mae:.2f})")
    logger.info(f"앙상블 모델 성능: MAE: {results['ensemble']['mae_original']:.2f}, R²: {results['ensemble']['r2']:.4f}")
    
    state = checkpoint_manager.load_state() or {}
    state['training_completed'] = True
    checkpoint_manager.save_state(state)
    logger.info("\n모든 작업이 성공적으로 완료되었습니다.")


def calculate_metrics(y_true, y_pred, model_name, inverse_transform_func):
    """성능 지표를 계산하고 로그를 출력하는 함수"""
    y_true_orig = inverse_transform_func(y_true)
    y_pred_orig = inverse_transform_func(y_pred)
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mae_original': mean_absolute_error(y_true_orig, y_pred_orig)
    }
    
    logger.info(f"\n--- {model_name} 모델 성능 ---")
    logger.info(f" - R² (Scaled): {metrics['r2']:.4f}")
    logger.info(f" - MAE (원래 값): {metrics['mae_original']:.2f} (가장 중요한 지표)")
    return metrics


def visualize_and_save_results(models_obj, results, y_test, individual_preds, ensemble_pred, inverse_transform, saved_data):
    """결과를 시각화하고 파일로 저장하는 함수"""
    os.makedirs('model', exist_ok=True)
    os.makedirs('scaler', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 1. 학습 곡선 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    for idx, (name, history) in enumerate(models_obj.histories.items()):
        if idx < 4:
            axes[idx].plot(history['loss'], label='Training Loss')
            axes[idx].plot(history['val_loss'], label='Validation Loss')
            axes[idx].set_title(f'{name.upper()} Training & Validation Loss')
            axes[idx].legend()
            axes[idx].grid(True)
    plt.tight_layout()
    plt.savefig('results/training_curves.png')
    plt.close()

    # 2. 예측 결과 시각화
    plt.figure(figsize=(20, 10))
    sample_size = min(300, len(y_test))
    y_test_orig = inverse_transform(y_test[:sample_size])
    ensemble_orig = inverse_transform(ensemble_pred[:sample_size])

    plt.plot(y_test_orig, label='Actual', color='black', marker='o', markersize=3, linestyle='')
    plt.plot(ensemble_orig, label='Ensemble Prediction', color='red', linewidth=2)
    plt.title('Logistics Volume Prediction (10 Minutes Ahead)', fontsize=16)
    plt.ylabel('Volume (TOTALCNT)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig('results/prediction_results.png')
    plt.close()

    # 3. 모델 및 결과 저장
    for name, model in models_obj.models.items():
        model.save(f'model/{name}_final_hybrid.keras')
        with open(f'results/{name}_history.json', 'w') as f:
            json.dump(models_obj.histories[name], f, indent=4)
            
    joblib.dump(saved_data['scaler'], 'scaler/robust_scaler_hybrid.pkl')
    pd.DataFrame(results).T.to_csv('results/model_performance.csv')
    logger.info("모델, 스케일러, 학습 이력, 성능 평가 결과가 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='반도체 물류 예측 하이브리드 모델 학습')
    parser.add_argument('--resume', action='store_true', help='이전 학습을 이어서 진행')
    parser.add_argument('--reset', action='store_true', help='체크포인트와 결과물을 모두 삭제하고 처음부터 시작')
    args = parser.parse_args()

    if args.reset:
        logger.info("'--reset' 옵션이 감지되었습니다. 기존 체크포인트와 결과물을 모두 삭제합니다.")
        for folder in ['checkpoints', 'model', 'results', 'scaler']:
            if os.path.exists(folder):
                shutil.rmtree(folder)
    
    main(resume=args.resume and not args.reset)

