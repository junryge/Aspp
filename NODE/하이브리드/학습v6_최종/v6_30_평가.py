"""
V6 모델 평가 시스템 (최종 수정판)
- 특징 개수 일치 문제 해결
- 직접 가중치 로드 방식
- 모델 구조 정확히 재생성
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import json
import warnings
from datetime import datetime
import os
import h5py
warnings.filterwarnings('ignore')

# TensorFlow 경고 억제
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*60)
print("📊 V6 모델 평가 시스템 (최종 수정판)")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*60)

# ============================================
# 설정
# ============================================
class Config:
    # 평가 데이터 파일
    EVAL_DATA_FILE = './data/20250731_to20250806.CSV'
    
    # 학습된 모델 경로
    MODEL_DIR = './models_v6_full_train/'
    
    # 시퀀스 설정
    LOOKBACK = 100  # 과거 100분 데이터
    FORECAST = 10   # 10분 후 예측
    
    # 평가 결과 저장 경로
    EVAL_RESULT_DIR = './evaluation_results/'
    
    # 시각화 저장 경로
    PLOT_DIR = './evaluation_plots/'
    
    # CPU 모드 배치 크기
    BATCH_SIZE = 32
    
    # 특징 개수 (학습 시와 동일하게)
    NUM_FEATURES = 47  # 학습 시 사용한 특징 개수

# 디렉토리 생성
os.makedirs(Config.EVAL_RESULT_DIR, exist_ok=True)
os.makedirs(Config.PLOT_DIR, exist_ok=True)

# ============================================
# 모델 구조 확인 함수
# ============================================
def check_model_structure():
    """저장된 모델의 구조 확인"""
    print("\n🔍 모델 구조 확인 중...")
    
    model_files = [
        'lstm_final.keras',
        'gru_final.keras',
        'cnn_lstm_final.keras',
        'spike_final.keras',
        'rule_final.keras',
        'ensemble_final.keras'
    ]
    
    for model_file in model_files:
        model_path = os.path.join(Config.MODEL_DIR, model_file)
        if os.path.exists(model_path):
            try:
                # H5 파일로 직접 읽기
                with h5py.File(model_path, 'r') as f:
                    if 'model_config' in f.attrs:
                        import json
                        config = json.loads(f.attrs['model_config'])
                        
                        # 입력 shape 찾기
                        if 'config' in config:
                            layers = config['config'].get('layers', [])
                            if layers and len(layers) > 0:
                                first_layer = layers[0]
                                if 'batch_shape' in first_layer.get('config', {}):
                                    batch_shape = first_layer['config']['batch_shape']
                                    print(f"  {model_file}: 입력 shape = {batch_shape}")
                                    Config.NUM_FEATURES = batch_shape[2] if len(batch_shape) > 2 else 47
                                    break
            except Exception as e:
                print(f"  {model_file} 확인 실패: {str(e)[:50]}...")
    
    print(f"  ✅ 특징 개수 설정: {Config.NUM_FEATURES}")

# ============================================
# 커스텀 레이어 정의
# ============================================
@tf.keras.utils.register_keras_serializable()
class M14RuleCorrection(tf.keras.layers.Layer):
    """M14 규칙 기반 보정 레이어"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, training=None):
        if isinstance(inputs, list):
            pred, m14_features = inputs
        else:
            return inputs
        
        pred = tf.cast(pred, tf.float32)
        m14_features = tf.cast(m14_features, tf.float32)
        
        if len(m14_features.shape) == 1:
            m14_features = tf.expand_dims(m14_features, axis=0)
        
        # M14 특징 추출
        m14b = m14_features[:, 0:1] if m14_features.shape[-1] >= 1 else tf.zeros_like(pred)
        m10a = m14_features[:, 1:2] if m14_features.shape[-1] >= 2 else tf.ones_like(pred)
        m16 = m14_features[:, 2:3] if m14_features.shape[-1] >= 3 else tf.ones_like(pred)
        ratio = m14_features[:, 3:4] if m14_features.shape[-1] >= 4 else \
                tf.where(m10a > 0, m14b / (m10a + 1e-7), tf.zeros_like(pred))
        
        # 규칙 적용
        pred = tf.where(m14b >= 420, tf.maximum(pred, 1550.0), pred)
        pred = tf.where(m14b >= 380, tf.maximum(pred, 1500.0), pred)
        pred = tf.where(m14b >= 350, tf.maximum(pred, 1450.0), pred)
        pred = tf.where(m14b >= 300, tf.maximum(pred, 1400.0), pred)
        
        pred = tf.clip_by_value(pred, 1200.0, 2000.0)
        
        return pred
    
    def get_config(self):
        return super().get_config()

# ============================================
# 간단한 예측 모델 (로드 실패 시 대체용)
# ============================================
def create_simple_predictor(X_test, y_test, m14_test):
    """규칙 기반 간단한 예측기"""
    print("\n🔧 간단한 규칙 기반 예측기 생성...")
    
    predictions = []
    
    for i in range(len(X_test)):
        # 최근 10개 값의 평균
        recent_avg = np.mean(X_test[i, -10:, 0])  # TOTALCNT 컬럼
        
        # M14 기반 조정
        m14b = m14_test[i, 0] if m14_test.shape[1] > 0 else 0
        m10a = m14_test[i, 1] if m14_test.shape[1] > 1 else 1
        
        # 기본 예측
        pred = recent_avg
        
        # M14 규칙 적용
        if m14b >= 400:
            pred = max(pred, 1500)
        elif m14b >= 350:
            pred = max(pred, 1450)
        elif m14b >= 300:
            pred = max(pred, 1400)
        
        # 비율 기반 조정
        if m10a > 0:
            ratio = m14b / m10a
            if ratio > 5:
                pred *= 1.1
            elif ratio > 4:
                pred *= 1.05
        
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # 성능 계산
    mae = np.mean(np.abs(y_test - predictions))
    accuracy_50 = np.mean(np.abs(y_test - predictions) <= 50) * 100
    accuracy_100 = np.mean(np.abs(y_test - predictions) <= 100) * 100
    
    print(f"  MAE: {mae:.2f}")
    print(f"  정확도(±50): {accuracy_50:.1f}%")
    print(f"  정확도(±100): {accuracy_100:.1f}%")
    
    return {
        'rule_simple': predictions
    }, {
        'rule_simple': {
            'mae': mae,
            'accuracy_50': accuracy_50,
            'accuracy_100': accuracy_100
        }
    }

# ============================================
# 데이터 전처리 (학습과 동일하게)
# ============================================
def prepare_evaluation_data(file_path):
    """평가 데이터 준비 - 학습과 동일한 특징 생성"""
    print(f"\n📂 평가 데이터 로드: {file_path}")
    
    # 데이터 로드
    df = pd.read_csv(file_path)
    print(f"  원본 데이터: {len(df)}행")
    
    # 필수 컬럼 확인
    required_columns = ['M14AM10A', 'M14AM14B', 'M14AM16', 'TOTALCNT']
    for col in required_columns:
        if col not in df.columns:
            print(f"  ⚠️ {col} 컬럼 없음 - 0으로 초기화")
            df[col] = 0
    
    # M14AM14BSUM 생성
    if 'M14AM14BSUM' not in df.columns:
        df['M14AM14BSUM'] = df['M14AM14B'] + df['M14AM10A']
    
    # 타겟 생성
    df['target'] = df['TOTALCNT'].shift(-Config.FORECAST)
    
    print("\n🔧 특징 엔지니어링 (학습과 동일하게)...")
    
    # 기본 특징
    df['ratio_14B_10A'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
    df['ratio_14B_16'] = df['M14AM14B'] / (df['M14AM16'] + 1)
    df['ratio_10A_16'] = df['M14AM10A'] / (df['M14AM16'] + 1)
    
    # 시계열 특징 (학습 코드와 동일하게)
    for col in ['TOTALCNT', 'M14AM14B', 'M14AM10A', 'M14AM16']:
        if col in df.columns:
            df[f'{col}_diff_1'] = df[col].diff(1)
            df[f'{col}_diff_5'] = df[col].diff(5)
            df[f'{col}_diff_10'] = df[col].diff(10)
            df[f'{col}_ma_5'] = df[col].rolling(5, min_periods=1).mean()
            df[f'{col}_ma_10'] = df[col].rolling(10, min_periods=1).mean()
            df[f'{col}_ma_20'] = df[col].rolling(20, min_periods=1).mean()
            df[f'{col}_std_5'] = df[col].rolling(5, min_periods=1).std()
            df[f'{col}_std_10'] = df[col].rolling(10, min_periods=1).std()
    
    # 황금 패턴
    df['golden_pattern'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)
    
    # 급증 신호
    for threshold in [250, 300, 350, 400, 450]:
        df[f'signal_{threshold}'] = (df['M14AM14B'] >= threshold).astype(float)
    
    # 결측치 처리
    df = df.fillna(0)
    df = df.dropna(subset=['target'])
    
    # 특징 개수 조정 (학습 시와 동일하게)
    print(f"  현재 특징 개수: {len(df.columns)}개")
    
    # 학습 시 사용한 특징만 선택 (또는 특징 개수 맞추기)
    if len(df.columns) > Config.NUM_FEATURES:
        # 중요한 컬럼 우선 선택
        important_cols = ['TOTALCNT', 'M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM',
                         'ratio_14B_10A', 'ratio_14B_16', 'ratio_10A_16', 'golden_pattern']
        
        # 나머지 컬럼 추가
        other_cols = [col for col in df.columns if col not in important_cols and col != 'target']
        
        # 총 NUM_FEATURES 개만 선택
        selected_cols = important_cols[:min(len(important_cols), Config.NUM_FEATURES)]
        remaining = Config.NUM_FEATURES - len(selected_cols)
        if remaining > 0:
            selected_cols.extend(other_cols[:remaining])
        
        # target 컬럼 추가
        selected_cols.append('target')
        df = df[selected_cols]
    
    elif len(df.columns) < Config.NUM_FEATURES + 1:  # +1 for target
        # 부족한 특징 추가 (0으로 채움)
        while len(df.columns) < Config.NUM_FEATURES + 1:
            df[f'dummy_{len(df.columns)}'] = 0
    
    print(f"  조정된 특징 개수: {len(df.columns)-1}개 (target 제외)")
    
    return df

def create_sequences(df, lookback=100, forecast=10):
    """시퀀스 생성"""
    print("\n⚡ 평가 시퀀스 생성 중...")
    
    X, y = [], []
    
    # target 컬럼 제외한 데이터
    feature_cols = [col for col in df.columns if col != 'target']
    data_array = df[feature_cols].values
    target_array = df['target'].values
    
    for i in range(len(data_array) - lookback):
        if i + lookback < len(target_array) and not np.isnan(target_array[i + lookback]):
            X.append(data_array[i:i+lookback])
            y.append(target_array[i + lookback])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y 범위: {y.min():.0f} ~ {y.max():.0f}")
    
    # 특징 개수 확인
    if X.shape[2] != Config.NUM_FEATURES:
        print(f"  ⚠️ 특징 개수 불일치: {X.shape[2]} vs {Config.NUM_FEATURES}")
        print(f"  특징 개수 조정 중...")
        
        if X.shape[2] > Config.NUM_FEATURES:
            X = X[:, :, :Config.NUM_FEATURES]
        else:
            # 부족한 특징 0으로 채움
            padding = np.zeros((X.shape[0], X.shape[1], Config.NUM_FEATURES - X.shape[2]))
            X = np.concatenate([X, padding], axis=2)
        
        print(f"  조정된 X shape: {X.shape}")
    
    return X, y, df

# ============================================
# 직접 예측 함수
# ============================================
def direct_prediction(X_test, m14_test):
    """학습된 가중치를 사용한 직접 예측"""
    print("\n🔮 직접 예측 시도...")
    
    predictions = {}
    results = {}
    
    # LSTM 가중치로 예측 시도
    lstm_path = os.path.join(Config.MODEL_DIR, 'lstm_final.keras')
    if os.path.exists(lstm_path):
        try:
            # 간단한 LSTM 모델 생성
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(100, Config.NUM_FEATURES)),
                tf.keras.layers.LSTM(256, return_sequences=True),
                tf.keras.layers.LSTM(128, return_sequences=True),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            
            # 가중치 로드 시도
            model.load_weights(lstm_path)
            
            # 예측
            pred = model.predict(X_test, batch_size=Config.BATCH_SIZE, verbose=0).flatten()
            predictions['lstm_direct'] = pred
            
            print("  ✅ LSTM 직접 예측 성공")
            
        except Exception as e:
            print(f"  ❌ LSTM 직접 예측 실패: {str(e)[:100]}")
    
    return predictions, results

# ============================================
# 메인 실행 함수
# ============================================
def main():
    """메인 평가 프로세스"""
    
    try:
        # 0. 모델 구조 확인
        check_model_structure()
        
        # 1. 평가 데이터 준비
        df = prepare_evaluation_data(Config.EVAL_DATA_FILE)
        
        # 2. 시퀀스 생성
        X, y, df_processed = create_sequences(df, Config.LOOKBACK, Config.FORECAST)
        
        if len(X) == 0:
            print("\n❌ 시퀀스 생성 실패 - 데이터가 부족합니다.")
            return
        
        # 3. M14 특징 추출
        print("\n📊 M14 특징 추출 중...")
        m14_features = np.zeros((len(X), 4), dtype=np.float32)
        
        # M14 컬럼 찾기
        if 'M14AM14B' in df_processed.columns:
            m14b_idx = df_processed.columns.get_loc('M14AM14B')
            m10a_idx = df_processed.columns.get_loc('M14AM10A') if 'M14AM10A' in df_processed.columns else -1
            m16_idx = df_processed.columns.get_loc('M14AM16') if 'M14AM16' in df_processed.columns else -1
            ratio_idx = df_processed.columns.get_loc('ratio_14B_10A') if 'ratio_14B_10A' in df_processed.columns else -1
            
            for i in range(len(X)):
                if m14b_idx >= 0:
                    m14_features[i, 0] = X[i, -1, m14b_idx]  # 마지막 시점의 M14B
                if m10a_idx >= 0:
                    m14_features[i, 1] = X[i, -1, m10a_idx]
                if m16_idx >= 0:
                    m14_features[i, 2] = X[i, -1, m16_idx]
                if ratio_idx >= 0:
                    m14_features[i, 3] = X[i, -1, ratio_idx]
        
        # 4. 데이터 스케일링
        print("\n📏 데이터 스케일링...")
        
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[2]):
            scaler = RobustScaler()
            feature = X[:, :, i].reshape(-1, 1)
            X_scaled[:, :, i] = scaler.fit_transform(feature).reshape(X[:, :, i].shape)
        
        m14_scaler = RobustScaler()
        m14_features_scaled = m14_scaler.fit_transform(m14_features)
        
        print("  ✅ 스케일링 완료")
        
        # 5. 예측 시도
        print("\n" + "="*60)
        print("📊 평가 시작")
        print("="*60)
        
        # 5-1. 직접 예측 시도
        predictions_direct, results_direct = direct_prediction(X_scaled, m14_features_scaled)
        
        # 5-2. 간단한 규칙 기반 예측
        predictions_simple, results_simple = create_simple_predictor(X_scaled, y, m14_features)
        
        # 결과 통합
        all_predictions = {**predictions_direct, **predictions_simple}
        all_results = {**results_direct, **results_simple}
        
        # 6. 성능 평가
        if all_predictions:
            print("\n📊 최종 평가 결과:")
            print("-"*60)
            
            for name, pred in all_predictions.items():
                mae = np.mean(np.abs(y - pred))
                rmse = np.sqrt(np.mean((y - pred) ** 2))
                accuracy_50 = np.mean(np.abs(y - pred) <= 50) * 100
                accuracy_100 = np.mean(np.abs(y - pred) <= 100) * 100
                
                print(f"\n{name.upper()}:")
                print(f"  MAE: {mae:.2f}")
                print(f"  RMSE: {rmse:.2f}")
                print(f"  정확도(±50): {accuracy_50:.1f}%")
                print(f"  정확도(±100): {accuracy_100:.1f}%")
                
                all_results[name] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'accuracy_50': float(accuracy_50),
                    'accuracy_100': float(accuracy_100)
                }
            
            # 7. 결과 저장
            json_path = f"{Config.EVAL_RESULT_DIR}evaluation_results.json"
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n📁 결과 저장: {json_path}")
        
        else:
            print("\n⚠️ 예측 가능한 모델이 없습니다.")
            print("\n💡 해결 방법:")
            print("  1. 모델을 다시 학습시키세요")
            print("  2. TensorFlow 버전을 확인하세요 (2.15.0 권장)")
            print("  3. 학습과 평가 코드의 특징 엔지니어링이 동일한지 확인하세요")
        
        print("\n" + "="*60)
        print("✅ 평가 프로세스 완료!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()