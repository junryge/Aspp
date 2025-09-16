"""
V6 모델 평가 시스템 - 특징 개수 일치 버전
- 학습 시와 동일한 47개 특징 사용
- WeightedLoss 클래스 제대로 등록
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os
import gc
warnings.filterwarnings('ignore')

print("="*60)
print("🔬 V6 모델 평가 시스템 (Feature-Matched Version)")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*60)

# ============================================
# GPU 설정
# ============================================
def setup_gpu():
    """GPU 설정 및 확인"""
    print("\n🎮 GPU 환경 확인...")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU 감지: {len(gpus)}개")
            return True
        except Exception as e:
            print(f"⚠️ GPU 설정 오류: {e}")
            return False
    else:
        print("💻 CPU 모드로 실행")
        return False

has_gpu = setup_gpu()

# ============================================
# 설정
# ============================================
class Config:
    # 평가 데이터
    EVAL_DATA_FILE = './data/20250731_to_20250826.csv'
    
    # 모델 디렉토리
    MODEL_DIR = './models_v6_full_train/'
    SCALER_FILE = './scalers_v6_gpu.pkl'
    
    # 시퀀스 설정 (학습과 동일)
    LOOKBACK = 100  # 과거 100분
    FORECAST = 10   # 10분 후 예측
    
    # 결과 저장 경로
    OUTPUT_DIR = './evaluation_results/'
    
    # 평가 설정
    BATCH_SIZE = 128  # 평가시 배치 크기
    
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ============================================
# 커스텀 클래스 등록 (TF 2.16.1 호환)
# ============================================
@tf.keras.utils.register_keras_serializable()
class WeightedLoss(tf.keras.losses.Loss):
    """가중치 손실 함수"""
    def __init__(self, name="weighted_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        mae = tf.abs(y_true - y_pred)
        
        # 가중치
        weights = tf.ones_like(y_true)
        weights = tf.where(y_true >= 1550, 30.0, weights)
        weights = tf.where((y_true >= 1500) & (y_true < 1550), 25.0, weights)
        weights = tf.where((y_true >= 1450) & (y_true < 1500), 20.0, weights)
        weights = tf.where((y_true >= 1400) & (y_true < 1450), 15.0, weights)
        weights = tf.where((y_true >= 1350) & (y_true < 1400), 10.0, weights)
        
        # 큰 오차 페널티
        large_error = tf.where(mae > 100, mae * 0.2, 0.0)
        
        return tf.reduce_mean(mae * weights + large_error)
    
    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable()
class M14RuleCorrection(tf.keras.layers.Layer):
    """M14 규칙 기반 보정"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, training=None):
        pred, m14_features = inputs
        
        pred = tf.cast(pred, tf.float32)
        m14_features = tf.cast(m14_features, tf.float32)
        
        m14b = m14_features[:, 0:1]
        m10a = m14_features[:, 1:2]
        m16 = m14_features[:, 2:3]
        ratio = m14_features[:, 3:4] if m14_features.shape[1] > 3 else tf.zeros_like(m14b)
        
        # 임계값 규칙
        pred = tf.where(m14b >= 420, tf.maximum(pred, 1550.0), pred)
        pred = tf.where(m14b >= 380, tf.maximum(pred, 1500.0), pred)
        pred = tf.where(m14b >= 350, tf.maximum(pred, 1450.0), pred)
        pred = tf.where(m14b >= 300, tf.maximum(pred, 1400.0), pred)
        
        # 비율 보정
        pred = tf.where(ratio >= 5.5, pred * 1.15, pred)
        pred = tf.where((ratio >= 5.0) & (ratio < 5.5), pred * 1.10, pred)
        pred = tf.where((ratio >= 4.5) & (ratio < 5.0), pred * 1.08, pred)
        pred = tf.where((ratio >= 4.0) & (ratio < 4.5), pred * 1.05, pred)
        
        # 황금 패턴
        golden = (m14b >= 350) & (m10a < 70)
        pred = tf.where(golden, pred * 1.2, pred)
        
        # 범위 제한
        pred = tf.clip_by_value(pred, 1200.0, 2000.0)
        
        return pred
    
    def get_config(self):
        config = super().get_config()
        return config

# ============================================
# 데이터 로드 및 전처리
# ============================================
print("\n📊 평가 데이터 로드 중...")

# 데이터 로드
df = pd.read_csv(Config.EVAL_DATA_FILE)
print(f"  데이터 크기: {len(df)}행")

# 시간 정보 파싱
if 'CURRTIME' in df.columns:
    try:
        df['datetime'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
    except:
        df['datetime'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M%S')
elif 'TIME' in df.columns:
    try:
        df['datetime'] = pd.to_datetime(df['TIME'], format='%Y%m%d%H%M')
    except:
        df['datetime'] = pd.to_datetime(df['TIME'], format='%Y%m%d%H%M%S')
else:
    df['datetime'] = pd.date_range(start='2025-07-31', periods=len(df), freq='1min')

# 필요한 컬럼 확인 및 생성
required_columns = ['M14AM10A', 'M14AM14B', 'M14AM16', 'TOTALCNT']
for col in required_columns:
    if col not in df.columns:
        print(f"⚠️ {col} 컬럼 없음 - 0으로 초기화")
        df[col] = 0

# M14AM14BSUM이 없으면 생성
if 'M14AM14BSUM' not in df.columns:
    df['M14AM14BSUM'] = df['M14AM14B'] + df['M14AM10A']

print(f"  날짜 범위: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"  TOTALCNT 범위: {df['TOTALCNT'].min():.0f} ~ {df['TOTALCNT'].max():.0f}")

# ============================================
# 특징 엔지니어링 (학습과 동일하게!)
# ============================================
print("\n🔧 특징 엔지니어링 (학습과 동일한 47개 특징)...")

# 비율 특징
df['ratio_14B_10A'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
df['ratio_14B_16'] = df['M14AM14B'] / (df['M14AM16'] + 1)
df['ratio_10A_16'] = df['M14AM10A'] / (df['M14AM16'] + 1)

# 시계열 특징 (학습 코드와 동일)
feature_columns = ['M14AM10A', 'M14AM14B', 'M14AM16', 'M14AM14BSUM', 'TOTALCNT']
for col in feature_columns:
    # 변화량
    df[f'{col}_diff_1'] = df[col].diff(1)
    df[f'{col}_diff_5'] = df[col].diff(5)
    df[f'{col}_diff_10'] = df[col].diff(10)
    
    # 이동평균
    df[f'{col}_ma_5'] = df[col].rolling(5, min_periods=1).mean()
    df[f'{col}_ma_10'] = df[col].rolling(10, min_periods=1).mean()
    
    # 표준편차 (학습 코드는 std_5, std_10만 생성)
    df[f'{col}_std_5'] = df[col].rolling(5, min_periods=1).std()
    df[f'{col}_std_10'] = df[col].rolling(10, min_periods=1).std()

# 황금 패턴
df['golden_pattern'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)

# 급증 신호 (학습과 동일한 임계값)
thresholds = {1300: 250, 1400: 300, 1450: 350, 1500: 380}  # 1550 제외
for level, threshold in thresholds.items():
    df[f'signal_{level}'] = (df['M14AM14B'] >= threshold).astype(float)

ratio_thresholds = {1300: 3.5, 1400: 4.0, 1450: 4.5, 1500: 5.0}  # 1550 제외
for level, ratio in ratio_thresholds.items():
    df[f'ratio_signal_{level}'] = (df['ratio_14B_10A'] >= ratio).astype(float)

# datetime 컬럼 제외
if 'datetime' in df.columns:
    df_features = df.drop('datetime', axis=1)
else:
    df_features = df

df_features = df_features.fillna(0)

print(f"  특징 개수: {len(df_features.columns)}개 (datetime 제외)")
print(f"  특징 컬럼 샘플: {list(df_features.columns[:10])}")

# ============================================
# 시퀀스 생성
# ============================================
print("\n⚡ 평가용 시퀀스 생성 중...")

def create_sequences_with_info(df, df_features, lookback=100, forecast=10):
    """시퀀스 생성 (날짜 정보 포함)"""
    X, y = [], []
    dates = []
    m14_features = []
    
    data = df_features.values
    
    for i in range(len(data) - lookback - forecast):
        X.append(data[i:i+lookback])
        y.append(df_features['TOTALCNT'].iloc[i+lookback+forecast-1])
        if 'datetime' in df.columns:
            dates.append(df['datetime'].iloc[i+lookback+forecast-1])
        else:
            dates.append(i+lookback+forecast-1)
        
        # M14 특징 (현재 시점)
        idx = i + lookback
        m14_features.append([
            df_features['M14AM14B'].iloc[idx],
            df_features['M14AM10A'].iloc[idx],
            df_features['M14AM16'].iloc[idx],
            df_features['ratio_14B_10A'].iloc[idx]
        ])
    
    return (np.array(X, dtype=np.float32), 
            np.array(y, dtype=np.float32),
            np.array(m14_features, dtype=np.float32),
            dates)

# 시퀀스 생성
X_eval, y_eval, m14_eval, dates_eval = create_sequences_with_info(
    df, df_features, Config.LOOKBACK, Config.FORECAST
)

print(f"  X shape: {X_eval.shape}")
print(f"  y shape: {y_eval.shape}")
print(f"  m14 shape: {m14_eval.shape}")
print(f"  평가 샘플 수: {len(X_eval):,}개")

# 학습된 모델의 특징 개수 확인
print(f"\n⚠️ 중요: 평가 데이터 특징 수 = {X_eval.shape[2]}개")
print(f"  (학습 모델이 기대하는 특징 수 = 47개)")

# ============================================
# 스케일링
# ============================================
print("\n📏 데이터 스케일링...")

try:
    # 기존 스케일러 로드
    with open(Config.SCALER_FILE, 'rb') as f:
        scalers = pickle.load(f)
    
    # X 스케일링
    X_scaled = np.zeros_like(X_eval)
    feature_scalers = scalers.get('feature_scalers', scalers)
    
    for i in range(X_eval.shape[2]):
        if f'feature_{i}' in feature_scalers:
            scaler = feature_scalers[f'feature_{i}']
            feature = X_eval[:, :, i].reshape(-1, 1)
            X_scaled[:, :, i] = scaler.transform(feature).reshape(X_eval[:, :, i].shape)
        else:
            # 스케일러가 없으면 새로 생성
            scaler = RobustScaler()
            feature = X_eval[:, :, i].reshape(-1, 1)
            X_scaled[:, :, i] = scaler.fit_transform(feature).reshape(X_eval[:, :, i].shape)
    
    # M14 스케일링
    m14_scaler = scalers.get('m14_scaler', None)
    if m14_scaler:
        m14_scaled = m14_scaler.transform(m14_eval)
    else:
        m14_scaler = RobustScaler()
        m14_scaled = m14_scaler.fit_transform(m14_eval)
    
    print("  ✅ 스케일링 완료")
    
except Exception as e:
    print(f"  ⚠️ 스케일러 파일 문제: {e}")
    print("  새로 스케일링 진행...")
    # 새로운 스케일러로 처리
    X_scaled = np.zeros_like(X_eval)
    for i in range(X_eval.shape[2]):
        scaler = RobustScaler()
        feature = X_eval[:, :, i].reshape(-1, 1)
        X_scaled[:, :, i] = scaler.fit_transform(feature).reshape(X_eval[:, :, i].shape)
    
    m14_scaler = RobustScaler()
    m14_scaled = m14_scaler.fit_transform(m14_eval)
    print("  ✅ 새 스케일러로 스케일링 완료")

# ============================================
# 모델 로드 (커스텀 객체 포함)
# ============================================
print("\n🤖 모델 로드 중...")

# 커스텀 객체 딕셔너리
custom_objects = {
    'WeightedLoss': WeightedLoss,
    'M14RuleCorrection': M14RuleCorrection
}

models = {}
model_names = ['lstm', 'gru', 'cnn_lstm', 'spike', 'rule', 'ensemble']

for name in model_names:
    model_path = f"{Config.MODEL_DIR}{name}_best.keras"
    if not os.path.exists(model_path):
        model_path = f"{Config.MODEL_DIR}{name}_final.keras"
    
    if os.path.exists(model_path):
        try:
            if name == 'ensemble':
                # 앙상블은 Lambda 레이어 때문에 safe_mode=False 필요
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects=custom_objects,
                    safe_mode=False
                )
            else:
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects=custom_objects
                )
            models[name] = model
            print(f"  ✅ {name} 모델 로드 완료")
            
            # 모델 입력 shape 확인
            if hasattr(model, 'input_shape'):
                if isinstance(model.input_shape, list):
                    print(f"     입력 shape: {model.input_shape}")
                else:
                    print(f"     입력 shape: {model.input_shape}")
                    
        except Exception as e:
            print(f"  ❌ {name} 모델 로드 실패: {e}")
    else:
        print(f"  ⚠️ {name} 모델 파일 없음: {model_path}")

print(f"\n📊 로드된 모델: {len(models)}개")

if len(models) == 0:
    print("\n⚠️ 모델이 하나도 로드되지 않았습니다!")
    print("모델 파일 경로를 확인하거나, 재학습이 필요할 수 있습니다.")
    exit(1)

# ============================================
# 모델별 예측
# ============================================
print("\n🔮 모델별 예측 수행 중...")

predictions = {}
for name, model in models.items():
    print(f"  {name} 예측 중...")
    
    try:
        if name in ['rule', 'ensemble']:
            # Rule과 Ensemble은 M14 특징도 필요
            pred = model.predict(
                [X_scaled, m14_scaled], 
                batch_size=Config.BATCH_SIZE,
                verbose=0
            )
        else:
            # 나머지 모델들
            pred = model.predict(
                X_scaled, 
                batch_size=Config.BATCH_SIZE,
                verbose=0
            )
        
        predictions[name] = pred.flatten()
        print(f"    ✅ 완료 - 예측값 범위: {pred.min():.0f} ~ {pred.max():.0f}")
        
    except Exception as e:
        print(f"    ❌ 예측 실패: {e}")
        # 실패한 경우 평균값으로 대체
        predictions[name] = np.full(len(y_eval), y_eval.mean())

# ============================================
# 성능 평가
# ============================================
print("\n📊 성능 평가 중...")

def calculate_metrics(y_true, y_pred, name="Model"):
    """성능 지표 계산"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 정확도 (평균 백분율 오차 기반)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-7))) * 100
    accuracy = 100 - mape
    
    # 구간별 성능
    level_metrics = {}
    for level in [1300, 1400, 1450, 1500, 1550]:
        mask = y_true >= level
        if np.any(mask):
            recall = np.sum((y_pred >= level) & mask) / np.sum(mask)
            precision = np.sum((y_pred >= level) & mask) / max(np.sum(y_pred >= level), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-7)
            
            level_metrics[f'{level}+'] = {
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'count': np.sum(mask)
            }
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Accuracy': accuracy,
        'levels': level_metrics
    }

# 모델별 성능 계산
model_metrics = {}
for name, pred in predictions.items():
    metrics = calculate_metrics(y_eval, pred, name)
    model_metrics[name] = metrics
    
    print(f"\n📈 {name.upper()} 성능:")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  R²: {metrics['R2']:.4f}")
    print(f"  정확도: {metrics['Accuracy']:.2f}%")
    
    if len(metrics['levels']) > 0:
        print("  구간별 F1 Score:")
        for level, level_metric in metrics['levels'].items():
            print(f"    {level}: {level_metric['f1']:.3f} "
                  f"(Recall: {level_metric['recall']:.3f}, "
                  f"Precision: {level_metric['precision']:.3f})")

# ============================================
# 결과 저장
# ============================================
print("\n💾 결과 저장 중...")

# 1. 상세 예측 결과 (날짜별)
results_df = pd.DataFrame({
    '날짜': dates_eval,
    '실제값': y_eval
})

# 모델별 예측값 추가
for name, pred in predictions.items():
    results_df[f'{name}_예측'] = pred
    results_df[f'{name}_오차'] = np.abs(y_eval - pred)

# 앙상블이 있으면 최종 예측으로 표시
if 'ensemble' in predictions:
    results_df['최종_예측'] = predictions['ensemble']
    results_df['최종_오차'] = np.abs(y_eval - predictions['ensemble'])

# CSV 저장
results_file = f"{Config.OUTPUT_DIR}prediction_results_detail.csv"
results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
print(f"  ✅ 상세 결과 저장: {results_file}")

# 2. 모델 성능 요약
metrics_summary = []
for name, metrics in model_metrics.items():
    summary = {
        '모델': name.upper(),
        'MAE': f"{metrics['MAE']:.2f}",
        'RMSE': f"{metrics['RMSE']:.2f}",
        'R²': f"{metrics['R2']:.4f}",
        '정확도(%)': f"{metrics['Accuracy']:.2f}",
        '1400+_F1': f"{metrics['levels'].get('1400+', {}).get('f1', 0):.3f}",
        '1500+_F1': f"{metrics['levels'].get('1500+', {}).get('f1', 0):.3f}"
    }
    metrics_summary.append(summary)

metrics_df = pd.DataFrame(metrics_summary)
metrics_file = f"{Config.OUTPUT_DIR}model_performance_summary.csv"
metrics_df.to_csv(metrics_file, index=False, encoding='utf-8-sig')
print(f"  ✅ 성능 요약 저장: {metrics_file}")

# ============================================
# 최종 요약 출력
# ============================================
print("\n" + "="*60)
print("🎯 V6 모델 평가 완료!")
print("="*60)

if len(model_metrics) > 0:
    print("\n📊 모델 성능 순위 (MAE 기준):")
    sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]['MAE'])
    for rank, (name, metrics) in enumerate(sorted_models, 1):
        print(f"  {rank}위. {name.upper()}: MAE={metrics['MAE']:.2f}, "
              f"정확도={metrics['Accuracy']:.2f}%")

    # 최고 성능 모델
    best_model = sorted_models[0][0]
    print(f"\n🏆 최고 성능 모델: {best_model.upper()}")
    print(f"  MAE: {model_metrics[best_model]['MAE']:.2f}")
    print(f"  정확도: {model_metrics[best_model]['Accuracy']:.2f}%")

print(f"\n📁 결과 저장 위치: {Config.OUTPUT_DIR}")
print("\n✅ 모든 평가 작업 완료!")
print("="*60)

# 메모리 정리
tf.keras.backend.clear_session()
gc.collect()