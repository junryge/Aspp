"""
V6 모델 평가 시스템 - TensorFlow 2.16.1 전용
- Keras 3.0 호환
- 모델 재생성 후 수동 예측
- 규칙 기반 평가 포함
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
warnings.filterwarnings('ignore')

print("="*60)
print("📊 V6 모델 평가 시스템 - TF 2.16.1 전용")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"📦 Keras: {tf.keras.__version__}")
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
    
    # 결과 저장 경로
    EVAL_RESULT_DIR = './evaluation_results/'
    PLOT_DIR = './evaluation_plots/'
    
    # 배치 크기
    BATCH_SIZE = 32
    
    # 특징 개수
    NUM_FEATURES = 47  # 학습 시 사용한 특징 개수

# 디렉토리 생성
os.makedirs(Config.EVAL_RESULT_DIR, exist_ok=True)
os.makedirs(Config.PLOT_DIR, exist_ok=True)

# ============================================
# TensorFlow 2.16.1용 모델 재생성
# ============================================
def create_models_tf216():
    """TensorFlow 2.16.1용 모델 구조 생성"""
    
    models = {}
    
    # 1. LSTM 모델
    lstm = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(Config.LOOKBACK, Config.NUM_FEATURES)),
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2),
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2),
        tf.keras.layers.LSTM(64, dropout=0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ], name='LSTM_Model_v216')
    models['lstm'] = lstm
    
    # 2. GRU 모델
    gru = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(Config.LOOKBACK, Config.NUM_FEATURES)),
        tf.keras.layers.GRU(256, return_sequences=True, dropout=0.15),
        tf.keras.layers.GRU(128, return_sequences=True, dropout=0.15),
        tf.keras.layers.GRU(64, dropout=0.15),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ], name='GRU_Model_v216')
    models['gru'] = gru
    
    print("✅ TF 2.16.1용 모델 구조 생성 완료")
    
    return models

# ============================================
# 가중치 추출 및 적용 시도
# ============================================
def try_load_weights(models):
    """저장된 모델에서 가중치 추출 시도"""
    
    loaded_models = {}
    
    for name, model in models.items():
        weight_files = [
            f"{Config.MODEL_DIR}{name}_final.keras",
            f"{Config.MODEL_DIR}{name}_best.keras",
            f"{Config.MODEL_DIR}{name}.weights.h5",
        ]
        
        loaded = False
        for weight_file in weight_files:
            if os.path.exists(weight_file):
                try:
                    # 방법 1: 직접 가중치 로드
                    model.load_weights(weight_file)
                    loaded_models[name] = model
                    loaded = True
                    print(f"✅ {name} 가중치 로드 성공")
                    break
                except:
                    try:
                        # 방법 2: 레이어별 가중치 복사
                        temp_model = tf.keras.models.load_model(weight_file, compile=False)
                        for i, layer in enumerate(model.layers):
                            if i < len(temp_model.layers):
                                try:
                                    layer.set_weights(temp_model.layers[i].get_weights())
                                except:
                                    pass
                        loaded_models[name] = model
                        loaded = True
                        print(f"✅ {name} 가중치 부분 로드")
                        break
                    except:
                        pass
        
        if not loaded:
            print(f"⚠️ {name} 가중치 로드 실패 - 랜덤 초기화 사용")
            loaded_models[name] = model
    
    return loaded_models

# ============================================
# 강력한 규칙 기반 예측기
# ============================================
class RuleBasedPredictor:
    """M14 규칙 기반 예측기"""
    
    def __init__(self):
        self.thresholds = {
            'M14B': [250, 300, 350, 400, 450],
            'predictions': [1350, 1400, 1450, 1500, 1550],
            'ratios': [3.0, 4.0, 4.5, 5.0, 5.5],
            'adjustments': [1.02, 1.05, 1.08, 1.10, 1.15]
        }
    
    def predict(self, X, m14_features):
        """규칙 기반 예측"""
        predictions = []
        
        for i in range(len(X)):
            # 최근 값들
            recent_values = X[i, -20:, 0]  # 최근 20개 TOTALCNT
            current_value = recent_values[-1]
            recent_avg = np.mean(recent_values)
            recent_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            # M14 특징
            m14b = m14_features[i, 0] if m14_features.shape[1] > 0 else 0
            m10a = m14_features[i, 1] if m14_features.shape[1] > 1 else 1
            m16 = m14_features[i, 2] if m14_features.shape[1] > 2 else 0
            
            # 기본 예측 (트렌드 기반)
            base_pred = recent_avg + recent_trend * Config.FORECAST
            
            # M14B 임계값 기반 조정
            for j, threshold in enumerate(self.thresholds['M14B']):
                if m14b >= threshold:
                    base_pred = max(base_pred, self.thresholds['predictions'][j])
            
            # 비율 기반 조정
            if m10a > 0:
                ratio = m14b / m10a
                for j, ratio_threshold in enumerate(self.thresholds['ratios']):
                    if ratio >= ratio_threshold:
                        base_pred *= self.thresholds['adjustments'][j]
            
            # 황금 패턴 (M14B 높고 M10A 낮음)
            if m14b >= 350 and m10a < 70:
                base_pred *= 1.2
            
            # 안정화
            base_pred = np.clip(base_pred, 1200, 2000)
            
            predictions.append(base_pred)
        
        return np.array(predictions)

# ============================================
# 데이터 전처리
# ============================================
def prepare_data(file_path):
    """데이터 준비"""
    print(f"\n📂 데이터 로드: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"  원본: {len(df)}행")
    
    # 필수 컬럼
    required = ['M14AM10A', 'M14AM14B', 'M14AM16', 'TOTALCNT']
    for col in required:
        if col not in df.columns:
            df[col] = 0
    
    if 'M14AM14BSUM' not in df.columns:
        df['M14AM14BSUM'] = df['M14AM14B'] + df['M14AM10A']
    
    # 특징 생성
    print("  특징 생성 중...")
    
    # 비율
    df['ratio_14B_10A'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
    df['ratio_14B_16'] = df['M14AM14B'] / (df['M14AM16'] + 1)
    
    # 시계열
    for col in ['TOTALCNT', 'M14AM14B', 'M14AM10A', 'M14AM16']:
        if col in df.columns:
            for period in [1, 5, 10]:
                df[f'{col}_diff_{period}'] = df[col].diff(period)
            for window in [5, 10, 20]:
                df[f'{col}_ma_{window}'] = df[col].rolling(window, min_periods=1).mean()
                df[f'{col}_std_{window}'] = df[col].rolling(window, min_periods=1).std()
    
    # 신호
    df['golden'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)
    
    for t in [250, 300, 350, 400, 450]:
        df[f'sig_{t}'] = (df['M14AM14B'] >= t).astype(float)
    
    df = df.fillna(0)
    
    # 특징 개수 맞추기
    if len(df.columns) > Config.NUM_FEATURES:
        df = df.iloc[:, :Config.NUM_FEATURES]
    elif len(df.columns) < Config.NUM_FEATURES:
        for i in range(Config.NUM_FEATURES - len(df.columns)):
            df[f'pad_{i}'] = 0
    
    print(f"  최종: {len(df.columns)}개 특징")
    
    return df

def create_sequences(df):
    """시퀀스 생성"""
    X, y = [], []
    
    data = df.values
    
    for i in range(len(data) - Config.LOOKBACK - Config.FORECAST):
        X.append(data[i:i+Config.LOOKBACK])
        # 타겟: 10분 후 TOTALCNT
        y.append(data[i+Config.LOOKBACK+Config.FORECAST-1, 3])  # TOTALCNT 인덱스
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    return X, y

# ============================================
# 평가 함수
# ============================================
def evaluate(y_true, y_pred, name):
    """성능 평가"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    acc_50 = np.mean(np.abs(y_true - y_pred) <= 50) * 100
    acc_100 = np.mean(np.abs(y_true - y_pred) <= 100) * 100
    
    # 급증 감지 성능
    spike_levels = [1400, 1450, 1500]
    spike_performance = {}
    
    for level in spike_levels:
        actual_spike = y_true >= level
        pred_spike = y_pred >= level
        
        if np.sum(actual_spike) > 0:
            recall = np.sum(actual_spike & pred_spike) / np.sum(actual_spike)
            if np.sum(pred_spike) > 0:
                precision = np.sum(actual_spike & pred_spike) / np.sum(pred_spike)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            else:
                precision = 0
                f1 = 0
            
            spike_performance[level] = {
                'recall': recall * 100,
                'precision': precision * 100,
                'f1': f1 * 100
            }
    
    print(f"\n📊 {name} 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  정확도(±50): {acc_50:.1f}%")
    print(f"  정확도(±100): {acc_100:.1f}%")
    
    for level, perf in spike_performance.items():
        print(f"  {level}+ 감지: F1={perf['f1']:.1f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'acc_50': acc_50,
        'acc_100': acc_100,
        'spike': spike_performance
    }

# ============================================
# 시각화
# ============================================
def visualize_results(y_true, predictions, results):
    """결과 시각화"""
    
    plt.figure(figsize=(15, 10))
    
    # 1. 예측 vs 실제
    plt.subplot(2, 3, 1)
    best_model = min(results.keys(), key=lambda x: results[x]['mae'])
    plt.scatter(y_true[:500], predictions[best_model][:500], alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{best_model} - Predictions')
    
    # 2. MAE 비교
    plt.subplot(2, 3, 2)
    names = list(results.keys())
    maes = [results[n]['mae'] for n in names]
    plt.bar(names, maes)
    plt.ylabel('MAE')
    plt.title('Model Comparison')
    plt.xticks(rotation=45)
    
    # 3. 시계열
    plt.subplot(2, 3, 3)
    sample = 200
    plt.plot(y_true[:sample], label='Actual', linewidth=2)
    plt.plot(predictions[best_model][:sample], label=f'{best_model}', alpha=0.7)
    plt.legend()
    plt.title('Time Series')
    
    # 4. 오차 분포
    plt.subplot(2, 3, 4)
    errors = predictions[best_model] - y_true
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    
    # 5. 정확도
    plt.subplot(2, 3, 5)
    acc_50 = [results[n]['acc_50'] for n in names]
    acc_100 = [results[n]['acc_100'] for n in names]
    x = np.arange(len(names))
    plt.bar(x - 0.2, acc_50, 0.4, label='±50')
    plt.bar(x + 0.2, acc_100, 0.4, label='±100')
    plt.xticks(x, names, rotation=45)
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Prediction Accuracy')
    
    # 6. 급증 감지
    plt.subplot(2, 3, 6)
    if 1400 in results[best_model].get('spike', {}):
        f1_scores = []
        for name in names:
            if 'spike' in results[name] and 1400 in results[name]['spike']:
                f1_scores.append(results[name]['spike'][1400]['f1'])
            else:
                f1_scores.append(0)
        plt.bar(names, f1_scores)
        plt.ylabel('F1 Score (%)')
        plt.title('1400+ Spike Detection')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_path = f'{Config.PLOT_DIR}evaluation_tf216.png'
    plt.savefig(save_path, dpi=150)
    print(f"\n📈 시각화 저장: {save_path}")
    plt.close()

# ============================================
# 메인 함수
# ============================================
def main():
    print("\n🚀 평가 시작...")
    
    try:
        # 1. 데이터 준비
        df = prepare_data(Config.EVAL_DATA_FILE)
        X, y = create_sequences(df)
        
        print(f"\n📊 데이터 shape:")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")
        print(f"  y 범위: {y.min():.0f} ~ {y.max():.0f}")
        
        # 2. M14 특징 추출
        m14_features = np.zeros((len(X), 4))
        m14_features[:, 0] = X[:, -1, 1]  # M14AM14B (마지막 시점)
        m14_features[:, 1] = X[:, -1, 0]  # M14AM10A
        m14_features[:, 2] = X[:, -1, 2]  # M14AM16
        m14_features[:, 3] = X[:, -1, 1] / (X[:, -1, 0] + 1)  # 비율
        
        # 3. 스케일링
        print("\n📏 스케일링...")
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[2]):
            scaler = RobustScaler()
            feature = X[:, :, i].reshape(-1, 1)
            X_scaled[:, :, i] = scaler.fit_transform(feature).reshape(X[:, :, i].shape)
        
        # 4. 예측
        predictions = {}
        results = {}
        
        # 4-1. 딥러닝 모델 시도
        print("\n🤖 딥러닝 모델 예측 시도...")
        models = create_models_tf216()
        loaded_models = try_load_weights(models)
        
        for name, model in loaded_models.items():
            try:
                pred = model.predict(X_scaled, batch_size=Config.BATCH_SIZE, verbose=0)
                predictions[name] = pred.flatten()
                results[name] = evaluate(y, predictions[name], name)
            except Exception as e:
                print(f"  ❌ {name} 예측 실패: {str(e)[:50]}")
        
        # 4-2. 규칙 기반 예측 (항상 실행)
        print("\n📊 규칙 기반 예측...")
        rule_predictor = RuleBasedPredictor()
        rule_pred = rule_predictor.predict(X, m14_features)
        predictions['rule_based'] = rule_pred
        results['rule_based'] = evaluate(y, rule_pred, 'Rule-Based')
        
        # 4-3. 단순 평균 예측 (베이스라인)
        print("\n📊 베이스라인 예측...")
        baseline_pred = np.array([np.mean(X[i, -10:, 3]) for i in range(len(X))])
        predictions['baseline'] = baseline_pred
        results['baseline'] = evaluate(y, baseline_pred, 'Baseline')
        
        # 5. 최고 모델
        if results:
            best = min(results.keys(), key=lambda x: results[x]['mae'])
            print("\n" + "="*60)
            print(f"🏆 최고 성능: {best.upper()}")
            print(f"   MAE: {results[best]['mae']:.2f}")
            print(f"   정확도(±50): {results[best]['acc_50']:.1f}%")
            print("="*60)
        
        # 6. 결과 저장
        with open(f'{Config.EVAL_RESULT_DIR}results_tf216.json', 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        # 7. 시각화
        if len(predictions) > 0:
            visualize_results(y, predictions, results)
        
        print("\n✅ 평가 완료!")
        print(f"📁 결과: {Config.EVAL_RESULT_DIR}")
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()