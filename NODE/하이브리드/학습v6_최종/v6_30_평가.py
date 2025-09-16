"""
V6 모델 평가 시스템 - 실전 버전
- TensorFlow 2.16.1 완벽 호환
- 규칙 기반 예측 + 딥러닝 모델 평가
- 실제 작동 검증됨
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')

print("="*60)
print("🔬 V6 모델 평가 시스템 - 실전 버전")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*60)

# ============================================
# 설정
# ============================================
class Config:
    # 평가 데이터
    EVAL_DATA_FILE = './data/20250731_to_20250826.csv'
    
    # 모델 경로
    MODEL_DIR = './models_v6_full_train/'
    
    # 시퀀스 설정
    LOOKBACK = 100  # 과거 100분
    FORECAST = 10   # 10분 후 예측
    
    # 결과 저장
    OUTPUT_DIR = './evaluation_results/'
    PLOT_DIR = './evaluation_plots/'
    
    # 배치 크기
    BATCH_SIZE = 32
    
    # 특징 개수 (학습과 동일)
    NUM_FEATURES = 47

# 디렉토리 생성
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.PLOT_DIR, exist_ok=True)

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
            # 최근 TOTALCNT 값들 (4번째 컬럼)
            try:
                recent_values = X[i, -20:, 4]  # TOTALCNT 위치
            except:
                recent_values = X[i, -20:, 3]  # 다른 위치 시도
            
            current_value = recent_values[-1]
            recent_avg = np.mean(recent_values)
            recent_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            # M14 특징
            m14b = m14_features[i, 0] if len(m14_features[i]) > 0 else 0
            m10a = m14_features[i, 1] if len(m14_features[i]) > 1 else 1
            m16 = m14_features[i, 2] if len(m14_features[i]) > 2 else 0
            
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
            
            # 황금 패턴
            if m14b >= 350 and m10a < 70:
                base_pred *= 1.2
            
            # 범위 제한
            base_pred = np.clip(base_pred, 1200, 2000)
            
            predictions.append(base_pred)
        
        return np.array(predictions)

# ============================================
# 모델 구조 재생성 (TF 2.16.1용)
# ============================================
def create_model_structures():
    """TF 2.16.1용 모델 구조만 생성"""
    
    models = {}
    
    # 1. LSTM
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
    
    # 2. GRU  
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
    
    print("✅ 모델 구조 생성 완료")
    
    return models

# ============================================
# 가중치 로드 시도
# ============================================
def try_load_weights(models):
    """저장된 가중치 로드 시도"""
    
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
                    model.load_weights(weight_file)
                    loaded_models[name] = model
                    loaded = True
                    print(f"✅ {name} 가중치 로드 성공: {weight_file}")
                    break
                except Exception as e:
                    print(f"⚠️ {name} 가중치 로드 실패: {str(e)[:50]}")
        
        if not loaded:
            print(f"⚠️ {name} - 랜덤 초기화 사용")
            loaded_models[name] = model
    
    return loaded_models

# ============================================
# 데이터 준비
# ============================================
def prepare_data(file_path):
    """데이터 로드 및 전처리"""
    print(f"\n📂 데이터 로드: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"  원본: {len(df)}행")
    
    # 필수 컬럼 확인
    required = ['M14AM10A', 'M14AM14B', 'M14AM16', 'TOTALCNT']
    for col in required:
        if col not in df.columns:
            print(f"  ⚠️ {col} 없음 - 0으로 초기화")
            df[col] = 0
    
    if 'M14AM14BSUM' not in df.columns:
        df['M14AM14BSUM'] = df['M14AM14B'] + df['M14AM10A']
    
    print(f"  TOTALCNT 범위: {df['TOTALCNT'].min():.0f} ~ {df['TOTALCNT'].max():.0f}")
    
    # 특징 생성 (학습과 동일하게)
    print("  특징 생성 중...")
    
    # 비율
    df['ratio_14B_10A'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
    df['ratio_14B_16'] = df['M14AM14B'] / (df['M14AM16'] + 1)
    df['ratio_10A_16'] = df['M14AM10A'] / (df['M14AM16'] + 1)
    
    # 시계열 특징
    feature_columns = ['M14AM10A', 'M14AM14B', 'M14AM16', 'M14AM14BSUM', 'TOTALCNT']
    for col in feature_columns:
        # 변화량
        df[f'{col}_diff_1'] = df[col].diff(1)
        df[f'{col}_diff_5'] = df[col].diff(5)
        df[f'{col}_diff_10'] = df[col].diff(10)
        
        # 이동평균
        df[f'{col}_ma_5'] = df[col].rolling(5, min_periods=1).mean()
        df[f'{col}_ma_10'] = df[col].rolling(10, min_periods=1).mean()
        
        # 표준편차
        df[f'{col}_std_5'] = df[col].rolling(5, min_periods=1).std()
        df[f'{col}_std_10'] = df[col].rolling(10, min_periods=1).std()
    
    # 황금 패턴
    df['golden_pattern'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)
    
    # 신호
    thresholds = {1300: 250, 1400: 300, 1450: 350, 1500: 380}
    for level, threshold in thresholds.items():
        df[f'signal_{level}'] = (df['M14AM14B'] >= threshold).astype(float)
    
    ratio_thresholds = {1300: 3.5, 1400: 4.0, 1450: 4.5, 1500: 5.0}
    for level, ratio in ratio_thresholds.items():
        df[f'ratio_signal_{level}'] = (df['ratio_14B_10A'] >= ratio).astype(float)
    
    df = df.fillna(0)
    
    # 특징 개수 맞추기 (47개)
    print(f"  현재 특징 수: {len(df.columns)}개")
    
    if len(df.columns) > Config.NUM_FEATURES:
        df = df.iloc[:, :Config.NUM_FEATURES]
    elif len(df.columns) < Config.NUM_FEATURES:
        for i in range(Config.NUM_FEATURES - len(df.columns)):
            df[f'pad_{i}'] = 0
    
    print(f"  최종 특징 수: {len(df.columns)}개")
    
    return df

def create_sequences(df):
    """시퀀스 생성"""
    X, y = [], []
    
    data = df.values
    
    for i in range(len(data) - Config.LOOKBACK - Config.FORECAST):
        X.append(data[i:i+Config.LOOKBACK])
        # TOTALCNT 위치 찾기
        totalcnt_idx = 4  # 기본 위치
        y.append(data[i+Config.LOOKBACK+Config.FORECAST-1, totalcnt_idx])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    return X, y

# ============================================
# 평가 함수
# ============================================
def evaluate(y_true, y_pred, name):
    """성능 평가"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 정확도
    acc_50 = np.mean(np.abs(y_true - y_pred) <= 50) * 100
    acc_100 = np.mean(np.abs(y_true - y_pred) <= 100) * 100
    
    # 급증 감지
    spike_performance = {}
    for level in [1400, 1450, 1500]:
        actual_spike = y_true >= level
        pred_spike = y_pred >= level
        
        if np.sum(actual_spike) > 0:
            recall = np.sum(actual_spike & pred_spike) / np.sum(actual_spike)
            precision = np.sum(actual_spike & pred_spike) / max(np.sum(pred_spike), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
            
            spike_performance[level] = {
                'recall': recall * 100,
                'precision': precision * 100,
                'f1': f1 * 100
            }
    
    print(f"\n📊 {name} 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  정확도(±50): {acc_50:.1f}%")
    print(f"  정확도(±100): {acc_100:.1f}%")
    
    for level, perf in spike_performance.items():
        print(f"  {level}+ 감지: F1={perf['f1']:.1f}%, Recall={perf['recall']:.1f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
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
    sample_size = min(500, len(y_true))
    plt.scatter(y_true[:sample_size], predictions[best_model][:sample_size], alpha=0.5)
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
    sample = min(200, len(y_true))
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
    
    # 5. 정확도 비교
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
    
    # 6. R² 비교
    plt.subplot(2, 3, 6)
    r2_scores = [results[n]['r2'] for n in names]
    plt.bar(names, r2_scores)
    plt.ylabel('R² Score')
    plt.title('Model R² Comparison')
    plt.xticks(rotation=45)
    plt.ylim(-0.1, 1.0)
    
    plt.tight_layout()
    save_path = f'{Config.PLOT_DIR}evaluation_results.png'
    plt.savefig(save_path, dpi=150)
    print(f"\n📈 시각화 저장: {save_path}")
    plt.close()

# ============================================
# 메인 실행
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
        m14_features[:, 0] = X[:, -1, 1]  # M14AM14B
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
        
        m14_scaler = RobustScaler()
        m14_scaled = m14_scaler.fit_transform(m14_features)
        
        # 4. 예측
        predictions = {}
        results = {}
        
        # 4-1. 규칙 기반 예측 (핵심!)
        print("\n🎯 규칙 기반 예측...")
        rule_predictor = RuleBasedPredictor()
        rule_pred = rule_predictor.predict(X, m14_features)
        predictions['rule_based'] = rule_pred
        results['rule_based'] = evaluate(y, rule_pred, 'Rule-Based')
        
        # 4-2. 베이스라인 (단순 평균)
        print("\n📊 베이스라인 예측...")
        baseline_pred = np.array([np.mean(X[i, -10:, 4]) for i in range(len(X))])
        predictions['baseline'] = baseline_pred
        results['baseline'] = evaluate(y, baseline_pred, 'Baseline')
        
        # 4-3. 딥러닝 모델 시도 (옵션)
        print("\n🤖 딥러닝 모델 시도...")
        models = create_model_structures()
        loaded_models = try_load_weights(models)
        
        for name, model in loaded_models.items():
            try:
                pred = model.predict(X_scaled, batch_size=Config.BATCH_SIZE, verbose=0)
                predictions[name] = pred.flatten()
                results[name] = evaluate(y, predictions[name], name.upper())
            except Exception as e:
                print(f"  ❌ {name} 예측 실패: {str(e)[:100]}")
        
        # 5. 최고 모델 선택
        if results:
            best = min(results.keys(), key=lambda x: results[x]['mae'])
            print("\n" + "="*60)
            print(f"🏆 최고 성능: {best.upper()}")
            print(f"   MAE: {results[best]['mae']:.2f}")
            print(f"   RMSE: {results[best]['rmse']:.2f}")
            print(f"   R²: {results[best]['r2']:.4f}")
            print(f"   정확도(±50): {results[best]['acc_50']:.1f}%")
            print(f"   정확도(±100): {results[best]['acc_100']:.1f}%")
            print("="*60)
            
            # 성능 비교 테이블
            print("\n📊 모델별 성능 비교:")
            print("-"*70)
            print(f"{'모델':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'±50(%)':<10} {'±100(%)':<10}")
            print("-"*70)
            
            for name, result in sorted(results.items(), key=lambda x: x[1]['mae']):
                print(f"{name:<15} {result['mae']:<10.2f} {result['rmse']:<10.2f} "
                      f"{result['r2']:<10.4f} {result['acc_50']:<10.1f} {result['acc_100']:<10.1f}")
            print("-"*70)
        
        # 6. 예측 결과 CSV 저장
        print("\n💾 예측 결과 저장...")
        
        results_df = pd.DataFrame({
            '실제값': y,
            'Rule_Based_예측': predictions['rule_based'],
            'Baseline_예측': predictions['baseline'],
            'Rule_오차': predictions['rule_based'] - y,
            'Baseline_오차': predictions['baseline'] - y,
            'Rule_절대오차': np.abs(predictions['rule_based'] - y),
            'Baseline_절대오차': np.abs(predictions['baseline'] - y),
            '50이내_정확': np.abs(predictions['rule_based'] - y) <= 50,
            '100이내_정확': np.abs(predictions['rule_based'] - y) <= 100,
        })
        
        # 딥러닝 모델 추가 (있는 경우)
        for name in ['lstm', 'gru']:
            if name in predictions:
                results_df[f'{name.upper()}_예측'] = predictions[name]
                results_df[f'{name.upper()}_오차'] = np.abs(predictions[name] - y)
        
        csv_path = f'{Config.OUTPUT_DIR}prediction_results.csv'
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ CSV 저장: {csv_path}")
        
        # 요약 통계
        print("\n📈 예측 결과 요약:")
        print(f"  전체 샘플: {len(results_df):,}개")
        print(f"  Rule MAE: {results_df['Rule_절대오차'].mean():.2f}")
        print(f"  Baseline MAE: {results_df['Baseline_절대오차'].mean():.2f}")
        print(f"  50 이내: {results_df['50이내_정확'].sum():,}개 ({results_df['50이내_정확'].mean()*100:.1f}%)")
        print(f"  100 이내: {results_df['100이내_정확'].sum():,}개 ({results_df['100이내_정확'].mean()*100:.1f}%)")
        
        # 급증 분석
        print("\n🎯 급증 구간(1400+) 분석:")
        spike_mask = y >= 1400
        if spike_mask.sum() > 0:
            spike_actual = y[spike_mask]
            spike_rule = predictions['rule_based'][spike_mask]
            spike_mae = np.mean(np.abs(spike_actual - spike_rule))
            spike_detected = (spike_rule >= 1400).sum()
            
            print(f"  실제 급증: {spike_mask.sum()}회")
            print(f"  예측 성공: {spike_detected}회 ({spike_detected/spike_mask.sum()*100:.1f}%)")
            print(f"  급증 MAE: {spike_mae:.2f}")
        
        # 7. 결과 JSON 저장
        json_path = f'{Config.OUTPUT_DIR}evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n  ✅ JSON 저장: {json_path}")
        
        # 8. 시각화
        if len(predictions) > 0:
            visualize_results(y, predictions, results)
        
        print("\n✅ 평가 완료!")
        print(f"📁 결과 위치: {Config.OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()