"""
🚀 EXTREME_NET 전용 반도체 물류 예측 시스템
==========================================
100분 데이터 → 10분 후 예측
TensorFlow 2.16.1 CPU/GPU 지원
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import warnings
import os
import gc
import pickle
import json
from datetime import datetime

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# ====================================
# GPU 설정
# ====================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"🔧 GPU {len(gpus)}개 발견")
    for i, gpu in enumerate(gpus):
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPU:{i} 메모리 동적 할당 활성화")
    strategy = tf.distribute.MirroredStrategy()
    device = f'GPU x {len(gpus)}개'
else:
    print("⚠️ GPU 없음, CPU 사용")
    strategy = tf.distribute.get_strategy()
    device = 'CPU'

print("="*80)
print("🚀 EXTREME_NET 반도체 물류 예측 시스템")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🔧 Device: {device}")
print("="*80)

# ====================================
# EXTREME_NET 모델 정의
# ====================================
def build_extreme_net(input_shape):
    """
    EXTREME_NET - LSTM + Multi-Head Attention 융합
    특징: 장기 패턴(LSTM) + 중요 시점 포착(Attention)
    """
    inputs = Input(shape=input_shape, name='extreme_inputs')
    
    # LSTM 브랜치 (시계열 패턴 학습)
    lstm = LSTM(48, return_sequences=True, name='lstm1')(inputs)
    lstm = LSTM(24, name='lstm2')(lstm)
    
    # Attention 브랜치 (중요 시점 포착)
    attn = MultiHeadAttention(
        num_heads=4, 
        key_dim=12,
        name='multi_head_attention'
    )(inputs, inputs)
    attn = GlobalAveragePooling1D(name='global_avg_pool')(attn)
    
    # 융합
    x = Concatenate(name='fusion')([lstm, attn])
    x = BatchNormalization(name='batch_norm')(x)
    x = Dense(48, activation='relu', name='dense1')(x)
    x = Dropout(0.2, name='dropout')(x)
    
    # 출력층
    out_reg = Dense(1, name='regression')(x)  # 회귀 (물류량 예측)
    out_cls = Dense(3, activation='softmax', name='classification')(x)  # 분류 (구간)
    
    model = Model(inputs, [out_reg, out_cls], name='ExtremeNet')
    
    print("\n📊 EXTREME_NET 구조:")
    print(f"  - LSTM 레이어: 48 → 24")
    print(f"  - Attention Heads: 4개 (key_dim=12)")
    print(f"  - 융합 Dense: 48")
    print(f"  - Dropout: 20%")
    print(f"  - 출력: 회귀 + 3구간 분류")
    
    return model

# ====================================
# 데이터 처리 클래스
# ====================================
class ExtremeDataProcessor:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_scaler = StandardScaler()  # 특징용
        self.target_scaler = RobustScaler()     # 타겟용 (이상치에 강건)
        self.feature_columns = None
        self.statistics = {}
        
    def save_all(self, path='extreme_net_model/'):
        """모든 설정 저장"""
        os.makedirs(path, exist_ok=True)
        
        # 스케일러 저장
        with open(f'{path}feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(f'{path}target_scaler.pkl', 'wb') as f:
            pickle.dump(self.target_scaler, f)
            
        # 설정 저장
        config = {
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'feature_columns': self.feature_columns,
            'statistics': self.statistics
        }
        with open(f'{path}config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"✅ 모든 설정 저장: {path}")
    
    def load_all(self, path='extreme_net_model/'):
        """모든 설정 로드"""
        with open(f'{path}feature_scaler.pkl', 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(f'{path}target_scaler.pkl', 'rb') as f:
            self.target_scaler = pickle.load(f)
        with open(f'{path}config.json', 'r') as f:
            config = json.load(f)
            self.seq_len = config['seq_len']
            self.pred_len = config['pred_len']
            self.feature_columns = config['feature_columns']
            self.statistics = config['statistics']
        print(f"✅ 모든 설정 로드: {path}")
        
    def load_and_process(self, filepath):
        print(f"\n📂 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✅ 원본: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        # 0값 제거
        before = len(df)
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        if before - len(df) > 0:
            print(f"✅ 0값 제거: {before - len(df):,}개")
        
        # 시간 정렬
        if 'CURRTIME' in df.columns:
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                           format='%Y%m%d%H%M', errors='coerce')
            df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 통계 저장
        self.statistics = {
            'mean': float(df['TOTALCNT'].mean()),
            'std': float(df['TOTALCNT'].std()),
            'min': float(df['TOTALCNT'].min()),
            'max': float(df['TOTALCNT'].max()),
            'median': float(df['TOTALCNT'].median())
        }
        
        print(f"\n📊 TOTALCNT 통계:")
        print(f"  평균: {self.statistics['mean']:,.0f}")
        print(f"  표준편차: {self.statistics['std']:,.0f}")
        print(f"  최소: {self.statistics['min']:,}")
        print(f"  최대: {self.statistics['max']:,}")
        print(f"  중앙값: {self.statistics['median']:,}")
        
        # 구간 분포
        normal = (df['TOTALCNT'] < 1400).sum()
        warning = ((df['TOTALCNT'] >= 1400) & (df['TOTALCNT'] < 1700)).sum()
        critical = (df['TOTALCNT'] >= 1700).sum()
        
        print(f"\n📊 3구간 분포:")
        print(f"  Level 0 (정상 < 1400): {normal:,}개 ({normal/len(df)*100:.1f}%)")
        print(f"  Level 1 (주의 1400-1699): {warning:,}개 ({warning/len(df)*100:.1f}%)")
        print(f"  Level 2 (위험 ≥ 1700): {critical:,}개 ({critical/len(df)*100:.1f}%)")
        
        # 급증 패턴 감지
        spike_count = 0
        for i in range(10, len(df)):
            if df.loc[i, 'TOTALCNT'] - df.loc[i-10, 'TOTALCNT'] > 100:
                spike_count += 1
        print(f"\n⚠️ 급증 패턴 (10분간 100+ 증가): {spike_count}회")
        
        return df
    
    def create_features(self, df):
        print("\n⚙️ EXTREME_NET용 특성 생성...")
        
        # 핵심 특성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        df['SPIKE'] = ((df['M14AM14B'] / (df['M14AM10A'] + 1)) > 4).astype(float)
        
        # 시간 특성
        if 'CURRTIME' in df.columns:
            df['HOUR'] = df['CURRTIME'].dt.hour
            df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
            df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        else:
            df['HOUR_SIN'] = 0
            df['HOUR_COS'] = 1
        
        # 이동평균 & 변동성
        for w in [5, 10, 20, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
        
        # 변화율
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_5'] = df['TOTALCNT'].diff(5).fillna(0)
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        df['CHANGE_RATE'] = df['TOTALCNT'].pct_change(10).fillna(0) * 100
        
        # 컬럼별 특성
        for col in ['M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16']:
            if col in df.columns:
                df[f'{col}_MA10'] = df[col].rolling(10, min_periods=1).mean()
                df[f'{col}_CHANGE'] = df[col].diff(10).fillna(0)
        
        # 타겟 생성
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        df['LEVEL'] = 0
        df.loc[df['TARGET'] >= 1400, 'LEVEL'] = 1
        df.loc[df['TARGET'] >= 1700, 'LEVEL'] = 2
        
        # 이상신호 감지
        df['ANOMALY'] = ((df['TARGET'] >= 1651) & (df['TARGET'] <= 1682)).astype(int)
        
        df = df.dropna()
        print(f"✅ 특성 생성 완료: {len(df.columns)}개 컬럼")
        print(f"✅ 유효 데이터: {len(df):,}행")
        
        return df
    
    def create_sequences(self, df):
        print("\n🔄 EXTREME_NET 시퀀스 생성...")
        
        # 특성 선택
        base_features = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN', 'SPIKE',
            'HOUR_SIN', 'HOUR_COS'
        ]
        
        ma_features = [f'MA_{w}' for w in [5, 10, 20, 30]]
        std_features = [f'STD_{w}' for w in [5, 10, 20, 30]]
        change_features = ['CHANGE_1', 'CHANGE_5', 'CHANGE_10', 'CHANGE_RATE']
        
        col_features = []
        for col in ['M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16']:
            if f'{col}_MA10' in df.columns:
                col_features.extend([f'{col}_MA10', f'{col}_CHANGE'])
        
        self.feature_columns = base_features + ma_features + std_features + change_features + col_features
        self.feature_columns = [f for f in self.feature_columns if f in df.columns]
        
        print(f"  선택된 특성: {len(self.feature_columns)}개")
        
        X_data = df[self.feature_columns].values.astype(np.float32)
        y_reg = df['TARGET'].values.astype(np.float32)
        y_cls = df['LEVEL'].values.astype(np.int32)
        y_anomaly = df['ANOMALY'].values.astype(np.int32)
        
        # 분할 비율
        total_samples = len(X_data) - self.seq_len
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)
        
        # 스케일링 (훈련 데이터로만 학습)
        print("  스케일링 적용...")
        self.feature_scaler.fit(X_data[:train_end + self.seq_len])
        self.target_scaler.fit(y_reg[:train_end].reshape(-1, 1))
        
        X_scaled = self.feature_scaler.transform(X_data)
        y_reg_scaled = self.target_scaler.transform(y_reg.reshape(-1, 1)).flatten()
        
        # 시퀀스 생성
        X, y_r, y_c, y_a = [], [], [], []
        for i in range(total_samples):
            X.append(X_scaled[i:i+self.seq_len])
            y_r.append(y_reg_scaled[i+self.seq_len-1])
            y_c.append(y_cls[i+self.seq_len-1])
            y_a.append(y_anomaly[i+self.seq_len-1])
        
        X = np.array(X, dtype=np.float32)
        y_r = np.array(y_r, dtype=np.float32)
        y_c = np.array(y_c, dtype=np.int32)
        y_a = np.array(y_a, dtype=np.int32)
        
        # 시계열 순서 유지 분할
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = (y_r[:train_end], y_c[:train_end], y_a[:train_end])
        y_val = (y_r[train_end:val_end], y_c[train_end:val_end], y_a[train_end:val_end])
        y_test = (y_r[val_end:], y_c[val_end:], y_a[val_end:])
        
        print(f"\n📊 데이터 분할:")
        print(f"  Train: {X_train.shape} (70%)")
        print(f"  Val: {X_val.shape} (15%)")
        print(f"  Test: {X_test.shape} (15%)")
        
        # 구간 분포 확인
        for name, y_cls_data in [('Train', y_train[1]), ('Val', y_val[1]), ('Test', y_test[1])]:
            counts = np.bincount(y_cls_data)
            print(f"\n  {name} 구간 분포:")
            for i, count in enumerate(counts):
                print(f"    Level {i}: {count:,}개 ({count/len(y_cls_data)*100:.1f}%)")
        
        # 모든 설정 저장
        self.save_all()
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ====================================
# EXTREME_NET 학습 함수
# ====================================
def train_extreme_net(train_data, val_data, test_data, processor):
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # 모델 생성
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    with strategy.scope():
        model = build_extreme_net(input_shape)
        
        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'regression': 'mae',
                'classification': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'regression': 0.7,
                'classification': 0.3
            },
            metrics={
                'regression': ['mae', 'mse'],
                'classification': ['accuracy']
            }
        )
    
    print("\n" + "="*60)
    print("🎯 EXTREME_NET 학습 시작")
    print("="*60)
    
    # 콜백 설정
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'extreme_net_model/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 배치 크기
    batch_size = 64 * strategy.num_replicas_in_sync if gpus else 32
    
    # 학습
    print(f"\n📈 학습 설정:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: 50 (Early Stopping)")
    print(f"  Learning Rate: 0.001 (ReduceLROnPlateau)")
    
    history = model.fit(
        X_train,
        {
            'regression': y_train[0],
            'classification': y_train[1]
        },
        validation_data=(
            X_val,
            {
                'regression': y_val[0],
                'classification': y_val[1]
            }
        ),
        epochs=50,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    print("\n" + "="*60)
    print("📊 EXTREME_NET 성능 평가")
    print("="*60)
    
    # 예측
    preds = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_reg_pred = preds[0].flatten()
    y_cls_pred = np.argmax(preds[1], axis=1)
    
    # 역변환
    y_reg_true_orig = processor.target_scaler.inverse_transform(
        y_test[0].reshape(-1, 1)
    ).flatten()
    y_reg_pred_orig = processor.target_scaler.inverse_transform(
        y_reg_pred.reshape(-1, 1)
    ).flatten()
    
    # 회귀 메트릭
    mae = mean_absolute_error(y_reg_true_orig, y_reg_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_reg_true_orig, y_reg_pred_orig))
    r2 = r2_score(y_reg_true_orig, y_reg_pred_orig)
    mape = np.mean(np.abs((y_reg_true_orig - y_reg_pred_orig) / y_reg_true_orig)) * 100
    
    print(f"\n📈 회귀 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # 분류 메트릭
    acc = accuracy_score(y_test[1], y_cls_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test[1], y_cls_pred, average='weighted'
    )
    
    print(f"\n📊 3구간 분류 성능:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # 구간별 성능
    cm = confusion_matrix(y_test[1], y_cls_pred)
    print(f"\n📋 Confusion Matrix:")
    print("  실제\\예측  Level0  Level1  Level2")
    for i in range(3):
        print(f"  Level{i}:   {cm[i, 0]:6d}  {cm[i, 1]:6d}  {cm[i, 2]:6d}")
    
    # 위험 감지 성능 (Level 2)
    if 2 in y_test[1]:
        danger_true = (y_test[1] == 2)
        danger_pred = (y_cls_pred == 2)
        
        tp = np.sum(danger_true & danger_pred)
        fp = np.sum(~danger_true & danger_pred)
        fn = np.sum(danger_true & ~danger_pred)
        
        danger_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        danger_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        danger_f1 = 2 * (danger_precision * danger_recall) / (danger_precision + danger_recall) \
                    if (danger_precision + danger_recall) > 0 else 0
        
        print(f"\n⚠️ 위험 구간(Level 2) 감지 성능:")
        print(f"  Precision: {danger_precision:.4f}")
        print(f"  Recall: {danger_recall:.4f}")
        print(f"  F1-Score: {danger_f1:.4f}")
        print(f"  정확 예측: {tp}개, 오탐: {fp}개, 놓침: {fn}개")
    
    # 급증 감지 성능
    spike_detected = 0
    spike_total = 0
    for i in range(len(y_reg_pred_orig)):
        if i > 0:
            actual_change = y_reg_true_orig[i] - y_reg_true_orig[i-1]
            pred_change = y_reg_pred_orig[i] - y_reg_pred_orig[i-1]
            if actual_change > 50:  # 급증 기준
                spike_total += 1
                if pred_change > 30:  # 예측 급증
                    spike_detected += 1
    
    if spike_total > 0:
        spike_detection_rate = spike_detected / spike_total
        print(f"\n📈 급증 감지 성능:")
        print(f"  급증 감지율: {spike_detection_rate:.2%}")
        print(f"  감지된 급증: {spike_detected}/{spike_total}")
    
    # 모델 저장
    os.makedirs('extreme_net_model', exist_ok=True)
    model.save('extreme_net_model/extreme_net_final.keras')
    print(f"\n💾 최종 모델 저장: extreme_net_model/extreme_net_final.keras")
    
    # 결과 저장
    results = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Danger_Precision': danger_precision if 'danger_precision' in locals() else 0,
        'Danger_Recall': danger_recall if 'danger_recall' in locals() else 0,
        'Danger_F1': danger_f1 if 'danger_f1' in locals() else 0,
        'Spike_Detection': spike_detection_rate if 'spike_detection_rate' in locals() else 0
    }
    
    # CSV 저장
    pd.DataFrame([results]).to_csv('extreme_net_results.csv', index=False)
    print(f"💾 결과 저장: extreme_net_results.csv")
    
    return model, results, history

# ====================================
# 실시간 예측 클래스
# ====================================
class ExtremeNetPredictor:
    def __init__(self, model_path='extreme_net_model/extreme_net_final.keras'):
        """EXTREME_NET 실시간 예측기"""
        self.model = tf.keras.models.load_model(model_path)
        self.processor = ExtremeDataProcessor()
        self.processor.load_all('extreme_net_model/')
        print(f"✅ EXTREME_NET 예측기 준비 완료")
        
    def predict(self, data_100min):
        """100분 데이터로 10분 후 예측"""
        # 데이터 검증
        if len(data_100min) != 100:
            raise ValueError(f"100분 데이터 필요 (현재: {len(data_100min)})")
        
        # 특성 추출
        X = data_100min[self.processor.feature_columns].values
        
        # 스케일링
        X_scaled = self.processor.feature_scaler.transform(X)
        X_seq = X_scaled.reshape(1, 100, -1).astype(np.float32)
        
        # 예측
        preds = self.model.predict(X_seq, verbose=0)
        y_reg_scaled = preds[0][0, 0]
        y_cls_probs = preds[1][0]
        y_cls = np.argmax(y_cls_probs)
        
        # 역변환
        y_reg = self.processor.target_scaler.inverse_transform([[y_reg_scaled]])[0, 0]
        
        # 레벨 정의
        levels = {
            0: '정상 (< 1400)',
            1: '주의 (1400-1699)',
            2: '위험 (≥ 1700)'
        }
        
        # 이상신호 감지
        is_anomaly = (1651 <= y_reg <= 1682)
        
        # 급증 여부
        current = data_100min['TOTALCNT'].iloc[-1]
        is_spike = (y_reg - current > 100)
        
        return {
            'prediction': float(y_reg),
            'current': float(current),
            'change': float(y_reg - current),
            'change_rate': float((y_reg - current) / current * 100),
            'level': y_cls,
            'level_name': levels[y_cls],
            'level_probs': {
                '정상': float(y_cls_probs[0]),
                '주의': float(y_cls_probs[1]),
                '위험': float(y_cls_probs[2])
            },
            'confidence': float(np.max(y_cls_probs)),
            'is_anomaly': is_anomaly,
            'is_spike': is_spike,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def batch_predict(self, df, save_path='predictions.csv'):
        """배치 예측"""
        predictions = []
        
        for i in range(100, len(df)):
            data_100min = df.iloc[i-100:i]
            pred = self.predict(data_100min)
            
            # 실제값 추가
            if i + 10 < len(df):
                pred['actual'] = float(df.iloc[i + 10]['TOTALCNT'])
                pred['error'] = abs(pred['prediction'] - pred['actual'])
                pred['error_rate'] = pred['error'] / pred['actual'] * 100
            
            predictions.append(pred)
            
            if len(predictions) % 100 == 0:
                print(f"  처리: {len(predictions)}개")
        
        # 저장
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(save_path, index=False)
        print(f"\n💾 예측 결과 저장: {save_path}")
        
        return pred_df

# ====================================
# 메인 실행
# ====================================
def main():
    print("\n" + "="*80)
    print("🚀 EXTREME_NET 학습 시작")
    print("="*80)
    
    # 데이터 찾기
    data_paths = [
        '/mnt/user-data/uploads/gs.CSV',
        'data/20240201_TO_202507281705.csv',
        'data/20250731_to20250806.csv',
        'uu.csv',
        'uu2.csv',
        'data.csv'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            print(f"✅ 데이터 발견: {path}")
            break
    
    if not data_path:
        print("❌ 데이터 파일을 찾을 수 없습니다!")
        print("다음 경로 중 하나에 데이터를 배치하세요:")
        for p in data_paths:
            print(f"  - {p}")
        return
    
    # 데이터 처리
    processor = ExtremeDataProcessor()
    df = processor.load_and_process(data_path)
    df = processor.create_features(df)
    train_data, val_data, test_data = processor.create_sequences(df)
    
    # EXTREME_NET 학습
    model, results, history = train_extreme_net(
        train_data, val_data, test_data, processor
    )
    
    # 최종 결과 출력
    print("\n" + "="*80)
    print("🏆 EXTREME_NET 최종 성능")
    print("="*80)
    
    print("\n📊 핵심 지표:")
    print(f"  예측 정확도 (R²): {results['R2']:.4f}")
    print(f"  평균 오차 (MAE): {results['MAE']:.2f}")
    print(f"  3구간 정확도: {results['Accuracy']:.4f}")
    print(f"  위험 감지 F1: {results['Danger_F1']:.4f}")
    print(f"  급증 감지율: {results['Spike_Detection']:.2%}")
    
    # 실시간 예측 테스트
    print("\n" + "="*80)
    print("🔮 실시간 예측 테스트")
    print("="*80)
    
    predictor = ExtremeNetPredictor()
    
    # 테스트 샘플
    if len(df) >= 100:
        test_sample = df.iloc[-100:].copy()
        result = predictor.predict(test_sample)
        
        print(f"\n📈 예측 결과:")
        print(f"  현재: {result['current']:.0f}")
        print(f"  10분 후 예측: {result['prediction']:.0f}")
        print(f"  변화량: {result['change']:+.0f} ({result['change_rate']:+.1f}%)")
        print(f"  레벨: {result['level_name']}")
        print(f"  신뢰도: {result['confidence']:.2%}")
        
        print(f"\n  구간별 확률:")
        for level, prob in result['level_probs'].items():
            print(f"    {level}: {prob:.2%}")
        
        if result['is_spike']:
            print(f"  ⚠️ 급증 경고!")
        if result['is_anomaly']:
            print(f"  ⚠️ 이상신호 감지 (1651-1682)")
    
    print("\n✅ EXTREME_NET 학습 및 평가 완료!")
    print("="*80)
    
    # Patch Time Series Transformer 안내
    print("\n" + "="*80)
    print("📌 Patch Time Series Transformer 안내")
    print("="*80)
    print("100만개 데이터가 준비되면 알려주세요!")
    print("PatchTST 모델로 더 높은 성능을 달성할 수 있습니다.")
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = main()