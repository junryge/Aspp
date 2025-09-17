"""
🚀 반도체 물류 TOTALCNT 예측 시스템 - Keras 3.0 호환 버전
==========================================================
TensorFlow 2.16 + Keras 3.0 완벽 호환

예측 구간:
- 정상: 0-1399 (56.53%)
- 주의: 1400-1699 (43.12%)  
- 심각: 1700+ (0.35%) ← 핵심 목표

평가 지표:
- MAE, RMSE, R² Score, 정확도
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
import os
import json
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# 설정
tf.random.set_seed(42)
np.random.seed(42)

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 사용: {len(gpus)}개")
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

print("="*80)
print("🚀 반도체 물류 예측 시스템 v15.0 - Keras 3.0 Compatible")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🔧 Device: {'GPU' if gpus else 'CPU'}")
print(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# =========================================
# 체크포인트 관리자
# =========================================

class CheckpointManager:
    """학습 중단/재개 관리"""
    
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_training_state(self, epoch, model_name, history, data_info):
        """학습 상태 저장"""
        state = {
            'epoch': epoch,
            'model_name': model_name,
            'history': history,
            'data_info': data_info,
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.checkpoint_dir, f'{model_name}_state.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 학습 상태 저장: {filepath}")
        
    def load_training_state(self, model_name):
        """학습 상태 로드"""
        filepath = os.path.join(self.checkpoint_dir, f'{model_name}_state.pkl')
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            print(f"📂 학습 상태 로드: {filepath}")
            return state
        return None
    
    def check_existing_model(self, model_name):
        """기존 모델 체크"""
        model_path = os.path.join(self.checkpoint_dir, f'{model_name}_model.keras')
        if os.path.exists(model_path):
            return model_path
        return None

# =========================================
# 데이터 처리 클래스
# =========================================

class DataProcessor:
    """황금 패턴 기반 데이터 처리"""
    
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_scaler = RobustScaler()
        self.target_scaler = MinMaxScaler()
        self.scaler_fitted = False
        
    def save_scalers(self, filepath='scalers.pkl'):
        """스케일러 저장"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'scaler_fitted': self.scaler_fitted
            }, f)
        print(f"💾 스케일러 저장: {filepath}")
    
    def load_scalers(self, filepath='scalers.pkl'):
        """스케일러 로드"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                scalers = pickle.load(f)
                self.feature_scaler = scalers['feature_scaler']
                self.target_scaler = scalers['target_scaler']
                self.scaler_fitted = scalers['scaler_fitted']
            print(f"📂 스케일러 로드: {filepath}")
            return True
        return False
    
    def load_data(self, filepath):
        """데이터 로드"""
        print(f"\n📂 데이터 로딩: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"✅ 데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        # 시간 변환
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 통계
        print(f"\n📊 TOTALCNT 분포:")
        print(f"  최소: {df['TOTALCNT'].min():,}")
        print(f"  평균: {df['TOTALCNT'].mean():,.0f} ± {df['TOTALCNT'].std():.0f}")
        print(f"  최대: {df['TOTALCNT'].max():,}")
        
        # 구간별 분포
        normal_cnt = (df['TOTALCNT'] < 1400).sum()
        warning_cnt = ((df['TOTALCNT'] >= 1400) & (df['TOTALCNT'] < 1700)).sum()
        critical_cnt = (df['TOTALCNT'] >= 1700).sum()
        
        print(f"\n📊 구간별 분포:")
        print(f"  정상(0-1399): {normal_cnt:,}개 ({normal_cnt/len(df)*100:.2f}%)")
        print(f"  주의(1400-1699): {warning_cnt:,}개 ({warning_cnt/len(df)*100:.2f}%)")
        print(f"  심각(1700+): {critical_cnt:,}개 ({critical_cnt/len(df)*100:.2f}%)")
        
        return df
    
    def create_features(self, df):
        """특성 생성"""
        print("\n⚙️ 황금 패턴 특성 생성...")
        
        # 1. 핵심 비율
        df['RATIO_M14B_M14A'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        
        # 2. 황금 패턴
        df['GOLDEN_1'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(int)
        df['GOLDEN_2'] = (df['M14AM14BSUM'] > 500).astype(int)
        df['GOLDEN_3'] = (df['RATIO_M14B_M14A'] > 5).astype(int)
        
        # 3. M14AM14B 구간
        bins = [0, 200, 250, 300, 350, 400, 450, 999999]
        df['M14B_RANGE'] = pd.cut(df['M14AM14B'], bins=bins, labels=False)
        df['M14B_RANGE'] = df['M14B_RANGE'].fillna(0).astype(int)
        
        # 4. 시간 특성
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['DAYOFWEEK'] = df['CURRTIME'].dt.dayofweek
        df['IS_PEAK'] = df['HOUR'].isin([3, 4, 5, 8]).astype(int)
        df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
        df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        
        # 5. 이동 통계
        for w in [10, 30, 60]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
            
        # 6. 변화율
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        df['PCT_CHANGE'] = df['TOTALCNT'].pct_change().fillna(0).clip(-1, 1)
        
        # 7. 타겟 생성
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        
        # 8. 구간 레이블
        def classify_level(x):
            if pd.isna(x):
                return np.nan
            elif x < 1400:
                return 0  # 정상
            elif x < 1700:
                return 1  # 주의
            else:
                return 2  # 심각
        
        df['LEVEL'] = df['TARGET'].apply(classify_level)
        
        print(f"✅ 특성 생성 완료: {len(df.columns)}개 컬럼")
        
        return df
    
    def prepare_sequences(self, df):
        """시퀀스 생성"""
        print("\n🔄 시퀀스 생성...")
        
        # NaN 제거
        df = df.dropna()
        
        # 특성 선택
        feature_cols = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'M14AM16SUM', 'M10AM14A', 'M14BM14A', 'M16M14A', 'M14AM10ASUM',
            'RATIO_M14B_M14A', 'GOLDEN_1', 'GOLDEN_2', 'GOLDEN_3',
            'M14B_RANGE', 'IS_PEAK', 'HOUR_SIN', 'HOUR_COS',
            'MA_10', 'MA_30', 'MA_60', 'STD_10', 'STD_30', 'STD_60',
            'CHANGE_1', 'CHANGE_10', 'PCT_CHANGE'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        print(f"  사용 특성: {len(available_cols)}개")
        
        # 데이터 준비
        X_data = df[available_cols].values
        y_reg = df['TARGET'].values
        y_cls = df['LEVEL'].values
        
        # 스케일링
        if not self.scaler_fitted:
            X_scaled = self.feature_scaler.fit_transform(X_data)
            y_reg_scaled = self.target_scaler.fit_transform(y_reg.reshape(-1, 1)).flatten()
            self.scaler_fitted = True
            self.save_scalers()
        else:
            X_scaled = self.feature_scaler.transform(X_data)
            y_reg_scaled = self.target_scaler.transform(y_reg.reshape(-1, 1)).flatten()
        
        # 시퀀스 생성
        X, y_regression, y_classification = [], [], []
        
        for i in range(len(X_scaled) - self.seq_len):
            X.append(X_scaled[i:i+self.seq_len])
            y_regression.append(y_reg_scaled[i+self.seq_len-1])
            y_classification.append(int(y_cls[i+self.seq_len-1]))
        
        X = np.array(X, dtype=np.float32)
        y_regression = np.array(y_regression, dtype=np.float32)
        y_classification = np.array(y_classification, dtype=np.int32)
        
        print(f"✅ 시퀀스 생성 완료: {X.shape}")
        
        return X, y_regression, y_classification, available_cols

# =========================================
# 모델 정의 (Keras 3.0 호환)
# =========================================

class PatchEmbedding(Layer):
    """패치 임베딩 레이어"""
    def __init__(self, n_patches, patch_len, n_features, d_model=128):
        super().__init__()
        self.n_patches = n_patches
        self.patch_len = patch_len
        self.n_features = n_features
        self.d_model = d_model
        self.linear = Dense(d_model)
        
    def call(self, x):
        # (batch, seq_len, features) -> (batch, n_patches, patch_len*features)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, self.n_patches, self.patch_len * self.n_features])
        return self.linear(x)

def build_patch_tst(input_shape, patch_len=10):
    """PatchTST - Keras 3.0 호환"""
    inputs = Input(shape=input_shape)
    seq_len, n_features = input_shape
    n_patches = seq_len // patch_len
    
    # 패치 임베딩
    patch_embed = PatchEmbedding(n_patches, patch_len, n_features, 128)
    x = patch_embed(inputs)
    
    # Positional Encoding
    positions = Embedding(n_patches, 128)(tf.range(n_patches))
    x = x + positions[tf.newaxis, :, :]
    
    # Transformer Blocks
    for _ in range(3):
        # Multi-Head Attention
        attn_output = MultiHeadAttention(
            num_heads=8, 
            key_dim=16,
            dropout=0.1
        )(x, x)
        x = LayerNormalization()(x + attn_output)
        
        # Feed Forward
        ffn_output = Dense(512, activation='gelu')(x)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = Dense(128)(ffn_output)
        x = LayerNormalization()(x + ffn_output)
    
    # Global Pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 출력층
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    
    model = Model(inputs, [out_reg, out_cls], name='PatchTST')
    return model

def build_stable_lstm(input_shape):
    """StableLSTM - 양방향 LSTM"""
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # 출력층
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    
    model = Model(inputs, [out_reg, out_cls], name='StableLSTM')
    return model

def build_spike_detector(input_shape):
    """SpikeDetector - 급변 감지"""
    inputs = Input(shape=input_shape)
    
    # 최근 20개 시점에 집중
    recent = Lambda(lambda x: x[:, -20:, :])(inputs)
    
    # 1D CNN
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(recent)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # GRU
    x = GRU(64, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    
    # 출력층
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    
    model = Model(inputs, [out_reg, out_cls], name='SpikeDetector')
    return model

def build_extreme_net(input_shape):
    """ExtremeNet - 극단값 전문"""
    inputs = Input(shape=input_shape)
    
    # Branch 1: Normal pattern
    normal_branch = LSTM(64, return_sequences=True)(inputs)
    normal_branch = LSTM(32, return_sequences=False)(normal_branch)
    normal_branch = Dense(32, activation='relu')(normal_branch)
    
    # Branch 2: Extreme pattern
    extreme_branch = MultiHeadAttention(
        num_heads=4, 
        key_dim=16
    )(inputs, inputs)
    extreme_branch = GlobalAveragePooling1D()(extreme_branch)
    extreme_branch = Dense(32, activation='relu')(extreme_branch)
    
    # Merge
    combined = Concatenate()([normal_branch, extreme_branch])
    x = BatchNormalization()(combined)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # 출력층
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    
    model = Model(inputs, [out_reg, out_cls], name='ExtremeNet')
    return model

def build_ensemble_base(input_shape):
    """EnsembleBase - 앙상블 기본"""
    inputs = Input(shape=input_shape)
    
    # Multiple pooling
    avg_pool = GlobalAveragePooling1D()(inputs)
    max_pool = GlobalMaxPooling1D()(inputs)
    
    # Combine
    combined = Concatenate()([avg_pool, max_pool])
    x = BatchNormalization()(combined)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 출력층
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    
    model = Model(inputs, [out_reg, out_cls], name='EnsembleBase')
    return model

# =========================================
# 모델 빌더
# =========================================

def build_models(input_shape):
    """모든 모델 빌드"""
    models = {}
    
    print("\n📦 모델 생성 중...")
    
    # 1. StableLSTM
    models['StableLSTM'] = build_stable_lstm(input_shape)
    print("  ✅ StableLSTM 생성 완료")
    
    # 2. PatchTST
    models['PatchTST'] = build_patch_tst(input_shape)
    print("  ✅ PatchTST 생성 완료")
    
    # 3. SpikeDetector
    models['SpikeDetector'] = build_spike_detector(input_shape)
    print("  ✅ SpikeDetector 생성 완료")
    
    # 4. ExtremeNet
    models['ExtremeNet'] = build_extreme_net(input_shape)
    print("  ✅ ExtremeNet 생성 완료")
    
    # 5. EnsembleBase
    models['EnsembleBase'] = build_ensemble_base(input_shape)
    print("  ✅ EnsembleBase 생성 완료")
    
    return models

# =========================================
# 학습 및 평가 클래스
# =========================================

class ModelTrainer:
    """모델 학습 및 평가"""
    
    def __init__(self, models_dict, processor, checkpoint_manager):
        self.models = models_dict
        self.processor = processor
        self.checkpoint_manager = checkpoint_manager
        self.results = {}
        
    def train_model(self, model_name, X_train, y_train, X_val, y_val, 
                   epochs=30, initial_epoch=0):
        """모델 학습"""
        
        model = self.models[model_name]
        
        # 커스텀 손실 함수 (클래스 불균형 처리)
        def weighted_sparse_categorical_crossentropy(y_true, y_pred):
            # 클래스별 가중치 적용
            weights = tf.constant([1.0, 5.0, 200.0])
            
            # sparse_categorical_crossentropy 계산
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            
            # 가중치 적용
            weight_map = tf.gather(weights, tf.cast(y_true, tf.int32))
            weighted_loss = loss * weight_map
            
            return tf.reduce_mean(weighted_loss)
        
        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'regression': 'mse',
                'classification': weighted_sparse_categorical_crossentropy
            },
            loss_weights={'regression': 0.3, 'classification': 0.7},
            metrics={
                'regression': ['mae'],
                'classification': ['accuracy']
            }
        )
        
        print(f"\n{'='*60}")
        print(f"🎯 {model_name} 학습 시작")
        print(f"{'='*60}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                f'./checkpoints/{model_name}_best.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 클래스별 샘플 가중치 생성 (class_weight 대체)
        class_weight_dict = {0: 1.0, 1: 5.0, 2: 200.0}
        sample_weights = np.array([class_weight_dict[cls] for cls in y_train[1]])
        
        # 학습 데이터 준비
        y_train_dict = {'regression': y_train[0], 'classification': y_train[1]}
        y_val_dict = {'regression': y_val[0], 'classification': y_val[1]}
        
        history = model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=256,
            callbacks=callbacks,
            # sample_weight 제거 (커스텀 손실 함수에서 처리)
            verbose=1
        )
        
        # 최종 저장
        os.makedirs('models', exist_ok=True)
        model.save(f'./models/{model_name}_final.keras')
        print(f"💾 모델 저장: ./models/{model_name}_final.keras")
        
        return history
    
    def evaluate_model(self, model_name, X_test, y_test):
        """모델 평가"""
        
        model = self.models[model_name]
        
        # 예측
        predictions = model.predict(X_test, batch_size=256, verbose=0)
        y_reg_pred = predictions[0].flatten()
        y_cls_pred = np.argmax(predictions[1], axis=1)
        
        # 실제값
        y_reg_true = y_test[0]
        y_cls_true = y_test[1]
        
        # 역변환
        y_reg_true_orig = self.processor.target_scaler.inverse_transform(
            y_reg_true.reshape(-1, 1)
        ).flatten()
        y_reg_pred_orig = self.processor.target_scaler.inverse_transform(
            y_reg_pred.reshape(-1, 1)
        ).flatten()
        
        # 회귀 메트릭
        mae = mean_absolute_error(y_reg_true_orig, y_reg_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_reg_true_orig, y_reg_pred_orig))
        r2 = r2_score(y_reg_true_orig, y_reg_pred_orig)
        
        # 분류 메트릭
        accuracy = accuracy_score(y_cls_true, y_cls_pred)
        
        # 구간별 정확도
        cm = confusion_matrix(y_cls_true, y_cls_pred)
        class_accuracies = []
        for i in range(len(cm)):
            if cm[i].sum() > 0:
                class_accuracies.append(cm[i, i] / cm[i].sum())
            else:
                class_accuracies.append(0)
        
        # 1700+ 성능
        if 2 in y_cls_true:
            precision_1700 = precision_score(y_cls_true, y_cls_pred, labels=[2], average='macro', zero_division=0)
            recall_1700 = recall_score(y_cls_true, y_cls_pred, labels=[2], average='macro', zero_division=0)
            f1_1700 = f1_score(y_cls_true, y_cls_pred, labels=[2], average='macro', zero_division=0)
        else:
            precision_1700 = recall_1700 = f1_1700 = 0
        
        # 결과 저장
        results = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Accuracy': accuracy,
            'Accuracy_Normal': class_accuracies[0] if len(class_accuracies) > 0 else 0,
            'Accuracy_Warning': class_accuracies[1] if len(class_accuracies) > 1 else 0,
            'Accuracy_Critical': class_accuracies[2] if len(class_accuracies) > 2 else 0,
            'Precision_1700': precision_1700,
            'Recall_1700': recall_1700,
            'F1_1700': f1_1700
        }
        
        self.results[model_name] = results
        
        # 출력
        print(f"\n{'='*60}")
        print(f"📊 {model_name} 평가 결과")
        print(f"{'='*60}")
        print(f"\n[회귀 성능]")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²:   {r2:.4f}")
        
        print(f"\n[분류 성능]")
        print(f"  전체 정확도: {accuracy:.2%}")
        print(f"  정상 정확도: {results['Accuracy_Normal']:.2%}")
        print(f"  주의 정확도: {results['Accuracy_Warning']:.2%}")
        print(f"  심각 정확도: {results['Accuracy_Critical']:.2%}")
        
        print(f"\n[1700+ 감지]")
        print(f"  Precision: {precision_1700:.2%}")
        print(f"  Recall:    {recall_1700:.2%}")
        print(f"  F1-Score:  {f1_1700:.2%}")
        
        return results

# =========================================
# 메인 실행
# =========================================

def main():
    """메인 실행"""
    
    # 디렉토리 생성
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 체크포인트 관리자
    checkpoint_manager = CheckpointManager()
    
    # 1. 데이터 준비
    print("\n📁 데이터 로드...")
    
    # 데이터 경로
    data_paths = [
        'data/20250731_to20250806.csv',  # 실제 경로
        '/mnt/user-data/uploads/gs.CSV',
        './data.csv'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            print(f"✅ 데이터 발견: {path}")
            break
    
    if data_path is None:
        print("❌ 데이터 파일을 찾을 수 없습니다!")
        return
    
    # 데이터 처리
    processor = DataProcessor(seq_len=100, pred_len=10)
    processor.load_scalers()
    
    df = processor.load_data(data_path)
    df = processor.create_features(df)
    X, y_reg, y_cls, feature_cols = processor.prepare_sequences(df)
    
    # 2. 데이터 분할
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train = X[:n_train]
    y_reg_train = y_reg[:n_train]
    y_cls_train = y_cls[:n_train]
    
    X_val = X[n_train:n_train+n_val]
    y_reg_val = y_reg[n_train:n_train+n_val]
    y_cls_val = y_cls[n_train:n_train+n_val]
    
    X_test = X[n_train+n_val:]
    y_reg_test = y_reg[n_train+n_val:]
    y_cls_test = y_cls[n_train+n_val:]
    
    print(f"\n📊 데이터 분할:")
    print(f"  Train: {X_train.shape} (70%)")
    print(f"  Val:   {X_val.shape} (15%)")
    print(f"  Test:  {X_test.shape} (15%)")
    
    # 3. 모델 생성
    input_shape = (X.shape[1], X.shape[2])
    models = build_models(input_shape)
    
    # 4. 학습 관리자
    trainer = ModelTrainer(models, processor, checkpoint_manager)
    
    # 5. 각 모델 학습 및 평가
    for model_name in models.keys():
        # 학습
        trainer.train_model(
            model_name,
            X_train,
            (y_reg_train, y_cls_train),
            X_val,
            (y_reg_val, y_cls_val),
            epochs=30
        )
        
        # 평가
        trainer.evaluate_model(
            model_name,
            X_test,
            (y_reg_test, y_cls_test)
        )
    
    # 6. 최종 결과
    print("\n" + "="*80)
    print("📊 최종 성능 비교")
    print("="*80)
    
    results_df = pd.DataFrame(trainer.results).T
    print("\n[회귀 성능]")
    print(results_df[['MAE', 'RMSE', 'R2']].round(3))
    
    print("\n[분류 정확도]")
    print(results_df[['Accuracy', 'Accuracy_Normal', 'Accuracy_Warning', 'Accuracy_Critical']].round(3))
    
    print("\n[1700+ 감지]")
    print(results_df[['Precision_1700', 'Recall_1700', 'F1_1700']].round(3))
    
    # CSV 저장
    results_df.to_csv('model_results.csv')
    print("\n💾 결과 저장: model_results.csv")
    
    print("\n✅ 학습 완료!")
    
    return models, results_df

if __name__ == "__main__":
    models, results = main()