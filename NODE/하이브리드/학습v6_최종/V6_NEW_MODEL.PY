"""
🚀 Ultimate 1700+ Predictor v8.0 - Production Ready
==================================================
- Infinity/NaN 오류 완전 해결
- 전체 구간별 평가 (일반, 중간, 1700+)
- 안정적인 데이터 처리
- [수정] EnsembleBase 모델 구조 오류 해결
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
from sklearn.model_selection import train_test_split
import warnings
import logging
from datetime import datetime
import os
import json

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TensorFlow 설정
tf.random.set_seed(42)
np.random.seed(42)

print("\n" + "="*80)
print("🚀 Ultimate 1700+ Predictor v8.0")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*80)

# ========================================
# 안전한 데이터 처리
# ========================================

class SafeDataProcessor:
    """Infinity/NaN 안전 처리"""
    
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = MinMaxScaler()  # RobustScaler 대신 MinMaxScaler 사용
        self.target_scaler = MinMaxScaler()
        
    def load_and_process(self, filepath):
        """데이터 로드 - 안전 처리"""
        logger.info("📂 데이터 로딩...")
        
        # CSV 로드
        df = pd.read_csv(filepath)
        logger.info(f"✅ 원본 데이터: {df.shape}")
        
        # 시간 변환
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')
        df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m%d%H%M', errors='coerce')
        
        # NaT 제거
        df = df.dropna(subset=['CURRTIME', 'TIME'])
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 수치형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Infinity와 NaN 체크 및 제거
        logger.info("🔍 이상치 체크...")
        for col in numeric_cols:
            # Infinity를 NaN으로 변경
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # NaN을 중앙값으로 채우기
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
                logger.info(f"   {col}: {df[col].isna().sum()}개 NaN → {median_val:.2f}로 대체")
        
        # 극단적 이상치 제거 (IQR 방법)
        for col in ['TOTALCNT']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            
            before = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            after = len(df)
            
            if before != after:
                logger.info(f"   {col}: {before-after}개 극단 이상치 제거")
        
        # 타겟 생성
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        df['IS_1700'] = (df['TARGET'] >= 1700).astype(int)
        
        # 특성 생성 (안전하게)
        df = self._create_safe_features(df)
        
        # 최종 NaN 제거
        df = df.dropna()
        
        logger.info(f"✅ 전처리 완료: {df.shape}")
        logger.info(f"✅ 1700+ 비율: {df['IS_1700'].mean():.2%} ({df['IS_1700'].sum()}개)")
        
        return df
    
    def _create_safe_features(self, df):
        """안전한 특성 생성"""
        logger.info("⚙️ 특성 생성...")
        
        # 이동평균 (안전한 범위)
        for window in [10, 30]:
            df[f'MA_{window}'] = df['TOTALCNT'].rolling(window, min_periods=1).mean()
            df[f'MA_{window}'].fillna(df['TOTALCNT'].mean(), inplace=True)
        
        # 변화율 (0으로 나누기 방지)
        df['CHANGE'] = df['TOTALCNT'].diff().fillna(0)
        df['PCT_CHANGE'] = df['TOTALCNT'].pct_change().fillna(0)
        
        # 극단값 방지
        df['PCT_CHANGE'] = df['PCT_CHANGE'].clip(-1, 1)
        
        # 비율 계산 (0으로 나누기 방지)
        if 'M14AM14B' in df.columns and 'M14AM10A' in df.columns:
            df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)  # +1로 0 방지
            df['RATIO'] = df['RATIO'].clip(0, 20)  # 극단값 제한
        
        # 시간 특성
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['DAYOFWEEK'] = df['CURRTIME'].dt.dayofweek
        df['IS_DAWN'] = df['HOUR'].isin([4, 5, 6]).astype(int)
        
        # 임계값 신호
        if 'M14AM14B' in df.columns:
            df['M14B_HIGH'] = (df['M14AM14B'] > df['M14AM14B'].quantile(0.9)).astype(int)
        
        # 모든 값 체크
        for col in df.select_dtypes(include=[np.number]).columns:
            # Infinity 체크
            df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
            # NaN 체크
            df[col] = df[col].fillna(df[col].median())
            
        return df
    
    def create_sequences(self, df):
        """안전한 시퀀스 생성"""
        logger.info("🔄 시퀀스 생성...")
        
        # 특성 선택 (문제 될 수 있는 컬럼 제외)
        exclude_cols = ['CURRTIME', 'TIME', 'TARGET', 'IS_1700']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 수치형만 선택
        feature_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        logger.info(f"   사용할 특성: {len(feature_cols)}개")
        
        # 데이터 복사
        data = df[feature_cols].copy()
        targets = df['TARGET'].copy()
        is_1700 = df['IS_1700'].copy()
        
        # 스케일링 전 체크
        logger.info("   스케일링 전 체크...")
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.median())
        
        # 스케일링
        try:
            data_scaled = self.scaler.fit_transform(data)
            targets_scaled = self.target_scaler.fit_transform(targets.values.reshape(-1, 1)).flatten()
        except Exception as e:
            logger.error(f"❌ 스케일링 오류: {e}")
            # 대체 방법: 수동 정규화
            data_scaled = (data - data.mean()) / (data.std() + 1e-7)
            targets_scaled = (targets - targets.mean()) / (targets.std() + 1e-7)
            data_scaled = data_scaled.fillna(0).values
            targets_scaled = targets_scaled.fillna(0).values
        
        # 시퀀스 생성
        X, y, y_binary = [], [], []
        
        for i in range(len(data_scaled) - self.seq_len - self.pred_len):
            seq = data_scaled[i:i+self.seq_len]
            
            # 시퀀스 체크
            if not np.any(np.isnan(seq)) and not np.any(np.isinf(seq)):
                X.append(seq)
                y.append(targets_scaled[i+self.seq_len])
                y_binary.append(is_1700.iloc[i+self.seq_len])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        y_binary = np.array(y_binary, dtype=np.float32)
        
        # 최종 체크
        logger.info("   최종 체크...")
        logger.info(f"   X: NaN={np.isnan(X).sum()}, Inf={np.isinf(X).sum()}")
        logger.info(f"   y: NaN={np.isnan(y).sum()}, Inf={np.isinf(y).sum()}")
        
        logger.info(f"✅ 시퀀스 생성 완료: {X.shape}")
        
        return X, y, y_binary, feature_cols

# ========================================
# 안정적인 모델
# ========================================

class StableModels:
    """안정적인 모델 구현"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.seq_len = input_shape[0]
        self.n_features = input_shape[1]
    
    def build_stable_lstm(self):
        """안정적인 LSTM"""
        inputs = Input(shape=self.input_shape)
        
        # Batch Normalization으로 안정화
        x = BatchNormalization()(inputs)
        
        # LSTM layers with careful dropout
        x = LSTM(128, return_sequences=True, 
                kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        
        x = LSTM(64, return_sequences=False,
                kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        
        # Dense layers
        x = Dense(32, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Dropout(0.1)(x)
        
        # Output with sigmoid to limit range
        outputs = Dense(1, activation='sigmoid')(x)  # 0-1 범위로 제한
        
        model = Model(inputs, outputs, name='StableLSTM')
        return model
    
    def build_spike_detector(self):
        """급변 감지 모델"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # CNN layers
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        
        x = Conv1D(32, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # LSTM
        x = LSTM(64, return_sequences=False)(x)
        x = BatchNormalization()(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs, outputs, name='SpikeDetector')
        return model
    
    def build_extreme_net(self):
        """극단값 전문 모델"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # GRU backbone
        x = GRU(128, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = GRU(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        
        # Two branches
        normal_branch = Dense(32, activation='relu')(x)
        normal_out = Dense(1, activation='linear')(normal_branch)
        
        extreme_branch = Dense(32, activation='relu')(x)
        extreme_factor = Dense(1, activation='sigmoid')(extreme_branch)
        
        # Combine
        outputs = Add()([normal_out, 
                        Multiply()([normal_out, extreme_factor])])
        outputs = Activation('sigmoid')(outputs)  # 최종 범위 제한
        
        model = Model(inputs, outputs, name='ExtremeNet')
        return model
    
    def build_ensemble_base(self):
        """앙상블용 베이스 모델"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)

        # [수정된 부분] 3D 시퀀스 입력을 2D로 변환하여 Dense Layer에 전달합니다.
        x = GlobalAveragePooling1D()(x)
        
        # Simple but effective
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs, outputs, name='EnsembleBase')
        return model

# ========================================
# 평가 시스템
# ========================================

class ComprehensiveEvaluator:
    """전체 구간 평가"""
    
    def __init__(self, target_scaler):
        self.target_scaler = target_scaler
        self.results = {}
    
    def evaluate(self, model, X_test, y_test, y_binary, model_name):
        """종합 평가"""
        
        # 예측
        y_pred = model.predict(X_test, batch_size=256, verbose=0).flatten()
        
        # Clip predictions to valid range
        y_pred = np.clip(y_pred, 0, 1)
        
        # 역변환
        y_test_real = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_real = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # ===== 전체 성능 =====
        mae_total = np.mean(np.abs(y_test_real - y_pred_real))
        rmse_total = np.sqrt(np.mean((y_test_real - y_pred_real) ** 2))
        
        # ===== 구간별 분석 =====
        metrics = {
            'total': {'mae': mae_total, 'rmse': rmse_total}
        }
        
        # 구간 정의
        ranges = [
            ('low', 0, 1200),
            ('normal', 1200, 1400),
            ('medium', 1400, 1600),
            ('high', 1600, 1700),
            ('extreme', 1700, 9999)
        ]
        
        for name, low, high in ranges:
            mask = (y_test_real >= low) & (y_test_real < high)
            if mask.sum() > 0:
                mae = np.mean(np.abs(y_test_real[mask] - y_pred_real[mask]))
                rmse = np.sqrt(np.mean((y_test_real[mask] - y_pred_real[mask]) ** 2))
                count = mask.sum()
                metrics[name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'count': count,
                    'percentage': (count / len(y_test_real)) * 100
                }
        
        # ===== 1700+ 특별 분석 =====
        actual_1700 = y_test_real >= 1700
        pred_1700 = y_pred_real >= 1700
        
        tp = np.sum(pred_1700 & actual_1700)
        fp = np.sum(pred_1700 & ~actual_1700)
        fn = np.sum(~pred_1700 & actual_1700)
        tn = np.sum(~pred_1700 & ~actual_1700)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # ===== 정상 구간 오탐 =====
        normal_mask = (y_test_real >= 1200) & (y_test_real < 1400)
        if normal_mask.sum() > 0:
            false_alarm = np.sum(y_pred_real[normal_mask] >= 1700)
            false_alarm_rate = false_alarm / normal_mask.sum()
        else:
            false_alarm_rate = 0
        
        # 결과 저장
        result = {
            'metrics': metrics,
            'precision_1700': precision,
            'recall_1700': recall,
            'f1_1700': f1,
            'false_alarm_rate': false_alarm_rate,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
        }
        
        self.results[model_name] = result
        
        # 출력
        self._print_results(model_name, result)
        
        return result
    
    def _print_results(self, model_name, result):
        """결과 출력"""
        print(f"\n{'='*60}")
        print(f"📊 {model_name} 평가 결과")
        print(f"{'='*60}")
        
        print("\n[전체 성능]")
        print(f"  MAE: {result['metrics']['total']['mae']:.2f}")
        print(f"  RMSE: {result['metrics']['total']['rmse']:.2f}")
        
        print("\n[구간별 성능]")
        for range_name in ['low', 'normal', 'medium', 'high', 'extreme']:
            if range_name in result['metrics']:
                m = result['metrics'][range_name]
                print(f"  {range_name:8s}: MAE={m['mae']:6.2f}, "
                      f"Count={m['count']:4d} ({m['percentage']:5.2f}%)")
        
        print("\n[1700+ 예측 성능]")
        print(f"  Precision: {result['precision_1700']:.2%}")
        print(f"  Recall: {result['recall_1700']:.2%}")
        print(f"  F1 Score: {result['f1_1700']:.2%}")
        
        print(f"\n[정상→1700+ 오탐률]")
        print(f"  False Alarm Rate: {result['false_alarm_rate']:.2%}")
        
        # 종합 점수
        balanced_score = (
            result['f1_1700'] * 0.4 +  # 1700+ 예측
            (1 - result['false_alarm_rate']) * 0.3 +  # 오탐 방지
            (1 - result['metrics']['normal']['mae']/100) * 0.3  # 정상 구간 정확도
        )
        print(f"\n[종합 점수]")
        print(f"  Balanced Score: {balanced_score:.2%}")

# ========================================
# 메인 실행
# ========================================

def main():
    """메인 함수"""
    
    print("\n🚀 시작...")
    
    # 1. 데이터 처리
    processor = SafeDataProcessor(seq_len=100, pred_len=10)
    
    # 데이터 경로 확인 (실제 환경에 맞게 경로를 수정하세요)
    data_paths = [
        'data/20240201_TO_202507281705.csv',
        '/mnt/user-data/uploads/gs.CSV',
        '/mnt/user-data/uploads/20250807_DATA.CSV',
        'V6_NEW_학습.py와 동일한 위치에 있는 데이터 파일.csv' # 예시 경로
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        logger.error("❌ 데이터 파일을 찾을 수 없습니다! 'data_paths' 변수 내의 경로를 확인해주세요.")
        return
    
    df = processor.load_and_process(data_path)
    X, y, y_binary, feature_cols = processor.create_sequences(df)
    
    # 2. 데이터 분할
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    y_test_binary = y_binary[train_size+val_size:]
    
    print(f"\n📊 데이터 분할:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # 3. 모델 생성
    model_builder = StableModels(input_shape=(100, X.shape[2]))
    
    models = {
        'StableLSTM': model_builder.build_stable_lstm(),
        'SpikeDetector': model_builder.build_spike_detector(),
        'ExtremeNet': model_builder.build_extreme_net(),
        'EnsembleBase': model_builder.build_ensemble_base()
    }
    
    # 4. 평가자 생성
    evaluator = ComprehensiveEvaluator(processor.target_scaler)
    
    # 5. 각 모델 학습 및 평가
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🎯 {name} 학습")
        print(f"{'='*60}")
        
        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',  # 안정적인 MSE 사용
            metrics=['mae']
        )
        
        # 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=256,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
            ],
            verbose=1
        )
        
        # 평가
        evaluator.evaluate(model, X_test, y_test, y_test_binary, name)
    
    # 6. 최종 순위
    print("\n" + "="*80)
    print("🏆 최종 순위")
    print("="*80)
    
    # F1 기준 정렬
    sorted_models = sorted(
        evaluator.results.items(),
        key=lambda x: x[1]['f1_1700'],
        reverse=True
    )
    
    for i, (name, result) in enumerate(sorted_models, 1):
        print(f"{i}. {name:15s} | F1: {result['f1_1700']:.2%} | "
              f"정상MAE: {result['metrics']['normal']['mae']:.2f} | "
              f"오탐률: {result['false_alarm_rate']:.2%}")
    
    print("\n✅ 완료!")
    
    return evaluator.results

if __name__ == "__main__":
    results = main()
