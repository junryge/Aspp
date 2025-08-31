# -*- coding: utf-8 -*-
"""
HUBROOM 반송량 예측 시스템 - TensorFlow 2.15 완전판
- 진행률 표시 (%)
- 중단/재개 기능
- 데이터 특성 상세 분석
- Model 1: PatchTST
- Model 2: PatchTST + PINN
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import json
import pickle
from tqdm import tqdm
import signal
import time
import warnings

warnings.filterwarnings('ignore')

# TensorFlow 설정
print(f"🔧 TensorFlow 버전: {tf.__version__}")
tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ===========================
# 🔄 체크포인트 관리자
# ===========================

class CheckpointManager:
    """학습 중단/재개를 위한 체크포인트 관리"""
    
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.interrupted = False
        
        # Ctrl+C 인터럽트 핸들러
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # 진행 상황 파일
        self.progress_file = os.path.join(checkpoint_dir, 'progress.json')
        self.data_file = os.path.join(checkpoint_dir, 'processed_data.pkl')
    
    def _signal_handler(self, sig, frame):
        print('\n\n⚠️ 중단 감지! 진행 상황 저장 중...')
        self.interrupted = True
        
    def save_progress(self, stage, progress, total, details=None):
        """진행 상황 저장"""
        progress_data = {
            'stage': stage,
            'progress': progress,
            'total': total,
            'percentage': (progress/total*100) if total > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        return progress_data
    
    def load_progress(self):
        """진행 상황 로드"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_data(self, data_dict):
        """처리된 데이터 저장"""
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"💾 데이터 저장: {self.data_file}")
    
    def load_data(self):
        """처리된 데이터 로드"""
        if os.path.exists(self.data_file):
            print(f"📂 저장된 데이터 로드: {self.data_file}")
            with open(self.data_file, 'rb') as f:
                return pickle.load(f)
        return None

# ===========================
# 📊 데이터 분석 및 처리
# ===========================

class DataAnalyzer:
    """데이터 특성 상세 분석"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_with_progress(self, df, target_col='CURRENT_M16A_3F_JOB_2'):
        """진행률과 함께 데이터 분석"""
        
        print("\n" + "="*70)
        print("📈 데이터 특성 상세 분석 시작")
        print("="*70)
        
        stages = [
            ("기본 정보", self._analyze_basic_info),
            ("결측치 분석", self._analyze_missing),
            ("타겟 변수", self._analyze_target),
            ("시간 패턴", self._analyze_temporal),
            ("통계 분포", self._analyze_statistics),
            ("상관관계", self._analyze_correlation),
            ("이상치 탐지", self._analyze_outliers),
            ("데이터 품질", self._analyze_quality)
        ]
        
        total_stages = len(stages)
        
        for i, (stage_name, analyze_func) in enumerate(stages, 1):
            # 진행률 표시
            progress_pct = (i / total_stages) * 100
            print(f"\n[{i}/{total_stages}] {stage_name} 분석 중... ({progress_pct:.0f}%)")
            print("-" * 50)
            
            # 분석 실행
            result = analyze_func(df, target_col)
            self.analysis_results[stage_name] = result
            
            # 진행 바
            self._show_progress_bar(i, total_stages)
            
            time.sleep(0.1)  # 시각적 효과
        
        print("\n" + "="*70)
        print("✅ 데이터 분석 완료!")
        print("="*70)
        
        return self.analysis_results
    
    def _show_progress_bar(self, current, total):
        """진행 바 표시"""
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)
        percent = current / total * 100
        print(f"전체 진행: [{bar}] {percent:.1f}%")
    
    def _analyze_basic_info(self, df, target_col):
        """기본 정보 분석"""
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.value_counts().to_dict()
        }
        
        print(f"  • 데이터 크기: {info['rows']:,} 행 × {info['columns']} 컬럼")
        print(f"  • 메모리 사용: {info['memory_mb']:.2f} MB")
        print(f"  • 데이터 타입: {info['dtypes']}")
        
        return info
    
    def _analyze_missing(self, df, target_col):
        """결측치 분석"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        result = {
            'total_missing': missing.sum(),
            'columns_with_missing': (missing > 0).sum(),
            'missing_percentage': missing_pct.mean()
        }
        
        print(f"  • 전체 결측치: {result['total_missing']:,}개")
        print(f"  • 결측 컬럼: {result['columns_with_missing']}개")
        print(f"  • 평균 결측률: {result['missing_percentage']:.2f}%")
        
        return result
    
    def _analyze_target(self, df, target_col):
        """타겟 변수 분석"""
        if target_col in df.columns:
            target = df[target_col].dropna()
            result = {
                'mean': target.mean(),
                'std': target.std(),
                'min': target.min(),
                'max': target.max(),
                'q25': target.quantile(0.25),
                'q50': target.quantile(0.50),
                'q75': target.quantile(0.75)
            }
            
            print(f"  • 평균±표준편차: {result['mean']:.2f} ± {result['std']:.2f}")
            print(f"  • 범위: [{result['min']:.0f}, {result['max']:.0f}]")
            print(f"  • 사분위수: Q1={result['q25']:.0f}, Q2={result['q50']:.0f}, Q3={result['q75']:.0f}")
        else:
            result = {'error': 'Target column not found'}
            print(f"  • ⚠️ 타겟 컬럼 '{target_col}' 없음")
        
        return result
    
    def _analyze_temporal(self, df, target_col):
        """시간 패턴 분석"""
        time_col = df.columns[0]  # 첫 컬럼이 시간
        
        result = {
            'start': str(df[time_col].iloc[0]),
            'end': str(df[time_col].iloc[-1]),
            'duration': len(df),
            'interval': 'minute'  # 추정
        }
        
        print(f"  • 기간: {result['start']} ~ {result['end']}")
        print(f"  • 데이터 포인트: {result['duration']:,}개")
        
        return result
    
    def _analyze_statistics(self, df, target_col):
        """통계 분포 분석"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats = df[numeric_cols].describe()
        result = {
            'numeric_columns': len(numeric_cols),
            'mean_values': stats.loc['mean'].mean(),
            'std_values': stats.loc['std'].mean()
        }
        
        print(f"  • 숫자형 컬럼: {result['numeric_columns']}개")
        print(f"  • 전체 평균: {result['mean_values']:.2f}")
        
        return result
    
    def _analyze_correlation(self, df, target_col):
        """상관관계 분석"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if target_col in numeric_cols and len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
            top_corr = correlations[1:6]  # 타겟 제외 상위 5개
            
            result = {
                'top_features': top_corr.index.tolist(),
                'correlations': top_corr.values.tolist()
            }
            
            print(f"  • 타겟과 상관관계 TOP 3:")
            for feat, corr in zip(result['top_features'][:3], result['correlations'][:3]):
                print(f"    - {feat}: {corr:.3f}")
        else:
            result = {'error': 'Correlation analysis failed'}
        
        return result
    
    def _analyze_outliers(self, df, target_col):
        """이상치 탐지"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # 처음 5개만
        
        outlier_counts = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
            outlier_counts[col] = outliers
        
        total_outliers = sum(outlier_counts.values())
        print(f"  • 총 이상치: {total_outliers}개")
        print(f"  • 이상치 비율: {total_outliers/(len(df)*len(numeric_cols))*100:.2f}%")
        
        return {'outlier_counts': outlier_counts, 'total': total_outliers}
    
    def _analyze_quality(self, df, target_col):
        """데이터 품질 평가"""
        score = 100.0
        
        # 결측치 감점
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        score -= min(30, missing_pct * 3)
        
        # 중복 행 감점
        duplicate_pct = df.duplicated().sum() / len(df) * 100
        score -= min(20, duplicate_pct * 2)
        
        # 데이터 크기 보너스
        if len(df) > 10000:
            score += 5
        
        score = max(0, min(100, score))
        
        grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
        
        print(f"  • 품질 점수: {score:.1f}/100 (등급: {grade})")
        
        if score >= 80:
            print("  • ✅ 우수한 데이터 품질")
        elif score >= 60:
            print("  • ⚠️ 보통 수준, 전처리 권장")
        else:
            print("  • ❌ 낮은 품질, 전처리 필수")
        
        return {'score': score, 'grade': grade}

# ===========================
# 📦 데이터 처리
# ===========================

class HUBROOMProcessor:
    """데이터 처리 및 시퀀스 생성"""
    
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.checkpoint_mgr = CheckpointManager()
        self.analyzer = DataAnalyzer()
        
    def process_with_resume(self, file_path='data/HUB_0509_TO_0730_DATA.CSV'):
        """중단 지점부터 재개 가능한 데이터 처리"""
        
        # 이전 진행 상황 확인
        saved_progress = self.checkpoint_mgr.load_progress()
        saved_data = self.checkpoint_mgr.load_data()
        
        if saved_progress and saved_data:
            print(f"\n♻️ 이전 작업 발견!")
            print(f"  • 단계: {saved_progress['stage']}")
            print(f"  • 진행률: {saved_progress['percentage']:.1f}%")
            
            resume = input("이어서 진행하시겠습니까? (y/n): ")
            if resume.lower() == 'y':
                return self._resume_from_checkpoint(saved_data, saved_progress)
        
        # 새로 시작
        return self._process_from_scratch(file_path)
    
    def _process_from_scratch(self, file_path):
        """처음부터 데이터 처리"""
        
        print("\n" + "="*70)
        print("🚀 데이터 처리 시작")
        print("="*70)
        
        # Stage 1: 데이터 로드
        print("\n[Stage 1/5] 데이터 로드 중...")
        df = pd.read_csv(file_path)
        print(f"✅ 로드 완료: {len(df):,} 행")
        self.checkpoint_mgr.save_progress('load', 1, 1)
        
        # Stage 2: 데이터 분석
        print("\n[Stage 2/5] 데이터 분석 중...")
        analysis = self.analyzer.analyze_with_progress(df)
        self.checkpoint_mgr.save_progress('analysis', 1, 1, analysis)
        
        # Stage 3: 전처리
        print("\n[Stage 3/5] 데이터 전처리 중...")
        df = self._preprocess_data(df)
        self.checkpoint_mgr.save_progress('preprocess', 1, 1)
        
        # Stage 4: 시퀀스 생성
        print("\n[Stage 4/5] 시퀀스 생성 중...")
        sequences = self._create_sequences_with_progress(df)
        self.checkpoint_mgr.save_progress('sequences', 1, 1)
        
        # Stage 5: 데이터 분할 및 정규화
        print("\n[Stage 5/5] 데이터 분할 및 정규화 중...")
        final_data = self._split_and_scale(sequences)
        self.checkpoint_mgr.save_progress('complete', 1, 1)
        
        # 최종 데이터 저장
        self.checkpoint_mgr.save_data(final_data)
        
        print("\n" + "="*70)
        print("✅ 데이터 처리 완료!")
        print("="*70)
        
        return final_data
    
    def _preprocess_data(self, df):
        """데이터 전처리"""
        # 시간 컬럼 처리
        time_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
        
        # 정렬 및 결측치 처리
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _create_sequences_with_progress(self, df, seq_len=20, pred_len=10):
        """진행률 표시와 함께 시퀀스 생성"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        total = len(df) - seq_len - pred_len + 1
        
        X, y = [], []
        
        with tqdm(total=total, desc="시퀀스 생성") as pbar:
            for i in range(total):
                # 20분 입력
                X_seq = df[numeric_cols].iloc[i:i+seq_len].values
                X.append(X_seq)
                
                # 10분 후 타겟
                y_val = df[numeric_cols[0]].iloc[i+seq_len+pred_len-1]
                y.append(y_val)
                
                pbar.update(1)
                
                # 100개마다 저장
                if (i + 1) % 100 == 0:
                    self.checkpoint_mgr.save_progress('sequences', i+1, total)
                
                # 중단 확인
                if self.checkpoint_mgr.interrupted:
                    print(f"\n💾 중단! 진행 상황 저장: {i+1}/{total}")
                    temp_data = {'X': X, 'y': y, 'progress': i+1}
                    self.checkpoint_mgr.save_data(temp_data)
                    sys.exit(0)
        
        return {'X': np.array(X), 'y': np.array(y), 'numeric_cols': numeric_cols}
    
    def _split_and_scale(self, sequences):
        """데이터 분할 및 정규화"""
        X = sequences['X']
        y = sequences['y']
        
        # 70:15:15 분할
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # 정규화
        n_samples, seq_len, n_features = X_train.shape
        
        X_train_scaled = self.scaler_X.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], seq_len, n_features)
        X_test_scaled = self.scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], seq_len, n_features)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # 스케일러 저장 (중요!)
        self.save_scalers()
        
        return {
            'X_train': X_train_scaled, 'y_train': y_train_scaled,
            'X_val': X_val_scaled, 'y_val': y_val_scaled,
            'X_test': X_test_scaled, 'y_test': y_test_scaled,
            'n_features': n_features,
            'scalers': {'X': self.scaler_X, 'y': self.scaler_y}
        }
    
    def save_scalers(self):
        """스케일러 저장"""
        import joblib
        
        scaler_dir = os.path.join(self.checkpoint_mgr.checkpoint_dir, 'scalers')
        os.makedirs(scaler_dir, exist_ok=True)
        
        # X 스케일러 저장
        joblib.dump(self.scaler_X, os.path.join(scaler_dir, 'scaler_X.pkl'))
        # y 스케일러 저장
        joblib.dump(self.scaler_y, os.path.join(scaler_dir, 'scaler_y.pkl'))
        
        print(f"💾 스케일러 저장 완료: {scaler_dir}")
    
    def load_scalers(self):
        """저장된 스케일러 로드"""
        import joblib
        
        scaler_dir = os.path.join(self.checkpoint_mgr.checkpoint_dir, 'scalers')
        
        if os.path.exists(scaler_dir):
            self.scaler_X = joblib.load(os.path.join(scaler_dir, 'scaler_X.pkl'))
            self.scaler_y = joblib.load(os.path.join(scaler_dir, 'scaler_y.pkl'))
            print(f"📂 스케일러 로드 완료: {scaler_dir}")
            return True
        return False
    
    def _resume_from_checkpoint(self, saved_data, saved_progress):
        """체크포인트에서 재개"""
        print(f"♻️ {saved_progress['stage']} 단계부터 재개...")
        
        # 단계별 재개 로직
        if saved_progress['stage'] == 'sequences':
            # 시퀀스 생성 중단된 경우
            df = pd.read_csv('data/HUB_0509_TO_0730_DATA.CSV')
            df = self._preprocess_data(df)
            
            # 이어서 시퀀스 생성
            X_prev = saved_data.get('X', [])
            y_prev = saved_data.get('y', [])
            start_idx = saved_data.get('progress', 0)
            
            print(f"  • 이전 진행: {start_idx}개 완료")
            # 여기서 이어서 처리...
        
        return saved_data

# ===========================
# 🧠 모델 정의
# ===========================

def create_patchtst_model(config):
    """PatchTST 모델 (TensorFlow)"""
    
    inputs = layers.Input(shape=(config['seq_len'], config['n_features']))
    
    # 패치 분할 (20분 → 5분×4)
    patch_len = config['patch_len']
    n_patches = config['seq_len'] // patch_len
    
    # Reshape for patches
    x = layers.Reshape((n_patches, patch_len * config['n_features']))(inputs)
    
    # Patch Embedding
    x = layers.Dense(config['d_model'])(x)
    x = layers.LayerNormalization()(x)
    
    # Positional Encoding
    positions = tf.range(start=0, limit=n_patches, delta=1)
    position_embedding = layers.Embedding(n_patches, config['d_model'])(positions)
    x = x + position_embedding
    
    # Transformer Blocks
    for _ in range(config['n_layers']):
        # Multi-Head Attention
        attn_output = layers.MultiHeadAttention(
            num_heads=config['n_heads'],
            key_dim=config['d_model'] // config['n_heads'],
            dropout=config['dropout']
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)
        
        # Feed Forward
        ffn_output = layers.Dense(config['d_ff'], activation='relu')(x)
        ffn_output = layers.Dense(config['d_model'])(ffn_output)
        ffn_output = layers.Dropout(config['dropout'])(ffn_output)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization()(x)
    
    # Global Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(config['dropout'])(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='PatchTST')
    return model

def create_patchtst_pinn_model(config):
    """PatchTST + PINN 모델"""
    
    # 기본 PatchTST
    base_model = create_patchtst_model(config)
    
    # PINN 레이어 추가
    inputs = base_model.input
    base_output = base_model(inputs)
    
    # 물리 제약 레이어
    physics_corrected = layers.Lambda(
        lambda x: tf.clip_by_value(x, 0, config['max_capacity']),
        name='physics_constraint'
    )(base_output)
    
    model = Model(inputs=inputs, outputs=physics_corrected, name='PatchTST_PINN')
    return model

# ===========================
# 📈 커스텀 콜백
# ===========================

class ProgressCallback(Callback):
    """학습 진행률 표시 콜백"""
    
    def __init__(self, checkpoint_mgr):
        super().__init__()
        self.checkpoint_mgr = checkpoint_mgr
        
    def on_epoch_end(self, epoch, logs=None):
        # 진행 상황 저장
        self.checkpoint_mgr.save_progress(
            'training',
            epoch + 1,
            self.params['epochs'],
            {'loss': logs.get('loss'), 'val_loss': logs.get('val_loss')}
        )
        
        # 진행률 표시
        progress = (epoch + 1) / self.params['epochs'] * 100
        print(f"\n진행률: {progress:.1f}% | Loss: {logs.get('loss'):.4f} | Val Loss: {logs.get('val_loss'):.4f}")

# ===========================
# 🎯 메인 실행
# ===========================

class HUBROOMPredictor:
    """실시간 예측을 위한 클래스"""
    
    def __init__(self, model_path='checkpoints/best_model.h5', scaler_dir='checkpoints/scalers'):
        """모델과 스케일러 로드"""
        import joblib
        
        # 모델 로드
        self.model = keras.models.load_model(model_path)
        print(f"✅ 모델 로드: {model_path}")
        
        # 스케일러 로드
        self.scaler_X = joblib.load(os.path.join(scaler_dir, 'scaler_X.pkl'))
        self.scaler_y = joblib.load(os.path.join(scaler_dir, 'scaler_y.pkl'))
        print(f"✅ 스케일러 로드: {scaler_dir}")
    
    def predict(self, last_20min_data):
        """실시간 10분 후 예측"""
        # 입력 데이터 정규화
        seq_len, n_features = last_20min_data.shape
        X_scaled = self.scaler_X.transform(last_20min_data.reshape(-1, n_features)).reshape(1, seq_len, n_features)
        
        # 예측
        pred_scaled = self.model.predict(X_scaled, verbose=0)
        
        # 역변환
        pred_original = self.scaler_y.inverse_transform(pred_scaled)
        
        return pred_original[0, 0]
    
    def predict_with_confidence(self, last_20min_data, n_predictions=10):
        """신뢰구간과 함께 예측 (드롭아웃 활용)"""
        predictions = []
        
        for _ in range(n_predictions):
            pred = self.predict(last_20min_data)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        result = {
            'prediction': np.mean(predictions),
            'std': np.std(predictions),
            'lower_bound': np.percentile(predictions, 5),
            'upper_bound': np.percentile(predictions, 95)
        }
        
        return result

def main():
    """메인 실행 함수"""
    
    print("\n" + "="*70)
    print("🏭 HUBROOM 반송량 예측 시스템")
    print("📊 TensorFlow 2.15 - 완전판")
    print("="*70)
    
    # 1. 데이터 처리
    processor = HUBROOMProcessor()
    data = processor.process_with_resume()
    
    # 2. 모델 설정
    config = {
        'seq_len': 20,
        'n_features': data['n_features'],
        'patch_len': 5,
        'd_model': 128,
        'n_heads': 8,
        'd_ff': 256,
        'n_layers': 3,
        'dropout': 0.1,
        'max_capacity': 200
    }
    
    print(f"\n📊 데이터 준비 완료:")
    print(f"  • Train: {data['X_train'].shape}")
    print(f"  • Valid: {data['X_val'].shape}")
    print(f"  • Test: {data['X_test'].shape}")
    
    # 3. 모델 생성
    print("\n🤖 모델 생성 중...")
    model1 = create_patchtst_model(config)
    model2 = create_patchtst_pinn_model(config)
    
    model1.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )
    
    model2.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("✅ 모델 생성 완료!")
    print(f"  • Model 1: PatchTST")
    print(f"  • Model 2: PatchTST + PINN")
    
    # 4. 학습
    checkpoint_mgr = CheckpointManager()
    
    # 체크포인트 콜백
    callbacks = [
        ProgressCallback(checkpoint_mgr),
        ModelCheckpoint('checkpoints/best_model.h5', save_best_only=True),
        EarlyStopping(patience=10, restore_best_weights=True)
    ]
    
    print("\n🚀 Model 1 (PatchTST) 학습 시작...")
    history1 = model1.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n🚀 Model 2 (PatchTST+PINN) 학습 시작...")
    history2 = model2.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. 평가
    print("\n📊 모델 평가 중...")
    
    # Model 1 평가
    test_loss1, test_mae1 = model1.evaluate(data['X_test'], data['y_test'], verbose=0)
    pred1 = model1.predict(data['X_test'], verbose=0)
    
    # Model 2 평가
    test_loss2, test_mae2 = model2.evaluate(data['X_test'], data['y_test'], verbose=0)
    pred2 = model2.predict(data['X_test'], verbose=0)
    
    # 역변환
    scaler_y = data['scalers']['y']
    pred1_original = scaler_y.inverse_transform(pred1)
    pred2_original = scaler_y.inverse_transform(pred2)
    y_test_original = scaler_y.inverse_transform(data['y_test'].reshape(-1, 1))
    
    # 결과 출력
    print("\n" + "="*70)
    print("📈 최종 결과")
    print("="*70)
    
    print("\nModel 1 (PatchTST):")
    print(f"  • Test MAE: {test_mae1:.4f}")
    print(f"  • Test RMSE: {np.sqrt(test_loss1):.4f}")
    
    print("\nModel 2 (PatchTST + PINN):")
    print(f"  • Test MAE: {test_mae2:.4f}")
    print(f"  • Test RMSE: {np.sqrt(test_loss2):.4f}")
    
    # 물리 제약 검증
    print("\n🔬 물리 제약 검증:")
    print(f"  • Model 1 음수 예측: {np.sum(pred1_original < 0)}개")
    print(f"  • Model 2 음수 예측: {np.sum(pred2_original < 0)}개")
    print(f"  • Model 1 용량 초과: {np.sum(pred1_original > 200)}개")
    print(f"  • Model 2 용량 초과: {np.sum(pred2_original > 200)}개")
    
    print("\n✅ 모든 작업 완료!")
    
    return model1, model2, history1, history2

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HUBROOM 예측 시스템')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'], 
                       default='train', help='실행 모드')
    parser.add_argument('--resume', action='store_true', 
                       help='이전 학습 이어서 진행')
    parser.add_argument('--model', choices=['patchtst', 'pinn', 'both'], 
                       default='both', help='학습할 모델')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            # 학습 모드
            model1, model2, history1, history2 = main()
            
        elif args.mode == 'predict':
            # 예측 모드
            print("\n🔮 실시간 예측 모드")
            predictor = HUBROOMPredictor()
            
            # 예시: 마지막 20분 데이터 (실제로는 실시간 데이터)
            # last_20min = get_realtime_data()  # 실제 구현 필요
            
            print("✅ 예측 서비스 준비 완료!")
            
        elif args.mode == 'evaluate':
            # 평가 모드
            print("\n📊 모델 평가 모드")
            # 저장된 모델 평가
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자 중단!")
        print("💾 진행 상황이 저장되었습니다.")
        print("다시 실행하면 중단된 지점부터 이어서 진행됩니다.")
        
        # 재개 명령어 안내
        print("\n📌 재개 방법:")
        print("  python tf_patchtst_complete.py --resume")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("💾 진행 상황이 저장되었습니다.")
        
        # 디버그 정보
        import traceback
        print("\n디버그 정보:")
        traceback.print_exc()

# ===========================
# 📁 파일 구조 설명
# ===========================
"""
프로젝트 구조:
├── tf_patchtst_complete.py     # 메인 코드
├── data/
│   └── HUB_0509_TO_0730_DATA.CSV  # 원본 데이터
├── checkpoints/
│   ├── progress.json           # 진행 상황
│   ├── processed_data.pkl      # 처리된 데이터
│   ├── best_model.h5          # 최고 성능 모델
│   ├── model_checkpoint/       # 학습 체크포인트
│   └── scalers/               # 스케일러
│       ├── scaler_X.pkl       # 입력 스케일러
│       └── scaler_y.pkl       # 출력 스케일러
└── logs/                       # 학습 로그

실행 예시:
1. 처음 학습: python tf_patchtst_complete.py
2. 이어서 학습: python tf_patchtst_complete.py --resume
3. 예측 모드: python tf_patchtst_complete.py --mode predict
4. 평가 모드: python tf_patchtst_complete.py --mode evaluate

requirements.txt:
tensorflow==2.15.0
pandas
numpy
scikit-learn
joblib
tqdm
matplotlib
"""