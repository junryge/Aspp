# -*- coding: utf-8 -*-
"""
반도체 물류 예측 시스템 - 280분 → 10분 후 예측
체크포인트 시스템으로 중단/재개 가능
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
                             RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                           classification_report, confusion_matrix, accuracy_score,
                           precision_recall_fscore_support)
import joblib
import pickle
import json
import os
import sys
import time
import warnings
import traceback
from datetime import datetime
import platform
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil 미설치 - 시스템 정보 제한적")
warnings.filterwarnings('ignore')

# ======================== 시스템 정보 확인 ========================
def check_system_info():
    """시스템 및 GPU 정보 확인"""
    print("\n" + "="*70)
    print(" 📌 시스템 정보")
    print("="*70)
    
    # CPU 정보
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version.split()[0]}")
    
    if PSUTIL_AVAILABLE:
        print(f"  CPU: {psutil.cpu_count()} cores")
        print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"  사용 가능 RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    else:
        import multiprocessing
        print(f"  CPU: {multiprocessing.cpu_count()} cores")
        print("  RAM: psutil 필요 (pip install psutil)")
    
    # GPU 확인 (sklearn은 GPU를 직접 사용하지 않음)
    print("\n  🔍 GPU 확인:")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"    TensorFlow GPU 발견: {len(gpus)}개")
            for i, gpu in enumerate(gpus):
                print(f"      GPU {i}: {gpu}")
        else:
            print("    TensorFlow GPU 없음 (CPU 모드)")
    except ImportError:
        print("    TensorFlow 미설치 - Sklearn은 CPU만 사용")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"    PyTorch GPU 발견: {torch.cuda.device_count()}개")
            print(f"      GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("    PyTorch GPU 없음 (CPU 모드)")
    except ImportError:
        pass
    
    # sklearn 정보
    import sklearn
    print(f"\n  📚 Scikit-learn 버전: {sklearn.__version__}")
    print("  ⚠️ 주의: Sklearn 모델은 CPU만 사용합니다")
    print("  💡 GPU 사용하려면 TensorFlow/PyTorch 모델 필요")
    
    print("="*70)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (15, 10)

# ======================== 설정 ========================
MODEL_DIR = 'regression_classification_models_280to10'
CHECKPOINT_DIR = 'checkpoint'
BACKUP_DIR = 'checkpoint_backup'
SEQ_LENGTH = 280  # 과거 280분
PRED_HORIZON = 10  # 10분 후 예측

# 폴더 생성
for dir_path in [MODEL_DIR, CHECKPOINT_DIR, BACKUP_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# ======================== 체크포인트 관리 ========================
class CheckpointManager:
    def __init__(self):
        self.checkpoint_file = os.path.join(CHECKPOINT_DIR, 'state.json')
        self.start_time = time.time()
        
    def load_checkpoint(self):
        """체크포인트 로드"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_checkpoint(self, state):
        """체크포인트 저장"""
        state['last_saved'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        state['elapsed_time'] = time.time() - self.start_time
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # 백업 (1시간마다)
        if state['elapsed_time'] > 3600:
            backup_dir = os.path.join(BACKUP_DIR, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            import shutil
            shutil.copy(self.checkpoint_file, backup_dir)
    
    def create_new_state(self):
        """새로운 상태 생성"""
        return {
            'version': '1.0',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stage': 'INIT',
            'completed_steps': {
                'data_loading': False,
                'sequence_creation': False,
                'train_test_split': False,
                'scaling': False
            },
            'completed_models': [],
            'current_model': None,
            'models_metrics': {},
            'data_info': {},
            'elapsed_time': 0
        }

# ======================== 데이터 처리 함수 ========================
def assign_level(totalcnt_value):
    """3구간 분류"""
    if totalcnt_value < 1400:
        return 0  # 정상
    elif totalcnt_value < 1700:
        return 1  # 확인
    else:
        return 2  # 위험

def detect_anomaly_signal(totalcnt_value):
    """이상신호 감지 (1651-1682)"""
    return 1 if 1651 <= totalcnt_value <= 1682 else 0

def create_sequences_280to10(data, seq_length=280, pred_horizon=10):
    """280분 시퀀스로 10분 후 예측 데이터 생성"""
    
    print(f"\n📊 시퀀스 생성 중... (280분 → 10분 후)")
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 
                   'M16M14A', 'TOTALCNT']
    
    data = data.copy()
    
    # 파생 변수 생성
    data['ratio_M14B_M14A'] = np.clip(data['M14AM14B'] / (data['M14AM10A'] + 1), 0, 1000)
    data['ratio_M14B_M16'] = np.clip(data['M14AM14B'] / (data['M14AM16'] + 1), 0, 1000)
    data['totalcnt_change'] = data['TOTALCNT'].diff().fillna(0)
    data['totalcnt_pct_change'] = data['TOTALCNT'].pct_change().fillna(0)
    data['totalcnt_pct_change'] = np.clip(data['totalcnt_pct_change'], -10, 10)
    
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)
    
    X_list = []
    y_reg_list = []
    y_cls_list = []
    y_anomaly_list = []
    
    n_sequences = len(data) - seq_length - pred_horizon + 1
    print(f"✅ 생성 가능한 시퀀스: {n_sequences:,}개")
    
    # 체크포인트 확인
    seq_checkpoint = os.path.join(CHECKPOINT_DIR, 'sequences_progress.json')
    start_idx = 0
    
    if os.path.exists(seq_checkpoint):
        with open(seq_checkpoint, 'r') as f:
            seq_progress = json.load(f)
            start_idx = seq_progress.get('last_processed', 0)
            
            if start_idx > 0:
                print(f"✅ 체크포인트 발견: {start_idx}/{n_sequences} 완료")
                # 기존 데이터 로드
                X_list = np.load(os.path.join(CHECKPOINT_DIR, 'X_partial.npy')).tolist()
                y_reg_list = np.load(os.path.join(CHECKPOINT_DIR, 'y_reg_partial.npy')).tolist()
                y_cls_list = np.load(os.path.join(CHECKPOINT_DIR, 'y_cls_partial.npy')).tolist()
                y_anomaly_list = np.load(os.path.join(CHECKPOINT_DIR, 'y_anomaly_partial.npy')).tolist()
    
    # 시퀀스 생성 (이어서)
    for i in range(start_idx, n_sequences):
        if i % 1000 == 0:
            print(f"  진행률: {i}/{n_sequences} ({i/n_sequences*100:.1f}%)", end='\r')
            
            # 중간 저장 (5000개마다)
            if i > 0 and i % 5000 == 0:
                np.save(os.path.join(CHECKPOINT_DIR, 'X_partial.npy'), X_list)
                np.save(os.path.join(CHECKPOINT_DIR, 'y_reg_partial.npy'), y_reg_list)
                np.save(os.path.join(CHECKPOINT_DIR, 'y_cls_partial.npy'), y_cls_list)
                np.save(os.path.join(CHECKPOINT_DIR, 'y_anomaly_partial.npy'), y_anomaly_list)
                
                with open(seq_checkpoint, 'w') as f:
                    json.dump({'last_processed': i}, f)
                print(f"\n💾 체크포인트 저장: {i}/{n_sequences}")
        
        start = i
        end = i + seq_length
        seq_data = data.iloc[start:end]
        
        features = []
        
        # 각 컬럼별 특징 추출
        for col in feature_cols:
            values = seq_data[col].values
            
            # 기본 통계
            features.extend([
                np.mean(values),
                np.std(values) if len(values) > 1 else 0,
                np.min(values),
                np.max(values),
                np.percentile(values, 25),
                np.percentile(values, 50),
                np.percentile(values, 75),
                values[-1],  # 현재값
                values[-1] - values[0],  # 전체 변화량
                np.mean(values[-60:]),  # 최근 1시간 평균
                np.max(values[-60:]),  # 최근 1시간 최대
                np.mean(values[-30:]),  # 최근 30분 평균
                np.max(values[-30:])   # 최근 30분 최대
            ])
            
            # TOTALCNT 특별 처리
            if col == 'TOTALCNT':
                features.append(np.sum((values >= 1650) & (values < 1700)))
                features.append(np.sum(values >= 1700))
                features.append(np.max(values[-20:]))
                features.append(np.sum(values < 1400))
                features.append(np.sum((values >= 1400) & (values < 1700)))
                features.append(np.sum(values >= 1700))
                features.append(np.sum((values >= 1651) & (values <= 1682)))
                
                anomaly_values = values[(values >= 1651) & (values <= 1682)]
                features.append(np.max(anomaly_values) if len(anomaly_values) > 0 else 0)
                
                normal_vals = values[values < 1400]
                check_vals = values[(values >= 1400) & (values < 1700)]
                danger_vals = values[values >= 1700]
                
                features.append(np.mean(normal_vals) if len(normal_vals) > 0 else 0)
                features.append(np.mean(check_vals) if len(check_vals) > 0 else 0)
                features.append(np.mean(danger_vals) if len(danger_vals) > 0 else 0)
                
                try:
                    x = np.arange(len(values))
                    slope, _ = np.polyfit(x, values, 1)
                    features.append(np.clip(slope, -100, 100))
                except:
                    features.append(0)
                
                try:
                    recent_slope = np.polyfit(np.arange(60), values[-60:], 1)[0]
                    features.append(np.clip(recent_slope, -100, 100))
                except:
                    features.append(0)
        
        # 파생 변수
        last_idx = end - 1
        features.extend([
            np.clip(data['ratio_M14B_M14A'].iloc[last_idx], 0, 1000),
            np.clip(data['ratio_M14B_M16'].iloc[last_idx], 0, 1000),
            np.clip(data['totalcnt_change'].iloc[last_idx], -1000, 1000),
            np.clip(data['totalcnt_pct_change'].iloc[last_idx], -10, 10),
        ])
        
        # 타겟: 10분 후 TOTALCNT
        target_idx = end + pred_horizon - 1
        if target_idx < len(data):
            future_totalcnt = data['TOTALCNT'].iloc[target_idx]
            
            X_list.append(features)
            y_reg_list.append(future_totalcnt)
            y_cls_list.append(assign_level(future_totalcnt))
            y_anomaly_list.append(detect_anomaly_signal(future_totalcnt))
    
    X = np.array(X_list)
    y_reg = np.array(y_reg_list)
    y_cls = np.array(y_cls_list)
    y_anomaly = np.array(y_anomaly_list)
    
    X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    # 완료 후 임시 파일 삭제
    for file in ['X_partial.npy', 'y_reg_partial.npy', 'y_cls_partial.npy', 'y_anomaly_partial.npy']:
        filepath = os.path.join(CHECKPOINT_DIR, file)
        if os.path.exists(filepath):
            os.remove(filepath)
    
    if os.path.exists(seq_checkpoint):
        os.remove(seq_checkpoint)
    
    print(f"\n✅ 시퀀스 생성 완료!")
    print(f"  특징 개수: {X.shape[1]}개")
    
    return X, y_reg, y_cls, y_anomaly

# ======================== 진행 상황 표시 ========================
def display_progress(state, checkpoint_manager):
    """실시간 진행 상황 표시"""
    
    os.system('cls' if os.name == 'nt' else 'clear')  # 화면 클리어
    
    print("\n" + "="*70)
    print(" 🚀 반도체 물류 예측 시스템 - 280분 → 10분 후 예측")
    print("="*70)
    
    # 데이터 처리 상태
    print("\n📊 데이터 전처리:")
    steps = state['completed_steps']
    print(f"  {'✅' if steps['data_loading'] else '⏸️'} 데이터 로딩")
    print(f"  {'✅' if steps['sequence_creation'] else '⏸️'} 시퀀스 생성 (280분)")
    print(f"  {'✅' if steps['train_test_split'] else '⏸️'} Train/Test 분할")
    print(f"  {'✅' if steps['scaling'] else '⏸️'} 데이터 스케일링")
    
    # 모델 학습 상태
    regression_models = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 
                        'MLP', 'Ridge', 'ElasticNet']
    classification_models = ['RandomForest_Cls', 'ExtraTrees_Cls', 
                            'GradientBoosting_Cls', 'MLP_Cls', 'LogisticRegression']
    
    print("\n📈 회귀 모델 (6개):")
    for model in regression_models:
        if model in state['completed_models']:
            metrics = state['models_metrics'].get(model, {})
            print(f"  ✅ {model:20} MAE: {metrics.get('mae', 0):.2f}, R²: {metrics.get('r2', 0):.4f}")
        elif model == state['current_model']:
            print(f"  ⏳ {model:20} [학습 중...]")
        else:
            print(f"  ⏸️ {model:20}")
    
    print("\n📊 분류 모델 (5개):")
    for model in classification_models:
        if model in state['completed_models']:
            metrics = state['models_metrics'].get(model, {})
            print(f"  ✅ {model:20} 정확도: {metrics.get('accuracy', 0):.3f}")
        elif model == state['current_model']:
            print(f"  ⏳ {model:20} [학습 중...]")
        else:
            print(f"  ⏸️ {model:20}")
    
    # 전체 진행률
    total_models = 11
    completed = len(state['completed_models'])
    progress = completed / total_models * 100 if total_models > 0 else 0
    
    print(f"\n📊 전체 진행률: {progress:.1f}% [{completed}/{total_models}]")
    print(f"[{'█' * int(progress/2)}{' ' * (50-int(progress/2))}]")
    
    # 시간 정보
    elapsed = state.get('elapsed_time', 0)
    print(f"\n⏱️ 경과 시간: {elapsed//60:.0f}분 {elapsed%60:.0f}초")
    
    if 'last_saved' in state:
        print(f"💾 마지막 저장: {state['last_saved']}")
    
    print("="*70)

# ======================== 메인 학습 함수 ========================
def train_with_checkpoint():
    """체크포인트 시스템을 사용한 학습"""
    
    checkpoint_manager = CheckpointManager()
    
    # 체크포인트 확인
    state = checkpoint_manager.load_checkpoint()
    
    if state is None:
        print("🆕 새로운 학습 시작...")
        state = checkpoint_manager.create_new_state()
    else:
        print("✅ 체크포인트 발견! 이어서 학습합니다...")
        print(f"  완료된 모델: {len(state['completed_models'])}개")
    
    try:
        # ============ STEP 1: 데이터 로딩 ============
        if not state['completed_steps']['data_loading']:
            print("\n[Step 1] 데이터 로딩...")
            
            df1 = pd.read_csv('uu.csv')
            df2 = pd.read_csv('uu2.csv')
            df = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)
            
            # 필수 컬럼 확인
            required_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                           'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 
                           'M16M14A', 'TOTALCNT']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ 필수 컬럼 누락: {missing_cols}")
                print(f"   현재 컬럼: {list(df.columns)}")
                raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")
            
            print(f"✅ 전체 데이터: {len(df):,}행")
            print(f"✅ 시간 범위: {len(df)} 분 ({len(df)/60:.1f} 시간)")
            print(f"✅ 모든 필수 컬럼 확인 완료")
            
            # 데이터 저장
            df.to_pickle(os.path.join(CHECKPOINT_DIR, 'df_combined.pkl'))
            
            state['data_info']['total_rows'] = len(df)
            state['completed_steps']['data_loading'] = True
            checkpoint_manager.save_checkpoint(state)
        else:
            print("\n✅ 데이터 로딩 완료 (체크포인트에서 복원)")
            df = pd.read_pickle(os.path.join(CHECKPOINT_DIR, 'df_combined.pkl'))
        
        # ============ STEP 2: 시퀀스 생성 ============
        if not state['completed_steps']['sequence_creation']:
            print("\n[Step 2] 280분 → 10분 후 시퀀스 생성...")
            
            X, y_reg, y_cls, y_anomaly = create_sequences_280to10(df, SEQ_LENGTH, PRED_HORIZON)
            
            # 저장
            np.save(os.path.join(CHECKPOINT_DIR, 'X.npy'), X)
            np.save(os.path.join(CHECKPOINT_DIR, 'y_reg.npy'), y_reg)
            np.save(os.path.join(CHECKPOINT_DIR, 'y_cls.npy'), y_cls)
            np.save(os.path.join(CHECKPOINT_DIR, 'y_anomaly.npy'), y_anomaly)
            
            state['data_info']['num_sequences'] = X.shape[0]
            state['data_info']['num_features'] = X.shape[1]
            state['completed_steps']['sequence_creation'] = True
            checkpoint_manager.save_checkpoint(state)
            
            # 데이터 분석
            print(f"\n📊 타겟(10분 후 TOTALCNT) 분석:")
            print(f"  최소값: {y_reg.min():.0f}")
            print(f"  최대값: {y_reg.max():.0f}")
            print(f"  평균값: {y_reg.mean():.0f} (±{y_reg.std():.0f})")
            
            print(f"\n📊 3구간 분포:")
            for i in range(3):
                count = np.sum(y_cls == i)
                names = ["정상(900-1399)", "확인(1400-1699)", "위험(1700+)"]
                print(f"  {names[i]}: {count:,}개 ({count/len(y_cls)*100:.2f}%)")
        else:
            print("\n✅ 시퀀스 생성 완료 (체크포인트에서 복원)")
            X = np.load(os.path.join(CHECKPOINT_DIR, 'X.npy'))
            y_reg = np.load(os.path.join(CHECKPOINT_DIR, 'y_reg.npy'))
            y_cls = np.load(os.path.join(CHECKPOINT_DIR, 'y_cls.npy'))
            y_anomaly = np.load(os.path.join(CHECKPOINT_DIR, 'y_anomaly.npy'))
        
        # ============ STEP 3: Train/Test 분할 ============
        if not state['completed_steps']['train_test_split']:
            print("\n[Step 3] Train/Test 분할...")
            
            X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, y_anomaly_train, y_anomaly_test = train_test_split(
                X, y_reg, y_cls, y_anomaly, test_size=0.2, random_state=42, shuffle=True
            )
            
            # 저장
            np.save(os.path.join(CHECKPOINT_DIR, 'X_train.npy'), X_train)
            np.save(os.path.join(CHECKPOINT_DIR, 'X_test.npy'), X_test)
            np.save(os.path.join(CHECKPOINT_DIR, 'y_reg_train.npy'), y_reg_train)
            np.save(os.path.join(CHECKPOINT_DIR, 'y_reg_test.npy'), y_reg_test)
            np.save(os.path.join(CHECKPOINT_DIR, 'y_cls_train.npy'), y_cls_train)
            np.save(os.path.join(CHECKPOINT_DIR, 'y_cls_test.npy'), y_cls_test)
            
            print(f"✅ 훈련 데이터: {X_train.shape[0]:,}개")
            print(f"✅ 테스트 데이터: {X_test.shape[0]:,}개")
            
            state['data_info']['train_samples'] = X_train.shape[0]
            state['data_info']['test_samples'] = X_test.shape[0]
            state['completed_steps']['train_test_split'] = True
            checkpoint_manager.save_checkpoint(state)
        else:
            print("\n✅ Train/Test 분할 완료 (체크포인트에서 복원)")
            X_train = np.load(os.path.join(CHECKPOINT_DIR, 'X_train.npy'))
            X_test = np.load(os.path.join(CHECKPOINT_DIR, 'X_test.npy'))
            y_reg_train = np.load(os.path.join(CHECKPOINT_DIR, 'y_reg_train.npy'))
            y_reg_test = np.load(os.path.join(CHECKPOINT_DIR, 'y_reg_test.npy'))
            y_cls_train = np.load(os.path.join(CHECKPOINT_DIR, 'y_cls_train.npy'))
            y_cls_test = np.load(os.path.join(CHECKPOINT_DIR, 'y_cls_test.npy'))
        
        # ============ STEP 4: 스케일링 ============
        if not state['completed_steps']['scaling']:
            print("\n[Step 4] 데이터 스케일링...")
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 저장
            joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, 'scaler.pkl'))
            np.save(os.path.join(CHECKPOINT_DIR, 'X_train_scaled.npy'), X_train_scaled)
            np.save(os.path.join(CHECKPOINT_DIR, 'X_test_scaled.npy'), X_test_scaled)
            
            state['completed_steps']['scaling'] = True
            checkpoint_manager.save_checkpoint(state)
        else:
            print("\n✅ 스케일링 완료 (체크포인트에서 복원)")
            scaler = joblib.load(os.path.join(CHECKPOINT_DIR, 'scaler.pkl'))
            X_train_scaled = np.load(os.path.join(CHECKPOINT_DIR, 'X_train_scaled.npy'))
            X_test_scaled = np.load(os.path.join(CHECKPOINT_DIR, 'X_test_scaled.npy'))
        
        # ============ STEP 5: 모델 학습 ============
        print("\n" + "="*70)
        print(" 📚 모델 학습 시작")
        print("="*70)
        
        # XGBoost 모델 추가 (GPU 우선, 실패시 CPU)
        xgboost_models = {}
        try:
            import xgboost as xgb
            print("\n✅ XGBoost 발견! GPU 가속 확인 중...")
            
            # GPU 테스트
            test_X = np.random.rand(100, 10)
            test_y = np.random.rand(100)
            
            try:
                # GPU 버전 시도
                test_model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=10)
                test_model.fit(test_X, test_y)
                print("  🎮 GPU 사용 가능! XGBoost GPU 모델 추가")
                
                # GPU 회귀 모델
                xgboost_models['XGBoost_GPU'] = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=10, 
                    learning_rate=0.1,
                    tree_method='gpu_hist',  # GPU 사용
                    gpu_id=0,
                    random_state=42,
                    verbosity=0
                )
                
                # GPU 분류 모델
                xgboost_models['XGBoost_GPU_Cls'] = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=10,
                    learning_rate=0.1,
                    tree_method='gpu_hist',  # GPU 사용
                    gpu_id=0,
                    random_state=42,
                    verbosity=0
                )
                
            except Exception as gpu_error:
                print(f"  ⚠️ GPU 사용 불가: {str(gpu_error)[:50]}")
                print("  💻 CPU 모드로 XGBoost 실행")
                
                # CPU 버전으로 폴백 (2코어 제한)
                xgboost_models['XGBoost_CPU'] = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=10,
                    learning_rate=0.1,
                    tree_method='hist',  # CPU 사용
                    nthread=2,  # CPU 2코어만
                    random_state=42,
                    verbosity=0
                )
                
                xgboost_models['XGBoost_CPU_Cls'] = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=10,
                    learning_rate=0.1,
                    tree_method='hist',  # CPU 사용
                    nthread=2,  # CPU 2코어만
                    random_state=42,
                    verbosity=0
                )
                
        except ImportError:
            print("❌ XGBoost 미설치! 설치 명령: pip install xgboost")
        
        try:
            import lightgbm as lgb
            print("✅ LightGBM 발견 - GPU 가속 시도")
            xgboost_models['LightGBM_GPU'] = lgb.LGBMRegressor(
                n_estimators=300, max_depth=10, learning_rate=0.1,
                device='gpu', gpu_platform_id=0, gpu_device_id=0,
                random_state=42, verbose=-1
            )
            xgboost_models['LightGBM_GPU_Cls'] = lgb.LGBMClassifier(
                n_estimators=300, max_depth=10, learning_rate=0.1,
                device='gpu', gpu_platform_id=0, gpu_device_id=0,
                random_state=42, verbose=-1
            )
        except:
            print("⚠️ LightGBM GPU 사용 불가 또는 미설치")
        
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
            print("✅ CatBoost 발견 - GPU 가속 시도")
            xgboost_models['CatBoost_GPU'] = CatBoostRegressor(
                iterations=300, depth=10, learning_rate=0.1,
                task_type='GPU', devices='0', random_state=42, verbose=False
            )
            xgboost_models['CatBoost_GPU_Cls'] = CatBoostClassifier(
                iterations=300, depth=10, learning_rate=0.1,
                task_type='GPU', devices='0', random_state=42, verbose=False
            )
        except:
            print("⚠️ CatBoost GPU 사용 불가 또는 미설치")
        
        # 회귀 모델 정의 (CPU 2코어 제한)
        regression_models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=300, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=2  # CPU 2코어만
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=300, max_depth=20, random_state=42, n_jobs=2  # CPU 2코어만
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=10,
                subsample=0.8, random_state=42
                # GradientBoosting은 n_jobs 파라미터 없음
            ),
            'MLP': MLPRegressor(
                hidden_layer_sizes=(200, 100, 50), activation='relu',
                max_iter=1000, early_stopping=True,
                validation_fraction=0.1, random_state=42
                # MLPRegressor는 n_jobs 파라미터 없음
            ),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        # 분류 모델 정의 (CPU 2코어 제한)
        classification_models = {
            'RandomForest_Cls': RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=2,  # CPU 2코어만
                class_weight='balanced'
            ),
            'ExtraTrees_Cls': ExtraTreesClassifier(
                n_estimators=300, max_depth=20, random_state=42,
                n_jobs=2, class_weight='balanced'  # CPU 2코어만
            ),
            'GradientBoosting_Cls': GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=10,
                subsample=0.8, random_state=42
                # GradientBoostingClassifier는 n_jobs 파라미터 없음
            ),
            'MLP_Cls': MLPClassifier(
                hidden_layer_sizes=(200, 100, 50), activation='relu',
                max_iter=1000, early_stopping=True,
                validation_fraction=0.1, random_state=42
                # MLPClassifier는 n_jobs 파라미터 없음
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced',
                n_jobs=2  # CPU 2코어만
            )
        }
        
        # 모든 모델 통합
        all_models = list(regression_models.items()) + list(classification_models.items()) + list(xgboost_models.items())
        
        if xgboost_models:
            print(f"\n🎮 GPU/고성능 모델 {len(xgboost_models)}개 추가됨!")
            for model_name in xgboost_models.keys():
                print(f"    - {model_name}")
            print(f"\n📊 총 {len(all_models)}개 모델 학습 예정\n")
        
        # 모델 학습 (이어서)
        for model_name, model in all_models:
            if model_name in state['completed_models']:
                continue  # 이미 완료된 모델 건너뛰기
            
            state['current_model'] = model_name
            checkpoint_manager.save_checkpoint(state)
            display_progress(state, checkpoint_manager)
            
            print(f"\n🚀 [{model_name}] 학습 시작...")
            start_time = time.time()
            
            try:
                if 'Cls' in model_name or model_name == 'LogisticRegression':
                    # 분류 모델
                    model.fit(X_train_scaled, y_cls_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    accuracy = accuracy_score(y_cls_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_cls_test, y_pred, average='weighted'
                    )
                    
                    metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                    
                    print(f"  ✅ 정확도: {accuracy:.3f}")
                    print(f"  ✅ F1-Score: {f1:.3f}")
                else:
                    # 회귀 모델
                    model.fit(X_train_scaled, y_reg_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    mae = mean_absolute_error(y_reg_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
                    r2 = r2_score(y_reg_test, y_pred)
                    
                    # 3구간 정확도
                    pred_cls = np.array([assign_level(val) for val in y_pred])
                    cls_accuracy = accuracy_score(y_cls_test, pred_cls)
                    
                    metrics = {
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'cls_accuracy': cls_accuracy
                    }
                    
                    print(f"  ✅ MAE: {mae:.2f}")
                    print(f"  ✅ R²: {r2:.4f}")
                    print(f"  ✅ 3구간 정확도: {cls_accuracy:.3f}")
                
                # 모델 저장
                model_path = os.path.join(MODEL_DIR, f'{model_name}_model.pkl')
                joblib.dump(model, model_path)
                
                # 상태 업데이트
                state['completed_models'].append(model_name)
                state['models_metrics'][model_name] = metrics
                state['current_model'] = None
                
                elapsed = time.time() - start_time
                print(f"  ⏱️ 소요 시간: {elapsed:.1f}초")
                
                checkpoint_manager.save_checkpoint(state)
                
            except Exception as e:
                print(f"\n❌ {model_name} 오류 발생: {str(e)}")
                print(f"   체크포인트 저장 후 이 모델은 건너뜁니다.")
                traceback.print_exc()
                
                # 오류 발생해도 다음 모델로 진행
                state['completed_models'].append(model_name + "_ERROR")
                checkpoint_manager.save_checkpoint(state)
                continue
        
        # ============ 최종 결과 ============
        print("\n" + "="*70)
        print(" 🎉 학습 완료!")
        print("="*70)
        
        # 최고 성능 모델 찾기
        reg_metrics = {k: v for k, v in state['models_metrics'].items() 
                      if 'Cls' not in k and 'LogisticRegression' not in k}
        cls_metrics = {k: v for k, v in state['models_metrics'].items() 
                      if 'Cls' in k or k == 'LogisticRegression'}
        
        if reg_metrics:
            best_r2_model = max(reg_metrics.items(), key=lambda x: x[1].get('r2', 0))[0]
            best_mae_model = min(reg_metrics.items(), key=lambda x: x[1].get('mae', float('inf')))[0]
            
            print(f"\n📈 회귀 모델 최고 성능:")
            print(f"  최고 R²: {best_r2_model} (R²={reg_metrics[best_r2_model]['r2']:.4f})")
            print(f"  최저 MAE: {best_mae_model} (MAE={reg_metrics[best_mae_model]['mae']:.2f})")
        
        if cls_metrics:
            best_cls_model = max(cls_metrics.items(), key=lambda x: x[1].get('accuracy', 0))[0]
            
            print(f"\n📊 분류 모델 최고 성능:")
            print(f"  최고 정확도: {best_cls_model} ({cls_metrics[best_cls_model]['accuracy']:.3f})")
        
        # 최종 스케일러 저장
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
        
        # 최종 설정 저장
        config = {
            'sequence_length': SEQ_LENGTH,
            'prediction_horizon': PRED_HORIZON,
            'num_features': X.shape[1],
            'best_models': {
                'r2': best_r2_model if reg_metrics else None,
                'mae': best_mae_model if reg_metrics else None,
                'accuracy': best_cls_model if cls_metrics else None
            },
            'all_metrics': state['models_metrics'],
            'training_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(MODEL_DIR, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        print(f"\n✅ 모든 모델 저장 완료: {MODEL_DIR}/")
        print(f"✅ 총 {len(state['completed_models'])}개 모델 학습 완료")
        
        # 체크포인트 정리
        print("\n🧹 체크포인트 파일 정리...")
        for file in os.listdir(CHECKPOINT_DIR):
            if file not in ['state.json']:
                os.remove(os.path.join(CHECKPOINT_DIR, file))
        
        print("\n✨ 시스템 준비 완료! 280분 데이터로 10분 후 예측 가능!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자 중단 감지!")
        print("💾 현재 상태를 저장하고 있습니다...")
        checkpoint_manager.save_checkpoint(state)
        print("✅ 체크포인트 저장 완료. 다시 실행하면 이어서 학습합니다.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ 예기치 않은 오류 발생: {str(e)}")
        print("💾 현재 상태를 저장하고 있습니다...")
        checkpoint_manager.save_checkpoint(state)
        traceback.print_exc()
        print("\n✅ 체크포인트 저장 완료. 다시 실행하면 이어서 학습합니다.")
        sys.exit(1)

# ======================== 실시간 예측 함수 ========================
def extract_features_from_sequence(data_280min):
    """280분 시퀀스에서 특징 추출 (실제 구현)"""
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 
                   'M16M14A', 'TOTALCNT']
    
    # DataFrame으로 변환
    if isinstance(data_280min, np.ndarray):
        df = pd.DataFrame(data_280min, columns=feature_cols)
    else:
        df = data_280min
    
    # 파생 변수
    df['ratio_M14B_M14A'] = np.clip(df['M14AM14B'] / (df['M14AM10A'] + 1), 0, 1000)
    df['ratio_M14B_M16'] = np.clip(df['M14AM14B'] / (df['M14AM16'] + 1), 0, 1000)
    df['totalcnt_change'] = df['TOTALCNT'].diff().fillna(0)
    df['totalcnt_pct_change'] = df['TOTALCNT'].pct_change().fillna(0).clip(-10, 10)
    
    features = []
    
    # 각 컬럼별 특징 추출 (create_sequences_280to10과 동일)
    for col in feature_cols:
        values = df[col].values
        
        features.extend([
            np.mean(values),
            np.std(values) if len(values) > 1 else 0,
            np.min(values),
            np.max(values),
            np.percentile(values, 25),
            np.percentile(values, 50),
            np.percentile(values, 75),
            values[-1],
            values[-1] - values[0],
            np.mean(values[-60:]),
            np.max(values[-60:]),
            np.mean(values[-30:]),
            np.max(values[-30:])
        ])
        
        if col == 'TOTALCNT':
            features.append(np.sum((values >= 1650) & (values < 1700)))
            features.append(np.sum(values >= 1700))
            features.append(np.max(values[-20:]))
            features.append(np.sum(values < 1400))
            features.append(np.sum((values >= 1400) & (values < 1700)))
            features.append(np.sum(values >= 1700))
            features.append(np.sum((values >= 1651) & (values <= 1682)))
            
            anomaly_values = values[(values >= 1651) & (values <= 1682)]
            features.append(np.max(anomaly_values) if len(anomaly_values) > 0 else 0)
            
            normal_vals = values[values < 1400]
            check_vals = values[(values >= 1400) & (values < 1700)]
            danger_vals = values[values >= 1700]
            
            features.append(np.mean(normal_vals) if len(normal_vals) > 0 else 0)
            features.append(np.mean(check_vals) if len(check_vals) > 0 else 0)
            features.append(np.mean(danger_vals) if len(danger_vals) > 0 else 0)
            
            try:
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                features.append(np.clip(slope, -100, 100))
            except:
                features.append(0)
            
            try:
                recent_slope = np.polyfit(np.arange(60), values[-60:], 1)[0]
                features.append(np.clip(recent_slope, -100, 100))
            except:
                features.append(0)
    
    # 파생 변수 추가
    features.extend([
        np.clip(df['ratio_M14B_M14A'].iloc[-1], 0, 1000),
        np.clip(df['ratio_M14B_M16'].iloc[-1], 0, 1000),
        np.clip(df['totalcnt_change'].iloc[-1], -1000, 1000),
        np.clip(df['totalcnt_pct_change'].iloc[-1], -10, 10),
    ])
    
    return np.array(features)

def predict_10min_later_integrated(sequence_280min, model_name=None, model_type='regression'):
    """
    280분 데이터로 10분 후 TOTALCNT 예측 + 3구간 분류
    
    Parameters:
    -----------
    sequence_280min : DataFrame or array
        280분 시퀀스 데이터 (280행 x 10컬럼)
    model_name : str
        사용할 모델 (None이면 best 모델 사용)
    model_type : str
        'regression' 또는 'classification'
    
    Returns:
    --------
    dict : 예측 결과
    """
    try:
        # 설정 로드
        config_path = os.path.join(MODEL_DIR, 'config.pkl')
        if not os.path.exists(config_path):
            raise FileNotFoundError("학습된 모델이 없습니다. 먼저 학습을 실행하세요.")
        
        config = pickle.load(open(config_path, 'rb'))
        
        # 모델 선택
        if model_name is None:
            if model_type == 'regression':
                model_name = config['best_models'].get('r2', 'RandomForest')
            else:
                model_name = config['best_models'].get('accuracy', 'RandomForest_Cls')
        
        # 모델과 스케일러 로드
        model_path = os.path.join(MODEL_DIR, f'{model_name}_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # 특징 추출
        features = extract_features_from_sequence(sequence_280min)
        features_scaled = scaler.transform([features])
        
        # 예측
        if 'Cls' in model_name or model_name == 'LogisticRegression':
            # 분류 모델
            predicted_level = model.predict(features_scaled)[0]
            level_values = {0: 1150, 1: 1550, 2: 1850}  # 각 레벨의 대표값
            predicted_value = level_values[predicted_level]
            
            if hasattr(model, 'predict_proba'):
                confidence = np.max(model.predict_proba(features_scaled)[0])
            else:
                confidence = 0.8
        else:
            # 회귀 모델
            predicted_value = model.predict(features_scaled)[0]
            predicted_level = assign_level(predicted_value)
            confidence = 0.85  # 회귀 모델의 기본 신뢰도
        
        level_names = {0: '정상 (900-1399)', 1: '확인 (1400-1699)', 2: '위험 (1700+)'}
        
        return {
            'predicted_value': float(predicted_value),
            'predicted_level': int(predicted_level),
            'level_name': level_names[predicted_level],
            'is_danger': predicted_value >= 1700,
            'is_anomaly': detect_anomaly_signal(predicted_value),
            'confidence': float(confidence),
            'model_used': model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    except Exception as e:
        print(f"❌ 예측 오류: {str(e)}")
        return None

# ======================== 시각화 함수 ========================
def create_visualizations(state):
    """학습 결과 시각화"""
    if not state['models_metrics']:
        print("시각화할 메트릭이 없습니다.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 회귀 모델 성능
    reg_models = [k for k in state['models_metrics'].keys() 
                  if 'Cls' not in k and 'LogisticRegression' not in k]
    
    if reg_models:
        # MAE 비교
        ax = axes[0, 0]
        mae_values = [state['models_metrics'][m].get('mae', 0) for m in reg_models]
        ax.bar(reg_models, mae_values, color='skyblue')
        ax.set_title('회귀 모델 - MAE')
        ax.set_ylabel('MAE')
        ax.tick_params(axis='x', rotation=45)
        
        # R² 비교
        ax = axes[0, 1]
        r2_values = [state['models_metrics'][m].get('r2', 0) for m in reg_models]
        ax.bar(reg_models, r2_values, color='green', alpha=0.7)
        ax.set_title('회귀 모델 - R² Score')
        ax.set_ylabel('R²')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # 3구간 정확도
        ax = axes[0, 2]
        cls_acc_values = [state['models_metrics'][m].get('cls_accuracy', 0) for m in reg_models]
        ax.bar(reg_models, cls_acc_values, color='purple', alpha=0.7)
        ax.set_title('회귀 모델 - 3구간 정확도')
        ax.set_ylabel('정확도')
        ax.tick_params(axis='x', rotation=45)
    
    # 분류 모델 성능
    cls_models = [k for k in state['models_metrics'].keys() 
                  if 'Cls' in k or k == 'LogisticRegression']
    
    if cls_models:
        # 정확도 비교
        ax = axes[1, 0]
        acc_values = [state['models_metrics'][m].get('accuracy', 0) for m in cls_models]
        ax.bar(cls_models, acc_values, color='orange', alpha=0.7)
        ax.set_title('분류 모델 - 정확도')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=45)
        
        # F1 Score 비교
        ax = axes[1, 1]
        f1_values = [state['models_metrics'][m].get('f1', 0) for m in cls_models]
        ax.bar(cls_models, f1_values, color='red', alpha=0.7)
        ax.set_title('분류 모델 - F1 Score')
        ax.set_ylabel('F1')
        ax.tick_params(axis='x', rotation=45)
    
    # 전체 진행 상황
    ax = axes[1, 2]
    completed = len(state['completed_models'])
    remaining = 11 - completed
    ax.pie([completed, remaining], labels=['완료', '남음'], 
           colors=['green', 'gray'], autopct='%1.1f%%')
    ax.set_title('전체 진행 상황')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_results.png'), dpi=150)
    plt.close()
    print(f"✅ 시각화 저장: {MODEL_DIR}/training_results.png")

# ======================== 자동 재시작 기능 통합 ========================
def main_with_auto_restart():
    """자동 재시작 기능이 통합된 메인 함수"""
    max_retries = 10
    retry_count = 0
    
    print("\n" + "="*70)
    print(" 🚀 반도체 물류 예측 시스템 v1.0")
    print(" 280분 데이터 → 10분 후 예측")
    print(" 체크포인트 + 자동 재시작 시스템")
    print("="*70)
    
    # 시스템 정보 확인
    check_system_info()
    
    # 사용 모델 정보 표시
    print("\n" + "="*70)
    print(" 📚 사용 모델 정보")
    print("="*70)
    print("\n  📈 회귀 모델 (6개) - TOTALCNT 값 예측:")
    print("    1. RandomForest       - 앙상블 트리 (300개)")
    print("    2. ExtraTrees        - 극단 랜덤 트리 (300개)")
    print("    3. GradientBoosting  - 부스팅 (300 estimators)")
    print("    4. MLP               - 신경망 (200-100-50 hidden layers)")
    print("    5. Ridge             - L2 정규화 선형 회귀")
    print("    6. ElasticNet        - L1+L2 정규화")
    
    print("\n  📊 분류 모델 (5개) - 3구간 분류:")
    print("    1. RandomForest_Cls      - 앙상블 분류기")
    print("    2. ExtraTrees_Cls       - 극단 랜덤 트리 분류기")
    print("    3. GradientBoosting_Cls - 부스팅 분류기")
    print("    4. MLP_Cls              - 신경망 분류기")
    print("    5. LogisticRegression   - 로지스틱 회귀")
    
    print("\n  💡 모든 Sklearn 모델은 CPU 2코어만 사용합니다 (n_jobs=2)")
    print("  💡 XGBoost는 GPU 있으면 GPU, 없으면 CPU 2코어 사용")
    print("="*70)
    
    print("\n💡 Ctrl+C로 언제든 중단 가능 (자동 저장됩니다)")
    print("💡 오류 발생 시 자동으로 재시작합니다 (최대 10회)")
    print("-"*70)
    
    while retry_count < max_retries:
        try:
            print(f"\n📊 학습 시작 (시도 {retry_count + 1}/{max_retries})")
            print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 학습 실행
            train_with_checkpoint()
            
            # 성공적으로 완료
            print("\n✅ 학습이 성공적으로 완료되었습니다!")
            
            # 시각화 생성
            checkpoint_manager = CheckpointManager()
            state = checkpoint_manager.load_checkpoint()
            if state:
                create_visualizations(state)
            
            break
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 사용자가 중단했습니다.")
            print("체크포인트가 저장되었으므로 나중에 이어서 학습할 수 있습니다.")
            break
            
        except Exception as e:
            retry_count += 1
            print(f"\n❌ 오류 발생: {str(e)}")
            
            if retry_count < max_retries:
                print(f"⏳ 5초 후 자동으로 재시작합니다... (남은 시도: {max_retries - retry_count})")
                time.sleep(5)
            else:
                print(f"\n❌ 최대 재시도 횟수({max_retries})에 도달했습니다.")
                print("수동으로 문제를 확인해주세요.")
                traceback.print_exc()
                break
    
    print("\n" + "="*70)
    print(" 프로그램 종료")
    print("="*70)

# ======================== 실행 ========================
if __name__ == "__main__":
    # 데이터 파일 확인
    required_files = ['uu.csv', 'uu2.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n❌ 필요한 파일이 없습니다: {missing_files}")
        print("uu.csv와 uu2.csv 파일이 필요합니다.")
        sys.exit(1)
    
    # 자동 재시작 모드로 실행
    main_with_auto_restart()