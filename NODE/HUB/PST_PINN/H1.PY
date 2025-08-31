# -*- coding: utf-8 -*-
"""
개선된 HUBROOM 반송량 예측 시스템
- 진행률 표시 기능 추가
- 체크포인트 저장/복원 기능
- 중단 후 재개 가능
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import os
import json
from tqdm import tqdm
import time
import signal
import sys

warnings.filterwarnings('ignore')

# ===========================
# 🔄 체크포인트 관리
# ===========================

class CheckpointManager:
    """학습 중단/재개를 위한 체크포인트 관리"""
    
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.interrupted = False
        
        # Ctrl+C 인터럽트 핸들러
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        print('\n\n⚠️ 학습 중단 감지! 체크포인트 저장 중...')
        self.interrupted = True
    
    def save_checkpoint(self, state, filename):
        """체크포인트 저장"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)
        print(f"💾 체크포인트 저장: {filepath}")
    
    def load_checkpoint(self, filename):
        """체크포인트 로드"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            print(f"📂 체크포인트 로드: {filepath}")
            return torch.load(filepath)
        return None
    
    def save_data_progress(self, data_dict, filename='data_progress.pkl'):
        """데이터 처리 진행 상황 저장"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"💾 데이터 진행 상황 저장: {filepath}")
    
    def load_data_progress(self, filename='data_progress.pkl'):
        """데이터 처리 진행 상황 로드"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            print(f"📂 데이터 진행 상황 로드: {filepath}")
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None

# ===========================
# 📊 개선된 데이터 처리
# ===========================

class ImprovedHUBROOMProcessor:
    """진행률 표시가 추가된 데이터 처리 클래스"""
    
    def __init__(self, file_path='data/HUB_0509_TO_0730_DATA.CSV'):
        self.file_path = file_path
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.checkpoint_manager = CheckpointManager()
        
        # 물리적 계산을 위한 컬럼 그룹
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'
        ]
        
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'
        ]
    
    def analyze_data(self, df):
        """데이터 특성 상세 분석"""
        print("\n" + "="*60)
        print("📈 데이터 특성 분석")
        print("="*60)
        
        total_steps = 8
        current_step = 0
        
        # 1. 기본 정보
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 기본 정보 분석... ({current_step/total_steps*100:.0f}%)")
        print(f"  - 전체 행 수: {df.shape[0]:,}")
        print(f"  - 전체 컬럼 수: {df.shape[1]}")
        print(f"  - 메모리 사용량: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # 2. 결측치 분석
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 결측치 분석... ({current_step/total_steps*100:.0f}%)")
        missing_ratio = (df.isnull().sum() / len(df)) * 100
        missing_cols = missing_ratio[missing_ratio > 0]
        if len(missing_cols) > 0:
            print(f"  - 결측치 있는 컬럼: {len(missing_cols)}개")
            for col, ratio in missing_cols.head(5).items():
                print(f"    • {col}: {ratio:.2f}%")
        else:
            print("  - ✅ 결측치 없음")
        
        # 3. 타겟 변수 분석
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 타겟 변수 분석... ({current_step/total_steps*100:.0f}%)")
        if self.target_col in df.columns:
            target = df[self.target_col]
            print(f"  - 타겟: {self.target_col}")
            print(f"  - 평균: {target.mean():.2f}")
            print(f"  - 표준편차: {target.std():.2f}")
            print(f"  - 최소/최대: {target.min():.0f} / {target.max():.0f}")
            print(f"  - 분포: Q1={target.quantile(0.25):.0f}, Q2={target.median():.0f}, Q3={target.quantile(0.75):.0f}")
        else:
            print(f"  - ⚠️ 타겟 컬럼 '{self.target_col}' 없음")
        
        # 4. 유입/유출 분석
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 유입/유출 균형 분석... ({current_step/total_steps*100:.0f}%)")
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        if available_inflow and available_outflow:
            total_inflow = df[available_inflow].sum().sum()
            total_outflow = df[available_outflow].sum().sum()
            balance = total_inflow - total_outflow
            print(f"  - 총 유입: {total_inflow:,.0f}")
            print(f"  - 총 유출: {total_outflow:,.0f}")
            print(f"  - 순 변화: {balance:+,.0f} ({balance/total_inflow*100:+.1f}%)")
        
        # 5. 시간 패턴 분석
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 시간 패턴 분석... ({current_step/total_steps*100:.0f}%)")
        if 'timestamp' in df.columns or df.columns[0].lower().find('time') >= 0:
            time_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
            print(f"  - 시작: {df[time_col].iloc[0]}")
            print(f"  - 종료: {df[time_col].iloc[-1]}")
            print(f"  - 데이터 간격: 분 단위 (추정)")
        
        # 6. 상관관계 분석
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 주요 상관관계 분석... ({current_step/total_steps*100:.0f}%)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if self.target_col in numeric_cols and len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()[self.target_col].sort_values(ascending=False)
            print(f"  타겟과 상관관계 높은 TOP 5:")
            for col, corr in correlations[1:6].items():
                print(f"    • {col}: {corr:.3f}")
        
        # 7. 이상치 탐지
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 이상치 탐지... ({current_step/total_steps*100:.0f}%)")
        for col in numeric_cols[:3]:  # 처음 3개 컬럼만
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
            if outliers > 0:
                print(f"  - {col}: {outliers}개 ({outliers/len(df)*100:.1f}%)")
        
        # 8. 데이터 품질 점수
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 데이터 품질 평가... ({current_step/total_steps*100:.0f}%)")
        quality_score = 100
        quality_score -= missing_ratio.mean() * 10  # 결측치 페널티
        quality_score -= min(20, outliers/len(df)*100*5)  # 이상치 페널티
        print(f"  - 데이터 품질 점수: {quality_score:.1f}/100")
        
        if quality_score >= 80:
            print("  - ✅ 우수한 데이터 품질")
        elif quality_score >= 60:
            print("  - ⚠️ 보통 데이터 품질 (전처리 권장)")
        else:
            print("  - ❌ 낮은 데이터 품질 (전처리 필수)")
        
        print("\n" + "="*60)
        
        return {
            'total_rows': df.shape[0],
            'total_cols': df.shape[1],
            'missing_ratio': missing_ratio.mean(),
            'quality_score': quality_score
        }
    
    def create_sequences_with_progress(self, df, numeric_cols, seq_len=20, pred_len=10, resume_from=0):
        """진행률 표시와 함께 시퀀스 생성"""
        
        print("\n📦 시퀀스 데이터 생성 중...")
        
        X, y, X_physics = [], [], []
        
        # 체크포인트에서 이전 진행 상황 로드
        saved_progress = self.checkpoint_manager.load_data_progress()
        if saved_progress and resume_from > 0:
            print(f"♻️ 이전 진행 상황 복원: {resume_from}개 완료")
            X = saved_progress['X']
            y = saved_progress['y']
            X_physics = saved_progress['X_physics']
        
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        total_sequences = len(df) - seq_len - pred_len + 1
        
        # tqdm으로 진행률 표시
        with tqdm(total=total_sequences, desc="시퀀스 생성", initial=resume_from) as pbar:
            for i in range(resume_from, total_sequences):
                # 입력: 과거 20분
                X_seq = df[numeric_cols].iloc[i:i+seq_len].values
                X.append(X_seq)
                
                # 타겟: 10분 후
                if self.target_col in df.columns:
                    y_val = df[self.target_col].iloc[i+seq_len+pred_len-1]
                else:
                    y_val = df[numeric_cols[0]].iloc[i+seq_len+pred_len-1]
                y.append(y_val)
                
                # 물리 데이터
                physics_data = {
                    'current_hubroom': df[numeric_cols[0]].iloc[i+seq_len-1],
                    'inflow_sum': df[available_inflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_inflow else 0,
                    'outflow_sum': df[available_outflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_outflow else 0,
                }
                X_physics.append(physics_data)
                
                pbar.update(1)
                
                # 100개마다 체크포인트 저장
                if (i + 1) % 100 == 0:
                    self.checkpoint_manager.save_data_progress({
                        'X': X, 'y': y, 'X_physics': X_physics,
                        'progress': i + 1
                    })
                
                # 인터럽트 확인
                if self.checkpoint_manager.interrupted:
                    print(f"\n💾 중단 지점 저장: {i+1}/{total_sequences}")
                    self.checkpoint_manager.save_data_progress({
                        'X': X, 'y': y, 'X_physics': X_physics,
                        'progress': i + 1
                    })
                    sys.exit(0)
        
        print(f"✅ 시퀀스 생성 완료: {len(X)}개")
        
        return np.array(X), np.array(y), X_physics

# ===========================
# 🧠 모델 정의 (기존과 동일)
# ===========================

class PatchTST(nn.Module):
    """PatchTST: 패치 기반 시계열 Transformer"""
    
    def __init__(self, config):
        super().__init__()
        
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.d_ff = config['d_ff']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        
        # 패치 임베딩
        self.patch_embedding = nn.Linear(
            self.patch_len * self.n_features, 
            self.d_model
        )
        
        # 위치 인코딩
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, self.d_model)
        )
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.n_layers
        )
        
        # 출력 레이어
        self.flatten = nn.Flatten()
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model * self.n_patches, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def create_patches(self, x):
        batch_size = x.shape[0]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(batch_size, self.n_patches, -1)
        return x
    
    def forward(self, x):
        x_patches = self.create_patches(x)
        x_embed = self.patch_embedding(x_patches)
        x_embed = x_embed + self.pos_embedding
        x_transformed = self.transformer(x_embed)
        x_flat = self.flatten(x_transformed)
        output = self.output_layer(x_flat)
        return output.squeeze(-1)

# ===========================
# 📈 개선된 학습 클래스
# ===========================

class ImprovedTrainer:
    """체크포인트와 진행률 표시가 추가된 학습 클래스"""
    
    def __init__(self, model, model_name, device='cpu'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}
        self.checkpoint_manager = CheckpointManager()
        self.best_val_loss = float('inf')
        
    def train_with_checkpoint(self, train_loader, val_loader, epochs=50, lr=0.001, resume=False):
        """체크포인트 지원 학습"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        start_epoch = 0
        
        # 체크포인트 복원
        if resume:
            checkpoint = self.checkpoint_manager.load_checkpoint(f'{self.model_name}_checkpoint.pth')
            if checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                self.history = checkpoint['history']
                self.best_val_loss = checkpoint['best_val_loss']
                print(f"♻️ 학습 재개: Epoch {start_epoch}부터 시작")
        
        print(f"\n🚀 {self.model_name} 학습 시작...")
        print("="*60)
        
        for epoch in range(start_epoch, epochs):
            # Training
            self.model.train()
            train_losses = []
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_x, batch_y, _ in train_pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]')
                for batch_x, batch_y, _ in val_pbar:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    pred = self.model(batch_x)
                    loss = criterion(pred, batch_y)
                    val_losses.append(loss.item())
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            
            # 결과 출력
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Best: {self.best_val_loss:.4f}")
            
            # 최고 모델 저장
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.checkpoint_manager.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'history': self.history
                }, f'{self.model_name}_best.pth')
                print(f"  🏆 새로운 최고 성능 달성!")
            
            # 주기적 체크포인트
            if (epoch + 1) % 5 == 0:
                self.checkpoint_manager.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'history': self.history
                }, f'{self.model_name}_checkpoint.pth')
            
            # 인터럽트 확인
            if self.checkpoint_manager.interrupted:
                print(f"\n💾 학습 중단, 체크포인트 저장...")
                self.checkpoint_manager.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'history': self.history
                }, f'{self.model_name}_interrupted.pth')
                sys.exit(0)
        
        print("✅ 학습 완료!")
        return self.history

# ===========================
# 🎯 메인 실행
# ===========================

def main_with_progress():
    """진행률 표시와 체크포인트가 추가된 메인 함수"""
    
    print("=" * 80)
    print("🏭 HUBROOM 반송량 예측 시스템 (개선판)")
    print("💡 특징: 진행률 표시, 중단/재개 가능")
    print("=" * 80)
    
    # 데이터 처리
    processor = ImprovedHUBROOMProcessor()
    
    try:
        # 1. 데이터 로드
        print("\n📂 데이터 로드 중...")
        df = pd.read_csv(processor.file_path)
        print(f"✅ 데이터 로드 완료: {df.shape[0]:,} 행 × {df.shape[1]} 컬럼")
        
        # 2. 데이터 특성 분석
        data_stats = processor.analyze_data(df)
        
        # 3. 전처리
        print("\n🔧 데이터 전처리 중...")
        time_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.fillna(method='ffill').fillna(0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"✅ 전처리 완료: {len(numeric_cols)}개 숫자형 컬럼")
        
        # 4. 시퀀스 생성 (체크포인트 확인)
        saved_progress = processor.checkpoint_manager.load_data_progress()
        resume_from = saved_progress['progress'] if saved_progress else 0
        
        X, y, physics_data = processor.create_sequences_with_progress(
            df, numeric_cols, seq_len=20, pred_len=10, resume_from=resume_from
        )
        
        print(f"\n📊 데이터 준비 완료:")
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        
        # 5. 데이터 분할
        print("\n🔀 데이터 분할 중...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        print(f"  - Train: {len(X_train):,} samples")
        print(f"  - Valid: {len(X_val):,} samples")
        print(f"  - Test: {len(X_test):,} samples")
        
        # 6. 정규화
        print("\n📏 데이터 정규화 중...")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        n_samples_train, seq_len, n_features = X_train.shape
        X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples_train, seq_len, n_features)
        X_val_scaled = scaler_X.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], seq_len, n_features)
        X_test_scaled = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], seq_len, n_features)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # 7. 데이터로더 생성
        class HUBROOMDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.FloatTensor(X)
                self.y = torch.FloatTensor(y)
                
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx], None
        
        train_dataset = HUBROOMDataset(X_train_scaled, y_train_scaled)
        val_dataset = HUBROOMDataset(X_val_scaled, y_val_scaled)
        test_dataset = HUBROOMDataset(X_test_scaled, y_test_scaled)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 8. 모델 학습
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️ 디바이스: {device}")
        
        config = {
            'seq_len': 20,
            'n_features': n_features,
            'patch_len': 5,
            'd_model': 128,
            'n_heads': 8,
            'd_ff': 256,
            'n_layers': 3,
            'dropout': 0.1
        }
        
        # PatchTST 모델 학습
        model = PatchTST(config)
        trainer = ImprovedTrainer(model, "PatchTST", device)
        
        # 이전 학습 이어서 하기
        resume_training = input("\n이전 학습을 이어서 하시겠습니까? (y/n): ").lower() == 'y'
        
        history = trainer.train_with_checkpoint(
            train_loader, val_loader, 
            epochs=30, lr=0.001, 
            resume=resume_training
        )
        
        print("\n✅ 모든 작업 완료!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자 중단 감지")
        print("💾 진행 상황이 저장되었습니다. 다시 실행하면 이어서 진행됩니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("💾 진행 상황이 저장되었습니다.")

if __name__ == "__main__":
    main_with_progress()