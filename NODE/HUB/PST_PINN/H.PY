# -*- coding: utf-8 -*-
"""
HUBROOM 반송량 예측 시스템
Model 1: PatchTST (순수 데이터 기반)
Model 2: PatchTST + PINN (물리 제약 통합)

시나리오: 과거 20분 데이터 → 10분 후 CURRENT_M16A_3F_JOB_2 예측
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
warnings.filterwarnings('ignore')

# ===========================
# 📊 데이터 준비 및 전처리
# ===========================

class HUBROOMDataProcessor:
    """HUBROOM 데이터 전처리 클래스"""
    
    def __init__(self, file_path='data/HUB_0509_TO_0730_DATA.CSV'):
        self.file_path = file_path
        self.scaler = StandardScaler()
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # 물리적 계산을 위한 컬럼 그룹 정의
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2',
            'M16B_10F_TO_HUB_JOB'
        ]
        
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB',
            'M16A_3F_TO_3F_MLUD_JOB'
        ]
        
        self.capacity_cols = [
            'M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA',
            'M14A_3F_CNV_MAXCAPA', 'M14B_7F_LFT_MAXCAPA'
        ]
        
    def load_and_preprocess(self):
        """데이터 로드 및 전처리"""
        print("📂 데이터 로딩 중...")
        df = pd.read_csv(self.file_path)
        
        # 시간 컬럼 처리
        time_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M')
        df = df.sort_values('timestamp')
        
        # 결측치 처리
        df = df.fillna(method='ffill').fillna(0)
        
        # 숫자형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"✅ 데이터 로드 완료: {df.shape[0]} 행 × {len(numeric_cols)} 특성")
        
        return df, numeric_cols
    
    def create_sequences(self, df, numeric_cols, seq_len=20, pred_len=10):
        """
        시퀀스 데이터 생성
        - seq_len: 과거 관측 길이 (20분)
        - pred_len: 미래 예측 길이 (10분)
        """
        X, y = [], []
        X_physics = []  # 물리 계산용 데이터
        
        # 필요한 컬럼이 있는지 확인
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        for i in range(len(df) - seq_len - pred_len + 1):
            # 입력: 과거 20분
            X_seq = df[numeric_cols].iloc[i:i+seq_len].values
            X.append(X_seq)
            
            # 타겟: 10분 후 값
            if self.target_col in df.columns:
                y_val = df[self.target_col].iloc[i+seq_len+pred_len-1]
            else:
                # 타겟 컬럼이 없으면 임의 생성 (예시)
                y_val = df[numeric_cols[0]].iloc[i+seq_len+pred_len-1]
            y.append(y_val)
            
            # 물리 데이터 (유입/유출)
            physics_data = {
                'current_hubroom': df[numeric_cols[0]].iloc[i+seq_len-1],
                'inflow_sum': df[available_inflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_inflow else 0,
                'outflow_sum': df[available_outflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_outflow else 0,
            }
            X_physics.append(physics_data)
        
        return np.array(X), np.array(y), X_physics

# ===========================
# 🧠 Model 1: PatchTST
# ===========================

class PatchTST(nn.Module):
    """
    PatchTST: 패치 기반 시계열 Transformer
    - 20분을 5분씩 4개 패치로 분할
    - Multi-head Attention으로 패턴 학습
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.seq_len = config['seq_len']  # 20
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']  # 5
        self.n_patches = self.seq_len // self.patch_len  # 4
        
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
        """시계열을 패치로 분할"""
        batch_size = x.shape[0]
        
        # (batch, seq_len, features) -> (batch, n_patches, patch_len, features)
        x = x.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        
        # (batch, n_patches, features, patch_len) -> (batch, n_patches, patch_len*features)
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(batch_size, self.n_patches, -1)
        
        return x
    
    def forward(self, x):
        # 패치 생성
        x_patches = self.create_patches(x)
        
        # 패치 임베딩
        x_embed = self.patch_embedding(x_patches)
        
        # 위치 인코딩 추가
        x_embed = x_embed + self.pos_embedding
        
        # Transformer 인코더
        x_transformed = self.transformer(x_embed)
        
        # 출력 생성
        x_flat = self.flatten(x_transformed)
        output = self.output_layer(x_flat)
        
        return output.squeeze(-1)

# ===========================
# 🔬 Model 2: PatchTST + PINN
# ===========================

class PhysicsInformedLoss(nn.Module):
    """
    물리 정보 손실 함수
    - 질량 보존: HUBROOM_변화 = 유입 - 유출
    - 용량 제약: 0 ≤ HUBROOM ≤ MAX_CAPACITY
    """
    
    def __init__(self, max_capacity=200, alpha_physics=0.3, alpha_capacity=0.2):
        super().__init__()
        self.max_capacity = max_capacity
        self.alpha_physics = alpha_physics
        self.alpha_capacity = alpha_capacity
        
    def forward(self, pred, target, physics_data):
        # 기본 예측 손실 (MSE)
        mse_loss = nn.MSELoss()(pred, target)
        
        # 물리 손실 계산
        physics_loss = 0
        capacity_loss = 0
        
        if physics_data is not None:
            # 질량 보존 검증
            expected = physics_data['current'] + physics_data['inflow'] - physics_data['outflow']
            physics_violation = torch.abs(pred - expected)
            physics_loss = torch.mean(physics_violation)
            
            # 용량 제약 검증
            over_capacity = torch.relu(pred - self.max_capacity)
            under_zero = torch.relu(-pred)
            capacity_loss = torch.mean(over_capacity + under_zero)
        
        # 전체 손실
        total_loss = mse_loss + self.alpha_physics * physics_loss + self.alpha_capacity * capacity_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'physics': physics_loss.item() if isinstance(physics_loss, torch.Tensor) else 0,
            'capacity': capacity_loss.item() if isinstance(capacity_loss, torch.Tensor) else 0
        }

class PatchTST_PINN(nn.Module):
    """
    PatchTST + PINN 하이브리드 모델
    - PatchTST 백본 + 물리 제약 레이어
    """
    
    def __init__(self, config):
        super().__init__()
        
        # PatchTST 백본
        self.patchtst = PatchTST(config)
        
        # 물리 보정 레이어
        self.physics_correction = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 0-1 스케일링
        )
        
        self.max_capacity = config.get('max_capacity', 200)
        
    def forward(self, x, physics_data=None):
        # PatchTST 예측
        base_pred = self.patchtst(x)
        
        # 물리 보정 적용
        if physics_data is not None and self.training:
            # 물리 기반 예상값 계산
            expected = physics_data['current'] + physics_data['inflow'] - physics_data['outflow']
            
            # 보정 계수 계산
            correction = self.physics_correction(base_pred.unsqueeze(-1)).squeeze(-1)
            
            # 가중 평균: 데이터 기반 예측과 물리 기반 예측
            pred = base_pred * (1 - correction) + expected * correction
        else:
            pred = base_pred
        
        # 용량 제약 적용 (클리핑)
        pred = torch.clamp(pred, min=0, max=self.max_capacity)
        
        return pred

# ===========================
# 📈 학습 및 평가
# ===========================

class HUBROOMTrainer:
    """모델 학습 및 평가 클래스"""
    
    def __init__(self, model, model_name, device='cpu'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}
        
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, use_physics=False):
        """모델 학습"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        if use_physics:
            criterion = PhysicsInformedLoss()
        else:
            criterion = nn.MSELoss()
        
        print(f"\n🚀 {self.model_name} 학습 시작...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y, batch_physics in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                if use_physics:
                    # Physics-informed 학습
                    physics_tensor = {
                        'current': torch.tensor([p['current_hubroom'] for p in batch_physics]).to(self.device),
                        'inflow': torch.tensor([p['inflow_sum'] for p in batch_physics]).to(self.device),
                        'outflow': torch.tensor([p['outflow_sum'] for p in batch_physics]).to(self.device)
                    }
                    pred = self.model(batch_x, physics_tensor)
                    loss, loss_components = criterion(pred, batch_y, physics_tensor)
                else:
                    # 일반 학습
                    pred = self.model(batch_x)
                    loss = criterion(pred, batch_y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y, _ in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    pred = self.model(batch_x)
                    
                    if use_physics:
                        loss = nn.MSELoss()(pred, batch_y)
                    else:
                        loss = criterion(pred, batch_y)
                    
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    def evaluate(self, test_loader):
        """모델 평가"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y, _ in test_loader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x).cpu().numpy()
                predictions.extend(pred)
                actuals.extend(batch_y.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 평가 메트릭
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
        
        print(f"\n📊 {self.model_name} 평가 결과:")
        print(f"  - MAE: {mae:.2f}")
        print(f"  - RMSE: {rmse:.2f}")
        print(f"  - MAPE: {mape:.2f}%")
        
        return predictions, actuals, {'mae': mae, 'rmse': rmse, 'mape': mape}

# ===========================
# 🎯 메인 실행 시나리오
# ===========================

def main_scenario():
    """
    시나리오: 과거 20분 데이터로 10분 후 예측
    Model 1: PatchTST (데이터 기반)
    Model 2: PatchTST + PINN (물리 제약 통합)
    """
    
    print("=" * 80)
    print("🏭 HUBROOM 반송량 예측 시스템")
    print("시나리오: 과거 20분 → 10분 후 예측")
    print("=" * 80)
    
    # 1. 데이터 준비
    processor = HUBROOMDataProcessor()
    df, numeric_cols = processor.load_and_preprocess()
    
    # 시뮬레이션용 데이터 생성 (실제 데이터가 없을 경우)
    if len(df) < 100:
        print("⚠️ 데이터 부족, 시뮬레이션 데이터 생성...")
        # 시뮬레이션 데이터 생성
        n_samples = 1000
        n_features = 38
        
        # 가상의 시계열 데이터 생성
        data = np.random.randn(n_samples, n_features) * 10 + 100
        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
        numeric_cols = df.columns.tolist()
    
    # 2. 시퀀스 생성
    X, y, physics_data = processor.create_sequences(df, numeric_cols, seq_len=20, pred_len=10)
    
    print(f"\n📦 시퀀스 데이터 생성 완료:")
    print(f"  - X shape: {X.shape} (샘플, 시퀀스, 특성)")
    print(f"  - y shape: {y.shape} (샘플,)")
    
    # 3. 데이터 분할
    X_train, X_temp, y_train, y_temp, physics_train, physics_temp = train_test_split(
        X, y, physics_data, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test, physics_val, physics_test = train_test_split(
        X_temp, y_temp, physics_temp, test_size=0.5, random_state=42
    )
    
    # 4. 데이터 정규화
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Reshape for scaling
    n_samples_train, seq_len, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(n_samples_train, seq_len, n_features)
    
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], seq_len, n_features)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], seq_len, n_features)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # 5. 데이터로더 생성
    class HUBROOMDataset(Dataset):
        def __init__(self, X, y, physics_data=None):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            self.physics_data = physics_data if physics_data else [None] * len(X)
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.physics_data[idx]
    
    train_dataset = HUBROOMDataset(X_train_scaled, y_train_scaled, physics_train)
    val_dataset = HUBROOMDataset(X_val_scaled, y_val_scaled, physics_val)
    test_dataset = HUBROOMDataset(X_test_scaled, y_test_scaled, physics_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 6. 모델 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'seq_len': 20,
        'n_features': n_features,
        'patch_len': 5,
        'd_model': 128,
        'n_heads': 8,
        'd_ff': 256,
        'n_layers': 3,
        'dropout': 0.1,
        'max_capacity': 200
    }
    
    # 7. Model 1: PatchTST 학습
    print("\n" + "="*60)
    print("🔷 Model 1: PatchTST (순수 데이터 기반)")
    print("="*60)
    
    model1 = PatchTST(config)
    trainer1 = HUBROOMTrainer(model1, "PatchTST", device)
    trainer1.train(train_loader, val_loader, epochs=30, lr=0.001, use_physics=False)
    
    # 평가
    pred1, actual1, metrics1 = trainer1.evaluate(test_loader)
    
    # 8. Model 2: PatchTST + PINN 학습
    print("\n" + "="*60)
    print("🔶 Model 2: PatchTST + PINN (물리 제약 통합)")
    print("="*60)
    
    model2 = PatchTST_PINN(config)
    trainer2 = HUBROOMTrainer(model2, "PatchTST+PINN", device)
    trainer2.train(train_loader, val_loader, epochs=30, lr=0.001, use_physics=True)
    
    # 평가
    pred2, actual2, metrics2 = trainer2.evaluate(test_loader)
    
    # 9. 결과 비교
    print("\n" + "="*60)
    print("📊 모델 성능 비교")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Model': ['PatchTST', 'PatchTST+PINN'],
        'MAE': [metrics1['mae'], metrics2['mae']],
        'RMSE': [metrics1['rmse'], metrics2['rmse']],
        'MAPE(%)': [metrics1['mape'], metrics2['mape']]
    })
    
    print(comparison_df.to_string(index=False))
    
    # 10. 예측 시각화
    plt.figure(figsize=(15, 5))
    
    # 실제 vs 예측 (처음 100개만)
    n_show = min(100, len(actual1))
    
    plt.subplot(1, 3, 1)
    plt.plot(actual1[:n_show], label='실제값', alpha=0.7)
    plt.plot(pred1[:n_show], label='PatchTST', alpha=0.7)
    plt.title('Model 1: PatchTST')
    plt.legend()
    plt.xlabel('시간')
    plt.ylabel('HUBROOM JOB 수')
    
    plt.subplot(1, 3, 2)
    plt.plot(actual2[:n_show], label='실제값', alpha=0.7)
    plt.plot(pred2[:n_show], label='PatchTST+PINN', alpha=0.7)
    plt.title('Model 2: PatchTST+PINN')
    plt.legend()
    plt.xlabel('시간')
    plt.ylabel('HUBROOM JOB 수')
    
    plt.subplot(1, 3, 3)
    # 학습 곡선
    epochs_range = range(1, len(trainer1.history['train_loss']) + 1)
    plt.plot(epochs_range, trainer1.history['val_loss'], label='PatchTST Val Loss')
    plt.plot(epochs_range, trainer2.history['val_loss'], label='PatchTST+PINN Val Loss')
    plt.title('검증 손실 비교')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hubroom_prediction_results.png', dpi=100)
    plt.show()
    
    # 11. 물리 제약 검증
    print("\n" + "="*60)
    print("🔬 물리 제약 검증")
    print("="*60)
    
    # 음수 예측 체크
    negative_pred1 = np.sum(pred1 < 0)
    negative_pred2 = np.sum(pred2 < 0)
    
    # 용량 초과 체크 (가정: 최대 200)
    over_capacity1 = np.sum(pred1 > 200)
    over_capacity2 = np.sum(pred2 > 200)
    
    print(f"음수 예측:")
    print(f"  - PatchTST: {negative_pred1}개")
    print(f"  - PatchTST+PINN: {negative_pred2}개")
    print(f"\n용량 초과 예측 (>200):")
    print(f"  - PatchTST: {over_capacity1}개")
    print(f"  - PatchTST+PINN: {over_capacity2}개")
    
    # 12. 실시간 예측 시뮬레이션
    print("\n" + "="*60)
    print("🎯 실시간 예측 시뮬레이션")
    print("="*60)
    
    # 마지막 20분 데이터로 10분 후 예측
    if len(X_test) > 0:
        last_20min = X_test_scaled[-1:, :, :]
        last_20min_tensor = torch.FloatTensor(last_20min).to(device)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            pred_model1 = model1(last_20min_tensor).cpu().numpy()[0]
            pred_model2 = model2(last_20min_tensor).cpu().numpy()[0]
        
        # 역변환
        pred_model1_original = scaler_y.inverse_transform([[pred_model1]])[0][0]
        pred_model2_original = scaler_y.inverse_transform([[pred_model2]])[0][0]
        actual_original = scaler_y.inverse_transform([[y_test_scaled[-1]]])[0][0]
        
        print(f"현재 시점 HUBROOM 상태:")
        print(f"  - 실제값: {actual_original:.0f}")
        print(f"\n10분 후 예측:")
        print(f"  - PatchTST: {pred_model1_original:.0f}")
        print(f"  - PatchTST+PINN: {pred_model2_original:.0f}")
        print(f"  - 차이: {abs(pred_model1_original - pred_model2_original):.0f}")
    
    print("\n" + "="*60)
    print("✅ 시나리오 완료!")
    print("="*60)
    
    return model1, model2, comparison_df

# 실행
if __name__ == "__main__":
    model1, model2, results = main_scenario()