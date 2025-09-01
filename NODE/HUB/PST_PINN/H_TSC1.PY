import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class ScalerSaver:
    """학습에 사용된 스케일러를 저장하는 클래스"""
    
    def __init__(self):
        self.file_path = 'data/HUB_0509_TO_0730_DATA.CSV'
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.seq_len = 20
        self.pred_len = 10
        
        # 물리적 계산을 위한 컬럼 그룹
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'
        ]
        
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'
        ]
        
        # 체크포인트 디렉토리 생성
        os.makedirs('./checkpoints', exist_ok=True)
    
    def create_sequences(self, df, numeric_cols, seq_len=20, pred_len=10):
        """시퀀스 데이터 생성"""
        X_list = []
        y_list = []
        physics_list = []
        
        data = df[numeric_cols].values
        target_idx = numeric_cols.index(self.target_col)
        
        # 사용 가능한 컬럼 확인
        available_inflow = [col for col in self.inflow_cols if col in numeric_cols]
        available_outflow = [col for col in self.outflow_cols if col in numeric_cols]
        
        print(f"📊 시퀀스 생성 중...")
        print(f"  - 유입 컬럼: {len(available_inflow)}개")
        print(f"  - 유출 컬럼: {len(available_outflow)}개")
        
        for i in range(len(data) - seq_len - pred_len + 1):
            # 입력 시퀀스
            X_seq = data[i:i+seq_len]
            
            # 타겟 (10분 후)
            y_val = data[i+seq_len+pred_len-1, target_idx]
            
            # 물리 데이터 (현재 상태)
            current_state = data[i+seq_len-1]
            current_hubroom = current_state[target_idx]
            
            # 유입/유출 합계
            inflow_sum = sum([current_state[numeric_cols.index(col)] 
                            for col in available_inflow])
            outflow_sum = sum([current_state[numeric_cols.index(col)] 
                             for col in available_outflow])
            
            physics = [current_hubroom, inflow_sum, outflow_sum]
            
            X_list.append(X_seq)
            y_list.append(y_val)
            physics_list.append(physics)
        
        return np.array(X_list), np.array(y_list), np.array(physics_list)
    
    def save_scalers(self):
        """스케일러 생성 및 저장"""
        print("="*60)
        print("💾 HUBROOM 스케일러 저장 프로그램")
        print("="*60)
        
        # 1. 데이터 로드
        print("\n📂 데이터 로드 중...")
        df = pd.read_csv(self.file_path)
        print(f"✅ 데이터 로드 완료: {df.shape[0]:,} 행 × {df.shape[1]} 컬럼")
        
        # 2. 전처리
        print("\n🔧 데이터 전처리 중...")
        time_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.fillna(method='ffill').fillna(0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"✅ 전처리 완료: {len(numeric_cols)}개 숫자형 컬럼")
        
        # 3. 시퀀스 생성
        print("\n📊 시퀀스 생성 중...")
        X, y, physics_data = self.create_sequences(df, numeric_cols, self.seq_len, self.pred_len)
        
        print(f"✅ 시퀀스 생성 완료:")
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        print(f"  - physics shape: {physics_data.shape}")
        
        # 4. 데이터 분할 (학습 데이터와 동일한 방식)
        print("\n🔀 데이터 분할 중...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        physics_train = physics_data[:len(X_train)]
        
        print(f"  - Train 데이터: {len(X_train):,} samples")
        
        # 5. 스케일러 생성 및 fitting
        print("\n📏 스케일러 생성 및 fitting...")
        
        # X 스케일러
        scaler_X = StandardScaler()
        n_samples_train, seq_len, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        scaler_X.fit(X_train_reshaped)
        
        # y 스케일러
        scaler_y = StandardScaler()
        y_train_reshaped = y_train.reshape(-1, 1)
        scaler_y.fit(y_train_reshaped)
        
        # physics 스케일러
        scaler_physics = StandardScaler()
        scaler_physics.fit(physics_train)
        
        print("✅ 스케일러 fitting 완료")
        
        # 6. 스케일러 저장
        print("\n💾 스케일러 저장 중...")
        
        # scaler_X 저장
        with open('./checkpoints/scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)
        print("  - scaler_X.pkl 저장 완료")
        
        # scaler_y 저장
        with open('./checkpoints/scaler_y.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)
        print("  - scaler_y.pkl 저장 완료")
        
        # scaler_physics 저장
        with open('./checkpoints/scaler_physics.pkl', 'wb') as f:
            pickle.dump(scaler_physics, f)
        print("  - scaler_physics.pkl 저장 완료")
        
        # 7. 검증
        print("\n🔍 저장된 스케일러 검증...")
        
        # 스케일러 다시 로드
        with open('./checkpoints/scaler_X.pkl', 'rb') as f:
            loaded_scaler_X = pickle.load(f)
        with open('./checkpoints/scaler_y.pkl', 'rb') as f:
            loaded_scaler_y = pickle.load(f)
        with open('./checkpoints/scaler_physics.pkl', 'rb') as f:
            loaded_scaler_physics = pickle.load(f)
        
        # 검증 테스트
        test_X = X_train[0:1].reshape(-1, n_features)
        test_y = y_train[0:1].reshape(-1, 1)
        test_physics = physics_train[0:1]
        
        # 변환 테스트
        X_scaled = loaded_scaler_X.transform(test_X)
        y_scaled = loaded_scaler_y.transform(test_y)
        physics_scaled = loaded_scaler_physics.transform(test_physics)
        
        print("✅ 스케일러 검증 완료")
        print(f"  - X 평균: {loaded_scaler_X.mean_[:3]}... (처음 3개)")
        print(f"  - y 평균: {loaded_scaler_y.mean_[0]:.4f}")
        print(f"  - physics 평균: {loaded_scaler_physics.mean_}")
        
        print("\n✅ 모든 스케일러가 성공적으로 저장되었습니다!")
        print(f"📁 저장 위치: ./checkpoints/")
        
        return scaler_X, scaler_y, scaler_physics

def main():
    """메인 실행 함수"""
    saver = ScalerSaver()
    
    try:
        scaler_X, scaler_y, scaler_physics = saver.save_scalers()
        
        print("\n" + "="*60)
        print("📋 저장된 파일 목록:")
        print("  1. ./checkpoints/scaler_X.pkl")
        print("  2. ./checkpoints/scaler_y.pkl")
        print("  3. ./checkpoints/scaler_physics.pkl")
        print("\n이제 평가 스크립트에서 이 스케일러들을 사용할 수 있습니다!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()