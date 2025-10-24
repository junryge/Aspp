import requests
import pickle
import base64
import numpy as np
import pandas as pd
import xgboost as xgb
import json

class RemoteGPU:
    """원격 GPU를 로컬처럼 사용하는 클래스"""
    
    def __init__(self, server_url='http://localhost:5000'):
        self.server_url = server_url
        self._check_connection()
    
    def _check_connection(self):
        """서버 연결 확인"""
        try:
            response = requests.get(f'{self.server_url}/ping', timeout=5)
            info = response.json()
            print(f"✓ 서버 연결됨: {self.server_url}")
            print(f"  GPU: {info['gpu_name']}")
        except Exception as e:
            print(f"✗ 서버 연결 실패: {e}")
    
    def compute(self, data, operation='multiply'):
        """GPU 연산"""
        tensor_encoded = base64.b64encode(pickle.dumps(data)).decode()
        
        response = requests.post(
            f'{self.server_url}/gpu/tensor',
            json={'tensor': tensor_encoded, 'operation': operation}
        )
        
        result = response.json()
        if result['success']:
            return pickle.loads(base64.b64decode(result['result']))
        else:
            raise Exception(result['error'])
    
    def train_xgboost(self, X, y, params=None):
        """XGBoost 학습"""
        if params is None:
            params = {
                'max_depth': 6,
                'eta': 0.3,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'num_boost_round': 100
            }
        
        print("원격 GPU로 데이터 전송 중...")
        X_encoded = base64.b64encode(pickle.dumps(X)).decode()
        y_encoded = base64.b64encode(pickle.dumps(y)).decode()
        params_json = json.dumps(params)
        
        print("GPU에서 학습 중...")
        response = requests.post(
            f'{self.server_url}/xgboost/train',
            json={
                'X': X_encoded,
                'y': y_encoded,
                'params': params_json
            },
            timeout=300
        )
        
        result = response.json()
        if result['success']:
            model_bytes = pickle.loads(base64.b64decode(result['model']))
            print(f"✓ 학습 완료! 최종 RMSE: {result['train_loss'][-1]:.4f}")
            return model_bytes, result['train_loss']
        else:
            raise Exception(result['error'])
    
    def predict_xgboost(self, model, X):
        """XGBoost 예측"""
        model_encoded = base64.b64encode(pickle.dumps(model)).decode()
        X_encoded = base64.b64encode(pickle.dumps(X)).decode()
        
        response = requests.post(
            f'{self.server_url}/xgboost/predict',
            json={
                'model': model_encoded,
                'X': X_encoded
            }
        )
        
        result = response.json()
        if result['success']:
            return pickle.loads(base64.b64decode(result['predictions']))
        else:
            raise Exception(result['error'])


# 사용 예시
if __name__ == '__main__':
    # 원격 GPU 연결
    gpu = RemoteGPU('http://YOUR_SERVER_IP:5000')
    
    # 1. 간단한 GPU 연산 테스트
    print("\n[테스트 1] GPU 연산")
    data = np.random.randn(1000, 1000)
    result = gpu.compute(data, operation='square')
    print(f"결과 shape: {result.shape}")
    
    # 2. XGBoost 학습
    print("\n[테스트 2] XGBoost GPU 학습")
    X_train = np.random.randn(10000, 50)
    y_train = np.random.randn(10000)
    
    model, losses = gpu.train_xgboost(X_train, y_tra