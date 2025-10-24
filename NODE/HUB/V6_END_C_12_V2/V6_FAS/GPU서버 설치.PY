'''
#GPU 있는 서버에서 실행 GPU 서버에서 가져와서 사용 
'''

from flask import Flask, request, jsonify
import torch
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import base64
import json

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """서버 상태 확인"""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
    return jsonify({
        'status': 'online',
        'gpu_available': gpu_available,
        'gpu_name': gpu_name
    })

@app.route('/gpu/tensor', methods=['POST'])
def gpu_tensor_compute():
    """간단한 텐서 연산"""
    try:
        data = request.json
        tensor_data = pickle.loads(base64.b64decode(data['tensor']))
        operation = data.get('operation', 'multiply')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor = torch.tensor(tensor_data, dtype=torch.float32).to(device)
        
        if operation == 'multiply':
            result = tensor * 2
        elif operation == 'square':
            result = tensor ** 2
        elif operation == 'sum':
            result = tensor.sum()
        else:
            result = tensor
        
        result_np = result.cpu().numpy()
        result_encoded = base64.b64encode(pickle.dumps(result_np)).decode()
        
        return jsonify({'success': True, 'result': result_encoded})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/xgboost/train', methods=['POST'])
def xgboost_train():
    """XGBoost 학습"""
    try:
        data = request.json
        X = pickle.loads(base64.b64decode(data['X']))
        y = pickle.loads(base64.b64decode(data['y']))
        params = json.loads(data.get('params', '{}'))
        
        # GPU 파라미터 추가
        params['tree_method'] = 'gpu_hist'
        params['gpu_id'] = 0
        
        dtrain = xgb.DMatrix(X, label=y)
        
        evals = [(dtrain, 'train')]
        evals_result = {}
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('num_boost_round', 100),
            evals=evals,
            evals_result=evals_result,
            verbose_eval=False
        )
        
        # 모델 저장
        model_bytes = pickle.dumps(model)
        model_encoded = base64.b64encode(model_bytes).decode()
        
        return jsonify({
            'success': True,
            'model': model_encoded,
            'train_loss': evals_result['train'].get('rmse', [])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/xgboost/predict', methods=['POST'])
def xgboost_predict():
    """XGBoost 예측"""
    try:
        data = request.json
        model_bytes = pickle.loads(base64.b64decode(data['model']))
        X = pickle.loads(base64.b64decode(data['X']))
        
        dtest = xgb.DMatrix(X)
        predictions = model_bytes.predict(dtest)
        
        pred_encoded = base64.b64encode(pickle.dumps(predictions)).decode()
        
        return jsonify({
            'success': True,
            'predictions': pred_encoded
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("=" * 50)
    print("GPU 서버 시작 중...")
    print(f"GPU 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("서버 주소: http://0.0.0.0:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)