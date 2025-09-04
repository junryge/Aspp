# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import joblib
from tqdm import tqdm
import os

# ======================================================================================
# [ CONFIG ] : 2개 모델 및 데이터 경로 설정 (⚠️ 사용 전 반드시 확인 및 수정!)
# ======================================================================================
# --- 공통 설정 ---
# 1. 평가할 데이터 파일 경로
FILE_PATH = 'DATA/20250801_20250831.CSV'

# 2. 학습 데이터에 사용했던 스케일러 파일 경로 (두 모델이 동일한 스케일러를 사용했다고 가정)
SCALER_PATH = 'models/scaler.pkl'

# 3. 예측 결과가 저장될 파일 경로
OUTPUT_PATH = 'results/comparison_predictions_202508.csv'

# --- 모델 1 (V3) 설정 ---
MODEL_1_NAME = 'V3'
MODEL_1_PATH = 'models/v3_model.pth' # ⚠️ V3 모델 실제 경로로 수정

# --- 모델 2 (PINN) 설정 ---
MODEL_2_NAME = 'PatchTST_PINN'
MODEL_2_PATH = 'models/patchtst_pinn_model.pth' # ⚠️ PINN 모델 실제 경로로 수정

# --- 데이터 입출력 설정 ---
LOOKBACK_WINDOW = 20  # 과거 20분
FORECAST_HORIZON = 10 # 10분 후 예측

# --- 피처 컬럼 설정 (두 모델이 동일한 피처를 사용했다고 가정) ---
FEATURE_COLUMNS = [
    'CURRENT_M16A_3F_JOB_2', 'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
    'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2', 'M16A_3F_TO_M16A_6F_JOB',
    'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB',
    'M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD', 'M16A_3F_STORAGE_UTIL', 'BRIDGE_TIME'
]
TARGET_COLUMN = 'CURRENT_M16A_3F_JOB_2'
# ======================================================================================


def generate_comparison_predictions():
    """
    2개의 학습된 모델을 로드하여 평가 데이터셋에 대한 예측을 수행하고 비교 결과를 CSV로 저장합니다.
    """
    print("="*50)
    print("2개 모델 비교 평가용 예측 데이터 생성을 시작합니다.")
    print("="*50)

    # --- 1. 필수 파일 확인 ---
    if not all(map(os.path.exists, [FILE_PATH, MODEL_1_PATH, MODEL_2_PATH, SCALER_PATH])):
        print("❌ 오류: 필수 파일(데이터, 모델 2개, 스케일러)이 존재하지 않습니다. 경로를 확인해주세요.")
        return

    # --- 2. 모델 및 스케일러 로딩 ---
    try:
        print(f"🔄 모델 1 ({MODEL_1_NAME}) 로딩 중... ({MODEL_1_PATH})")
        model_1 = torch.load(MODEL_1_PATH)
        model_1.eval()
        
        print(f"🔄 모델 2 ({MODEL_2_NAME}) 로딩 중... ({MODEL_2_PATH})")
        model_2 = torch.load(MODEL_2_PATH)
        model_2.eval()
        
        print(f"🔄 스케일러 로딩 중... ({SCALER_PATH})")
        scaler = joblib.load(SCALER_PATH)
        print("✅ 모든 모델 및 스케일러 로딩 완료.")
    except Exception as e:
        print(f"❌ 오류: 모델 또는 스케일러 로딩에 실패했습니다. {e}")
        return

    # --- 3. 평가 데이터 준비 ---
    print(f"🔄 평가 데이터 로딩 및 전처리 중... ({FILE_PATH})")
    df = pd.read_csv(FILE_PATH, encoding='utf-8-sig')
    df['time'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
    df.set_index('time', inplace=True)
    
    df_features = df[FEATURE_COLUMNS]
    df_scaled = df.copy()
    df_scaled[FEATURE_COLUMNS] = scaler.transform(df_features)
    print("✅ 데이터 준비 완료.")

    # --- 4. 슬라이딩 윈도우를 이용한 예측 생성 ---
    print("🚀 2개 모델에 대한 예측 동시 생성을 시작합니다...")
    results = []
    
    num_predictions = len(df) - LOOKBACK_WINDOW - FORECAST_HORIZON + 1
    for i in tqdm(range(num_predictions), desc="예측 진행률"):
        input_start = i
        input_end = i + LOOKBACK_WINDOW
        input_data = df_scaled.iloc[input_start:input_end][FEATURE_COLUMNS]
        target_time = df.index[input_end + FORECAST_HORIZON - 1]
        
        input_tensor = torch.FloatTensor(input_data.values).unsqueeze(0)

        with torch.no_grad():
            pred_1_scaled = model_1(input_tensor).item()
            pred_2_scaled = model_2(input_tensor).item()

        results.append({
            'timestamp': target_time,
            'prediction_1_scaled': pred_1_scaled,
            'prediction_2_scaled': pred_2_scaled
        })
    print("✅ 예측 생성 완료.")

    # --- 5. 결과 데이터 후처리 및 저장 ---
    print("🔄 결과 데이터 후처리 및 저장 중...")
    result_df = pd.DataFrame(results)

    # 스케일 복원을 위한 임시 배열 생성
    target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
    
    # 모델 1 예측 복원
    temp_pred_1 = np.zeros((len(result_df), len(FEATURE_COLUMNS)))
    temp_pred_1[:, target_idx] = result_df['prediction_1_scaled']
    result_df[f'prediction_{MODEL_1_NAME}'] = scaler.inverse_transform(temp_pred_1)[:, target_idx]
    
    # 모델 2 예측 복원
    temp_pred_2 = np.zeros((len(result_df), len(FEATURE_COLUMNS)))
    temp_pred_2[:, target_idx] = result_df['prediction_2_scaled']
    result_df[f'prediction_{MODEL_2_NAME}'] = scaler.inverse_transform(temp_pred_2)[:, target_idx]

    # 실제 값 및 오차(MAE) 계산
    result_df['true_value'] = result_df['timestamp'].map(df[TARGET_COLUMN])
    result_df[f'MAE_{MODEL_1_NAME}'] = (result_df['true_value'] - result_df[f'prediction_{MODEL_1_NAME}']).abs()
    result_df[f'MAE_{MODEL_2_NAME}'] = (result_df['true_value'] - result_df[f'prediction_{MODEL_2_NAME}']).abs()
    
    # 최종 결과 컬럼 정리
    final_cols = [
        'true_value', 
        f'prediction_{MODEL_1_NAME}', 
        f'MAE_{MODEL_1_NAME}', 
        f'prediction_{MODEL_2_NAME}', 
        f'MAE_{MODEL_2_NAME}'
    ]
    final_df = result_df.set_index('timestamp')[final_cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"🎉 모든 작업이 완료되었습니다! 비교 결과가 아래 파일에 저장되었습니다.")
    print(f"   - 파일 경로: {os.path.abspath(OUTPUT_PATH)}")
    print("="*50)
    print("\n생성된 파일 상위 5개 행:")
    print(final_df.head())


if __name__ == '__main__':
    generate_comparison_predictions()
