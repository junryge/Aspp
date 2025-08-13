"""
모델 평가 빠른 실행 가이드
========================
"""

import os

# 1. 필요한 라이브러리 설치
# pip install tensorflow==2.18.0 pandas numpy scikit-learn matplotlib seaborn joblib

# 2. 경로 설정 (실제 경로로 변경 필요)
BASE_PATH = r"D:\하이닉스\6.연구_항목\CODE\202508051차_POC구축\앙상블_하이브리드200회학습_90_학습\모델별_성능평가"

# 앙상블 신형 모델 경로
ENSEMBLE_MODELS = {
    'lstm': os.path.join(BASE_PATH, "AL", "model_2", "lee_lstm_final_hybrid.keras"),
    'gru': os.path.join(BASE_PATH, "AL", "model_2", "lee_gru_final_hybrid.keras"),
    'rnn': os.path.join(BASE_PATH, "AL", "model_2", "lee_rnn_final_hybrid.keras"),
    'bi_lstm': os.path.join(BASE_PATH, "AL", "model_2", "lee_bi_lstm_final_hybrid.keras")
}

# LSTM 구형 모델 경로
OLD_MODEL = os.path.join(BASE_PATH, "LSTM", "model_1", "Model_s30f10_0724_2079936.keras")

# 스케일러 경로
SCALER_NEW = os.path.join(BASE_PATH, "AL", "scaler_2", "standard_scaler_hybrid.pkl")
SCALER_OLD = os.path.join(BASE_PATH, "LSTM", "scaler_1", "StdScaler_s30f10_0724_2079936.save")

# 테스트 데이터 경로
TEST_DATA = "data/20250731_to_20250806.csv"

# 3. 간단한 평가 실행 함수
def quick_evaluate():
    """빠른 모델 평가 실행"""
    from semiconductor_model_evaluation import ModelComparator
    
    # 비교기 초기화
    comparator = ModelComparator()
    
    # 모델 로드
    model_paths_new = [ENSEMBLE_MODELS['lstm'], ENSEMBLE_MODELS['gru'], ENSEMBLE_MODELS['rnn'], ENSEMBLE_MODELS['bi_lstm']]
    comparator.load_models(model_paths_new, OLD_MODEL, SCALER_NEW, SCALER_OLD)
    
    # 데이터 준비
    data = comparator.prepare_data(TEST_DATA)
    
    # 예측 수행
    print("\n예측 수행 중...")
    
    # 앙상블 신형
    X_new, y_new, times, actuals = comparator.create_sequences(data, model_type='new')
    ensemble_pred_scaled, _ = comparator.predict_ensemble_new(X_new)
    ensemble_pred = comparator.inverse_scale(ensemble_pred_scaled, comparator.scaler_new)
    
    # LSTM 구형
    X_old, y_old, _, _ = comparator.create_sequences(data, model_type='old')
    old_pred_scaled = comparator.predict_old(X_old)
    old_pred = comparator.inverse_scale(old_pred_scaled, comparator.scaler_old)
    
    # 간단한 성능 비교
    from sklearn.metrics import mean_absolute_error
    
    mae_new = mean_absolute_error(actuals, ensemble_pred)
    mae_old = mean_absolute_error(actuals, old_pred)
    
    print("\n" + "="*50)
    print("빠른 성능 비교 결과")
    print("="*50)
    print(f"앙상블 신형 MAE: {mae_new:.2f}")
    print(f"LSTM 구형 MAE: {mae_old:.2f}")
    print(f"성능 개선: {((mae_old - mae_new) / mae_old * 100):.1f}%")
    
    # 최근 10개 예측 결과 출력
    print("\n최근 10개 예측 결과:")
    print(f"{'시간':^20} | {'실제값':^10} | {'앙상블 신형':^12} | {'LSTM 구형':^12}")
    print("-" * 60)
    
    for i in range(-10, 0):
        print(f"{str(times[i]):^20} | {actuals[i]:^10.0f} | {ensemble_pred[i]:^12.0f} | {old_pred[i]:^12.0f}")

# 4. 실시간 예측 함수 (최신 데이터로 예측)
def predict_next_10min(current_data_path):
    """최신 100분 데이터로 다음 10분 예측"""
    from semiconductor_model_evaluation import ModelComparator
    import pandas as pd
    
    comparator = ModelComparator()
    
    # 모델 로드
    model_paths_new = [ENSEMBLE_MODELS['lstm'], ENSEMBLE_MODELS['gru'], ENSEMBLE_MODELS['rnn'], ENSEMBLE_MODELS['bi_lstm']]
    comparator.load_models(model_paths_new, OLD_MODEL, SCALER_NEW, SCALER_OLD)
    
    # 데이터 준비
    data = comparator.prepare_data(current_data_path)
    
    # 마지막 30개 데이터로 예측
    if len(data) >= 30:
        # 앙상블 신형 예측
        X_new, _, _, _ = comparator.create_sequences(data.tail(31), model_type='new')
        if len(X_new) > 0:
            pred_scaled, _ = comparator.predict_ensemble_new(X_new[-1:])
            pred_new = comparator.inverse_scale(pred_scaled, comparator.scaler_new)[0]
            
            # LSTM 구형 예측
            X_old, _, _, _ = comparator.create_sequences(data.tail(31), model_type='old')
            pred_scaled_old = comparator.predict_old(X_old[-1:])
            pred_old = comparator.inverse_scale(pred_scaled_old, comparator.scaler_old)[0]
            
            current_time = data.index[-1]
            predict_time = current_time + pd.Timedelta(minutes=10)
            
            print("\n" + "="*50)
            print("10분 후 예측 결과")
            print("="*50)
            print(f"현재 시간: {current_time}")
            print(f"예측 시간: {predict_time}")
            print(f"현재 물류량: {data['TOTALCNT'].iloc[-1]:.0f}")
            print(f"\n앙상블 신형 예측: {pred_new:.0f}")
            print(f"LSTM 구형 예측: {pred_old:.0f}")
            print(f"예측 차이: {abs(pred_new - pred_old):.0f}")
            
            return {
                'time': predict_time,
                'ensemble_new': pred_new,
                'lstm_old': pred_old
            }
    
    return None

if __name__ == "__main__":
    # 빠른 평가 실행
    quick_evaluate()
    
    # 실시간 예측 (옵션)
    # result = predict_next_10min(TEST_DATA)
    # if result:
    #     print(f"\n10분 후 예측: {result}")