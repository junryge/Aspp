"""
반도체 물류 예측 평가 시스템
================================
100분간의 실제 데이터를 사용하여 10분 후를 예측하고
실제값과 비교하여 모델 성능을 평가합니다.

CPU 기반 LSTM, RNN, GRU 하이브리드 모델 사용
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
from datetime import datetime, timedelta
import joblib
import warnings
import platform

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 고정
tf.random.set_seed(2079936)
np.random.seed(2079936)

# 한글 폰트 설정
def set_korean_font():
    """운영체제별 한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Windows':
        font_paths = ['C:/Windows/Fonts/malgun.ttf', 'C:/Windows/Fonts/NanumGothic.ttf']
        font_family = 'Malgun Gothic'
    elif system == 'Darwin':
        font_paths = ['/System/Library/Fonts/Supplemental/AppleGothic.ttf']
        font_family = 'AppleGothic'
    else:
        font_paths = ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
        font_family = 'NanumGothic'
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            plt.rcParams['font.family'] = font_family
            plt.rcParams['axes.unicode_minus'] = False
            return True
    return False

set_korean_font()

class LogisticsPredictionEvaluator:
    """100분 데이터로 10분 예측을 평가하는 클래스"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.seq_length = 30  # 시퀀스 길이
        self.future_minutes = 10  # 예측 시간
        
    def load_models(self):
        """학습된 모델들 로드"""
        print("="*70)
        print("학습된 모델 로딩 중...")
        print("="*70)
        
        # 스크립트 디렉토리
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 모델 로드 시도
        model_names = ['lstm', 'gru', 'rnn', 'bi_lstm']
        
        for model_name in model_names:
            model_paths = [
                os.path.join(script_dir, "model", f"{model_name}_final_hybrid.keras"),
                os.path.join(script_dir, "model", "BM_s30f10_0731_2079936.keras"),
                os.path.join(script_dir, "model", "Model_s30f10_0731_2079936.keras"),
                # 대체 경로들
                f"model/{model_name}_final_hybrid.keras",
                "model/BM_s30f10_0731_2079936.keras",
                "model/Model_s30f10_0731_2079936.keras"
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        self.models[model_name] = load_model(model_path, compile=False)
                        print(f"✓ {model_name.upper()} 모델 로드 완료")
                        model_loaded = True
                        break
                    except Exception as e:
                        continue
            
            if not model_loaded and len(self.models) == 0:
                # 첫 번째 모델이면 반드시 하나는 로드해야 함
                for model_path in model_paths:
                    if os.path.exists(model_path):
                        try:
                            # 모든 모델 타입에 동일한 모델 사용
                            loaded_model = load_model(model_path, compile=False)
                            for name in model_names:
                                self.models[name] = loaded_model
                            print(f"✓ 단일 모델을 모든 타입에 사용")
                            break
                        except:
                            continue
                break
        
        # 스케일러 로드
        scaler_paths = [
            os.path.join(script_dir, "scaler", "standard_scaler_hybrid.pkl"),
            os.path.join(script_dir, "scaler", "StdScaler_s30f10_0731_2079936.save"),
            "scaler/standard_scaler_hybrid.pkl",
            "scaler/StdScaler_s30f10_0731_2079936.save"
        ]
        
        for scaler_path in scaler_paths:
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    print("✓ 스케일러 로드 완료")
                    break
                except:
                    continue
        
        if not self.models:
            raise Exception("모델을 로드할 수 없습니다!")
        if self.scaler is None:
            raise Exception("스케일러를 로드할 수 없습니다!")
            
        print(f"\n로드된 모델 수: {len(self.models)}개")
        
    def prepare_data(self, data_path):
        """데이터 전처리"""
        print("\n데이터 전처리 중...")
        
        # 데이터 로드
        data = pd.read_csv(data_path)
        
        # 시간 컬럼 변환
        data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
        data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
        
        # SUM 컬럼 제거
        columns_to_drop = [col for col in data.columns if 'SUM' in col]
        data = data.drop(columns=columns_to_drop)
        
        # 필요한 컬럼만 선택
        data = data[['CURRTIME', 'TOTALCNT', 'TIME']]
        data.set_index('CURRTIME', inplace=True)
        
        # 특징 엔지니어링
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        data['MA_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
        data['MA_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
        data['MA_30'] = data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
        data['STD_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).std()
        data['STD_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).std()
        data['change_rate'] = data['TOTALCNT'].pct_change()
        data['change_rate_5'] = data['TOTALCNT'].pct_change(5)
        data = data.fillna(method='ffill').fillna(0)
        
        print(f"전처리 완료 - 데이터 shape: {data.shape}")
        
        return data
    
    def scale_features(self, data):
        """특징 스케일링"""
        # 스케일링할 컬럼
        scale_columns = ['TOTALCNT', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
        available_columns = [col for col in scale_columns if col in data.columns]
        
        # FUTURE 컬럼 더미로 추가 (스케일러가 기대하는 경우)
        if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ == 7:
            data = data.copy()
            data['FUTURE'] = data['TOTALCNT']
            available_columns = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
            available_columns = [col for col in available_columns if col in data.columns]
        
        # 스케일링
        scaled_data = self.scaler.transform(data[available_columns])
        scaled_df = pd.DataFrame(scaled_data, columns=[f'scaled_{col}' for col in available_columns])
        scaled_df.index = data.index
        
        # 병합
        result = pd.merge(data, scaled_df, left_index=True, right_index=True, how='left')
        
        return result, available_columns
    
    def predict_next_10min(self, data, start_idx):
        """특정 시점에서 10분 후 예측"""
        # 시퀀스 데이터 추출
        end_idx = start_idx + self.seq_length
        if end_idx > len(data):
            return None
        
        # 입력 특징 선택
        input_features = [col for col in data.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']
        
        # 시퀀스 생성
        X_seq = data[input_features].iloc[start_idx:end_idx].values
        X_seq = X_seq.reshape(1, self.seq_length, len(input_features))
        
        # 각 모델로 예측
        predictions = {}
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_seq, verbose=0)
                predictions[model_name] = pred[0][0]
            except:
                pass
        
        # 앙상블 예측 (가중 평균)
        if predictions:
            weights = {'lstm': 0.3, 'gru': 0.25, 'rnn': 0.15, 'bi_lstm': 0.3}
            ensemble_pred = 0
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = weights.get(model_name, 1/len(predictions))
                ensemble_pred += pred * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_pred = ensemble_pred / total_weight
                
            return ensemble_pred, predictions
        
        return None
    
    def inverse_scale_prediction(self, scaled_pred, available_columns):
        """예측값 역스케일링"""
        n_features = len(available_columns)
        dummy_array = np.zeros((1, n_features))
        
        # FUTURE 또는 TOTALCNT 위치 찾기
        if 'FUTURE' in available_columns:
            pred_idx = available_columns.index('FUTURE')
        elif 'TOTALCNT' in available_columns:
            pred_idx = available_columns.index('TOTALCNT')
        else:
            pred_idx = 0
        
        dummy_array[0, pred_idx] = scaled_pred
        return self.scaler.inverse_transform(dummy_array)[0, pred_idx]
    
    def evaluate_100min_predictions(self, data_path):
        """100분 데이터로 10분 예측 평가"""
        # 모델 로드
        self.load_models()
        
        # 데이터 준비
        data = self.prepare_data(data_path)
        scaled_data, available_columns = self.scale_features(data)
        
        print("\n" + "="*70)
        print("100분 데이터 → 10분 예측 평가 시작")
        print("="*70)
        
        # 결과 저장용 리스트
        results = []
        
        # 100분 데이터로 예측 (0~99분 데이터로 10~109분 예측)
        for i in range(min(100, len(data) - self.seq_length - 10)):
            # 현재 시점
            current_time = data.index[i + self.seq_length - 1]
            current_value = data['TOTALCNT'].iloc[i + self.seq_length - 1]
            
            # 10분 후 실제값
            future_idx = i + self.seq_length - 1 + 10
            if future_idx < len(data):
                future_time = data.index[future_idx]
                actual_value = data['TOTALCNT'].iloc[future_idx]
                
                # 예측
                pred_result = self.predict_next_10min(scaled_data, i)
                if pred_result:
                    ensemble_pred_scaled, individual_preds = pred_result
                    
                    # 역스케일링
                    ensemble_pred = self.inverse_scale_prediction(ensemble_pred_scaled, available_columns)
                    
                    # 개별 모델 예측값도 역스케일링
                    individual_predictions = {}
                    for model_name, pred in individual_preds.items():
                        individual_predictions[model_name] = self.inverse_scale_prediction(pred, available_columns)
                    
                    # 결과 저장
                    result = {
                        'current_time': current_time,
                        'current_value': current_value,
                        'future_time': future_time,
                        'actual_value': actual_value,
                        'ensemble_pred': ensemble_pred,
                        'lstm_pred': individual_predictions.get('lstm', ensemble_pred),
                        'gru_pred': individual_predictions.get('gru', ensemble_pred),
                        'rnn_pred': individual_predictions.get('rnn', ensemble_pred),
                        'bi_lstm_pred': individual_predictions.get('bi_lstm', ensemble_pred)
                    }
                    results.append(result)
        
        # DataFrame으로 변환
        results_df = pd.DataFrame(results)
        
        # 평가 지표 계산
        print(f"\n총 {len(results_df)}개 예측 수행")
        
        # 전체 성능 지표
        mae = mean_absolute_error(results_df['actual_value'], results_df['ensemble_pred'])
        mse = mean_squared_error(results_df['actual_value'], results_df['ensemble_pred'])
        rmse = np.sqrt(mse)
        r2 = r2_score(results_df['actual_value'], results_df['ensemble_pred'])
        
        # MAPE 계산
        mape = np.mean(np.abs((results_df['actual_value'] - results_df['ensemble_pred']) / results_df['actual_value'])) * 100
        
        print("\n" + "="*50)
        print("앙상블 모델 성능 평가 결과")
        print("="*50)
        print(f"MAE  : {mae:.2f}")
        print(f"MSE  : {mse:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R²   : {r2:.4f}")
        print(f"MAPE : {mape:.2f}%")
        
        # 개별 모델 성능
        print("\n" + "="*50)
        print("개별 모델 성능 비교")
        print("="*50)
        
        for model_name in ['lstm', 'gru', 'rnn', 'bi_lstm']:
            col_name = f'{model_name}_pred'
            if col_name in results_df.columns:
                model_mae = mean_absolute_error(results_df['actual_value'], results_df[col_name])
                model_r2 = r2_score(results_df['actual_value'], results_df[col_name])
                print(f"{model_name.upper():8} - MAE: {model_mae:6.2f}, R²: {model_r2:.4f}")
        
        return results_df
    
    def visualize_results(self, results_df):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 실제값 vs 예측값 (앙상블)
        ax = axes[0, 0]
        ax.plot(results_df.index[:50], results_df['actual_value'][:50], 
                label='실제값', color='blue', linewidth=2)
        ax.plot(results_df.index[:50], results_df['ensemble_pred'][:50], 
                label='예측값 (앙상블)', color='red', linewidth=1.5, linestyle='--')
        ax.set_title('실제값 vs 예측값 (첫 50개)', fontsize=14)
        ax.set_xlabel('예측 순서')
        ax.set_ylabel('물류량')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 예측 오차 분포
        ax = axes[0, 1]
        errors = results_df['actual_value'] - results_df['ensemble_pred']
        ax.hist(errors, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_title('예측 오차 분포', fontsize=14)
        ax.set_xlabel('오차 (실제값 - 예측값)')
        ax.set_ylabel('빈도')
        ax.grid(True, alpha=0.3)
        
        # 3. 산점도
        ax = axes[1, 0]
        ax.scatter(results_df['actual_value'], results_df['ensemble_pred'], 
                  alpha=0.5, color='blue')
        
        # 완벽한 예측선
        min_val = min(results_df['actual_value'].min(), results_df['ensemble_pred'].min())
        max_val = max(results_df['actual_value'].max(), results_df['ensemble_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_title('실제값 vs 예측값 산점도', fontsize=14)
        ax.set_xlabel('실제값')
        ax.set_ylabel('예측값')
        ax.grid(True, alpha=0.3)
        
        # 4. 모델별 MAE 비교
        ax = axes[1, 1]
        model_names = ['ensemble', 'lstm', 'gru', 'rnn', 'bi_lstm']
        mae_values = []
        
        for model_name in model_names:
            if model_name == 'ensemble':
                col_name = 'ensemble_pred'
            else:
                col_name = f'{model_name}_pred'
            
            if col_name in results_df.columns:
                mae = mean_absolute_error(results_df['actual_value'], results_df[col_name])
                mae_values.append(mae)
            else:
                mae_values.append(0)
        
        bars = ax.bar(model_names, mae_values, color=['darkred', 'blue', 'green', 'orange', 'purple'])
        ax.set_title('모델별 MAE 비교', fontsize=14)
        ax.set_ylabel('MAE')
        
        # 값 표시
        for bar, mae in zip(bars, mae_values):
            if mae > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{mae:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'evaluation_100min_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 상세 결과 출력
        print("\n" + "="*70)
        print("예측 결과 상세 (처음 20개)")
        print("="*70)
        print(f"{'현재시간':^16} | {'현재값':^8} | {'10분후':^16} | {'실제값':^8} | {'예측값':^8} | {'오차':^8}")
        print("-"*70)
        
        for i in range(min(20, len(results_df))):
            row = results_df.iloc[i]
            error = row['actual_value'] - row['ensemble_pred']
            print(f"{row['current_time'].strftime('%m-%d %H:%M'):^16} | "
                  f"{row['current_value']:^8.0f} | "
                  f"{row['future_time'].strftime('%m-%d %H:%M'):^16} | "
                  f"{row['actual_value']:^8.0f} | "
                  f"{row['ensemble_pred']:^8.0f} | "
                  f"{error:^8.1f}")
        
        # 결과 저장
        results_df.to_csv(f'prediction_results_100min_{timestamp}.csv', index=False)
        print(f"\n결과가 'prediction_results_100min_{timestamp}.csv'에 저장되었습니다.")
        
        return results_df

def main():
    """메인 실행 함수"""
    print("\n" + "="*70)
    print("반도체 물류 예측 평가 시스템")
    print("100분 데이터 → 10분 예측")
    print("="*70)
    
    evaluator = LogisticsPredictionEvaluator()
    
    # 데이터 파일 경로 설정
    data_path = '20250731_to_20250806.csv'
    
    # 다양한 경로 시도
    possible_paths = [
        data_path,
        os.path.join('data', data_path),
        os.path.join(os.path.dirname(__file__), data_path),
        os.path.join(os.path.dirname(__file__), 'data', data_path)
    ]
    
    actual_path = None
    for path in possible_paths:
        if os.path.exists(path):
            actual_path = path
            break
    
    if actual_path is None:
        print(f"\n❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("다음 위치 중 하나에 파일을 놓아주세요:")
        for path in possible_paths:
            print(f"  - {path}")
        return
    
    print(f"\n데이터 파일 경로: {actual_path}")
    
    try:
        # 평가 실행
        results_df = evaluator.evaluate_100min_predictions(actual_path)
        
        # 결과 시각화
        evaluator.visualize_results(results_df)
        
        print("\n✅ 평가가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()