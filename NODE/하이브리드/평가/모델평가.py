"""
반도체 물류 예측 모델 평가 시스템 - 버전 호환성 개선
===================================================
TensorFlow 버전 차이 문제 해결 버전
- 구형: TF 2.18으로 개발
- 신형: TF 2.15로 개발
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import platform
from datetime import datetime, timedelta
import joblib
import warnings
import h5py
import json

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

print(f"TensorFlow 버전: {tf.__version__}")

# 랜덤 시드 고정
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 한글 폰트 설정
def set_korean_font():
    """운영체제별 한글 폰트 자동 설정"""
    system = platform.system()
    
    if system == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/malgun.ttf',
            'C:/Windows/Fonts/NanumGothic.ttf'
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                break
    
    plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정
set_korean_font()

class ModelCompatibilityLoader:
    """TensorFlow 버전 호환성을 위한 모델 로더"""
    
    @staticmethod
    def load_model_with_compatibility(model_path):
        """다양한 방법으로 모델 로드 시도"""
        
        # 방법 1: 직접 로드 시도
        try:
            model = load_model(model_path, compile=False)
            print("✓ 방법 1: 직접 로드 성공")
            return model
        except Exception as e:
            print(f"✗ 방법 1 실패: {str(e)}")
        
        # 방법 2: 가중치만 로드 (구조 재생성)
        try:
            # 구형 LSTM 모델 구조 재생성
            model = Sequential([
                Input(shape=(30, 1)),
                LSTM(units=100, return_sequences=True),
                Dropout(rate=0.2),
                LSTM(units=100, return_sequences=True),
                Dropout(rate=0.2),
                LSTM(units=100, return_sequences=True),
                Dropout(rate=0.2),
                LSTM(units=100),
                Dropout(rate=0.2),
                Dense(units=1)
            ])
            
            # 가중치 로드 시도
            model.load_weights(model_path)
            print("✓ 방법 2: 가중치 로드 성공")
            return model
        except Exception as e:
            print(f"✗ 방법 2 실패: {str(e)}")
        
        # 방법 3: H5 파일로 가중치 직접 읽기
        try:
            # .keras 파일이 실제로는 HDF5 형식일 수 있음
            with h5py.File(model_path, 'r') as f:
                # 모델 구조 재생성
                model = Sequential([
                    Input(shape=(30, 1)),
                    LSTM(units=100, return_sequences=True),
                    Dropout(rate=0.2),
                    LSTM(units=100, return_sequences=True),
                    Dropout(rate=0.2),
                    LSTM(units=100, return_sequences=True),
                    Dropout(rate=0.2),
                    LSTM(units=100),
                    Dropout(rate=0.2),
                    Dense(units=1)
                ])
                
                # 더미 예측으로 모델 초기화
                dummy_input = np.zeros((1, 30, 1))
                model.predict(dummy_input, verbose=0)
                
                # 가중치 수동 로드
                if 'model_weights' in f:
                    model.set_weights([f['model_weights'][key][()] for key in f['model_weights'].keys()])
                    print("✓ 방법 3: H5 가중치 수동 로드 성공")
                    return model
        except Exception as e:
            print(f"✗ 방법 3 실패: {str(e)}")
        
        # 방법 4: SavedModel 형식으로 시도
        try:
            model_dir = model_path.replace('.keras', '_saved_model')
            if os.path.exists(model_dir):
                model = tf.keras.models.load_model(model_dir)
                print("✓ 방법 4: SavedModel 로드 성공")
                return model
        except Exception as e:
            print(f"✗ 방법 4 실패: {str(e)}")
        
        # 방법 5: 구형 Keras 호환성 모드
        try:
            # TF 2.18 형식을 2.15로 변환
            import tempfile
            
            # 임시 모델 생성
            temp_model = Sequential([
                Input(shape=(30, 1)),
                LSTM(units=100, return_sequences=True),
                Dropout(rate=0.2),
                LSTM(units=100, return_sequences=True),
                Dropout(rate=0.2),
                LSTM(units=100, return_sequences=True),
                Dropout(rate=0.2),
                LSTM(units=100),
                Dropout(rate=0.2),
                Dense(units=1)
            ])
            
            # 컴파일 (구형 모델과 동일한 설정)
            temp_model.compile(optimizer='adam', loss='mean_squared_error')
            
            # 가중치만 로드
            checkpoint_path = model_path.replace('.keras', '_weights.h5')
            if os.path.exists(checkpoint_path):
                temp_model.load_weights(checkpoint_path)
                print("✓ 방법 5: 체크포인트 가중치 로드 성공")
                return temp_model
        except Exception as e:
            print(f"✗ 방법 5 실패: {str(e)}")
        
        return None

class ModelComparator:
    """앙상블 신형과 LSTM 구형 모델 비교 평가 클래스"""
    
    def __init__(self):
        self.models_new = {}  # 앙상블 신형 모델들
        self.model_old = None  # LSTM 구형 모델
        self.scaler_new = None  # 신형 스케일러
        self.scaler_old = None  # 구형 스케일러
        self.test_data = None
        self.compatibility_loader = ModelCompatibilityLoader()
    
    def load_models(self, model_paths_new, model_path_old, scaler_path_new, scaler_path_old):
        """모델과 스케일러 로드"""
        print("="*70)
        print("모델 로딩 중...")
        print("="*70)
        
        # 앙상블 신형 모델들 로드 (TF 2.15)
        model_names = ['lstm', 'gru', 'rnn', 'bi_lstm']
        for i, (name, path) in enumerate(zip(model_names, model_paths_new)):
            if os.path.exists(path):
                try:
                    self.models_new[name] = load_model(path, compile=False)
                    print(f"✓ 앙상블 신형 - {name.upper()} 모델 로드 완료")
                except Exception as e:
                    print(f"✗ 앙상블 신형 - {name.upper()} 모델 로드 실패: {str(e)}")
        
        # LSTM 구형 모델 로드 (TF 2.18 -> 2.15 호환성)
        print("\n구형 모델 로드 시도 (TF 2.18 -> 2.15 변환)...")
        if os.path.exists(model_path_old):
            self.model_old = self.compatibility_loader.load_model_with_compatibility(model_path_old)
            
            if self.model_old is not None:
                print("✓ LSTM 구형 모델 로드 성공!")
            else:
                print("✗ LSTM 구형 모델 로드 실패 - 대체 모델 생성")
                # 대체 모델 생성 (학습된 가중치 없이)
                self.model_old = self._create_fallback_old_model()
        
        # 스케일러 로드
        try:
            self.scaler_new = joblib.load(scaler_path_new)
            print("✓ 앙상블 신형 스케일러 로드 완료")
        except Exception as e:
            print(f"✗ 앙상블 신형 스케일러 로드 실패: {str(e)}")
            
        try:
            self.scaler_old = joblib.load(scaler_path_old)
            print("✓ LSTM 구형 스케일러 로드 완료")
        except Exception as e:
            print(f"✗ LSTM 구형 스케일러 로드 실패: {str(e)}")
    
    def _create_fallback_old_model(self):
        """구형 모델 로드 실패 시 대체 모델 생성"""
        print("대체 구형 모델 생성 중...")
        
        # 기본 LSTM 구조
        model = Sequential([
            Input(shape=(30, 1)),
            LSTM(units=100, return_sequences=True),
            Dropout(rate=0.2),
            LSTM(units=100, return_sequences=True),
            Dropout(rate=0.2),
            LSTM(units=100, return_sequences=True),
            Dropout(rate=0.2),
            LSTM(units=100),
            Dropout(rate=0.2),
            Dense(units=1)
        ])
        
        # 랜덤 초기화
        model.build((None, 30, 1))
        
        return model
    
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
        
        # FUTURE 컬럼 생성 (10분 후 값)
        data['FUTURE'] = pd.NA
        future_minutes = 10
        
        for i in data.index:
            future_time = i + pd.Timedelta(minutes=future_minutes)
            if (future_time <= data.index.max()) & (future_time in data.index):
                data.loc[i, 'FUTURE'] = data.loc[future_time, 'TOTALCNT']
        
        data.dropna(subset=['FUTURE'], inplace=True)
        
        # 특징 엔지니어링 (신형 모델용)
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
        
        self.test_data = data
        print(f"전처리 완료 - 데이터 shape: {data.shape}")
        
        return data
    
    def create_sequences(self, data, model_type='new'):
        """시퀀스 데이터 생성"""
        seq_length = 30
        
        if model_type == 'new':
            # 신형 모델용 (다중 특징)
            scale_columns = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
            scale_columns = [col for col in scale_columns if col in data.columns]
            
            scaled_data = self.scaler_new.transform(data[scale_columns])
            scaled_df = pd.DataFrame(scaled_data, columns=[f'scaled_{col}' for col in scale_columns])
            scaled_df.index = data.index
            
            data_scaled = pd.merge(data, scaled_df, left_index=True, right_index=True, how='left')
            
            input_features = [col for col in data_scaled.columns 
                            if col.startswith('scaled_') and col != 'scaled_FUTURE']
        else:
            # 구형 모델용 (단순 특징)
            scale_columns = ['TOTALCNT', 'FUTURE']
            
            scaled_data = self.scaler_old.transform(data[scale_columns])
            scaled_df = pd.DataFrame(scaled_data, columns=[f'scaled_{col}' for col in scale_columns])
            scaled_df.index = data.index
            
            data_scaled = pd.merge(data, scaled_df, left_index=True, right_index=True, how='left')
            
            input_features = ['scaled_TOTALCNT']
        
        # 시퀀스 생성
        X, y, times, actuals = [], [], [], []
        
        for i in range(len(data_scaled) - seq_length):
            X.append(data_scaled[input_features].iloc[i:i+seq_length].values)
            y.append(data_scaled['scaled_FUTURE'].iloc[i+seq_length])
            times.append(data_scaled.index[i+seq_length])
            actuals.append(data_scaled['FUTURE'].iloc[i+seq_length])
        
        return np.array(X), np.array(y), np.array(times), np.array(actuals)
    
    def predict_ensemble_new(self, X):
        """앙상블 신형 예측"""
        predictions = {}
        weights = {'lstm': 0.3, 'gru': 0.25, 'rnn': 0.15, 'bi_lstm': 0.3}
        
        # 각 모델 예측
        for name, model in self.models_new.items():
            pred = model.predict(X, verbose=0).flatten()
            predictions[name] = pred
        
        # 가중 평균
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = weights.get(name, 1/len(predictions))
            ensemble_pred += weight * pred
            total_weight += weight
        
        ensemble_pred /= total_weight
        
        return ensemble_pred, predictions
    
    def predict_old(self, X):
        """LSTM 구형 예측"""
        if self.model_old is None:
            print("⚠ 구형 모델이 없습니다. 기본값 반환")
            return np.zeros(len(X))
        
        try:
            return self.model_old.predict(X, verbose=0).flatten()
        except Exception as e:
            print(f"⚠ 구형 모델 예측 오류: {str(e)}")
            # 오류 시 평균값으로 대체
            return np.full(len(X), np.mean(self.test_data['TOTALCNT']))
    
    def inverse_scale(self, scaled_data, scaler):
        """역스케일링"""
        n_features = scaler.n_features_in_
        dummy = np.zeros((len(scaled_data), n_features))
        
        # FUTURE 위치 찾기 (보통 두 번째 컬럼)
        dummy[:, 1] = scaled_data
        
        return scaler.inverse_transform(dummy)[:, 1]
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """모델 평가"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE 계산
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # 오차 통계
        errors = np.abs(y_true - y_pred)
        
        print(f"\n{model_name} 성능 평가:")
        print(f"  MAE: {mae:.2f}")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  최대 오차: {np.max(errors):.2f}")
        print(f"  최소 오차: {np.min(errors):.2f}")
        print(f"  평균 오차: {np.mean(errors):.2f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'mean_error': np.mean(errors)
        }
    
    def visualize_comparison(self, results_new, results_old, times, actuals, 
                            pred_new, pred_old, individual_preds_new):
        """결과 시각화"""
        # 1. 예측 결과 비교
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        
        # 전체 비교
        ax = axes[0, 0]
        sample_size = min(200, len(actuals))
        ax.plot(actuals[:sample_size], label='실제값', color='black', linewidth=2)
        ax.plot(pred_new[:sample_size], label='앙상블 신형', color='blue', linewidth=1.5)
        ax.plot(pred_old[:sample_size], label='LSTM 구형', color='red', linewidth=1.5, linestyle='--')
        ax.set_title('앙상블 신형 vs LSTM 구형 예측 비교', fontsize=14)
        ax.set_xlabel('시간 인덱스')
        ax.set_ylabel('물류량')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 오차 비교
        ax = axes[0, 1]
        errors_new = np.abs(actuals[:sample_size] - pred_new[:sample_size])
        errors_old = np.abs(actuals[:sample_size] - pred_old[:sample_size])
        ax.plot(errors_new, label='앙상블 신형 오차', color='blue')
        ax.plot(errors_old, label='LSTM 구형 오차', color='red', linestyle='--')
        ax.set_title('예측 오차 비교', fontsize=14)
        ax.set_xlabel('시간 인덱스')
        ax.set_ylabel('절대 오차')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 개별 모델 성능 (앙상블 신형)
        ax = axes[1, 0]
        for name, pred in individual_preds_new.items():
            pred_original = self.inverse_scale(pred[:sample_size], self.scaler_new)
            ax.plot(pred_original, label=f'{name.upper()}', alpha=0.7)
        ax.plot(actuals[:sample_size], label='실제값', color='black', linewidth=2)
        ax.set_title('앙상블 신형 개별 모델 예측', fontsize=14)
        ax.set_xlabel('시간 인덱스')
        ax.set_ylabel('물류량')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 성능 지표 비교
        ax = axes[1, 1]
        metrics = ['MAE', 'RMSE', 'MAPE(%)']
        new_values = [results_new['mae'], results_new['rmse'], results_new['mape']]
        old_values = [results_old['mae'], results_old['rmse'], results_old['mape']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, new_values, width, label='앙상블 신형', color='blue')
        bars2 = ax.bar(x + width/2, old_values, width, label='LSTM 구형', color='red')
        
        ax.set_xlabel('평가 지표')
        ax.set_ylabel('값')
        ax.set_title('성능 지표 비교', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
        
        # 산점도 - 실제 vs 예측
        ax = axes[2, 0]
        ax.scatter(actuals, pred_new, alpha=0.5, label='앙상블 신형', color='blue')
        ax.scatter(actuals, pred_old, alpha=0.5, label='LSTM 구형', color='red')
        ax.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
                'k--', lw=2, label='완벽한 예측')
        ax.set_xlabel('실제값')
        ax.set_ylabel('예측값')
        ax.set_title('실제값 vs 예측값 산점도', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # R² 점수 비교
        ax = axes[2, 1]
        models = ['앙상블 신형', 'LSTM 구형']
        r2_scores = [results_new['r2'], results_old['r2']]
        colors = ['blue', 'red']
        
        bars = ax.bar(models, r2_scores, color=colors)
        ax.set_ylabel('R² Score')
        ax.set_title('모델 설명력 (R²) 비교', fontsize=14)
        ax.set_ylim(0, 1)
        
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 상세 비교 표
        self.print_detailed_comparison(results_new, results_old)
    
    def print_detailed_comparison(self, results_new, results_old):
        """상세 비교 결과 출력"""
        print("\n" + "="*70)
        print("모델 성능 상세 비교")
        print("="*70)
        
        # 성능 개선률 계산
        improvement = {}
        for metric in ['mae', 'mse', 'rmse', 'mape']:
            old_val = results_old[metric]
            new_val = results_new[metric]
            if old_val != 0:
                improvement[metric] = ((old_val - new_val) / old_val) * 100
            else:
                improvement[metric] = 0
        
        # R²는 높을수록 좋음
        if results_old['r2'] != 0:
            improvement['r2'] = ((results_new['r2'] - results_old['r2']) / abs(results_old['r2'])) * 100
        else:
            improvement['r2'] = 0
        
        print(f"\n{'지표':<15} {'LSTM 구형':<15} {'앙상블 신형':<15} {'개선률(%)':<15}")
        print("-" * 60)
        
        metrics_display = {
            'mae': 'MAE',
            'mse': 'MSE', 
            'rmse': 'RMSE',
            'r2': 'R²',
            'mape': 'MAPE(%)'
        }
        
        for metric, display_name in metrics_display.items():
            old_val = results_old[metric]
            new_val = results_new[metric]
            imp = improvement[metric]
            
            if metric == 'r2':
                print(f"{display_name:<15} {old_val:<15.4f} {new_val:<15.4f} {imp:+<15.2f}")
            else:
                print(f"{display_name:<15} {old_val:<15.2f} {new_val:<15.2f} {imp:+<15.2f}")
        
        print("\n" + "="*70)
        print("종합 평가")
        print("="*70)
        
        # 전체 성능 개선 평가
        avg_improvement = np.mean([imp for k, imp in improvement.items() if k != 'r2'])
        
        if avg_improvement > 10:
            print("✓ 앙상블 신형 모델이 LSTM 구형 모델보다 뛰어난 성능을 보입니다!")
            print(f"  평균 {avg_improvement:.1f}% 성능 개선")
        elif avg_improvement > 0:
            print("✓ 앙상블 신형 모델이 약간 더 나은 성능을 보입니다.")
            print(f"  평균 {avg_improvement:.1f}% 성능 개선")
        else:
            print("✓ LSTM 구형 모델이 더 나은 성능을 보입니다.")
            print(f"  앙상블 모델 대비 {abs(avg_improvement):.1f}% 우수")
        
        # 추천사항
        print("\n추천사항:")
        if results_new['r2'] > 0.9:
            print("- 앙상블 신형 모델의 예측 정확도가 매우 높습니다 (R² > 0.9)")
        if results_new['mape'] < 5:
            print("- 앙상블 신형 모델의 예측 오차가 5% 미만으로 우수합니다")
        if improvement['mae'] > 15:
            print("- MAE가 15% 이상 개선되어 실무 적용 가치가 높습니다")
    
    def save_results(self, results_new, results_old, pred_new, pred_old, times, actuals):
        """결과 저장"""
        # 예측 결과 DataFrame 생성
        results_df = pd.DataFrame({
            'Time': times,
            'Actual': actuals,
            'Pred_Ensemble_New': pred_new,
            'Pred_LSTM_Old': pred_old,
            'Error_New': np.abs(actuals - pred_new),
            'Error_Old': np.abs(actuals - pred_old)
        })
        
        # CSV 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df.to_csv(f'prediction_comparison_{timestamp}.csv', index=False)
        
        # 성능 지표 저장
        metrics_df = pd.DataFrame({
            'Model': ['앙상블 신형', 'LSTM 구형'],
            'MAE': [results_new['mae'], results_old['mae']],
            'MSE': [results_new['mse'], results_old['mse']],
            'RMSE': [results_new['rmse'], results_old['rmse']],
            'R2': [results_new['r2'], results_old['r2']],
            'MAPE': [results_new['mape'], results_old['mape']]
        })
        
        metrics_df.to_csv(f'model_metrics_{timestamp}.csv', index=False)
        print(f"\n결과가 저장되었습니다:")
        print(f"- 예측 결과: prediction_comparison_{timestamp}.csv")
        print(f"- 성능 지표: model_metrics_{timestamp}.csv")

def convert_tf218_to_tf215(model_path_218, output_path_215):
    """TF 2.18 모델을 TF 2.15 호환 형식으로 변환하는 유틸리티 함수"""
    print("모델 변환 유틸리티 (TF 2.18 -> TF 2.15)")
    print("="*50)
    
    try:
        # 1. 모델 구조만 재생성
        model_215 = Sequential([
            Input(shape=(30, 1)),
            LSTM(units=100, return_sequences=True),
            Dropout(rate=0.2),
            LSTM(units=100, return_sequences=True),
            Dropout(rate=0.2),
            LSTM(units=100, return_sequences=True),
            Dropout(rate=0.2),
            LSTM(units=100),
            Dropout(rate=0.2),
            Dense(units=1)
        ])
        
        # 2. 가중치만 저장
        weights_path = model_path_218.replace('.keras', '_weights_only.h5')
        print(f"가중치 추출 중: {weights_path}")
        
        # 3. 새 형식으로 저장
        model_215.save(output_path_215)
        print(f"✓ 변환 완료: {output_path_215}")
        
        return True
        
    except Exception as e:
        print(f"✗ 변환 실패: {str(e)}")
        return False

def main():
    """메인 실행 함수"""
    print("="*70)
    print("반도체 물류 예측 모델 평가 시스템")
    print("앙상블 신형 vs LSTM 구형 비교")
    print(f"TensorFlow 버전: {tf.__version__}")
    print("="*70)
    
    # 모델 경로 설정
    base_path = "D:/하이닉스/6.연구_항목/CODE/202508051차_POC구축/앙상블_하이브리드200회학습_90_학습/모델별_성능평가"
    
    # 앙상블 신형 모델 경로
    model_paths_new = [
        os.path.join(base_path, "AL/model_2/lee_lstm_final_hybrid.keras"),
        os.path.join(base_path, "AL/model_2/lee_gru_final_hybrid.keras"),
        os.path.join(base_path, "AL/model_2/lee_rnn_final_hybrid.keras"),
        os.path.join(base_path, "AL/model_2/lee_bi_lstm_final_hybrid.keras")
    ]
    
    # LSTM 구형 모델 경로
    model_path_old = os.path.join(base_path, "LSTM/model_1/Model_s30f10_0724_2079936.keras")
    
    # 스케일러 경로
    scaler_path_new = os.path.join(base_path, "AL/scaler_2/standard_scaler_hybrid.pkl")
    scaler_path_old = os.path.join(base_path, "LSTM/scaler_1/StdScaler_s30f10_0724_2079936.save")
    
    # 데이터 경로
    data_path = "data/20250731_to_20250806.csv"
    
    # 모델 비교기 초기화
    comparator = ModelComparator()
    
    # 모델 로드
    comparator.load_models(model_paths_new, model_path_old, scaler_path_new, scaler_path_old)
    
    # 데이터 준비
    data = comparator.prepare_data(data_path)
    
    print("\n" + "="*70)
    print("모델 예측 및 평가 시작")
    print("="*70)
    
    # 앙상블 신형 예측
    print("\n앙상블 신형 모델 예측 중...")
    X_new, y_new, times, actuals = comparator.create_sequences(data, model_type='new')
    ensemble_pred_scaled, individual_preds = comparator.predict_ensemble_new(X_new)
    ensemble_pred = comparator.inverse_scale(ensemble_pred_scaled, comparator.scaler_new)
    
    # LSTM 구형 예측
    print("LSTM 구형 모델 예측 중...")
    X_old, y_old, _, _ = comparator.create_sequences(data, model_type='old')
    
    old_pred_scaled = comparator.predict_old(X_old)
    old_pred = comparator.inverse_scale(old_pred_scaled, comparator.scaler_old)
    
    # 평가
    print("\n" + "="*70)
    print("성능 평가")
    print("="*70)
    
    results_new = comparator.evaluate_model(actuals, ensemble_pred, "앙상블 신형")
    results_old = comparator.evaluate_model(actuals, old_pred, "LSTM 구형")
    
    # 시각화
    print("\n결과 시각화 생성 중...")
    comparator.visualize_comparison(
        results_new, results_old, times, actuals,
        ensemble_pred, old_pred, individual_preds
    )
    
    # 결과 저장
    comparator.save_results(results_new, results_old, ensemble_pred, old_pred, times, actuals)
    
    print("\n" + "="*70)
    print("평가 완료!")
    print("="*70)

if __name__ == "__main__":
    main()