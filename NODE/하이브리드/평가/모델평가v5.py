"""
학습된 모델을 로드하여 평가만 수행하는 코드
TensorFlow 2.15.0
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print(f"TensorFlow Version: {tf.__version__}")

class ModelEvaluator:
    """학습된 모델을 로드하여 평가"""
    
    def __init__(self, model_dir=None, data_path=None):
        # 절대 경로 설정
        if model_dir is None:
            self.model_dir = r'D:\하이닉스\6.연구_항목\CODE\202508051차_POC구축\앙상블_하이브리드v5_150g학습\models_v5'
        else:
            self.model_dir = model_dir
            
        if data_path is None:
            self.data_path = r'D:\하이닉스\6.연구_항목\CODE\202508051차_POC구축\앙상블_하이브리드v5_150g학습\data\20250731_to20250806.csv'
        else:
            self.data_path = data_path
            
        self.sequence_length = 100  # 과거 100분
        self.prediction_horizon = 10  # 10분 후 예측
        self.spike_threshold = 1400
        self.models = {}
        self.scaler = None
        
        print(f"모델 디렉토리: {self.model_dir}")
        print(f"데이터 경로: {self.data_path}")
        
    def load_models(self):
        """저장된 모델들 로드"""
        print("=" * 60)
        print("학습된 모델 로드 중...")
        print("=" * 60)
        
        model_files = {
            'lstm': 'lstm_final.h5',
            'gru': 'gru_final.h5', 
            'cnn_lstm': 'cnn_lstm_final.h5',
            'spike_detector': 'spike_detector_final.h5'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(self.model_dir, filename)
            if os.path.exists(filepath):
                try:
                    # compile=False로 로드 (custom_objects 없이)
                    self.models[name] = keras.models.load_model(filepath, compile=False)
                    
                    # 모델 다시 컴파일
                    if name == 'spike_detector':
                        # spike_detector는 다중 출력
                        self.models[name].compile(
                            optimizer='adam',
                            loss=['mae', 'binary_crossentropy'],
                            metrics=['mae', 'accuracy']
                        )
                    else:
                        self.models[name].compile(
                            optimizer='adam',
                            loss='mae',
                            metrics=['mae']
                        )
                    
                    print(f"✓ {name} 모델 로드 완료")
                except Exception as e:
                    # 다른 방법 시도
                    try:
                        self.models[name] = tf.saved_model.load(filepath)
                        print(f"✓ {name} 모델 로드 완료 (saved_model)")
                    except:
                        print(f"✗ {name} 모델 로드 실패: {e}")
            else:
                print(f"✗ {name} 모델 파일 없음: {filepath}")
        
        # 스케일러 로드
        scaler_loaded = False
        
        # joblib 먼저 시도 (더 안정적)
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"✓ 스케일러 로드 완료")
                scaler_loaded = True
            except:
                try:
                    # pickle로 재시도
                    import pickle
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f, encoding='latin1')  # encoding 추가
                    print(f"✓ 스케일러 로드 완료 (pickle)")
                    scaler_loaded = True
                except Exception as e:
                    print(f"⚠ 스케일러 로드 실패: {e}")
        
        # 스케일러가 없으면 새로 생성
        if not scaler_loaded:
            print(f"⚠ 스케일러 새로 생성")
            self.scaler = MinMaxScaler()
        
        print()
        return len(self.models) > 0
    
    def load_evaluation_data(self):
        """평가용 데이터 로드 및 전처리"""
        print("평가 데이터 로드 중...")
        
        # 파일 존재 확인
        if not os.path.exists(self.data_path):
            print(f"✗ 파일이 존재하지 않습니다: {self.data_path}")
            return None
        
        # TSV 파일 읽기 시도 (탭 구분)
        try:
            df = pd.read_csv(self.data_path, sep='\t', encoding='utf-8')
            print("✓ TSV 형식으로 로드 성공")
        except:
            # CSV 파일 읽기 (콤마 구분)
            try:
                df = pd.read_csv(self.data_path, encoding='utf-8')
                print("✓ CSV 형식으로 로드 성공")
            except Exception as e:
                print(f"✗ 파일 로드 실패: {e}")
                return None
        
        print(f"데이터 shape: {df.shape}")
        print(f"컬럼: {df.columns.tolist()}")
        
        # 필수 컬럼 확인
        if 'current_value' not in df.columns and 'TOTALCNT' in df.columns:
            df['current_value'] = df['TOTALCNT']
        
        # 실제값 컬럼 확인
        if '실제' not in df.columns:
            # 10분 후 값을 실제값으로 사용
            df['실제'] = df['current_value'].shift(-self.prediction_horizon)
        
        # 시간 컬럼 처리
        if 'current_time' in df.columns:
            df['current_time'] = pd.to_datetime(df['current_time'])
        elif 'datetime' in df.columns:
            df['current_time'] = pd.to_datetime(df['datetime'])
        else:
            # 시간 컬럼이 없으면 인덱스 사용
            df['current_time'] = pd.date_range(start='2025-07-31', periods=len(df), freq='1min')
        
        if 'future_time' not in df.columns:
            df['future_time'] = df['current_time'] + pd.Timedelta(minutes=self.prediction_horizon)
        
        print(f"데이터 기간: {df['current_time'].iloc[0]} ~ {df['current_time'].iloc[-1]}")
        
        # 데이터 분포 확인
        df = df.dropna(subset=['실제'])  # NaN 제거
        spike_count = (df['실제'] >= self.spike_threshold).sum()
        print(f"전체 데이터: {len(df):,}개")
        print(f"1400+ 급증 구간: {spike_count:,}개 ({spike_count/len(df)*100:.2f}%)")
        print()
        
        return df
    
    def create_features(self, df):
        """특징 생성 (학습 시와 동일하게)"""
        df = df.copy()
        
        # 시간 특징
        df['hour'] = df['current_time'].dt.hour
        df['minute'] = df['current_time'].dt.minute
        df['day_of_week'] = df['current_time'].dt.dayofweek
        
        # 이동평균 및 표준편차
        for window in [5, 10, 20]:
            df[f'ma_{window}'] = df['current_value'].rolling(window=window, min_periods=1).mean()
            df[f'std_{window}'] = df['current_value'].rolling(window=window, min_periods=1).std().fillna(0)
        
        # 변화율
        df['change_rate'] = df['current_value'].pct_change().fillna(0)
        
        return df
    
    def prepare_sequences(self, df):
        """시퀀스 데이터 준비 (100분 -> 10분 후)"""
        print("시퀀스 데이터 생성 중...")
        
        # 특징 컬럼 선택
        feature_cols = ['current_value', 'hour', 'minute', 'day_of_week',
                       'ma_5', 'ma_10', 'ma_20', 
                       'std_5', 'std_10', 'std_20',
                       'change_rate']
        
        sequences = []
        actuals = []
        timestamps = []
        
        for i in range(len(df) - self.sequence_length - self.prediction_horizon + 1):
            # 과거 100분 데이터
            seq_data = df[feature_cols].iloc[i:i+self.sequence_length].values
            
            # 10분 후 실제 값
            future_idx = i + self.sequence_length + self.prediction_horizon - 1
            if future_idx < len(df):
                sequences.append(seq_data)
                actuals.append(df.iloc[future_idx]['실제'])
                timestamps.append({
                    'start_time': df.iloc[i]['current_time'],
                    'end_time': df.iloc[i+self.sequence_length-1]['current_time'],
                    'future_time': df.iloc[future_idx]['future_time']
                })
        
        X = np.array(sequences)
        y = np.array(actuals)
        
        # 스케일링 (학습 시 사용한 스케일러 사용)
        if self.scaler is not None:
            n_samples, n_timesteps, n_features = X.shape
            X_reshaped = X.reshape(n_samples * n_timesteps, n_features)
            
            # 스케일러가 학습되지 않았다면 fit
            try:
                X_scaled = self.scaler.transform(X_reshaped)
            except:
                X_scaled = self.scaler.fit_transform(X_reshaped)
            
            X = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        print(f"생성된 시퀀스: {len(X):,}개")
        print(f"입력 shape: {X.shape}")
        print(f"타겟 shape: {y.shape}")
        print()
        
        return X, y, timestamps
    
    def evaluate_models(self, X, y, timestamps):
        """각 모델 평가"""
        print("=" * 60)
        print("모델 평가 시작")
        print("=" * 60)
        
        results = {}
        predictions = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name} 모델 평가 중...")
            
            try:
                # 예측
                if model_name == 'spike_detector':
                    # Spike detector는 두 개 출력 (regression, classification)
                    pred = model.predict(X, batch_size=256, verbose=0)
                    if isinstance(pred, list):
                        y_pred = pred[0].flatten()  # regression 출력만 사용
                    else:
                        y_pred = pred.flatten()
                else:
                    y_pred = model.predict(X, batch_size=256, verbose=0).flatten()
                
                predictions[model_name] = y_pred
                
                # 평가 지표 계산
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)
                
                # 1400+ 급증 구간 평가
                spike_mask = y >= self.spike_threshold
                spike_mae = mean_absolute_error(y[spike_mask], y_pred[spike_mask]) if spike_mask.sum() > 0 else 0
                
                # 급증 감지 성능
                pred_spike = y_pred >= self.spike_threshold
                actual_spike = y >= self.spike_threshold
                
                tp = np.sum((pred_spike == True) & (actual_spike == True))
                fp = np.sum((pred_spike == True) & (actual_spike == False))
                fn = np.sum((pred_spike == False) & (actual_spike == True))
                tn = np.sum((pred_spike == False) & (actual_spike == False))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                results[model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'Spike_MAE': spike_mae,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1,
                    'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
                }
                
                print(f"  MAE: {mae:.2f}")
                print(f"  RMSE: {rmse:.2f}")
                print(f"  R²: {r2:.4f}")
                print(f"  1400+ MAE: {spike_mae:.2f}")
                print(f"  Precision: {precision*100:.2f}%")
                print(f"  Recall: {recall*100:.2f}%")
                print(f"  F1-Score: {f1:.4f}")
                
            except Exception as e:
                print(f"  ✗ 평가 실패: {e}")
                continue
        
        # 앙상블 예측 (가중 평균)
        if len(predictions) > 0:
            print(f"\n앙상블 모델 평가 중...")
            
            # 모델별 가중치 (GRU가 가장 좋았으므로 높은 가중치)
            weights = {
                'lstm': 0.2,
                'gru': 0.4,
                'cnn_lstm': 0.2,
                'spike_detector': 0.2
            }
            
            ensemble_pred = np.zeros_like(y)
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = weights.get(model_name, 0.25)
                ensemble_pred += pred * weight
                total_weight += weight
            
            ensemble_pred = ensemble_pred / total_weight
            predictions['ensemble'] = ensemble_pred
            
            # 앙상블 평가
            mae = mean_absolute_error(y, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
            r2 = r2_score(y, ensemble_pred)
            
            spike_mask = y >= self.spike_threshold
            spike_mae = mean_absolute_error(y[spike_mask], ensemble_pred[spike_mask]) if spike_mask.sum() > 0 else 0
            
            pred_spike = ensemble_pred >= self.spike_threshold
            actual_spike = y >= self.spike_threshold
            
            tp = np.sum((pred_spike == True) & (actual_spike == True))
            fp = np.sum((pred_spike == True) & (actual_spike == False))
            fn = np.sum((pred_spike == False) & (actual_spike == True))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results['ensemble'] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Spike_MAE': spike_mae,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }
            
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  1400+ MAE: {spike_mae:.2f}")
            print(f"  Precision: {precision*100:.2f}%")
            print(f"  Recall: {recall*100:.2f}%")
            print(f"  F1-Score: {f1:.4f}")
        
        return results, predictions
    
    def save_evaluation_results(self, y, predictions, timestamps, original_df):
        """평가 결과를 CSV로 저장"""
        print("\n평가 결과 저장 중...")
        
        # 결과 데이터프레임 생성
        eval_df = pd.DataFrame({
            'start_time': [t['start_time'] for t in timestamps],
            'end_time': [t['end_time'] for t in timestamps],
            'future_time': [t['future_time'] for t in timestamps],
            '실제값': y,
            '급증여부': (y >= self.spike_threshold).astype(int)
        })
        
        # 각 모델의 예측값 추가
        for model_name, pred in predictions.items():
            eval_df[f'{model_name}_예측'] = pred
            eval_df[f'{model_name}_오차'] = y - pred
            eval_df[f'{model_name}_절대오차'] = np.abs(y - pred)
        
        # 통계 정보 추가
        eval_df['실제값_구간'] = pd.cut(y, bins=[0, 1000, 1200, 1400, 1600, 2000, 10000],
                                    labels=['0-1000', '1000-1200', '1200-1400', 
                                           '1400-1600', '1600-2000', '2000+'])
        
        # CSV 저장 (한글 깨짐 방지)
        output_file = f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        eval_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 평가 결과 저장: {output_file}")
        
        # 요약 통계 저장
        summary_file = f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("반도체 물류 예측 모델 평가 결과\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"평가 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"평가 데이터: {self.data_path}\n")
            f.write(f"평가 샘플 수: {len(eval_df):,}개\n")
            f.write(f"급증 구간(≥1400): {(y >= self.spike_threshold).sum():,}개 ({(y >= self.spike_threshold).mean()*100:.2f}%)\n\n")
            
            # 구간별 분포
            f.write("구간별 데이터 분포:\n")
            for interval in eval_df['실제값_구간'].value_counts().sort_index().items():
                f.write(f"  {interval[0]}: {interval[1]:,}개\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("모델별 성능 요약\n")
            f.write("=" * 80 + "\n\n")
            
            # 모델별 성능 테이블
            for model_name in predictions.keys():
                if f'{model_name}_오차' in eval_df.columns:
                    f.write(f"\n{model_name.upper()} 모델:\n")
                    f.write(f"  평균 절대 오차: {eval_df[f'{model_name}_절대오차'].mean():.2f}\n")
                    f.write(f"  최대 오차: {eval_df[f'{model_name}_절대오차'].max():.2f}\n")
                    f.write(f"  최소 오차: {eval_df[f'{model_name}_절대오차'].min():.2f}\n")
                    f.write(f"  오차 표준편차: {eval_df[f'{model_name}_오차'].std():.2f}\n")
        
        print(f"✓ 평가 요약 저장: {summary_file}")
        
        return eval_df
    
    def visualize_results(self, eval_df, results):
        """결과 시각화"""
        print("\n결과 시각화 생성 중...")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('반도체 물류 예측 모델 평가 결과', fontsize=16, fontweight='bold')
        
        # 1. 시계열 예측 비교 (샘플)
        ax1 = axes[0, 0]
        sample_size = min(500, len(eval_df))
        x_range = range(sample_size)
        
        ax1.plot(x_range, eval_df['실제값'].iloc[:sample_size], 
                label='실제', color='black', linewidth=2, alpha=0.8)
        
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for i, model in enumerate(['lstm', 'gru', 'cnn_lstm', 'ensemble']):
            if f'{model}_예측' in eval_df.columns:
                ax1.plot(x_range, eval_df[f'{model}_예측'].iloc[:sample_size],
                        label=model.upper(), alpha=0.6, linewidth=1, color=colors[i])
        
        ax1.axhline(y=self.spike_threshold, color='red', linestyle='--', 
                   alpha=0.5, label='급증 임계값(1400)')
        ax1.set_xlabel('시간 인덱스')
        ax1.set_ylabel('물류량')
        ax1.set_title(f'예측 비교 (첫 {sample_size}개 샘플)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 모델별 MAE 비교
        ax2 = axes[0, 1]
        model_names = list(results.keys())
        mae_values = [results[m]['MAE'] for m in model_names]
        spike_mae_values = [results[m]['Spike_MAE'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, mae_values, width, label='전체 MAE', color='skyblue')
        bars2 = ax2.bar(x + width/2, spike_mae_values, width, label='1400+ MAE', color='coral')
        
        ax2.set_xlabel('모델')
        ax2.set_ylabel('MAE')
        ax2.set_title('모델별 평균 절대 오차')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.upper() for m in model_names], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 3. 급증 감지 성능 (Precision/Recall)
        ax3 = axes[1, 0]
        precision_values = [results[m]['Precision'] for m in model_names]
        recall_values = [results[m]['Recall'] for m in model_names]
        
        bars1 = ax3.bar(x - width/2, precision_values, width, label='Precision', color='lightgreen')
        bars2 = ax3.bar(x + width/2, recall_values, width, label='Recall', color='lightcoral')
        
        ax3.set_xlabel('모델')
        ax3.set_ylabel('Score')
        ax3.set_title('급증 감지 성능 (1400+)')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.upper() for m in model_names], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 값 표시
        for bar in bars1:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.1f}%', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 4. R² Score 비교
        ax4 = axes[1, 1]
        r2_values = [results[m]['R2'] for m in model_names]
        bars = ax4.bar(model_names, r2_values, color='mediumpurple')
        
        ax4.set_xlabel('모델')
        ax4.set_ylabel('R² Score')
        ax4.set_title('모델별 R² Score')
        ax4.set_xticklabels([m.upper() for m in model_names], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        for bar, r2 in zip(bars, r2_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{r2:.3f}', ha='center', va='bottom')
        
        # 5. 오차 분포 (히스토그램)
        ax5 = axes[2, 0]
        if 'ensemble_오차' in eval_df.columns:
            ax5.hist(eval_df['ensemble_오차'], bins=50, alpha=0.7, 
                    color='blue', edgecolor='black')
            ax5.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax5.set_xlabel('예측 오차')
            ax5.set_ylabel('빈도')
            ax5.set_title('앙상블 모델 오차 분포')
            ax5.grid(True, alpha=0.3)
            
            # 통계 정보 추가
            mean_error = eval_df['ensemble_오차'].mean()
            std_error = eval_df['ensemble_오차'].std()
            ax5.text(0.02, 0.98, f'평균: {mean_error:.2f}\n표준편차: {std_error:.2f}',
                    transform=ax5.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 6. F1 Score 비교
        ax6 = axes[2, 1]
        f1_values = [results[m]['F1'] for m in model_names]
        bars = ax6.bar(model_names, f1_values, color='orange')
        
        ax6.set_xlabel('모델')
        ax6.set_ylabel('F1 Score')
        ax6.set_title('급증 감지 F1 Score')
        ax6.set_xticklabels([m.upper() for m in model_names], rotation=45)
        ax6.grid(True, alpha=0.3)
        
        for bar, f1 in zip(bars, f1_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 그래프 저장
        output_file = f'evaluation_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 시각화 저장: {output_file}")
        
        plt.show()
        
        return fig

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("반도체 물류 예측 모델 평가 프로그램")
    print("=" * 60)
    print()
    
    # 평가기 초기화 - 경로 자동 설정
    evaluator = ModelEvaluator()
    
    # 1. 학습된 모델 로드
    if not evaluator.load_models():
        print("⚠ 모델을 로드할 수 없습니다. 모델 디렉토리를 확인하세요.")
        print("학습v5.py를 실행하여 모델을 먼저 학습시켜주세요.")
        return None, None  # 두 개 반환
    
    # 2. 평가 데이터 로드
    df = evaluator.load_evaluation_data()
    if df is None:
        print("⚠ 데이터를 로드할 수 없습니다.")
        return None, None  # 두 개 반환
    
    # 3. 특징 생성
    df = evaluator.create_features(df)
    
    # 4. 시퀀스 데이터 준비
    X, y, timestamps = evaluator.prepare_sequences(df)
    
    # 5. 모델 평가
    results, predictions = evaluator.evaluate_models(X, y, timestamps)
    
    # 6. 결과 저장
    eval_df = evaluator.save_evaluation_results(y, predictions, timestamps, df)
    
    # 7. 시각화
    fig = evaluator.visualize_results(eval_df, results)
    
    # 8. 최종 요약 출력
    print("\n" + "=" * 60)
    print("평가 완료 - 최종 요약")
    print("=" * 60)
    
    if 'ensemble' in results:
        best_model = min(results.items(), key=lambda x: x[1]['MAE'])
        print(f"\n최고 성능 모델 (MAE 기준): {best_model[0].upper()}")
        print(f"  - MAE: {best_model[1]['MAE']:.2f}")
        print(f"  - RMSE: {best_model[1]['RMSE']:.2f}")
        print(f"  - R²: {best_model[1]['R2']:.4f}")
        
        print(f"\n앙상블 모델 성능:")
        print(f"  - MAE: {results['ensemble']['MAE']:.2f}")
        print(f"  - RMSE: {results['ensemble']['RMSE']:.2f}")
        print(f"  - R²: {results['ensemble']['R2']:.4f}")
        print(f"  - 1400+ Recall: {results['ensemble']['Recall']*100:.2f}%")
    
    print("\n✓ 모든 평가가 완료되었습니다!")
    print("✓ 결과 파일들이 현재 디렉토리에 저장되었습니다.")
    
    return eval_df, results

if __name__ == "__main__":
    eval_df, results = main()