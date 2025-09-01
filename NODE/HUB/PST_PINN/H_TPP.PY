import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class HUBROOMEvaluator:
    """HUBROOM 예측 모델 평가 클래스"""
    
    def __init__(self, data_path='202509.csv'):
        self.data_path = data_path
        self.seq_len = 20
        self.pred_len = 10
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.critical_threshold = 300
        
        # 모델 경로
        self.patchtst_path = './checkpoints/PatchTST_best.h5'
        self.pinn_path = './checkpoints/PatchTST_PINN_best.h5'
        
        # 스케일러 로드
        self.scaler_X = self.load_scaler('scaler_X.pkl')
        self.scaler_y = self.load_scaler('scaler_y.pkl')
        self.scaler_physics = self.load_scaler('scaler_physics.pkl')
        
        # 물리 컬럼
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'
        ]
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'
        ]
    
    def load_scaler(self, filename):
        """스케일러 로드"""
        filepath = f'./checkpoints/{filename}'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                print(f"✅ {filename} 로드 완료")
                return pickle.load(f)
        else:
            print(f"⚠️ {filename}이 없습니다. save_scalers.py를 먼저 실행하세요!")
            return None
    
    def prepare_data(self):
        """2025년 9월 데이터 준비"""
        print("📂 2025년 9월 데이터 로드 중...")
        df = pd.read_csv(self.data_path)
        
        # 시간 컬럼 처리
        time_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 결측치 처리
        df = df.fillna(method='ffill').fillna(0)
        
        # 숫자형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"✅ 데이터 로드 완료: {len(df)} 행")
        print(f"📅 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        
        return df, numeric_cols
    
    def create_evaluation_sequences(self, df, numeric_cols):
        """평가용 시퀀스 생성"""
        X_list = []
        y_actual_list = []
        y_time_list = []
        physics_list = []
        timestamps = []
        
        data = df[numeric_cols].values
        target_idx = numeric_cols.index(self.target_col)
        
        # 사용 가능한 컬럼 확인
        available_inflow = [col for col in self.inflow_cols if col in numeric_cols]
        available_outflow = [col for col in self.outflow_cols if col in numeric_cols]
        
        print(f"\n📊 시퀀스 생성 중...")
        total_sequences = len(data) - self.seq_len - self.pred_len + 1
        
        # 시퀀스 생성
        for i in range(total_sequences):
            # 입력 시퀀스 (과거 20분)
            X_seq = data[i:i+self.seq_len]
            
            # 실제 값 (미래 10분)
            y_actual = data[i+self.seq_len:i+self.seq_len+self.pred_len, target_idx]
            
            # 타임스탬프
            start_time = df['timestamp'].iloc[i+self.seq_len-1]
            pred_times = [start_time + timedelta(minutes=j+1) for j in range(self.pred_len)]
            
            # 물리 데이터 (현재 상태)
            current_state = data[i+self.seq_len-1]
            current_hubroom = current_state[target_idx]
            
            inflow_sum = sum([current_state[numeric_cols.index(col)] 
                            for col in available_inflow])
            outflow_sum = sum([current_state[numeric_cols.index(col)] 
                             for col in available_outflow])
            
            physics = np.array([current_hubroom, inflow_sum, outflow_sum])
            
            X_list.append(X_seq)
            y_actual_list.append(y_actual)
            y_time_list.append(pred_times)
            physics_list.append(physics)
            timestamps.append(start_time)
        
        print(f"✅ 시퀀스 생성 완료: {len(X_list)}개")
        
        return (np.array(X_list), np.array(y_actual_list), 
                np.array(physics_list), timestamps, y_time_list)
    
    def load_models(self):
        """학습된 모델 로드"""
        print("\n🤖 모델 로드 중...")
        
        models = {}
        
        # PatchTST 모델 로드
        if os.path.exists(self.patchtst_path):
            try:
                models['PatchTST'] = tf.keras.models.load_model(
                    self.patchtst_path,
                    compile=False
                )
                print("✅ PatchTST 모델 로드 완료")
            except:
                # 가중치만 로드하는 방식 시도
                print("⚠️ 전체 모델 로드 실패, 가중치만 로드 시도...")
        else:
            print("❌ PatchTST 모델 파일이 없습니다.")
        
        # PatchTST+PINN 모델 로드
        if os.path.exists(self.pinn_path):
            try:
                models['PatchTST_PINN'] = tf.keras.models.load_model(
                    self.pinn_path,
                    compile=False
                )
                print("✅ PatchTST+PINN 모델 로드 완료")
            except:
                print("⚠️ 전체 모델 로드 실패, 가중치만 로드 시도...")
        else:
            print("❌ PatchTST+PINN 모델 파일이 없습니다.")
        
        return models
    
    def predict_and_evaluate(self, models, X, y_actual, physics_data, timestamps):
        """모델별 예측 및 평가"""
        results = {}
        
        # 데이터 정규화
        n_samples, seq_len, n_features = X.shape
        
        # 스케일러 확인
        if self.scaler_X is None or self.scaler_y is None:
            print("❌ 스케일러가 로드되지 않았습니다!")
            return results
        
        X_scaled = self.scaler_X.transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
        physics_scaled = self.scaler_physics.transform(physics_data) if self.scaler_physics else physics_data
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"📊 {model_name} 모델 평가")
            print(f"{'='*60}")
            
            try:
                # 예측
                if model_name == 'PatchTST_PINN':
                    # PINN 모델은 물리 데이터도 필요
                    y_pred_scaled = model.predict([X_scaled, physics_scaled], verbose=0)
                else:
                    # PatchTST는 시퀀스 데이터만
                    y_pred_scaled = model.predict(X_scaled, verbose=0)
                
                # 역정규화 (10분 후 예측값)
                y_pred_10min = self.scaler_y.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).flatten()
                
                # 실제값은 10분 후 값만 추출
                y_true_10min = y_actual[:, -1]  # 마지막 시점 (10분 후)
                
                # 메트릭 계산
                mae = mean_absolute_error(y_true_10min, y_pred_10min)
                mse = mean_squared_error(y_true_10min, y_pred_10min)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true_10min, y_pred_10min)
                
                # 300 이상 예측 분석
                over_300_pred = np.sum(y_pred_10min >= 300)
                over_300_true = np.sum(y_true_10min >= 300)
                
                # 300 이상일 때의 정확도
                mask_300 = y_true_10min >= 300
                if np.sum(mask_300) > 0:
                    mae_300 = mean_absolute_error(y_true_10min[mask_300], y_pred_10min[mask_300])
                    acc_300 = np.sum((y_pred_10min >= 300) & (y_true_10min >= 300)) / np.sum(mask_300)
                else:
                    mae_300 = 0
                    acc_300 = 0
                
                # 결과 저장
                results[model_name] = {
                    'y_true': y_true_10min,
                    'y_pred': y_pred_10min,
                    'timestamps': timestamps,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'over_300_pred': over_300_pred,
                    'over_300_true': over_300_true,
                    'mae_300': mae_300,
                    'acc_300': acc_300
                }
                
                # 성능 출력
                print(f"\n📈 전체 성능:")
                print(f"  - MAE: {mae:.4f}")
                print(f"  - RMSE: {rmse:.4f}")
                print(f"  - R²: {r2:.4f}")
                
                print(f"\n🚨 300 이상 예측 분석:")
                print(f"  - 실제 300 이상: {over_300_true}개")
                print(f"  - 예측 300 이상: {over_300_pred}개")
                print(f"  - 300 이상일 때 MAE: {mae_300:.4f}")
                print(f"  - 300 감지 정확도: {acc_300:.2%}")
                
                # 샘플 출력
                print(f"\n📝 예측 샘플 (처음 10개):")
                for i in range(min(10, len(y_pred_10min))):
                    status = "⚠️ 경고" if y_pred_10min[i] >= 300 else "✅ 정상"
                    print(f"  [{i+1}] 실제: {y_true_10min[i]:.1f}, 예측: {y_pred_10min[i]:.1f} {status}")
                
            except Exception as e:
                print(f"❌ {model_name} 예측 중 오류 발생: {e}")
                continue
        
        return results
    
    def visualize_results(self, results):
        """결과 시각화"""
        if not results:
            print("시각화할 결과가 없습니다.")
            return
        
        # 모델별 비교 그래프
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('HUBROOM 반송량 예측 모델 비교 (2025년 9월)', fontsize=16)
        
        # 1. 시계열 예측 비교 (처음 100개)
        ax1 = axes[0, 0]
        for model_name, result in results.items():
            ax1.plot(result['y_true'][:100], label=f'실제값', alpha=0.7, linewidth=2)
            ax1.plot(result['y_pred'][:100], label=f'{model_name} 예측', alpha=0.7, linestyle='--')
        ax1.axhline(y=300, color='red', linestyle=':', label='위험 임계값 (300)')
        ax1.set_title('시계열 예측 비교 (처음 100개)')
        ax1.set_xlabel('시간 인덱스')
        ax1.set_ylabel('HUBROOM 반송량')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 산점도
        ax2 = axes[0, 1]
        colors = ['blue', 'green', 'orange']
        for idx, (model_name, result) in enumerate(results.items()):
            ax2.scatter(result['y_true'], result['y_pred'], 
                       alpha=0.5, label=model_name, color=colors[idx % len(colors)])
        ax2.plot([0, 600], [0, 600], 'r--', alpha=0.5, label='Perfect Prediction')
        ax2.axvline(x=300, color='red', linestyle=':', alpha=0.5)
        ax2.axhline(y=300, color='red', linestyle=':', alpha=0.5)
        ax2.set_title('예측값 vs 실제값')
        ax2.set_xlabel('실제값')
        ax2.set_ylabel('예측값')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 오차 분포
        ax3 = axes[1, 0]
        for model_name, result in results.items():
            errors = result['y_pred'] - result['y_true']
            ax3.hist(errors, bins=50, alpha=0.5, label=f'{model_name} (MAE: {result["mae"]:.2f})')
        ax3.set_title('예측 오차 분포')
        ax3.set_xlabel('예측 오차')
        ax3.set_ylabel('빈도')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 성능 메트릭 비교
        ax4 = axes[1, 1]
        metrics = ['MAE', 'RMSE', 'R²', 'MAE@300+']
        model_names = list(results.keys())
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for idx, model_name in enumerate(model_names):
            values = [
                results[model_name]['mae'],
                results[model_name]['rmse'],
                results[model_name]['r2'] * 100,  # R²를 백분율로
                results[model_name]['mae_300']
            ]
            ax4.bar(x + idx * width, values, width, label=model_name)
        
        ax4.set_title('모델 성능 메트릭 비교')
        ax4.set_xlabel('메트릭')
        ax4.set_ylabel('값')
        ax4.set_xticks(x + width / 2)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hubroom_evaluation_202509.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 300 이상 예측 상세 분석
        self.analyze_critical_predictions(results)
    
    def analyze_critical_predictions(self, results):
        """300 이상 예측 상세 분석"""
        print("\n" + "="*60)
        print("🚨 300 이상 예측 상세 분석")
        print("="*60)
        
        for model_name, result in results.items():
            print(f"\n📊 {model_name} 모델:")
            
            y_true = result['y_true']
            y_pred = result['y_pred']
            
            # 300 이상 케이스 분석
            true_over_300 = y_true >= 300
            pred_over_300 = y_pred >= 300
            
            # 혼동 행렬
            tp = np.sum(true_over_300 & pred_over_300)  # True Positive
            fp = np.sum(~true_over_300 & pred_over_300)  # False Positive
            tn = np.sum(~true_over_300 & ~pred_over_300)  # True Negative
            fn = np.sum(true_over_300 & ~pred_over_300)  # False Negative
            
            # 메트릭 계산
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
                
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
                
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            print(f"  - True Positive (정확히 예측한 위험): {tp}개")
            print(f"  - False Positive (잘못된 경보): {fp}개")
            print(f"  - True Negative (정확히 예측한 정상): {tn}개")
            print(f"  - False Negative (놓친 위험): {fn}개")
            print(f"  - Precision (정밀도): {precision:.2%}")
            print(f"  - Recall (재현율): {recall:.2%}")
            print(f"  - F1-Score: {f1:.2%}")
            
            # 극단값 분석
            extreme_cases = y_true > 400
            if np.sum(extreme_cases) > 0:
                extreme_mae = mean_absolute_error(y_true[extreme_cases], y_pred[extreme_cases])
                print(f"  - 400 초과 극단값 MAE: {extreme_mae:.2f}")
    
    def save_predictions(self, results, output_path='predictions_202509.csv'):
        """예측 결과를 CSV로 저장"""
        print(f"\n💾 예측 결과 저장 중...")
        
        # 첫 번째 모델의 타임스탬프 사용
        first_model = list(results.keys())[0]
        timestamps = results[first_model]['timestamps']
        
        # 데이터프레임 생성
        df_results = pd.DataFrame({
            'timestamp': timestamps,
            'actual': results[first_model]['y_true']
        })
        
        # 각 모델의 예측값 추가
        for model_name, result in results.items():
            df_results[f'pred_{model_name}'] = result['y_pred']
            df_results[f'error_{model_name}'] = result['y_pred'] - result['y_true']
        
        # 300 이상 플래그
        df_results['is_critical'] = df_results['actual'] >= 300
        
        # 저장
        df_results.to_csv(output_path, index=False)
        print(f"✅ 예측 결과 저장 완료: {output_path}")
        
        # 요약 통계
        print(f"\n📊 저장된 데이터 요약:")
        print(f"  - 전체 예측 수: {len(df_results)}개")
        print(f"  - 300 이상 실제값: {df_results['is_critical'].sum()}개")
        print(f"  - 기간: {df_results['timestamp'].min()} ~ {df_results['timestamp'].max()}")

def main():
    """메인 실행 함수"""
    print("="*80)
    print("🏭 HUBROOM 반송량 예측 평가 시스템")
    print("📅 대상: 2025년 9월 데이터")
    print("="*80)
    
    # 평가기 생성
    evaluator = HUBROOMEvaluator()
    
    # 스케일러 확인
    if evaluator.scaler_X is None:
        print("\n❌ 스케일러가 없습니다. save_scalers.py를 먼저 실행하세요!")
        return
    
    try:
        # 1. 데이터 준비
        df, numeric_cols = evaluator.prepare_data()
        
        # 2. 시퀀스 생성
        X, y_actual, physics_data, timestamps, y_times = evaluator.create_evaluation_sequences(df, numeric_cols)
        
        # 3. 모델 로드
        models = evaluator.load_models()
        
        if not models:
            print("\n❌ 로드된 모델이 없습니다!")
            return
        
        # 4. 예측 및 평가
        results = evaluator.predict_and_evaluate(models, X, y_actual, physics_data, timestamps)
        
        # 5. 결과 시각화
        evaluator.visualize_results(results)
        
        # 6. 예측 결과 저장
        evaluator.save_predictions(results)
        
        # 7. 최종 요약
        print("\n" + "="*80)
        print("📊 최종 모델 비교 요약")
        print("="*80)
        
        print(f"\n{'모델':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'300+ 정확도':<15}")
        print("-"*65)
        
        for model_name, result in results.items():
            print(f"{model_name:<20} {result['mae']:<10.2f} {result['rmse']:<10.2f} "
                  f"{result['r2']:<10.2f} {result['acc_300']:<15.2%}")
        
        print("\n✅ 평가 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()