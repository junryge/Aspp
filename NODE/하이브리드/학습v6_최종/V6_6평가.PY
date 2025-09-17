"""
📊 모든 모델 비교 평가 시스템
============================
6개 모델 전체를 평가하고 비교
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
import os
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class MultiModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """다중 모델 평가기 초기화"""
        print("="*80)
        print("📊 다중 모델 평가 시스템")
        print("="*80)
        
        # 스케일러 로드
        with open(f'{scaler_path}feature_scaler.pkl', 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(f'{scaler_path}target_scaler.pkl', 'rb') as f:
            self.target_scaler = pickle.load(f)
        with open(f'{scaler_path}config.json', 'r') as f:
            config = json.load(f)
            self.seq_len = config['seq_len']
            self.pred_len = config['pred_len']
            self.feature_columns = config['feature_columns']
        print(f"✅ 스케일러 로드 완료")
        
        self.models = {}
        self.results = {}
        
    def load_all_models(self, model_dir='models/'):
        """모든 모델 로드"""
        print(f"\n📁 모델 로딩...")
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        
        for model_file in model_files:
            model_name = model_file.replace('.keras', '')
            model_path = os.path.join(model_dir, model_file)
            
            try:
                self.models[model_name] = tf.keras.models.load_model(model_path)
                print(f"  ✅ {model_name} 로드 완료")
            except Exception as e:
                print(f"  ❌ {model_name} 로드 실패: {e}")
        
        print(f"\n총 {len(self.models)}개 모델 로드 완료")
        return self.models
    
    def load_test_data(self, filepath):
        """테스트 데이터 로드"""
        print(f"\n📂 평가 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        print(f"  원본: {df.shape[0]:,}행")
        
        # 0값 제거
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        # 시간 변환
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        print(f"  유효: {df.shape[0]:,}행")
        return df
    
    def create_features(self, df):
        """특성 생성"""
        # 기본 특성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        
        # 시간
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
        df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        
        # 이동평균
        for w in [10, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
        
        # 변화율
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        
        return df
    
    def evaluate_single_model(self, model_name, model, df, save_details=False):
        """단일 모델 평가"""
        print(f"\n🎯 {model_name} 평가 중...")
        
        # 예측 가능 범위
        start_idx = self.seq_len
        end_idx = len(df) - self.pred_len
        
        predictions = []
        actuals = []
        timestamps = []
        detailed_results = []
        
        # 배치 예측
        batch_size = 500
        for i in range(start_idx, end_idx, batch_size):
            batch_end = min(i + batch_size, end_idx)
            
            # 배치 데이터 준비
            X_batch = []
            for j in range(i, batch_end):
                seq_data = df.iloc[j-self.seq_len:j][self.feature_columns].values
                X_batch.append(seq_data)
            
            if len(X_batch) == 0:
                continue
                
            # 스케일링
            X_batch = np.array(X_batch)
            X_batch_scaled = []
            for seq in X_batch:
                seq_scaled = self.feature_scaler.transform(seq)
                X_batch_scaled.append(seq_scaled)
            X_batch_scaled = np.array(X_batch_scaled)
            
            # 예측
            preds = model.predict(X_batch_scaled, verbose=0)
            
            # 회귀 예측값 추출
            if isinstance(preds, list):
                y_pred_scaled = preds[0].flatten()
            else:
                y_pred_scaled = preds.flatten()
            
            # 역변환
            y_pred = self.target_scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)).flatten()
            
            # 실제값 수집
            for k, j in enumerate(range(i, batch_end)):
                actual_time = df.iloc[j]['CURRTIME']
                pred_time = actual_time + timedelta(minutes=self.pred_len)
                
                actual_idx = j + self.pred_len
                if actual_idx < len(df):
                    actual_value = df.iloc[actual_idx]['TOTALCNT']
                    pred_value = y_pred[k]
                    
                    predictions.append(pred_value)
                    actuals.append(actual_value)
                    timestamps.append(pred_time)
                    
                    if save_details:
                        detailed_results.append({
                            '모델': model_name,
                            '예측시점': actual_time.strftime('%Y-%m-%d %H:%M'),
                            '예측대상시간': pred_time.strftime('%Y-%m-%d %H:%M'),
                            '실제값': actual_value,
                            '예측값': round(pred_value),
                            '오차': round(pred_value - actual_value),
                            '오차율(%)': round(abs(pred_value - actual_value) / actual_value * 100, 2)
                        })
        
        # 성능 계산
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        results = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            '정확도(%)': 100 - mape,
            '예측개수': len(predictions)
        }
        
        print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
        
        if save_details:
            return results, detailed_results
        return results
    
    def evaluate_all_models(self, test_file):
        """모든 모델 평가"""
        # 데이터 로드
        df = self.load_test_data(test_file)
        df = self.create_features(df)
        
        print("\n" + "="*80)
        print("📊 모델별 평가 시작")
        print("="*80)
        
        all_results = {}
        all_details = []
        
        # 각 모델 평가
        for model_name, model in self.models.items():
            results = self.evaluate_single_model(model_name, model, df, save_details=False)
            all_results[model_name] = results
        
        # 결과 DataFrame 생성
        results_df = pd.DataFrame(all_results).T
        results_df = results_df.sort_values('R2', ascending=False)
        
        print("\n" + "="*80)
        print("📊 모델 성능 비교")
        print("="*80)
        
        # 성능 테이블 출력
        print(f"\n{'모델':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE(%)':>8} {'정확도(%)':>8}")
        print("-" * 60)
        
        for model_name, row in results_df.iterrows():
            print(f"{model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                  f"{row['R2']:8.4f} {row['MAPE']:8.2f} {row['정확도(%)']:8.2f}")
        
        # 최고 성능
        print("\n🏆 최고 성능:")
        print(f"  MAE 최저: {results_df['MAE'].idxmin()} ({results_df['MAE'].min():.2f})")
        print(f"  RMSE 최저: {results_df['RMSE'].idxmin()} ({results_df['RMSE'].min():.2f})")
        print(f"  R² 최고: {results_df['R2'].idxmax()} ({results_df['R2'].max():.4f})")
        print(f"  MAPE 최저: {results_df['MAPE'].idxmin()} ({results_df['MAPE'].min():.2f}%)")
        print(f"  정확도 최고: {results_df['정확도(%)'].idxmax()} ({results_df['정확도(%)'].max():.2f}%)")
        
        # CSV 저장
        output_file = f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, encoding='utf-8-sig')
        print(f"\n💾 비교 결과 저장: {output_file}")
        
        # 시각화
        self.visualize_comparison(results_df)
        
        # 최고 모델로 상세 평가
        best_model_name = results_df['R2'].idxmax()
        print(f"\n📝 최고 모델({best_model_name}) 상세 평가...")
        
        _, detailed = self.evaluate_single_model(
            best_model_name, 
            self.models[best_model_name], 
            df, 
            save_details=True
        )
        
        # 상세 결과 저장
        if detailed:
            detailed_df = pd.DataFrame(detailed)
            detail_file = f'best_model_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            
            # 처음 100개와 마지막 100개만 저장
            sample_df = pd.concat([detailed_df.head(100), detailed_df.tail(100)])
            sample_df.to_csv(detail_file, index=False, encoding='utf-8-sig')
            print(f"💾 상세 예측 결과 저장: {detail_file}")
            
            # 샘플 출력
            print(f"\n📝 {best_model_name} 예측 샘플 (최근 10개):")
            print("="*100)
            print(f"{'예측시점':^19} | {'대상시간':^19} | {'실제값':>8} | {'예측값':>8} | {'오차':>8} | {'오차율':>8}")
            print("-"*100)
            
            for _, row in sample_df.tail(10).iterrows():
                print(f"{row['예측시점']:^19} | {row['예측대상시간']:^19} | "
                      f"{row['실제값']:8.0f} | {row['예측값']:8.0f} | "
                      f"{row['오차']:8.0f} | {row['오차율(%)']:7.1f}%")
        
        return results_df
    
    def visualize_comparison(self, results_df):
        """모델 비교 시각화"""
        print("\n📈 비교 그래프 생성 중...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. MAE 비교
        axes[0, 0].bar(results_df.index, results_df['MAE'], color='skyblue')
        axes[0, 0].set_title('MAE Comparison')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_xticklabels(results_df.index, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. R² 비교
        axes[0, 1].bar(results_df.index, results_df['R2'], color='lightgreen')
        axes[0, 1].set_title('R² Score Comparison')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].set_xticklabels(results_df.index, rotation=45)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. MAPE 비교
        axes[1, 0].bar(results_df.index, results_df['MAPE'], color='salmon')
        axes[1, 0].set_title('MAPE Comparison (%)')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].set_xticklabels(results_df.index, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 정확도 비교
        axes[1, 1].bar(results_df.index, results_df['정확도(%)'], color='gold')
        axes[1, 1].set_title('Accuracy Comparison (%)')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_xticklabels(results_df.index, rotation=45)
        axes[1, 1].axhline(y=95, color='g', linestyle='--', alpha=0.5, label='95% line')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f'model_comparison_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        print(f"📊 비교 그래프 저장: {plot_file}")
        plt.show()

def main():
    """메인 실행"""
    # 평가기 생성
    evaluator = MultiModelEvaluator()
    
    # 모든 모델 로드
    models = evaluator.load_all_models('models/')
    
    if not models:
        print("❌ 모델이 없습니다!")
        return
    
    # 테스트 파일 찾기
    test_files = [
        'data/20250731_to20250806.csv',
        'data/test_data.csv',
        '/mnt/user-data/uploads/test.csv'
    ]
    
    test_file = None
    for file in test_files:
        if os.path.exists(file):
            test_file = file
            break
    
    if not test_file:
        print("❌ 테스트 데이터를 찾을 수 없습니다!")
        return
    
    # 모든 모델 평가
    results = evaluator.evaluate_all_models(test_file)
    
    print("\n✅ 모든 모델 평가 완료!")
    print("="*80)

if __name__ == "__main__":
    main()