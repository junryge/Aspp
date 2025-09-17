"""
📊 모든 모델 예측값 비교 평가
============================
6개 모델의 예측값을 하나의 CSV에 모두 저장
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
import os
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
tf.keras.config.enable_unsafe_deserialization()

class CompleteModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """평가기 초기화"""
        print("="*80)
        print("📊 전체 모델 예측 평가 시스템")
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
        
    def load_all_models(self, model_dir='models/'):
        """모든 모델 로드"""
        print(f"\n📁 모델 로딩...")
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        
        for model_file in model_files:
            model_name = model_file.replace('.keras', '')
            model_path = os.path.join(model_dir, model_file)
            
            try:
                self.models[model_name] = tf.keras.models.load_model(
                    model_path, safe_mode=False
                )
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
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
        df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        
        for w in [10, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
        
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        
        return df
    
    def evaluate_all_models(self, test_file):
        """모든 모델 평가 및 예측값 저장"""
        
        # 데이터 로드
        df = self.load_test_data(test_file)
        df = self.create_features(df)
        
        # 예측 가능 범위
        start_idx = self.seq_len
        end_idx = len(df) - self.pred_len
        total = end_idx - start_idx
        
        print(f"\n🔮 예측 시작...")
        print(f"  시퀀스: {self.seq_len}분 → 예측: {self.pred_len}분 후")
        print(f"  예측 개수: {total:,}개")
        
        # 모든 예측을 저장할 DataFrame 준비
        all_predictions = pd.DataFrame()
        
        # 시간 정보 수집
        timestamps_pred = []  # 예측 시점
        timestamps_target = []  # 예측 대상 시간
        actuals = []  # 실제값
        
        print("\n📊 시간 정보 수집 중...")
        for i in range(start_idx, end_idx):
            pred_time = df.iloc[i]['CURRTIME']
            target_time = pred_time + timedelta(minutes=self.pred_len)
            
            actual_idx = i + self.pred_len
            if actual_idx < len(df):
                timestamps_pred.append(pred_time)
                timestamps_target.append(target_time)
                actuals.append(df.iloc[actual_idx]['TOTALCNT'])
        
        # 기본 정보 저장
        all_predictions['예측시점'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_pred]
        all_predictions['예측대상시간'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_target]
        all_predictions['실제값'] = actuals
        
        print(f"  예측할 데이터: {len(all_predictions)}개")
        
        # 각 모델별 예측
        model_metrics = {}
        
        for model_name, model in self.models.items():
            print(f"\n🎯 {model_name} 예측 중...")
            predictions = []
            
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
                
                if isinstance(preds, list):
                    y_pred_scaled = preds[0].flatten()
                else:
                    y_pred_scaled = preds.flatten()
                
                # 역변환
                y_pred = self.target_scaler.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)).flatten()
                
                # 수집
                for k in range(len(y_pred)):
                    actual_idx = i - start_idx + k
                    if actual_idx < len(all_predictions):
                        predictions.append(y_pred[k])
                
                if len(predictions) % 2000 == 0:
                    print(f"    {len(predictions):,}/{len(all_predictions):,} 완료")
            
            # 예측값 컬럼 추가
            predictions = predictions[:len(all_predictions)]  # 길이 맞추기
            all_predictions[f'{model_name}_예측'] = [round(p) for p in predictions]
            all_predictions[f'{model_name}_오차'] = all_predictions[f'{model_name}_예측'] - all_predictions['실제값']
            all_predictions[f'{model_name}_오차율(%)'] = round(
                abs(all_predictions[f'{model_name}_오차']) / all_predictions['실제값'] * 100, 2
            )
            
            # 성능 계산
            mae = mean_absolute_error(all_predictions['실제값'], predictions)
            rmse = np.sqrt(mean_squared_error(all_predictions['실제값'], predictions))
            r2 = r2_score(all_predictions['실제값'], predictions)
            mape = np.mean(abs(all_predictions[f'{model_name}_오차']) / all_predictions['실제값']) * 100
            
            model_metrics[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                '정확도(%)': 100 - mape
            }
            
            print(f"  ✅ {model_name} 완료: MAE={mae:.2f}, R²={r2:.4f}, 정확도={100-mape:.2f}%")
        
        # 앙상블 예측 추가
        print("\n🔮 앙상블 예측 생성...")
        
        # 사용 가능한 모델들의 예측값으로 앙상블
        model_weights = {
            'PatchTST': 0.30,      # 최고 성능 30%
            'ExtremeNet': 0.25,    # 두번째 25%
            'StableLSTM': 0.20,    # 세번째 20%
            'SpikeDetector': 0.15, # 네번째 15%
            'GoldenRule': 0.10     # 다섯째 10%
        }
        
        # 앙상블 계산
        ensemble_pred = np.zeros(len(all_predictions))
        total_weight = 0
        
        for model_name in self.models.keys():
            if model_name in model_weights:
                weight = model_weights[model_name]
                ensemble_pred += all_predictions[f'{model_name}_예측'] * weight
                total_weight += weight
            else:
                # 가중치 없는 모델은 균등 가중치
                ensemble_pred += all_predictions[f'{model_name}_예측'] * 0.1
                total_weight += 0.1
        
        # 정규화 (가중치 합이 1이 되도록)
        if total_weight > 0:
            ensemble_pred = ensemble_pred / total_weight
        
        # 앙상블 결과 추가
        all_predictions['앙상블_예측'] = [round(p) for p in ensemble_pred]
        all_predictions['앙상블_오차'] = all_predictions['앙상블_예측'] - all_predictions['실제값']
        all_predictions['앙상블_오차율(%)'] = round(
            abs(all_predictions['앙상블_오차']) / all_predictions['실제값'] * 100, 2
        )
        
        # 앙상블 성능
        ensemble_mae = mean_absolute_error(all_predictions['실제값'], ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(all_predictions['실제값'], ensemble_pred))
        ensemble_r2 = r2_score(all_predictions['실제값'], ensemble_pred)
        ensemble_mape = np.mean(abs(all_predictions['앙상블_오차']) / all_predictions['실제값']) * 100
        
        model_metrics['앙상블'] = {
            'MAE': ensemble_mae,
            'RMSE': ensemble_rmse,
            'R2': ensemble_r2,
            'MAPE': ensemble_mape,
            '정확도(%)': 100 - ensemble_mape
        }
        
        print(f"\n✅ 앙상블 완료: MAE={ensemble_mae:.2f}, R²={ensemble_r2:.4f}, 정확도={100-ensemble_mape:.2f}%")
        
        # CSV 저장
        output_file = f'all_models_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        all_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 모든 모델 예측값 저장: {output_file}")
        
        # 성능 요약
        print("\n" + "="*80)
        print("📊 모델 성능 요약")
        print("="*80)
        
        metrics_df = pd.DataFrame(model_metrics).T
        metrics_df = metrics_df.sort_values('R2', ascending=False)
        
        print(f"\n{'모델':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE(%)':>8} {'정확도(%)':>10}")
        print("-" * 65)
        
        for model_name, row in metrics_df.iterrows():
            if model_name == '앙상블':
                # 앙상블 강조
                print(f"{'🎯 ' + model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                      f"{row['R2']:8.4f} {row['MAPE']:8.2f} {row['정확도(%)']:10.2f} ⭐")
            else:
                print(f"{model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                      f"{row['R2']:8.4f} {row['MAPE']:8.2f} {row['정확도(%)']:10.2f}")
        
        # 성능 비교 CSV 저장
        metrics_file = f'model_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        metrics_df.to_csv(metrics_file, encoding='utf-8-sig')
        print(f"\n💾 성능 비교 저장: {metrics_file}")
        
        # 샘플 출력
        print("\n📝 예측 샘플 (처음 5개)")
        print("="*120)
        
        # 컬럼 선택 (앙상블 포함)
        display_cols = ['예측시점', '예측대상시간', '실제값', '앙상블_예측']
        for model_name in self.models.keys():
            display_cols.append(f'{model_name}_예측')
        
        # 컬럼이 실제로 존재하는지 확인
        display_cols = [col for col in display_cols if col in all_predictions.columns]
        
        print(all_predictions[display_cols].head())
        
        print("\n📝 예측 샘플 (마지막 5개)")
        print("="*120)
        print(all_predictions[display_cols].tail())
        
        # 최고 모델 강조
        best_model = metrics_df.index[0]
        print(f"\n🏆 최고 성능:")
        if best_model == '앙상블':
            print(f"  📌 앙상블 (5개 모델 조합)")
            print(f"  MAE: {metrics_df.loc[best_model, 'MAE']:.2f}")
            print(f"  R²: {metrics_df.loc[best_model, 'R2']:.4f}")
            print(f"  정확도: {metrics_df.loc[best_model, '정확도(%)']:.2f}%")
            print(f"\n  앙상블 구성:")
            print(f"    - PatchTST: 30%")
            print(f"    - ExtremeNet: 25%")
            print(f"    - StableLSTM: 20%")
            print(f"    - SpikeDetector: 15%")
            print(f"    - GoldenRule: 10%")
        else:
            print(f"  📌 {best_model}")
            print(f"  MAE: {metrics_df.loc[best_model, 'MAE']:.2f}")
            print(f"  R²: {metrics_df.loc[best_model, 'R2']:.4f}")
            print(f"  정확도: {metrics_df.loc[best_model, '정확도(%)']:.2f}%")
        
        # 간단한 통계
        print("\n📊 예측 통계:")
        print(f"  총 예측 건수: {len(all_predictions):,}개")
        print(f"  평가 기간: {all_predictions['예측시점'].iloc[0]} ~ {all_predictions['예측시점'].iloc[-1]}")
        
        return all_predictions, metrics_df

def main():
    """메인 실행"""
    
    # 평가기 생성
    evaluator = CompleteModelEvaluator()
    
    # 모든 모델 로드
    models = evaluator.load_all_models('models/')
    
    if not models:
        print("❌ 모델이 없습니다!")
        return
    
    # 테스트 파일
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
    
    # 평가 실행
    all_predictions, metrics = evaluator.evaluate_all_models(test_file)
    
    print("\n✅ 평가 완료!")
    print(f"📁 저장된 파일:")
    print(f"  1. all_models_predictions_YYYYMMDD.csv - 모든 모델 예측값")
    print(f"  2. model_performance_YYYYMMDD.csv - 성능 비교")
    print("="*80)

if __name__ == "__main__":
    main()