"""
📊 V6.6 모든 모델 예측값 비교 평가 (극단값 개선 버전)
========================================================
6개 모델의 예측값을 하나의 CSV에 모두 저장 + 1700+ 극단값 개선
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

# ====================== 극단값 보정 클래스 추가 ======================
class ExtremeValueBooster:
    """1700+ 구간 극단적 예측 개선"""
    
    def __init__(self):
        # M14B 매핑 테이블
        self.m14b_thresholds = {
            450: 1700,
            500: 1750, 
            550: 1850
        }
        
    def boost_prediction(self, pred, m14b_value, m14a_value=None):
        """M14B 값에 따른 극단적 부스팅"""
        boosted = pred
        
        # M14B가 극도로 높으면 강제 상향
        if m14b_value > 550:
            boosted = max(pred * 1.45, 1850)
        elif m14b_value > 500:
            boosted = max(pred * 1.35, 1750)
        elif m14b_value > 450:
            boosted = max(pred * 1.25, 1700)
        elif m14b_value > 400:
            boosted = max(pred * 1.15, 1650)
        elif m14b_value > 350:
            boosted = max(pred * 1.08, 1550)
            
        # 황금 패턴 추가 부스팅
        if m14b_value > 300 and m14a_value and m14a_value < 80:
            boosted *= 1.1
            
        return boosted

class CompleteModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """평가기 초기화"""
        print("="*80)
        print("📊 전체 모델 예측 평가 시스템 (극단값 개선 버전)")
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
        self.extreme_booster = ExtremeValueBooster()  # 극단값 부스터 추가
        
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
        
        # 고값 통계 출력
        high_count = (df['TOTALCNT'] >= 1700).sum()
        print(f"  🎯 고값(1700+): {high_count}개 ({high_count/len(df)*100:.1f}%)")
        
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
        """모든 모델 평가 및 예측값 저장 (극단값 개선 포함)"""
        
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
        
        # 시간 및 특징 정보 수집
        timestamps_pred = []
        timestamps_target = []
        actuals = []
        m14b_values = []
        m14a_values = []
        
        print("\n📊 데이터 수집 중...")
        for i in range(start_idx, end_idx):
            pred_time = df.iloc[i]['CURRTIME']
            target_time = pred_time + timedelta(minutes=self.pred_len)
            
            actual_idx = i + self.pred_len
            if actual_idx < len(df):
                timestamps_pred.append(pred_time)
                timestamps_target.append(target_time)
                actuals.append(df.iloc[actual_idx]['TOTALCNT'])
                m14b_values.append(df.iloc[i]['M14AM14B'])
                m14a_values.append(df.iloc[i]['M14AM10A'])
        
        # 기본 정보 저장
        all_predictions['예측시점'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_pred]
        all_predictions['예측대상시간'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_target]
        all_predictions['실제값'] = actuals
        all_predictions['M14AM14B'] = m14b_values
        all_predictions['M14AM10A'] = m14a_values
        
        print(f"  예측할 데이터: {len(all_predictions)}개")
        
        # 각 모델별 예측
        model_metrics = {}
        model_predictions = {}
        
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
            
            # 예측값 저장
            predictions = predictions[:len(all_predictions)]
            model_predictions[model_name] = predictions
            
            # 원본 예측값 저장
            all_predictions[f'{model_name}_원본'] = [round(p) for p in predictions]
            
            # 극단값 부스팅 적용
            boosted_predictions = []
            for i in range(len(predictions)):
                m14b = all_predictions.iloc[i]['M14AM14B']
                m14a = all_predictions.iloc[i]['M14AM10A']
                boosted = self.extreme_booster.boost_prediction(predictions[i], m14b, m14a)
                boosted_predictions.append(boosted)
            
            all_predictions[f'{model_name}_예측'] = [round(p) for p in boosted_predictions]
            all_predictions[f'{model_name}_오차'] = all_predictions[f'{model_name}_예측'] - all_predictions['실제값']
            all_predictions[f'{model_name}_오차율(%)'] = round(
                abs(all_predictions[f'{model_name}_오차']) / all_predictions['실제값'] * 100, 2
            )
            
            # 성능 계산
            mae = mean_absolute_error(all_predictions['실제값'], boosted_predictions)
            rmse = np.sqrt(mean_squared_error(all_predictions['실제값'], boosted_predictions))
            r2 = r2_score(all_predictions['실제값'], boosted_predictions)
            mape = np.mean(abs(all_predictions[f'{model_name}_오차']) / all_predictions['실제값']) * 100
            
            model_metrics[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                '정확도(%)': 100 - mape
            }
            
            print(f"  ✅ {model_name} 완료: MAE={mae:.2f}, R²={r2:.4f}, 정확도={100-mape:.2f}%")
        
        # 극단값 특화 앙상블 (SpikeDetector와 GoldenRule 강화)
        print("\n🔥 극단값 특화 앙상블 생성...")
        
        extreme_ensemble = []
        for i in range(len(all_predictions)):
            m14b = all_predictions.iloc[i]['M14AM14B']
            
            # M14B가 높을 때는 SpikeDetector와 GoldenRule 위주
            if m14b > 450:
                weights = {
                    'SpikeDetector': 0.40,
                    'GoldenRule': 0.30,
                    'PatchTST': 0.15,
                    'StableLSTM': 0.10,
                    'ExtremeNet': 0.05
                }
            else:
                weights = {
                    'PatchTST': 0.30,
                    'StableLSTM': 0.25,
                    'ExtremeNet': 0.20,
                    'SpikeDetector': 0.15,
                    'GoldenRule': 0.10
                }
            
            ensemble_pred = 0
            total_weight = 0
            
            for model_name in self.models.keys():
                if model_name in weights:
                    weight = weights[model_name]
                    ensemble_pred += all_predictions[f'{model_name}_예측'] .iloc[i] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_pred = ensemble_pred / total_weight
            
            # 극단값 최종 체크
            if m14b > 500:
                ensemble_pred = max(ensemble_pred, 1750)
            elif m14b > 450:
                ensemble_pred = max(ensemble_pred, 1700)
            elif m14b > 400:
                ensemble_pred = max(ensemble_pred, 1650)
            
            extreme_ensemble.append(ensemble_pred)
        
        # 극단값 앙상블 결과 추가
        all_predictions['극단앙상블_예측'] = [round(p) for p in extreme_ensemble]
        all_predictions['극단앙상블_오차'] = all_predictions['극단앙상블_예측'] - all_predictions['실제값']
        all_predictions['극단앙상블_오차율(%)'] = round(
            abs(all_predictions['극단앙상블_오차']) / all_predictions['실제값'] * 100, 2
        )
        
        # 극단 앙상블 성능
        extreme_mae = mean_absolute_error(all_predictions['실제값'], extreme_ensemble)
        extreme_rmse = np.sqrt(mean_squared_error(all_predictions['실제값'], extreme_ensemble))
        extreme_r2 = r2_score(all_predictions['실제값'], extreme_ensemble)
        extreme_mape = np.mean(abs(all_predictions['극단앙상블_오차']) / all_predictions['실제값']) * 100
        
        model_metrics['극단앙상블'] = {
            'MAE': extreme_mae,
            'RMSE': extreme_rmse,
            'R2': extreme_r2,
            'MAPE': extreme_mape,
            '정확도(%)': 100 - extreme_mape
        }
        
        print(f"✅ 극단앙상블: MAE={extreme_mae:.2f}, R²={extreme_r2:.4f}, 정확도={100-extreme_mape:.2f}%")
        
        # CSV 저장
        output_file = f'v66_extreme_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        all_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 예측값 저장: {output_file}")
        
        # 고값 구간 분석 (핵심!)
        print("\n" + "="*80)
        print("🎯 고값 구간 (1700+) 분석")
        print("="*80)
        
        high_mask = all_predictions['실제값'] >= 1700
        if high_mask.any():
            high_df = all_predictions[high_mask]
            print(f"\n전체 고값 샘플: {high_mask.sum()}개")
            
            # 각 모델별 고값 적중률 계산
            print(f"\n[1700+ 적중률]")
            for model_name in list(self.models.keys()) + ['극단앙상블']:
                high_preds = high_df[f'{model_name}_예측'].values
                hit_1700 = (high_preds >= 1700).sum()
                hit_1650 = (high_preds >= 1650).sum()
                
                print(f"  {model_name:15s}: 1700+={hit_1700:2d}/{len(high_df):2d} ({hit_1700/len(high_df)*100:5.1f}%), "
                      f"1650+={hit_1650:2d}/{len(high_df):2d} ({hit_1650/len(high_df)*100:5.1f}%)")
            
            # 극단앙상블 상세
            print(f"\n[극단앙상블 고값 상세]")
            for idx in high_df.head(10).index:
                row = all_predictions.loc[idx]
                print(f"  실제:{row['실제값']:4.0f}, 예측:{row['극단앙상블_예측']:4.0f}, "
                      f"M14B:{row['M14AM14B']:3.0f}, 오차:{row['극단앙상블_오차']:+4.0f}")
        
        # 성능 요약 테이블
        print("\n" + "="*80)
        print("📊 전체 모델 성능 요약")
        print("="*80)
        
        metrics_df = pd.DataFrame(model_metrics).T
        metrics_df = metrics_df.sort_values('R2', ascending=False)
        
        print(f"\n{'모델':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE(%)':>8} {'정확도(%)':>10}")
        print("-" * 65)
        
        for model_name, row in metrics_df.iterrows():
            if model_name == '극단앙상블':
                print(f"{'🔥 ' + model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                      f"{row['R2']:8.4f} {row['MAPE']:8.2f} {row['정확도(%)']:10.2f} ⭐⭐⭐")
            else:
                print(f"{model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                      f"{row['R2']:8.4f} {row['MAPE']:8.2f} {row['정확도(%)']:10.2f}")
        
        return all_predictions, metrics_df

def main():
    """메인 실행"""
    
    print("\n🚀 V6.6 극단값 개선 평가 시작!")
    print("목표: 1700+ 구간 예측률 40% → 80% 상향")
    
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
    
    print("\n✅ 극단값 개선 평가 완료!")
    print(f"\n📁 저장된 파일:")
    print(f"  v66_extreme_predictions_YYYYMMDD.csv - 극단값 개선된 예측값")
    print("\n🔥 핵심 개선:")
    print("  - M14B > 450 → 최소 1700 보장")
    print("  - M14B > 500 → 최소 1750 보장")
    print("  - 고값 구간에서 SpikeDetector 40%, GoldenRule 30% 가중치")
    print("="*80)

if __name__ == "__main__":
    main()