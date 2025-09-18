"""
📊 V6.8 극단값 예측 개선 - 버그 수정 버전
========================================================
원본과 부스팅 예측값이 제대로 구분되도록 수정
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
        print("🔥 V6.8 극단값 개선 평가 (버그 수정)")
        print("="*80)
        
        # 스케일러 로드
        with open(f'{scaler_path}feature_scaler.pkl', 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(f'{scaler_path}target_scaler.pkl', 'rb') as f:
            self.target_scaler = pickle.load(f)
        with open(f'{scaler_path}config.json', 'r') as f:
            config = json.load(f)
            self.seq_len = config['seq_len']  # 100
            self.pred_len = config['pred_len']  # 10
            self.feature_columns = config['feature_columns']
        
        print(f"✅ 설정: {self.seq_len}분 시퀀스 → {self.pred_len}분 후 예측")
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
                print(f"  ✅ {model_name} 로드")
            except Exception as e:
                print(f"  ❌ {model_name} 실패: {e}")
        
        print(f"총 {len(self.models)}개 모델 로드")
        return self.models
    
    def load_test_data(self, filepath):
        """테스트 데이터 로드"""
        df = pd.read_csv(filepath)
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        print(f"\n📊 데이터: {len(df)}행")
        high_count = (df['TOTALCNT'] >= 1700).sum()
        print(f"  고값(1700+): {high_count}개 ({high_count/len(df)*100:.1f}%)")
        
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
    
    def apply_extreme_boosting(self, pred, m14b, m14a, model_name):
        """극단값 부스팅 적용"""
        original = pred
        
        # 1. ExtremeNet - 고값에서 약하므로 강화 필요
        if model_name == 'ExtremeNet':
            if m14b > 500:
                boosted = max(pred * 1.3, 1750)
            elif m14b > 450:
                boosted = max(pred * 1.2, 1700)
            elif m14b > 400:
                boosted = max(pred * 1.1, 1650)
            else:
                boosted = pred * 1.05
                
        # 2. SpikeDetector, GoldenRule - 이미 잘하므로 최소값만
        elif model_name in ['SpikeDetector', 'GoldenRule']:
            if m14b > 500:
                boosted = max(pred, 1750)
            elif m14b > 450:
                boosted = max(pred, 1700)
            elif m14b > 400:
                boosted = max(pred, 1650)
            else:
                boosted = pred
                
        # 3. 나머지 모델들
        else:
            if m14b > 500:
                boosted = max(pred * 1.15, 1750)
            elif m14b > 450:
                boosted = max(pred * 1.1, 1700)
            elif m14b > 400:
                boosted = max(pred * 1.05, 1650)
            else:
                boosted = pred
        
        # 황금 패턴
        if m14b > 300 and m14a < 80:
            boosted *= 1.08
        
        return boosted
    
    def evaluate_all_models(self, test_file):
        """모든 모델 평가"""
        
        # 데이터 준비
        df = self.load_test_data(test_file)
        df = self.create_features(df)
        
        start_idx = self.seq_len
        end_idx = len(df) - self.pred_len
        total = end_idx - start_idx
        
        if total <= 0:
            print("❌ 데이터 부족")
            return None, None
        
        print(f"\n🔮 예측 시작: {total}개")
        
        # 결과 DataFrame 준비
        results = []
        
        # 각 시점별 예측
        for idx in range(start_idx, end_idx):
            pred_time = df.iloc[idx]['CURRTIME']
            target_time = pred_time + timedelta(minutes=self.pred_len)
            
            actual_idx = idx + self.pred_len
            if actual_idx >= len(df):
                continue
            
            # 현재 정보
            row_data = {
                '예측시점': pred_time.strftime('%Y-%m-%d %H:%M'),
                '예측대상시간': target_time.strftime('%Y-%m-%d %H:%M'),
                '실제값': df.iloc[actual_idx]['TOTALCNT'],
                'M14AM14B': df.iloc[idx]['M14AM14B'],
                'M14AM10A': df.iloc[idx]['M14AM10A']
            }
            
            # 시퀀스 데이터 추출
            seq_data = df.iloc[idx-self.seq_len:idx][self.feature_columns].values
            if len(seq_data) != self.seq_len:
                continue
            
            seq_scaled = self.feature_scaler.transform(seq_data)
            seq_scaled = seq_scaled.reshape(1, self.seq_len, -1)
            
            # 각 모델 예측
            model_preds = {}
            
            for model_name, model in self.models.items():
                # 원본 예측
                pred = model.predict(seq_scaled, verbose=0)
                if isinstance(pred, list):
                    pred = pred[0]
                pred_value = self.target_scaler.inverse_transform(pred.reshape(-1, 1))[0, 0]
                
                # 원본 저장
                row_data[f'{model_name}_원본'] = round(pred_value)
                
                # 부스팅 적용
                boosted_value = self.apply_extreme_boosting(
                    pred_value, 
                    row_data['M14AM14B'],
                    row_data['M14AM10A'],
                    model_name
                )
                
                # 부스팅 저장
                row_data[f'{model_name}_예측'] = round(boosted_value)
                row_data[f'{model_name}_오차'] = round(boosted_value - row_data['실제값'])
                row_data[f'{model_name}_오차율(%)'] = round(
                    abs(boosted_value - row_data['실제값']) / row_data['실제값'] * 100, 2
                )
                
                model_preds[model_name] = boosted_value
            
            # 극단값 앙상블
            m14b = row_data['M14AM14B']
            
            # M14B에 따른 가중치
            if m14b > 500:
                weights = {
                    'SpikeDetector': 0.35,
                    'GoldenRule': 0.30,
                    'PatchTST': 0.15,
                    'StableLSTM': 0.10,
                    'ExtremeNet': 0.10
                }
            elif m14b > 450:
                weights = {
                    'SpikeDetector': 0.30,
                    'GoldenRule': 0.25,
                    'PatchTST': 0.20,
                    'StableLSTM': 0.15,
                    'ExtremeNet': 0.10
                }
            else:
                weights = {
                    'PatchTST': 0.30,
                    'StableLSTM': 0.25,
                    'ExtremeNet': 0.20,
                    'SpikeDetector': 0.15,
                    'GoldenRule': 0.10
                }
            
            # 앙상블 계산
            ensemble = 0
            total_weight = 0
            
            for model_name, pred in model_preds.items():
                weight = weights.get(model_name, 0.1)
                ensemble += pred * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble = ensemble / total_weight
            
            # 최종 조정
            if m14b > 500:
                ensemble = max(ensemble, 1750)
            elif m14b > 450:
                ensemble = max(ensemble, 1700)
            elif m14b > 400:
                ensemble = max(ensemble, 1650)
            
            row_data['극단앙상블_예측'] = round(ensemble)
            row_data['극단앙상블_오차'] = round(ensemble - row_data['실제값'])
            row_data['극단앙상블_오차율(%)'] = round(
                abs(ensemble - row_data['실제값']) / row_data['실제값'] * 100, 2
            )
            
            results.append(row_data)
            
            # 진행 상황
            if len(results) % 100 == 0:
                print(f"  {len(results)}/{total} 완료...")
        
        # DataFrame 변환
        all_predictions = pd.DataFrame(results)
        
        # 성능 분석
        print("\n" + "="*80)
        print("📊 성능 분석")
        print("="*80)
        
        # 전체 성능
        models_to_eval = list(self.models.keys()) + ['극단앙상블']
        
        print(f"\n[전체 성능]")
        print(f"{'모델':<15} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>8} {'정확도(%)':>10}")
        print("-" * 55)
        
        for model_name in models_to_eval:
            col_name = f'{model_name}_예측'
            if col_name in all_predictions.columns:
                preds = all_predictions[col_name].values
                actuals = all_predictions['실제값'].values
                
                mae = mean_absolute_error(actuals, preds)
                rmse = np.sqrt(mean_squared_error(actuals, preds))
                mape = np.mean(abs(preds - actuals) / actuals) * 100
                
                if model_name == '극단앙상블':
                    print(f"{'🔥 ' + model_name:<15} {mae:8.2f} {rmse:8.2f} {mape:8.2f} {100-mape:10.2f} ⭐")
                else:
                    print(f"{model_name:<15} {mae:8.2f} {rmse:8.2f} {mape:8.2f} {100-mape:10.2f}")
        
        # 고값 구간 분석
        high_mask = all_predictions['실제값'] >= 1700
        if high_mask.any():
            high_df = all_predictions[high_mask]
            
            print(f"\n[고값(1700+) 구간 분석] - {len(high_df)}개 샘플")
            print("-" * 70)
            
            for model_name in models_to_eval:
                col_pred = f'{model_name}_예측'
                col_orig = f'{model_name}_원본'
                
                if col_pred in high_df.columns:
                    preds = high_df[col_pred].values
                    hit_1700 = (preds >= 1700).sum()
                    hit_1650 = (preds >= 1650).sum()
                    
                    # 원본과 비교
                    if col_orig in high_df.columns:
                        orig_preds = high_df[col_orig].values
                        orig_hit = (orig_preds >= 1700).sum()
                        
                        print(f"{model_name:15}: 원본 {orig_hit:2d} → 부스팅 {hit_1700:2d}/{len(high_df)} "
                              f"({hit_1700/len(high_df)*100:.1f}%) [개선 +{hit_1700-orig_hit}]")
                    else:
                        print(f"{'🔥 ' + model_name:15}: {hit_1700:2d}/{len(high_df)} "
                              f"({hit_1700/len(high_df)*100:.1f}%)")
            
            # 상세 샘플
            print(f"\n[극단앙상블 고값 예측 상세]")
            print(f"{'실제':>6} {'예측':>6} {'M14B':>5} {'M14A':>5} {'오차':>6} {'결과':>8}")
            print("-" * 45)
            
            for idx in high_df.head(15).index:
                row = all_predictions.loc[idx]
                actual = row['실제값']
                pred = row['극단앙상블_예측']
                m14b = row['M14AM14B']
                m14a = row['M14AM10A']
                error = pred - actual
                
                if pred >= 1700:
                    result = "✅ HIT"
                elif pred >= 1650:
                    result = "⚠️ NEAR"
                else:
                    result = "❌ MISS"
                
                print(f"{actual:6.0f} {pred:6.0f} {m14b:5.0f} {m14a:5.0f} {error:+6.0f} {result:>8}")
        
        # CSV 저장
        output_file = f'v68_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        all_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: {output_file}")
        
        return all_predictions, None

def main():
    """메인 실행"""
    
    print("\n🚀 V6.8 극단값 개선 평가 (버그 수정)")
    
    # 평가기 생성
    evaluator = CompleteModelEvaluator()
    
    # 모델 로드
    models = evaluator.load_all_models('models/')
    
    if not models:
        print("❌ 모델 없음")
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
        print("❌ 테스트 데이터 없음")
        return
    
    # 평가 실행
    all_predictions, _ = evaluator.evaluate_all_models(test_file)
    
    print("\n✅ 완료!")
    print("\n핵심:")
    print("  ✅ 원본과 부스팅 예측값 확실히 구분")
    print("  ✅ ExtremeNet 적절한 부스팅 (1.1~1.3배)")
    print("  ✅ M14B > 450 → 최소 1700")
    print("  ✅ 균형잡힌 앙상블 가중치")

if __name__ == "__main__":
    main()