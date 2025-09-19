"""
📊 V6.7 UU1 보수적 버전 - 과도한 1682+ 예측 방지
========================================================
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

warnings.filterwarnings('ignore')
tf.keras.config.enable_unsafe_deserialization()

# ====================== 보수적 극단값 보정 클래스 ======================
class ConservativeExtremeValueBooster:
    """보수적 부스팅"""
    
    def __init__(self):
        print("🔥 보수적 극단값 부스터 초기화")
        print("  - 부스팅 최대 5%로 제한")
        print("  - M14B 조건 강화")
        
    def analyze_sequence(self, sequence_data):
        """시퀀스 분석"""
        if len(sequence_data) == 0:
            return {'max': 0, 'trend': 'stable', 'consecutive_rises': 0}
        
        seq_max = np.max(sequence_data)
        consecutive_rises = 0
        for i in range(len(sequence_data)-1, 0, -1):
            if sequence_data[i] > sequence_data[i-1]:
                consecutive_rises += 1
            else:
                break
        
        # 추세 분석 (엄격한 기준)
        if len(sequence_data) >= 30:
            recent = sequence_data[-30:]
            x = np.arange(len(recent))
            coeffs = np.polyfit(x, recent, 1)
            slope = coeffs[0]
            
            if slope > 3 and consecutive_rises >= 10:
                trend = 'increasing'
            elif slope < -2:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'max': seq_max,
            'trend': trend,
            'consecutive_rises': consecutive_rises
        }
    
    def boost_prediction(self, pred, m14b_value, m14a_value=None, model_name=None, 
                        sequence_info=None):
        """보수적 부스팅"""
        if not sequence_info:
            return pred
            
        seq_max = sequence_info.get('max', 0)
        seq_trend = sequence_info.get('trend', 'stable')
        consecutive_rises = sequence_info.get('consecutive_rises', 0)
        
        boosted = pred
        
        # ExtremeNet 보수적 처리
        if model_name == 'ExtremeNet':
            if seq_max >= 1680 and seq_trend == 'increasing' and m14b_value > 250:
                boosted = pred * 1.05  # 최대 5%
            elif seq_max >= 1651 and consecutive_rises >= 15:
                boosted = pred * 1.02  # 2%
            else:
                boosted = pred
                
        # 다른 모델들
        elif model_name in ['SpikeDetector', 'GoldenRule']:
            if seq_max >= 1680 and m14b_value > 250:
                boosted = pred * 1.02
            else:
                boosted = pred
        else:
            boosted = pred
        
        # 상한선
        if boosted > 1750:
            boosted = 1750
            
        return boosted

class ImprovedModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """평가기 초기화"""
        print("="*80)
        print("🔥 V6.7 UU1 보수적 버전")
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
        
        self.models = {}
        self.extreme_booster = ConservativeExtremeValueBooster()
        
    def load_all_models(self, model_dir='models/'):
        """모델 로드"""
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
        
        return self.models
    
    def load_test_data(self, filepath):
        """데이터 로드"""
        print(f"\n📂 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        print(f"  유효: {df.shape[0]:,}행")
        print(f"  1700+: {(df['TOTALCNT'] >= 1700).sum()}개")
        
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
        """모델 평가"""
        
        # 데이터 로드
        df = self.load_test_data(test_file)
        df = self.create_features(df)
        
        # 예측 범위
        start_idx = self.seq_len
        end_idx = len(df) - self.pred_len
        total = end_idx - start_idx
        
        print(f"\n🔮 예측 시작...")
        print(f"  예측 개수: {total:,}개")
        
        # 결과 DataFrame
        all_predictions = pd.DataFrame()
        
        # 데이터 수집
        timestamps_pred = []
        actuals = []
        m14b_values = []
        sequence_maxes = []
        sequence_trends = []
        
        for i in range(start_idx, end_idx):
            pred_time = df.iloc[i]['CURRTIME']
            actual_idx = i + self.pred_len
            
            if actual_idx < len(df):
                timestamps_pred.append(pred_time)
                actuals.append(df.iloc[actual_idx]['TOTALCNT'])
                m14b_values.append(df.iloc[i]['M14AM14B'])
                
                # 시퀀스 분석
                seq_data = df.iloc[i-self.seq_len:i]['TOTALCNT'].values
                seq_info = self.extreme_booster.analyze_sequence(seq_data)
                sequence_maxes.append(seq_info.get('max'))
                sequence_trends.append(seq_info.get('trend'))
        
        # 기본 정보 저장
        all_predictions['예측시점'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_pred]
        all_predictions['실제값'] = actuals
        all_predictions['M14AM14B'] = m14b_values
        all_predictions['시퀀스_MAX'] = sequence_maxes
        all_predictions['시퀀스_추세'] = sequence_trends
        
        # 각 모델 예측
        model_metrics = {}
        
        for model_name, model in self.models.items():
            print(f"\n🎯 {model_name} 예측 중...")
            predictions = []
            
            # 배치 예측
            batch_size = 500
            for i in range(start_idx, end_idx, batch_size):
                batch_end = min(i + batch_size, end_idx)
                
                X_batch = []
                for j in range(i, batch_end):
                    seq_data = df.iloc[j-self.seq_len:j][self.feature_columns].values
                    X_batch.append(seq_data)
                
                if len(X_batch) == 0:
                    continue
                
                X_batch = np.array(X_batch)
                X_batch_scaled = []
                for seq in X_batch:
                    seq_scaled = self.feature_scaler.transform(seq)
                    X_batch_scaled.append(seq_scaled)
                X_batch_scaled = np.array(X_batch_scaled)
                
                preds = model.predict(X_batch_scaled, verbose=0)
                
                if isinstance(preds, list):
                    y_pred_scaled = preds[0].flatten()
                else:
                    y_pred_scaled = preds.flatten()
                
                y_pred = self.target_scaler.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)).flatten()
                
                for k in range(len(y_pred)):
                    actual_idx = i - start_idx + k
                    if actual_idx < len(all_predictions):
                        predictions.append(y_pred[k])
            
            # 보수적 부스팅
            boosted_predictions = []
            for i in range(len(predictions)):
                seq_info = {
                    'max': all_predictions.iloc[i]['시퀀스_MAX'],
                    'trend': all_predictions.iloc[i]['시퀀스_추세'],
                    'consecutive_rises': 0
                }
                
                boosted = self.extreme_booster.boost_prediction(
                    predictions[i],
                    all_predictions.iloc[i]['M14AM14B'],
                    None,
                    model_name,
                    seq_info
                )
                boosted_predictions.append(boosted)
            
            all_predictions[f'{model_name}_예측'] = [round(p) for p in boosted_predictions]
            
            # 성능
            mae = mean_absolute_error(all_predictions['실제값'], boosted_predictions)
            print(f"  MAE: {mae:.2f}")
            
            model_metrics[model_name] = {'MAE': mae}
        
        # 보수적 앙상블
        print("\n🔥 보수적 앙상블 생성...")
        ensemble = []
        
        for i in range(len(all_predictions)):
            m14b = all_predictions.iloc[i]['M14AM14B']
            seq_max = all_predictions.iloc[i]['시퀀스_MAX']
            
            # 보수적 가중치 (StableLSTM, PatchTST 강화)
            if seq_max >= 1650:
                weights = {
                    'StableLSTM': 0.30,
                    'PatchTST': 0.25,
                    'GoldenRule': 0.20,
                    'SpikeDetector': 0.15,
                    'ExtremeNet': 0.10
                }
            else:
                weights = {
                    'StableLSTM': 0.22,
                    'PatchTST': 0.22,
                    'GoldenRule': 0.22,
                    'ExtremeNet': 0.17,
                    'SpikeDetector': 0.17
                }
            
            pred = 0
            for model_name, w in weights.items():
                if f'{model_name}_예측' in all_predictions.columns:
                    pred += all_predictions.iloc[i][f'{model_name}_예측'] * w
            
            ensemble.append(pred)
        
        all_predictions['앙상블_예측'] = [round(p) for p in ensemble]
        
        # 최종 성능
        mae = mean_absolute_error(all_predictions['실제값'], ensemble)
        print(f"✅ 보수적 앙상블: MAE={mae:.2f}")
        
        # CSV 저장
        output_file = f'v67_uu1_conservative_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        all_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: {output_file}")
        
        return all_predictions, model_metrics

def main():
    evaluator = ImprovedModelEvaluator()
    models = evaluator.load_all_models('models/')
    
    if not models:
        print("❌ 모델 없음")
        return
    
    test_files = ['uu.csv', 'uu1.csv', 'data/uu.csv']
    
    for file in test_files:
        if os.path.exists(file):
            evaluator.evaluate_all_models(file)
            break

if __name__ == "__main__":
    main()