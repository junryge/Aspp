"""
📊 V6.7 개선 - ExtremeNet 범위값 추가 버전
========================================================
원본 V6.7 + ExtremeNet_범위값 계산 기능만 추가
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

# ====================== 개선된 극단값 보정 클래스 ======================
class ImprovedExtremeValueBooster:
    """시퀀스 기반 스마트 극단값 예측"""
    
    def __init__(self):
        print("🔥 개선된 극단값 부스터 초기화")
        print("  - 시퀀스 max값 1651 이상 체크")
        print("  - 증가/하락 추세 분석")
        print("  - ExtremeNet 범위값 계산")
        
    def analyze_sequence(self, sequence_data):
        """시퀀스 분석"""
        if len(sequence_data) == 0:
            return {'max': 0, 'min': 0, 'trend': 'stable', 'is_high_plateau': False,
                   'consecutive_rises': 0, 'consecutive_falls': 0, 
                   'rise_strength': 0, 'fall_strength': 0, 'volatility': 0}
        
        seq_max = np.max(sequence_data)
        seq_min = np.min(sequence_data)
        seq_mean = np.mean(sequence_data[-30:]) if len(sequence_data) >= 30 else np.mean(sequence_data)
        
        is_high_plateau = seq_mean >= 1700
        
        # 연속 상승/하락 카운트
        consecutive_rises = 0
        consecutive_falls = 0
        
        for i in range(len(sequence_data)-1, 0, -1):
            if sequence_data[i] > sequence_data[i-1]:
                consecutive_rises += 1
            else:
                break
        
        for i in range(len(sequence_data)-1, 0, -1):
            if sequence_data[i] < sequence_data[i-1]:
                consecutive_falls += 1
            else:
                break
        
        # 상승/하락 강도
        rise_strength = 0
        fall_strength = 0
        if len(sequence_data) >= 10:
            recent_10 = sequence_data[-10:]
            change = recent_10[-1] - recent_10[0]
            if change > 0:
                rise_strength = change
            else:
                fall_strength = abs(change)
        
        # 추세 분석
        if len(sequence_data) >= 30:
            recent = sequence_data[-30:]
            x = np.arange(len(recent))
            coeffs = np.polyfit(x, recent, 1)
            slope = coeffs[0]
            
            if is_high_plateau:
                if consecutive_rises >= 10 and rise_strength > 50:
                    trend = 'extreme_rising'
                elif consecutive_falls >= 10 and fall_strength > 50:
                    trend = 'extreme_falling'
                elif slope > 1 or consecutive_rises >= 5:
                    trend = 'high_increasing'
                elif slope < -1 or consecutive_falls >= 5:
                    trend = 'high_decreasing'
                else:
                    trend = 'high_stable'
            else:
                if consecutive_rises >= 10 and rise_strength > 50:
                    trend = 'strong_rising'
                elif consecutive_falls >= 10 and fall_strength > 50:
                    trend = 'strong_falling'
                elif consecutive_rises >= 7 and rise_strength > 30:
                    trend = 'rapid_increasing'
                elif consecutive_falls >= 7 and fall_strength > 30:
                    trend = 'rapid_decreasing'
                elif slope > 2:
                    trend = 'increasing'
                elif slope < -2:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
        else:
            trend = 'stable'
        
        volatility = np.std(sequence_data[-10:]) if len(sequence_data) >= 10 else 0
        
        return {
            'max': seq_max,
            'min': seq_min,
            'trend': trend,
            'is_high_plateau': is_high_plateau,
            'consecutive_rises': consecutive_rises,
            'consecutive_falls': consecutive_falls,
            'rise_strength': rise_strength,
            'fall_strength': fall_strength,
            'volatility': volatility
        }
    
    def boost_prediction(self, pred, m14b_value, m14a_value=None, model_name=None, 
                        sequence_info=None):
        """시퀀스 기반 스마트 부스팅"""
        if not sequence_info:
            return pred
            
        seq_max = sequence_info.get('max', 0)
        seq_trend = sequence_info.get('trend', 'stable')
        
        boosted = pred
        
        # ExtremeNet 특별 처리
        if model_name == 'ExtremeNet':
            if seq_max >= 1651 and seq_trend == 'increasing':
                if m14b_value > 200:
                    boosted = max(pred * 1.2, 1700)
                else:
                    boosted = max(pred * 1.15, 1700)
        
        # SpikeDetector, GoldenRule
        elif model_name in ['SpikeDetector', 'GoldenRule']:
            if seq_max >= 1651 and seq_trend == 'increasing':
                if m14b_value > 200:
                    boosted = max(pred * 1.15, 1700)
                else:
                    boosted = max(pred * 1.1, 1700)
        
        # 기타 모델
        else:
            if seq_max >= 1651 and seq_trend == 'increasing':
                if m14b_value > 200:
                    boosted = max(pred * 1.1, 1700)
                else:
                    boosted = max(pred * 1.05, 1700)
        
        return boosted

class ImprovedModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """개선된 평가기 초기화"""
        print("="*80)
        print("🔥 V6.7 개선 - ExtremeNet 범위값 포함")
        print("  핵심: 시퀀스 max 1651+ & 증가 추세 → ExtremeNet 부스팅")
        print("  추가: ExtremeNet_범위값 계산")
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
        self.extreme_booster = ImprovedExtremeValueBooster()
        
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
        
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        print(f"  유효: {df.shape[0]:,}행")
        
        # 고값 통계
        high_count = (df['TOTALCNT'] >= 1700).sum()
        very_high_count = (df['TOTALCNT'] >= 1750).sum()
        extreme_count = (df['TOTALCNT'] >= 1800).sum()
        
        print(f"\n🎯 고값 구간 분포:")
        print(f"  1700+: {high_count}개 ({high_count/len(df)*100:.1f}%)")
        print(f"  1750+: {very_high_count}개 ({very_high_count/len(df)*100:.1f}%)")
        print(f"  1800+: {extreme_count}개 ({extreme_count/len(df)*100:.1f}%)")
        
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
        """개선된 모델 평가 - ExtremeNet 범위값 포함"""
        
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
        
        # 모든 예측을 저장할 DataFrame
        all_predictions = pd.DataFrame()
        
        # 데이터 수집
        timestamps_pred = []
        timestamps_target = []
        actuals = []
        m14b_values = []
        m14a_values = []
        sequence_maxes = []
        sequence_trends = []
        
        print("\n📊 데이터 수집 및 시퀀스 분석 중...")
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
                
                # 시퀀스 분석
                seq_data = df.iloc[i-self.seq_len:i]['TOTALCNT'].values
                seq_info = self.extreme_booster.analyze_sequence(seq_data)
                sequence_maxes.append(seq_info.get('max'))
                sequence_trends.append(seq_info.get('trend'))
        
        # 기본 정보 저장
        all_predictions['예측시점'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_pred]
        all_predictions['예측대상시간'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_target]
        all_predictions['실제값'] = actuals
        all_predictions['M14AM14B'] = m14b_values
        all_predictions['M14AM10A'] = m14a_values
        all_predictions['시퀀스_MAX'] = sequence_maxes
        all_predictions['시퀀스_추세'] = sequence_trends
        
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
                
                if len(predictions) % 2000 == 0:
                    print(f"    {len(predictions):,}/{len(all_predictions):,} 완료")
            
            # 예측값 저장
            predictions = predictions[:len(all_predictions)]
            model_predictions[model_name] = predictions
            
            # 원본 예측값 저장
            all_predictions[f'{model_name}_원본'] = [round(p) for p in predictions]
            
            # 🔥 ExtremeNet 범위값 계산 (추가 부분)
            if model_name == 'ExtremeNet':
                print(f"  📊 ExtremeNet 범위값 계산 중...")
                extreme_percents = []
                extreme_min_values = []
                extreme_max_values = []
                extreme_ranges = []
                
                for i in range(len(predictions)):
                    seq_max = all_predictions.iloc[i]['시퀀스_MAX']
                    extreme_orig = predictions[i]
                    
                    if extreme_orig > 0:
                        # 1. 비율 계산
                        비율 = seq_max - extreme_orig
                        
                        # 2. 퍼센트 계산  
                        퍼센트 = (비율 - 1) * 100
                        
                        # 3. 범위 계산
                        최소값 = extreme_orig
                        최대값 = extreme_orig * 비율
                        
                        # 음수 방지
                        if 최대값 < 최소값:
                            최대값 = 최소값 * 1.1
                        
                        extreme_percents.append(round(퍼센트, 2))
                        extreme_min_values.append(round(최소값))
                        extreme_max_values.append(round(최대값))
                        extreme_ranges.append(f"{round(최소값)}~{round(최대값)}")
                    else:
                        extreme_percents.append(0)
                        extreme_min_values.append(0)
                        extreme_max_values.append(0)
                        extreme_ranges.append("0~0")
                
                # ExtremeNet 관련 컬럼 추가
                all_predictions['ExtremeNet_퍼센트'] = extreme_percents
                all_predictions['ExtremeNet_최소값'] = extreme_min_values
                all_predictions['ExtremeNet_최대값'] = extreme_max_values
                all_predictions['ExtremeNet_범위값'] = extreme_ranges
                
                print(f"    ✅ 범위값 계산 완료")
                print(f"    공식: 비율 = 시퀀스_MAX - 원본")
            
            # 시퀀스 기반 스마트 부스팅
            print(f"  🔥 {model_name} 시퀀스 기반 부스팅 적용 중...")
            boosted_predictions = []
            boost_applied_count = 0
            
            for i in range(len(predictions)):
                m14b = all_predictions.iloc[i]['M14AM14B']
                m14a = all_predictions.iloc[i]['M14AM10A']
                seq_max = all_predictions.iloc[i]['시퀀스_MAX']
                seq_trend = all_predictions.iloc[i]['시퀀스_추세']
                original = predictions[i]
                
                # 시퀀스 정보 딕셔너리 생성
                seq_info = {
                    'max': seq_max,
                    'trend': seq_trend,
                    'consecutive_rises': 0,
                    'consecutive_falls': 0,
                    'rise_strength': 0,
                    'fall_strength': 0
                }
                
                # 시퀀스 기반 부스팅 적용
                boosted = self.extreme_booster.boost_prediction(
                    original, m14b, m14a, model_name,
                    sequence_info=seq_info
                )
                
                # ExtremeNet 부스팅 통계
                if model_name == 'ExtremeNet' and boosted != original:
                    boost_applied_count += 1
                    
                boosted_predictions.append(boosted)
            
            if model_name == 'ExtremeNet':
                print(f"    ExtremeNet 부스팅 적용: {boost_applied_count}개 "
                      f"({boost_applied_count/len(predictions)*100:.1f}%)")
            
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
        
        # 개선된 극단값 앙상블
        print("\n🔥 개선된 극단값 앙상블 생성...")
        
        extreme_ensemble = []
        for i in range(len(all_predictions)):
            m14b = all_predictions.iloc[i]['M14AM14B']
            m14a = all_predictions.iloc[i]['M14AM10A']
            seq_max = all_predictions.iloc[i]['시퀀스_MAX']
            seq_trend = all_predictions.iloc[i]['시퀀스_추세']
            
            # 각 모델 예측값
            extreme_pred = all_predictions.iloc[i]['ExtremeNet_예측']
            spike_pred = all_predictions.iloc[i]['SpikeDetector_예측'] if 'SpikeDetector_예측' in all_predictions.columns else extreme_pred
            golden_pred = all_predictions.iloc[i]['GoldenRule_예측'] if 'GoldenRule_예측' in all_predictions.columns else extreme_pred
            patch_pred = all_predictions.iloc[i]['PatchTST_예측'] if 'PatchTST_예측' in all_predictions.columns else extreme_pred
            stable_pred = all_predictions.iloc[i]['StableLSTM_예측'] if 'StableLSTM_예측' in all_predictions.columns else extreme_pred
            
            # 핵심: 시퀀스 MAX가 1650+ 이고 increasing이면 최소값 보장
            if seq_max >= 1650 and seq_trend == 'increasing':
                max_pred = max(extreme_pred, spike_pred, golden_pred, patch_pred, stable_pred)
                ensemble_pred = max(max_pred, seq_max * 1.03, 1700)
                
            elif seq_max >= 1600:
                weights = {
                    'ExtremeNet': 0.25,
                    'SpikeDetector': 0.20,
                    'GoldenRule': 0.20,
                    'PatchTST': 0.20,
                    'StableLSTM': 0.15
                }
                
                ensemble_pred = (
                    extreme_pred * weights['ExtremeNet'] +
                    spike_pred * weights['SpikeDetector'] +
                    golden_pred * weights['GoldenRule'] +
                    patch_pred * weights['PatchTST'] +
                    stable_pred * weights['StableLSTM']
                )
                
                ensemble_pred = max(ensemble_pred, seq_max, 1650)
                
            else:
                if m14b > 200:
                    weights = {
                        'SpikeDetector': 0.30,
                        'GoldenRule': 0.25,
                        'ExtremeNet': 0.20,
                        'PatchTST': 0.15,
                        'StableLSTM': 0.10
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
                for model_name in ['ExtremeNet', 'SpikeDetector', 'GoldenRule', 'PatchTST', 'StableLSTM']:
                    if f'{model_name}_예측' in all_predictions.columns:
                        ensemble_pred += all_predictions.iloc[i][f'{model_name}_예측'] * weights[model_name]
            
            extreme_ensemble.append(ensemble_pred)
        
        # 극단 앙상블 결과 추가
        all_predictions['개선앙상블_예측'] = [round(p) for p in extreme_ensemble]
        all_predictions['개선앙상블_오차'] = all_predictions['개선앙상블_예측'] - all_predictions['실제값']
        all_predictions['개선앙상블_오차율(%)'] = round(
            abs(all_predictions['개선앙상블_오차']) / all_predictions['실제값'] * 100, 2
        )
        
        # CSV 저장
        output_file = f'v67_range_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        all_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 예측값 저장: {output_file}")
        
        # 성능 요약
        print("\n" + "="*80)
        print("📊 모델 성능 요약")
        print("="*80)
        
        metrics_df = pd.DataFrame(model_metrics).T
        metrics_df = metrics_df.sort_values('R2', ascending=False)
        
        print(f"\n{'모델':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'정확도(%)':>10}")
        print("-" * 55)
        
        for model_name, row in metrics_df.iterrows():
            print(f"{model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                  f"{row['R2']:8.4f} {row['정확도(%)']:10.2f}")
        
        return all_predictions, metrics_df

def main():
    """메인 실행"""
    
    print("\n🚀 V6.7 개선 - ExtremeNet 범위값 포함 평가 시작!")
    print("핵심: 시퀀스 max 1651+ & 증가 추세 → ExtremeNet 부스팅")
    print("추가: ExtremeNet_범위값 계산 (원본 ~ 최대값)")
    
    # 평가기 생성
    evaluator = ImprovedModelEvaluator()
    
    # 모든 모델 로드
    models = evaluator.load_all_models('models/')
    
    if not models:
        print("❌ 모델이 없습니다!")
        return
    
    # 테스트 파일
    test_files = [
        'data/M14_20250916_20250817.csv',
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
    
    print("\n" + "="*80)
    print("🏆 V6.7 ExtremeNet 범위값 평가 완료!")
    print("="*80)
    print("\n📁 저장된 파일:")
    print(f"  1. v67_range_predictions_YYYYMMDD.csv")
    print("\n🔥 ExtremeNet 추가 컬럼:")
    print("  ✅ ExtremeNet_퍼센트")
    print("  ✅ ExtremeNet_최소값")
    print("  ✅ ExtremeNet_최대값")
    print("  ✅ ExtremeNet_범위값")
    print("="*80)

if __name__ == "__main__":
    main()