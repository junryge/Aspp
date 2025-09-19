"""
📊 V6.7 개선 - ExtremeNet 범위값 추가
========================================================
핵심: 시퀀스 MAX와 원본 예측값으로 범위 계산
공식: (시퀀스_MAX / ExtremeNet_원본) = 퍼센트
      ExtremeNet_범위값 = 원본 ~ (원본 * (1 + 퍼센트))
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
        
        # 연속 상승 카운트
        consecutive_rises = 0
        for i in range(len(sequence_data)-1, 0, -1):
            if sequence_data[i] > sequence_data[i-1]:
                consecutive_rises += 1
            else:
                break
        
        # 연속 하락 카운트
        consecutive_falls = 0
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
    
    def calculate_extreme_range(self, original_pred, seq_max, seq_trend):
        """ExtremeNet 범위값 계산 - 추세별 공격적 범위 설정"""
        if original_pred <= 0:
            return 0, original_pred, original_pred, "0~0"
        
        # 기본 비율 계산
        base_ratio = seq_max / original_pred
        
        # 🔥 추세별 공격적 범위 설정
        if 'extreme_rising' in seq_trend or 'strong_rising' in seq_trend:
            # 극단 상승: 원본의 100-150% 범위
            min_ratio = 1.0
            max_ratio = 1.5
            
            if seq_max >= 1800:
                max_ratio = 1.6  # 60% 상향
            elif seq_max >= 1750:
                max_ratio = 1.5  # 50% 상향
            elif seq_max >= 1700:
                max_ratio = 1.4  # 40% 상향
            elif seq_max >= 1650:
                max_ratio = 1.3  # 30% 상향
                
        elif 'rapid_increasing' in seq_trend or 'high_increasing' in seq_trend:
            # 빠른 상승: 원본의 100-140% 범위
            min_ratio = 1.0
            max_ratio = 1.4
            
            if seq_max >= 1700:
                max_ratio = 1.45
            elif seq_max >= 1650:
                max_ratio = 1.35
                
        elif seq_trend == 'increasing':
            # 일반 상승: 원본의 100-130% 범위
            min_ratio = 1.0
            max_ratio = 1.3
            
            if seq_max >= 1650:
                max_ratio = 1.35
                
        elif 'extreme_falling' in seq_trend or 'strong_falling' in seq_trend:
            # 극단 하락: 원본의 80-100% 범위
            min_ratio = 0.8
            max_ratio = 1.0
            
        elif 'rapid_decreasing' in seq_trend or 'high_decreasing' in seq_trend:
            # 빠른 하락: 원본의 85-105% 범위
            min_ratio = 0.85
            max_ratio = 1.05
            
        elif seq_trend == 'decreasing':
            # 일반 하락: 원본의 90-110% 범위
            min_ratio = 0.9
            max_ratio = 1.1
            
        elif 'high_stable' in seq_trend:
            # 고평원: 원본의 95-120% 범위
            min_ratio = 0.95
            max_ratio = 1.2
            
            if seq_max >= 1750:
                max_ratio = 1.25
                
        else:  # stable
            # 안정: 원본의 95-115% 범위
            min_ratio = 0.95
            max_ratio = 1.15
        
        # 시퀀스 MAX가 원본보다 크면 최대값 추가 상향
        if seq_max > original_pred:
            extra_boost = (seq_max / original_pred - 1) * 0.5
            max_ratio = max(max_ratio, 1 + extra_boost)
        
        min_value = original_pred * min_ratio
        max_value = original_pred * max_ratio
        
        # 최대값이 시퀀스 MAX보다 낮으면 조정
        if max_value < seq_max and 'rising' in seq_trend:
            max_value = max(seq_max * 1.05, max_value)
        
        percent = (max_ratio - 1) * 100
        range_str = f"{round(min_value)}~{round(max_value)}"
        
        return percent, min_value, max_value, range_str

class ImprovedModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """개선된 평가기 초기화"""
        print("="*80)
        print("🔥 V6.7 개선 - ExtremeNet 범위값 시스템")
        print("  핵심: 시퀀스 MAX 기반 예측 범위 계산")
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
        sequence_infos = []
        
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
                sequence_infos.append(seq_info)
        
        # sequence_infos를 self에 저장 (나중에 사용)
        self.sequence_infos = sequence_infos
        
        # 기본 정보 저장
        all_predictions['예측시점'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_pred]
        all_predictions['예측대상시간'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_target]
        all_predictions['실제값'] = actuals
        all_predictions['M14AM14B'] = m14b_values
        all_predictions['M14AM10A'] = m14a_values
        all_predictions['시퀀스_MAX'] = [info['max'] for info in sequence_infos]
        all_predictions['시퀀스_추세'] = [info['trend'] for info in sequence_infos]
        
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
                
                predictions.extend(y_pred[:batch_end - i])
            
            predictions = predictions[:len(all_predictions)]
            
            # 원본 예측값 저장
            all_predictions[f'{model_name}_원본'] = [round(p) for p in predictions]
            
            # 🔥 ExtremeNet 범위값 계산
            if model_name == 'ExtremeNet':
                print(f"  📊 ExtremeNet 범위값 계산 중...")
                extreme_percents = []
                extreme_min_values = []
                extreme_max_values = []
                extreme_ranges = []
                
                for i in range(len(predictions)):
                    original = predictions[i]
                    seq_max = sequence_infos[i]['max']
                    seq_trend = sequence_infos[i]['trend']
                    
                    percent, min_val, max_val, range_str = self.extreme_booster.calculate_extreme_range(
                        original, seq_max, seq_trend
                    )
                    
                    extreme_percents.append(round(percent, 2))
                    extreme_min_values.append(round(min_val))
                    extreme_max_values.append(round(max_val))
                    extreme_ranges.append(range_str)
                
                all_predictions['ExtremeNet_퍼센트'] = extreme_percents
                all_predictions['ExtremeNet_최소값'] = extreme_min_values
                all_predictions['ExtremeNet_최대값'] = extreme_max_values
                all_predictions['ExtremeNet_범위값'] = extreme_ranges
                
                print(f"    ✅ 범위값 계산 완료")
                
                # 범위 통계
                avg_percent = np.mean(extreme_percents)
                avg_range = np.mean([max_val - min_val for max_val, min_val in 
                                    zip(extreme_max_values, extreme_min_values)])
                
                print(f"    평균 퍼센트: {avg_percent:.1f}%")
                print(f"    평균 범위 폭: {avg_range:.0f}")
            
            # 부스팅 적용
            print(f"  🔥 {model_name} 부스팅 적용 중...")
            boosted_predictions = []
            
            for i in range(len(predictions)):
                m14b = all_predictions.iloc[i]['M14AM14B']
                m14a = all_predictions.iloc[i]['M14AM10A']
                original = predictions[i]
                seq_trend = sequence_infos[i]['trend']
                consecutive_rises = sequence_infos[i]['consecutive_rises']
                consecutive_falls = sequence_infos[i]['consecutive_falls']
                
                boosted = self.extreme_booster.boost_prediction(
                    original, m14b, m14a, model_name,
                    sequence_info=sequence_infos[i]
                )
                
                # 🔥 ExtremeNet의 경우 추세별 최대값 적용
                if model_name == 'ExtremeNet':
                    min_val = all_predictions.iloc[i]['ExtremeNet_최소값']
                    max_val = all_predictions.iloc[i]['ExtremeNet_최대값']
                    
                    # 추세별 예측값 조정
                    if 'extreme_rising' in seq_trend or 'strong_rising' in seq_trend:
                        # 극단 상승: 최대값의 95-100%
                        boosted = max_val * 0.95 + (max_val - min_val) * 0.05
                        if consecutive_rises >= 15:
                            boosted = max_val  # 15회 이상 연속 상승시 최대값
                        elif consecutive_rises >= 10:
                            boosted = max_val * 0.98
                            
                    elif 'rapid_increasing' in seq_trend or 'high_increasing' in seq_trend:
                        # 빠른 상승: 최대값의 85-95%
                        boosted = min_val + (max_val - min_val) * 0.85
                        if consecutive_rises >= 7:
                            boosted = min_val + (max_val - min_val) * 0.90
                            
                    elif seq_trend == 'increasing':
                        # 일반 상승: 최대값의 70-85%
                        boosted = min_val + (max_val - min_val) * 0.70
                        if consecutive_rises >= 5:
                            boosted = min_val + (max_val - min_val) * 0.80
                            
                    elif 'extreme_falling' in seq_trend or 'strong_falling' in seq_trend:
                        # 극단 하락: 최소값의 100-120%
                        boosted = min_val * 1.05
                        if consecutive_falls >= 15:
                            boosted = min_val  # 15회 이상 연속 하락시 최소값
                        elif consecutive_falls >= 10:
                            boosted = min_val * 1.10
                            
                    elif 'rapid_decreasing' in seq_trend or 'high_decreasing' in seq_trend:
                        # 빠른 하락: 최소값의 120-140%
                        boosted = min_val * 1.20
                        if consecutive_falls >= 7:
                            boosted = min_val * 1.15
                            
                    elif seq_trend == 'decreasing':
                        # 일반 하락: 최소값의 140-160%
                        boosted = min_val * 1.40
                        if consecutive_falls >= 5:
                            boosted = min_val * 1.30
                            
                    elif 'high_stable' in seq_trend:
                        # 고평원: 범위 중간값
                        boosted = (min_val + max_val) / 2
                        
                    else:  # stable
                        # 안정: 원본과 최대값의 중간
                        boosted = (original + max_val) / 2
                    
                    # 최종적으로 범위 내로 제한
                    boosted = np.clip(boosted, min_val, max_val)
                
                boosted_predictions.append(boosted)
            
            all_predictions[f'{model_name}_예측'] = [round(p) for p in boosted_predictions]
            
            # ExtremeNet 범위 내 여부 체크
            if model_name == 'ExtremeNet':
                in_range = []
                for i in range(len(all_predictions)):
                    pred = all_predictions.iloc[i]['ExtremeNet_예측']
                    min_val = all_predictions.iloc[i]['ExtremeNet_최소값']
                    max_val = all_predictions.iloc[i]['ExtremeNet_최대값']
                    in_range.append('O' if min_val <= pred <= max_val else 'X')
                
                all_predictions['ExtremeNet_범위내'] = in_range
                
                in_range_count = in_range.count('O')
                print(f"    범위 내 예측: {in_range_count}/{len(in_range)} "
                      f"({in_range_count/len(in_range)*100:.1f}%)")
            
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
            
            print(f"  ✅ {model_name} 완료: MAE={mae:.2f}, 정확도={100-mape:.2f}%")
        
        # 앙상블 생성
        print("\n🔥 개선된 앙상블 생성...")
        
        extreme_ensemble = []
        for i in range(len(all_predictions)):
            m14b = all_predictions.iloc[i]['M14AM14B']
            seq_max = all_predictions.iloc[i]['시퀀스_MAX']
            seq_trend = all_predictions.iloc[i]['시퀀스_추세']
            
            # 각 모델 예측값
            extreme_pred = all_predictions.iloc[i]['ExtremeNet_예측']
            spike_pred = all_predictions.iloc[i]['SpikeDetector_예측'] if 'SpikeDetector_예측' in all_predictions.columns else extreme_pred
            golden_pred = all_predictions.iloc[i]['GoldenRule_예측'] if 'GoldenRule_예측' in all_predictions.columns else extreme_pred
            patch_pred = all_predictions.iloc[i]['PatchTST_예측'] if 'PatchTST_예측' in all_predictions.columns else extreme_pred
            stable_pred = all_predictions.iloc[i]['StableLSTM_예측'] if 'StableLSTM_예측' in all_predictions.columns else extreme_pred
            
            # 동적 가중치
            if seq_max >= 1650 and 'increasing' in seq_trend:
                weights = [0.35, 0.25, 0.20, 0.10, 0.10]  # ExtremeNet 중심
            elif seq_max >= 1600:
                weights = [0.25, 0.20, 0.20, 0.20, 0.15]  # 균형
            else:
                if m14b > 200:
                    weights = [0.20, 0.30, 0.25, 0.15, 0.10]  # Spike/Golden 중심
                else:
                    weights = [0.20, 0.15, 0.15, 0.30, 0.20]  # Patch/Stable 중심
            
            ensemble_pred = (
                extreme_pred * weights[0] +
                spike_pred * weights[1] +
                golden_pred * weights[2] +
                patch_pred * weights[3] +
                stable_pred * weights[4]
            )
            
            extreme_ensemble.append(ensemble_pred)
        
        all_predictions['앙상블_예측'] = [round(p) for p in extreme_ensemble]
        all_predictions['앙상블_오차'] = all_predictions['앙상블_예측'] - all_predictions['실제값']
        all_predictions['앙상블_오차율(%)'] = round(
            abs(all_predictions['앙상블_오차']) / all_predictions['실제값'] * 100, 2
        )
        
        # 앙상블 성능
        ensemble_mae = mean_absolute_error(all_predictions['실제값'], extreme_ensemble)
        ensemble_rmse = np.sqrt(mean_squared_error(all_predictions['실제값'], extreme_ensemble))
        ensemble_r2 = r2_score(all_predictions['실제값'], extreme_ensemble)
        ensemble_mape = np.mean(abs(all_predictions['앙상블_오차']) / all_predictions['실제값']) * 100
        
        model_metrics['앙상블'] = {
            'MAE': ensemble_mae,
            'RMSE': ensemble_rmse,
            'R2': ensemble_r2,
            'MAPE': ensemble_mape,
            '정확도(%)': 100 - ensemble_mape
        }
        
        print(f"✅ 앙상블: MAE={ensemble_mae:.2f}, 정확도={100-ensemble_mape:.2f}%")
        
        # CSV 저장
        output_file = f'v67_extreme_range_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        all_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 예측값 저장: {output_file}")
        
        # ==================== ExtremeNet 범위값 분석 ====================
        print("\n" + "="*80)
        print("🎯 ExtremeNet 범위값 상세 분석")
        print("="*80)
        
        # 범위 내 실제값 분석
        actual_in_range = 0
        for i in range(len(all_predictions)):
            actual = all_predictions.iloc[i]['실제값']
            min_val = all_predictions.iloc[i]['ExtremeNet_최소값']
            max_val = all_predictions.iloc[i]['ExtremeNet_최대값']
            if min_val <= actual <= max_val:
                actual_in_range += 1
        
        print(f"\n📊 범위값 성능:")
        print(f"  실제값이 범위 내: {actual_in_range}/{len(all_predictions)} "
              f"({actual_in_range/len(all_predictions)*100:.1f}%)")
        
        # 고값 구간 범위 분석
        high_mask = all_predictions['실제값'] >= 1700
        if high_mask.any():
            high_df = all_predictions[high_mask]
            
            print(f"\n🎯 고값 구간(1700+) 범위 분석:")
            print(f"  샘플 수: {len(high_df)}개")
            
            # 고값 구간 범위 내 실제값
            high_actual_in_range = 0
            for idx in high_df.index:
                actual = high_df.loc[idx, '실제값']
                min_val = high_df.loc[idx, 'ExtremeNet_최소값']
                max_val = high_df.loc[idx, 'ExtremeNet_최대값']
                if min_val <= actual <= max_val:
                    high_actual_in_range += 1
            
            print(f"  실제값 범위 내: {high_actual_in_range}/{len(high_df)} "
                  f"({high_actual_in_range/len(high_df)*100:.1f}%)")
            
            # 범위 폭 통계
            range_widths = high_df['ExtremeNet_최대값'] - high_df['ExtremeNet_최소값']
            print(f"  평균 범위 폭: {range_widths.mean():.0f}")
            print(f"  최대 범위 폭: {range_widths.max():.0f}")
            print(f"  최소 범위 폭: {range_widths.min():.0f}")
            
            # 퍼센트 통계
            print(f"  평균 퍼센트: {high_df['ExtremeNet_퍼센트'].mean():.1f}%")
            print(f"  최대 퍼센트: {high_df['ExtremeNet_퍼센트'].max():.1f}%")
            
            # 범위 예시
            print(f"\n[고값 구간 ExtremeNet 추세별 예측] (처음 15개)")
            print("-" * 110)
            print(f"{'실제값':>7} {'원본':>7} {'예측':>7} {'범위값':>15} {'추세':>15} {'연속↑':>6} {'연속↓':>6} {'적중':>6}")
            print("-" * 110)
            
            for idx in high_df.head(15).index:
                row = all_predictions.loc[idx]
                actual = row['실제값']
                orig = row['ExtremeNet_원본']
                pred = row['ExtremeNet_예측']
                range_val = row['ExtremeNet_범위값']
                seq_trend = row['시퀀스_추세']
                
                # 연속 상승/하락 카운트 (self.sequence_infos에서 가져오기)
                idx_pos = list(all_predictions.index).index(idx)
                consecutive_rises = self.sequence_infos[idx_pos]['consecutive_rises']
                consecutive_falls = self.sequence_infos[idx_pos]['consecutive_falls']
                
                # 적중 판정
                error = abs(pred - actual)
                if error <= 50:
                    hit = '✅'
                elif error <= 100:
                    hit = '⚠️'
                else:
                    hit = '❌'
                
                # 추세 아이콘
                if 'rising' in seq_trend:
                    trend_icon = '🚀'
                elif 'increasing' in seq_trend:
                    trend_icon = '📈'
                elif 'falling' in seq_trend:
                    trend_icon = '💥'
                elif 'decreasing' in seq_trend:
                    trend_icon = '📉'
                elif 'stable' in seq_trend:
                    trend_icon = '➡️'
                else:
                    trend_icon = '🔄'
                
                print(f"{actual:7.0f} {orig:7.0f} {pred:7.0f} {range_val:>15} "
                      f"{trend_icon} {seq_trend[:12]:>12} {consecutive_rises:6} "
                      f"{consecutive_falls:6} {hit:>6}")
            
            # 추세별 성능 분석
            print(f"\n[추세별 ExtremeNet 성능]")
            print("-" * 80)
            
            trend_types = ['extreme_rising', 'strong_rising', 'increasing', 
                          'stable', 'decreasing', 'strong_falling']
            
            for trend in trend_types:
                trend_mask = high_df['시퀀스_추세'] == trend
                if trend_mask.sum() > 0:
                    trend_df = high_df[trend_mask]
                    
                    # ExtremeNet 예측 정확도
                    mae = mean_absolute_error(trend_df['실제값'], trend_df['ExtremeNet_예측'])
                    
                    # 평균 예측값
                    avg_pred = trend_df['ExtremeNet_예측'].mean()
                    avg_actual = trend_df['실제값'].mean()
                    
                    print(f"  {trend:20s}: {trend_mask.sum():3d}개, "
                          f"MAE={mae:6.1f}, 평균예측={avg_pred:7.0f}, 평균실제={avg_actual:7.0f}")
        
        # 성능 요약
        print("\n" + "="*80)
        print("📊 전체 모델 성능 요약")
        print("="*80)
        
        metrics_df = pd.DataFrame(model_metrics).T
        metrics_df = metrics_df.sort_values('R2', ascending=False)
        
        print(f"\n{'모델':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'정확도(%)':>10}")
        print("-" * 55)
        
        for model_name, row in metrics_df.iterrows():
            if model_name == '앙상블':
                print(f"{'🔥 ' + model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                      f"{row['R2']:8.4f} {row['정확도(%)']:10.2f} ⭐")
            else:
                print(f"{model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                      f"{row['R2']:8.4f} {row['정확도(%)']:10.2f}")
        
        return all_predictions, metrics_df

def main():
    """메인 실행"""
    
    print("\n🚀 V6.7 ExtremeNet 추세별 최대값 예측 시작!")
    print("="*60)
    print("🔥 핵심 전략:")
    print("  - 극단 상승(15회+): 최대값의 95-100% 예측")
    print("  - 강한 상승(10회+): 최대값의 85-95% 예측")
    print("  - 일반 상승(5회+): 최대값의 70-85% 예측")
    print("  - 극단 하락(15회+): 최소값의 100-120% 예측")
    print("  - 범위: 원본의 80-160% (추세별 동적)")
    print("="*60)
    
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
    print(f"  1. v67_extreme_range_YYYYMMDD.csv")
    print("\n🔥 핵심 기능:")
    print("  ✅ ExtremeNet_퍼센트: 시퀀스 MAX 대비 증가율")
    print("  ✅ ExtremeNet_범위값: 예측 가능 범위 (최소~최대)")
    print("  ✅ ExtremeNet_범위내: 예측값이 범위 내인지 체크")
    print("  ✅ 실제값 범위 포함률 분석")
    print("="*80)

if __name__ == "__main__":
    main()