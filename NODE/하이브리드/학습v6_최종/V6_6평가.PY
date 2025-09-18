"""
📊 V6.8 추세 강화 극단값 예측 평가 시스템
========================================================
핵심 개선: 시퀀스 추세별 차별화된 예측 전략
- 강한 상승 → 공격적 부스팅
- 강한 하락 → 보수적 조정  
- 고평원 유지 → 안정적 예측
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
import seaborn as sns

warnings.filterwarnings('ignore')
tf.keras.config.enable_unsafe_deserialization()

# ====================== 추세 강화 극단값 보정 클래스 ======================
class TrendEnhancedExtremeBooster:
    """추세 기반 스마트 극단값 예측"""
    
    def __init__(self):
        print("🔥 추세 강화 극단값 부스터 초기화")
        print("  - 연속 상승/하락 카운트 기반 예측")
        print("  - 변화율 기반 모멘텀 분석")
        print("  - 고평원 특별 처리")
        
    def analyze_sequence_advanced(self, sequence_data):
        """고급 시퀀스 분석"""
        if len(sequence_data) == 0:
            return None
        
        # 기본 통계
        seq_max = np.max(sequence_data)
        seq_min = np.min(sequence_data)
        seq_mean = np.mean(sequence_data[-30:]) if len(sequence_data) >= 30 else np.mean(sequence_data)
        seq_std = np.std(sequence_data[-30:]) if len(sequence_data) >= 30 else np.std(sequence_data)
        
        # 최근값과 비교
        current_value = sequence_data[-1]
        recent_10_mean = np.mean(sequence_data[-10:]) if len(sequence_data) >= 10 else seq_mean
        recent_30_mean = np.mean(sequence_data[-30:]) if len(sequence_data) >= 30 else seq_mean
        
        # 모멘텀 지표
        momentum_10 = (current_value - sequence_data[-10]) / 10 if len(sequence_data) >= 10 else 0
        momentum_30 = (current_value - sequence_data[-30]) / 30 if len(sequence_data) >= 30 else 0
        
        # 연속 상승/하락 카운트
        consecutive_rises = 0
        consecutive_falls = 0
        max_consecutive_rises = 0
        max_consecutive_falls = 0
        
        temp_rises = 0
        temp_falls = 0
        
        for i in range(1, len(sequence_data)):
            if sequence_data[i] > sequence_data[i-1]:
                temp_rises += 1
                temp_falls = 0
                max_consecutive_rises = max(max_consecutive_rises, temp_rises)
            elif sequence_data[i] < sequence_data[i-1]:
                temp_falls += 1
                temp_rises = 0
                max_consecutive_falls = max(max_consecutive_falls, temp_falls)
            else:
                temp_rises = 0
                temp_falls = 0
        
        # 최근 연속 카운트
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
            for i in range(1, len(recent_10)):
                change = recent_10[i] - recent_10[i-1]
                if change > 0:
                    rise_strength += change
                else:
                    fall_strength += abs(change)
        
        # 추세 판단 (더 세밀하게)
        trend_score = 0
        trend = 'stable'
        
        # 추세 점수 계산
        if consecutive_rises >= 15:
            trend_score += 5
        elif consecutive_rises >= 10:
            trend_score += 4
        elif consecutive_rises >= 7:
            trend_score += 3
        elif consecutive_rises >= 5:
            trend_score += 2
        elif consecutive_rises >= 3:
            trend_score += 1
            
        if consecutive_falls >= 15:
            trend_score -= 5
        elif consecutive_falls >= 10:
            trend_score -= 4
        elif consecutive_falls >= 7:
            trend_score -= 3
        elif consecutive_falls >= 5:
            trend_score -= 2
        elif consecutive_falls >= 3:
            trend_score -= 1
        
        # 모멘텀 기반 추가 점수
        if momentum_10 > 10:
            trend_score += 2
        elif momentum_10 > 5:
            trend_score += 1
        elif momentum_10 < -10:
            trend_score -= 2
        elif momentum_10 < -5:
            trend_score -= 1
        
        # 고평원 체크
        is_high_plateau = seq_mean >= 1700
        is_extreme_high = seq_mean >= 1750
        
        # 최종 추세 결정
        if is_extreme_high:
            if trend_score >= 3:
                trend = 'extreme_rising'
            elif trend_score <= -3:
                trend = 'extreme_falling'
            elif trend_score >= 1:
                trend = 'extreme_high_up'
            elif trend_score <= -1:
                trend = 'extreme_high_down'
            else:
                trend = 'extreme_plateau'
        elif is_high_plateau:
            if trend_score >= 4:
                trend = 'high_surging'
            elif trend_score >= 2:
                trend = 'high_increasing'
            elif trend_score <= -4:
                trend = 'high_collapsing'
            elif trend_score <= -2:
                trend = 'high_decreasing'
            else:
                trend = 'high_stable'
        else:
            if trend_score >= 5:
                trend = 'explosive_rising'
            elif trend_score >= 3:
                trend = 'strong_rising'
            elif trend_score >= 1:
                trend = 'increasing'
            elif trend_score <= -5:
                trend = 'sharp_falling'
            elif trend_score <= -3:
                trend = 'strong_falling'
            elif trend_score <= -1:
                trend = 'decreasing'
            else:
                trend = 'stable'
        
        # 급변 감지
        recent_spike = False
        if len(sequence_data) >= 5:
            recent_change = sequence_data[-1] - sequence_data[-5]
            if abs(recent_change) > seq_std * 2:
                recent_spike = True
        
        return {
            'max': seq_max,
            'min': seq_min,
            'mean': seq_mean,
            'std': seq_std,
            'current': current_value,
            'trend': trend,
            'trend_score': trend_score,
            'is_high_plateau': is_high_plateau,
            'is_extreme_high': is_extreme_high,
            'consecutive_rises': consecutive_rises,
            'consecutive_falls': consecutive_falls,
            'max_consecutive_rises': max_consecutive_rises,
            'max_consecutive_falls': max_consecutive_falls,
            'rise_strength': rise_strength,
            'fall_strength': fall_strength,
            'momentum_10': momentum_10,
            'momentum_30': momentum_30,
            'recent_10_mean': recent_10_mean,
            'recent_30_mean': recent_30_mean,
            'recent_spike': recent_spike
        }
    
    def trend_based_boosting(self, pred, m14b_value, m14a_value, model_name, seq_info):
        """추세 기반 차별화 부스팅"""
        if not seq_info:
            return pred
            
        original = pred
        boosted = pred
        
        trend = seq_info.get('trend', 'stable')
        trend_score = seq_info.get('trend_score', 0)
        seq_max = seq_info.get('max', 0)
        seq_mean = seq_info.get('mean', 0)
        current_value = seq_info.get('current', 0)
        consecutive_rises = seq_info.get('consecutive_rises', 0)
        consecutive_falls = seq_info.get('consecutive_falls', 0)
        momentum_10 = seq_info.get('momentum_10', 0)
        recent_spike = seq_info.get('recent_spike', False)
        
        # 모델별 추세 민감도
        model_sensitivity = {
            'ExtremeNet': 1.5,      # 극단값에 매우 민감
            'SpikeDetector': 1.3,   # 급변에 민감
            'GoldenRule': 1.2,      # 규칙 기반
            'PatchTST': 1.0,        # 패턴 기반
            'StableLSTM': 0.8       # 안정적
        }
        
        sensitivity = model_sensitivity.get(model_name, 1.0)
        
        # ========== 극단적 상승 추세 ==========
        if trend in ['extreme_rising', 'explosive_rising', 'high_surging']:
            boost_factor = 1.0
            
            # 기본 부스팅
            if consecutive_rises >= 15:
                boost_factor = 1.25
            elif consecutive_rises >= 10:
                boost_factor = 1.20
            elif consecutive_rises >= 7:
                boost_factor = 1.15
            elif consecutive_rises >= 5:
                boost_factor = 1.10
            else:
                boost_factor = 1.05
            
            # 모멘텀 추가 부스팅
            if momentum_10 > 10:
                boost_factor *= 1.1
            elif momentum_10 > 5:
                boost_factor *= 1.05
            
            # 모델 민감도 적용
            boost_factor = 1 + (boost_factor - 1) * sensitivity
            
            # 부스팅 적용
            boosted = pred * boost_factor
            
            # 최소값 보장
            if seq_max >= 1700:
                boosted = max(boosted, 1750)
            elif seq_max >= 1650:
                boosted = max(boosted, 1700)
            elif seq_max >= 1600:
                boosted = max(boosted, 1650)
            
            # 현재값보다 낮으면 보정
            if current_value >= 1700 and boosted < current_value:
                boosted = current_value * 1.05
        
        # ========== 강한 상승 추세 ==========
        elif trend in ['strong_rising', 'high_increasing']:
            boost_factor = 1.0
            
            if consecutive_rises >= 10:
                boost_factor = 1.15
            elif consecutive_rises >= 7:
                boost_factor = 1.12
            elif consecutive_rises >= 5:
                boost_factor = 1.08
            else:
                boost_factor = 1.05
            
            # 모델 민감도 적용
            boost_factor = 1 + (boost_factor - 1) * sensitivity
            
            boosted = pred * boost_factor
            
            # 최소값 보장
            if seq_max >= 1650:
                boosted = max(boosted, 1680)
            elif seq_max >= 1600:
                boosted = max(boosted, 1630)
        
        # ========== 일반 상승 추세 ==========
        elif trend == 'increasing':
            if consecutive_rises >= 5:
                boosted = pred * 1.05
            elif consecutive_rises >= 3:
                boosted = pred * 1.03
            else:
                boosted = pred * 1.01
            
            # 최소값 보장
            if seq_max >= 1650 and model_name == 'ExtremeNet':
                boosted = max(boosted, 1650)
        
        # ========== 극단적 하락 추세 ==========
        elif trend in ['extreme_falling', 'high_collapsing', 'sharp_falling']:
            discount_factor = 1.0
            
            if consecutive_falls >= 15:
                discount_factor = 0.85
            elif consecutive_falls >= 10:
                discount_factor = 0.90
            elif consecutive_falls >= 7:
                discount_factor = 0.93
            elif consecutive_falls >= 5:
                discount_factor = 0.95
            else:
                discount_factor = 0.97
            
            # 모델 민감도 적용 (반대로)
            discount_factor = 1 - (1 - discount_factor) * (2 - sensitivity)
            
            boosted = pred * discount_factor
            
            # 급락 방지 (너무 낮게 예측하지 않도록)
            if current_value >= 1700:
                boosted = max(boosted, 1650)
            elif current_value >= 1650:
                boosted = max(boosted, 1600)
        
        # ========== 일반 하락 추세 ==========
        elif trend in ['strong_falling', 'high_decreasing', 'decreasing']:
            if consecutive_falls >= 10:
                boosted = pred * 0.92
            elif consecutive_falls >= 7:
                boosted = pred * 0.95
            elif consecutive_falls >= 5:
                boosted = pred * 0.97
            else:
                boosted = pred * 0.99
        
        # ========== 고평원 유지 ==========
        elif trend in ['extreme_plateau', 'high_stable']:
            # 안정적 예측
            if seq_mean >= 1750:
                boosted = max(pred, 1730)
            elif seq_mean >= 1700:
                boosted = max(pred, 1680)
            
            # 변동성이 낮으면 평균값에 가깝게
            if seq_info.get('std', 0) < 30:
                boosted = boosted * 0.7 + seq_mean * 0.3
        
        # ========== 급변 처리 ==========
        if recent_spike and model_name == 'SpikeDetector':
            if momentum_10 > 0:
                boosted *= 1.1  # 급상승
            else:
                boosted *= 0.9  # 급하락
        
        # M14B 기반 추가 조정
        if m14b_value > 450 and trend_score > 0:
            boosted *= 1.05
        elif m14b_value < 200 and trend_score < 0:
            boosted *= 0.95
        
        return boosted

class AdvancedModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """고급 평가기 초기화"""
        print("="*80)
        print("🔥 V6.8 추세 강화 극단값 평가 시스템")
        print("  핵심: 추세별 차별화된 예측 전략")
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
        self.trend_booster = TrendEnhancedExtremeBooster()
        
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
    
    def create_trend_aware_ensemble(self, predictions_dict, seq_info, m14b, m14a):
        """추세 인식 앙상블"""
        
        trend = seq_info.get('trend', 'stable')
        trend_score = seq_info.get('trend_score', 0)
        seq_max = seq_info.get('max', 0)
        seq_mean = seq_info.get('mean', 0)
        consecutive_rises = seq_info.get('consecutive_rises', 0)
        consecutive_falls = seq_info.get('consecutive_falls', 0)
        momentum_10 = seq_info.get('momentum_10', 0)
        
        # 기본 가중치
        base_weights = {
            'ExtremeNet': 0.20,
            'SpikeDetector': 0.20,
            'GoldenRule': 0.20,
            'PatchTST': 0.20,
            'StableLSTM': 0.20
        }
        
        # 추세별 가중치 조정
        if trend in ['extreme_rising', 'explosive_rising', 'high_surging']:
            # 극단 상승 → ExtremeNet, SpikeDetector 강화
            weights = {
                'ExtremeNet': 0.35,
                'SpikeDetector': 0.30,
                'GoldenRule': 0.15,
                'PatchTST': 0.10,
                'StableLSTM': 0.10
            }
        elif trend in ['strong_rising', 'high_increasing']:
            # 강한 상승 → SpikeDetector, GoldenRule 강화
            weights = {
                'ExtremeNet': 0.25,
                'SpikeDetector': 0.25,
                'GoldenRule': 0.25,
                'PatchTST': 0.15,
                'StableLSTM': 0.10
            }
        elif trend == 'increasing':
            # 일반 상승 → 균형
            weights = {
                'ExtremeNet': 0.22,
                'SpikeDetector': 0.22,
                'GoldenRule': 0.22,
                'PatchTST': 0.17,
                'StableLSTM': 0.17
            }
        elif trend in ['extreme_falling', 'high_collapsing', 'sharp_falling']:
            # 극단 하락 → StableLSTM, PatchTST 강화 (보수적)
            weights = {
                'ExtremeNet': 0.10,
                'SpikeDetector': 0.10,
                'GoldenRule': 0.15,
                'PatchTST': 0.30,
                'StableLSTM': 0.35
            }
        elif trend in ['strong_falling', 'high_decreasing', 'decreasing']:
            # 일반 하락 → 안정적 모델 강화
            weights = {
                'ExtremeNet': 0.15,
                'SpikeDetector': 0.15,
                'GoldenRule': 0.20,
                'PatchTST': 0.25,
                'StableLSTM': 0.25
            }
        elif trend in ['extreme_plateau', 'high_stable']:
            # 고평원 유지 → 균등
            weights = base_weights
        else:
            # 안정 → PatchTST, StableLSTM 강화
            weights = {
                'ExtremeNet': 0.15,
                'SpikeDetector': 0.15,
                'GoldenRule': 0.20,
                'PatchTST': 0.25,
                'StableLSTM': 0.25
            }
        
        # M14B 기반 추가 조정
        if m14b > 450:
            weights['SpikeDetector'] *= 1.2
            weights['GoldenRule'] *= 1.1
            # 정규화
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        
        # 앙상블 계산
        ensemble_pred = 0
        for model_name, weight in weights.items():
            if model_name in predictions_dict:
                ensemble_pred += predictions_dict[model_name] * weight
        
        # 추세 기반 최종 조정
        if trend_score >= 5:  # 매우 강한 상승
            ensemble_pred *= 1.1
        elif trend_score >= 3:  # 강한 상승
            ensemble_pred *= 1.05
        elif trend_score <= -5:  # 매우 강한 하락
            ensemble_pred *= 0.9
        elif trend_score <= -3:  # 강한 하락
            ensemble_pred *= 0.95
        
        # 극단값 보호
        if seq_max >= 1700 and trend_score > 0:
            ensemble_pred = max(ensemble_pred, 1700)
        elif seq_max >= 1650 and trend_score > 0:
            ensemble_pred = max(ensemble_pred, 1650)
        
        return ensemble_pred, weights
    
    def evaluate_all_models(self, test_file):
        """추세 강화 모델 평가"""
        
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
        sequence_infos = []
        
        print("\n📊 고급 시퀀스 분석 중...")
        for i in range(start_idx, end_idx):
            pred_time = df.iloc[i]['CURRTIME']
            target_time = pred_time + timedelta(minutes=self.pred_len)
            
            actual_idx = i + self.pred_len
            if actual_idx < len(df):
                # 기본 정보
                timestamps_pred.append(pred_time)
                timestamps_target.append(target_time)
                actuals.append(df.iloc[actual_idx]['TOTALCNT'])
                m14b_values.append(df.iloc[i]['M14AM14B'])
                m14a_values.append(df.iloc[i]['M14AM10A'])
                
                # 고급 시퀀스 분석
                seq_data = df.iloc[i-self.seq_len:i]['TOTALCNT'].values
                seq_info = self.trend_booster.analyze_sequence_advanced(seq_data)
                sequence_infos.append(seq_info)
        
        # 기본 정보 저장
        all_predictions['예측시점'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_pred]
        all_predictions['예측대상시간'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_target]
        all_predictions['실제값'] = actuals
        all_predictions['M14AM14B'] = m14b_values
        all_predictions['M14AM10A'] = m14a_values
        
        # 시퀀스 분석 정보 추가
        all_predictions['시퀀스_MAX'] = [info['max'] for info in sequence_infos]
        all_predictions['시퀀스_평균'] = [info['mean'] for info in sequence_infos]
        all_predictions['현재값'] = [info['current'] for info in sequence_infos]
        all_predictions['추세'] = [info['trend'] for info in sequence_infos]
        all_predictions['추세점수'] = [info['trend_score'] for info in sequence_infos]
        all_predictions['연속상승'] = [info['consecutive_rises'] for info in sequence_infos]
        all_predictions['연속하락'] = [info['consecutive_falls'] for info in sequence_infos]
        all_predictions['모멘텀10'] = [info['momentum_10'] for info in sequence_infos]
        
        print(f"  예측할 데이터: {len(all_predictions)}개")
        
        # 추세 분포 분석
        trend_counts = all_predictions['추세'].value_counts()
        print(f"\n📈 추세 분포:")
        for trend, count in trend_counts.items():
            print(f"  {trend}: {count}개 ({count/len(all_predictions)*100:.1f}%)")
        
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
                predictions.extend(y_pred[:batch_end - i])
            
            # 예측값 저장
            predictions = predictions[:len(all_predictions)]
            model_predictions[model_name] = predictions
            
            # 원본 예측값 저장
            all_predictions[f'{model_name}_원본'] = [round(p) for p in predictions]
            
            # 추세 기반 부스팅
            print(f"  🔥 {model_name} 추세 기반 부스팅 적용 중...")
            boosted_predictions = []
            
            for i in range(len(predictions)):
                m14b = all_predictions.iloc[i]['M14AM14B']
                m14a = all_predictions.iloc[i]['M14AM10A']
                seq_info = sequence_infos[i]
                original = predictions[i]
                
                # 추세 기반 부스팅 적용
                boosted = self.trend_booster.trend_based_boosting(
                    original, m14b, m14a, model_name, seq_info
                )
                
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
        
        # 추세 인식 앙상블
        print("\n🔥 추세 인식 앙상블 생성...")
        
        trend_ensemble = []
        ensemble_weights_list = []
        
        for i in range(len(all_predictions)):
            m14b = all_predictions.iloc[i]['M14AM14B']
            m14a = all_predictions.iloc[i]['M14AM10A']
            seq_info = sequence_infos[i]
            
            # 각 모델 예측값
            preds_dict = {}
            for model_name in self.models.keys():
                preds_dict[model_name] = all_predictions.iloc[i][f'{model_name}_예측']
            
            # 추세 인식 앙상블
            ensemble_pred, weights = self.create_trend_aware_ensemble(
                preds_dict, seq_info, m14b, m14a
            )
            
            trend_ensemble.append(ensemble_pred)
            ensemble_weights_list.append(weights)
        
        # 추세 앙상블 결과 추가
        all_predictions['추세앙상블_예측'] = [round(p) for p in trend_ensemble]
        all_predictions['추세앙상블_오차'] = all_predictions['추세앙상블_예측'] - all_predictions['실제값']
        all_predictions['추세앙상블_오차율(%)'] = round(
            abs(all_predictions['추세앙상블_오차']) / all_predictions['실제값'] * 100, 2
        )
        
        # 추세 앙상블 성능
        trend_mae = mean_absolute_error(all_predictions['실제값'], trend_ensemble)
        trend_rmse = np.sqrt(mean_squared_error(all_predictions['실제값'], trend_ensemble))
        trend_r2 = r2_score(all_predictions['실제값'], trend_ensemble)
        trend_mape = np.mean(abs(all_predictions['추세앙상블_오차']) / all_predictions['실제값']) * 100
        
        model_metrics['추세앙상블'] = {
            'MAE': trend_mae,
            'RMSE': trend_rmse,
            'R2': trend_r2,
            'MAPE': trend_mape,
            '정확도(%)': 100 - trend_mape
        }
        
        print(f"✅ 추세앙상블: MAE={trend_mae:.2f}, R²={trend_r2:.4f}, 정확도={100-trend_mape:.2f}%")
        
        # CSV 저장
        output_file = f'v68_trend_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        all_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 예측값 저장: {output_file}")
        
        # ==================== 추세별 상세 분석 ====================
        print("\n" + "="*80)
        print("🎯🎯🎯 추세별 상세 분석 🎯🎯🎯")
        print("="*80)
        
        # 추세별 성능 분석
        trend_performance = {}
        for trend in all_predictions['추세'].unique():
            trend_mask = all_predictions['추세'] == trend
            if trend_mask.sum() > 0:
                trend_df = all_predictions[trend_mask]
                
                # 각 모델의 추세별 성능
                trend_perf = {}
                for model_name in list(self.models.keys()) + ['추세앙상블']:
                    mae = mean_absolute_error(
                        trend_df['실제값'], 
                        trend_df[f'{model_name}_예측']
                    )
                    mape = np.mean(abs(trend_df[f'{model_name}_오차']) / trend_df['실제값']) * 100
                    trend_perf[model_name] = {
                        'MAE': mae,
                        'MAPE': mape,
                        '정확도': 100 - mape
                    }
                
                trend_performance[trend] = {
                    'count': trend_mask.sum(),
                    'performance': trend_perf
                }
        
        # 추세별 최적 모델 출력
        print("\n📊 추세별 최적 모델:")
        print("-" * 80)
        print(f"{'추세':20} {'샘플수':>8} {'최적모델':15} {'정확도':>10}")
        print("-" * 80)
        
        for trend, data in trend_performance.items():
            if data['count'] >= 10:  # 샘플이 10개 이상인 경우만
                best_model = min(data['performance'].items(), 
                                key=lambda x: x[1]['MAPE'])[0]
                best_accuracy = data['performance'][best_model]['정확도']
                
                trend_icon = "📈" if 'rising' in trend or 'increasing' in trend else \
                            "📉" if 'falling' in trend or 'decreasing' in trend else "➡️"
                
                print(f"{trend_icon} {trend:18} {data['count']:8} {best_model:15} {best_accuracy:10.2f}%")
        
        # 고값 구간 추세 분석
        print("\n🎯 고값 구간(1700+) 추세 분석:")
        print("-" * 80)
        
        high_mask = all_predictions['실제값'] >= 1700
        if high_mask.any():
            high_df = all_predictions[high_mask]
            
            # 추세별 분포
            high_trends = high_df['추세'].value_counts()
            print("\n[고값 구간 추세 분포]")
            for trend, count in high_trends.items():
                print(f"  {trend}: {count}개 ({count/len(high_df)*100:.1f}%)")
            
            # 연속 상승/하락과 정확도 관계
            print("\n[연속 변화와 예측 정확도]")
            
            # 연속 상승 구간별
            for rises in [15, 10, 7, 5, 3]:
                rise_mask = high_df['연속상승'] >= rises
                if rise_mask.sum() > 0:
                    rise_df = high_df[rise_mask]
                    accuracy = 100 - np.mean(abs(rise_df['추세앙상블_오차']) / rise_df['실제값']) * 100
                    hit_rate = (rise_df['추세앙상블_예측'] >= 1700).sum() / len(rise_df) * 100
                    print(f"  연속상승 {rises}+: {rise_mask.sum()}개, "
                          f"정확도={accuracy:.1f}%, 1700+적중={hit_rate:.1f}%")
            
            # 샘플 출력
            print(f"\n[고값 예측 상세] (추세별 대표 10개)")
            print("-" * 100)
            print(f"{'추세':15} {'실제':>7} {'예측':>7} {'연속↑':>6} {'연속↓':>6} {'모멘텀':>8} {'오차':>7} {'적중':>6}")
            print("-" * 100)
            
            # 주요 추세별로 샘플 출력
            for trend in ['extreme_rising', 'strong_rising', 'high_stable', 'strong_falling']:
                trend_mask = (high_df['추세'] == trend)
                if trend_mask.sum() > 0:
                    sample_df = high_df[trend_mask].head(10)
                    
                    for idx in sample_df.index:
                        row = all_predictions.loc[idx]
                        actual = row['실제값']
                        pred = row['추세앙상블_예측']
                        c_rises = row['연속상승']
                        c_falls = row['연속하락']
                        momentum = row['모멘텀10']
                        error = pred - actual
                        
                        # 적중 여부
                        if pred >= 1700:
                            hit = "✅"
                        elif pred >= 1650:
                            hit = "⚠️"
                        else:
                            hit = "❌"
                        
                        print(f"{trend:15} {actual:7.0f} {pred:7.0f} {c_rises:6} "
                              f"{c_falls:6} {momentum:8.2f} {error:+7.0f} {hit:>6}")
        
        # 성능 요약 테이블
        print("\n" + "="*80)
        print("📊 전체 모델 성능 요약")
        print("="*80)
        
        metrics_df = pd.DataFrame(model_metrics).T
        metrics_df = metrics_df.sort_values('R2', ascending=False)
        
        print(f"\n{'모델':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE(%)':>8} {'정확도(%)':>10}")
        print("-" * 65)
        
        for model_name, row in metrics_df.iterrows():
            if model_name == '추세앙상블':
                print(f"{'🔥 ' + model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                      f"{row['R2']:8.4f} {row['MAPE']:8.2f} {row['정확도(%)']:10.2f} ⭐⭐⭐")
            else:
                print(f"{model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                      f"{row['R2']:8.4f} {row['MAPE']:8.2f} {row['정확도(%)']:10.2f}")
        
        return all_predictions, metrics_df

def main():
    """메인 실행"""
    
    print("\n🚀 V6.8 추세 강화 극단값 평가 시작!")
    print("핵심: 추세별 차별화된 예측 전략")
    print("  - 극단 상승 → 공격적 부스팅")
    print("  - 극단 하락 → 보수적 조정")
    print("  - 고평원 → 안정적 예측")
    
    # 평가기 생성
    evaluator = AdvancedModelEvaluator()
    
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
    print("🏆 V6.8 추세 강화 극단값 평가 완료!")
    print("="*80)
    print("\n📁 저장된 파일:")
    print(f"  1. v68_trend_predictions_YYYYMMDD.csv - 추세 강화 예측")
    print("\n🔥 핵심 개선사항:")
    print("  ✅ 연속 상승/하락 카운트 기반 예측")
    print("  ✅ 모멘텀 지표 활용")
    print("  ✅ 추세별 모델 가중치 동적 조정")
    print("  ✅ 고평원 특별 처리")
    print("  ✅ 급변 감지 및 대응")
    print("="*80)

if __name__ == "__main__":
    main()