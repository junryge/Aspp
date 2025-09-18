"""
📊 V6.7 개선 - 추세 기반 동적 앙상블 시스템
========================================================
핵심 개선: 시퀀스 추세(증가/하락)에 따라 앙상블 가중치 동적 조정
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

# ====================== 개선된 극단값 보정 클래스 ======================
class ImprovedExtremeValueBooster:
    """시퀀스 기반 스마트 극단값 예측"""
    
    def __init__(self):
        print("🔥 개선된 극단값 부스터 초기화")
        print("  - 시퀀스 max값 1651 이상 체크")
        print("  - 증가/하락 추세 분석")
        print("  - 추세별 차별화된 부스팅")
        
    def analyze_sequence(self, sequence_data):
        """시퀀스 분석: max값, 추세, 연속 상승/하락 정도 계산"""
        if len(sequence_data) == 0:
            return None, 'stable', False, 0, 0, 0, 0
        
        # 1. 시퀀스 max값과 평균값
        seq_max = np.max(sequence_data)
        seq_min = np.min(sequence_data)
        seq_mean = np.mean(sequence_data[-30:]) if len(sequence_data) >= 30 else np.mean(sequence_data)
        
        # 2. 고평원 상태 체크 (최근 30개 평균이 1700 이상)
        is_high_plateau = seq_mean >= 1700
        
        # 3. 연속 상승 카운트 계산
        consecutive_rises = 0
        for i in range(len(sequence_data)-1, 0, -1):
            if sequence_data[i] > sequence_data[i-1]:
                consecutive_rises += 1
            else:
                break
        
        # 4. 연속 하락 카운트 계산
        consecutive_falls = 0
        for i in range(len(sequence_data)-1, 0, -1):
            if sequence_data[i] < sequence_data[i-1]:
                consecutive_falls += 1
            else:
                break
        
        # 5. 상승/하락 강도 계산 (최근 10개 데이터)
        rise_strength = 0
        fall_strength = 0
        if len(sequence_data) >= 10:
            recent_10 = sequence_data[-10:]
            change = recent_10[-1] - recent_10[0]  # 10분간 총 변화량
            if change > 0:
                rise_strength = change
            else:
                fall_strength = abs(change)
        
        # 6. 추세 분석 (마지막 30개 데이터)
        if len(sequence_data) >= 30:
            recent = sequence_data[-30:]
            x = np.arange(len(recent))
            coeffs = np.polyfit(x, recent, 1)
            slope = coeffs[0]
            
            # 고평원 상태에서의 추세 판단
            if is_high_plateau:
                # 고평원(1700+)에서의 극단적 변화 감지
                if consecutive_rises >= 10 and rise_strength > 50:
                    trend = 'extreme_rising'  # 극단적 상승
                elif consecutive_falls >= 10 and fall_strength > 50:
                    trend = 'extreme_falling'  # 극단적 하락 (고평원에서 급락)
                elif slope > 1 or consecutive_rises >= 5:
                    trend = 'high_increasing'  # 고평원 상승
                elif slope < -1 or consecutive_falls >= 5:
                    trend = 'high_decreasing'  # 고평원 하락
                else:
                    trend = 'high_stable'  # 고평원 유지
            else:
                # 일반 구간 추세 판단
                if consecutive_rises >= 10 and rise_strength > 50:
                    trend = 'strong_rising'  # 강한 연속 상승
                elif consecutive_falls >= 10 and fall_strength > 50:
                    trend = 'strong_falling'  # 강한 연속 하락
                elif consecutive_rises >= 7 and rise_strength > 30:
                    trend = 'rapid_increasing'  # 빠른 상승
                elif consecutive_falls >= 7 and fall_strength > 30:
                    trend = 'rapid_decreasing'  # 빠른 하락
                elif slope > 2:
                    trend = 'increasing'
                elif slope < -2:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
        else:
            trend = 'stable'
        
        # 변동성 지표 (최근 10개의 표준편차)
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
        """시퀀스 기반 스마트 부스팅 (상승/하락 감도 포함)"""
        if not sequence_info:
            return pred
            
        original = pred
        boosted = pred
        seq_max = sequence_info.get('max', 0)
        seq_trend = sequence_info.get('trend', 'stable')
        is_high_plateau = sequence_info.get('is_high_plateau', False)
        consecutive_rises = sequence_info.get('consecutive_rises', 0)
        consecutive_falls = sequence_info.get('consecutive_falls', 0)
        rise_strength = sequence_info.get('rise_strength', 0)
        fall_strength = sequence_info.get('fall_strength', 0)
        
        # ========== ExtremeNet 특별 처리 ==========
        if model_name == 'ExtremeNet':
            # 일반 상승 (기존 조건)
            if seq_max >= 1651 and seq_trend == 'increasing':
                print(f"    📈 ExtremeNet 부스팅 활성화! (max={seq_max:.0f})")
                if m14b_value > 200:  # M14B 기준을 200으로 낮춤
                    boosted = max(pred * 1.2, 1700)
                else:
                    boosted = max(pred * 1.15, 1700)
            else:
                boosted = pred
                
        # ========== SpikeDetector, GoldenRule ==========
        elif model_name in ['SpikeDetector', 'GoldenRule']:
            if seq_max >= 1651 and seq_trend == 'increasing':
                if m14b_value > 200:
                    boosted = max(pred * 1.15, 1700)
                else:
                    boosted = max(pred * 1.1, 1700)
            else:
                boosted = pred
            
            # 하락 추세에서는 추가 조정 없음
                
        # ========== 기타 모델 (PatchTST, StableLSTM) ==========
        else:
            if seq_max >= 1651 and seq_trend == 'increasing':
                if m14b_value > 200:
                    boosted = max(pred * 1.1, 1700)
                else:
                    boosted = max(pred * 1.05, 1700)
            else:
                boosted = pred
        
        return boosted

class ImprovedModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """개선된 평가기 초기화"""
        print("="*80)
        print("🔥 V6.7 개선 - 추세 기반 동적 앙상블")
        print("  핵심: 시퀀스 추세별 차별화된 앙상블 전략")
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
        max_1651_count = (df['TOTALCNT'] >= 1651).sum()
        
        print(f"\n🎯 고값 구간 분포:")
        print(f"  1651+: {max_1651_count}개 ({max_1651_count/len(df)*100:.1f}%) ← 기준값")
        print(f"  1700+: {high_count}개 ({high_count/len(df)*100:.1f}%)")
        print(f"  1750+: {very_high_count}개 ({very_high_count/len(df)*100:.1f}%)")
        print(f"  1800+: {extreme_count}개 ({extreme_count/len(df)*100:.1f}%)")
        
        # M14B 분포
        m14b_high = (df['M14AM14B'] > 450).sum()
        print(f"\n📊 M14AM14B 분포:")
        print(f"  450+: {m14b_high}개 ({m14b_high/len(df)*100:.1f}%)")
        
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
        """개선된 모델 평가 - 추세 기반 동적 앙상블"""
        
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
        sequence_maxes = []
        sequence_trends = []
        consecutive_rises_list = []
        consecutive_falls_list = []
        
        print("\n📊 데이터 수집 및 시퀀스 분석 중...")
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
                
                # 시퀀스 분석 (과거 100분 TOTALCNT)
                seq_data = df.iloc[i-self.seq_len:i]['TOTALCNT'].values
                seq_info = self.extreme_booster.analyze_sequence(seq_data)
                sequence_maxes.append(seq_info.get('max'))
                sequence_trends.append(seq_info.get('trend'))
                consecutive_rises_list.append(seq_info.get('consecutive_rises', 0))
                consecutive_falls_list.append(seq_info.get('consecutive_falls', 0))
        
        # 기본 정보 저장
        all_predictions['예측시점'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_pred]
        all_predictions['예측대상시간'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_target]
        all_predictions['실제값'] = actuals
        all_predictions['M14AM14B'] = m14b_values
        all_predictions['M14AM10A'] = m14a_values
        all_predictions['시퀀스_MAX'] = sequence_maxes
        all_predictions['시퀀스_추세'] = sequence_trends
        all_predictions['연속상승'] = consecutive_rises_list
        all_predictions['연속하락'] = consecutive_falls_list
        
        print(f"  예측할 데이터: {len(all_predictions)}개")
        
        # 시퀀스 분석 통계
        high_seq_count = (np.array(sequence_maxes) >= 1651).sum()
        inc_trend_count = (np.array(sequence_trends) == 'increasing').sum()
        dec_trend_count = (np.array(sequence_trends) == 'decreasing').sum()
        
        print(f"\n📈 시퀀스 분석 결과:")
        print(f"  max값 1651+: {high_seq_count}개 ({high_seq_count/len(all_predictions)*100:.1f}%)")
        print(f"  증가 추세: {inc_trend_count}개 ({inc_trend_count/len(all_predictions)*100:.1f}%)")
        print(f"  하락 추세: {dec_trend_count}개 ({dec_trend_count/len(all_predictions)*100:.1f}%)")
        
        # ExtremeNet 부스팅 조건 만족 개수
        boost_condition = ((np.array(sequence_maxes) >= 1651) & 
                          (np.array(sequence_trends) == 'increasing')).sum()
        print(f"  🔥 ExtremeNet 부스팅 조건 충족: {boost_condition}개 ({boost_condition/len(all_predictions)*100:.1f}%)")
        
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
                    'consecutive_rises': all_predictions.iloc[i]['연속상승'],
                    'consecutive_falls': all_predictions.iloc[i]['연속하락'],
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
        
        # ==================== 추세 기반 동적 앙상블 ====================
        print("\n🔥 추세 기반 동적 앙상블 생성...")
        print("  상승 추세 → 공격적 모델 강화")
        print("  하락 추세 → 보수적 모델 강화")
        
        trend_counts = {}
        extreme_ensemble = []
        
        for i in range(len(all_predictions)):
            m14b = all_predictions.iloc[i]['M14AM14B']
            m14a = all_predictions.iloc[i]['M14AM10A']
            seq_max = all_predictions.iloc[i]['시퀀스_MAX']
            seq_trend = all_predictions.iloc[i]['시퀀스_추세']
            consecutive_rises = all_predictions.iloc[i]['연속상승']
            consecutive_falls = all_predictions.iloc[i]['연속하락']
            
            # 각 모델 예측값
            extreme_pred = all_predictions.iloc[i]['ExtremeNet_예측']
            spike_pred = all_predictions.iloc[i]['SpikeDetector_예측']
            golden_pred = all_predictions.iloc[i]['GoldenRule_예측']
            patch_pred = all_predictions.iloc[i]['PatchTST_예측']
            stable_pred = all_predictions.iloc[i]['StableLSTM_예측']
            
            # 추세 카운트
            if seq_trend not in trend_counts:
                trend_counts[seq_trend] = 0
            trend_counts[seq_trend] += 1
            
            # ========== 추세별 가중치 전략 ==========
            
            # 1️⃣ 극단적 상승 추세 (연속 10회+ 상승)
            if seq_trend in ['extreme_rising', 'strong_rising'] or consecutive_rises >= 10:
                weights = {
                    'ExtremeNet': 0.35,
                    'SpikeDetector': 0.30,
                    'GoldenRule': 0.20,
                    'PatchTST': 0.10,
                    'StableLSTM': 0.05
                }
                
                ensemble_pred = (
                    extreme_pred * weights['ExtremeNet'] +
                    spike_pred * weights['SpikeDetector'] +
                    golden_pred * weights['GoldenRule'] +
                    patch_pred * weights['PatchTST'] +
                    stable_pred * weights['StableLSTM']
                )
                
                # 연속 상승 정도에 따른 추가 부스팅
                if consecutive_rises >= 15:
                    ensemble_pred *= 1.12
                elif consecutive_rises >= 10:
                    ensemble_pred *= 1.08
                else:
                    ensemble_pred *= 1.05
                    
                # 최소값 보장
                if seq_max >= 1700:
                    ensemble_pred = max(ensemble_pred, 1750)
                elif seq_max >= 1650:
                    ensemble_pred = max(ensemble_pred, 1700)
            
            # 2️⃣ 빠른 상승 추세 (연속 5-9회 상승)
            elif seq_trend in ['rapid_increasing', 'high_increasing'] or consecutive_rises >= 5:
                weights = {
                    'ExtremeNet': 0.25,
                    'SpikeDetector': 0.25,
                    'GoldenRule': 0.25,
                    'PatchTST': 0.15,
                    'StableLSTM': 0.10
                }
                
                ensemble_pred = (
                    extreme_pred * weights['ExtremeNet'] +
                    spike_pred * weights['SpikeDetector'] +
                    golden_pred * weights['GoldenRule'] +
                    patch_pred * weights['PatchTST'] +
                    stable_pred * weights['StableLSTM']
                )
                
                if consecutive_rises >= 7:
                    ensemble_pred *= 1.06
                else:
                    ensemble_pred *= 1.03
                
                if seq_max >= 1650:
                    ensemble_pred = max(ensemble_pred, 1680)
            
            # 3️⃣ 일반 상승 추세
            elif seq_trend == 'increasing':
                weights = {
                    'ExtremeNet': 0.22,
                    'SpikeDetector': 0.22,
                    'GoldenRule': 0.22,
                    'PatchTST': 0.17,
                    'StableLSTM': 0.17
                }
                
                ensemble_pred = (
                    extreme_pred * weights['ExtremeNet'] +
                    spike_pred * weights['SpikeDetector'] +
                    golden_pred * weights['GoldenRule'] +
                    patch_pred * weights['PatchTST'] +
                    stable_pred * weights['StableLSTM']
                )
                
                ensemble_pred *= 1.02
                
                if seq_max >= 1650:
                    ensemble_pred = max(ensemble_pred, seq_max * 1.03)
            
            # 4️⃣ 극단적 하락 추세 (연속 10회+ 하락)
            elif seq_trend in ['extreme_falling', 'strong_falling'] or consecutive_falls >= 10:
                weights = {
                    'StableLSTM': 0.35,
                    'PatchTST': 0.30,
                    'GoldenRule': 0.15,
                    'SpikeDetector': 0.10,
                    'ExtremeNet': 0.10
                }
                
                ensemble_pred = (
                    extreme_pred * weights['ExtremeNet'] +
                    spike_pred * weights['SpikeDetector'] +
                    golden_pred * weights['GoldenRule'] +
                    patch_pred * weights['PatchTST'] +
                    stable_pred * weights['StableLSTM']
                )
                
                # 연속 하락 정도에 따른 디스카운트
                if consecutive_falls >= 15:
                    ensemble_pred *= 0.88
                elif consecutive_falls >= 10:
                    ensemble_pred *= 0.92
                else:
                    ensemble_pred *= 0.95
                    
                # 급락 방지
                if seq_max >= 1700:
                    ensemble_pred = max(ensemble_pred, seq_max * 0.90)
            
            # 5️⃣ 빠른 하락 추세 (연속 5-9회 하락)
            elif seq_trend in ['rapid_decreasing', 'high_decreasing'] or consecutive_falls >= 5:
                weights = {
                    'StableLSTM': 0.30,
                    'PatchTST': 0.25,
                    'GoldenRule': 0.20,
                    'SpikeDetector': 0.15,
                    'ExtremeNet': 0.10
                }
                
                ensemble_pred = (
                    extreme_pred * weights['ExtremeNet'] +
                    spike_pred * weights['SpikeDetector'] +
                    golden_pred * weights['GoldenRule'] +
                    patch_pred * weights['PatchTST'] +
                    stable_pred * weights['StableLSTM']
                )
                
                if consecutive_falls >= 7:
                    ensemble_pred *= 0.94
                else:
                    ensemble_pred *= 0.97
            
            # 6️⃣ 일반 하락 추세
            elif seq_trend == 'decreasing':
                weights = {
                    'StableLSTM': 0.25,
                    'PatchTST': 0.25,
                    'GoldenRule': 0.20,
                    'SpikeDetector': 0.15,
                    'ExtremeNet': 0.15
                }
                
                ensemble_pred = (
                    extreme_pred * weights['ExtremeNet'] +
                    spike_pred * weights['SpikeDetector'] +
                    golden_pred * weights['GoldenRule'] +
                    patch_pred * weights['PatchTST'] +
                    stable_pred * weights['StableLSTM']
                )
                
                ensemble_pred *= 0.98
            
            # 7️⃣ 고평원 안정 상태
            elif seq_trend in ['high_stable']:
                weights = {
                    'ExtremeNet': 0.20,
                    'SpikeDetector': 0.20,
                    'GoldenRule': 0.20,
                    'PatchTST': 0.20,
                    'StableLSTM': 0.20
                }
                
                ensemble_pred = (
                    extreme_pred * weights['ExtremeNet'] +
                    spike_pred * weights['SpikeDetector'] +
                    golden_pred * weights['GoldenRule'] +
                    patch_pred * weights['PatchTST'] +
                    stable_pred * weights['StableLSTM']
                )
                
                if seq_max >= 1700:
                    all_preds = [extreme_pred, spike_pred, golden_pred, patch_pred, stable_pred]
                    median_pred = np.median(all_preds)
                    ensemble_pred = ensemble_pred * 0.7 + median_pred * 0.3
                    ensemble_pred = max(ensemble_pred, 1680)
            
            # 8️⃣ 일반 안정 상태
            else:
                if m14b > 350:
                    weights = {
                        'SpikeDetector': 0.25,
                        'GoldenRule': 0.25,
                        'ExtremeNet': 0.20,
                        'PatchTST': 0.15,
                        'StableLSTM': 0.15
                    }
                elif m14b > 250:
                    weights = {
                        'GoldenRule': 0.22,
                        'ExtremeNet': 0.22,
                        'SpikeDetector': 0.22,
                        'PatchTST': 0.17,
                        'StableLSTM': 0.17
                    }
                else:
                    weights = {
                        'PatchTST': 0.25,
                        'StableLSTM': 0.25,
                        'ExtremeNet': 0.20,
                        'SpikeDetector': 0.15,
                        'GoldenRule': 0.15
                    }
                
                ensemble_pred = (
                    extreme_pred * weights['ExtremeNet'] +
                    spike_pred * weights['SpikeDetector'] +
                    golden_pred * weights['GoldenRule'] +
                    patch_pred * weights['PatchTST'] +
                    stable_pred * weights['StableLSTM']
                )
            
            # M14B/M14A 비율 기반 추가 조정
            ratio = m14b / (m14a + 1)
            if ratio > 5 and ('increasing' in seq_trend or consecutive_rises >= 3):
                ensemble_pred *= 1.05
            elif ratio < 2 and ('decreasing' in seq_trend or consecutive_falls >= 3):
                ensemble_pred *= 0.95
            
            # 극단값 보호
            if seq_max >= 1800 and ensemble_pred < 1750:
                ensemble_pred = 1750
            elif seq_max >= 1700 and ensemble_pred < 1650:
                ensemble_pred = 1650
            
            extreme_ensemble.append(ensemble_pred)
        
        # 추세별 통계 출력
        print("\n📊 추세별 앙상블 적용 통계:")
        for trend, count in sorted(trend_counts.items(), key=lambda x: x[1], reverse=True):
            icon = "📈" if 'rising' in trend or 'increasing' in trend else \
                   "📉" if 'falling' in trend or 'decreasing' in trend else "➡️"
            print(f"  {icon} {trend:20s}: {count:4d}개 ({count/len(all_predictions)*100:5.1f}%)")
        
        # 연속 변화 통계
        print("\n📈 연속 변화 분포:")
        rises_10plus = (np.array(consecutive_rises_list) >= 10).sum()
        rises_5plus = (np.array(consecutive_rises_list) >= 5).sum()
        falls_10plus = (np.array(consecutive_falls_list) >= 10).sum()
        falls_5plus = (np.array(consecutive_falls_list) >= 5).sum()
        
        print(f"  연속상승 10+: {rises_10plus}개 ({rises_10plus/len(all_predictions)*100:.1f}%)")
        print(f"  연속상승 5+: {rises_5plus}개 ({rises_5plus/len(all_predictions)*100:.1f}%)")
        print(f"  연속하락 10+: {falls_10plus}개 ({falls_10plus/len(all_predictions)*100:.1f}%)")
        print(f"  연속하락 5+: {falls_5plus}개 ({falls_5plus/len(all_predictions)*100:.1f}%)")
        
        # 극단 앙상블 결과 추가
        all_predictions['추세앙상블_예측'] = [round(p) for p in extreme_ensemble]
        all_predictions['추세앙상블_오차'] = all_predictions['추세앙상블_예측'] - all_predictions['실제값']
        all_predictions['추세앙상블_오차율(%)'] = round(
            abs(all_predictions['추세앙상블_오차']) / all_predictions['실제값'] * 100, 2
        )
        
        # 추세 앙상블 성능
        extreme_mae = mean_absolute_error(all_predictions['실제값'], extreme_ensemble)
        extreme_rmse = np.sqrt(mean_squared_error(all_predictions['실제값'], extreme_ensemble))
        extreme_r2 = r2_score(all_predictions['실제값'], extreme_ensemble)
        extreme_mape = np.mean(abs(all_predictions['추세앙상블_오차']) / all_predictions['실제값']) * 100
        
        model_metrics['추세앙상블'] = {
            'MAE': extreme_mae,
            'RMSE': extreme_rmse,
            'R2': extreme_r2,
            'MAPE': extreme_mape,
            '정확도(%)': 100 - extreme_mape
        }
        
        print(f"\n✅ 추세앙상블: MAE={extreme_mae:.2f}, R²={extreme_r2:.4f}, 정확도={100-extreme_mape:.2f}%")
        
        # CSV 저장
        output_file = f'v67_trend_ensemble_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        all_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 예측값 저장: {output_file}")
        
        # ==================== 고값 구간 상세 분석 ====================
        print("\n" + "="*80)
        print("🎯🎯🎯 고값 구간 (1700+) 추세별 분석 🎯🎯🎯")
        print("="*80)
        
        high_mask = all_predictions['실제값'] >= 1700
        if high_mask.any():
            high_df = all_predictions[high_mask]
            print(f"\n📊 전체 고값 샘플: {high_mask.sum()}개")
            
            # 추세별 분포
            high_trends = {}
            for trend in high_df['시퀀스_추세'].unique():
                count = (high_df['시퀀스_추세'] == trend).sum()
                high_trends[trend] = count
            
            print(f"\n[고값 구간 추세 분포]")
            for trend, count in sorted(high_trends.items(), key=lambda x: x[1], reverse=True):
                print(f"  {trend}: {count}개 ({count/len(high_df)*100:.1f}%)")
            
            # 연속 상승과 정확도 관계
            print(f"\n[연속 상승과 예측 정확도]")
            for rises in [15, 10, 7, 5, 3]:
                rise_mask = high_df['연속상승'] >= rises
                if rise_mask.sum() > 0:
                    rise_df = high_df[rise_mask]
                    accuracy = 100 - np.mean(abs(rise_df['추세앙상블_오차']) / rise_df['실제값']) * 100
                    hit_rate = (rise_df['추세앙상블_예측'] >= 1700).sum() / len(rise_df) * 100
                    print(f"  연속상승 {rises:2d}+: {rise_mask.sum():3d}개, "
                          f"정확도={accuracy:.1f}%, 1700+적중={hit_rate:.1f}%")
            
            # 각 모델별 고값 적중률 계산
            print(f"\n[모델별 1700+ 적중률]")
            print("-" * 70)
            
            for model_name in list(self.models.keys()) + ['추세앙상블']:
                col_pred = f'{model_name}_예측'
                
                high_preds = high_df[col_pred].values
                hit_1700 = (high_preds >= 1700).sum()
                
                print(f"  {model_name:15s}: {hit_1700:3d}/{len(high_df):3d} ({hit_1700/len(high_df)*100:5.1f}%)")
            
            # 추세별 앙상블 성능
            print(f"\n[추세별 앙상블 성능]")
            print("-" * 80)
            print(f"{'추세':20s} {'샘플수':>8} {'MAE':>8} {'MAPE(%)':>8} {'정확도(%)':>10}")
            print("-" * 80)
            
            for trend in ['extreme_rising', 'strong_rising', 'increasing', 
                         'stable', 'decreasing', 'strong_falling']:
                trend_mask = high_df['시퀀스_추세'] == trend
                if trend_mask.sum() >= 5:  # 샘플 5개 이상만
                    trend_df = high_df[trend_mask]
                    mae = mean_absolute_error(trend_df['실제값'], trend_df['추세앙상블_예측'])
                    mape = np.mean(abs(trend_df['추세앙상블_오차']) / trend_df['실제값']) * 100
                    accuracy = 100 - mape
                    
                    print(f"{trend:20s} {trend_mask.sum():8d} {mae:8.2f} {mape:8.2f} {accuracy:10.2f}%")
        
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
    
    print("\n🚀 V6.7 개선 - 추세 기반 동적 앙상블!")
    print("핵심: 시퀀스 추세별 차별화된 앙상블 전략")
    print("  - 연속 상승 → 공격적 모델 강화 + 부스팅")
    print("  - 연속 하락 → 보수적 모델 강화 + 디스카운트")
    
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
    print("🏆 V6.7 추세 기반 동적 앙상블 평가 완료!")
    print("="*80)
    print("\n📁 저장된 파일:")
    print(f"  1. v67_trend_ensemble_YYYYMMDD.csv - 추세 기반 예측")
    print("\n🔥 핵심 개선사항:")
    print("  ✅ 연속 상승/하락 카운트 기반 가중치 조정")
    print("  ✅ 상승 추세: ExtremeNet/SpikeDetector 최대 65% 비중")
    print("  ✅ 하락 추세: StableLSTM/PatchTST 최대 65% 비중")
    print("  ✅ 연속 변화 정도에 따른 추가 부스팅/디스카운트")
    print("="*80)

if __name__ == "__main__":
    main()