"""
📊 V6.7 개선된 극단값 예측 평가 시스템
========================================================
핵심 개선: 시퀀스 max값 1651+ & 증가 추세일 때만 ExtremeNet 부스팅
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
        consecutive_rises = sequence_info.get('consecutive_rises', 0)
        consecutive_falls = sequence_info.get('consecutive_falls', 0)
        rise_strength = sequence_info.get('rise_strength', 0)
        fall_strength = sequence_info.get('fall_strength', 0)
        
        # ========== ExtremeNet 특별 처리 ==========
        if model_name == 'ExtremeNet':
            
            # 🚀 극단적 상승 (10분+ 연속 상승)
            if seq_trend == 'extreme_rising':
                print(f"    🚀🚀 ExtremeNet 극단 상승! (연속{consecutive_rises}분↑, 강도{rise_strength:.0f})")
                # 상승 강도에 비례한 부스팅
                boost_factor = 1.5 + (rise_strength / 100) * 0.3  # 최대 1.8배
                boosted = pred * min(boost_factor, 1.8)
                boosted = max(boosted, 1900)
                
            # 💥 극단적 하락 (10분+ 연속 하락)
            elif seq_trend == 'extreme_falling':
                print(f"    💥💥 ExtremeNet 극단 하락! (연속{consecutive_falls}분↓, 강도{fall_strength:.0f})")
                # 하락 강도에 비례한 감소
                reduction_factor = 0.9 - (fall_strength / 100) * 0.1  # 최대 20% 감소
                boosted = pred * max(reduction_factor, 0.8)
                
            # 📈 강한 상승 추세
            elif seq_trend in ['strong_rising', 'increasing']:
                print(f"    📈 ExtremeNet 상승 (연속{consecutive_rises}분↑)")
                boost_factor = 1.3 + (consecutive_rises / 20) * 0.2
                boosted = pred * min(boost_factor, 1.5)
                if seq_max >= 1651:
                    boosted = max(boosted, 1750)
                    
            # 📉 강한 하락 추세
            elif seq_trend in ['strong_falling', 'rapid_decreasing']:
                print(f"    📉 ExtremeNet 강한 하락 (연속{consecutive_falls}분↓)")
                reduction_factor = 0.95 - (consecutive_falls / 20) * 0.05
                boosted = pred * max(reduction_factor, 0.85)
                
            # 🔥 고평원 상태별 처리
            elif seq_trend == 'high_increasing':
                print(f"    📈 ExtremeNet 고평원 상승! (max={seq_max:.0f})")
                if m14b_value > 500:
                    boosted = max(pred * 1.6, 1850)
                else:
                    boosted = max(pred * 1.4, 1750)
                    
            elif seq_trend == 'high_stable':
                print(f"    ➡️ ExtremeNet 고평원 유지 (max={seq_max:.0f})")
                boosted = max(pred, 1700)  # 최소 1700 보장
                
            elif seq_trend == 'high_decreasing':
                print(f"    📉 ExtremeNet 고평원 하락 (max={seq_max:.0f})")
                boosted = pred * 0.98
                
            # 일반 상승 (기존 조건)
            elif seq_max >= 1651 and seq_trend == 'increasing':
                print(f"    📈 ExtremeNet 부스팅 활성화! (max={seq_max:.0f})")
                if m14b_value > 450:
                    boosted = max(pred * 1.4, 1700)
                else:
                    boosted = pred * 1.2
                    
            # 일반 하락
            elif seq_trend == 'decreasing':
                if seq_max >= 1651:
                    boosted = pred * 0.95
                else:
                    boosted = pred
            
            # 안정적이거나 max값 미달
            else:
                if seq_max < 1651:
                    print(f"    ⚪ ExtremeNet 원본 사용 (max={seq_max:.0f} < 1651)")
                boosted = pred
                
        # ========== SpikeDetector, GoldenRule ==========
        elif model_name in ['SpikeDetector', 'GoldenRule']:
            # sequence_info에서 추세 가져오기
            seq_trend = sequence_info.get('trend', 'stable')
            
            # 고평원 상태 처리
            if seq_trend in ['high_increasing', 'high_stable', 'high_decreasing']:
                # 고평원에서는 이미 높은 예측을 유지
                if seq_trend == 'high_increasing':
                    # 고평원 상승 → 추가 부스팅
                    if m14b_value > 500:
                        boosted = max(pred * 1.1, 1900)
                    else:
                        boosted = max(pred * 1.05, 1800)
                elif seq_trend == 'high_stable':
                    # 고평원 유지 → 현재 수준
                    boosted = max(pred, 1700)
                else:  # high_decreasing
                    # 고평원 하락 → 약간 보수적
                    boosted = pred * 0.98
            else:
                # 일반 상태 (기존 로직)
                if m14b_value > 550:
                    boosted = max(pred, 1850)
                elif m14b_value > 500:
                    boosted = max(pred, 1750)
                elif m14b_value > 450:
                    boosted = max(pred, 1700)
                elif m14b_value > 400:
                    boosted = max(pred * 1.05, 1650)
            
            # 하락 추세에서는 추가 조정 없음
                
        # ========== 기타 모델 (PatchTST, StableLSTM) ==========
        else:
            # sequence_info에서 추세 가져오기
            seq_trend = sequence_info.get('trend', 'stable')
            
            # 고평원 상태 우선 처리
            if seq_trend in ['high_increasing', 'high_stable', 'high_decreasing']:
                if seq_trend == 'high_increasing':
                    # 고평원 상승 → 강한 부스팅
                    if m14b_value > 500:
                        boosted = max(pred * 1.5, 1850)
                    else:
                        boosted = max(pred * 1.3, 1750)
                elif seq_trend == 'high_stable':
                    # 고평원 유지 → 안정적 예측
                    boosted = max(pred, 1700)
                else:  # high_decreasing
                    # 고평원 하락 → 보수적
                    boosted = pred * 0.97
            # 일반 증가 추세
            elif seq_trend == 'increasing':
                # 증가 추세 → 적극적 부스팅
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
                    
            elif seq_trend == 'decreasing':
                # 하락 추세 → 보수적 접근
                if m14b_value > 450:
                    boosted = max(pred * 1.05, 1650)  # 약한 부스팅
                else:
                    boosted = pred  # 원본 유지
            else:
                # 안정적 → 기본 부스팅
                if m14b_value > 450:
                    boosted = max(pred * 1.15, 1700)
                elif m14b_value > 400:
                    boosted = max(pred * 1.10, 1650)
        
        # 황금 패턴 추가 부스팅 (모든 모델 공통)
        if m14b_value > 300 and m14a_value and m14a_value < 80:
            if seq_trend == 'increasing':
                boosted = boosted * 1.15  # 증가 추세면 더 강하게
            else:
                boosted = boosted * 1.08  # 아니면 약하게
            
        return boosted

class ImprovedModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """개선된 평가기 초기화"""
        print("="*80)
        print("🔥 V6.7 개선된 극단값 평가 시스템")
        print("  핵심: 시퀀스 max 1651+ & 증가 추세 → ExtremeNet 부스팅")
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
        """개선된 모델 평가 - 시퀀스 기반 부스팅"""
        
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
        
        # 기본 정보 저장
        all_predictions['예측시점'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_pred]
        all_predictions['예측대상시간'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_target]
        all_predictions['실제값'] = actuals
        all_predictions['M14AM14B'] = m14b_values
        all_predictions['M14AM10A'] = m14a_values
        all_predictions['시퀀스_MAX'] = sequence_maxes
        all_predictions['시퀀스_추세'] = sequence_trends
        
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
            
            # 각 모델 예측값 확인
            extreme_pred = all_predictions.iloc[i]['ExtremeNet_예측']
            spike_pred = all_predictions.iloc[i]['SpikeDetector_예측']
            golden_pred = all_predictions.iloc[i]['GoldenRule_예측']
            patch_pred = all_predictions.iloc[i]['PatchTST_예측']
            stable_pred = all_predictions.iloc[i]['StableLSTM_예측']
            
            # 🔥 핵심: 시퀀스 MAX가 1650+ 이고 increasing이면 최소값 보장
            if seq_max >= 1650 and seq_trend == 'increasing':
                # 가장 높은 예측값을 우선 사용
                max_pred = max(extreme_pred, spike_pred, golden_pred, patch_pred, stable_pred)
                
                # 최소 1700 보장
                ensemble_pred = max(max_pred, seq_max * 1.03, 1700)
                
            # 시퀀스 MAX가 1600+ 이면 좀 더 보수적
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
                
                # 최소값 보장
                ensemble_pred = max(ensemble_pred, seq_max, 1650)
                
            # 일반 경우
            else:
                # 동적 가중치 (기존 로직)
                if m14b > 200:  # 기준을 낮춤
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
                for model_name in self.models.keys():
                    if model_name in weights:
                        ensemble_pred += all_predictions.iloc[i][f'{model_name}_예측'] * weights[model_name]
            
            extreme_ensemble.append(ensemble_pred)
        
        # 극단 앙상블 결과 추가
        all_predictions['개선앙상블_예측'] = [round(p) for p in extreme_ensemble]
        all_predictions['개선앙상블_오차'] = all_predictions['개선앙상블_예측'] - all_predictions['실제값']
        all_predictions['개선앙상블_오차율(%)'] = round(
            abs(all_predictions['개선앙상블_오차']) / all_predictions['실제값'] * 100, 2
        )
        
        # 개선 앙상블 성능
        extreme_mae = mean_absolute_error(all_predictions['실제값'], extreme_ensemble)
        extreme_rmse = np.sqrt(mean_squared_error(all_predictions['실제값'], extreme_ensemble))
        extreme_r2 = r2_score(all_predictions['실제값'], extreme_ensemble)
        extreme_mape = np.mean(abs(all_predictions['개선앙상블_오차']) / all_predictions['실제값']) * 100
        
        model_metrics['개선앙상블'] = {
            'MAE': extreme_mae,
            'RMSE': extreme_rmse,
            'R2': extreme_r2,
            'MAPE': extreme_mape,
            '정확도(%)': 100 - extreme_mape
        }
        
        print(f"✅ 개선앙상블: MAE={extreme_mae:.2f}, R²={extreme_r2:.4f}, 정확도={100-extreme_mape:.2f}%")
        
        # CSV 저장
        output_file = f'v67_improved_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        all_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 예측값 저장: {output_file}")
        
        # ==================== 고값 구간 상세 분석 ====================
        print("\n" + "="*80)
        print("🎯🎯🎯 고값 구간 (1700+) 상세 분석 🎯🎯🎯")
        print("="*80)
        
        high_mask = all_predictions['실제값'] >= 1700
        if high_mask.any():
            high_df = all_predictions[high_mask]
            print(f"\n📊 전체 고값 샘플: {high_mask.sum()}개")
            
            # 추세별 분포
            high_inc = (high_df['시퀀스_추세'] == 'increasing').sum()
            high_dec = (high_df['시퀀스_추세'] == 'decreasing').sum()
            high_stable = (high_df['시퀀스_추세'] == 'stable').sum()
            
            print(f"\n[고값 구간 추세 분포]")
            print(f"  증가 추세: {high_inc}개 ({high_inc/len(high_df)*100:.1f}%)")
            print(f"  하락 추세: {high_dec}개 ({high_dec/len(high_df)*100:.1f}%)")
            print(f"  안정 추세: {high_stable}개 ({high_stable/len(high_df)*100:.1f}%)")
            
            # ExtremeNet 원본 vs 부스팅 비교
            print(f"\n[ExtremeNet 부스팅 효과 분석]")
            extreme_orig = high_df['ExtremeNet_원본'].values
            extreme_boost = high_df['ExtremeNet_예측'].values
            
            orig_hit = (extreme_orig >= 1700).sum()
            boost_hit = (extreme_boost >= 1700).sum()
            
            print(f"  원본 1700+ 적중: {orig_hit}/{len(high_df)} ({orig_hit/len(high_df)*100:.1f}%)")
            print(f"  부스팅 1700+ 적중: {boost_hit}/{len(high_df)} ({boost_hit/len(high_df)*100:.1f}%)")
            print(f"  개선: +{boost_hit - orig_hit}개")
            
            # 시퀀스 조건별 분석
            seq_condition_mask = ((high_df['시퀀스_MAX'] >= 1651) & 
                                 (high_df['시퀀스_추세'] == 'increasing'))
            if seq_condition_mask.any():
                condition_df = high_df[seq_condition_mask]
                print(f"\n  조건 충족 샘플 (max≥1651 & 증가): {seq_condition_mask.sum()}개")
                cond_hit = (condition_df['ExtremeNet_예측'] >= 1700).sum()
                print(f"    → ExtremeNet 적중: {cond_hit}/{len(condition_df)} "
                      f"({cond_hit/len(condition_df)*100:.1f}%)")
            
            # 각 모델별 고값 적중률 계산
            print(f"\n[모델별 1700+ 적중률]")
            print("-" * 70)
            
            for model_name in list(self.models.keys()) + ['개선앙상블']:
                col_pred = f'{model_name}_예측'
                
                high_preds = high_df[col_pred].values
                hit_1700 = (high_preds >= 1700).sum()
                hit_1650 = (high_preds >= 1650).sum()
                
                print(f"  {model_name:15s}: "
                      f"{hit_1700:2d}/{len(high_df):2d} ({hit_1700/len(high_df)*100:5.1f}%)")
            
            # 개선앙상블 상세 결과 (추세 포함)
            print(f"\n[개선앙상블 고값 예측 상세] (처음 20개)")
            print("-" * 90)
            print(f"{'실제값':>7} {'예측값':>7} {'M14B':>6} {'시퀀스MAX':>10} {'추세':>10} {'오차':>7} {'적중':>6}")
            print("-" * 90)
            
            for idx in high_df.head(20).index:
                row = all_predictions.loc[idx]
                actual = row['실제값']
                pred = row['개선앙상블_예측']
                m14b = row['M14AM14B']
                seq_max = row['시퀀스_MAX']
                seq_trend = row['시퀀스_추세']
                error = pred - actual
                
                # 적중 여부
                if pred >= 1700:
                    hit = "✅ HIT"
                elif pred >= 1650:
                    hit = "⚠️ NEAR"
                else:
                    hit = "❌ MISS"
                
                trend_icon = "📈" if seq_trend == 'increasing' else "📉" if seq_trend == 'decreasing' else "➡️"
                
                print(f"{actual:7.0f} {pred:7.0f} {m14b:6.0f} {seq_max:10.0f} "
                      f"{trend_icon:>2}{seq_trend:>8} {error:+7.0f} {hit:>6}")
        
        # 성능 요약 테이블
        print("\n" + "="*80)
        print("📊 전체 모델 성능 요약")
        print("="*80)
        
        metrics_df = pd.DataFrame(model_metrics).T
        metrics_df = metrics_df.sort_values('R2', ascending=False)
        
        print(f"\n{'모델':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE(%)':>8} {'정확도(%)':>10}")
        print("-" * 65)
        
        for model_name, row in metrics_df.iterrows():
            if model_name == '개선앙상블':
                print(f"{'🔥 ' + model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                      f"{row['R2']:8.4f} {row['MAPE']:8.2f} {row['정확도(%)']:10.2f} ⭐⭐⭐")
            else:
                print(f"{model_name:<15} {row['MAE']:8.2f} {row['RMSE']:8.2f} "
                      f"{row['R2']:8.4f} {row['MAPE']:8.2f} {row['정확도(%)']:10.2f}")
        
        return all_predictions, metrics_df

def main():
    """메인 실행"""
    
    print("\n🚀 V6.7 개선된 극단값 평가 시작!")
    print("핵심: 시퀀스 max 1651+ & 증가 추세 → ExtremeNet 부스팅")
    print("      하락 추세 → 보수적 예측")
    
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
    print("🏆 V6.7 개선된 극단값 평가 완료!")
    print("="*80)
    print("\n📁 저장된 파일:")
    print(f"  1. v67_improved_predictions_YYYYMMDD.csv - 개선된 예측")
    print("\n🔥 핵심 개선사항:")
    print("  ✅ 시퀀스 max값 1651 이상 체크")
    print("  ✅ 증가 추세일 때만 ExtremeNet 부스팅")
    print("  ✅ 하락 추세에서는 보수적 예측")
    print("  ✅ 추세별 차별화된 앙상블 가중치")
    print("="*80)

if __name__ == "__main__":
    main()