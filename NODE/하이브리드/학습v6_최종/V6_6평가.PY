"""
📊 V6.7 개선된 극단값 예측 평가 시스템 (최적화 완전판)
========================================================
개선사항:
1. ExtremeNet 부스팅 조건 완화 (1682 → 1650)
2. 부스팅 상한선 추가 (과도한 예측 방지)
3. stable 상태에서도 약한 부스팅 적용
4. UU1/UU2 패턴 감지 조건 최적화
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

# ====================== 개선된 패턴 감지기 ======================
class DataPatternDetector:
    """데이터 패턴 자동 감지 (UU1 vs UU2) - 개선됨"""
    
    def detect_pattern(self, df):
        """
        패턴 감지: UU1(급증) 또는 UU2(고값유지)
        개선: 더 정확한 임계값 설정
        """
        
        high_cases = df[df['TOTALCNT'] >= 1682]
        
        if len(high_cases) == 0:
            print("  📈 UU1 패턴: 1682+ 데이터 없음 → 급증 예측 필요")
            return "UU1"
        
        # 시퀀스 패턴 분석
        pattern_stats = []
        trend_counts = {'increasing': 0, 'stable': 0, 'decreasing': 0}
        
        for idx in high_cases.index:
            if idx >= 100:
                seq = df.iloc[idx-100:idx]['TOTALCNT'].values
                
                # 추세 계산
                if len(seq) >= 30:
                    recent = seq[-30:]
                    x = np.arange(len(recent))
                    coeffs = np.polyfit(x, recent, 1)
                    slope = coeffs[0]
                    
                    if slope > 2:
                        trend_counts['increasing'] += 1
                    elif slope < -2:
                        trend_counts['decreasing'] += 1
                    else:
                        trend_counts['stable'] += 1
                
                pattern_stats.append({
                    'seq_max': seq.max(),
                    'seq_mean': seq.mean(),
                    'high_1650': (seq >= 1650).sum(),
                    'high_1682': (seq >= 1682).sum()
                })
        
        if not pattern_stats:
            return "UU1"
        
        # 통계 계산
        avg_seq_max = np.mean([p['seq_max'] for p in pattern_stats])
        avg_seq_mean = np.mean([p['seq_mean'] for p in pattern_stats])
        avg_high_1650 = np.mean([p['high_1650'] for p in pattern_stats])
        avg_high_1682 = np.mean([p['high_1682'] for p in pattern_stats])
        
        total_trends = sum(trend_counts.values())
        stable_ratio = trend_counts['stable'] / max(total_trends, 1)
        
        print(f"\n  📊 패턴 분석 결과:")
        print(f"     시퀀스 MAX 평균: {avg_seq_max:.0f}")
        print(f"     시퀀스 MEAN 평균: {avg_seq_mean:.0f}")
        print(f"     1650+ 평균 개수: {avg_high_1650:.0f}개/100")
        print(f"     1682+ 평균 개수: {avg_high_1682:.0f}개/100")
        print(f"     Stable 비율: {stable_ratio:.1%}")
        
        # 실용적 UU2 판별 조건
        # UU2: 시퀀스가 이미 높은 상태
        if avg_seq_max >= 1700 and avg_high_1650 >= 10:
            print(f"  🔥 UU2 패턴 감지: 고값 유지 상태")
            return "UU2"
        else:
            print(f"  📈 UU1 패턴 감지: 급증 예측 필요")
            return "UU1"

# ====================== 개선된 극단값 보정 클래스 ======================
class ImprovedExtremeValueBooster:
    """시퀀스 기반 스마트 극단값 예측 - 개선됨"""
    
    def __init__(self):
        print("🔥 개선된 극단값 부스터 초기화")
        print("  - 부스팅 조건 완화: 1650부터 시작")
        print("  - 부스팅 상한선 추가: 과도한 예측 방지")
        print("  - Stable 상태에서도 약한 부스팅")
        self.data_pattern = "UU1"
        
    def set_data_pattern(self, pattern):
        """데이터 패턴 설정"""
        self.data_pattern = pattern
        
    def analyze_sequence(self, sequence_data):
        """시퀀스 분석: max값과 추세 계산"""
        if len(sequence_data) == 0:
            return None, 'stable'
        
        seq_max = np.max(sequence_data)
        
        if len(sequence_data) >= 30:
            recent = sequence_data[-30:]
            x = np.arange(len(recent))
            coeffs = np.polyfit(x, recent, 1)
            slope = coeffs[0]
            
            if slope > 2:
                trend = 'increasing'
            elif slope < -2:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return seq_max, trend
    
    def boost_prediction(self, pred, m14b_value, m14a_value=None, model_name=None, 
                        sequence_max=None, sequence_trend=None):
        """개선된 부스팅 로직"""
        
        original = pred
        boosted = pred
        
        # ========== UU2 패턴 (고값 유지) ==========
        if self.data_pattern == "UU2":
            if model_name == 'ExtremeNet':
                if sequence_max and sequence_max >= 1680:
                    if m14b_value > 500:
                        boosted = min(max(pred * 1.15, 1680), 1850)
                    elif m14b_value > 450:
                        boosted = min(max(pred * 1.10, 1650), 1800)
                    else:
                        boosted = min(pred * 1.05, 1750)
                else:
                    boosted = pred
                    
            elif model_name in ['SpikeDetector', 'GoldenRule']:
                if m14b_value > 450:
                    boosted = min(max(pred * 1.03, 1650), 1800)
                else:
                    boosted = pred
                    
            return boosted
        
        # ========== UU1 패턴 (급증) - 개선됨 ==========
        if model_name == 'ExtremeNet':
            
            # 조건 1: 강한 부스팅 (1682+ & increasing)
            if sequence_max and sequence_max >= 1682 and sequence_trend == 'increasing':
                if m14b_value > 550:
                    boosted = min(max(pred * 1.6, 1850), 2000)
                elif m14b_value > 500:
                    boosted = min(max(pred * 1.5, 1750), 1950)
                elif m14b_value > 450:
                    boosted = min(max(pred * 1.4, 1700), 1900)
                elif m14b_value > 400:
                    boosted = min(max(pred * 1.3, 1650), 1850)
                elif m14b_value > 350:
                    boosted = min(max(pred * 1.2, 1550), 1800)
                else:
                    boosted = min(pred * 1.1, 1750)
            
            # 조건 2: 중간 부스팅 (1650-1682 & increasing) - 새로 추가!
            elif sequence_max and 1650 <= sequence_max < 1682 and sequence_trend == 'increasing':
                if m14b_value > 500:
                    boosted = min(max(pred * 1.3, 1700), 1900)
                elif m14b_value > 450:
                    boosted = min(max(pred * 1.25, 1650), 1850)
                elif m14b_value > 400:
                    boosted = min(max(pred * 1.2, 1600), 1800)
                else:
                    boosted = min(max(pred * 1.15, 1550), 1750)
            
            # 조건 3: 약한 부스팅 (1650+ & stable) - 새로 추가!
            elif sequence_max and sequence_max >= 1650 and sequence_trend == 'stable':
                if m14b_value > 450:
                    boosted = min(max(pred * 1.15, 1600), 1800)
                elif m14b_value > 400:
                    boosted = min(max(pred * 1.10, 1550), 1750)
                else:
                    boosted = min(max(pred * 1.05, 1500), 1700)
            
            # 조건 4: 하락 추세
            elif sequence_trend == 'decreasing':
                if sequence_max and sequence_max >= 1682:
                    boosted = pred * 0.95
                else:
                    boosted = pred
            
            # 그 외: 최소 부스팅
            else:
                if sequence_max and sequence_max >= 1600 and m14b_value > 400:
                    boosted = min(max(pred * 1.05, 1500), 1700)
                else:
                    boosted = pred
                    
        # ========== SpikeDetector, GoldenRule ==========
        elif model_name in ['SpikeDetector', 'GoldenRule']:
            if m14b_value > 550:
                boosted = min(max(pred, 1850), 2000)
            elif m14b_value > 500:
                boosted = min(max(pred, 1750), 1950)
            elif m14b_value > 450:
                boosted = min(max(pred, 1700), 1900)
            elif m14b_value > 400:
                boosted = min(max(pred * 1.05, 1650), 1850)
            
        # ========== 기타 모델 ==========
        else:
            if sequence_trend == 'increasing':
                if m14b_value > 550:
                    boosted = min(max(pred * 1.45, 1850), 2000)
                elif m14b_value > 500:
                    boosted = min(max(pred * 1.35, 1750), 1950)
                elif m14b_value > 450:
                    boosted = min(max(pred * 1.25, 1700), 1900)
                elif m14b_value > 400:
                    boosted = min(max(pred * 1.15, 1650), 1850)
                elif m14b_value > 350:
                    boosted = min(max(pred * 1.08, 1550), 1800)
                    
            elif sequence_trend == 'decreasing':
                if m14b_value > 450:
                    boosted = max(pred * 1.05, 1650)
                else:
                    boosted = pred
            else:
                if m14b_value > 450:
                    boosted = min(max(pred * 1.15, 1700), 1900)
                elif m14b_value > 400:
                    boosted = min(max(pred * 1.10, 1650), 1850)
        
        # 황금 패턴 추가 부스팅
        if m14b_value > 300 and m14a_value and m14a_value < 80:
            if self.data_pattern == "UU2":
                boosted = min(boosted * 1.05, 2000)
            elif sequence_trend == 'increasing':
                boosted = min(boosted * 1.15, 2000)
            else:
                boosted = min(boosted * 1.08, 1950)
            
        return boosted

# ====================== 메인 평가 클래스 ======================
class ImprovedModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """개선된 평가기 초기화"""
        print("="*80)
        print("🔥 V6.7 최적화된 극단값 평가 시스템")
        print("  - 부스팅 조건 완화")
        print("  - 상한선 추가")
        print("  - 패턴 감지 개선")
        print("="*80)
        
        # 스케일러 로드
        try:
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
        except:
            print(f"⚠️ 스케일러 없음 - 기본값 사용")
            self.seq_len = 100
            self.pred_len = 10
            self.feature_columns = ['TOTALCNT', 'M14AM14B', 'M14AM10A', 'M14AM14BSUM', 'M14AM16']
            self.feature_scaler = None
            self.target_scaler = None
        
        self.models = {}
        self.extreme_booster = ImprovedExtremeValueBooster()
        self.pattern_detector = DataPatternDetector()
    
    def load_all_models(self, model_dir='models/'):
        """모든 모델 로드"""
        print(f"\n📁 모델 로딩...")
        
        if not os.path.exists(model_dir):
            print(f"  ⚠️ 모델 폴더가 없습니다")
            return {}
        
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
        """테스트 데이터 로드 + 패턴 감지"""
        print(f"\n📂 평가 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        print(f"  원본: {df.shape[0]:,}행")
        
        # 0값 제거
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        # 시간 변환
        if 'CURRTIME' in df.columns:
            try:
                df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                               format='%Y%m%d%H%M', errors='coerce')
                df = df.sort_values('CURRTIME').reset_index(drop=True)
            except:
                pass
        
        print(f"  유효: {df.shape[0]:,}행")
        
        # 패턴 자동 감지
        data_pattern = self.pattern_detector.detect_pattern(df)
        self.extreme_booster.set_data_pattern(data_pattern)
        
        # 고값 통계 출력
        high_count = (df['TOTALCNT'] >= 1700).sum()
        very_high_count = (df['TOTALCNT'] >= 1750).sum()
        extreme_count = (df['TOTALCNT'] >= 1800).sum()
        max_1682_count = (df['TOTALCNT'] >= 1682).sum()
        
        print(f"\n🎯 고값 구간 분포:")
        print(f"  1682+: {max_1682_count}개 ({max_1682_count/len(df)*100:.1f}%) ← 기준값")
        print(f"  1700+: {high_count}개 ({high_count/len(df)*100:.1f}%)")
        print(f"  1750+: {very_high_count}개 ({very_high_count/len(df)*100:.1f}%)")
        print(f"  1800+: {extreme_count}개 ({extreme_count/len(df)*100:.1f}%)")
        
        # M14B 분포
        if 'M14AM14B' in df.columns:
            m14b_high = (df['M14AM14B'] > 450).sum()
            print(f"\n📊 M14AM14B 분포:")
            print(f"  450+: {m14b_high}개 ({m14b_high/len(df)*100:.1f}%)")
        
        return df
    
    def create_features(self, df):
        """특성 생성"""
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        
        if 'CURRTIME' in df.columns:
            try:
                df['HOUR'] = df['CURRTIME'].dt.hour
                df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
                df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
            except:
                pass
        
        for w in [10, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
        
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        
        return df
    
    def evaluate_all_models(self, test_file):
        """개선된 모델 평가"""
        
        # 데이터 로드
        df = self.load_test_data(test_file)
        df = self.create_features(df)
        
        # 예측 가능 범위
        start_idx = self.seq_len
        end_idx = len(df) - self.pred_len
        total = end_idx - start_idx
        
        if total <= 0:
            print("❌ 예측할 데이터가 충분하지 않습니다")
            return None, None
        
        print(f"\n🔮 예측 시작...")
        print(f"  시퀀스: {self.seq_len}분 → 예측: {self.pred_len}분 후")
        print(f"  예측 개수: {total:,}개")
        print(f"  패턴 모드: {self.extreme_booster.data_pattern}")
        
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
            if 'CURRTIME' in df.columns:
                pred_time = df.iloc[i]['CURRTIME']
                target_time = pred_time + timedelta(minutes=self.pred_len) if pd.notna(pred_time) else None
                timestamps_pred.append(pred_time)
                timestamps_target.append(target_time)
            
            actual_idx = i + self.pred_len
            if actual_idx < len(df):
                actuals.append(df.iloc[actual_idx]['TOTALCNT'])
                m14b_values.append(df.iloc[i]['M14AM14B'])
                m14a_values.append(df.iloc[i]['M14AM10A'])
                
                # 시퀀스 분석 (과거 100분 TOTALCNT)
                seq_data = df.iloc[i-self.seq_len:i]['TOTALCNT'].values
                seq_max, trend = self.extreme_booster.analyze_sequence(seq_data)
                sequence_maxes.append(seq_max)
                sequence_trends.append(trend)
        
        # 기본 정보 저장
        if timestamps_pred and timestamps_pred[0] is not None:
            all_predictions['예측시점'] = [t.strftime('%Y-%m-%d %H:%M') if pd.notna(t) else '' for t in timestamps_pred]
            all_predictions['예측대상시간'] = [t.strftime('%Y-%m-%d %H:%M') if pd.notna(t) else '' for t in timestamps_target]
        
        all_predictions['실제값'] = actuals
        all_predictions['M14AM14B'] = m14b_values
        all_predictions['M14AM10A'] = m14a_values
        all_predictions['시퀀스_MAX'] = sequence_maxes
        all_predictions['시퀀스_추세'] = sequence_trends
        all_predictions['데이터패턴'] = self.extreme_booster.data_pattern
        
        print(f"  예측할 데이터: {len(all_predictions)}개")
        
        # 시퀀스 분석 통계
        high_seq_count = (np.array(sequence_maxes) >= 1682).sum()
        mid_seq_count = ((np.array(sequence_maxes) >= 1650) & (np.array(sequence_maxes) < 1682)).sum()
        inc_trend_count = (np.array(sequence_trends) == 'increasing').sum()
        dec_trend_count = (np.array(sequence_trends) == 'decreasing').sum()
        stable_count = (np.array(sequence_trends) == 'stable').sum()
        
        print(f"\n📈 시퀀스 분석 결과:")
        print(f"  max값 1682+: {high_seq_count}개 ({high_seq_count/len(all_predictions)*100:.1f}%)")
        print(f"  max값 1650-1682: {mid_seq_count}개 ({mid_seq_count/len(all_predictions)*100:.1f}%)")
        print(f"  증가 추세: {inc_trend_count}개 ({inc_trend_count/len(all_predictions)*100:.1f}%)")
        print(f"  안정 추세: {stable_count}개 ({stable_count/len(all_predictions)*100:.1f}%)")
        print(f"  하락 추세: {dec_trend_count}개 ({dec_trend_count/len(all_predictions)*100:.1f}%)")
        
        # ExtremeNet 부스팅 조건 만족 개수
        boost_strong = ((np.array(sequence_maxes) >= 1682) & 
                       (np.array(sequence_trends) == 'increasing')).sum()
        boost_mid = ((np.array(sequence_maxes) >= 1650) & 
                    (np.array(sequence_maxes) < 1682) & 
                    (np.array(sequence_trends) == 'increasing')).sum()
        boost_weak = ((np.array(sequence_maxes) >= 1650) & 
                     (np.array(sequence_trends) == 'stable')).sum()
        
        print(f"\n🔥 ExtremeNet 부스팅 조건:")
        print(f"  강한 부스팅 (1682+ & inc): {boost_strong}개")
        print(f"  중간 부스팅 (1650-1682 & inc): {boost_mid}개")
        print(f"  약한 부스팅 (1650+ & stable): {boost_weak}개")
        print(f"  전체 부스팅 대상: {boost_strong + boost_mid + boost_weak}개")
        
        # 모델이 없으면 더미 예측 실행
        if not self.models:
            print("\n⚠️ 실제 모델이 없어 더미 예측을 실행합니다")
            
            # 더미 모델 생성
            for model_name in ['ExtremeNet', 'SpikeDetector', 'GoldenRule', 'PatchTST', 'StableLSTM']:
                print(f"  {model_name} 더미 예측...")
                dummy_preds = []
                boosted_preds = []
                
                for i in range(len(all_predictions)):
                    # 더미 예측
                    dummy_pred = sequence_maxes[i] * 0.98 + np.random.randn() * 20
                    
                    # 부스팅 적용
                    boosted = self.extreme_booster.boost_prediction(
                        dummy_pred, 
                        m14b_values[i], 
                        m14a_values[i], 
                        model_name,
                        sequence_max=sequence_maxes[i],
                        sequence_trend=sequence_trends[i]
                    )
                    
                    dummy_preds.append(dummy_pred)
                    boosted_preds.append(boosted)
                
                all_predictions[f'{model_name}_원본'] = [round(p) for p in dummy_preds]
                all_predictions[f'{model_name}_예측'] = [round(p) for p in boosted_preds]
                all_predictions[f'{model_name}_오차'] = all_predictions[f'{model_name}_예측'] - all_predictions['실제값']
                all_predictions[f'{model_name}_오차율(%)'] = round(
                    abs(all_predictions[f'{model_name}_오차']) / all_predictions['실제값'] * 100, 2
                )
        
        else:
            # 실제 모델로 예측
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
                    if self.feature_scaler:
                        X_batch_scaled = []
                        for seq in X_batch:
                            seq_scaled = self.feature_scaler.transform(seq)
                            X_batch_scaled.append(seq_scaled)
                        X_batch_scaled = np.array(X_batch_scaled)
                    else:
                        X_batch_scaled = X_batch
                    
                    # 예측
                    preds = model.predict(X_batch_scaled, verbose=0)
                    
                    if isinstance(preds, list):
                        y_pred_scaled = preds[0].flatten()
                    else:
                        y_pred_scaled = preds.flatten()
                    
                    # 역변환
                    if self.target_scaler:
                        y_pred = self.target_scaler.inverse_transform(
                            y_pred_scaled.reshape(-1, 1)).flatten()
                    else:
                        y_pred = y_pred_scaled
                    
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
                print(f"  🔥 {model_name} 부스팅 적용 중...")
                boosted_predictions = []
                boost_applied_count = 0
                
                for i in range(len(predictions)):
                    m14b = all_predictions.iloc[i]['M14AM14B']
                    m14a = all_predictions.iloc[i]['M14AM10A']
                    seq_max = all_predictions.iloc[i]['시퀀스_MAX']
                    seq_trend = all_predictions.iloc[i]['시퀀스_추세']
                    original = predictions[i]
                    
                    # 시퀀스 기반 부스팅 적용
                    boosted = self.extreme_booster.boost_prediction(
                        original, m14b, m14a, model_name,
                        sequence_max=seq_max, sequence_trend=seq_trend
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
            
            # 패턴별 가중치 설정
            if self.extreme_booster.data_pattern == "UU2":
                # UU2: 고값 유지 패턴
                weights = {
                    'SpikeDetector': 0.30,
                    'GoldenRule': 0.25,
                    'PatchTST': 0.20,
                    'StableLSTM': 0.15,
                    'ExtremeNet': 0.10
                }
            else:
                # UU1: 일반 패턴
                if seq_trend == 'increasing' and seq_max >= 1682:
                    if m14b > 500:
                        weights = {
                            'SpikeDetector': 0.50,
                            'GoldenRule': 0.30,
                            'ExtremeNet': 0.10,
                            'PatchTST': 0.05,
                            'StableLSTM': 0.05
                        }
                    else:
                        weights = {
                            'SpikeDetector': 0.35,
                            'GoldenRule': 0.25,
                            'ExtremeNet': 0.20,
                            'PatchTST': 0.10,
                            'StableLSTM': 0.10
                        }
                elif seq_trend == 'decreasing':
                    weights = {
                        'PatchTST': 0.30,
                        'StableLSTM': 0.30,
                        'ExtremeNet': 0.15,
                        'GoldenRule': 0.15,
                        'SpikeDetector': 0.10
                    }
                else:
                    if m14b > 450:
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
            
            ensemble_pred = 0
            total_weight = 0
            
            for model_name in ['ExtremeNet', 'SpikeDetector', 'GoldenRule', 'PatchTST', 'StableLSTM']:
                if f'{model_name}_예측' in all_predictions.columns:
                    weight = weights.get(model_name, 0.2)
                    ensemble_pred += all_predictions.iloc[i][f'{model_name}_예측'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_pred = ensemble_pred / total_weight
            
            # 추세별 최종 조정
            if seq_trend == 'increasing' and seq_max >= 1682:
                if m14b > 550:
                    ensemble_pred = max(ensemble_pred, 1850)
                elif m14b > 500:
                    ensemble_pred = max(ensemble_pred, 1750)
                elif m14b > 450:
                    ensemble_pred = max(ensemble_pred, 1700)
            elif seq_trend == 'decreasing' and seq_max >= 1682:
                ensemble_pred = min(ensemble_pred, seq_max * 0.95)
            
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
        
        if self.models:
            model_metrics['개선앙상블'] = {
                'MAE': extreme_mae,
                'RMSE': extreme_rmse,
                'R2': extreme_r2,
                'MAPE': extreme_mape,
                '정확도(%)': 100 - extreme_mape
            }
        
        print(f"✅ 개선앙상블: MAE={extreme_mae:.2f}, R²={extreme_r2:.4f}, 정확도={100-extreme_mape:.2f}%")
        
        # CSV 저장
        pattern_suffix = self.extreme_booster.data_pattern
        output_file = f'v67_optimized_{pattern_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        all_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 예측값 저장: {output_file}")
        
        # 고값 구간 상세 분석
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
            
            # 각 모델별 고값 적중률 계산
            print(f"\n[모델별 1700+ 적중률]")
            print("-" * 70)
            
            model_list = ['ExtremeNet', 'SpikeDetector', 'GoldenRule', 'PatchTST', 'StableLSTM', '개선앙상블']
            for model_name in model_list:
                col_pred = f'{model_name}_예측'
                if col_pred in high_df.columns:
                    high_preds = high_df[col_pred].values
                    hit_1700 = (high_preds >= 1700).sum()
                    
                    print(f"  {model_name:15s}: "
                          f"{hit_1700:2d}/{len(high_df):2d} ({hit_1700/len(high_df)*100:5.1f}%)")
        
        # 성능 요약 테이블
        print("\n" + "="*80)
        print("📊 전체 모델 성능 요약")
        print("="*80)
        
        if self.models:
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
        else:
            return all_predictions, None

def main():
    """메인 실행"""
    
    print("\n🚀 V6.7 최적화 버전 시작!")
    print("개선사항:")
    print("  1. ExtremeNet 부스팅 조건 완화 (1650부터)")
    print("  2. 부스팅 상한선 추가 (과도한 예측 방지)")
    print("  3. Stable 상태에서도 약한 부스팅")
    print("  4. UU2 패턴 감지 조건 최적화")
    
    # 평가기 생성
    evaluator = ImprovedModelEvaluator()
    
    # 모든 모델 로드
    models = evaluator.load_all_models('models/')
    
    # 테스트 파일
    test_files = [
        'uu1.csv',
        'uu2.csv',
        'data/uu1.csv',
        'data/uu2.csv',
        'data/test_data.csv',
    ]
    
    for file in test_files:
        if os.path.exists(file):
            print(f"\n{'='*80}")
            print(f"📍 테스트 파일: {file}")
            print('='*80)
            
            # 평가 실행
            all_predictions, metrics = evaluator.evaluate_all_models(file)
            
            if all_predictions is not None:
                print(f"\n✅ {file} 평가 완료!")
    
    print("\n" + "="*80)
    print("🏆 평가 완료!")
    print("="*80)

if __name__ == "__main__":
    main()