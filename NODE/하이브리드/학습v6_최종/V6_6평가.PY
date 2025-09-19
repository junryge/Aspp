"""
📊 적응형 딥러닝 앙상블 평가 시스템 V2.2 (Fixed)
feature_columns 자동 매칭
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import json
import os
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
tf.keras.config.enable_unsafe_deserialization()

print("=" * 80)
print("🚀 적응형 평가 시스템 V2.2")
print("  - feature_columns 자동 매칭")
print("  - 시퀀스 100개로 데이터 타입 판별")
print("  - 10분 후 예측")
print("=" * 80)
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ====================== 시퀀스 기반 데이터 타입 감지기 ======================
class SequenceBasedDetector:
    """100개 시퀀스로 데이터 타입 판별"""
    
    def __init__(self):
        self.seq_len = 100
        self.pred_len = 10
        
    def analyze_sequence(self, sequence_data):
        """100개 시퀀스 분석"""
        
        if len(sequence_data) < 100:
            print(f"⚠️ 시퀀스 길이 부족: {len(sequence_data)}개")
            return None
            
        seq = sequence_data[-100:]
        
        analysis = {
            'mean': np.mean(seq['TOTALCNT']),
            'std': np.std(seq['TOTALCNT']),
            'max': np.max(seq['TOTALCNT']),
            'min': np.min(seq['TOTALCNT']),
            'noise_level': np.std(seq['TOTALCNT']) / np.mean(seq['TOTALCNT']) if np.mean(seq['TOTALCNT']) > 0 else 0,
        }
        
        if 'M14AM14B' in seq.columns:
            analysis['corr_m14b'] = seq['TOTALCNT'].corr(seq['M14AM14B'])
        
        if 'M14AM10A' in seq.columns:
            analysis['corr_m14a'] = seq['TOTALCNT'].corr(seq['M14AM10A'])
        
        analysis['volatility'] = seq['TOTALCNT'].diff().std()
        
        x = np.arange(len(seq))
        y = seq['TOTALCNT'].values
        coeffs = np.polyfit(x, y, 1)
        analysis['trend_slope'] = coeffs[0]
        
        if 'M14AM14B' in seq.columns and 'M14AM10A' in seq.columns:
            pattern1 = ((seq['M14AM14B'] > 300) & (seq['M14AM10A'] < 70)).sum()
            pattern2 = ((seq['M14AM14B'] > 350) & (seq['M14AM10A'] < 65)).sum()
            
            analysis['spike_pattern_uu1'] = pattern1
            analysis['spike_pattern_uu2'] = pattern2
        
        return analysis
    
    def detect_data_type(self, sequence_df):
        """데이터 타입 판별"""
        
        analysis = self.analyze_sequence(sequence_df)
        if not analysis:
            return 'unknown', {}
        
        print("\n🔍 시퀀스 분석 결과:")
        print(f"  평균: {analysis['mean']:.0f}")
        print(f"  표준편차: {analysis['std']:.0f}")
        print(f"  노이즈: {analysis['noise_level']:.3f}")
        print(f"  M14B 상관: {analysis.get('corr_m14b', 0):.3f}")
        
        score_uu1 = 0
        score_uu2 = 0
        
        if analysis.get('corr_m14b', 0) > 0.8:
            score_uu1 += 3
        elif analysis.get('corr_m14b', 0) < 0.6:
            score_uu2 += 3
        
        if analysis['noise_level'] < 0.15:
            score_uu1 += 2
        elif analysis['noise_level'] > 0.18:
            score_uu2 += 2
        
        if analysis['volatility'] < 50:
            score_uu1 += 1
        elif analysis['volatility'] > 70:
            score_uu2 += 1
        
        print(f"\n📊 판별 점수: uu1={score_uu1}, uu2={score_uu2}")
        
        if score_uu1 > score_uu2:
            data_type = 'uu1'
            print("  ✅ 판별: uu1 타입 (고상관, 저노이즈)")
        elif score_uu2 > score_uu1:
            data_type = 'uu2'
            print("  ✅ 판별: uu2 타입 (저상관, 고노이즈)")
        else:
            data_type = 'balanced'
            print("  ✅ 판별: 균형 타입")
        
        return data_type, analysis

# ====================== 적응형 극단값 부스터 ======================
class AdaptiveExtremeBooster:
    """데이터 타입별 적응형 부스팅"""
    
    def __init__(self, data_type='unknown'):
        self.data_type = data_type
        print(f"\n🔥 부스터 초기화 ({data_type} 타입)")
        
        if data_type == 'uu1':
            self.spike_threshold_b = 300
            self.spike_threshold_a = 70
            self.spike_ratio = 1.3
            self.extreme_threshold = 1651
        elif data_type == 'uu2':
            self.spike_threshold_b = 350
            self.spike_threshold_a = 65
            self.spike_ratio = 1.4
            self.extreme_threshold = 1700
        else:
            self.spike_threshold_b = 325
            self.spike_threshold_a = 67
            self.spike_ratio = 1.35
            self.extreme_threshold = 1675
    
    def analyze_sequence_for_boost(self, sequence_data):
        """부스팅용 시퀀스 분석"""
        
        seq_max = np.max(sequence_data)
        seq_min = np.min(sequence_data)
        seq_mean = np.mean(sequence_data[-30:]) if len(sequence_data) >= 30 else np.mean(sequence_data)
        
        is_extreme = seq_max >= self.extreme_threshold
        
        if len(sequence_data) >= 10:
            recent_10 = sequence_data[-10:]
            trend_change = recent_10[-1] - recent_10[0]
            
            if trend_change > 50:
                trend = 'strong_rising'
            elif trend_change > 20:
                trend = 'rising'
            elif trend_change < -50:
                trend = 'strong_falling'
            elif trend_change < -20:
                trend = 'falling'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        consecutive_rises = 0
        for i in range(len(sequence_data)-1, 0, -1):
            if sequence_data[i] > sequence_data[i-1]:
                consecutive_rises += 1
            else:
                break
        
        return {
            'max': seq_max,
            'min': seq_min,
            'mean': seq_mean,
            'trend': trend,
            'is_extreme': is_extreme,
            'consecutive_rises': consecutive_rises
        }
    
    def boost_prediction(self, pred, m14b, m14a, model_name, seq_info):
        """데이터 타입별 부스팅"""
        
        boosted = pred
        
        if model_name == 'ExtremeNet':
            if self.data_type == 'uu1':
                if seq_info['max'] >= 1651 and seq_info['trend'] in ['rising', 'strong_rising']:
                    if m14b > 300:
                        boosted = max(pred * 1.25, 1700)
                    else:
                        boosted = max(pred * 1.20, 1680)
                elif seq_info['is_extreme']:
                    boosted = max(pred * 1.15, seq_info['max'])
            
            elif self.data_type == 'uu2':
                if seq_info['max'] >= 1700 and seq_info['trend'] in ['rising', 'strong_rising']:
                    if m14b > 350:
                        boosted = max(pred * 1.30, 1750)
                    else:
                        boosted = max(pred * 1.25, 1720)
                elif seq_info['is_extreme']:
                    boosted = max(pred * 1.20, seq_info['max'])
            
            else:
                if seq_info['is_extreme'] and seq_info['trend'] in ['rising', 'strong_rising']:
                    boosted = max(pred * 1.22, seq_info['max'] * 1.05)
        
        elif model_name == 'SpikeDetector':
            if m14b > self.spike_threshold_b and m14a < self.spike_threshold_a:
                boosted = max(pred * self.spike_ratio, seq_info['max'] * 1.05)
            elif seq_info['is_extreme']:
                boosted = max(pred * 1.10, seq_info['max'])
        
        elif model_name == 'GoldenRule':
            if self.data_type == 'uu1' and m14b > 300 and m14a < 70:
                boosted = max(pred * 1.30, seq_info['max'] * 1.05)
            elif self.data_type == 'uu2' and m14b > 350 and m14a < 65:
                boosted = max(pred * 1.40, seq_info['max'] * 1.08)
            elif seq_info['is_extreme']:
                boosted = max(pred * 1.08, seq_info['max'])
        
        return boosted

# ====================== 통합 평가 시스템 ======================
class UnifiedEvaluator:
    """단일 CSV 파일 적응형 평가"""
    
    def __init__(self, scaler_path='scalers/', model_path='models/'):
        self.scaler_path = scaler_path
        self.model_path = model_path
        self.seq_len = 100
        self.pred_len = 10
        self.models = {}
        self.detector = SequenceBasedDetector()
        self.booster = None
        self.data_type = None
        self.feature_columns = None
        self.scaler_loaded = False
        
        print("\n📁 평가 시스템 초기화...")
        self.load_scalers()
        self.load_models()
    
    def load_scalers(self):
        """스케일러 로드"""
        try:
            with open(f'{self.scaler_path}feature_scaler.pkl', 'rb') as f:
                self.feature_scaler = pickle.load(f)
            with open(f'{self.scaler_path}target_scaler.pkl', 'rb') as f:
                self.target_scaler = pickle.load(f)
            
            config_file = f'{self.scaler_path}config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.feature_columns = config.get('feature_columns')
                    self.seq_len = config.get('seq_len', 100)
                    self.pred_len = config.get('pred_len', 10)
                    print(f"  ✅ 스케일러 로드 완료")
                    print(f"  ✅ Feature columns: {len(self.feature_columns)}개")
            else:
                print("  ⚠️ config.json 없음")
                self.scaler_loaded = False
                return
            
            self.scaler_loaded = True
            
        except Exception as e:
            print(f"  ⚠️ 스케일러 로드 실패: {str(e)[:50]}")
            self.scaler_loaded = False
    
    def load_models(self):
        """실제 존재하는 모델만 로드"""
        print("\n📂 모델 로딩...")
        
        if not os.path.exists(self.model_path):
            print(f"  ❌ 모델 디렉토리 없음: {self.model_path}")
            return
        
        model_files = [f for f in os.listdir(self.model_path) if f.endswith('.keras')]
        
        if not model_files:
            print("  ❌ 모델 파일이 없습니다!")
            return
        
        loaded_count = 0
        for model_file in model_files:
            model_name = model_file.replace('.keras', '')
            full_path = os.path.join(self.model_path, model_file)
            
            try:
                self.models[model_name] = tf.keras.models.load_model(
                    full_path, safe_mode=False
                )
                print(f"  ✅ {model_name} 로드")
                loaded_count += 1
            except Exception as e:
                print(f"  ❌ {model_name} 실패: {str(e)[:50]}")
        
        print(f"\n총 {loaded_count}개 모델 로드 완료")
    
    def create_features(self, df):
        """필수 특징 생성"""
        # 시간 특징
        if 'CURRTIME' in df.columns:
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                           format='%Y%m%d%H%M', errors='coerce')
            df['HOUR'] = df['CURRTIME'].dt.hour
            df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
            df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        else:
            df['HOUR'] = 12
            df['HOUR_SIN'] = np.sin(2 * np.pi * 12 / 24)
            df['HOUR_COS'] = np.cos(2 * np.pi * 12 / 24)
        
        # 비율 특징
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        
        # 이동평균
        for w in [10, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
        
        # 변화량
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        
        return df
    
    def load_and_prepare_data(self, csv_path):
        """데이터 로드 및 전처리"""
        print(f"\n📊 데이터 로드: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"  원본: {len(df)}행")
        
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        if 'CURRTIME' in df.columns:
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                           format='%Y%m%d%H%M', errors='coerce')
            df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 필수 특징 생성
        df = self.create_features(df)
        
        print(f"  유효: {len(df)}행")
        print(f"  컬럼: {len(df.columns)}개")
        
        # 필요한 컬럼 확인
        if self.feature_columns:
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_cols:
                print(f"  ⚠️ 누락 컬럼: {missing_cols}")
            
            available_cols = [col for col in self.feature_columns if col in df.columns]
            print(f"  ✅ 사용 가능 컬럼: {len(available_cols)}개")
        
        return df
    
    def evaluate(self, csv_path):
        """통합 평가 실행"""
        
        # 1. 데이터 로드
        df = self.load_and_prepare_data(csv_path)
        
        if len(df) < self.seq_len + self.pred_len:
            print(f"❌ 데이터 부족: 최소 {self.seq_len + self.pred_len}행 필요")
            return None, None
        
        if not self.models:
            print("❌ 로드된 모델이 없습니다!")
            return None, None
        
        if not self.scaler_loaded:
            print("❌ 스케일러가 로드되지 않았습니다!")
            return None, None
        
        # 2. 데이터 타입 판별
        first_seq = df.iloc[:self.seq_len]
        self.data_type, seq_analysis = self.detector.detect_data_type(first_seq)
        
        # 3. 부스터 설정
        self.booster = AdaptiveExtremeBooster(self.data_type)
        
        # 4. 예측 준비
        print(f"\n🔮 예측 시작...")
        print(f"  시퀀스: {self.seq_len}개 → 예측: {self.pred_len}분 후")
        
        start_idx = self.seq_len
        end_idx = len(df) - self.pred_len
        total_predictions = min(end_idx - start_idx, 3000)  # 최대 3000개
        
        print(f"  예측 개수: {total_predictions}개")
        
        # 5. 데이터 수집
        results = pd.DataFrame()
        timestamps = []
        actuals = []
        m14b_values = []
        m14a_values = []
        seq_maxes = []
        seq_trends = []
        
        print("\n📈 데이터 수집 중...")
        for i in range(start_idx, start_idx + total_predictions):
            if i + self.pred_len >= len(df):
                break
                
            seq_data = df.iloc[i-self.seq_len:i]
            target_idx = i + self.pred_len
            
            if 'CURRTIME' in df.columns:
                timestamps.append(df.iloc[i]['CURRTIME'])
            else:
                timestamps.append(i)
                
            actuals.append(df.iloc[target_idx]['TOTALCNT'])
            m14b_values.append(df.iloc[i]['M14AM14B'])
            m14a_values.append(df.iloc[i]['M14AM10A'])
            
            seq_info = self.booster.analyze_sequence_for_boost(seq_data['TOTALCNT'].values)
            seq_maxes.append(seq_info['max'])
            seq_trends.append(seq_info['trend'])
        
        results['시점'] = timestamps
        results['실제값'] = actuals
        results['M14AM14B'] = m14b_values
        results['M14AM10A'] = m14a_values
        results['시퀀스_MAX'] = seq_maxes
        results['시퀀스_추세'] = seq_trends
        
        # 6. 모델별 예측
        print("\n🎯 모델별 예측...")
        model_predictions = {}
        
        for model_name, model in self.models.items():
            print(f"  {model_name} 예측 중...")
            predictions = []
            
            try:
                # 배치 예측
                batch_size = 32
                for batch_start in range(0, len(actuals), batch_size):
                    batch_end = min(batch_start + batch_size, len(actuals))
                    batch_X = []
                    
                    for j in range(batch_start, batch_end):
                        idx = start_idx + j
                        
                        # 시퀀스 데이터 준비
                        seq_data = df.iloc[idx-self.seq_len:idx][self.feature_columns].values
                        
                        # 스케일링
                        seq_scaled = self.feature_scaler.transform(seq_data)
                        batch_X.append(seq_scaled)
                    
                    if len(batch_X) == 0:
                        continue
                    
                    batch_X = np.array(batch_X)
                    
                    # 예측
                    batch_pred = model.predict(batch_X, verbose=0)
                    
                    # 역변환
                    if isinstance(batch_pred, list):
                        batch_pred = batch_pred[0]
                    
                    batch_pred = batch_pred.reshape(-1, 1)
                    batch_pred_original = self.target_scaler.inverse_transform(batch_pred)
                    
                    predictions.extend(batch_pred_original.flatten())
                    
                    if len(predictions) % 500 == 0:
                        print(f"    {len(predictions)}/{len(actuals)} 완료")
                
                # 부스팅 적용
                boosted_preds = []
                for j in range(len(predictions)):
                    seq_info = {
                        'max': seq_maxes[j],
                        'trend': seq_trends[j],
                        'is_extreme': seq_maxes[j] >= self.booster.extreme_threshold
                    }
                    
                    boosted = self.booster.boost_prediction(
                        predictions[j],
                        m14b_values[j],
                        m14a_values[j],
                        model_name,
                        seq_info
                    )
                    boosted_preds.append(boosted)
                
                # 결과 저장
                results[f'{model_name}_원본'] = predictions
                results[f'{model_name}_예측'] = boosted_preds
                results[f'{model_name}_오차'] = np.array(boosted_preds) - np.array(actuals)
                
                # ExtremeNet 범위값
                if model_name == 'ExtremeNet':
                    ranges = []
                    percents = []
                    min_vals = []
                    max_vals = []
                    
                    for j in range(len(predictions)):
                        원본 = predictions[j]
                        seq_max = seq_maxes[j]
                        
                        if 원본 > 0:
                            차이 = seq_max - 원본
                            퍼센트 = (차이 / 원본) * 100
                            최소값 = 원본
                            최대값 = 원본 * (1 + 퍼센트/100)
                            
                            if 최대값 < 최소값:
                                최대값 = 최소값 * 1.1
                        else:
                            퍼센트 = 0
                            최소값 = 0
                            최대값 = 0
                        
                        percents.append(round(퍼센트, 2))
                        min_vals.append(round(최소값))
                        max_vals.append(round(최대값))
                        ranges.append(f"{int(최소값)}~{int(최대값)}")
                    
                    results['ExtremeNet_퍼센트'] = percents
                    results['ExtremeNet_최소값'] = min_vals
                    results['ExtremeNet_최대값'] = max_vals
                    results['ExtremeNet_범위값'] = ranges
                
                model_predictions[model_name] = boosted_preds
                print(f"    ✅ {model_name} 완료")
                
            except Exception as e:
                print(f"    ❌ {model_name} 예측 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 7. 앙상블 생성
        print("\n🔥 앙상블 생성...")
        
        available_models = list(model_predictions.keys())
        
        if len(available_models) > 0:
            # 균등 가중치
            equal_weight = 1.0 / len(available_models)
            
            ensemble_preds = []
            for i in range(len(actuals)):
                ensemble = 0
                for model_name in available_models:
                    ensemble += model_predictions[model_name][i] * equal_weight
                ensemble_preds.append(ensemble)
            
            results['앙상블_예측'] = ensemble_preds
            results['앙상블_오차'] = np.array(ensemble_preds) - np.array(actuals)
            
            print(f"  ✅ {len(available_models)}개 모델로 앙상블 생성")
        else:
            print("  ❌ 앙상블 생성 실패: 예측된 모델 없음")
        
        # 8. 성능 계산
        print("\n📊 성능 평가...")
        metrics = {}
        
        for col in results.columns:
            if '_예측' in col:
                model_name = col.replace('_예측', '')
                mae = mean_absolute_error(actuals, results[col])
                rmse = np.sqrt(mean_squared_error(actuals, results[col]))
                mape = np.mean(np.abs((np.array(actuals) - results[col]) / np.array(actuals))) * 100
                
                metrics[model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    '정확도': 100 - mape
                }
        
        # 9. 모델별 CSV 저장
        print("\n💾 결과 저장 중...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        
        for model_name in model_predictions.keys():
            model_df = pd.DataFrame()
            model_df['시점'] = results['시점']
            model_df['실제값'] = results['실제값']
            model_df['M14AM14B'] = results['M14AM14B']
            model_df['M14AM10A'] = results['M14AM10A']
            model_df['시퀀스_MAX'] = results['시퀀스_MAX']
            model_df['시퀀스_추세'] = results['시퀀스_추세']
            model_df[f'{model_name}_원본예측'] = results[f'{model_name}_원본']
            model_df[f'{model_name}_부스팅예측'] = results[f'{model_name}_예측']
            model_df[f'{model_name}_오차'] = results[f'{model_name}_오차']
            model_df[f'{model_name}_오차율(%)'] = abs(results[f'{model_name}_오차']) / results['실제값'] * 100
            
            if model_name == 'ExtremeNet' and 'ExtremeNet_범위값' in results.columns:
                model_df['ExtremeNet_범위값'] = results['ExtremeNet_범위값']
                model_df['ExtremeNet_퍼센트'] = results['ExtremeNet_퍼센트']
                model_df['ExtremeNet_최소값'] = results['ExtremeNet_최소값']
                model_df['ExtremeNet_최대값'] = results['ExtremeNet_최대값']
            
            if model_name in metrics:
                model_df.loc[0, 'MAE'] = metrics[model_name]['MAE']
                model_df.loc[0, 'RMSE'] = metrics[model_name]['RMSE']
                model_df.loc[0, 'MAPE(%)'] = metrics[model_name]['MAPE']
                model_df.loc[0, '정확도(%)'] = metrics[model_name]['정확도']
            
            model_file = f'{model_name}_{self.data_type}_{timestamp}.csv'
            model_df.to_csv(model_file, index=False, encoding='utf-8-sig')
            saved_files.append(model_file)
            print(f"  ✅ {model_file}")
        
        # 앙상블 저장
        if '앙상블_예측' in results.columns:
            ensemble_df = pd.DataFrame()
            ensemble_df['시점'] = results['시점']
            ensemble_df['실제값'] = results['실제값']
            ensemble_df['앙상블_예측'] = results['앙상블_예측']
            ensemble_df['앙상블_오차'] = results['앙상블_오차']
            ensemble_df['앙상블_오차율(%)'] = abs(results['앙상블_오차']) / results['실제값'] * 100
            
            if '앙상블' in metrics:
                ensemble_df.loc[0, 'MAE'] = metrics['앙상블']['MAE']
                ensemble_df.loc[0, 'RMSE'] = metrics['앙상블']['RMSE']
                ensemble_df.loc[0, 'MAPE(%)'] = metrics['앙상블']['MAPE']
                ensemble_df.loc[0, '정확도(%)'] = metrics['앙상블']['정확도']
            
            ensemble_file = f'Ensemble_{self.data_type}_{timestamp}.csv'
            ensemble_df.to_csv(ensemble_file, index=False, encoding='utf-8-sig')
            saved_files.append(ensemble_file)
            print(f"  ✅ {ensemble_file}")
        
        # 10. 결과 출력
        print("\n" + "="*80)
        print("📊 평가 결과")
        print("="*80)
        print(f"데이터 타입: {self.data_type}")
        print(f"평가 데이터: {len(results)}개")
        print(f"저장 파일: {len(saved_files)}개")
        
        if metrics:
            print(f"\n{'모델':<15} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>10} {'정확도(%)':>10}")
            print("-"*60)
            
            sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]['정확도'], reverse=True)
            for model_name, m in sorted_metrics:
                print(f"{model_name:<15} {m['MAE']:10.2f} {m['RMSE']:10.2f} "
                      f"{m['MAPE']:10.2f} {m['정확도']:10.2f}")
        
        return results, metrics

# ====================== 메인 실행 ======================
def main():
    """메인 실행"""
    
    print("\n" + "="*80)
    print("🚀 적응형 평가 시스템 시작!")
    print("="*80)
    
    evaluator = UnifiedEvaluator()
    
    test_files = [
        'data/M14_20250916_20250918.csv',
        'test_data.csv',
        'uu.csv',
        'uu2.csv'
    ]
    
    test_file = None
    for file in test_files:
        if os.path.exists(file):
            test_file = file
            print(f"✅ 평가 파일: {file}")
            break
    
    if not test_file:
        print("❌ 평가 파일이 없습니다!")
        return
    
    results, metrics = evaluator.evaluate(test_file)
    
    if results is not None:
        print("\n✅ 평가 완료!")
    else:
        print("\n❌ 평가 실패!")

if __name__ == "__main__":
    main()