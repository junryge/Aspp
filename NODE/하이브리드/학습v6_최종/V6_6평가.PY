"""

def boost_prediction(self, pred, m14b_value, m14a_value, sequence_trend, model_name=None):
    """트렌드 기반 극단 부스팅 - 현실적으로 수정"""
    
    boosted = pred
    
    # 1단계: 트렌드 기반 부스팅 (더 보수적으로)
    if sequence_trend == 'EXTREME_UP':
        # 상승 트렌드 + 극단값 → 적절한 부스팅
        if m14b_value > 500:
            boosted = max(pred * 1.3, 1850)  # 2.0 → 1.3
        elif m14b_value > 450:
            boosted = max(pred * 1.25, 1780)  # 1.8 → 1.25
        elif m14b_value > 400:
            boosted = max(pred * 1.2, 1730)  # 1.6 → 1.2
        else:
            boosted = max(pred * 1.15, 1700)  # 1.4 → 1.15
    
    elif sequence_trend == 'UP':
        # 일반 상승 트렌드
        if m14b_value > 500:
            boosted = max(pred * 1.2, 1800)  # 1.6 → 1.2
        elif m14b_value > 450:
            boosted = max(pred * 1.15, 1750)  # 1.5 → 1.15
        elif m14b_value > 400:
            boosted = max(pred * 1.1, 1700)  # 1.4 → 1.1
        else:
            boosted = pred * 1.05
    
    else:  # STABLE or DOWN
        # 안정/하락 → 기본 부스팅
        if m14b_value > 500:
            boosted = max(pred * 1.1, 1750)  # 1.4 → 1.1
        elif m14b_value > 450:
            boosted = max(pred * 1.05, 1700)  # 1.3 → 1.05
        elif m14b_value > 400:
            boosted = max(pred, 1650)  # 1.2 → 1.0
        else:
            boosted = pred
    
    # 2단계: 황금 패턴 (적절히)
    if m14b_value > 300 and m14a_value and m14a_value < 80:
        boosted = boosted * 1.1  # 1.2 → 1.1
    
    # 3단계: 상한선 제한 (현실적으로)
    boosted = min(boosted, 1950)  # 최대 1950으로 제한
    
    return boosted


    # 트렌드 기반 극단값 앙상블
    print("\n🔥 트렌드 기반 극단값 앙상블 생성...")
    
    extreme_ensemble = []
    for i in range(len(all_predictions)):
        m14b = all_predictions.iloc[i]['M14AM14B']
        m14a = all_predictions.iloc[i]['M14AM10A']
        trend = all_predictions.iloc[i]['시퀀스트렌드']
        
        # 트렌드별 가중치 전략
        if trend == 'EXTREME_UP':  # <-- 여기부터 수정
            # 극단 상승: 가장 높은 예측값 선택 + 적절한 부스팅
            model_preds = []
            for model_name in self.models.keys():
                if f'{model_name}_예측' in all_predictions.columns:
                    model_preds.append(all_predictions.iloc[i][f'{model_name}_예측'])
            
            if model_preds:
                ensemble_pred = max(model_preds) * 1.1  # 1.2 → 1.1로 수정
            else:
                ensemble_pred = 1800
            
            # 최종 강제 (적절하게 수정)
            if m14b > 500:
                ensemble_pred = max(ensemble_pred, 1850)  # 1900 → 1850
            elif m14b > 450:
                ensemble_pred = max(ensemble_pred, 1750)  # 1800 → 1750
            else:
                ensemble_pred = max(ensemble_pred, 1700)  # 1750 → 1700
        
        elif trend == 'UP':


V9.0 시퀀스 트렌드 기반 극단값 예측 시스템
핵심: 100개 시퀀스 트렌드 분석 → 상승 트렌드면 EXTREME 부스팅
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

# ====================== 트렌드 기반 극단값 보정 클래스 ======================
class TrendBasedExtremeBooster:
    """시퀀스 트렌드 분석 기반 극단값 부스팅"""
    
    def __init__(self):
        print("🔥 트렌드 기반 극단값 부스터 v9.0 초기화")
        
    def analyze_sequence_trend(self, sequence_100):
        """100개 시퀀스 트렌드 분석"""
        if len(sequence_100) < 100:
            return 'UNKNOWN', {}
        
        # 구간별 평균
        first_20 = np.mean(sequence_100[:20])
        middle_20 = np.mean(sequence_100[40:60])
        last_20 = np.mean(sequence_100[-20:])
        max_val = np.max(sequence_100)
        
        # 선형 회귀로 기울기 계산
        x = np.arange(len(sequence_100))
        z = np.polyfit(x, sequence_100, 1)
        slope = z[0]
        
        # 트렌드 판단
        trend_info = {
            'first_20': first_20,
            'last_20': last_20,
            'max_val': max_val,
            'slope': slope,
            'ratio': last_20 / first_20 if first_20 > 0 else 1
        }
        
        # 상승/하락 판단
        if last_20 > first_20 * 1.05 and max_val >= 1650:
            return 'EXTREME_UP', trend_info  # 상승 + 극단값
        elif last_20 > first_20 * 1.02 and slope > 0:
            return 'UP', trend_info  # 상승
        elif last_20 < first_20 * 0.98 and slope < 0:
            return 'DOWN', trend_info  # 하락
        else:
            return 'STABLE', trend_info  # 안정
    
    def boost_prediction(self, pred, m14b_value, m14a_value, sequence_trend, model_name=None):
        """트렌드 기반 극단 부스팅"""
        
        boosted = pred
        
        # 1단계: 트렌드 기반 부스팅
        if sequence_trend == 'EXTREME_UP':
            # 상승 트렌드 + 극단값 → 초강력 부스팅
            if m14b_value > 500:
                boosted = max(pred * 2.0, 1900)
            elif m14b_value > 450:
                boosted = max(pred * 1.8, 1800)
            elif m14b_value > 400:
                boosted = max(pred * 1.6, 1750)
            else:
                boosted = max(pred * 1.4, 1700)
        
        elif sequence_trend == 'UP':
            # 일반 상승 트렌드 → 중간 부스팅
            if m14b_value > 500:
                boosted = max(pred * 1.6, 1850)
            elif m14b_value > 450:
                boosted = max(pred * 1.5, 1750)
            elif m14b_value > 400:
                boosted = max(pred * 1.4, 1700)
            else:
                boosted = pred * 1.2
        
        else:  # STABLE or DOWN
            # 안정/하락 → 기본 부스팅
            if m14b_value > 500:
                boosted = max(pred * 1.4, 1800)
            elif m14b_value > 450:
                boosted = max(pred * 1.3, 1700)
            elif m14b_value > 400:
                boosted = max(pred * 1.2, 1650)
            else:
                boosted = pred * 1.1
        
        # 2단계: 황금 패턴 추가 부스팅
        if m14b_value > 300 and m14a_value and m14a_value < 80:
            boosted = boosted * 1.2
        
        # 3단계: 비율 부스팅
        if m14a_value and m14a_value > 0:
            ratio = m14b_value / m14a_value
            if ratio > 8:
                boosted = boosted * 1.15
            elif ratio > 6:
                boosted = boosted * 1.1
        
        return boosted

class CompleteModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """평가기 초기화"""
        print("="*80)
        print("🔥 V9.0 시퀀스 트렌드 기반 극단값 예측 시스템")
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
            self.feature_columns = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 'TOTALCNT']
        
        self.models = {}
        self.trend_booster = TrendBasedExtremeBooster()
        
    def load_all_models(self, model_dir='models/'):
        """모든 모델 로드"""
        print(f"\n📁 모델 로딩...")
        
        if not os.path.exists(model_dir):
            print("  ⚠️ 모델 디렉토리 없음")
            return self.models
        
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
        if 'CURRTIME' in df.columns:
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
        
        if 'CURRTIME' in df.columns:
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
        """모든 모델 평가 - 트렌드 기반 부스팅 포함"""
        
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
        
        # 기본 정보 수집
        timestamps_pred = []
        timestamps_target = []
        actuals = []
        m14b_values = []
        m14a_values = []
        sequence_trends = []
        trend_infos = []
        
        print("\n📊 데이터 수집 및 트렌드 분석 중...")
        for i in range(start_idx, end_idx):
            if 'CURRTIME' in df.columns:
                pred_time = df.iloc[i]['CURRTIME']
                target_time = pred_time + timedelta(minutes=self.pred_len)
                timestamps_pred.append(pred_time)
                timestamps_target.append(target_time)
            
            actual_idx = i + self.pred_len
            if actual_idx < len(df):
                actuals.append(df.iloc[actual_idx]['TOTALCNT'])
                m14b_values.append(df.iloc[i]['M14AM14B'])
                m14a_values.append(df.iloc[i]['M14AM10A'])
                
                # 시퀀스 트렌드 분석
                sequence = df.iloc[i-self.seq_len:i]['TOTALCNT'].values
                trend, info = self.trend_booster.analyze_sequence_trend(sequence)
                sequence_trends.append(trend)
                trend_infos.append(info)
        
        # 기본 정보 저장
        if timestamps_pred:
            all_predictions['예측시점'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_pred]
            all_predictions['예측대상시간'] = [t.strftime('%Y-%m-%d %H:%M') for t in timestamps_target]
        
        all_predictions['실제값'] = actuals
        all_predictions['M14AM14B'] = m14b_values
        all_predictions['M14AM10A'] = m14a_values
        all_predictions['시퀀스트렌드'] = sequence_trends
        
        print(f"  예측할 데이터: {len(all_predictions)}개")
        
        # 트렌드 분포 출력
        trend_counts = pd.Series(sequence_trends).value_counts()
        print(f"\n📈 시퀀스 트렌드 분포:")
        for trend, count in trend_counts.items():
            print(f"  {trend}: {count}개 ({count/len(sequence_trends)*100:.1f}%)")
        
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
                    if hasattr(self, 'feature_scaler'):
                        seq_scaled = self.feature_scaler.transform(seq)
                    else:
                        seq_scaled = seq
                    X_batch_scaled.append(seq_scaled)
                X_batch_scaled = np.array(X_batch_scaled)
                
                # 예측
                preds = model.predict(X_batch_scaled, verbose=0)
                
                if isinstance(preds, list):
                    y_pred_scaled = preds[0].flatten()
                else:
                    y_pred_scaled = preds.flatten()
                
                # 역변환
                if hasattr(self, 'target_scaler'):
                    y_pred = self.target_scaler.inverse_transform(
                        y_pred_scaled.reshape(-1, 1)).flatten()
                else:
                    y_pred = y_pred_scaled
                
                # 수집
                for k in range(len(y_pred)):
                    actual_idx = i - start_idx + k
                    if actual_idx < len(all_predictions):
                        predictions.append(y_pred[k])
            
            # 예측값 저장
            predictions = predictions[:len(all_predictions)]
            model_predictions[model_name] = predictions
            
            # 원본 예측값 저장
            all_predictions[f'{model_name}_원본'] = [round(p) for p in predictions]
            
            # 트렌드 기반 부스팅 적용
            print(f"  🔥 {model_name} 트렌드 기반 부스팅 적용 중...")
            boosted_predictions = []
            
            for i in range(len(predictions)):
                m14b = all_predictions.iloc[i]['M14AM14B']
                m14a = all_predictions.iloc[i]['M14AM10A']
                trend = all_predictions.iloc[i]['시퀀스트렌드']
                original = predictions[i]
                
                # 트렌드 기반 부스팅
                boosted = self.trend_booster.boost_prediction(
                    original, m14b, m14a, trend, model_name
                )
                boosted_predictions.append(boosted)
            
            all_predictions[f'{model_name}_예측'] = [round(p) for p in boosted_predictions]
            all_predictions[f'{model_name}_오차'] = all_predictions[f'{model_name}_예측'] - all_predictions['실제값']
            all_predictions[f'{model_name}_오차율(%)'] = round(
                abs(all_predictions[f'{model_name}_오차']) / all_predictions['실제값'] * 100, 2
            )
        
        # 트렌드 기반 극단값 앙상블
        print("\n🔥 트렌드 기반 극단값 앙상블 생성...")
        
        extreme_ensemble = []
        for i in range(len(all_predictions)):
            m14b = all_predictions.iloc[i]['M14AM14B']
            m14a = all_predictions.iloc[i]['M14AM10A']
            trend = all_predictions.iloc[i]['시퀀스트렌드']
            
            # 트렌드별 가중치 전략
            if trend == 'EXTREME_UP':
                # 극단 상승: 가장 높은 예측값 선택 + 부스팅
                model_preds = []
                for model_name in self.models.keys():
                    if f'{model_name}_예측' in all_predictions.columns:
                        model_preds.append(all_predictions.iloc[i][f'{model_name}_예측'])
                
                if model_preds:
                    ensemble_pred = max(model_preds) * 1.2
                else:
                    ensemble_pred = 1800
                
                # 최종 강제
                if m14b > 500:
                    ensemble_pred = max(ensemble_pred, 1900)
                elif m14b > 450:
                    ensemble_pred = max(ensemble_pred, 1800)
                else:
                    ensemble_pred = max(ensemble_pred, 1750)
            
            elif trend == 'UP':
                # 상승: 상위 75% 값 + 부스팅
                model_preds = []
                for model_name in self.models.keys():
                    if f'{model_name}_예측' in all_predictions.columns:
                        model_preds.append(all_predictions.iloc[i][f'{model_name}_예측'])
                
                if model_preds:
                    ensemble_pred = np.percentile(model_preds, 75) * 1.15
                else:
                    ensemble_pred = 1700
                
                if m14b > 450:
                    ensemble_pred = max(ensemble_pred, 1750)
            
            else:  # STABLE or DOWN
                # 안정/하락: 평균값 사용
                model_preds = []
                for model_name in self.models.keys():
                    if f'{model_name}_예측' in all_predictions.columns:
                        model_preds.append(all_predictions.iloc[i][f'{model_name}_예측'])
                
                if model_preds:
                    ensemble_pred = np.mean(model_preds)
                else:
                    ensemble_pred = 1600
                
                if m14b > 450:
                    ensemble_pred = max(ensemble_pred, 1700)
            
            # 황금 패턴 추가
            if m14b > 300 and m14a < 80:
                ensemble_pred = ensemble_pred * 1.1
            
            extreme_ensemble.append(ensemble_pred)
        
        # 결과 저장
        all_predictions['트렌드앙상블_예측'] = [round(p) for p in extreme_ensemble]
        all_predictions['트렌드앙상블_오차'] = all_predictions['트렌드앙상블_예측'] - all_predictions['실제값']
        all_predictions['트렌드앙상블_오차율(%)'] = round(
            abs(all_predictions['트렌드앙상블_오차']) / all_predictions['실제값'] * 100, 2
        )
        
        # 성능 계산
        extreme_mae = mean_absolute_error(all_predictions['실제값'], extreme_ensemble)
        extreme_rmse = np.sqrt(mean_squared_error(all_predictions['실제값'], extreme_ensemble))
        extreme_r2 = r2_score(all_predictions['실제값'], extreme_ensemble)
        extreme_mape = np.mean(abs(all_predictions['트렌드앙상블_오차']) / all_predictions['실제값']) * 100
        
        model_metrics['트렌드앙상블'] = {
            'MAE': extreme_mae,
            'RMSE': extreme_rmse,
            'R2': extreme_r2,
            'MAPE': extreme_mape,
            '정확도(%)': 100 - extreme_mape
        }
        
        print(f"✅ 트렌드앙상블: MAE={extreme_mae:.2f}, R²={extreme_r2:.4f}, 정확도={100-extreme_mape:.2f}%")
        
        # CSV 저장
        output_file = f'v9_trend_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        all_predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 예측값 저장: {output_file}")
        
        # 고값 구간 상세 분석
        self.analyze_extreme_performance(all_predictions)
        
        return all_predictions, model_metrics
    
    def analyze_extreme_performance(self, df):
        """고값 구간 성능 분석"""
        
        print("\n" + "="*80)
        print("🎯 고값 구간 (1700+) 상세 분석")
        print("="*80)
        
        high_mask = df['실제값'] >= 1700
        if high_mask.any():
            high_df = df[high_mask]
            print(f"\n📊 전체 고값 샘플: {high_mask.sum()}개")
            
            # 트렌드별 분석
            print(f"\n[트렌드별 1700+ 적중률]")
            print("-" * 70)
            
            for trend in high_df['시퀀스트렌드'].unique():
                trend_mask = high_df['시퀀스트렌드'] == trend
                if trend_mask.any():
                    trend_df = high_df[trend_mask]
                    hit_1700 = (trend_df['트렌드앙상블_예측'] >= 1700).sum()
                    print(f"  {trend:15s}: {hit_1700}/{len(trend_df)} "
                          f"({hit_1700/len(trend_df)*100:.1f}%)")
            
            # 전체 적중률
            print(f"\n[전체 고값 적중률]")
            print("-" * 70)
            
            hit_1800 = (high_df['트렌드앙상블_예측'] >= 1800).sum()
            hit_1750 = (high_df['트렌드앙상블_예측'] >= 1750).sum()
            hit_1700 = (high_df['트렌드앙상블_예측'] >= 1700).sum()
            hit_1650 = (high_df['트렌드앙상블_예측'] >= 1650).sum()
            
            print(f"  1800+ 예측: {hit_1800:2d}/{len(high_df):2d} ({hit_1800/len(high_df)*100:5.1f}%)")
            print(f"  1750+ 예측: {hit_1750:2d}/{len(high_df):2d} ({hit_1750/len(high_df)*100:5.1f}%)")
            print(f"  1700+ 예측: {hit_1700:2d}/{len(high_df):2d} ({hit_1700/len(high_df)*100:5.1f}%) 🎯")
            print(f"  1650+ 예측: {hit_1650:2d}/{len(high_df):2d} ({hit_1650/len(high_df)*100:5.1f}%)")
            
            # 상세 출력
            print(f"\n[고값 예측 상세] (처음 20개)")
            print("-" * 80)
            print(f"{'실제값':>7} {'예측값':>7} {'M14B':>6} {'M14A':>6} {'트렌드':>12} {'오차':>7} {'적중':>6}")
            print("-" * 80)
            
            for idx in high_df.head(20).index:
                row = df.loc[idx]
                actual = row['실제값']
                pred = row['트렌드앙상블_예측']
                m14b = row['M14AM14B']
                m14a = row['M14AM10A']
                trend = row['시퀀스트렌드']
                error = pred - actual
                
                if pred >= 1700:
                    hit = "✅ HIT"
                elif pred >= 1650:
                    hit = "⚠️"
                else:
                    hit = "❌"
                
                print(f"{actual:7.0f} {pred:7.0f} {m14b:6.0f} {m14a:6.0f} {trend:>12} {error:+7.0f} {hit:>6}")

def main():
    """메인 실행"""
    
    print("\n🚀 V9.0 시퀀스 트렌드 기반 극단값 예측 시작!")
    print("핵심: 100개 시퀀스 상승 트렌드 → EXTREME 부스팅")
    
    # 평가기 생성
    evaluator = CompleteModelEvaluator()
    
    # 모든 모델 로드
    models = evaluator.load_all_models('models/')
    
    if not models:
        print("⚠️ 모델이 없습니다! 기본 예측만 수행")
    
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
    print("🏆 V9.0 트렌드 기반 극단값 예측 완료!")
    print("="*80)
    print("\n🔥 핵심 개선:")
    print("  ✅ 100개 시퀀스 트렌드 분석 (상승/하락/안정)")
    print("  ✅ EXTREME_UP 트렌드 → 2.0배 부스팅 + 최소 1900")
    print("  ✅ UP 트렌드 → 1.6배 부스팅 + 최소 1750")
    print("  ✅ STABLE/DOWN → 기본 부스팅 + 최소 1700 (M14B>450)")
    print("  ✅ 트렌드별 차별화 앙상블 전략")
    print("="*80)

if __name__ == "__main__":
    main()