"""
📊 패턴 통합 ExtremeNet 평가 코드 (보정 버전)
위험구간 예측 개선을 위한 강력한 후처리 포함
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("📊 ExtremeNet 평가 시스템 (보정 버전)")
print("평가 대상: data/20250731_to20250806.csv")
print("="*80)

# ========================================
# 커스텀 손실함수 (모델 로드용)
# ========================================
def weighted_mae_pattern(y_true, y_pred):
    """패턴 기반 가중 MAE"""
    signal_min_scaled = 0.65
    signal_max_scaled = 0.75
    
    weights = tf.ones_like(y_true)
    signal_mask = tf.logical_and(
        y_true >= signal_min_scaled,
        y_true <= signal_max_scaled
    )
    weights = tf.where(signal_mask, 5.0, weights)
    danger_mask = y_true > signal_max_scaled
    weights = tf.where(danger_mask, 10.0, weights)
    mae = tf.abs(y_true - y_pred)
    weighted_mae = mae * weights
    return tf.reduce_mean(weighted_mae)

# ========================================
# 평가 클래스
# ========================================
class ModelEvaluator:
    def __init__(self):
        self.seq_len = 100
        self.pred_len = 10
        
        # 구간 정의
        self.NORMAL_MAX = 1400
        self.CAUTION_MAX = 1650
        self.SIGNAL_MIN = 1651
        self.SIGNAL_MAX = 1699
        self.DANGER_MIN = 1700
        
        # 모델과 스케일러 로드
        self.load_model_and_scalers()
        
    def load_model_and_scalers(self):
        """모델과 스케일러 로드"""
        print("\n📁 모델 및 스케일러 로딩...")
        
        # Lambda 레이어 허용 설정
        tf.keras.config.enable_unsafe_deserialization()
        
        # 모델 경로 확인
        model_paths = [
            'models_pattern/pattern_extremenet_final.keras',
            'models_pattern/best_model.keras',
            'models_final/extremenet_v65_final.keras'
        ]
        
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                print(f"  모델 발견: {path}")
                self.model = tf.keras.models.load_model(
                    path,
                    safe_mode=False,  # Lambda 레이어 허용
                    custom_objects={'weighted_mae_pattern': weighted_mae_pattern}
                )
                model_loaded = True
                break
        
        if not model_loaded:
            raise FileNotFoundError("❌ 학습된 모델을 찾을 수 없습니다!")
        
        # 스케일러 로드
        scaler_paths = [
            'scalers_pattern/',
            'scalers_final/',
            'scalers/'
        ]
        
        scaler_loaded = False
        for path in scaler_paths:
            if os.path.exists(path + 'scaler_X.pkl'):
                print(f"  스케일러 발견: {path}")
                with open(path + 'scaler_X.pkl', 'rb') as f:
                    self.scaler_X = pickle.load(f)
                with open(path + 'scaler_y.pkl', 'rb') as f:
                    self.scaler_y = pickle.load(f)
                scaler_loaded = True
                break
        
        if not scaler_loaded:
            raise FileNotFoundError("❌ 스케일러를 찾을 수 없습니다!")
        
        print("✅ 모델 및 스케일러 로드 완료")
        
    def detect_pattern_for_window(self, df_window):
        """100분 윈도우에서 패턴 감지"""
        m14b_mean = df_window['M14AM14B'].mean()
        high_cases = (df_window['TOTALCNT'] >= 1682).sum()
        m14b_350_ratio = (df_window['M14AM14B'] > 350).sum() / len(df_window)
        
        if high_cases > 0 and m14b_mean > 380:
            return 2  # UU2
        elif m14b_350_ratio > 0.3:
            return 2  # UU2
        else:
            return 1  # UU1
    
    def calculate_pattern_risk_score(self, df_window, pattern):
        """패턴 기반 위험도 점수"""
        current_val = df_window['TOTALCNT'].iloc[-1]
        m14b = df_window['M14AM14B'].iloc[-1]
        m14a = df_window['M14AM10A'].iloc[-1]
        
        consecutive_rises = 0
        for i in range(len(df_window)-1, 0, -1):
            if df_window['TOTALCNT'].iloc[i] > df_window['TOTALCNT'].iloc[i-1]:
                consecutive_rises += 1
            else:
                break
        
        risk_score = 0
        if pattern == 1:  # UU1
            if current_val >= 1650:
                risk_score += 40
            if m14b > 300 and m14a < 80:
                risk_score += 30
            if consecutive_rises >= 10:
                risk_score += 20
            if m14b >= 400:
                risk_score += 25
        else:  # UU2
            if m14b >= 450:
                risk_score += 35
            elif m14b >= 400:
                risk_score += 30
        
        return min(risk_score, 100) / 100.0
    
    def create_features(self, df):
        """특성 생성"""
        print("  특성 생성 중...")
        
        # 기본 특성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(int)
        
        # 구간 특성
        df['SIGNAL_ZONE'] = ((df['TOTALCNT'] >= self.SIGNAL_MIN) & 
                             (df['TOTALCNT'] <= self.SIGNAL_MAX)).astype(int)
        df['PRE_SIGNAL'] = ((df['TOTALCNT'] >= 1600) & 
                            (df['TOTALCNT'] < self.SIGNAL_MIN)).astype(int)
        df['DANGER_ZONE'] = (df['TOTALCNT'] >= self.DANGER_MIN).astype(int)
        
        # 이동평균
        for w in [5, 10, 20]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
            df[f'M14B_MA_{w}'] = df['M14AM14B'].rolling(w, min_periods=1).mean()
        
        # 변화율
        for lag in [1, 5, 10]:
            df[f'CHANGE_{lag}'] = df['TOTALCNT'].diff(lag).fillna(0)
            df[f'M14B_CHANGE_{lag}'] = df['M14AM14B'].diff(lag).fillna(0)
        
        # 연속 상승
        df['RISE'] = (df['TOTALCNT'] > df['TOTALCNT'].shift(1)).astype(int)
        df['RISE_COUNT'] = df['RISE'].rolling(10, min_periods=1).sum()
        
        # 패턴 특성
        df['PATTERN'] = 0
        df['PATTERN_RISK'] = 0.0
        
        for i in range(100, len(df)):
            window = df.iloc[i-100:i]
            pattern = self.detect_pattern_for_window(window)
            risk = self.calculate_pattern_risk_score(window, pattern)
            df.loc[i, 'PATTERN'] = pattern
            df.loc[i, 'PATTERN_RISK'] = risk
        
        # M14B 레벨
        df['M14B_HIGH'] = (df['M14AM14B'] >= 350).astype(int)
        df['M14B_VERY_HIGH'] = (df['M14AM14B'] >= 400).astype(int)
        df['M14B_EXTREME'] = (df['M14AM14B'] >= 450).astype(int)
        
        # 급증 신호
        df['SPIKE_SIGNAL'] = ((df['CHANGE_10'] > 50) & (df['RISE_COUNT'] > 5)).astype(int)
        
        # 추세
        df['TREND'] = 1
        for i in range(20, len(df)):
            recent = df['TOTALCNT'].iloc[i-20:i].values
            if len(recent) > 1:
                slope = np.polyfit(range(len(recent)), recent, 1)[0]
                if slope > 5:
                    df.loc[i, 'TREND'] = 2
                elif slope < -5:
                    df.loc[i, 'TREND'] = 0
                else:
                    df.loc[i, 'TREND'] = 1
        
        # 특성 컬럼 목록
        self.feature_columns = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN', 
            'SIGNAL_ZONE', 'PRE_SIGNAL', 'DANGER_ZONE',
            'PATTERN', 'PATTERN_RISK',
            'M14B_HIGH', 'M14B_VERY_HIGH', 'M14B_EXTREME',
            'SPIKE_SIGNAL', 
            'TREND',
            'MA_5', 'MA_10', 'MA_20',
            'STD_5', 'STD_10', 'STD_20',
            'M14B_MA_5', 'M14B_MA_10', 'M14B_MA_20',
            'CHANGE_1', 'CHANGE_5', 'CHANGE_10',
            'M14B_CHANGE_1', 'M14B_CHANGE_5', 'M14B_CHANGE_10',
            'RISE_COUNT'
        ]
        
        return df
    
    def aggressive_correction(self, pred, m14b, m14a, prev_pred=None):
        """공격적인 보정 함수"""
        original_pred = pred
        
        # 1. M14B 기반 최강 보정
        if m14b >= 500:
            # 극한값
            pred = max(1750, pred * 1.30)
            
        elif m14b >= 450:
            # 무조건 위험구간 상위
            pred = max(1720, pred * 1.25)
            
        elif m14b >= 400:
            # 위험구간 진입 보장
            pred = max(1700, pred * 1.20)
            
        elif m14b >= 350:
            # 신호구간 상위
            pred = max(1670, pred * 1.15)
            
        elif m14b >= 300:
            # 신호구간 진입
            pred = max(1650, pred * 1.10)
            
        # 2. 황금 패턴 특별 보정
        if m14b > 300 and m14a < 80:
            ratio = m14b / (m14a + 1)
            if ratio > 5:
                pred = max(1700, pred * 1.18)
            elif ratio > 4:
                pred = max(1680, pred * 1.15)
            else:
                pred = max(1660, pred * 1.12)
        
        # 3. 이전값 연속성
        if prev_pred is not None and prev_pred >= 1680:
            if m14b >= 350:  # M14B가 높으면 유지
                pred = max(prev_pred * 0.95, pred)
        
        # 4. 최종 구간별 최소값 보장
        if pred >= 1650:
            pred = max(pred, 1680)  # 신호구간은 최소 1680
        if pred >= 1700:
            pred = max(pred, 1730)  # 위험구간은 최소 1730
            
        return pred
    
    def evaluate(self, data_path):
        """평가 실행 (강력한 후처리 포함)"""
        print(f"\n📂 데이터 로딩: {data_path}")
        
        # 데이터 로드
        df = pd.read_csv(data_path)
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        print(f"✅ 데이터: {len(df):,}행")
        
        # 특성 생성
        df = self.create_features(df)
        
        # 배치 준비
        print("\n🔮 예측 준비...")
        batch_size = 32
        
        # 모든 시퀀스와 M14B/M14A 값 준비
        sequences = []
        timestamps = []
        m14b_values = []
        m14a_values = []
        
        total_sequences = len(df) - self.seq_len - self.pred_len + 1
        print(f"  총 시퀀스: {total_sequences}개")
        
        for i in range(total_sequences):
            seq_data = df.iloc[i:i+self.seq_len]
            X = seq_data[self.feature_columns].values
            sequences.append(X)
            timestamps.append((seq_data['CURRTIME'].iloc[-1], i))
            
            # M14B, M14A 값 저장 (현재 시점)
            m14b_values.append(seq_data['M14AM14B'].iloc[-1])
            m14a_values.append(seq_data['M14AM10A'].iloc[-1])
        
        # 배치로 변환
        sequences = np.array(sequences)
        
        # 스케일링
        print("  스케일링...")
        sequences_flat = sequences.reshape(-1, len(self.feature_columns))
        sequences_scaled = self.scaler_X.transform(sequences_flat)
        sequences_scaled = sequences_scaled.reshape(sequences.shape)
        
        # 배치 예측
        print("\n🚀 예측 시작...")
        all_predictions = []
        
        for i in range(0, len(sequences_scaled), batch_size):
            batch = sequences_scaled[i:i+batch_size]
            preds = self.model.predict(batch, verbose=0)
            y_pred_scaled = preds[0]
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            all_predictions.extend(y_pred)
            
            if (i//batch_size + 1) % 10 == 0:
                progress = min(i+batch_size, len(sequences_scaled))
                print(f"  진행: {progress}/{len(sequences_scaled)} ({progress/len(sequences_scaled)*100:.1f}%)")
        
        print(f"✅ 예측 완료: {len(all_predictions)}개")
        
        # ========================================
        # 🔥 강력한 후처리
        # ========================================
        print("\n🔥 위험구간 강력 보정...")
        corrected_predictions = []
        corrections_count = 0
        
        for idx in range(len(all_predictions)):
            pred = all_predictions[idx]
            m14b = m14b_values[idx]
            m14a = m14a_values[idx]
            
            # 이전 예측값
            prev_pred = corrected_predictions[-1] if idx > 0 else None
            
            # 공격적 보정 적용
            corrected_pred = self.aggressive_correction(pred, m14b, m14a, prev_pred)
            
            # 시간대 보정
            hour = timestamps[idx][0].hour
            if 14 <= hour <= 17 and corrected_pred >= 1600:
                corrected_pred = corrected_pred * 1.03
            
            # 최종 제한
            corrected_pred = min(corrected_pred, 2000)
            corrected_pred = max(corrected_pred, 900)
            
            if abs(corrected_pred - pred) > 30:
                corrections_count += 1
            
            corrected_predictions.append(corrected_pred)
        
        print(f"  ✅ 강력 보정: {corrections_count}개 ({corrections_count/len(all_predictions)*100:.1f}%)")
        
        # 결과 정리
        print("\n📊 결과 정리 중...")
        results = []
        
        for idx, (pred, (current_time, orig_idx)) in enumerate(zip(corrected_predictions, timestamps)):
            target_time = current_time + timedelta(minutes=10)
            
            # 실제값
            if orig_idx + self.seq_len + self.pred_len - 1 < len(df):
                actual = df['TOTALCNT'].iloc[orig_idx + self.seq_len + self.pred_len - 1]
            else:
                actual = np.nan
            
            # 당일 최대/최소
            current_date = current_time.date()
            day_data = df[df['CURRTIME'].dt.date == current_date]
            daily_max = day_data['TOTALCNT'].max()
            daily_min = day_data['TOTALCNT'].min()
            
            results.append({
                '날짜': current_date.strftime('%Y-%m-%d'),
                '시간날짜': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                '타겟날짜': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                '타겟MAX': daily_max,
                '타겟MIN': daily_min,
                '예측값': round(pred, 2),
                '예측값_보정': round(corrected_predictions[idx], 2),
                '실제값': round(actual, 2) if not np.isnan(actual) else None,
                '오차': round(actual - corrected_predictions[idx], 2) if not np.isnan(actual) else None,
                'M14B': m14b_values[idx],
                'M14A': m14a_values[idx]
            })
        
        results_df = pd.DataFrame(results)
        
        # 성능 평가
        valid_results = results_df.dropna(subset=['실제값'])
        if len(valid_results) > 0:
            # 원본 성능
            mae_orig = mean_absolute_error(valid_results['실제값'], valid_results['예측값'])
            rmse_orig = np.sqrt(mean_squared_error(valid_results['실제값'], valid_results['예측값']))
            r2_orig = r2_score(valid_results['실제값'], valid_results['예측값'])
            
            # 보정 후 성능
            mae = mean_absolute_error(valid_results['실제값'], valid_results['예측값_보정'])
            rmse = np.sqrt(mean_squared_error(valid_results['실제값'], valid_results['예측값_보정']))
            r2 = r2_score(valid_results['실제값'], valid_results['예측값_보정'])
            
            print(f"\n📊 전체 성능:")
            print(f"  예측 수: {len(results)}개")
            print(f"  검증 가능: {len(valid_results)}개")
            print(f"  ──────────────────────────────")
            print(f"  MAE:  {mae_orig:.2f} → {mae:.2f} [개선: {mae_orig-mae:+.2f}]")
            print(f"  RMSE: {rmse_orig:.2f} → {rmse:.2f} [개선: {rmse_orig-rmse:+.2f}]")
            print(f"  R²:   {r2_orig:.4f} → {r2:.4f} [개선: {r2-r2_orig:+.4f}]")
            
            # 구간별 성능
            signal_data = valid_results[(valid_results['실제값'] >= self.SIGNAL_MIN) & 
                                       (valid_results['실제값'] <= self.SIGNAL_MAX)]
            danger_data = valid_results[valid_results['실제값'] >= self.DANGER_MIN]
            
            if len(signal_data) > 0:
                signal_mae_orig = mean_absolute_error(signal_data['실제값'], signal_data['예측값'])
                signal_mae = mean_absolute_error(signal_data['실제값'], signal_data['예측값_보정'])
                hits = np.abs(signal_data['예측값_보정'] - signal_data['실제값']) < 50
                
                print(f"\n🟠 신호구간(1651~1699):")
                print(f"  샘플: {len(signal_data)}개")
                print(f"  MAE: {signal_mae_orig:.2f} → {signal_mae:.2f} [개선: {signal_mae_orig-signal_mae:+.2f}]")
                print(f"  적중률(±50): {hits.mean()*100:.1f}%")
            
            if len(danger_data) > 0:
                danger_mae_orig = mean_absolute_error(danger_data['실제값'], danger_data['예측값'])
                danger_mae = mean_absolute_error(danger_data['실제값'], danger_data['예측값_보정'])
                hits = np.abs(danger_data['예측값_보정'] - danger_data['실제값']) < 50
                
                print(f"\n🔴 위험구간(1700+):")
                print(f"  샘플: {len(danger_data)}개")
                print(f"  MAE: {danger_mae_orig:.2f} → {danger_mae:.2f} [개선: {danger_mae_orig-danger_mae:+.2f}]")
                print(f"  적중률(±50): {hits.mean()*100:.1f}%")
                
                # M14B별 분석
                high_m14b = danger_data[danger_data['M14B'] >= 400]
                if len(high_m14b) > 0:
                    high_mae = mean_absolute_error(high_m14b['실제값'], high_m14b['예측값_보정'])
                    print(f"  M14B≥400: {len(high_m14b)}개, MAE: {high_mae:.2f}")
        
        return results_df
    
    def save_results(self, results_df, output_path='evaluation_results.csv'):
        """결과 저장"""
        # CSV용 컬럼 (예측값_보정을 예측값으로)
        output_df = results_df.copy()
        output_df['예측값'] = output_df['예측값_보정']
        
        output_columns = ['날짜', '시간날짜', '타겟날짜', '타겟MAX', '타겟MIN', '예측값']
        if '실제값' in output_df.columns:
            output_columns.extend(['실제값', '오차'])
        
        output_df[output_columns].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 결과 저장: {output_path}")
        
        # 샘플 출력
        print("\n📋 위험구간 예측 샘플:")
        danger_samples = results_df[results_df['예측값_보정'] >= 1700].head(10)
        if len(danger_samples) > 0:
            display_cols = ['시간날짜', '예측값', '예측값_보정', 'M14B']
            if '실제값' in danger_samples.columns:
                display_cols.append('실제값')
            print(danger_samples[display_cols].to_string(index=False))
        
        return output_path

# ========================================
# 메인 실행
# ========================================
def main():
    """메인 실행 함수"""
    
    eval_path = 'data/20250731_to20250806.csv'
    
    if not os.path.exists(eval_path):
        # 대체 경로 시도
        alt_paths = ['data/20250732_to20250806.csv', '20250731_to20250806.csv']
        for path in alt_paths:
            if os.path.exists(path):
                eval_path = path
                break
        else:
            print(f"❌ 평가 데이터 없음")
            return None
    
    try:
        evaluator = ModelEvaluator()
        results_df = evaluator.evaluate(eval_path)
        
        output_file = evaluator.save_results(
            results_df, 
            output_path='evaluation_results_corrected.csv'
        )
        
        print("\n" + "="*80)
        print("✅ 평가 완료!")
        print(f"📊 결과 파일: {output_file}")
        print("="*80)
        
        return results_df
        
    except Exception as e:
        print(f"\n❌ 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()