"""
📊 패턴 통합 ExtremeNet 평가 코드 (고속 버전)
배치 처리로 속도 대폭 개선
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

# Lambda 레이어 허용
tf.keras.config.enable_unsafe_deserialization()

print("="*80)
print("📊 ExtremeNet 평가 시스템 (고속)")
print("평가 대상: data/20250731_to20250806.csv")
print("="*80)

# ========================================
# 커스텀 손실함수
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
# 고속 평가 클래스
# ========================================
class FastEvaluator:
    def __init__(self):
        self.seq_len = 100
        self.pred_len = 10
        self.batch_size = 64  # 배치 크기 증가
        
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
        
        # 모델 경로
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
                    safe_mode=False,
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
        
    def create_features_batch(self, df):
        """특성 생성 (벡터화로 속도 개선)"""
        print("  특성 생성 중...")
        start = time.time()
        
        # 기본 특성 (벡터화)
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(int)
        
        # 구간 특성 (벡터화)
        df['SIGNAL_ZONE'] = ((df['TOTALCNT'] >= self.SIGNAL_MIN) & 
                             (df['TOTALCNT'] <= self.SIGNAL_MAX)).astype(int)
        df['PRE_SIGNAL'] = ((df['TOTALCNT'] >= 1600) & 
                            (df['TOTALCNT'] < self.SIGNAL_MIN)).astype(int)
        df['DANGER_ZONE'] = (df['TOTALCNT'] >= self.DANGER_MIN).astype(int)
        
        # 이동평균 (한번에)
        for w in [5, 10, 20]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
            df[f'M14B_MA_{w}'] = df['M14AM14B'].rolling(w, min_periods=1).mean()
        
        # 변화율 (한번에)
        for lag in [1, 5, 10]:
            df[f'CHANGE_{lag}'] = df['TOTALCNT'].diff(lag).fillna(0)
            df[f'M14B_CHANGE_{lag}'] = df['M14AM14B'].diff(lag).fillna(0)
        
        # 연속 상승
        df['RISE'] = (df['TOTALCNT'] > df['TOTALCNT'].shift(1)).astype(int)
        df['RISE_COUNT'] = df['RISE'].rolling(10, min_periods=1).sum()
        
        # 패턴 (간소화)
        df['PATTERN'] = 1
        df['PATTERN_RISK'] = 0.5
        
        # M14B 레벨
        df['M14B_HIGH'] = (df['M14AM14B'] >= 350).astype(int)
        df['M14B_VERY_HIGH'] = (df['M14AM14B'] >= 400).astype(int)
        df['M14B_EXTREME'] = (df['M14AM14B'] >= 450).astype(int)
        
        # 급증 신호
        df['SPIKE_SIGNAL'] = ((df['CHANGE_10'] > 50) & (df['RISE_COUNT'] > 5)).astype(int)
        
        # 추세 (간소화)
        df['TREND'] = 1
        
        # 특성 컬럼
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
        
        print(f"  특성 생성 완료 ({time.time()-start:.1f}초)")
        return df
    
    def evaluate_fast(self, data_path):
        """고속 평가"""
        print(f"\n📂 데이터 로딩: {data_path}")
        
        # 데이터 로드
        df = pd.read_csv(data_path)
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        print(f"✅ 데이터: {len(df):,}행")
        
        # 특성 생성
        df = self.create_features_batch(df)
        
        # 시퀀스 준비
        total_sequences = len(df) - self.seq_len - self.pred_len + 1
        if total_sequences <= 0:
            print("❌ 데이터 부족")
            return pd.DataFrame()
        
        print(f"\n🚀 고속 예측 시작")
        print(f"  총 시퀀스: {total_sequences:,}개")
        print(f"  배치 크기: {self.batch_size}")
        
        start_time = time.time()
        
        # 모든 시퀀스 준비 (벡터화)
        print("  시퀀스 준비 중...")
        indices = np.arange(total_sequences)
        
        # 배치 단위로 처리
        all_predictions = []
        
        for batch_start in range(0, len(indices), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            # 배치 데이터 준비
            batch_X = []
            for i in batch_indices:
                X = df[self.feature_columns].iloc[i:i+self.seq_len].values
                batch_X.append(X)
            
            batch_X = np.array(batch_X, dtype=np.float32)
            
            # 스케일링
            batch_X_flat = batch_X.reshape(-1, len(self.feature_columns))
            batch_X_scaled = self.scaler_X.transform(batch_X_flat)
            batch_X_scaled = batch_X_scaled.reshape(batch_X.shape)
            
            # 예측
            preds = self.model.predict(batch_X_scaled, verbose=0)
            y_pred_scaled = preds[0][:, 0]
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            all_predictions.extend(y_pred)
            
            # 진행 상황 (10배치마다)
            if (batch_start // self.batch_size + 1) % 10 == 0:
                elapsed = time.time() - start_time
                progress = len(all_predictions)
                speed = progress / elapsed
                eta = (total_sequences - progress) / speed
                print(f"    진행: {progress:,}/{total_sequences:,} "
                      f"({progress/total_sequences*100:.1f}%) "
                      f"속도: {speed:.0f}개/초 "
                      f"남은시간: {eta:.0f}초")
        
        # 결과 정리
        print("\n  결과 정리 중...")
        results = []
        
        # 일별 통계 미리 계산
        daily_stats = df.groupby(df['CURRTIME'].dt.date)['TOTALCNT'].agg(['max', 'min']).to_dict()
        
        for i in range(len(all_predictions)):
            seq_data = df.iloc[i:i+self.seq_len]
            current_time = seq_data['CURRTIME'].iloc[-1]
            target_time = current_time + timedelta(minutes=10)
            current_date = current_time.date()
            
            # 실제값
            if i + self.seq_len + self.pred_len - 1 < len(df):
                actual = df['TOTALCNT'].iloc[i + self.seq_len + self.pred_len - 1]
            else:
                actual = np.nan
            
            results.append({
                '날짜': current_date.strftime('%Y-%m-%d'),
                '시간날짜': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                '타겟날짜': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                '타겟MAX': daily_stats['max'][current_date],
                '타겟MIN': daily_stats['min'][current_date],
                '예측값': round(all_predictions[i], 2),
                '실제값': round(actual, 2) if not np.isnan(actual) else None,
                '오차': round(actual - all_predictions[i], 2) if not np.isnan(actual) else None
            })
        
        # 데이터프레임 생성
        results_df = pd.DataFrame(results)
        
        total_time = time.time() - start_time
        print(f"\n⏱️ 처리 시간: {total_time:.1f}초")
        print(f"⚡ 처리 속도: {len(all_predictions)/total_time:.0f}개/초")
        
        # 성능 평가
        self.evaluate_performance(results_df)
        
        return results_df
    
    def evaluate_performance(self, results_df):
        """성능 평가"""
        valid_results = results_df.dropna(subset=['실제값'])
        
        if len(valid_results) > 0:
            mae = mean_absolute_error(valid_results['실제값'], valid_results['예측값'])
            rmse = np.sqrt(mean_squared_error(valid_results['실제값'], valid_results['예측값']))
            r2 = r2_score(valid_results['실제값'], valid_results['예측값'])
            
            print(f"\n📊 전체 성능:")
            print(f"  예측 수: {len(results_df):,}개")
            print(f"  검증 가능: {len(valid_results):,}개")
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.4f}")
            
            # 구간별 성능
            signal_data = valid_results[(valid_results['실제값'] >= self.SIGNAL_MIN) & 
                                       (valid_results['실제값'] <= self.SIGNAL_MAX)]
            danger_data = valid_results[valid_results['실제값'] >= self.DANGER_MIN]
            
            if len(signal_data) > 0:
                signal_mae = mean_absolute_error(signal_data['실제값'], signal_data['예측값'])
                print(f"\n🟠 신호구간(1651~1699):")
                print(f"  샘플: {len(signal_data)}개")
                print(f"  MAE: {signal_mae:.2f}")
            
            if len(danger_data) > 0:
                danger_mae = mean_absolute_error(danger_data['실제값'], danger_data['예측값'])
                print(f"\n🔴 위험구간(1700+):")
                print(f"  샘플: {len(danger_data)}개")
                print(f"  MAE: {danger_mae:.2f}")
    
    def save_results(self, results_df, output_path='evaluation_results.csv'):
        """결과 저장"""
        output_columns = ['날짜', '시간날짜', '타겟날짜', '타겟MAX', '타겟MIN', '예측값']
        
        if '실제값' in results_df.columns:
            output_columns.extend(['실제값', '오차'])
        
        output_df = results_df[output_columns]
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 결과 저장: {output_path}")
        print("\n📋 결과 샘플:")
        print(output_df.head(10))
        
        return output_path

# ========================================
# 메인
# ========================================
def main():
    """메인 실행"""
    
    eval_path = 'data/20250731_to20250806.csv'
    
    if not os.path.exists(eval_path):
        print(f"❌ 평가 데이터 없음: {eval_path}")
        
        # 다른 경로 시도
        alt_paths = [
            'data/20250732_to20250806.csv',
            '20250731_to20250806.csv',
            '20250732_to20250806.csv'
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                eval_path = path
                print(f"✅ 대체 경로 발견: {path}")
                break
        else:
            return None
    
    try:
        evaluator = FastEvaluator()
        results_df = evaluator.evaluate_fast(eval_path)
        
        if not results_df.empty:
            output_file = evaluator.save_results(
                results_df, 
                output_path='evaluation_results_fast.csv'
            )
            
            print("\n" + "="*80)
            print("✅ 평가 완료!")
            print(f"📊 결과 파일: {output_file}")
            print("="*80)
        
        return results_df
        
    except Exception as e:
        print(f"\n❌ 평가 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()