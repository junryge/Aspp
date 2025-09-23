"""
ExtremeNet V6.5 - 새 데이터 평가 (TF 2.16.1 Lambda 레이어 해결)
==================================================
data/20250732_to20250806.csv 데이터 평가
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import pickle
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# TensorFlow 2.16.1 Lambda 레이어 로드 허용
os.environ['TF_USE_LEGACY_KERAS'] = '1'

print("="*80)
print("🔮 ExtremeNet V6.5 - 새 데이터 평가")
print(f"📊 TensorFlow {tf.__version__}")
print("="*80)

# ========================================
# 데이터 처리 클래스
# ========================================
class DataEvaluator:
    def __init__(self, model_path='models_final/extremenet_v65_final.keras',
                 scaler_path='scalers_final/'):
        """평가기 초기화"""
        
        # 모델 속성 초기화
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
        # 모델 로드
        print("\n📦 모델 로딩 중...")
        
        # 모델 경로들
        model_paths = [
            model_path,
            'models_final/best_extremenet.keras',
            'checkpoints_final/extremenet_final.keras',
        ]
        
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                try:
                    print(f"  시도: {path}")
                    
                    # TensorFlow 2.16+ safe_mode=False로 Lambda 레이어 로드
                    self.model = tf.keras.models.load_model(
                        path,
                        safe_mode=False  # Lambda 레이어 로드 허용
                    )
                    
                    print(f"✅ 모델 로드 성공: {path}")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    print(f"⚠️ 로드 실패: {str(e)[:100]}")
                    continue
        
        if not model_loaded:
            print("\n❌ 모델 로드 실패!")
            print("확인한 경로:")
            for path in model_paths:
                print(f"  - {path}: {'[존재]' if os.path.exists(path) else '[없음]'}")
            raise FileNotFoundError("모델을 로드할 수 없습니다.")
        
        # 스케일러 로드
        print("\n📦 스케일러 로딩 중...")
        
        scaler_paths = [
            (f'{scaler_path}scaler_X.pkl', f'{scaler_path}scaler_y.pkl'),
            ('scalers/scaler_X.pkl', 'scalers/scaler_y.pkl'),
        ]
        
        scaler_loaded = False
        for x_path, y_path in scaler_paths:
            if os.path.exists(x_path) and os.path.exists(y_path):
                try:
                    with open(x_path, 'rb') as f:
                        self.scaler_X = pickle.load(f)
                    with open(y_path, 'rb') as f:
                        self.scaler_y = pickle.load(f)
                    print(f"✅ 스케일러 로드 성공")
                    scaler_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ 스케일러 로드 실패: {e}")
        
        if not scaler_loaded:
            print("⚠️ 스케일러 없음, 새로 생성")
            self.scaler_X = StandardScaler()
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))
            self.need_fit_scaler = True
        else:
            self.need_fit_scaler = False
        
        # 특성 컬럼
        self.feature_columns = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN', 'SIGNAL_ZONE', 'PRE_SIGNAL', 'DANGER_ZONE',
            'PATTERN', 'TREND',
            'MA_5', 'MA_10', 'MA_20',
            'STD_5', 'STD_10', 'STD_20',
            'M14B_MA_5', 'M14B_MA_10', 'M14B_MA_20',
            'CHANGE_1', 'CHANGE_5', 'CHANGE_10',
            'M14B_CHANGE_1', 'M14B_CHANGE_5', 'M14B_CHANGE_10',
            'RISE_COUNT'
        ]
        
        self.seq_len = 100
        self.pred_len = 10
        
    def load_and_process_data(self, filepath):
        """데이터 로드 및 전처리"""
        print(f"\n📂 데이터 로딩: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"✅ 원본 데이터: {len(df):,}행")
        
        # 시간 정렬
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 0값 제거
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        print(f"✅ 유효 데이터: {len(df):,}행")
        
        # 구간 통계
        normal = (df['TOTALCNT'] < 1400).sum()
        caution = ((df['TOTALCNT'] >= 1400) & (df['TOTALCNT'] <= 1650)).sum()
        signal = ((df['TOTALCNT'] >= 1651) & (df['TOTALCNT'] <= 1699)).sum()
        danger = (df['TOTALCNT'] >= 1700).sum()
        
        print(f"\n📊 데이터 분포:")
        print(f"  정상(~1400): {normal:,}개 ({normal/len(df)*100:.1f}%)")
        print(f"  주의(1400~1650): {caution:,}개 ({caution/len(df)*100:.1f}%)")
        print(f"  신호(1651~1699): {signal:,}개 ({signal/len(df)*100:.1f}%)")
        print(f"  위험(1700+): {danger:,}개 ({danger/len(df)*100:.1f}%)")
        
        return df
    
    def create_features(self, df):
        """특성 생성"""
        print("\n⚙️ 특성 생성 중...")
        
        # 기본 특성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(int)
        
        # 구간 특성
        df['SIGNAL_ZONE'] = ((df['TOTALCNT'] >= 1651) & (df['TOTALCNT'] <= 1699)).astype(int)
        df['PRE_SIGNAL'] = ((df['TOTALCNT'] >= 1600) & (df['TOTALCNT'] < 1651)).astype(int)
        df['DANGER_ZONE'] = (df['TOTALCNT'] >= 1700).astype(int)
        
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
        
        # 패턴
        df['PATTERN'] = 0
        signal_mask = ((df['TOTALCNT'] >= 1651) & (df['TOTALCNT'] <= 1699))
        high_m14b = df['M14AM14B'].rolling(100, min_periods=1).mean() > 380
        df.loc[signal_mask, 'PATTERN'] = 1
        df.loc[high_m14b, 'PATTERN'] = 2
        
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
        
        print(f"✅ 특성 생성 완료: {len(self.feature_columns)}개")
        
        # 스케일러 학습 필요시
        if hasattr(self, 'need_fit_scaler') and self.need_fit_scaler:
            print("⚙️ 스케일러 학습 중...")
            X_sample = df[self.feature_columns].values[:1000]
            y_sample = df['TOTALCNT'].values[:1000].reshape(-1, 1)
            self.scaler_X.fit(X_sample)
            self.scaler_y.fit(y_sample)
            print("✅ 스케일러 학습 완료")
        
        return df
    
    def predict_sequences(self, df):
        """시퀀스 단위로 예측"""
        
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다!")
        
        print("\n🔮 예측 시작...")
        
        results = []
        max_idx = len(df) - self.seq_len - self.pred_len + 1
        
        if max_idx <= 0:
            print(f"⚠️ 데이터 부족. 최소 {self.seq_len + self.pred_len}개 필요")
            return pd.DataFrame()
        
        for i in range(max_idx):
            # 100분 데이터
            seq_data = df.iloc[i:i+self.seq_len]
            current_time = seq_data['CURRTIME'].iloc[-1]
            target_time = current_time + timedelta(minutes=10)
            
            # 실제값
            if i+self.seq_len+self.pred_len-1 < len(df):
                actual_value = df['TOTALCNT'].iloc[i+self.seq_len+self.pred_len-1]
            else:
                actual_value = np.nan
            
            # 특성 추출 및 스케일링
            X = seq_data[self.feature_columns].values
            
            try:
                X_scaled = self.scaler_X.transform(X.reshape(-1, len(self.feature_columns)))
                X_scaled = X_scaled.reshape(1, self.seq_len, len(self.feature_columns))
            except:
                X_scaled = X.reshape(1, self.seq_len, len(self.feature_columns))
            
            # 예측
            try:
                preds = self.model.predict(X_scaled, verbose=0)
                
                if isinstance(preds, list):
                    y_pred_scaled = preds[0][0, 0]
                else:
                    y_pred_scaled = preds[0, 0]
                
                try:
                    y_pred = self.scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
                except:
                    y_pred = y_pred_scaled * 900 + 900
                
            except Exception as e:
                print(f"⚠️ 예측 오류 (인덱스 {i}): {e}")
                continue
            
            current_value = seq_data['TOTALCNT'].iloc[-1]
            
            # 일별 최대값
            current_date = current_time.date()
            day_data = df[df['CURRTIME'].dt.date == current_date]
            daily_max = day_data['TOTALCNT'].max() if len(day_data) > 0 else current_value
            
            results.append({
                'DateTime': current_time.strftime('%Y-%m-%d %H:%M'),
                'TargetDateTime': target_time.strftime('%Y-%m-%d %H:%M'),
                'CurrentValue': current_value,
                'ActualValue': actual_value,
                'PredictedValue': y_pred,
                'DailyMax': daily_max,
                'Error': abs(actual_value - y_pred) if not np.isnan(actual_value) else np.nan
            })
            
            if (i+1) % 100 == 0:
                print(f"  진행: {i+1}/{max_idx}")
        
        print(f"✅ 예측 완료: {len(results)}개")
        return pd.DataFrame(results)
    
    def evaluate_results(self, results_df):
        """평가 메트릭 계산"""
        print("\n📊 평가 메트릭:")
        
        if results_df.empty:
            print("⚠️ 예측 결과 없음")
            return
        
        valid_results = results_df.dropna(subset=['ActualValue', 'PredictedValue'])
        
        if len(valid_results) > 0:
            mae = mean_absolute_error(valid_results['ActualValue'], 
                                     valid_results['PredictedValue'])
            rmse = np.sqrt(mean_squared_error(valid_results['ActualValue'], 
                                             valid_results['PredictedValue']))
            r2 = r2_score(valid_results['ActualValue'], 
                         valid_results['PredictedValue'])
            
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.4f}")
            
            # 구간별 평가
            signal_mask = (valid_results['ActualValue'] >= 1651) & \
                         (valid_results['ActualValue'] <= 1699)
            danger_mask = valid_results['ActualValue'] >= 1700
            
            if signal_mask.sum() > 0:
                signal_mae = mean_absolute_error(
                    valid_results[signal_mask]['ActualValue'],
                    valid_results[signal_mask]['PredictedValue']
                )
                print(f"\n🟠 신호구간(1651~1699) MAE: {signal_mae:.2f}")
                print(f"  샘플 수: {signal_mask.sum()}개")
            
            if danger_mask.sum() > 0:
                danger_mae = mean_absolute_error(
                    valid_results[danger_mask]['ActualValue'],
                    valid_results[danger_mask]['PredictedValue']
                )
                print(f"🔴 위험구간(1700+) MAE: {danger_mae:.2f}")
                print(f"  샘플 수: {danger_mask.sum()}개")

# ========================================
# 메인 실행
# ========================================
def main():
    try:
        # 평가기 생성
        evaluator = DataEvaluator()
        
        # 데이터 경로 찾기
        data_paths = [
            'data/20250732_to20250806.csv',
            '20250732_to20250806.csv', 
            'data/20250807_DATA.CSV',
            '20250807_DATA.CSV',
            'data/20240201_TO_202507281705.csv',
            '20240201_TO_202507281705.csv'
        ]
        
        data_path = None
        for path in data_paths:
            if os.path.exists(path):
                data_path = path
                print(f"📌 데이터: {data_path}")
                break
        
        if not data_path:
            print("❌ 데이터 파일 없음!")
            return None
        
        # 데이터 처리
        df = evaluator.load_and_process_data(data_path)
        df = evaluator.create_features(df)
        
        # 예측
        results_df = evaluator.predict_sequences(df)
        
        if results_df.empty:
            print("⚠️ 예측 결과 없음")
            return results_df
        
        # 평가
        evaluator.evaluate_results(results_df)
        
        # 저장
        output_file = 'prediction_results.csv'
        save_columns = ['DateTime', 'TargetDateTime', 'DailyMax', 'ActualValue', 
                       'PredictedValue', 'Error'] if 'ActualValue' in results_df.columns \
                      else ['DateTime', 'TargetDateTime', 'DailyMax', 'PredictedValue']
        
        results_df[save_columns].to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: {output_file}")
        
        # 샘플 출력
        print("\n📋 결과 샘플:")
        print(results_df[save_columns].head(10))
        
        # 통계
        print(f"\n📊 예측값 요약:")
        print(f"  평균: {results_df['PredictedValue'].mean():.2f}")
        print(f"  표준편차: {results_df['PredictedValue'].std():.2f}")
        print(f"  최소: {results_df['PredictedValue'].min():.2f}")
        print(f"  최대: {results_df['PredictedValue'].max():.2f}")
        
        print("\n✅ 완료!")
        return results_df
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()