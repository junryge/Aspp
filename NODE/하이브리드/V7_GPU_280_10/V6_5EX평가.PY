"""
ExtremeNet V6.5 - 새 데이터 평가 및 예측 결과 저장 (완전 수정판)
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

print("="*80)
print("🔮 ExtremeNet V6.5 - 새 데이터 평가")
print("📊 평가 데이터: data/20250732_to20250806.csv")
print("="*80)

# ========================================
# 데이터 처리 클래스
# ========================================
class DataEvaluator:
    def __init__(self, model_path='models_final/extremenet_v65_final.keras',
                 scaler_path='scalers_final/'):
        """평가기 초기화"""
        
        # 모델 속성 먼저 초기화 (중요!)
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
        # 모델 로드
        print("\n📦 모델 로딩 중...")
        
        # custom_objects 정의
        custom_objects = {
            'weighted_mae_signal': self.weighted_mae_signal,
            'weighted_mae': self.weighted_mae
        }
        
        # 여러 경로 시도
        model_paths = [
            model_path,
            'models_final/best_extremenet.keras',
            'models/extremenet_v65_fixed_final.keras', 
            'models/best_extremenet_fixed.keras',
            'checkpoints_final/extremenet_final.keras',
            'extremenet_v65_final.keras',
            'best_extremenet.keras'
        ]
        
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.model = tf.keras.models.load_model(path, custom_objects=custom_objects)
                    print(f"✅ 모델 로드 성공: {path}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ {path} 로드 실패: {str(e)[:100]}")
                    continue
        
        if not model_loaded:
            print("\n❌ 모델을 찾을 수 없습니다!")
            print("다음 경로들을 확인했습니다:")
            for path in model_paths:
                print(f"  - {path}: {'[존재]' if os.path.exists(path) else '[없음]'}")
            raise FileNotFoundError("모델 파일을 찾을 수 없습니다. 먼저 모델을 학습시켜주세요.")
        
        # 스케일러 로드
        print("\n📦 스케일러 로딩 중...")
        
        scaler_paths = [
            (f'{scaler_path}scaler_X.pkl', f'{scaler_path}scaler_y.pkl'),
            ('scalers/scaler_X.pkl', 'scalers/scaler_y.pkl'),
            ('scalers_final/scaler_X.pkl', 'scalers_final/scaler_y.pkl')
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
                    continue
        
        if not scaler_loaded:
            print("⚠️ 스케일러 없음, 새로 생성 (예측 정확도 낮을 수 있음)")
            self.scaler_X = StandardScaler()
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))
            self.need_fit_scaler = True
        else:
            self.need_fit_scaler = False
        
        # 특성 컬럼 정의
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
        
    def weighted_mae_signal(self, y_true, y_pred):
        """커스텀 손실함수 - 신호구간 강조"""
        weights = tf.ones_like(y_true)
        signal_mask = tf.logical_and(y_true >= 0.65, y_true <= 0.75)
        weights = tf.where(signal_mask, 5.0, weights)
        danger_mask = y_true > 0.75
        weights = tf.where(danger_mask, 10.0, weights)
        mae = tf.abs(y_true - y_pred)
        weighted_mae = mae * weights
        return tf.reduce_mean(weighted_mae)
    
    def weighted_mae(self, y_true, y_pred):
        """커스텀 손실함수 (대체)"""
        return self.weighted_mae_signal(y_true, y_pred)
        
    def load_and_process_data(self, filepath):
        """데이터 로드 및 전처리"""
        print(f"\n📂 데이터 로딩: {filepath}")
        
        # CSV 로드
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
        df['TREND'] = 1  # 기본값 보합
        for i in range(20, len(df)):
            recent = df['TOTALCNT'].iloc[i-20:i].values
            if len(recent) > 1:
                slope = np.polyfit(range(len(recent)), recent, 1)[0]
                if slope > 5:
                    df.loc[i, 'TREND'] = 2  # 상승
                elif slope < -5:
                    df.loc[i, 'TREND'] = 0  # 하락
        
        print(f"✅ 특성 생성 완료: {len(self.feature_columns)}개")
        
        # 스케일러가 학습되지 않은 경우 학습
        if hasattr(self, 'need_fit_scaler') and self.need_fit_scaler:
            print("⚙️ 스케일러 학습 중...")
            X_sample = df[self.feature_columns].values[:1000]  # 샘플로 학습
            y_sample = df['TOTALCNT'].values[:1000].reshape(-1, 1)
            self.scaler_X.fit(X_sample)
            self.scaler_y.fit(y_sample)
            print("✅ 스케일러 학습 완료")
        
        return df
    
    def predict_sequences(self, df):
        """시퀀스 단위로 예측"""
        
        # 모델 체크
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다!")
        
        print("\n🔮 예측 시작...")
        
        results = []
        
        # 예측 가능한 범위 계산
        max_idx = len(df) - self.seq_len - self.pred_len + 1
        if max_idx <= 0:
            print(f"⚠️ 데이터가 부족합니다. 최소 {self.seq_len + self.pred_len}개 필요")
            return pd.DataFrame()
        
        # 100분 시퀀스로 10분 후 예측
        for i in range(max_idx):
            # 100분 데이터
            seq_data = df.iloc[i:i+self.seq_len]
            
            # 현재 시간
            current_time = seq_data['CURRTIME'].iloc[-1]
            
            # 10분 후 시간
            target_time = current_time + timedelta(minutes=10)
            
            # 실제값 (10분 후)
            if i+self.seq_len+self.pred_len-1 < len(df):
                actual_value = df['TOTALCNT'].iloc[i+self.seq_len+self.pred_len-1]
            else:
                actual_value = np.nan
            
            # 특성 추출
            X = seq_data[self.feature_columns].values
            
            # 스케일링
            try:
                X_scaled = self.scaler_X.transform(X.reshape(-1, len(self.feature_columns)))
                X_scaled = X_scaled.reshape(1, self.seq_len, len(self.feature_columns))
            except Exception as e:
                print(f"⚠️ 스케일링 오류: {e}")
                X_scaled = X.reshape(1, self.seq_len, len(self.feature_columns))
            
            # 예측
            try:
                preds = self.model.predict(X_scaled, verbose=0)
                
                # 모델 출력이 여러 개인 경우 첫 번째 출력 사용
                if isinstance(preds, list):
                    y_pred_scaled = preds[0][0, 0]
                else:
                    y_pred_scaled = preds[0, 0]
                
                # 역스케일링
                try:
                    y_pred = self.scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
                except:
                    # 스케일러가 없는 경우 대략적 변환
                    y_pred = y_pred_scaled * 900 + 900
                
            except Exception as e:
                print(f"⚠️ 예측 오류 (인덱스 {i}): {e}")
                continue
            
            # 현재값
            current_value = seq_data['TOTALCNT'].iloc[-1]
            
            # 일별 최대값 계산 (당일 최대값)
            current_date = current_time.date()
            day_data = df[df['CURRTIME'].dt.date == current_date]
            if len(day_data) > 0:
                daily_max = day_data['TOTALCNT'].max()
            else:
                daily_max = current_value
            
            # 결과 저장
            results.append({
                'DateTime': current_time.strftime('%Y-%m-%d %H:%M'),
                'TargetDateTime': target_time.strftime('%Y-%m-%d %H:%M'),
                'CurrentValue': current_value,
                'ActualValue': actual_value,
                'PredictedValue': y_pred,
                'DailyMax': daily_max,
                'Error': abs(actual_value - y_pred) if not np.isnan(actual_value) else np.nan
            })
            
            # 진행 상황 출력
            if (i+1) % 100 == 0:
                print(f"  진행: {i+1}/{max_idx}")
        
        print(f"✅ 예측 완료: {len(results)}개")
        return pd.DataFrame(results)
    
    def evaluate_results(self, results_df):
        """평가 메트릭 계산"""
        print("\n📊 평가 메트릭:")
        
        if results_df.empty:
            print("⚠️ 예측 결과가 없습니다.")
            return
        
        # NaN 제거
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
        else:
            print("⚠️ 유효한 예측 결과가 없습니다.")

# ========================================
# 메인 실행
# ========================================
def main():
    try:
        # 평가기 생성
        evaluator = DataEvaluator()
        
        # 데이터 경로
        data_path = 'data/20250732_to20250806.csv'
        
        # 대체 경로
        if not os.path.exists(data_path):
            alt_paths = [
                '20250732_to20250806.csv', 
                'data/20250807_DATA.CSV',
                '20250807_DATA.CSV',
                'data/20240201_TO_202507281705.csv',
                '20240201_TO_202507281705.csv'
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    data_path = alt
                    print(f"📌 대체 경로 사용: {data_path}")
                    break
        
        if not os.path.exists(data_path):
            # 테스트용 샘플 데이터 생성
            print("⚠️ 데이터 파일이 없어서 샘플 데이터를 생성합니다.")
            
            # 기존 데이터에서 일부 추출
            original_paths = [
                'data/20240201_TO_202507281705.csv',
                '20240201_TO_202507281705.csv'
            ]
            
            for original_path in original_paths:
                if os.path.exists(original_path):
                    print(f"📂 원본 데이터에서 샘플 추출: {original_path}")
                    df_original = pd.read_csv(original_path)
                    # 마지막 1000개 행을 테스트 데이터로 사용
                    df_test = df_original.tail(1000)
                    df_test.to_csv('test_data.csv', index=False)
                    data_path = 'test_data.csv'
                    print(f"✅ 테스트 데이터 생성: {data_path}")
                    break
            
            if not os.path.exists(data_path):
                print("❌ 데이터 파일을 찾을 수 없습니다!")
                return None
        
        # 데이터 로드 및 처리
        df = evaluator.load_and_process_data(data_path)
        df = evaluator.create_features(df)
        
        # 예측 수행
        results_df = evaluator.predict_sequences(df)
        
        if results_df.empty:
            print("⚠️ 예측 결과가 없습니다.")
            return results_df
        
        # 평가
        evaluator.evaluate_results(results_df)
        
        # 결과 저장
        output_file = 'prediction_results.csv'
        
        # 저장할 컬럼 선택
        save_columns = ['DateTime', 'TargetDateTime', 'DailyMax', 'PredictedValue']
        if 'ActualValue' in results_df.columns:
            # 실제값이 있으면 포함
            save_columns = ['DateTime', 'TargetDateTime', 'DailyMax', 'ActualValue', 'PredictedValue', 'Error']
        
        results_df[save_columns].to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 결과 저장: {output_file}")
        
        # 샘플 출력
        print("\n📋 예측 결과 샘플 (처음 10개):")
        print(results_df[save_columns].head(10))
        
        # 요약 통계
        print("\n📊 예측값 요약:")
        print(f"  평균: {results_df['PredictedValue'].mean():.2f}")
        print(f"  표준편차: {results_df['PredictedValue'].std():.2f}")
        print(f"  최소: {results_df['PredictedValue'].min():.2f}")
        print(f"  최대: {results_df['PredictedValue'].max():.2f}")
        
        if 'Error' in results_df.columns:
            valid_errors = results_df['Error'].dropna()
            if len(valid_errors) > 0:
                print(f"\n📊 오차 분포:")
                print(f"  평균 오차: {valid_errors.mean():.2f}")
                print(f"  오차 < 50: {(valid_errors < 50).sum()/len(valid_errors)*100:.1f}%")
                print(f"  오차 < 100: {(valid_errors < 100).sum()/len(valid_errors)*100:.1f}%")
        
        print("\n✅ 평가 완료!")
        return results_df
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()