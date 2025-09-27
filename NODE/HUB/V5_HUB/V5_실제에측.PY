#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM V4 Ultimate - 실시간 예측 시스템 (간단 버전)
================================================================================
기반: 제공된 점프 감지 80% 시스템 학습 코드
목표: 과거 20분 데이터로 10분 후 CURRENT_M16A_3F_JOB_2 예측
- 데이터 부족 시 알림
- 바로 예측 실행
- 간단한 요약 제공: 예측값, {안정/주의/위험, 퍼센트}
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor
from xgboost import XGBClassifier
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM V4 Ultimate 실시간 예측 시스템")
print("📊 간단 요약: 예측값 + 위험도 + 퍼센트")
print("="*80)

# ==============================================================================
# 🧠 예측용 데이터 처리기
# ==============================================================================

class HubRoomPredictionProcessor:
    """예측용 데이터 처리 - 학습 코드와 동일한 특징 엔지니어링"""
    
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # 21개 필수 컬럼
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2', 'M16B_10F_TO_HUB_JOB'
        ]
        
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB', 'M16A_3F_TO_3F_MLUD_JOB'
        ]
        
        self.cmd_cols = [
            'M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD', 'M16A_2F_TO_HUB_CMD',
            'M14A_3F_TO_HUB_CMD', 'M14B_7F_TO_HUB_CMD'
        ]
        
        self.capa_cols = ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
        self.other_cols = ['M16A_3F_STORAGE_UTIL', 'M14_TO_M16_OFS_CUR', 'M16_TO_M14_OFS_CUR']
        
        # 확률 맵
        self.probability_map = {
            0: 0.003, 1: 0.15, 2: 0.25, 3: 0.31, 4: 0.43, 5: 0.43,
            6: 0.35, 7: 0.42, 8: 0.53, 9: 0.49, 10: 0.42,
            11: 0.47, 12: 0.52, 13: 0.60, 14: 0.54, 15: 0.66,
            16: 0.62, 17: 0.71, 18: 0.79, 19: 0.83, 20: 0.987,
            21: 0.99, 22: 0.99, 23: 0.99, 24: 0.99, 25: 0.99,
            26: 0.99, 27: 0.99, 28: 0.99, 29: 0.99, 30: 0.99
        }
    
    def load_and_process_data(self, file_path='data/HUBROOM_PIVOT_DATA.csv'):
        """데이터 로드 및 기본 처리"""
        print(f"📂 데이터 로드: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ 데이터 로드: {df.shape}")
            
            # 시간 컬럼 처리
            time_col = df.columns[0]
            try:
                df['datetime'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M')
            except:
                df['datetime'] = pd.to_datetime(df[time_col])
            
            # BRIDGE_TIME 처리
            if 'BRIDGE_TIME' not in df.columns:
                df['BRIDGE_TIME'] = 3.5
            else:
                df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)
            
            # 필수 컬럼 체크
            all_required_cols = [self.target_col] + self.inflow_cols + self.outflow_cols + self.cmd_cols + self.capa_cols + self.other_cols
            missing_cols = [col for col in all_required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ 누락 컬럼: {missing_cols[:5]}..." if len(missing_cols) > 5 else f"⚠️ 누락 컬럼: {missing_cols}")
                for col in missing_cols:
                    df[col] = 0
            
            return df
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            raise
    
    def create_prediction_features(self, df):
        """특징 생성"""
        print("🔧 특징 생성 중...")
        
        # 기본 특징들
        df['flow_balance'] = df[self.inflow_cols].sum(axis=1) - df[self.outflow_cols].sum(axis=1)
        df['flow_ratio'] = df[self.inflow_cols].sum(axis=1) / (df[self.outflow_cols].sum(axis=1) + 1)
        df['trend_20min'] = df[self.target_col].diff(20)
        df['trend_10min'] = df[self.target_col].diff(10)
        df['acceleration'] = df['trend_10min'] - df['trend_10min'].shift(10)
        df['consecutive_250+'] = (df[self.target_col] > 250).rolling(10).sum()
        df['consecutive_270+'] = (df[self.target_col] > 270).rolling(10).sum()
        df['cmd_sync_count'] = (df[self.cmd_cols] > 235).sum(axis=1)
        df['cmd_max'] = df[self.cmd_cols].max(axis=1)
        df['bridge_diff'] = df['BRIDGE_TIME'].diff(5)
        df['bridge_high'] = (df['BRIDGE_TIME'] > 4.0).astype(int)
        df['storage_x_bridge'] = df['M16A_3F_STORAGE_UTIL'] * df['BRIDGE_TIME']
        
        # 연속 300+ 확률 (간소화)
        df['consecutive_300_count'] = 0
        df['consecutive_300_prob'] = 0.003
        for i in range(30, len(df)):
            window = df[self.target_col].iloc[i-30:i].values
            count = sum(1 for v in window if v >= 300)
            prob = self.probability_map.get(count, 0.5)
            df.loc[df.index[i], 'consecutive_300_count'] = count
            df.loc[df.index[i], 'consecutive_300_prob'] = prob
        
        # 기타 특징들
        conditions = [df[self.target_col] < 150, (df[self.target_col] >= 150) & (df[self.target_col] < 300), df[self.target_col] >= 300]
        df['range_class'] = np.select(conditions, [0, 1, 2], default=1)
        
        df['past_30min_max'] = df[self.target_col].rolling(30).max()
        df['is_jump'] = ((df['past_30min_max'].shift(10) < 280) & (df[self.target_col] >= 300)).astype(int)
        
        df['change_20min'] = df[self.target_col] - df[self.target_col].shift(20)
        trend_conditions = [df['change_20min'] < -20, (df['change_20min'] >= -20) & (df['change_20min'] < 20), (df['change_20min'] >= 20) & (df['change_20min'] < 50), df['change_20min'] >= 50]
        df['trend_pattern'] = np.select(trend_conditions, [0, 1, 2, 3], default=1)
        
        df['change_rate_10min'] = ((df[self.target_col] - df[self.target_col].shift(10)) / (df[self.target_col].shift(10) + 1)) * 100
        df['change_rate_20min'] = ((df[self.target_col] - df[self.target_col].shift(20)) / (df[self.target_col].shift(20) + 1)) * 100
        df['change_rate_30min'] = ((df[self.target_col] - df[self.target_col].shift(30)) / (df[self.target_col].shift(30) + 1)) * 100
        
        df['volatility_10min'] = df[self.target_col].rolling(10).std()
        df['volatility_20min'] = df[self.target_col].rolling(20).std()
        df['volatility_30min'] = df[self.target_col].rolling(30).std()
        
        df['distance_to_300'] = 300 - df[self.target_col]
        df['near_extreme'] = (df[self.target_col] > 280).astype(int)
        df['recent_5min_mean'] = df[self.target_col].rolling(5).mean()
        df['recent_5min_max'] = df[self.target_col].rolling(5).max()
        df['recent_10min_mean'] = df[self.target_col].rolling(10).mean()
        df['in_jump_zone'] = ((df[self.target_col] >= 275) & (df[self.target_col] <= 279)).astype(int)
        
        # NaN 처리
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(0)
        
        print(f"✅ 특징 완료: {len(df.columns)}개")
        return df

# ==============================================================================
# 🤖 예측 시스템
# ==============================================================================

class HUBROOMPredictionSystem:
    """HUBROOM 예측 시스템"""
    
    def __init__(self, model_dir='./checkpoints_jump80'):
        self.model_dir = model_dir
        self.models = {}
        self.target_col = 'CURRENT_M16A_3F_JOB_2'  # 누락된 속성 추가
        self.load_models()
    
    def load_models(self):
        """모델 로드"""
        print(f"🔄 모델 로드: {self.model_dir}")
        
        model_files = {'model_value': 'model_value.pkl'}
        models_dir = os.path.join(self.model_dir, 'models')
        
        if not os.path.exists(models_dir):
            print(f"❌ 모델 디렉토리 없음: {models_dir}")
            return
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                print(f"  ✅ {model_name} 로드 완료")
            else:
                print(f"  ❌ {model_name} 파일 없음")
    
    def prepare_features_for_prediction(self, X_seq):
        """시퀀스를 특징으로 변환"""
        last_features = X_seq[:, -1, :]
        mean_features = np.mean(X_seq, axis=1)
        std_features = np.std(X_seq, axis=1)
        max_features = np.max(X_seq, axis=1)
        min_features = np.min(X_seq, axis=1)
        trend_features = X_seq[:, -1, :] - X_seq[:, 0, :]
        
        features = np.hstack([last_features, mean_features, std_features, max_features, min_features, trend_features])
        return features
    
    def get_risk_classification(self, value):
        """위험도 분류"""
        if value < 220:
            return {'level': 'SAFE', 'korean': '안정', 'percentage': max(10, min(30, (220 - value) / 220 * 100))}
        elif value < 300:
            return {'level': 'WARNING', 'korean': '주의', 'percentage': max(30, min(70, (value - 220) / 80 * 100))}
        else:
            return {'level': 'DANGER', 'korean': '위험', 'percentage': max(70, min(99, 70 + (value - 300) / 100 * 29))}
    
    def predict_sequence(self, df, start_idx):
        """단일 시퀀스 예측"""
        seq_len = 20
        
        if start_idx < seq_len:
            return None
            
        # 특징 컬럼 선택
        feature_cols = [col for col in df.columns if col not in ['datetime', 'range_class', 'is_jump', 'trend_pattern']]
        
        # 시퀀스 생성
        sequence = df[feature_cols].iloc[start_idx-seq_len:start_idx].values
        sequence = sequence.reshape(1, seq_len, -1)
        
        # 특징 준비
        features = self.prepare_features_for_prediction(sequence)
        
        # 예측 수행
        predictions = {}
        
        if 'model_value' in self.models:
            value_pred = self.models['model_value'].predict(features)[0]
            risk_info = self.get_risk_classification(value_pred)
            
            predictions['value'] = {
                'prediction': value_pred,
                'risk_korean': risk_info['korean'],
                'risk_percentage': risk_info['percentage']
            }
        
        # 현재 정보
        current_value = df[self.target_col].iloc[start_idx-1]
        predictions['current'] = {
            'value': current_value,
            'datetime': df['datetime'].iloc[start_idx-1]
        }
        
        return predictions

# ==============================================================================
# 🎯 실시간 예측 실행기
# ==============================================================================

class RealTimePredictionRunner:
    """실시간 예측 실행"""
    
    def __init__(self):
        self.processor = HubRoomPredictionProcessor()
        self.predictor = HUBROOMPredictionSystem()
    
    def run_prediction(self, file_path='data/HUBROOM_PIVOT_DATA.csv', num_predictions=5):
        """예측 실행"""
        try:
            # 데이터 로드
            df = self.processor.load_and_process_data(file_path)
            
            # 데이터 부족 체크
            if len(df) < 30:
                print(f"데이터 부족: {len(df)}개 (최소 30개 필요)")
                return []
            
            df = self.processor.create_prediction_features(df)
            
            # 모델 체크
            if not self.predictor.models:
                print("모델 없음")
                return []
            
            # 예측 수행
            results = []
            start_idx = len(df) - num_predictions
            
            for i in range(num_predictions):
                idx = start_idx + i
                if idx >= 20 and idx < len(df):
                    try:
                        result = self.predictor.predict_sequence(df, idx)
                        if result:
                            results.append(result)
                    except:
                        continue
            
            # 결과 표시
            self.display_simple_results(results)
            return results
            
        except Exception as e:
            print(f"오류: {e}")
            return []
    
    def display_simple_results(self, results):
        """초간단 결과 표시"""
        if not results:
            print("예측 결과 없음")
            return
        
        # 최신 예측만 간단하게
        latest = results[-1]
        if 'value' in latest:
            pred_value = latest['value']['prediction']
            risk_korean = latest['value']['risk_korean']
            risk_percentage = latest['value']['risk_percentage']
            
            print(f"{{{pred_value:.1f},{risk_korean}({risk_percentage:.0f}%)}}")
        else:
            print("예측 실패")

# ==============================================================================
# 🚀 메인 실행
# ==============================================================================

def main():
    """메인 실행"""
    try:
        runner = RealTimePredictionRunner()
        results = runner.run_prediction('data/HUBROOM_PIVOT_DATA.csv', 5)
        
        if results:
            print(f"\n✅ 예측 완료: {len(results)}개 결과")
        
    except FileNotFoundError:
        print("❌ 파일 없음: data/HUBROOM_PIVOT_DATA.csv")
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()