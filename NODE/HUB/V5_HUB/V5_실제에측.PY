#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM V4 Ultimate - 실시간 예측 시스템
================================================================================
기반: 제공된 점프 감지 80% 시스템 학습 코드
목표: 과거 20분 데이터로 10분 후 CURRENT_M16A_3F_JOB_2 예측
- Model 1: PatchTST (안정형, 200-300 구간 전문)  
- Model 2: PatchTST+PINN (극단형, 300+ 전문)
- 예측 구간: 90-220 안정, 220-299 주의, 300+ 위험
- 파일: data/HUBROOM_PIVOT_DATA.csv
================================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor
from xgboost import XGBClassifier
import joblib
import os
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM V4 Ultimate 실시간 예측 시스템")
print("📊 3구간 분류 + 점프 감지 + 상승/하락 패턴")
print("⏰ 과거 20분 → 10분 후 예측")
print("="*80)

# ==============================================================================
# 🧠 예측용 데이터 처리기 (학습 코드 기반)
# ==============================================================================

class HubRoomPredictionProcessor:
    """예측용 데이터 처리 - 학습 코드와 동일한 특징 엔지니어링"""
    
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # 21개 필수 컬럼 (학습 코드와 동일)
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB',
            'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2',
            'M14B_7F_TO_HUB_JOB2',
            'M16B_10F_TO_HUB_JOB'
        ]
        
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB',
            'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB',
            'M16A_3F_TO_M14B_7F_JOB',
            'M16A_3F_TO_3F_MLUD_JOB'
        ]
        
        self.cmd_cols = [
            'M16A_3F_CMD',
            'M16A_6F_TO_HUB_CMD',
            'M16A_2F_TO_HUB_CMD',
            'M14A_3F_TO_HUB_CMD',
            'M14B_7F_TO_HUB_CMD'
        ]
        
        self.capa_cols = [
            'M16A_6F_LFT_MAXCAPA',
            'M16A_2F_LFT_MAXCAPA'
        ]
        
        self.other_cols = [
            'M16A_3F_STORAGE_UTIL',
            'M14_TO_M16_OFS_CUR',
            'M16_TO_M14_OFS_CUR'
        ]
        
        # 확률 맵 (학습 코드와 동일)
        self.probability_map = {
            0: 0.003, 1: 0.15, 2: 0.25, 3: 0.31, 4: 0.43, 5: 0.43,
            6: 0.35, 7: 0.42, 8: 0.53, 9: 0.49, 10: 0.42,
            11: 0.47, 12: 0.52, 13: 0.60, 14: 0.54, 15: 0.66,
            16: 0.62, 17: 0.71, 18: 0.79, 19: 0.83, 20: 0.987,
            21: 0.99, 22: 0.99, 23: 0.99, 24: 0.99, 25: 0.99,
            26: 0.99, 27: 0.99, 28: 0.99, 29: 0.99, 30: 0.99
        }
    
    def load_and_process_data(self, file_path='data/HUBROOM_PIVOT_DATA.csv'):
        """데이터 로드 및 BRIDGE_TIME 처리"""
        print(f"\n📂 데이터 로드: {file_path}")
        
        # 메인 데이터 로드
        df = pd.read_csv(file_path)
        print(f"✅ 데이터 로드 완료: {df.shape}")
        
        # 시간 컬럼 처리
        time_col = df.columns[0]
        df['datetime'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M')
        
        # BRIDGE_TIME 처리 (있다면)
        if 'BRIDGE_TIME' not in df.columns:
            # BRIDGE_TIME이 없으면 기본값으로 설정
            df['BRIDGE_TIME'] = 3.5
            print("⚠️ BRIDGE_TIME 컬럼이 없어 기본값(3.5)으로 설정")
        else:
            # 결측치 처리
            df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)
            print(f"✅ BRIDGE_TIME 처리 완료")
        
        return df
    
    def create_prediction_features(self, df):
        """예측용 특징 생성 (학습 코드와 동일한 특징 엔지니어링)"""
        print("\n🔧 특징 엔지니어링 중...")
        
        # 1. 유입/유출 밸런스
        df['flow_balance'] = df[self.inflow_cols].sum(axis=1) - df[self.outflow_cols].sum(axis=1)
        df['flow_ratio'] = df[self.inflow_cols].sum(axis=1) / (df[self.outflow_cols].sum(axis=1) + 1)
        
        # 2. 추세 특징
        df['trend_20min'] = df[self.target_col].diff(20)
        df['trend_10min'] = df[self.target_col].diff(10)
        df['acceleration'] = df['trend_10min'] - df['trend_10min'].shift(10)
        
        # 3. 연속 패턴
        df['consecutive_250+'] = (df[self.target_col] > 250).rolling(10).sum()
        df['consecutive_270+'] = (df[self.target_col] > 270).rolling(10).sum()
        
        # 4. CMD 동기화
        df['cmd_sync_count'] = (df[self.cmd_cols] > 235).sum(axis=1)
        df['cmd_max'] = df[self.cmd_cols].max(axis=1)
        
        # 5. 브릿지타임 변화
        df['bridge_diff'] = df['BRIDGE_TIME'].diff(5)
        df['bridge_high'] = (df['BRIDGE_TIME'] > 4.0).astype(int)
        
        # 6. storage x bridge 상호작용
        df['storage_x_bridge'] = df['M16A_3F_STORAGE_UTIL'] * df['BRIDGE_TIME']
        
        # 7. 연속 300+ 카운트와 확률
        consecutive_300_counts = []
        consecutive_300_probs = []
        
        for i in range(len(df)):
            if i < 30:
                count = 0
                prob = 0.003
            else:
                window = df[self.target_col].iloc[i-30:i].values
                count = sum(1 for v in window if v >= 300)
                prob = self.probability_map.get(count, 0.5)
            
            consecutive_300_counts.append(count)
            consecutive_300_probs.append(prob)
        
        df['consecutive_300_count'] = consecutive_300_counts
        df['consecutive_300_prob'] = consecutive_300_probs
        
        # 8. 3구간 분류
        conditions = [
            df[self.target_col] < 150,
            (df[self.target_col] >= 150) & (df[self.target_col] < 300),
            df[self.target_col] >= 300
        ]
        choices = [0, 1, 2]  # 0: 저위험, 1: 중위험, 2: 고위험
        df['range_class'] = np.select(conditions, choices, default=1)
        
        # 9. 점프 여부
        df['past_30min_max'] = df[self.target_col].rolling(30).max()
        df['is_jump'] = ((df['past_30min_max'].shift(10) < 280) & 
                        (df[self.target_col] >= 300)).astype(int)
        
        # 10. 상승/하락 패턴
        df['change_20min'] = df[self.target_col] - df[self.target_col].shift(20)
        
        trend_conditions = [
            df['change_20min'] < -20,
            (df['change_20min'] >= -20) & (df['change_20min'] < 20),
            (df['change_20min'] >= 20) & (df['change_20min'] < 50),
            df['change_20min'] >= 50
        ]
        trend_choices = [0, 1, 2, 3]  # 0:down, 1:stable, 2:gradual_up, 3:rapid_up
        df['trend_pattern'] = np.select(trend_conditions, trend_choices, default=1)
        
        # 11. 변화율 (%)
        df['change_rate_10min'] = ((df[self.target_col] - df[self.target_col].shift(10)) / 
                                   (df[self.target_col].shift(10) + 1)) * 100
        df['change_rate_20min'] = ((df[self.target_col] - df[self.target_col].shift(20)) / 
                                   (df[self.target_col].shift(20) + 1)) * 100
        df['change_rate_30min'] = ((df[self.target_col] - df[self.target_col].shift(30)) / 
                                   (df[self.target_col].shift(30) + 1)) * 100
        
        # 12. 변동성
        df['volatility_10min'] = df[self.target_col].rolling(10).std()
        df['volatility_20min'] = df[self.target_col].rolling(20).std()
        df['volatility_30min'] = df[self.target_col].rolling(30).std()
        
        # 13. 극단값 근접도
        df['distance_to_300'] = 300 - df[self.target_col]
        df['near_extreme'] = (df[self.target_col] > 280).astype(int)
        
        # 14. 최근 통계
        df['recent_5min_mean'] = df[self.target_col].rolling(5).mean()
        df['recent_5min_max'] = df[self.target_col].rolling(5).max()
        df['recent_10min_mean'] = df[self.target_col].rolling(10).mean()
        
        # 15. 277 구간 특별 지표
        df['in_jump_zone'] = ((df[self.target_col] >= 275) & (df[self.target_col] <= 279)).astype(int)
        
        # NaN 처리
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(0)
        
        print(f"✅ 특징 생성 완료: {len(df.columns)}개 컬럼")
        return df

# ==============================================================================
# 🤖 예측 시스템 (학습된 모델 로드)
# ==============================================================================

class HUBROOMPredictionSystem:
    """HUBROOM 예측 시스템 - 학습된 모델 사용"""
    
    def __init__(self, model_dir='./checkpoints_jump80'):
        self.model_dir = model_dir
        self.models = {}
        self.feature_indices = {}
        
        # 모델 로드
        self.load_models()
    
    def load_models(self):
        """학습된 모델들 로드"""
        print("\n🔄 학습된 모델 로드 중...")
        
        model_files = {
            'model_jump': 'model_jump.pkl',
            'model_range': 'model_range.pkl', 
            'model_trend': 'model_trend.pkl',
            'model_value': 'model_value.pkl'
        }
        
        models_dir = os.path.join(self.model_dir, 'models')
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                print(f"  ✅ {model_name} 로드 완료")
            else:
                print(f"  ❌ {model_name} 파일 없음: {filepath}")
    
    def prepare_features_for_prediction(self, X_seq):
        """시퀀스를 예측용 특징으로 변환 (학습 코드와 동일)"""
        # 마지막 시점 특징
        last_features = X_seq[:, -1, :]
        
        # 통계 특징
        mean_features = np.mean(X_seq, axis=1)
        std_features = np.std(X_seq, axis=1)
        max_features = np.max(X_seq, axis=1)
        min_features = np.min(X_seq, axis=1)
        
        # 추세 특징
        trend_features = X_seq[:, -1, :] - X_seq[:, 0, :]
        
        # 모든 특징 결합
        features = np.hstack([
            last_features,
            mean_features, 
            std_features,
            max_features,
            min_features,
            trend_features
        ])
        
        return features
    
    def get_risk_classification(self, value):
        """위험도 분류"""
        if value < 220:
            return {
                'level': 'SAFE',
                'korean': '안정', 
                'color': 'GREEN',
                'probability': max(0.1, min(0.3, (220 - value) / 220))
            }
        elif value < 300:
            return {
                'level': 'WARNING',
                'korean': '주의',
                'color': 'YELLOW', 
                'probability': max(0.3, min(0.7, (value - 220) / 80))
            }
        else:
            return {
                'level': 'DANGER',
                'korean': '위험',
                'color': 'RED',
                'probability': max(0.7, min(0.99, 0.7 + (value - 300) / 100))
            }
    
    def predict_sequence(self, df, start_idx):
        """단일 시퀀스 예측"""
        seq_len = 20
        
        if start_idx < seq_len:
            return None
            
        # 특징 컬럼 선택 (학습 시와 동일)
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime', 'range_class', 'is_jump', 'trend_pattern']]
        
        # 시퀀스 생성 (과거 20분)
        sequence = df[feature_cols].iloc[start_idx-seq_len:start_idx].values
        sequence = sequence.reshape(1, seq_len, -1)
        
        # 특징 준비
        features = self.prepare_features_for_prediction(sequence)
        
        # 예측 수행
        predictions = {}
        
        # 1. 점프 감지
        if 'model_jump' in self.models:
            jump_prob = self.models['model_jump'].predict_proba(features)[0, 1]
            jump_pred = self.models['model_jump'].predict(features)[0]
            predictions['jump'] = {
                'probability': jump_prob,
                'prediction': bool(jump_pred),
                'confidence': 'HIGH' if jump_prob > 0.7 else 'MEDIUM' if jump_prob > 0.3 else 'LOW'
            }
        
        # 2. 3구간 분류  
        if 'model_range' in self.models:
            range_prob = self.models['model_range'].predict_proba(features)[0]
            range_pred = self.models['model_range'].predict(features)[0]
            predictions['range'] = {
                'prediction': int(range_pred),
                'probabilities': {
                    'low_risk': range_prob[0],
                    'medium_risk': range_prob[1], 
                    'high_risk': range_prob[2]
                },
                'class_names': ['저위험(<150)', '중위험(150-299)', '고위험(300+)']
            }
        
        # 3. 상승/하락 패턴
        if 'model_trend' in self.models:
            trend_prob = self.models['model_trend'].predict_proba(features)[0]
            trend_pred = self.models['model_trend'].predict(features)[0]
            predictions['trend'] = {
                'prediction': int(trend_pred),
                'probabilities': {
                    'down': trend_prob[0],
                    'stable': trend_prob[1],
                    'gradual_up': trend_prob[2] if len(trend_prob) > 2 else 0,
                    'rapid_up': trend_prob[3] if len(trend_prob) > 3 else 0
                },
                'class_names': ['하락', '안정', '점진상승', '급상승']
            }
        
        # 4. 값 예측
        if 'model_value' in self.models:
            value_pred = self.models['model_value'].predict(features)[0]
            
            # 위험도 분류
            risk_info = self.get_risk_classification(value_pred)
            
            predictions['value'] = {
                'prediction': value_pred,
                'risk_level': risk_info['level'],
                'risk_korean': risk_info['korean'],
                'risk_color': risk_info['color'],
                'risk_probability': risk_info['probability']
            }
        
        # 현재 상태 정보
        current_value = df[df.columns[1]].iloc[start_idx-1]  # CURRENT_M16A_3F_JOB_2
        predictions['current'] = {
            'value': current_value,
            'datetime': df['datetime'].iloc[start_idx-1],
            'risk_info': self.get_risk_classification(current_value)
        }
        
        return predictions

# ==============================================================================
# 🎯 실시간 예측 실행기
# ==============================================================================

class RealTimePredictionRunner:
    """실시간 예측 실행 및 결과 표시"""
    
    def __init__(self):
        self.processor = HubRoomPredictionProcessor()
        self.predictor = HUBROOMPredictionSystem()
    
    def run_prediction(self, file_path='data/HUBROOM_PIVOT_DATA.csv', num_predictions=10):
        """예측 실행"""
        print(f"\n🚀 실시간 예측 시작 (최근 {num_predictions}개 시점)")
        print("="*80)
        
        # 1. 데이터 로드 및 처리
        df = self.processor.load_and_process_data(file_path)
        df = self.processor.create_prediction_features(df)
        
        # 2. 최근 데이터 예측
        results = []
        start_idx = len(df) - num_predictions
        
        print(f"\n📊 예측 수행 중...")
        for i in range(num_predictions):
            idx = start_idx + i
            if idx >= 20:  # 최소 20개 시점 필요
                result = self.predictor.predict_sequence(df, idx)
                if result:
                    result['index'] = idx
                    result['sequence_end'] = df['datetime'].iloc[idx-1]
                    results.append(result)
        
        # 3. 결과 출력
        self.display_results(results)
        
        return results
    
    def display_results(self, results):
        """결과 표시"""
        print("\n" + "="*100)
        print("📈 HUBROOM V4 Ultimate 예측 결과")
        print("="*100)
        
        for i, result in enumerate(results, 1):
            print(f"\n[예측 {i}] 시간: {result['sequence_end']}")
            print("-" * 60)
            
            # 현재 상태
            current = result['current']
            print(f"🔸 현재값: {current['value']:.1f} ({current['risk_info']['korean']})")
            
            # 값 예측  
            if 'value' in result:
                value_pred = result['value']
                print(f"🔹 예측값: {value_pred['prediction']:.1f}")
                print(f"🔹 위험도: {value_pred['risk_korean']} ({value_pred['risk_level']})")
                print(f"🔹 위험확률: {value_pred['risk_probability']:.1%}")
                
                # 구간별 분류
                if value_pred['prediction'] < 220:
                    status = "✅ 안정 구간"
                elif value_pred['prediction'] < 300:
                    status = "⚠️ 주의 구간" 
                else:
                    status = "🚨 위험 구간"
                print(f"🔹 상태: {status}")
            
            # 점프 위험
            if 'jump' in result:
                jump = result['jump']
                jump_risk = "🚨 점프 위험" if jump['prediction'] else "✅ 점프 안전"
                print(f"🔹 점프 예측: {jump_risk} (확률: {jump['probability']:.1%})")
            
            # 3구간 분류
            if 'range' in result:
                range_pred = result['range']
                class_name = range_pred['class_names'][range_pred['prediction']]
                probabilities = range_pred['probabilities']
                print(f"🔹 위험 분류: {class_name}")
                print(f"   - 저위험: {probabilities['low_risk']:.1%}")
                print(f"   - 중위험: {probabilities['medium_risk']:.1%}")
                print(f"   - 고위험: {probabilities['high_risk']:.1%}")
            
            # 추세 분석
            if 'trend' in result:
                trend = result['trend']
                trend_name = trend['class_names'][trend['prediction']]
                print(f"🔹 추세 패턴: {trend_name}")
        
        # 전체 요약
        print("\n" + "="*100)
        print("📋 전체 요약")
        print("="*100)
        
        # 위험 구간 통계
        danger_count = sum(1 for r in results if r.get('value', {}).get('prediction', 0) >= 300)
        warning_count = sum(1 for r in results if 220 <= r.get('value', {}).get('prediction', 0) < 300)
        safe_count = sum(1 for r in results if r.get('value', {}).get('prediction', 0) < 220)
        
        print(f"🔸 안정 구간 (< 220): {safe_count}개 ({safe_count/len(results):.1%})")
        print(f"🔸 주의 구간 (220-299): {warning_count}개 ({warning_count/len(results):.1%})")  
        print(f"🔸 위험 구간 (≥ 300): {danger_count}개 ({danger_count/len(results):.1%})")
        
        # 점프 위험 통계
        jump_risk_count = sum(1 for r in results if r.get('jump', {}).get('prediction', False))
        print(f"🔸 점프 위험: {jump_risk_count}개 ({jump_risk_count/len(results):.1%})")
        
        # 권장 조치
        if danger_count > 0:
            print(f"\n🚨 권장 조치: M16 6F AI MAXCAPA 50% DOWN 즉시 실행")
        elif warning_count > len(results) * 0.5:
            print(f"\n⚠️ 권장 조치: 모니터링 강화 및 예방 조치 준비")
        else:
            print(f"\n✅ 상태: 정상 범위 내 운영 중")

# ==============================================================================
# 🚀 메인 실행 함수
# ==============================================================================

def main():
    """메인 실행 함수"""
    try:
        # 예측 실행기 초기화
        runner = RealTimePredictionRunner()
        
        # 예측 수행
        results = runner.run_prediction(
            file_path='data/HUBROOM_PIVOT_DATA.csv',
            num_predictions=10
        )
        
        print(f"\n✅ 예측 완료! 총 {len(results)}개 결과 생성")
        
        # 개별 예측 상세 정보 (옵션)
        detail_view = input("\n상세 예측 정보를 보시겠습니까? (y/n): ")
        if detail_view.lower() == 'y':
            for i, result in enumerate(results, 1):
                print(f"\n[상세 {i}] ===================")
                print(f"예측값: {result.get('value', {}).get('prediction', 'N/A'):.1f}")
                print(f"점프확률: {result.get('jump', {}).get('probability', 0):.1%}")
                print(f"위험확률: {result.get('value', {}).get('risk_probability', 0):.1%}")
        
    except FileNotFoundError:
        print("❌ 오류: data/HUBROOM_PIVOT_DATA.csv 파일을 찾을 수 없습니다.")
        print("파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()