"""
제조공정 병목 예측 시스템 (LLM 제외 버전)
Python 3.11.4 호환
MCS 로그 + 진동센서 데이터 기반 병목 예측
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional
import json
import pickle
from pathlib import Path

# 기본 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# 벡터 저장을 위한 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 시계열 분석
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# LSTM 시뮬레이션을 위한 간단한 구현
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# ===== 1. MCS 로그 + 진동센서 샘플 데이터 생성 =====
def generate_sample_data(n_days=30, interval_minutes=5):
    """
    MCS 로그 + 진동센서 샘플 데이터 생성
    - 30일간의 데이터
    - 5분 간격으로 수집
    - 3개의 생산 라인 (LINE_A, LINE_B, LINE_C)
    - 진동센서 데이터 추가
    """
    
    print("MCS 로그 + 진동센서 샘플 데이터 생성 중...")
    
    # 시간 범위 설정
    start_date = datetime.now() - timedelta(days=n_days)
    end_date = datetime.now()
    
    # 타임스탬프 생성
    timestamps = pd.date_range(start=start_date, end=end_date, freq=f'{interval_minutes}min')
    
    # 라인 정보
    lines = ['LINE_A', 'LINE_B', 'LINE_C']
    
    data = []
    
    for timestamp in timestamps:
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        
        for line_id in lines:
            # 기본 정상 값
            normal_throughput = 100  # 개/시간
            normal_cycle_time = 5    # 분
            normal_wait_time = 10    # 분
            normal_utilization = 95  # %
            normal_vibration = 2.5   # mm/s (진동 속도)
            normal_temperature = 65  # °C
            
            # 시간대별 병목 발생 패턴
            if (9 <= hour <= 11) or (14 <= hour <= 16):
                bottleneck_prob = 0.35
            elif 12 <= hour <= 13:
                bottleneck_prob = 0.15
            elif hour >= 22 or hour <= 6:
                bottleneck_prob = 0.05
            else:
                bottleneck_prob = 0.1
            
            # 주말은 병목 가능성 낮음
            if day_of_week in [5, 6]:
                bottleneck_prob *= 0.5
            
            # 특정 라인의 고장 패턴 시뮬레이션
            if line_id == 'LINE_B' and hour in [10, 15]:
                bottleneck_prob += 0.2
            
            # 병목 상황 결정
            is_bottleneck = random.random() < bottleneck_prob
            
            if is_bottleneck:
                # 병목 상황 데이터
                throughput = normal_throughput * random.uniform(0.5, 0.75)
                cycle_time = normal_cycle_time * random.uniform(1.2, 1.8)
                wait_time = normal_wait_time * random.uniform(1.5, 2.5)
                utilization = normal_utilization * random.uniform(0.6, 0.8)
                error_count = random.randint(2, 8)
                downtime_minutes = random.uniform(5, 30)
                
                # 진동센서 데이터 (병목시 증가)
                vibration_x = normal_vibration * random.uniform(1.5, 3.0)
                vibration_y = normal_vibration * random.uniform(1.5, 3.0)
                vibration_z = normal_vibration * random.uniform(1.5, 3.0)
                temperature = normal_temperature * random.uniform(1.1, 1.3)
                noise_level = random.uniform(85, 95)  # dB
            else:
                # 정상 상황 데이터
                throughput = normal_throughput * random.uniform(0.95, 1.05)
                cycle_time = normal_cycle_time * random.uniform(0.95, 1.05)
                wait_time = normal_wait_time * random.uniform(0.9, 1.1)
                utilization = normal_utilization * random.uniform(0.95, 1.0)
                error_count = random.randint(0, 2)
                downtime_minutes = random.uniform(0, 5)
                
                # 진동센서 데이터 (정상)
                vibration_x = normal_vibration * random.uniform(0.8, 1.2)
                vibration_y = normal_vibration * random.uniform(0.8, 1.2)
                vibration_z = normal_vibration * random.uniform(0.8, 1.2)
                temperature = normal_temperature * random.uniform(0.95, 1.05)
                noise_level = random.uniform(70, 80)  # dB
            
            # 데이터 저장
            data.append({
                'timestamp': timestamp,
                'line_id': line_id,
                'throughput': round(throughput, 2),
                'cycle_time': round(cycle_time, 2),
                'wait_time': round(wait_time, 2),
                'utilization': round(utilization, 2),
                'error_count': error_count,
                'downtime_minutes': round(downtime_minutes, 2),
                'vibration_x': round(vibration_x, 3),
                'vibration_y': round(vibration_y, 3),
                'vibration_z': round(vibration_z, 3),
                'vibration_rms': round(np.sqrt(vibration_x**2 + vibration_y**2 + vibration_z**2), 3),
                'temperature': round(temperature, 1),
                'noise_level': round(noise_level, 1),
                'is_bottleneck': int(is_bottleneck),
                'shift': 'day' if 8 <= hour <= 16 else 'evening' if 16 < hour <= 24 else 'night'
            })
    
    df = pd.DataFrame(data)
    print(f"생성된 데이터: {len(df)} 행")
    print(f"병목 비율: {df['is_bottleneck'].mean():.2%}")
    
    return df

# ===== 2. 병목 위험도 지수(BRI) 계산 (진동센서 포함) =====
def calculate_bri(row, weights=None):
    """
    병목 위험도 지수(Bottleneck Risk Index) 계산
    진동센서 데이터 포함
    """
    if weights is None:
        weights = {
            'utilization': 0.25,
            'wait_time': 0.20,
            'cycle_time': 0.20,
            'throughput': 0.15,
            'vibration': 0.15,
            'temperature': 0.05
        }
    
    # 각 지수 계산 (정규화)
    U = (100 - row['utilization']) / 100
    W = row['wait_time'] / 10
    C = row['cycle_time'] / 5
    T = 100 / max(row['throughput'], 1)
    V = row['vibration_rms'] / 10  # 10mm/s 기준
    Temp = (row['temperature'] - 50) / 50  # 50-100°C 범위
    
    # BRI 계산
    bri = (weights['utilization'] * U + 
           weights['wait_time'] * W + 
           weights['cycle_time'] * C + 
           weights['throughput'] * T +
           weights['vibration'] * V +
           weights['temperature'] * Temp)
    
    return bri

# ===== 3. 데이터 전처리 (진동센서 포함) =====
class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess(self, df):
        """데이터 전처리 수행"""
        print("\n데이터 전처리 시작...")
        
        # 1. 이상치 제거 (IQR 방법)
        df_clean = self.remove_outliers(df)
        
        # 2. 시간 특징 추가
        df_clean = self.add_time_features(df_clean)
        
        # 3. BRI 계산
        df_clean['bri'] = df_clean.apply(calculate_bri, axis=1)
        
        # 4. 경로별 집계 특징 추가
        df_clean = self.add_path_features(df_clean)
        
        # 5. 진동센서 특징 추가
        df_clean = self.add_vibration_features(df_clean)
        
        print(f"전처리 완료: {len(df_clean)} 행")
        
        return df_clean
    
    def remove_outliers(self, df, columns=['throughput', 'cycle_time', 'wait_time', 'vibration_rms']):
        """IQR 방법으로 이상치 제거"""
        df_clean = df.copy()
        
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 이상치 제거
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        print(f"이상치 제거: {len(df) - len(df_clean)} 행 제거됨")
        
        return df_clean
    
    def add_time_features(self, df):
        """시간 관련 특징 추가"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        
        # 시간대별 구분
        df['time_period'] = pd.cut(df['hour'], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['dawn', 'morning', 'afternoon', 'night'])
        
        return df
    
    def add_path_features(self, df):
        """경로별 집계 특징 추가"""
        # 라인별 1시간 이동평균
        for col in ['throughput', 'wait_time', 'utilization', 'vibration_rms', 'temperature']:
            df[f'{col}_ma_1h'] = df.groupby('line_id')[col].transform(
                lambda x: x.rolling('1H', on=df['timestamp']).mean()
            )
        
        # 라인별 누적 에러 수
        df['cumulative_errors'] = df.groupby('line_id')['error_count'].cumsum()
        
        return df
    
    def add_vibration_features(self, df):
        """진동센서 특징 추가"""
        # 진동 변화율
        df['vibration_change_rate'] = df.groupby('line_id')['vibration_rms'].pct_change()
        
        # 진동 표준편차 (1시간 윈도우)
        df['vibration_std_1h'] = df.groupby('line_id')['vibration_rms'].transform(
            lambda x: x.rolling('1H', on=df['timestamp']).std()
        )
        
        # 온도 변화율
        df['temp_change_rate'] = df.groupby('line_id')['temperature'].pct_change()
        
        return df

# ===== 4. 시계열 모델 구현 =====
class TimeSeriesModels:
    """시계열 모델 (ARIMA, LSTM 시뮬레이션, RNN 시뮬레이션)"""
    
    def __init__(self):
        self.arima_models = {}
        self.lstm_simulator = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rnn_simulator = LinearRegression()
        
    def train_arima(self, df, target_col='bri'):
        """ARIMA 모델 학습"""
        print("\nARIMA 모델 학습 중...")
        
        for line_id in df['line_id'].unique():
            line_data = df[df['line_id'] == line_id].set_index('timestamp')[target_col]
            
            # 정상성 검정
            adf_result = adfuller(line_data)
            print(f"{line_id} ADF 통계량: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
            
            # ARIMA 모델 학습
            try:
                model = ARIMA(line_data, order=(2, 1, 2))
                fitted_model = model.fit()
                self.arima_models[line_id] = fitted_model
                print(f"{line_id} ARIMA 모델 학습 완료")
            except Exception as e:
                print(f"{line_id} ARIMA 학습 실패: {e}")
    
    def train_lstm_simulator(self, X_train, y_train):
        """LSTM 시뮬레이터 학습 (RandomForest로 대체)"""
        print("\nLSTM 시뮬레이터 학습 중...")
        self.lstm_simulator.fit(X_train, y_train)
        print("LSTM 시뮬레이터 학습 완료!")
    
    def train_rnn_simulator(self, X_train, y_train):
        """RNN 시뮬레이터 학습 (Linear Regression으로 대체)"""
        print("\nRNN 시뮬레이터 학습 중...")
        self.rnn_simulator.fit(X_train, y_train)
        print("RNN 시뮬레이터 학습 완료!")
    
    def predict_arima(self, line_id, steps=24):
        """ARIMA 예측 (24시간)"""
        if line_id not in self.arima_models:
            return None
        
        forecast = self.arima_models[line_id].forecast(steps=steps)
        return forecast
    
    def create_sequences(self, data, sequence_length=24):
        """시계열 데이터를 시퀀스로 변환"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data[i:i+sequence_length]
            target = data[i+sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

# ===== 5. XGBoost 모델 (1차 필터) =====
class XGBoostBottleneckDetector:
    """XGBoost 기반 실시간 병목 감지"""
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """모델 학습용 특징 준비"""
        feature_cols = [
            'throughput', 'cycle_time', 'wait_time', 'utilization',
            'error_count', 'downtime_minutes', 'bri',
            'hour', 'day_of_week', 'is_weekend',
            'throughput_ma_1h', 'wait_time_ma_1h', 'utilization_ma_1h',
            'cumulative_errors',
            'vibration_rms', 'temperature', 'noise_level',
            'vibration_x', 'vibration_y', 'vibration_z',
            'vibration_rms_ma_1h', 'temperature_ma_1h',
            'vibration_change_rate', 'vibration_std_1h', 'temp_change_rate'
        ]
        
        # NaN 값 처리
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        
        self.feature_cols = [col for col in feature_cols if col in df.columns]
        
        return df[self.feature_cols]
    
    def train(self, X_train, y_train):
        """모델 학습"""
        print("\nXGBoost 모델 학습 중...")
        
        # 데이터 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # XGBoost 파라미터
        params = {
            'objective': 'binary:logistic',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 150,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        # 모델 학습
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train_scaled, y_train)
        
        print("XGBoost 학습 완료!")
        
    def predict(self, X):
        """예측 수행 (0.01초 이내)"""
        X_scaled = self.scaler.transform(X)
        
        # 예측 확률
        proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # 위험도 레벨 분류
        risk_levels = []
        for p in proba:
            if p < 0.3:
                risk_levels.append('정상')
            elif p < 0.5:
                risk_levels.append('주의')
            elif p < 0.7:
                risk_levels.append('경고')
            else:
                risk_levels.append('위험')
        
        return proba, risk_levels
    
    def get_feature_importance(self):
        """특징 중요도 반환"""
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance

# ===== 6. RAG 시스템 (과거 사례 검색) =====
class RAGSystem:
    """과거 병목 사례 검색 시스템"""
    
    def __init__(self):
        self.case_database = []
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.vectors = None
        
    def build_case_database(self, df):
        """병목 사례 데이터베이스 구축"""
        print("\nRAG 사례 데이터베이스 구축 중...")
        
        bottleneck_cases = df[df['is_bottleneck'] == 1]
        
        for _, row in bottleneck_cases.iterrows():
            # 사례 설명 생성
            case_description = f"""
            라인: {row['line_id']}
            시간: {row['timestamp']}
            처리량: {row['throughput']:.1f}
            대기시간: {row['wait_time']:.1f}분
            가동률: {row['utilization']:.1f}%
            에러수: {row['error_count']}
            진동: {row['vibration_rms']:.2f}mm/s
            온도: {row['temperature']:.1f}°C
            """
            
            # 해결 방안 시뮬레이션
            if row['vibration_rms'] > 7.5:
                solution = "베어링 교체 및 축 정렬 점검"
                root_cause = "과도한 진동 - 기계적 마모"
            elif row['temperature'] > 80:
                solution = "냉각 시스템 점검 및 윤활유 교체"
                root_cause = "과열 - 냉각 시스템 이상"
            elif row['error_count'] > 5:
                solution = "설비 점검 및 재시작 필요"
                root_cause = "설비 오류 누적"
            elif row['wait_time'] > 20:
                solution = "버퍼 용량 증설 또는 라인 밸런싱"
                root_cause = "공정간 불균형"
            elif row['utilization'] < 70:
                solution = "예방 정비 실시"
                root_cause = "설비 노후화"
            else:
                solution = "생산 일정 조정"
                root_cause = "과부하"
            
            self.case_database.append({
                'description': case_description,
                'solution': solution,
                'root_cause': root_cause,
                'line_id': row['line_id'],
                'bri': row['bri'],
                'vibration': row['vibration_rms'],
                'temperature': row['temperature']
            })
        
        # 벡터화
        descriptions = [case['description'] for case in self.case_database]
        self.vectors = self.vectorizer.fit_transform(descriptions)
        
        print(f"RAG 데이터베이스 구축 완료: {len(self.case_database)}개 사례")
    
    def search_similar_cases(self, current_situation, top_k=5):
        """유사 사례 검색"""
        # 현재 상황 설명
        query = f"""
        처리량: {current_situation['throughput']:.1f}
        대기시간: {current_situation['wait_time']:.1f}분
        가동률: {current_situation['utilization']:.1f}%
        에러수: {current_situation['error_count']}
        진동: {current_situation['vibration_rms']:.2f}mm/s
        온도: {current_situation['temperature']:.1f}°C
        """
        
        # 벡터화 및 유사도 계산
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # 상위 k개 사례 선택
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        similar_cases = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # 최소 유사도
                case = self.case_database[idx].copy()
                case['similarity'] = similarities[idx]
                similar_cases.append(case)
        
        return similar_cases

# ===== 7. 규칙 기반 종합 판단 시스템 (LLM 대체) =====
class RuleBasedDiagnostics:
    """규칙 기반 병목 진단 시스템"""
    
    def __init__(self):
        self.diagnosis_rules = self._create_diagnosis_rules()
    
    def _create_diagnosis_rules(self):
        """진단 규칙 정의"""
        return {
            'high_vibration': {
                'condition': lambda d: d['vibration_rms'] > 7.5,
                'root_cause': '과도한 진동 - 베어링 마모 또는 축 정렬 불량',
                'immediate_action': '진동 측정 및 베어링 상태 점검',
                'long_term_solution': '베어링 교체 및 정기적인 축 정렬 점검 프로그램 도입',
                'priority': 1
            },
            'high_temperature': {
                'condition': lambda d: d['temperature'] > 80,
                'root_cause': '과열 - 냉각 시스템 이상 또는 과부하',
                'immediate_action': '부하 감소 및 냉각 시스템 점검',
                'long_term_solution': '냉각 시스템 업그레이드 및 온도 모니터링 강화',
                'priority': 1
            },
            'high_error_count': {
                'condition': lambda d: d['error_count'] > 5,
                'root_cause': '제어 시스템 오류 또는 센서 이상',
                'immediate_action': '시스템 재시작 및 오류 로그 분석',
                'long_term_solution': '제어 시스템 펌웨어 업데이트 및 센서 교체',
                'priority': 2
            },
            'high_wait_time': {
                'condition': lambda d: d['wait_time'] > 20,
                'root_cause': '공정간 처리 속도 불균형',
                'immediate_action': '병목 공정 확인 및 버퍼 조정',
                'long_term_solution': '라인 밸런싱 재설계 및 버퍼 용량 최적화',
                'priority': 3
            },
            'low_utilization': {
                'condition': lambda d: d['utilization'] < 70,
                'root_cause': '설비 효율 저하 - 유지보수 필요',
                'immediate_action': '설비 상태 점검 및 간단한 정비',
                'long_term_solution': '예방 정비 계획 수립 및 설비 현대화',
                'priority': 3
            },
            'complex_issue': {
                'condition': lambda d: d['bri'] > 0.7,
                'root_cause': '복합적 문제 - 다각도 분석 필요',
                'immediate_action': '전체 라인 긴급 점검',
                'long_term_solution': '통합 모니터링 시스템 구축 및 AI 기반 예측 정비',
                'priority': 1
            }
        }
    
    def diagnose(self, current_data, predictions):
        """규칙 기반 진단"""
        diagnosis = {
            'bottleneck_probability': 0,
            'root_causes': [],
            'immediate_actions': [],
            'long_term_solutions': [],
            'risk_score': 0
        }
        
        # 각 모델의 예측 결과 종합
        xgb_prob = predictions.get('xgboost_probability', 0.5)
        arima_trend = predictions.get('arima_trend', 0)
        pattern_similarity = predictions.get('pattern_similarity', 0.5)
        
        # 가중 평균으로 최종 확률 계산
        weights = {'xgb': 0.5, 'arima': 0.2, 'pattern': 0.3}
        final_prob = (weights['xgb'] * xgb_prob + 
                     weights['arima'] * arima_trend + 
                     weights['pattern'] * pattern_similarity)
        
        diagnosis['bottleneck_probability'] = final_prob
        
        # 규칙 기반 진단 수행
        triggered_rules = []
        for rule_name, rule in self.diagnosis_rules.items():
            if rule['condition'](current_data):
                triggered_rules.append({
                    'name': rule_name,
                    'priority': rule['priority'],
                    'root_cause': rule['root_cause'],
                    'immediate_action': rule['immediate_action'],
                    'long_term_solution': rule['long_term_solution']
                })
        
        # 우선순위별로 정렬
        triggered_rules.sort(key=lambda x: x['priority'])
        
        # 진단 결과 정리
        if triggered_rules:
            for rule in triggered_rules[:3]:  # 상위 3개만
                diagnosis['root_causes'].append(rule['root_cause'])
                diagnosis['immediate_actions'].append(rule['immediate_action'])
                diagnosis['long_term_solutions'].append(rule['long_term_solution'])
            
            # 위험도 점수 계산
            diagnosis['risk_score'] = min(len(triggered_rules) * 0.25 + final_prob, 1.0)
        else:
            diagnosis['root_causes'] = ['정상 운영 상태']
            diagnosis['immediate_actions'] = ['지속적인 모니터링']
            diagnosis['long_term_solutions'] = ['현재 운영 방식 유지']
            diagnosis['risk_score'] = final_prob
        
        return diagnosis

# ===== 8. 통합 예측 시스템 =====
class BottleneckPredictionSystem:
    """병목 예측 통합 시스템 (LLM 없음)"""
    
    def __init__(self):
        self.xgboost_model = XGBoostBottleneckDetector()
        self.time_series_models = TimeSeriesModels()
        self.rag_system = RAGSystem()
        self.preprocessor = DataPreprocessor()
        self.diagnostics = RuleBasedDiagnostics()
        
    def train(self, df):
        """전체 시스템 학습"""
        print("\n=== 병목 예측 시스템 학습 시작 ===")
        
        # 데이터 전처리
        df_processed = self.preprocessor.preprocess(df)
        
        # 학습/테스트 데이터 분할
        train_df = df_processed[df_processed['timestamp'] < df_processed['timestamp'].max() - timedelta(days=7)]
        test_df = df_processed[df_processed['timestamp'] >= df_processed['timestamp'].max() - timedelta(days=7)]
        
        # 1. XGBoost 학습
        X_train = self.xgboost_model.prepare_features(train_df)
        y_train = train_df['is_bottleneck']
        self.xgboost_model.train(X_train, y_train)
        
        # 2. 시계열 모델 학습
        # ARIMA
        self.time_series_models.train_arima(train_df, target_col='bri')
        
        # LSTM/RNN 시뮬레이터 학습
        # 시퀀스 데이터 생성
        for line_id in train_df['line_id'].unique():
            line_data = train_df[train_df['line_id'] == line_id].sort_values('timestamp')
            if len(line_data) > 48:  # 최소 48개 데이터 필요
                sequences, targets = self.time_series_models.create_sequences(
                    line_data['bri'].values, sequence_length=24
                )
                if len(sequences) > 0:
                    # LSTM 시뮬레이터
                    self.time_series_models.train_lstm_simulator(
                        sequences.reshape(sequences.shape[0], -1), targets
                    )
                    # RNN 시뮬레이터
                    self.time_series_models.train_rnn_simulator(
                        sequences.reshape(sequences.shape[0], -1), targets
                    )
                    break  # 첫 번째 라인만 학습 (데모용)
        
        # 3. RAG 데이터베이스 구축
        self.rag_system.build_case_database(train_df)
        
        print("\n=== 시스템 학습 완료 ===")
        
        return test_df
    
    def predict(self, current_data):
        """통합 예측 수행"""
        # 1차 필터: XGBoost
        X = self.xgboost_model.prepare_features(current_data)
        xgb_proba, risk_levels = self.xgboost_model.predict(X)
        
        results = []
        
        for idx, (_, row) in enumerate(current_data.iterrows()):
            result = {
                'timestamp': row['timestamp'],
                'line_id': row['line_id'],
                'xgboost_probability': xgb_proba[idx],
                'risk_level': risk_levels[idx],
                'bri': row['bri']
            }
            
            # 2차 분석: 위험도 50% 이상일 때
            if xgb_proba[idx] > 0.5:
                # ARIMA 예측
                try:
                    arima_forecast = self.time_series_models.predict_arima(row['line_id'], steps=24)
                    if arima_forecast is not None:
                        result['arima_trend'] = float(arima_forecast.mean() > row['bri'])
                    else:
                        result['arima_trend'] = 0.5
                except:
                    result['arima_trend'] = 0.5
                
                # RAG 검색
                similar_cases = self.rag_system.search_similar_cases(row, top_k=3)
                result['similar_cases'] = similar_cases
                
                # 패턴 유사도 계산
                if similar_cases:
                    result['pattern_similarity'] = np.mean([case['similarity'] for case in similar_cases[:3]])
                else:
                    result['pattern_similarity'] = 0.5
                
                # 규칙 기반 진단
                predictions_dict = {
                    'xgboost_probability': xgb_proba[idx],
                    'arima_trend': result.get('arima_trend', 0.5),
                    'pattern_similarity': result.get('pattern_similarity', 0.5)
                }
                
                result['diagnosis'] = self.diagnostics.diagnose(row, predictions_dict)
            
            results.append(result)
        
        return pd.DataFrame(results)

# ===== 9. 실행 예제 =====
def main():
    """메인 실행 함수"""
    print("=== 제조공정 병목 예측 시스템 (LLM 제외) ===")
    print("Python 3.11.4 호환 버전")
    print("="*50)
    
    # 1. 샘플 데이터 생성
    df = generate_sample_data(n_days=30, interval_minutes=5)
    
    # 2. 데이터 미리보기
    print("\n데이터 미리보기:")
    print(df.head())
    print(f"\n데이터 shape: {df.shape}")
    print(f"\n컬럼 목록: {df.columns.tolist()}")
    
    # 3. 시스템 초기화 및 학습
    system = BottleneckPredictionSystem()
    test_df = system.train(df)
    
    # 4. 예측 수행
    print("\n=== 예측 수행 중 ===")
    
    # 최근 24시간 데이터로 예측
    recent_data = test_df.tail(24*12)  # 24시간 * (60분/5분)
    predictions = system.predict(recent_data)
    
    # 5. 결과 출력
    print("\n=== 예측 결과 ===")
    
    # 위험도가 높은 케이스만 출력
    high_risk = predictions[predictions['xgboost_probability'] > 0.5]
    
    if not high_risk.empty:
        print(f"\n⚠️  병목 위험 감지: {len(high_risk)}건")
        
        for _, pred in high_risk.iterrows():
            print(f"\n{'='*50}")
            print(f"라인: {pred['line_id']}")
            print(f"시간: {pred['timestamp']}")
            print(f"위험도: {pred['risk_level']} ({pred['xgboost_probability']:.1%})")
            print(f"BRI 지수: {pred['bri']:.2f}")
            
            if 'diagnosis' in pred and pred['diagnosis']:
                diag = pred['diagnosis']
                print(f"\n📊 종합 진단:")
                print(f"  - 병목 확률: {diag['bottleneck_probability']:.1%}")
                print(f"  - 위험도 점수: {diag['risk_score']:.2f}")
                
                print(f"\n🔍 근본 원인:")
                for i, cause in enumerate(diag['root_causes'], 1):
                    print(f"  {i}. {cause}")
                
                print(f"\n⚡ 즉시 조치:")
                for i, action in enumerate(diag['immediate_actions'], 1):
                    print(f"  {i}. {action}")
                
                print(f"\n🛠️ 장기 해결책:")
                for i, solution in enumerate(diag['long_term_solutions'], 1):
                    print(f"  {i}. {solution}")
                
                if 'similar_cases' in pred and pred['similar_cases']:
                    print(f"\n📚 유사 사례 (상위 3개):")
                    for i, case in enumerate(pred['similar_cases'][:3], 1):
                        print(f"  {i}. 유사도: {case['similarity']:.1%}")
                        print(f"     - 원인: {case['root_cause']}")
                        print(f"     - 해결: {case['solution']}")
    else:
        print("\n✅ 현재 모든 라인 정상 운영 중")
    
    # 6. 모델 성능 평가
    print("\n\n=== 모델 성능 평가 ===")
    
    # XGBoost 특징 중요도
    importance = system.xgboost_model.get_feature_importance()
    print("\nXGBoost 주요 특징 (상위 10개):")
    print(importance.head(10))
    
    # 전체 정확도
    X_test = system.xgboost_model.prepare_features(test_df)
    y_test = test_df['is_bottleneck']
    
    X_test_scaled = system.xgboost_model.scaler.transform(X_test)
    y_pred = system.xgboost_model.model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n모델 성능:")
    print(f"  - 정확도: {accuracy:.1%}")
    print(f"  - 정밀도: {precision:.1%}")
    print(f"  - 재현율: {recall:.1%}")
    print(f"  - F1 점수: {f1:.3f}")
    
    # 7. 시스템 상태 요약
    print("\n=== 시스템 상태 요약 ===")
    print(f"총 처리 데이터: {len(df):,} 행")
    print(f"학습 데이터: {len(df) - len(test_df):,} 행")
    print(f"테스트 데이터: {len(test_df):,} 행")
    print(f"RAG 사례 DB: {len(system.rag_system.case_database)} 건")
    print(f"ARIMA 모델: {len(system.time_series_models.arima_models)} 개")
    print("\n✅ 시스템 준비 완료! 실시간 예측 가능 상태입니다.")
    print("📌 참고: LLM 없이 규칙 기반 진단 시스템으로 작동 중")

if __name__ == "__main__":
    main()