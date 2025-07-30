"""
제조공정 병목 예측 시스템
Python 3.11.4 호환
MCS 로그 데이터 기반 병목 예측
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# 벡터 저장을 위한 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 시계열 분석
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# ===== 1. MCS 로그 샘플 데이터 생성 =====
def generate_mcs_sample_data(n_days=30, interval_minutes=5):
    """
    MCS 로그 샘플 데이터 생성
    - 30일간의 데이터
    - 5분 간격으로 수집
    - 3개의 생산 라인 (LINE_A, LINE_B, LINE_C)
    """
    
    print("MCS 로그 샘플 데이터 생성 중...")
    
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
            
            # 시간대별 병목 발생 패턴
            # 오전 9-11시, 오후 2-4시에 병목 가능성 높음
            if (9 <= hour <= 11) or (14 <= hour <= 16):
                bottleneck_prob = 0.35
            # 점심시간
            elif 12 <= hour <= 13:
                bottleneck_prob = 0.15
            # 야간
            elif hour >= 22 or hour <= 6:
                bottleneck_prob = 0.05
            else:
                bottleneck_prob = 0.1
            
            # 주말은 병목 가능성 낮음
            if day_of_week in [5, 6]:  # 토, 일
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
            else:
                # 정상 상황 데이터 (약간의 변동 포함)
                throughput = normal_throughput * random.uniform(0.95, 1.05)
                cycle_time = normal_cycle_time * random.uniform(0.95, 1.05)
                wait_time = normal_wait_time * random.uniform(0.9, 1.1)
                utilization = normal_utilization * random.uniform(0.95, 1.0)
                error_count = random.randint(0, 2)
                downtime_minutes = random.uniform(0, 5)
            
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
                'is_bottleneck': int(is_bottleneck),
                'shift': 'day' if 8 <= hour <= 16 else 'evening' if 16 < hour <= 24 else 'night'
            })
    
    df = pd.DataFrame(data)
    print(f"생성된 데이터: {len(df)} 행")
    print(f"병목 비율: {df['is_bottleneck'].mean():.2%}")
    
    return df

# ===== 2. 병목 위험도 지수(BRI) 계산 =====
def calculate_bri(row, weights=None):
    """
    병목 위험도 지수(Bottleneck Risk Index) 계산
    BRI = α1*U + α2*W + α3*C + α4*T
    """
    if weights is None:
        weights = {
            'utilization': 0.3,
            'wait_time': 0.25,
            'cycle_time': 0.25,
            'throughput': 0.2
        }
    
    # 각 지수 계산 (정규화)
    U = (100 - row['utilization']) / 100  # 가동률 지수 (낮을수록 위험)
    W = row['wait_time'] / 10  # 대기시간 지수 (10분 기준)
    C = row['cycle_time'] / 5  # 사이클타임 지수 (5분 기준)
    T = 100 / max(row['throughput'], 1)  # 처리량 지수 (역수)
    
    # BRI 계산
    bri = (weights['utilization'] * U + 
           weights['wait_time'] * W + 
           weights['cycle_time'] * C + 
           weights['throughput'] * T)
    
    return bri

# ===== 3. 데이터 전처리 =====
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
        
        print(f"전처리 완료: {len(df_clean)} 행")
        
        return df_clean
    
    def remove_outliers(self, df, columns=['throughput', 'cycle_time', 'wait_time']):
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
        for col in ['throughput', 'wait_time', 'utilization']:
            df[f'{col}_ma_1h'] = df.groupby('line_id')[col].transform(
                lambda x: x.rolling('1H', on=df['timestamp']).mean()
            )
        
        # 라인별 누적 에러 수
        df['cumulative_errors'] = df.groupby('line_id')['error_count'].cumsum()
        
        return df

# ===== 4. XGBoost 모델 (1차 필터) =====
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
            'cumulative_errors'
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
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
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

# ===== 5. GRU 모델 (2차 정밀분석) - 간단한 구현 =====
class SimpleGRUPredictor:
    """간단한 GRU 시뮬레이션 (실제로는 TensorFlow/PyTorch 사용)"""
    
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length  # 24시간 데이터 사용
        self.patterns = {}  # 패턴 저장
        
    def learn_patterns(self, df):
        """과거 패턴 학습"""
        print("\nGRU 패턴 학습 중...")
        
        # 라인별로 병목 패턴 학습
        for line_id in df['line_id'].unique():
            line_data = df[df['line_id'] == line_id].sort_values('timestamp')
            
            # 병목 발생 전 24시간 패턴 수집
            bottleneck_indices = line_data[line_data['is_bottleneck'] == 1].index
            
            patterns = []
            for idx in bottleneck_indices:
                if idx >= self.sequence_length:
                    pattern = line_data.loc[idx-self.sequence_length:idx-1, 
                                           ['throughput', 'wait_time', 'utilization']].values
                    patterns.append(pattern)
            
            if patterns:
                self.patterns[line_id] = np.array(patterns).mean(axis=0)
        
        print("GRU 패턴 학습 완료!")
    
    def predict_bottleneck_time(self, current_data, line_id):
        """병목 발생 시간 예측"""
        if line_id not in self.patterns:
            return None, 0.5
        
        # 현재 패턴과 학습된 패턴 비교 (간단한 유사도 계산)
        learned_pattern = self.patterns[line_id]
        
        if len(current_data) < self.sequence_length:
            return None, 0.5
        
        current_pattern = current_data[-self.sequence_length:][['throughput', 'wait_time', 'utilization']].values
        
        # 패턴 유사도 계산
        similarity = 1 - np.mean(np.abs(current_pattern - learned_pattern))
        
        # 예측 시간 (유사도가 높을수록 빨리 발생)
        if similarity > 0.8:
            predicted_hours = random.uniform(1, 3)
        elif similarity > 0.6:
            predicted_hours = random.uniform(3, 6)
        else:
            predicted_hours = None
        
        return predicted_hours, similarity

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
            """
            
            # 해결 방안 시뮬레이션
            if row['error_count'] > 5:
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
                'bri': calculate_bri(row)
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

# ===== 7. 통합 예측 시스템 =====
class BottleneckPredictionSystem:
    """병목 예측 통합 시스템"""
    
    def __init__(self):
        self.xgboost_model = XGBoostBottleneckDetector()
        self.gru_model = SimpleGRUPredictor()
        self.rag_system = RAGSystem()
        self.preprocessor = DataPreprocessor()
        
    def train(self, df):
        """전체 시스템 학습"""
        print("\n=== 병목 예측 시스템 학습 시작 ===")
        
        # 데이터 전처리
        df_processed = self.preprocessor.preprocess(df)
        
        # 학습/테스트 데이터 분할
        train_df = df_processed[df_processed['timestamp'] < df_processed['timestamp'].max() - timedelta(days=7)]
        test_df = df_processed[df_processed['timestamp'] >= df_processed['timestamp'].max() - timedelta(days=7)]
        
        # XGBoost 학습
        X_train = self.xgboost_model.prepare_features(train_df)
        y_train = train_df['is_bottleneck']
        self.xgboost_model.train(X_train, y_train)
        
        # GRU 패턴 학습
        self.gru_model.learn_patterns(train_df)
        
        # RAG 데이터베이스 구축
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
            
            # 2차 분석: 위험도 70% 이상일 때만
            if xgb_proba[idx] > 0.7:
                # GRU 예측
                line_data = current_data[current_data['line_id'] == row['line_id']]
                predicted_hours, pattern_similarity = self.gru_model.predict_bottleneck_time(
                    line_data, row['line_id']
                )
                
                result['gru_predicted_hours'] = predicted_hours
                result['pattern_similarity'] = pattern_similarity
                
                # RAG 검색
                similar_cases = self.rag_system.search_similar_cases(row, top_k=3)
                result['similar_cases'] = similar_cases
                
                # 최종 판단 (PHI-4 시뮬레이션)
                result['final_diagnosis'] = self._phi4_analysis(result, row)
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _phi4_analysis(self, prediction_result, current_data):
        """PHI-4 LLM 분석 시뮬레이션"""
        diagnosis = {
            'bottleneck_probability': 0,
            'root_cause': '',
            'immediate_action': '',
            'long_term_solution': ''
        }
        
        # 예측 결과 종합
        xgb_weight = 0.4
        gru_weight = 0.3
        rag_weight = 0.3
        
        # XGBoost 확률
        xgb_prob = prediction_result['xgboost_probability']
        
        # GRU 확률 (패턴 유사도 기반)
        gru_prob = prediction_result.get('pattern_similarity', 0.5)
        
        # RAG 확률 (유사 사례 기반)
        if prediction_result.get('similar_cases'):
            rag_prob = np.mean([case['similarity'] for case in prediction_result['similar_cases'][:3]])
        else:
            rag_prob = 0.5
        
        # 최종 확률 계산
        final_prob = (xgb_weight * xgb_prob + 
                     gru_weight * gru_prob + 
                     rag_weight * rag_prob)
        
        diagnosis['bottleneck_probability'] = final_prob
        
        # 근본 원인 분석
        if current_data['error_count'] > 5:
            diagnosis['root_cause'] = "설비 오류 누적으로 인한 성능 저하"
            diagnosis['immediate_action'] = "해당 라인 긴급 점검 및 에러 로그 확인"
            diagnosis['long_term_solution'] = "예방 정비 주기 단축 및 모니터링 시스템 강화"
        elif current_data['wait_time'] > 20:
            diagnosis['root_cause'] = "공정간 처리 속도 불균형"
            diagnosis['immediate_action'] = "대체 라인으로 생산 분산"
            diagnosis['long_term_solution'] = "라인 밸런싱 재설계 및 버퍼 용량 증대"
        elif current_data['utilization'] < 70:
            diagnosis['root_cause'] = "설비 노후화로 인한 효율 저하"
            diagnosis['immediate_action'] = "가동 속도 조정 및 부분 정비"
            diagnosis['long_term_solution'] = "핵심 부품 교체 계획 수립"
        else:
            diagnosis['root_cause'] = "일시적 과부하"
            diagnosis['immediate_action'] = "생산 일정 조정"
            diagnosis['long_term_solution'] = "수요 예측 정확도 향상"
        
        return diagnosis

# ===== 8. 실행 예제 =====
def main():
    """메인 실행 함수"""
    print("=== 제조공정 병목 예측 시스템 ===")
    print("Python 3.11.4 호환 버전")
    print("="*40)
    
    # 1. 샘플 데이터 생성
    df = generate_mcs_sample_data(n_days=30, interval_minutes=5)
    
    # 2. 데이터 미리보기
    print("\n데이터 미리보기:")
    print(df.head())
    print(f"\n데이터 shape: {df.shape}")
    
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
    high_risk = predictions[predictions['xgboost_probability'] > 0.7]
    
    if not high_risk.empty:
        print(f"\n⚠️  고위험 병목 감지: {len(high_risk)}건")
        
        for _, pred in high_risk.iterrows():
            print(f"\n라인: {pred['line_id']}")
            print(f"시간: {pred['timestamp']}")
            print(f"위험도: {pred['risk_level']} ({pred['xgboost_probability']:.1%})")
            print(f"BRI 지수: {pred['bri']:.2f}")
            
            if 'final_diagnosis' in pred and pred['final_diagnosis']:
                diag = pred['final_diagnosis']
                print(f"\n📊 최종 진단:")
                print(f"  - 병목 확률: {diag['bottleneck_probability']:.1%}")
                print(f"  - 근본 원인: {diag['root_cause']}")
                print(f"  - 즉시 조치: {diag['immediate_action']}")
                print(f"  - 장기 해결책: {diag['long_term_solution']}")
    else:
        print("\n✅ 현재 모든 라인 정상 운영 중")
    
    # 6. 모델 성능 평가
    print("\n=== 모델 성능 평가 ===")
    
    # XGBoost 특징 중요도
    importance = system.xgboost_model.get_feature_importance()
    print("\nXGBoost 주요 특징 (상위 5개):")
    print(importance.head())
    
    # 전체 정확도
    X_test = system.xgboost_model.prepare_features(test_df)
    y_test = test_df['is_bottleneck']
    
    X_test_scaled = system.xgboost_model.scaler.transform(X_test)
    y_pred = system.xgboost_model.model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
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
    print("\n시스템 준비 완료! 실시간 예측 가능 상태입니다.")

if __name__ == "__main__":
    main()