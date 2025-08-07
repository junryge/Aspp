"""
CNN-LSTM Multi-Task 모델 종합 평가 시스템
=========================================
학습된 CNN-LSTM Multi-Task 모델의 성능을 다각도로 평가합니다.

평가 항목:
1. 물류량 예측 성능 (회귀 태스크)
   - MAE, MSE, RMSE, R² Score
   - MAPE, SMAPE
   - 예측 정확도 (오차 범위별)

2. 병목 위치 예측 성능 (분류 태스크)
   - Accuracy, Precision, Recall, F1-Score
   - 클래스별 성능
   - 혼동 행렬

3. 종합 성능 평가
   - 두 태스크의 통합 성능 점수
   - 실시간 예측 능력
   - 병목 예방 효과

사용 데이터: data/0730to31.csv
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, precision_recall_fscore_support,
                           confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import platform
from datetime import datetime, timedelta
import joblib
import json
import warnings
import logging

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# ===================================
# 1. 환경 설정
# ===================================

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 고정
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 한글 폰트 설정
def set_korean_font():
    """운영체제별 한글 폰트 자동 설정"""
    system = platform.system()
    
    try:
        if system == 'Windows':
            font_path = 'C:/Windows/Fonts/malgun.ttf'
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.family'] = 'AppleGothic'
        else:  # Linux
            plt.rcParams['font.family'] = 'NanumGothic'
    except:
        print("한글 폰트 설정 실패. 영문으로 표시됩니다.")
        return False
    
    plt.rcParams['axes.unicode_minus'] = False
    return True

USE_KOREAN = set_korean_font()

# ===================================
# 2. 데이터 전처리 함수 (학습 시와 동일)
# ===================================

def load_and_preprocess_data(data_path):
    """데이터 로드 및 전처리"""
    logger.info("평가 데이터 로딩 중...")
    
    # 데이터 로드
    data = pd.read_csv(data_path)
    logger.info(f"원본 데이터 shape: {data.shape}")
    
    # 시간 컬럼 변환
    data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
    data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
    
    # SUM 컬럼 제거
    columns_to_drop = [col for col in data.columns if 'SUM' in col]
    data = data.drop(columns=columns_to_drop)
    
    # 인덱스 설정
    data.set_index('CURRTIME', inplace=True)
    
    return data

def create_features(data):
    """특징 엔지니어링 (학습 시와 동일)"""
    logger.info("특징 생성 중...")
    
    features_data = data.copy()
    
    # 시간 특징
    features_data['hour'] = features_data.index.hour
    features_data['dayofweek'] = features_data.index.dayofweek
    features_data['is_weekend'] = (features_data.index.dayofweek >= 5).astype(int)
    features_data['month'] = features_data.index.month
    features_data['day'] = features_data.index.day
    features_data['is_peak_hour'] = features_data.index.hour.isin([8, 9, 14, 15, 16, 17]).astype(int)
    
    # 팹 간 불균형 지표
    features_data['imbalance_M14A_M10A'] = features_data['M14AM10A'] - features_data['M10AM14A']
    features_data['imbalance_M14A_M14B'] = features_data['M14AM14B'] - features_data['M14BM14A']
    features_data['imbalance_M14A_M16'] = features_data['M14AM16'] - features_data['M16M14A']
    
    # 이동 평균
    for window in [5, 10, 30, 60]:
        features_data[f'MA_{window}'] = features_data['TOTALCNT'].rolling(window=window, min_periods=1).mean()
    
    # 표준편차
    for window in [5, 10, 30]:
        features_data[f'STD_{window}'] = features_data['TOTALCNT'].rolling(window=window, min_periods=1).std()
    
    # 최대/최소값
    features_data['MAX_10'] = features_data['TOTALCNT'].rolling(window=10, min_periods=1).max()
    features_data['MIN_10'] = features_data['TOTALCNT'].rolling(window=10, min_periods=1).min()
    
    # 팹별 부하율
    total_safe = features_data['TOTALCNT'].replace(0, 1)
    features_data['load_M14A_out'] = (features_data['M14AM10A'] + features_data['M14AM14B'] + 
                                      features_data['M14AM16']) / total_safe
    features_data['load_M14A_in'] = (features_data['M10AM14A'] + features_data['M14BM14A'] + 
                                     features_data['M16M14A']) / total_safe
    
    # 경로별 비율
    features_data['ratio_M14A_M10A'] = (features_data['M14AM10A'] + features_data['M10AM14A']) / total_safe
    features_data['ratio_M14A_M14B'] = (features_data['M14AM14B'] + features_data['M14BM14A']) / total_safe
    features_data['ratio_M14A_M16'] = (features_data['M14AM16'] + features_data['M16M14A']) / total_safe
    
    # 변화율
    features_data['change_rate'] = features_data['TOTALCNT'].pct_change()
    features_data['change_rate_5'] = features_data['TOTALCNT'].pct_change(5)
    features_data['change_rate_10'] = features_data['TOTALCNT'].pct_change(10)
    features_data['acceleration'] = features_data['change_rate'].diff()
    
    # 결측값 및 무한대 처리
    features_data = features_data.fillna(method='ffill').fillna(0)
    features_data = features_data.replace([np.inf, -np.inf], 0)
    
    # 이상치 클리핑
    numeric_columns = features_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col not in ['TIME', 'CURRTIME']:
            upper_limit = features_data[col].quantile(0.999)
            lower_limit = features_data[col].quantile(0.001)
            features_data[col] = features_data[col].clip(lower=lower_limit, upper=upper_limit)
    
    return features_data

def create_targets(data, future_minutes=10):
    """타겟 변수 생성"""
    logger.info("타겟 변수 생성 중...")
    
    # 물류량 타겟
    data['FUTURE_TOTALCNT'] = pd.NA
    
    for i in data.index:
        future_time = i + pd.Timedelta(minutes=future_minutes)
        if (future_time <= data.index.max()) & (future_time in data.index):
            data.loc[i, 'FUTURE_TOTALCNT'] = data.loc[future_time, 'TOTALCNT']
    
    # 병목 위치 타겟
    thresholds = {
        'total': data['TOTALCNT'].quantile(0.90),
        'm14a_m10a': np.percentile(data['M14AM10A'] + data['M10AM14A'], 90),
        'm14a_m14b': np.percentile(data['M14AM14B'] + data['M14BM14A'], 90),
        'm14a_m16': np.percentile(data['M14AM16'] + data['M16M14A'], 90)
    }
    
    data['BOTTLENECK_LOCATION'] = 0
    
    for i in data.index:
        future_time = i + pd.Timedelta(minutes=future_minutes)
        if (future_time <= data.index.max()) & (future_time in data.index):
            future_total = data.loc[future_time, 'TOTALCNT']
            
            if future_total > thresholds['total']:
                route_loads = {
                    1: data.loc[future_time, 'M14AM10A'] + data.loc[future_time, 'M10AM14A'],
                    2: data.loc[future_time, 'M14AM14B'] + data.loc[future_time, 'M14BM14A'],
                    3: data.loc[future_time, 'M14AM16'] + data.loc[future_time, 'M16M14A']
                }
                
                max_route = max(route_loads.items(), key=lambda x: x[1])
                if max_route[0] == 1 and max_route[1] > thresholds['m14a_m10a']:
                    data.loc[i, 'BOTTLENECK_LOCATION'] = 1
                elif max_route[0] == 2 and max_route[1] > thresholds['m14a_m14b']:
                    data.loc[i, 'BOTTLENECK_LOCATION'] = 2
                elif max_route[0] == 3 and max_route[1] > thresholds['m14a_m16']:
                    data.loc[i, 'BOTTLENECK_LOCATION'] = 3
    
    # NA 제거
    data = data.dropna(subset=['FUTURE_TOTALCNT'])
    
    return data, thresholds

# ===================================
# 3. Multi-Task 모델 평가 클래스
# ===================================

class MultiTaskModelEvaluator:
    """CNN-LSTM Multi-Task 모델 평가 클래스"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.class_mapping = None
        self.reverse_mapping = None
        
    def load_model_and_config(self):
        """모델과 설정 파일 로드"""
        logger.info("="*60)
        logger.info("모델 및 설정 로딩 중...")
        logger.info("="*60)
        
        # 모델 로드
        model_paths = [
            'model/cnn_lstm_multitask_final.keras',
            'model/cnn_lstm_multitask_best.keras'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                self.model = load_model(path, compile=False)
                logger.info(f"✓ 모델 로드 완료: {path}")
                break
        
        if self.model is None:
            raise FileNotFoundError("모델 파일을 찾을 수 없습니다.")
        
        # 스케일러 로드
        scaler_path = 'scaler/multitask_scaler.pkl'
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✓ 스케일러 로드 완료: {scaler_path}")
        else:
            raise FileNotFoundError("스케일러 파일을 찾을 수 없습니다.")
        
        # 클래스 매핑 로드
        mapping_path = 'config/class_mapping.json'
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                # JSON에서 문자열 키를 정수로 변환
                string_mapping = json.load(f)
                self.class_mapping = {int(k): v for k, v in string_mapping.items()}
                self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}
            logger.info(f"✓ 클래스 매핑 로드 완료: {self.class_mapping}")
        else:
            logger.warning("클래스 매핑 파일을 찾을 수 없습니다. 기본 매핑 사용")
            self.class_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
            self.reverse_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
    
    def prepare_evaluation_data(self, data_path):
        """평가 데이터 준비"""
        # 데이터 로드 및 전처리
        data = load_and_preprocess_data(data_path)
        data = create_features(data)
        data, thresholds = create_targets(data)
        
        # 스케일링할 특징 목록 (학습 시와 동일)
        scale_features_list = [
            'TOTALCNT', 'M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A',
            'imbalance_M14A_M10A', 'imbalance_M14A_M14B', 'imbalance_M14A_M16',
            'MA_5', 'MA_10', 'MA_30', 'MA_60',
            'STD_5', 'STD_10', 'STD_30',
            'MAX_10', 'MIN_10',
            'load_M14A_out', 'load_M14A_in',
            'ratio_M14A_M10A', 'ratio_M14A_M14B', 'ratio_M14A_M16',
            'change_rate', 'change_rate_5', 'change_rate_10',
            'acceleration'
        ]
        
        # 실제 존재하는 컬럼만 선택
        scale_features_list = [col for col in scale_features_list if col in data.columns]
        
        # 스케일링
        data_scaled, _ = self.scale_features(data, scale_features_list)
        
        # 시퀀스 생성
        sequence_features = [col for col in data_scaled.columns if col.startswith('scaled_')]
        target_features = ['FUTURE_TOTALCNT', 'BOTTLENECK_LOCATION']
        
        X, y_regression, y_classification = self.create_sequences(
            data_scaled,
            sequence_features,
            target_features,
            seq_length=60
        )
        
        # 클래스 레이블 재매핑
        unique_classes = np.unique(y_classification)
        if len(self.class_mapping) > 0:
            y_classification_mapped = np.array([self.class_mapping.get(cls, cls) for cls in y_classification])
        else:
            # 기본 매핑
            class_mapping_temp = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
            y_classification_mapped = np.array([class_mapping_temp[cls] for cls in y_classification])
        
        logger.info(f"평가 데이터 준비 완료")
        logger.info(f"  - X shape: {X.shape}")
        logger.info(f"  - y_regression shape: {y_regression.shape}")
        logger.info(f"  - y_classification shape: {y_classification_mapped.shape}")
        logger.info(f"  - 병목 클래스 분포: {np.unique(y_classification_mapped, return_counts=True)}")
        
        return X, y_regression, y_classification_mapped, data, thresholds
    
    def scale_features(self, data, feature_columns):
        """특징 스케일링"""
        scale_columns = [col for col in feature_columns if col in data.columns]
        scale_data = data[scale_columns].copy()
        
        # 무한대 및 NaN 처리
        scale_data = scale_data.replace([np.inf, -np.inf], np.nan)
        scale_data = scale_data.fillna(scale_data.mean())
        
        # 스케일링
        scaled_data = self.scaler.transform(scale_data)
        
        # 스케일링된 데이터프레임 생성
        scaled_df = pd.DataFrame(
            scaled_data,
            columns=[f'scaled_{col}' for col in scale_columns],
            index=data.index
        )
        
        # 원본 데이터와 병합
        result = pd.concat([data, scaled_df], axis=1)
        
        return result, self.scaler
    
    def create_sequences(self, data, feature_cols, target_cols, seq_length=60):
        """시퀀스 데이터 생성"""
        X, y_regression, y_classification = [], [], []
        
        # 연속성 확인
        time_diff = data.index.to_series().diff()
        split_points = time_diff > pd.Timedelta(minutes=1)
        segment_ids = split_points.cumsum()
        
        for segment_id in segment_ids.unique():
            segment = data[segment_ids == segment_id]
            
            if len(segment) > seq_length:
                feature_data = segment[feature_cols].values
                regression_data = segment[target_cols[0]].values
                classification_data = segment[target_cols[1]].values
                
                for i in range(len(segment) - seq_length):
                    X.append(feature_data[i:i+seq_length])
                    y_regression.append(regression_data[i+seq_length])
                    y_classification.append(classification_data[i+seq_length])
        
        return np.array(X), np.array(y_regression), np.array(y_classification)
    
    def evaluate_regression_task(self, y_true, y_pred):
        """회귀 태스크 평가 (물류량 예측)"""
        logger.info("\n" + "="*60)
        logger.info("물류량 예측 성능 평가 (회귀 태스크)")
        logger.info("="*60)
        
        # 기본 메트릭
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE, SMAPE
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask_smape = denominator != 0
        smape = np.mean(np.abs(y_true[mask_smape] - y_pred[mask_smape]) / denominator[mask_smape]) * 100
        
        # 오차 범위별 정확도
        def accuracy_within_threshold(y_true, y_pred, threshold_percent):
            threshold = np.mean(y_true) * (threshold_percent / 100)
            within_threshold = np.abs(y_true - y_pred) <= threshold
            return np.mean(within_threshold) * 100
        
        acc_5 = accuracy_within_threshold(y_true, y_pred, 5)
        acc_10 = accuracy_within_threshold(y_true, y_pred, 10)
        acc_15 = accuracy_within_threshold(y_true, y_pred, 15)
        
        # 결과 출력
        print("\n📊 물류량 예측 성능:")
        print(f"  • MAE: {mae:.2f}")
        print(f"  • MSE: {mse:.2f}")
        print(f"  • RMSE: {rmse:.2f}")
        print(f"  • R² Score: {r2:.4f} ({r2*100:.1f}%)")
        print(f"  • MAPE: {mape:.2f}%")
        print(f"  • SMAPE: {smape:.2f}%")
        
        print("\n🎯 예측 정확도:")
        print(f"  • 5% 오차 범위 내: {acc_5:.1f}%")
        print(f"  • 10% 오차 범위 내: {acc_10:.1f}%")
        print(f"  • 15% 오차 범위 내: {acc_15:.1f}%")
        
        regression_metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'smape': smape,
            'acc_5': acc_5,
            'acc_10': acc_10,
            'acc_15': acc_15
        }
        
        return regression_metrics
    
    def evaluate_classification_task(self, y_true, y_pred_probs):
        """분류 태스크 평가 (병목 위치 예측)"""
        logger.info("\n" + "="*60)
        logger.info("병목 위치 예측 성능 평가 (분류 태스크)")
        logger.info("="*60)
        
        # 예측 클래스
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # 전체 정확도
        accuracy = accuracy_score(y_true, y_pred)
        
        # 클래스별 성능
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # 가중 평균 성능
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        # 클래스 이름 설정
        unique_classes = np.unique(y_true)
        num_classes = len(unique_classes)
        
        if self.reverse_mapping:
            # 원본 클래스로 역매핑
            original_classes = sorted([self.reverse_mapping.get(i, i) for i in unique_classes])
            
            if set(original_classes) == {0, 2, 3}:
                class_names = ['정상', 'M14A-M14B', 'M14A-M16']
            elif set(original_classes) == {0, 1, 2, 3}:
                class_names = ['정상', 'M14A-M10A', 'M14A-M14B', 'M14A-M16']
            else:
                class_names = [f'클래스_{i}' for i in unique_classes]
        else:
            class_names = [f'클래스_{i}' for i in unique_classes]
        
        # 결과 출력
        print(f"\n📊 병목 위치 예측 성능:")
        print(f"  • 전체 정확도: {accuracy:.2%}")
        print(f"  • 가중 평균 정밀도: {weighted_precision:.2%}")
        print(f"  • 가중 평균 재현율: {weighted_recall:.2%}")
        print(f"  • 가중 평균 F1 Score: {weighted_f1:.2%}")
        
        print("\n📈 클래스별 성능:")
        for i, class_name in enumerate(class_names):
            if i < len(precision):  # 클래스가 존재하는 경우만
                print(f"\n  {class_name}:")
                print(f"    - 정밀도: {precision[i]:.2%}")
                print(f"    - 재현율: {recall[i]:.2%}")
                print(f"    - F1 Score: {f1[i]:.2%}")
                print(f"    - 샘플 수: {int(support[i])}")
        
        # 병목 탐지 성능 (병목 vs 정상)
        is_bottleneck_true = y_true > 0
        is_bottleneck_pred = y_pred > 0
        
        bottleneck_accuracy = np.mean(is_bottleneck_true == is_bottleneck_pred)
        
        # True Positive, False Positive 등 계산
        tp = np.sum((is_bottleneck_true == True) & (is_bottleneck_pred == True))
        tn = np.sum((is_bottleneck_true == False) & (is_bottleneck_pred == False))
        fp = np.sum((is_bottleneck_true == False) & (is_bottleneck_pred == True))
        fn = np.sum((is_bottleneck_true == True) & (is_bottleneck_pred == False))
        
        bottleneck_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        bottleneck_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n🚨 병목 탐지 성능 (병목 vs 정상):")
        print(f"  • 병목 탐지 정확도: {bottleneck_accuracy:.2%}")
        print(f"  • 병목 탐지 정밀도: {bottleneck_precision:.2%}")
        print(f"  • 병목 탐지 재현율: {bottleneck_recall:.2%}")
        
        classification_metrics = {
            'accuracy': accuracy,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'bottleneck_accuracy': bottleneck_accuracy,
            'bottleneck_precision': bottleneck_precision,
            'bottleneck_recall': bottleneck_recall,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        return classification_metrics, class_names
    
    def calculate_integrated_score(self, regression_metrics, classification_metrics):
        """통합 성능 점수 계산"""
        # 회귀 태스크 점수 (50%)
        regression_score = (
            (100 - min(regression_metrics['mape'], 100)) * 0.3 +  # MAPE
            regression_metrics['acc_10'] * 0.4 +                   # 10% 정확도
            regression_metrics['r2'] * 100 * 0.3                   # R² Score
        ) * 0.5
        
        # 분류 태스크 점수 (50%)
        classification_score = (
            classification_metrics['accuracy'] * 100 * 0.4 +
            classification_metrics['weighted_f1'] * 100 * 0.3 +
            classification_metrics['bottleneck_recall'] * 100 * 0.3
        ) * 0.5
        
        # 통합 점수
        integrated_score = regression_score + classification_score
        
        # 등급 판정
        if integrated_score >= 90:
            grade = "A+ (탁월함)"
        elif integrated_score >= 85:
            grade = "A (우수함)"
        elif integrated_score >= 80:
            grade = "B+ (매우 좋음)"
        elif integrated_score >= 75:
            grade = "B (좋음)"
        elif integrated_score >= 70:
            grade = "C+ (양호)"
        elif integrated_score >= 65:
            grade = "C (보통)"
        else:
            grade = "D (개선 필요)"
        
        return integrated_score, grade, regression_score, classification_score
    
    def visualize_results(self, y_true_reg, y_pred_reg, y_true_cls, y_pred_cls,
                         regression_metrics, classification_metrics, class_names):
        """평가 결과 시각화"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 물류량 예측 비교 (시계열)
        ax1 = plt.subplot(3, 3, 1)
        sample_size = min(300, len(y_true_reg))
        ax1.plot(y_true_reg[:sample_size], label='실제값', color='blue', linewidth=2)
        ax1.plot(y_pred_reg[:sample_size], label='예측값', color='red', alpha=0.7)
        ax1.set_title('물류량 예측 결과 (시계열)', fontsize=14)
        ax1.set_xlabel('시간')
        ax1.set_ylabel('물류량')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 물류량 예측 산점도
        ax2 = plt.subplot(3, 3, 2)
        ax2.scatter(y_true_reg, y_pred_reg, alpha=0.5, s=10)
        ax2.plot([y_true_reg.min(), y_true_reg.max()],
                [y_true_reg.min(), y_true_reg.max()],
                'r--', lw=2)
        ax2.set_title('물류량 예측 산점도', fontsize=14)
        ax2.set_xlabel('실제값')
        ax2.set_ylabel('예측값')
        ax2.text(0.05, 0.95, f'R² = {regression_metrics["r2"]:.3f}',
                transform=ax2.transAxes, verticalalignment='top')
        ax2.grid(True, alpha=0.3)
        
        # 3. 오차 분포
        ax3 = plt.subplot(3, 3, 3)
        errors = y_true_reg - y_pred_reg
        ax3.hist(errors, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--')
        ax3.set_title('예측 오차 분포', fontsize=14)
        ax3.set_xlabel('오차 (실제값 - 예측값)')
        ax3.set_ylabel('빈도')
        ax3.text(0.05, 0.95, f'MAE = {regression_metrics["mae"]:.2f}',
                transform=ax3.transAxes, verticalalignment='top')
        
        # 4. 병목 예측 혼동 행렬
        ax4 = plt.subplot(3, 3, 4)
        y_pred_cls_labels = np.argmax(y_pred_cls, axis=1)
        cm = classification_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title('병목 위치 예측 혼동 행렬', fontsize=14)
        ax4.set_xlabel('예측값')
        ax4.set_ylabel('실제값')
        ax4.set_xticklabels(class_names, rotation=45)
        ax4.set_yticklabels(class_names, rotation=0)
        
        # 5. 클래스별 F1 Score
        ax5 = plt.subplot(3, 3, 5)
        f1_scores = classification_metrics['f1']
        bars = ax5.bar(class_names, f1_scores, color=['green', 'orange', 'red', 'purple'][:len(f1_scores)])
        ax5.set_title('클래스별 F1 Score', fontsize=14)
        ax5.set_ylabel('F1 Score')
        ax5.set_ylim(0, 1)
        
        # 값 표시
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 6. 시간대별 예측 정확도
        ax6 = plt.subplot(3, 3, 6)
        # 오차를 시간 단위로 그룹화 (예: 100개씩)
        chunk_size = 100
        num_chunks = len(errors) // chunk_size
        chunk_mae = []
        
        for i in range(num_chunks):
            chunk_errors = errors[i*chunk_size:(i+1)*chunk_size]
            chunk_mae.append(np.mean(np.abs(chunk_errors)))
        
        ax6.plot(chunk_mae)
        ax6.set_title('시간대별 MAE 변화', fontsize=14)
        ax6.set_xlabel('시간 구간')
        ax6.set_ylabel('MAE')
        ax6.grid(True, alpha=0.3)
        
        # 7. 병목 예측 확률 분포
        ax7 = plt.subplot(3, 3, 7)
        # 병목 클래스(1,2,3)에 대한 예측 확률의 최대값
        bottleneck_probs = np.max(y_pred_cls[:, 1:], axis=1) if y_pred_cls.shape[1] > 1 else y_pred_cls[:, 0]
        ax7.hist(bottleneck_probs, bins=50, color='orange', alpha=0.7, edgecolor='black')
        ax7.set_title('병목 예측 확률 분포', fontsize=14)
        ax7.set_xlabel('병목 예측 확률')
        ax7.set_ylabel('빈도')
        ax7.axvline(x=0.5, color='red', linestyle='--', label='임계값 0.5')
        ax7.legend()
        
        # 8. 성능 메트릭 요약
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        summary_text = f"""
        물류량 예측 성능:
        • MAE: {regression_metrics['mae']:.2f}
        • RMSE: {regression_metrics['rmse']:.2f}
        • R² Score: {regression_metrics['r2']:.3f}
        • MAPE: {regression_metrics['mape']:.1f}%
        • 10% 정확도: {regression_metrics['acc_10']:.1f}%
        
        병목 예측 성능:
        • 정확도: {classification_metrics['accuracy']:.1%}
        • F1 Score: {classification_metrics['weighted_f1']:.1%}
        • 병목 재현율: {classification_metrics['bottleneck_recall']:.1%}
        """
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 9. 실제 vs 예측 병목 시점
        ax9 = plt.subplot(3, 3, 9)
        sample_size = min(500, len(y_true_cls))
        
        # 실제 병목
        actual_bottleneck = y_true_cls[:sample_size] > 0
        ax9.scatter(np.where(actual_bottleneck)[0], 
                   y_true_reg[:sample_size][actual_bottleneck],
                   color='blue', s=50, alpha=0.6, label='실제 병목')
        
        # 예측 병목
        pred_bottleneck = y_pred_cls_labels[:sample_size] > 0
        ax9.scatter(np.where(pred_bottleneck)[0],
                   y_pred_reg[:sample_size][pred_bottleneck],
                   color='red', s=30, alpha=0.6, label='예측 병목')
        
        ax9.plot(y_true_reg[:sample_size], color='gray', alpha=0.3, linewidth=1)
        ax9.set_title('병목 탐지 결과', fontsize=14)
        ax9.set_xlabel('시간')
        ax9.set_ylabel('물류량')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'multitask_evaluation_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_evaluation_results(self, regression_metrics, classification_metrics,
                               integrated_score, grade):
        """평가 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 결과 딕셔너리 생성
        results = {
            'evaluation_time': timestamp,
            'integrated_score': integrated_score,
            'grade': grade,
            'regression_metrics': regression_metrics,
            'classification_metrics': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in classification_metrics.items()
            }
        }
        
        # JSON 저장
        json_path = f'multitask_evaluation_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        logger.info(f"평가 결과 저장: {json_path}")
        
        # 텍스트 보고서 저장
        report_path = f'multitask_evaluation_report_{timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("CNN-LSTM Multi-Task 모델 평가 보고서\n")
            f.write(f"평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"통합 성능 점수: {integrated_score:.1f}%\n")
            f.write(f"성능 등급: {grade}\n\n")
            
            f.write("물류량 예측 성능:\n")
            f.write(f"  - MAE: {regression_metrics['mae']:.2f}\n")
            f.write(f"  - MAPE: {regression_metrics['mape']:.1f}%\n")
            f.write(f"  - R² Score: {regression_metrics['r2']:.3f}\n")
            f.write(f"  - 10% 정확도: {regression_metrics['acc_10']:.1f}%\n\n")
            
            f.write("병목 예측 성능:\n")
            f.write(f"  - 정확도: {classification_metrics['accuracy']:.1%}\n")
            f.write(f"  - F1 Score: {classification_metrics['weighted_f1']:.1%}\n")
            f.write(f"  - 병목 재현율: {classification_metrics['bottleneck_recall']:.1%}\n")
        
        logger.info(f"평가 보고서 저장: {report_path}")
    
    def evaluate(self, data_path='data/0730to31.csv'):
        """전체 평가 실행"""
        try:
            # 모델 및 설정 로드
            self.load_model_and_config()
            
            # 평가 데이터 준비
            X_test, y_true_reg, y_true_cls, original_data, thresholds = self.prepare_evaluation_data(data_path)
            
            # 모델 예측
            logger.info("\n모델 예측 수행 중...")
            predictions = self.model.predict(X_test, verbose=1)
            y_pred_reg = predictions[0].flatten()
            y_pred_cls = predictions[1]
            
            # 회귀 태스크 평가
            regression_metrics = self.evaluate_regression_task(y_true_reg, y_pred_reg)
            
            # 분류 태스크 평가
            classification_metrics, class_names = self.evaluate_classification_task(y_true_cls, y_pred_cls)
            
            # 통합 점수 계산
            integrated_score, grade, reg_score, cls_score = self.calculate_integrated_score(
                regression_metrics, classification_metrics
            )
            
            # 최종 결과 출력
            logger.info("\n" + "="*70)
            logger.info("종합 평가 결과")
            logger.info("="*70)
            print(f"\n⭐ 통합 성능 점수: {integrated_score:.1f}%")
            print(f"   - 물류량 예측 점수: {reg_score*2:.1f}% (50% 가중치)")
            print(f"   - 병목 예측 점수: {cls_score*2:.1f}% (50% 가중치)")
            print(f"\n📊 성능 등급: {grade}")
            
            # 병목 임계값 정보
            print(f"\n🚨 병목 판정 임계값:")
            print(f"   - 전체 물류량: {thresholds['total']:.0f}")
            print(f"   - M14A-M10A 경로: {thresholds['m14a_m10a']:.0f}")
            print(f"   - M14A-M14B 경로: {thresholds['m14a_m14b']:.0f}")
            print(f"   - M14A-M16 경로: {thresholds['m14a_m16']:.0f}")
            
            # 시각화
            logger.info("\n결과 시각화 생성 중...")
            self.visualize_results(
                y_true_reg, y_pred_reg, y_true_cls, y_pred_cls,
                regression_metrics, classification_metrics, class_names
            )
            
            # 결과 저장
            self.save_evaluation_results(
                regression_metrics, classification_metrics,
                integrated_score, grade
            )
            
            logger.info("\n" + "="*70)
            logger.info("평가 완료!")
            logger.info("="*70)
            
            return {
                'integrated_score': integrated_score,
                'grade': grade,
                'regression_metrics': regression_metrics,
                'classification_metrics': classification_metrics
            }
            
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# ===================================
# 4. 메인 실행 함수
# ===================================

def main(data_path='data/0730to31.csv'):
    """메인 실행 함수"""
    print("\n" + "="*70)
    print("CNN-LSTM Multi-Task 모델 평가 시스템")
    print("="*70)
    
    # 평가기 초기화
    evaluator = MultiTaskModelEvaluator()
    
    # 데이터 파일 확인
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return None
    
    # 평가 실행
    results = evaluator.evaluate(data_path)
    
    if results:
        print("\n✅ 평가가 성공적으로 완료되었습니다!")
        print("\n생성된 파일:")
        print("  - multitask_evaluation_YYYYMMDD_HHMMSS.png (시각화)")
        print("  - multitask_evaluation_YYYYMMDD_HHMMSS.json (결과 데이터)")
        print("  - multitask_evaluation_report_YYYYMMDD_HHMMSS.txt (보고서)")
        
        return results
    else:
        print("\n❌ 평가 실행 중 오류가 발생했습니다.")
        print("\n확인 사항:")
        print("  1. model/cnn_lstm_multitask_final.keras 파일이 있는지 확인")
        print("  2. scaler/multitask_scaler.pkl 파일이 있는지 확인")
        print("  3. 데이터 파일 형식이 올바른지 확인")
        return None

# ===================================
# 5. 스크립트 실행
# ===================================

if __name__ == "__main__":
    import sys
    
    # 명령줄 인자로 데이터 경로 받기
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"사용자 지정 데이터: {data_path}")
        main(data_path)
    else:
        # 기본 경로 사용
        main('data/0730to31.csv')