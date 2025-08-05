"""
하이브리드 딥러닝 모델 성능 평가 시스템 (퍼센트 기반)
=======================================================
학습이 완료된 하이브리드 모델(LSTM, RNN, GRU, Bi-LSTM)의 
성능을 다양한 퍼센트 지표로 평가합니다.

주요 평가 지표:
1. MAPE (Mean Absolute Percentage Error) - 평균 절대 퍼센트 오차
2. 예측 정확도 - 일정 오차 범위 내 예측 비율
3. R² Score - 모델 설명력
4. 방향성 정확도 - 증감 예측 정확도
5. 병목 예측 정확도 - 병목 구간 예측 성공률

개발일: 2024년
버전: 1.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import platform
from datetime import datetime, timedelta
import joblib
import json
import warnings
import traceback

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# ===================================
# 한글 폰트 설정
# ===================================
def set_korean_font():
    """운영체제별 한글 폰트 자동 설정"""
    system = platform.system()
    
    if system == 'Windows':
        # 윈도우
        font_paths = [
            'C:/Windows/Fonts/malgun.ttf',  # 맑은 고딕
            'C:/Windows/Fonts/ngulim.ttf',  # 나눔고딕
            'C:/Windows/Fonts/NanumGothic.ttf'
        ]
        font_family = 'Malgun Gothic'
    elif system == 'Darwin':
        # 맥OS
        font_paths = [
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
            '/Library/Fonts/NanumGothic.ttf'
        ]
        font_family = 'AppleGothic'
    else:
        # 리눅스
        font_paths = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
        ]
        font_family = 'NanumGothic'
    
    # 사용 가능한 폰트 찾기
    font_set = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                font_set = True
                print(f"✓ 한글 폰트 설정: {font_prop.get_name()}")
                break
            except:
                continue
    
    # 폰트를 찾지 못한 경우 기본 설정
    if not font_set:
        try:
            plt.rcParams['font.family'] = font_family
            print(f"✓ 한글 폰트 설정: {font_family}")
        except:
            print("⚠ 한글 폰트를 찾을 수 없습니다. 영문으로 표시됩니다.")
            # 영문 라벨 사용을 위한 플래그
            return False
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    return True

# 한글 폰트 설정 실행
USE_KOREAN = set_korean_font()

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 고정
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class ModelPercentEvaluator:
    """모델 성능을 퍼센트로 평가하는 클래스"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.config = None
        
    def load_models_and_config(self):
        """학습된 모델과 설정 로드"""
        print("="*70)
        print("학습된 모델 로딩 중..." if USE_KOREAN else "Loading trained models...")
        print("="*70)
        
        # 딥러닝 모델들 로드
        model_names = ['lstm', 'gru', 'rnn', 'bi_lstm']
        for model_name in model_names:
            try:
                # 첫 번째 경로 시도
                model_path = f'model/{model_name}_final_hybrid.keras'
                if os.path.exists(model_path):
                    self.models[model_name] = load_model(model_path, compile=False)
                    print(f"✓ {model_name.upper()} {'모델 로드 완료' if USE_KOREAN else 'model loaded'}")
                else:
                    # 대체 경로 시도
                    alt_paths = [
                        f'model/BM_s30f10_0731_2079936.keras',
                        f'model/Model_s30f10_0731_2079936.keras',
                        f'model/{model_name}_best.keras'
                    ]
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            self.models[model_name] = load_model(alt_path, compile=False)
                            print(f"✓ {model_name.upper()} {'모델 로드 완료 (대체 경로)' if USE_KOREAN else 'model loaded (alt path)'}")
                            break
            except Exception as e:
                print(f"⚠ {model_name.upper()} {'모델 로드 실패' if USE_KOREAN else 'model load failed'}: {str(e)}")
        
        # 스케일러 로드
        try:
            scaler_paths = [
                'scaler/standard_scaler_hybrid.pkl',
                'scaler/StdScaler_s30f10_0731_2079936.save'
            ]
            for scaler_path in scaler_paths:
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    print(f"✓ {'스케일러 로드 완료' if USE_KOREAN else 'Scaler loaded'}")
                    break
        except Exception as e:
            print(f"⚠ {'스케일러 로드 실패' if USE_KOREAN else 'Scaler load failed'}: {str(e)}")
        
        # 설정 로드
        try:
            config_path = 'results/training_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    'seq_length': 30,
                    'future_minutes': 10,
                    'bottleneck_threshold': 2000
                }
            print(f"✓ {'설정 파일 로드 완료' if USE_KOREAN else 'Config file loaded'}")
        except:
            self.config = {'seq_length': 30, 'future_minutes': 10, 'bottleneck_threshold': 2000}
            
    def calculate_mape(self, y_true, y_pred):
        """MAPE (Mean Absolute Percentage Error) 계산"""
        # 0값 제외 (0으로 나누기 방지)
        mask = y_true != 0
        if not np.any(mask):
            return 0
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape
    
    def calculate_smape(self, y_true, y_pred):
        """Symmetric MAPE 계산 (0값 문제 해결)"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not np.any(mask):
            return 0
            
        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        return smape
    
    def calculate_accuracy_within_threshold(self, y_true, y_pred, threshold_percent=5):
        """특정 오차 범위 내 예측 정확도 계산"""
        threshold = np.mean(y_true) * (threshold_percent / 100)
        within_threshold = np.abs(y_true - y_pred) <= threshold
        accuracy = np.mean(within_threshold) * 100
        return accuracy
    
    def calculate_directional_accuracy(self, y_true, y_pred, y_previous=None):
        """방향성 정확도 (증가/감소 예측 정확도)"""
        if y_previous is None:
            # 이전 값이 없으면 첫 번째 값을 제외하고 계산
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
        else:
            true_direction = (y_true - y_previous) > 0
            pred_direction = (y_pred - y_previous) > 0
        
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
        return directional_accuracy
    
    def calculate_bottleneck_accuracy(self, y_true, y_pred, threshold):
        """병목 구간 예측 정확도"""
        true_bottleneck = y_true > threshold
        pred_bottleneck = y_pred > threshold
        
        # True Positive, True Negative, False Positive, False Negative
        tp = np.sum((true_bottleneck == True) & (pred_bottleneck == True))
        tn = np.sum((true_bottleneck == False) & (pred_bottleneck == False))
        fp = np.sum((true_bottleneck == False) & (pred_bottleneck == True))
        fn = np.sum((true_bottleneck == True) & (pred_bottleneck == False))
        
        # 정확도
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0
        
        # 정밀도 (Precision)
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        
        # 재현율 (Recall)
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def comprehensive_evaluation(self, y_true, y_pred, model_name="Model"):
        """종합적인 퍼센트 기반 평가"""
        print(f"\n{'='*70}")
        print(f"{model_name} {'성능 평가 (퍼센트 기준)' if USE_KOREAN else 'Performance Evaluation (Percentage)'}")
        print(f"{'='*70}")
        
        # 1. MAPE와 SMAPE
        mape = self.calculate_mape(y_true, y_pred)
        smape = self.calculate_smape(y_true, y_pred)
        
        # 2. 정확도 (다양한 오차 범위)
        accuracy_5 = self.calculate_accuracy_within_threshold(y_true, y_pred, 5)
        accuracy_10 = self.calculate_accuracy_within_threshold(y_true, y_pred, 10)
        accuracy_15 = self.calculate_accuracy_within_threshold(y_true, y_pred, 15)
        
        # 3. R² Score (퍼센트로 변환)
        r2 = r2_score(y_true, y_pred) * 100
        
        # 4. 방향성 정확도
        directional_acc = self.calculate_directional_accuracy(y_true, y_pred)
        
        # 5. 병목 예측 정확도
        threshold = self.config.get('bottleneck_threshold', 2000)
        bottleneck_metrics = self.calculate_bottleneck_accuracy(y_true, y_pred, threshold)
        
        # 6. 전체 성능 점수 (가중 평균)
        overall_score = (
            (100 - min(mape, 100)) * 0.2 +  # MAPE는 낮을수록 좋음
            accuracy_10 * 0.3 +               # 10% 오차 내 정확도
            r2 * 0.2 +                        # R² 점수
            directional_acc * 0.15 +          # 방향성 정확도
            bottleneck_metrics['accuracy'] * 0.15  # 병목 예측 정확도
        )
        
        # 결과 출력
        if USE_KOREAN:
            print("\n📊 예측 오차 지표:")
            print(f"  • MAPE (평균 절대 퍼센트 오차): {mape:.2f}%")
            print(f"  • SMAPE (대칭 평균 절대 퍼센트 오차): {smape:.2f}%")
            
            print("\n🎯 예측 정확도:")
            print(f"  • 5% 오차 범위 내 정확도: {accuracy_5:.1f}%")
            print(f"  • 10% 오차 범위 내 정확도: {accuracy_10:.1f}%")
            print(f"  • 15% 오차 범위 내 정확도: {accuracy_15:.1f}%")
            
            print("\n📈 모델 설명력:")
            print(f"  • R² Score: {r2:.1f}%")
            print(f"  • 방향성 예측 정확도: {directional_acc:.1f}%")
            
            print("\n🚨 병목 구간 예측 성능:")
            print(f"  • 병목 예측 정확도: {bottleneck_metrics['accuracy']:.1f}%")
            print(f"  • 병목 예측 정밀도: {bottleneck_metrics['precision']:.1f}%")
            print(f"  • 병목 예측 재현율: {bottleneck_metrics['recall']:.1f}%")
            print(f"  • F1 Score: {bottleneck_metrics['f1_score']:.1f}%")
            
            print("\n⭐ 종합 성능 점수: {:.1f}%".format(overall_score))
        else:
            print("\n📊 Prediction Error Metrics:")
            print(f"  • MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
            print(f"  • SMAPE (Symmetric MAPE): {smape:.2f}%")
            
            print("\n🎯 Prediction Accuracy:")
            print(f"  • Accuracy within 5% error: {accuracy_5:.1f}%")
            print(f"  • Accuracy within 10% error: {accuracy_10:.1f}%")
            print(f"  • Accuracy within 15% error: {accuracy_15:.1f}%")
            
            print("\n📈 Model Performance:")
            print(f"  • R² Score: {r2:.1f}%")
            print(f"  • Directional Accuracy: {directional_acc:.1f}%")
            
            print("\n🚨 Bottleneck Prediction Performance:")
            print(f"  • Bottleneck Accuracy: {bottleneck_metrics['accuracy']:.1f}%")
            print(f"  • Bottleneck Precision: {bottleneck_metrics['precision']:.1f}%")
            print(f"  • Bottleneck Recall: {bottleneck_metrics['recall']:.1f}%")
            print(f"  • F1 Score: {bottleneck_metrics['f1_score']:.1f}%")
            
            print("\n⭐ Overall Performance Score: {:.1f}%".format(overall_score))
        
        # 성능 등급 판정
        if overall_score >= 90:
            grade = "A+ (탁월함)" if USE_KOREAN else "A+ (Excellent)"
        elif overall_score >= 85:
            grade = "A (우수함)" if USE_KOREAN else "A (Outstanding)"
        elif overall_score >= 80:
            grade = "B+ (매우 좋음)" if USE_KOREAN else "B+ (Very Good)"
        elif overall_score >= 75:
            grade = "B (좋음)" if USE_KOREAN else "B (Good)"
        elif overall_score >= 70:
            grade = "C+ (양호)" if USE_KOREAN else "C+ (Satisfactory)"
        elif overall_score >= 65:
            grade = "C (보통)" if USE_KOREAN else "C (Average)"
        else:
            grade = "D (개선 필요)" if USE_KOREAN else "D (Needs Improvement)"
        
        print(f"📊 {'성능 등급' if USE_KOREAN else 'Performance Grade'}: {grade}")
        
        return {
            'mape': mape,
            'smape': smape,
            'accuracy_5': accuracy_5,
            'accuracy_10': accuracy_10,
            'accuracy_15': accuracy_15,
            'r2_score': r2,
            'directional_accuracy': directional_acc,
            'bottleneck_metrics': bottleneck_metrics,
            'overall_score': overall_score,
            'grade': grade
        }
    
    def plot_evaluation_results(self, evaluation_results):
        """평가 결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 한글/영문 라벨 설정
        if USE_KOREAN:
            labels = {
                'overall_score': '모델별 종합 성능 점수',
                'mape': 'MAPE (낮을수록 좋음)',
                'accuracy': '10% 오차 범위 내 정확도',
                'r2': 'R² Score (모델 설명력)',
                'bottleneck': '병목 구간 예측 정확도',
                'best_model': '최고 성능 모델',
                'categories': ['MAPE\n(역)', '10% 정확도', 'R² Score', '방향성\n정확도', '병목 예측'],
                'good_line': '우수 기준선',
                'overall_score_y': '종합 점수 (%)',
                'accuracy_y': '정확도 (%)'
            }
        else:
            labels = {
                'overall_score': 'Overall Performance Score by Model',
                'mape': 'MAPE (Lower is Better)',
                'accuracy': 'Accuracy within 10% Error Range',
                'r2': 'R² Score (Model Explanatory Power)',
                'bottleneck': 'Bottleneck Prediction Accuracy',
                'best_model': 'Best Performance Model',
                'categories': ['MAPE\n(inv)', '10% Acc', 'R² Score', 'Direction\nAcc', 'Bottleneck'],
                'good_line': 'Excellence Baseline',
                'overall_score_y': 'Overall Score (%)',
                'accuracy_y': 'Accuracy (%)'
            }
        
        # 1. 모델별 종합 점수
        ax = axes[0, 0]
        models = list(evaluation_results.keys())
        scores = [evaluation_results[m]['overall_score'] for m in models]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        bars = ax.bar(models, scores, color=colors[:len(models)])
        ax.set_title(labels['overall_score'], fontsize=14, fontweight='bold')
        ax.set_ylabel(labels['overall_score_y'], fontsize=12)
        ax.set_ylim(0, 100)
        ax.axhline(y=80, color='r', linestyle='--', alpha=0.5, label=labels['good_line'])
        
        # 막대 위에 점수 표시
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 2. MAPE 비교
        ax = axes[0, 1]
        mapes = [evaluation_results[m]['mape'] for m in models]
        bars = ax.bar(models, mapes, color='coral')
        ax.set_title(labels['mape'], fontsize=14, fontweight='bold')
        ax.set_ylabel('MAPE (%)', fontsize=12)
        
        for bar, mape in zip(bars, mapes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{mape:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 3. 정확도 비교 (10% 오차 범위)
        ax = axes[0, 2]
        accuracies = [evaluation_results[m]['accuracy_10'] for m in models]
        bars = ax.bar(models, accuracies, color='lightgreen')
        ax.set_title(labels['accuracy'], fontsize=14, fontweight='bold')
        ax.set_ylabel(labels['accuracy_y'], fontsize=12)
        ax.set_ylim(0, 100)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 4. R² Score 비교
        ax = axes[1, 0]
        r2_scores = [evaluation_results[m]['r2_score'] for m in models]
        bars = ax.bar(models, r2_scores, color='skyblue')
        ax.set_title(labels['r2'], fontsize=14, fontweight='bold')
        ax.set_ylabel('R² (%)', fontsize=12)
        ax.set_ylim(0, 100)
        
        for bar, r2 in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{r2:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 5. 병목 예측 정확도
        ax = axes[1, 1]
        bottleneck_accs = [evaluation_results[m]['bottleneck_metrics']['accuracy'] for m in models]
        bars = ax.bar(models, bottleneck_accs, color='salmon')
        ax.set_title(labels['bottleneck'], fontsize=14, fontweight='bold')
        ax.set_ylabel(labels['accuracy_y'], fontsize=12)
        ax.set_ylim(0, 100)
        
        for bar, acc in zip(bars, bottleneck_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 6. 레이더 차트 (최고 성능 모델)
        ax = axes[1, 2]
        best_model = max(evaluation_results.items(), key=lambda x: x[1]['overall_score'])
        best_model_name, best_model_data = best_model
        
        categories = labels['categories']
        values = [
            100 - min(best_model_data['mape'], 100),  # MAPE는 역으로
            best_model_data['accuracy_10'],
            best_model_data['r2_score'],
            best_model_data['directional_accuracy'],
            best_model_data['bottleneck_metrics']['accuracy']
        ]
        
        # 레이더 차트를 위한 각도
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 닫힌 도형을 위해
        angles += angles[:1]
        
        ax = plt.subplot(2, 3, 6, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, label=best_model_name.upper())
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title(f"{labels['best_model']}: {best_model_name.upper()}", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'model_evaluation_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_all_models(self, test_data_path=None):
        """모든 모델 평가 실행"""
        # 데이터 경로 자동 탐색
        if test_data_path is None:
            possible_paths = [
                'data/0730to31.csv',
                'data/Final_Predicted_Data_0731_BM.csv',
                'data/20240201_TO_202507281705.csv'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    test_data_path = path
                    print(f"테스트 데이터 경로: {test_data_path}")
                    break
            
            if test_data_path is None:
                print("❌ 테스트 데이터 파일을 찾을 수 없습니다.")
                print("   다음 경로 중 하나에 데이터를 배치하세요:")
                for path in possible_paths:
                    print(f"   - {path}")
                return None
        # 모델 로드
        self.load_models_and_config()
        
        if not self.models:
            print("❌ 로드된 모델이 없습니다. 학습을 먼저 완료하세요.")
            return None
        
        print(f"\n로드된 모델 수: {len(self.models)}개")
        
        # 테스트 데이터 준비
        print("\n테스트 데이터 준비 중...")
        X_test, y_test = self.prepare_test_data(test_data_path)
        
        if X_test is None or y_test is None:
            print("❌ 테스트 데이터 준비 실패")
            return None
        
        # 각 모델 평가
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()} 모델 평가 중...")
            
            # 예측
            y_pred_scaled = model.predict(X_test, verbose=0).flatten()
            
            # 역스케일링
            y_pred = self.inverse_scale(y_pred_scaled)
            y_true = self.inverse_scale(y_test)
            
            # 평가
            results = self.comprehensive_evaluation(y_true, y_pred, model_name.upper())
            evaluation_results[model_name] = results
        
        # 앙상블 모델 평가
        if len(self.models) > 1:
            print("\n앙상블 모델 평가 중...")
            ensemble_pred = self.ensemble_predict(X_test)
            ensemble_pred = self.inverse_scale(ensemble_pred)
            
            results = self.comprehensive_evaluation(y_true, ensemble_pred, "ENSEMBLE")
            evaluation_results['ensemble'] = results
        
        # 시각화
        self.plot_evaluation_results(evaluation_results)
        
        # 결과 저장
        self.save_evaluation_results(evaluation_results)
        
        # 최종 요약
        print("\n" + "="*70)
        print("평가 완료 - 최종 요약")
        print("="*70)
        
        best_model = max(evaluation_results.items(), key=lambda x: x[1]['overall_score'])
        print(f"\n🏆 최고 성능 모델: {best_model[0].upper()}")
        print(f"   종합 점수: {best_model[1]['overall_score']:.1f}%")
        print(f"   성능 등급: {best_model[1]['grade']}")
        
        return evaluation_results
    
    def prepare_test_data(self, data_path):
        """테스트 데이터 준비"""
        try:
            # 데이터 로드
            data = pd.read_csv(data_path)
            
            # 시간 컬럼 변환
            data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
            data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
            
            # SUM 컬럼 제거
            columns_to_drop = [col for col in data.columns if 'SUM' in col]
            data = data.drop(columns=columns_to_drop)
            
            # 필요한 컬럼만 선택
            data = data[['CURRTIME', 'TOTALCNT', 'TIME']]
            data.set_index('CURRTIME', inplace=True)
            
            # FUTURE 컬럼 생성
            data['FUTURE'] = pd.NA
            future_minutes = self.config.get('future_minutes', 10)
            
            for i in data.index:
                future_time = i + pd.Timedelta(minutes=future_minutes)
                if (future_time <= data.index.max()) & (future_time in data.index):
                    data.loc[i, 'FUTURE'] = data.loc[future_time, 'TOTALCNT']
            
            data.dropna(subset=['FUTURE'], inplace=True)
            
            # ===== 특징 엔지니어링 추가 (학습 시와 동일하게) =====
            print(f"{'특징 엔지니어링 수행 중...' if USE_KOREAN else 'Performing feature engineering...'}")
            
            # 시간 특징
            data['hour'] = data.index.hour
            data['dayofweek'] = data.index.dayofweek
            data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
            
            # 이동 평균 특징
            data['MA_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
            data['MA_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
            data['MA_30'] = data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
            
            # 표준편차 특징
            data['STD_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).std()
            data['STD_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).std()
            
            # 변화율 특징
            data['change_rate'] = data['TOTALCNT'].pct_change()
            data['change_rate_5'] = data['TOTALCNT'].pct_change(5)
            
            # 결측값 처리
            data = data.ffill().fillna(0)
            
            # 스케일러가 기대하는 컬럼 확인
            expected_features = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else None
            
            if expected_features is not None:
                print(f"{'스케일러가 기대하는 특징' if USE_KOREAN else 'Expected features by scaler'}: {list(expected_features)}")
                scale_columns = list(expected_features)
            else:
                # 기본 스케일링 컬럼 (학습 시와 동일하게)
                scale_columns = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
            
            # 사용 가능한 컬럼만 선택
            scale_columns = [col for col in scale_columns if col in data.columns]
            print(f"{'스케일링할 컬럼' if USE_KOREAN else 'Columns to scale'}: {scale_columns}")
            
            # 스케일링
            scaled_data = self.scaler.transform(data[scale_columns])
            scaled_df = pd.DataFrame(scaled_data, columns=[f'scaled_{col}' for col in scale_columns], index=data.index)
            
            # 원본 데이터와 병합
            data = pd.merge(data, scaled_df, left_index=True, right_index=True, how='left')
            
            # 연속성 확인하여 데이터 분할
            time_diff = data.index.to_series().diff()
            split_points = time_diff > pd.Timedelta(minutes=1)
            segment_ids = split_points.cumsum()
            
            # 입력 특징 선택 (scaled_FUTURE 제외)
            input_features = [col for col in data.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']
            print(f"{'입력 특징' if USE_KOREAN else 'Input features'}: {input_features}")
            
            # 시퀀스 생성
            seq_length = self.config.get('seq_length', 30)
            X, y = [], []
            
            for segment_id in segment_ids.unique():
                segment = data[segment_ids == segment_id]
                
                if len(segment) > seq_length:
                    X_data = segment[input_features].values
                    y_data = segment['scaled_FUTURE'].values if 'scaled_FUTURE' in segment.columns else segment['FUTURE'].values
                    
                    for i in range(len(segment) - seq_length):
                        X.append(X_data[i:i+seq_length])
                        y.append(y_data[i+seq_length])
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"{'테스트 데이터 shape' if USE_KOREAN else 'Test data shape'}: X={X.shape}, y={y.shape}")
            
            return X, y
            
        except Exception as e:
            print(f"{'테스트 데이터 준비 실패' if USE_KOREAN else 'Test data preparation failed'}: {str(e)}")
            traceback.print_exc()
            return None, None
    
    def inverse_scale(self, scaled_data):
        """역스케일링"""
        try:
            n_features = self.scaler.n_features_in_
            dummy = np.zeros((len(scaled_data), n_features))
            
            # FUTURE가 몇 번째 컬럼인지 찾기
            if hasattr(self.scaler, 'feature_names_in_'):
                feature_names = list(self.scaler.feature_names_in_)
                if 'FUTURE' in feature_names:
                    future_idx = feature_names.index('FUTURE')
                elif 'TOTALCNT' in feature_names:
                    # FUTURE가 없으면 TOTALCNT 위치 사용
                    future_idx = feature_names.index('TOTALCNT')
                else:
                    future_idx = 0
            else:
                # 기본적으로 첫 번째 컬럼 사용
                future_idx = 0
            
            dummy[:, future_idx] = scaled_data
            return self.scaler.inverse_transform(dummy)[:, future_idx]
            
        except Exception as e:
            print(f"{'역스케일링 실패' if USE_KOREAN else 'Inverse scaling failed'}: {str(e)}")
            # 실패 시 원본 데이터 반환
            return scaled_data
    
    def ensemble_predict(self, X_test):
        """앙상블 예측"""
        predictions = []
        weights = self.config.get('model_weights', {
            'lstm': 0.3,
            'gru': 0.25,
            'rnn': 0.15,
            'bi_lstm': 0.3
        })
        
        total_weight = 0
        for model_name, model in self.models.items():
            pred = model.predict(X_test, verbose=0).flatten()
            weight = weights.get(model_name, 1/len(self.models))
            predictions.append(pred * weight)
            total_weight += weight
        
        # 가중치 정규화
        if total_weight > 0:
            ensemble_pred = np.sum(predictions, axis=0) / total_weight
        else:
            ensemble_pred = np.mean(predictions, axis=0)
            
        return ensemble_pred
    
    def save_evaluation_results(self, results):
        """평가 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 결과를 DataFrame으로 변환
        rows = []
        for model_name, metrics in results.items():
            row = {
                'Model': model_name.upper(),
                'Overall Score (%)': metrics['overall_score'],
                'Grade': metrics['grade'],
                'MAPE (%)': metrics['mape'],
                'SMAPE (%)': metrics['smape'],
                'Accuracy 5% (%)': metrics['accuracy_5'],
                'Accuracy 10% (%)': metrics['accuracy_10'],
                'Accuracy 15% (%)': metrics['accuracy_15'],
                'R² Score (%)': metrics['r2_score'],
                'Directional Accuracy (%)': metrics['directional_accuracy'],
                'Bottleneck Accuracy (%)': metrics['bottleneck_metrics']['accuracy'],
                'Bottleneck Precision (%)': metrics['bottleneck_metrics']['precision'],
                'Bottleneck Recall (%)': metrics['bottleneck_metrics']['recall'],
                'Bottleneck F1 (%)': metrics['bottleneck_metrics']['f1_score']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Overall Score (%)', ascending=False)
        
        # CSV로 저장
        csv_path = f'model_evaluation_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n📁 {'평가 결과 저장' if USE_KOREAN else 'Evaluation results saved'}: {csv_path}")
        
        # 요약 텍스트 저장
        txt_path = f'model_evaluation_summary_{timestamp}.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            if USE_KOREAN:
                f.write("하이브리드 모델 성능 평가 결과\n")
                f.write(f"평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            else:
                f.write("Hybrid Model Performance Evaluation Results\n")
                f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            for model_name, metrics in sorted(results.items(), 
                                             key=lambda x: x[1]['overall_score'], 
                                             reverse=True):
                f.write(f"\n{model_name.upper()} {'모델' if USE_KOREAN else 'Model'}\n")
                f.write("-"*40 + "\n")
                if USE_KOREAN:
                    f.write(f"종합 점수: {metrics['overall_score']:.1f}%\n")
                    f.write(f"성능 등급: {metrics['grade']}\n")
                    f.write(f"MAPE: {metrics['mape']:.2f}%\n")
                    f.write(f"10% 오차 내 정확도: {metrics['accuracy_10']:.1f}%\n")
                    f.write(f"R² Score: {metrics['r2_score']:.1f}%\n")
                    f.write(f"병목 예측 정확도: {metrics['bottleneck_metrics']['accuracy']:.1f}%\n")
                else:
                    f.write(f"Overall Score: {metrics['overall_score']:.1f}%\n")
                    f.write(f"Performance Grade: {metrics['grade']}\n")
                    f.write(f"MAPE: {metrics['mape']:.2f}%\n")
                    f.write(f"Accuracy within 10% error: {metrics['accuracy_10']:.1f}%\n")
                    f.write(f"R² Score: {metrics['r2_score']:.1f}%\n")
                    f.write(f"Bottleneck Prediction Accuracy: {metrics['bottleneck_metrics']['accuracy']:.1f}%\n")
        
        print(f"📁 {'평가 요약 저장' if USE_KOREAN else 'Evaluation summary saved'}: {txt_path}")

# 메인 실행 함수
def main(test_data_path=None):
    """메인 실행 함수
    
    Args:
        test_data_path: 테스트 데이터 경로 (None이면 자동 탐색)
    """
    print("\n" + "="*70)
    if USE_KOREAN:
        print("하이브리드 딥러닝 모델 퍼센트 평가 시스템")
    else:
        print("Hybrid Deep Learning Model Percentage Evaluation System")
    print("="*70)
    
    evaluator = ModelPercentEvaluator()
    
    # 평가 실행
    results = evaluator.evaluate_all_models(test_data_path)
    
    if results:
        if USE_KOREAN:
            print("\n✅ 모든 평가가 성공적으로 완료되었습니다!")
            print("   - 평가 결과 CSV 파일 저장 완료")
            print("   - 평가 요약 텍스트 파일 저장 완료")
            print("   - 시각화 이미지 저장 완료")
        else:
            print("\n✅ All evaluations completed successfully!")
            print("   - Evaluation results CSV saved")
            print("   - Evaluation summary text saved")
            print("   - Visualization images saved")
        
        # 최고 성능 모델 강조
        best_model = max(results.items(), key=lambda x: x[1]['overall_score'])
        if USE_KOREAN:
            print(f"\n🏆 최고 성능 모델: {best_model[0].upper()}")
            print(f"   종합 점수: {best_model[1]['overall_score']:.1f}%")
            print(f"   성능 등급: {best_model[1]['grade']}")
        else:
            print(f"\n🏆 Best Performance Model: {best_model[0].upper()}")
            print(f"   Overall Score: {best_model[1]['overall_score']:.1f}%")
            print(f"   Performance Grade: {best_model[1]['grade']}")
        
        return results
    else:
        if USE_KOREAN:
            print("\n❌ 평가 실행 중 오류가 발생했습니다.")
            print("\n💡 해결 방법:")
            print("   1. 학습이 완료되었는지 확인하세요")
            print("   2. model/ 폴더에 .keras 파일이 있는지 확인하세요")
            print("   3. scaler/ 폴더에 스케일러 파일이 있는지 확인하세요")
            print("   4. 테스트 데이터 파일이 올바른 형식인지 확인하세요")
        else:
            print("\n❌ Error occurred during evaluation.")
            print("\n💡 Solutions:")
            print("   1. Check if training is completed")
            print("   2. Check if .keras files exist in model/ folder")
            print("   3. Check if scaler file exists in scaler/ folder")
            print("   4. Check if test data file has correct format")
        return None

if __name__ == "__main__":
    import sys
    
    # 명령줄 인자로 데이터 경로 받기
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"사용자 지정 테스트 데이터: {test_path}")
        main(test_path)
    else:
        main()