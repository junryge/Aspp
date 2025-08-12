"""
급증 예측 하이브리드 딥러닝 모델 성능 평가 시스템 v3.0
=======================================================
이중 출력 하이브리드 모델(LSTM, RNN, GRU, Bi-LSTM)의 
급증 예측 성능을 중심으로 평가합니다.

주요 평가 지표:
1. 급증 예측 정확도 (TOTALCNT > 1400)
2. Precision, Recall, F1-Score
3. 수치 예측 정확도 (MAPE, MAE)
4. 급증 구간별 성능 분석
5. 오탐/미탐 분석

개발일: 2024년
버전: 3.0 (급증 예측 특화)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           classification_report, confusion_matrix, roc_auc_score, roc_curve)
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
        font_paths = [
            'C:/Windows/Fonts/malgun.ttf',
            'C:/Windows/Fonts/ngulim.ttf',
            'C:/Windows/Fonts/NanumGothic.ttf'
        ]
        font_family = 'Malgun Gothic'
    elif system == 'Darwin':
        font_paths = [
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
            '/Library/Fonts/NanumGothic.ttf'
        ]
        font_family = 'AppleGothic'
    else:
        font_paths = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
        ]
        font_family = 'NanumGothic'
    
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
    
    if not font_set:
        try:
            plt.rcParams['font.family'] = font_family
            print(f"✓ 한글 폰트 설정: {font_family}")
        except:
            print("⚠ 한글 폰트를 찾을 수 없습니다. 영문으로 표시됩니다.")
            return False
    
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

class SpikeModelEvaluator:
    """급증 예측 모델 성능을 평가하는 클래스"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.config = None
        self.spike_threshold = 1400  # 급증 임계값
        
    def load_models_and_config(self):
        """학습된 모델과 설정 로드"""
        print("="*70)
        print("학습된 급증 예측 모델 로딩 중..." if USE_KOREAN else "Loading trained spike prediction models...")
        print("="*70)
        
        # v3 이중 출력 모델들 로드
        model_names = ['dual_lstm', 'dual_gru', 'dual_rnn', 'dual_bilstm']
        for model_name in model_names:
            try:
                model_path = f'model_v3/{model_name}_final.keras'
                if os.path.exists(model_path):
                    self.models[model_name] = load_model(model_path, compile=False)
                    print(f"✓ {model_name.upper()} {'모델 로드 완료' if USE_KOREAN else 'model loaded'}")
                else:
                    # 기존 모델 경로 시도
                    alt_path = f'model/{model_name.replace("dual_", "")}_final_hybrid.keras'
                    if os.path.exists(alt_path):
                        print(f"⚠ {model_name} v3 모델이 없어 기존 모델 사용")
            except Exception as e:
                print(f"⚠ {model_name.upper()} {'모델 로드 실패' if USE_KOREAN else 'model load failed'}: {str(e)}")
        
        # 스케일러 로드
        try:
            scaler_paths = [
                'scaler_v3/scaler_v3.pkl',
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
            config_paths = [
                'results_v3/training_config.json',
                'results/training_config.json'
            ]
            for config_path in config_paths:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    break
            
            if not self.config:
                self.config = {
                    'seq_length': 30,
                    'future_minutes': 10,
                    'spike_threshold': 1400
                }
            print(f"✓ {'설정 파일 로드 완료' if USE_KOREAN else 'Config file loaded'}")
        except:
            self.config = {'seq_length': 30, 'future_minutes': 10, 'spike_threshold': 1400}
    
    def calculate_spike_metrics(self, y_true, y_pred_prob, threshold=0.5):
        """급증 예측 성능 지표 계산"""
        # 이진 분류로 변환
        y_pred = (y_pred_prob > threshold).astype(int)
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 기본 지표
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 추가 지표
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        
        # AUC 계산
        try:
            auc = roc_auc_score(y_true, y_pred_prob) * 100
        except:
            auc = 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'auc': auc,
            'confusion_matrix': cm,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    
    def comprehensive_spike_evaluation(self, y_true_reg, y_pred_reg, y_true_cls, y_pred_cls, model_name="Model"):
        """급증 예측 중심의 종합 평가"""
        print(f"\n{'='*70}")
        print(f"{model_name} {'급증 예측 성능 평가' if USE_KOREAN else 'Spike Prediction Performance'}")
        print(f"{'='*70}")
        
        # 1. 급증 예측 성능
        spike_metrics = self.calculate_spike_metrics(y_true_cls, y_pred_cls)
        
        # 2. 다양한 임계값에서의 성능
        thresholds = [0.3, 0.5, 0.7]
        threshold_metrics = {}
        for thr in thresholds:
            threshold_metrics[thr] = self.calculate_spike_metrics(y_true_cls, y_pred_cls, thr)
        
        # 3. 수치 예측 성능 (원본 스케일)
        y_true_original = self.inverse_scale(y_true_reg)
        y_pred_original = self.inverse_scale(y_pred_reg)
        
        mae = mean_absolute_error(y_true_original, y_pred_original)
        mape = np.mean(np.abs((y_true_original - y_pred_original) / (y_true_original + 1e-8))) * 100
        rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
        
        # 4. 급증 구간에서의 예측 성능
        spike_mask = y_true_cls == 1
        if np.any(spike_mask):
            spike_mae = mean_absolute_error(y_true_original[spike_mask], y_pred_original[spike_mask])
            spike_mape = np.mean(np.abs((y_true_original[spike_mask] - y_pred_original[spike_mask]) / 
                                       (y_true_original[spike_mask] + 1e-8))) * 100
        else:
            spike_mae = 0
            spike_mape = 0
        
        # 5. 종합 점수 (급증 예측 중심)
        overall_score = (
            spike_metrics['recall'] * 0.35 +          # 재현율 최우선
            spike_metrics['precision'] * 0.25 +       # 정밀도
            spike_metrics['f1_score'] * 0.2 +         # F1 점수
            (100 - min(spike_mape, 100)) * 0.1 +     # 급증 구간 예측 정확도
            spike_metrics['auc'] * 0.1                # AUC
        )
        
        # 결과 출력
        if USE_KOREAN:
            print("\n🎯 급증 예측 성능 (임계값 0.5):")
            print(f"  • 정확도: {spike_metrics['accuracy']:.1f}%")
            print(f"  • 정밀도: {spike_metrics['precision']:.1f}%")
            print(f"  • 재현율: {spike_metrics['recall']:.1f}% ⭐")
            print(f"  • F1-Score: {spike_metrics['f1_score']:.1f}%")
            print(f"  • 특이도: {spike_metrics['specificity']:.1f}%")
            print(f"  • AUC: {spike_metrics['auc']:.1f}%")
            
            print("\n📊 임계값별 재현율:")
            for thr, metrics in threshold_metrics.items():
                print(f"  • 임계값 {thr}: {metrics['recall']:.1f}%")
            
            print("\n💹 급증 구간 예측 정확도:")
            print(f"  • 급증 구간 MAE: {spike_mae:.0f}")
            print(f"  • 급증 구간 MAPE: {spike_mape:.1f}%")
            
            print("\n🔢 전체 수치 예측 성능:")
            print(f"  • MAE: {mae:.0f}")
            print(f"  • MAPE: {mape:.1f}%")
            print(f"  • RMSE: {rmse:.0f}")
            
            print("\n⚠️ 오탐/미탐 분석:")
            print(f"  • 미탐 (False Negative): {spike_metrics['fn']}건")
            print(f"  • 오탐 (False Positive): {spike_metrics['fp']}건")
            
            print("\n⭐ 종합 점수: {:.1f}%".format(overall_score))
        else:
            print("\n🎯 Spike Prediction Performance (threshold 0.5):")
            print(f"  • Accuracy: {spike_metrics['accuracy']:.1f}%")
            print(f"  • Precision: {spike_metrics['precision']:.1f}%")
            print(f"  • Recall: {spike_metrics['recall']:.1f}% ⭐")
            print(f"  • F1-Score: {spike_metrics['f1_score']:.1f}%")
            print(f"  • Specificity: {spike_metrics['specificity']:.1f}%")
            print(f"  • AUC: {spike_metrics['auc']:.1f}%")
            
            print("\n⭐ Overall Score: {:.1f}%".format(overall_score))
        
        # 성능 등급
        if spike_metrics['recall'] >= 70:
            grade = "A (목표 달성)" if USE_KOREAN else "A (Target Achieved)"
        elif spike_metrics['recall'] >= 60:
            grade = "B (양호)" if USE_KOREAN else "B (Good)"
        elif spike_metrics['recall'] >= 50:
            grade = "C (보통)" if USE_KOREAN else "C (Average)"
        else:
            grade = "D (개선 필요)" if USE_KOREAN else "D (Needs Improvement)"
        
        print(f"📊 {'급증 예측 등급' if USE_KOREAN else 'Spike Prediction Grade'}: {grade}")
        
        return {
            'spike_metrics': spike_metrics,
            'threshold_metrics': threshold_metrics,
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'spike_mae': spike_mae,
            'spike_mape': spike_mape,
            'overall_score': overall_score,
            'grade': grade
        }
    
    def plot_spike_evaluation_results(self, evaluation_results):
        """급증 예측 결과 시각화"""
        fig = plt.figure(figsize=(20, 15))
        
        # 한글/영문 라벨
        if USE_KOREAN:
            labels = {
                'recall_comparison': '모델별 급증 예측 재현율',
                'confusion_matrix': '혼동 행렬',
                'threshold_analysis': '임계값별 성능 분석',
                'spike_timeline': '급증 예측 타임라인',
                'feature_importance': '구간별 급증 기여도',
                'overall_performance': '종합 성능 비교'
            }
        else:
            labels = {
                'recall_comparison': 'Spike Prediction Recall by Model',
                'confusion_matrix': 'Confusion Matrix',
                'threshold_analysis': 'Performance by Threshold',
                'spike_timeline': 'Spike Prediction Timeline',
                'feature_importance': 'Segment Contribution to Spikes',
                'overall_performance': 'Overall Performance Comparison'
            }
        
        # 1. 모델별 재현율 비교
        ax1 = plt.subplot(3, 3, 1)
        models = list(evaluation_results.keys())
        recalls = [evaluation_results[m]['spike_metrics']['recall'] for m in models]
        precisions = [evaluation_results[m]['spike_metrics']['precision'] for m in models]
        f1_scores = [evaluation_results[m]['spike_metrics']['f1_score'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax1.bar(x - width, recalls, width, label='Recall', color='green')
        ax1.bar(x, precisions, width, label='Precision', color='blue')
        ax1.bar(x + width, f1_scores, width, label='F1-Score', color='orange')
        
        ax1.set_title(labels['recall_comparison'], fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.upper() for m in models], rotation=45)
        ax1.set_ylabel('Score (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 목표선 표시
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Target (70%)')
        
        # 2. 최고 모델의 혼동 행렬
        ax2 = plt.subplot(3, 3, 2)
        best_model = max(evaluation_results.items(), 
                        key=lambda x: x[1]['spike_metrics']['recall'])
        cm = best_model[1]['spike_metrics']['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Spike'],
                   yticklabels=['Normal', 'Spike'],
                   ax=ax2)
        ax2.set_title(f"{labels['confusion_matrix']} - {best_model[0].upper()}", fontsize=14)
        
        # 3. 임계값 분석
        ax3 = plt.subplot(3, 3, 3)
        thresholds = [0.3, 0.5, 0.7]
        for model_name, results in evaluation_results.items():
            recalls = [results['threshold_metrics'][t]['recall'] for t in thresholds]
            ax3.plot(thresholds, recalls, marker='o', label=model_name.upper())
        
        ax3.set_title(labels['threshold_analysis'], fontsize=14)
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Recall (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 모델별 종합 점수
        ax4 = plt.subplot(3, 3, 4)
        overall_scores = [evaluation_results[m]['overall_score'] for m in models]
        bars = ax4.bar(models, overall_scores, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        
        ax4.set_title(labels['overall_performance'], fontsize=14)
        ax4.set_ylabel('Overall Score (%)')
        ax4.set_ylim(0, 100)
        
        # 점수 표시
        for bar, score in zip(bars, overall_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}%', ha='center', va='bottom')
        
        # 5. ROC 곡선 (가능한 경우)
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax5.set_xlim([0, 1])
        ax5.set_ylim([0, 1])
        ax5.set_xlabel('False Positive Rate')
        ax5.set_ylabel('True Positive Rate')
        ax5.set_title('ROC Curves', fontsize=14)
        ax5.grid(True, alpha=0.3)
        
        # 6. 급증 예측 성능 레이더 차트
        ax6 = plt.subplot(3, 3, 6, projection='polar')
        
        # 최고 모델의 성능 지표
        metrics = best_model[1]['spike_metrics']
        categories = ['Recall', 'Precision', 'F1-Score', 'Specificity', 'AUC']
        values = [
            metrics['recall'],
            metrics['precision'],
            metrics['f1_score'],
            metrics['specificity'],
            metrics['auc']
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax6.plot(angles, values, 'o-', linewidth=2, color='darkred')
        ax6.fill(angles, values, alpha=0.25, color='darkred')
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 100)
        ax6.set_title(f"Best Model Performance: {best_model[0].upper()}", fontsize=14)
        
        # 7. 급증 구간 MAE 비교
        ax7 = plt.subplot(3, 3, 7)
        spike_maes = [evaluation_results[m]['spike_mae'] for m in models]
        bars = ax7.bar(models, spike_maes, color='coral')
        
        ax7.set_title('Spike Period MAE' if not USE_KOREAN else '급증 구간 MAE', fontsize=14)
        ax7.set_ylabel('MAE')
        
        for bar, mae in zip(bars, spike_maes):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{mae:.0f}', ha='center', va='bottom')
        
        # 8. 미탐/오탐 분석
        ax8 = plt.subplot(3, 3, 8)
        fn_counts = [evaluation_results[m]['spike_metrics']['fn'] for m in models]
        fp_counts = [evaluation_results[m]['spike_metrics']['fp'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax8.bar(x - width/2, fn_counts, width, label='False Negative', color='red', alpha=0.7)
        ax8.bar(x + width/2, fp_counts, width, label='False Positive', color='orange', alpha=0.7)
        
        ax8.set_title('False Predictions Analysis' if not USE_KOREAN else '오탐/미탐 분석', fontsize=14)
        ax8.set_xticks(x)
        ax8.set_xticklabels([m.upper() for m in models])
        ax8.set_ylabel('Count')
        ax8.legend()
        
        # 9. 성능 등급 표시
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # 등급별 색상
        grade_colors = {
            'A': 'darkgreen',
            'B': 'green',
            'C': 'orange',
            'D': 'red'
        }
        
        y_pos = 0.9
        for model_name, results in evaluation_results.items():
            grade = results['grade']
            grade_letter = grade.split()[0]
            color = grade_colors.get(grade_letter, 'gray')
            
            ax9.text(0.1, y_pos, f"{model_name.upper()}:", fontsize=12, fontweight='bold')
            ax9.text(0.5, y_pos, grade, fontsize=12, color=color, fontweight='bold')
            ax9.text(0.8, y_pos, f"{results['spike_metrics']['recall']:.1f}%", fontsize=12)
            y_pos -= 0.15
        
        ax9.set_title('Model Grades' if not USE_KOREAN else '모델 등급', fontsize=14)
        ax9.text(0.1, 0.2, 'Target: Recall ≥ 70%' if not USE_KOREAN else '목표: 재현율 ≥ 70%', 
                fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'spike_evaluation_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_test_data(self, data_path):
        """급증 예측을 위한 테스트 데이터 준비"""
        try:
            # 데이터 로드
            data = pd.read_csv(data_path)
            
            # 시간 컬럼 변환
            data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
            data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
            
            # 필요한 컬럼 선택
            required_columns = ['CURRTIME', 'TOTALCNT', 'M14AM10A', 'M10AM14A', 
                              'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A', 'TIME']
            available_columns = [col for col in required_columns if col in data.columns]
            data = data[available_columns]
            data.set_index('CURRTIME', inplace=True)
            
            # FUTURE 컬럼 생성 (10분 후)
            data['FUTURE'] = pd.NA
            future_minutes = self.config.get('future_minutes', 10)
            
            for i in data.index:
                future_time = i + pd.Timedelta(minutes=future_minutes)
                if (future_time <= data.index.max()) & (future_time in data.index):
                    data.loc[i, 'FUTURE'] = data.loc[future_time, 'TOTALCNT']
            
            data.dropna(subset=['FUTURE'], inplace=True)
            
            # 급증 라벨 생성
            data['future_spike'] = (data['FUTURE'] > self.spike_threshold).astype(int)
            
            print(f"급증 비율: {data['future_spike'].mean():.2%}")
            
            # 개별 구간 특징 생성
            segment_columns = ['M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A']
            available_segments = [col for col in segment_columns if col in data.columns]
            
            for col in available_segments:
                # 비율
                data[f'{col}_ratio'] = data[col] / (data['TOTALCNT'] + 1e-6)
                # 변화율
                data[f'{col}_change_10'] = data[col].pct_change(10).fillna(0)
                # 이동평균
                data[f'{col}_MA5'] = data[col].rolling(window=5, min_periods=1).mean()
            
            # 급증 신호 특징
            if 'M14AM14B' in data.columns:
                data['M14AM14B_spike_signal'] = (data['M14AM14B_change_10'] > 0.5).astype(int)
            if 'M16M14A' in data.columns:
                data['M16M14A_spike_signal'] = (data['M16M14A_change_10'] > 0.5).astype(int)
            
            # 기본 특징
            data['hour'] = data.index.hour
            data['dayofweek'] = data.index.dayofweek
            data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
            data['MA_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
            data['MA_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
            data['MA_30'] = data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
            data['STD_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).std()
            data['STD_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).std()
            data['change_rate'] = data['TOTALCNT'].pct_change()
            data['change_rate_5'] = data['TOTALCNT'].pct_change(5)
            
            # 결측값 처리
            data = data.ffill().fillna(0)
            
            # 스케일링
            scaling_columns = ['TOTALCNT', 'FUTURE'] + available_segments
            scaling_columns += [col for col in data.columns if 'MA' in col or 'STD' in col]
            scaling_columns += [f'{seg}_MA5' for seg in available_segments if f'{seg}_MA5' in data.columns]
            scaling_columns = list(set(scaling_columns))
            
            if hasattr(self.scaler, 'feature_names_in_'):
                expected_columns = list(self.scaler.feature_names_in_)
                scaling_columns = [col for col in expected_columns if col in data.columns]
            
            scaled_data = self.scaler.transform(data[scaling_columns])
            scaled_df = pd.DataFrame(scaled_data, columns=[f'scaled_{col}' for col in scaling_columns], 
                                   index=data.index)
            
            # 비스케일 특징
            non_scaled_features = [col for col in data.columns 
                                 if ('ratio' in col or 'change' in col or 'signal' in col or 
                                     col in ['hour', 'dayofweek', 'is_weekend', 'future_spike'])]
            
            # 최종 데이터
            final_data = pd.concat([data[non_scaled_features], scaled_df], axis=1)
            
            # 시퀀스 생성
            def split_data_by_continuity(data):
                time_diff = data.index.to_series().diff()
                split_points = time_diff > pd.Timedelta(minutes=1)
                segment_ids = split_points.cumsum()
                segments = []
                for segment_id in segment_ids.unique():
                    segment = data[segment_ids == segment_id].copy()
                    if len(segment) > 30:
                        segments.append(segment)
                return segments
            
            data_segments = split_data_by_continuity(final_data)
            
            # 시퀀스 생성
            def create_sequences(data, feature_cols, target_col_reg, target_col_cls, seq_length=30):
                X, y_reg, y_cls = [], [], []
                feature_data = data[feature_cols].values
                target_data_reg = data[target_col_reg].values
                target_data_cls = data[target_col_cls].values
                
                for i in range(len(data) - seq_length):
                    X.append(feature_data[i:i+seq_length])
                    y_reg.append(target_data_reg[i+seq_length])
                    y_cls.append(target_data_cls[i+seq_length])
                
                return np.array(X), np.array(y_reg), np.array(y_cls)
            
            seq_length = self.config.get('seq_length', 30)
            input_features = [col for col in final_data.columns 
                            if col not in ['scaled_FUTURE', 'future_spike']]
            
            all_X, all_y_reg, all_y_cls = [], [], []
            for segment in data_segments:
                X_seg, y_reg_seg, y_cls_seg = create_sequences(
                    segment, input_features, 'scaled_FUTURE', 'future_spike', seq_length
                )
                if len(X_seg) > 0:
                    all_X.append(X_seg)
                    all_y_reg.append(y_reg_seg)
                    all_y_cls.append(y_cls_seg)
            
            X = np.concatenate(all_X, axis=0)
            y_reg = np.concatenate(all_y_reg, axis=0)
            y_cls = np.concatenate(all_y_cls, axis=0)
            
            print(f"테스트 데이터 shape: X={X.shape}, y_reg={y_reg.shape}, y_cls={y_cls.shape}")
            print(f"테스트 데이터 급증 비율: {y_cls.mean():.2%}")
            
            return X, y_reg, y_cls, data
            
        except Exception as e:
            print(f"테스트 데이터 준비 실패: {str(e)}")
            traceback.print_exc()
            return None, None, None, None
    
    def inverse_scale(self, scaled_data):
        """역스케일링"""
        try:
            if hasattr(self.scaler, 'feature_names_in_'):
                feature_names = list(self.scaler.feature_names_in_)
                n_features = len(feature_names)
                
                dummy = np.zeros((len(scaled_data), n_features))
                
                if 'FUTURE' in feature_names:
                    future_idx = feature_names.index('FUTURE')
                else:
                    future_idx = 0
                
                dummy[:, future_idx] = scaled_data
                return self.scaler.inverse_transform(dummy)[:, future_idx]
            else:
                # 기본 역스케일링
                n_features = self.scaler.n_features_in_
                dummy = np.zeros((len(scaled_data), n_features))
                dummy[:, 0] = scaled_data
                return self.scaler.inverse_transform(dummy)[:, 0]
                
        except Exception as e:
            print(f"역스케일링 실패: {str(e)}")
            return scaled_data
    
    def enhanced_ensemble_predict(self, models, X_test, spike_weight_threshold=0.7):
        """급증 예측을 강화한 앙상블"""
        regression_preds = {}
        spike_preds = {}
        
        # 각 모델별 예측
        for model_name, model in models.items():
            pred = model.predict(X_test, verbose=0)
            
            # 이중 출력 모델인 경우
            if isinstance(pred, list) and len(pred) == 2:
                regression_preds[model_name] = pred[0].flatten()
                spike_preds[model_name] = pred[1].flatten()
            else:
                # 단일 출력 모델인 경우 (기존 모델)
                regression_preds[model_name] = pred.flatten()
                spike_preds[model_name] = np.zeros_like(pred.flatten())
        
        # 가중 평균
        weights = self.config.get('model_weights', {
            'dual_lstm': 0.35,
            'dual_gru': 0.25,
            'dual_rnn': 0.15,
            'dual_bilstm': 0.25
        })
        
        ensemble_regression = np.zeros_like(list(regression_preds.values())[0])
        ensemble_spike = np.zeros_like(list(spike_preds.values())[0])
        total_weight = 0
        
        for model_name in regression_preds:
            weight = weights.get(model_name, 1/len(models))
            ensemble_regression += weight * regression_preds[model_name]
            ensemble_spike += weight * spike_preds[model_name]
            total_weight += weight
        
        ensemble_regression /= total_weight
        ensemble_spike /= total_weight
        
        # 급증 확률이 높으면 예측값 상향 조정
        spike_mask = ensemble_spike > spike_weight_threshold
        ensemble_regression[spike_mask] *= 1.15
        
        return ensemble_regression, ensemble_spike, regression_preds, spike_preds
    
    def evaluate_all_models(self, test_data_path=None):
        """모든 모델 평가 실행"""
        # 데이터 경로 설정
        if test_data_path is None:
            possible_paths = [
                'data/20250731_to_20250806.csv',
                'data/0730to31.csv',
                'data/20240201_TO_202507281705.csv'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    test_data_path = path
                    print(f"테스트 데이터 경로: {test_data_path}")
                    break
            
            if test_data_path is None:
                print("❌ 테스트 데이터 파일을 찾을 수 없습니다.")
                return None
        
        # 모델 로드
        self.load_models_and_config()
        
        if not self.models:
            print("❌ 로드된 모델이 없습니다.")
            return None
        
        print(f"\n로드된 모델 수: {len(self.models)}개")
        
        # 테스트 데이터 준비
        print("\n테스트 데이터 준비 중...")
        X_test, y_test_reg, y_test_cls, test_data = self.prepare_test_data(test_data_path)
        
        if X_test is None:
            print("❌ 테스트 데이터 준비 실패")
            return None
        
        # 각 모델 평가
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper()} 모델 평가 중...")
            
            # 예측
            pred = model.predict(X_test, verbose=0)
            
            if isinstance(pred, list) and len(pred) == 2:
                # 이중 출력 모델
                y_pred_reg = pred[0].flatten()
                y_pred_cls = pred[1].flatten()
            else:
                # 단일 출력 모델 (급증 예측 없음)
                y_pred_reg = pred.flatten()
                y_pred_cls = np.zeros_like(y_pred_reg)
            
            # 평가
            results = self.comprehensive_spike_evaluation(
                y_test_reg, y_pred_reg, y_test_cls, y_pred_cls, model_name.upper()
            )
            evaluation_results[model_name] = results
        
        # 앙상블 평가
        if len(self.models) > 1:
            print("\n앙상블 모델 평가 중...")
            ensemble_reg, ensemble_spike, _, _ = self.enhanced_ensemble_predict(self.models, X_test)
            
            results = self.comprehensive_spike_evaluation(
                y_test_reg, ensemble_reg, y_test_cls, ensemble_spike, "ENSEMBLE"
            )
            evaluation_results['ensemble'] = results
        
        # 시각화
        self.plot_spike_evaluation_results(evaluation_results)
        
        # 결과 저장
        self.save_evaluation_results(evaluation_results)
        
        # 급증 예측 샘플 분석
        self.analyze_spike_predictions(X_test, y_test_reg, y_test_cls, test_data, evaluation_results)
        
        # 최종 요약
        print("\n" + "="*70)
        print("평가 완료 - 최종 요약")
        print("="*70)
        
        # 목표 달성 확인
        target_recall = 70
        achieved_models = [(name, res) for name, res in evaluation_results.items() 
                          if res['spike_metrics']['recall'] >= target_recall]
        
        if achieved_models:
            print(f"\n✅ 목표 달성 모델 (재현율 ≥ {target_recall}%):")
            for name, res in achieved_models:
                print(f"   • {name.upper()}: {res['spike_metrics']['recall']:.1f}%")
        else:
            print(f"\n❌ 목표 미달성 (재현율 < {target_recall}%)")
        
        # 최고 성능 모델
        best_model = max(evaluation_results.items(), 
                        key=lambda x: x[1]['spike_metrics']['recall'])
        print(f"\n🏆 최고 급증 예측 모델: {best_model[0].upper()}")
        print(f"   재현율: {best_model[1]['spike_metrics']['recall']:.1f}%")
        print(f"   정밀도: {best_model[1]['spike_metrics']['precision']:.1f}%")
        print(f"   F1-Score: {best_model[1]['spike_metrics']['f1_score']:.1f}%")
        
        return evaluation_results
    
    def analyze_spike_predictions(self, X_test, y_test_reg, y_test_cls, test_data, evaluation_results):
        """급증 예측 상세 분석"""
        print("\n" + "="*70)
        print("급증 예측 상세 분석")
        print("="*70)
        
        # 최고 모델 선택
        best_model_name = max(evaluation_results.items(), 
                             key=lambda x: x[1]['spike_metrics']['recall'])[0]
        best_model = self.models[best_model_name]
        
        # 예측 수행
        pred = best_model.predict(X_test[:100], verbose=0)  # 샘플 100개
        if isinstance(pred, list) and len(pred) == 2:
            y_pred_cls = pred[1].flatten()
        else:
            return
        
        # 급증 예측 샘플 분석
        spike_pred_indices = np.where(y_pred_cls > 0.7)[0]
        actual_spike_indices = np.where(y_test_cls[:100] == 1)[0]
        
        print(f"\n예측된 급증 수: {len(spike_pred_indices)}")
        print(f"실제 급증 수: {len(actual_spike_indices)}")
        
        # 정확히 예측한 급증
        correct_spikes = np.intersect1d(spike_pred_indices, actual_spike_indices)
        print(f"정확히 예측한 급증: {len(correct_spikes)}건")
        
        # 미탐 사례
        missed_spikes = np.setdiff1d(actual_spike_indices, spike_pred_indices)
        print(f"놓친 급증 (미탐): {len(missed_spikes)}건")
        
        # 오탐 사례
        false_alarms = np.setdiff1d(spike_pred_indices, actual_spike_indices)
        print(f"잘못된 급증 예측 (오탐): {len(false_alarms)}건")
    
    def save_evaluation_results(self, results):
        """평가 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # DataFrame 생성
        rows = []
        for model_name, metrics in results.items():
            row = {
                'Model': model_name.upper(),
                'Overall Score (%)': metrics['overall_score'],
                'Grade': metrics['grade'],
                'Spike Recall (%)': metrics['spike_metrics']['recall'],
                'Spike Precision (%)': metrics['spike_metrics']['precision'],
                'Spike F1 (%)': metrics['spike_metrics']['f1_score'],
                'Spike AUC (%)': metrics['spike_metrics']['auc'],
                'False Negatives': metrics['spike_metrics']['fn'],
                'False Positives': metrics['spike_metrics']['fp'],
                'MAE': metrics['mae'],
                'MAPE (%)': metrics['mape'],
                'Spike MAE': metrics['spike_mae'],
                'Spike MAPE (%)': metrics['spike_mape']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Spike Recall (%)', ascending=False)
        
        # CSV 저장
        csv_path = f'spike_evaluation_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n📁 평가 결과 저장: {csv_path}")

# 메인 실행 함수
def main(test_data_path=None):
    """메인 실행 함수"""
    print("\n" + "="*70)
    print("급증 예측 모델 성능 평가 시스템 v3.0")
    print("="*70)
    
    evaluator = SpikeModelEvaluator()
    
    # 평가 실행
    results = evaluator.evaluate_all_models(test_data_path)
    
    if results:
        print("\n✅ 모든 평가가 성공적으로 완료되었습니다!")
        return results
    else:
        print("\n❌ 평가 실행 중 오류가 발생했습니다.")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        print(f"사용자 지정 테스트 데이터: {test_path}")
        main(test_path)
    else:
        main()