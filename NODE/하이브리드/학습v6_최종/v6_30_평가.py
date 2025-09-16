"""
V6 모델 평가 시스템 (수정판)
- 학습된 모델 로드
- 새로운 평가 데이터로 성능 측정
- 상세한 분석 리포트 생성
- CPU 모드 최적화
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# TensorFlow 경고 억제
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*60)
print("📊 V6 모델 평가 시스템 (수정판)")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*60)

# ============================================
# GPU/CPU 설정
# ============================================
def setup_compute():
    """계산 환경 설정"""
    print("\n🎮 계산 환경 확인...")
    
    # CPU 모드 강제 (필요시)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU 감지: {len(gpus)}개")
            return True
        except Exception as e:
            print(f"⚠️ GPU 설정 오류: {e}")
            print("💻 CPU 모드로 전환")
            return False
    else:
        print("💻 CPU 모드로 실행")
        return False

has_gpu = setup_compute()

# ============================================
# 설정
# ============================================
class Config:
    # 평가 데이터 파일
    EVAL_DATA_FILE = './data/20250731_to20250806.CSV'  # 실제 파일명에 맞게 수정
    
    # 학습된 모델 경로
    MODEL_DIR = './models_v6_full_train/'
    
    # 시퀀스 설정
    LOOKBACK = 100  # 과거 100분 데이터
    FORECAST = 10   # 10분 후 예측
    
    # 평가 결과 저장 경로
    EVAL_RESULT_DIR = './evaluation_results/'
    
    # 시각화 저장 경로
    PLOT_DIR = './evaluation_plots/'
    
    # CPU 모드 배치 크기
    BATCH_SIZE = 32  # CPU에서는 작은 배치 크기 사용

# 디렉토리 생성
os.makedirs(Config.EVAL_RESULT_DIR, exist_ok=True)
os.makedirs(Config.PLOT_DIR, exist_ok=True)

# ============================================
# 커스텀 레이어 정의 (모델 로드용)
# ============================================
class M14RuleCorrection(tf.keras.layers.Layer):
    """M14 규칙 기반 보정 레이어"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, training=None):
        pred, m14_features = inputs
        
        pred = tf.cast(pred, tf.float32)
        m14_features = tf.cast(m14_features, tf.float32)
        
        m14b = m14_features[:, 0:1]
        m10a = m14_features[:, 1:2]
        m16 = m14_features[:, 2:3]
        ratio = m14_features[:, 3:4] if m14_features.shape[1] > 3 else \
                tf.where(m10a > 0, m14b / (m10a + 1e-7), 0.0)
        
        # 임계값 규칙
        pred = tf.where(m14b >= 420, tf.maximum(pred, 1550.0), pred)
        pred = tf.where(m14b >= 380, tf.maximum(pred, 1500.0), pred)
        pred = tf.where(m14b >= 350, tf.maximum(pred, 1450.0), pred)
        pred = tf.where(m14b >= 300, tf.maximum(pred, 1400.0), pred)
        
        # 비율 보정
        pred = tf.where(ratio >= 5.5, pred * 1.15, pred)
        pred = tf.where((ratio >= 5.0) & (ratio < 5.5), pred * 1.10, pred)
        pred = tf.where((ratio >= 4.5) & (ratio < 5.0), pred * 1.08, pred)
        pred = tf.where((ratio >= 4.0) & (ratio < 4.5), pred * 1.05, pred)
        
        # 황금 패턴
        golden = (m14b >= 350) & (m10a < 70)
        pred = tf.where(golden, pred * 1.2, pred)
        
        # 범위 제한
        pred = tf.clip_by_value(pred, 1200.0, 2000.0)
        
        return pred

# 커스텀 손실 함수 (모델 로드용)
class WeightedLoss(tf.keras.losses.Loss):
    """가중치 손실 함수"""
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        mae = tf.abs(y_true - y_pred)
        
        # 가중치
        weights = tf.ones_like(y_true)
        weights = tf.where(y_true >= 1550, 30.0, weights)
        weights = tf.where((y_true >= 1500) & (y_true < 1550), 25.0, weights)
        weights = tf.where((y_true >= 1450) & (y_true < 1500), 20.0, weights)
        weights = tf.where((y_true >= 1400) & (y_true < 1450), 15.0, weights)
        weights = tf.where((y_true >= 1350) & (y_true < 1400), 10.0, weights)
        
        # 큰 오차 페널티
        large_error = tf.where(mae > 100, mae * 0.2, 0.0)
        
        return tf.reduce_mean(mae * weights + large_error)

# ============================================
# 데이터 전처리 함수
# ============================================
def prepare_evaluation_data(file_path):
    """평가 데이터 준비"""
    print(f"\n📂 평가 데이터 로드: {file_path}")
    
    # 데이터 로드
    df = pd.read_csv(file_path)
    print(f"  원본 데이터: {len(df)}행")
    
    # 필요한 컬럼 확인 및 처리
    required_columns = ['M14AM10A', 'M14AM14B', 'M14AM16', 'TOTALCNT']
    
    # 컬럼 존재 여부 확인
    for col in required_columns:
        if col not in df.columns:
            print(f"  ⚠️ {col} 컬럼 없음 - 0으로 초기화")
            df[col] = 0
    
    # M14AM14BSUM 생성
    if 'M14AM14BSUM' not in df.columns:
        df['M14AM14BSUM'] = df['M14AM14B'] + df['M14AM10A']
    
    # 타겟 생성 (10분 후 TOTALCNT)
    df['target'] = df['TOTALCNT'].shift(-Config.FORECAST)
    
    # 특징 엔지니어링
    print("\n🔧 특징 엔지니어링...")
    
    # 기본 특징
    df['ratio_14B_10A'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
    df['ratio_14B_16'] = df['M14AM14B'] / (df['M14AM16'] + 1)
    df['ratio_10A_16'] = df['M14AM10A'] / (df['M14AM16'] + 1)
    
    # 시계열 특징
    for col in ['TOTALCNT', 'M14AM14B', 'M14AM10A', 'M14AM16']:
        if col in df.columns:
            # 변화량
            df[f'{col}_diff_1'] = df[col].diff(1)
            df[f'{col}_diff_5'] = df[col].diff(5)
            df[f'{col}_diff_10'] = df[col].diff(10)
            
            # 이동평균
            df[f'{col}_ma_5'] = df[col].rolling(5, min_periods=1).mean()
            df[f'{col}_ma_10'] = df[col].rolling(10, min_periods=1).mean()
            df[f'{col}_ma_20'] = df[col].rolling(20, min_periods=1).mean()
            
            # 표준편차
            df[f'{col}_std_5'] = df[col].rolling(5, min_periods=1).std()
            df[f'{col}_std_10'] = df[col].rolling(10, min_periods=1).std()
    
    # 황금 패턴
    df['golden_pattern'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)
    
    # 급증 신호
    thresholds_14b = [250, 300, 350, 400, 450]
    for threshold in thresholds_14b:
        df[f'signal_{threshold}'] = (df['M14AM14B'] >= threshold).astype(float)
    
    thresholds_ratio = [3.5, 4.0, 4.5, 5.0, 5.5]
    for threshold in thresholds_ratio:
        df[f'ratio_signal_{threshold}'] = (df['ratio_14B_10A'] >= threshold).astype(float)
    
    # 결측치 처리
    df = df.fillna(0)
    
    # 타겟이 있는 데이터만 선택
    df = df.dropna(subset=['target'])
    
    print(f"  전처리 완료: {len(df)}행, {len(df.columns)}개 특징")
    
    return df

def create_sequences(df, lookback=100, forecast=10):
    """시퀀스 생성 (수정된 버전)"""
    print("\n⚡ 평가 시퀀스 생성 중...")
    
    X, y = [], []
    
    # numpy array로 변환
    data_array = df.values
    
    # TOTALCNT 컬럼 인덱스 찾기
    totalcnt_idx = df.columns.get_loc('TOTALCNT')
    
    for i in range(len(data_array) - lookback - forecast + 1):
        # 시퀀스 입력 (100분)
        X.append(data_array[i:i+lookback])
        
        # 타겟 (10분 후 TOTALCNT)
        target_idx = i + lookback + forecast - 1
        if target_idx < len(data_array):
            y.append(data_array[target_idx, totalcnt_idx])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y 범위: {y.min():.0f} ~ {y.max():.0f}")
    
    return X, y, df

# ============================================
# 모델 로드 함수
# ============================================
def load_models():
    """학습된 모델 로드"""
    print("\n📦 학습된 모델 로드 중...")
    
    models = {}
    model_names = ['lstm', 'gru', 'cnn_lstm', 'spike', 'rule', 'ensemble']
    
    # 커스텀 객체 정의
    custom_objects = {
        'M14RuleCorrection': M14RuleCorrection,
        'WeightedLoss': WeightedLoss,
    }
    
    for name in model_names:
        model_path = f"{Config.MODEL_DIR}{name}_final.keras"
        if os.path.exists(model_path):
            try:
                models[name] = tf.keras.models.load_model(
                    model_path,
                    custom_objects=custom_objects,
                    compile=False
                )
                # 재컴파일 (평가용)
                models[name].compile(
                    optimizer='adam',
                    loss='mae',
                    metrics=['mae']
                )
                print(f"  ✅ {name} 모델 로드 성공")
            except Exception as e:
                print(f"  ❌ {name} 모델 로드 실패: {e}")
        else:
            print(f"  ⚠️ {name} 모델 파일 없음: {model_path}")
    
    return models

# ============================================
# 평가 함수
# ============================================
def evaluate_models(models, X_test, y_test, m14_test):
    """모델 평가"""
    print("\n📊 모델 평가 시작...")
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\n  평가 중: {name.upper()}")
        
        try:
            # 예측 (배치 처리로 메모리 효율 개선)
            if name in ['ensemble', 'rule']:
                # Rule과 Ensemble은 두 개의 입력 필요
                pred = model.predict(
                    [X_test, m14_test], 
                    batch_size=Config.BATCH_SIZE,
                    verbose=0
                ).flatten()
            else:
                # 나머지 모델은 하나의 입력
                pred = model.predict(
                    X_test, 
                    batch_size=Config.BATCH_SIZE,
                    verbose=0
                ).flatten()
            
            predictions[name] = pred
            
            # 전체 성능 지표
            mae = np.mean(np.abs(y_test - pred))
            mse = np.mean((y_test - pred) ** 2)
            rmse = np.sqrt(mse)
            
            # MAPE 계산 (0 division 방지)
            non_zero_mask = y_test != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_test[non_zero_mask] - pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
            else:
                mape = 0
            
            # 정확도 (오차 기준)
            accuracy_50 = np.mean(np.abs(y_test - pred) <= 50) * 100
            accuracy_100 = np.mean(np.abs(y_test - pred) <= 100) * 100
            
            # 구간별 성능
            level_performance = {}
            for level in [1300, 1400, 1450, 1500, 1550]:
                mask = y_test >= level
                if np.any(mask):
                    # Recall: 실제 급증 중 예측 성공
                    tp = np.sum((pred >= level) & mask)
                    fn = np.sum((pred < level) & mask)
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    # Precision: 예측 급증 중 실제 급증
                    fp = np.sum((pred >= level) & ~mask)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    
                    # F1 Score
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # 구간 MAE
                    level_mae = np.mean(np.abs(y_test[mask] - pred[mask]))
                    
                    level_performance[level] = {
                        'recall': recall,
                        'precision': precision,
                        'f1': f1,
                        'mae': level_mae,
                        'count': np.sum(mask),
                        'tp': int(tp),
                        'fp': int(fp),
                        'fn': int(fn)
                    }
            
            results[name] = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'accuracy_50': float(accuracy_50),
                'accuracy_100': float(accuracy_100),
                'levels': level_performance
            }
            
            print(f"    MAE: {mae:.2f}")
            print(f"    RMSE: {rmse:.2f}")
            print(f"    MAPE: {mape:.2f}%")
            print(f"    정확도(±50): {accuracy_50:.1f}%")
            print(f"    정확도(±100): {accuracy_100:.1f}%")
            
        except Exception as e:
            print(f"    ❌ 평가 실패: {e}")
            import traceback
            traceback.print_exc()
    
    return results, predictions

# ============================================
# 시각화 함수
# ============================================
def create_visualizations(y_test, predictions, results):
    """평가 결과 시각화"""
    print("\n📈 시각화 생성 중...")
    
    try:
        # 폰트 설정 (한글 지원)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 모델별 성능 비교
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1-1. MAE 비교
        ax = axes[0, 0]
        model_names = list(results.keys())
        mae_values = [results[name]['mae'] for name in model_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(model_names, mae_values, color=colors)
        ax.set_title('Model MAE Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE')
        ax.set_xticklabels(model_names, rotation=45)
        for bar, value in zip(bars, mae_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{value:.1f}', ha='center', va='bottom')
        
        # 1-2. 정확도 비교
        ax = axes[0, 1]
        acc_50 = [results[name]['accuracy_50'] for name in model_names]
        acc_100 = [results[name]['accuracy_100'] for name in model_names]
        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width/2, acc_50, width, label='±50 Accuracy', color='skyblue')
        ax.bar(x + width/2, acc_100, width, label='±100 Accuracy', color='lightcoral')
        ax.set_title('Model Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        
        # 1-3. 1400+ 급증 감지 성능
        ax = axes[0, 2]
        f1_scores = []
        for name in model_names:
            if 1400 in results[name]['levels']:
                f1_scores.append(results[name]['levels'][1400]['f1'] * 100)
            else:
                f1_scores.append(0)
        bars = ax.bar(model_names, f1_scores, color='green', alpha=0.7)
        ax.set_title('1400+ Spike Detection F1 Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score (%)')
        ax.set_xticklabels(model_names, rotation=45)
        for bar, value in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{value:.1f}%', ha='center', va='bottom')
        
        # 2. 예측 vs 실제 (최고 성능 모델)
        if results:
            best_model = min(results.keys(), key=lambda x: results[x]['mae'])
            
            # 2-1. 산점도
            ax = axes[1, 0]
            sample_size = min(500, len(y_test))
            sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
            ax.scatter(y_test[sample_idx], predictions[best_model][sample_idx], 
                      alpha=0.5, s=10)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', label='Perfect Prediction')
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{best_model.upper()} - Predicted vs Actual', fontsize=12, fontweight='bold')
            ax.legend()
            
            # 2-2. 시계열 예측
            ax = axes[1, 1]
            time_sample = min(200, len(y_test))
            time_range = range(time_sample)
            ax.plot(time_range, y_test[:time_sample], label='Actual', linewidth=2)
            ax.plot(time_range, predictions[best_model][:time_sample], 
                   label=f'{best_model} Predicted', linewidth=2, alpha=0.7)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('TOTALCNT')
            ax.set_title('Time Series Prediction Sample', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2-3. 오차 분포
            ax = axes[1, 2]
            errors = predictions[best_model] - y_test
            ax.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', label='Error = 0')
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{best_model.upper()} Error Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            ax.text(0.05, 0.95, f'Mean: {mean_error:.1f}\nStd: {std_error:.1f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 저장
        save_path = f'{Config.PLOT_DIR}model_evaluation_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ 시각화 저장: {save_path}")
        
        # 표시 (선택사항)
        # plt.show()
        plt.close()
        
    except Exception as e:
        print(f"  ❌ 시각화 생성 실패: {e}")
        import traceback
        traceback.print_exc()

def create_detailed_report(results, y_test):
    """상세 평가 리포트 생성"""
    print("\n📝 상세 리포트 생성 중...")
    
    report = []
    report.append("="*80)
    report.append("V6 모델 평가 리포트")
    report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80)
    report.append("")
    
    # 데이터 개요
    report.append("📊 평가 데이터 개요")
    report.append(f"  - 총 샘플 수: {len(y_test):,}개")
    report.append(f"  - TOTALCNT 범위: {y_test.min():.0f} ~ {y_test.max():.0f}")
    report.append(f"  - TOTALCNT 평균: {y_test.mean():.0f}")
    report.append(f"  - TOTALCNT 표준편차: {y_test.std():.0f}")
    report.append("")
    
    # 급증 데이터 분포
    report.append("📈 급증 데이터 분포")
    for level in [1300, 1400, 1450, 1500, 1550]:
        count = np.sum(y_test >= level)
        ratio = count / len(y_test) * 100
        report.append(f"  - {level}+ : {count:,}개 ({ratio:.1f}%)")
    report.append("")
    
    # 모델별 성능
    report.append("🏆 모델별 성능 평가")
    report.append("-"*80)
    
    if results:
        # 최고 성능 모델 찾기
        best_model = min(results.keys(), key=lambda x: results[x]['mae'])
        
        for name, metrics in results.items():
            is_best = " ⭐ BEST" if name == best_model else ""
            report.append(f"\n📍 {name.upper()} 모델{is_best}")
            report.append(f"  전체 성능:")
            report.append(f"    - MAE: {metrics['mae']:.2f}")
            report.append(f"    - RMSE: {metrics['rmse']:.2f}")
            report.append(f"    - MAPE: {metrics['mape']:.2f}%")
            report.append(f"    - 정확도(±50): {metrics['accuracy_50']:.1f}%")
            report.append(f"    - 정확도(±100): {metrics['accuracy_100']:.1f}%")
            
            report.append(f"  \n  급증 감지 성능:")
            for level, perf in metrics['levels'].items():
                if perf['count'] > 0:
                    report.append(f"    {level}+ 감지:")
                    report.append(f"      - Recall: {perf['recall']:.1%}")
                    report.append(f"      - Precision: {perf['precision']:.1%}")
                    report.append(f"      - F1 Score: {perf['f1']:.1%}")
                    report.append(f"      - MAE: {perf['mae']:.1f}")
                    report.append(f"      - 샘플 수: {perf['count']:,}개")
        
        report.append("")
        report.append("="*80)
        report.append("🎯 결론")
        report.append(f"  최고 성능 모델: {best_model.upper()}")
        report.append(f"  MAE: {results[best_model]['mae']:.2f}")
        report.append(f"  정확도(±50): {results[best_model]['accuracy_50']:.1f}%")
        
        # 개선 권장사항
        report.append("")
        report.append("💡 개선 권장사항")
        
        # CNN-LSTM이나 Spike 성능이 낮은 경우
        if 'cnn_lstm' in results and results['cnn_lstm']['mae'] > 1000:
            report.append("  - CNN-LSTM 모델 재학습 필요 (구조 단순화 권장)")
        if 'spike' in results and results['spike']['mae'] > 1000:
            report.append("  - Spike Detector 임계값 재조정 필요")
        
        # 앙상블 개선 여부
        if 'ensemble' in results and 'lstm' in results:
            if results['ensemble']['mae'] > results['lstm']['mae']:
                report.append("  - 앙상블 가중치 재조정 권장 (CNN/Spike 제외)")
    
    report.append("="*80)
    
    # 리포트 저장
    report_text = "\n".join(report)
    report_path = f"{Config.EVAL_RESULT_DIR}evaluation_report.txt"
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"  ✅ 리포트 저장: {report_path}")
    except Exception as e:
        print(f"  ❌ 리포트 저장 실패: {e}")
    
    # 결과 JSON 저장
    json_path = f"{Config.EVAL_RESULT_DIR}evaluation_results.json"
    try:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  ✅ JSON 결과 저장: {json_path}")
    except Exception as e:
        print(f"  ❌ JSON 저장 실패: {e}")
    
    return report_text

# ============================================
# 메인 실행 함수
# ============================================
def main():
    """메인 평가 프로세스"""
    
    try:
        # 1. 평가 데이터 준비
        df = prepare_evaluation_data(Config.EVAL_DATA_FILE)
        
        # 2. 시퀀스 생성
        X, y, df_processed = create_sequences(df, Config.LOOKBACK, Config.FORECAST)
        
        if len(X) == 0:
            print("\n❌ 시퀀스 생성 실패 - 데이터가 부족합니다.")
            print(f"   필요한 최소 데이터: {Config.LOOKBACK + Config.FORECAST}행")
            print(f"   현재 데이터: {len(df)}행")
            return
        
        # 3. M14 특징 추출
        print("\n📊 M14 특징 추출 중...")
        m14_features = np.zeros((len(X), 4), dtype=np.float32)
        
        for i in range(len(X)):
            idx = i + Config.LOOKBACK
            if idx < len(df_processed):
                m14_features[i] = [
                    df_processed['M14AM14B'].iloc[idx],
                    df_processed['M14AM10A'].iloc[idx],
                    df_processed['M14AM16'].iloc[idx],
                    df_processed['ratio_14B_10A'].iloc[idx]
                ]
        
        # 4. 데이터 스케일링
        print("\n📏 데이터 스케일링...")
        
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[2]):
            scaler = RobustScaler()
            feature = X[:, :, i].reshape(-1, 1)
            X_scaled[:, :, i] = scaler.fit_transform(feature).reshape(X[:, :, i].shape)
        
        m14_scaler = RobustScaler()
        m14_features_scaled = m14_scaler.fit_transform(m14_features)
        
        print("  ✅ 스케일링 완료")
        
        # 5. 모델 로드
        models = load_models()
        
        if not models:
            print("\n❌ 로드된 모델이 없습니다. 학습을 먼저 실행하세요.")
            return
        
        # 6. 모델 평가
        results, predictions = evaluate_models(models, X_scaled, y, m14_features_scaled)
        
        # 7. 시각화
        if results and predictions:
            create_visualizations(y, predictions, results)
        
        # 8. 상세 리포트
        if results:
            report = create_detailed_report(results, y)
        
        # 9. 결과 출력
        print("\n" + "="*60)
        print("📊 평가 완료!")
        print("="*60)
        
        # 최고 모델 강조
        if results:
            best_model = min(results.keys(), key=lambda x: results[x]['mae'])
            print(f"\n🏆 최고 성능 모델: {best_model.upper()}")
            print(f"  - MAE: {results[best_model]['mae']:.2f}")
            print(f"  - RMSE: {results[best_model]['rmse']:.2f}")
            print(f"  - 정확도(±50): {results[best_model]['accuracy_50']:.1f}%")
            print(f"  - 정확도(±100): {results[best_model]['accuracy_100']:.1f}%")
            
            # 급증 감지 성능
            if 1400 in results[best_model]['levels']:
                perf_1400 = results[best_model]['levels'][1400]
                print(f"\n  1400+ 급증 감지:")
                print(f"    - Recall: {perf_1400['recall']:.1%}")
                print(f"    - Precision: {perf_1400['precision']:.1%}")
                print(f"    - F1 Score: {perf_1400['f1']:.1%}")
        
        print("\n📁 결과 저장 위치:")
        print(f"  - 리포트: {Config.EVAL_RESULT_DIR}evaluation_report.txt")
        print(f"  - JSON: {Config.EVAL_RESULT_DIR}evaluation_results.json")
        print(f"  - 시각화: {Config.PLOT_DIR}model_evaluation_summary.png")
        
        print("\n✅ 모든 평가 작업 완료!")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n❌ 평가 데이터 파일을 찾을 수 없습니다: {Config.EVAL_DATA_FILE}")
        print("파일 경로를 확인하세요.")
    except Exception as e:
        print(f"\n❌ 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()