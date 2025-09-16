"""
V6 모델 평가 시스템
- 학습된 모델 로드
- 새로운 평가 데이터로 성능 측정
- 상세한 분석 리포트 생성
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

print("="*60)
print("📊 V6 모델 평가 시스템")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*60)

# ============================================
# GPU 설정
# ============================================
def setup_gpu():
    """GPU 설정 및 확인"""
    print("\n🎮 GPU 환경 확인...")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU 감지: {len(gpus)}개")
            return True
        except Exception as e:
            print(f"⚠️ GPU 설정 오류: {e}")
            return False
    else:
        print("💻 CPU 모드로 실행")
        return False

has_gpu = setup_gpu()

# ============================================
# 설정
# ============================================
class Config:
    # 평가 데이터 파일
    EVAL_DATA_FILE = './data/20250731_to20250826.csv'
    
    # 학습된 모델 경로
    MODEL_DIR = './models_v6_full_train/'
    
    # 시퀀스 설정
    LOOKBACK = 100  # 과거 100분 데이터
    FORECAST = 10   # 10분 후 예측
    
    # 평가 결과 저장 경로
    EVAL_RESULT_DIR = './evaluation_results/'
    
    # 시각화 저장 경로
    PLOT_DIR = './evaluation_plots/'

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

def create_sequences(data, lookback=100, forecast=10):
    """시퀀스 생성"""
    X, y = [], []
    
    for i in range(len(data) - lookback - forecast):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback+forecast-1, data.columns.get_loc('TOTALCNT')])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

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
                print(f"  ✅ {name} 모델 로드 성공")
            except Exception as e:
                print(f"  ❌ {name} 모델 로드 실패: {e}")
        else:
            print(f"  ⚠️ {name} 모델 파일 없음")
    
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
        
        # 예측
        try:
            if name in ['ensemble', 'rule']:
                pred = model.predict([X_test, m14_test], verbose=0).flatten()
            else:
                pred = model.predict(X_test, verbose=0).flatten()
            
            predictions[name] = pred
            
            # 전체 성능 지표
            mae = np.mean(np.abs(y_test - pred))
            mse = np.mean((y_test - pred) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - pred) / (y_test + 1e-7))) * 100
            
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
                        'tp': tp,
                        'fp': fp,
                        'fn': fn
                    }
            
            results[name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'accuracy_50': accuracy_50,
                'accuracy_100': accuracy_100,
                'levels': level_performance
            }
            
            print(f"    MAE: {mae:.2f}")
            print(f"    RMSE: {rmse:.2f}")
            print(f"    MAPE: {mape:.2f}%")
            print(f"    정확도(±50): {accuracy_50:.1f}%")
            print(f"    정확도(±100): {accuracy_100:.1f}%")
            
        except Exception as e:
            print(f"    ❌ 평가 실패: {e}")
    
    return results, predictions

# ============================================
# 시각화 함수
# ============================================
def create_visualizations(y_test, predictions, results):
    """평가 결과 시각화"""
    print("\n📈 시각화 생성 중...")
    
    # 1. 모델별 성능 비교
    plt.figure(figsize=(15, 10))
    
    # 1-1. MAE 비교
    plt.subplot(2, 3, 1)
    model_names = list(results.keys())
    mae_values = [results[name]['mae'] for name in model_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    bars = plt.bar(model_names, mae_values, color=colors)
    plt.title('모델별 MAE (평균 절대 오차)', fontsize=12, fontweight='bold')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    for bar, value in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:.1f}', ha='center', va='bottom')
    
    # 1-2. 정확도 비교
    plt.subplot(2, 3, 2)
    acc_50 = [results[name]['accuracy_50'] for name in model_names]
    acc_100 = [results[name]['accuracy_100'] for name in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    plt.bar(x - width/2, acc_50, width, label='±50 정확도', color='skyblue')
    plt.bar(x + width/2, acc_100, width, label='±100 정확도', color='lightcoral')
    plt.title('모델별 예측 정확도', fontsize=12, fontweight='bold')
    plt.ylabel('정확도 (%)')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    
    # 1-3. 1400+ 급증 감지 성능
    plt.subplot(2, 3, 3)
    f1_scores = []
    for name in model_names:
        if 1400 in results[name]['levels']:
            f1_scores.append(results[name]['levels'][1400]['f1'] * 100)
        else:
            f1_scores.append(0)
    bars = plt.bar(model_names, f1_scores, color='green', alpha=0.7)
    plt.title('1400+ 급증 감지 F1 Score', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score (%)')
    plt.xticks(rotation=45)
    for bar, value in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:.1f}%', ha='center', va='bottom')
    
    # 2. 예측 vs 실제 (최고 성능 모델)
    best_model = min(results.keys(), key=lambda x: results[x]['mae'])
    
    plt.subplot(2, 3, 4)
    sample_size = min(500, len(y_test))
    sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
    plt.scatter(y_test[sample_idx], predictions[best_model][sample_idx], 
               alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
            'r--', label='완벽한 예측')
    plt.xlabel('실제값')
    plt.ylabel('예측값')
    plt.title(f'{best_model.upper()} - 예측 vs 실제', fontsize=12, fontweight='bold')
    plt.legend()
    
    # 3. 시계열 예측 (샘플)
    plt.subplot(2, 3, 5)
    time_sample = 200
    time_range = range(time_sample)
    plt.plot(time_range, y_test[:time_sample], label='실제', linewidth=2)
    plt.plot(time_range, predictions[best_model][:time_sample], 
            label=f'{best_model} 예측', linewidth=2, alpha=0.7)
    plt.xlabel('시간 (분)')
    plt.ylabel('TOTALCNT')
    plt.title('시계열 예측 샘플 (200분)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 오차 분포
    plt.subplot(2, 3, 6)
    errors = predictions[best_model] - y_test
    plt.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='오차 = 0')
    plt.xlabel('예측 오차')
    plt.ylabel('빈도')
    plt.title(f'{best_model.upper()} 오차 분포', fontsize=12, fontweight='bold')
    plt.legend()
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.text(0.05, 0.95, f'평균: {mean_error:.1f}\n표준편차: {std_error:.1f}',
            transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'{Config.PLOT_DIR}model_evaluation_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  ✅ 시각화 저장: {Config.PLOT_DIR}model_evaluation_summary.png")

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
            report.append("  - 앙상블 가중치 재조정 권장")
    
    report.append("="*80)
    
    # 리포트 저장
    report_text = "\n".join(report)
    report_path = f"{Config.EVAL_RESULT_DIR}evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"  ✅ 리포트 저장: {report_path}")
    
    # 결과 JSON 저장
    json_path = f"{Config.EVAL_RESULT_DIR}evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  ✅ JSON 결과 저장: {json_path}")
    
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
        print("\n⚡ 평가 시퀀스 생성 중...")
        data = df.values
        X, y = create_sequences(df, Config.LOOKBACK, Config.FORECAST)
        
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        
        # 3. M14 특징 추출
        m14_features = np.zeros((len(X), 4), dtype=np.float32)
        for i in range(len(X)):
            idx = i + Config.LOOKBACK
            if idx < len(df):
                m14_features[i] = [
                    df['M14AM14B'].iloc[idx],
                    df['M14AM10A'].iloc[idx],
                    df['M14AM16'].iloc[idx],
                    df['ratio_14B_10A'].iloc[idx]
                ]
        
        # 4. 데이터 스케일링
        print("\n📏 데이터 스케일링...")
        
        # 스케일러 로드 또는 새로 생성
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
        create_visualizations(y, predictions, results)
        
        # 8. 상세 리포트
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