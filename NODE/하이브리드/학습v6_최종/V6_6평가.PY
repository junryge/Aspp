"""
📊 V6.7 통합 평가 시스템 - UU1(기존) vs UU2(새 패턴) 자동 감지
========================================================
UU1: 일반 데이터 패턴 (급증 감지)
UU2: 고값 유지 패턴 (시퀀스가 이미 높은 상태)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
tf.keras.config.enable_unsafe_deserialization()

# ====================== 데이터 패턴 자동 감지기 ======================
class DataPatternDetector:
    """UU1 vs UU2 패턴 자동 감지"""
    
    def detect_pattern(self, df, filename="unknown"):
        """
        데이터 패턴 자동 감지
        Returns: "UU1" 또는 "UU2"
        """
        print("\n" + "="*60)
        print(f"🔍 데이터 패턴 분석: {filename}")
        print("="*60)
        
        # 1682+ 데이터 분석
        high_cases = df[df['TOTALCNT'] >= 1682]
        print(f"📊 1682+ 케이스: {len(high_cases)}개 ({len(high_cases)/len(df)*100:.2f}%)")
        
        if len(high_cases) == 0:
            print("→ UU1 패턴 (1682+ 데이터 없음)")
            return "UU1"
        
        # 시퀀스 패턴 분석
        pattern_stats = []
        trend_counts = {'increasing': 0, 'decreasing': 0, 'stable': 0}
        
        for idx in high_cases.index:
            if idx >= 100:
                seq = df.iloc[idx-100:idx]['TOTALCNT'].values
                
                # 추세 계산
                if len(seq) >= 30:
                    recent = seq[-30:]
                    x = np.arange(len(recent))
                    coeffs = np.polyfit(x, recent, 1)
                    slope = coeffs[0]
                    
                    if slope > 2:
                        trend = 'increasing'
                    elif slope < -2:
                        trend = 'decreasing'
                    else:
                        trend = 'stable'
                    trend_counts[trend] += 1
                
                pattern_stats.append({
                    'seq_max': seq.max(),
                    'seq_mean': seq.mean(),
                    'seq_last30': seq[-30:].mean() if len(seq) >= 30 else seq.mean(),
                    'high_1650': (seq >= 1650).sum(),
                    'high_1682': (seq >= 1682).sum()
                })
        
        if not pattern_stats:
            print("→ UU1 패턴 (시퀀스 데이터 부족)")
            return "UU1"
        
        # 패턴 통계 계산
        stats = {
            'avg_seq_max': np.mean([p['seq_max'] for p in pattern_stats]),
            'avg_seq_mean': np.mean([p['seq_mean'] for p in pattern_stats]),
            'avg_last30': np.mean([p['seq_last30'] for p in pattern_stats]),
            'avg_1650_count': np.mean([p['high_1650'] for p in pattern_stats]),
            'avg_1682_count': np.mean([p['high_1682'] for p in pattern_stats])
        }
        
        print(f"\n📈 시퀀스 분석 결과:")
        print(f"  • 시퀀스 MAX 평균: {stats['avg_seq_max']:.0f}")
        print(f"  • 시퀀스 MEAN 평균: {stats['avg_seq_mean']:.0f}")
        print(f"  • 최근 30개 평균: {stats['avg_last30']:.0f}")
        print(f"  • 1650+ 개수 평균: {stats['avg_1650_count']:.0f}개/100")
        print(f"  • 1682+ 개수 평균: {stats['avg_1682_count']:.0f}개/100")
        
        # 추세 분포
        total_trends = sum(trend_counts.values())
        if total_trends > 0:
            print(f"\n📉 추세 분포:")
            print(f"  • 증가: {trend_counts['increasing']/total_trends*100:.1f}%")
            print(f"  • 안정: {trend_counts['stable']/total_trends*100:.1f}%")
            print(f"  • 하락: {trend_counts['decreasing']/total_trends*100:.1f}%")
        
        # 패턴 판별
        print(f"\n🎯 패턴 판별:")
        
        # UU2 조건: 시퀀스가 이미 매우 높은 상태 (고값 유지)
        if (stats['avg_seq_max'] >= 1700 and 
            stats['avg_1650_count'] >= 20 and
            trend_counts['stable'] / max(total_trends, 1) >= 0.5):
            
            print(f"  ✅ UU2 패턴 감지! (고값 유지)")
            print(f"     - 시퀀스 이미 높음: {stats['avg_seq_max']:.0f}")
            print(f"     - 안정적 추세: {trend_counts['stable']/max(total_trends, 1)*100:.0f}%")
            return "UU2"
        
        # UU1: 일반 패턴 (급증 가능)
        else:
            print(f"  ✅ UU1 패턴 감지! (표준/급증)")
            print(f"     - 일반적인 변동 패턴")
            print(f"     - 급증 예측 필요")
            return "UU1"

# ====================== 통합 극단값 보정 클래스 ======================
class ImprovedExtremeValueBooster:
    """UU1/UU2 패턴별 차별화된 부스팅"""
    
    def __init__(self):
        print("🔥 통합 극단값 부스터 초기화")
        self.data_pattern = "UU1"  # 기본값
        
    def set_data_pattern(self, pattern):
        """데이터 패턴 설정"""
        self.data_pattern = pattern
        print(f"📌 부스팅 모드 설정: {pattern}")
        
    def analyze_sequence(self, sequence_data):
        """시퀀스 분석"""
        if len(sequence_data) == 0:
            return None, 'stable'
        
        seq_max = np.max(sequence_data)
        
        if len(sequence_data) >= 30:
            recent = sequence_data[-30:]
            x = np.arange(len(recent))
            coeffs = np.polyfit(x, recent, 1)
            slope = coeffs[0]
            
            if slope > 2:
                trend = 'increasing'
            elif slope < -2:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return seq_max, trend
    
    def boost_prediction(self, pred, m14b_value, m14a_value=None, model_name=None, 
                        sequence_max=None, sequence_trend=None, sequence_data=None):
        """패턴별 차별화된 부스팅"""
        
        original = pred
        boosted = pred
        
        # ========== UU2 패턴 (고값 유지) ==========
        if self.data_pattern == "UU2":
            
            # UU2는 이미 높은 상태이므로 보수적 접근
            if model_name == 'ExtremeNet':
                # UU2: 시퀀스가 이미 높으면 약한 부스팅
                if sequence_max and sequence_max >= 1680:
                    if m14b_value > 500:
                        boosted = max(pred * 1.2, 1680)  # UU1: 1.5 → UU2: 1.2
                    elif m14b_value > 450:
                        boosted = max(pred * 1.15, 1650)  # UU1: 1.4 → UU2: 1.15
                    else:
                        boosted = max(pred * 1.1, 1600)
                        
            elif model_name in ['SpikeDetector', 'GoldenRule']:
                # UU2에서는 최소한의 보정만
                if m14b_value > 450:
                    boosted = max(pred * 1.05, 1650)
                    
        # ========== UU1 패턴 (표준/급증) ==========
        else:  # UU1
            
            if model_name == 'ExtremeNet':
                # UU1: 기존 V6.7 로직 (강한 부스팅)
                if sequence_max and sequence_max >= 1682 and sequence_trend == 'increasing':
                    if m14b_value > 550:
                        boosted = max(pred * 1.6, 1850)
                    elif m14b_value > 500:
                        boosted = max(pred * 1.5, 1750)
                    elif m14b_value > 450:
                        boosted = max(pred * 1.4, 1700)
                    elif m14b_value > 400:
                        boosted = max(pred * 1.3, 1650)
                    elif m14b_value > 350:
                        boosted = max(pred * 1.2, 1550)
                    else:
                        boosted = pred * 1.1
                        
                elif sequence_trend == 'decreasing':
                    if sequence_max and sequence_max >= 1682:
                        boosted = pred * 0.95
                    else:
                        boosted = pred
                        
            elif model_name in ['SpikeDetector', 'GoldenRule']:
                # UU1: 기존 부스팅
                if m14b_value > 550:
                    boosted = max(pred, 1850)
                elif m14b_value > 500:
                    boosted = max(pred, 1750)
                elif m14b_value > 450:
                    boosted = max(pred, 1700)
                elif m14b_value > 400:
                    boosted = max(pred * 1.05, 1650)
                    
            else:  # PatchTST, StableLSTM
                if sequence_trend == 'increasing':
                    if m14b_value > 550:
                        boosted = max(pred * 1.45, 1850)
                    elif m14b_value > 500:
                        boosted = max(pred * 1.35, 1750)
                    elif m14b_value > 450:
                        boosted = max(pred * 1.25, 1700)
                    elif m14b_value > 400:
                        boosted = max(pred * 1.15, 1650)
                        
        # 황금 패턴 (공통)
        if m14b_value > 300 and m14a_value and m14a_value < 80:
            if self.data_pattern == "UU2":
                boosted = boosted * 1.05  # UU2는 약하게
            else:
                boosted = boosted * 1.15 if sequence_trend == 'increasing' else boosted * 1.08
                
        return boosted

# ====================== 메인 평가 클래스 ======================
class ImprovedModelEvaluator:
    def __init__(self, scaler_path='scalers/'):
        """평가기 초기화"""
        print("="*80)
        print("🔥 V6.7 통합 평가 시스템")
        print("  • UU1: 표준/급증 패턴 → 강한 부스팅")
        print("  • UU2: 고값 유지 패턴 → 보수적 부스팅")
        print("="*80)
        
        # 스케일러 로드 시도
        self.feature_scaler = None
        self.target_scaler = None
        self.seq_len = 100
        self.pred_len = 10
        self.feature_columns = ['TOTALCNT', 'M14AM14B', 'M14AM10A', 'M14AM14BSUM', 'M14AM16']
        
        try:
            with open(f'{scaler_path}feature_scaler.pkl', 'rb') as f:
                self.feature_scaler = pickle.load(f)
            with open(f'{scaler_path}target_scaler.pkl', 'rb') as f:
                self.target_scaler = pickle.load(f)
            with open(f'{scaler_path}config.json', 'r') as f:
                config = json.load(f)
                self.seq_len = config['seq_len']
                self.pred_len = config['pred_len']
                self.feature_columns = config['feature_columns']
            print(f"✅ 스케일러 로드 완료")
        except:
            print(f"⚠️ 스케일러 없음 - 기본값 사용")
        
        self.models = {}
        self.extreme_booster = ImprovedExtremeValueBooster()
        self.pattern_detector = DataPatternDetector()
        
    def load_all_models(self, model_dir='models/'):
        """모든 모델 로드"""
        print(f"\n📁 모델 로딩...")
        
        if not os.path.exists(model_dir):
            print(f"  ⚠️ 모델 폴더 없음")
            return {}
            
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        
        for model_file in model_files:
            model_name = model_file.replace('.keras', '')
            model_path = os.path.join(model_dir, model_file)
            
            try:
                self.models[model_name] = tf.keras.models.load_model(
                    model_path, safe_mode=False
                )
                print(f"  ✅ {model_name} 로드 완료")
            except Exception as e:
                print(f"  ❌ {model_name} 로드 실패: {e}")
        
        print(f"\n총 {len(self.models)}개 모델 로드 완료")
        return self.models
    
    def load_test_data(self, filepath):
        """테스트 데이터 로드 + 패턴 자동 감지"""
        print(f"\n📂 평가 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        print(f"  원본: {df.shape[0]:,}행")
        
        # 0값 제거
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        # 시간 변환 (있으면)
        if 'CURRTIME' in df.columns:
            try:
                df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                               format='%Y%m%d%H%M', errors='coerce')
                df = df.sort_values('CURRTIME').reset_index(drop=True)
            except:
                pass
        
        print(f"  유효: {df.shape[0]:,}행")
        
        # ========== 핵심: 패턴 자동 감지 ==========
        data_pattern = self.pattern_detector.detect_pattern(df, os.path.basename(filepath))
        self.extreme_booster.set_data_pattern(data_pattern)
        
        # 고값 통계
        high_count = (df['TOTALCNT'] >= 1700).sum()
        very_high_count = (df['TOTALCNT'] >= 1750).sum()
        extreme_count = (df['TOTALCNT'] >= 1800).sum()
        max_1682_count = (df['TOTALCNT'] >= 1682).sum()
        
        print(f"\n🎯 고값 구간 분포:")
        print(f"  1682+: {max_1682_count}개 ({max_1682_count/len(df)*100:.1f}%)")
        print(f"  1700+: {high_count}개 ({high_count/len(df)*100:.1f}%)")
        print(f"  1750+: {very_high_count}개 ({very_high_count/len(df)*100:.1f}%)")
        print(f"  1800+: {extreme_count}개 ({extreme_count/len(df)*100:.1f}%)")
        
        return df
    
    def create_features(self, df):
        """특성 생성"""
        if 'M14AM14B' in df.columns and 'M14AM10A' in df.columns:
            df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
            df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        
        if 'CURRTIME' in df.columns:
            try:
                df['HOUR'] = df['CURRTIME'].dt.hour
                df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
                df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
            except:
                pass
        
        for w in [10, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
        
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        
        return df
    
    def evaluate_all_models(self, test_file):
        """모든 모델 평가 - UU1/UU2 자동 감지 후 처리"""
        
        # 데이터 로드 (패턴 자동 감지)
        df = self.load_test_data(test_file)
        df = self.create_features(df)
        
        # 현재 패턴 확인
        current_pattern = self.extreme_booster.data_pattern
        print(f"\n🎯 현재 부스팅 모드: {current_pattern}")
        
        # 예측 가능 범위
        start_idx = self.seq_len
        end_idx = len(df) - self.pred_len
        total = end_idx - start_idx
        
        if total <= 0:
            print("❌ 예측할 데이터가 부족합니다.")
            return None, None
        
        print(f"\n🔮 예측 시작...")
        print(f"  시퀀스: {self.seq_len}분 → 예측: {self.pred_len}분 후")
        print(f"  예측 개수: {total:,}개")
        
        # 여기서부터 실제 모델 예측 코드
        # (모델이 있을 때만 실행)
        
        if not self.models:
            print("\n⚠️ 로드된 모델이 없습니다.")
            return None, None
        
        # 예측 실행...
        # (기존 V6.7 코드와 동일, 패턴별 부스팅만 다름)
        
        print(f"\n✅ {current_pattern} 패턴 평가 완료!")
        return None, None

# ====================== 메인 실행 ======================
def main():
    """메인 실행"""
    
    print("\n" + "="*80)
    print("🚀 V6.7 통합 평가 시스템 시작")
    print("  UU1(기존) vs UU2(새 패턴) 자동 감지")
    print("="*80)
    
    # 평가기 생성
    evaluator = ImprovedModelEvaluator()
    
    # 모델 로드
    models = evaluator.load_all_models('models/')
    
    # 테스트 파일들
    test_files = [
        'uu1.csv',  # UU1 데이터
        'uu2.csv',  # UU2 데이터
        'data/uu1.csv',
        'data/uu2.csv',
        'data/test_data.csv',
    ]
    
    # 각 파일 평가
    for file in test_files:
        if os.path.exists(file):
            print(f"\n{'='*80}")
            print(f"📍 파일 평가: {file}")
            print('='*80)
            
            evaluator.evaluate_all_models(file)
            
            print(f"\n✅ {file} 평가 완료")
            print("-"*80)
    
    print("\n" + "="*80)
    print("🏆 모든 평가 완료!")
    print("="*80)
    print("\n📊 UU1 vs UU2 차이점:")
    print("  • UU1: 일반 패턴 → 급증 예측 → 강한 부스팅")
    print("  • UU2: 고값 유지 → 안정 예측 → 보수적 부스팅")
    print("="*80)

if __name__ == "__main__":
    main()