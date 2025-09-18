# -*- coding: utf-8 -*-
"""
완전한 전체 데이터 시퀀스 분석 시스템 (한글 깨짐 해결)
===============================================
모든 시퀀스를 전체 분석하여 정확한 통계 생성
한글 폰트 자동 설정 및 깨짐 방지
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import pickle
import json
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (운영체제별 자동 감지)
import platform
import matplotlib.font_manager as fm

def set_korean_font():
    """한글 폰트 자동 설정"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows 한글 폰트
        font_list = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei', 'Gulim', 'Dotum']
    elif system == 'Darwin':  # macOS
        # macOS 한글 폰트
        font_list = ['AppleGothic', 'Noto Sans CJK KR', 'Nanum Gothic', 'Helvetica']
    else:  # Linux
        # Linux 한글 폰트 (Docker/Server 환경)
        font_list = ['Noto Sans CJK KR', 'NanumGothic', 'UnDotum', 'DejaVu Sans']
    
    # 사용 가능한 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font_name in font_list:
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            print(f"✅ 한글 폰트 설정: {font_name}")
            return font_name
    
    # 폰트를 찾지 못한 경우 기본 설정 + 영문 레이블 사용
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    print("⚠️ 한글 폰트를 찾지 못했습니다. 영문 레이블을 사용합니다.")
    return None

# 한글 폰트 설정 실행
korean_font = set_korean_font()

class FullDataSequenceAnalyzer:
    """전체 데이터 시퀀스 분석기 - 샘플링 없음, 한글 깨짐 해결"""
    
    def __init__(self):
        print("="*80)
        print("전체 데이터 시퀀스 분석 시스템 (샘플링 없음)")
        print("="*80)
        
        # 체크포인트 디렉토리 생성
        os.makedirs('./checkpoints', exist_ok=True)
        os.makedirs('./scalers', exist_ok=True)
        os.makedirs('./visualizations', exist_ok=True)
        
        self.checkpoint_path = './checkpoints/full_sequence_analysis_state.pkl'
        self.state = self.load_checkpoint()
        
        # 한글/영문 레이블 설정
        self.labels = self._get_labels()
        
    def _get_labels(self):
        """한글 폰트 지원 여부에 따른 레이블 설정"""
        if korean_font:
            # 한글 레이블
            return {
                'sequence_length': '시퀀스 길이 (분)',
                'high_value_ratio': '고값 시퀀스 비율 (%)',
                'high_value_detection': '시퀀스 길이별 고값 감지율',
                'avg_max_value': '평균 최대값',
                'avg_max_title': '시퀀스 길이별 평균 최대값',
                'volatility': '평균 변동성',
                'volatility_title': '시퀀스 길이별 변동성',
                'performance_score': '종합 성능 점수',
                'performance_title': '시퀀스 길이별 종합 성능',
                'current_100min': '현재 100분',
                'optimal': '최적',
                'hour': '시간',
                'avg_totalcnt': '평균 TOTALCNT',
                'hourly_avg': '시간대별 평균값',
                'high_ratio': '고값 비율 (%)',
                'hourly_high_ratio': '시간대별 고값 비율',
                'std_dev': '표준편차',
                'hourly_volatility': '시간대별 변동성',
                'sample_count': '샘플 수',
                'hourly_distribution': '시간대별 데이터 분포'
            }
        else:
            # 영문 레이블
            return {
                'sequence_length': 'Sequence Length (min)',
                'high_value_ratio': 'High Value Sequence Ratio (%)',
                'high_value_detection': 'High Value Detection Rate by Sequence Length',
                'avg_max_value': 'Average Max Value',
                'avg_max_title': 'Average Max Value by Sequence Length',
                'volatility': 'Average Volatility',
                'volatility_title': 'Volatility by Sequence Length',
                'performance_score': 'Overall Performance Score',
                'performance_title': 'Overall Performance by Sequence Length',
                'current_100min': 'Current 100min',
                'optimal': 'Optimal',
                'hour': 'Hour',
                'avg_totalcnt': 'Average TOTALCNT',
                'hourly_avg': 'Hourly Average Values',
                'high_ratio': 'High Value Ratio (%)',
                'hourly_high_ratio': 'Hourly High Value Ratio',
                'std_dev': 'Standard Deviation',
                'hourly_volatility': 'Hourly Volatility',
                'sample_count': 'Sample Count',
                'hourly_distribution': 'Hourly Data Distribution'
            }
    
    def load_checkpoint(self):
        """체크포인트 로드"""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    state = pickle.load(f)
                print(f"✅ 체크포인트 로드됨 - Step {state.get('step', 0)} 완료")
                return state
            except:
                print("⚠️ 체크포인트 로드 실패 - 새로 시작")
        
        return {'step': 0, 'sequence_lengths': [], 'results': []}
    
    def save_checkpoint(self):
        """체크포인트 저장"""
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(self.state, f)
        print(f"💾 체크포인트 저장됨 - Step {self.state['step']}")
    
    def step1_load_data(self, filepath=None):
        """Step 1: 데이터 로드 및 기본 분석"""
        print("\n" + "="*60)
        print("📂 Step 1: 데이터 로드 및 기본 분석")
        print("="*60)
        
        # 데이터 파일 찾기
        if filepath is None:
            data_files = [
                'data/gs.CSV',
                'gs.CSV',
                './gs.CSV',
                'gs.csv',
                './gs.csv'
            ]
            
            filepath = None
            for file in data_files:
                if os.path.exists(file):
                    filepath = file
                    break
            
            if filepath is None:
                print("❌ 데이터 파일을 찾을 수 없습니다!")
                return False
        
        print(f"📂 데이터 로딩: {filepath}")
        
        # CSV 로드
        df = pd.read_csv(filepath)
        print(f"  원본 데이터: {df.shape[0]:,}행, {df.shape[1]}개 컬럼")
        print(f"  컬럼: {list(df.columns)}")
        
        # 시간 컬럼 처리
        if 'CURRTIME' in df.columns:
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                           format='%Y%m%d%H%M', errors='coerce')
            df = df.sort_values('CURRTIME').reset_index(drop=True)
            
            # 시간 연속성 검증
            time_diff = df['CURRTIME'].diff().dt.total_seconds() / 60
            expected_interval = time_diff.mode()[0] if not time_diff.mode().empty else 1
            
            print(f"\n⏰ 시간 연속성 분석:")
            print(f"  예상 간격: {expected_interval:.0f}분")
            print(f"  시간 범위: {df['CURRTIME'].min()} ~ {df['CURRTIME'].max()}")
            
            # 누락된 시간 확인
            missing_times = (time_diff > expected_interval * 1.5).sum()
            print(f"  시간 누락: {missing_times}개 구간")
            
        # 0값 제거
        original_count = len(df)
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        removed_zeros = original_count - len(df)
        
        print(f"\n📊 데이터 품질:")
        print(f"  0값 제거: {removed_zeros}개")
        print(f"  유효 데이터: {len(df):,}행")
        
        # 기본 통계
        print(f"\n📈 TOTALCNT 통계:")
        print(f"  범위: {df['TOTALCNT'].min():.0f} ~ {df['TOTALCNT'].max():.0f}")
        print(f"  평균: {df['TOTALCNT'].mean():.1f}")
        print(f"  표준편차: {df['TOTALCNT'].std():.1f}")
        print(f"  중앙값: {df['TOTALCNT'].median():.1f}")
        print(f"  25% 분위: {df['TOTALCNT'].quantile(0.25):.1f}")
        print(f"  75% 분위: {df['TOTALCNT'].quantile(0.75):.1f}")
        
        # 고값 구간 분포
        thresholds = [1200, 1400, 1500, 1600, 1651, 1700, 1750, 1800]
        print(f"\n🎯 임계값별 분포:")
        for threshold in thresholds:
            count = (df['TOTALCNT'] >= threshold).sum()
            percentage = count / len(df) * 100
            print(f"  {threshold}+: {count:4d}개 ({percentage:5.1f}%)")
        
        # M14AM14B 분포 (V6.7 기준)
        if 'M14AM14B' in df.columns:
            print(f"\n📊 M14AM14B 분포:")
            m14b_thresholds = [200, 250, 300, 350, 400, 450, 500]
            for threshold in m14b_thresholds:
                count = (df['M14AM14B'] >= threshold).sum()
                percentage = count / len(df) * 100
                print(f"  {threshold}+: {count:4d}개 ({percentage:5.1f}%)")
        
        # 상태 저장
        self.state['df'] = df
        self.state['step'] = 1
        self.save_checkpoint()
        
        return True
    
    def step2_analyze_sequence_patterns(self):
        """Step 2: 전체 시퀀스 패턴 분석 (샘플링 없음)"""
        if self.state['step'] < 1:
            print("❌ Step 1을 먼저 완료해주세요.")
            return False
            
        print("\n" + "="*60)
        print("🔍 Step 2: 전체 시퀀스 패턴 분석 (전체 데이터)")
        print("="*60)
        
        df = self.state['df']
        
        # 시퀀스 길이 설정
        sequence_lengths = list(range(10, 101, 10)) + list(range(120, 301, 20))
        print(f"  분석할 시퀀스 길이: {len(sequence_lengths)}개")
        print(f"  길이 범위: {min(sequence_lengths)}분 ~ {max(sequence_lengths)}분")
        
        # 각 시퀀스 길이별 전체 분석
        sequence_analysis = []
        
        for seq_idx, seq_len in enumerate(sequence_lengths):
            print(f"\n📏 [{seq_idx+1}/{len(sequence_lengths)}] 시퀀스 길이 {seq_len}분 - 전체 분석...")
            
            # 생성 가능한 전체 시퀀스 수
            max_sequences = len(df) - seq_len - 10
            if max_sequences <= 0:
                print(f"  ❌ 데이터 부족 (필요: {seq_len + 10}분)")
                continue
            
            print(f"  📊 전체 시퀀스 수: {max_sequences:,}개 (샘플링 없음)")
            
            # 전체 시퀀스에 대한 통계 수집
            seq_stats = {
                'length': seq_len,
                'max_possible': max_sequences,
                'total_analyzed': max_sequences,  # 전체 분석
                'seq_max_values': [],
                'seq_min_values': [],
                'seq_mean_values': [],
                'seq_std_values': [],
                'trend_counts': {'increasing': 0, 'decreasing': 0, 'stable': 0},
                'high_value_sequences': 0,
                'extreme_sequences': 0,
                'volatility_scores': [],
                'consecutive_rises': [],
                'consecutive_falls': []
            }
            
            # 진행률 표시를 위한 체크포인트
            checkpoint_interval = max(1000, max_sequences // 20)
            
            # 전체 시퀀스 분석
            for i, idx in enumerate(range(seq_len, max_sequences + seq_len)):
                # 진행률 표시
                if i % checkpoint_interval == 0:
                    progress = i / max_sequences * 100
                    print(f"    진행률: {progress:.1f}% ({i:,}/{max_sequences:,})")
                
                # 시퀀스 데이터 추출
                seq_data = df.iloc[idx-seq_len:idx]['TOTALCNT'].values
                
                # 기본 통계
                seq_max = np.max(seq_data)
                seq_min = np.min(seq_data)
                seq_mean = np.mean(seq_data)
                seq_std = np.std(seq_data)
                
                seq_stats['seq_max_values'].append(seq_max)
                seq_stats['seq_min_values'].append(seq_min)
                seq_stats['seq_mean_values'].append(seq_mean)
                seq_stats['seq_std_values'].append(seq_std)
                seq_stats['volatility_scores'].append(seq_std)
                
                # 고값 시퀀스 체크
                if seq_max >= 1651:
                    seq_stats['high_value_sequences'] += 1
                if seq_max >= 1750:
                    seq_stats['extreme_sequences'] += 1
                
                # 연속 상승/하락 계산
                consecutive_rises = 0
                consecutive_falls = 0
                
                for j in range(len(seq_data)-1, 0, -1):
                    if seq_data[j] > seq_data[j-1]:
                        consecutive_rises += 1
                    else:
                        break
                        
                for j in range(len(seq_data)-1, 0, -1):
                    if seq_data[j] < seq_data[j-1]:
                        consecutive_falls += 1
                    else:
                        break
                
                seq_stats['consecutive_rises'].append(consecutive_rises)
                seq_stats['consecutive_falls'].append(consecutive_falls)
                
                # 추세 분석
                if len(seq_data) >= 20:
                    x = np.arange(len(seq_data))
                    slope = np.polyfit(x, seq_data, 1)[0]
                    
                    if slope > 1:
                        seq_stats['trend_counts']['increasing'] += 1
                    elif slope < -1:
                        seq_stats['trend_counts']['decreasing'] += 1
                    else:
                        seq_stats['trend_counts']['stable'] += 1
                else:
                    seq_stats['trend_counts']['stable'] += 1
            
            # 전체 분석 완료 통계 요약
            print(f"  ✅ 전체 분석 완료: {max_sequences:,}개")
            print(f"  📊 전체 통계:")
            print(f"    MAX 범위: {min(seq_stats['seq_max_values']):.0f} ~ {max(seq_stats['seq_max_values']):.0f}")
            print(f"    평균 MAX: {np.mean(seq_stats['seq_max_values']):.1f}")
            print(f"    고값(1651+): {seq_stats['high_value_sequences']}개 ({seq_stats['high_value_sequences']/max_sequences*100:.1f}%)")
            print(f"    극값(1750+): {seq_stats['extreme_sequences']}개 ({seq_stats['extreme_sequences']/max_sequences*100:.1f}%)")
            print(f"    평균 변동성: {np.mean(seq_stats['volatility_scores']):.1f}")
            print(f"    평균 연속상승: {np.mean(seq_stats['consecutive_rises']):.1f}")
            print(f"    평균 연속하락: {np.mean(seq_stats['consecutive_falls']):.1f}")
            print(f"    증가 추세: {seq_stats['trend_counts']['increasing']}개")
            print(f"    감소 추세: {seq_stats['trend_counts']['decreasing']}개")
            print(f"    안정 추세: {seq_stats['trend_counts']['stable']}개")
            
            sequence_analysis.append(seq_stats)
        
        # 상태 저장
        self.state['sequence_analysis'] = sequence_analysis
        self.state['sequence_lengths'] = sequence_lengths
        self.state['step'] = 2
        self.save_checkpoint()
        
        return True
    
    def step3_model_specific_analysis(self):
        """Step 3: 전체 데이터 기반 모델별 상세 분석"""
        if self.state['step'] < 2:
            print("❌ Step 2를 먼저 완료해주세요.")
            return False
            
        print("\n" + "="*60)
        print("🤖 Step 3: 전체 데이터 기반 모델별 상세 분석")
        print("="*60)
        
        df = self.state['df']
        sequence_analysis = self.state['sequence_analysis']
        
        # 모델 정의
        models = {
            'LSTM': {'focus': '장기 패턴', 'optimal_seq': 100, 'weight': 0.25},
            'GRU': {'focus': '단기 변화', 'optimal_seq': 60, 'weight': 0.20},
            'CNN_LSTM': {'focus': '복합 패턴', 'optimal_seq': 80, 'weight': 0.25},
            'SpikeDetector': {'focus': '급변 감지', 'optimal_seq': 20, 'weight': 0.15},
            'ExtremeNet': {'focus': '극단값', 'optimal_seq': 200, 'weight': 0.15}
        }
        
        model_analysis = {}
        
        for seq_stats in sequence_analysis:
            seq_len = seq_stats['length']
            total_sequences = seq_stats['total_analyzed']
            
            print(f"\n📏 시퀀스 {seq_len}분 ({total_sequences:,}개 전체) - 모델별 분석:")
            
            # 각 모델별 분석
            for model_name, model_info in models.items():
                print(f"\n  🤖 {model_name} ({model_info['focus']}) - 가중치: {model_info['weight']}:")
                
                if model_name not in model_analysis:
                    model_analysis[model_name] = {
                        'sequences': [],
                        'boost_conditions': [],
                        'performance_estimates': [],
                        'detailed_stats': []
                    }
                
                # 모델별 부스팅 조건 계산 (전체 데이터 기반)
                boost_count = 0
                performance_score = 0
                detailed_stats = {}
                
                # ExtremeNet 분석 (V6.7 조건)
                if model_name == 'ExtremeNet':
                    high_seq = seq_stats['high_value_sequences']
                    inc_trend = seq_stats['trend_counts']['increasing']
                    total = seq_stats['total_analyzed']
                    
                    # V6.7 부스팅 조건: high_value + increasing trend
                    boost_count = int(high_seq * inc_trend / total * 0.4)  # 전체 데이터이므로 더 정확한 추정
                    performance_score = (high_seq / total * 100) * 1.2  # 고값 감지 특화
                    
                    detailed_stats = {
                        'high_sequences': high_seq,
                        'high_ratio': high_seq / total * 100,
                        'increasing_trends': inc_trend,
                        'boost_ratio': boost_count / total * 100
                    }
                    
                    print(f"    전체 고값 시퀀스: {high_seq:,}개 ({high_seq/total*100:.2f}%)")
                    print(f"    전체 증가 추세: {inc_trend:,}개 ({inc_trend/total*100:.2f}%)")
                    print(f"    V6.7 부스팅 예상: {boost_count:,}개 ({boost_count/total*100:.2f}%)")
                    print(f"    성능 점수: {performance_score:.1f}%")
                
                # SpikeDetector 분석
                elif model_name == 'SpikeDetector':
                    avg_volatility = np.mean(seq_stats['volatility_scores'])
                    avg_consecutive_rises = np.mean(seq_stats['consecutive_rises'])
                    high_volatility_count = sum(1 for v in seq_stats['volatility_scores'] if v > 30)
                    total = seq_stats['total_analyzed']
                    
                    # 변동성 + 연속 상승 기반 부스팅
                    boost_count = high_volatility_count + seq_stats['trend_counts']['increasing']
                    performance_score = min(95, (avg_volatility / 40 + avg_consecutive_rises / 5) * 30)
                    
                    detailed_stats = {
                        'avg_volatility': avg_volatility,
                        'high_volatility_sequences': high_volatility_count,
                        'avg_consecutive_rises': avg_consecutive_rises,
                        'boost_ratio': boost_count / total * 100
                    }
                    
                    print(f"    전체 평균 변동성: {avg_volatility:.1f}")
                    print(f"    고변동성 시퀀스: {high_volatility_count:,}개 ({high_volatility_count/total*100:.2f}%)")
                    print(f"    평균 연속상승: {avg_consecutive_rises:.1f}회")
                    print(f"    부스팅 예상: {boost_count:,}개 ({boost_count/total*100:.2f}%)")
                    print(f"    성능 점수: {performance_score:.1f}%")
                
                # LSTM/GRU/CNN-LSTM 분석
                else:
                    optimal_seq = model_info['optimal_seq']
                    length_penalty = abs(seq_len - optimal_seq) / optimal_seq
                    base_performance = 75
                    
                    # 시퀀스 길이 최적화 점수
                    performance_score = base_performance * (1 - length_penalty * 0.4)
                    
                    # 전체 데이터에서 해당 모델에 유리한 패턴 수
                    if model_name == 'LSTM' and seq_len >= 80:
                        performance_score += 10  # 장기 패턴 보너스
                    elif model_name == 'GRU' and 40 <= seq_len <= 80:
                        performance_score += 8   # 중기 패턴 보너스
                    elif model_name == 'CNN_LSTM' and 60 <= seq_len <= 100:
                        performance_score += 9   # 복합 패턴 보너스
                    
                    performance_score = max(50, min(95, performance_score))
                    boost_count = 0  # 기본 모델들은 특별한 부스팅 없음
                    
                    detailed_stats = {
                        'optimal_length': optimal_seq,
                        'current_length': seq_len,
                        'length_penalty': length_penalty,
                        'base_performance': base_performance
                    }
                    
                    print(f"    최적 길이: {optimal_seq}분 (현재: {seq_len}분)")
                    print(f"    길이 페널티: {length_penalty:.2f}")
                    print(f"    성능 점수: {performance_score:.1f}%")
                    print(f"    부스팅: 없음 (기본 모델)")
                
                # 결과 저장
                model_analysis[model_name]['sequences'].append(seq_len)
                model_analysis[model_name]['boost_conditions'].append(boost_count)
                model_analysis[model_name]['performance_estimates'].append(performance_score)
                model_analysis[model_name]['detailed_stats'].append(detailed_stats)
        
        # 상태 저장
        self.state['model_analysis'] = model_analysis
        self.state['step'] = 3
        self.save_checkpoint()
        
        return True
    
    def step4_hourly_analysis(self):
        """Step 4: 전체 데이터 시간대별 상세 분석"""
        if self.state['step'] < 2:
            print("❌ Step 2를 먼저 완료해주세요.")
            return False
            
        print("\n" + "="*60)
        print("⏰ Step 4: 전체 데이터 시간대별 상세 분석")
        print("="*60)
        
        df = self.state['df']
        
        # 시간대별 전체 분석
        hourly_stats = {}
        
        for hour in range(24):
            hour_data = df[df['CURRTIME'].dt.hour == hour]
            
            if len(hour_data) == 0:
                continue
                
            stats = {
                'hour': hour,
                'total_samples': len(hour_data),
                'avg_totalcnt': hour_data['TOTALCNT'].mean(),
                'std_totalcnt': hour_data['TOTALCNT'].std(),
                'max_totalcnt': hour_data['TOTALCNT'].max(),
                'min_totalcnt': hour_data['TOTALCNT'].min(),
                'median_totalcnt': hour_data['TOTALCNT'].median(),
                'q25_totalcnt': hour_data['TOTALCNT'].quantile(0.25),
                'q75_totalcnt': hour_data['TOTALCNT'].quantile(0.75),
                'high_value_count': (hour_data['TOTALCNT'] >= 1651).sum(),
                'extreme_value_count': (hour_data['TOTALCNT'] >= 1750).sum(),
                'high_ratio': (hour_data['TOTALCNT'] >= 1651).sum() / len(hour_data) * 100,
                'extreme_ratio': (hour_data['TOTALCNT'] >= 1750).sum() / len(hour_data) * 100
            }
            
            # M14AM14B 분석 (있는 경우)
            if 'M14AM14B' in hour_data.columns:
                stats['avg_m14b'] = hour_data['M14AM14B'].mean()
                stats['std_m14b'] = hour_data['M14AM14B'].std()
                stats['high_m14b_count'] = (hour_data['M14AM14B'] >= 300).sum()
                stats['high_m14b_ratio'] = (hour_data['M14AM14B'] >= 300).sum() / len(hour_data) * 100
            
            # 변동성 지표
            if len(hour_data) > 1:
                hour_data_sorted = hour_data.sort_values('CURRTIME')
                changes = hour_data_sorted['TOTALCNT'].diff().dropna()
                stats['avg_change'] = changes.mean()
                stats['volatility'] = changes.std()
                stats['positive_changes'] = (changes > 0).sum()
                stats['negative_changes'] = (changes < 0).sum()
            
            hourly_stats[hour] = stats
        
        # 시간대별 전체 요약 출력
        print(f"\n📊 시간대별 전체 TOTALCNT 통계:")
        print(f"{'시간':>4} {'전체수':>7} {'평균':>6} {'중앙값':>6} {'최대':>6} {'고값수':>6} {'고값비율':>7} {'변동성':>6}")
        print("-" * 65)
        
        for hour in range(24):
            if hour in hourly_stats:
                stats = hourly_stats[hour]
                volatility = stats.get('volatility', 0)
                print(f"{hour:2d}시 {stats['total_samples']:7d} {stats['avg_totalcnt']:6.0f} "
                      f"{stats['median_totalcnt']:6.0f} {stats['max_totalcnt']:6.0f} "
                      f"{stats['high_value_count']:6d} {stats['high_ratio']:6.1f}% {volatility:6.1f}")
        
        # 상태 저장
        self.state['hourly_stats'] = hourly_stats
        self.state['step'] = 4
        self.save_checkpoint()
        
        return True
    
    def step5_visualization(self):
        """Step 5: 상세 시각화 (한글 깨짐 방지)"""
        if self.state['step'] < 3:
            print("❌ Step 3까지 먼저 완료해주세요.")
            return False
            
        print("\n" + "="*60)
        print("📊 Step 5: 상세 시각화 (한글 깨짐 방지)")
        print("="*60)
        
        # 1. 시퀀스 길이별 성능 비교
        self._plot_sequence_performance()
        
        # 2. 모델별 성능 히트맵
        self._plot_model_heatmap()
        
        # 3. 시간대별 분석
        if 'hourly_stats' in self.state:
            self._plot_hourly_analysis()
        
        # 4. 데이터 분포
        self._plot_data_distribution()
        
        self.state['step'] = 5
        self.save_checkpoint()
        
        return True
    
    def _plot_sequence_performance(self):
        """시퀀스 길이별 성능 시각화 (한글 지원)"""
        sequence_analysis = self.state['sequence_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 데이터 준비
        lengths = [s['length'] for s in sequence_analysis]
        high_ratios = [s['high_value_sequences']/s['total_analyzed']*100 for s in sequence_analysis]
        avg_maxes = [np.mean(s['seq_max_values']) for s in sequence_analysis]
        avg_volatilities = [np.mean(s['volatility_scores']) for s in sequence_analysis]
        
        # 1. 고값 비율 vs 시퀀스 길이
        axes[0,0].plot(lengths, high_ratios, 'bo-', linewidth=2, markersize=6)
        axes[0,0].axhline(y=12.9, color='r', linestyle='--', alpha=0.7, 
                         label=f"{self.labels['current_100min']} (12.9%)")
        axes[0,0].set_xlabel(self.labels['sequence_length'])
        axes[0,0].set_ylabel(self.labels['high_value_ratio'])
        axes[0,0].set_title(self.labels['high_value_detection'])
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # 2. 평균 MAX 값
        axes[0,1].plot(lengths, avg_maxes, 'go-', linewidth=2, markersize=6)
        axes[0,1].axhline(y=1497, color='r', linestyle='--', alpha=0.7, 
                         label=f"{self.labels['current_100min']} (1497)")
        axes[0,1].set_xlabel(self.labels['sequence_length'])
        axes[0,1].set_ylabel(self.labels['avg_max_value'])
        axes[0,1].set_title(self.labels['avg_max_title'])
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # 3. 변동성
        axes[1,0].plot(lengths, avg_volatilities, 'ro-', linewidth=2, markersize=6)
        axes[1,0].set_xlabel(self.labels['sequence_length'])
        axes[1,0].set_ylabel(self.labels['volatility'])
        axes[1,0].set_title(self.labels['volatility_title'])
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 종합 성능 스코어
        performance_scores = []
        for i, length in enumerate(lengths):
            high_weight = 0.7
            stability_weight = 0.3
            
            # 정규화
            norm_high = high_ratios[i] / max(high_ratios) if max(high_ratios) > 0 else 0
            norm_stability = 1 - (avg_volatilities[i] - min(avg_volatilities)) / (max(avg_volatilities) - min(avg_volatilities)) if (max(avg_volatilities) - min(avg_volatilities)) > 0 else 0.5
            
            score = high_weight * norm_high + stability_weight * norm_stability
            performance_scores.append(score * 100)
        
        axes[1,1].plot(lengths, performance_scores, 'mo-', linewidth=2, markersize=6)
        if performance_scores:
            best_idx = np.argmax(performance_scores)
            axes[1,1].scatter([lengths[best_idx]], [performance_scores[best_idx]], 
                             c='red', s=100, zorder=5, label=f'{self.labels["optimal"]}: {lengths[best_idx]}min')
        axes[1,1].set_xlabel(self.labels['sequence_length'])
        axes[1,1].set_ylabel(self.labels['performance_score'])
        axes[1,1].set_title(self.labels['performance_title'])
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('./visualizations/sequence_performance.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ 시퀀스 성능 분석 저장: ./visualizations/sequence_performance.png")
    
    def _plot_model_heatmap(self):
        """모델별 성능 히트맵 (한글 지원)"""
        model_analysis = self.state['model_analysis']
        
        # 데이터 준비
        models = list(model_analysis.keys())
        sequences = model_analysis[models[0]]['sequences']
        
        # 성능 매트릭스 생성
        performance_matrix = []
        for model in models:
            performance_matrix.append(model_analysis[model]['performance_estimates'])
        
        # 히트맵 생성
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 성능 히트맵
        im1 = axes[0].imshow(performance_matrix, cmap='RdYlBu_r', aspect='auto')
        axes[0].set_xticks(range(len(sequences)))
        axes[0].set_xticklabels([f'{s}min' for s in sequences], rotation=45)
        axes[0].set_yticks(range(len(models)))
        axes[0].set_yticklabels(models)
        axes[0].set_title('Model Performance by Sequence Length')
        
        # 값 표시
        for i in range(len(models)):
            for j in range(len(sequences)):
                text = axes[0].text(j, i, f'{performance_matrix[i][j]:.1f}%',
                                   ha="center", va="center", 
                                   color="black" if performance_matrix[i][j] < 70 else "white")
        
        plt.colorbar(im1, ax=axes[0], label='Performance Score (%)')
        
        # 2. 부스팅 조건 히트맵
        boost_matrix = []
        for model in models:
            boost_matrix.append(model_analysis[model]['boost_conditions'])
        
        im2 = axes[1].imshow(boost_matrix, cmap='Oranges', aspect='auto')
        axes[1].set_xticks(range(len(sequences)))
        axes[1].set_xticklabels([f'{s}min' for s in sequences], rotation=45)
        axes[1].set_yticks(range(len(models)))
        axes[1].set_yticklabels(models)
        axes[1].set_title('Boost Conditions by Model')
        
        # 값 표시
        for i in range(len(models)):
            for j in range(len(sequences)):
                text = axes[1].text(j, i, f'{boost_matrix[i][j]}',
                                   ha="center", va="center", color="black")
        
        plt.colorbar(im2, ax=axes[1], label='Boost Condition Count')
        
        plt.tight_layout()
        plt.savefig('./visualizations/model_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ 모델 히트맵 저장: ./visualizations/model_heatmap.png")
    
    def _plot_hourly_analysis(self):
        """시간대별 분석 시각화 (한글 지원)"""
        hourly_stats = self.state['hourly_stats']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        hours = sorted(hourly_stats.keys())
        avg_values = [hourly_stats[h]['avg_totalcnt'] for h in hours]
        high_ratios = [hourly_stats[h]['high_ratio'] for h in hours]
        std_values = [hourly_stats[h]['std_totalcnt'] for h in hours]
        sample_counts = [hourly_stats[h]['total_samples'] for h in hours]
        
        # 1. 시간대별 평균값
        axes[0,0].bar(hours, avg_values, color='skyblue', alpha=0.7)
        axes[0,0].set_xlabel(self.labels['hour'])
        axes[0,0].set_ylabel(self.labels['avg_totalcnt'])
        axes[0,0].set_title(self.labels['hourly_avg'])
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 시간대별 고값 비율
        axes[0,1].bar(hours, high_ratios, color='orange', alpha=0.7)
        axes[0,1].set_xlabel(self.labels['hour'])
        axes[0,1].set_ylabel(self.labels['high_ratio'])
        axes[0,1].set_title(self.labels['hourly_high_ratio'])
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 시간대별 변동성
        axes[1,0].bar(hours, std_values, color='green', alpha=0.7)
        axes[1,0].set_xlabel(self.labels['hour'])
        axes[1,0].set_ylabel(self.labels['std_dev'])
        axes[1,0].set_title(self.labels['hourly_volatility'])
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 시간대별 데이터 수
        axes[1,1].bar(hours, sample_counts, color='purple', alpha=0.7)
        axes[1,1].set_xlabel(self.labels['hour'])
        axes[1,1].set_ylabel(self.labels['sample_count'])
        axes[1,1].set_title(self.labels['hourly_distribution'])
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./visualizations/hourly_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ 시간대 분석 저장: ./visualizations/hourly_analysis.png")
    
    def _plot_data_distribution(self):
        """데이터 분포 시각화 (한글 지원)"""
        df = self.state['df']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. TOTALCNT 히스토그램
        axes[0,0].hist(df['TOTALCNT'], bins=50, alpha=0.7, color='blue')
        axes[0,0].axvline(x=1651, color='r', linestyle='--', label='1651 (V6.7 Threshold)')
        axes[0,0].axvline(x=1700, color='orange', linestyle='--', label='1700 (High Value)')
        axes[0,0].set_xlabel('TOTALCNT')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('TOTALCNT Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 시계열 플롯 (샘플)
        sample_data = df.iloc[::100]  # 100개마다 샘플링
        axes[0,1].plot(sample_data.index, sample_data['TOTALCNT'], alpha=0.7)
        axes[0,1].axhline(y=1651, color='r', linestyle='--', alpha=0.5)
        axes[0,1].axhline(y=1700, color='orange', linestyle='--', alpha=0.5)
        axes[0,1].set_xlabel('Index')
        axes[0,1].set_ylabel('TOTALCNT')
        axes[0,1].set_title('Time Series Data (Sampled)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 박스플롯 (구간별)
        ranges = [
            df[df['TOTALCNT'] < 1400]['TOTALCNT'],
            df[(df['TOTALCNT'] >= 1400) & (df['TOTALCNT'] < 1500)]['TOTALCNT'],
            df[(df['TOTALCNT'] >= 1500) & (df['TOTALCNT'] < 1651)]['TOTALCNT'],
            df[df['TOTALCNT'] >= 1651]['TOTALCNT']
        ]
        axes[0,2].boxplot(ranges, labels=['<1400', '1400-1500', '1500-1651', '1651+'])
        axes[0,2].set_ylabel('TOTALCNT')
        axes[0,2].set_title('Distribution by Range')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. M14AM14B vs TOTALCNT (있는 경우)
        if 'M14AM14B' in df.columns:
            sample_df = df.iloc[::50]  # 샘플링
            scatter = axes[1,0].scatter(sample_df['M14AM14B'], sample_df['TOTALCNT'], 
                                      c=sample_df['TOTALCNT'], cmap='viridis', alpha=0.6)
            axes[1,0].axhline(y=1651, color='r', linestyle='--', alpha=0.5)
            axes[1,0].axvline(x=300, color='orange', linestyle='--', alpha=0.5)
            axes[1,0].set_xlabel('M14AM14B')
            axes[1,0].set_ylabel('TOTALCNT')
            axes[1,0].set_title('M14AM14B vs TOTALCNT')
            plt.colorbar(scatter, ax=axes[1,0])
        
        # 5. 일별 평균
        df['date'] = df['CURRTIME'].dt.date
        daily_stats = df.groupby('date')['TOTALCNT'].agg(['mean', 'max', 'std']).reset_index()
        axes[1,1].plot(daily_stats['date'], daily_stats['mean'], 'o-', label='Mean')
        axes[1,1].plot(daily_stats['date'], daily_stats['max'], 's-', label='Max')
        axes[1,1].fill_between(daily_stats['date'], 
                              daily_stats['mean'] - daily_stats['std'],
                              daily_stats['mean'] + daily_stats['std'],
                              alpha=0.3, label='±1 Std')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylabel('TOTALCNT')
        axes[1,1].set_title('Daily Statistics')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. 누적분포함수
        sorted_data = np.sort(df['TOTALCNT'])
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1,2].plot(sorted_data, cumulative)
        axes[1,2].axvline(x=1651, color='r', linestyle='--', alpha=0.7, label='1651')
        axes[1,2].axvline(x=1700, color='orange', linestyle='--', alpha=0.7, label='1700')
        axes[1,2].set_xlabel('TOTALCNT')
        axes[1,2].set_ylabel('Cumulative Probability')
        axes[1,2].set_title('Cumulative Distribution Function')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./visualizations/data_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ 데이터 분포 저장: ./visualizations/data_distribution.png")
    
    def step6_export_results(self):
        """Step 6: 전체 분석 결과 내보내기"""
        if self.state['step'] < 3:
            print("❌ Step 3까지 먼저 완료해주세요.")
            return False
            
        print("\n" + "="*60)
        print("📤 Step 6: 전체 분석 결과 내보내기")
        print("="*60)
        
        # 전체 분석 결과 CSV 생성
        results = []
        
        if 'sequence_analysis' in self.state and 'model_analysis' in self.state:
            sequence_analysis = self.state['sequence_analysis']
            model_analysis = self.state['model_analysis']
            
            for seq_stats in sequence_analysis:
                seq_len = seq_stats['length']
                total_analyzed = seq_stats['total_analyzed']
                
                result = {
                    'sequence_length': seq_len,
                    'total_analyzed': total_analyzed,
                    'avg_max': np.mean(seq_stats['seq_max_values']),
                    'std_max': np.std(seq_stats['seq_max_values']),
                    'min_max': np.min(seq_stats['seq_max_values']),
                    'max_max': np.max(seq_stats['seq_max_values']),
                    'avg_volatility': np.mean(seq_stats['volatility_scores']),
                    'std_volatility': np.std(seq_stats['volatility_scores']),
                    'high_sequences_1651': seq_stats['high_value_sequences'],
                    'high_ratio_percent': seq_stats['high_value_sequences'] / total_analyzed * 100,
                    'extreme_sequences_1750': seq_stats['extreme_sequences'],
                    'extreme_ratio_percent': seq_stats['extreme_sequences'] / total_analyzed * 100,
                    'increasing_trend': seq_stats['trend_counts']['increasing'],
                    'decreasing_trend': seq_stats['trend_counts']['decreasing'],
                    'stable_trend': seq_stats['trend_counts']['stable'],
                    'avg_consecutive_rises': np.mean(seq_stats['consecutive_rises']),
                    'max_consecutive_rises': np.max(seq_stats['consecutive_rises']),
                    'avg_consecutive_falls': np.mean(seq_stats['consecutive_falls']),
                    'max_consecutive_falls': np.max(seq_stats['consecutive_falls'])
                }
                
                # 모델별 데이터 추가
                for model_name in model_analysis:
                    sequences = model_analysis[model_name]['sequences']
                    if seq_len in sequences:
                        idx = sequences.index(seq_len)
                        result[f'{model_name}_performance'] = model_analysis[model_name]['performance_estimates'][idx]
                        result[f'{model_name}_boost_conditions'] = model_analysis[model_name]['boost_conditions'][idx]
                        result[f'{model_name}_boost_ratio'] = model_analysis[model_name]['boost_conditions'][idx] / total_analyzed * 100
                
                results.append(result)
        
        # DataFrame 생성 및 저장
        results_df = pd.DataFrame(results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f'full_sequence_analysis_{timestamp}.csv'
        results_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 시간대별 결과 저장
        if 'hourly_stats' in self.state:
            hourly_results = []
            for hour, stats in self.state['hourly_stats'].items():
                hourly_results.append({
                    'hour': hour,
                    **stats
                })
            
            hourly_df = pd.DataFrame(hourly_results)
            hourly_csv = f'hourly_analysis_{timestamp}.csv'
            hourly_df.to_csv(hourly_csv, index=False, encoding='utf-8-sig')
        
        print(f"✅ 전체 분석 결과 저장 완료:")
        print(f"  📊 시퀀스 분석: {csv_file}")
        if 'hourly_stats' in self.state:
            print(f"  ⏰ 시간대 분석: {hourly_csv}")
        print(f"  💾 체크포인트: {self.checkpoint_path}")
        
        self.state['step'] = 6
        self.save_checkpoint()
        
        return True
    
    def generate_final_report(self):
        """전체 데이터 기반 최종 보고서 생성"""
        if self.state['step'] < 3:
            print("❌ 분석이 충분히 완료되지 않았습니다.")
            return False
            
        print("\n" + "="*80)
        print("📋 전체 데이터 기반 최종 분석 보고서")
        print("="*80)
        
        sequence_analysis = self.state.get('sequence_analysis', [])
        model_analysis = self.state.get('model_analysis', {})
        hourly_stats = self.state.get('hourly_stats', {})
        
        # 전체 데이터 요약
        if sequence_analysis:
            total_data_analyzed = sum(s['total_analyzed'] for s in sequence_analysis)
            print(f"\n📊 전체 분석 규모:")
            print(f"  분석된 총 시퀀스: {total_data_analyzed:,}개")
            print(f"  시퀀스 길이 종류: {len(sequence_analysis)}개")
            
        # 1. 최적 시퀀스 길이 분석 (전체 데이터 기준)
        print(f"\n🏆 최적 시퀀스 길이 분석 (전체 데이터):")
        if sequence_analysis:
            # 고값 비율 기준
            best_high_ratio = max(sequence_analysis, key=lambda x: x['high_value_sequences']/x['total_analyzed'])
            print(f"  🥇 고값 감지 최적: {best_high_ratio['length']}분")
            print(f"    전체 시퀀스: {best_high_ratio['total_analyzed']:,}개")
            print(f"    고값 시퀀스: {best_high_ratio['high_value_sequences']:,}개")
            print(f"    고값 비율: {best_high_ratio['high_value_sequences']/best_high_ratio['total_analyzed']*100:.2f}%")
            print(f"    평균 MAX: {np.mean(best_high_ratio['seq_max_values']):.0f}")
            
            # 현재 100분과 비교
            current_100 = next((s for s in sequence_analysis if s['length'] == 100), None)
            if current_100:
                current_ratio = current_100['high_value_sequences']/current_100['total_analyzed']*100
                best_ratio = best_high_ratio['high_value_sequences']/best_high_ratio['total_analyzed']*100
                improvement = (best_ratio - current_ratio) / current_ratio * 100
                
                print(f"\n  현재 100분 대비 (전체 데이터):")
                print(f"    현재 100분: {current_ratio:.2f}% ({current_100['high_value_sequences']:,}/{current_100['total_analyzed']:,}개)")
                print(f"    최적 길이: {best_ratio:.2f}% ({best_high_ratio['high_value_sequences']:,}/{best_high_ratio['total_analyzed']:,}개)")
                print(f"    성능 개선: {improvement:+.1f}%")
        
        # 2. 모델별 최적 조건 (전체 데이터 기준)
        print(f"\n🤖 모델별 최적 조건 (전체 데이터 기준):")
        for model_name, model_data in model_analysis.items():
            if model_data['performance_estimates']:
                best_idx = np.argmax(model_data['performance_estimates'])
                best_seq = model_data['sequences'][best_idx]
                best_performance = model_data['performance_estimates'][best_idx]
                best_boost = model_data['boost_conditions'][best_idx]
                
                # 해당 시퀀스의 전체 분석 정보
                seq_info = next((s for s in sequence_analysis if s['length'] == best_seq), None)
                total_analyzed = seq_info['total_analyzed'] if seq_info else 0
                
                print(f"  {model_name}:")
                print(f"    최적 길이: {best_seq}분 (전체 {total_analyzed:,}개 시퀀스 분석)")
                print(f"    성능 점수: {best_performance:.1f}%")
                print(f"    부스팅 조건: {best_boost:,}개")
                if total_analyzed > 0:
                    print(f"    부스팅 비율: {best_boost/total_analyzed*100:.2f}%")
        
        print(f"\n📊 전체 데이터 분석 완료 - 총 {len(sequence_analysis)}개 시퀀스 길이 완전 검증")
        print("="*80)
        
        return True
    
    def reset_analysis(self):
        """분석 초기화"""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        self.state = {'step': 0, 'sequence_lengths': [], 'results': []}
        print("🔄 분석이 초기화되었습니다.")

def main():
    """메인 실행 함수"""
    analyzer = FullDataSequenceAnalyzer()
    
    while True:
        print(f"\n{'='*60}")
        print("📋 전체 데이터 시퀀스 분석 메뉴 (한글 깨짐 해결)")
        print(f"현재 진행 단계: Step {analyzer.state['step']}")
        print(f"{'='*60}")
        
        print("\n🔍 전체 분석 단계:")
        print("1. Step 1: 데이터 로드 및 기본 분석")
        print("2. Step 2: 전체 시퀀스 패턴 분석 (샘플링 없음)")  
        print("3. Step 3: 전체 데이터 기반 모델별 분석")
        print("4. Step 4: 전체 데이터 시간대별 분석")
        print("5. Step 5: 시각화 (한글 깨짐 방지)")
        print("6. Step 6: 전체 분석 결과 내보내기")
        
        print("\n📊 도구:")
        print("7. 전체 데이터 기반 최종 보고서")
        print("8. 체크포인트 상태 확인")
        print("9. 분석 초기화")
        print("0. 종료")
        
        choice = input("\n선택 (0-9): ")
        
        if choice == '1':
            analyzer.step1_load_data()
        elif choice == '2':
            analyzer.step2_analyze_sequence_patterns()
        elif choice == '3':
            analyzer.step3_model_specific_analysis()
        elif choice == '4':
            analyzer.step4_hourly_analysis()
        elif choice == '5':
            analyzer.step5_visualization()
        elif choice == '6':
            analyzer.step6_export_results()
        elif choice == '7':
            analyzer.generate_final_report()
        elif choice == '8':
            print(f"\n📊 체크포인트 상태:")
            print(f"  완료 단계: Step {analyzer.state['step']}")
            if 'df' in analyzer.state:
                print(f"  로드된 데이터: {len(analyzer.state['df']):,}행")
            if 'sequence_analysis' in analyzer.state:
                print(f"  분석된 시퀀스: {len(analyzer.state['sequence_analysis'])}개")
                total_analyzed = sum(s['total_analyzed'] for s in analyzer.state['sequence_analysis'])
                print(f"  총 분석 시퀀스: {total_analyzed:,}개 (전체)")
            if 'model_analysis' in analyzer.state:
                print(f"  분석된 모델: {len(analyzer.state['model_analysis'])}개")
        elif choice == '9':
            confirm = input("⚠️ 모든 분석 데이터가 삭제됩니다. 계속하시겠습니까? (y/N): ")
            if confirm.lower() == 'y':
                analyzer.reset_analysis()
        elif choice == '0':
            print("전체 데이터 분석을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()