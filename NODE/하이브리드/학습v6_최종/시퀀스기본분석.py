# -*- coding: utf-8 -*-
"""
CSV 데이터 시퀀스 검증 시스템 (전체 데이터 분석)
===============================================
샘플링 없이 모든 시퀀스를 완전 분석
10분부터 300분까지 모든 시퀀스 길이 검증
모델별, 시간별 상세 분석 및 CSV 저장
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

class FullSequenceVerifier:
    """CSV 데이터의 모든 시퀀스 패턴 검증 (샘플링 없음)"""
    
    def __init__(self):
        print("="*80)
        print("CSV 데이터 전체 시퀀스 검증 시스템 (샘플링 없음)")
        print("="*80)
        self.results = []
        
    def load_and_analyze_data(self, filepath):
        """데이터 로드 및 기본 분석"""
        print(f"\n📂 데이터 로딩: {filepath}")
        
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
            time_diff = df['CURRTIME'].diff().dt.total_seconds() / 60  # 분 단위
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
        
        # 고값 구간 분포
        high_1651 = (df['TOTALCNT'] >= 1651).sum()
        high_1700 = (df['TOTALCNT'] >= 1700).sum()
        high_1750 = (df['TOTALCNT'] >= 1750).sum()
        
        print(f"\n🎯 고값 구간 분포:")
        print(f"  1651+: {high_1651}개 ({high_1651/len(df)*100:.1f}%)")
        print(f"  1700+: {high_1700}개 ({high_1700/len(df)*100:.1f}%)")
        print(f"  1750+: {high_1750}개 ({high_1750/len(df)*100:.1f}%)")
        
        return df
    
    def analyze_sequence_detailed(self, sequence_data):
        """V6.7 시퀀스 분석 기능"""
        if len(sequence_data) == 0:
            return {'max': 0, 'min': 0, 'trend': 'stable', 'is_high_plateau': False,
                   'consecutive_rises': 0, 'consecutive_falls': 0, 
                   'rise_strength': 0, 'fall_strength': 0, 'volatility': 0,
                   'mean': 0, 'std': 0, 'slope': 0}
        
        # 1. 시퀀스 기본 통계
        seq_max = np.max(sequence_data)
        seq_min = np.min(sequence_data)
        seq_mean = np.mean(sequence_data)
        seq_std = np.std(sequence_data)
        
        # 2. 고평원 상태 체크 (최근 30개 평균이 1700 이상)
        recent_mean = np.mean(sequence_data[-30:]) if len(sequence_data) >= 30 else seq_mean
        is_high_plateau = recent_mean >= 1700
        
        # 3. 연속 상승 카운트 계산
        consecutive_rises = 0
        for i in range(len(sequence_data)-1, 0, -1):
            if sequence_data[i] > sequence_data[i-1]:
                consecutive_rises += 1
            else:
                break
        
        # 4. 연속 하락 카운트 계산
        consecutive_falls = 0
        for i in range(len(sequence_data)-1, 0, -1):
            if sequence_data[i] < sequence_data[i-1]:
                consecutive_falls += 1
            else:
                break
        
        # 5. 상승/하락 강도 계산
        rise_strength = 0
        fall_strength = 0
        if len(sequence_data) >= 10:
            recent_10 = sequence_data[-10:]
            change = recent_10[-1] - recent_10[0]
            if change > 0:
                rise_strength = change
            else:
                fall_strength = abs(change)
        
        # 6. 추세 분석
        slope = 0
        if len(sequence_data) >= 30:
            recent = sequence_data[-30:]
            x = np.arange(len(recent))
            coeffs = np.polyfit(x, recent, 1)
            slope = coeffs[0]
            
            if is_high_plateau:
                if consecutive_rises >= 10 and rise_strength > 50:
                    trend = 'extreme_rising'
                elif consecutive_falls >= 10 and fall_strength > 50:
                    trend = 'extreme_falling'
                elif slope > 1 or consecutive_rises >= 5:
                    trend = 'high_increasing'
                elif slope < -1 or consecutive_falls >= 5:
                    trend = 'high_decreasing'
                else:
                    trend = 'high_stable'
            else:
                if consecutive_rises >= 10 and rise_strength > 50:
                    trend = 'strong_rising'
                elif consecutive_falls >= 10 and fall_strength > 50:
                    trend = 'strong_falling'
                elif consecutive_rises >= 7 and rise_strength > 30:
                    trend = 'rapid_increasing'
                elif consecutive_falls >= 7 and fall_strength > 30:
                    trend = 'rapid_decreasing'
                elif slope > 2:
                    trend = 'increasing'
                elif slope < -2:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
        else:
            trend = 'stable'
        
        # 변동성 지표
        volatility = np.std(sequence_data[-10:]) if len(sequence_data) >= 10 else seq_std
        
        return {
            'max': seq_max,
            'min': seq_min,
            'mean': seq_mean,
            'std': seq_std,
            'trend': trend,
            'is_high_plateau': is_high_plateau,
            'consecutive_rises': consecutive_rises,
            'consecutive_falls': consecutive_falls,
            'rise_strength': rise_strength,
            'fall_strength': fall_strength,
            'volatility': volatility,
            'slope': slope
        }
    
    def verify_sequences(self, df):
        """전체 데이터 시퀀스 길이별 검증 실행 (샘플링 없음)"""
        print(f"\n🔍 전체 데이터 시퀀스 길이별 검증 시작!")
        
        # 시퀀스 길이 설정 (10분부터 300분까지)
        sequence_lengths = list(range(10, 101, 10)) + list(range(120, 301, 20))  
        print(f"  검증할 시퀀스 길이: {len(sequence_lengths)}개")
        print(f"  길이 목록: {sequence_lengths}")
        
        verification_results = []
        
        for seq_idx, seq_len in enumerate(sequence_lengths):
            print(f"\n{'='*60}")
            print(f"🎯 [{seq_idx+1}/{len(sequence_lengths)}] 시퀀스 길이: {seq_len}분 전체 검증")
            print(f"{'='*60}")
            
            # 시퀀스별 분석 결과 저장
            seq_analyses = []
            valid_sequences = 0
            
            # 각 시점에서 시퀀스 생성 가능 여부 확인
            max_start_idx = len(df) - seq_len - 10  # 10분 후 예측을 위한 여유
            
            if max_start_idx <= 0:
                print(f"  ❌ 시퀀스 길이 {seq_len}분: 데이터 부족 (필요: {seq_len + 10}분, 보유: {len(df)}분)")
                continue
            
            # 전체 데이터 분석 (샘플링 제거)
            total_sequences = max_start_idx
            sample_indices = list(range(seq_len, max_start_idx + seq_len))
            
            print(f"  📊 분석할 전체 시퀀스: {len(sample_indices):,}개 (샘플링 없음)")
            print(f"  📈 예상 분석 시간: {len(sample_indices) // 1000:.1f}초 (대략)")
            
            # 각 모델별 시퀀스 분석
            model_analysis = {
                'LSTM': {'high_seq_count': 0, 'trend_counts': {}, 'boost_conditions': 0},
                'GRU': {'high_seq_count': 0, 'trend_counts': {}, 'boost_conditions': 0},
                'CNN_LSTM': {'high_seq_count': 0, 'trend_counts': {}, 'boost_conditions': 0},
                'SpikeDetector': {'high_seq_count': 0, 'trend_counts': {}, 'boost_conditions': 0},
                'ExtremeNet': {'high_seq_count': 0, 'trend_counts': {}, 'boost_conditions': 0}
            }
            
            # 시간대별 분석 (시간별)
            hourly_analysis = {}
            for hour in range(24):
                hourly_analysis[hour] = {
                    'count': 0, 'high_seq_count': 0, 'trend_counts': {},
                    'avg_max': 0, 'avg_volatility': 0, 'total_max': 0, 'total_volatility': 0
                }
            
            # 진행률 표시를 위한 체크포인트
            checkpoint_interval = max(1000, len(sample_indices) // 20)
            
            # 전체 시퀀스 분석 (샘플링 없음)
            for seq_count, idx in enumerate(sample_indices):
                # 진행률 표시
                if seq_count % checkpoint_interval == 0:
                    progress = seq_count / len(sample_indices) * 100
                    print(f"    진행률: {progress:.1f}% ({seq_count:,}/{len(sample_indices):,})")
                
                current_time = df.iloc[idx]['CURRTIME']
                hour = current_time.hour
                
                # 시퀀스 데이터 추출 (TOTALCNT)
                seq_data = df.iloc[idx-seq_len:idx]['TOTALCNT'].values
                
                # 시퀀스 분석
                analysis = self.analyze_sequence_detailed(seq_data)
                seq_analyses.append(analysis)
                
                # 유효한 시퀀스 카운트
                if analysis['max'] > 0:
                    valid_sequences += 1
                    
                    # 시간대별 집계
                    hourly_analysis[hour]['count'] += 1
                    hourly_analysis[hour]['total_max'] += analysis['max']
                    hourly_analysis[hour]['total_volatility'] += analysis['volatility']
                    
                    if analysis['max'] >= 1651:
                        hourly_analysis[hour]['high_seq_count'] += 1
                    
                    trend = analysis['trend']
                    if trend not in hourly_analysis[hour]['trend_counts']:
                        hourly_analysis[hour]['trend_counts'][trend] = 0
                    hourly_analysis[hour]['trend_counts'][trend] += 1
                    
                    # 모델별 분석 (각 모델이 이 시퀀스를 어떻게 처리할지)
                    m14b_value = df.iloc[idx]['M14AM14B'] if 'M14AM14B' in df.columns else 300
                    
                    for model_name in model_analysis.keys():
                        # 고값 시퀀스 카운트
                        if analysis['max'] >= 1651:
                            model_analysis[model_name]['high_seq_count'] += 1
                        
                        # 추세별 카운트
                        if trend not in model_analysis[model_name]['trend_counts']:
                            model_analysis[model_name]['trend_counts'][trend] = 0
                        model_analysis[model_name]['trend_counts'][trend] += 1
                        
                        # ExtremeNet 부스팅 조건 (V6.7)
                        if model_name == 'ExtremeNet':
                            if analysis['max'] >= 1651 and ('increasing' in trend or trend == 'extreme_rising'):
                                model_analysis[model_name]['boost_conditions'] += 1
                        
                        # SpikeDetector 조건 (최근 20분 중점)
                        elif model_name == 'SpikeDetector':
                            if analysis['consecutive_rises'] >= 5 or analysis['rise_strength'] > 30:
                                model_analysis[model_name]['boost_conditions'] += 1
            
            # 시간대별 평균 계산 (전체 데이터 기준)
            for hour in hourly_analysis:
                if hourly_analysis[hour]['count'] > 0:
                    hourly_analysis[hour]['avg_max'] = hourly_analysis[hour]['total_max'] / hourly_analysis[hour]['count']
                    hourly_analysis[hour]['avg_volatility'] = hourly_analysis[hour]['total_volatility'] / hourly_analysis[hour]['count']
            
            # 전체 데이터 시퀀스 검증 결과 출력
            print(f"\n  ✅ 전체 분석 완료: {len(sample_indices):,}개")
            print(f"\n📈 시퀀스 {seq_len}분 전체 검증 결과:")
            print(f"  유효 시퀀스: {valid_sequences:,}개 / {len(sample_indices):,}개")
            
            if valid_sequences > 0:
                # 전체 통계
                all_max_values = [a['max'] for a in seq_analyses if a['max'] > 0]
                all_trends = [a['trend'] for a in seq_analyses if a['max'] > 0]
                all_volatilities = [a['volatility'] for a in seq_analyses if a['volatility'] > 0]
                all_consecutive_rises = [a['consecutive_rises'] for a in seq_analyses if a['max'] > 0]
                all_consecutive_falls = [a['consecutive_falls'] for a in seq_analyses if a['max'] > 0]
                
                print(f"  MAX값 범위: {min(all_max_values):.0f} ~ {max(all_max_values):.0f}")
                print(f"  평균 MAX: {np.mean(all_max_values):.1f}")
                print(f"  평균 변동성: {np.mean(all_volatilities):.1f}")
                print(f"  평균 연속상승: {np.mean(all_consecutive_rises):.1f}")
                print(f"  평균 연속하락: {np.mean(all_consecutive_falls):.1f}")
                
                # 고값 구간 비율 (전체 데이터 기준)
                high_1651_count = sum(1 for v in all_max_values if v >= 1651)
                high_1700_count = sum(1 for v in all_max_values if v >= 1700)
                high_1750_count = sum(1 for v in all_max_values if v >= 1750)
                
                print(f"  고값 시퀀스 분포 (전체 {valid_sequences:,}개 중):")
                print(f"    1651+: {high_1651_count:,}개 ({high_1651_count/valid_sequences*100:.2f}%)")
                print(f"    1700+: {high_1700_count:,}개 ({high_1700_count/valid_sequences*100:.2f}%)")
                print(f"    1750+: {high_1750_count:,}개 ({high_1750_count/valid_sequences*100:.2f}%)")
                
                # 추세 분포 (전체 데이터 기준)
                trend_counts = {}
                for trend in all_trends:
                    trend_counts[trend] = trend_counts.get(trend, 0) + 1
                
                print(f"  추세 분포 (전체):")
                for trend, count in sorted(trend_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {trend}: {count:,}개 ({count/valid_sequences*100:.2f}%)")
                
                # 모델별 부스팅 조건 분석 (전체 데이터 기준)
                print(f"\n🤖 모델별 분석 (시퀀스 {seq_len}분, 전체 데이터):")
                for model_name, model_data in model_analysis.items():
                    boost_ratio = model_data['boost_conditions'] / valid_sequences * 100 if valid_sequences > 0 else 0
                    high_ratio = model_data['high_seq_count'] / valid_sequences * 100 if valid_sequences > 0 else 0
                    
                    print(f"  {model_name}:")
                    print(f"    고값 대상: {model_data['high_seq_count']:,}개 ({high_ratio:.2f}%)")
                    print(f"    부스팅 조건: {model_data['boost_conditions']:,}개 ({boost_ratio:.2f}%)")
                
                # 시간대별 분석 (전체 데이터 기준, 상위 시간대만)
                print(f"\n⏰ 시간대별 분석 (전체 데이터, 상위 5시간):")
                hour_summary = []
                for hour, data in hourly_analysis.items():
                    if data['count'] > 0:
                        hour_summary.append((hour, data))
                
                # 시퀀스 개수 기준으로 정렬
                hour_summary.sort(key=lambda x: x[1]['count'], reverse=True)
                
                for hour, data in hour_summary[:5]:
                    high_ratio = data['high_seq_count'] / data['count'] * 100 if data['count'] > 0 else 0
                    print(f"  {hour:02d}시: {data['count']:,}개, 고값 {data['high_seq_count']:,}개({high_ratio:.1f}%), "
                          f"평균MAX {data['avg_max']:.0f}")
            
            # 결과 저장 (CSV용, 전체 데이터 기준)
            seq_result = {
                'sequence_length': seq_len,
                'total_sequences': len(sample_indices),
                'valid_sequences': valid_sequences,
                'high_sequences_1651': sum(1 for a in seq_analyses if a['max'] >= 1651),
                'high_sequences_1700': sum(1 for a in seq_analyses if a['max'] >= 1700),
                'high_sequences_1750': sum(1 for a in seq_analyses if a['max'] >= 1750),
                'avg_max': np.mean([a['max'] for a in seq_analyses if a['max'] > 0]) if valid_sequences > 0 else 0,
                'std_max': np.std([a['max'] for a in seq_analyses if a['max'] > 0]) if valid_sequences > 0 else 0,
                'min_max': np.min([a['max'] for a in seq_analyses if a['max'] > 0]) if valid_sequences > 0 else 0,
                'max_max': np.max([a['max'] for a in seq_analyses if a['max'] > 0]) if valid_sequences > 0 else 0,
                'avg_volatility': np.mean([a['volatility'] for a in seq_analyses if a['volatility'] > 0]) if valid_sequences > 0 else 0,
                'avg_consecutive_rises': np.mean([a['consecutive_rises'] for a in seq_analyses if a['max'] > 0]) if valid_sequences > 0 else 0,
                'avg_consecutive_falls': np.mean([a['consecutive_falls'] for a in seq_analyses if a['max'] > 0]) if valid_sequences > 0 else 0,
                'increasing_trend': sum(1 for a in seq_analyses if 'increasing' in a['trend']),
                'decreasing_trend': sum(1 for a in seq_analyses if 'decreasing' in a['trend']),
                'stable_trend': sum(1 for a in seq_analyses if a['trend'] == 'stable'),
                'extreme_rising': sum(1 for a in seq_analyses if a['trend'] == 'extreme_rising'),
                'extreme_falling': sum(1 for a in seq_analyses if a['trend'] == 'extreme_falling')
            }
            
            # 모델별 데이터 추가 (전체 데이터 기준)
            for model_name, model_data in model_analysis.items():
                seq_result[f'{model_name}_high_seq'] = model_data['high_seq_count']
                seq_result[f'{model_name}_boost_conditions'] = model_data['boost_conditions']
                seq_result[f'{model_name}_boost_ratio'] = model_data['boost_conditions'] / valid_sequences * 100 if valid_sequences > 0 else 0
            
            # 시간대별 최고 데이터 추가
            if hour_summary:
                best_hour, best_data = hour_summary[0]
                seq_result['best_hour'] = best_hour
                seq_result['best_hour_count'] = best_data['count']
                seq_result['best_hour_high_count'] = best_data['high_seq_count']
                seq_result['best_hour_high_ratio'] = best_data['high_seq_count'] / best_data['count'] * 100 if best_data['count'] > 0 else 0
            
            verification_results.append(seq_result)
        
        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(verification_results)
        
        # CSV 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'full_sequence_verification_{timestamp}.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n" + "="*80)
        print(f"💾 전체 데이터 검증 결과 저장: {output_file}")
        print(f"📊 총 {len(verification_results)}개 시퀀스 길이 전체 검증 완료")
        
        # 전체 분석 통계 요약
        if verification_results:
            total_analyzed = sum(r['total_sequences'] for r in verification_results)
            total_valid = sum(r['valid_sequences'] for r in verification_results)
            print(f"🎯 전체 분석 규모: {total_analyzed:,}개 시퀀스 (유효: {total_valid:,}개)")
        
        print("="*80)
        
        return results_df, output_file
    
    def generate_summary_report(self, results_df):
        """전체 데이터 기반 요약 보고서 생성"""
        print(f"\n📋 전체 데이터 시퀀스 검증 요약 보고서")
        print("="*60)
        
        if results_df.empty:
            print("❌ 검증 결과가 없습니다!")
            return
        
        # 전체 분석 규모 출력
        total_sequences = results_df['total_sequences'].sum()
        total_valid = results_df['valid_sequences'].sum()
        print(f"\n🔍 전체 분석 규모:")
        print(f"  총 분석 시퀀스: {total_sequences:,}개")
        print(f"  유효 시퀀스: {total_valid:,}개")
        print(f"  시퀀스 길이 종류: {len(results_df)}개")
        
        # 1. 최적 시퀀스 길이 분석 (전체 데이터 기준)
        print(f"\n🏆 최적 시퀀스 길이 분석 (전체 데이터 기준):")
        
        # 유효 시퀀스 비율이 높은 순
        results_df['valid_ratio'] = results_df['valid_sequences'] / results_df['total_sequences'] * 100
        
        # 고값 시퀀스 비율 (1651+)
        results_df['high_ratio_1651'] = results_df['high_sequences_1651'] / results_df['valid_sequences'] * 100
        results_df['high_ratio_1651'] = results_df['high_ratio_1651'].fillna(0)
        
        # 극값 시퀀스 비율 (1750+)
        results_df['extreme_ratio_1750'] = results_df['high_sequences_1750'] / results_df['valid_sequences'] * 100
        results_df['extreme_ratio_1750'] = results_df['extreme_ratio_1750'].fillna(0)
        
        print(f"\n  고값 시퀀스 비율(1651+) 상위 5개 (전체 데이터):")
        top_high = results_df.nlargest(5, 'high_ratio_1651')
        for _, row in top_high.iterrows():
            print(f"    {int(row['sequence_length'])}분: {row['high_ratio_1651']:.2f}% "
                  f"({int(row['high_sequences_1651']):,}/{int(row['valid_sequences']):,}개)")
        
        print(f"\n  극값 시퀀스 비율(1750+) 상위 3개:")
        top_extreme = results_df.nlargest(3, 'extreme_ratio_1750')
        for _, row in top_extreme.iterrows():
            if row['extreme_ratio_1750'] > 0:
                print(f"    {int(row['sequence_length'])}분: {row['extreme_ratio_1750']:.2f}% "
                      f"({int(row['high_sequences_1750']):,}개)")
        
        # 2. 모델별 부스팅 조건 분석 (전체 데이터 기준)
        print(f"\n🤖 모델별 부스팅 조건 분석 (전체 데이터):")
        model_names = ['ExtremeNet', 'SpikeDetector', 'LSTM', 'GRU', 'CNN_LSTM']
        
        for model in model_names:
            boost_col = f'{model}_boost_conditions'
            high_col = f'{model}_high_seq'
            
            if boost_col in results_df.columns:
                total_boost = results_df[boost_col].sum()
                total_high = results_df[high_col].sum()
                
                if total_boost > 0:
                    best_seq = results_df.loc[results_df[boost_col].idxmax()]
                    print(f"  {model}:")
                    print(f"    총 부스팅 조건: {total_boost:,}개")
                    print(f"    총 고값 대상: {total_high:,}개")
                    print(f"    최적 길이: {int(best_seq['sequence_length'])}분 "
                          f"(부스팅 {int(best_seq[boost_col]):,}개, {best_seq[f'{model}_boost_ratio']:.2f}%)")
        
        # 3. 시간대 분석 (전체 데이터 기준)
        print(f"\n⏰ 시간대별 분석 (전체 데이터 기준):")
        if 'best_hour' in results_df.columns:
            hour_counts = results_df['best_hour'].value_counts().head(5)
            print(f"  최다 고값 시간대:")
            for hour, count in hour_counts.items():
                avg_high_ratio = results_df[results_df['best_hour'] == hour]['best_hour_high_ratio'].mean()
                total_sequences = results_df[results_df['best_hour'] == hour]['best_hour_count'].sum()
                print(f"    {int(hour):02d}시: {count}개 길이에서 최고성능, "
                      f"평균 고값비율 {avg_high_ratio:.1f}%, 총 {total_sequences:,}개 시퀀스")
        
        # 4. 권장 사항 (전체 데이터 기준)
        print(f"\n💡 전체 데이터 기반 권장 사항:")
        
        # 현재 100분과 비교
        current_100 = results_df[results_df['sequence_length'] == 100]
        if not current_100.empty:
            current_performance = current_100.iloc[0]
            print(f"  현재 100분 설정 (전체 데이터):")
            print(f"    전체 시퀀스: {int(current_performance['total_sequences']):,}개")
            print(f"    고값 비율: {current_performance['high_ratio_1651']:.2f}%")
            print(f"    평균 MAX: {current_performance['avg_max']:.0f}")
            print(f"    ExtremeNet 부스팅: {int(current_performance.get('ExtremeNet_boost_conditions', 0)):,}개 "
                  f"({current_performance.get('ExtremeNet_boost_ratio', 0):.2f}%)")
        
        # 최고 성능 시퀀스 추천
        best_overall = results_df.loc[results_df['high_ratio_1651'].idxmax()]
        print(f"\n  🥇 추천 시퀀스 길이: {int(best_overall['sequence_length'])}분")
        print(f"    전체 분석: {int(best_overall['total_sequences']):,}개 시퀀스")
        print(f"    고값 비율: {best_overall['high_ratio_1651']:.2f}% "
              f"({int(best_overall['high_sequences_1651']):,}개)")
        print(f"    평균 MAX: {best_overall['avg_max']:.0f}")
        print(f"    ExtremeNet 부스팅: {int(best_overall.get('ExtremeNet_boost_conditions', 0)):,}개 "
              f"({best_overall.get('ExtremeNet_boost_ratio', 0):.2f}%)")
        
        # 현재 대비 개선 효과
        if not current_100.empty:
            improvement = (best_overall['high_ratio_1651'] - current_performance['high_ratio_1651']) / current_performance['high_ratio_1651'] * 100
            boost_improvement = (best_overall.get('ExtremeNet_boost_conditions', 0) - current_performance.get('ExtremeNet_boost_conditions', 0)) / max(current_performance.get('ExtremeNet_boost_conditions', 1), 1) * 100
            
            print(f"\n  📈 현재 100분 대비 개선 효과:")
            print(f"    고값 감지율: {improvement:+.1f}% 향상")
            print(f"    ExtremeNet 부스팅: {boost_improvement:+.1f}% 향상")
        
        # 안정성과 성능의 균형
        balanced = results_df[(results_df['valid_ratio'] >= 95) & (results_df['high_ratio_1651'] >= 10)]
        if not balanced.empty:
            balanced_best = balanced.loc[balanced['high_ratio_1651'].idxmax()]
            print(f"\n  ⚖️ 균형잡힌 선택: {int(balanced_best['sequence_length'])}분")
            print(f"    전체 분석: {int(balanced_best['total_sequences']):,}개")
            print(f"    고값 비율: {balanced_best['high_ratio_1651']:.2f}%")
            print(f"    유효 비율: {balanced_best['valid_ratio']:.2f}%")

def main():
    """메인 실행 함수"""
    print("🚀 CSV 데이터 전체 시퀀스 검증 시작!")
    
    # 검증기 생성
    verifier = FullSequenceVerifier()
    
    # 데이터 파일 경로
    data_files = [
        'data/gs.CSV',
        'gs.CSV',
        './gs.CSV',
        'gs.csv',
        './gs.csv'
    ]
    
    data_file = None
    for file in data_files:
        if os.path.exists(file):
            data_file = file
            break
    
    if not data_file:
        print("❌ 데이터 파일을 찾을 수 없습니다!")
        print("업로드 가능한 파일:")
        for file in data_files:
            print(f"  - {file}")
        return
    
    # 데이터 로드 및 분석
    df = verifier.load_and_analyze_data(data_file)
    
    # 전체 시퀀스 검증 실행 (샘플링 없음)
    results_df, output_file = verifier.verify_sequences(df)
    
    # 전체 데이터 기반 요약 보고서 생성
    verifier.generate_summary_report(results_df)
    
    print(f"\n✅ 전체 데이터 시퀀스 검증 완료!")
    print(f"📁 결과 파일: {output_file}")
    print(f"📊 {len(results_df)}개 시퀀스 길이 전체 검증")
    print(f"🎯 최적 시퀀스 길이와 모델별 부스팅 조건 완전 분석 완료")

if __name__ == "__main__":
    main()