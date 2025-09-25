#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
📊 모든 컬럼 활용 점프 신호 종합 분석
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_all_columns_for_jump_signals(filepath='evaluation_results.csv'):
    """모든 컬럼을 활용한 점프 신호 분석"""
    
    print("="*80)
    print("🔍 전체 컬럼 기반 점프 신호 분석")
    print("="*80)
    
    # 데이터 로드
    df = pd.read_csv(filepath)
    
    # 타겟 컬럼
    target_col = 'CURRENT_M16A_3F_JOB_2'
    
    # 시간 처리
    df['datetime'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
    
    # 모든 컬럼 출력
    print("\n📋 전체 컬럼 리스트")
    print("-" * 60)
    for i, col in enumerate(df.columns, 1):
        if col not in ['STAT_DT', 'datetime']:
            print(f"  {i:2d}. {col}")
    
    # 주요 컬럼 그룹 정의
    inflow_cols = [
        'M16A_6F_TO_HUB_JOB',
        'M16A_2F_TO_HUB_JOB2', 
        'M14A_3F_TO_HUB_JOB2',
        'M14B_7F_TO_HUB_JOB2',
        'M16B_10F_TO_HUB_JOB'
    ]
    
    outflow_cols = [
        'M16A_3F_TO_M16A_6F_JOB',
        'M16A_3F_TO_M16A_2F_JOB',
        'M16A_3F_TO_M14A_3F_JOB',
        'M16A_3F_TO_M14B_7F_JOB',
        'M16A_3F_TO_3F_MLUD_JOB'
    ]
    
    cmd_cols = [
        'M16A_3F_CMD',
        'M16A_6F_TO_HUB_CMD',
        'M16A_2F_TO_HUB_CMD',
        'M14A_3F_TO_HUB_CMD',
        'M14B_7F_TO_HUB_CMD'
    ]
    
    maxcapa_cols = [
        'M16A_6F_LFT_MAXCAPA',
        'M16A_2F_LFT_MAXCAPA',
        'M14A_3F_CNV_MAXCAPA',
        'M14B_7F_LFT_MAXCAPA',
        'M16A_3F_LFT_MAXCAPA',
        'M16A_3F_CNV_MAXCAPA'
    ]
    
    special_cols = [
        'M16A_3F_STORAGE_UTIL',
        'M14_TO_M16_OFS_CUR',
        'M16_TO_M14_OFS_CUR'
    ]
    
    # ========================================
    # 1. 점프 케이스 찾기
    # ========================================
    print(f"\n🚀 점프 케이스 탐색 (30분 시퀀스)")
    print("-" * 60)
    
    seq_len = 30
    jump_cases = []
    
    for i in range(seq_len, len(df) - 10):
        # 과거 30분 데이터
        past_30_target = df[target_col].iloc[i-seq_len:i].values
        
        # 10분 후 값
        if i + 10 >= len(df):
            continue
        future_val = df[target_col].iloc[i+10]
        
        # 점프 조건
        past_max = max(past_30_target)
        if past_max < 280 and future_val >= 300:
            
            # 모든 컬럼의 30분 데이터 수집
            jump_data = {
                'index': i,
                'time': df['datetime'].iloc[i],
                'past_max': past_max,
                'past_mean': np.mean(past_30_target),
                'future_value': future_val,
                'jump_amount': future_val - past_max
            }
            
            # 각 컬럼 그룹별 통계
            for col_group, col_list in [
                ('inflow', inflow_cols),
                ('outflow', outflow_cols),
                ('cmd', cmd_cols)
            ]:
                # 그룹 내 컬럼들의 합계
                group_sum = 0
                group_mean = 0
                for col in col_list:
                    if col in df.columns:
                        past_values = df[col].iloc[i-seq_len:i].values
                        # 숫자형 데이터만 처리
                        try:
                            numeric_values = pd.to_numeric(past_values, errors='coerce')
                            numeric_values = numeric_values[~np.isnan(numeric_values)]
                            if len(numeric_values) > 0:
                                group_sum += np.sum(numeric_values[-5:])  # 마지막 5분 합
                                group_mean += np.mean(numeric_values)
                        except:
                            pass
                
                jump_data[f'{col_group}_last5_sum'] = group_sum
                jump_data[f'{col_group}_mean'] = group_mean
            
            # 특수 컬럼들
            for col in special_cols:
                if col in df.columns:
                    try:
                        past_values = pd.to_numeric(df[col].iloc[i-seq_len:i], errors='coerce')
                        past_values = past_values[~np.isnan(past_values)]
                        if len(past_values) > 0:
                            jump_data[f'{col}_mean'] = np.mean(past_values)
                            jump_data[f'{col}_max'] = np.max(past_values)
                    except:
                        jump_data[f'{col}_mean'] = 0
                        jump_data[f'{col}_max'] = 0
            
            jump_cases.append(jump_data)
    
    print(f"  발견된 점프 케이스: {len(jump_cases)}개")
    
    if not jump_cases:
        print("  ⚠️ 점프 케이스를 찾을 수 없습니다.")
        return None, df
    
    # ========================================
    # 2. 컬럼별 상관관계 분석
    # ========================================
    print(f"\n📊 점프와 각 컬럼의 관계")
    print("-" * 60)
    
    # DataFrame 변환
    jump_df = pd.DataFrame(jump_cases)
    
    # 유입/유출 분석
    if 'inflow_last5_sum' in jump_df.columns and 'outflow_last5_sum' in jump_df.columns:
        print("\n  [유입/유출 패턴]")
        print(f"    평균 유입(마지막5분): {jump_df['inflow_last5_sum'].mean():.1f}")
        print(f"    평균 유출(마지막5분): {jump_df['outflow_last5_sum'].mean():.1f}")
        
        # 유입 > 유출인 케이스
        inflow_dominant = (jump_df['inflow_last5_sum'] > jump_df['outflow_last5_sum']).sum()
        print(f"    유입 > 유출: {inflow_dominant}/{len(jump_df)} ({inflow_dominant/len(jump_df)*100:.1f}%)")
    
    # CMD 분석
    if 'cmd_mean' in jump_df.columns:
        print(f"\n  [CMD 패턴]")
        print(f"    평균 CMD: {jump_df['cmd_mean'].mean():.1f}")
    
    # STORAGE_UTIL 분석
    if 'M16A_3F_STORAGE_UTIL_mean' in jump_df.columns:
        print(f"\n  [STORAGE_UTIL 패턴]")
        print(f"    평균: {jump_df['M16A_3F_STORAGE_UTIL_mean'].mean():.1f}")
        print(f"    최대: {jump_df['M16A_3F_STORAGE_UTIL_max'].mean():.1f}")
    
    # ========================================
    # 3. 점프 전 특징적 패턴
    # ========================================
    print(f"\n🎯 점프 전 특징적 신호")
    print("-" * 60)
    
    # 각 점프 케이스의 특징 점수 계산
    for idx, jump in enumerate(jump_cases):
        signals = []
        
        # 신호 1: 과거 MAX가 높음
        if jump['past_max'] >= 270:
            signals.append("MAX>=270")
        
        # 신호 2: 유입 > 유출
        if 'inflow_last5_sum' in jump and 'outflow_last5_sum' in jump:
            if jump['inflow_last5_sum'] > jump['outflow_last5_sum']:
                signals.append("유입우세")
        
        # 신호 3: STORAGE_UTIL 높음
        if 'M16A_3F_STORAGE_UTIL_max' in jump:
            if jump['M16A_3F_STORAGE_UTIL_max'] > 50:
                signals.append("STORAGE>50")
        
        jump_cases[idx]['signals'] = signals
        jump_cases[idx]['signal_count'] = len(signals)
    
    # 신호 개수별 통계
    signal_counts = [j['signal_count'] for j in jump_cases]
    print(f"  평균 신호 개수: {np.mean(signal_counts):.1f}")
    print(f"  신호 2개 이상: {sum(1 for s in signal_counts if s >= 2)}/{len(jump_cases)}")
    
    # ========================================
    # 4. 상위 점프 케이스 상세
    # ========================================
    print(f"\n🔥 대표 점프 케이스 (상위 5개)")
    print("-" * 60)
    
    sorted_jumps = sorted(jump_cases, key=lambda x: x['jump_amount'], reverse=True)[:5]
    
    for idx, jump in enumerate(sorted_jumps, 1):
        print(f"\n  [{idx}] {jump['time'].strftime('%m/%d %H:%M')}")
        print(f"      과거: MAX={jump['past_max']:.0f}, MEAN={jump['past_mean']:.0f}")
        print(f"      미래: {jump['future_value']:.0f} (+{jump['jump_amount']:.0f})")
        
        if 'inflow_last5_sum' in jump:
            print(f"      유입/유출: {jump['inflow_last5_sum']:.0f}/{jump.get('outflow_last5_sum', 0):.0f}")
        
        if jump.get('signals'):
            print(f"      신호: {', '.join(jump['signals'])}")
    
    # ========================================
    # 5. 컬럼 중요도 순위
    # ========================================
    print(f"\n📈 점프 예측에 중요한 컬럼")
    print("-" * 60)
    
    # 각 컬럼과 점프량의 상관관계
    correlations = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != target_col and col != 'STAT_DT':
            try:
                # 점프 시점의 컬럼 값들
                jump_col_values = []
                for jump in jump_cases:
                    idx = jump['index']
                    if idx < len(df):
                        val = df[col].iloc[idx]
                        if pd.notna(val):
                            jump_col_values.append(float(val))
                
                if len(jump_col_values) >= 5:
                    # 점프량과의 상관관계
                    jump_amounts = [j['jump_amount'] for j in jump_cases[:len(jump_col_values)]]
                    corr, _ = stats.pearsonr(jump_col_values, jump_amounts)
                    if not np.isnan(corr):
                        correlations[col] = abs(corr)
            except:
                pass
    
    # 상위 10개 컬럼
    if correlations:
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n  상관관계 TOP 10:")
        for col, corr in sorted_corr:
            print(f"    {col:30s}: {corr:.3f}")
    
    # ========================================
    # 6. 실시간 감지 규칙 제안
    # ========================================
    print(f"\n💡 점프 감지 규칙 제안")
    print("-" * 60)
    
    print("\n  IF (다음 조건 중 2개 이상):")
    print("    1. past_max >= 265 AND gap > 20")
    print("    2. 마지막 5분 유입 > 유출")
    print("    3. STORAGE_UTIL > 50")
    print("    4. 마지막 5분 상승 추세")
    print("  THEN:")
    print("    → 점프 가능성 HIGH")
    print("    → 예측 = mean × 1.15~1.25")
    
    return jump_cases, df

# 실행
if __name__ == "__main__":
    jump_cases, df = analyze_all_columns_for_jump_signals('evaluation_results.csv')
    
    print("\n" + "="*80)
    print("✅ 전체 컬럼 분석 완료!")
    print("="*80)
    
    if jump_cases:
        # 결과 저장
        jump_df = pd.DataFrame(jump_cases)
        jump_df.to_csv('jump_analysis_all_columns.csv', index=False)
        print(f"\n💾 분석 결과 저장: jump_analysis_all_columns.csv")