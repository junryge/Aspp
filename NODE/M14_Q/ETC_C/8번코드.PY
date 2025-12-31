#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🚀 8단계: Feature Engineering 수행
================================================================================
7단계 계획서 기반 ~297개 Feature 생성
- 12개 핵심 컬럼 × 등급별 Feature
- 상호작용 Feature 25개
- TOTALCNT 자체 Feature 15개
================================================================================
"""

import numpy as np
import pandas as pd
import glob
import os
import warnings
from datetime import datetime
import gc

warnings.filterwarnings('ignore')

print("="*80)
print("🚀 8단계: Feature Engineering 수행")
print("   7단계 계획서 기반 ~297개 Feature 생성")
print("="*80)


# ==============================================================================
# 1. 설정
# ==============================================================================

# 시퀀스 설정
SEQ_LEN = 280  # 280분 시퀀스
PRED_OFFSETS = [10, 15, 25]  # 예측 오프셋 (분)

# 핵심 컬럼 정의 (12개)
CORE_COLUMNS = {
    # 기존 6개
    'M14AM14B': {'grade': 'S', 'thresholds': [450, 500, 520, 540, 560]},
    'M14AM14BSUM': {'grade': 'S', 'thresholds': [550, 600, 620, 650, 700]},
    'M10AM14A': {'grade': 'A', 'thresholds': [45, 50, 55, 60]},
    'M14.QUE.OHT.OHTUTIL': {'grade': 'A', 'thresholds': [80, 85, 90, 95]},
    'M14.QUE.ALL.TRANSPORT4MINOVERCNT': {'grade': 'B', 'thresholds': [100, 120, 151, 180]},
    # queue_gap은 파생 (아래에서 생성)
    
    # 신규 6개
    'M14.QUE.SENDFAB.VERTICALQUEUECOUNT': {'grade': 'S', 'thresholds': [150, 200, 230, 250, 280]},
    'M14AM16SUM': {'grade': 'A', 'thresholds': [150, 180, 200, 220]},
    'M16M14A': {'grade': 'A', 'thresholds': [80, 100, 110, 120]},
    'M14.QUE.CNV.SOUTHCURRENTQCNT': {'grade': 'A', 'thresholds': [40, 50, 55]},
    'M14.QUE.CNV.NORTHCURRENTQCNT': {'grade': 'A', 'thresholds': [45, 50, 55]},
    'M14.QUE.OHT.CURRENTOHTQCNT': {'grade': 'A', 'thresholds': [800, 900, 950]},
}

# queue_gap 설정 (파생변수)
QUEUE_GAP_CONFIG = {
    'grade': 'B',
    'thresholds': [150, 200, 250, 300, 350]
}


# ==============================================================================
# 2. 데이터 로딩
# ==============================================================================

def load_data(path_pattern='*.CSV'):
    """CSV 파일 로드"""
    print(f"\n📂 데이터 로딩: {path_pattern}")
    
    files = glob.glob(path_pattern)
    if not files:
        files = glob.glob('data/*.CSV') + glob.glob('M14_Q_*.CSV')
    
    if not files:
        print("⚠️ CSV 파일을 찾을 수 없습니다.")
        return None
    
    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, on_bad_lines='skip', encoding='utf-8')
            print(f"  ✅ {os.path.basename(f)}: {len(df):,}행")
            dfs.append(df)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(f, on_bad_lines='skip', encoding='cp949')
                print(f"  ✅ {os.path.basename(f)} (cp949): {len(df):,}행")
                dfs.append(df)
            except Exception as e:
                print(f"  ❌ {f}: {e}")
    
    if not dfs:
        return None
    
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 총 데이터: {len(df_all):,}행")
    return df_all


def preprocess_data(df):
    """데이터 전처리"""
    print("\n🔧 데이터 전처리...")
    
    df = df.copy()
    
    # CURRTIME 파싱
    if 'CURRTIME' in df.columns:
        df['CURRTIME'] = df['CURRTIME'].astype(str).str.strip()
        df = df[df['CURRTIME'].str.len() == 12].copy()
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')
        df = df.dropna(subset=['CURRTIME']).copy()
        df = df.sort_values('CURRTIME').reset_index(drop=True)
    
    # 숫자형 변환
    exclude_cols = ['CURRTIME']
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 결측치 처리
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # 파생변수 생성
    if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
        df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    else:
        df['queue_gap'] = 0
    
    if 'M14B.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14B.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
        df['queue_gap_B'] = df['M14B.QUE.ALL.CURRENTQCREATED'] - df['M14B.QUE.ALL.CURRENTQCOMPLETED']
    else:
        df['queue_gap_B'] = 0
    
    print(f"  ✅ 전처리 완료: {len(df):,}행")
    return df


# ==============================================================================
# 3. Feature 생성 함수들
# ==============================================================================

def create_stat_features(seq, prefix, grade):
    """통계 Feature 생성"""
    features = {}
    
    features[f'{prefix}_mean'] = np.mean(seq)
    features[f'{prefix}_max'] = np.max(seq)
    features[f'{prefix}_min'] = np.min(seq)
    features[f'{prefix}_std'] = np.std(seq)
    
    if grade == 'S':
        features[f'{prefix}_median'] = np.median(seq)
        features[f'{prefix}_skew'] = pd.Series(seq).skew() if len(seq) > 2 else 0
    
    return features


def create_lag_features(seq, prefix, grade):
    """시차(Lag) Feature 생성"""
    features = {}
    
    features[f'{prefix}_current'] = seq[-1]
    features[f'{prefix}_last5'] = np.mean(seq[-5:])
    features[f'{prefix}_last10'] = np.mean(seq[-10:])
    features[f'{prefix}_last20'] = np.mean(seq[-20:])
    
    if grade in ['S', 'A']:
        features[f'{prefix}_last30'] = np.mean(seq[-30:])
    
    if grade == 'S':
        features[f'{prefix}_last60'] = np.mean(seq[-60:])
    
    return features


def create_trend_features(seq, prefix, grade):
    """추세 Feature 생성"""
    features = {}
    seq_len = len(seq)
    
    # slope (전체 기울기)
    try:
        features[f'{prefix}_slope'] = np.polyfit(np.arange(seq_len), seq, 1)[0]
    except:
        features[f'{prefix}_slope'] = 0
    
    # trend (최근 변화량)
    features[f'{prefix}_trend10'] = seq[-1] - seq[-10] if len(seq) >= 10 else 0
    
    if grade in ['S', 'A']:
        features[f'{prefix}_trend20'] = seq[-1] - seq[-20] if len(seq) >= 20 else 0
    
    if grade == 'S':
        features[f'{prefix}_trend5'] = seq[-1] - seq[-5] if len(seq) >= 5 else 0
        # 가속도 (2차 미분)
        if len(seq) >= 3:
            diff1 = seq[-1] - seq[-2]
            diff2 = seq[-2] - seq[-3]
            features[f'{prefix}_accel'] = diff1 - diff2
        else:
            features[f'{prefix}_accel'] = 0
    
    return features


def create_threshold_features(seq, prefix, thresholds):
    """임계값 Feature 생성"""
    features = {}
    
    for th in thresholds:
        features[f'{prefix}_over_{th}'] = np.sum(seq > th)
    
    return features


def create_rolling_features(seq, prefix, grade):
    """롤링 통계 Feature 생성"""
    features = {}
    
    windows = [10, 20, 30]
    if grade == 'S':
        windows = [5, 10, 20, 30, 60]
    
    for w in windows:
        if len(seq) >= w:
            rolling_mean = np.mean(seq[-w:])
            rolling_std = np.std(seq[-w:])
            features[f'{prefix}_roll{w}_mean'] = rolling_mean
            features[f'{prefix}_roll{w}_std'] = rolling_std
    
    return features


def create_special_features(seq, prefix, grade):
    """특수 Feature 생성"""
    features = {}
    
    if grade in ['S', 'B']:  # S등급 또는 queue_gap(B)
        # 급등 감지 (최근 10분 대비 현재값 급등)
        recent_mean = np.mean(seq[-10:])
        if recent_mean > 0:
            surge_ratio = seq[-1] / recent_mean
            features[f'{prefix}_surge'] = 1 if surge_ratio > 1.3 else 0
        else:
            features[f'{prefix}_surge'] = 0
    
    if grade == 'S':
        # 피크 카운트 (상위 10% 초과 횟수)
        threshold_90 = np.percentile(seq, 90)
        features[f'{prefix}_peak_count'] = np.sum(seq > threshold_90)
        
        # 변동성 (표준편차 / 평균)
        mean_val = np.mean(seq)
        if mean_val > 0:
            features[f'{prefix}_volatility'] = np.std(seq) / mean_val
        else:
            features[f'{prefix}_volatility'] = 0
    
    if prefix == 'queue_gap':
        # 누적 증가량
        features[f'{prefix}_cumsum'] = np.sum(np.diff(seq)[np.diff(seq) > 0])
        
        # 연속 증가 횟수
        diff = np.diff(seq)
        consecutive = 0
        max_consecutive = 0
        for d in diff:
            if d > 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        features[f'{prefix}_consec_increase'] = max_consecutive
    
    return features


# ==============================================================================
# 4. 컬럼별 Feature 생성
# ==============================================================================

def create_column_features(seq, col_name, config):
    """단일 컬럼의 모든 Feature 생성"""
    features = {}
    grade = config['grade']
    thresholds = config['thresholds']
    
    # 짧은 prefix 생성
    prefix = col_name.replace('M14.QUE.', '').replace('M14B.QUE.', 'B_')
    prefix = prefix.replace('.', '_').replace('CURRENTQCNT', 'QCNT')
    prefix = prefix[:30]  # 최대 30자
    
    # 각 종류별 Feature 생성
    features.update(create_stat_features(seq, prefix, grade))
    features.update(create_lag_features(seq, prefix, grade))
    features.update(create_trend_features(seq, prefix, grade))
    features.update(create_threshold_features(seq, prefix, thresholds))
    features.update(create_rolling_features(seq, prefix, grade))
    features.update(create_special_features(seq, prefix, grade))
    
    return features


# ==============================================================================
# 5. 상호작용 Feature 생성
# ==============================================================================

def create_interaction_features(row_dict):
    """상호작용 Feature 생성 (25개)"""
    features = {}
    
    # 현재값 추출
    m14b = row_dict.get('M14AM14B', np.zeros(1))[-1]
    m14bsum = row_dict.get('M14AM14BSUM', np.zeros(1))[-1]
    m10a = row_dict.get('M10AM14A', np.zeros(1))[-1]
    m16m14a = row_dict.get('M16M14A', np.zeros(1))[-1]
    m14am16sum = row_dict.get('M14AM16SUM', np.zeros(1))[-1]
    vertical = row_dict.get('M14.QUE.SENDFAB.VERTICALQUEUECOUNT', np.zeros(1))[-1]
    south = row_dict.get('M14.QUE.CNV.SOUTHCURRENTQCNT', np.zeros(1))[-1]
    north = row_dict.get('M14.QUE.CNV.NORTHCURRENTQCNT', np.zeros(1))[-1]
    ohtutil = row_dict.get('M14.QUE.OHT.OHTUTIL', np.zeros(1))[-1]
    ohtqcnt = row_dict.get('M14.QUE.OHT.CURRENTOHTQCNT', np.zeros(1))[-1]
    transport = row_dict.get('M14.QUE.ALL.TRANSPORT4MINOVERCNT', np.zeros(1))[-1]
    gap = row_dict.get('queue_gap', np.zeros(1))[-1]
    
    # gap trend (최근 10분 변화)
    gap_seq = row_dict.get('queue_gap', np.zeros(10))
    gap_trend10 = gap_seq[-1] - gap_seq[-10] if len(gap_seq) >= 10 else 0
    
    # =========================================================================
    # 패턴 Feature (기존 4개 + 신규 7개)
    # =========================================================================
    
    # 기존 패턴
    features['gold_strict'] = 1 if (m14b > 540 and m14bsum > 620) else 0
    features['gold_normal'] = 1 if (m14b > 520 and m14bsum > 600) else 0
    features['triple_check'] = 1 if (m14b > 520 and m14bsum > 600 and gap > 250) else 0
    features['quad_check'] = 1 if (m14b > 520 and m14bsum > 600 and gap > 250 and transport > 145) else 0
    
    # 신규 패턴
    features['vertical_gold'] = 1 if (vertical > 230 and m14b > 500) else 0
    features['cnv_imbalance'] = 1 if abs(south - north) > 20 else 0
    features['cnv_total_high'] = 1 if (south + north) > 100 else 0
    features['oht_overload'] = 1 if (ohtutil > 90 and ohtqcnt > 900) else 0
    features['m16_surge'] = 1 if (m16m14a > 110 and m14am16sum > 200) else 0
    features['inflow_surge'] = 1 if (m10a > 55 and m16m14a > 100) else 0
    features['queue_accel_danger'] = 1 if (gap > 200 and gap_trend10 > 30) else 0
    
    # =========================================================================
    # 곱셈/비율 Feature (14개)
    # =========================================================================
    
    features['m14b_x_sum'] = m14b * m14bsum / 1000
    features['m14b_x_vertical'] = m14b * vertical / 1000
    features['gap_x_m14b'] = gap * m14b / 1000
    features['gap_x_oht'] = gap * ohtqcnt / 10000
    features['cnv_ratio'] = south / (north + 1)
    features['cnv_total'] = south + north
    features['inflow_total'] = m10a + m16m14a
    features['inflow_ratio'] = m16m14a / (m10a + 1)
    features['oht_efficiency'] = ohtqcnt / (ohtutil + 1)
    features['m14_m16_ratio'] = m14bsum / (m14am16sum + 1)
    features['m14b_x_sum_mean'] = np.mean(row_dict.get('M14AM14B', [0]) * row_dict.get('M14AM14BSUM', [0])) / 1000
    features['ratio_m14b_total'] = m14b / (row_dict.get('TOTALCNT', [1])[-1] + 1)
    features['ratio_sum_total'] = m14bsum / (row_dict.get('TOTALCNT', [1])[-1] + 1)
    features['gap_x_trans'] = gap * transport / 100
    
    return features


# ==============================================================================
# 6. TOTALCNT 자체 Feature 생성
# ==============================================================================

def create_totalcnt_features(seq):
    """TOTALCNT 자체 Feature 생성 (15개)"""
    features = {}
    
    features['total_mean'] = np.mean(seq)
    features['total_max'] = np.max(seq)
    features['total_min'] = np.min(seq)
    features['total_std'] = np.std(seq)
    features['total_median'] = np.median(seq)
    
    features['total_current'] = seq[-1]
    features['total_last5'] = np.mean(seq[-5:])
    features['total_last10'] = np.mean(seq[-10:])
    features['total_last30'] = np.mean(seq[-30:])
    
    try:
        features['total_slope'] = np.polyfit(np.arange(len(seq)), seq, 1)[0]
    except:
        features['total_slope'] = 0
    
    features['total_trend10'] = seq[-1] - seq[-10]
    features['total_trend20'] = seq[-1] - seq[-20]
    
    features['total_over_1600'] = np.sum(seq >= 1600)
    features['total_over_1650'] = np.sum(seq >= 1650)
    features['total_over_1700'] = np.sum(seq >= 1700)
    
    return features


# ==============================================================================
# 7. 전체 Feature 생성 (시퀀스 → Feature 벡터)
# ==============================================================================

def create_all_features(row_dict):
    """모든 Feature 생성 (하나의 시퀀스 → 하나의 Feature 벡터)"""
    features = {}
    
    # 1. 각 핵심 컬럼별 Feature
    for col_name, config in CORE_COLUMNS.items():
        if col_name in row_dict:
            seq = np.array(row_dict[col_name])
            col_features = create_column_features(seq, col_name, config)
            features.update(col_features)
    
    # 2. queue_gap Feature (파생변수)
    if 'queue_gap' in row_dict:
        seq = np.array(row_dict['queue_gap'])
        gap_features = create_column_features(seq, 'queue_gap', QUEUE_GAP_CONFIG)
        features.update(gap_features)
    
    # 3. 상호작용 Feature
    interaction_features = create_interaction_features(row_dict)
    features.update(interaction_features)
    
    # 4. TOTALCNT 자체 Feature
    if 'TOTALCNT' in row_dict:
        totalcnt_features = create_totalcnt_features(np.array(row_dict['TOTALCNT']))
        features.update(totalcnt_features)
    
    return features


# ==============================================================================
# 8. 전체 데이터셋 Feature 생성
# ==============================================================================

def generate_feature_dataset(df, pred_offset=10):
    """전체 데이터셋에 대해 Feature 생성"""
    print(f"\n🔄 Feature 생성 시작 (예측 오프셋: {pred_offset}분)")
    print(f"   시퀀스 길이: {SEQ_LEN}분")
    
    results = []
    total = len(df) - SEQ_LEN - pred_offset
    
    # 필요한 컬럼 목록
    required_cols = list(CORE_COLUMNS.keys()) + ['TOTALCNT', 'queue_gap', 
                                                  'M14.QUE.ALL.CURRENTQCREATED', 
                                                  'M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 없는 컬럼 확인
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ⚠️ 누락 컬럼: {missing}")
    
    for i in range(SEQ_LEN, len(df) - pred_offset):
        # 시퀀스 데이터 추출
        row_dict = {}
        
        for col in df.columns:
            if col not in ['CURRTIME']:
                row_dict[col] = df[col].iloc[i-SEQ_LEN:i].values
        
        # Feature 생성
        features = create_all_features(row_dict)
        
        # 메타 정보 추가
        if 'CURRTIME' in df.columns:
            features['CURRTIME'] = df['CURRTIME'].iloc[i-1]
        
        features['current_TOTALCNT'] = df['TOTALCNT'].iloc[i-1]
        features['target_TOTALCNT'] = df['TOTALCNT'].iloc[i + pred_offset - 1]
        features['target_delta'] = features['target_TOTALCNT'] - features['current_TOTALCNT']
        features['is_danger'] = 1 if features['target_TOTALCNT'] >= 1700 else 0
        
        results.append(features)
        
        # 진행 상황 출력
        if (i - SEQ_LEN) % 5000 == 0:
            print(f"  진행: {i-SEQ_LEN:,}/{total:,} ({(i-SEQ_LEN)/total*100:.1f}%)")
            gc.collect()
    
    df_features = pd.DataFrame(results)
    print(f"\n✅ Feature 생성 완료!")
    print(f"   총 샘플: {len(df_features):,}개")
    print(f"   총 Feature: {len(df_features.columns)}개")
    
    return df_features


# ==============================================================================
# 9. 메인 실행
# ==============================================================================

if __name__ == '__main__':
    # 1. 데이터 로드
    df = load_data('*.CSV')
    
    if df is None:
        print("\n⚠️ CSV 파일을 찾을 수 없습니다.")
        print("   사용법: 스크립트와 같은 폴더에 CSV 파일을 두세요.")
        exit(1)
    
    # 2. 전처리
    df = preprocess_data(df)
    
    # 3. Feature 생성 (10분 예측)
    df_features = generate_feature_dataset(df, pred_offset=10)
    
    # 4. 저장
    output_file = 'step8_features_10min.csv'
    df_features.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장 완료: {output_file}")
    
    # 5. Feature 목록 저장
    feature_cols = [c for c in df_features.columns if c not in ['CURRTIME', 'current_TOTALCNT', 
                                                                  'target_TOTALCNT', 'target_delta', 'is_danger']]
    with open('step8_feature_list.txt', 'w', encoding='utf-8') as f:
        f.write(f"총 Feature 수: {len(feature_cols)}\n\n")
        for i, col in enumerate(feature_cols, 1):
            f.write(f"{i:3d}. {col}\n")
    
    print(f"📋 Feature 목록 저장: step8_feature_list.txt")
    
    # 6. 요약 통계
    print("\n" + "="*80)
    print("📊 생성 결과 요약")
    print("="*80)
    print(f"  총 샘플 수: {len(df_features):,}개")
    print(f"  총 Feature 수: {len(feature_cols)}개")
    print(f"  1700+ 샘플 수: {df_features['is_danger'].sum():,}개 ({df_features['is_danger'].mean()*100:.2f}%)")
    print(f"  타겟 평균: {df_features['target_TOTALCNT'].mean():.2f}")
    print(f"  타겟 표준편차: {df_features['target_TOTALCNT'].std():.2f}")
    
    print("\n✅ 8단계 완료!")
    print("   → 9단계: 생성된 Feature로 2차 Selection 진행")