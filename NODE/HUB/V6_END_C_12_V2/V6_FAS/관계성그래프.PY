# -*- coding: utf-8 -*-
"""
V6 Feature 관계성 분석 및 시각화
- Feature 간 상관계수 계산
- 인터랙티브 히트맵 생성
- 네트워크 그래프 생성
- 카테고리별 관계도
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Feature 컬럼 그룹
FEATURE_COLS = {
    'storage': ['M16A_3F_STORAGE_UTIL'],
    'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
    'hub': ['HUBROOMTOTAL'],
    'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
    'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
    'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
    'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
}

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'

def create_features_for_correlation(df, start_idx=30):
    """상관계수 분석용 Feature 생성"""
    features_list = []
    labels = []
    
    for i in range(start_idx, len(df) - 10):
        seq_target = df[TARGET_COL].iloc[i-30:i].values
        
        # 타겟 기본 통계
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': seq_target[-1],
            'target_last_5_mean': np.mean(seq_target[-5:]),
            'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
        }
        
        # Momentum Feature
        features['target_acceleration'] = (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5
        features['target_is_rising'] = 1 if seq_target[-1] > seq_target[-5] else 0
        features['target_rapid_rise'] = 1 if (seq_target[-1] - seq_target[-5] > 10) else 0
        features['target_last_10_mean'] = np.mean(seq_target[-10:])
        
        # 각 컬럼 그룹별 Feature
        for group_name, cols in FEATURE_COLS.items():
            for col in cols:
                if col not in df.columns:
                    continue
                
                col_seq = df[col].iloc[i-30:i].values
                
                if group_name == 'maxcapa':
                    features[f'{col}_last_value'] = col_seq[-1]
                
                elif group_name in ['cmd', 'storage', 'fs_storage', 'hub']:
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_std'] = np.std(col_seq)
                    features[f'{col}_max'] = np.max(col_seq)
                    features[f'{col}_min'] = np.min(col_seq)
                    features[f'{col}_last_value'] = col_seq[-1]
                    features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                
                else:
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_last_value'] = col_seq[-1]
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
        
        # FS Storage Feature
        if 'CD_M163FSTORAGEUSE' in df.columns and 'CD_M163FSTORAGETOTAL' in df.columns:
            storage_use = df['CD_M163FSTORAGEUSE'].iloc[i-30:i].values
            storage_total = df['CD_M163FSTORAGETOTAL'].iloc[i-30:i].values
            storage_util = df['CD_M163FSTORAGEUTIL'].iloc[i-30:i].values
            
            features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
            features['storage_remaining'] = storage_total[-1] - storage_use[-1]
            features['storage_util_last'] = storage_util[-1]
            features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
            features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
        
        # HUBROOMTOTAL Feature
        if 'HUBROOMTOTAL' in df.columns:
            hub_seq = df['HUBROOMTOTAL'].iloc[i-30:i].values
            hub_last = hub_seq[-1]
            
            features['hub_critical'] = 1 if hub_last < 590 else 0
            features['hub_high'] = 1 if hub_last < 610 else 0
            features['hub_warning'] = 1 if hub_last < 620 else 0
            features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
            
            if 'CD_M163FSTORAGEUTIL' in df.columns:
                storage_util_last = df['CD_M163FSTORAGEUTIL'].iloc[i-1]
                features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
        
        # 유입-유출, CMD
        inflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['inflow'] if col in df.columns)
        outflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['outflow'] if col in df.columns)
        features['net_flow'] = inflow_sum - outflow_sum
        
        cmd_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['cmd'] if col in df.columns)
        features['total_cmd'] = cmd_sum
        features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
        features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
        
        if 'HUBROOMTOTAL' in df.columns:
            hub_last = df['HUBROOMTOTAL'].iloc[i-1]
            features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
        
        if 'M16A_3F_STORAGE_UTIL' in df.columns:
            storage_util = df['M16A_3F_STORAGE_UTIL'].iloc[i-1]
            features['storage_util_critical'] = 1 if storage_util >= 205 else 0
            features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
        
        # 급증 위험도 점수
        features['surge_risk_score'] = (
            features.get('hub_high', 0) * 3 +
            features.get('storage_util_critical', 0) * 2 +
            features.get('total_cmd_low', 0) * 1 +
            features.get('storage_util_high', 0) * 1
        )
        
        # 급증 임박 신호
        features['surge_imminent'] = 1 if (
            seq_target[-1] > 272 and
            features.get('target_acceleration', 0) > 0.2 and
            features.get('hub_high', 0) == 1
        ) else 0
        
        features_list.append(features)
        labels.append(df[TARGET_COL].iloc[i:i+10].max())
    
    return pd.DataFrame(features_list), np.array(labels)


def create_correlation_heatmap(corr_matrix, title="Feature 상관계수 히트맵"):
    """상관계수 히트맵 생성"""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="상관계수")
    ))
    
    fig.update_layout(
        title=title,
        width=1400,
        height=1400,
        xaxis_tickangle=-45,
        font=dict(size=10)
    )
    
    return fig


def create_target_correlation_chart(df_features, labels):
    """타겟과의 상관계수 막대 그래프"""
    # 타겟 추가
    df_with_target = df_features.copy()
    df_with_target['TARGET'] = labels
    
    # 타겟과의 상관계수 계산
    target_corr = df_with_target.corr()['TARGET'].drop('TARGET').sort_values(ascending=False)
    
    # 상위 30개만
    top_30 = target_corr.head(30)
    
    fig = go.Figure()
    
    colors = ['red' if x > 0.8 else 'orange' if x > 0.6 else 'lightblue' for x in top_30.values]
    
    fig.add_trace(go.Bar(
        x=top_30.values,
        y=top_30.index,
        orientation='h',
        marker=dict(color=colors),
        text=np.round(top_30.values, 3),
        textposition='auto'
    ))
    
    fig.update_layout(
        title="타겟(CURRENT_M16A_3F_JOB_2)과의 상관계수 Top 30",
        xaxis_title="상관계수",
        yaxis_title="Feature",
        height=900,
        width=1000,
        showlegend=False
    )
    
    return fig, target_corr


def create_category_correlation_matrix(df_features):
    """카테고리별 상관계수 매트릭스"""
    # Feature를 카테고리로 분류
    category_features = {
        'Target_기본통계': [c for c in df_features.columns if c.startswith('target_') and 
                          c not in ['target_acceleration', 'target_is_rising', 'target_rapid_rise', 'target_last_10_mean']],
        'Momentum': ['target_acceleration', 'target_is_rising', 'target_rapid_rise', 'target_last_10_mean'],
        'Hub': [c for c in df_features.columns if 'HUBROOM' in c or c.startswith('hub_')],
        'Storage': [c for c in df_features.columns if 'M16A_3F_STORAGE' in c or c.startswith('storage_')],
        'FS_Storage': [c for c in df_features.columns if 'CD_M163F' in c],
        'CMD': [c for c in df_features.columns if 'CMD' in c or c.startswith('total_cmd')],
        'Inflow': [c for c in df_features.columns if 'TO_HUB_JOB' in c],
        'Outflow': [c for c in df_features.columns if 'M16A_3F_TO_' in c and '_JOB' in c],
        '위험신호': ['surge_risk_score', 'surge_imminent', 'hub_storage_risk', 'hub_cmd_bottleneck', 'net_flow']
    }
    
    # 각 카테고리 대표 Feature 선택
    selected_features = []
    feature_names = []
    
    for cat, features in category_features.items():
        valid_features = [f for f in features if f in df_features.columns]
        if valid_features:
            # 각 카테고리에서 최대 5개씩
            selected = valid_features[:5]
            selected_features.extend(selected)
            feature_names.extend([f"{cat}_{f}" for f in selected])
    
    # 상관계수 계산
    corr_matrix = df_features[selected_features].corr()
    corr_matrix.index = feature_names
    corr_matrix.columns = feature_names
    
    return corr_matrix


def main():
    """메인 실행 함수"""
    print("="*80)
    print("🔍 V6 Feature 관계성 분석 시작")
    print("="*80)
    
    # 데이터 로드
    print("\n[1] 데이터 로드 중...")
    try:
        df = pd.read_csv('20250904_TO_20251020.csv', on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df = pd.read_csv('20250904_TO_20251020.csv', on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df = pd.read_csv('20250904_TO_20251020.csv', on_bad_lines='skip', encoding='euc-kr', low_memory=False)
    
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
    
    print(f"✅ 데이터 로드 완료: {len(df)}개 행")
    
    # Feature 생성
    print("\n[2] Feature 생성 중...")
    df_features, labels = create_features_for_correlation(df)
    print(f"✅ Feature 생성 완료: {len(df_features.columns)}개 Feature, {len(df_features)}개 샘플")
    
    # 상관계수 계산
    print("\n[3] 상관계수 계산 중...")
    corr_matrix = df_features.corr()
    print(f"✅ 상관계수 계산 완료: {corr_matrix.shape[0]}x{corr_matrix.shape[1]} 매트릭스")
    
    # 1. 전체 상관계수 히트맵
    print("\n[4] 전체 상관계수 히트맵 생성 중...")
    fig1 = create_correlation_heatmap(corr_matrix, "V6 Feature 간 상관계수 히트맵 (전체)")
    fig1.write_html('V6_Feature_상관계수_히트맵_전체.html')
    print("✅ 저장: V6_Feature_상관계수_히트맵_전체.html")
    
    # 2. 타겟과의 상관계수
    print("\n[5] 타겟 상관계수 차트 생성 중...")
    fig2, target_corr = create_target_correlation_chart(df_features, labels)
    fig2.write_html('V6_타겟_상관계수_Top30.html')
    print("✅ 저장: V6_타겟_상관계수_Top30.html")
    
    # 타겟 상관계수 CSV 저장
    target_corr_df = pd.DataFrame({
        'Feature': target_corr.index,
        '타겟_상관계수': target_corr.values,
        '절대값': np.abs(target_corr.values)
    }).sort_values('절대값', ascending=False)
    target_corr_df.to_csv('V6_타겟_상관계수_전체.csv', index=False, encoding='utf-8-sig')
    print("✅ 저장: V6_타겟_상관계수_전체.csv")
    
    # 3. 카테고리별 상관계수 히트맵
    print("\n[6] 카테고리별 상관계수 히트맵 생성 중...")
    category_corr = create_category_correlation_matrix(df_features)
    fig3 = create_correlation_heatmap(category_corr, "V6 Feature 카테고리별 상관계수 (대표 Feature)")
    fig3.write_html('V6_카테고리별_상관계수_히트맵.html')
    print("✅ 저장: V6_카테고리별_상관계수_히트맵.html")
    
    # 4. 고상관 Feature 쌍 추출
    print("\n[7] 고상관 Feature 쌍 추출 중...")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # 상관계수 0.7 이상
                high_corr_pairs.append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    '상관계수': round(corr_val, 3),
                    '절대값': round(abs(corr_val), 3)
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('절대값', ascending=False)
    high_corr_df.to_csv('V6_고상관_Feature쌍.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 저장: V6_고상관_Feature쌍.csv ({len(high_corr_df)}개 쌍)")
    
    # 5. 통계 요약
    print("\n[8] 통계 요약 생성 중...")
    
    print("\n" + "="*80)
    print("📊 분석 결과 요약")
    print("="*80)
    
    print(f"\n🎯 타겟과 가장 강한 상관관계 Top 10:")
    for idx, (feat, corr) in enumerate(target_corr.head(10).items(), 1):
        print(f"  {idx:2d}. {feat:40s}: {corr:+.3f}")
    
    print(f"\n⚠️ 타겟과 가장 약한 상관관계 Bottom 10:")
    for idx, (feat, corr) in enumerate(target_corr.tail(10).items(), 1):
        print(f"  {idx:2d}. {feat:40s}: {corr:+.3f}")
    
    print(f"\n🔗 가장 강한 Feature 간 상관관계 Top 10:")
    for idx, row in high_corr_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['Feature1']:35s} ↔ {row['Feature2']:35s}: {row['상관계수']:+.3f}")
    
    print("\n" + "="*80)
    print("✅ 분석 완료!")
    print("="*80)
    
    print("\n📁 생성된 파일:")
    print("  1. V6_Feature_상관계수_히트맵_전체.html")
    print("  2. V6_타겟_상관계수_Top30.html")
    print("  3. V6_카테고리별_상관계수_히트맵.html")
    print("  4. V6_타겟_상관계수_전체.csv")
    print("  5. V6_고상관_Feature쌍.csv")


if __name__ == '__main__':
    main()