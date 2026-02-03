# -*- coding: utf-8 -*-
"""
================================================================================
V10_4 신규 컬럼 분석 코드
추가된 컬럼: AHTO+AHSO, REWORK, AHSO

[컬럼 설명]
- AHTO+AHSO: Q-Time 위반 알람 합계 (늦음+빠름)
- REWORK: Q-Time 위반 후 재작업 처분된 Lot 수
- AHSO: Q-Time 하한 미달 알람 (너무 빨리 이동)
================================================================================
"""

import os
import sys
import warnings
import glob
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
CONFIG = {
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

# 신규 추가 컬럼 (Q-Time Interlock 관련)
NEW_COLUMNS = ['AHTO+AHSO', 'REWORK', 'AHSO']

print("=" * 70)
print("🔍 V10_4 신규 컬럼 분석 (Q-Time Interlock 알람)")
print("   AHTO+AHSO: 대기시간 위반 알람 합계")
print("   REWORK: 재작업 처분 Lot")
print("   AHSO: 대기시간 미달 (빠름) 알람")
print("=" * 70)

# ============================================================================
# 데이터 로드
# ============================================================================
def load_csv(filepath):
    """CSV 파일 로드 (여러 인코딩 시도)"""
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, on_bad_lines='skip')
            return df
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"파일 인코딩을 알 수 없습니다: {filepath}")


def load_data(path_input):
    """단일 파일 또는 패턴으로 데이터 로드"""
    print(f"\n[1/6] 데이터 로드 중...")
    
    # 패턴이면 glob, 아니면 단일 파일
    if '*' in path_input:
        files = sorted(glob.glob(path_input))
    elif os.path.exists(path_input):
        files = [path_input]
    else:
        raise ValueError(f"파일을 찾을 수 없습니다: {path_input}")
    
    if not files:
        raise ValueError(f"매칭되는 파일이 없습니다: {path_input}")
    
    dfs = []
    for f in files:
        df = load_csv(f)
        print(f"  - {os.path.basename(f)}: {len(df):,}행")
        dfs.append(df)
    
    result = pd.concat(dfs, ignore_index=True)
    print(f"  → 총 데이터: {len(result):,}행, {len(result.columns)}컬럼")
    return result


def analyze_new_columns(df):
    """신규 컬럼 기본 통계 분석"""
    print("\n[2/6] 신규 컬럼 기본 통계...")
    print("-" * 60)
    
    results = {}
    for col in NEW_COLUMNS:
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            non_zero = data[data > 0]
            
            stats = {
                '컬럼명': col,
                '존재': '✅',
                '개수': len(data.dropna()),
                '평균': data.mean(),
                '표준편차': data.std(),
                '최소': data.min(),
                '25%': data.quantile(0.25),
                '중앙값': data.median(),
                '75%': data.quantile(0.75),
                '최대': data.max(),
                '0값비율': (data == 0).sum() / len(data) * 100,
                '양수평균': non_zero.mean() if len(non_zero) > 0 else 0,
                '양수개수': len(non_zero),
            }
            results[col] = stats
            
            print(f"\n📊 {col}")
            print(f"   평균: {stats['평균']:.3f} | 표준편차: {stats['표준편차']:.3f}")
            print(f"   범위: {stats['최소']:.0f} ~ {stats['최대']:.0f}")
            print(f"   0값 비율: {stats['0값비율']:.1f}% ({len(data) - stats['양수개수']:,}개)")
            print(f"   양수일때 평균: {stats['양수평균']:.2f} ({stats['양수개수']:,}개)")
        else:
            print(f"\n⚠️ {col}: 컬럼 없음")
            results[col] = {'컬럼명': col, '존재': '❌'}
    
    return results


def analyze_correlation_with_target(df):
    """TOTALCNT와의 상관관계 분석"""
    print("\n[3/6] TOTALCNT와의 상관관계 분석...")
    print("-" * 60)
    
    target = pd.to_numeric(df[CONFIG['target_column']], errors='coerce')
    
    correlations = {}
    for col in NEW_COLUMNS:
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            corr = target.corr(data)
            correlations[col] = corr
            
            # 상관관계 해석
            if abs(corr) >= 0.4:
                strength = "🔥 강함"
            elif abs(corr) >= 0.2:
                strength = "⚡ 보통"
            else:
                strength = "💤 약함"
            
            print(f"  {col}: {corr:.4f} {strength}")
    
    return correlations


def analyze_danger_zone(df):
    """1700+ 위험 구간에서의 신규 컬럼 특성"""
    print("\n[4/6] 위험 구간(1700+) 분석...")
    print("-" * 60)
    
    target = pd.to_numeric(df[CONFIG['target_column']], errors='coerce')
    danger_mask = target >= CONFIG['limit_value']
    safe_mask = target < CONFIG['limit_value']
    
    danger_count = danger_mask.sum()
    safe_count = safe_mask.sum()
    
    print(f"  전체: {len(df):,}개")
    print(f"  위험(1700+): {danger_count:,}개 ({100*danger_count/len(df):.2f}%)")
    print(f"  안전(<1700): {safe_count:,}개")
    
    analysis = {}
    for col in NEW_COLUMNS:
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            
            danger_mean = data[danger_mask].mean()
            safe_mean = data[safe_mask].mean()
            diff = danger_mean - safe_mean
            diff_pct = (diff / safe_mean * 100) if safe_mean != 0 else 0
            
            analysis[col] = {
                '위험평균': danger_mean,
                '안전평균': safe_mean,
                '차이': diff,
                '차이%': diff_pct,
            }
            
            # 차이 해석
            if abs(diff_pct) >= 30:
                indicator = "🔥 유의미한 차이!"
            elif abs(diff_pct) >= 10:
                indicator = "⚡ 주목할 만한 차이"
            else:
                indicator = "💤 차이 적음"
            
            print(f"\n  📊 {col}")
            print(f"     위험 구간 평균: {danger_mean:.2f}")
            print(f"     안전 구간 평균: {safe_mean:.2f}")
            print(f"     차이: {diff:+.2f} ({diff_pct:+.1f}%) {indicator}")
    
    return analysis


def analyze_lead_indicator(df, look_ahead=10):
    """선행 지표 분석 (10분 후 1700+ 예측력)"""
    print(f"\n[5/6] 선행 지표 분석 ({look_ahead}분 후 예측력)...")
    print("-" * 60)
    
    target = pd.to_numeric(df[CONFIG['target_column']], errors='coerce')
    
    # look_ahead분 후 위험 여부
    future_danger = (target.shift(-look_ahead) >= CONFIG['limit_value']).astype(int)
    
    lead_analysis = {}
    for col in NEW_COLUMNS:
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            
            # 현재 값과 미래 위험의 상관관계
            valid_mask = ~(data.isna() | future_danger.isna())
            if valid_mask.sum() > 100:
                corr = data[valid_mask].corr(future_danger[valid_mask])
                
                # 임계값 분석
                thresholds = [
                    data.quantile(0.5),
                    data.quantile(0.75),
                    data.quantile(0.9),
                ]
                
                lead_analysis[col] = {
                    '선행상관': corr,
                    '임계값분석': []
                }
                
                print(f"\n  📊 {col}")
                print(f"     {look_ahead}분 후 위험과의 상관: {corr:.4f}")
                
                for thresh in thresholds:
                    high_mask = (data >= thresh) & valid_mask
                    if high_mask.sum() > 0:
                        danger_rate = future_danger[high_mask].mean() * 100
                        print(f"     {col} >= {thresh:.1f}일 때 위험 확률: {danger_rate:.2f}%")
    
    return lead_analysis


def generate_report(df, stats, correlations, danger_analysis, lead_analysis):
    """분석 보고서 생성"""
    print("\n[6/6] 분석 요약 보고서...")
    print("=" * 70)
    print("📋 V10_4 모델 통합 권장사항")
    print("=" * 70)
    
    recommendations = []
    
    for col in NEW_COLUMNS:
        if col in correlations:
            corr = correlations[col]
            danger_diff = danger_analysis.get(col, {}).get('차이%', 0)
            
            score = abs(corr) * 50 + abs(danger_diff) * 0.5
            
            if score >= 15:
                rec = f"✅ {col}: 강력 추천 (상관={corr:.3f}, 위험구간차이={danger_diff:+.1f}%)"
                recommendations.append((col, 'important', score))
            elif score >= 5:
                rec = f"⚡ {col}: 보조 변수로 추천 (상관={corr:.3f}, 위험구간차이={danger_diff:+.1f}%)"
                recommendations.append((col, 'auxiliary', score))
            else:
                rec = f"💤 {col}: 낮은 우선순위 (상관={corr:.3f}, 위험구간차이={danger_diff:+.1f}%)"
                recommendations.append((col, 'low', score))
            
            print(f"  {rec}")
    
    # FEATURE_GROUPS 추가 코드 출력
    print("\n" + "-" * 60)
    print("📝 V10_4 FEATURE_GROUPS 추가 코드:")
    print("-" * 60)
    
    important_cols = [r[0] for r in recommendations if r[1] == 'important']
    auxiliary_cols = [r[0] for r in recommendations if r[1] in ['auxiliary', 'low']]
    
    if important_cols:
        print(f"\n# important 그룹에 추가:")
        for col in important_cols:
            print(f"    '{col}',")
    
    if auxiliary_cols:
        print(f"\n# auxiliary 또는 pdt_new 그룹에 추가:")
        for col in auxiliary_cols:
            print(f"    '{col}',")
    
    return recommendations


def analyze_time_pattern(df):
    """시간대별 알람 패턴 분석"""
    print("\n[추가] 시간대별 Q-Time 알람 패턴...")
    print("-" * 60)
    
    if 'CURRTIME' not in df.columns:
        print("  ⚠️ CURRTIME 컬럼 없음")
        return None
    
    # 시간 파싱
    df['CURRTIME_DT'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    df['HOUR'] = df['CURRTIME_DT'].dt.hour
    
    for col in NEW_COLUMNS:
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            hourly = df.groupby('HOUR')[col].agg(['mean', 'sum', 'max'])
            
            # 알람이 많은 시간대 찾기
            top_hours = hourly['sum'].nlargest(3)
            if top_hours.sum() > 0:
                print(f"\n  📊 {col} - 알람 많은 시간대:")
                for hour, val in top_hours.items():
                    print(f"     {hour:02d}시: 총 {val:.0f}건")
    
    return df


def main():
    # CSV 파일 경로 입력
    print("\n" + "=" * 70)
    csv_path = input("📂 CSV 파일 경로 입력 (예: M14_학습.CSV 또는 *.CSV): ").strip()
    
    if not csv_path:
        print("❌ 파일 경로를 입력하세요.")
        return None
    
    # 데이터 로드
    try:
        df = load_data(csv_path)
    except ValueError as e:
        print(f"\n❌ 오류: {e}")
        return None
    
    # 신규 컬럼 존재 확인
    print("\n[컬럼 확인]")
    found_cols = []
    for col in NEW_COLUMNS:
        if col in df.columns:
            print(f"  ✅ {col} 존재")
            found_cols.append(col)
        else:
            print(f"  ❌ {col} 없음")
    
    if not found_cols:
        print("\n⚠️ 분석할 신규 컬럼이 없습니다!")
        return None
    
    # 분석 실행
    stats = analyze_new_columns(df)
    correlations = analyze_correlation_with_target(df)
    danger_analysis = analyze_danger_zone(df)
    lead_analysis = analyze_lead_indicator(df, look_ahead=10)
    
    # 시간대별 패턴 분석
    df = analyze_time_pattern(df)
    
    # 보고서 생성
    recommendations = generate_report(df, stats, correlations, danger_analysis, lead_analysis)
    
    print("\n" + "=" * 70)
    print("✅ 분석 완료!")
    print("=" * 70)
    
    return {
        'df': df,
        'stats': stats,
        'correlations': correlations,
        'danger_analysis': danger_analysis,
        'lead_analysis': lead_analysis,
        'recommendations': recommendations
    }


if __name__ == "__main__":
    results = main()