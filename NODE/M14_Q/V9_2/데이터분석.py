#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 AMHS 3개월 데이터 종합 분석
- EDA 및 변수 타입 정의
- 데이터 정합성 확인
- Feature Selection (XGBoost, SHAP, Boruta)
- 시각화 + 리포트 생성
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("🔬 AMHS 3개월 데이터 종합 분석")
print("="*80)

# ============================================================
# 1. 데이터 로드
# ============================================================
def load_data(file_path):
    """데이터 로드"""
    print("\n" + "="*60)
    print("📂 1. 데이터 로드")
    print("="*60)
    
    df = pd.read_csv(file_path, on_bad_lines='skip')
    print(f"✅ 파일: {file_path}")
    print(f"✅ 행: {len(df):,}개, 열: {df.shape[1]}개")
    
    # CURRTIME 파싱
    if 'CURRTIME' in df.columns:
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
        if df['CURRTIME'].notna().sum() > 0:
            print(f"✅ 기간: {df['CURRTIME'].min()} ~ {df['CURRTIME'].max()}")
            days = (df['CURRTIME'].max() - df['CURRTIME'].min()).days
            print(f"   총 {days}일 ({days/30:.1f}개월)")
    
    # queue_gap 생성 (핵심 파생변수!)
    if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
        df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
        print(f"✅ queue_gap 생성 완료")
    
    return df


# ============================================================
# 2. EDA - 변수 타입 분석
# ============================================================
def analyze_variable_types(df):
    """변수 타입 분석 (카테고리 vs 연속형)"""
    print("\n" + "="*60)
    print("📊 2. EDA - 변수 타입 분석")
    print("="*60)
    
    results = []
    for col in df.columns:
        if col == 'CURRTIME':
            continue
        
        unique_count = df[col].nunique()
        total_count = len(df)
        unique_ratio = unique_count / total_count * 100
        
        # 변수 타입 판정
        if unique_count <= 10:
            var_type = '카테고리(명목)'
        elif unique_count <= 30:
            var_type = '카테고리(순서)'
        elif unique_ratio < 5:
            var_type = '준카테고리'
        else:
            var_type = '연속형'
        
        results.append({
            '컬럼명': col,
            '데이터타입': str(df[col].dtype),
            '고유값수': unique_count,
            '고유값비율(%)': round(unique_ratio, 2),
            '변수타입': var_type,
            '결측치': df[col].isna().sum(),
            '최소값': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else '-',
            '최대값': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else '-',
            '평균': round(df[col].mean(), 2) if pd.api.types.is_numeric_dtype(df[col]) else '-'
        })
    
    df_types = pd.DataFrame(results)
    
    print(f"\n📌 변수 타입 분포:")
    print(df_types['변수타입'].value_counts().to_string())
    
    return df_types


# ============================================================
# 3. 타겟 변수 분포 분석
# ============================================================
def analyze_target(df, target='TOTALCNT'):
    """TOTALCNT 분포 분석"""
    print("\n" + "="*60)
    print(f"📊 3. 타겟 변수 분포: {target}")
    print("="*60)
    
    if target not in df.columns:
        print(f"❌ {target} 없음")
        return None
    
    print(f"  평균: {df[target].mean():.2f}")
    print(f"  중앙값: {df[target].median():.2f}")
    print(f"  표준편차: {df[target].std():.2f}")
    print(f"  최소/최대: {df[target].min():.0f} ~ {df[target].max():.0f}")
    
    # 위험 구간 분석
    print(f"\n📌 위험 구간 분포:")
    thresholds = [1600, 1650, 1700, 1750, 1800]
    for t in thresholds:
        cnt = (df[target] >= t).sum()
        print(f"  {t}+ : {cnt:,}개 ({cnt/len(df)*100:.3f}%)")
    
    return df[target].describe()


# ============================================================
# 4. 상관관계 분석
# ============================================================
def correlation_analysis(df, target='TOTALCNT', threshold=0.95):
    """상관관계 분석"""
    print("\n" + "="*60)
    print("📊 4. 상관관계 분석")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'CURRTIME']
    
    print(f"✅ 숫자형 변수: {len(numeric_cols)}개")
    
    corr_matrix = df[numeric_cols].corr()
    
    # 타겟과의 상관관계
    if target in corr_matrix.columns:
        target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
        
        print(f"\n📌 {target}과의 상관관계 TOP 20:")
        for i, (col, corr) in enumerate(target_corr.head(20).items()):
            print(f"  {i+1:2d}. {col}: {corr:.4f}")
    
    # 높은 상관관계 쌍 (제거 후보)
    print(f"\n📌 상관계수 {threshold} 이상 변수쌍 (제거 후보):")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                high_corr.append({
                    '변수1': corr_matrix.columns[i],
                    '변수2': corr_matrix.columns[j],
                    '상관계수': round(corr_matrix.iloc[i, j], 4)
                })
    
    if high_corr:
        df_high = pd.DataFrame(high_corr).sort_values('상관계수', ascending=False)
        print(f"  총 {len(df_high)}개 쌍")
        print(df_high.head(15).to_string(index=False))
    else:
        print(f"  없음")
    
    return corr_matrix, high_corr


# ============================================================
# 5. 데이터 정합성 확인
# ============================================================
def check_data_integrity(df):
    """결측치, 이상치, 중복값 확인"""
    print("\n" + "="*60)
    print("🔍 5. 데이터 정합성 확인")
    print("="*60)
    
    # 5-1. 결측치
    print("\n📌 5-1. 결측치 분석")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].sort_values(ascending=False)
    if len(missing_cols) > 0:
        print(f"  결측치 있는 컬럼: {len(missing_cols)}개")
        for col, cnt in missing_cols.head(10).items():
            print(f"    {col}: {cnt:,}개 ({cnt/len(df)*100:.2f}%)")
    else:
        print("  ✅ 결측치 없음")
    
    # 5-2. 이상치 (IQR)
    print("\n📌 5-2. 이상치 분석 (IQR)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_cols = []
    
    for col in numeric_cols[:50]:  # 상위 50개만
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        if outliers > len(df) * 0.05:  # 5% 이상
            outlier_cols.append({'컬럼': col, '이상치수': outliers, '비율(%)': round(outliers/len(df)*100, 2)})
    
    if outlier_cols:
        print(f"  이상치 5%+ 컬럼: {len(outlier_cols)}개")
        for o in outlier_cols[:10]:
            print(f"    {o['컬럼']}: {o['이상치수']:,}개 ({o['비율(%)']}%)")
    
    # 5-3. 중복값
    print("\n📌 5-3. 중복값 분석")
    full_dup = df.duplicated().sum()
    cols_no_time = [c for c in df.columns if c != 'CURRTIME']
    time_only_dup = df.duplicated(subset=cols_no_time).sum()
    
    print(f"  완전 중복: {full_dup:,}개")
    print(f"  시간만 다른 중복: {time_only_dup:,}개")
    
    # 5-4. 시간 연속성
    if 'CURRTIME' in df.columns and df['CURRTIME'].notna().sum() > 0:
        print("\n📌 5-4. 시간 연속성")
        df_sorted = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME')
        time_diff = df_sorted['CURRTIME'].diff().dt.total_seconds() / 60
        gaps = (time_diff > 1).sum()
        print(f"  누락 구간: {gaps:,}개")
        if gaps > 0:
            print(f"  최대 누락: {time_diff.max():.0f}분")
    
    return {'missing': len(missing_cols), 'duplicates': full_dup}


# ============================================================
# 6. queue_gap 분석 (핵심!)
# ============================================================
def analyze_queue_gap(df, target='TOTALCNT'):
    """queue_gap 특별 분석"""
    print("\n" + "="*60)
    print("🔥 6. queue_gap 분석 (핵심 Feature!)")
    print("="*60)
    
    if 'queue_gap' not in df.columns:
        print("❌ queue_gap 없음")
        return None
    
    print(f"  평균: {df['queue_gap'].mean():.2f}")
    print(f"  표준편차: {df['queue_gap'].std():.2f}")
    print(f"  최소/최대: {df['queue_gap'].min():.0f} ~ {df['queue_gap'].max():.0f}")
    
    # TOTALCNT 상관계수
    if target in df.columns:
        corr = df['queue_gap'].corr(df[target])
        print(f"\n📌 TOTALCNT 상관계수: {corr:.4f}")
    
    # 구간별 분석
    print(f"\n📌 구간별 평균 TOTALCNT:")
    bins = [-999, 100, 150, 200, 250, 300, 350, 400, 9999]
    labels = ['<100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400+']
    df['gap_bin'] = pd.cut(df['queue_gap'], bins=bins, labels=labels)
    
    for label in labels:
        mask = df['gap_bin'] == label
        if mask.sum() > 0:
            avg = df[mask][target].mean()
            rate_1700 = (df[mask][target] >= 1700).sum() / mask.sum() * 100
            print(f"    {label}: 평균 {avg:.0f}, 1700+ 확률 {rate_1700:.1f}%")
    
    df.drop('gap_bin', axis=1, inplace=True)
    return df['queue_gap'].describe()


# ============================================================
# 7. 황금 패턴 분석
# ============================================================
def analyze_golden_pattern(df, target='TOTALCNT'):
    """황금 패턴 분석"""
    print("\n" + "="*60)
    print("✨ 7. 황금 패턴 분석")
    print("="*60)
    
    if 'M14AM14B' not in df.columns or 'M14AM14BSUM' not in df.columns:
        print("❌ M14AM14B/M14AM14BSUM 없음")
        return None
    
    patterns = {
        '엄격(540/620)': (df['M14AM14B'] > 540) & (df['M14AM14BSUM'] > 620),
        '보통(520/600)': (df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600),
        '완화(500/580)': (df['M14AM14B'] > 500) & (df['M14AM14BSUM'] > 580),
    }
    
    for name, mask in patterns.items():
        cnt = mask.sum()
        if cnt > 0:
            avg = df[mask][target].mean()
            rate_1700 = (df[mask][target] >= 1700).sum() / cnt * 100
            print(f"  {name}:")
            print(f"    발생: {cnt:,}개 ({cnt/len(df)*100:.2f}%)")
            print(f"    평균 TOTALCNT: {avg:.0f}")
            print(f"    1700+ 확률: {rate_1700:.1f}%")
    
    return patterns


# ============================================================
# 8. Feature Importance (XGBoost)
# ============================================================
def feature_importance_analysis(df, target='TOTALCNT'):
    """XGBoost Feature Importance"""
    print("\n" + "="*60)
    print("🎯 8. Feature Importance (XGBoost)")
    print("="*60)
    
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
    except ImportError:
        print("❌ xgboost 필요: pip install xgboost")
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target and c != 'CURRTIME']
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target]
    
    print(f"✅ Feature: {len(feature_cols)}개, 샘플: {len(X):,}개")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=100, 
                              random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # 성능
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n📌 모델 성능: MAE={mae:.2f}, R²={r2:.4f}")
    
    # Importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📌 Feature Importance TOP 30:")
    print(importance.head(30).to_string(index=False))
    
    return importance, model


# ============================================================
# 9. SHAP 분석 (선택)
# ============================================================
def shap_analysis(df, target='TOTALCNT', sample_size=3000):
    """SHAP 분석"""
    print("\n" + "="*60)
    print("🎯 9. SHAP 분석")
    print("="*60)
    
    try:
        import shap
        import xgboost as xgb
    except ImportError:
        print("❌ shap 필요: pip install shap")
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target and c != 'CURRTIME']
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target]
    
    # 샘플링
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample, y_sample = X.iloc[idx], y.iloc[idx]
    else:
        X_sample, y_sample = X, y
    
    print(f"✅ SHAP 샘플: {len(X_sample):,}개")
    
    model = xgb.XGBRegressor(max_depth=6, n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_sample, y_sample, verbose=False)
    
    print("🔄 SHAP 계산 중...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    shap_imp = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    print(f"\n📌 SHAP 중요도 TOP 30:")
    print(shap_imp.head(30).to_string(index=False))
    
    return shap_imp


# ============================================================
# 10. Boruta (선택)
# ============================================================
def boruta_analysis(df, target='TOTALCNT', max_iter=30):
    """Boruta Feature Selection"""
    print("\n" + "="*60)
    print("🎯 10. Boruta Feature Selection")
    print("="*60)
    
    try:
        from boruta import BorutaPy
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        print("❌ boruta 필요: pip install boruta")
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target and c != 'CURRTIME']
    
    X = df[feature_cols].fillna(df[feature_cols].median()).values
    y = df[target].values
    
    print(f"✅ Feature: {len(feature_cols)}개")
    print(f"🔄 Boruta 실행 중... (시간 소요)")
    
    rf = RandomForestRegressor(n_jobs=-1, max_depth=7, random_state=42)
    boruta = BorutaPy(rf, n_estimators='auto', max_iter=max_iter, random_state=42, verbose=0)
    boruta.fit(X, y)
    
    results = pd.DataFrame({
        'feature': feature_cols,
        'ranking': boruta.ranking_,
        'confirmed': boruta.support_,
        'tentative': boruta.support_weak_
    }).sort_values('ranking')
    
    confirmed = results[results['confirmed']]['feature'].tolist()
    tentative = results[results['tentative']]['feature'].tolist()
    
    print(f"\n📌 Boruta 결과:")
    print(f"  Confirmed: {len(confirmed)}개")
    print(f"  Tentative: {len(tentative)}개")
    print(f"  Rejected: {len(results) - len(confirmed) - len(tentative)}개")
    
    print(f"\n📌 Confirmed Features:")
    for i, f in enumerate(confirmed[:20]):
        print(f"  {i+1}. {f}")
    
    return results


# ============================================================
# 11. 시각화
# ============================================================
def create_visualizations(df, target='TOTALCNT', output_prefix='fig'):
    """시각화 생성"""
    print("\n" + "="*60)
    print("📈 11. 시각화 생성")
    print("="*60)
    
    # 1. TOTALCNT 분포
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.hist(df[target], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(1600, color='orange', linestyle='--', label='1600')
    ax1.axvline(1700, color='red', linestyle='--', label='1700')
    ax1.set_xlabel(target)
    ax1.set_title(f'{target} Distribution')
    ax1.legend()
    
    # 2. 시계열
    ax2 = axes[0, 1]
    if 'CURRTIME' in df.columns and df['CURRTIME'].notna().any():
        df_plot = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME')
        sample_idx = np.linspace(0, len(df_plot)-1, min(3000, len(df_plot)), dtype=int)
        ax2.plot(df_plot['CURRTIME'].iloc[sample_idx], df_plot[target].iloc[sample_idx], lw=0.5)
        ax2.axhline(1700, color='red', linestyle='--', alpha=0.7)
    ax2.set_title(f'{target} Time Series')
    
    # 3. queue_gap vs TOTALCNT
    ax3 = axes[1, 0]
    if 'queue_gap' in df.columns:
        sample = df.sample(min(3000, len(df)))
        ax3.scatter(sample['queue_gap'], sample[target], alpha=0.3, s=10)
        ax3.axhline(1700, color='red', linestyle='--')
        ax3.axvline(300, color='red', linestyle='--')
        ax3.set_xlabel('queue_gap')
        ax3.set_ylabel(target)
    ax3.set_title('queue_gap vs TOTALCNT')
    
    # 4. 황금 패턴
    ax4 = axes[1, 1]
    if 'M14AM14B' in df.columns:
        sample = df.sample(min(3000, len(df)))
        colors = ['red' if t >= 1700 else 'blue' for t in sample[target]]
        ax4.scatter(sample['M14AM14B'], sample['M14AM14BSUM'], c=colors, alpha=0.3, s=10)
        ax4.axvline(520, color='orange', linestyle='--')
        ax4.axhline(600, color='orange', linestyle='--')
        ax4.set_xlabel('M14AM14B')
        ax4.set_ylabel('M14AM14BSUM')
    ax4.set_title('Golden Pattern (Red=1700+)')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=150)
    plt.close()
    print(f"✅ 저장: {output_prefix}_analysis.png")


# ============================================================
# 12. 결과 저장
# ============================================================
def save_results(df_types, importance, corr_high, output_prefix='results'):
    """결과 저장"""
    print("\n" + "="*60)
    print("💾 12. 결과 저장")
    print("="*60)
    
    if df_types is not None:
        df_types.to_csv(f'{output_prefix}_variable_types.csv', index=False, encoding='utf-8-sig')
        print(f"✅ {output_prefix}_variable_types.csv")
    
    if importance is not None:
        importance.to_csv(f'{output_prefix}_feature_importance.csv', index=False, encoding='utf-8-sig')
        print(f"✅ {output_prefix}_feature_importance.csv")
    
    if corr_high:
        pd.DataFrame(corr_high).to_csv(f'{output_prefix}_high_correlation.csv', index=False, encoding='utf-8-sig')
        print(f"✅ {output_prefix}_high_correlation.csv")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == '__main__':
    # ========================================
    # 📌 여기 파일 경로만 수정하세요!
    # ========================================
    DATA_FILE = "data/M14_Q_3MONTH.csv"  # 3개월 데이터 파일 경로
    
    # 1. 데이터 로드
    df = load_data(DATA_FILE)
    
    # 2. 변수 타입 분석
    df_types = analyze_variable_types(df)
    
    # 3. 타겟 분포
    analyze_target(df, 'TOTALCNT')
    
    # 4. 상관관계
    corr_matrix, high_corr = correlation_analysis(df, 'TOTALCNT', threshold=0.95)
    
    # 5. 정합성 확인
    check_data_integrity(df)
    
    # 6. queue_gap 분석
    analyze_queue_gap(df, 'TOTALCNT')
    
    # 7. 황금 패턴
    analyze_golden_pattern(df, 'TOTALCNT')
    
    # 8. Feature Importance
    importance, model = feature_importance_analysis(df, 'TOTALCNT')
    
    # 9. SHAP (시간 오래 걸림 - 필요시 주석 해제)
    # shap_imp = shap_analysis(df, 'TOTALCNT', sample_size=3000)
    
    # 10. Boruta (시간 매우 오래 걸림 - 필요시 주석 해제)
    # boruta_results = boruta_analysis(df, 'TOTALCNT', max_iter=30)
    
    # 11. 시각화
    create_visualizations(df, 'TOTALCNT', output_prefix='amhs')
    
    # 12. 결과 저장
    save_results(df_types, importance, high_corr, output_prefix='amhs')
    
    print("\n" + "="*80)
    print("✅ 분석 완료!")
    print("="*80)#