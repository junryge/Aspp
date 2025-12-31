#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Baseline 모델 및 Feature Selection 전략 수립
목적: 3개월 데이터 기반 최적 Feature Set 도출
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
import glob
import os
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("🚀 Baseline 모델 및 Feature Selection 전략 수립")
print("="*80)

# ========================================
# 1. 데이터 로딩
# ========================================
def load_all_data(path_pattern='*.CSV'):
    """3개월치 CSV 파일 로드"""
    print(f"\n📂 데이터 로딩: {path_pattern}")
    
    files = glob.glob(path_pattern)
    if not files:
        # 기본 경로 시도
        files = glob.glob('data/*.CSV') + glob.glob('M14_Q_*.CSV')
    
    if not files:
        print("⚠️ CSV 파일을 찾을 수 없습니다. 샘플 데이터로 진행합니다.")
        return None
    
    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, on_bad_lines='skip', encoding='utf-8')
            print(f"  ✅ {os.path.basename(f)}: {len(df):,}행, {df.shape[1]}컬럼")
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
    print(f"\n📊 총 데이터: {len(df_all):,}행, {df_all.shape[1]}컬럼")
    return df_all


# ========================================
# 2. 데이터 전처리
# ========================================
def preprocess_data(df):
    """데이터 전처리 및 기본 Feature 생성"""
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
    numeric_cols = [c for c in df.columns if c not in exclude_cols]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 결측치 처리 (forward fill → backward fill → 0)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # 핵심 파생 변수 생성
    if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
        df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    if 'M14B.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14B.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
        df['queue_gap_B'] = df['M14B.QUE.ALL.CURRENTQCREATED'] - df['M14B.QUE.ALL.CURRENTQCOMPLETED']
    
    print(f"  ✅ 전처리 완료: {len(df):,}행")
    return df


# ========================================
# 3. EDA (탐색적 데이터 분석)
# ========================================
def perform_eda(df, target_col='TOTALCNT'):
    """탐색적 데이터 분석"""
    print("\n📊 EDA (탐색적 데이터 분석)")
    print("-"*60)
    
    # 타겟 분포
    print(f"\n🎯 타겟 변수 ({target_col}) 분포:")
    print(f"  평균: {df[target_col].mean():.2f}")
    print(f"  중앙값: {df[target_col].median():.2f}")
    print(f"  표준편차: {df[target_col].std():.2f}")
    print(f"  최소: {df[target_col].min():.2f}")
    print(f"  최대: {df[target_col].max():.2f}")
    
    # 위험 구간 분포
    print(f"\n⚠️ 위험 구간 분포:")
    total = len(df)
    print(f"  < 900 (LOW): {(df[target_col] < 900).sum():,}개 ({(df[target_col] < 900).sum()/total*100:.2f}%)")
    print(f"  900-1599 (NORMAL): {((df[target_col] >= 900) & (df[target_col] < 1600)).sum():,}개")
    print(f"  1600-1699 (CAUTION): {((df[target_col] >= 1600) & (df[target_col] < 1700)).sum():,}개")
    print(f"  >= 1700 (CRITICAL): {(df[target_col] >= 1700).sum():,}개 ({(df[target_col] >= 1700).sum()/total*100:.2f}%)")
    
    # 결측치 분석
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️ 결측치가 있는 컬럼: {(missing > 0).sum()}개")
    else:
        print(f"\n✅ 결측치 없음")
    
    return {
        'target_mean': df[target_col].mean(),
        'target_std': df[target_col].std(),
        'critical_count': (df[target_col] >= 1700).sum(),
        'critical_ratio': (df[target_col] >= 1700).sum() / total * 100
    }


# ========================================
# 4. 상관관계 분석
# ========================================
def analyze_correlations(df, target_col='TOTALCNT', top_n=None):
    """타겟과의 상관관계 분석"""
    print(f"\n🔗 상관관계 분석 (타겟: {target_col})")
    print("-"*60)
    
    # 숫자형 컬럼만
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 타겟과의 상관계수
    correlations = numeric_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    
    print(f"\n📈 상관 Feature (전체 {len(correlations)}개):")
    display_n = top_n if top_n else len(correlations)
    for i, (col, corr) in enumerate(correlations.head(display_n).items(), 1):
        print(f"  {i:3d}. {col:<55} : {corr:.4f}")
    
    # 0.998 이상 (데이터 누수 의심)
    leakage_candidates = correlations[correlations > 0.95]
    if len(leakage_candidates) > 0:
        print(f"\n⚠️ 데이터 누수 의심 (상관계수 > 0.95):")
        for col, corr in leakage_candidates.items():
            print(f"  - {col}: {corr:.4f}")
    
    return correlations


# ========================================
# 5. Feature Selection 방법들
# ========================================
class FeatureSelector:
    """다양한 Feature Selection 방법 통합"""
    
    def __init__(self, df, target_col='TOTALCNT', exclude_cols=None):
        self.df = df
        self.target_col = target_col
        self.exclude_cols = exclude_cols or ['CURRTIME', target_col]
        
        # Feature/Target 분리
        self.feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                           if c not in self.exclude_cols]
        self.X = df[self.feature_cols].values
        self.y = df[target_col].values
        
        # 결과 저장
        self.results = {}
    
    def variance_threshold(self, threshold=0.01):
        """분산 기반 필터링"""
        print("\n📐 1. Variance Threshold")
        
        # 정규화 후 분산 계산
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_scaled)
        
        selected_mask = selector.get_support()
        selected_features = [self.feature_cols[i] for i, sel in enumerate(selected_mask) if sel]
        removed_features = [self.feature_cols[i] for i, sel in enumerate(selected_mask) if not sel]
        
        print(f"  선택: {len(selected_features)}개, 제거: {len(removed_features)}개")
        if removed_features:
            print(f"  제거된 Feature: {removed_features[:10]}...")
        
        self.results['variance'] = selected_features
        return selected_features
    
    def correlation_filter(self, corr_threshold=0.3, multicollinearity_threshold=0.95):
        """상관관계 기반 필터링"""
        print("\n📊 2. Correlation Filter")
        
        df_numeric = self.df[self.feature_cols + [self.target_col]]
        
        # 타겟과의 상관관계
        target_corr = df_numeric.corr()[self.target_col].drop(self.target_col).abs()
        
        # 타겟과 상관관계가 높은 Feature 선택
        selected = target_corr[target_corr >= corr_threshold].index.tolist()
        print(f"  타겟 상관 >= {corr_threshold}: {len(selected)}개")
        
        # 다중공선성 제거
        if len(selected) > 1:
            corr_matrix = df_numeric[selected].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            to_drop = set()
            for col in upper.columns:
                highly_correlated = upper.index[upper[col] > multicollinearity_threshold].tolist()
                for hc in highly_correlated:
                    # 타겟 상관이 낮은 쪽 제거
                    if target_corr[col] >= target_corr[hc]:
                        to_drop.add(hc)
                    else:
                        to_drop.add(col)
            
            selected = [f for f in selected if f not in to_drop]
            print(f"  다중공선성 제거 후: {len(selected)}개 (제거: {len(to_drop)}개)")
        
        self.results['correlation'] = selected
        return selected
    
    def mutual_info_selection(self, k=None):
        """Mutual Information 기반 선택"""
        print("\n🔮 3. Mutual Information")
        
        k = k if k else len(self.feature_cols)
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(self.X, self.y)
        
        scores = pd.Series(selector.scores_, index=self.feature_cols).sort_values(ascending=False)
        selected = scores.index.tolist()
        
        print(f"  전체 {len(selected)}개 분석 완료")
        print(f"  전체 Feature MI 점수:")
        for i, (col, score) in enumerate(scores.items(), 1):
            print(f"    {i:3d}. {col:<55} : {score:.4f}")
        
        self.results['mutual_info'] = selected
        return selected
    
    def tree_importance(self, n_estimators=100, top_n=None):
        """Tree 기반 Feature Importance"""
        print("\n🌲 4. Tree-based Feature Importance")
        
        # RandomForest
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=10, 
                                   n_jobs=-1, random_state=42)
        rf.fit(self.X, self.y)
        
        rf_importance = pd.Series(rf.feature_importances_, index=self.feature_cols).sort_values(ascending=False)
        
        # ExtraTrees
        et = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=10,
                                 n_jobs=-1, random_state=42)
        et.fit(self.X, self.y)
        
        et_importance = pd.Series(et.feature_importances_, index=self.feature_cols).sort_values(ascending=False)
        
        # 평균 중요도
        avg_importance = (rf_importance + et_importance) / 2
        avg_importance = avg_importance.sort_values(ascending=False)
        
        top_n = top_n if top_n else len(avg_importance)
        selected = avg_importance.head(top_n).index.tolist()
        
        print(f"  전체 {len(avg_importance)}개 분석 완료")
        print(f"  전체 Feature 중요도:")
        for i, (col, imp) in enumerate(avg_importance.items(), 1):
            print(f"    {i:3d}. {col:<55} : {imp:.4f}")
        
        self.results['tree_importance'] = selected
        self.results['tree_scores'] = avg_importance
        return selected
    
    def lasso_selection(self, cv=5):
        """LASSO 기반 Feature Selection"""
        print("\n📉 5. LASSO Selection")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        lasso = LassoCV(cv=cv, random_state=42, n_jobs=-1)
        lasso.fit(X_scaled, self.y)
        
        coef = pd.Series(np.abs(lasso.coef_), index=self.feature_cols).sort_values(ascending=False)
        selected = coef[coef > 0].index.tolist()
        
        print(f"  선택: {len(selected)}개 (비영 계수)")
        print(f"  최적 alpha: {lasso.alpha_:.4f}")
        print(f"  전체 Feature LASSO 계수:")
        for i, (col, c) in enumerate(coef.items(), 1):
            marker = "✓" if c > 0 else " "
            print(f"    {marker} {i:3d}. {col:<55} : {c:.4f}")
        
        self.results['lasso'] = selected
        return selected
    
    def rfe_selection(self, n_features=None, step=5):
        """Recursive Feature Elimination"""
        print("\n🔄 6. RFE (Recursive Feature Elimination)")
        
        n_features = n_features if n_features else len(self.feature_cols) // 2
        
        estimator = RandomForestRegressor(n_estimators=50, max_depth=8, 
                                         n_jobs=-1, random_state=42)
        
        rfe = RFE(estimator, n_features_to_select=n_features, step=step)
        rfe.fit(self.X, self.y)
        
        selected = [self.feature_cols[i] for i, sel in enumerate(rfe.support_) if sel]
        ranking = pd.Series(rfe.ranking_, index=self.feature_cols).sort_values()
        
        print(f"  선택: {len(selected)}개")
        print(f"  전체 Feature RFE 순위:")
        for i, (col, rank) in enumerate(ranking.items(), 1):
            marker = "✓" if rank == 1 else " "
            print(f"    {marker} {i:3d}. {col:<55} : rank {rank}")
        
        self.results['rfe'] = selected
        return selected
    
    def ensemble_selection(self, min_methods=3):
        """여러 방법의 교집합/합집합 분석"""
        print("\n🎯 7. Ensemble Selection (다중 방법 통합)")
        print("-"*60)
        
        if len(self.results) < 2:
            print("  ⚠️ 최소 2개 이상의 방법을 먼저 실행하세요")
            return None
        
        # 각 Feature가 몇 개의 방법에서 선택됐는지
        all_features = set()
        for method, features in self.results.items():
            if isinstance(features, list):
                all_features.update(features)
        
        feature_counts = {}
        for feat in all_features:
            count = sum(1 for method, features in self.results.items() 
                       if isinstance(features, list) and feat in features)
            feature_counts[feat] = count
        
        feature_counts = pd.Series(feature_counts).sort_values(ascending=False)
        
        # min_methods개 이상에서 선택된 Feature
        consensus = feature_counts[feature_counts >= min_methods].index.tolist()
        
        print(f"\n📌 {min_methods}개 이상 방법에서 선택된 Feature: {len(consensus)}개")
        print("\n  전체 Feature별 선택 횟수:")
        for i, (feat, cnt) in enumerate(feature_counts.items(), 1):
            methods = [m for m, f in self.results.items() if isinstance(f, list) and feat in f]
            marker = "✓" if cnt >= min_methods else " "
            print(f"    {marker} {i:3d}. {feat:<50} : {cnt}회 ({', '.join(methods)})")
        
        self.results['ensemble'] = consensus
        return consensus
    
    def get_final_features(self, strategy='ensemble', **kwargs):
        """최종 Feature Set 반환"""
        if strategy in self.results:
            return self.results[strategy]
        else:
            return self.ensemble_selection(**kwargs)


# ========================================
# 6. Baseline 모델 평가
# ========================================
def evaluate_baseline_models(df, feature_cols, target_col='TOTALCNT', pred_offset=10):
    """Baseline 모델 성능 평가"""
    print("\n🧪 Baseline 모델 평가")
    print("-"*60)
    
    # 시계열 분할 (마지막 20%를 테스트)
    train_size = int(len(df) * 0.8)
    
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    
    # 타겟 생성 (pred_offset분 후)
    df_train['target'] = df_train[target_col].shift(-pred_offset)
    df_test['target'] = df_test[target_col].shift(-pred_offset)
    
    df_train = df_train.dropna(subset=['target'])
    df_test = df_test.dropna(subset=['target'])
    
    X_train = df_train[feature_cols].values
    y_train = df_train['target'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['target'].values
    
    print(f"  Train: {len(X_train):,}개, Test: {len(X_test):,}개")
    
    results = {}
    
    # 1. Naive Baseline (현재값 = 예측값)
    naive_pred = df_test[target_col].values[:len(y_test)]
    results['Naive (현재값)'] = {
        'MAE': mean_absolute_error(y_test, naive_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, naive_pred)),
        'R2': r2_score(y_test, naive_pred)
    }
    
    # 2. Ridge Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_pred = ridge.predict(X_test_scaled)
    results['Ridge'] = {
        'MAE': mean_absolute_error(y_test, ridge_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, ridge_pred)),
        'R2': r2_score(y_test, ridge_pred)
    }
    
    # 3. RandomForest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['RandomForest'] = {
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R2': r2_score(y_test, rf_pred)
    }
    
    # 4. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results['XGBoost'] = {
        'MAE': mean_absolute_error(y_test, xgb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
        'R2': r2_score(y_test, xgb_pred)
    }
    
    # 결과 출력
    print("\n📊 모델별 성능:")
    print(f"  {'모델':<20} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    print("  " + "-"*52)
    for model, metrics in results.items():
        print(f"  {model:<20} {metrics['MAE']:>10.2f} {metrics['RMSE']:>10.2f} {metrics['R2']:>10.4f}")
    
    # 1700+ 감지율 분석
    print("\n🔥 1700+ 감지율 (XGBoost 기준):")
    actual_danger = y_test >= 1700
    pred_danger = xgb_pred >= 1700
    
    if actual_danger.sum() > 0:
        detection_rate = (actual_danger & pred_danger).sum() / actual_danger.sum() * 100
        print(f"  실제 1700+ : {actual_danger.sum()}개")
        print(f"  감지 : {(actual_danger & pred_danger).sum()}개 ({detection_rate:.1f}%)")
        
        # 오탐
        false_positive = (pred_danger & ~actual_danger).sum()
        print(f"  오탐(FP) : {false_positive}개")
    else:
        print("  테스트 데이터에 1700+ 케이스 없음")
    
    return results, xgb_model


# ========================================
# 7. 최종 추천 Feature Set 생성
# ========================================
def generate_recommended_features(selector, correlations, top_n=None):
    """최종 추천 Feature Set 생성"""
    print("\n" + "="*80)
    print("🏆 최종 추천 Feature Set")
    print("="*80)
    
    # 1. 핵심 컬럼 (도메인 지식 기반)
    core_features = [
        'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A',
        'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
        'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED',
        'queue_gap'  # 파생 변수
    ]
    
    # 2. 상관관계 기반 전체
    corr_all = correlations.index.tolist()
    
    # 3. Ensemble Selection
    ensemble = selector.results.get('ensemble', [])
    
    # 4. Tree Importance
    tree_all = selector.results.get('tree_importance', [])
    
    # 통합
    all_recommended = set(core_features)
    all_recommended.update(corr_all)
    all_recommended.update(ensemble)
    all_recommended.update(tree_all)
    
    # 존재하는 컬럼만
    available_cols = set(selector.feature_cols)
    final_features = [f for f in all_recommended if f in available_cols or f in ['queue_gap', 'queue_gap_B']]
    
    # 상관관계 순 정렬
    final_features = sorted(final_features, 
                           key=lambda x: correlations.get(x, 0) if x in correlations else 0,
                           reverse=True)
    
    if top_n:
        final_features = final_features[:top_n]
    
    print(f"\n📌 전체 추천 Feature ({len(final_features)}개):")
    print("-"*80)
    
    for i, feat in enumerate(final_features, 1):
        corr = correlations.get(feat, 0)
        source = []
        if feat in core_features:
            source.append('Core')
        if feat in corr_all:
            source.append('Corr')
        if feat in ensemble:
            source.append('Ensemble')
        if feat in tree_all:
            source.append('Tree')
        
        print(f"  {i:3d}. {feat:<55} (상관:{corr:.4f}, {'/'.join(source)})")
    
    return final_features


# ========================================
# 8. 결과 저장
# ========================================
def save_results(selector, correlations, recommended_features, output_dir='feature_selection_results'):
    """분석 결과 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 상관관계 저장
    correlations.to_csv(f'{output_dir}/correlations.csv', encoding='utf-8-sig')
    
    # 2. 추천 Feature 저장
    pd.DataFrame({'feature': recommended_features}).to_csv(
        f'{output_dir}/recommended_features.csv', index=False, encoding='utf-8-sig'
    )
    
    # 3. 모든 방법 결과 저장
    all_results = []
    for method, features in selector.results.items():
        if isinstance(features, list):
            for feat in features:
                all_results.append({'method': method, 'feature': feat})
    
    pd.DataFrame(all_results).to_csv(
        f'{output_dir}/all_method_results.csv', index=False, encoding='utf-8-sig'
    )
    
    print(f"\n💾 결과 저장 완료: {output_dir}/")


# ========================================
# Main 실행
# ========================================
if __name__ == '__main__':
    # 1. 데이터 로딩
    df = load_all_data('*.CSV')
    
    if df is None:
        print("\n⚠️ CSV 파일을 찾을 수 없습니다.")
        print("   사용법: 스크립트와 같은 폴더에 CSV 파일을 두거나,")
        print("          load_all_data('경로/*.CSV')로 경로를 지정하세요.")
        exit(1)
    
    # 2. 전처리
    df = preprocess_data(df)
    
    # 3. EDA
    eda_results = perform_eda(df)
    
    # 4. 상관관계 분석
    correlations = analyze_correlations(df)
    
    # 5. Feature Selection
    print("\n" + "="*80)
    print("🔍 Feature Selection 시작")
    print("="*80)
    
    selector = FeatureSelector(df, target_col='TOTALCNT')
    
    # 각 방법 실행 (전체 분석!)
    selector.variance_threshold()
    selector.correlation_filter()
    selector.mutual_info_selection()  # 전체
    selector.tree_importance()  # 전체
    selector.lasso_selection()
    
    # RFE는 시간이 오래 걸려서 선택적
    print("\n⏳ RFE 실행 중 (시간 소요)...")
    try:
        selector.rfe_selection()  # 전체
    except Exception as e:
        print(f"  RFE 스킵: {e}")
    
    # Ensemble
    selector.ensemble_selection(min_methods=3)
    
    # 6. 추천 Feature Set (전체!)
    recommended = generate_recommended_features(selector, correlations)
    
    # 7. Baseline 모델 평가
    print("\n" + "="*80)
    print("🧪 선택된 Feature로 Baseline 모델 평가")
    print("="*80)
    
    # queue_gap 제외 (실제 컬럼만)
    eval_features = [f for f in recommended if f in df.columns]
    
    if len(eval_features) >= 5:
        model_results, best_model = evaluate_baseline_models(df, eval_features)
    else:
        print("  ⚠️ 평가할 Feature가 부족합니다.")
    
    # 8. 결과 저장
    save_results(selector, correlations, recommended)
    
    # 9. 최종 요약
    print("\n" + "="*80)
    print("📋 최종 요약")
    print("="*80)
    print(f"""
    ✅ 분석 완료!
    
    📊 데이터:
       - 총 {len(df):,}행, {df.shape[1]}컬럼
       - 1700+ 케이스: {(df['TOTALCNT'] >= 1700).sum()}개 ({eda_results['critical_ratio']:.2f}%)
    
    🎯 추천 Feature: {len(recommended)}개
       - 핵심: M14AM14B, M14AM14BSUM, queue_gap, TRANSPORT
    
    📁 저장된 파일:
       - feature_selection_results/correlations.csv
       - feature_selection_results/recommended_features.csv
       - feature_selection_results/all_method_results.csv
    
    🚀 다음 단계:
       1. 추천 Feature로 V9.0 모델 학습
       2. 10분/15분/25분 예측 모델 구축
       3. 실시간 평가 시스템 적용
    """)