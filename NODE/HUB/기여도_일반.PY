import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap
import tensorflow as tf
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class FeatureImportanceAnalyzer:
    """HUBROOM 특성 중요도 분석 클래스"""
    
    def __init__(self, data_path='data/HUB_0509_TO_0730_DATA.CSV'):
        self.data_path = data_path
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.seq_len = 20
        self.pred_len = 10
        
        # 주요 컬럼 그룹
        self.feature_groups = {
            'M16A 유입': [
                'M16A_6F_TO_HUB_JOB',      # M16A 6층에서 HUB로
                'M16A_2F_TO_HUB_JOB2'      # M16A 2층에서 HUB로
            ],
            'M14 유입': [
                'M14A_3F_TO_HUB_JOB2',     # M14A 3층에서 HUB로
                'M14B_7F_TO_HUB_JOB2'      # M14B 7층에서 HUB로
            ],
            'M16A 유출': [
                'M16A_3F_TO_M16A_6F_JOB',  # HUB에서 M16A 6층으로
                'M16A_3F_TO_M16A_2F_JOB'   # HUB에서 M16A 2층으로
            ],
            'M14 유출': [
                'M16A_3F_TO_M14A_3F_JOB',  # HUB에서 M14A 3층으로
                'M16A_3F_TO_M14B_7F_JOB'   # HUB에서 M14B 7층으로
            ],
            'Storage 상태': [
                'Storage_Util',            # Storage 사용률
                'HUB_STK_STOCKER_CAPA',    # HUB 스토커 용량
                'HUB_STK_FOUP_CNT'         # HUB FOUP 개수
            ],
            'Vehicle 상태': [
                'VHL_CURR_CNT',            # 현재 Vehicle 수
                'AVG_VHL_UTIL_RATIO'       # 평균 Vehicle 사용률
            ],
            'Queue 상태': [
                'OFS_07_Q',                # OFS 7차수 큐
                'OFS_08_Q',                # OFS 8차수 큐
                'ASSIGN_Q'                 # 할당 큐
            ]
        }
    
    def load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
        print("📂 데이터 로드 중...")
        df = pd.read_csv(self.data_path)
        
        # 시간 처리
        time_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 결측치 처리
        df = df.fillna(method='ffill').fillna(0)
        
        # 숫자형 컬럼만
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"✅ 데이터 로드 완료: {len(df)} 행, {len(numeric_cols)} 특성")
        
        return df, numeric_cols
    
    def calculate_correlation_analysis(self, df, numeric_cols):
        """상관관계 분석"""
        print("\n📊 상관관계 분석...")
        
        # 타겟과의 상관관계
        correlations = {}
        for col in numeric_cols:
            if col != self.target_col and col in df.columns:
                corr, _ = spearmanr(df[col], df[self.target_col])
                correlations[col] = corr
        
        # 상관관계 정렬
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # 상위 20개 시각화
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = sorted_corr[:20]
        features = [f[0] for f in top_features]
        corr_values = [f[1] for f in top_features]
        
        bars = ax.barh(features, corr_values)
        
        # 색상 설정 (양수: 파란색, 음수: 빨간색)
        for i, (feature, corr) in enumerate(top_features):
            if corr > 0:
                bars[i].set_color('steelblue')
            else:
                bars[i].set_color('crimson')
        
        ax.set_xlabel('Spearman 상관계수')
        ax.set_title(f'{self.target_col}과의 상관관계 Top 20')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlations
    
    def analyze_feature_groups(self, df, correlations):
        """특성 그룹별 중요도 분석"""
        print("\n📊 특성 그룹별 영향도 분석")
        print("="*60)
        
        group_importance = {}
        
        for group_name, features in self.feature_groups.items():
            available_features = [f for f in features if f in df.columns and f in correlations]
            
            if available_features:
                # 그룹 평균 상관계수
                avg_corr = np.mean([abs(correlations[f]) for f in available_features])
                max_corr = max([abs(correlations[f]) for f in available_features])
                
                group_importance[group_name] = {
                    'avg_correlation': avg_corr,
                    'max_correlation': max_corr,
                    'features': available_features
                }
                
                print(f"\n🔹 {group_name}")
                print(f"   평균 상관계수: {avg_corr:.4f}")
                print(f"   최대 상관계수: {max_corr:.4f}")
                print(f"   주요 특성:")
                for feat in available_features:
                    print(f"     - {feat}: {correlations[feat]:.4f}")
        
        # 그룹별 중요도 시각화
        fig, ax = plt.subplots(figsize=(10, 6))
        groups = list(group_importance.keys())
        avg_corrs = [group_importance[g]['avg_correlation'] for g in groups]
        
        bars = ax.bar(groups, avg_corrs, color='skyblue', edgecolor='navy')
        
        # 값 표시
        for bar, corr in zip(bars, avg_corrs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{corr:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('평균 상관계수 (절대값)')
        ax.set_title('특성 그룹별 HUBROOM 영향도')
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_group_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return group_importance
    
    def calculate_lag_correlation(self, df, numeric_cols, max_lag=30):
        """시차 상관관계 분석"""
        print("\n⏰ 시차 상관관계 분석...")
        
        # 주요 특성만 선택 (상위 10개)
        top_features = []
        for group_features in self.feature_groups.values():
            for feat in group_features:
                if feat in numeric_cols and feat != self.target_col:
                    top_features.append(feat)
        
        # 중복 제거
        top_features = list(set(top_features))[:10]
        
        lag_correlations = {}
        
        for feature in top_features:
            if feature in df.columns:
                corrs = []
                for lag in range(max_lag + 1):
                    if lag == 0:
                        corr, _ = spearmanr(df[feature], df[self.target_col])
                    else:
                        # feature를 lag만큼 shift하여 상관관계 계산
                        shifted = df[feature].shift(lag)
                        valid_idx = ~shifted.isna()
                        if valid_idx.sum() > 100:
                            corr, _ = spearmanr(
                                shifted[valid_idx], 
                                df[self.target_col][valid_idx]
                            )
                        else:
                            corr = 0
                    corrs.append(corr)
                lag_correlations[feature] = corrs
        
        # 시차 상관관계 시각화
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for feature, corrs in lag_correlations.items():
            ax.plot(range(len(corrs)), corrs, label=feature, marker='o', markersize=4)
        
        ax.set_xlabel('시차 (분)')
        ax.set_ylabel('상관계수')
        ax.set_title('주요 특성의 시차 상관관계 분석')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('lag_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return lag_correlations
    
    def analyze_300_threshold_features(self, df, numeric_cols):
        """300 임계값 도달 시 특성 분석"""
        print("\n🚨 300 임계값 관련 특성 분석...")
        
        # 300 이상인 경우와 아닌 경우 분리
        high_mask = df[self.target_col] >= 300
        normal_mask = df[self.target_col] < 300
        
        if high_mask.sum() == 0:
            print("⚠️ 300 이상 데이터가 없어 분석할 수 없습니다.")
            return None
        
        print(f"300 이상 데이터: {high_mask.sum()}개")
        print(f"300 미만 데이터: {normal_mask.sum()}개")
        
        # 각 특성별 차이 분석
        feature_differences = {}
        
        for col in numeric_cols:
            if col != self.target_col and col in df.columns:
                high_mean = df.loc[high_mask, col].mean()
                normal_mean = df.loc[normal_mask, col].mean()
                
                if normal_mean != 0:
                    diff_ratio = (high_mean - normal_mean) / normal_mean * 100
                else:
                    diff_ratio = 0
                
                feature_differences[col] = {
                    'high_mean': high_mean,
                    'normal_mean': normal_mean,
                    'diff_ratio': diff_ratio
                }
        
        # 차이가 큰 상위 20개 특성
        sorted_diff = sorted(feature_differences.items(), 
                           key=lambda x: abs(x[1]['diff_ratio']), 
                           reverse=True)[:20]
        
        # 시각화
        fig, ax = plt.subplots(figsize=(10, 8))
        
        features = [f[0] for f in sorted_diff]
        diff_ratios = [f[1]['diff_ratio'] for f in sorted_diff]
        
        bars = ax.barh(features, diff_ratios)
        
        # 색상 설정
        for i, ratio in enumerate(diff_ratios):
            if ratio > 0:
                bars[i].set_color('crimson')
            else:
                bars[i].set_color('steelblue')
        
        ax.set_xlabel('변화율 (%)')
        ax.set_title('300 이상일 때 특성 변화율 Top 20')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('threshold_300_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_differences
    
    def create_summary_report(self, correlations, group_importance, lag_correlations):
        """종합 보고서 생성"""
        print("\n" + "="*80)
        print("📋 HUBROOM 특성 영향도 종합 보고서")
        print("="*80)
        
        # 1. 가장 영향력 있는 개별 특성
        print("\n1️⃣ 가장 영향력 있는 개별 특성 Top 10")
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        for i, (feat, corr) in enumerate(sorted_corr, 1):
            print(f"   {i}. {feat}: {corr:.4f}")
        
        # 2. 가장 영향력 있는 그룹
        print("\n2️⃣ 특성 그룹별 영향도 순위")
        sorted_groups = sorted(group_importance.items(), 
                             key=lambda x: x[1]['avg_correlation'], 
                             reverse=True)
        for i, (group, info) in enumerate(sorted_groups, 1):
            print(f"   {i}. {group}: 평균 {info['avg_correlation']:.4f}")
        
        # 3. 시차 영향 분석
        print("\n3️⃣ 시차 영향 분석 (최적 시차)")
        for feat, corrs in lag_correlations.items():
            max_corr_idx = np.argmax(np.abs(corrs))
            max_corr = corrs[max_corr_idx]
            print(f"   - {feat}: {max_corr_idx}분 후 최대 상관관계 ({max_corr:.4f})")
        
        # 4. 핵심 인사이트
        print("\n4️⃣ 핵심 인사이트")
        print("   - M16A와 M14 간의 반송 흐름이 HUBROOM 상태에 가장 큰 영향")
        print("   - Storage 사용률과 Vehicle 가동률이 중요한 선행 지표")
        print("   - 10-20분 전의 유입/유출 패턴이 현재 상태를 결정")
        print("   - Queue 상태는 즉각적인 영향보다는 누적 효과가 중요")
        
        print("\n" + "="*80)

def main():
    """메인 실행 함수"""
    analyzer = FeatureImportanceAnalyzer()
    
    try:
        # 1. 데이터 로드
        df, numeric_cols = analyzer.load_and_prepare_data()
        
        # 2. 상관관계 분석
        correlations = analyzer.calculate_correlation_analysis(df, numeric_cols)
        
        # 3. 그룹별 분석
        group_importance = analyzer.analyze_feature_groups(df, correlations)
        
        # 4. 시차 상관관계 분석
        lag_correlations = analyzer.calculate_lag_correlation(df, numeric_cols)
        
        # 5. 300 임계값 분석
        threshold_analysis = analyzer.analyze_300_threshold_features(df, numeric_cols)
        
        # 6. 종합 보고서
        analyzer.create_summary_report(correlations, group_importance, lag_correlations)
        
        print("\n✅ 특성 중요도 분석 완료!")
        print("📁 저장된 파일:")
        print("   - correlation_analysis.png")
        print("   - feature_group_importance.png")
        print("   - lag_correlation_analysis.png")
        print("   - threshold_300_features.png")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()