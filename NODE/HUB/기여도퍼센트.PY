import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class FeaturePercentageImportance:
    """컬럼별 영향도를 퍼센트로 분석하는 클래스"""
    
    def __init__(self, data_path='data/HUB_0509_TO_0730_DATA.CSV'):
        self.data_path = data_path
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.seq_len = 20
        self.pred_len = 10
        
    def prepare_data_for_importance(self):
        """중요도 분석을 위한 데이터 준비"""
        print("📂 데이터 준비 중...")
        
        # 데이터 로드
        df = pd.read_csv(self.data_path)
        
        # 시간 처리
        time_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 결측치 처리
        df = df.fillna(method='ffill').fillna(0)
        
        # 숫자형 컬럼만
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 타겟 제외
        feature_cols = [col for col in numeric_cols if col != self.target_col]
        
        # 데이터 준비
        X = df[feature_cols].values
        y = df[self.target_col].values
        
        # 학습/테스트 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ 데이터 준비 완료: {len(feature_cols)}개 특성")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def calculate_rf_importance(self, X_train, y_train, feature_cols):
        """Random Forest 기반 중요도 계산"""
        print("\n🌲 Random Forest 중요도 계산 중...")
        
        # 모델 학습
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Feature importance
        importances = rf_model.feature_importances_
        
        # 퍼센트로 변환
        importance_percent = (importances / importances.sum()) * 100
        
        # DataFrame 생성
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances,
            'percentage': importance_percent
        }).sort_values('percentage', ascending=False)
        
        return importance_df, rf_model
    
    def calculate_permutation_importance(self, rf_model, X_test, y_test, feature_cols):
        """Permutation 중요도 계산"""
        print("\n🔄 Permutation 중요도 계산 중...")
        
        # Permutation importance 계산
        perm_importance = permutation_importance(
            rf_model, X_test, y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        # 퍼센트로 변환
        perm_importances = perm_importance.importances_mean
        perm_importances[perm_importances < 0] = 0  # 음수는 0으로
        perm_percent = (perm_importances / perm_importances.sum()) * 100
        
        # DataFrame 생성
        perm_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': perm_importances,
            'percentage': perm_percent
        }).sort_values('percentage', ascending=False)
        
        return perm_df
    
    def calculate_variance_contribution(self, X_train, y_train, feature_cols):
        """각 특성의 분산 기여도 계산"""
        print("\n📊 분산 기여도 계산 중...")
        
        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # 각 특성과 타겟의 공분산
        covariances = []
        for i in range(X_scaled.shape[1]):
            cov = np.abs(np.cov(X_scaled[:, i], y_train)[0, 1])
            covariances.append(cov)
        
        # 퍼센트로 변환
        covariances = np.array(covariances)
        var_percent = (covariances / covariances.sum()) * 100
        
        # DataFrame 생성
        var_df = pd.DataFrame({
            'feature': feature_cols,
            'covariance': covariances,
            'percentage': var_percent
        }).sort_values('percentage', ascending=False)
        
        return var_df
    
    def visualize_top_features(self, importance_df, title, top_n=20):
        """상위 특성 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 상위 N개 선택
        top_df = importance_df.head(top_n)
        
        # 1. 막대 그래프
        bars = ax1.barh(top_df['feature'], top_df['percentage'])
        
        # 색상 그라데이션
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 값 표시
        for i, (feat, pct) in enumerate(zip(top_df['feature'], top_df['percentage'])):
            ax1.text(pct + 0.1, i, f'{pct:.1f}%', va='center')
        
        ax1.set_xlabel('영향도 (%)')
        ax1.set_title(f'{title} - Top {top_n}')
        ax1.grid(True, alpha=0.3)
        
        # 2. 파이 차트 (상위 10개 + 기타)
        top10 = importance_df.head(10)
        others_pct = importance_df.iloc[10:]['percentage'].sum()
        
        labels = list(top10['feature']) + ['기타']
        sizes = list(top10['percentage']) + [others_pct]
        
        # 가장 큰 조각 강조
        explode = [0.1 if i == 0 else 0 for i in range(len(labels))]
        
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
                explode=explode, colors=plt.cm.Set3(range(len(labels))))
        ax2.set_title('전체 영향도 분포')
        
        plt.tight_layout()
        return fig
    
    def create_comparison_report(self, rf_df, perm_df, var_df):
        """종합 비교 보고서 생성"""
        print("\n" + "="*80)
        print("📊 컬럼별 영향도 퍼센트 종합 보고서")
        print("="*80)
        
        # 상위 10개 특성 비교
        print("\n🏆 TOP 10 특성 비교 (단위: %)")
        print("-"*80)
        print(f"{'순위':<5} {'특성명':<30} {'RF 중요도':<15} {'Permutation':<15} {'분산 기여도':<15}")
        print("-"*80)
        
        # 각 방법의 상위 10개
        for i in range(10):
            rf_feat = rf_df.iloc[i]['feature'] if i < len(rf_df) else ''
            rf_pct = rf_df.iloc[i]['percentage'] if i < len(rf_df) else 0
            
            perm_feat = perm_df.iloc[i]['feature'] if i < len(perm_df) else ''
            perm_pct = perm_df.iloc[i]['percentage'] if i < len(perm_df) else 0
            
            var_feat = var_df.iloc[i]['feature'] if i < len(var_df) else ''
            var_pct = var_df.iloc[i]['percentage'] if i < len(var_df) else 0
            
            # RF 기준으로 출력
            print(f"{i+1:<5} {rf_feat:<30} {rf_pct:<15.1f} ", end='')
            
            # 같은 특성 찾기
            perm_match = perm_df[perm_df['feature'] == rf_feat]
            var_match = var_df[var_df['feature'] == rf_feat]
            
            perm_val = perm_match['percentage'].values[0] if len(perm_match) > 0 else 0
            var_val = var_match['percentage'].values[0] if len(var_match) > 0 else 0
            
            print(f"{perm_val:<15.1f} {var_val:<15.1f}")
        
        # 평균 중요도 계산
        print("\n📈 통합 중요도 (3가지 방법 평균)")
        print("-"*60)
        
        # 모든 특성에 대해 평균 계산
        all_features = set(rf_df['feature']) | set(perm_df['feature']) | set(var_df['feature'])
        
        combined_importance = []
        for feat in all_features:
            rf_val = rf_df[rf_df['feature'] == feat]['percentage'].values
            perm_val = perm_df[perm_df['feature'] == feat]['percentage'].values
            var_val = var_df[var_df['feature'] == feat]['percentage'].values
            
            rf_pct = rf_val[0] if len(rf_val) > 0 else 0
            perm_pct = perm_val[0] if len(perm_val) > 0 else 0
            var_pct = var_val[0] if len(var_val) > 0 else 0
            
            avg_pct = (rf_pct + perm_pct + var_pct) / 3
            
            combined_importance.append({
                'feature': feat,
                'avg_percentage': avg_pct,
                'rf': rf_pct,
                'perm': perm_pct,
                'var': var_pct
            })
        
        # 정렬
        combined_df = pd.DataFrame(combined_importance).sort_values('avg_percentage', ascending=False)
        
        # 상위 15개 출력
        for i, row in combined_df.head(15).iterrows():
            print(f"{row['feature']:<35} 평균: {row['avg_percentage']:>5.1f}% "
                  f"(RF: {row['rf']:>5.1f}%, Perm: {row['perm']:>5.1f}%, Var: {row['var']:>5.1f}%)")
        
        # 누적 영향도
        print("\n📊 누적 영향도 분석")
        print("-"*40)
        cumsum = combined_df['avg_percentage'].cumsum()
        milestones = [50, 70, 80, 90]
        for milestone in milestones:
            n_features = (cumsum <= milestone).sum()
            print(f"상위 {n_features}개 특성이 전체의 {milestone}% 설명")
        
        return combined_df
    
    def visualize_combined_importance(self, combined_df):
        """통합 중요도 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 통합 중요도 막대 그래프
        ax1 = axes[0, 0]
        top15 = combined_df.head(15)
        bars = ax1.barh(top15['feature'], top15['avg_percentage'])
        
        # 그라데이션 색상
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(bars)))
        for bar, color in zip(bars, colors[::-1]):
            bar.set_color(color)
        
        # 값 표시
        for i, (idx, row) in enumerate(top15.iterrows()):
            ax1.text(row['avg_percentage'] + 0.1, i, f"{row['avg_percentage']:.1f}%", va='center')
        
        ax1.set_xlabel('평균 영향도 (%)')
        ax1.set_title('통합 중요도 Top 15')
        ax1.grid(True, alpha=0.3)
        
        # 2. 방법별 비교 (상위 10개)
        ax2 = axes[0, 1]
        top10 = combined_df.head(10)
        x = np.arange(len(top10))
        width = 0.25
        
        ax2.bar(x - width, top10['rf'], width, label='Random Forest', color='skyblue')
        ax2.bar(x, top10['perm'], width, label='Permutation', color='lightcoral')
        ax2.bar(x + width, top10['var'], width, label='Variance', color='lightgreen')
        
        ax2.set_xlabel('특성')
        ax2.set_ylabel('영향도 (%)')
        ax2.set_title('방법별 중요도 비교')
        ax2.set_xticks(x)
        ax2.set_xticklabels(top10['feature'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 누적 중요도 곡선
        ax3 = axes[1, 0]
        cumsum = combined_df['avg_percentage'].cumsum()
        ax3.plot(range(1, len(cumsum)+1), cumsum, 'b-', linewidth=2)
        ax3.fill_between(range(1, len(cumsum)+1), cumsum, alpha=0.3)
        
        # 주요 지점 표시
        milestones = [50, 70, 80, 90]
        for milestone in milestones:
            idx = (cumsum <= milestone).sum()
            ax3.plot(idx, milestone, 'ro', markersize=8)
            ax3.annotate(f'{idx}개 특성\n{milestone}%', 
                        xy=(idx, milestone), 
                        xytext=(idx+5, milestone-5),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax3.set_xlabel('특성 수')
        ax3.set_ylabel('누적 영향도 (%)')
        ax3.set_title('누적 영향도 곡선')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, min(50, len(cumsum)))
        
        # 4. 영향도 분포
        ax4 = axes[1, 1]
        ax4.hist(combined_df['avg_percentage'], bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.axvline(combined_df['avg_percentage'].mean(), color='red', linestyle='--', 
                   label=f'평균: {combined_df["avg_percentage"].mean():.2f}%')
        ax4.axvline(combined_df['avg_percentage'].median(), color='green', linestyle='--', 
                   label=f'중앙값: {combined_df["avg_percentage"].median():.2f}%')
        
        ax4.set_xlabel('영향도 (%)')
        ax4.set_ylabel('특성 수')
        ax4.set_title('영향도 분포')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def main():
    """메인 실행 함수"""
    analyzer = FeaturePercentageImportance()
    
    try:
        # 1. 데이터 준비
        X_train, X_test, y_train, y_test, feature_cols = analyzer.prepare_data_for_importance()
        
        # 2. Random Forest 중요도
        rf_importance_df, rf_model = analyzer.calculate_rf_importance(X_train, y_train, feature_cols)
        rf_fig = analyzer.visualize_top_features(rf_importance_df, "Random Forest 중요도")
        plt.savefig('rf_importance_percentage.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Permutation 중요도
        perm_importance_df = analyzer.calculate_permutation_importance(
            rf_model, X_test, y_test, feature_cols
        )
        perm_fig = analyzer.visualize_top_features(perm_importance_df, "Permutation 중요도")
        plt.savefig('permutation_importance_percentage.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. 분산 기여도
        var_importance_df = analyzer.calculate_variance_contribution(X_train, y_train, feature_cols)
        var_fig = analyzer.visualize_top_features(var_importance_df, "분산 기여도")
        plt.savefig('variance_importance_percentage.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. 종합 비교 보고서
        combined_df = analyzer.create_comparison_report(
            rf_importance_df, perm_importance_df, var_importance_df
        )
        
        # 6. 통합 시각화
        combined_fig = analyzer.visualize_combined_importance(combined_df)
        plt.savefig('combined_importance_percentage.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 7. CSV 저장
        combined_df.to_csv('feature_importance_percentages.csv', index=False)
        print("\n✅ 분석 완료! 결과가 CSV 파일로 저장되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()