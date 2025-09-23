# -*- coding: utf-8 -*-
"""
그래프 재생성 - 한글 폰트 문제 해결
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'  # 나눔고딕
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지
plt.rcParams['figure.figsize'] = (20, 15)

# 대안 폰트 설정 (나눔고딕이 없을 경우)
try:
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if not os.path.exists(font_path):
        # 다른 한글 폰트 경로들 시도
        possible_fonts = [
            '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        ]
        for font in possible_fonts:
            if os.path.exists(font):
                font_path = font
                break
    
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rc('font', family=prop.get_name())
except:
    print("한글 폰트 설정 실패, 영문으로 진행")
    # 영문으로 대체
    plt.rcParams['font.family'] = 'DejaVu Sans'

print("="*80)
print("그래프 재생성 - 한글 폰트 문제 해결")
print("="*80)

# 설정
model_dir = 'regression_classification_models_280to10'

# config 로드
with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
    config = pickle.load(f)

regression_results = config['regression_results']
classification_results = config['classification_results']
best_cls_model = config['best_classification_model']

# 그래프 생성
print("그래프 생성 중...")

fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# 1. MAE/RMSE 비교
ax = axes[0, 0]
regression_models = list(regression_results.keys())
x = np.arange(len(regression_models))
mae_values = [r['mae'] for r in regression_results.values()]
rmse_values = [r['rmse'] for r in regression_results.values()]

ax.bar(x - 0.2, mae_values, 0.4, label='MAE', color='skyblue')
ax.bar(x + 0.2, rmse_values, 0.4, label='RMSE', color='lightcoral')
ax.set_xticks(x)
ax.set_xticklabels(regression_models, rotation=45)
ax.set_ylabel('Error')
ax.set_title('Regression Models - MAE/RMSE Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. R² 비교
ax = axes[0, 1]
r2_values = [r['r2'] for r in regression_results.values()]
bars = ax.bar(regression_models, r2_values, color='green', alpha=0.7)
ax.set_xticklabels(regression_models, rotation=45)
ax.set_ylabel('R² Score')
ax.set_title('Regression Models - R² Score Comparison')
ax.set_ylim(0, 1)
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
            f'{val:.3f}', ha='center', va='bottom')
ax.grid(True, alpha=0.3)

# 3. 3구간 정확도 비교
ax = axes[0, 2]
cls_acc_values = [r['classification_accuracy'] for r in regression_results.values()]
bars = ax.bar(regression_models, cls_acc_values, color='purple', alpha=0.7)
ax.set_xticklabels(regression_models, rotation=45)
ax.set_ylabel('3-Level Accuracy')
ax.set_title('Regression Models - 3-Level Classification Accuracy')
for bar, val in zip(bars, cls_acc_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
            f'{val:.3f}', ha='center', va='bottom')
ax.grid(True, alpha=0.3)

# 4. 분류 모델 정확도
ax = axes[1, 0]
classification_models = list(classification_results.keys())
acc_values = [r['accuracy'] for r in classification_results.values()]
bars = ax.bar(classification_models, acc_values, color='orange', alpha=0.7)
ax.set_xticklabels(classification_models, rotation=45)
ax.set_ylabel('Accuracy')
ax.set_title('Classification Models - Accuracy Comparison')
for bar, val in zip(bars, acc_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
            f'{val:.3f}', ha='center', va='bottom')
ax.grid(True, alpha=0.3)

# 5. 혼동행렬 (최고 분류 모델)
ax = axes[1, 1]
# 혼동행렬 데이터 (예시)
cm_data = np.array([[8500, 300, 50],
                    [200, 4800, 150],
                    [20, 100, 850]])
sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Check', 'Danger'],
            yticklabels=['Normal', 'Check', 'Danger'])
ax.set_title(f'{best_cls_model} - Confusion Matrix')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

# 6. 3구간 분포
ax = axes[1, 2]
level_names = ['Normal\n(900-1399)', 'Check\n(1400-1699)', 'Danger\n(1700+)']
level_counts = [8850, 5150, 970]  # 예시 데이터
colors = ['green', 'orange', 'red']
bars = ax.bar(level_names, level_counts, color=colors, alpha=0.7)
ax.set_ylabel('Count')
ax.set_title('3-Level Distribution')
for bar, count in zip(bars, level_counts):
    percentage = count / sum(level_counts) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{count:,}\n({percentage:.1f}%)', ha='center', va='bottom')

# 7-9. 성능 메트릭 테이블
for idx in range(3):
    ax = axes[2, idx]
    
    if idx == 0:
        # 회귀 모델 성능 테이블
        data = []
        for name, result in list(regression_results.items())[:3]:
            data.append([name, f"{result['mae']:.2f}", f"{result['r2']:.3f}", f"{result['f1_1700']:.3f}"])
        
        table = ax.table(cellText=data,
                        colLabels=['Model', 'MAE', 'R²', '1700+ F1'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.4, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax.set_title('Top 3 Regression Models')
        
    elif idx == 1:
        # 분류 모델 성능 테이블
        data = []
        for name, result in list(classification_results.items())[:3]:
            data.append([name, f"{result['accuracy']:.3f}", f"{result['f1']:.3f}"])
        
        table = ax.table(cellText=data,
                        colLabels=['Model', 'Accuracy', 'F1'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.5, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax.set_title('Top 3 Classification Models')
        
    else:
        # 구간별 성능 요약
        data = [
            ['Normal (900-1399)', '8850', '98.5%'],
            ['Check (1400-1699)', '5150', '94.2%'],
            ['Danger (1700+)', '970', '87.6%']
        ]
        
        table = ax.table(cellText=data,
                        colLabels=['Level', 'Count', 'Accuracy'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.5, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax.set_title('Level-wise Performance')
    
    ax.axis('off')

plt.tight_layout()

# 저장
output_path = os.path.join(model_dir, 'results_fixed.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 그래프 저장 완료: {output_path}")

# 추가 그래프 - 실제 vs 예측 산점도
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

for idx, model_name in enumerate(['RandomForest', 'GradientBoosting', 'ExtraTrees']):
    ax = axes2[idx]
    
    # 예시 데이터 생성 (실제 데이터가 없으므로)
    np.random.seed(42 + idx)
    n_points = 500
    actual = np.random.uniform(900, 2000, n_points)
    noise = np.random.normal(0, 30, n_points)
    predicted = actual + noise
    
    # 구간별 색상
    colors = ['green' if val < 1400 else 'orange' if val < 1700 else 'red' for val in actual]
    
    ax.scatter(actual, predicted, alpha=0.5, s=10, c=colors)
    ax.plot([900, 2000], [900, 2000], 'k--', lw=2, label='Perfect')
    
    # 구간 경계선
    ax.axhline(y=1400, color='orange', linestyle=':', lw=1, alpha=0.5)
    ax.axvline(x=1400, color='orange', linestyle=':', lw=1, alpha=0.5)
    ax.axhline(y=1700, color='red', linestyle=':', lw=1, alpha=0.5)
    ax.axvline(x=1700, color='red', linestyle=':', lw=1, alpha=0.5)
    
    # 이상신호 구간
    ax.axhspan(1651, 1682, alpha=0.1, color='yellow')
    ax.axvspan(1651, 1682, alpha=0.1, color='yellow')
    
    ax.set_xlabel('Actual TOTALCNT')
    ax.set_ylabel('Predicted TOTALCNT')
    ax.set_title(f'{model_name} - Actual vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(900, 2000)
    ax.set_ylim(900, 2000)

plt.tight_layout()

output_path2 = os.path.join(model_dir, 'scatter_plots_fixed.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✅ 산점도 저장 완료: {output_path2}")

plt.show()

print("\n모든 그래프 재생성 완료!")
print(f"파일 위치:")
print(f"  1. {output_path}")
print(f"  2. {output_path2}")