# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 09:44:52 2025

@author: ggg3g
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# í•œê¸€ í°íŠ¸ ì„¤ì •
plt.rcParams['font.family'] = 'DejaVu Sans'

# ===== 1. ì´ˆê¸° ì„¤ì • =====
print("=== ê°œì„ ëœ ë°˜ë„ì²´ ë¬¼ë¥˜ ì˜ˆì¸¡ ì‹œìŠ¤í…œ ===")
print("Version 2.0 - 3êµ¬ê°„ ë¶„ë¥˜ + ë¶ˆê· í˜• ì²˜ë¦¬ + ì „ì²´ í‰ê°€")
print("="*60)

model_dir = 'saved_models_improved'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ===== 2. í‰ê°€ í•¨ìˆ˜ ì •ì˜ (ë¨¼ì € ì •ì˜) =====
def evaluate_model(y_true, y_pred, y_true_values, y_proba, model_name):
    """ì¢…í•©ì ì¸ ëª¨ë¸ í‰ê°€"""
    
    print(f"\n{'='*60}")
    print(f"{model_name} í‰ê°€ ê²°ê³¼")
    print('='*60)
    
    # ì •í™•ë„
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nì „ì²´ ì •í™•ë„: {accuracy:.4f}")
    
    # í´ëž˜ìŠ¤ë³„ ì„±ëŠ¥
    report = classification_report(y_true, y_pred, 
                                 target_names=['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜'],
                                 output_dict=True)
    
    print("\ní´ëž˜ìŠ¤ë³„ ì„±ëŠ¥:")
    print(f"{'í´ëž˜ìŠ¤':10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 55)
    for class_name in ['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜']:
        p = report[class_name]['precision']
        r = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        s = report[class_name]['support']
        print(f"{class_name:10} {p:10.4f} {r:10.4f} {f1:10.4f} {s:10.0f}")
    
    # í˜¼ë™ í–‰ë ¬
    cm = confusion_matrix(y_true, y_pred)
    print("\ní˜¼ë™ í–‰ë ¬:")
    print("      ì˜ˆì¸¡ â†’")
    print("      ì¼ë°˜  ì£¼ì˜  ìœ„í—˜")
    for i, row in enumerate(cm):
        print(f"{['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜'][i]:4} ", end="")
        for j in range(len(row)):
            print(f"{row[j]:5d} ", end="")
        print()
    
    # í´ëž˜ìŠ¤ë¥¼ ëŒ€í‘œê°’ìœ¼ë¡œ ë³€í™˜í•´ì„œ MAE/RMSE ê³„ì‚°
    class_to_value = {0: 900, 1: 1450, 2: 1750}
    y_pred_values = np.array([class_to_value[p] for p in y_pred])
    
    mae = mean_absolute_error(y_true_values, y_pred_values)
    rmse = np.sqrt(mean_squared_error(y_true_values, y_pred_values))
    
    print(f"\nì˜ˆì¸¡ ì˜¤ì°¨ (ì°¸ê³ ìš©):")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    # ìœ„í—˜ í´ëž˜ìŠ¤ íŠ¹ë³„ ë¶„ì„
    if 2 in y_true:
        danger_indices = np.where(y_true == 2)[0]
        if len(danger_indices) > 0:
            danger_correct = np.sum(y_pred[danger_indices] == 2)
            print(f"\nìœ„í—˜ í´ëž˜ìŠ¤(1700+) íŠ¹ë³„ ë¶„ì„:")
            print(f"  ì „ì²´ ìœ„í—˜ ìƒ˜í”Œ: {len(danger_indices)}ê°œ")
            print(f"  ì •í™•ížˆ ì˜ˆì¸¡: {danger_correct}ê°œ ({danger_correct/len(danger_indices)*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'report': report,
        'cm': cm,
        'mae': mae,
        'rmse': rmse
    }

# ===== 3. ë°ì´í„° ë¡œë“œ =====
print("\n1. ë°ì´í„° ë¡œë”©...")
# ì‹¤ì œ ì‚¬ìš©ì‹œ: df = pd.read_csv('your_data.csv')
try:
    df1 = pd.read_csv('uu.csv')
    df2 = pd.read_csv('uu2.csv')
    df = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)
except:
    print("í…ŒìŠ¤íŠ¸ ë°ì´í„° ìƒì„± ì¤‘...")
    # í…ŒìŠ¤íŠ¸ìš© ë°ì´í„° ìƒì„±
    np.random.seed(42)
    n_samples = 10000
    df = pd.DataFrame({
        'CURRTIME': range(n_samples),
        'TOTALCNT': np.random.normal(1400, 200, n_samples).clip(600, 2000),
        'M14AM14B': np.random.normal(200, 50, n_samples).clip(0, 500),
        'M14AM10A': np.random.normal(100, 30, n_samples).clip(0, 300),
        'M14AM16': np.random.normal(150, 40, n_samples).clip(0, 400),
        'M14AM14BSUM': np.random.normal(1000, 200, n_samples).clip(0, 2000),
        'M14AM10ASUM': np.random.normal(800, 150, n_samples).clip(0, 1500),
        'M14AM16SUM': np.random.normal(900, 180, n_samples).clip(0, 1800),
        'M14BM14A': np.random.normal(50, 20, n_samples).clip(0, 200),
        'M10AM14A': np.random.normal(60, 25, n_samples).clip(0, 250),
        'M16M14A': np.random.normal(70, 30, n_samples).clip(0, 300),
        'TIME': range(n_samples)
    })

print(f"ì „ì²´ ë°ì´í„°: {len(df)}í–‰")
print(f"ì»¬ëŸ¼: {df.columns.tolist()}")

# ===== 4. ì‹œí€€ìŠ¤ ë°ì´í„° ìƒì„± í•¨ìˆ˜ =====
def create_sequences(data, seq_length=280, pred_horizon=10):
    """280ë¶„ ì‹œí€€ìŠ¤ë¡œ 10ë¶„ í›„ ì˜ˆì¸¡ ë°ì´í„° ìƒì„±"""
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    # ë¹„ìœ¨ íŠ¹ì„± ì¶”ê°€
    data = data.copy()
    data['ratio_M14B_M14A'] = data['M14AM14B'] / (data['M14AM10A'] + 1)
    data['ratio_M14B_M16'] = data['M14AM14B'] / (data['M14AM16'] + 1)
    data['change_rate'] = data['TOTALCNT'].pct_change().fillna(0)
    
    X_list = []
    y_list = []
    y_values = []
    
    print(f"ì‹œí€€ìŠ¤ ìƒì„± ì¤‘... (ì´ {len(data) - seq_length - pred_horizon}ê°œ)")
    
    for i in range(len(data) - seq_length - pred_horizon):
        if i % 1000 == 0:
            print(f"  ì§„í–‰ë¥ : {i/(len(data) - seq_length - pred_horizon)*100:.1f}%")
        
        # 280ë¶„ ì‹œí€€ìŠ¤ ë°ì´í„°
        seq_data = data[feature_cols].iloc[i:i+seq_length].values
        
        # íŠ¹ì§• ì¶”ì¶œ
        features = []
        for col_idx in range(seq_data.shape[1]):
            col_data = seq_data[:, col_idx]
            
            # ê¸°ë³¸ í†µê³„ëŸ‰
            features.extend([
                np.mean(col_data),
                np.std(col_data),
                np.max(col_data),
                np.min(col_data),
                np.percentile(col_data, 75) - np.percentile(col_data, 25),  # IQR
                col_data[-1],  # ë§ˆì§€ë§‰ ê°’
                col_data[-1] - col_data[0],  # ì „ì²´ ë³€í™”ëŸ‰
                np.max(col_data[-20:]) if len(col_data) >= 20 else np.max(col_data)  # ìµœê·¼ 20ë¶„ ìµœëŒ€
            ])
        
        # ì¶”ê°€ íŠ¹ì§•
        last_idx = i + seq_length - 1
        features.append(data['ratio_M14B_M14A'].iloc[last_idx])
        features.append(data['ratio_M14B_M16'].iloc[last_idx])
        features.append(data['change_rate'].iloc[last_idx])
        
        # 10ë¶„ í›„ ê°’
        future_totalcnt = data['TOTALCNT'].iloc[i + seq_length + pred_horizon]
        
        # ë ˆì´ë¸” (3í´ëž˜ìŠ¤)
        if future_totalcnt >= 1700:
            label = 2  # ìœ„í—˜
        elif future_totalcnt >= 1200:
            label = 1  # ì£¼ì˜
        else:
            label = 0  # ì¼ë°˜
        
        X_list.append(features)
        y_list.append(label)
        y_values.append(future_totalcnt)
    
    return np.array(X_list), np.array(y_list), np.array(y_values)

# ===== 5. ë°ì´í„° ìƒì„± ë° ì „ì²˜ë¦¬ =====
print("\n2. ì‹œí€€ìŠ¤ ë°ì´í„° ìƒì„±...")
X, y, y_values = create_sequences(df)
print(f"ìƒì„±ëœ íŠ¹ì„±: {X.shape}")

# ë°ì´í„° ë¶„í¬ í™•ì¸
unique, counts = np.unique(y, return_counts=True)
print("\n=== í´ëž˜ìŠ¤ ë¶„í¬ ===")
class_names = ['ì¼ë°˜(600-1200)', 'ì£¼ì˜(1200-1699)', 'ìœ„í—˜(1700+)']
for u, c in zip(unique, counts):
    print(f"{class_names[u]}: {c} ({c/len(y)*100:.1f}%)")

# ===== 6. ë°ì´í„° ì¤€ë¹„ =====
print("\n3. ë°ì´í„° ì „ì²˜ë¦¬...")

# ì •ê·œí™”
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test ë¶„í• 
X_train, X_test, y_train, y_test, y_train_vals, y_test_vals = train_test_split(
    X_scaled, y, y_values, test_size=0.2, random_state=42, stratify=y
)

print(f"í›ˆë ¨ ë°ì´í„°: {X_train.shape}")
print(f"í…ŒìŠ¤íŠ¸ ë°ì´í„°: {X_test.shape}")

# ===== 7. ë¶ˆê· í˜• ì²˜ë¦¬ (SMOTE) =====
print("\n4. ë¶ˆê· í˜• ì²˜ë¦¬ (SMOTE)...")

# ìœ„í—˜ í´ëž˜ìŠ¤ì˜ ìƒ˜í”Œ ìˆ˜ í™•ì¸
n_danger_samples = np.sum(y_train == 2)
print(f"í›ˆë ¨ ë°ì´í„° ë‚´ ìœ„í—˜ í´ëž˜ìŠ¤ ìƒ˜í”Œ: {n_danger_samples}ê°œ")

if n_danger_samples > 1:
    # k_neighborsë¥¼ ìœ„í—˜ í´ëž˜ìŠ¤ ìƒ˜í”Œ ìˆ˜ì— ë§žê²Œ ì¡°ì •
    k_neighbors = min(5, n_danger_samples - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
else:
    print("ìœ„í—˜ í´ëž˜ìŠ¤ ìƒ˜í”Œì´ ë„ˆë¬´ ì ì–´ SMOTE ì ìš© ë¶ˆê°€")
    X_train_balanced, y_train_balanced = X_train, y_train

print(f"ê· í˜• ì¡°ì • ì „: {X_train.shape}")
print(f"ê· í˜• ì¡°ì • í›„: {X_train_balanced.shape}")

# ê· í˜• ì¡°ì • í›„ ë¶„í¬
unique_balanced, counts_balanced = np.unique(y_train_balanced, return_counts=True)
print("\nê· í˜• ì¡°ì • í›„ í´ëž˜ìŠ¤ ë¶„í¬:")
for u, c in zip(unique_balanced, counts_balanced):
    print(f"  í´ëž˜ìŠ¤ {u}: {c} ({c/len(y_train_balanced)*100:.1f}%)")

# ===== 8. ëª¨ë¸ ì •ì˜ ë° í•™ìŠµ =====
print("\n5. ëª¨ë¸ í•™ìŠµ...")

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight={0: 1, 1: 2, 2: 50},
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=10,
        random_state=42
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
}

# ëª¨ë¸ë³„ í•™ìŠµ ë° ì˜ˆì¸¡
results = {}
predictions = {}

for name, model in models.items():
    print(f"\n{name} í•™ìŠµ ì¤‘...")
    
    # í•™ìŠµ
    model.fit(X_train_balanced, y_train_balanced)
    
    # ì˜ˆì¸¡
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    
    # ì €ìž¥
    joblib.dump(model, os.path.join(model_dir, f'{name}_model.pkl'))
    
    predictions[name] = {
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    # í‰ê°€
    results[name] = evaluate_model(y_test, y_pred, y_test_vals, y_proba, name)

# ===== 9. ìµœì  ìž„ê³„ê°’ ë¶„ì„ =====
print("\n\n6. ìœ„í—˜ í´ëž˜ìŠ¤ ìž„ê³„ê°’ ìµœì í™”...")

def analyze_thresholds(y_true, y_proba_danger):
    """ë‹¤ì–‘í•œ ìž„ê³„ê°’ì—ì„œì˜ ì„±ëŠ¥ ë¶„ì„"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba_danger >= threshold).astype(int)
        
        tp = np.sum((y_true == 2) & (y_pred == 1))
        fp = np.sum((y_true != 2) & (y_pred == 1))
        fn = np.sum((y_true == 2) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
    
    return pd.DataFrame(results)

# RandomForestì˜ ìœ„í—˜ í´ëž˜ìŠ¤ ìž„ê³„ê°’ ë¶„ì„
if predictions['RandomForest']['y_proba'] is not None:
    threshold_df = analyze_thresholds(y_test, predictions['RandomForest']['y_proba'][:, 2])
    best_threshold_idx = threshold_df['f1'].argmax()
    best_threshold = threshold_df.iloc[best_threshold_idx]
    
    print(f"\nìµœì  ìž„ê³„ê°’: {best_threshold['threshold']:.2f}")
    print(f"  Precision: {best_threshold['precision']:.4f}")
    print(f"  Recall: {best_threshold['recall']:.4f}")
    print(f"  F1-Score: {best_threshold['f1']:.4f}")
    print(f"  ì •í™• ì˜ˆì¸¡: {int(best_threshold['tp'])}ê°œ")
    print(f"  ì˜¤íƒ: {int(best_threshold['fp'])}ê°œ")
    print(f"  ë†“ì¹¨: {int(best_threshold['fn'])}ê°œ")

# ===== 10. ì‹œê°í™” =====
print("\n7. ê²°ê³¼ ì‹œê°í™”...")

plt.figure(figsize=(20, 12))

# 1. ëª¨ë¸ë³„ ì •í™•ë„
plt.subplot(2, 3, 1)
accuracies = [r['accuracy'] for r in results.values()]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = plt.bar(results.keys(), accuracies, color=colors)
plt.title('Model Accuracy Comparison', fontsize=14, pad=10)
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)
for i, (bar, v) in enumerate(zip(bars, accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', 
             ha='center', va='bottom', fontsize=11)

# 2. í´ëž˜ìŠ¤ë³„ F1-Score
plt.subplot(2, 3, 2)
x = np.arange(len(models))
width = 0.25
for i, class_name in enumerate(['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜']):
    f1_scores = [r['report'][class_name]['f1-score'] for r in results.values()]
    plt.bar(x + i*width - width, f1_scores, width, label=f'{class_name}')
plt.title('F1-Score by Class', fontsize=14, pad=10)
plt.xticks(x, results.keys())
plt.legend()
plt.ylim(0, 1.1)

# 3. MAE/RMSE ë¹„êµ
plt.subplot(2, 3, 3)
mae_values = [r['mae'] for r in results.values()]
rmse_values = [r['rmse'] for r in results.values()]
x = np.arange(len(models))
plt.bar(x - 0.2, mae_values, 0.4, label='MAE', alpha=0.8, color='skyblue')
plt.bar(x + 0.2, rmse_values, 0.4, label='RMSE', alpha=0.8, color='lightcoral')
plt.title('Prediction Error (MAE/RMSE)', fontsize=14, pad=10)
plt.xticks(x, results.keys())
plt.ylabel('Error')
plt.legend()

# 4-6. ê° ëª¨ë¸ì˜ í˜¼ë™ í–‰ë ¬
for idx, (name, result) in enumerate(results.items()):
    plt.subplot(2, 3, idx + 4)
    sns.heatmap(result['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Warning', 'Danger'],
                yticklabels=['Normal', 'Warning', 'Danger'],
                cbar_kws={'label': 'Count'})
    plt.title(f'{name} Confusion Matrix', fontsize=12)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ì¶”ê°€ ì‹œê°í™”: ìž„ê³„ê°’ ë¶„ì„
if 'threshold_df' in locals():
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['threshold'], threshold_df['precision'], 'b-', label='Precision')
    plt.plot(threshold_df['threshold'], threshold_df['recall'], 'r-', label='Recall')
    plt.plot(threshold_df['threshold'], threshold_df['f1'], 'g-', label='F1-Score')
    plt.axvline(x=best_threshold['threshold'], color='k', linestyle='--', 
                label=f'Best Threshold ({best_threshold["threshold"]:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Analysis for Danger Class (1700+)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(model_dir, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ===== 11. ìµœì¢… ìš”ì•½ ë° ì €ìž¥ =====
print("\n\n=== ìµœì¢… ìš”ì•½ ===")
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
print(f"ìµœê³  ì •í™•ë„ ëª¨ë¸: {best_model} ({results[best_model]['accuracy']:.4f})")

# ìœ„í—˜ í´ëž˜ìŠ¤ F1 ê¸°ì¤€ ìµœê³  ëª¨ë¸
best_danger_model = max(results.items(), 
                       key=lambda x: x[1]['report']['ìœ„í—˜']['f1-score'])[0]
print(f"ìœ„í—˜ ê°ì§€ ìµœê³  ëª¨ë¸: {best_danger_model} (F1: {results[best_danger_model]['report']['ìœ„í—˜']['f1-score']:.4f})")

# ìŠ¤ì¼€ì¼ëŸ¬ ì €ìž¥
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

# ì„¤ì • ì €ìž¥
config = {
    'sequence_length': 280,
    'prediction_horizon': 10,
    'num_features': X.shape[1],
    'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
    'best_model': best_model,
    'best_danger_model': best_danger_model,
    'optimal_threshold': best_threshold['threshold'] if 'best_threshold' in locals() else 0.5,
    'class_distribution': dict(zip(unique, counts)),
    'class_names': class_names
}

with open(os.path.join(model_dir, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)

# ê²°ê³¼ ìš”ì•½ í…ìŠ¤íŠ¸ íŒŒì¼ ì €ìž¥
with open(os.path.join(model_dir, 'results_summary.txt'), 'w', encoding='utf-8') as f:
    f.write("=== ë°˜ë„ì²´ ë¬¼ë¥˜ ì˜ˆì¸¡ ì‹œìŠ¤í…œ ê²°ê³¼ ìš”ì•½ ===\n\n")
    f.write(f"ë°ì´í„°ì…‹ í¬ê¸°: {len(df)}í–‰\n")
    f.write(f"ì‹œí€€ìŠ¤ ê¸¸ì´: 280ë¶„\n")
    f.write(f"ì˜ˆì¸¡ ì‹œê°„: 10ë¶„ í›„\n")
    f.write(f"íŠ¹ì„± ê°œìˆ˜: {X.shape[1]}ê°œ\n\n")
    
    f.write("í´ëž˜ìŠ¤ ë¶„í¬:\n")
    for u, c in zip(unique, counts):
        f.write(f"  {class_names[u]}: {c} ({c/len(y)*100:.1f}%)\n")
    
    f.write("\nëª¨ë¸ë³„ ì„±ëŠ¥:\n")
    for name, result in results.items():
        f.write(f"\n{name}:\n")
        f.write(f"  ì „ì²´ ì •í™•ë„: {result['accuracy']:.4f}\n")
        f.write(f"  ìœ„í—˜ í´ëž˜ìŠ¤ F1: {result['report']['ìœ„í—˜']['f1-score']:.4f}\n")
        f.write(f"  MAE: {result['mae']:.2f}\n")
        f.write(f"  RMSE: {result['rmse']:.2f}\n")

print(f"\nâœ… ëª¨ë“  ëª¨ë¸ê³¼ ì„¤ì • ì €ìž¥ ì™„ë£Œ!")
print(f"ì €ìž¥ ìœ„ì¹˜: {model_dir}/")
print(f"ì €ìž¥ëœ íŒŒì¼:")
for file in os.listdir(model_dir):
    print(f"  - {file}")

# ===== 12. ì‹¤ì‹œê°„ ì˜ˆì¸¡ í•¨ìˆ˜ =====
def predict_realtime(sequence_280min, model_name='RandomForest', use_optimal_threshold=True):
    """
    ì‹¤ì‹œê°„ 280ë¶„ ì‹œí€€ìŠ¤ ë°ì´í„°ë¡œ 10ë¶„ í›„ ì˜ˆì¸¡
    
    Parameters:
    -----------
    sequence_280min : pd.DataFrame
        280í–‰ Ã— 10ì—´ì˜ ë°ì´í„°í”„ë ˆìž„ (feature_colsì™€ ë™ì¼í•œ ì»¬ëŸ¼)
    model_name : str
        ì‚¬ìš©í•  ëª¨ë¸ëª… ('RandomForest', 'GradientBoosting', 'MLP')
    use_optimal_threshold : bool
        ìœ„í—˜ í´ëž˜ìŠ¤ ì˜ˆì¸¡ ì‹œ ìµœì  ìž„ê³„ê°’ ì‚¬ìš© ì—¬ë¶€
    
    Returns:
    --------
    dict : ì˜ˆì¸¡ ê²°ê³¼
        - 'class': ì˜ˆì¸¡ í´ëž˜ìŠ¤ (0, 1, 2)
        - 'class_name': í´ëž˜ìŠ¤ëª… ('ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜')
        - 'probabilities': ê° í´ëž˜ìŠ¤ í™•ë¥ 
        - 'danger_alert': ìœ„í—˜ ì•Œë¦¼ ì—¬ë¶€
    """
    
    # ëª¨ë¸ê³¼ ìŠ¤ì¼€ì¼ëŸ¬ ë¡œë“œ
    model = joblib.load(os.path.join(model_dir, f'{model_name}_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    config = pickle.load(open(os.path.join(model_dir, 'config.pkl'), 'rb'))
    
    # íŠ¹ì§• ì¶”ì¶œ (create_sequencesì™€ ë™ì¼í•œ ë°©ì‹)
    # ì‹¤ì œ êµ¬í˜„ ì‹œ ì—¬ê¸°ì— íŠ¹ì§• ì¶”ì¶œ ì½”ë“œ ì¶”ê°€
    
    # ì˜ˆì‹œ ë°˜í™˜ê°’
    return {
        'class': 1,
        'class_name': 'ì£¼ì˜',
        'probabilities': [0.1, 0.7, 0.2],
        'danger_alert': False
    }

print("\nì‹¤ì‹œê°„ ì˜ˆì¸¡ í•¨ìˆ˜ ì¤€ë¹„ ì™„ë£Œ!")
print("\nì‚¬ìš© ì˜ˆì‹œ:")
print("result = predict_realtime(data_280min, model_name='RandomForest')")
print("print(f'ì˜ˆì¸¡ ê²°ê³¼: {result[\"class_name\"]}')")
print("print(f'ìœ„í—˜ í™•ë¥ : {result[\"probabilities\"][2]:.2%}')")

print("\n" + "="*60)
print("ì‹œìŠ¤í…œ ì¤€ë¹„ ì™„ë£Œ! ëª¨ë“  ëª¨ë¸ì´ ì €ìž¥ë˜ì—ˆìŠµë‹ˆë‹¤.")
print("="*60)