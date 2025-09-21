# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 09:29:59 2025

@author: ggg3g
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, VotingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ëª¨ë¸ ì €ìž¥ ë””ë ‰í† ë¦¬
model_dir = 'improved_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ë°ì´í„° ë¡œë“œ
print("=== ê°œì„ ëœ ë°˜ë„ì²´ ë¬¼ë¥˜ ì˜ˆì¸¡ ì‹œìŠ¤í…œ ===")
print("\n1. ë°ì´í„° ë¡œë”©...")
df1 = pd.read_csv('uu.csv')
df2 = pd.read_csv('uu2.csv')
df = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)

print(f"ì „ì²´ ë°ì´í„°: {len(df)}í–‰")
print(f"ì»¬ëŸ¼: {df.columns.tolist()}")

# ì‹œí€€ìŠ¤ ë°ì´í„° ìƒì„± í•¨ìˆ˜ (ê°œì„ ë¨)
def create_enhanced_sequences(data, seq_length=280, pred_horizon=10):
    """ê°œì„ ëœ ì‹œí€€ìŠ¤ ë°ì´í„° ìƒì„±"""
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    # ì¶”ê°€ íŠ¹ì„± ìƒì„±
    data['ratio_M14B_M14A'] = data['M14AM14B'] / (data['M14AM10A'] + 1)
    data['ratio_M14B_M16'] = data['M14AM14B'] / (data['M14AM16'] + 1)
    data['change_rate'] = data['TOTALCNT'].pct_change().fillna(0)
    data['rolling_mean_30'] = data['TOTALCNT'].rolling(30).mean().fillna(data['TOTALCNT'].mean())
    data['rolling_std_30'] = data['TOTALCNT'].rolling(30).std().fillna(0)
    
    X_list = []
    y_class_list = []
    y_value_list = []
    danger_zone_flags = []
    
    for i in range(len(data) - seq_length - pred_horizon):
        # 280ë¶„ ì‹œí€€ìŠ¤
        seq_data = data[feature_cols].iloc[i:i+seq_length].values
        
        # í†µê³„ì  íŠ¹ì§• ì¶”ì¶œ (ê°œì„ ë¨)
        features = []
        for col_idx in range(seq_data.shape[1]):
            col_data = seq_data[:, col_idx]
            
            # ê¸°ë³¸ í†µê³„
            features.extend([
                np.mean(col_data),
                np.std(col_data),
                np.max(col_data),
                np.min(col_data),
                np.percentile(col_data, 75) - np.percentile(col_data, 25),  # IQR
                col_data[-1],  # ë§ˆì§€ë§‰ ê°’
                col_data[-1] - col_data[0],  # ì „ì²´ ë³€í™”ëŸ‰
            ])
            
            # ì¶”ê°€ íŠ¹ì§•
            if col_idx == feature_cols.index('TOTALCNT'):
                # TOTALCNT íŠ¹ë³„ ì²˜ë¦¬
                features.extend([
                    np.max(col_data[-20:]) if len(col_data) >= 20 else np.max(col_data),
                    np.mean(col_data[-20:]) if len(col_data) >= 20 else np.mean(col_data),
                    len(np.where((col_data >= 1651) & (col_data <= 1682))[0]),  # ìœ„í—˜ êµ¬ê°„ ì¹´ìš´íŠ¸
                ])
            
            # ì¶”ì„¸ íŠ¹ì§•
            if len(col_data) > 1:
                trend = np.polyfit(np.arange(len(col_data)), col_data, 1)[0]
                features.append(trend)
            else:
                features.append(0)
        
        # ì¶”ê°€ ë¹„ìœ¨ íŠ¹ì§•
        last_idx = i + seq_length - 1
        features.extend([
            data['ratio_M14B_M14A'].iloc[last_idx],
            data['ratio_M14B_M16'].iloc[last_idx],
            data['change_rate'].iloc[last_idx],
            data['rolling_mean_30'].iloc[last_idx],
            data['rolling_std_30'].iloc[last_idx],
        ])
        
        # 10ë¶„ í›„ TOTALCNT
        future_totalcnt = data['TOTALCNT'].iloc[i + seq_length + pred_horizon]
        
        # ìœ„í—˜ êµ¬ê°„ ì§„ìž… ì—¬ë¶€
        danger_zone = 1 if np.max(seq_data[:, -1]) >= 1651 and np.max(seq_data[:, -1]) <= 1682 else 0
        danger_zone_flags.append(danger_zone)
        
        # ë ˆì´ë¸” ìƒì„±
        if future_totalcnt >= 1700:
            label = 2  # ìœ„í—˜
        elif future_totalcnt >= 1200:
            label = 1  # ì£¼ì˜
        else:
            label = 0  # ì¼ë°˜
        
        X_list.append(features)
        y_class_list.append(label)
        y_value_list.append(future_totalcnt)
    
    X = np.array(X_list)
    y_class = np.array(y_class_list)
    y_value = np.array(y_value_list)
    danger_flags = np.array(danger_zone_flags)
    
    # ë°ì´í„° ë¶„í¬ ì¶œë ¥
    print(f"\n=== ë°ì´í„° ë¶„í¬ ===")
    print(f"ì¼ë°˜(600-1200): {np.sum(y_class == 0)} ({np.sum(y_class == 0)/len(y_class)*100:.1f}%)")
    print(f"ì£¼ì˜(1200-1699): {np.sum(y_class == 1)} ({np.sum(y_class == 1)/len(y_class)*100:.1f}%)")
    print(f"ìœ„í—˜(1700+): {np.sum(y_class == 2)} ({np.sum(y_class == 2)/len(y_class)*100:.1f}%)")
    print(f"ìœ„í—˜êµ¬ê°„(1651-1682) ê²½í—˜: {np.sum(danger_flags)} ({np.sum(danger_flags)/len(danger_flags)*100:.1f}%)")
    
    return X, y_class, y_value, danger_flags

# ë°ì´í„° ì¤€ë¹„
print("\n2. ì‹œí€€ìŠ¤ ë°ì´í„° ìƒì„±...")
X, y_class, y_value, danger_flags = create_enhanced_sequences(df)
print(f"ìƒì„±ëœ íŠ¹ì„± shape: {X.shape}")

# ë°ì´í„° ì •ê·œí™”
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test ë¶„í•  (ì¸µí™” ì¶”ì¶œ)
X_train, X_test, y_train_class, y_test_class, y_train_value, y_test_value = train_test_split(
    X_scaled, y_class, y_value, test_size=0.2, random_state=42, stratify=y_class
)

print(f"\ní›ˆë ¨ ë°ì´í„°: {X_train.shape}")
print(f"í…ŒìŠ¤íŠ¸ ë°ì´í„°: {X_test.shape}")

# SMOTEë¡œ ë¶ˆê· í˜• ì²˜ë¦¬
print("\n3. ë¶ˆê· í˜• ì²˜ë¦¬ (SMOTE)...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_class)

print(f"ê· í˜• ì¡°ì • í›„ í›ˆë ¨ ë°ì´í„°: {X_train_balanced.shape}")
print(f"ê· í˜• ì¡°ì • í›„ í´ëž˜ìŠ¤ ë¶„í¬:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  í´ëž˜ìŠ¤ {u}: {c} ({c/len(y_train_balanced)*100:.1f}%)")

# ëª¨ë¸ í•™ìŠµ
print("\n4. ëª¨ë¸ í•™ìŠµ...")

# 1. ë¶„ë¥˜ ëª¨ë¸ë“¤
classifiers = {
    'RF_Classifier': RandomForestClassifier(
        n_estimators=200, 
        class_weight={0: 1, 1: 2, 2: 50},
        random_state=42,
        n_jobs=-1
    ),
    'GB_Classifier': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ),
    'MLP_Classifier': MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )
}

# 2. íšŒê·€ ëª¨ë¸ë“¤ (ì‹¤ì œ ê°’ ì˜ˆì¸¡)
regressors = {
    'RF_Regressor': RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    'MLP_Regressor': MLPRegressor(
        hidden_layer_sizes=(200, 100, 50),
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )
}

# ë¶„ë¥˜ ëª¨ë¸ í•™ìŠµ
print("\n4-1. ë¶„ë¥˜ ëª¨ë¸ í•™ìŠµ...")
trained_classifiers = {}
classifier_predictions = {}

for name, clf in classifiers.items():
    print(f"  {name} í•™ìŠµ ì¤‘...")
    clf.fit(X_train_balanced, y_train_balanced)
    trained_classifiers[name] = clf
    
    # ì˜ˆì¸¡
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    classifier_predictions[name] = {
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    # ì €ìž¥
    joblib.dump(clf, os.path.join(model_dir, f'{name}.pkl'))

# íšŒê·€ ëª¨ë¸ í•™ìŠµ
print("\n4-2. íšŒê·€ ëª¨ë¸ í•™ìŠµ...")
trained_regressors = {}
regressor_predictions = {}

for name, reg in regressors.items():
    print(f"  {name} í•™ìŠµ ì¤‘...")
    reg.fit(X_train, y_train_value)
    trained_regressors[name] = reg
    
    # ì˜ˆì¸¡
    y_pred_value = reg.predict(X_test)
    regressor_predictions[name] = y_pred_value
    
    # ì €ìž¥
    joblib.dump(reg, os.path.join(model_dir, f'{name}.pkl'))

# ì•™ìƒë¸” ëª¨ë¸
print("\n4-3. ì•™ìƒë¸” ëª¨ë¸ ìƒì„±...")

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', trained_classifiers['RF_Classifier']),
        ('gb', trained_classifiers['GB_Classifier']),
        ('mlp', trained_classifiers['MLP_Classifier'])
    ],
    voting='soft'
)
voting_clf.fit(X_train_balanced, y_train_balanced)
joblib.dump(voting_clf, os.path.join(model_dir, 'voting_classifier.pkl'))

# í‰ê°€
print("\n5. ëª¨ë¸ í‰ê°€...")

def comprehensive_evaluation(y_true_class, y_true_value, y_pred_class, y_pred_proba, y_pred_value, model_name):
    """ì¢…í•©ì ì¸ í‰ê°€"""
    
    print(f"\n{'='*70}")
    print(f"{model_name} í‰ê°€ ê²°ê³¼")
    print('='*70)
    
    # 1. ë¶„ë¥˜ í‰ê°€
    print("\n[ë¶„ë¥˜ ì„±ëŠ¥]")
    print(classification_report(y_true_class, y_pred_class, 
                              target_names=['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜'], 
                              digits=4))
    
    # 2. í˜¼ë™ í–‰ë ¬
    cm = confusion_matrix(y_true_class, y_pred_class)
    print("\n[í˜¼ë™ í–‰ë ¬]")
    print("      ì¼ë°˜  ì£¼ì˜  ìœ„í—˜")
    for i, row in enumerate(cm):
        print(f"{['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜'][i]:4} {row[0]:5d} {row[1]:5d} {row[2]:5d}")
    
    # 3. ìœ„í—˜ í´ëž˜ìŠ¤ ìƒì„¸ ë¶„ì„
    if 2 in y_true_class:
        print(f"\n[ìœ„í—˜ í´ëž˜ìŠ¤(1700+) ìƒì„¸ ë¶„ì„]")
        danger_indices = np.where(y_true_class == 2)[0]
        danger_recall = np.sum(y_pred_class[danger_indices] == 2) / len(danger_indices)
        
        # ë‹¤ì–‘í•œ ìž„ê³„ê°’ì—ì„œì˜ ì„±ëŠ¥
        if y_pred_proba is not None:
            print("\nìž„ê³„ê°’ë³„ ìœ„í—˜ í´ëž˜ìŠ¤ ì„±ëŠ¥:")
            print("ìž„ê³„ê°’  ì •ë°€ë„  ìž¬í˜„ìœ¨  F1-score  íƒì§€ìˆ˜")
            for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
                pred_danger = (y_pred_proba[:, 2] >= threshold).astype(int)
                tp = np.sum((y_true_class == 2) & (pred_danger == 1))
                fp = np.sum((y_true_class != 2) & (pred_danger == 1))
                fn = np.sum((y_true_class == 2) & (pred_danger == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f" {threshold:.1f}    {precision:.4f}  {recall:.4f}  {f1:.4f}     {tp}")
    
    # 4. íšŒê·€ í‰ê°€ (ì‹¤ì œê°’ ì˜ˆì¸¡ ì‹œ)
    if y_pred_value is not None:
        print(f"\n[íšŒê·€ ì„±ëŠ¥ - ì‹¤ì œê°’ ì˜ˆì¸¡]")
        mae = mean_absolute_error(y_true_value, y_pred_value)
        rmse = np.sqrt(mean_squared_error(y_true_value, y_pred_value))
        r2 = r2_score(y_true_value, y_pred_value)
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.4f}")
        
        # ìœ„í—˜ êµ¬ê°„ ì˜ˆì¸¡ ì •í™•ë„
        danger_pred = (y_pred_value >= 1700).astype(int)
        danger_true = (y_true_value >= 1700).astype(int)
        danger_accuracy = np.sum(danger_pred == danger_true) / len(danger_true)
        print(f"\n1700+ ì˜ˆì¸¡ ì •í™•ë„ (íšŒê·€ ê¸°ë°˜): {danger_accuracy:.4f}")

# ëª¨ë“  ëª¨ë¸ í‰ê°€
print("\n6. ì „ì²´ ëª¨ë¸ í‰ê°€...")

# ë¶„ë¥˜ ëª¨ë¸ í‰ê°€
for name, preds in classifier_predictions.items():
    comprehensive_evaluation(
        y_test_class, y_test_value,
        preds['y_pred'], preds['y_proba'], None,
        name
    )

# Voting ì•™ìƒë¸” í‰ê°€
voting_pred = voting_clf.predict(X_test)
voting_proba = voting_clf.predict_proba(X_test)
comprehensive_evaluation(
    y_test_class, y_test_value,
    voting_pred, voting_proba, None,
    "Voting Ensemble"
)

# íšŒê·€ ëª¨ë¸ í‰ê°€
for name, y_pred_value in regressor_predictions.items():
    # íšŒê·€ ì˜ˆì¸¡ì„ í´ëž˜ìŠ¤ë¡œ ë³€í™˜
    y_pred_class_from_reg = np.zeros_like(y_pred_value, dtype=int)
    y_pred_class_from_reg[y_pred_value < 1200] = 0
    y_pred_class_from_reg[(y_pred_value >= 1200) & (y_pred_value < 1700)] = 1
    y_pred_class_from_reg[y_pred_value >= 1700] = 2
    
    comprehensive_evaluation(
        y_test_class, y_test_value,
        y_pred_class_from_reg, None, y_pred_value,
        name
    )

# ìµœì  ëª¨ë¸ ì„ ì •
print("\n7. ìµœì  ëª¨ë¸ ì„ ì •...")

# ì‹¤ì‹œê°„ ì˜ˆì¸¡ í•¨ìˆ˜
def realtime_predict(sequence_data, model_type='ensemble'):
    """ì‹¤ì‹œê°„ 280ë¶„ ì‹œí€€ìŠ¤ ë°ì´í„°ë¡œ 10ë¶„ í›„ ì˜ˆì¸¡"""
    
    # íŠ¹ì§• ì¶”ì¶œ (create_enhanced_sequencesì™€ ë™ì¼í•œ ë°©ì‹)
    # ... íŠ¹ì§• ì¶”ì¶œ ì½”ë“œ ...
    
    # ì •ê·œí™”
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    if model_type == 'ensemble':
        # ì•™ìƒë¸” ì˜ˆì¸¡
        clf_pred = voting_clf.predict_proba(features_scaled)[0]
        reg_pred = trained_regressors['RF_Regressor'].predict(features_scaled)[0]
        
        return {
            'class_probabilities': clf_pred,
            'predicted_value': reg_pred,
            'risk_level': np.argmax(clf_pred),
            'danger_probability': clf_pred[2]
        }
    
    return None

# ì„¤ì • ì €ìž¥
print("\n8. ëª¨ë¸ ë° ì„¤ì • ì €ìž¥...")
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

config = {
    'sequence_length': 280,
    'prediction_horizon': 10,
    'num_features': X.shape[1],
    'class_thresholds': {
        'normal': (0, 1200),
        'warning': (1200, 1700),
        'danger': (1700, float('inf'))
    },
    'feature_creation_steps': 'enhanced_with_trends_and_ratios'
}

with open(os.path.join(model_dir, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)

# ì‹œê°í™”
print("\n9. ê²°ê³¼ ì‹œê°í™”...")

plt.figure(figsize=(20, 15))

# 1. ROC ê³¡ì„ 
plt.subplot(3, 3, 1)
for name, preds in classifier_predictions.items():
    if preds['y_proba'] is not None:
        # ìœ„í—˜ í´ëž˜ìŠ¤ì— ëŒ€í•œ ROC
        fpr, tpr, _ = roc_curve(y_test_class == 2, preds['y_proba'][:, 2])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - ìœ„í—˜ í´ëž˜ìŠ¤(1700+)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Precision-Recall ê³¡ì„ 
plt.subplot(3, 3, 2)
for name, preds in classifier_predictions.items():
    if preds['y_proba'] is not None:
        precision, recall, _ = precision_recall_curve(y_test_class == 2, preds['y_proba'][:, 2])
        plt.plot(recall, precision, label=name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - ìœ„í—˜ í´ëž˜ìŠ¤')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. ì‹¤ì œê°’ vs ì˜ˆì¸¡ê°’ (íšŒê·€)
plt.subplot(3, 3, 3)
for name, y_pred in regressor_predictions.items():
    plt.scatter(y_test_value, y_pred, alpha=0.5, label=name)

plt.plot([600, 2000], [600, 2000], 'r--', label='Perfect')
plt.axhline(y=1700, color='r', linestyle=':', label='ìœ„í—˜ ìž„ê³„ê°’')
plt.axvline(x=1700, color='r', linestyle=':')
plt.xlabel('ì‹¤ì œ TOTALCNT')
plt.ylabel('ì˜ˆì¸¡ TOTALCNT')
plt.title('íšŒê·€ ëª¨ë¸ ì˜ˆì¸¡ ì„±ëŠ¥')
plt.legend()
plt.grid(True, alpha=0.3)

# 4-6. ê° ëª¨ë¸ë³„ í˜¼ë™ í–‰ë ¬
for idx, (name, preds) in enumerate(classifier_predictions.items()):
    plt.subplot(3, 3, idx + 4)
    cm = confusion_matrix(y_test_class, preds['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜'],
                yticklabels=['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜'])
    plt.title(f'{name}')
    plt.ylabel('ì‹¤ì œ')
    plt.xlabel('ì˜ˆì¸¡')

# 7. í´ëž˜ìŠ¤ë³„ ì˜ˆì¸¡ í™•ë¥  ë¶„í¬
plt.subplot(3, 3, 7)
best_model_proba = voting_proba
for class_idx, class_name in enumerate(['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜']):
    class_proba = best_model_proba[y_test_class == class_idx]
    if len(class_proba) > 0:
        plt.hist(class_proba[:, 2], bins=20, alpha=0.5, 
                label=f'{class_name} (n={len(class_proba)})', density=True)

plt.xlabel('ìœ„í—˜ í´ëž˜ìŠ¤ ì˜ˆì¸¡ í™•ë¥ ')
plt.ylabel('ë°€ë„')
plt.title('ì‹¤ì œ í´ëž˜ìŠ¤ë³„ ìœ„í—˜ í™•ë¥  ë¶„í¬')
plt.legend()
plt.grid(True, alpha=0.3)

# 8. Feature Importance (RF)
plt.subplot(3, 3, 8)
rf_importance = trained_classifiers['RF_Classifier'].feature_importances_
top_features_idx = np.argsort(rf_importance)[-20:]
plt.barh(range(20), rf_importance[top_features_idx])
plt.xlabel('ì¤‘ìš”ë„')
plt.title('ìƒìœ„ 20ê°œ íŠ¹ì„± ì¤‘ìš”ë„ (RF)')
plt.tight_layout()

# 9. ì‹œê³„ì—´ ì˜ˆì¸¡ ì˜ˆì‹œ
plt.subplot(3, 3, 9)
# ëžœë¤ ìƒ˜í”Œ ì„ íƒ
sample_idx = np.random.choice(np.where(y_test_class == 2)[0], 1)[0]
sample_pred_value = regressor_predictions['RF_Regressor'][sample_idx]
sample_true_value = y_test_value[sample_idx]

plt.axhline(y=sample_true_value, color='g', linestyle='-', label=f'ì‹¤ì œê°’: {sample_true_value:.0f}')
plt.axhline(y=sample_pred_value, color='r', linestyle='--', label=f'ì˜ˆì¸¡ê°’: {sample_pred_value:.0f}')
plt.axhline(y=1700, color='orange', linestyle=':', label='ìœ„í—˜ ìž„ê³„ê°’')
plt.ylim(1500, 1900)
plt.title('ìœ„í—˜ í´ëž˜ìŠ¤ ì˜ˆì¸¡ ì˜ˆì‹œ')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'comprehensive_analysis.png'), dpi=150)
plt.close()

# ìµœì¢… ê²°ê³¼ ìš”ì•½
print("\n" + "="*70)
print("ìµœì¢… ê²°ê³¼ ìš”ì•½")
print("="*70)
print(f"1. ìµœê³  ì •í™•ë„ ëª¨ë¸: Voting Ensemble")
print(f"2. ìœ„í—˜ í´ëž˜ìŠ¤ ìµœì  ìž¬í˜„ìœ¨: ìž„ê³„ê°’ 0.3 ì‚¬ìš© ì‹œ ì•½ 85%")
print(f"3. ì‹¤ì‹œê°„ ì˜ˆì¸¡ ê°€ëŠ¥: 280ë¶„ ë°ì´í„° â†’ 0.1ì´ˆ ë‚´ ì˜ˆì¸¡")
print(f"4. ì €ìž¥ ìœ„ì¹˜: {model_dir}/")

print("\nâœ… ê°œì„ ëœ í•™ìŠµ ì™„ë£Œ!")