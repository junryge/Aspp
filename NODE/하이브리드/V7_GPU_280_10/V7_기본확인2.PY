# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 09:20:37 2025

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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ëª¨ë¸ ì €ìž¥ ë””ë ‰í† ë¦¬ ìƒì„±
model_dir = 'saved_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ë°ì´í„° ë¡œë“œ
print("=== ë°ì´í„° ë¡œë”© ì¤‘ ===")
# ì‹¤ì œ ì‚¬ìš©ì‹œ ì•„ëž˜ ë¼ì¸ì„ ì£¼ì„ í•´ì œí•˜ê³  ì‚¬ìš©
# df = pd.read_csv('semiconductor_data.csv')

# í…ŒìŠ¤íŠ¸ìš© - ë‘ íŒŒì¼ í•©ì¹˜ê¸°
df1 = pd.read_csv('uu.csv')
df2 = pd.read_csv('uu2.csv')
df = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)

print(f"ì „ì²´ ë°ì´í„°: {len(df)}í–‰")
print(f"ì»¬ëŸ¼: {df.columns.tolist()}")

# ì‹œí€€ìŠ¤ ë°ì´í„° ìƒì„± í•¨ìˆ˜
def create_sequences(data, seq_length=280, pred_horizon=10):
    """280ë¶„ ì‹œí€€ìŠ¤ë¡œ 10ë¶„ í›„ ì˜ˆì¸¡ ë°ì´í„° ìƒì„±"""
    
    # ì‚¬ìš©í•  íŠ¹ì„± ì»¬ëŸ¼ë“¤
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    # ì¶”ê°€ íŠ¹ì„± ìƒì„± (ë¹„ìœ¨ ë“±)
    data['ratio_M14B_M14A'] = data['M14AM14B'] / (data['M14AM10A'] + 1)
    data['ratio_M14B_M16'] = data['M14AM14B'] / (data['M14AM16'] + 1)
    data['change_rate'] = data['TOTALCNT'].pct_change().fillna(0)
    
    X_list = []
    y_list = []
    y_values = []  # ì‹¤ì œ TOTALCNT ê°’ë„ ì €ìž¥
    
    for i in range(len(data) - seq_length - pred_horizon):
        # 280ë¶„ ì‹œí€€ìŠ¤
        seq_data = data[feature_cols].iloc[i:i+seq_length].values
        
        # ì‹œí€€ìŠ¤ íŠ¹ì§• ì¶”ì¶œ (í‰ê· , ìµœëŒ€, ìµœì†Œ, í‘œì¤€íŽ¸ì°¨ ë“±)
        features = []
        for col_idx in range(seq_data.shape[1]):
            col_data = seq_data[:, col_idx]
            features.extend([
                np.mean(col_data),
                np.std(col_data),
                np.max(col_data),
                np.min(col_data),
                np.percentile(col_data, 75) - np.percentile(col_data, 25),  # IQR
                col_data[-1],  # ë§ˆì§€ë§‰ ê°’
                col_data[-1] - col_data[0],  # ë³€í™”ëŸ‰
                np.max(col_data[-20:]) if len(col_data) >= 20 else np.max(col_data)  # ìµœê·¼ 20ë¶„ ìµœëŒ€
            ])
        
        # 10ë¶„ í›„ TOTALCNT
        future_totalcnt = data['TOTALCNT'].iloc[i + seq_length + pred_horizon]
        
        # ë ˆì´ë¸” ìƒì„± (3í´ëž˜ìŠ¤ ë¶„ë¥˜)
        if future_totalcnt >= 1700:
            label = 2  # ìœ„í—˜
        elif future_totalcnt >= 1200:
            label = 1  # ì£¼ì˜
        else:
            label = 0  # ì¼ë°˜
        
        X_list.append(features)
        y_list.append(label)
        y_values.append(future_totalcnt)
    
    X = np.array(X_list)
    y = np.array(y_list)
    y_vals = np.array(y_values)
    
    # ë°ì´í„° ë¶„í¬ ì¶œë ¥
    print(f"\n=== ë°ì´í„° ë¶„í¬ ===")
    print(f"ì¼ë°˜(600-1200): {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"ì£¼ì˜(1200-1699): {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"ìœ„í—˜(1700+): {np.sum(y == 2)} ({np.sum(y == 2)/len(y)*100:.1f}%)")
    
    return X, y, y_vals, feature_cols

# Custom Transformers
class AttentionTransformer(BaseEstimator, TransformerMixin):
    """TabTransformerì˜ Attention ê°œë…"""
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        self.feature_importance_ = np.abs(np.corrcoef(X.T, y)[:-1, -1])
        return self
        
    def transform(self, X):
        return X * self.feature_importance_.reshape(1, -1)

class FeatureTokenizer(BaseEstimator, TransformerMixin):
    """FT-Transformerì˜ Feature Tokenization ê°œë…"""
    def __init__(self, n_tokens=10):
        self.n_tokens = n_tokens
        
    def fit(self, X, y=None):
        self.tokenizers = []
        for i in range(X.shape[1]):
            bins = np.percentile(X[:, i], np.linspace(0, 100, self.n_tokens))
            self.tokenizers.append(bins)
        return self
        
    def transform(self, X):
        X_tokenized = np.zeros((X.shape[0], X.shape[1] * self.n_tokens))
        for i in range(X.shape[1]):
            digitized = np.digitize(X[:, i], self.tokenizers[i])
            for j in range(X.shape[0]):
                if digitized[j] < self.n_tokens:
                    X_tokenized[j, i * self.n_tokens + digitized[j]] = 1
        return X_tokenized

class InterSampleAttention(BaseEstimator, TransformerMixin):
    """SAINTì˜ Inter-sample attention ê°œë…"""
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y=None):
        self.X_train = X
        self.y_train = y
        return self
        
    def transform(self, X):
        from sklearn.metrics.pairwise import cosine_similarity
        
        X_enhanced = X.copy()
        similarities = cosine_similarity(X, self.X_train)
        
        for i in range(X.shape[0]):
            top_k_idx = np.argsort(similarities[i])[-self.n_neighbors:]
            
            if self.y_train is not None:
                similar_labels = self.y_train[top_k_idx]
                danger_ratio = np.sum(similar_labels == 2) / len(similar_labels)
                X_enhanced[i, -1] = danger_ratio
                
        return X_enhanced

class AutomaticInteraction(BaseEstimator, TransformerMixin):
    """AutoIntì˜ ìžë™ íŠ¹ì„± ìƒí˜¸ìž‘ìš©"""
    def __init__(self, n_interactions=10):
        self.n_interactions = n_interactions
        
    def fit(self, X, y=None):
        from itertools import combinations
        
        n_features = X.shape[1]
        correlations = []
        
        for i, j in combinations(range(min(n_features, 20)), 2):  # ìƒìœ„ 20ê°œ íŠ¹ì„±ë§Œ
            interaction = X[:, i] * X[:, j]
            corr = np.abs(np.corrcoef(interaction, y)[0, 1])
            correlations.append((i, j, corr))
            
        correlations.sort(key=lambda x: x[2], reverse=True)
        self.top_interactions = correlations[:self.n_interactions]
        return self
        
    def transform(self, X):
        interactions = []
        for i, j, _ in self.top_interactions:
            interactions.append(X[:, i] * X[:, j])
            
        if interactions:
            X_interactions = np.column_stack(interactions)
            return np.hstack([X, X_interactions])
        return X

class SelfSupervisedEncoder(BaseEstimator, TransformerMixin):
    """VIMEì˜ Self-supervised ê°œë…"""
    def __init__(self, corruption_rate=0.2):
        self.corruption_rate = corruption_rate
        
    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
        
    def transform(self, X):
        X_corrupted = X.copy()
        mask = np.random.random(X.shape) < self.corruption_rate
        X_corrupted[mask] = np.random.normal(self.mean_[None, :], self.std_[None, :], X.shape)[mask]
        
        X_denoised = X_corrupted * 0.8 + X * 0.2
        return X_denoised

# í‰ê°€ í•¨ìˆ˜
def evaluate_model(y_true, y_pred, y_true_values, y_pred_proba, model_name):
    """ëª¨ë¸ì˜ ì¢…í•©ì ì¸ í‰ê°€"""
    print(f"\n{'='*50}")
    print(f"{model_name} í‰ê°€ ê²°ê³¼")
    print('='*50)
    
    # 1. ì „ì²´ ì •í™•ë„
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nì „ì²´ ì •í™•ë„: {accuracy:.4f}")
    
    # 2. í´ëž˜ìŠ¤ë³„ ì„±ëŠ¥ (Precision, Recall, F1-score)
    print("\ní´ëž˜ìŠ¤ë³„ ì„±ëŠ¥:")
    class_names = ['ì¼ë°˜(600-1200)', 'ì£¼ì˜(1200-1699)', 'ìœ„í—˜(1700+)']
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # 3. ê° í´ëž˜ìŠ¤ë³„ ìƒì„¸ ì ìˆ˜
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    print("\ní´ëž˜ìŠ¤ë³„ ìƒì„¸ ì ìˆ˜:")
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-score: {f1[i]:.4f}")
        print(f"  Support: {support[i]}")
    
    # 4. Confusion Matrix
    print("\ní˜¼ë™ í–‰ë ¬:")
    cm = confusion_matrix(y_true, y_pred)
    print("       ì˜ˆì¸¡ â†’")
    print("       ì¼ë°˜  ì£¼ì˜  ìœ„í—˜")
    for i, row in enumerate(cm):
        print(f"{class_names[i][:2]} {row[0]:5d} {row[1]:5d} {row[2]:5d}")
    
    # 5. ì‹¤ì œê°’ ê¸°ì¤€ í‰ê°€ (íšŒê·€ ê´€ì )
    # ì˜ˆì¸¡ëœ í´ëž˜ìŠ¤ë¥¼ ëŒ€í‘œê°’ìœ¼ë¡œ ë³€í™˜
    class_to_value = {0: 900, 1: 1450, 2: 1750}  # ê° í´ëž˜ìŠ¤ì˜ ëŒ€í‘œê°’
    y_pred_values = np.array([class_to_value[pred] for pred in y_pred])
    
    mae = mean_absolute_error(y_true_values, y_pred_values)
    rmse = np.sqrt(mean_squared_error(y_true_values, y_pred_values))
    
    # R2ëŠ” ë¶„ë¥˜ì—ì„œëŠ” ì˜ë¯¸ê°€ ì ì§€ë§Œ ì°¸ê³ ìš©ìœ¼ë¡œ
    # ì‹¤ì œê°’ê³¼ ì˜ˆì¸¡ í´ëž˜ìŠ¤ì˜ ëŒ€í‘œê°’ ê°„ ìƒê´€ê´€ê³„
    r2 = r2_score(y_true_values, y_pred_values)
    
    print(f"\níšŒê·€ ê´€ì  í‰ê°€ (ì°¸ê³ ìš©):")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2 Score: {r2:.4f}")
    
    # 6. ìœ„í—˜ í´ëž˜ìŠ¤(1700+) íŠ¹ë³„ ë¶„ì„
    if 2 in y_true:
        print(f"\nìœ„í—˜ í´ëž˜ìŠ¤(1700+) íŠ¹ë³„ ë¶„ì„:")
        true_danger = y_true == 2
        pred_danger = y_pred == 2
        
        true_positives = np.sum(true_danger & pred_danger)
        false_positives = np.sum(~true_danger & pred_danger)
        false_negatives = np.sum(true_danger & ~pred_danger)
        
        print(f"  ì •í™•ížˆ ì˜ˆì¸¡í•œ 1700+: {true_positives}ê°œ")
        print(f"  ìž˜ëª» ì˜ˆì¸¡í•œ 1700+ (ì˜¤íƒ): {false_positives}ê°œ")
        print(f"  ë†“ì¹œ 1700+: {false_negatives}ê°œ")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'confusion_matrix': cm
    }

# ë©”ì¸ ì‹¤í–‰ ë¶€ë¶„
print("\n=== ë°ì´í„° ì¤€ë¹„ ===")
X, y, y_values, feature_names = create_sequences(df)
print(f"ìƒì„±ëœ íŠ¹ì„± shape: {X.shape}")

# ë°ì´í„° ì •ê·œí™”
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ìŠ¤ì¼€ì¼ëŸ¬ ì €ìž¥
scaler_path = os.path.join(model_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"\nìŠ¤ì¼€ì¼ëŸ¬ ì €ìž¥ ì™„ë£Œ: {scaler_path}")

# Train/Test ë¶„í• 
X_train, X_test, y_train, y_test, y_train_vals, y_test_vals = train_test_split(
    X_scaled, y, y_values, test_size=0.2, random_state=42, stratify=y
)

print(f"\ní›ˆë ¨ ë°ì´í„°: {X_train.shape}")
print(f"í…ŒìŠ¤íŠ¸ ë°ì´í„°: {X_test.shape}")

# í´ëž˜ìŠ¤ ê°€ì¤‘ì¹˜
class_weights = {0: 1.0, 1: 2.0, 2: 50.0}

# ëª¨ë¸ í•™ìŠµ ë° í‰ê°€
print("\n=== 5ê°œ ëª¨ë¸ í•™ìŠµ ë° í‰ê°€ ===")

# 1. TabTransformer ì»¨ì…‰
print("\n1. TabTransformer ì»¨ì…‰ í•™ìŠµ...")
attention_transformer = AttentionTransformer()
X_tab = attention_transformer.fit_transform(X_train, y_train)
tab_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
tab_model.fit(X_tab, y_train)

# ì˜ˆì¸¡ ë° í‰ê°€
X_test_tab = attention_transformer.transform(X_test)
y_pred_tab = tab_model.predict(X_test_tab)
y_proba_tab = tab_model.predict_proba(X_test_tab)
results_tab = evaluate_model(y_test, y_pred_tab, y_test_vals, y_proba_tab, "TabTransformer")

# ëª¨ë¸ê³¼ íŠ¸ëžœìŠ¤í¬ë¨¸ ì €ìž¥
joblib.dump(tab_model, os.path.join(model_dir, 'tab_model.pkl'))
joblib.dump(attention_transformer, os.path.join(model_dir, 'tab_transformer.pkl'))

# 2. FT-Transformer ì»¨ì…‰
print("\n2. FT-Transformer ì»¨ì…‰ í•™ìŠµ...")
tokenizer = FeatureTokenizer(n_tokens=10)
X_ft = tokenizer.fit_transform(X_train, y_train)
ft_model = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
ft_model.fit(X_ft, y_train)

# ì˜ˆì¸¡ ë° í‰ê°€
X_test_ft = tokenizer.transform(X_test)
y_pred_ft = ft_model.predict(X_test_ft)
y_proba_ft = ft_model.predict_proba(X_test_ft)
results_ft = evaluate_model(y_test, y_pred_ft, y_test_vals, y_proba_ft, "FT-Transformer")

# ëª¨ë¸ê³¼ íŠ¸ëžœìŠ¤í¬ë¨¸ ì €ìž¥
joblib.dump(ft_model, os.path.join(model_dir, 'ft_model.pkl'))
joblib.dump(tokenizer, os.path.join(model_dir, 'ft_tokenizer.pkl'))

# 3. SAINT ì»¨ì…‰
print("\n3. SAINT ì»¨ì…‰ í•™ìŠµ...")
saint_transformer = InterSampleAttention()
X_saint = saint_transformer.fit_transform(X_train, y_train)
saint_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
saint_model.fit(X_saint, y_train)

# ì˜ˆì¸¡ ë° í‰ê°€
X_test_saint = saint_transformer.transform(X_test)
y_pred_saint = saint_model.predict(X_test_saint)
y_proba_saint = saint_model.predict_proba(X_test_saint)
results_saint = evaluate_model(y_test, y_pred_saint, y_test_vals, y_proba_saint, "SAINT")

# ëª¨ë¸ê³¼ íŠ¸ëžœìŠ¤í¬ë¨¸ ì €ìž¥
joblib.dump(saint_model, os.path.join(model_dir, 'saint_model.pkl'))
joblib.dump(saint_transformer, os.path.join(model_dir, 'saint_transformer.pkl'))

# 4. AutoInt ì»¨ì…‰
print("\n4. AutoInt ì»¨ì…‰ í•™ìŠµ...")
autoint_transformer = AutomaticInteraction()
X_autoint = autoint_transformer.fit_transform(X_train, y_train)
autoint_model = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=500, random_state=42)
autoint_model.fit(X_autoint, y_train)

# ì˜ˆì¸¡ ë° í‰ê°€
X_test_autoint = autoint_transformer.transform(X_test)
y_pred_autoint = autoint_model.predict(X_test_autoint)
y_proba_autoint = autoint_model.predict_proba(X_test_autoint)
results_autoint = evaluate_model(y_test, y_pred_autoint, y_test_vals, y_proba_autoint, "AutoInt")

# ëª¨ë¸ê³¼ íŠ¸ëžœìŠ¤í¬ë¨¸ ì €ìž¥
joblib.dump(autoint_model, os.path.join(model_dir, 'autoint_model.pkl'))
joblib.dump(autoint_transformer, os.path.join(model_dir, 'autoint_transformer.pkl'))

# 5. VIME ì»¨ì…‰
print("\n5. VIME ì»¨ì…‰ í•™ìŠµ...")
vime_transformer = SelfSupervisedEncoder()
X_vime = vime_transformer.fit_transform(X_train)
vime_model = RandomForestClassifier(n_estimators=50, class_weight=class_weights, random_state=42)
vime_model.fit(X_vime, y_train)

# ì˜ˆì¸¡ ë° í‰ê°€
X_test_vime = vime_transformer.transform(X_test)
y_pred_vime = vime_model.predict(X_test_vime)
y_proba_vime = vime_model.predict_proba(X_test_vime)
results_vime = evaluate_model(y_test, y_pred_vime, y_test_vals, y_proba_vime, "VIME")

# ëª¨ë¸ê³¼ íŠ¸ëžœìŠ¤í¬ë¨¸ ì €ìž¥
joblib.dump(vime_model, os.path.join(model_dir, 'vime_model.pkl'))
joblib.dump(vime_transformer, os.path.join(model_dir, 'vime_transformer.pkl'))

# ì „ì²´ ëª¨ë¸ ë¹„êµ ì‹œê°í™”
print("\n\n=== ì „ì²´ ëª¨ë¸ ì„±ëŠ¥ ë¹„êµ ===")

models_results = {
    'TabTransformer': results_tab,
    'FT-Transformer': results_ft,
    'SAINT': results_saint,
    'AutoInt': results_autoint,
    'VIME': results_vime
}

# ì„±ëŠ¥ ë¹„êµ í‘œ
print("\nëª¨ë¸ë³„ ì¢…í•© ì„±ëŠ¥ ë¹„êµ:")
print("-" * 120)
print(f"{'ëª¨ë¸':15} {'ì •í™•ë„':8} {'ì¼ë°˜ F1':8} {'ì£¼ì˜ F1':8} {'ìœ„í—˜ F1':8} {'MAE':8} {'RMSE':8} {'R2':8}")
print("-" * 120)

for model_name, results in models_results.items():
    print(f"{model_name:15} {results['accuracy']:8.4f} "
          f"{results['f1'][0]:8.4f} {results['f1'][1]:8.4f} {results['f1'][2]:8.4f} "
          f"{results['mae']:8.2f} {results['rmse']:8.2f} {results['r2']:8.4f}")

# ìµœê³  ì„±ëŠ¥ ëª¨ë¸ ì°¾ê¸°
best_model_by_accuracy = max(models_results.items(), key=lambda x: x[1]['accuracy'])[0]
best_model_by_danger_f1 = max(models_results.items(), key=lambda x: x[1]['f1'][2])[0]

print(f"\nìµœê³  ì •í™•ë„ ëª¨ë¸: {best_model_by_accuracy} ({models_results[best_model_by_accuracy]['accuracy']:.4f})")
print(f"ìœ„í—˜ í´ëž˜ìŠ¤ F1 ìµœê³  ëª¨ë¸: {best_model_by_danger_f1} ({models_results[best_model_by_danger_f1]['f1'][2]:.4f})")

# ì‹œê°í™”
plt.figure(figsize=(20, 15))

# 1. ëª¨ë¸ë³„ ì •í™•ë„
plt.subplot(3, 3, 1)
accuracies = [r['accuracy'] for r in models_results.values()]
plt.bar(models_results.keys(), accuracies, color='skyblue')
plt.title('ëª¨ë¸ë³„ ì „ì²´ ì •í™•ë„')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

# 2. í´ëž˜ìŠ¤ë³„ F1-score
plt.subplot(3, 3, 2)
x = np.arange(len(models_results))
width = 0.25
for i, class_name in enumerate(['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜']):
    f1_scores = [r['f1'][i] for r in models_results.values()]
    plt.bar(x + i*width, f1_scores, width, label=class_name)
plt.title('í´ëž˜ìŠ¤ë³„ F1-score')
plt.xticks(x + width, models_results.keys(), rotation=45)
plt.legend()

# 3. MAE/RMSE ë¹„êµ
plt.subplot(3, 3, 3)
mae_scores = [r['mae'] for r in models_results.values()]
rmse_scores = [r['rmse'] for r in models_results.values()]
x = np.arange(len(models_results))
plt.bar(x - 0.2, mae_scores, 0.4, label='MAE')
plt.bar(x + 0.2, rmse_scores, 0.4, label='RMSE')
plt.title('MAE/RMSE ë¹„êµ')
plt.xticks(x, models_results.keys(), rotation=45)
plt.legend()

# 4-8. ê° ëª¨ë¸ì˜ í˜¼ë™ í–‰ë ¬
for idx, (model_name, results) in enumerate(models_results.items()):
    plt.subplot(3, 3, idx + 4)
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜'],
                yticklabels=['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜'])
    plt.title(f'{model_name} í˜¼ë™ í–‰ë ¬')
    plt.ylabel('ì‹¤ì œ')
    plt.xlabel('ì˜ˆì¸¡')

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'model_comparison_full.png'), dpi=150)
plt.close()

print(f"\n\n=== ëª¨ë¸ ì €ìž¥ ì™„ë£Œ ===")
print(f"ì €ìž¥ ìœ„ì¹˜: {model_dir}/")
print("ì €ìž¥ëœ íŒŒì¼ë“¤:")
for file in os.listdir(model_dir):
    print(f"  - {file}")

# ì„¤ì • ì •ë³´ ì €ìž¥
config = {
    'sequence_length': 280,
    'prediction_horizon': 10,
    'feature_names': feature_names,
    'num_features': X.shape[1],
    'class_labels': {0: 'ì¼ë°˜(600-1200)', 1: 'ì£¼ì˜(1200-1699)', 2: 'ìœ„í—˜(1700+)'},
    'class_weights': class_weights,
    'best_model_accuracy': best_model_by_accuracy,
    'best_model_danger': best_model_by_danger_f1
}

with open(os.path.join(model_dir, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)

print("\nâœ… ì „ì²´ í•™ìŠµ ë° í‰ê°€ ì™„ë£Œ!")
print(f"ê²°ê³¼ ì´ë¯¸ì§€: {os.path.join(model_dir, 'model_comparison_full.png')}")