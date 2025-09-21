# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 08:51:39 2025

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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

# ë°ì´í„° ë¡œë“œ (CSV íŒŒì¼ 1ê°œ ì‚¬ìš©)
print("=== ë°ì´í„° ë¡œë”© ì¤‘ ===")
# íŒŒì¼ ê²½ë¡œë¥¼ ì‹¤ì œ íŒŒì¼ë¡œ ë³€ê²½í•˜ì„¸ìš”
df = pd.read_csv('AS.csv')  # ì‹¤ì œ íŒŒì¼ëª…ìœ¼ë¡œ ë³€ê²½

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
    data['change_rate'] = data['TOTALCNT'].pct_change()
    
    X_list = []
    y_list = []
    
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
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # ë°ì´í„° ë¶„í¬ ì¶œë ¥
    print(f"\n=== ë°ì´í„° ë¶„í¬ ===")
    print(f"ì¼ë°˜(600-1200): {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"ì£¼ì˜(1200-1699): {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"ìœ„í—˜(1700+): {np.sum(y == 2)} ({np.sum(y == 2)/len(y)*100:.1f}%)")
    
    return X, y, feature_cols

# 1651-1682 ì „ì¡°ì¦ìƒ íŒ¨í„´ ë¶„ì„
def analyze_precursor_patterns(df):
    """1651-1682 êµ¬ê°„ê³¼ 1700+ ê´€ê³„ ë¶„ì„"""
    print("\n=== ì „ì¡°ì¦ìƒ íŒ¨í„´ ë¶„ì„ ===")
    
    patterns = []
    for i in range(280, len(df)-10):
        # í˜„ìž¬ ì‹œì ë¶€í„° ê³¼ê±° 280ë¶„
        window = df.iloc[i-280:i]
        
        # 10ë¶„ í›„ ê°’
        future_val = df['TOTALCNT'].iloc[i+10]
        
        # 280ë¶„ ë‚´ 1651-1682 êµ¬ê°„ ì§„ìž… ì—¬ë¶€
        max_in_window = window['TOTALCNT'].max()
        entered_danger_zone = (1651 <= max_in_window <= 1682)
        
        if future_val >= 1700:
            # ì¶”ê°€ ì¡°ê±´ë“¤ í™•ì¸
            last_m14b = window['M14AM14B'].iloc[-1]
            avg_m14b = window['M14AM14B'].mean()
            max_m14b = window['M14AM14B'].max()
            
            last_ratio = window['M14AM14B'].iloc[-1] / (window['M14AM10A'].iloc[-1] + 1)
            
            patterns.append({
                'future_value': future_val,
                'entered_zone': entered_danger_zone,
                'last_m14b': last_m14b,
                'avg_m14b': avg_m14b,
                'max_m14b': max_m14b,
                'last_ratio': last_ratio,
                'max_totalcnt': max_in_window
            })
    
    if patterns:
        pattern_df = pd.DataFrame(patterns)
        
        # íŒ¨í„´ 1: 1651-1682 ì§„ìž…
        pattern1_count = len(pattern_df[pattern_df['entered_zone']])
        print(f"\níŒ¨í„´ 1 - 1651-1682 ì§„ìž… í›„ 1700+: {pattern1_count}/{len(pattern_df)} ({pattern1_count/len(pattern_df)*100:.1f}%)")
        
        # íŒ¨í„´ 2: M14AM14B ê³ ê°’
        pattern2 = pattern_df[pattern_df['last_m14b'] > 300]
        print(f"íŒ¨í„´ 2 - M14AM14B > 300: {len(pattern2)}/{len(pattern_df)} ({len(pattern2)/len(pattern_df)*100:.1f}%)")
        
        # ë³µí•© íŒ¨í„´
        complex_pattern = pattern_df[(pattern_df['last_ratio'] > 4) & (pattern_df['max_m14b'] > 250)]
        print(f"ë³µí•© íŒ¨í„´ - ë¹„ìœ¨>4 & M14BìµœëŒ€>250: {len(complex_pattern)}/{len(pattern_df)} ({len(complex_pattern)/len(pattern_df)*100:.1f}%)")

# ë°ì´í„° ì¤€ë¹„
X, y, feature_names = create_sequences(df)
print(f"\nìƒì„±ëœ íŠ¹ì„± shape: {X.shape}")

# íŒ¨í„´ ë¶„ì„
analyze_precursor_patterns(df)

# ë°ì´í„° ì •ê·œí™” ë° ë¶„í• 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test ë¶„í• 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\ní›ˆë ¨ ë°ì´í„°: {X_train.shape}")
print(f"í…ŒìŠ¤íŠ¸ ë°ì´í„°: {X_test.shape}")

# í´ëž˜ìŠ¤ ê°€ì¤‘ì¹˜
class_weights = {
    0: 1.0,  # ì¼ë°˜
    1: 2.0,  # ì£¼ì˜
    2: 50.0  # ìœ„í—˜ (ë§¤ìš° ì¤‘ìš”!)
}

# Custom Transformers for model concepts
class AttentionTransformer(BaseEstimator, TransformerMixin):
    """TabTransformerì˜ Attention ê°œë…ì„ ê°„ë‹¨ížˆ êµ¬í˜„"""
    def __init__(self, n_components=50):
        self.n_components = n_components
        
    def fit(self, X, y=None):
        # íŠ¹ì„± ì¤‘ìš”ë„ë¥¼ í•™ìŠµ (ê°„ë‹¨í•œ ë²„ì „)
        self.feature_importance_ = np.abs(np.corrcoef(X.T, y)[:-1, -1])
        return self
        
    def transform(self, X):
        # Attention weights ì ìš©
        weighted_X = X * self.feature_importance_.reshape(1, -1)
        return weighted_X

class FeatureTokenizer(BaseEstimator, TransformerMixin):
    """FT-Transformerì˜ Feature Tokenization ê°œë…"""
    def __init__(self, n_tokens=20):
        self.n_tokens = n_tokens
        
    def fit(self, X, y=None):
        # ê° íŠ¹ì„±ì„ í† í°ìœ¼ë¡œ ë³€í™˜í•˜ëŠ” ë°©ë²• í•™ìŠµ
        self.tokenizers = []
        for i in range(X.shape[1]):
            bins = np.percentile(X[:, i], np.linspace(0, 100, self.n_tokens))
            self.tokenizers.append(bins)
        return self
        
    def transform(self, X):
        # íŠ¹ì„±ì„ í† í°í™”
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
        # ìœ ì‚¬í•œ ìƒ˜í”Œë“¤ì˜ ì •ë³´ë¥¼ í™œìš©
        from sklearn.metrics.pairwise import cosine_similarity
        
        X_enhanced = X.copy()
        similarities = cosine_similarity(X, self.X_train)
        
        for i in range(X.shape[0]):
            # ê°€ìž¥ ìœ ì‚¬í•œ ìƒ˜í”Œë“¤ ì°¾ê¸°
            top_k_idx = np.argsort(similarities[i])[-self.n_neighbors:]
            
            # ìœ ì‚¬ ìƒ˜í”Œë“¤ì˜ ë ˆì´ë¸” ì •ë³´ í™œìš©
            if self.y_train is not None:
                similar_labels = self.y_train[top_k_idx]
                danger_ratio = np.sum(similar_labels == 2) / len(similar_labels)
                X_enhanced[i, -1] = danger_ratio  # ìœ„í—˜ë„ ì¶”ê°€
                
        return X_enhanced

class AutomaticInteraction(BaseEstimator, TransformerMixin):
    """AutoIntì˜ ìžë™ íŠ¹ì„± ìƒí˜¸ìž‘ìš©"""
    def __init__(self, n_interactions=10):
        self.n_interactions = n_interactions
        
    def fit(self, X, y=None):
        # ì¤‘ìš”í•œ íŠ¹ì„± ì¡°í•© ì°¾ê¸°
        from itertools import combinations
        
        n_features = X.shape[1]
        correlations = []
        
        for i, j in combinations(range(n_features), 2):
            interaction = X[:, i] * X[:, j]
            corr = np.abs(np.corrcoef(interaction, y)[0, 1])
            correlations.append((i, j, corr))
            
        # ìƒìœ„ nê°œ ìƒí˜¸ìž‘ìš© ì„ íƒ
        correlations.sort(key=lambda x: x[2], reverse=True)
        self.top_interactions = correlations[:self.n_interactions]
        return self
        
    def transform(self, X):
        # ìƒí˜¸ìž‘ìš© íŠ¹ì„± ì¶”ê°€
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
        # Autoencoder ìŠ¤íƒ€ì¼ë¡œ í•™ìŠµ (ê°„ë‹¨í•œ ë²„ì „)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
        
    def transform(self, X):
        # ë…¸ì´ì¦ˆ ì¶”ê°€ í›„ ë³µì›í•˜ëŠ” ê³¼ì •ì„ í†µí•´ robustí•œ íŠ¹ì„± ìƒì„±
        X_corrupted = X.copy()
        mask = np.random.random(X.shape) < self.corruption_rate
        X_corrupted[mask] = np.random.normal(self.mean_[None, :], self.std_[None, :], X.shape)[mask]
        
        # ê°„ë‹¨í•œ denoising
        X_denoised = X_corrupted * 0.8 + X * 0.2
        return X_denoised

# ëª¨ë¸ ìƒì„± ë° í•™ìŠµ
print("\n=== 5ê°œ ëª¨ë¸ ì»¨ì…‰ êµ¬í˜„ ë° í•™ìŠµ ===")

# 1. TabTransformer ì»¨ì…‰
print("\n1. TabTransformer ì»¨ì…‰ (Attention + MLP)")
attention_transformer = AttentionTransformer()
X_tab = attention_transformer.fit_transform(X_train, y_train)
tab_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
tab_model.fit(X_tab, y_train)

# 2. FT-Transformer ì»¨ì…‰
print("\n2. FT-Transformer ì»¨ì…‰ (Feature Tokenization + RF)")
tokenizer = FeatureTokenizer(n_tokens=10)
X_ft = tokenizer.fit_transform(X_train, y_train)
ft_model = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
ft_model.fit(X_ft, y_train)

# 3. SAINT ì»¨ì…‰
print("\n3. SAINT ì»¨ì…‰ (Inter-sample Attention + GB)")
saint_transformer = InterSampleAttention()
X_saint = saint_transformer.fit_transform(X_train, y_train)
saint_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
saint_model.fit(X_saint, y_train)

# 4. AutoInt ì»¨ì…‰
print("\n4. AutoInt ì»¨ì…‰ (Feature Interactions + MLP)")
autoint_transformer = AutomaticInteraction()
X_autoint = autoint_transformer.fit_transform(X_train, y_train)
autoint_model = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=500, random_state=42)
autoint_model.fit(X_autoint, y_train)

# 5. VIME ì»¨ì…‰
print("\n5. VIME ì»¨ì…‰ (Self-supervised + Ensemble)")
vime_transformer = SelfSupervisedEncoder()
X_vime = vime_transformer.fit_transform(X_train)
# ì—¬ëŸ¬ ëª¨ë¸ì˜ ì•™ìƒë¸”
vime_rf = RandomForestClassifier(n_estimators=50, class_weight=class_weights, random_state=42)
vime_gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
vime_rf.fit(X_vime, y_train)
vime_gb.fit(X_vime, y_train)

# ëª¨ë¸ í‰ê°€
print("\n\n=== ëª¨ë¸ ì„±ëŠ¥ ë¹„êµ ===")

models = {
    'TabTransformerì»¨ì…‰': (tab_model, attention_transformer),
    'FT-Transformerì»¨ì…‰': (ft_model, tokenizer),
    'SAINTì»¨ì…‰': (saint_model, saint_transformer),
    'AutoIntì»¨ì…‰': (autoint_model, autoint_transformer),
    'VIMEì»¨ì…‰': (vime_rf, vime_transformer)  # RFë§Œ ì‚¬ìš©
}

results = {}

for name, (model, transformer) in models.items():
    # í…ŒìŠ¤íŠ¸ ë°ì´í„° ë³€í™˜
    if transformer:
        X_test_transformed = transformer.transform(X_test)
    else:
        X_test_transformed = X_test
    
    # ì˜ˆì¸¡
    y_pred = model.predict(X_test_transformed)
    
    # ì„±ëŠ¥ í‰ê°€
    report = classification_report(y_test, y_pred, 
                                 target_names=['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜'],
                                 output_dict=True)
    
    # 1700+ ì˜ˆì¸¡ ì„±ëŠ¥ (í´ëž˜ìŠ¤ 2)
    danger_precision = report['ìœ„í—˜']['precision'] if 'ìœ„í—˜' in report else 0
    danger_recall = report['ìœ„í—˜']['recall'] if 'ìœ„í—˜' in report else 0
    danger_f1 = report['ìœ„í—˜']['f1-score'] if 'ìœ„í—˜' in report else 0
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'danger_precision': danger_precision,
        'danger_recall': danger_recall,
        'danger_f1': danger_f1
    }
    
    print(f"\n{name}:")
    print(f"  ì „ì²´ ì •í™•ë„: {results[name]['accuracy']:.3f}")
    print(f"  1700+ ì •ë°€ë„: {danger_precision:.3f}")
    print(f"  1700+ ìž¬í˜„ìœ¨: {danger_recall:.3f}")
    print(f"  1700+ F1-score: {danger_f1:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    ì¼ë°˜  ì£¼ì˜  ìœ„í—˜")
    for i, row in enumerate(cm):
        if i < len(row):
            print(f"    {row[0]:4d} {row[1] if len(row) > 1 else 0:4d} {row[2] if len(row) > 2 else 0:4d}  <- {['ì¼ë°˜', 'ì£¼ì˜', 'ìœ„í—˜'][i]}")

# ì„±ëŠ¥ ì‹œê°í™”
plt.figure(figsize=(15, 10))

# 1. ëª¨ë¸ë³„ 1700+ ì˜ˆì¸¡ ì„±ëŠ¥
plt.subplot(2, 2, 1)
models_names = list(results.keys())
danger_recalls = [results[m]['danger_recall'] for m in models_names]
danger_precisions = [results[m]['danger_precision'] for m in models_names]
danger_f1s = [results[m]['danger_f1'] for m in models_names]

x = np.arange(len(models_names))
width = 0.25

plt.bar(x - width, danger_recalls, width, label='Recall', alpha=0.8)
plt.bar(x, danger_precisions, width, label='Precision', alpha=0.8)
plt.bar(x + width, danger_f1s, width, label='F1-score', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('1700+ Detection Performance')
plt.xticks(x, models_names, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 2. ì „ì²´ ì •í™•ë„
plt.subplot(2, 2, 2)
accuracies = [results[m]['accuracy'] for m in models_names]
plt.bar(models_names, accuracies, color='skyblue', alpha=0.8)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Overall Accuracy')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# 3. íŠ¹ì„± ì¤‘ìš”ë„ (TabTransformerì˜ attention weights)
plt.subplot(2, 2, 3)
attention_weights = attention_transformer.feature_importance_[:20]  # ìƒìœ„ 20ê°œ
plt.bar(range(len(attention_weights)), attention_weights, alpha=0.8)
plt.xlabel('Feature Index')
plt.ylabel('Attention Weight')
plt.title('Feature Importance (TabTransformer Concept)')
plt.grid(axis='y', alpha=0.3)

# 4. ëª¨ë¸ë³„ ì˜ˆì¸¡ ë¶„í¬
plt.subplot(2, 2, 4)
# ê°€ìž¥ ì¢‹ì€ ëª¨ë¸ì˜ ì˜ˆì¸¡ í™•ë¥  ë¶„í¬
best_model_name = max(results.items(), key=lambda x: x[1]['danger_f1'])[0]
best_model, best_transformer = models[best_model_name]

if hasattr(best_model, 'predict_proba'):
    X_test_best = best_transformer.transform(X_test) if best_transformer else X_test
    y_proba = best_model.predict_proba(X_test_best)
    
    if y_proba.shape[1] > 2:
        danger_proba = y_proba[:, 2]  # ìœ„í—˜ í´ëž˜ìŠ¤ í™•ë¥ 
        
        # ì‹¤ì œ ìœ„í—˜ vs ì•„ë‹Œ ê²½ìš°
        actual_danger = y_test == 2
        plt.hist(danger_proba[actual_danger], bins=20, alpha=0.5, label='ì‹¤ì œ 1700+', density=True)
        plt.hist(danger_proba[~actual_danger], bins=20, alpha=0.5, label='ì‹¤ì œ <1700', density=True)
        plt.xlabel('Predicted Probability of 1700+')
        plt.ylabel('Density')
        plt.title(f'Probability Distribution ({best_model_name})')
        plt.legend()
        plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.close()

# ìµœì¢… ë¶„ì„
print("\n\n=== ìµœì¢… ë¶„ì„ ===")

# ìµœê³  ì„±ëŠ¥ ëª¨ë¸ ì°¾ê¸°
best_model = max(results.items(), key=lambda x: x[1]['danger_f1'])[0]
print(f"1700+ ì˜ˆì¸¡ ìµœê³  ì„±ëŠ¥ ëª¨ë¸: {best_model}")
print(f"F1-score: {results[best_model]['danger_f1']:.3f}")

# ì£¼ìš” ë°œê²¬ì‚¬í•­ ì •ë¦¬
print("\n=== í•µì‹¬ ë°œê²¬ì‚¬í•­ ===")
print("1. 1651-1682 êµ¬ê°„ì´ 1700+ ê¸‰ì¦ì˜ ì „ì¡° ì‹ í˜¸ ì¤‘ í•˜ë‚˜")
print("2. M14AM14B > 300 ì¡°ê±´ë„ ì¤‘ìš”í•œ ì§€í‘œ")
print("3. ë³µí•© íŒ¨í„´ (ë¹„ìœ¨>4 & M14BìµœëŒ€>250)ì´ ê°•ë ¥í•œ ì˜ˆì¸¡ ì¸ìž")
print("4. 280ë¶„ ì‹œí€€ìŠ¤ ë°ì´í„°ê°€ 10ë¶„ í›„ ì˜ˆì¸¡ì— íš¨ê³¼ì ")

# ê²°ê³¼ íŒŒì¼ ì €ìž¥
with open('model_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== 5ê°œ ëª¨ë¸ ì»¨ì…‰ ì„±ëŠ¥ ë¹„êµ ===\n\n")
    for name, metrics in results.items():
        f.write(f"{name}:\n")
        f.write(f"  ì „ì²´ ì •í™•ë„: {metrics['accuracy']:.3f}\n")
        f.write(f"  1700+ ì •ë°€ë„: {metrics['danger_precision']:.3f}\n")
        f.write(f"  1700+ ìž¬í˜„ìœ¨: {metrics['danger_recall']:.3f}\n")
        f.write(f"  1700+ F1-score: {metrics['danger_f1']:.3f}\n\n")
    
    f.write(f"\nìµœê³  ì„±ëŠ¥ ëª¨ë¸: {best_model}\n")
    f.write(f"F1-score: {results[best_model]['danger_f1']:.3f}\n")

print("\nâœ… ë¶„ì„ ì™„ë£Œ!")
print("ê²°ê³¼ íŒŒì¼:")
print("- model_comparison.png")
print("- model_results.txt")