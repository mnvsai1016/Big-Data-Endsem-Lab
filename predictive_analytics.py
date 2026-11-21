import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from mlxtend.frequent_patterns import apriori, association_rules

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.autolayout'] = True

os.makedirs("report_images", exist_ok=True)

# -------------------------------------------------------------------------
# TASK 1: DATA PREPARATION & FEATURE SELECTION
# -------------------------------------------------------------------------
print("=" * 60)
print("TASK 1: DATA PREPARATION & FEATURE SELECTION")
print("=" * 60)

csv_path = "datasets/fifa21_male2.csv"
df = pd.read_csv(csv_path, low_memory=False)
print(f"Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

def clean_currency(val):
    if pd.isna(val):
        return 0.0
    val = str(val).strip().replace('€', '')
    if 'M' in val:
        return float(val.replace('M', '')) * 1e6
    elif 'K' in val:
        return float(val.replace('K', '')) * 1e3
    try:
        return float(val)
    except:
        return 0.0

df['Value_Clean'] = df['Value'].apply(clean_currency)
df['Wage_Clean'] = df['Wage'].apply(clean_currency)
print(f"Cleaned Value (Mean: €{df['Value_Clean'].mean():,.2f}, Max: €{df['Value_Clean'].max():,.2f})")
print(f"Cleaned Wage (Mean: €{df['Wage_Clean'].mean():,.2f}, Max: €{df['Wage_Clean'].max():,.2f})")

# Map BP to Role_Group
role_map = {
    'ST': 'Forward', 'CF': 'Forward', 'RW': 'Forward', 'LW': 'Forward',
    'CAM': 'Midfielder', 'CM': 'Midfielder', 'CDM': 'Midfielder', 'RM': 'Midfielder', 'LM': 'Midfielder',
    'CB': 'Defender', 'LB': 'Defender', 'RB': 'Defender', 'LWB': 'Defender', 'RWB': 'Defender',
    'GK': 'Goalkeeper'
}
df['Role_Group'] = df['BP'].map(role_map)
print("\nRole_Group Distribution:")
print(df['Role_Group'].value_counts())

# Select technical sub-stats
technical_substats = [
    # Attacking
    'Crossing', 'Finishing', 'Heading Accuracy', 'Short Passing', 'Volleys',
    # Skill
    'Dribbling', 'Curve', 'FK Accuracy', 'Long Passing', 'Ball Control',
    # Movement
    'Acceleration', 'Sprint Speed', 'Agility', 'Reactions', 'Balance',
    # Power
    'Shot Power', 'Jumping', 'Stamina', 'Strength', 'Long Shots',
    # Mentality
    'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
    # Defending
    'Marking', 'Standing Tackle', 'Sliding Tackle',
    # Goalkeeping
    'GK Diving', 'GK Handling', 'GK Kicking', 'GK Positioning', 'GK Reflexes'
]
base_stats = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY']

# Ensure numeric
for col in technical_substats + base_stats:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

print(f"Verified {len(technical_substats)} technical sub-stats and {len(base_stats)} base stats.")

# -------------------------------------------------------------------------
# TASK 2: ASSOCIATION RULE MINING (APRIORI)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2: ASSOCIATION RULE MINING (APRIORI ALGORITHM)")
print("=" * 60)

# Discretize attributes into binary transactions
arm_df = pd.DataFrame()
# Statistical thresholds: Pace, Shooting, Passing, Dribbling, Defending, Physical
arm_df['High_Pace'] = df['PAC'] >= df['PAC'].median()
arm_df['High_Shooting'] = df['SHO'] >= df['SHO'].median()
arm_df['High_Passing'] = df['PAS'] >= df['PAS'].median()
arm_df['High_Dribbling'] = df['DRI'] >= df['DRI'].median()
arm_df['High_Defending'] = df['DEF'] >= df['DEF'].median()
arm_df['High_Physical'] = df['PHY'] >= df['PHY'].median()
arm_df['High_Value'] = df['Value_Clean'] >= df['Value_Clean'].median()
arm_df['High_Wage'] = df['Wage_Clean'] >= df['Wage_Clean'].median()

# Role groups
arm_df['Role_Forward'] = df['Role_Group'] == 'Forward'
arm_df['Role_Midfielder'] = df['Role_Group'] == 'Midfielder'
arm_df['Role_Defender'] = df['Role_Group'] == 'Defender'
arm_df['Role_GK'] = df['Role_Group'] == 'Goalkeeper'

print("Binary Itemset Column Item Frequencies:")
item_freqs = arm_df.mean()
for k, v in item_freqs.items():
    print(f"  {k:18s}: {v:.3f} ({v*100:.1f}%)")

# Grid search across Support (15%, 20%, 25%) and Confidence (70%, 80%, 85%)
support_thresholds = [0.15, 0.20, 0.25]
confidence_thresholds = [0.70, 0.80, 0.85]

arm_results_summary = []
best_rules_df = None

for sup in support_thresholds:
    freq_items = apriori(arm_df, min_support=sup, use_colnames=True)
    for conf in confidence_thresholds:
        if len(freq_items) > 0:
            rules = association_rules(freq_items, metric="confidence", min_threshold=conf)
            arm_results_summary.append({
                "min_support": sup,
                "min_confidence": conf,
                "num_frequent_itemsets": len(freq_items),
                "num_rules": len(rules)
            })
            if sup == 0.15 and conf == 0.70:
                best_rules_df = rules.copy()

print("\nARM Parameter Grid Evaluation:")
print(pd.DataFrame(arm_results_summary))

# Rank top rules by Lift and Confidence
best_rules_df['antecedents_str'] = best_rules_df['antecedents'].apply(lambda x: ", ".join(list(x)))
best_rules_df['consequents_str'] = best_rules_df['consequents'].apply(lambda x: ", ".join(list(x)))
top_rules = best_rules_df.sort_values(by=['lift', 'confidence'], ascending=[False, False]).head(10)

print("\nTop 10 Association Rules Ranked by Lift & Confidence:")
for idx, r in top_rules.reset_index(drop=True).iterrows():
    print(f"Rule {idx+1}: {{{r['antecedents_str']}}} -> {{{r['consequents_str']}}} "
          f"[Supp: {r['support']:.3f}, Conf: {r['confidence']:.3f}, Lift: {r['lift']:.3f}]")

# -------------------------------------------------------------------------
# TASK 3: CLOSED FREQUENT ITEMSETS (CFIs) & GALOIS CLOSURE
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 3: CLOSED FREQUENT ITEMSETS (GALOIS CLOSURE / ACLOSE)")
print("=" * 60)

# Implementation of Galois Closure h(X) = t^-1(t(X))
# In formal concept analysis, continuous big data is structured into multi-level / hierarchical attributes
# where specialized skill tiers logically entail foundational skill competence.
hierarchical_df = pd.DataFrame()
for stat, cutoffs in [
    ('PAC', [55, 68, 76]),
    ('SHO', [45, 60, 72]),
    ('PAS', [45, 60, 72]),
    ('DRI', [50, 64, 74]),
    ('DEF', [40, 55, 70]),
    ('PHY', [52, 65, 75])
]:
    for idx, c in enumerate(cutoffs):
        hierarchical_df[f"{stat}_T{idx+1}"] = df[stat] >= c

hierarchical_df['Role_Forward'] = df['Role_Group'] == 'Forward'
hierarchical_df['Role_Midfielder'] = df['Role_Group'] == 'Midfielder'
hierarchical_df['Role_Defender'] = df['Role_Group'] == 'Defender'

items_list = list(hierarchical_df.columns)
bit_matrix = hierarchical_df.values.astype(bool)
item_to_idx = {item: i for i, item in enumerate(items_list)}

def compute_galois_closure(itemset):
    """Computes Galois closure h(X) = t^-1(t(X))"""
    if len(itemset) == 0:
        return frozenset()
    indices = [item_to_idx[item] for item in itemset]
    tx_mask = bit_matrix[:, indices].all(axis=1)
    if not np.any(tx_mask):
        return frozenset(items_list)
    common_items_mask = bit_matrix[tx_mask, :].all(axis=0)
    return frozenset([items_list[i] for i in range(len(items_list)) if common_items_mask[i]])

cfi_comparison = []
for sup in support_thresholds:
    fi_df = apriori(hierarchical_df, min_support=sup, use_colnames=True)
    num_fi = len(fi_df)
    
    cfi_list = []
    for itemset in fi_df['itemsets']:
        h_x = compute_galois_closure(itemset)
        if h_x == itemset:
            cfi_list.append(itemset)
            
    num_cfi = len(cfi_list)
    compression = (1.0 - (num_cfi / num_fi)) * 100 if num_fi > 0 else 0.0
    
    cfi_comparison.append({
        "min_support": sup,
        "Frequent_Itemsets (FIs)": num_fi,
        "Closed_Frequent_Itemsets (CFIs)": num_cfi,
        "Lossless_Compression_%": round(compression, 2)
    })

cfi_comp_df = pd.DataFrame(cfi_comparison)
print("\nLossless Pattern Compression Table (FIs vs CFIs):")
print(cfi_comp_df)

# -------------------------------------------------------------------------
# TASK 4: DECISION TREE CLASSIFIER (PRUNING & RULES)
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 4: DECISION TREE CLASSIFIER WITH CCP PRUNING")
print("=" * 60)

feature_cols = technical_substats + base_stats
X = df[feature_cols].copy()
y = df['Role_Group'].copy()

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = list(le.classes_)
print(f"Target Classes: {class_names}")

# Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_list, bacc_list, prec_list, rec_list, f1_list = [], [], [], [], []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X.iloc[train_idx], y_encoded[train_idx])
    y_pred = clf.predict(X.iloc[val_idx])
    
    acc_list.append(accuracy_score(y_encoded[val_idx], y_pred))
    bacc_list.append(balanced_accuracy_score(y_encoded[val_idx], y_pred))
    prec_list.append(precision_score(y_encoded[val_idx], y_pred, average='macro'))
    rec_list.append(recall_score(y_encoded[val_idx], y_pred, average='macro'))
    f1_list.append(f1_score(y_encoded[val_idx], y_pred, average='macro'))

print("\nUnpruned Decision Tree 5-Fold CV Performance:")
print(f"  Accuracy:          {np.mean(acc_list)*100:.2f}% ± {np.std(acc_list)*100:.2f}%")
print(f"  Balanced Accuracy: {np.mean(bacc_list)*100:.2f}% ± {np.std(bacc_list)*100:.2f}%")
print(f"  Macro Precision:   {np.mean(prec_list)*100:.2f}% ± {np.std(prec_list)*100:.2f}%")
print(f"  Macro Recall:      {np.mean(rec_list)*100:.2f}% ± {np.std(rec_list)*100:.2f}%")
print(f"  Macro F1-Score:    {np.mean(f1_list)*100:.2f}% ± {np.std(f1_list)*100:.2f}%")

# Cost-Complexity Pruning Path
base_clf = DecisionTreeClassifier(random_state=42)
path = base_clf.cost_complexity_pruning_path(X, y_encoded)
ccp_alphas = path.ccp_alphas
sampled_alphas = np.quantile(ccp_alphas[ccp_alphas > 0], np.linspace(0.1, 0.95, 15))
alpha_scores = []

for alpha in sampled_alphas:
    fold_f1 = []
    for train_idx, val_idx in skf.split(X, y_encoded):
        pruned_clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        pruned_clf.fit(X.iloc[train_idx], y_encoded[train_idx])
        y_pred = pruned_clf.predict(X.iloc[val_idx])
        fold_f1.append(f1_score(y_encoded[val_idx], y_pred, average='macro'))
    alpha_scores.append(np.mean(fold_f1))

best_alpha = sampled_alphas[np.argmax(alpha_scores)]
print(f"\nOptimal ccp_alpha from Cost-Complexity Pruning: {best_alpha:.6f} (Macro F1: {max(alpha_scores)*100:.2f}%)")

# Train final pruned model with max_depth limit for human-interpretable splits & tree plot
final_clf = DecisionTreeClassifier(max_depth=4, ccp_alpha=best_alpha, random_state=42)
final_clf.fit(X, y_encoded)
y_all_pred = final_clf.predict(X)

final_acc = accuracy_score(y_encoded, y_all_pred)
final_bacc = balanced_accuracy_score(y_encoded, y_all_pred)
final_prec = precision_score(y_encoded, y_all_pred, average='macro')
final_rec = recall_score(y_encoded, y_all_pred, average='macro')
final_f1 = f1_score(y_encoded, y_all_pred, average='macro')

print("\nFinal Pruned Model Performance:")
print(f"  Accuracy:          {final_acc*100:.2f}%")
print(f"  Balanced Accuracy: {final_bacc*100:.2f}%")
print(f"  Macro Precision:   {final_prec*100:.2f}%")
print(f"  Macro Recall:      {final_rec*100:.2f}%")
print(f"  Macro F1-Score:    {final_f1*100:.2f}%")

# Save Figure 11: Decision Tree Splits
fig, ax = plt.subplots(figsize=(18, 9), dpi=300)
plot_tree(final_clf, feature_names=feature_cols, class_names=class_names,
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("Decision Tree Tactical Hierarchy Splits (Pruned FIFA 21 Role Classifier)", fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("report_images/fig11_decision_tree_splits.png", dpi=300)
plt.close()
print("Saved report_images/fig11_decision_tree_splits.png")

# Save Figure 12: Confusion Matrix
cm = confusion_matrix(y_encoded, y_all_pred)
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=False)
ax.set_title("Decision Tree Confusion Matrix (Role Group Prediction)", fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("report_images/fig12_confusion_matrix.png", dpi=300)
plt.close()
print("Saved report_images/fig12_confusion_matrix.png")

# Extract and print explicit If-Then rules
tree_rules_text = export_text(final_clf, feature_names=feature_cols)
print("\nExplicit Decision Tree If-Then Rules (Split Trace):")
print(tree_rules_text[:1500] + ("\n... [truncated for brevity]" if len(tree_rules_text) > 1500 else ""))

# -------------------------------------------------------------------------
# TASK 5: K-MEANS CLUSTERING & PCA TACTICAL PROJECTION
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 5: K-MEANS CLUSTERING (K=2 to 8) & PCA PROJECTION")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_range = range(2, 9)
inertias = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sample_indices = np.random.RandomState(42).choice(len(X_scaled), size=min(3000, len(X_scaled)), replace=False)
    sil = silhouette_score(X_scaled[sample_indices], labels[sample_indices])
    silhouettes.append(sil)
    print(f"K = {k}: Inertia = {km.inertia_:,.1f}, Silhouette = {sil:.4f}")

# Figure 13: Elbow & Silhouette Plot
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (K)', fontweight='bold')
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color, fontweight='bold')
line1 = ax1.plot(k_range, inertias, marker='o', linewidth=2, color=color, label='Inertia (Elbow)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color, fontweight='bold')
line2 = ax2.plot(k_range, silhouettes, marker='s', linewidth=2, linestyle='--', color=color, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')
ax1.set_title("K-Means Inertia (Elbow) & Silhouette Analysis (K=2 to 8)", fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("report_images/fig13_kmeans_elbow_silhouette.png", dpi=300)
plt.close()
print("Saved report_images/fig13_kmeans_elbow_silhouette.png")

# Optimal K = 4 Tactical Clustering
optimal_k = 4
km_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = km_opt.fit_predict(X_scaled)
df['Tactical_Cluster'] = cluster_labels

# Characterize clusters
cluster_profile = df.groupby('Tactical_Cluster')[base_stats].mean()
print("\nTactical Cluster Centroid Profiles (Averages):")
print(cluster_profile)

archetype_names = {}
for c in range(optimal_k):
    stats = cluster_profile.loc[c]
    if stats['DEF'] > 60:
        archetype_names[c] = f"Cluster {c}: Ball-Winning Defensive Specialists"
    elif stats['SHO'] > 60 and stats['PAC'] > 70:
        archetype_names[c] = f"Cluster {c}: Explosive Clinical Finishers"
    elif stats['PAS'] > 65 and stats['DRI'] > 65:
        archetype_names[c] = f"Cluster {c}: Technical Creative Playmakers"
    else:
        archetype_names[c] = f"Cluster {c}: Goalkeeping / Stopper Core"

print("\nTactical Archetype Mapping:")
for c, name in archetype_names.items():
    print(f"  {name} (Size: {(cluster_labels == c).sum()} players)")

# PCA 2D Projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.4, s=15)
centroids_pca = pca.transform(km_opt.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=180, edgecolors='black', linewidth=1.5, label='Cluster Centroids')

ax.set_xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% Variance)", fontweight='bold')
ax.set_ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% Variance)", fontweight='bold')
ax.set_title("2D PCA Projection of FIFA 21 Tactical Player Archetypes (K=4)", fontsize=12, fontweight='bold', pad=12)

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / (optimal_k - 1)), markersize=8, label=archetype_names[i]) for i in range(optimal_k)]
legend_elements.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markeredgecolor='black', markersize=10, label='Centroids'))
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("report_images/fig14_pca_clusters.png", dpi=300)
plt.close()
print("Saved report_images/fig14_pca_clusters.png")

# Compile summary JSON
summary_output = {
    "student_roll": "CS23I1016",
    "student_name": "M Naga Venkata Sai",
    "course": "CS 5010 Analytics & Systems of Big Data Practice Endsem",
    "dataset_rows": len(df),
    "arm_grid": arm_results_summary,
    "top_rules": top_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].to_dict(orient='records'),
    "cfi_compression": cfi_comparison,
    "decision_tree_metrics": {
        "accuracy": round(final_acc, 4),
        "balanced_accuracy": round(final_bacc, 4),
        "macro_precision": round(final_prec, 4),
        "macro_recall": round(final_rec, 4),
        "macro_f1": round(final_f1, 4),
        "optimal_ccp_alpha": round(best_alpha, 6),
        "rules_text": tree_rules_text
    },
    "kmeans_metrics": {
        "k_range": list(k_range),
        "inertias": [round(x, 1) for x in inertias],
        "silhouettes": [round(x, 4) for x in silhouettes],
        "archetypes": archetype_names
    }
}

with open("predictive_results_summary.json", "w") as f:
    json.dump(summary_output, f, indent=2)

print("\nSaved predictive_results_summary.json successfully!")
