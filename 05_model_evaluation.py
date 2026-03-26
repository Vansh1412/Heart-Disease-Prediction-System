# ============================================================
#  05_model_evaluation.py
#  Heart Disease Prediction — Detailed Evaluation & Visualizations
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve,
                              ConfusionMatrixDisplay)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 120})

print("=" * 65)
print("       HEART DISEASE — MODEL EVALUATION")
print("=" * 65)

# ── Load data & models ───────────────────────────────────────────────────────
X_test  = np.load('models/X_test.npy')
X_test_raw = joblib.load('models/X_test_raw.pkl')
y_test  = np.load('models/y_test.npy')

model_files = {
    'Logistic Regression':    ('models/logistic_regression.pkl', True),
    'K-Nearest Neighbors':    ('models/k-nearest_neighbors.pkl', True),
    'Decision Tree':          ('models/decision_tree.pkl', False),
    'Random Forest':          ('models/random_forest.pkl', False),
    'Gradient Boosting':      ('models/gradient_boosting.pkl', False),
    'Support Vector Machine': ('models/support_vector_machine.pkl', True),
}

models = {}
for name, (path, scaled) in model_files.items():
    try:
        models[name] = {'model': joblib.load(path), 'scaled': scaled}
    except FileNotFoundError:
        print(f"  ⚠️  {name} not found, skipping")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Confusion Matrices (all models)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, (name, info) in enumerate(models.items()):
    X_te = X_test if info['scaled'] else X_test_raw
    y_pred = info['model'].predict(X_te)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No Disease','Disease'],
                yticklabels=['No Disease','Disease'],
                linewidths=1, linecolor='white',
                annot_kws={'size': 11, 'weight': 'bold'})
    acc = (cm[0,0]+cm[1,1])/cm.sum()
    axes[i].set_title(f'{name}\nAccuracy: {acc*100:.2f}%', fontweight='bold', fontsize=9)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('plots/09_confusion_matrices.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Plot: Confusion matrices saved")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — ROC Curves (all models on one plot)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

line_styles = ['-', '--', '-.', ':', '-', '--']
colors = ['#1565C0', '#C62828', '#2E7D32', '#F57F17', '#6A1B9A', '#00838F']

for (name, info), ls, color in zip(models.items(), line_styles, colors):
    X_te  = X_test if info['scaled'] else X_test_raw
    proba = info['model'].predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, lw=2, ls=ls, color=color,
                 label=f'{name} (AUC = {roc_auc:.3f})')

axes[0].plot([0,1],[0,1], 'k--', lw=1, label='Random Classifier')
axes[0].fill_between([0,1],[0,1], alpha=0.05, color='grey')
axes[0].set_xlabel('False Positive Rate', fontsize=11)
axes[0].set_ylabel('True Positive Rate', fontsize=11)
axes[0].set_title('ROC Curves — All Models', fontweight='bold', fontsize=12)
axes[0].legend(loc='lower right', fontsize=8)
axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1.02])

# Precision-Recall Curves
for (name, info), ls, color in zip(models.items(), line_styles, colors):
    X_te  = X_test if info['scaled'] else X_test_raw
    proba = info['model'].predict_proba(X_te)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, proba)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, lw=2, ls=ls, color=color,
                 label=f'{name} (AUC = {pr_auc:.3f})')

axes[1].axhline(y=y_test.mean(), color='k', linestyle='--', lw=1, label='No Skill')
axes[1].set_xlabel('Recall', fontsize=11)
axes[1].set_ylabel('Precision', fontsize=11)
axes[1].set_title('Precision-Recall Curves', fontweight='bold', fontsize=12)
axes[1].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('plots/10_roc_pr_curves.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Plot: ROC & Precision-Recall curves saved")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — Model Comparison Bar Chart
# ════════════════════════════════════════════════════════════════════════════
results_df = pd.read_csv('outputs/model_comparison.csv')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']
x = np.arange(len(results_df))
width = 0.15
metric_colors = ['#1976D2','#388E3C','#F57C00','#7B1FA2','#D32F2F']

for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
    axes[0].bar(x + i*width, results_df[metric]*100, width,
                label=metric, color=color, alpha=0.85, edgecolor='white')

axes[0].set_xlabel('Model')
axes[0].set_ylabel('Score (%)')
axes[0].set_title('All Metrics per Model')
axes[0].set_xticks(x + width*2)
axes[0].set_xticklabels(results_df['Model'], rotation=30, ha='right', fontsize=8)
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 110)

# Accuracy comparison horizontal bar
results_sorted = results_df.sort_values('Accuracy')
bar_colors = ['#EF9A9A' if v < 0.74 else '#A5D6A7' if v > 0.76 else '#90CAF9'
              for v in results_sorted['Accuracy']]
axes[1].barh(results_sorted['Model'], results_sorted['Accuracy']*100,
             color=bar_colors, edgecolor='white', height=0.5)
for i, (_, row) in enumerate(results_sorted.iterrows()):
    axes[1].text(row['Accuracy']*100 + 0.2, i,
                 f"{row['Accuracy']*100:.2f}%", va='center', fontsize=9, fontweight='bold')
axes[1].axvline(75, color='#F44336', linestyle='--', linewidth=1.5, label='75% threshold')
axes[1].axvline(78, color='#4CAF50', linestyle='--', linewidth=1.5, label='78% threshold')
axes[1].set_xlabel('Accuracy (%)')
axes[1].set_title('Accuracy Comparison')
axes[1].legend(fontsize=9)
axes[1].set_xlim(60, 90)

plt.tight_layout()
plt.savefig('plots/11_model_comparison.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Plot: Model comparison chart saved")

# ════════════════════════════════════════════════════════════════════════════
#  Detailed Classification Reports
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  DETAILED CLASSIFICATION REPORTS")
print("=" * 65)
with open('outputs/classification_reports.txt', 'w') as f:
    for name, info in models.items():
        X_te   = X_test if info['scaled'] else X_test_raw
        y_pred = info['model'].predict(X_te)
        report = classification_report(y_test, y_pred,
                    target_names=['No Disease', 'Heart Disease'])
        block  = f"\n{'='*55}\n  {name}\n{'='*55}\n{report}"
        print(block)
        f.write(block)

print("\n💾 Classification reports → outputs/classification_reports.txt")
print("\n✅ Evaluation complete. Run 06_feature_importance.py next.")
