# ============================================================
#  11_model_stacking.py
#  Heart Disease Prediction — Voting + Stacking Ensembles
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (RandomForestClassifier, GradientBoostingClassifier,
                                     VotingClassifier, StackingClassifier)
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics         import (accuracy_score, f1_score, roc_auc_score,
                                     roc_curve, auc, confusion_matrix)
import seaborn as sns

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 120})

print("=" * 65)
print("       HEART DISEASE — ENSEMBLE & STACKING MODELS")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────────────────
X_train     = np.load('models/X_train.npy')
X_test      = np.load('models/X_test.npy')
X_train_raw = joblib.load('models/X_train_raw.pkl')
X_test_raw  = joblib.load('models/X_test_raw.pkl')
y_train     = np.load('models/y_train.npy')
y_test      = np.load('models/y_test.npy')
feature_cols = joblib.load('models/feature_cols.pkl')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Base estimators (pipelines so each handles its own scaling) ───────────────
base_estimators = [
    ('lr', Pipeline([('s', StandardScaler()),
                     ('m', LogisticRegression(C=1.0, max_iter=2000, random_state=42))])),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=12,
                                   random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                       max_depth=5, random_state=42)),
    ('knn', Pipeline([('s', StandardScaler()),
                      ('m', KNeighborsClassifier(n_neighbors=7))])),
    ('svm', Pipeline([('s', StandardScaler()),
                      ('m', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))])),
]

# ════════════════════════════════════════════════════════════════════════════
#  1. SOFT VOTING ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Soft Voting Ensemble")
print("-" * 55)

voting_soft = VotingClassifier(estimators=base_estimators, voting='soft', n_jobs=-1)
voting_soft.fit(X_train_raw, y_train)
y_pred_vs   = voting_soft.predict(X_test_raw)
y_prob_vs   = voting_soft.predict_proba(X_test_raw)[:, 1]

acc_vs  = accuracy_score(y_test, y_pred_vs)
f1_vs   = f1_score(y_test, y_pred_vs)
auc_vs  = roc_auc_score(y_test, y_prob_vs)
cv_vs   = cross_val_score(voting_soft, X_train_raw, y_train,
                          cv=cv, scoring='accuracy', n_jobs=-1).mean()

print(f"  CV Accuracy    : {cv_vs*100:.2f}%")
print(f"  Test Accuracy  : {acc_vs*100:.2f}%")
print(f"  F1 Score       : {f1_vs*100:.2f}%")
print(f"  AUC-ROC        : {auc_vs:.4f}")
joblib.dump(voting_soft, 'models/ensemble_soft_voting.pkl')

# ════════════════════════════════════════════════════════════════════════════
#  2. HARD VOTING ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Hard Voting Ensemble")
print("-" * 55)

voting_hard = VotingClassifier(estimators=base_estimators, voting='hard', n_jobs=-1)
voting_hard.fit(X_train_raw, y_train)
y_pred_vh   = voting_hard.predict(X_test_raw)

acc_vh = accuracy_score(y_test, y_pred_vh)
f1_vh  = f1_score(y_test, y_pred_vh)
cv_vh  = cross_val_score(voting_hard, X_train_raw, y_train,
                         cv=cv, scoring='accuracy', n_jobs=-1).mean()

print(f"  CV Accuracy    : {cv_vh*100:.2f}%")
print(f"  Test Accuracy  : {acc_vh*100:.2f}%")
print(f"  F1 Score       : {f1_vh*100:.2f}%")
joblib.dump(voting_hard, 'models/ensemble_hard_voting.pkl')

# ════════════════════════════════════════════════════════════════════════════
#  3. STACKING CLASSIFIER (meta-learner = Logistic Regression)
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Stacking Classifier (meta-learner: Logistic Regression)")
print("-" * 55)

stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)
stacking.fit(X_train_raw, y_train)
y_pred_st  = stacking.predict(X_test_raw)
y_prob_st  = stacking.predict_proba(X_test_raw)[:, 1]

acc_st  = accuracy_score(y_test, y_pred_st)
f1_st   = f1_score(y_test, y_pred_st)
auc_st  = roc_auc_score(y_test, y_prob_st)
cv_st   = cross_val_score(stacking, X_train_raw, y_train,
                          cv=cv, scoring='accuracy', n_jobs=-1).mean()

print(f"  CV Accuracy    : {cv_st*100:.2f}%")
print(f"  Test Accuracy  : {acc_st*100:.2f}%")
print(f"  F1 Score       : {f1_st*100:.2f}%")
print(f"  AUC-ROC        : {auc_st:.4f}")
joblib.dump(stacking, 'models/ensemble_stacking.pkl')

# ════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════
# Load individual model results for comparison
base_results = pd.read_csv('outputs/model_comparison.csv')

ensemble_results = pd.DataFrame([
    {'Model': 'Hard Voting',  'Accuracy': acc_vh, 'F1': f1_vh,  'AUC': None},
    {'Model': 'Soft Voting',  'Accuracy': acc_vs, 'F1': f1_vs,  'AUC': auc_vs},
    {'Model': 'Stacking (LR)','Accuracy': acc_st, 'F1': f1_st,  'AUC': auc_st},
])

# ── Plot 1: Accuracy comparison bar ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Ensemble Models vs Individual Models', fontsize=13, fontweight='bold')

all_models = (list(base_results['Model']) + list(ensemble_results['Model']))
all_accs   = (list(base_results['Accuracy'] * 100) +
              list(ensemble_results['Accuracy'] * 100))
colors_bar  = ['#90CAF9'] * len(base_results) + ['#EF5350', '#FF9800', '#4CAF50']
sorted_pairs = sorted(zip(all_models, all_accs, colors_bar), key=lambda x: x[1])
smod, sacc, scol = zip(*sorted_pairs)

axes[0].barh(smod, sacc, color=scol, edgecolor='white', height=0.55)
for i, (m, a) in enumerate(zip(smod, sacc)):
    axes[0].text(a + 0.15, i, f'{a:.2f}%', va='center', fontsize=8, fontweight='bold')
axes[0].axvline(80, color='red', lw=1.5, linestyle='--', label='80% threshold')
axes[0].set_xlabel('Accuracy (%)')
axes[0].set_title('Accuracy: Ensemble vs Individual')
axes[0].legend(fontsize=8)
axes[0].set_xlim(60, 87)

# ── Plot 2: ROC comparison ───────────────────────────────────────────────────
individual_models = {
    'Logistic Regression': (joblib.load('models/logistic_regression.pkl'), True),
    'Random Forest':        (joblib.load('models/random_forest.pkl'), False),
    'Gradient Boosting':    (joblib.load('models/gradient_boosting.pkl'), False),
}
for name, (model, scaled) in individual_models.items():
    X_te = X_test if scaled else X_test_raw
    prob = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[1].plot(fpr, tpr, lw=1.5, linestyle='--', alpha=0.6, label=f'{name} ({auc(fpr,tpr):.3f})')

for name, prob, ls, color in [
    ('Soft Voting',   y_prob_vs, '-',  '#FF5722'),
    ('Stacking (LR)', y_prob_st, '-',  '#4CAF50'),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[1].plot(fpr, tpr, lw=2.5, linestyle=ls, color=color,
                 label=f'{name} ({auc(fpr,tpr):.3f})')

axes[1].plot([0,1],[0,1],'k--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC: Ensemble vs Individual')
axes[1].legend(loc='lower right', fontsize=7)

plt.tight_layout()
plt.savefig('plots/17_ensemble_comparison.png', bbox_inches='tight', dpi=150)
plt.close()
print("\n✅ Plot: Ensemble comparison saved")

# ── Plot 2: Stacking confusion matrix ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Ensemble Confusion Matrices', fontsize=12, fontweight='bold')
for ax, pred, title in zip(axes,
    [y_pred_vh, y_pred_vs, y_pred_st],
    ['Hard Voting', 'Soft Voting', 'Stacking (LR)']):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease','Disease'],
                yticklabels=['No Disease','Disease'],
                linewidths=1, annot_kws={'size':12,'weight':'bold'})
    acc = (cm[0,0]+cm[1,1])/cm.sum()
    ax.set_title(f'{title}\nAcc: {acc*100:.2f}%', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('plots/18_ensemble_confusion.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Plot: Ensemble confusion matrices saved")

# ════════════════════════════════════════════════════════════════════════════
#  Save results
# ════════════════════════════════════════════════════════════════════════════
ensemble_results.to_csv('outputs/ensemble_results.csv', index=False)
print(f"\n💾 Ensemble results → outputs/ensemble_results.csv")
print("💾 Models saved → models/ensemble_*.pkl")

print("\n" + "=" * 65)
print("  FINAL ENSEMBLE SUMMARY")
print("=" * 65)
print(f"  {'Model':<22} {'Accuracy':>9} {'F1':>8} {'AUC':>10}")
print("-" * 55)
print(f"  {'Hard Voting':<22} {acc_vh*100:>8.2f}%  {f1_vh*100:>7.2f}%  {'N/A':>10}")
print(f"  {'Soft Voting':<22} {acc_vs*100:>8.2f}%  {f1_vs*100:>7.2f}%  {auc_vs:>10.4f}")
print(f"  {'Stacking (LR)':<22} {acc_st*100:>8.2f}%  {f1_st*100:>7.2f}%  {auc_st:>10.4f}")
print("\n✅ Ensemble complete. Run 12_risk_scoring.py next.")
