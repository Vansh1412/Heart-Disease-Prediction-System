# ============================================================
#  08_cross_validation.py
#  Heart Disease Prediction — K-Fold & Stratified K-Fold CV
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (KFold, StratifiedKFold,
                                     cross_val_score, cross_validate,
                                     learning_curve)
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 120})

print("=" * 65)
print("       HEART DISEASE — CROSS VALIDATION ANALYSIS")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────────────────
X_train_raw = joblib.load('models/X_train_raw.pkl')
X_test_raw  = joblib.load('models/X_test_raw.pkl')
y_train     = np.load('models/y_train.npy')
y_test      = np.load('models/y_test.npy')
feature_cols = joblib.load('models/feature_cols.pkl')

X_all = np.vstack([X_train_raw, X_test_raw])
y_all = np.concatenate([y_train, y_test])
print(f"✅ Full dataset for CV: {X_all.shape}")

# ════════════════════════════════════════════════════════════════════════════
#  Pipelines (scaler + model bundled)
# ════════════════════════════════════════════════════════════════════════════
pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(C=1.0, max_iter=2000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('model', RandomForestClassifier(n_estimators=200, max_depth=12,
                                         random_state=42, n_jobs=-1))
    ]),
    'Gradient Boosting': Pipeline([
        ('model', GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                              max_depth=5, random_state=42))
    ]),
}

# ════════════════════════════════════════════════════════════════════════════
#  K-Fold vs Stratified K-Fold (5 and 10 fold)
# ════════════════════════════════════════════════════════════════════════════
cv_configs = {
    'KFold-5':           KFold(n_splits=5,  shuffle=True, random_state=42),
    'KFold-10':          KFold(n_splits=10, shuffle=True, random_state=42),
    'StratifiedKFold-5': StratifiedKFold(n_splits=5,  shuffle=True, random_state=42),
    'StratifiedKFold-10':StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
}

print("\n📊 CROSS VALIDATION RESULTS")
print("=" * 80)
cv_results = []

for model_name, pipe in pipelines.items():
    print(f"\n  🤖 {model_name}")
    print(f"  {'CV Strategy':<22} {'Mean Acc':>9} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 57)
    for cv_name, cv_obj in cv_configs.items():
        scores = cross_val_score(pipe, X_all, y_all, cv=cv_obj,
                                 scoring='accuracy', n_jobs=-1)
        print(f"  {cv_name:<22} {scores.mean()*100:>8.2f}%  "
              f"{scores.std()*100:>6.2f}%  "
              f"{scores.min()*100:>6.2f}%  "
              f"{scores.max()*100:>6.2f}%")
        cv_results.append({
            'Model': model_name, 'CV_Strategy': cv_name,
            'Mean_Accuracy': scores.mean(), 'Std': scores.std(),
            'Min': scores.min(), 'Max': scores.max(),
            'Scores': scores.tolist()
        })

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — CV Score Distribution (violin)
# ════════════════════════════════════════════════════════════════════════════
flat_rows = []
for r in cv_results:
    for s in r['Scores']:
        flat_rows.append({
            'Model': r['Model'],
            'CV_Strategy': r['CV_Strategy'],
            'Accuracy': s * 100
        })
flat_df = pd.DataFrame(flat_rows)

fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
fig.suptitle('Cross-Validation Score Distributions', fontsize=14, fontweight='bold')

model_names = list(pipelines.keys())
colors = ['#1976D2', '#388E3C', '#F57C00']

for i, (model_name, color) in enumerate(zip(model_names, colors)):
    data = flat_df[flat_df['Model'] == model_name]
    sns.violinplot(data=data, x='CV_Strategy', y='Accuracy',
                   ax=axes[i], color=color, alpha=0.7, inner='box')
    axes[i].set_title(model_name, fontweight='bold', fontsize=10)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].set_ylim(55, 90)
    axes[i].axhline(75, color='red', linestyle='--', lw=1, alpha=0.7, label='75%')
    axes[i].axhline(78, color='green', linestyle='--', lw=1, alpha=0.7, label='78%')
    if i == 0:
        axes[i].set_ylabel('Accuracy (%)')
    axes[i].legend(fontsize=7)

plt.tight_layout()
plt.savefig('plots/15_cv_distributions.png', bbox_inches='tight', dpi=150)
plt.close()
print("\n✅ Plot: CV distributions saved")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Learning Curves
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Learning Curves — Bias vs Variance Analysis', fontsize=13, fontweight='bold')

train_sizes = np.linspace(0.1, 1.0, 10)
cv_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (model_name, pipe, color) in enumerate(zip(model_names, pipelines.values(), colors)):
    t_sizes, t_scores, v_scores = learning_curve(
        pipe, X_all, y_all,
        train_sizes=train_sizes,
        cv=cv_lc, scoring='accuracy',
        n_jobs=-1, shuffle=True, random_state=42
    )
    t_mean = t_scores.mean(axis=1) * 100
    t_std  = t_scores.std(axis=1) * 100
    v_mean = v_scores.mean(axis=1) * 100
    v_std  = v_scores.std(axis=1) * 100

    axes[i].plot(t_sizes, t_mean, 'o-', color=color, label='Training Score', lw=2)
    axes[i].fill_between(t_sizes, t_mean-t_std, t_mean+t_std, alpha=0.15, color=color)
    axes[i].plot(t_sizes, v_mean, 's--', color='tomato', label='Validation Score', lw=2)
    axes[i].fill_between(t_sizes, v_mean-v_std, v_mean+v_std, alpha=0.15, color='tomato')

    axes[i].set_title(model_name, fontweight='bold')
    axes[i].set_xlabel('Training Set Size')
    if i == 0: axes[i].set_ylabel('Accuracy (%)')
    axes[i].legend(fontsize=8)
    axes[i].set_ylim(55, 100)

    gap = t_mean[-1] - v_mean[-1]
    bias_var = 'Overfitting' if gap > 5 else ('Underfitting' if v_mean[-1] < 70 else 'Well-fitted')
    axes[i].set_xlabel(f'Training Samples\n[{bias_var}  |  Gap={gap:.1f}%]')

plt.tight_layout()
plt.savefig('plots/16_learning_curves.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Plot: Learning curves saved")

# ════════════════════════════════════════════════════════════════════════════
#  Save Results
# ════════════════════════════════════════════════════════════════════════════
cv_df = pd.DataFrame([{k: v for k,v in r.items() if k != 'Scores'} for r in cv_results])
cv_df.to_csv('outputs/cross_validation_results.csv', index=False)

print(f"\n💾 CV results → outputs/cross_validation_results.csv")
print("\n✅ Cross-validation complete. Run 09_predict_new_patient.py next.")
