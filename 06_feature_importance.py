# ============================================================
#  06_feature_importance.py
#  Heart Disease Prediction — Feature Importance & SHAP Values
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 120})

print("=" * 65)
print("       HEART DISEASE — FEATURE IMPORTANCE ANALYSIS")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────────────────
feature_cols = joblib.load('models/feature_cols.pkl')
X_test_raw   = joblib.load('models/X_test_raw.pkl')
X_test_sc    = np.load('models/X_test.npy')
y_test       = np.load('models/y_test.npy')

rf_model = joblib.load('models/random_forest.pkl')
gb_model = joblib.load('models/gradient_boosting.pkl')

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Random Forest Feature Importance
# ════════════════════════════════════════════════════════════════════════════
rf_importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
rf_sorted      = rf_importances.sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
fig.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')

colors = cm.RdYlGn(np.linspace(0.3, 0.9, len(rf_sorted)))
axes[0].barh(rf_sorted.index, rf_sorted.values * 100, color=colors)
for i, (idx, val) in enumerate(rf_sorted.items()):
    axes[0].text(val*100 + 0.1, i, f'{val*100:.2f}%', va='center', fontsize=8)
axes[0].set_xlabel('Feature Importance (%)')
axes[0].set_title('Random Forest Feature Importance', fontweight='bold')
axes[0].set_xlim(0, rf_sorted.max()*110)

# Gradient Boosting
gb_importances = pd.Series(gb_model.feature_importances_, index=feature_cols)
gb_sorted      = gb_importances.sort_values(ascending=True)
colors2 = cm.PuBuGn(np.linspace(0.3, 0.9, len(gb_sorted)))
axes[1].barh(gb_sorted.index, gb_sorted.values * 100, color=colors2)
for i, (idx, val) in enumerate(gb_sorted.items()):
    axes[1].text(val*100 + 0.1, i, f'{val*100:.2f}%', va='center', fontsize=8)
axes[1].set_xlabel('Feature Importance (%)')
axes[1].set_title('Gradient Boosting Feature Importance', fontweight='bold')
axes[1].set_xlim(0, gb_sorted.max()*110)

plt.tight_layout()
plt.savefig('plots/12_feature_importance.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Plot: Feature importance saved")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — SHAP Summary (using tree explainer on RF)
# ════════════════════════════════════════════════════════════════════════════
try:
    import shap
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_raw)

    # SHAP returns list for multi-class; pick class 1 (disease)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # SHAP bar summary
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('SHAP Value Analysis (Random Forest)', fontsize=14, fontweight='bold')

    shap_mean = np.abs(sv).mean(axis=0)
    shap_df   = pd.Series(shap_mean, index=feature_cols).sort_values()
    shap_colors = cm.Reds(np.linspace(0.3, 0.9, len(shap_df)))
    axes[0].barh(shap_df.index, shap_df.values, color=shap_colors)
    axes[0].set_xlabel('Mean |SHAP Value|')
    axes[0].set_title('Feature Importance (SHAP)', fontweight='bold')

    # SHAP beeswarm (manual)
    top_n = 10
    top_feats = shap_df.sort_values(ascending=False).head(top_n).index.tolist()[::-1]
    top_idx   = [list(feature_cols).index(f) for f in top_feats]

    for i, (feat, idx) in enumerate(zip(top_feats, top_idx)):
        vals     = sv[:, idx]
        feat_val = X_test_raw.iloc[:, idx] if hasattr(X_test_raw, 'iloc') else X_test_raw[:, idx]
        sc = axes[1].scatter(vals, [i]*len(vals),
                             c=feat_val, cmap='RdYlBu_r', alpha=0.3, s=15)

    axes[1].set_yticks(range(len(top_feats)))
    axes[1].set_yticklabels(top_feats, fontsize=9)
    axes[1].set_xlabel('SHAP Value (impact on model output)')
    axes[1].set_title('SHAP Beeswarm Plot (Top 10 Features)', fontweight='bold')
    axes[1].axvline(0, color='black', lw=0.8, linestyle='--')
    plt.colorbar(sc, ax=axes[1], label='Feature Value (low → high)')

    plt.tight_layout()
    plt.savefig('plots/13_shap_analysis.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✅ Plot: SHAP analysis saved")

except ImportError:
    print("⚠️  SHAP not installed. Skipping SHAP plot.")
    print("    Install with: pip install shap")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — Logistic Regression Coefficients
# ════════════════════════════════════════════════════════════════════════════
try:
    lr_model = joblib.load('models/logistic_regression.pkl')
    coefs    = pd.Series(lr_model.coef_[0], index=feature_cols).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = ['#F44336' if v > 0 else '#2196F3' for v in coefs.values]
    ax.barh(coefs.index, coefs.values, color=colors, edgecolor='white')
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Coefficient Value (positive = ↑ disease risk)')
    ax.set_title('Logistic Regression Coefficients\n(Red = Increases Risk, Blue = Decreases Risk)',
                 fontweight='bold', fontsize=11)
    for i, (idx, val) in enumerate(coefs.items()):
        ax.text(val + (0.01 if val >= 0 else -0.01), i,
                f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)
    plt.tight_layout()
    plt.savefig('plots/14_lr_coefficients.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("✅ Plot: LR coefficients saved")
except Exception as e:
    print(f"⚠️  LR coefficients plot skipped: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  Save Feature Importance Summary CSV
# ════════════════════════════════════════════════════════════════════════════
summary = pd.DataFrame({
    'Feature': feature_cols,
    'RF_Importance_%': rf_importances.values * 100,
    'GB_Importance_%': gb_importances.values * 100,
})
summary['Avg_Importance_%'] = (summary['RF_Importance_%'] + summary['GB_Importance_%']) / 2
summary = summary.sort_values('Avg_Importance_%', ascending=False)
summary.to_csv('outputs/feature_importance_summary.csv', index=False)

print("\n📊 TOP 10 MOST IMPORTANT FEATURES:")
print("-" * 50)
for _, row in summary.head(10).iterrows():
    bar = '█' * int(row['Avg_Importance_%'] * 2)
    print(f"  {row['Feature']:<30} {row['Avg_Importance_%']:5.2f}%  {bar}")

print(f"\n💾 Feature importance summary → outputs/feature_importance_summary.csv")
print("\n✅ Feature importance complete. Run 07_hyperparameter_tuning.py next.")
