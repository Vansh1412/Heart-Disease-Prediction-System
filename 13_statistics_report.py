# ============================================================
#  13_statistics_report.py
#  Heart Disease Prediction — Full Statistical Analysis Report
#  Chi-square, T-tests, Mann-Whitney, Correlation significance
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, pointbiserialr

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 120})

print("=" * 70)
print("       HEART DISEASE — STATISTICAL ANALYSIS REPORT")
print("=" * 70)

df = pd.read_csv('data/heart_dataset.csv')

# Split by target
pos = df[df['Target'] == 1]
neg = df[df['Target'] == 0]

print(f"\n  Groups: Disease (n={len(pos):,})  |  Healthy (n={len(neg):,})")
print(f"  Significance threshold: α = 0.05\n")

# ════════════════════════════════════════════════════════════════════════════
#  1. T-TESTS & MANN-WHITNEY (continuous variables)
# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  CONTINUOUS VARIABLES — T-Test & Mann-Whitney U")
print("=" * 70)
print(f"  {'Variable':<28} {'Dis Mean':>9} {'Hlth Mean':>10} {'p-value':>10} {'Sig':>5} {'Effect'}")
print("  " + "-" * 70)

cont_stats = []
cont_vars = ['Age','Trestbps','Cholesterol','Max_Heart_Rate',
             'ST_Depression','Major_Vessels']

for col in cont_vars:
    d, n   = pos[col].dropna(), neg[col].dropna()
    t_stat, p_t = ttest_ind(d, n, equal_var=False)
    u_stat, p_u = mannwhitneyu(d, n, alternative='two-sided')
    p_val  = min(p_t, p_u)

    # Cohen's d effect size
    pooled_sd = np.sqrt((d.std()**2 + n.std()**2) / 2)
    cohen_d   = (d.mean() - n.mean()) / pooled_sd if pooled_sd > 0 else 0
    effect    = 'Large' if abs(cohen_d) > 0.8 else ('Medium' if abs(cohen_d) > 0.5 else 'Small')
    sig       = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))

    print(f"  {col:<28} {d.mean():>9.2f}  {n.mean():>9.2f}  {p_val:>10.4e}  {sig:>4}  {effect} (d={cohen_d:.2f})")
    cont_stats.append({'Variable': col, 'Disease_Mean': d.mean(), 'Healthy_Mean': n.mean(),
                        'p_value': p_val, 'Significance': sig, 'Cohen_d': cohen_d, 'Effect': effect})

cont_df = pd.DataFrame(cont_stats)

# ════════════════════════════════════════════════════════════════════════════
#  2. CHI-SQUARE TESTS (categorical variables)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  CATEGORICAL VARIABLES — Chi-Square Test of Independence")
print("=" * 70)
print(f"  {'Variable':<28} {'Chi2':>8} {'df':>4} {'p-value':>12} {'Sig':>5} {'Cramer V'}")
print("  " + "-" * 70)

cat_vars = ['Sex_Label','Chest_Pain_Type','Fasting_Blood_Sugar','Resting_ECG',
            'Exercise_Induced_Angina','Slope','Major_Vessels','Thalassemia',
            'Smoking_Status','Exercise_Level','BMI_Category']

cat_stats = []
for col in cat_vars:
    ct = pd.crosstab(df[col], df['Target'])
    if ct.shape[0] < 2: continue
    chi2, p, dof, _ = chi2_contingency(ct)
    n_total  = ct.sum().sum()
    cramer_v = np.sqrt(chi2 / (n_total * (min(ct.shape) - 1)))
    sig      = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    assoc    = 'Strong' if cramer_v > 0.3 else ('Moderate' if cramer_v > 0.15 else 'Weak')

    print(f"  {col:<28} {chi2:>8.2f}  {dof:>3}  {p:>12.4e}  {sig:>4}  {cramer_v:.3f} ({assoc})")
    cat_stats.append({'Variable': col, 'Chi2': chi2, 'df': dof, 'p_value': p,
                      'Significance': sig, 'Cramer_V': cramer_v, 'Association': assoc})

cat_df = pd.DataFrame(cat_stats)

# ════════════════════════════════════════════════════════════════════════════
#  3. POINT-BISERIAL CORRELATIONS (with target)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  POINT-BISERIAL CORRELATION WITH TARGET")
print("=" * 70)
pb_results = []
numeric_cols = ['Age','Sex','Chest_Pain_Type','Trestbps','Cholesterol','Fasting_Blood_Sugar',
                'Resting_ECG','Max_Heart_Rate','Exercise_Induced_Angina','ST_Depression',
                'Slope','Major_Vessels','Thalassemia']
for col in numeric_cols:
    r, p = pointbiserialr(df['Target'], df[col])
    direction = '↑ risk' if r > 0 else '↓ risk'
    print(f"  {col:<30} r = {r:+.4f}  p = {p:.4e}  ({direction})")
    pb_results.append({'Feature': col, 'Correlation': r, 'p_value': p})

pb_df = pd.DataFrame(pb_results).sort_values('Correlation')

# ════════════════════════════════════════════════════════════════════════════
#  4. GROUP STATISTICS SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  DESCRIPTIVE STATISTICS BY GROUP")
print("=" * 70)
for col in ['Age','Trestbps','Cholesterol','Max_Heart_Rate','ST_Depression']:
    print(f"\n  {col}:")
    print(f"    {'':8}  {'Mean':>8}  {'Median':>8}  {'Std':>8}  {'Min':>6}  {'Max':>6}")
    for label, sub in [('Disease', pos), ('Healthy', neg)]:
        s = sub[col]
        print(f"    {label:<8}  {s.mean():>8.2f}  {s.median():>8.2f}  "
              f"{s.std():>8.2f}  {s.min():>6.0f}  {s.max():>6.0f}")

# ════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Statistical Analysis — Heart Disease Predictors', fontsize=13, fontweight='bold')

# Effect sizes (Cohen's d)
ax = axes[0, 0]
sorted_cont = cont_df.sort_values('Cohen_d')
colors_cd = ['#F44336' if v > 0 else '#2196F3' for v in sorted_cont['Cohen_d']]
ax.barh(sorted_cont['Variable'], sorted_cont['Cohen_d'], color=colors_cd, edgecolor='white')
ax.axvline(0, color='black', lw=0.8)
ax.axvline(0.5, color='green', lw=1, linestyle='--', alpha=0.6, label='Medium effect')
ax.axvline(-0.5, color='green', lw=1, linestyle='--', alpha=0.6)
ax.set_title("Cohen's d Effect Size\n(+ve = higher in disease group)", fontweight='bold')
ax.set_xlabel("Cohen's d")
ax.legend(fontsize=8)

# Cramer's V
ax = axes[0, 1]
sorted_cat = cat_df.sort_values('Cramer_V', ascending=True)
colors_cv = ['#4CAF50' if v > 0.3 else '#FF9800' if v > 0.15 else '#90CAF9'
             for v in sorted_cat['Cramer_V']]
ax.barh(sorted_cat['Variable'].astype(str), sorted_cat['Cramer_V'],
        color=colors_cv, edgecolor='white')
ax.axvline(0.15, color='orange', lw=1.5, linestyle='--', label='Moderate (0.15)')
ax.axvline(0.30, color='green',  lw=1.5, linestyle='--', label='Strong (0.30)')
ax.set_title("Cramér's V — Categorical Association\n(Green=Strong, Orange=Moderate)",
             fontweight='bold')
ax.set_xlabel("Cramér's V")
ax.legend(fontsize=7)

# Point-biserial correlation tornado
ax = axes[1, 0]
colors_pb = ['#F44336' if r > 0 else '#2196F3' for r in pb_df['Correlation']]
ax.barh(pb_df['Feature'], pb_df['Correlation'], color=colors_pb, edgecolor='white')
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Point-Biserial Correlation with Target\n(Red=+risk, Blue=−risk)', fontweight='bold')
ax.set_xlabel('Correlation Coefficient r')

# p-value significance landscape
ax = axes[1, 1]
all_vars   = list(cont_df['Variable']) + list(cat_df['Variable'])
all_pvals  = list(cont_df['p_value']) + list(cat_df['p_value'])
log_pvals  = [-np.log10(max(p, 1e-50)) for p in all_pvals]
colors_pv  = ['#F44336' if p < 0.001 else '#FF9800' if p < 0.01 else
              '#FFC107' if p < 0.05 else '#90CAF9' for p in all_pvals]
ax.barh(all_vars, log_pvals, color=colors_pv, edgecolor='white')
ax.axvline(-np.log10(0.05),  color='orange', lw=1.5, linestyle='--', label='p=0.05')
ax.axvline(-np.log10(0.001), color='red',    lw=1.5, linestyle='--', label='p=0.001')
ax.set_title('-log₁₀(p-value) — Statistical Significance\n(Higher = more significant)',
             fontweight='bold')
ax.set_xlabel('-log₁₀(p-value)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('plots/21_statistical_analysis.png', bbox_inches='tight', dpi=150)
plt.close()
print("\n✅ Plot: Statistical analysis dashboard saved")

# ════════════════════════════════════════════════════════════════════════════
#  Save CSVs
# ════════════════════════════════════════════════════════════════════════════
cont_df.to_csv('outputs/stats_continuous.csv', index=False)
cat_df.to_csv('outputs/stats_categorical.csv', index=False)
pb_df.to_csv('outputs/stats_pointbiserial.csv', index=False)

print(f"\n💾 Stats CSVs → outputs/stats_continuous.csv / categorical / pointbiserial")
print("\n✅ Statistical analysis complete. Run 14_hospital_doctor_report.py next.")
