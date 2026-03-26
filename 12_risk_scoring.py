# ============================================================
#  12_risk_scoring.py
#  Heart Disease Prediction — Clinical Risk Scoring Engine
#  Builds a weighted composite risk score (0–100) and tiered
#  risk bands, mimicking the Framingham / TIMI approach.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 120})

print("=" * 65)
print("       HEART DISEASE — CLINICAL RISK SCORING ENGINE")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/heart_dataset.csv')
print(f"\n✅ Dataset: {df.shape[0]:,} patients")

# ════════════════════════════════════════════════════════════════════════════
#  FRAMINGHAM-STYLE RISK SCORE  (0–100 points)
#  Each factor adds points; total → probability tier
# ════════════════════════════════════════════════════════════════════════════
def compute_risk_score(row):
    score = 0

    # Age (max 20 pts)
    if   row['Age'] < 40:  score += 0
    elif row['Age'] < 50:  score += 5
    elif row['Age'] < 55:  score += 8
    elif row['Age'] < 60:  score += 12
    elif row['Age'] < 65:  score += 16
    else:                   score += 20

    # Sex (max 6 pts)
    score += 6 if row['Sex'] == 1 else 0

    # Chest pain type (max 14 pts)
    cp_map = {0: 14, 1: 8, 2: 4, 3: 0}
    score += cp_map.get(int(row['Chest_Pain_Type']), 0)

    # Resting BP (max 10 pts)
    bp = row['Trestbps']
    if   bp < 120: score += 0
    elif bp < 130: score += 2
    elif bp < 140: score += 5
    elif bp < 160: score += 8
    else:           score += 10

    # Cholesterol (max 8 pts)
    chol = row['Cholesterol']
    if   chol < 200: score += 0
    elif chol < 240: score += 3
    elif chol < 280: score += 6
    else:             score += 8

    # Fasting Blood Sugar > 120 (max 4 pts)
    score += 4 if row['Fasting_Blood_Sugar'] == 1 else 0

    # Max Heart Rate — lower is riskier (max 10 pts)
    hr = row['Max_Heart_Rate']
    if   hr > 170: score += 0
    elif hr > 150: score += 2
    elif hr > 130: score += 5
    elif hr > 110: score += 8
    else:           score += 10

    # Exercise Induced Angina (max 8 pts)
    score += 8 if row['Exercise_Induced_Angina'] == 1 else 0

    # ST Depression (max 8 pts)
    op = row['ST_Depression']
    if   op == 0:  score += 0
    elif op < 1.0: score += 3
    elif op < 2.0: score += 6
    else:           score += 8

    # Major Vessels (max 8 pts)
    score += int(row['Major_Vessels']) * 2

    # Thalassemia (max 4 pts)
    thal_map = {3: 0, 6: 2, 7: 4}
    score += thal_map.get(int(row['Thalassemia']), 0)

    # Lifestyle adjustments
    if row.get('Smoking_Status') == 'Current Smoker': score += 5
    elif row.get('Smoking_Status') == 'Former Smoker': score += 2

    if row.get('BMI_Category') == 'Obese':      score += 4
    elif row.get('BMI_Category') == 'Overweight': score += 2

    if row.get('Exercise_Level') == 'Low':    score += 3
    elif row.get('Exercise_Level') == 'High': score -= 2

    return min(max(score, 0), 100)

def score_to_band(score):
    if score < 25:   return 'Very Low Risk',    '#4CAF50'
    elif score < 40: return 'Low Risk',         '#8BC34A'
    elif score < 55: return 'Moderate Risk',    '#FFC107'
    elif score < 70: return 'High Risk',        '#FF5722'
    else:            return 'Very High Risk',   '#F44336'

print("\n⚙️  Computing clinical risk scores for all patients...")
df['Risk_Score'] = df.apply(compute_risk_score, axis=1)
df['Risk_Band']  = df['Risk_Score'].apply(lambda s: score_to_band(s)[0])

print(f"  Score range   : {df['Risk_Score'].min()} – {df['Risk_Score'].max()}")
print(f"  Mean score    : {df['Risk_Score'].mean():.1f}")
print(f"  Disease median: {df[df['Target']==1]['Risk_Score'].median():.1f}")
print(f"  Healthy median: {df[df['Target']==0]['Risk_Score'].median():.1f}")

# ════════════════════════════════════════════════════════════════════════════
#  Evaluate: risk score vs actual disease (AUC)
# ════════════════════════════════════════════════════════════════════════════
auc_score = roc_auc_score(df['Target'], df['Risk_Score'])
print(f"\n  📊 Risk Score AUC-ROC: {auc_score:.4f}")

# Band distribution
print("\n  Risk Band Distribution:")
band_order = ['Very Low Risk','Low Risk','Moderate Risk','High Risk','Very High Risk']
for band in band_order:
    sub   = df[df['Risk_Band'] == band]
    dis   = sub['Target'].mean() * 100 if len(sub) else 0
    bar   = '█' * int(dis/3)
    print(f"    {band:<18}: {len(sub):>5} patients | Disease rate: {dis:5.1f}%  {bar}")

# ════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════

# ── Plot 1: Risk score distribution ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Clinical Risk Scoring Engine', fontsize=14, fontweight='bold')

# Score histogram by target
ax = axes[0, 0]
for t, color, label in zip([0, 1], ['#2196F3', '#F44336'], ['No Disease', 'Heart Disease']):
    ax.hist(df[df['Target']==t]['Risk_Score'], bins=40, alpha=0.55,
            color=color, label=label, edgecolor='white')
    ax.axvline(df[df['Target']==t]['Risk_Score'].mean(),
               color=color, linestyle='--', lw=2)
ax.set_title('Risk Score Distribution by Disease Status', fontweight='bold')
ax.set_xlabel('Risk Score (0–100)')
ax.set_ylabel('Patient Count')
ax.legend()

# Add band boundaries
for boundary, label in zip([25, 40, 55, 70], ['VL|L', 'L|M', 'M|H', 'H|VH']):
    ax.axvline(boundary, color='grey', linestyle=':', lw=1)
    ax.text(boundary+0.5, ax.get_ylim()[1]*0.92, label, fontsize=7, color='grey')

# Disease rate by band
ax = axes[0, 1]
band_stats = df.groupby('Risk_Band').agg(
    Count=('Target','count'), DiseaseRate=('Target','mean')).reindex(band_order)
band_colors_list = ['#4CAF50','#8BC34A','#FFC107','#FF5722','#F44336']
bars = ax.bar(band_order, band_stats['DiseaseRate']*100,
              color=band_colors_list, edgecolor='white', width=0.6)
for bar, (_, row) in zip(bars, band_stats.iterrows()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{bar.get_height():.1f}%\n(n={row['Count']:,})",
            ha='center', fontsize=8, fontweight='bold')
ax.set_title('Disease Rate by Risk Band', fontweight='bold')
ax.set_ylabel('Disease Rate (%)')
ax.tick_params(axis='x', rotation=20)
ax.set_ylim(0, 110)

# Score vs Age scatter
ax = axes[1, 0]
sc = ax.scatter(df['Age'], df['Risk_Score'],
                c=df['Target'], cmap='RdYlBu_r',
                alpha=0.25, s=12, rasterized=True)
ax.set_xlabel('Age')
ax.set_ylabel('Risk Score')
ax.set_title('Risk Score vs Age (coloured by Disease)', fontweight='bold')
plt.colorbar(sc, ax=ax, label='0=Healthy / 1=Disease')

# Risk band pie
ax = axes[1, 1]
pie_data = df['Risk_Band'].value_counts().reindex(band_order).fillna(0)
wedges, texts, autotexts = ax.pie(
    pie_data.values, labels=band_order,
    autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
    colors=band_colors_list, startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for t in autotexts: t.set_fontsize(8)
ax.set_title('Patient Distribution by Risk Band', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/19_risk_scoring.png', bbox_inches='tight', dpi=150)
plt.close()
print("\n✅ Plot: Risk scoring dashboard saved")

# ── Plot 2: Score calibration ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
bins     = np.arange(0, 105, 10)
df['score_bin'] = pd.cut(df['Risk_Score'], bins=bins, right=False)
calib    = df.groupby('score_bin', observed=True)['Target'].mean() * 100
bin_mid  = [b.mid for b in calib.index]

ax.bar([str(b) for b in calib.index], calib.values,
       color=[f'#{max(0,min(255,int(v*2.55))):02X}{max(0,min(255,int((100-v)*2.55))):02X}00'
              for v in calib.values], edgecolor='white')
ax.plot([str(b) for b in calib.index], calib.values,
        'k-o', markersize=5, linewidth=1.5)
ax.set_title('Risk Score Calibration — Disease Rate per Score Band', fontweight='bold')
ax.set_xlabel('Risk Score Range')
ax.set_ylabel('Actual Disease Rate (%)')
ax.tick_params(axis='x', rotation=45)
ax.set_ylim(0, 110)

for i, (idx, val) in enumerate(calib.items()):
    ax.text(i, val+1.5, f'{val:.0f}%', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('plots/20_score_calibration.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Plot: Score calibration saved")

# ════════════════════════════════════════════════════════════════════════════
#  Save enriched dataset
# ════════════════════════════════════════════════════════════════════════════
df.drop(columns=['score_bin'], errors='ignore', inplace=True)
df.to_csv('outputs/dataset_with_risk_scores.csv', index=False)
print(f"\n💾 Enriched dataset saved → outputs/dataset_with_risk_scores.csv")
print("\n✅ Risk scoring complete. Run 13_statistics_report.py next.")
