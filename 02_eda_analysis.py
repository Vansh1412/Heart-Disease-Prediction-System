# ============================================================
#  02_eda_analysis.py
#  Heart Disease Prediction — Exploratory Data Analysis (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/heart_dataset.csv')

# Global style
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.dpi': 120
})
COLORS = ['#2196F3', '#F44336']
PALETTE = sns.color_palette(['#2196F3', '#F44336'])

print("🔍 Running EDA — generating plots to plots/ folder...")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Distribution of Key Numeric Features
# ════════════════════════════════════════════════════════════════════════════
num_feats = ['Age', 'Trestbps', 'Cholesterol', 'Max_Heart_Rate', 'ST_Depression']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribution of Key Clinical Features', fontsize=15, fontweight='bold', y=1.01)
axes = axes.flatten()

for i, feat in enumerate(num_feats):
    ax = axes[i]
    for t, color, label in zip([0, 1], COLORS, ['No Disease', 'Heart Disease']):
        subset = df[df['Target'] == t][feat]
        ax.hist(subset, bins=35, alpha=0.55, color=color, label=label, edgecolor='white')
        ax.axvline(subset.mean(), color=color, linestyle='--', linewidth=1.5)
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig('plots/01_numeric_distributions.png', bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ Plot 1: Numeric distributions saved")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Categorical Feature vs Target (stacked bar)
# ════════════════════════════════════════════════════════════════════════════
cat_feats = ['Sex_Label', 'Age_Group', 'Chest_Pain_Type',
             'BP_Category', 'Risk_Level', 'Smoking_Status',
             'Exercise_Level', 'BMI_Category']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Heart Disease Rate by Categorical Features', fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, feat in enumerate(cat_feats):
    ax = axes[i]
    rates = df.groupby(feat)['Target'].mean().sort_values() * 100
    bars = ax.barh(rates.index, rates.values,
                   color=[COLORS[1] if v > 50 else COLORS[0] for v in rates.values],
                   edgecolor='white', height=0.6)
    for bar, val in zip(bars, rates.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8)
    ax.set_title(feat, fontweight='bold', fontsize=9)
    ax.set_xlabel('Disease Rate (%)')
    ax.set_xlim(0, 105)

plt.tight_layout()
plt.savefig('plots/02_categorical_vs_target.png', bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ Plot 2: Categorical vs target saved")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — Correlation Heatmap
# ════════════════════════════════════════════════════════════════════════════
ml_cols = ['Age', 'Sex', 'Chest_Pain_Type', 'Trestbps', 'Cholesterol',
           'Fasting_Blood_Sugar', 'Resting_ECG', 'Max_Heart_Rate',
           'Exercise_Induced_Angina', 'ST_Depression', 'Slope',
           'Major_Vessels', 'Thalassemia', 'Target']

fig, ax = plt.subplots(figsize=(13, 10))
corr = df[ml_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 8},
            square=True, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('plots/03_correlation_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ Plot 3: Correlation heatmap saved")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 4 — Age Distribution by Disease Status
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Age, Gender & Risk Level Analysis', fontsize=13, fontweight='bold')

# Age KDE
ax = axes[0]
for t, color, label in zip([0, 1], COLORS, ['No Disease', 'Heart Disease']):
    subset = df[df['Target'] == t]['Age']
    ax.hist(subset, bins=30, alpha=0.5, color=color, label=label, density=True)
ax.set_title('Age Distribution by Disease Status')
ax.set_xlabel('Age')
ax.legend()

# Gender Pie
ax = axes[1]
gender_disease = df[df['Target'] == 1]['Sex_Label'].value_counts()
ax.pie(gender_disease.values, labels=gender_disease.index,
       autopct='%1.1f%%', colors=['#42A5F5', '#EF5350'],
       startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title('Gender Split (Heart Disease Patients)')

# Risk Level Bar
ax = axes[2]
rl = df['Risk_Level'].value_counts()
bars = ax.bar(rl.index, rl.values,
              color=['#4CAF50', '#FF9800', '#F44336'], edgecolor='white')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{bar.get_height():,}', ha='center', fontsize=9)
ax.set_title('Patient Count by Risk Level')
ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('plots/04_age_gender_risk.png', bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ Plot 4: Age/Gender/Risk analysis saved")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 5 — Boxplots: Clinical Values vs Target
# ════════════════════════════════════════════════════════════════════════════
box_feats = ['Age', 'Max_Heart_Rate', 'ST_Depression', 'Cholesterol',
             'Trestbps', 'Major_Vessels']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Clinical Features vs Heart Disease (Boxplots)', fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, feat in enumerate(box_feats):
    ax = axes[i]
    data_0 = df[df['Target'] == 0][feat]
    data_1 = df[df['Target'] == 1][feat]
    bp = ax.boxplot([data_0, data_1], labels=['No Disease', 'Heart Disease'],
                    patch_artist=True, notch=True,
                    medianprops={'color': 'black', 'linewidth': 2})
    bp['boxes'][0].set_facecolor('#90CAF9')
    bp['boxes'][1].set_facecolor('#EF9A9A')
    ax.set_title(feat, fontweight='bold')
    ax.set_ylabel(feat)

plt.tight_layout()
plt.savefig('plots/05_boxplots_vs_target.png', bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ Plot 5: Boxplots saved")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 6 — Patients Over Time (Year-wise Trend)
# ════════════════════════════════════════════════════════════════════════════
df['Visit_Date'] = pd.to_datetime(df['Visit_Date'])
df['Year'] = df['Visit_Date'].dt.year
df['Month'] = df['Visit_Date'].dt.month

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Temporal Analysis of Patient Visits', fontsize=13, fontweight='bold')

# Year-wise
year_counts = df.groupby(['Year', 'Target']).size().unstack(fill_value=0)
year_counts.plot(kind='bar', ax=axes[0], color=COLORS, edgecolor='white')
axes[0].set_title('Patients per Year by Disease Status')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Count')
axes[0].legend(['No Disease', 'Heart Disease'])
axes[0].tick_params(axis='x', rotation=0)

# Month-wise
month_disease = df.groupby('Month')['Target'].mean() * 100
axes[1].plot(month_disease.index, month_disease.values, 'o-',
             color='#F44336', linewidth=2.5, markersize=7)
axes[1].fill_between(month_disease.index, month_disease.values,
                     alpha=0.15, color='#F44336')
axes[1].set_title('Disease Rate (%) by Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Disease Rate (%)')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec'])

plt.tight_layout()
plt.savefig('plots/06_temporal_analysis.png', bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ Plot 6: Temporal analysis saved")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 7 — Hospital & Doctor Performance
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Hospital & Doctor Analysis', fontsize=13, fontweight='bold')

# Hospital patient count
hosp_cnt = df['Hospital_ID'].value_counts().sort_values()
colors_hosp = plt.cm.Blues(np.linspace(0.4, 0.9, len(hosp_cnt)))
axes[0].barh(hosp_cnt.index, hosp_cnt.values, color=colors_hosp)
for i, (idx, val) in enumerate(hosp_cnt.items()):
    axes[0].text(val + 20, i, f'{val:,}', va='center', fontsize=8)
axes[0].set_title('Patient Load per Hospital')
axes[0].set_xlabel('Number of Patients')

# Doctor patient count
doc_cnt = df['Doctor_ID'].value_counts().sort_values(ascending=False).head(15)
colors_doc = plt.cm.Oranges(np.linspace(0.4, 0.9, len(doc_cnt)))
axes[1].bar(doc_cnt.index, doc_cnt.values, color=colors_doc, edgecolor='white')
axes[1].set_title('Patient Load per Doctor (Top 15)')
axes[1].set_xlabel('Doctor ID')
axes[1].set_ylabel('Number of Patients')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/07_hospital_doctor_analysis.png', bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ Plot 7: Hospital/Doctor analysis saved")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 8 — Lifestyle Impact on Heart Disease
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Lifestyle Factors vs Heart Disease Risk', fontsize=13, fontweight='bold')
axes = axes.flatten()

lifestyle_cols = ['Smoking_Status', 'Alcohol_Consumption', 'Exercise_Level', 'BMI_Category']
for i, col in enumerate(lifestyle_cols):
    ax = axes[i]
    ct = pd.crosstab(df[col], df['Target'], normalize='index') * 100
    ct.columns = ['No Disease', 'Heart Disease']
    ct.plot(kind='bar', ax=ax, color=COLORS, edgecolor='white', width=0.6)
    ax.set_title(f'{col} vs Disease Rate', fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=25)
    ax.set_ylim(0, 110)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.0f}%', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig('plots/08_lifestyle_vs_disease.png', bbox_inches='tight', dpi=150)
plt.close()
print("  ✅ Plot 8: Lifestyle analysis saved")

print("\n✅ EDA COMPLETE — All 8 plots saved to plots/ folder")
print("   Run 03_preprocessing.py next.")
