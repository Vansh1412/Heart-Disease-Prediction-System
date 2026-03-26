# ============================================================
#  01_data_overview.py
#  Heart Disease Prediction — Dataset Overview & Summary
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ────────────────────────────────────────────────────────────────
df = pd.read_csv('data/heart_dataset.csv')

print("=" * 65)
print("       HEART DISEASE PREDICTION — DATA OVERVIEW")
print("=" * 65)

# ── Basic Shape ──────────────────────────────────────────────────────────────
print(f"\n📋 Dataset Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"📌 Total Patients  : {df['Patient_ID'].nunique():,}")
print(f"🏥 Hospitals       : {df['Hospital_ID'].nunique()}")
print(f"👨‍⚕️ Doctors          : {df['Doctor_ID'].nunique()}")
print(f"📅 Visit Date Range: {df['Visit_Date'].min()}  →  {df['Visit_Date'].max()}")

# ── Column Info ──────────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("📌 COLUMN INFORMATION")
print("-" * 65)
print(f"{'Column':<30} {'Dtype':<12} {'Non-Null':<10} {'Unique'}")
print("-" * 65)
for col in df.columns:
    print(f"{col:<30} {str(df[col].dtype):<12} {df[col].notna().sum():<10} {df[col].nunique()}")

# ── Missing Values ───────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("🔍 MISSING VALUES")
print("-" * 65)
nulls = df.isnull().sum()
if nulls.sum() == 0:
    print("✅ No missing values found!")
else:
    print(nulls[nulls > 0])

# ── Target Distribution ──────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("🎯 TARGET VARIABLE DISTRIBUTION")
print("-" * 65)
tc = df['Target'].value_counts()
print(f"  0 - No Heart Disease : {tc.get(0, 0):,}  ({tc.get(0,0)/len(df)*100:.1f}%)")
print(f"  1 - Heart Disease    : {tc.get(1, 0):,}  ({tc.get(1,0)/len(df)*100:.1f}%)")

# ── Numeric Summary ──────────────────────────────────────────────────────────
num_cols = ['Age', 'Trestbps', 'Cholesterol', 'Max_Heart_Rate',
            'ST_Depression', 'Major_Vessels']
print("\n" + "-" * 65)
print("📊 NUMERIC COLUMNS — STATISTICAL SUMMARY")
print("-" * 65)
print(df[num_cols].describe().round(2).to_string())

# ── Categorical Distributions ────────────────────────────────────────────────
cat_cols = ['Sex_Label', 'Age_Group', 'BP_Category', 'Cholesterol_Category',
            'Risk_Level', 'Smoking_Status', 'Exercise_Level', 'BMI_Category']
print("\n" + "-" * 65)
print("🗂️  CATEGORICAL COLUMNS — VALUE COUNTS")
print("-" * 65)
for col in cat_cols:
    print(f"\n  {col}:")
    vc = df[col].value_counts()
    for val, cnt in vc.items():
        bar = '█' * int(cnt / len(df) * 30)
        print(f"    {val:<28} {cnt:>5}  ({cnt/len(df)*100:5.1f}%)  {bar}")

# ── Disease Rate by Gender ────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("💡 DISEASE RATE BY KEY GROUPS")
print("-" * 65)
for col in ['Sex_Label', 'Age_Group', 'Smoking_Status', 'BMI_Category']:
    print(f"\n  Disease Rate by {col}:")
    rate = df.groupby(col)['Target'].mean().sort_values(ascending=False)
    for grp, val in rate.items():
        bar = '█' * int(val * 30)
        print(f"    {grp:<25} {val*100:5.1f}%  {bar}")

print("\n" + "=" * 65)
print("✅  Overview complete. Run 02_eda_analysis.py next.")
print("=" * 65)
