# ============================================================
#  03_preprocessing.py
#  Heart Disease Prediction — Data Preprocessing & Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

print("=" * 65)
print("       HEART DISEASE — DATA PREPROCESSING")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/heart_dataset.csv')
print(f"\n✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Drop Unnecessary Columns
# ════════════════════════════════════════════════════════════════════════════
drop_cols = ['Patient_ID', 'Hospital_ID', 'Doctor_ID', 'Visit_Date',
             'Age_Group', 'Sex_Label', 'BP_Category', 'Cholesterol_Category',
             'Heart_Rate_Level', 'Risk_Level']
df_ml = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"\n🗑️  Dropped {len(drop_cols)} non-ML columns")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Handle Missing Values
# ════════════════════════════════════════════════════════════════════════════
print(f"\n🔍 Missing Values Check:")
nulls = df_ml.isnull().sum()
if nulls.sum() == 0:
    print("   ✅ No missing values found!")
else:
    for col, cnt in nulls[nulls > 0].items():
        if df_ml[col].dtype == 'object' or pd.api.types.is_string_dtype(df_ml[col]):
            df_ml[col].fillna(df_ml[col].mode()[0], inplace=True)
        else:
            df_ml[col].fillna(df_ml[col].median(), inplace=True)
        print(f"   Filled {col}: {cnt} nulls → {'mode' if df_ml[col].dtype=='object' else 'median'}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Encode Categorical Columns
# ════════════════════════════════════════════════════════════════════════════
cat_cols = df_ml.select_dtypes(include='object').columns.tolist()
print(f"\n🔢 Encoding {len(cat_cols)} categorical columns: {cat_cols}")

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    encoders[col] = le
    print(f"   {col}: {list(le.classes_)}")

joblib.dump(encoders, 'models/label_encoders.pkl')
print("   💾 Label encoders saved → models/label_encoders.pkl")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Feature Engineering
# ════════════════════════════════════════════════════════════════════════════
print("\n⚙️  Feature Engineering:")

# Age-Sex interaction
df_ml['Age_Sex_Interact'] = df_ml['Age'] * df_ml['Sex']
print("   ✅ Age × Sex interaction term added")

# BP-Cholesterol risk score
df_ml['BP_Chol_Score'] = (df_ml['Trestbps'] / 100) * (df_ml['Cholesterol'] / 200)
print("   ✅ BP × Cholesterol risk score added")

# Max Heart Rate utilisation (220 - age formula)
df_ml['HR_Reserve'] = (220 - df_ml['Age']) - df_ml['Max_Heart_Rate']
print("   ✅ Heart Rate Reserve (HRR) added")

# Oldpeak × Slope interaction
df_ml['ST_Slope_Risk'] = df_ml['ST_Depression'] * (df_ml['Slope'] + 1)
print("   ✅ ST Depression × Slope risk term added")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Outlier Detection (IQR Method — clip, not drop)
# ════════════════════════════════════════════════════════════════════════════
print("\n📐 Outlier Clipping (IQR × 1.5):")
clip_cols = ['Trestbps', 'Cholesterol', 'Max_Heart_Rate', 'ST_Depression']
for col in clip_cols:
    Q1, Q3 = df_ml[col].quantile(0.25), df_ml[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    before = ((df_ml[col] < lo) | (df_ml[col] > hi)).sum()
    df_ml[col] = df_ml[col].clip(lo, hi)
    print(f"   {col}: clipped {before} outliers → [{lo:.1f}, {hi:.1f}]")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Feature / Target Split
# ════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [c for c in df_ml.columns if c != 'Target']
X = df_ml[FEATURE_COLS]
y = df_ml['Target']

print(f"\n📦 Features : {X.shape[1]} columns")
print(f"   {list(X.columns)}")
print(f"   Target  : {y.name}  |  Classes: {y.unique()}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Train / Test Split
# ════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\n✂️  Train/Test Split (80/20, stratified):")
print(f"   Train  : {X_train.shape[0]:,} rows  | Disease rate: {y_train.mean():.3f}")
print(f"   Test   : {X_test.shape[0]:,} rows   | Disease rate: {y_test.mean():.3f}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 8 — Feature Scaling
# ════════════════════════════════════════════════════════════════════════════
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, 'models/scaler.pkl')
print(f"\n📏 StandardScaler applied")
print(f"   💾 Scaler saved → models/scaler.pkl")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 9 — Save Processed Data
# ════════════════════════════════════════════════════════════════════════════
np.save('models/X_train.npy', X_train_scaled)
np.save('models/X_test.npy',  X_test_scaled)
np.save('models/y_train.npy', y_train.values)
np.save('models/y_test.npy',  y_test.values)

# Save feature names
joblib.dump(FEATURE_COLS, 'models/feature_cols.pkl')

# Save unscaled for tree-based models
joblib.dump(X_train, 'models/X_train_raw.pkl')
joblib.dump(X_test,  'models/X_test_raw.pkl')

print("\n💾 Saved processed arrays to models/:")
print("   X_train.npy, X_test.npy, y_train.npy, y_test.npy")
print("   X_train_raw.pkl, X_test_raw.pkl, feature_cols.pkl")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 10 — Summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("📊 PREPROCESSING SUMMARY")
print("=" * 65)
print(f"  Total features      : {X.shape[1]}")
print(f"  Engineered features : 4 new columns added")
print(f"  Encoding            : {len(cat_cols)} categorical columns encoded")
print(f"  Scaling             : StandardScaler (mean=0, std=1)")
print(f"  Train size          : {X_train.shape[0]:,}")
print(f"  Test size           : {X_test.shape[0]:,}")
print("\n✅ Preprocessing complete. Run 04_model_training.py next.")
