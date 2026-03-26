# ============================================================
#  10_full_pipeline.py
#  Heart Disease Prediction — Complete End-to-End Pipeline
#  Run this single file to execute everything at once.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     roc_curve, auc, classification_report)
from sklearn.pipeline        import Pipeline

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 120})
COLORS = ['#2196F3', '#F44336']

print("=" * 65)
print("  ❤️   HEART DISEASE PREDICTION — FULL PIPELINE")
print("=" * 65)
START_TIME = time.time()

# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv('data/heart_dataset.csv')
print(f"  ✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Disease rate: {df['Target'].mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Preprocessing...")

drop_cols = ['Patient_ID','Hospital_ID','Doctor_ID','Visit_Date',
             'Age_Group','Sex_Label','BP_Category','Cholesterol_Category',
             'Heart_Rate_Level','Risk_Level']
df_ml = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode categoricals
cat_cols  = df_ml.select_dtypes(include='object').columns.tolist()
encoders  = {}
for col in cat_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    encoders[col] = le
print(f"  ✅ Encoded {len(cat_cols)} categorical columns")

# Feature engineering
df_ml['Age_Sex_Interact'] = df_ml['Age'] * df_ml['Sex']
df_ml['BP_Chol_Score']    = (df_ml['Trestbps'] / 100) * (df_ml['Cholesterol'] / 200)
df_ml['HR_Reserve']       = (220 - df_ml['Age']) - df_ml['Max_Heart_Rate']
df_ml['ST_Slope_Risk']    = df_ml['ST_Depression'] * (df_ml['Slope'] + 1)
print(f"  ✅ 4 engineered features added")

# Clip outliers
for col in ['Trestbps', 'Cholesterol', 'Max_Heart_Rate', 'ST_Depression']:
    Q1, Q3 = df_ml[col].quantile(0.25), df_ml[col].quantile(0.75)
    df_ml[col] = df_ml[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))

FEATURE_COLS = [c for c in df_ml.columns if c != 'Target']
X = df_ml[FEATURE_COLS].values
y = df_ml['Target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

# Save artifacts
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders, 'models/label_encoders.pkl')
joblib.dump(FEATURE_COLS, 'models/feature_cols.pkl')
np.save('models/X_train.npy', X_train_sc)
np.save('models/X_test.npy',  X_test_sc)
np.save('models/y_train.npy', y_train)
np.save('models/y_test.npy',  y_test)
joblib.dump(pd.DataFrame(X_train, columns=FEATURE_COLS), 'models/X_train_raw.pkl')
joblib.dump(pd.DataFrame(X_test,  columns=FEATURE_COLS), 'models/X_test_raw.pkl')
print(f"  ✅ Train/Test split: {len(X_train):,} / {len(X_test):,}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — TRAIN MODELS
# ════════════════════════════════════════════════════════════════════════════
print("\n[STEP 3] Training models...")

models_cfg = {
    'Logistic Regression': (
        LogisticRegression(C=1.0, max_iter=2000, random_state=42), True),
    'K-Nearest Neighbors': (
        KNeighborsClassifier(n_neighbors=7), True),
    'Decision Tree': (
        DecisionTreeClassifier(max_depth=8, min_samples_split=20,
                               min_samples_leaf=10, random_state=42), False),
    'Random Forest': (
        RandomForestClassifier(n_estimators=200, max_depth=12,
                               random_state=42, n_jobs=-1), False),
    'Gradient Boosting': (
        GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                   max_depth=5, random_state=42), False),
    'Support Vector Machine': (
        SVC(kernel='rbf', C=1.0, probability=True, random_state=42), True),
}

results  = []
trained  = {}

print(f"\n  {'Model':<26} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'AUC':>8}")
print("  " + "-" * 72)

for name, (model, scaled) in models_cfg.items():
    X_tr = X_train_sc if scaled else X_train
    X_te = X_test_sc  if scaled else X_test
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_prob)

    results.append({'Model':name,'Accuracy':acc,'Precision':prec,
                    'Recall':rec,'F1_Score':f1,'AUC_ROC':roc})
    trained[name] = {'model': model, 'scaled': scaled, 'pred': y_pred, 'prob': y_prob}

    print(f"  {name:<26} {acc*100:>8.2f}%  {prec*100:>8.2f}%  "
          f"{rec*100:>7.2f}%  {f1*100:>6.2f}%  {roc:>8.4f}")
    joblib.dump(model, f"models/{name.replace(' ','_').lower()}.pkl")

print("  " + "-" * 72)
results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
best_model_name = results_df.iloc[0]['Model']
print(f"\n  🏆 Best Model: {best_model_name}  ({results_df.iloc[0]['Accuracy']*100:.2f}%)")
results_df.to_csv('outputs/model_comparison.csv', index=False)

# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — PLOTS
# ════════════════════════════════════════════════════════════════════════════
print("\n[STEP 4] Generating dashboard plots...")

# ── 4a. Model comparison ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Heart Disease Prediction — Model Performance', fontsize=14, fontweight='bold')

results_sorted = results_df.sort_values('Accuracy')
bar_colors = ['#F44336' if v < 0.73 else '#FF9800' if v < 0.76 else '#4CAF50'
              for v in results_sorted['Accuracy']]
axes[0].barh(results_sorted['Model'], results_sorted['Accuracy']*100,
             color=bar_colors, edgecolor='white', height=0.5)
for i, (_, row) in enumerate(results_sorted.iterrows()):
    axes[0].text(row['Accuracy']*100+0.2, i, f"{row['Accuracy']*100:.2f}%",
                 va='center', fontsize=9, fontweight='bold')
axes[0].axvline(75, color='grey', lw=1.5, linestyle='--')
axes[0].set_xlabel('Accuracy (%)')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_xlim(60, 88)

metrics = ['Accuracy','Precision','Recall','F1_Score','AUC_ROC']
x = np.arange(len(results_df))
width = 0.15
met_colors = ['#1976D2','#388E3C','#F57C00','#7B1FA2','#D32F2F']
for i, (m, c) in enumerate(zip(metrics, met_colors)):
    axes[1].bar(x+i*width, results_df[m].values*100, width,
                label=m, color=c, alpha=0.8, edgecolor='white')
axes[1].set_xticks(x+width*2)
axes[1].set_xticklabels(results_df['Model'], rotation=30, ha='right', fontsize=8)
axes[1].legend(fontsize=7)
axes[1].set_ylabel('Score (%)')
axes[1].set_title('All Metrics per Model')

plt.tight_layout()
plt.savefig('plots/A_model_comparison.png', bbox_inches='tight', dpi=150)
plt.close()

# ── 4b. Confusion matrices & ROC ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.4)
fig.suptitle('Confusion Matrices & ROC Curves', fontsize=14, fontweight='bold')

model_names_list = list(trained.keys())
roc_ax = fig.add_subplot(gs[:, 3])
line_styles = ['-','--','-.',':', '-', '--']
roc_colors  = ['#1565C0','#C62828','#2E7D32','#F57F17','#6A1B9A','#00838F']

for i, (name, info) in enumerate(trained.items()):
    row, col = divmod(i, 3)
    ax  = fig.add_subplot(gs[row, col])
    cm  = confusion_matrix(y_test, info['pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'],
                linewidths=1, annot_kws={'size':10,'weight':'bold'})
    acc = (cm[0,0]+cm[1,1])/cm.sum()
    ax.set_title(f'{name}\nAcc: {acc*100:.1f}%', fontsize=8, fontweight='bold')

    fpr, tpr, _ = roc_curve(y_test, info['prob'])
    roc_auc_val = auc(fpr, tpr)
    roc_ax.plot(fpr, tpr, lw=2, ls=line_styles[i], color=roc_colors[i],
                label=f'{name[:16]} ({roc_auc_val:.3f})')

roc_ax.plot([0,1],[0,1],'k--',lw=1)
roc_ax.set_xlabel('FPR'); roc_ax.set_ylabel('TPR')
roc_ax.set_title('ROC Curves', fontweight='bold')
roc_ax.legend(loc='lower right', fontsize=7)

plt.savefig('plots/B_confusion_roc.png', bbox_inches='tight', dpi=150)
plt.close()

# ── 4c. Feature importance ───────────────────────────────────────────────────
rf  = trained['Random Forest']['model']
gb  = trained['Gradient Boosting']['model']
rf_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
gb_imp = pd.Series(gb.feature_importances_, index=FEATURE_COLS).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Feature Importance', fontsize=13, fontweight='bold')
import matplotlib.cm as cm_mod
c1 = cm_mod.Greens(np.linspace(0.3, 0.9, len(rf_imp)))
c2 = cm_mod.Purples(np.linspace(0.3, 0.9, len(gb_imp)))
axes[0].barh(rf_imp.index, rf_imp.values*100, color=c1)
axes[0].set_title('Random Forest'); axes[0].set_xlabel('Importance (%)')
axes[1].barh(gb_imp.index, gb_imp.values*100, color=c2)
axes[1].set_title('Gradient Boosting'); axes[1].set_xlabel('Importance (%)')
plt.tight_layout()
plt.savefig('plots/C_feature_importance.png', bbox_inches='tight', dpi=150)
plt.close()

# ── 4d. Key clinical distributions ───────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Clinical Features vs Heart Disease', fontsize=13, fontweight='bold')
axes = axes.flatten()
feats_to_plot = ['Age','Trestbps','Cholesterol','Max_Heart_Rate','ST_Depression',
                 'Major_Vessels']
labels = ['No Disease', 'Heart Disease']
for i, feat in enumerate(feats_to_plot):
    for t, color, label in zip([0,1], COLORS, labels):
        data = df[df['Target']==t][feat]
        axes[i].hist(data, bins=30, alpha=0.55, color=color, label=label, edgecolor='white')
        axes[i].axvline(data.mean(), color=color, linestyle='--', lw=1.5)
    axes[i].set_title(feat, fontweight='bold')
    axes[i].legend(fontsize=8)
plt.tight_layout()
plt.savefig('plots/D_clinical_distributions.png', bbox_inches='tight', dpi=150)
plt.close()

print(f"  ✅ 4 dashboard plots saved to plots/")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 5 — CROSS VALIDATION
# ════════════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Cross-validation check (5-fold Stratified)...")
cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, (model, scaled) in models_cfg.items():
    X_cv = X_train_sc if scaled else X_train
    pipe_scores = cross_val_score(
        model.__class__(**model.get_params()), X_cv, y_train,
        cv=cv_obj, scoring='accuracy', n_jobs=-1)
    print(f"  {name:<26} CV={pipe_scores.mean()*100:.2f}% ± {pipe_scores.std()*100:.2f}%")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 6 — SAMPLE PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════
print("\n[STEP 6] Sample predictions from test set...")
best_model_info = trained[best_model_name]
X_te_best = X_test_sc if best_model_info['scaled'] else X_test

sample_idx = np.random.choice(len(X_test), 10, replace=False)
print(f"\n  Sample of 10 Test Patients ({best_model_name}):")
print(f"  {'#':<4} {'Actual':<10} {'Predicted':<12} {'P(Disease)':<12} {'Correct?'}")
print("  " + "-" * 50)
for idx in sample_idx:
    actual = y_test[idx]
    pred   = best_model_info['pred'][idx]
    prob   = best_model_info['prob'][idx]
    ok     = '✅' if actual == pred else '❌'
    print(f"  {idx:<4} {'Disease' if actual==1 else 'Healthy':<10} "
          f"{'Disease' if pred==1 else 'Healthy':<12} {prob*100:>8.1f}%    {ok}")

# ════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
total_time = time.time() - START_TIME
print("\n" + "=" * 65)
print("  📊 FINAL PIPELINE SUMMARY")
print("=" * 65)
print(f"  Dataset       : 12,000 patients, {len(FEATURE_COLS)} features")
print(f"  Train / Test  : {len(X_train):,} / {len(X_test):,}")
print(f"  Models Trained: {len(models_cfg)}")
print(f"  Best Model    : {best_model_name}")
print(f"  Best Accuracy : {results_df.iloc[0]['Accuracy']*100:.2f}%")
print(f"  Best AUC-ROC  : {results_df.iloc[0]['AUC_ROC']:.4f}")
print(f"  Total Time    : {total_time:.1f} seconds")
print(f"\n  📁 Outputs:")
print(f"     models/       → {len(list(__import__('pathlib').Path('models').glob('*.pkl')))} model files")
print(f"     outputs/      → CSV reports")
print(f"     plots/        → Charts & visualizations")
print("\n✅ PIPELINE COMPLETE!")
