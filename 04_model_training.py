# ============================================================
#  04_model_training.py
#  Heart Disease Prediction — Train & Compare 6 ML Models
# ============================================================

import numpy as np
import pandas as pd
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score)

print("=" * 70)
print("       HEART DISEASE — MODEL TRAINING (6 ALGORITHMS)")
print("=" * 70)

# ── Load preprocessed data ───────────────────────────────────────────────────
X_train = np.load('models/X_train.npy')
X_test  = np.load('models/X_test.npy')
y_train = np.load('models/y_train.npy')
y_test  = np.load('models/y_test.npy')
X_train_raw = joblib.load('models/X_train_raw.pkl')
X_test_raw  = joblib.load('models/X_test_raw.pkl')

print(f"\n✅ Data loaded: {X_train.shape[0]:,} train | {X_test.shape[0]:,} test")

# ════════════════════════════════════════════════════════════════════════════
#  Define Models
# ════════════════════════════════════════════════════════════════════════════
models = {
    'Logistic Regression': {
        'model': LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42),
        'uses_scaled': True
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(n_neighbors=7, metric='euclidean'),
        'uses_scaled': True
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(max_depth=8, min_samples_split=20,
                                        min_samples_leaf=10, random_state=42),
        'uses_scaled': False
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=200, max_depth=12,
                                        min_samples_split=10, min_samples_leaf=5,
                                        random_state=42, n_jobs=-1),
        'uses_scaled': False
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                            max_depth=5, min_samples_split=15,
                                            subsample=0.85, random_state=42),
        'uses_scaled': False
    },
    'Support Vector Machine': {
        'model': SVC(kernel='rbf', C=1.0, gamma='scale',
                     probability=True, random_state=42),
        'uses_scaled': True
    },
}

# ════════════════════════════════════════════════════════════════════════════
#  Train & Evaluate Each Model
# ════════════════════════════════════════════════════════════════════════════
results = []

print(f"\n{'Model':<26} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'AUC-ROC':>9} {'Time':>7}")
print("-" * 80)

for name, info in models.items():
    model     = info['model']
    X_tr      = X_train if info['uses_scaled'] else X_train_raw
    X_te      = X_test  if info['uses_scaled'] else X_test_raw

    t0 = time.time()
    model.fit(X_tr, y_train)
    elapsed = time.time() - t0

    y_pred      = model.predict(X_te)
    y_proba     = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    results.append({
        'Model': name, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1_Score': f1, 'AUC_ROC': auc, 'Train_Time_s': round(elapsed, 2)
    })

    print(f"{name:<26} {acc*100:>8.2f}%  {prec*100:>8.2f}%  {rec*100:>7.2f}%  "
          f"{f1*100:>6.2f}%  {auc:>9.4f}  {elapsed:>5.1f}s")

    # Save each trained model
    joblib.dump(model, f'models/{name.replace(" ","_").lower()}.pkl')

print("-" * 80)

# ════════════════════════════════════════════════════════════════════════════
#  Summary Table & Best Model
# ════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
results_df.to_csv('outputs/model_comparison.csv', index=False)

best = results_df.iloc[0]
print(f"\n🏆 BEST MODEL   : {best['Model']}")
print(f"   Accuracy     : {best['Accuracy']*100:.2f}%")
print(f"   Precision    : {best['Precision']*100:.2f}%")
print(f"   Recall       : {best['Recall']*100:.2f}%")
print(f"   F1 Score     : {best['F1_Score']*100:.2f}%")
print(f"   AUC-ROC      : {best['AUC_ROC']:.4f}")

# Save best model name
joblib.dump(best['Model'], 'models/best_model_name.pkl')

print("\n" + "=" * 70)
print("📊 FULL COMPARISON TABLE (sorted by Accuracy)")
print("=" * 70)
print(results_df.to_string(index=False))

print(f"\n💾 All models saved to models/")
print(f"💾 Comparison CSV → outputs/model_comparison.csv")
print("\n✅ Training complete. Run 05_model_evaluation.py next.")
