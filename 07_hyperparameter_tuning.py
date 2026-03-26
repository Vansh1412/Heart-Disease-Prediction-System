# ============================================================
#  07_hyperparameter_tuning.py
#  Heart Disease Prediction — GridSearchCV & RandomizedSearchCV
# ============================================================

import numpy as np
import pandas as pd
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics         import accuracy_score, f1_score, roc_auc_score

print("=" * 65)
print("       HEART DISEASE — HYPERPARAMETER TUNING")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────────────────
X_train     = np.load('models/X_train.npy')
X_test      = np.load('models/X_test.npy')
X_train_raw = joblib.load('models/X_train_raw.pkl')
X_test_raw  = joblib.load('models/X_test_raw.pkl')
y_train     = np.load('models/y_train.npy')
y_test      = np.load('models/y_test.npy')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tuning_results = []

# ════════════════════════════════════════════════════════════════════════════
#  1. Logistic Regression — GridSearchCV
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Logistic Regression — GridSearchCV")
print("-" * 55)

lr_grid = {
    'C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    'solver': ['lbfgs', 'liblinear'],
    'penalty': ['l2'],
    'max_iter': [2000]
}

t0 = time.time()
lr_search = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_grid, cv=cv, scoring='accuracy',
    n_jobs=-1, verbose=0, refit=True
)
lr_search.fit(X_train, y_train)
t1 = time.time()

best_lr     = lr_search.best_estimator_
y_pred_lr   = best_lr.predict(X_test)
y_proba_lr  = best_lr.predict_proba(X_test)[:, 1]

lr_acc  = accuracy_score(y_test, y_pred_lr)
lr_f1   = f1_score(y_test, y_pred_lr)
lr_auc  = roc_auc_score(y_test, y_proba_lr)
lr_cv   = lr_search.best_score_

print(f"  Best Params  : {lr_search.best_params_}")
print(f"  CV Score     : {lr_cv*100:.2f}%")
print(f"  Test Accuracy: {lr_acc*100:.2f}%")
print(f"  Test F1      : {lr_f1*100:.2f}%")
print(f"  AUC-ROC      : {lr_auc:.4f}")
print(f"  Time         : {t1-t0:.1f}s")

joblib.dump(best_lr, 'models/tuned_logistic_regression.pkl')
tuning_results.append({'Model':'Logistic Regression (Tuned)',
    'Best_Params': str(lr_search.best_params_),
    'CV_Score': lr_cv, 'Test_Accuracy': lr_acc, 'F1': lr_f1, 'AUC': lr_auc})

# ════════════════════════════════════════════════════════════════════════════
#  2. Random Forest — RandomizedSearchCV
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Random Forest — RandomizedSearchCV")
print("-" * 55)

rf_param_dist = {
    'n_estimators':      [100, 200, 300, 400],
    'max_depth':         [6, 8, 10, 12, 15, None],
    'min_samples_split': [5, 10, 15, 20, 30],
    'min_samples_leaf':  [2, 4, 6, 8, 10],
    'max_features':      ['sqrt', 'log2', 0.5, 0.7],
    'bootstrap':         [True, False],
}

t0 = time.time()
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_dist, n_iter=40, cv=cv, scoring='accuracy',
    random_state=42, n_jobs=-1, verbose=0, refit=True
)
rf_search.fit(X_train_raw, y_train)
t1 = time.time()

best_rf     = rf_search.best_estimator_
y_pred_rf   = best_rf.predict(X_test_raw)
y_proba_rf  = best_rf.predict_proba(X_test_raw)[:, 1]

rf_acc  = accuracy_score(y_test, y_pred_rf)
rf_f1   = f1_score(y_test, y_pred_rf)
rf_auc  = roc_auc_score(y_test, y_proba_rf)
rf_cv   = rf_search.best_score_

print(f"  Best Params  : {rf_search.best_params_}")
print(f"  CV Score     : {rf_cv*100:.2f}%")
print(f"  Test Accuracy: {rf_acc*100:.2f}%")
print(f"  Test F1      : {rf_f1*100:.2f}%")
print(f"  AUC-ROC      : {rf_auc:.4f}")
print(f"  Time         : {t1-t0:.1f}s")

joblib.dump(best_rf, 'models/tuned_random_forest.pkl')
tuning_results.append({'Model':'Random Forest (Tuned)',
    'Best_Params': str(rf_search.best_params_),
    'CV_Score': rf_cv, 'Test_Accuracy': rf_acc, 'F1': rf_f1, 'AUC': rf_auc})

# ════════════════════════════════════════════════════════════════════════════
#  3. Gradient Boosting — RandomizedSearchCV
# ════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Gradient Boosting — RandomizedSearchCV")
print("-" * 55)

gb_param_dist = {
    'n_estimators':      [100, 150, 200, 300],
    'learning_rate':     [0.03, 0.05, 0.08, 0.10, 0.15],
    'max_depth':         [3, 4, 5, 6],
    'min_samples_split': [10, 15, 20, 30],
    'min_samples_leaf':  [5, 8, 10, 15],
    'subsample':         [0.7, 0.8, 0.85, 0.9, 1.0],
    'max_features':      ['sqrt', 'log2', 0.7],
}

t0 = time.time()
gb_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_dist, n_iter=40, cv=cv, scoring='accuracy',
    random_state=42, n_jobs=-1, verbose=0, refit=True
)
gb_search.fit(X_train_raw, y_train)
t1 = time.time()

best_gb     = gb_search.best_estimator_
y_pred_gb   = best_gb.predict(X_test_raw)
y_proba_gb  = best_gb.predict_proba(X_test_raw)[:, 1]

gb_acc  = accuracy_score(y_test, y_pred_gb)
gb_f1   = f1_score(y_test, y_pred_gb)
gb_auc  = roc_auc_score(y_test, y_proba_gb)
gb_cv   = gb_search.best_score_

print(f"  Best Params  : {gb_search.best_params_}")
print(f"  CV Score     : {gb_cv*100:.2f}%")
print(f"  Test Accuracy: {gb_acc*100:.2f}%")
print(f"  Test F1      : {gb_f1*100:.2f}%")
print(f"  AUC-ROC      : {gb_auc:.4f}")
print(f"  Time         : {t1-t0:.1f}s")

joblib.dump(best_gb, 'models/tuned_gradient_boosting.pkl')
tuning_results.append({'Model':'Gradient Boosting (Tuned)',
    'Best_Params': str(gb_search.best_params_),
    'CV_Score': gb_cv, 'Test_Accuracy': gb_acc, 'F1': gb_f1, 'AUC': gb_auc})

# ════════════════════════════════════════════════════════════════════════════
#  Summary
# ════════════════════════════════════════════════════════════════════════════
tuning_df = pd.DataFrame(tuning_results)
tuning_df.to_csv('outputs/hyperparameter_tuning_results.csv', index=False)

print("\n" + "=" * 65)
print("  HYPERPARAMETER TUNING SUMMARY")
print("=" * 65)
print(tuning_df[['Model','CV_Score','Test_Accuracy','F1','AUC']].to_string(index=False))
print(f"\n💾 Results saved → outputs/hyperparameter_tuning_results.csv")
print("💾 Tuned models saved → models/tuned_*.pkl")
print("\n✅ Tuning complete. Run 08_cross_validation.py next.")
