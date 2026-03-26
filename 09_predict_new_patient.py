# ============================================================
#  09_predict_new_patient.py
#  Heart Disease Prediction — Single Patient Prediction Tool
# ============================================================

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Load all saved artifacts ─────────────────────────────────────────────────
scaler       = joblib.load('models/scaler.pkl')
encoders     = joblib.load('models/label_encoders.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

rf_model = joblib.load('models/random_forest.pkl')
gb_model = joblib.load('models/gradient_boosting.pkl')

try:
    rf_tuned = joblib.load('models/tuned_random_forest.pkl')
    gb_tuned = joblib.load('models/tuned_gradient_boosting.pkl')
    HAS_TUNED = True
except:
    HAS_TUNED = False

# ════════════════════════════════════════════════════════════════════════════
#  Patient Input Helper
# ════════════════════════════════════════════════════════════════════════════
def print_risk_banner(prob, prediction):
    print("\n" + "█" * 55)
    if prediction == 1:
        if prob > 0.80:
            print("█  🔴  VERY HIGH RISK — IMMEDIATE CARDIAC EVALUATION  █")
        else:
            print("█  🟠  HIGH RISK — HEART DISEASE LIKELY DETECTED      █")
    else:
        if prob < 0.20:
            print("█  🟢  LOW RISK  — HEART DISEASE UNLIKELY              █")
        else:
            print("█  🟡  MODERATE RISK — MONITOR CLOSELY                 █")
    print("█" * 55)

def preprocess_patient(patient_dict):
    """
    Convert raw patient input into a feature vector for prediction.
    """
    row = {
        'Age':                    patient_dict['age'],
        'Sex':                    patient_dict['sex'],
        'Chest_Pain_Type':        patient_dict['cp'],
        'Trestbps':               patient_dict['trestbps'],
        'Cholesterol':            patient_dict['chol'],
        'Fasting_Blood_Sugar':    patient_dict['fbs'],
        'Resting_ECG':            patient_dict['restecg'],
        'Max_Heart_Rate':         patient_dict['thalachh'],
        'Exercise_Induced_Angina':patient_dict['exang'],
        'ST_Depression':          patient_dict['oldpeak'],
        'Slope':                  patient_dict['slope'],
        'Major_Vessels':          patient_dict['ca'],
        'Thalassemia':            patient_dict['thal'],
    }

    # Encode lifestyle categorical fields if present
    for cat_col, raw_key in [('Smoking_Status', 'smoking'),
                              ('Alcohol_Consumption', 'alcohol'),
                              ('Exercise_Level', 'exercise'),
                              ('BMI_Category', 'bmi')]:
        if raw_key in patient_dict and cat_col in encoders:
            le  = encoders[cat_col]
            val = patient_dict[raw_key]
            if val in le.classes_:
                row[cat_col] = le.transform([val])[0]
            else:
                row[cat_col] = 0

    # Feature Engineering (must match 03_preprocessing.py)
    row['Age_Sex_Interact'] = row['Age'] * row['Sex']
    row['BP_Chol_Score']    = (row['Trestbps'] / 100) * (row['Cholesterol'] / 200)
    row['HR_Reserve']       = (220 - row['Age']) - row['Max_Heart_Rate']
    row['ST_Slope_Risk']    = row['ST_Depression'] * (row['Slope'] + 1)

    # Align to trained feature columns
    df_patient = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df_patient.columns:
            df_patient[col] = 0
    df_patient = df_patient[feature_cols]

    X_raw    = df_patient.copy()
    X_scaled = scaler.transform(df_patient)
    return X_raw, X_scaled

def predict_patient(patient_dict, verbose=True):
    """
    Run prediction for one patient and return results dict.
    """
    X_raw, X_scaled = preprocess_patient(patient_dict)

    # Collect predictions from all available models
    model_results = []
    for name, model, use_scaled in [
        ('Random Forest',          rf_model, False),
        ('Gradient Boosting',      gb_model, False),
    ] + ([('Tuned RF', rf_tuned, False), ('Tuned GB', gb_tuned, False)] if HAS_TUNED else []):
        X_in   = X_scaled if use_scaled else X_raw
        pred   = model.predict(X_in)[0]
        proba  = model.predict_proba(X_in)[0]
        model_results.append({'name': name, 'pred': pred,
                               'prob_disease': proba[1], 'prob_healthy': proba[0]})

    # Ensemble (majority vote + average probability)
    avg_prob   = np.mean([r['prob_disease'] for r in model_results])
    votes_yes  = sum(r['pred'] == 1 for r in model_results)
    final_pred = 1 if votes_yes > len(model_results) / 2 else 0

    if verbose:
        print("\n" + "=" * 55)
        print("  HEART DISEASE PREDICTION REPORT")
        print("=" * 55)
        print(f"\n  Patient Info:")
        print(f"    Age                  : {patient_dict['age']}")
        print(f"    Sex                  : {'Male' if patient_dict['sex']==1 else 'Female'}")
        print(f"    Chest Pain Type      : {patient_dict['cp']} "
              f"({'Typical Angina' if patient_dict['cp']==0 else 'Atypical Angina' if patient_dict['cp']==1 else 'Non-Anginal' if patient_dict['cp']==2 else 'Asymptomatic'})")
        print(f"    Resting BP           : {patient_dict['trestbps']} mmHg")
        print(f"    Cholesterol          : {patient_dict['chol']} mg/dL")
        print(f"    Fasting Blood Sugar  : {'> 120 mg/dL' if patient_dict['fbs']==1 else '≤ 120 mg/dL'}")
        print(f"    Max Heart Rate       : {patient_dict['thalachh']} bpm")
        print(f"    Exercise Angina      : {'Yes' if patient_dict['exang']==1 else 'No'}")
        print(f"    ST Depression        : {patient_dict['oldpeak']}")
        print(f"    Major Vessels (ca)   : {patient_dict['ca']}")

        print(f"\n  Model Predictions:")
        print(f"  {'Model':<22} {'Pred':<12} {'P(Disease)':<12} {'P(Healthy)'}")
        print("  " + "-" * 55)
        for r in model_results:
            label = '🔴 Disease' if r['pred'] == 1 else '🟢 Healthy'
            print(f"  {r['name']:<22} {label:<14} {r['prob_disease']*100:>8.1f}%     {r['prob_healthy']*100:>7.1f}%")

        print(f"\n  ━━━━ ENSEMBLE RESULT ━━━━")
        print(f"  Vote Count     : {votes_yes}/{len(model_results)} models predict Disease")
        print(f"  Avg Probability: {avg_prob*100:.1f}% chance of Heart Disease")
        print(f"  Final Decision : {'❤️  HEART DISEASE DETECTED' if final_pred==1 else '💚 NO HEART DISEASE DETECTED'}")

        print_risk_banner(avg_prob, final_pred)

        # Clinical recommendations
        print("\n  📋 RECOMMENDATIONS:")
        if final_pred == 1:
            print("    • Refer to cardiologist immediately")
            print("    • Schedule ECG, Echo & stress test")
            print("    • Review medication and lifestyle")
            if patient_dict['chol'] > 240:
                print("    • Cholesterol is HIGH → consider statin therapy")
            if patient_dict['trestbps'] > 140:
                print("    • BP is HIGH → consider antihypertensive medication")
        else:
            print("    • Continue healthy lifestyle")
            print("    • Annual cardiovascular checkup recommended")
            if patient_dict.get('smoking','') == 'Current Smoker':
                print("    • Smoking cessation strongly recommended")

    return {'prediction': final_pred, 'probability': avg_prob,
            'model_results': model_results}

# ════════════════════════════════════════════════════════════════════════════
#  Test Cases
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    print("\n" + "=" * 55)
    print("  TEST CASE 1: HIGH RISK MALE PATIENT (58 yr)")
    print("=" * 55)
    p1 = {
        'age': 58, 'sex': 1, 'cp': 0, 'trestbps': 158, 'chol': 275,
        'fbs': 1, 'restecg': 1, 'thalachh': 115, 'exang': 1,
        'oldpeak': 2.8, 'slope': 1, 'ca': 2, 'thal': 7,
        'smoking': 'Current Smoker', 'alcohol': 'High',
        'exercise': 'Low', 'bmi': 'Obese'
    }
    predict_patient(p1)

    print("\n" + "=" * 55)
    print("  TEST CASE 2: LOW RISK FEMALE PATIENT (35 yr)")
    print("=" * 55)
    p2 = {
        'age': 35, 'sex': 0, 'cp': 2, 'trestbps': 118, 'chol': 190,
        'fbs': 0, 'restecg': 0, 'thalachh': 175, 'exang': 0,
        'oldpeak': 0.0, 'slope': 2, 'ca': 0, 'thal': 3,
        'smoking': 'Non-Smoker', 'alcohol': 'None',
        'exercise': 'High', 'bmi': 'Normal'
    }
    predict_patient(p2)

    print("\n" + "=" * 55)
    print("  TEST CASE 3: BORDERLINE MALE PATIENT (50 yr)")
    print("=" * 55)
    p3 = {
        'age': 50, 'sex': 1, 'cp': 1, 'trestbps': 135, 'chol': 245,
        'fbs': 0, 'restecg': 0, 'thalachh': 148, 'exang': 0,
        'oldpeak': 1.2, 'slope': 1, 'ca': 1, 'thal': 7,
        'smoking': 'Former Smoker', 'alcohol': 'Moderate',
        'exercise': 'Moderate', 'bmi': 'Overweight'
    }
    predict_patient(p3)

    print("\n\n✅ Single patient prediction complete.")
    print("   Import predict_patient() from this file for custom inputs.")
