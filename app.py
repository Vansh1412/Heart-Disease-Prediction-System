"""
app.py  —  Heart Disease Prediction  |  Flask Web Server
Run:  python app.py
Then open:  http://localhost:5000
"""

import os, warnings, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__, template_folder='templates', static_folder='static')

# ─── Load Models ──────────────────────────────────────────────────────────────
def load_models():
    scaler       = joblib.load(os.path.join(BASE, 'models/scaler.pkl'))
    encoders     = joblib.load(os.path.join(BASE, 'models/label_encoders.pkl'))
    feature_cols = joblib.load(os.path.join(BASE, 'models/feature_cols.pkl'))
    model_files  = {
        'Logistic Regression':    ('models/logistic_regression.pkl',    True),
        'K-Nearest Neighbors':    ('models/k-nearest_neighbors.pkl',    True),
        'Decision Tree':          ('models/decision_tree.pkl',          False),
        'Random Forest':          ('models/random_forest.pkl',          False),
        'Gradient Boosting':      ('models/gradient_boosting.pkl',      False),
        'Support Vector Machine': ('models/support_vector_machine.pkl', True),
    }
    models = {}
    for name, (path, scaled) in model_files.items():
        full = os.path.join(BASE, path)
        if os.path.exists(full):
            models[name] = {'model': joblib.load(full), 'scaled': scaled}
    return scaler, encoders, feature_cols, models

scaler, encoders, feature_cols, MODELS = load_models()

# ─── Compatibility patch for older sklearn pickles ─────────────────────────────
def _patch_model(m):
    """Fix attributes removed in newer scikit-learn versions."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        if isinstance(m, LogisticRegression):
            if not hasattr(m, 'multi_class'):
                m.multi_class = 'auto'
            if not hasattr(m, 'l1_ratio'):
                m.l1_ratio = None
    except Exception:
        pass
    return m

for _name, _info in MODELS.items():
    _patch_model(_info['model'])

# ─── Load Dataset (summary stats only) ────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, 'data/heart_dataset.csv'))

# ─── Feature Engineering Helper ───────────────────────────────────────────────
def build_input(data):
    age        = int(data['age'])
    sex_v      = 1 if data['sex'] == 'Male' else 0
    cp         = int(data['cp'])
    trestbps   = int(data['trestbps'])
    chol       = int(data['chol'])
    fbs_v      = int(data['fbs'])
    restecg    = int(data['restecg'])
    thalachh   = int(data['thalachh'])
    exang      = int(data['exang'])
    oldpeak    = float(data['oldpeak'])
    slope      = int(data['slope'])
    ca         = int(data['ca'])
    thal       = int(data['thal'])
    smoking    = data['smoking']
    alcohol    = data['alcohol']
    exercise   = data['exercise']
    bmi_cat    = data['bmi']

    row = {
        'Age': age, 'Sex': sex_v, 'Chest_Pain_Type': cp,
        'Trestbps': trestbps, 'Cholesterol': chol,
        'Fasting_Blood_Sugar': fbs_v, 'Resting_ECG': restecg,
        'Max_Heart_Rate': thalachh, 'Exercise_Induced_Angina': exang,
        'ST_Depression': oldpeak, 'Slope': slope,
        'Major_Vessels': ca, 'Thalassemia': thal,
    }
    for col, val in [('Smoking_Status', smoking),
                     ('Alcohol_Consumption', alcohol),
                     ('Exercise_Level', exercise),
                     ('BMI_Category', bmi_cat)]:
        if col in encoders and val in encoders[col].classes_:
            row[col] = int(encoders[col].transform([val])[0])
        else:
            row[col] = 0

    row['Age_Sex_Interact'] = age * sex_v
    row['BP_Chol_Score']    = (trestbps / 100) * (chol / 200)
    row['HR_Reserve']       = (220 - age) - thalachh
    row['ST_Slope_Risk']    = oldpeak * (slope + 1)

    df_in = pd.DataFrame([row])
    for c in feature_cols:
        if c not in df_in.columns:
            df_in[c] = 0
    df_in = df_in[feature_cols]
    return df_in, scaler.transform(df_in)


def clinical_risk_score(d):
    age      = int(d['age'])
    sex      = d['sex']
    cp       = int(d['cp'])
    trestbps = int(d['trestbps'])
    chol     = int(d['chol'])
    fbs      = int(d['fbs'])
    thalachh = int(d['thalachh'])
    exang    = int(d['exang'])
    oldpeak  = float(d['oldpeak'])
    ca       = int(d['ca'])
    thal     = int(d['thal'])
    smoking  = d['smoking']
    bmi_cat  = d['bmi']
    exercise = d['exercise']

    s = 0
    if   age < 40: s += 0
    elif age < 50: s += 5
    elif age < 55: s += 8
    elif age < 60: s += 12
    elif age < 65: s += 16
    else:          s += 20
    s += 6 if sex == 'Male' else 0
    s += {0: 14, 1: 8, 2: 4, 3: 0}.get(cp, 0)
    s += 0 if trestbps < 120 else 2 if trestbps < 130 else 5 if trestbps < 140 else 8 if trestbps < 160 else 10
    s += 0 if chol < 200 else 3 if chol < 240 else 6 if chol < 280 else 8
    s += 4 if fbs else 0
    s += 0 if thalachh > 170 else 2 if thalachh > 150 else 5 if thalachh > 130 else 8 if thalachh > 110 else 10
    s += 8 if exang else 0
    s += 0 if oldpeak == 0 else 3 if oldpeak < 1 else 6 if oldpeak < 2 else 8
    s += ca * 2
    s += {3: 0, 6: 2, 7: 4}.get(thal, 0)
    if smoking == 'Current Smoker': s += 5
    elif smoking == 'Former Smoker': s += 2
    if bmi_cat == 'Obese': s += 4
    elif bmi_cat == 'Overweight': s += 2
    if exercise == 'Low': s += 3
    elif exercise == 'High': s -= 2
    return int(min(max(s, 0), 100))


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X_raw, X_scaled = build_input(data)

        preds = {}
        for name, info in MODELS.items():
            X_in  = X_scaled if info['scaled'] else X_raw
            pred  = int(info['model'].predict(X_in)[0])
            prob  = info['model'].predict_proba(X_in)[0].tolist()
            preds[name] = {'pred': pred, 'prob_disease': prob[1], 'prob_healthy': prob[0]}

        avg_prob   = float(np.mean([v['prob_disease'] for v in preds.values()]))
        votes_yes  = sum(v['pred'] == 1 for v in preds.values())
        final_pred = 1 if votes_yes > len(preds) / 2 else 0
        risk_score = clinical_risk_score(data)

        if risk_score < 25:   risk_band = 'Very Low Risk'
        elif risk_score < 40: risk_band = 'Low Risk'
        elif risk_score < 55: risk_band = 'Moderate Risk'
        elif risk_score < 70: risk_band = 'High Risk'
        else:                 risk_band = 'Very High Risk'

        # Recommendations
        chol     = int(data['chol'])
        trestbps = int(data['trestbps'])
        exang    = int(data['exang'])
        ca       = int(data['ca'])
        smoking  = data['smoking']
        bmi_cat  = data['bmi']

        recs = []
        if final_pred == 1:
            recs += ['Refer to cardiologist immediately',
                     'Schedule ECG, Echocardiogram & Stress Test',
                     'Review current medications']
            if chol > 240:     recs.append('Cholesterol HIGH — consider statin therapy')
            if trestbps > 140: recs.append('Blood Pressure HIGH — consider antihypertensive medication')
            if exang:          recs.append('Exercise angina present — restrict strenuous activity')
            if ca >= 2:        recs.append('Multiple vessels blocked — surgical evaluation advised')
        else:
            recs += ['Continue healthy lifestyle',
                     'Annual cardiovascular checkup recommended',
                     'Maintain regular moderate exercise']
            if smoking == 'Current Smoker': recs.append('Smoking cessation strongly advised')
            if bmi_cat == 'Obese':          recs.append('Weight management recommended')
            if chol > 200:                  recs.append('Dietary changes to reduce cholesterol')

        return jsonify({
            'success': True,
            'final_pred': final_pred,
            'avg_prob': avg_prob,
            'votes_yes': votes_yes,
            'total_models': len(preds),
            'risk_score': risk_score,
            'risk_band': risk_band,
            'recommendations': recs,
            'model_preds': {k: {'pred': v['pred'],
                               'prob_disease': round(v['prob_disease'] * 100, 2),
                               'prob_healthy': round(v['prob_healthy'] * 100, 2)}
                            for k, v in preds.items()}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dataset-stats')
def dataset_stats():
    total        = len(df)
    disease      = int(df['Target'].sum())
    healthy      = int((df['Target'] == 0).sum())
    disease_pct  = round(df['Target'].mean() * 100, 1)

    # Age group rates
    age_rate = df.groupby('Age_Group')['Target'].mean().reset_index()
    age_order = ['Young', 'Middle Age', 'Senior', 'Elderly']
    age_rate['sort'] = age_rate['Age_Group'].map(
        {k: i for i, k in enumerate(age_order)})
    age_rate = age_rate.sort_values('sort')

    # Gender
    gender_stats = df.groupby('Sex_Label').apply(
        lambda g: {'disease': int((g['Target'] == 1).sum()),
                   'healthy': int((g['Target'] == 0).sum())}
    ).to_dict()

    # Risk level distribution
    risk_dist = df['Risk_Level'].value_counts().to_dict()

    # Lifestyle
    lifestyle_rates = {}
    for col in ['Smoking_Status', 'Exercise_Level', 'BMI_Category', 'Alcohol_Consumption']:
        r = df.groupby(col)['Target'].mean().round(3).to_dict()
        lifestyle_rates[col] = r

    return jsonify({
        'total': total,
        'disease': disease,
        'healthy': healthy,
        'disease_pct': disease_pct,
        'age_rate': age_rate[['Age_Group', 'Target']].rename(
            columns={'Target': 'rate'}).to_dict(orient='records'),
        'gender_stats': gender_stats,
        'risk_dist': risk_dist,
        'lifestyle_rates': lifestyle_rates
    })


if __name__ == '__main__':
    print("=" * 55)
    print("  ❤️  Heart Disease Prediction System")
    print("  Open your browser: http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, port=5000)
