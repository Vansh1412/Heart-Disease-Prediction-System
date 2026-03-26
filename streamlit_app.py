# ============================================================
#  streamlit_app.py
#  Heart Disease Prediction — Interactive Teacher Demo
#  Run with:  streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import warnings, os, inspect
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(192,57,43,0.3);
    }
    .main-header h1 { color: white; margin:0; font-size: 2.2rem; }
    .main-header p  { color: rgba(255,255,255,0.85); margin:4px 0 0; font-size:1rem; }
    .metric-card {
        background: white; border-radius: 10px; padding: 18px 22px;
        border-left: 5px solid #e74c3c; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .metric-card h3 { margin:0 0 4px; font-size: 0.85rem; color: #7f8c8d; text-transform: uppercase; }
    .metric-card p  { margin:0; font-size: 1.8rem; font-weight: 700; color: #2c3e50; }
    .risk-high   { border-color: #e74c3c; background: #fdf0ed; }
    .risk-medium { border-color: #f39c12; background: #fef9ec; }
    .risk-low    { border-color: #27ae60; background: #edfaf1; }
    .code-box {
        background: #1e1e1e; color: #d4d4d4; border-radius: 8px;
        padding: 16px; font-family: monospace; font-size: 0.8rem;
        overflow-x: auto; line-height: 1.6;
    }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #2c3e50;
        border-bottom: 2px solid #e74c3c; padding-bottom: 6px; margin-bottom: 16px;
    }
    .stTab { font-size: 1rem !important; }
    div[data-testid="stSidebarContent"] { background: #fafafa; }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    scaler       = joblib.load(os.path.join(base, 'models/scaler.pkl'))
    encoders     = joblib.load(os.path.join(base, 'models/label_encoders.pkl'))
    feature_cols = joblib.load(os.path.join(base, 'models/feature_cols.pkl'))
    models = {}
    model_files = {
        'Logistic Regression':    ('models/logistic_regression.pkl',    True),
        'K-Nearest Neighbors':    ('models/k-nearest_neighbors.pkl',    True),
        'Decision Tree':          ('models/decision_tree.pkl',          False),
        'Random Forest':          ('models/random_forest.pkl',          False),
        'Gradient Boosting':      ('models/gradient_boosting.pkl',      False),
        'Support Vector Machine': ('models/support_vector_machine.pkl', True),
    }
    for name, (path, scaled) in model_files.items():
        full = os.path.join(base, path)
        if os.path.exists(full):
            models[name] = {'model': joblib.load(full), 'scaled': scaled}
    return scaler, encoders, feature_cols, models

@st.cache_data
def load_dataset():
    base = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(base, 'data/heart_dataset.csv'))

scaler, encoders, feature_cols, models = load_models()
df = load_dataset()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>❤️ Heart Disease Prediction System</h1>
  <p>ML-powered clinical decision support · 12,000 patients · 6 algorithms · Real-time prediction</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Patient Input
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏥 Enter Patient Details")
    st.markdown("---")

    st.markdown("**👤 Demographics**")
    age = st.slider("Age", 29, 77, 55, help="Patient age in years")
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)

    st.markdown("**🫀 Cardiac Measurements**")
    cp_labels = {
        "Typical Angina (0)": 0,
        "Atypical Angina (1)": 1,
        "Non-Anginal Pain (2)": 2,
        "Asymptomatic (3)": 3
    }
    cp_sel  = st.selectbox("Chest Pain Type", list(cp_labels.keys()))
    cp      = cp_labels[cp_sel]

    trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
    chol     = st.slider("Cholesterol (mg/dL)", 126, 400, 240)
    thalachh = st.slider("Max Heart Rate (bpm)", 70, 202, 150)
    oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)

    st.markdown("**🔬 Lab & Test Results**")
    fbs     = st.radio("Fasting Blood Sugar > 120 mg/dL", ["No (0)", "Yes (1)"], horizontal=True)
    fbs_val = 1 if "Yes" in fbs else 0

    restecg_labels = {"Normal (0)": 0, "ST-T Abnormality (1)": 1, "LV Hypertrophy (2)": 2}
    restecg_sel = st.selectbox("Resting ECG", list(restecg_labels.keys()))
    restecg     = restecg_labels[restecg_sel]

    exang_sel = st.radio("Exercise-Induced Angina", ["No (0)", "Yes (1)"], horizontal=True)
    exang     = 1 if "Yes" in exang_sel else 0

    slope_labels = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}
    slope_sel = st.selectbox("Slope of Peak ST", list(slope_labels.keys()))
    slope     = slope_labels[slope_sel]

    ca   = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3])
    thal_labels = {"Normal (3)": 3, "Fixed Defect (6)": 6, "Reversible Defect (7)": 7}
    thal_sel = st.selectbox("Thalassemia", list(thal_labels.keys()))
    thal = thal_labels[thal_sel]

    st.markdown("**🌿 Lifestyle**")
    smoking  = st.selectbox("Smoking Status", ["Non-Smoker", "Former Smoker", "Current Smoker"])
    alcohol  = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
    exercise = st.selectbox("Exercise Level", ["High", "Moderate", "Low"])
    bmi      = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])

    st.markdown("---")
    predict_btn = st.button("🔮 Run Prediction", type="primary", use_container_width=True)

# ── Preprocessing helper ──────────────────────────────────────────────────────
def build_input(age, sex_str, cp, trestbps, chol, fbs_v, restecg, thalachh,
                exang, oldpeak, slope, ca, thal, smoking, alcohol, exercise, bmi):
    sex_v = 1 if sex_str == "Male" else 0
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
                     ('BMI_Category', bmi)]:
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

def clinical_risk_score(row):
    s = 0
    age_v = row['age']
    if   age_v < 40:  s += 0
    elif age_v < 50:  s += 5
    elif age_v < 55:  s += 8
    elif age_v < 60:  s += 12
    elif age_v < 65:  s += 16
    else:              s += 20
    s += 6 if row['sex'] == "Male" else 0
    s += {0:14, 1:8, 2:4, 3:0}.get(row['cp'], 0)
    bp = row['trestbps']
    s += 0 if bp<120 else 2 if bp<130 else 5 if bp<140 else 8 if bp<160 else 10
    c = row['chol']
    s += 0 if c<200 else 3 if c<240 else 6 if c<280 else 8
    s += 4 if row['fbs'] else 0
    hr = row['thalachh']
    s += 0 if hr>170 else 2 if hr>150 else 5 if hr>130 else 8 if hr>110 else 10
    s += 8 if row['exang'] else 0
    op = row['oldpeak']
    s += 0 if op==0 else 3 if op<1 else 6 if op<2 else 8
    s += row['ca'] * 2
    s += {3:0, 6:2, 7:4}.get(row['thal'], 0)
    if row['smoking'] == 'Current Smoker': s += 5
    elif row['smoking'] == 'Former Smoker': s += 2
    if row['bmi'] == 'Obese': s += 4
    elif row['bmi'] == 'Overweight': s += 2
    if row['exercise'] == 'Low': s += 3
    elif row['exercise'] == 'High': s -= 2
    return min(max(s, 0), 100)

# ════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Prediction Results",
    "📊 Dataset Analytics",
    "🤖 Model Performance",
    "📈 Feature Analysis",
    "💻 Backend Code"
])

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION RESULTS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if not predict_btn:
        st.info("👈  Fill in patient details in the sidebar and click **Run Prediction**")

        # Show sample patients
        st.markdown("#### 📋 Sample Test Patients — click a preset to load")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="metric-card risk-high">
            <h3>🔴 High Risk Patient</h3>
            <p style="font-size:0.9rem">Male, 58yr · Typical Angina<br>
            BP=158 · Chol=275 · HR=115<br>
            Exang=Yes · Oldpeak=2.8 · ca=2</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card risk-low">
            <h3>🟢 Low Risk Patient</h3>
            <p style="font-size:0.9rem">Female, 35yr · Non-Anginal<br>
            BP=118 · Chol=190 · HR=175<br>
            Exang=No · Oldpeak=0.0 · ca=0</p>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="metric-card risk-medium">
            <h3>🟡 Borderline Patient</h3>
            <p style="font-size:0.9rem">Male, 50yr · Atypical Angina<br>
            BP=135 · Chol=245 · HR=148<br>
            Exang=No · Oldpeak=1.2 · ca=1</p>
            </div>""", unsafe_allow_html=True)

    else:
        # Build feature vector
        patient_input = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                         'chol': chol, 'fbs': fbs_val, 'restecg': restecg,
                         'thalachh': thalachh, 'exang': exang, 'oldpeak': oldpeak,
                         'slope': slope, 'ca': ca, 'thal': thal,
                         'smoking': smoking, 'alcohol': alcohol,
                         'exercise': exercise, 'bmi': bmi}

        X_raw, X_scaled = build_input(age, sex, cp, trestbps, chol, fbs_val,
                                       restecg, thalachh, exang, oldpeak,
                                       slope, ca, thal, smoking, alcohol, exercise, bmi)

        # Risk score
        risk_score = clinical_risk_score(patient_input)
        if risk_score < 25:   risk_band, risk_color = "Very Low Risk",   "#27ae60"
        elif risk_score < 40: risk_band, risk_color = "Low Risk",        "#2ecc71"
        elif risk_score < 55: risk_band, risk_color = "Moderate Risk",   "#f39c12"
        elif risk_score < 70: risk_band, risk_color = "High Risk",       "#e67e22"
        else:                 risk_band, risk_color = "Very High Risk",  "#e74c3c"

        # Run all models
        preds = {}
        for name, info in models.items():
            X_in = X_scaled if info['scaled'] else X_raw
            pred = info['model'].predict(X_in)[0]
            prob = info['model'].predict_proba(X_in)[0]
            preds[name] = {'pred': pred, 'prob_disease': prob[1], 'prob_healthy': prob[0]}

        avg_prob    = np.mean([v['prob_disease'] for v in preds.values()])
        votes_yes   = sum(v['pred'] == 1 for v in preds.values())
        final_pred  = 1 if votes_yes > len(preds) / 2 else 0
        verdict     = "❤️ HEART DISEASE DETECTED" if final_pred == 1 else "💚 NO HEART DISEASE"
        verdict_clr = "#e74c3c" if final_pred == 1 else "#27ae60"

        # ── TOP SUMMARY ROW ───────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{verdict_clr},{verdict_clr}cc);
            padding:20px 30px;border-radius:12px;margin-bottom:20px;
            box-shadow:0 4px 15px rgba(0,0,0,0.15);">
          <h2 style="color:white;margin:0;font-size:1.8rem">{verdict}</h2>
          <p style="color:rgba(255,255,255,0.9);margin:4px 0 0;font-size:1rem">
            Ensemble of {len(preds)} models · {votes_yes}/{len(preds)} votes for disease · 
            Avg probability: {avg_prob*100:.1f}%
          </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Disease Probability", f"{avg_prob*100:.1f}%",
                      delta=f"{(avg_prob-0.54)*100:+.1f}% vs avg")
        with col2:
            st.metric("🗳️ Model Votes",
                      f"{votes_yes}/{len(preds)} Disease", "")
        with col3:
            st.metric("⚠️ Clinical Risk Score", f"{risk_score}/100", risk_band)
        with col4:
            st.metric("📊 Risk Band", risk_band, "")

        st.markdown("---")

        # ── GAUGE + PROBABILITY BARS ──────────────────────────────────────────
        col_g, col_b = st.columns([1, 1])

        with col_g:
            st.markdown('<p class="section-title">🎯 Disease Probability Gauge</p>',
                        unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_prob * 100,
                delta={'reference': 54, 'valueformat': '.1f',
                       'increasing': {'color': '#e74c3c'},
                       'decreasing': {'color': '#27ae60'}},
                title={'text': "Heart Disease Probability (%)", 'font': {'size': 14}},
                number={'suffix': '%', 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': verdict_clr, 'thickness': 0.25},
                    'bgcolor': 'white',
                    'bordercolor': 'gray',
                    'steps': [
                        {'range': [0,  25], 'color': '#d5f5e3'},
                        {'range': [25, 50], 'color': '#fef9e7'},
                        {'range': [50, 75], 'color': '#fdebd0'},
                        {'range': [75, 100],'color': '#fadbd8'},
                    ],
                    'threshold': {
                        'line': {'color': 'black', 'width': 3},
                        'thickness': 0.8, 'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_b:
            st.markdown('<p class="section-title">🤖 All Model Predictions</p>',
                        unsafe_allow_html=True)
            model_names_list = list(preds.keys())
            probs_disease    = [preds[m]['prob_disease']*100 for m in model_names_list]
            bar_colors       = ['#e74c3c' if p > 50 else '#27ae60' for p in probs_disease]

            fig_bars = go.Figure()
            fig_bars.add_trace(go.Bar(
                y=model_names_list, x=probs_disease,
                orientation='h',
                marker_color=bar_colors,
                text=[f"{p:.1f}%" for p in probs_disease],
                textposition='outside',
            ))
            fig_bars.add_vline(x=50, line_dash='dash', line_color='black',
                               annotation_text='50% threshold')
            fig_bars.update_layout(
                xaxis=dict(range=[0, 115], title='P(Heart Disease) %'),
                height=300, margin=dict(l=10,r=10,t=10,b=30),
                plot_bgcolor='white', paper_bgcolor='white',
                showlegend=False
            )
            st.plotly_chart(fig_bars, use_container_width=True)

        # ── CLINICAL RISK SCORE GAUGE ─────────────────────────────────────────
        st.markdown("---")
        col_rs, col_rec = st.columns([1, 1])

        with col_rs:
            st.markdown('<p class="section-title">⚠️ Clinical Risk Score (0–100)</p>',
                        unsafe_allow_html=True)
            fig_rs = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': f"Risk Band: {risk_band}", 'font': {'size': 13}},
                number={'font': {'size': 42, 'color': risk_color}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color, 'thickness': 0.3},
                    'steps': [
                        {'range': [0,  25], 'color': '#d5f5e3'},
                        {'range': [25, 40], 'color': '#a9dfbf'},
                        {'range': [40, 55], 'color': '#fef9e7'},
                        {'range': [55, 70], 'color': '#fdebd0'},
                        {'range': [70, 100],'color': '#fadbd8'},
                    ],
                }
            ))
            fig_rs.update_layout(height=260, margin=dict(l=20,r=20,t=50,b=10))
            st.plotly_chart(fig_rs, use_container_width=True)

        with col_rec:
            st.markdown('<p class="section-title">📋 Clinical Recommendations</p>',
                        unsafe_allow_html=True)
            recs = []
            if final_pred == 1:
                recs += ["🔴 **Refer to cardiologist immediately**",
                         "📋 Schedule ECG, Echocardiogram & Stress Test",
                         "💊 Review current medications"]
                if chol > 240:  recs.append("⚠️ Cholesterol HIGH → consider statin therapy")
                if trestbps > 140: recs.append("⚠️ BP HIGH → consider antihypertensive medication")
                if exang:       recs.append("⚠️ Exercise angina present → restrict strenuous activity")
                if ca >= 2:     recs.append("⚠️ Multiple vessels blocked → surgical evaluation advised")
            else:
                recs += ["✅ **Continue healthy lifestyle**",
                         "📅 Annual cardiovascular checkup recommended",
                         "🏃 Maintain regular moderate exercise"]
                if smoking == 'Current Smoker': recs.append("🚭 Smoking cessation strongly advised")
                if bmi == 'Obese': recs.append("⚖️ Weight management recommended")
                if chol > 200:  recs.append("🥗 Dietary changes to reduce cholesterol")

            for r in recs:
                st.markdown(f"- {r}")

        # ── PATIENT INPUT SUMMARY TABLE ───────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-title">📋 Patient Input Summary</p>',
                    unsafe_allow_html=True)
        summary_data = {
            'Feature': ['Age','Sex','Chest Pain Type','Resting BP','Cholesterol',
                        'Fasting Blood Sugar','Resting ECG','Max Heart Rate',
                        'Exercise Angina','ST Depression','Slope',
                        'Major Vessels','Thalassemia','Smoking','Alcohol','Exercise','BMI'],
            'Value':   [age, sex, cp_sel, f"{trestbps} mmHg", f"{chol} mg/dL",
                        fbs, restecg_sel, f"{thalachh} bpm",
                        exang_sel, oldpeak, slope_sel,
                        ca, thal_sel, smoking, alcohol, exercise, bmi],
            'Normal Range / Note': [
                '29–77','—','0=Typical Angina is highest risk','90–120 mmHg','<200 mg/dL',
                '≤120 mg/dL is normal','0=Normal','100–170 bpm','No is better',
                '0 is ideal','Downsloping (2) is best','0 vessels is ideal',
                'Normal (3) is best','Non-smoker is ideal','None is ideal',
                'High is ideal','Normal weight is ideal'
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DATASET ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">📊 Dataset Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", f"{len(df):,}")
    c2.metric("Heart Disease Cases", f"{df['Target'].sum():,}", f"{df['Target'].mean()*100:.1f}%")
    c3.metric("Healthy Cases", f"{(df['Target']==0).sum():,}")
    c4.metric("Features", "28 raw · 21 ML features")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        # Disease by Age Group
        age_rate = df.groupby('Age_Group')['Target'].mean().reset_index()
        age_rate.columns = ['Age Group','Disease Rate']
        age_order = ['Young','Middle Age','Senior','Elderly']
        age_rate['Age Group'] = pd.Categorical(age_rate['Age Group'], categories=age_order, ordered=True)
        age_rate = age_rate.sort_values('Age Group')
        fig = px.bar(age_rate, x='Age Group', y='Disease Rate',
                     color='Disease Rate', color_continuous_scale='RdYlGn_r',
                     title='Disease Rate by Age Group',
                     text=age_rate['Disease Rate'].map(lambda x: f'{x*100:.1f}%'))
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis_tickformat='.0%', showlegend=False,
                          plot_bgcolor='white', height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Gender distribution
        gen_dis = df.groupby(['Sex_Label','Target']).size().reset_index(name='Count')
        gen_dis['Status'] = gen_dis['Target'].map({0:'No Disease',1:'Heart Disease'})
        fig = px.bar(gen_dis, x='Sex_Label', y='Count', color='Status',
                     barmode='group', title='Disease Count by Gender',
                     color_discrete_map={'No Disease':'#3498db','Heart Disease':'#e74c3c'})
        fig.update_layout(plot_bgcolor='white', height=320)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        # Cholesterol distribution
        fig = go.Figure()
        for t, color, name in [(0,'#3498db','No Disease'),(1,'#e74c3c','Heart Disease')]:
            fig.add_trace(go.Histogram(x=df[df['Target']==t]['Cholesterol'],
                nbinsx=40, opacity=0.6, name=name, marker_color=color))
        fig.update_layout(barmode='overlay', title='Cholesterol Distribution',
                          xaxis_title='Cholesterol (mg/dL)',
                          plot_bgcolor='white', height=320, legend=dict(x=0.7, y=0.95))
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        # Risk Level pie
        rl_counts = df['Risk_Level'].value_counts()
        fig = px.pie(values=rl_counts.values, names=rl_counts.index,
                     title='Patient Distribution by Risk Level',
                     color_discrete_map={'Low Risk':'#27ae60',
                                         'Moderate Risk':'#f39c12',
                                         'High Risk':'#e74c3c'},
                     hole=0.4)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Lifestyle vs disease
    st.markdown('<p class="section-title">🌿 Lifestyle Factors Impact</p>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    for col_widget, feature in zip([col1, col2, col3, col4],
        ['Smoking_Status','Exercise_Level','BMI_Category','Alcohol_Consumption']):
        rate = df.groupby(feature)['Target'].mean().reset_index()
        rate.columns = [feature, 'Rate']
        fig = px.bar(rate, x=feature, y='Rate', title=feature.replace('_',' '),
                     color='Rate', color_continuous_scale='RdYlGn_r',
                     text=rate['Rate'].map(lambda x: f'{x*100:.0f}%'))
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis_tickformat='.0%', showlegend=False,
                          plot_bgcolor='white', height=280,
                          margin=dict(t=40,b=40), xaxis_tickangle=-20)
        col_widget.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">🏆 Model Comparison Dashboard</p>',
                unsafe_allow_html=True)

    perf = {
        'Model':     ['Logistic Regression','KNN','Decision Tree',
                      'Random Forest','Gradient Boosting','SVM'],
        'Accuracy':  [78.92, 75.88, 73.54, 78.88, 79.71, 80.50],
        'Precision': [79.61, 76.03, 75.29, 79.20, 80.21, 80.99],
        'Recall':    [81.94, 80.79, 75.93, 82.56, 82.87, 83.49],
        'F1 Score':  [80.76, 78.34, 75.61, 80.85, 81.52, 82.22],
        'AUC-ROC':   [0.871, 0.824, 0.806, 0.875, 0.885, 0.882],
    }
    perf_df = pd.DataFrame(perf).sort_values('Accuracy', ascending=False)

    # Styled table
    st.dataframe(
        perf_df.style
            .background_gradient(subset=['Accuracy','F1 Score','AUC-ROC'], cmap='RdYlGn')
            .format({'Accuracy':'{:.2f}%','Precision':'{:.2f}%',
                     'Recall':'{:.2f}%','F1 Score':'{:.2f}%','AUC-ROC':'{:.3f}'}),
        use_container_width=True, hide_index=True
    )

    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        metrics = ['Accuracy','Precision','Recall','F1 Score']
        colors  = ['#3498db','#2ecc71','#e74c3c','#9b59b6']
        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Bar(name=metric, x=perf_df['Model'],
                                 y=perf_df[metric], marker_color=color, opacity=0.8))
        fig.update_layout(barmode='group', title='All Metrics per Model',
                          yaxis=dict(range=[60, 90], title='Score (%)'),
                          plot_bgcolor='white', height=380,
                          xaxis_tickangle=-20, legend=dict(x=0,y=1))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig = px.bar(perf_df.sort_values('Accuracy'),
                     x='Accuracy', y='Model', orientation='h',
                     color='Accuracy', color_continuous_scale='RdYlGn',
                     title='Accuracy Ranking',
                     text=perf_df.sort_values('Accuracy')['Accuracy'].map(lambda x: f'{x:.2f}%'))
        fig.add_vline(x=75, line_dash='dash', line_color='red',
                      annotation_text='75% realistic threshold')
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis=dict(range=[60, 88]), plot_bgcolor='white', height=380)
        st.plotly_chart(fig, use_container_width=True)

    # CV scores
    st.markdown("---")
    st.markdown("**5-Fold Stratified Cross-Validation Scores:**")
    cv_data = {
        'Model': ['Logistic Regression','Random Forest','Gradient Boosting'],
        'CV Mean': ['78.83%','78.88%','80.58%'],
        'CV Std':  ['±0.55%','±0.71%','±0.64%'],
        'Min':     ['77.83%','77.96%','79.71%'],
        'Max':     ['79.38%','80.04%','81.33%'],
    }
    st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — FEATURE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">📈 Feature Importance & Correlations</p>',
                unsafe_allow_html=True)

    feat_imp = {
        'Feature': ['Max_Heart_Rate','Age','Major_Vessels','Smoking_Status',
                    'Exercise_Level','Alcohol_Consumption','Chest_Pain_Type',
                    'Age_Sex_Interact','BP_Chol_Score','ST_Depression',
                    'Thalassemia','Trestbps','Exercise_Induced_Angina','Cholesterol'],
        'RF_Imp':  [13.4, 9.2, 9.0, 8.4, 7.4, 6.0, 5.8, 5.3, 5.2, 4.7,
                    4.1, 3.9, 3.7, 3.5],
        'GB_Imp':  [15.1, 10.2, 8.5, 7.9, 6.8, 5.5, 6.2, 5.0, 4.9, 5.1,
                    4.5, 4.2, 3.8, 3.2],
    }
    fi_df = pd.DataFrame(feat_imp)
    fi_df['Avg'] = (fi_df['RF_Imp'] + fi_df['GB_Imp']) / 2
    fi_df = fi_df.sort_values('Avg', ascending=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Random Forest', y=fi_df['Feature'],
                             x=fi_df['RF_Imp'], orientation='h', marker_color='#3498db'))
        fig.add_trace(go.Bar(name='Gradient Boosting', y=fi_df['Feature'],
                             x=fi_df['GB_Imp'], orientation='h', marker_color='#e74c3c'))
        fig.update_layout(barmode='group', title='Feature Importance Comparison',
                          xaxis_title='Importance (%)', plot_bgcolor='white',
                          height=450, legend=dict(x=0.6, y=0.05))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Correlation with target
        corr_data = {
            'Feature': ['Chest_Pain_Type','Exercise_Induced_Angina','ST_Depression',
                        'Major_Vessels','Thalassemia','Slope','Age','Sex',
                        'Trestbps','Fasting_Blood_Sugar','Max_Heart_Rate'],
            'Correlation': [-0.43, 0.43, 0.43, 0.48, 0.52, -0.35, 0.23, 0.27,
                            0.16, 0.09, -0.42]
        }
        cd = pd.DataFrame(corr_data).sort_values('Correlation')
        colors_corr = ['#e74c3c' if v > 0 else '#3498db' for v in cd['Correlation']]
        fig = go.Figure(go.Bar(
            y=cd['Feature'], x=cd['Correlation'],
            orientation='h', marker_color=colors_corr,
            text=cd['Correlation'].map(lambda x: f'{x:+.3f}'),
            textposition='outside'
        ))
        fig.add_vline(x=0, line_color='black', line_width=1)
        fig.update_layout(title='Correlation with Target\n(Red=↑Risk · Blue=↓Risk)',
                          xaxis_title='Correlation Coefficient',
                          plot_bgcolor='white', height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Key stats by feature
    st.markdown("---")
    st.markdown("**Key Clinical Statistics:**")
    stats_data = []
    for col_name in ['Age','Trestbps','Cholesterol','Max_Heart_Rate','ST_Depression']:
        d = df[df['Target']==1][col_name]
        h = df[df['Target']==0][col_name]
        stats_data.append({
            'Feature': col_name,
            'Disease Mean ± Std': f"{d.mean():.1f} ± {d.std():.1f}",
            'Healthy Mean ± Std': f"{h.mean():.1f} ± {h.std():.1f}",
            'Difference': f"{d.mean()-h.mean():+.1f}",
        })
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — BACKEND CODE VIEWER
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">💻 Backend Code — How It Works</p>',
                unsafe_allow_html=True)

    selected_script = st.selectbox("📂 Select a script to view:", [
        "03_preprocessing.py — Feature Engineering & Scaling",
        "04_model_training.py — Train 6 ML Models",
        "09_predict_new_patient.py — Single Patient Prediction",
        "12_risk_scoring.py — Clinical Risk Score Engine",
        "13_statistics_report.py — Statistical Tests",
    ])

    script_map = {
        "03_preprocessing.py — Feature Engineering & Scaling": "03_preprocessing.py",
        "04_model_training.py — Train 6 ML Models":            "04_model_training.py",
        "09_predict_new_patient.py — Single Patient Prediction":"09_predict_new_patient.py",
        "12_risk_scoring.py — Clinical Risk Score Engine":      "12_risk_scoring.py",
        "13_statistics_report.py — Statistical Tests":          "13_statistics_report.py",
    }

    fname = script_map[selected_script]
    fpath = os.path.join(os.path.dirname(__file__), fname)

    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            code = f.read()
        st.code(code, language='python', line_numbers=True)
        st.download_button(
            label=f"⬇️ Download {fname}",
            data=code, file_name=fname, mime='text/plain'
        )
    else:
        st.warning(f"File not found: {fname}")

    st.markdown("---")
    st.markdown("**🧠 How the Prediction Pipeline Works:**")
    st.markdown("""
    ```
    Patient Input (17 features)
         ↓
    Encode categorical (LabelEncoder)
         ↓
    Feature Engineering: Age×Sex, BP×Chol, HR Reserve, ST×Slope
         ↓
    Clip outliers (IQR × 1.5)
         ↓
    StandardScaler (mean=0, std=1)
         ↓
    ┌─────────────────────────────────────────────────────┐
    │  Logistic Regression   │  Random Forest             │
    │  K-Nearest Neighbors   │  Gradient Boosting         │
    │  Decision Tree         │  Support Vector Machine    │
    └─────────────────────────────────────────────────────┘
         ↓
    Ensemble: Average probability + Majority Vote
         ↓
    Clinical Risk Score (Framingham-inspired, 0–100)
         ↓
    Prediction + Risk Band + Clinical Recommendations
    ```
    """)
