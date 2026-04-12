"""
app.py  —  Heart Disease Prediction  |  Flask Web Server
Run:  python app.py
Then open:  http://localhost:5000
"""

import os, sys, io, warnings, json
# Ensure stdout handles unicode on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import joblib
import shap
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
try:
    from flask_mail import Mail, Message as MailMessage
    MAIL_AVAILABLE = True
except ImportError:
    MAIL_AVAILABLE = False
    print('[WARNING] flask-mail not installed - email notifications disabled. Run: pip install flask-mail')

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'super-secret-cardioai-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ── Email Configuration (set via environment variables or edit directly) ──────
app.config['MAIL_SERVER']   = os.environ.get('MAIL_SERVER',   'smtp.gmail.com')
app.config['MAIL_PORT']     = int(os.environ.get('MAIL_PORT', '587'))
app.config['MAIL_USE_TLS']  = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')   # your Gmail address
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')   # Gmail App Password
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME', 'cardioai@noreply.com')

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
mail = Mail(app) if MAIL_AVAILABLE else None
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'error'

# ─── Database Models ──────────────────────────────────────────────────────────
class User(db.Model, UserMixin):
    id               = db.Column(db.Integer, primary_key=True)
    username         = db.Column(db.String(50), unique=True, nullable=False)
    password_hash    = db.Column(db.String(100), nullable=False)
    role             = db.Column(db.String(20), default='doctor')
    full_name        = db.Column(db.String(120), default='')
    doctor_code      = db.Column(db.String(10),  default='')
    hospital_id      = db.Column(db.String(10),  default='')
    specialization   = db.Column(db.String(100), default='')
    experience_years = db.Column(db.Integer,     default=0)
    qualifications   = db.Column(db.String(200), default='')
    gender           = db.Column(db.String(10),  default='')
    patient_rating   = db.Column(db.Float,       default=0.0)
    last_login       = db.Column(db.DateTime,    nullable=True)
    is_active_acc    = db.Column(db.Boolean,     default=True)
    history          = db.relationship("PatientHistory", backref="doctor", lazy=True)

class PatientHistory(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    doctor_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_label = db.Column(db.String(80), default='')       # optional name/ID
    age           = db.Column(db.Integer)
    sex           = db.Column(db.String(10))
    risk_score    = db.Column(db.Integer)
    risk_band     = db.Column(db.String(30))
    avg_prob      = db.Column(db.Float)
    votes_yes     = db.Column(db.Integer)
    final_prediction = db.Column(db.String(50))
    timestamp     = db.Column(db.DateTime, default=datetime.utcnow)

class PatientProfile(db.Model):
    """Stores the patient's personal details — filled once, reused every session."""
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    full_name     = db.Column(db.String(120), default='')
    dob           = db.Column(db.String(20),  default='')
    sex           = db.Column(db.String(10),  default='')
    phone         = db.Column(db.String(20),  default='')
    email         = db.Column(db.String(120), default='')
    notify_email  = db.Column(db.Boolean,     default=True)   # opt-in email alerts
    updated_at    = db.Column(db.DateTime,    default=datetime.utcnow, onupdate=datetime.utcnow)

class Notification(db.Model):
    """In-app + email notifications for high-risk alerts."""
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title      = db.Column(db.String(120), default='')
    message    = db.Column(db.Text,        default='')
    ntype      = db.Column(db.String(20),  default='info')   # info | warning | danger
    is_read    = db.Column(db.Boolean,     default=False)
    timestamp  = db.Column(db.DateTime,    default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()
    # Migrate existing DB: add new columns if they don't exist (SQLite safe)
    from sqlalchemy import text
    new_cols = [
        "ALTER TABLE patient_history ADD COLUMN patient_label VARCHAR(80) DEFAULT ''",
        "ALTER TABLE patient_history ADD COLUMN risk_band VARCHAR(30)",
        "ALTER TABLE patient_history ADD COLUMN avg_prob FLOAT",
        "ALTER TABLE patient_history ADD COLUMN votes_yes INTEGER",
        "ALTER TABLE patient_profile ADD COLUMN email VARCHAR(120) DEFAULT ''",
        "ALTER TABLE patient_profile ADD COLUMN notify_email BOOLEAN DEFAULT 1",
    ]
    with db.engine.connect() as conn:
        for stmt in new_cols:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass  # column already exists

    # Migrate User table new columns
    user_new_cols = [
        "ALTER TABLE user ADD COLUMN full_name VARCHAR(120) DEFAULT ''",
        "ALTER TABLE user ADD COLUMN doctor_code VARCHAR(10) DEFAULT ''",
        "ALTER TABLE user ADD COLUMN hospital_id VARCHAR(10) DEFAULT ''",
        "ALTER TABLE user ADD COLUMN specialization VARCHAR(100) DEFAULT ''",
        "ALTER TABLE user ADD COLUMN experience_years INTEGER DEFAULT 0",
        "ALTER TABLE user ADD COLUMN qualifications VARCHAR(200) DEFAULT ''",
        "ALTER TABLE user ADD COLUMN gender VARCHAR(10) DEFAULT ''",
        "ALTER TABLE user ADD COLUMN patient_rating FLOAT DEFAULT 0.0",
        "ALTER TABLE user ADD COLUMN last_login DATETIME",
        "ALTER TABLE user ADD COLUMN is_active_acc BOOLEAN DEFAULT 1",
    ]
    with db.engine.connect() as conn:
        for stmt in user_new_cols:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass

    # ── Seed real doctors from dataset ─────────────────────────────────────
    DOCTORS_SEED = [
        {"code":"D01","name":"Dr. Rajesh Sharma",   "spec":"Cardiologist",              "exp":14, "qual":"MBBS, MD, DM Cardiology",            "hosp":"H01","gender":"Male",  "rating":4.8},
        {"code":"D02","name":"Dr. Priya Mehta",     "spec":"Interventional Cardiologist","exp":18, "qual":"MBBS, MD, DM Cardiology",            "hosp":"H01","gender":"Female","rating":4.9},
        {"code":"D03","name":"Dr. Anil Verma",      "spec":"General Physician",           "exp":10, "qual":"MBBS, MD General Medicine",           "hosp":"H01","gender":"Male",  "rating":4.5},
        {"code":"D04","name":"Dr. Sunita Rao",      "spec":"Cardiologist",              "exp":22, "qual":"MBBS, MD, DM Cardiology",            "hosp":"H02","gender":"Female","rating":4.7},
        {"code":"D05","name":"Dr. Vikram Nair",     "spec":"Cardiac Surgeon",             "exp":16, "qual":"MBBS, MS, MCh Cardiac Surgery",        "hosp":"H02","gender":"Male",  "rating":4.9},
        {"code":"D06","name":"Dr. Kavita Joshi",    "spec":"Cardiologist",              "exp":12, "qual":"MBBS, MD, DM Cardiology",            "hosp":"H03","gender":"Female","rating":4.6},
        {"code":"D07","name":"Dr. Suresh Iyer",     "spec":"General Physician",           "exp":8,  "qual":"MBBS, MD General Medicine",           "hosp":"H03","gender":"Male",  "rating":4.3},
        {"code":"D08","name":"Dr. Meena Gupta",     "spec":"Electrophysiologist",         "exp":20, "qual":"MBBS, MD, DM Cardiology, EP Fellowship","hosp":"H03","gender":"Female","rating":4.8},
        {"code":"D09","name":"Dr. Ramesh Patel",    "spec":"Cardiologist",              "exp":11, "qual":"MBBS, MD, DM Cardiology",            "hosp":"H04","gender":"Male",  "rating":4.5},
        {"code":"D10","name":"Dr. Anita Singh",     "spec":"General Physician",           "exp":9,  "qual":"MBBS, MD General Medicine",           "hosp":"H04","gender":"Female","rating":4.4},
        {"code":"D11","name":"Dr. Kiran Bose",      "spec":"Cardiologist",              "exp":25, "qual":"MBBS, MD, DM Cardiology",            "hosp":"H05","gender":"Male",  "rating":5.0},
        {"code":"D12","name":"Dr. Leela Chopra",    "spec":"Cardiac Rehabilitation",     "exp":13, "qual":"MBBS, MD, Cert. Cardiac Rehab",       "hosp":"H05","gender":"Female","rating":4.6},
        {"code":"D13","name":"Dr. Sanjay Malhotra", "spec":"Cardiologist",              "exp":17, "qual":"MBBS, MD, DM Cardiology",            "hosp":"H06","gender":"Male",  "rating":4.7},
        {"code":"D14","name":"Dr. Deepa Krishnan",  "spec":"General Physician",           "exp":7,  "qual":"MBBS, MD General Medicine",           "hosp":"H07","gender":"Female","rating":4.2},
        {"code":"D15","name":"Dr. Naresh Tiwari",   "spec":"Cardiac Surgeon",             "exp":21, "qual":"MBBS, MS, MCh Cardiac Surgery",        "hosp":"H08","gender":"Male",  "rating":4.8},
    ]
    HOSPITALS_META = {
        "H01":{"name":"Apollo Heart Institute",     "city":"Delhi",    "region":"North India","beds":850, "rating":4.8,"type":"Multi-Specialty",          "icu":"Yes","accr":"ISO 9001:2015",   "est":1998},
        "H02":{"name":"Fortis Cardiac Care",        "city":"Mumbai",   "region":"West India", "beds":620, "rating":4.7,"type":"Cardiac Specialty",         "icu":"Yes","accr":"NABH Accredited",  "est":2002},
        "H03":{"name":"AIIMS Cardiology Dept",      "city":"Chennai",  "region":"South India","beds":1200,"rating":4.9,"type":"Government Multi-Specialty","icu":"Yes","accr":"NABL Accredited",  "est":1956},
        "H04":{"name":"Max Super Specialty",        "city":"Bangalore","region":"South India","beds":780, "rating":4.6,"type":"Multi-Specialty",          "icu":"Yes","accr":"JCI Accredited",   "est":2000},
        "H05":{"name":"Medanta Heart Institute",    "city":"Gurugram", "region":"North India","beds":1050,"rating":4.9,"type":"Cardiac Specialty",         "icu":"Yes","accr":"ISO 9001:2015",   "est":2009},
        "H06":{"name":"Narayana Health",            "city":"Kolkata",  "region":"East India", "beds":540, "rating":4.7,"type":"Multi-Specialty",          "icu":"Yes","accr":"NABH Accredited",  "est":2001},
        "H07":{"name":"Wockhardt Hospital",         "city":"Hyderabad","region":"South India","beds":420, "rating":4.4,"type":"Multi-Specialty",          "icu":"No", "accr":"ISO 9001:2015",   "est":1999},
        "H08":{"name":"Ruby Hall Clinic",           "city":"Pune",     "region":"West India", "beds":380, "rating":4.5,"type":"Multi-Specialty",          "icu":"Yes","accr":"NABH Accredited",  "est":1959},
    }
    DEFAULT_PW = bcrypt.generate_password_hash("CardioAI@2024").decode("utf-8")
    ADMIN_PW   = bcrypt.generate_password_hash("Admin@2024").decode("utf-8")

    # Create admin if not exists
    if not User.query.filter_by(username="admin").first():
        db.session.add(User(username="admin", password_hash=ADMIN_PW, role="admin",
                            full_name="System Administrator"))
        db.session.commit()

    # Create each doctor if not exists
    for d in DOCTORS_SEED:
        uname = d["code"].lower()  # d01 … d15
        if not User.query.filter_by(username=uname).first():
            db.session.add(User(
                username=uname, password_hash=DEFAULT_PW, role="doctor",
                full_name=d["name"], doctor_code=d["code"],
                hospital_id=d["hosp"], specialization=d["spec"],
                experience_years=d["exp"], qualifications=d["qual"],
                gender=d["gender"], patient_rating=d["rating"],
                is_active_acc=True
            ))
    db.session.commit()

    # Store hospitals + doctors meta globally for fast API access
    import json as _json
    app.config["HOSPITALS_META"] = HOSPITALS_META
    app.config["DOCTORS_SEED"]   = DOCTORS_SEED

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

# ─── Build SHAP Explainers ──────────────────────────────────────────────────
def create_shap_explainers():
    explainers = {}
    X_train_path = os.path.join(BASE, 'models/X_train.npy')
    X_train = np.load(X_train_path) if os.path.exists(X_train_path) else None

    for name, info in MODELS.items():
        mdl = info['model']
        try:
            if name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
                explainers[name] = shap.TreeExplainer(mdl)
                print(f"  ✅ SHAP TreeExplainer ready: {name}")
            elif name == 'Logistic Regression' and X_train is not None:
                explainers[name] = shap.LinearExplainer(mdl, X_train)
                print(f"  ✅ SHAP LinearExplainer ready: {name}")
            else:
                print(f"  ⚠️  SHAP skipped (too slow for web): {name}")
        except Exception as e:
            print(f"  ❌ SHAP failed for {name}: {e}")
    return explainers

SHAP_EXPLAINERS = create_shap_explainers()

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
# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'patient':
        profile = PatientProfile.query.filter_by(user_id=current_user.id).first()
        return render_template('patient_dashboard.html', user=current_user, profile=profile)
    if current_user.role == 'admin':
        return render_template('admin_dashboard.html', user=current_user)
    # Doctor
    hosp = app.config.get('HOSPITALS_META', {}).get(current_user.hospital_id, {})
    return render_template('index.html', user=current_user, hospital=hosp)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role     = request.form.get('role', 'doctor')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
            
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, password_hash=hashed_pw, role=role)
        db.session.add(user)
        db.session.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    role_hint = request.args.get('role', '')
    return render_template('auth.html', action='register', role_hint=role_hint)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'error')
    role_hint = request.args.get('role', 'doctor')
    return render_template('auth.html', action='login', role_hint=role_hint)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/api/predict', methods=['POST'])
@login_required
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

        # Save prediction to history
        if current_user.is_authenticated:
            try:
                hist = PatientHistory(
                    doctor_id        = current_user.id,
                    patient_label    = data.get('patient_label', '').strip(),
                    age              = int(data['age']),
                    sex              = data['sex'],
                    risk_score       = risk_score,
                    risk_band        = risk_band,
                    avg_prob         = round(avg_prob * 100, 2),
                    votes_yes        = votes_yes,
                    final_prediction = 'Disease' if final_pred == 1 else 'Healthy'
                )
                db.session.add(hist)
                db.session.commit()
                # Fire high-risk notification if disease detected
                if final_pred == 1 and risk_score >= 55:
                    label = data.get('patient_label', 'Unlabelled Patient')
                    create_notification(
                        current_user.id,
                        title=f'⚠️ High-Risk Patient — {label or "Patient"}',
                        message=f'Risk Score {risk_score}/100 ({risk_band}). Run SHAP analysis and export a clinical report.',
                        ntype='danger'
                    )
            except Exception as e:
                print(f"Error saving patient history: {e}")

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
@login_required
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

    # New Data for Advanced Charts
    age_disease = df[df['Target'] == 1]['Age'].tolist()
    age_healthy = df[df['Target'] == 0]['Age'].tolist()

    # We can sample the scatter data to max 1000 points to ensure hyper-fast rendering without lag
    df_sample = df.sample(n=min(1500, len(df)), random_state=42)
    s_disease = df_sample[df_sample['Target'] == 1][['Cholesterol', 'Max_Heart_Rate']].to_dict(orient='records')
    s_healthy = df_sample[df_sample['Target'] == 0][['Cholesterol', 'Max_Heart_Rate']].to_dict(orient='records')

    return jsonify({
        'total': total,
        'disease': disease,
        'healthy': healthy,
        'disease_pct': disease_pct,
        'age_rate': age_rate[['Age_Group', 'Target']].rename(
            columns={'Target': 'rate'}).to_dict(orient='records'),
        'gender_stats': gender_stats,
        'risk_dist': risk_dist,
        'lifestyle_rates': lifestyle_rates,
        'age_dist': {
            'disease': age_disease,
            'healthy': age_healthy
        },
        'scatter_data': {
            'disease': s_disease,
            'healthy': s_healthy
        }
    })

@app.route('/api/prediction-stats')
@login_required
def prediction_stats():
    """Return prediction counts for the current logged-in doctor."""
    from datetime import date
    today_start = datetime.combine(date.today(), datetime.min.time())
    total_by_doctor = PatientHistory.query.filter_by(doctor_id=current_user.id).count()
    today_by_doctor = PatientHistory.query.filter(
        PatientHistory.doctor_id == current_user.id,
        PatientHistory.timestamp >= today_start
    ).count()
    disease_count = PatientHistory.query.filter_by(
        doctor_id=current_user.id, final_prediction='Disease'
    ).count()
    return jsonify({
        'total': total_by_doctor,
        'today': today_by_doctor,
        'disease': disease_count,
        'healthy': total_by_doctor - disease_count
    })


@app.route('/api/history')
@login_required
def get_history():
    """Return paginated, filterable prediction history for logged-in doctor."""
    from datetime import date as date_type
    start   = request.args.get('start')
    end     = request.args.get('end')
    risk    = request.args.get('risk', '')       # e.g. 'Disease' or 'Healthy'
    page    = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))

    q = PatientHistory.query.filter_by(doctor_id=current_user.id)
    if start:
        try:
            q = q.filter(PatientHistory.timestamp >= datetime.strptime(start, '%Y-%m-%d'))
        except ValueError:
            pass
    if end:
        try:
            end_dt = datetime.strptime(end, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
            q = q.filter(PatientHistory.timestamp <= end_dt)
        except ValueError:
            pass
    if risk in ('Disease', 'Healthy'):
        q = q.filter_by(final_prediction=risk)

    total   = q.count()
    records = q.order_by(PatientHistory.timestamp.desc()) \
               .offset((page - 1) * per_page).limit(per_page).all()

    def to_dict(r):
        return {
            'id':               r.id,
            'patient_label':    r.patient_label or '',
            'age':              r.age,
            'sex':              r.sex,
            'risk_score':       r.risk_score,
            'risk_band':        r.risk_band or '—',
            'avg_prob':         r.avg_prob,
            'votes_yes':        r.votes_yes,
            'final_prediction': r.final_prediction,
            'timestamp':        r.timestamp.strftime('%d %b %Y, %H:%M') if r.timestamp else '—',
            'timestamp_iso':    r.timestamp.isoformat() if r.timestamp else None,
        }

    # Trend data: all records (unfiltered except by doctor) sorted ascending for chart
    trend_q = PatientHistory.query.filter_by(doctor_id=current_user.id) \
                .order_by(PatientHistory.timestamp.asc()).limit(200).all()
    trend = [{'ts': r.timestamp.isoformat(), 'risk': r.risk_score,
              'pred': r.final_prediction} for r in trend_q if r.timestamp]

    return jsonify({
        'records': [to_dict(r) for r in records],
        'total':   total,
        'page':    page,
        'pages':   max(1, (total + per_page - 1) // per_page),
        'trend':   trend
    })


@app.route('/api/history/<int:record_id>', methods=['DELETE'])
@login_required
def delete_history(record_id):
    rec = PatientHistory.query.filter_by(id=record_id, doctor_id=current_user.id).first()
    if not rec:
        return jsonify({'success': False, 'error': 'Not found'}), 404
    db.session.delete(rec)
    db.session.commit()
    return jsonify({'success': True})


@app.route('/api/history/export-csv')
@login_required
def export_history_csv():
    """Stream the doctor's full prediction history as a downloadable CSV."""
    import csv, io
    start = request.args.get('start')
    end   = request.args.get('end')
    risk  = request.args.get('risk', '')

    q = PatientHistory.query.filter_by(doctor_id=current_user.id)
    if start:
        try: q = q.filter(PatientHistory.timestamp >= datetime.strptime(start, '%Y-%m-%d'))
        except ValueError: pass
    if end:
        try:
            q = q.filter(PatientHistory.timestamp <=
                         datetime.strptime(end, '%Y-%m-%d').replace(hour=23,minute=59,second=59))
        except ValueError: pass
    if risk in ('Disease', 'Healthy'):
        q = q.filter_by(final_prediction=risk)

    records = q.order_by(PatientHistory.timestamp.desc()).all()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['#', 'Date', 'Time', 'Patient Label', 'Age', 'Sex',
                     'Risk Score', 'Risk Band', 'Disease Probability (%)',
                     'Model Votes (Disease)', 'Verdict'])
    for i, r in enumerate(records, 1):
        ts = r.timestamp.strftime('%d/%m/%Y') if r.timestamp else ''
        tm = r.timestamp.strftime('%H:%M') if r.timestamp else ''
        writer.writerow([
            i, ts, tm, r.patient_label or '',
            r.age, r.sex, r.risk_score, r.risk_band or '',
            f'{r.avg_prob:.1f}' if r.avg_prob else '',
            f'{r.votes_yes}/6' if r.votes_yes is not None else '',
            r.final_prediction
        ])

    buf.seek(0)
    from flask import Response
    fname = f'CardioAI_History_{current_user.username}_{datetime.now().strftime("%Y%m%d")}.csv'
    return Response(
        buf.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{fname}"'}
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PATIENT ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/patient/profile', methods=['GET', 'POST'])
@login_required
def patient_profile():
    if current_user.role != 'patient':
        return jsonify({'success': False, 'error': 'Not a patient account'}), 403
    if request.method == 'POST':
        data = request.get_json()
        prof = PatientProfile.query.filter_by(user_id=current_user.id).first()
        if not prof:
            prof = PatientProfile(user_id=current_user.id)
            db.session.add(prof)
        prof.full_name     = data.get('full_name', '').strip()
        prof.dob           = data.get('dob', '').strip()
        prof.sex           = data.get('sex', '').strip()
        prof.phone         = data.get('phone', '').strip()
        prof.email         = data.get('email', '').strip()
        prof.notify_email  = bool(data.get('notify_email', True))
        prof.updated_at    = datetime.utcnow()
        db.session.commit()
        return jsonify({'success': True})
    prof = PatientProfile.query.filter_by(user_id=current_user.id).first()
    if not prof:
        return jsonify({'full_name': '', 'dob': '', 'sex': '', 'phone': '', 'email': '', 'notify_email': True})
    return jsonify({
        'full_name': prof.full_name, 'dob': prof.dob, 'sex': prof.sex,
        'phone': prof.phone, 'email': prof.email or '',
        'notify_email': prof.notify_email if prof.notify_email is not None else True
    })


@app.route('/api/patient/predict', methods=['POST'])
@login_required
def patient_predict():
    """Same ML engine as /api/predict but returns patient-friendly output."""
    if current_user.role != 'patient':
        return jsonify({'success': False, 'error': 'Not a patient account'}), 403
    try:
        data = request.get_json()

        # Fill clinical defaults for fields the patient may not have provided
        defaults = {
            'trestbps': 130, 'chol': 220, 'fbs': 0, 'restecg': 0,
            'thalachh': 150, 'exang': 0, 'oldpeak': 1.0, 'slope': 1,
            'ca': 0, 'thal': 3
        }
        for k, v in defaults.items():
            if k not in data or data[k] == '' or data[k] is None:
                data[k] = v

        X_raw, X_scaled = build_input(data)
        preds = {}
        for name, info in MODELS.items():
            X_in = X_scaled if info['scaled'] else X_raw
            pred = int(info['model'].predict(X_in)[0])
            prob = info['model'].predict_proba(X_in)[0].tolist()
            preds[name] = {'pred': pred, 'prob_disease': prob[1], 'prob_healthy': prob[0]}

        avg_prob   = float(np.mean([v['prob_disease'] for v in preds.values()]))
        votes_yes  = sum(v['pred'] == 1 for v in preds.values())
        final_pred = 1 if votes_yes > len(preds) / 2 else 0
        risk_score = clinical_risk_score(data)

        # Patient-friendly health score (inverted — higher = healthier)
        health_score = 100 - risk_score

        # Friendly verdict levels
        if health_score >= 75:
            verdict = 'Good'; verdict_color = 'green'
            verdict_msg = 'Your heart health looks good. Keep maintaining your healthy habits!'
        elif health_score >= 55:
            verdict = 'Moderate'; verdict_color = 'amber'
            verdict_msg = 'Some areas need attention. Small lifestyle changes can make a big difference.'
        elif health_score >= 35:
            verdict = 'Needs Attention'; verdict_color = 'orange'
            verdict_msg = 'Our analysis found several risk factors. Please consult your doctor soon.'
        else:
            verdict = 'High Concern'; verdict_color = 'red'
            verdict_msg = 'Our analysis found significant risk factors. Please see a doctor as soon as possible.'

        # Risk band mapping
        if risk_score < 25:   risk_band = 'Very Low Risk'
        elif risk_score < 40: risk_band = 'Low Risk'
        elif risk_score < 55: risk_band = 'Moderate Risk'
        elif risk_score < 70: risk_band = 'High Risk'
        else:                 risk_band = 'Very High Risk'

        # Plain-English SHAP-based top factors
        plain_factors = []
        try:
            avail = [n for n in ['Gradient Boosting', 'Random Forest', 'Decision Tree']
                     if n in SHAP_EXPLAINERS]
            if avail:
                exp    = SHAP_EXPLAINERS[avail[0]]
                X_in2  = X_raw.values if avail[0] in ['Gradient Boosting','Random Forest','Decision Tree'] else X_scaled
                sv     = exp.shap_values(X_in2)
                if isinstance(sv, list): sv = sv[1]
                if hasattr(sv, 'ndim') and sv.ndim == 3: sv = sv[0,:,1]
                elif hasattr(sv, 'ndim') and sv.ndim == 2: sv = sv[0]
                sv = np.array(sv, dtype=float)

                FRIENDLY_LABELS = {
                    'Smoking_Status':           ('Smoking',            'Your smoking habit is increasing your risk.'),
                    'Max_Heart_Rate':            ('Max Heart Rate',     'Your maximum heart rate during exercise is a key indicator.'),
                    'Age':                       ('Age',                'Age is a natural risk factor for heart disease.'),
                    'Chest_Pain_Type':           ('Chest Pain',         'The type of chest discomfort you experience matters.'),
                    'Exercise_Induced_Angina':   ('Exercise Symptoms',  'Breathlessness or pain during exercise is a warning sign.'),
                    'Cholesterol':               ('Cholesterol',        'Your cholesterol level affects heart health significantly.'),
                    'Trestbps':                  ('Blood Pressure',     'Your resting blood pressure is above the ideal range.'),
                    'ST_Depression':             ('ECG Reading',        'Your ECG shows some electrical changes in the heart.'),
                    'Major_Vessels':             ('Blood Vessels',      'The number of blocked vessels detected is a strong indicator.'),
                    'Thalassemia':               ('Thalassemia',        'Your thalassemia type affects blood oxygen supply.'),
                    'BMI_Category':              ('Body Weight',        'Your BMI category is affecting your cardiovascular risk.'),
                    'Exercise_Level':            ('Exercise Habits',    'Low physical activity increases your heart risk.'),
                    'Alcohol_Consumption':       ('Alcohol',            'Alcohol consumption is contributing to your risk score.'),
                    'Fasting_Blood_Sugar':       ('Blood Sugar',        'Elevated fasting blood sugar is a diabetes indicator.'),
                    'HR_Reserve':                ('Heart Rate Reserve', 'Your heart is working harder than expected for your age.'),
                    'BP_Chol_Score':             ('BP + Cholesterol',   'The combination of your BP and cholesterol is elevated.'),
                    'Age_Sex_Interact':          ('Age & Sex Profile',  'Your age and sex combination affects your risk profile.'),
                }
                pairs = list(zip(feature_cols, sv))
                top3  = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:3]
                for feat, val in top3:
                    if feat in FRIENDLY_LABELS:
                        label, desc = FRIENDLY_LABELS[feat]
                        plain_factors.append({
                            'label': label, 'desc': desc,
                            'direction': 'risk' if val > 0 else 'protective',
                            'magnitude': abs(float(val))
                        })
        except Exception as e:
            print(f'Patient SHAP error: {e}')

        # Friendly recommendations
        chol     = int(data.get('chol', 220))
        trestbps = int(data.get('trestbps', 130))
        smoking  = data.get('smoking', 'Non-Smoker')
        bmi_cat  = data.get('bmi', 'Normal')
        exercise = data.get('exercise', 'Moderate')
        alcohol  = data.get('alcohol', 'None')

        friendly_recs = []
        if final_pred == 1 or risk_score >= 55:
            friendly_recs.append({'icon': '🏥', 'title': 'See a Doctor Soon',
                'desc': 'Our AI has detected significant risk factors. Please schedule an appointment with your doctor.'})
        if smoking == 'Current Smoker':
            friendly_recs.append({'icon': '🚭', 'title': 'Quit Smoking',
                'desc': 'Smoking is one of the single biggest risk factors for heart disease. Even cutting down helps immediately.'})
        if bmi_cat in ('Obese', 'Overweight'):
            friendly_recs.append({'icon': '🥗', 'title': 'Healthy Weight',
                'desc': 'Even a 5–10% reduction in body weight can significantly reduce cardiovascular risk.'})
        if exercise == 'Low':
            friendly_recs.append({'icon': '🚶', 'title': 'Move More',
                'desc': 'Aim for 30 minutes of moderate exercise 5 days a week. Walking counts!'})
        if chol > 240:
            friendly_recs.append({'icon': '🥦', 'title': 'Lower Your Cholesterol',
                'desc': 'Cut down on saturated fats, eat more fibre, and consider talking to your doctor about medication.'})
        if trestbps > 140:
            friendly_recs.append({'icon': '🧘', 'title': 'Manage Blood Pressure',
                'desc': 'Reduce salt intake, manage stress, and monitor your BP regularly at home.'})
        if alcohol == 'High':
            friendly_recs.append({'icon': '🍹', 'title': 'Reduce Alcohol',
                'desc': 'Heavy drinking strains the heart. Try to stay within recommended weekly limits.'})
        if len(friendly_recs) == 0:
            friendly_recs.append({'icon': '✅', 'title': 'Keep It Up!',
                'desc': 'Your lifestyle choices are good. Keep up regular exercise, a balanced diet, and routine checkups.'})

        # Always append doctor consultation advice
        friendly_recs.append({'icon': '👨‍⚕️', 'title': 'Always Consult Your Doctor',
            'desc': 'This AI tool is for awareness only and is not a medical diagnosis. Please consult a licensed doctor for proper evaluation.'})

        # Save to history
        try:
            prof = PatientProfile.query.filter_by(user_id=current_user.id).first()
            label = prof.full_name if prof and prof.full_name else ''
            hist = PatientHistory(
                doctor_id        = current_user.id,
                patient_label    = label,
                age              = int(data.get('age', 0)),
                sex              = data.get('sex', ''),
                risk_score       = risk_score,
                risk_band        = risk_band,
                avg_prob         = round(avg_prob * 100, 2),
                votes_yes        = votes_yes,
                final_prediction = 'Disease' if final_pred == 1 else 'Healthy'
            )
            db.session.add(hist)
            db.session.commit()
            # Fire notification + email if high risk
            if health_score <= 40:
                create_notification(
                    current_user.id,
                    title='⚠️ Your Assessment — High Concern',
                    message=f'Your heart health score is {health_score}/100. Please consult a doctor.',
                    ntype='danger'
                )
                # Email alert if opted in
                prof2 = PatientProfile.query.filter_by(user_id=current_user.id).first()
                if prof2 and prof2.email and prof2.notify_email:
                    send_high_risk_email(
                        prof2.email,
                        prof2.full_name or current_user.username,
                        risk_score, risk_band, 'CardioAI'
                    )
        except Exception as e:
            print(f'Patient history save error: {e}')

        # Vitals comparison (only if lab data provided)
        has_labs = data.get('has_labs', False)
        vitals = []
        if has_labs:
            vitals = [
                {'label': 'Blood Pressure', 'value': data.get('trestbps', '—'),
                 'unit': 'mmHg', 'ideal': '< 120', 'status': 'ok' if int(data.get('trestbps',120)) < 130 else 'warn' if int(data.get('trestbps',130)) < 140 else 'bad'},
                {'label': 'Cholesterol', 'value': data.get('chol', '—'),
                 'unit': 'mg/dL', 'ideal': '< 200', 'status': 'ok' if int(data.get('chol',200)) < 200 else 'warn' if int(data.get('chol',200)) < 240 else 'bad'},
                {'label': 'Max Heart Rate', 'value': data.get('thalachh', '—'),
                 'unit': 'bpm', 'ideal': '> 150', 'status': 'ok' if int(data.get('thalachh',150)) > 150 else 'warn' if int(data.get('thalachh',150)) > 120 else 'bad'},
                {'label': 'Blood Sugar', 'value': 'High' if int(data.get('fbs',0)) else 'Normal',
                 'unit': '', 'ideal': 'Normal', 'status': 'bad' if int(data.get('fbs',0)) else 'ok'},
            ]

        return jsonify({
            'success':       True,
            'health_score':  health_score,
            'risk_score':    risk_score,
            'verdict':       verdict,
            'verdict_color': verdict_color,
            'verdict_msg':   verdict_msg,
            'final_pred':    final_pred,
            'avg_prob':      round(avg_prob * 100, 1),
            'top_factors':   plain_factors,
            'recommendations': friendly_recs,
            'vitals':        vitals,
            'has_labs':      has_labs,
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/patient/history')
@login_required
def patient_history():
    if current_user.role != 'patient':
        return jsonify({'success': False, 'error': 'Not a patient account'}), 403
    recs = PatientHistory.query.filter_by(doctor_id=current_user.id) \
               .order_by(PatientHistory.timestamp.asc()).all()
    trend = [{'ts':   r.timestamp.strftime('%d %b'),
              'hs':   100 - (r.risk_score or 0),
              'pred': r.final_prediction}
             for r in recs if r.timestamp]
    latest = None
    if recs:
        r = recs[-1]
        latest = {'health_score': 100 - (r.risk_score or 0),
                  'verdict': r.final_prediction,
                  'date': r.timestamp.strftime('%d %b %Y')}
    prev = None
    if len(recs) >= 2:
        prev = 100 - (recs[-2].risk_score or 0)
    return jsonify({'trend': trend, 'total': len(recs), 'latest': latest, 'prev': prev})


@app.route('/api/patient/export-pdf', methods=['POST'])
@login_required
def patient_export_pdf():
    if current_user.role != 'patient':
        return jsonify({'success': False, 'error': 'Not a patient account'}), 403
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)

        data         = request.get_json()
        health_score = int(data.get('health_score', 0))
        verdict      = data.get('verdict', '')
        verdict_msg  = data.get('verdict_msg', '')
        recs         = data.get('recommendations', [])
        top_factors  = data.get('top_factors', [])
        vitals       = data.get('vitals', [])
        prof_name    = data.get('patient_name', '')
        prof_dob     = data.get('dob', '')
        prof_sex     = data.get('sex', '')

        report_id   = datetime.now().strftime('PAT-%Y%m%d-%H%M%S')
        report_date = datetime.now().strftime('%d %B %Y  %H:%M')

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=18*mm, rightMargin=18*mm,
                                topMargin=15*mm, bottomMargin=15*mm)
        W      = 174*mm
        styles = getSampleStyleSheet()

        def ps(name, **kw):
            return ParagraphStyle(name, parent=styles['Normal'], **kw)

        # Colours
        TEAL   = colors.HexColor('#0F766E')
        GREEN  = colors.HexColor('#16A34A')
        AMBER  = colors.HexColor('#D97706')
        RED    = colors.HexColor('#DC2626')
        SLATE  = colors.HexColor('#1E293B')
        LIGHT  = colors.HexColor('#F1F5F9')
        BORDER = colors.HexColor('#CBD5E1')
        WHITE  = colors.white
        DARK   = colors.HexColor('#0F172A')
        SUB    = colors.HexColor('#475569')

        score_color = GREEN if health_score >= 75 else AMBER if health_score >= 55 else RED

        story = []

        # ── Header ────────────────────────────────────────────────────────
        hdr = Table([[
            Paragraph('<b>CardioAI Health</b>',
                      ps('ht', fontSize=20, textColor=WHITE, fontName='Helvetica-Bold')),
            Paragraph(f'<b>Personal Heart Health Summary</b><br/>'
                      f'Report ID: {report_id}<br/>Date: {report_date}',
                      ps('hs', fontSize=9, textColor=colors.HexColor('#CBD5E1'),
                         alignment=2, fontName='Helvetica'))
        ]], colWidths=[W*0.45, W*0.55])
        hdr.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), SLATE),
            ('TOPPADDING',    (0,0),(-1,-1), 10), ('BOTTOMPADDING',(0,0),(-1,-1), 10),
            ('LEFTPADDING',   (0,0),(-1,-1), 14), ('RIGHTPADDING', (0,0),(-1,-1), 14),
            ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ]))
        story.append(hdr); story.append(Spacer(1, 5*mm))

        # ── Patient Info ──────────────────────────────────────────────────
        info_rows = [[
            Paragraph(f'<b>Name:</b> {prof_name or "—"}',
                      ps('i', fontSize=10, textColor=DARK, fontName='Helvetica')),
            Paragraph(f'<b>Date of Birth:</b> {prof_dob or "—"}',
                      ps('i', fontSize=10, textColor=DARK, fontName='Helvetica')),
            Paragraph(f'<b>Sex:</b> {prof_sex or "—"}',
                      ps('i', fontSize=10, textColor=DARK, fontName='Helvetica')),
        ]]
        info_tbl = Table(info_rows, colWidths=[W/3, W/3, W/3])
        info_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(-1,-1), LIGHT),
            ('TOPPADDING', (0,0),(-1,-1), 8), ('BOTTOMPADDING',(0,0),(-1,-1), 8),
            ('LEFTPADDING',(0,0),(-1,-1), 10), ('BOX',(0,0),(-1,-1),0.5,BORDER),
        ]))
        story.append(info_tbl); story.append(Spacer(1, 5*mm))

        # ── Health Score Banner ───────────────────────────────────────────
        score_tbl = Table([[
            Paragraph(f'<b>Heart Health Score</b>',
                      ps('sl', fontSize=11, textColor=SUB, fontName='Helvetica-Bold')),
            Paragraph(f'<b>{health_score}/100</b>',
                      ps('sv', fontSize=28, textColor=score_color, fontName='Helvetica-Bold', alignment=1)),
            Paragraph(f'<b>{verdict}</b><br/>{verdict_msg}',
                      ps('sm', fontSize=10, textColor=DARK, fontName='Helvetica', leading=15))
        ]], colWidths=[W*0.28, W*0.2, W*0.52])
        score_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), LIGHT),
            ('TOPPADDING',    (0,0),(-1,-1), 12), ('BOTTOMPADDING',(0,0),(-1,-1), 12),
            ('LEFTPADDING',   (0,0),(-1,-1), 12), ('RIGHTPADDING', (0,0),(-1,-1), 12),
            ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
            ('BOX',           (0,0),(-1,-1), 1.5, score_color),
            ('LINEAFTER',     (0,0),(0,0),   0.5, BORDER),
            ('LINEAFTER',     (1,0),(1,0),   0.5, BORDER),
        ]))
        story.append(score_tbl); story.append(Spacer(1, 5*mm))

        # ── Top Risk Factors ──────────────────────────────────────────────
        if top_factors:
            story.append(Paragraph('Your Top Risk Factors',
                                   ps('h2', fontSize=12, textColor=SLATE, fontName='Helvetica-Bold',
                                      spaceBefore=4, spaceAfter=4)))
            story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
            story.append(Spacer(1, 2*mm))
            rf_style = ps('rf', fontSize=9, textColor=DARK, fontName='Helvetica', leading=14, leftIndent=8, spaceAfter=4)
            for f in top_factors:
                icon = '[!]' if f['direction'] == 'risk' else '[+]'
                col  = RED if f['direction'] == 'risk' else GREEN
                story.append(Paragraph(
                    f'<b>{icon}  {f["label"]}</b>  —  {f["desc"]}',
                    rf_style
                ))
            story.append(Spacer(1, 4*mm))

        # ── Vitals ───────────────────────────────────────────────────────
        if vitals:
            story.append(Paragraph('Your Key Vitals',
                                   ps('h2v', fontSize=12, textColor=SLATE, fontName='Helvetica-Bold',
                                      spaceBefore=4, spaceAfter=4)))
            story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
            story.append(Spacer(1, 2*mm))
            vrows = [['Vital', 'Your Value', 'Ideal Range', 'Status']]
            for v in vitals:
                sc = '[OK] Good' if v['status'] == 'ok' else '[!] Watch' if v['status'] == 'warn' else '[X] Attention'
                tc = GREEN if v['status'] == 'ok' else AMBER if v['status'] == 'warn' else RED
                vrows.append([v['label'], f"{v['value']} {v['unit']}", v['ideal'], sc])
            vtbl = Table(vrows, colWidths=[W*0.3, W*0.25, W*0.25, W*0.2])
            vstyle = TableStyle([
                ('BACKGROUND',    (0,0),(-1,0), SLATE), ('TEXTCOLOR',(0,0),(-1,0), WHITE),
                ('FONTNAME',      (0,0),(-1,0), 'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,-1), 9),
                ('ROWBACKGROUNDS',(0,1),(-1,-1), [WHITE, LIGHT]),
                ('GRID',          (0,0),(-1,-1), 0.3, BORDER),
                ('TOPPADDING',    (0,0),(-1,-1), 4), ('BOTTOMPADDING',(0,0),(-1,-1), 4),
                ('LEFTPADDING',   (0,0),(-1,-1), 6),
            ])
            for i, v in enumerate(vitals, 1):
                tc2 = GREEN if v['status']=='ok' else AMBER if v['status']=='warn' else RED
                vstyle.add('TEXTCOLOR', (3,i),(3,i), tc2)
                vstyle.add('FONTNAME',  (3,i),(3,i), 'Helvetica-Bold')
            vtbl.setStyle(vstyle)
            story.append(vtbl); story.append(Spacer(1, 5*mm))

        # ── Recommendations ───────────────────────────────────────────────
        story.append(Paragraph('Personalised Recommendations',
                               ps('h2r', fontSize=12, textColor=SLATE, fontName='Helvetica-Bold',
                                  spaceBefore=4, spaceAfter=4)))
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
        story.append(Spacer(1, 2*mm))
        rc_style = ps('rc', fontSize=9, textColor=DARK, fontName='Helvetica', leading=14, leftIndent=10, spaceAfter=5)
        for i, rec in enumerate(recs, 1):
            story.append(Paragraph(
                f'*  <b>{rec.get("title","")}</b>  —  {rec.get("desc","")}',
                rc_style
            ))

        story.append(Spacer(1, 5*mm))

        # ── Disclaimer ───────────────────────────────────────────────────
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            'This report is generated by CardioAI for personal health awareness only. '
            'It is NOT a medical diagnosis and does not replace consultation with a licensed doctor. '
            'CardioAI uses a 6-model machine learning ensemble trained on cardiovascular datasets.',
            ps('disc', fontSize=7.5, textColor=SUB, leading=10, fontName='Helvetica')
        ))

        doc.build(story)
        buf.seek(0)
        fname = f'CardioAI_MyHealthReport_{datetime.now().strftime("%Y%m%d")}.pdf'
        return send_file(buf, as_attachment=True, download_name=fname, mimetype='application/pdf')

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


@app.route('/api/shap', methods=['POST'])
@login_required
def shap_explain():
    try:
        data       = request.get_json()
        model_name = data.get('model', 'Gradient Boosting')

        # Fall back if requested model has no explainer
        if model_name not in SHAP_EXPLAINERS:
            available = list(SHAP_EXPLAINERS.keys())
            if not available:
                return jsonify({'success': False, 'error': 'No SHAP explainers loaded'}), 500
            model_name = available[0]

        X_raw, X_scaled = build_input(data)
        explainer = SHAP_EXPLAINERS[model_name]

        # Use scaled input for LR, raw for tree models
        X_in = X_scaled if MODELS[model_name]['scaled'] else X_raw.values

        shap_vals = explainer.shap_values(X_in)
        ev        = explainer.expected_value

        # ── Normalise shapes ────────────────────────────────────────────────
        # Modern SHAP TreeExplainer for RF/DT returns shape (n_samples, n_features, n_classes)
        # Older or single-output models return (n_samples, n_features) or a list of two arrays.

        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            # Shape: (samples, features, classes) — grab class=1 (Disease)
            sv   = shap_vals[0, :, 1]
            base = float(ev[1]) if hasattr(ev, '__len__') and len(ev) > 1 else float(ev)

        elif isinstance(shap_vals, list) and len(shap_vals) == 2:
            # Old TreeExplainer: [class0_array, class1_array], each shape (samples, features)
            sv   = np.array(shap_vals[1])[0]
            base = float(ev[1]) if hasattr(ev, '__len__') and len(ev) > 1 else float(ev)

        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            # LinearExplainer / single-output: (samples, features)
            sv   = shap_vals[0]
            if hasattr(ev, '__len__'):
                base = float(ev[1]) if len(ev) > 1 else float(ev[0])
            else:
                base = float(ev)

        else:
            # Fallback — try to flatten whatever we get
            sv   = np.array(shap_vals).flatten()[:len(feature_cols)]
            base = float(np.array(ev).flatten()[0])

        sv = np.array(sv, dtype=float)   # guarantee 1-D float array

        # Human-readable feature labels
        FEATURE_LABELS = {
            'Age': 'Age', 'Sex': 'Sex (M/F)',
            'Chest_Pain_Type': 'Chest Pain Type', 'Trestbps': 'Resting BP',
            'Cholesterol': 'Cholesterol', 'Fasting_Blood_Sugar': 'Fasting Blood Sugar',
            'Resting_ECG': 'Resting ECG', 'Max_Heart_Rate': 'Max Heart Rate',
            'Exercise_Induced_Angina': 'Exercise Angina', 'ST_Depression': 'ST Depression',
            'Slope': 'ST Slope', 'Major_Vessels': 'Major Vessels',
            'Thalassemia': 'Thalassemia', 'Smoking_Status': 'Smoking',
            'Alcohol_Consumption': 'Alcohol', 'Exercise_Level': 'Exercise Level',
            'BMI_Category': 'BMI Category', 'Age_Sex_Interact': 'Age × Sex',
            'BP_Chol_Score': 'BP × Chol Score', 'HR_Reserve': 'HR Reserve',
            'ST_Slope_Risk': 'ST × Slope Risk'
        }

        result = []
        raw_vals = X_raw.iloc[0]
        for fname, sval in zip(feature_cols, sv):
            result.append({
                'feature':  FEATURE_LABELS.get(fname, fname),
                'raw_name': fname,
                'value':    round(float(raw_vals.get(fname, 0)), 3),
                'shap':     round(float(sval), 5)
            })

        # Sort by absolute SHAP — most impactful first
        result.sort(key=lambda x: abs(x['shap']), reverse=True)

        return jsonify({
            'success':        True,
            'model':          model_name,
            'base_value':     round(base, 4),
            'features':       result[:15]
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


# ─── PDF Export ───────────────────────────────────────────────────────────────
@app.route('/api/export-pdf', methods=['POST'])
@login_required
def export_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable, KeepTogether)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

        data = request.get_json()

        # ── Pull prediction result fields from payload ─────────────────────
        final_pred   = int(data.get('final_pred', 0))
        avg_prob     = float(data.get('avg_prob', 0)) * 100
        risk_score   = int(data.get('risk_score', 0))
        risk_band    = data.get('risk_band', 'Unknown')
        votes_yes    = int(data.get('votes_yes', 0))
        total_models = int(data.get('total_models', 6))
        model_preds  = data.get('model_preds', {})
        recs         = data.get('recommendations', [])
        shap_feats   = data.get('shap_features', [])
        shap_model   = data.get('shap_model', 'Gradient Boosting')
        patient      = data.get('patient', {})

        report_id   = datetime.now().strftime('CAI-%Y%m%d-%H%M%S')
        report_date = datetime.now().strftime('%d %B %Y  %H:%M')

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=18*mm, rightMargin=18*mm,
                                topMargin=15*mm, bottomMargin=15*mm)

        # ── Colour palette ─────────────────────────────────────────────────
        CRIMSON  = colors.HexColor('#DC2626')
        DARKRED  = colors.HexColor('#991B1B')
        GREEN    = colors.HexColor('#16A34A')
        GOLD     = colors.HexColor('#D97706')
        SLATE    = colors.HexColor('#1E293B')
        LIGHTBG  = colors.HexColor('#F1F5F9')
        BORDER   = colors.HexColor('#CBD5E1')
        WHITE    = colors.white
        TEXTDARK = colors.HexColor('#0F172A')
        TEXTSUB  = colors.HexColor('#475569')

        styles = getSampleStyleSheet()

        def ps(name, parent='Normal', **kw):
            return ParagraphStyle(name, parent=styles[parent], **kw)

        title_s   = ps('title',   fontSize=22, textColor=WHITE,    alignment=TA_LEFT,   fontName='Helvetica-Bold', spaceAfter=0)
        sub_s     = ps('sub',     fontSize=9,  textColor=colors.HexColor('#CBD5E1'), alignment=TA_LEFT, fontName='Helvetica', spaceAfter=0)
        h2_s      = ps('h2',      fontSize=12, textColor=SLATE,    alignment=TA_LEFT,   fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=4)
        body_s    = ps('body',    fontSize=9,  textColor=TEXTDARK, alignment=TA_LEFT,   fontName='Helvetica',  leading=13)
        small_s   = ps('small',   fontSize=8,  textColor=TEXTSUB,  alignment=TA_LEFT,   fontName='Helvetica',  leading=11)
        center_s  = ps('center',  fontSize=9,  textColor=TEXTDARK, alignment=TA_CENTER, fontName='Helvetica')
        verdict_s = ps('verdict', fontSize=18, textColor=WHITE,    alignment=TA_CENTER, fontName='Helvetica-Bold', spaceAfter=0)
        rec_s     = ps('rec',     fontSize=9,  textColor=TEXTDARK, alignment=TA_LEFT,   fontName='Helvetica', leading=14, leftIndent=10)

        story = []
        W = 174*mm   # usable width

        # ═══════════════════ HEADER ═══════════════════════════════════════
        header_data = [[
            Paragraph('<b>❤ CardioAI</b>', title_s),
            Paragraph(f'<b>Clinical Cardiovascular Risk Report</b><br/>'
                      f'Report ID: {report_id}<br/>Generated: {report_date}', sub_s)
        ]]
        header_tbl = Table(header_data, colWidths=[W*0.4, W*0.6])
        header_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), SLATE),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING',   (0,0), (-1,-1), 12),
            ('RIGHTPADDING',  (0,0), (-1,-1), 12),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN',  (1,0), (1,0), 'RIGHT'),
        ]))
        story.append(header_tbl)
        story.append(Spacer(1, 6*mm))

        # ═══════════════════ VERDICT BANNER ═══════════════════════════════
        verdict_color = CRIMSON if final_pred == 1 else GREEN
        verdict_text  = ('⚠  HEART DISEASE DETECTED' if final_pred == 1
                         else '✔  NO HEART DISEASE DETECTED')
        sub_verdict   = (f'Ensemble vote: {votes_yes}/{total_models} models · '
                         f'Avg probability: {avg_prob:.1f}%')

        verd_tbl = Table([
            [Paragraph(verdict_text, verdict_s)],
            [Paragraph(sub_verdict,  center_s)],
        ], colWidths=[W])
        verd_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), verdict_color),
            ('TOPPADDING',    (0,0), (0,0), 12),
            ('BOTTOMPADDING', (0,0), (0,0), 4),
            ('TOPPADDING',    (0,1), (0,1), 2),
            ('BOTTOMPADDING', (0,1), (0,1), 12),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
            ('RIGHTPADDING',  (0,0), (-1,-1), 8),
            ('TEXTCOLOR',     (0,1), (0,1), WHITE),
        ]))
        story.append(verd_tbl)
        story.append(Spacer(1, 5*mm))

        # ═══════════════════ RISK SCORE BAR ═══════════════════════════════
        risk_color = CRIMSON if risk_score >= 55 else GOLD if risk_score >= 40 else GREEN
        bar_filled = risk_score / 100
        bar_data   = [[
            Paragraph(f'Clinical Risk Score', h2_s),
            Paragraph(f'<b>{risk_score}/100</b>  — {risk_band}', ps('rs', fontSize=12, textColor=risk_color, fontName='Helvetica-Bold'))
        ]]
        bar_tbl = Table(bar_data, colWidths=[W*0.5, W*0.5])
        bar_tbl.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(1,0),(1,0),'RIGHT')]))
        story.append(bar_tbl)

        # Visual bar
        bar_track = Table([['']], colWidths=[W], rowHeights=[8])
        bar_track.setStyle(TableStyle([('BACKGROUND',(0,0),(0,0), LIGHTBG),('LINEABOVE',(0,0),(0,0),0.5,BORDER)]))
        bar_fill  = Table([['']], colWidths=[W * bar_filled], rowHeights=[8])
        bar_fill.setStyle(TableStyle([('BACKGROUND',(0,0),(0,0), risk_color)]))

        # Wrap in a 2-col table
        bar_wrap = Table([[bar_fill, '']], colWidths=[W * bar_filled, W * (1 - bar_filled)], rowHeights=[8])
        bar_wrap.setStyle(TableStyle([
            ('BACKGROUND', (0,0),(0,0), risk_color),
            ('BACKGROUND', (1,0),(1,0), LIGHTBG),
            ('BOX', (0,0),(-1,-1), 0.5, BORDER),
        ]))
        story.append(bar_wrap)
        story.append(Spacer(1, 5*mm))

        # ═══════════════════ PATIENT INPUT SUMMARY ════════════════════════
        story.append(Paragraph('Patient Clinical Data', h2_s))
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
        story.append(Spacer(1, 2*mm))

        labels = {
            'age':'Age','sex':'Sex','cp':'Chest Pain Type','trestbps':'Resting BP (mmHg)',
            'chol':'Cholesterol (mg/dL)','thalachh':'Max Heart Rate','oldpeak':'ST Depression',
            'slope':'ST Slope','fbs':'Fasting Blood Sugar','restecg':'Resting ECG',
            'exang':'Exercise Angina','ca':'Major Vessels','thal':'Thalassemia',
            'smoking':'Smoking Status','alcohol':'Alcohol','exercise':'Exercise Level','bmi':'BMI Category'
        }
        cp_map = {'0':'Typical Angina','1':'Atypical Angina','2':'Non-Anginal Pain','3':'Asymptomatic'}
        slope_map = {'0':'Upsloping','1':'Flat','2':'Downsloping'}
        thal_map  = {'3':'Normal','6':'Fixed Defect','7':'Reversible Defect'}

        pat_items = []
        for k, label in labels.items():
            v = str(patient.get(k, '—'))
            if k == 'cp':    v = cp_map.get(v, v)
            if k == 'slope': v = slope_map.get(v, v)
            if k == 'thal':  v = thal_map.get(v, v)
            if k == 'sex':   v = 'Male' if v == '1' else 'Female' if v == '0' else v
            if k == 'fbs':   v = 'Yes (>120 mg/dL)' if v == '1' else 'No'
            if k == 'exang': v = 'Yes' if v == '1' else 'No'
            pat_items.append((label, v))

        # Split into two columns
        half = (len(pat_items) + 1) // 2
        left_col  = pat_items[:half]
        right_col = pat_items[half:]

        pat_rows = []
        for i in range(half):
            lk, lv = left_col[i] if i < len(left_col) else ('','')
            rk, rv = right_col[i] if i < len(right_col) else ('','')
            pat_rows.append([
                Paragraph(lk, small_s), Paragraph(f'<b>{lv}</b>', body_s),
                Paragraph(rk, small_s), Paragraph(f'<b>{rv}</b>', body_s),
            ])

        pat_tbl = Table(pat_rows, colWidths=[W*0.18, W*0.32, W*0.18, W*0.32])
        ts = TableStyle([
            ('FONTNAME',      (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE',      (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS',(0,0), (-1,-1), [WHITE, LIGHTBG]),
            ('TOPPADDING',    (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING',   (0,0), (-1,-1), 5),
            ('RIGHTPADDING',  (0,0), (-1,-1), 5),
            ('GRID',          (0,0), (-1,-1), 0.3, BORDER),
            ('TEXTCOLOR',     (0,0), (0,-1), TEXTSUB),
            ('TEXTCOLOR',     (2,0), (2,-1), TEXTSUB),
        ])
        pat_tbl.setStyle(ts)
        story.append(pat_tbl)
        story.append(Spacer(1, 5*mm))

        # ═══════════════════ MODEL RESULTS TABLE ══════════════════════════
        story.append(Paragraph('Ensemble Model Predictions', h2_s))
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
        story.append(Spacer(1, 2*mm))

        hdr = ['Model', 'Verdict', 'P(Disease)', 'P(Healthy)', 'Confidence']
        model_rows = [hdr]
        for mname, mdata in model_preds.items():
            verdict_str = '🔴 Disease' if mdata['pred'] == 1 else '🟢 Healthy'
            pd_val = mdata['prob_disease']
            ph_val = mdata['prob_healthy']
            conf   = max(pd_val, ph_val)
            model_rows.append([
                mname, verdict_str,
                f'{pd_val:.1f}%', f'{ph_val:.1f}%', f'{conf:.1f}%'
            ])

        m_tbl = Table(model_rows, colWidths=[W*0.28, W*0.18, W*0.18, W*0.18, W*0.18])
        m_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), SLATE),
            ('TEXTCOLOR',     (0,0), (-1,0), WHITE),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0,0), (-1,-1), 9),
            ('ALIGN',         (1,0), (-1,-1), 'CENTER'),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHTBG]),
            ('GRID',          (0,0), (-1,-1), 0.3, BORDER),
            ('TOPPADDING',    (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING',   (0,0), (-1,-1), 5),
        ]))
        story.append(m_tbl)
        story.append(Spacer(1, 5*mm))

        # ═══════════════════ SHAP FEATURE IMPACT ══════════════════════════
        if shap_feats:
            story.append(Paragraph(f'AI Explanation — SHAP Feature Impact  ({shap_model})', h2_s))
            story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
            story.append(Paragraph('Each value shows how much a feature pushed this prediction toward or away from Heart Disease.',
                                   small_s))
            story.append(Spacer(1, 2*mm))

            shap_rows = [['Feature', 'Patient Value', 'SHAP Impact', 'Direction']]
            for f in shap_feats[:12]:
                direction = '▲ Increases Risk' if f['shap'] > 0 else '▼ Reduces Risk'
                shap_rows.append([
                    f['feature'],
                    str(f['value']),
                    f"{f['shap']:+.4f}",
                    direction
                ])

            s_tbl = Table(shap_rows, colWidths=[W*0.32, W*0.20, W*0.20, W*0.28])
            s_ts = TableStyle([
                ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#374151')),
                ('TEXTCOLOR',     (0,0), (-1,0), WHITE),
                ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE',      (0,0), (-1,-1), 8.5),
                ('ALIGN',         (2,0), (2,-1), 'CENTER'),
                ('ALIGN',         (3,0), (3,-1), 'CENTER'),
                ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHTBG]),
                ('GRID',          (0,0), (-1,-1), 0.3, BORDER),
                ('TOPPADDING',    (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                ('LEFTPADDING',   (0,0), (-1,-1), 5),
            ])
            # colour code direction column
            for i, f in enumerate(shap_feats[:12], start=1):
                col = colors.HexColor('#FEE2E2') if f['shap'] > 0 else colors.HexColor('#DCFCE7')
                tcol = CRIMSON if f['shap'] > 0 else GREEN
                s_ts.add('BACKGROUND', (3,i), (3,i), col)
                s_ts.add('TEXTCOLOR',  (3,i), (3,i), tcol)
                s_ts.add('FONTNAME',   (3,i), (3,i), 'Helvetica-Bold')
            s_tbl.setStyle(s_ts)
            story.append(s_tbl)
            story.append(Spacer(1, 5*mm))

        # ═══════════════════ RECOMMENDATIONS ══════════════════════════════
        if recs:
            story.append(Paragraph('Clinical Recommendations', h2_s))
            story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
            story.append(Spacer(1, 2*mm))
            for i, rec in enumerate(recs, 1):
                story.append(Paragraph(f'{i}.  {rec}', rec_s))
            story.append(Spacer(1, 5*mm))

        # ═══════════════════ DISCLAIMER FOOTER ════════════════════════════
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
        story.append(Spacer(1, 2*mm))
        disc = ('⚕  This report is generated by CardioAI — an AI-assisted clinical decision support tool. '
                'It is intended for use by licensed healthcare professionals only. '
                'It does not constitute a medical diagnosis. Always apply clinical judgment. '
                'CardioAI employs a 6-model ensemble (SVM, Random Forest, Gradient Boosting, '
                'Logistic Regression, KNN, Decision Tree) trained on validated cardiovascular datasets.')
        story.append(Paragraph(disc, ps('disc', fontSize=7.5, textColor=TEXTSUB, leading=10)))

        doc.build(story)
        buf.seek(0)

        filename = f'CardioAI_Report_{report_id}.pdf'
        return send_file(buf, as_attachment=True,
                         download_name=filename,
                         mimetype='application/pdf')

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'trace': traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════════════════════
#  EMAIL & NOTIFICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def send_high_risk_email(to_email, patient_label, risk_score, risk_band, doctor_name='CardioAI'):
    """Send high-risk alert email. Non-blocking — catches all errors."""
    if not MAIL_AVAILABLE or not mail or not to_email:
        return False
    try:
        if not app.config.get('MAIL_USERNAME'):
            return False   # Email not configured
        subject = f'⚠️ CardioAI High-Risk Alert — {patient_label or "Patient"}'
        body = f"""
CardioAI Clinical Alert
========================
A high cardiovascular risk has been detected.

Patient:    {patient_label or 'Unlabelled Patient'}
Risk Score: {risk_score}/100
Risk Band:  {risk_band}
Doctor:     {doctor_name}
Time:       {datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}

This patient has been flagged as HIGH RISK by the CardioAI ensemble model.
Please review their clinical data and consider immediate follow-up.

— CardioAI Clinical Decision Support System
⚕ This is an automated alert. Not a substitute for clinical judgment.
"""
        msg = MailMessage(subject, recipients=[to_email], body=body)
        mail.send(msg)
        return True
    except Exception as e:
        print(f'Email send error: {e}')
        return False


def create_notification(user_id, title, message, ntype='info'):
    """Create an in-app notification for a user."""
    try:
        notif = Notification(user_id=user_id, title=title, message=message, ntype=ntype)
        db.session.add(notif)
        db.session.commit()
    except Exception as e:
        print(f'Notification create error: {e}')


@app.route('/api/notifications')
@login_required
def get_notifications():
    notifs = Notification.query.filter_by(user_id=current_user.id) \
                 .order_by(Notification.timestamp.desc()).limit(20).all()
    unread = Notification.query.filter_by(user_id=current_user.id, is_read=False).count()
    return jsonify({
        'notifications': [{
            'id': n.id, 'title': n.title, 'message': n.message,
            'type': n.ntype, 'is_read': n.is_read,
            'timestamp': n.timestamp.strftime('%d %b %Y %H:%M') if n.timestamp else '—'
        } for n in notifs],
        'unread': unread
    })


@app.route('/api/notifications/mark-read', methods=['POST'])
@login_required
def mark_notifications_read():
    Notification.query.filter_by(user_id=current_user.id, is_read=False) \
        .update({'is_read': True})
    db.session.commit()
    return jsonify({'success': True})


@app.route('/api/notifications/mark-one/<int:nid>', methods=['POST'])
@login_required
def mark_one_read(nid):
    n = Notification.query.filter_by(id=nid, user_id=current_user.id).first()
    if n:
        n.is_read = True
        db.session.commit()
    return jsonify({'success': True})


@app.route('/api/notifications/delete/<int:nid>', methods=['DELETE'])
@login_required
def delete_notification(nid):
    n = Notification.query.filter_by(id=nid, user_id=current_user.id).first()
    if n:
        db.session.delete(n)
        db.session.commit()
    return jsonify({'success': True})


# ── Patient profile (with email field) ────────────────────────────────────────
@app.route('/api/patient/email-settings', methods=['GET', 'POST'])
@login_required
def patient_email_settings():
    if current_user.role != 'patient':
        return jsonify({'error': 'Forbidden'}), 403
    prof = PatientProfile.query.filter_by(user_id=current_user.id).first()
    if request.method == 'POST':
        data = request.get_json()
        if not prof:
            prof = PatientProfile(user_id=current_user.id)
            db.session.add(prof)
        prof.email        = data.get('email', '').strip()
        prof.notify_email = bool(data.get('notify_email', True))
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({
        'email':        prof.email if prof else '',
        'notify_email': prof.notify_email if prof else True
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  DOCTOR-SPECIFIC ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-load dataset into memory once
import pandas as _pd_global
_HEART_DF = None
def _get_heart_df():
    global _HEART_DF
    if _HEART_DF is None:
        _path = os.path.join(BASE, 'data', 'heart_dataset.csv')
        if not os.path.exists(_path):
            _path = os.path.join(BASE, 'heart_dataset.csv')
        _HEART_DF = _pd_global.read_csv(_path)
    return _HEART_DF


@app.route('/api/doctor/stats')
@login_required
def doctor_stats():
    try:
        if current_user.role not in ('doctor', 'admin'):
            return jsonify({'error': 'Forbidden'}), 403
        dcode = (current_user.doctor_code or '').strip()
        hdf   = _get_heart_df()
        mine  = hdf[hdf['Doctor_ID'] == dcode].copy() if dcode else hdf.head(0).copy()

        total     = len(mine)
        disease   = int((mine['Target'] == 1).sum())
        healthy   = int((mine['Target'] == 0).sum())
        high_risk = int((mine['Risk_Level'] == 'High Risk').sum())
        avg_age   = round(float(mine['Age'].mean()), 1) if total else 0

        # Age group breakdown — safe sort without key lambda
        age_grp_raw = mine.groupby('Age_Group')['Target'].agg(['count','sum']).reset_index()
        age_grp_raw.columns = ['group','total','disease']
        age_grp_raw['healthy'] = age_grp_raw['total'] - age_grp_raw['disease']
        age_order = {'Young':0,'Middle Age':1,'Senior':2,'Elderly':3}
        age_grp_raw['_ord'] = age_grp_raw['group'].map(age_order).fillna(99)
        age_grp = age_grp_raw.sort_values('_ord').drop('_ord', axis=1).to_dict('records')

        sex_split = mine.groupby('Sex_Label')['Patient_ID'].count().to_dict() if total else {}
        risk_pie  = mine['Risk_Level'].value_counts().to_dict() if total else {}
        bp_dist   = mine['BP_Category'].value_counts().to_dict() if ('BP_Category' in mine.columns and total) else {}
        chol_dist = mine['Cholesterol_Category'].value_counts().to_dict() if ('Cholesterol_Category' in mine.columns and total) else {}

        trend = {}
        if 'Visit_Date' in mine.columns and total:
            mine2 = mine.copy()
            mine2['Visit_Date'] = _pd_global.to_datetime(mine2['Visit_Date'], errors='coerce')
            mine2['Month'] = mine2['Visit_Date'].dt.strftime('%b %Y')
            trend = mine2.groupby('Month')['Patient_ID'].count().to_dict()

        from datetime import date as _date
        today_start = datetime.combine(_date.today(), datetime.min.time())
        predictions_total = PatientHistory.query.filter_by(doctor_id=current_user.id).count()
        predictions_today = PatientHistory.query.filter(
            PatientHistory.doctor_id == current_user.id,
            PatientHistory.timestamp >= today_start).count()

        hosp = app.config.get('HOSPITALS_META', {}).get(current_user.hospital_id or '', {})

        return jsonify({
            'total': total, 'disease': disease, 'healthy': healthy,
            'high_risk': high_risk, 'avg_age': avg_age,
            'disease_pct': round(disease/total*100, 1) if total else 0,
            'age_groups': age_grp, 'sex_split': sex_split, 'risk_pie': risk_pie,
            'bp_dist': bp_dist, 'chol_dist': chol_dist, 'visit_trend': trend,
            'predictions_total': predictions_total, 'predictions_today': predictions_today,
            'hospital': hosp,
            'doctor': {
                'name': current_user.full_name or '',
                'code': dcode,
                'specialization': current_user.specialization or '',
                'experience': current_user.experience_years or 0,
                'qualifications': current_user.qualifications or '',
                'rating': current_user.patient_rating or 0,
            }
        })
    except Exception as _e:
        import traceback; print(f"doctor_stats error: {traceback.format_exc()}")
        return jsonify({'error': str(_e), 'total': 0, 'disease': 0, 'healthy': 0,
                        'high_risk': 0, 'avg_age': 0, 'disease_pct': 0,
                        'age_groups': [], 'sex_split': {}, 'risk_pie': {},
                        'bp_dist': {}, 'chol_dist': {}, 'visit_trend': {},
                        'predictions_total': 0, 'predictions_today': 0,
                        'hospital': {}, 'doctor': {}})


@app.route('/api/doctor/patients')
@login_required
def doctor_patients():
    try:
        if current_user.role not in ('doctor', 'admin'):
            return jsonify({'error': 'Forbidden'}), 403
        dcode    = request.args.get('dcode') or (current_user.doctor_code or '').strip()
        page     = max(1, int(request.args.get('page', 1)))
        per_page = int(request.args.get('per_page', 30))
        search   = request.args.get('search', '').strip()
        risk     = request.args.get('risk', '')
        sex      = request.args.get('sex', '')
        age_grp  = request.args.get('age_group', '')

        hdf  = _get_heart_df()
        mine = hdf[hdf['Doctor_ID'] == dcode].copy() if dcode else hdf.head(0).copy()

        if search:
            mine = mine[mine['Patient_ID'].str.contains(search, case=False, na=False)]
        if risk and 'Risk_Level' in mine.columns:
            mine = mine[mine['Risk_Level'] == risk]
        if sex and 'Sex_Label' in mine.columns:
            mine = mine[mine['Sex_Label'] == sex]
        if age_grp and 'Age_Group' in mine.columns:
            mine = mine[mine['Age_Group'] == age_grp]

        total   = len(mine)
        start   = (page - 1) * per_page
        page_df = mine.iloc[start:start+per_page]

        cols = ['Patient_ID','Age','Age_Group','Sex_Label','Chest_Pain_Type',
                'Trestbps','BP_Category','Cholesterol','Cholesterol_Category',
                'Max_Heart_Rate','Exercise_Induced_Angina','Risk_Level','Target',
                'Smoking_Status','Exercise_Level','BMI_Category','Visit_Date']
        cols = [c for c in cols if c in page_df.columns]
        records = page_df[cols].fillna('—').to_dict('records')

        return jsonify({
            'records': records, 'total': total, 'page': page,
            'pages': max(1, (total + per_page - 1) // per_page)
        })
    except Exception as _e:
        import traceback; print(f"doctor_patients error: {traceback.format_exc()}")
        return jsonify({'error': str(_e), 'records': [], 'total': 0, 'page': 1, 'pages': 1})


# ═══════════════════════════════════════════════════════════════════════════════
#  ADMIN-SPECIFIC ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

def _require_admin():
    if not current_user.is_authenticated or current_user.role != 'admin':
        from flask import abort; abort(403)

@app.route('/api/admin/overview')
@login_required
def admin_overview():
    _require_admin()
    hdf = _get_heart_df()
    total_patients = len(hdf)
    disease_count  = int((hdf['Target'] == 1).sum())
    healthy_count  = int((hdf['Target'] == 0).sum())
    high_risk      = int((hdf['Risk_Level'] == 'High Risk').sum())

    total_doctors  = User.query.filter_by(role='doctor').count()
    active_doctors = User.query.filter_by(role='doctor', is_active_acc=True).count()
    total_users    = User.query.count()
    total_preds    = PatientHistory.query.count()

    from datetime import date as _date
    today_start = datetime.combine(_date.today(), datetime.min.time())
    preds_today = PatientHistory.query.filter(PatientHistory.timestamp >= today_start).count()

    # Hospital summary
    hosp_meta = app.config.get('HOSPITALS_META', {})
    hospital_summary = []
    for hid, hmeta in hosp_meta.items():
        h_pats  = len(hdf[hdf['Hospital_ID'] == hid])
        h_dis   = int((hdf[hdf['Hospital_ID'] == hid]['Target'] == 1).sum())
        h_docs  = User.query.filter_by(role='doctor', hospital_id=hid).count()
        hospital_summary.append({**hmeta, 'id': hid,
            'patient_count': h_pats, 'disease_count': h_dis,
            'doctor_count': h_docs,
            'disease_pct': round(h_dis/h_pats*100, 1) if h_pats else 0})

    # Risk by region
    if 'Hospital_ID' in hdf.columns:
        hdf2 = hdf.copy()
        region_map = {hid: h['region'] for hid, h in hosp_meta.items()}
        hdf2['Region'] = hdf2['Hospital_ID'].map(region_map)
        region_risk = hdf2.groupby('Region').agg(
            total=('Patient_ID','count'),
            disease=('Target','sum')).reset_index()
        region_risk['disease_pct'] = (region_risk['disease']/region_risk['total']*100).round(1)
        region_data = region_risk.to_dict('records')
    else:
        region_data = []

    # Model performance (static from training)
    model_perf = [
        {'model':'Support Vector Machine',  'accuracy':80.50,'precision':80.99,'recall':83.49,'f1':82.22,'auc':0.882},
        {'model':'Gradient Boosting',       'accuracy':79.71,'precision':80.21,'recall':82.87,'f1':81.52,'auc':0.885},
        {'model':'Logistic Regression',     'accuracy':78.92,'precision':79.61,'recall':81.94,'f1':80.76,'auc':0.871},
        {'model':'Random Forest',           'accuracy':78.88,'precision':79.20,'recall':82.56,'f1':80.85,'auc':0.875},
        {'model':'K-Nearest Neighbors',     'accuracy':75.88,'precision':76.03,'recall':80.79,'f1':78.34,'auc':0.824},
        {'model':'Decision Tree',           'accuracy':73.54,'precision':75.29,'recall':75.93,'f1':75.61,'auc':0.806},
    ]

    # Dataset distributions
    age_dist    = hdf['Age_Group'].value_counts().to_dict()    if 'Age_Group'    in hdf.columns else {}
    risk_dist   = hdf['Risk_Level'].value_counts().to_dict()   if 'Risk_Level'   in hdf.columns else {}
    gender_dist = hdf['Sex_Label'].value_counts().to_dict()    if 'Sex_Label'    in hdf.columns else {}
    smoke_dist  = hdf['Smoking_Status'].value_counts().to_dict() if 'Smoking_Status' in hdf.columns else {}

    # Scatter data (sample)
    sample = hdf.sample(n=min(800, len(hdf)), random_state=42)
    scatter = {
        'disease': sample[sample['Target']==1][['Cholesterol','Max_Heart_Rate']].to_dict('records'),
        'healthy': sample[sample['Target']==0][['Cholesterol','Max_Heart_Rate']].to_dict('records'),
    }

    # Age histogram
    age_dist_raw = {
        'disease': hdf[hdf['Target']==1]['Age'].tolist(),
        'healthy': hdf[hdf['Target']==0]['Age'].tolist(),
    }

    return jsonify({
        'total_patients': total_patients,
        'disease_count':  disease_count,
        'healthy_count':  healthy_count,
        'high_risk':      high_risk,
        'disease_pct':    round(disease_count/total_patients*100, 1),
        'total_doctors':  total_doctors,
        'active_doctors': active_doctors,
        'total_users':    total_users,
        'total_preds':    total_preds,
        'preds_today':    preds_today,
        'hospital_summary': hospital_summary,
        'region_data':    region_data,
        'model_perf':     model_perf,
        'age_dist':       age_dist,
        'risk_dist':      risk_dist,
        'gender_dist':    gender_dist,
        'smoke_dist':     smoke_dist,
        'scatter':        scatter,
        'age_dist_raw':   age_dist_raw,
    })


@app.route('/api/admin/doctors')
@login_required
def admin_doctors():
    _require_admin()
    hdf     = _get_heart_df()
    doctors = User.query.filter_by(role='doctor').all()
    result  = []
    for d in doctors:
        dcode = d.doctor_code
        pats  = len(hdf[hdf['Doctor_ID'] == dcode]) if dcode else 0
        dis   = int((hdf[hdf['Doctor_ID'] == dcode]['Target'] == 1).sum()) if dcode and pats else 0
        preds = PatientHistory.query.filter_by(doctor_id=d.id).count()
        hosp  = app.config.get('HOSPITALS_META', {}).get(d.hospital_id, {})
        result.append({
            'id':            d.id,
            'username':      d.username,
            'full_name':     d.full_name,
            'doctor_code':   dcode,
            'hospital_id':   d.hospital_id,
            'hospital_name': hosp.get('name', '—'),
            'hospital_city': hosp.get('city', '—'),
            'specialization':d.specialization,
            'experience':    d.experience_years,
            'qualifications':d.qualifications,
            'gender':        d.gender,
            'rating':        d.patient_rating,
            'is_active':     d.is_active_acc,
            'last_login':    d.last_login.strftime('%d %b %Y %H:%M') if d.last_login else 'Never',
            'patient_count': pats,
            'disease_count': dis,
            'predictions':   preds,
        })
    return jsonify({'doctors': result})


@app.route('/api/admin/toggle-doctor/<int:uid>', methods=['POST'])
@login_required
def admin_toggle_doctor(uid):
    _require_admin()
    user = User.query.get_or_404(uid)
    user.is_active_acc = not user.is_active_acc
    db.session.commit()
    return jsonify({'success': True, 'is_active': user.is_active_acc})


@app.route('/api/admin/reset-password/<int:uid>', methods=['POST'])
@login_required
def admin_reset_password(uid):
    _require_admin()
    user = User.query.get_or_404(uid)
    user.password_hash = bcrypt.generate_password_hash('CardioAI@2024').decode('utf-8')
    db.session.commit()
    return jsonify({'success': True})


@app.route('/api/admin/all-predictions')
@login_required
def admin_all_predictions():
    _require_admin()
    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 40))
    records  = PatientHistory.query.order_by(PatientHistory.timestamp.desc()) \
                 .offset((page-1)*per_page).limit(per_page).all()
    total    = PatientHistory.query.count()
    def to_dict(r):
        doc = User.query.get(r.doctor_id)
        return {
            'id': r.id, 'timestamp': r.timestamp.strftime('%d %b %Y %H:%M') if r.timestamp else '—',
            'doctor': doc.full_name if doc else '—',
            'hospital': app.config.get('HOSPITALS_META',{}).get(doc.hospital_id,{}).get('name','—') if doc else '—',
            'patient_label': r.patient_label or '—',
            'age': r.age, 'sex': r.sex, 'risk_score': r.risk_score,
            'risk_band': r.risk_band or '—', 'avg_prob': r.avg_prob,
            'final_prediction': r.final_prediction,
        }
    return jsonify({'records': [to_dict(r) for r in records], 'total': total, 'page': page,
                    'pages': max(1,(total+per_page-1)//per_page)})


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — Must be at the very end after all routes are registered
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 55)
    print("  ❤️  Heart Disease Prediction System")
    print("  Open your browser: http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, port=5000)
