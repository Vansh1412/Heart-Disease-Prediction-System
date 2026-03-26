# ❤️ Heart Disease Prediction — Complete ML Project
## 12,000 Patients | 6 Models | 14 Scripts | 23 Charts

---

## 📁 Folder Structure

```
heart_prediction/
│
├── data/
│   └── heart_dataset.csv                  # 12,000 realistic patient records
│
├── models/                                # Saved after running scripts
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_cols.pkl
│   ├── logistic_regression.pkl
│   ├── k-nearest_neighbors.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── support_vector_machine.pkl
│   ├── ensemble_soft_voting.pkl
│   └── ensemble_hard_voting.pkl
│
├── outputs/                               # CSV reports & analytics
│   ├── model_comparison.csv
│   ├── classification_reports.txt
│   ├── feature_importance_summary.csv
│   ├── cross_validation_results.csv
│   ├── ensemble_results.csv
│   ├── dataset_with_risk_scores.csv
│   ├── hospital_stats.csv
│   ├── doctor_stats.csv
│   ├── stats_continuous.csv
│   ├── stats_categorical.csv
│   └── stats_pointbiserial.csv
│
├── plots/                                 # 23 auto-generated charts
│
├── 01_data_overview.py           Dataset shape, dtypes, nulls, distributions
├── 02_eda_analysis.py            8 EDA plots (histograms, heatmap, boxplots, trends)
├── 03_preprocessing.py           Encoding, outlier clipping, feature engineering, scaling
├── 04_model_training.py          Train 6 ML models, compare, save .pkl files
├── 05_model_evaluation.py        Confusion matrix, ROC curves, PR curves
├── 06_feature_importance.py      RF + GB importance, SHAP values, LR coefficients
├── 07_hyperparameter_tuning.py   GridSearchCV (LR) + RandomizedSearchCV (RF, GB)
├── 08_cross_validation.py        K-Fold, Stratified K-Fold, Learning Curves
├── 09_predict_new_patient.py     Single patient prediction with risk banner + advice
├── 10_full_pipeline.py           ONE FILE runs steps 1–9 end-to-end
├── 11_model_stacking.py          Soft Voting + Hard Voting Ensemble models
├── 12_risk_scoring.py            Framingham-style 0–100 clinical risk score engine
├── 13_statistics_report.py       Chi-square, T-tests, Mann-Whitney, Cohen's d, Cramer's V
├── 14_hospital_doctor_report.py  Hospital & doctor performance analytics + YoY trends
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

**Install dependencies first:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib openpyxl
pip install shap   # optional — for SHAP explainability plots
```

**Option A — Run everything at once:**
```bash
cd heart_prediction
python 10_full_pipeline.py
```

**Option B — Run step by step (recommended for learning):**
```bash
python 01_data_overview.py
python 02_eda_analysis.py
python 03_preprocessing.py
python 04_model_training.py
python 05_model_evaluation.py
python 06_feature_importance.py
python 07_hyperparameter_tuning.py
python 08_cross_validation.py
python 09_predict_new_patient.py
python 11_model_stacking.py
python 12_risk_scoring.py
python 13_statistics_report.py
python 14_hospital_doctor_report.py
```

---

## 📊 Dataset — 18 Features, 12,000 Patients

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Patient age (29–77) |
| Sex | Binary | 1=Male, 0=Female |
| Chest_Pain_Type | Ordinal | 0=Typical Angina → 3=Asymptomatic |
| Trestbps | Numeric | Resting blood pressure (mmHg) |
| Cholesterol | Numeric | Serum cholesterol (mg/dL) |
| Fasting_Blood_Sugar | Binary | 1 if > 120 mg/dL |
| Resting_ECG | Ordinal | 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy |
| Max_Heart_Rate | Numeric | Maximum exercise heart rate |
| Exercise_Induced_Angina | Binary | 1=Yes, 0=No |
| ST_Depression | Numeric | ST depression (oldpeak) |
| Slope | Ordinal | Peak ST slope (0=Up, 1=Flat, 2=Down) |
| Major_Vessels | Ordinal | Fluoroscopy vessels coloured (0–3) |
| Thalassemia | Categorical | 3=Normal, 6=Fixed defect, 7=Reversible |
| Smoking_Status | Categorical | Current / Former / Non-Smoker |
| Alcohol_Consumption | Categorical | High / Moderate / None |
| Exercise_Level | Categorical | High / Moderate / Low |
| BMI_Category | Categorical | Normal / Overweight / Obese |
| **Target** | **Binary** | **1=Heart Disease, 0=Healthy** |

---

## 🏆 Model Performance (tested on 2,400 patients)

| Model | Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|
| **Support Vector Machine** | **80.50%** | 82.2% | 0.882 |
| Gradient Boosting | 79.71% | 81.5% | 0.885 |
| Logistic Regression | 78.92% | 80.8% | 0.871 |
| Random Forest | 78.88% | 80.9% | 0.875 |
| Soft Voting Ensemble | 80.12% | — | 0.882 |
| Hard Voting Ensemble | 80.21% | — | — |
| K-Nearest Neighbors | 75.88% | 78.3% | 0.824 |
| Decision Tree | 73.54% | 75.6% | 0.806 |

> All values in clinically realistic **73–81% accuracy range** (not inflated).

---

## 🔑 Top 5 Most Important Features

1. **Max Heart Rate** — Lower max HR = higher disease risk
2. **Age** — Risk rises sharply after 50
3. **Major Vessels (ca)** — More blocked vessels = higher risk
4. **Smoking Status** — Current smokers have 72% disease rate
5. **Chest Pain Type** — Typical angina (type 0) is highest risk

---

## 🩺 Predict a New Patient

Edit the input dictionary in `09_predict_new_patient.py`:

```python
patient = {
    'age': 58,  'sex': 1,  'cp': 0,
    'trestbps': 145,  'chol': 270,  'fbs': 1,
    'restecg': 1,  'thalachh': 120,  'exang': 1,
    'oldpeak': 2.3,  'slope': 1,  'ca': 2,  'thal': 7,
    'smoking': 'Current Smoker',  'alcohol': 'High',
    'exercise': 'Low',  'bmi': 'Obese'
}
```

Output includes: per-model probability, ensemble vote, colour-coded risk banner, and specific clinical recommendations.

---

## 🏥 Hospitals & Doctors Covered

**8 Hospitals:** Apollo Delhi, Fortis Mumbai, AIIMS Chennai, Max Bangalore,
Medanta Gurugram, Narayana Kolkata, Wockhardt Hyderabad, Ruby Hall Pune

**15 Doctors:** Cardiologists, Interventional Cardiologists, Cardiac Surgeons,
Electrophysiologists, General Physicians, Cardiac Rehabilitation specialists

---

## 📈 Clinical Risk Score (Script 12)

Framingham-inspired **0–100 composite score** using 14 clinical + lifestyle factors:

| Band | Score | Disease Rate |
|---|---|---|
| Very Low Risk | 0–24 | ~9.5% |
| Low Risk | 25–39 | ~28% |
| Moderate Risk | 40–54 | ~53% |
| High Risk | 55–69 | ~78% |
| Very High Risk | 70–100 | ~92% |
