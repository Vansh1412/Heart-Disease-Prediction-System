# 🎓 How to Show the Prediction System to Your Teacher

## ────────────────────────────────────────────────────────
##  OPTION 1 — Browser Demo (EASIEST, zero setup)
## ────────────────────────────────────────────────────────

1. Extract the zip file
2. Double-click  ➜  **DEMO_open_in_browser.html**
3. It opens in Chrome / Edge / Firefox
4. Done! No Python needed.

### What to show:
- 🔴 Click "High Risk" preset → see disease detected
- 🟢 Click "Low Risk" preset → see healthy result
- 🟡 Click "Border" preset → see borderline case
- Move the sliders → watch predictions update live
- Click all 5 tabs to show Dataset, Models, Features, Code

---

## ────────────────────────────────────────────────────────
##  OPTION 2 — Streamlit App (MOST IMPRESSIVE)
## ────────────────────────────────────────────────────────

### Install (one time only):
```
pip install streamlit plotly scikit-learn pandas numpy joblib
```

### Run:
```
cd heart_prediction
streamlit run streamlit_app.py
```

Opens at http://localhost:8501 automatically.

### What to show:
1. Show the **sidebar input** — adjust patient sliders
2. Click **Run Prediction** — show the live gauges
3. Show **all 5 tabs**: Prediction, Dataset, Model Performance, Features, Code
4. Load presets (High/Low/Border) — show how it changes instantly
5. Click the **Code tab** → scroll through Python scripts live

---

## ────────────────────────────────────────────────────────
##  OPTION 3 — Python Scripts (for backend demo)
## ────────────────────────────────────────────────────────

```
cd heart_prediction
python 09_predict_new_patient.py     ← Shows 3 test patient predictions
python 04_model_training.py          ← Live model training with accuracy table
python 12_risk_scoring.py            ← Risk score computation
```

---

## ────────────────────────────────────────────────────────
##  SUGGESTED PRESENTATION ORDER (5 minutes)
## ────────────────────────────────────────────────────────

1. Open **DEMO_open_in_browser.html** (30 sec)
   → "This is our interactive prediction demo"

2. Load **High Risk preset** (1 min)
   → Show verdict banner, gauges, all 6 model probabilities
   → "All 6 models agree — high disease probability"

3. Load **Low Risk preset** (30 sec)
   → "Notice how the gauges change completely"

4. Switch to **Dataset tab** (1 min)
   → "We used 12,000 patients with real clinical patterns"
   → Show age group chart, lifestyle charts

5. Switch to **Models tab** (1 min)
   → "We trained 6 different algorithms"
   → "Accuracy range is 73-80% — realistic for heart disease"

6. Switch to **Code tab** (1 min)
   → "Here's the actual preprocessing and training code"
   → Show feature engineering code

