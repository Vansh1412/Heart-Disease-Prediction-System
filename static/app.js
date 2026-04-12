/* ==========================================================================
   CardioAI — app.js
   Handles: tab navigation, prediction API, Chart.js charts, Plotly gauges
   ========================================================================== */
'use strict';

/* ── Feature Importance Data ──────────────────────────────────────────── */
const FI_DATA = [
  { feature: 'Max_Heart_Rate', rf: 11.64, gb: 15.25 },
  { feature: 'Age', rf: 10.10, gb: 8.31 },
  { feature: 'Major_Vessels', rf: 8.64, gb: 9.35 },
  { feature: 'Smoking_Status', rf: 8.22, gb: 8.63 },
  { feature: 'Exercise_Level', rf: 6.01, gb: 8.84 },
  { feature: 'Alcohol_Consumption', rf: 5.99, gb: 6.04 },
  { feature: 'Age_Sex_Interact', rf: 6.51, gb: 4.18 },
  { feature: 'Chest_Pain_Type', rf: 5.09, gb: 6.46 },
  { feature: 'BP_Chol_Score', rf: 4.76, gb: 5.60 },
  { feature: 'ST_Depression', rf: 5.11, gb: 4.32 },
  { feature: 'Exercise_Induced_Angina', rf: 4.52, gb: 4.08 },
  { feature: 'Trestbps', rf: 4.18, gb: 3.66 },
  { feature: 'HR_Reserve', rf: 4.26, gb: 2.27 },
  { feature: 'ST_Slope_Risk', rf: 3.96, gb: 2.48 },
  { feature: 'Cholesterol', rf: 3.90, gb: 3.15 },
  { feature: 'BMI_Category', rf: 2.52, gb: 3.44 },
  { feature: 'Thalassemia', rf: 1.68, gb: 1.79 },
  { feature: 'Slope', rf: 1.15, gb: 0.95 },
  { feature: 'Resting_ECG', rf: 0.93, gb: 0.88 },
  { feature: 'Sex', rf: 0.60, gb: 0.03 },
  { feature: 'Fasting_Blood_Sugar', rf: 0.24, gb: 0.27 },
];

const CORR_DATA = [
  { feature: 'Thalassemia', corr: 0.52 },
  { feature: 'Major_Vessels', corr: 0.48 },
  { feature: 'Exercise_Induced_Angina', corr: 0.43 },
  { feature: 'ST_Depression', corr: 0.43 },
  { feature: 'Sex', corr: 0.27 },
  { feature: 'Age', corr: 0.23 },
  { feature: 'Trestbps', corr: 0.16 },
  { feature: 'Fasting_Blood_Sugar', corr: 0.09 },
  { feature: 'Slope', corr: -0.35 },
  { feature: 'Max_Heart_Rate', corr: -0.42 },
  { feature: 'Chest_Pain_Type', corr: -0.43 },
];

/* ── Chart Defaults ───────────────────────────────────────────────────── */
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = '#a1a1aa';
Chart.defaults.plugins.legend.labels.boxWidth = 12;

const CHART_DARK = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { labels: { color: '#a1a1aa' } },
    tooltip: {
      backgroundColor: '#232328',
      titleColor: '#f4f4f5',
      bodyColor: '#a1a1aa',
      borderColor: 'rgba(255,255,255,.08)',
      borderWidth: 1,
      padding: 12,
    }
  },
  scales: {
    x: {
      ticks: { color: '#71717a' },
      grid: { color: 'rgba(255,255,255,.05)' }
    },
    y: {
      ticks: { color: '#71717a' },
      grid: { color: 'rgba(255,255,255,.05)' }
    }
  }
};

/* ══════════════════════════════════════════════════════════════════════════
   TAB NAVIGATION
══════════════════════════════════════════════════════════════════════════ */
function switchTab(tabId) {
  document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  const section = document.getElementById('tab-' + tabId);
  if (section) section.classList.add('active');
  document.querySelectorAll('.nav-link').forEach(l => {
    if (l.dataset.tab === tabId) l.classList.add('active');
  });
  if (tabId === 'analytics') initAnalytics();
  if (tabId === 'performance') initPerformance();
  if (tabId === 'features') initFeatures();
}

document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    switchTab(link.dataset.tab);
  });
});

/* ══════════════════════════════════════════════════════════════════════════
   PRESET PATIENTS
══════════════════════════════════════════════════════════════════════════ */
const PRESETS = {
  high: {
    age: 58, sex: 'Male', cp: 0, trestbps: 158, chol: 275, thalachh: 115,
    oldpeak: 2.8, slope: 2, fbs: 0, restecg: 1, exang: 1, ca: 2, thal: 7,
    smoking: 'Current Smoker', alcohol: 'High', exercise: 'Low', bmi: 'Obese'
  },
  low: {
    age: 35, sex: 'Female', cp: 2, trestbps: 118, chol: 190, thalachh: 175,
    oldpeak: 0.0, slope: 0, fbs: 0, restecg: 0, exang: 0, ca: 0, thal: 3,
    smoking: 'Non-Smoker', alcohol: 'None', exercise: 'High', bmi: 'Normal'
  },
  moderate: {
    age: 50, sex: 'Male', cp: 1, trestbps: 135, chol: 245, thalachh: 148,
    oldpeak: 1.2, slope: 1, fbs: 0, restecg: 0, exang: 0, ca: 1, thal: 3,
    smoking: 'Former Smoker', alcohol: 'Moderate', exercise: 'Moderate', bmi: 'Overweight'
  }
};

function loadPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  document.getElementById('f-age').value = p.age;
  document.getElementById('f-sex').value = p.sex;
  document.getElementById('f-cp').value = p.cp;
  document.getElementById('f-trestbps').value = p.trestbps;
  document.getElementById('f-chol').value = p.chol;
  document.getElementById('f-thalachh').value = p.thalachh;
  document.getElementById('f-oldpeak').value = p.oldpeak;
  document.getElementById('f-slope').value = p.slope;
  document.getElementById('f-fbs').value = p.fbs;
  document.getElementById('f-restecg').value = p.restecg;
  document.getElementById('f-exang').value = p.exang;
  document.getElementById('f-ca').value = p.ca;
  document.getElementById('f-thal').value = p.thal;
  document.getElementById('f-smoking').value = p.smoking;
  document.getElementById('f-alcohol').value = p.alcohol;
  document.getElementById('f-exercise').value = p.exercise;
  document.getElementById('f-bmi').value = p.bmi;
}

/* ══════════════════════════════════════════════════════════════════════════
   PREDICTION FORM
══════════════════════════════════════════════════════════════════════════ */
document.getElementById('predict-form').addEventListener('submit', async e => {
  e.preventDefault();
  const payload = {
    age: document.getElementById('f-age').value,
    sex: document.getElementById('f-sex').value,
    cp: document.getElementById('f-cp').value,
    trestbps: document.getElementById('f-trestbps').value,
    chol: document.getElementById('f-chol').value,
    thalachh: document.getElementById('f-thalachh').value,
    oldpeak: document.getElementById('f-oldpeak').value,
    slope: document.getElementById('f-slope').value,
    fbs: document.getElementById('f-fbs').value,
    restecg: document.getElementById('f-restecg').value,
    exang: document.getElementById('f-exang').value,
    ca: document.getElementById('f-ca').value,
    thal: document.getElementById('f-thal').value,
    smoking: document.getElementById('f-smoking').value,
    alcohol: document.getElementById('f-alcohol').value,
    exercise: document.getElementById('f-exercise').value,
    bmi: document.getElementById('f-bmi').value,
    patient_label: (document.getElementById('f-patient-label')?.value || '').trim(),
  };

  showLoading(true);
  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();
    if (data.success) {
      renderResults(data, payload);
      lastPatientData  = payload;  // save for SHAP
      lastPredResult   = data;     // save for PDF export
      triggerShap();               // auto-run SHAP
      // Reset analytics so doctor's prediction count refreshes on next visit
      analyticsLoaded = false;
      // Signal history tab to reload
      historyLoaded = false;
      document.dispatchEvent(new CustomEvent('cardio:newprediction'));
      // Show export button
      const pdfWrap = document.getElementById('pdf-export-wrap');
      if (pdfWrap) pdfWrap.style.display = 'block';
    }
    else alert('Prediction error: ' + data.error);
  } catch (err) {
    alert('Network error: ' + err.message);
  } finally {
    showLoading(false);
  }
});

/* ── Shared Prediction State ─────────────────────────────────────────── */
let lastPatientData = null;
let lastPredResult  = null;
let lastShapResult  = null;

function triggerShap() {
  const box = document.getElementById('shap-box');
  if (box) {
    box.style.display = 'block';
    runShap();
  }
}

async function runShap() {
  if (!lastPatientData) return;

  const sel   = document.getElementById('shap-model-select');
  const model = sel ? sel.value : 'Gradient Boosting';
  const status  = document.getElementById('shap-status');
  const chartEl = document.getElementById('chart-shap');

  status.textContent  = '⏳ Computing SHAP values for ' + model + '…';
  status.style.display = 'block';
  chartEl.style.display = 'none';

  try {
    const res = await fetch('/api/shap', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ ...lastPatientData, model })
    });
    const d = await res.json();
    if (!d.success) throw new Error(d.error);

    lastShapResult = d;           // save for PDF export
    status.style.display  = 'none';
    chartEl.style.display = 'block';
    renderShapChart(d.features, d.base_value, d.model);

  } catch (err) {
    status.textContent = '⚠️ SHAP failed: ' + err.message;
    status.style.color = '#f87171';
  }
}

function renderShapChart(features, baseVal, modelName) {
  const top    = features.slice(0, 12).reverse();
  const vals   = top.map(f => f.shap);
  const labels = top.map(f => `${f.feature}  [${f.value}]`);
  const colors = vals.map(v => v > 0 ? 'rgba(239,68,68,0.82)' : 'rgba(59,130,246,0.82)');
  const border = vals.map(v => v > 0 ? '#ef4444' : '#3b82f6');

  Plotly.newPlot('chart-shap', [{
    type:        'bar',
    orientation: 'h',
    x:    vals,
    y:    labels,
    marker: { color: colors, line: { color: border, width: 1.2 } },
    hovertemplate: '<b>%{y}</b><br>SHAP impact: <b>%{x:.5f}</b><extra></extra>'
  }], {
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'transparent',
    font:   { color: '#cbd5e1', family: 'Inter, sans-serif', size: 12 },
    margin: { l: 210, r: 50, t: 30, b: 60 },
    title:  { text: `Model: ${modelName} · Base value: ${baseVal.toFixed(3)}`,
               font: { size: 11, color: '#64748b' }, x: 0.01 },
    xaxis: {
      title:      'SHAP Value  (← healthy  |  disease →)',
      gridcolor:  'rgba(255,255,255,0.07)',
      zerolinecolor: 'rgba(255,255,255,0.35)',
      zerolinewidth: 1.5
    },
    yaxis: { gridcolor: 'transparent' },
    shapes: [{
      type: 'line', x0: 0, x1: 0,
      y0: -0.5, y1: top.length - 0.5,
      line: { color: 'rgba(255,255,255,0.25)', width: 1, dash: 'dot' }
    }]
  }, { responsive: true, displayModeBar: false });
}

/* ── PDF Export ───────────────────────────────────────────────────────── */
async function exportPDF() {
  if (!lastPredResult || !lastPatientData) {
    alert('Please run a prediction first before exporting the report.');
    return;
  }

  const btn    = document.getElementById('btn-export-pdf');
  const status = document.getElementById('pdf-status');

  // Disable button & show spinner
  btn.disabled = true;
  btn.innerHTML = '<span class="pdf-icon">⏳</span><span>Generating Report…</span>';
  status.textContent = '';

  try {
    // Build the payload — combine prediction result + patient data + last SHAP
    const shapSel = document.getElementById('shap-model-select');
    const payload = {
      // Prediction fields
      final_pred:   lastPredResult.final_pred,
      avg_prob:     lastPredResult.avg_prob,
      risk_score:   lastPredResult.risk_score,
      risk_band:    lastPredResult.risk_band,
      votes_yes:    lastPredResult.votes_yes,
      total_models: lastPredResult.total_models,
      model_preds:  lastPredResult.model_preds,
      recommendations: lastPredResult.recommendations,
      // Patient data
      patient:      lastPatientData,
      // SHAP data (may be null if SHAP hasn't loaded yet)
      shap_features: lastShapResult ? lastShapResult.features  : [],
      shap_model:    lastShapResult ? lastShapResult.model      : (shapSel ? shapSel.value : 'Gradient Boosting'),
    };

    const resp = await fetch('/api/export-pdf', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: 'Unknown server error' }));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }

    // Trigger browser download
    const blob     = await resp.blob();
    const url      = URL.createObjectURL(blob);
    const a        = document.createElement('a');
    const filename = resp.headers.get('Content-Disposition')?.match(/filename="?([^"]+)"?/)?.[1]
                     || 'CardioAI_Clinical_Report.pdf';
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    status.textContent = '✅ Report downloaded successfully!';
    status.style.color = '#22c55e';

  } catch (err) {
    status.textContent = '⚠️ Export failed: ' + err.message;
    status.style.color = '#f87171';
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="pdf-icon">📄</span><span>Export Clinical PDF Report</span><span class="pdf-tag">Official Document</span>';
  }
}

function showLoading(on) {
  const ov = document.getElementById('loading-overlay');
  ov.classList.toggle('active', on);
}

/* ── Render results ───────────────────────────────────────────────────── */
function renderResults(data, payload) {
  document.getElementById('results-placeholder').style.display = 'none';
  document.getElementById('results-content').style.display = 'block';

  const isDisease = data.final_pred === 1;
  const prob = (data.avg_prob * 100).toFixed(1);
  const riskScore = data.risk_score;

  /* Verdict */
  const banner = document.getElementById('verdict-banner');
  const titleEl = document.getElementById('verdict-title');
  const subEl = document.getElementById('verdict-sub');
  const iconEl = document.getElementById('verdict-icon');

  if (isDisease) {
    banner.style.background = 'linear-gradient(135deg,rgba(220,38,38,.25),rgba(185,28,28,.15))';
    banner.style.border = '1px solid rgba(220,38,38,.4)';
    titleEl.textContent = '❤️ HEART DISEASE DETECTED';
    titleEl.style.color = '#ef4444';
    iconEl.style.color = '#ef4444';
  } else {
    banner.style.background = 'linear-gradient(135deg,rgba(34,197,94,.18),rgba(22,163,74,.08))';
    banner.style.border = '1px solid rgba(34,197,94,.35)';
    titleEl.textContent = '💚 NO HEART DISEASE DETECTED';
    titleEl.style.color = '#22c55e';
    iconEl.style.color = '#22c55e';
  }
  subEl.textContent = `Ensemble of ${data.total_models} models · ${data.votes_yes}/${data.total_models} votes for disease · Avg probability: ${prob}%`;

  /* KPI cards */
  document.getElementById('rm-prob').textContent = prob + '%';
  document.getElementById('rm-prob').style.color = isDisease ? '#ef4444' : '#22c55e';
  document.getElementById('rm-votes').textContent = `${data.votes_yes}/${data.total_models}`;
  document.getElementById('rm-risk').textContent = riskScore + '/100';
  document.getElementById('rm-band').textContent = data.risk_band;

  /* Gauges */
  buildGaugeProb(parseFloat(prob), isDisease);
  buildGaugeRisk(riskScore, data.risk_band);

  /* Per-model chart */
  buildModelBars(data.model_preds);

  /* Recommendations */
  const list = document.getElementById('recs-list');
  list.innerHTML = '';
  data.recommendations.forEach(rec => {
    const li = document.createElement('li');
    li.textContent = rec;
    list.appendChild(li);
  });

  /* Summary table */
  buildSummaryTable(payload);

  /* Scroll into view */
  document.getElementById('results-content').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Gauges (Plotly) ──────────────────────────────────────────────────── */
function buildGaugeProb(prob, isDisease) {
  const color = isDisease ? '#ef4444' : '#22c55e';
  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: { l: 20, r: 20, t: 40, b: 10 }, height: 220, font: { color: '#a1a1aa', family: 'Inter' }
  };
  const trace = {
    type: 'indicator', mode: 'gauge+number+delta',
    value: prob,
    delta: { reference: 50, increasing: { color: '#ef4444' }, decreasing: { color: '#22c55e' }, valueformat: '.1f' },
    number: { suffix: '%', font: { size: 34, color: '#f4f4f5' } },
    title: { text: 'Disease Probability (%)', font: { size: 12, color: '#71717a' } },
    gauge: {
      axis: { range: [0, 100], tickcolor: '#3f3f46', tickwidth: 1, tickfont: { color: '#71717a', size: 11 } },
      bar: { color, thickness: 0.28 },
      bgcolor: 'transparent',
      bordercolor: '#3f3f46',
      steps: [
        { range: [0, 25], color: 'rgba(34,197,94,.12)' },
        { range: [25, 50], color: 'rgba(245,158,11,.08)' },
        { range: [50, 75], color: 'rgba(220,38,38,.10)' },
        { range: [75, 100], color: 'rgba(220,38,38,.20)' },
      ],
      threshold: { line: { color: '#f4f4f5', width: 2 }, thickness: 0.8, value: 50 }
    }
  };
  Plotly.react('gauge-prob', [trace], layout, { displayModeBar: false, responsive: true });
}

function buildGaugeRisk(score, band) {
  const colorMap = {
    'Very Low Risk': '#22c55e', 'Low Risk': '#4ade80',
    'Moderate Risk': '#f59e0b', 'High Risk': '#f97316', 'Very High Risk': '#ef4444'
  };
  const color = colorMap[band] || '#ef4444';
  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: { l: 20, r: 20, t: 40, b: 10 }, height: 220, font: { color: '#a1a1aa', family: 'Inter' }
  };
  const trace = {
    type: 'indicator', mode: 'gauge+number',
    value: score,
    number: { font: { size: 34, color } },
    title: { text: `Risk Band: ${band}`, font: { size: 12, color: '#71717a' } },
    gauge: {
      axis: { range: [0, 100], tickcolor: '#3f3f46', tickwidth: 1, tickfont: { color: '#71717a', size: 11 } },
      bar: { color, thickness: 0.3 },
      bgcolor: 'transparent', bordercolor: '#3f3f46',
      steps: [
        { range: [0, 25], color: 'rgba(34,197,94,.12)' },
        { range: [25, 40], color: 'rgba(74,222,128,.08)' },
        { range: [40, 55], color: 'rgba(245,158,11,.10)' },
        { range: [55, 70], color: 'rgba(249,115,22,.12)' },
        { range: [70, 100], color: 'rgba(220,38,38,.18)' },
      ]
    }
  };
  Plotly.react('gauge-risk', [trace], layout, { displayModeBar: false, responsive: true });
}

/* ── Per-Model horizontal bars (Plotly) ─────────────────────────────────── */
function buildModelBars(modelPreds) {
  const names = Object.keys(modelPreds);
  const probs = names.map(n => modelPreds[n].prob_disease);
  const colors = probs.map(p => p > 50 ? '#ef4444' : '#22c55e');

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: { l: 180, r: 60, t: 10, b: 40 }, height: 260,
    xaxis: {
      range: [0, 110], title: { text: 'P(Heart Disease) %', font: { color: '#71717a', size: 11 } },
      tickcolor: '#3f3f46', gridcolor: 'rgba(255,255,255,.05)', color: '#71717a'
    },
    yaxis: { tickfont: { color: '#a1a1aa', size: 12 }, tickcolor: 'transparent' },
    shapes: [{
      type: 'line', x0: 50, x1: 50, y0: -0.5, y1: names.length - 0.5,
      line: { color: 'rgba(255,255,255,.35)', width: 1.5, dash: 'dot' }
    }],
    annotations: [{
      x: 51, y: names.length / 2, text: '50%', showarrow: false,
      font: { color: 'rgba(255,255,255,.4)', size: 10 }, xanchor: 'left'
    }],
    font: { family: 'Inter' }
  };

  const trace = {
    type: 'bar', orientation: 'h',
    x: probs, y: names,
    marker: { color: colors, cornerradius: 4 },
    text: probs.map(p => p.toFixed(1) + '%'), textposition: 'outside',
    textfont: { color: '#a1a1aa', size: 12 }, cliponaxis: false
  };
  Plotly.react('chart-models', [trace], layout, { displayModeBar: false, responsive: true });
}

/* ── Summary Table ────────────────────────────────────────────────────── */
function buildSummaryTable(p) {
  const rows = [
    ['Age', p.age, '29–77 yrs'],
    ['Sex', p.sex, '—'],
    ['Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic'][+p.cp], '0=Highest risk'],
    ['Resting BP', p.trestbps + ' mmHg', '90–120 mmHg normal'],
    ['Cholesterol', p.chol + ' mg/dL', '<200 mg/dL ideal'],
    ['Max Heart Rate', p.thalachh + ' bpm', '100–170 bpm'],
    ['ST Depression', p.oldpeak, '0 is ideal'],
    ['Slope', ['Upsloping', 'Flat', 'Downsloping'][+p.slope], 'Upsloping is best'],
    ['Fasting Blood Sugar >120', +p.fbs ? 'Yes' : 'No', '≤120 mg/dL normal'],
    ['Resting ECG', ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'][+p.restecg], 'Normal (0) ideal'],
    ['Exercise Angina', +p.exang ? 'Yes' : 'No', 'No is better'],
    ['Major Vessels', p.ca, '0 vessels ideal'],
    ['Thalassemia', { 3: 'Normal', 6: 'Fixed Defect', 7: 'Reversible Defect' }[p.thal], 'Normal (3) best'],
    ['Smoking Status', p.smoking, 'Non-smoker ideal'],
    ['Alcohol', p.alcohol, 'None ideal'],
    ['Exercise Level', p.exercise, 'High is ideal'],
    ['BMI Category', p.bmi, 'Normal ideal'],
  ];
  const tbl = document.getElementById('summary-table');
  tbl.innerHTML = `
    <thead><tr>
      <th>Feature</th><th>Patient Value</th><th>Normal Reference</th>
    </tr></thead>
    <tbody>
      ${rows.map(([f, v, n]) => `<tr><td>${f}</td><td><strong>${v}</strong></td><td style="color:var(--text2);font-size:.8rem">${n}</td></tr>`).join('')}
    </tbody>`;
}

/* ══════════════════════════════════════════════════════════════════════════
   ANALYTICS TAB
══════════════════════════════════════════════════════════════════════════ */
/* ── After prediction, reset analytics so re-visiting shows fresh data ── */
const _origExportState = window.switchTab;
let analyticsLoaded = false;
let chartAge, chartGender, chartRiskDist, chartLifestyle;

async function initAnalytics() {
  if (analyticsLoaded) return;
  analyticsLoaded = true;
  showLoading(true);
  try {
    const resp = await fetch('/api/dataset-stats');
    const d = await resp.json();

    /* KPI Cards */
    const healthyPct = (100 - +d.disease_pct).toFixed(1);

    // Also fetch this doctor's personal prediction history
    let predStats = { total: 0, today: 0, disease: 0, healthy: 0 };
    try {
      const pResp = await fetch('/api/prediction-stats');
      predStats = await pResp.json();
    } catch(e) { /* silent fallback */ }

    document.getElementById('analytics-stats').innerHTML = `
      <div class="as-card red">
        <div class="as-val" style="color:#ef4444">${d.disease.toLocaleString()}</div>
        <div class="as-lbl">Heart Disease Cases</div>
        <div class="as-note" style="color:#ef4444">${d.disease_pct}% of total</div>
      </div>
      <div class="as-card green">
        <div class="as-val" style="color:#22c55e">${d.healthy.toLocaleString()}</div>
        <div class="as-lbl">Healthy Cases</div>
        <div class="as-note" style="color:#22c55e">${healthyPct}% of total</div>
      </div>
      <div class="as-card gold">
        <div class="as-val" style="color:#f59e0b">${d.total.toLocaleString()}</div>
        <div class="as-lbl">Total Patients</div>
        <div class="as-note">Training + Test records</div>
      </div>
      <div class="as-card purple">
        <div class="as-val" style="color:#a855f7">${predStats.total}</div>
        <div class="as-lbl">My Predictions</div>
        <div class="as-note" style="color:#a855f7">${predStats.today} today</div>
      </div>`;

    /* Hero KPIs */
    document.getElementById('kpi-disease').textContent = d.disease.toLocaleString();
    document.getElementById('kpi-disease-pct').textContent = d.disease_pct + '%';
    document.getElementById('kpi-healthy').textContent = d.healthy.toLocaleString();
    document.getElementById('kpi-healthy-pct').textContent = healthyPct + '%';
    document.getElementById('h-total').textContent = d.total.toLocaleString();

    /* =========================================================================
       PowerBI Dense Charts
       ========================================================================= */
    const denseOpts = {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: CHART_DARK.plugins.tooltip },
      layout: { padding: 4 }
    };
    const denseScales = {
      x: { grid: { display: false }, ticks: { font: { size: 10 }, color: '#71717a' } },
      y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { font: { size: 10 }, color: '#71717a' } }
    };

    /* Age Chart */
    const ageLbls = d.age_rate.map(r => r.Age_Group);
    const ageRates = d.age_rate.map(r => +(r.rate * 100).toFixed(1));
    const ageCols = ageRates.map(r =>
      r < 40 ? 'rgba(34,197,94,.75)' : r < 55 ? 'rgba(245,158,11,.75)' : 'rgba(220,38,38,.8)');
    if (chartAge) chartAge.destroy();
    chartAge = new Chart(document.getElementById('chart-age'), {
      type: 'bar',
      data: {
        labels: ageLbls, datasets: [{
          label: 'Disease Rate (%)',
          data: ageRates, backgroundColor: ageCols, borderRadius: 4
        }]
      },
      options: { ...denseOpts, scales: denseScales }
    });

    /* Gender Chart */
    const genders = Object.keys(d.gender_stats);
    if (chartGender) chartGender.destroy();
    chartGender = new Chart(document.getElementById('chart-gender'), {
      type: 'bar',
      indexAxis: 'y', // Horizontal for tight fit
      data: {
        labels: genders,
        datasets: [
          {
            label: 'Heart Disease', data: genders.map(g => d.gender_stats[g].disease),
            backgroundColor: 'rgba(220,38,38,.75)', borderRadius: 4
          },
          {
            label: 'Healthy', data: genders.map(g => d.gender_stats[g].healthy),
            backgroundColor: 'rgba(34,197,94,.65)', borderRadius: 4
          }
        ]
      },
      options: {
        ...denseOpts, scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { font: { size: 10 }, color: '#71717a' } },
          y: { grid: { display: false }, ticks: { font: { size: 10 }, color: '#71717a' } }
        }
      }
    });

    /* Risk Dist Doughnut */
    const riskKeys = Object.keys(d.risk_dist);
    const riskVals = riskKeys.map(k => d.risk_dist[k]);
    const riskColors = {
      'Low Risk': 'rgba(34,197,94,.8)',
      'Moderate Risk': 'rgba(245,158,11,.8)', 'High Risk': 'rgba(220,38,38,.8)'
    };
    if (chartRiskDist) chartRiskDist.destroy();
    chartRiskDist = new Chart(document.getElementById('chart-risk-dist'), {
      type: 'doughnut',
      data: {
        labels: riskKeys, datasets: [{
          data: riskVals,
          backgroundColor: riskKeys.map(k => riskColors[k] || 'rgba(168,85,247,.8)'),
          borderColor: 'transparent', hoverOffset: 4
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false, cutout: '65%',
        plugins: { legend: { display: true, position: 'right', labels: { color: '#71717a', boxWidth: 10, font: { size: 10 } } }, tooltip: CHART_DARK.plugins.tooltip }
      }
    });

    /* ==========================================
       NEW ADVANCED CHARTS (Plotly)
       ========================================== */
    /* Detailed Age Histogram */
    if (document.getElementById('chart-age-hist')) {
      const traceAgeDisease = {
        x: d.age_dist.disease,
        type: 'histogram',
        name: '❤️ Heart Disease',
        opacity: 0.78,
        nbinsx: 25,
        marker: { color: 'rgba(239, 68, 68, 0.85)', line: { color: 'rgba(239,68,68,.4)', width: 0.5 } }
      };
      const traceAgeHealthy = {
        x: d.age_dist.healthy,
        type: 'histogram',
        name: '💚 Healthy',
        opacity: 0.6,
        nbinsx: 25,
        marker: { color: 'rgba(34, 197, 94, 0.7)', line: { color: 'rgba(34,197,94,.3)', width: 0.5 } }
      };
      const layoutAge = {
        barmode: 'overlay',
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        margin: { t: 20, l: 50, r: 20, b: 50 },
        font: { color: '#a1a1aa', family: 'Inter, sans-serif', size: 11 },
        legend: { x: 0.02, y: 0.96, bgcolor: 'rgba(0,0,0,0)', font: { color: '#e4e4e7' } },
        xaxis: { title: { text: 'Patient Age (years)', font: { color: '#71717a', size: 11 } }, gridcolor: 'rgba(255,255,255,0.05)', color: '#71717a' },
        yaxis: { title: { text: 'Patient Count', font: { color: '#71717a', size: 11 } }, gridcolor: 'rgba(255,255,255,0.05)', color: '#71717a' }
      };
      Plotly.newPlot('chart-age-hist', [traceAgeDisease, traceAgeHealthy], layoutAge, {responsive: true, displayModeBar: false});
    }

    /* Cholesterol vs Max HR Scatter */
    if (document.getElementById('chart-chol-scatter')) {
      const traceScatterDisease = {
        x: d.scatter_data.disease.map(p => p.Cholesterol),
        y: d.scatter_data.disease.map(p => p.Max_Heart_Rate),
        mode: 'markers', type: 'scatter', name: '❤️ Heart Disease',
        marker: { color: 'rgba(239, 68, 68, 0.65)', size: 5, line: { color: 'rgba(239,68,68,.3)', width: 0.5 } }
      };
      const traceScatterHealthy = {
        x: d.scatter_data.healthy.map(p => p.Cholesterol),
        y: d.scatter_data.healthy.map(p => p.Max_Heart_Rate),
        mode: 'markers', type: 'scatter', name: '💚 Healthy',
        marker: { color: 'rgba(34, 197, 94, 0.45)', size: 5, line: { color: 'rgba(34,197,94,.2)', width: 0.5 } }
      };
      const layoutScatter = {
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        margin: { t: 20, l: 55, r: 20, b: 50 },
        font: { color: '#a1a1aa', family: 'Inter, sans-serif', size: 11 },
        legend: { x: 0.02, y: 0.96, bgcolor: 'rgba(0,0,0,0)', font: { color: '#e4e4e7' } },
        xaxis: {
          title: { text: 'Cholesterol (mg/dL)', font: { color: '#71717a', size: 11 } },
          gridcolor: 'rgba(255,255,255,0.05)', color: '#71717a',
          zerolinecolor: 'rgba(255,255,255,0.1)'
        },
        yaxis: {
          title: { text: 'Max Heart Rate (bpm)', font: { color: '#71717a', size: 11 } },
          gridcolor: 'rgba(255,255,255,0.05)', color: '#71717a',
          zerolinecolor: 'rgba(255,255,255,0.1)'
        }
      };
      Plotly.newPlot('chart-chol-scatter', [traceScatterDisease, traceScatterHealthy], layoutScatter, {responsive: true, displayModeBar: false});
    }

    /* Lifestyle Chart */
    const smk = d.lifestyle_rates.Smoking_Status;
    const smkLabels = Object.keys(smk);
    if (chartLifestyle) chartLifestyle.destroy();
    chartLifestyle = new Chart(document.getElementById('chart-lifestyle'), {
      type: 'bar',
      data: {
        labels: smkLabels,
        datasets: [{
          label: 'Disease Rate (%)',
          data: smkLabels.map(k => (smk[k] * 100).toFixed(1)),
          backgroundColor: ['rgba(34,197,94,.75)', 'rgba(245,158,11,.75)', 'rgba(220,38,38,.75)'],
          borderRadius: 4
        }]
      },
      options: { ...denseOpts, scales: denseScales }
    });
  } catch (err) {
    console.error('Analytics error:', err);
  } finally {
    showLoading(false);
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   PERFORMANCE TAB
══════════════════════════════════════════════════════════════════════════ */
let perfLoaded = false;
let chartAcc, chartMet;

function initPerformance() {
  if (perfLoaded) return;
  perfLoaded = true;

  const models = ['SVM', 'Gradient Boosting', 'Logistic Reg.', 'Random Forest', 'KNN', 'Decision Tree'];
  const acc = [80.50, 79.71, 78.92, 78.88, 75.88, 73.54];
  const prec = [80.99, 80.21, 79.61, 79.20, 76.03, 75.29];
  const recall = [83.49, 82.87, 81.94, 82.56, 80.79, 75.93];
  const f1 = [82.22, 81.52, 80.76, 80.85, 78.34, 75.61];

  if (chartAcc) chartAcc.destroy();
  chartAcc = new Chart(document.getElementById('chart-accuracy'), {
    type: 'bar',
    data: {
      labels: models,
      datasets: [{
        label: 'Accuracy (%)', data: acc, borderRadius: 6,
        backgroundColor: acc.map((v, i) => i === 0
          ? 'rgba(220,38,38,.8)' : 'rgba(220,38,38,.38)')
      }]
    },
    options: {
      ...CHART_DARK,
      indexAxis: 'y',
      plugins: { ...CHART_DARK.plugins, legend: { display: false } },
      scales: {
        x: {
          ...CHART_DARK.scales.x, min: 65, max: 85,
          title: { display: true, text: 'Accuracy (%)', color: '#71717a' }
        },
        y: { ...CHART_DARK.scales.y }
      }
    }
  });

  if (chartMet) chartMet.destroy();
  chartMet = new Chart(document.getElementById('chart-metrics'), {
    type: 'bar',
    data: {
      labels: models,
      datasets: [
        { label: 'Accuracy', data: acc, backgroundColor: 'rgba(220,38,38,.7)', borderRadius: 4 },
        { label: 'Precision', data: prec, backgroundColor: 'rgba(168,85,247,.7)', borderRadius: 4 },
        { label: 'Recall', data: recall, backgroundColor: 'rgba(34,197,94,.7)', borderRadius: 4 },
        { label: 'F1 Score', data: f1, backgroundColor: 'rgba(245,158,11,.7)', borderRadius: 4 },
      ]
    },
    options: {
      ...CHART_DARK,
      scales: {
        x: { ...CHART_DARK.scales.x, ticks: { maxRotation: 30 } },
        y: {
          ...CHART_DARK.scales.y, min: 65, max: 88,
          title: { display: true, text: 'Score (%)', color: '#71717a' }
        }
      }
    }
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   FEATURES TAB (Dense PowerBI Style)
══════════════════════════════════════════════════════════════════════════ */
let featLoaded = false;
let chartFI, chartCorr;

function initFeatures() {
  if (featLoaded) return;
  featLoaded = true;

  /* PowerBI dense options */
  const denseOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false }, tooltip: CHART_DARK.plugins.tooltip },
    layout: { padding: 4 }
  };
  const denseScales = {
    x: { grid: { display: false }, ticks: { font: { size: 10 }, color: '#71717a' } },
    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { font: { size: 10 }, color: '#a1a1aa' } }
  };

  /* Feature Importance Chart (Top 8) */
  const fi = [...FI_DATA]
    .sort((a, b) => (b.rf + b.gb) / 2 - (a.rf + a.gb) / 2)
    .slice(0, 8); // Top 8 only

  if (chartFI) chartFI.destroy();
  chartFI = new Chart(document.getElementById('chart-fi'), {
    type: 'bar',
    data: {
      labels: fi.map(d => d.feature),
      datasets: [
        {
          label: 'Random Forest', data: fi.map(d => d.rf),
          backgroundColor: 'rgba(220,38,38,.75)', borderRadius: 3
        },
        {
          label: 'Gradient Boosting', data: fi.map(d => d.gb),
          backgroundColor: 'rgba(168,85,247,.65)', borderRadius: 3
        }
      ]
    },
    options: { ...denseOpts, indexAxis: 'y', scales: { x: denseScales.x, y: denseScales.y } }
  });

  /* Correlation Chart (Top 8 Absolute) */
  const corr = [...CORR_DATA]
    .sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr))
    .slice(0, 8); // Top 8 only

  if (chartCorr) chartCorr.destroy();
  chartCorr = new Chart(document.getElementById('chart-corr'), {
    type: 'bar',
    data: {
      labels: corr.map(d => d.feature),
      datasets: [{
        label: 'Correlation',
        data: corr.map(d => d.corr),
        backgroundColor: corr.map(d => d.corr > 0 ? 'rgba(220,38,38,.75)' : 'rgba(34,197,94,.7)'),
        borderRadius: 4
      }]
    },
    options: { ...denseOpts, indexAxis: 'y', scales: { x: denseScales.x, y: denseScales.y } }
  });

  /* Feature Importance Table (Top 8) */
  const sorted = [...FI_DATA].sort((a, b) => (b.rf + b.gb) / 2 - (a.rf + a.gb) / 2).slice(0, 8);
  const maxAvg = (sorted[0].rf + sorted[0].gb) / 2;
  document.getElementById('fi-table-body').innerHTML = sorted.map((d, i) => {
    const avg = ((d.rf + d.gb) / 2).toFixed(2);
    const pct = ((d.rf + d.gb) / 2 / maxAvg * 100).toFixed(1);
    return `<tr>
      <td><span class="rank ${i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : ''}">${i + 1}</span></td>
      <td><strong>${d.feature}</strong></td>
      <td>${d.rf.toFixed(2)}%</td>
      <td>${d.gb.toFixed(2)}%</td>
      <td><strong style="color:var(--text0)">${avg}%</strong></td>
      <td style="min-width:140px">
        <div class="fi-bar-wrap" style="height:6px; background:var(--bg3); border-radius:4px; overflow:hidden;">
          <div class="fi-bar" style="height:100%; width:${pct}%; background:linear-gradient(90deg,var(--accent1),var(--accent2))"></div>
        </div>
      </td>
    </tr>`;
  }).join('');
}

/* ══════════════════════════════════════════════════════════════════════════
   INITIAL LOAD
══════════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', async () => {
  /* Load dataset stats for hero KPIs */
  try {
    const r = await fetch('/api/dataset-stats');
    const d = await r.json();
    document.getElementById('h-total').textContent = d.total.toLocaleString();
    document.getElementById('kpi-disease').textContent = d.disease.toLocaleString();
    document.getElementById('kpi-disease-pct').textContent = d.disease_pct + '%';
    document.getElementById('kpi-healthy').textContent = d.healthy.toLocaleString();
    const hp = (100 - +d.disease_pct).toFixed(1);
    document.getElementById('kpi-healthy-pct').textContent = hp + '%';
  } catch (e) { /* silent */ }

  /* Open predict tab by default */
  switchTab('predict');
});
/* ══════════════════════════════════════════════════════════════════════════
   DOCUMENTATION TAB
══════════════════════════════════════════════════════════════════════════ */

/* Register docs tab in switchTab */
const _origSwitch = switchTab;
// Patch switchTab to handle 'docs'
window.switchTab = function (tabId) {
  document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  const section = document.getElementById('tab-' + tabId);
  if (section) section.classList.add('active');
  document.querySelectorAll('.nav-link').forEach(l => {
    if (l.dataset.tab === tabId) l.classList.add('active');
  });
  if (tabId === 'analytics') initAnalytics();
  if (tabId === 'performance') initPerformance();
  if (tabId === 'features') initFeatures();
};

/* Scroll to a specific doc section */
function scrollToDoc(id) {
  const el = document.getElementById('doc-' + id);
  if (!el) return;
  // Show docs tab first
  switchTab('docs');
  setTimeout(() => {
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    // Highlight briefly
    el.style.borderColor = 'rgba(168,85,247,.6)';
    setTimeout(() => { el.style.borderColor = ''; }, 1800);
    // Update TOC active state
    document.querySelectorAll('.docs-toc-link').forEach(l => l.classList.remove('active'));
  }, 100);
}

/* Search / filter docs */
function filterDocs(query) {
  const q = query.toLowerCase().trim();
  document.querySelectorAll('.doc-card').forEach(card => {
    if (!q) { card.classList.remove('hidden'); return; }
    const searchText = (card.dataset.search || '') + ' ' +
      (card.querySelector('.doc-title')?.textContent || '') + ' ' +
      (card.querySelector('.doc-body')?.textContent || '');
    card.classList.toggle('hidden', !searchText.toLowerCase().includes(q));
  });
}

/* ── FIX: Updated realistic presets ──────────────────────────────────────── */
/* Override PRESETS with calibrated real patient values */
Object.assign(PRESETS, {
  high: {
    /* Real test patient idx=10 · RF=77.9% GB=87.1% LR=75.4% SVM=83.1% · AVG ~81% */
    age: 63, sex: 'Male', cp: 3, trestbps: 147, chol: 141, thalachh: 139,
    oldpeak: 1.2, slope: 1, fbs: 0, restecg: 0, exang: 0, ca: 0, thal: 7,
    smoking: 'Non-Smoker', alcohol: 'High', exercise: 'High', bmi: 'Overweight'
  },
  low: {
    /* Real test patient idx=57 · RF=36.6% GB=13.6% LR=19.8% SVM=18.6% · AVG ~22% */
    age: 50, sex: 'Male', cp: 2, trestbps: 130, chol: 352, thalachh: 170,
    oldpeak: 0.8, slope: 2, fbs: 0, restecg: 0, exang: 0, ca: 0, thal: 3,
    smoking: 'Current Smoker', alcohol: 'None', exercise: 'High', bmi: 'Obese'
  },
  moderate: {
    /* Real test patient idx=29 · RF=52.2% GB=65.9% LR=50.7% SVM=59.9% · AVG ~57% */
    age: 58, sex: 'Male', cp: 2, trestbps: 143, chol: 248, thalachh: 150,
    oldpeak: 0.8, slope: 0, fbs: 0, restecg: 0, exang: 0, ca: 0, thal: 7,
    smoking: 'Former Smoker', alcohol: 'High', exercise: 'Low', bmi: 'Overweight'
  }
});

/* ══════════════════════════════════════════════════════════════════════════
   PATIENT HISTORY TAB
══════════════════════════════════════════════════════════════════════════ */
let historyPage = 1;
const HIST_PER_PAGE = 20;
let historyLoaded = false;

/* Called by switchTab patching below */
async function initHistory() {
  if (historyLoaded) return;
  historyLoaded = true;
  await loadHistory(1);
}

/* Reset so fresh data loads when returning after a new prediction */
document.addEventListener('cardio:newprediction', () => {
  historyLoaded = false;
});

function applyHistoryFilters() {
  historyLoaded = false;
  loadHistory(1);
}
function clearHistoryFilters() {
  document.getElementById('hist-start').value = '';
  document.getElementById('hist-end').value   = '';
  document.getElementById('hist-verdict').value = '';
  historyLoaded = false;
  loadHistory(1);
}

function buildHistoryURL(page) {
  const getVal = (id) => { const el = document.getElementById(id); return el ? el.value : ''; };
  const start   = getVal('hist-start') || getVal('rep-start');
  const end     = getVal('hist-end') || getVal('rep-end');
  const verdict = getVal('hist-verdict') || getVal('rep-verdict');
  let url = `/api/history?page=${page}&per_page=${HIST_PER_PAGE}`;
  if (start)   url += `&start=${start}`;
  if (end)     url += `&end=${end}`;
  if (verdict) url += `&risk=${verdict}`;
  return url;
}

async function loadHistory(page = 1) {
  historyPage = page;
  const tbody = document.getElementById('hist-tbody');
  tbody.innerHTML = `<tr><td colspan="11" style="text-align:center;color:var(--text2);padding:32px">⏳ Loading…</td></tr>`;

  try {
    const resp = await fetch(buildHistoryURL(page));
    const d    = await resp.json();

    /* ── KPI row ── */
    const total   = d.total;
    const disease = d.records.filter ? null : null; // full counts come from trend
    const allRecs = d.trend || [];
    const disCount = allRecs.filter(r => r.pred === 'Disease').length;
    const hlCount  = allRecs.filter(r => r.pred === 'Healthy').length;
    const avgRisk  = allRecs.length
      ? Math.round(allRecs.reduce((s, r) => s + (r.risk || 0), 0) / allRecs.length)
      : 0;

    // today count
    const todayStr = new Date().toISOString().slice(0, 10);
    const todayCount = allRecs.filter(r => r.ts && r.ts.slice(0, 10) === todayStr).length;

    const safeSetText = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
    safeSetText('hk-total', total);
    safeSetText('hk-disease', disCount);
    safeSetText('hk-healthy', hlCount);
    safeSetText('hk-today', todayCount);
    safeSetText('hk-avg-risk', avgRisk || '—');

    /* ── Trend chart ── */
    renderHistoryTrend(d.trend || []);

    /* ── Table ── */
    safeSetText('hist-count', total);
    safeSetText('hist-count-badge', total); // support doctor dashboard
    const pagesInfo = d.pages > 1
      ? `(Page ${d.page} of ${d.pages})`
      : '';
    safeSetText('hist-pages-info', pagesInfo);

    if (d.records.length === 0) {
      tbody.innerHTML = `
        <tr><td colspan="11">
          <div class="hist-empty">
            <div class="hist-empty-icon">📋</div>
            <div class="hist-empty-title">No predictions found</div>
            <div class="hist-empty-sub">Run a prediction on the Predict tab to see history here.</div>
          </div>
        </td></tr>`;
    } else {
      const riskColor = (score) => {
        if (score < 25)  return '#22c55e';
        if (score < 40)  return '#4ade80';
        if (score < 55)  return '#f59e0b';
        if (score < 70)  return '#f97316';
        return '#ef4444';
      };
      tbody.innerHTML = d.records.map((r, i) => {
        const score  = r.risk_score || 0;
        const color  = riskColor(score);
        const isDisease = r.final_prediction === 'Disease';
        const verdictHtml = isDisease
          ? `<span class="verdict-chip verdict-disease">❤️ Disease</span>`
          : `<span class="verdict-chip verdict-healthy">💚 Healthy</span>`;
        const probStr = r.avg_prob != null ? r.avg_prob.toFixed(1) + '%' : '—';
        const votesStr = r.votes_yes != null ? `${r.votes_yes}/6` : '—';
        const label = r.patient_label || `<span style="color:var(--text2);font-style:italic">Unlabelled</span>`;
        return `<tr>
          <td style="color:var(--text2);font-size:.78rem">${(page - 1) * HIST_PER_PAGE + i + 1}</td>
          <td style="white-space:nowrap;font-size:.82rem">${r.timestamp}</td>
          <td>${label}</td>
          <td>${r.age ?? '—'}</td>
          <td>${r.sex ?? '—'}</td>
          <td>
            <div class="risk-score-cell">
              <strong style="color:${color}">${score}</strong>
              <div class="risk-mini-bar">
                <div class="risk-mini-fill" style="width:${score}%;background:${color}"></div>
              </div>
            </div>
          </td>
          <td><span style="font-size:.78rem;color:${color}">${r.risk_band}</span></td>
          <td style="color:${isDisease ? '#f87171' : '#4ade80'}">${probStr}</td>
          <td style="font-size:.82rem">${votesStr}</td>
          <td>${verdictHtml}</td>
          <td>
            <button class="hist-delete-btn" onclick="deleteHistoryRecord(${r.id}, this)">🗑</button>
          </td>
        </tr>`;
      }).join('');
    }

    /* ── Pagination ── */
    renderHistoryPagination(d.page, d.pages);

  } catch(err) {
    tbody.innerHTML = `<tr><td colspan="11" style="text-align:center;color:#f87171;padding:32px">⚠️ Failed to load: ${err.message}</td></tr>`;
  }
}

function renderHistoryTrend(trend) {
  const el = document.getElementById('chart-history-trend');
  if (!el || !trend.length) {
    if (el) el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text2);font-size:.9rem">No data yet — run your first prediction!</div>`;
    return;
  }

  const xs      = trend.map(r => r.ts.slice(0, 10));
  const ys      = trend.map(r => r.risk);
  const colors  = trend.map(r => r.pred === 'Disease'
    ? 'rgba(239,68,68,0.85)'
    : 'rgba(34,197,94,0.75)');
  const borders = trend.map(r => r.pred === 'Disease' ? '#ef4444' : '#22c55e');

  const traceLine = {
    x: xs, y: ys, type: 'scatter', mode: 'lines',
    line: { color: 'rgba(255,255,255,0.08)', width: 1.5 },
    showlegend: false, hoverinfo: 'skip'
  };
  const traceDots = {
    x: xs, y: ys, type: 'scatter', mode: 'markers',
    name: 'Risk Score',
    marker: {
      color: colors,
      size: 9,
      line: { color: borders, width: 1.5 }
    },
    hovertemplate: '<b>Date:</b> %{x}<br><b>Risk Score:</b> %{y}<extra></extra>'
  };

  // 7-day rolling average
  const window7 = 7;
  const rolling = ys.map((_, i) => {
    const slice = ys.slice(Math.max(0, i - window7 + 1), i + 1);
    return Math.round(slice.reduce((a, b) => a + b, 0) / slice.length);
  });
  const traceAvg = {
    x: xs, y: rolling, type: 'scatter', mode: 'lines',
    name: '7-pt avg',
    line: { color: 'rgba(168,85,247,0.7)', width: 2, dash: 'dot' },
    hovertemplate: '<b>7-pt avg:</b> %{y}<extra></extra>'
  };

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: { t: 10, l: 45, r: 20, b: 45 },
    font: { color: '#a1a1aa', family: 'Inter, sans-serif', size: 11 },
    legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0)', font: { color: '#e4e4e7', size: 11 } },
    xaxis: { gridcolor: 'rgba(255,255,255,0.05)', color: '#71717a', type: 'category',
             tickangle: -30, nticks: Math.min(trend.length, 14) },
    yaxis: { title: { text: 'Risk Score (0–100)', font: { color: '#71717a', size: 10 } },
             range: [0, 105], gridcolor: 'rgba(255,255,255,0.05)', color: '#71717a' },
    shapes: [
      { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 55, y1: 55,
        line: { color: 'rgba(249,115,22,0.3)', width: 1, dash: 'dot' } },
      { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 70, y1: 70,
        line: { color: 'rgba(239,68,68,0.3)', width: 1, dash: 'dot' } },
    ]
  };
  Plotly.newPlot('chart-history-trend', [traceLine, traceDots, traceAvg], layout,
    { responsive: true, displayModeBar: false });
}

function renderHistoryPagination(page, pages) {
  const el = document.getElementById('hist-pagination');
  if (!el || pages <= 1) { if(el) el.innerHTML = ''; return; }
  let html = '';
  html += `<button class="hist-page-btn" onclick="loadHistory(${page - 1})" ${page === 1 ? 'disabled' : ''}>← Prev</button>`;
  for (let p = 1; p <= pages; p++) {
    if (pages > 7 && Math.abs(p - page) > 2 && p !== 1 && p !== pages) {
      if (p === 2 || p === pages - 1) html += `<span style="color:var(--text2);padding:0 4px">…</span>`;
      continue;
    }
    html += `<button class="hist-page-btn ${p === page ? 'active' : ''}" onclick="loadHistory(${p})">${p}</button>`;
  }
  html += `<button class="hist-page-btn" onclick="loadHistory(${page + 1})" ${page === pages ? 'disabled' : ''}>Next →</button>`;
  el.innerHTML = html;
}

async function deleteHistoryRecord(id, btn) {
  if (!confirm('Delete this record permanently?')) return;
  btn.disabled = true;
  try {
    const r = await fetch(`/api/history/${id}`, { method: 'DELETE' });
    const d = await r.json();
    if (d.success) {
      historyLoaded = false;
      loadHistory(historyPage);
    } else {
      alert('Delete failed: ' + d.error);
      btn.disabled = false;
    }
  } catch(e) {
    alert('Error: ' + e.message);
    btn.disabled = false;
  }
}

function exportHistoryCSV() {
  const start   = document.getElementById('hist-start').value;
  const end     = document.getElementById('hist-end').value;
  const verdict = document.getElementById('hist-verdict').value;
  let url = '/api/history/export-csv?';
  if (start)   url += `&start=${start}`;
  if (end)     url += `&end=${end}`;
  if (verdict) url += `&risk=${verdict}`;
  window.location.href = url;
}

/* Patch switchTab to handle 'history' */
const _histSwitchTab = window.switchTab;
window.switchTab = function(tabId) {
  document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  const section = document.getElementById('tab-' + tabId);
  if (section) section.classList.add('active');
  document.querySelectorAll('.nav-link').forEach(l => {
    if (l.dataset.tab === tabId) l.classList.add('active');
  });
  if (tabId === 'analytics')   initAnalytics();
  if (tabId === 'performance') initPerformance();
  if (tabId === 'features')    initFeatures();
  if (tabId === 'history')     initHistory();
};
