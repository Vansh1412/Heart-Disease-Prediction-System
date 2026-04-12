/* ==========================================================================
   CardioAI — admin_app.js  |  Admin Console Frontend
   ========================================================================== */
'use strict';

/* ── State ─────────────────────────────────────────────────────────────── */
let OVERVIEW_DATA = null;
let DOCTORS_DATA  = [];
let predPage      = 1;

const PAGES = {
  overview:    { title:'System Overview',      sub:'Real-time platform intelligence' },
  analytics:   { title:'Dataset Analytics',    sub:'Full exploration of 12,687 patient records' },
  ml:          { title:'ML Model Performance', sub:'6-algorithm ensemble analysis & feature importance' },
  doctors:     { title:'Doctor Management',    sub:'All 15 registered physicians — manage access & view cohorts' },
  hospitals:   { title:'Hospital Network',     sub:'8 partner hospitals across India' },
  predictions: { title:'Prediction Logs',      sub:'All AI assessments run by doctors on the platform' },
  predict:     { title:'Run Assessment',       sub:'Test the ensemble pipeline from admin console' },
};

const FI_DATA = [
  {feature:'Max_Heart_Rate',rf:11.64,gb:15.25},{feature:'Age',rf:10.10,gb:8.31},
  {feature:'Major_Vessels',rf:8.64,gb:9.35},{feature:'Smoking_Status',rf:8.22,gb:8.63},
  {feature:'Exercise_Level',rf:6.01,gb:8.84},{feature:'Alcohol_Consumption',rf:5.99,gb:6.04},
  {feature:'Age_Sex_Interact',rf:6.51,gb:4.18},{feature:'Chest_Pain_Type',rf:5.09,gb:6.46},
  {feature:'BP_Chol_Score',rf:4.76,gb:5.60},{feature:'ST_Depression',rf:5.11,gb:4.32},
];
const CORR_DATA = [
  {feature:'Thalassemia',corr:0.52},{feature:'Major_Vessels',corr:0.48},
  {feature:'Exercise_Induced_Angina',corr:0.43},{feature:'ST_Depression',corr:0.43},
  {feature:'Sex',corr:0.27},{feature:'Age',corr:0.23},{feature:'Trestbps',corr:0.16},
  {feature:'Slope',corr:-0.35},{feature:'Max_Heart_Rate',corr:-0.42},
  {feature:'Chest_Pain_Type',corr:-0.43},
];

/* ── Chart defaults ─────────────────────────────────────────────────────── */
Chart.defaults.font.family = "'DM Sans', sans-serif";
Chart.defaults.color = '#6e7681';

/* Global safe chart creator — handles zero-dimension containers in hidden tabs */
function safeAdmChart(id, config) {
  try {
    const el = document.getElementById(id);
    if (!el) return null;
    const ex = Chart.getChart(id);
    if (ex) ex.destroy();
    const par = el.parentElement;
    if (par) {
      const w = par.offsetWidth  || par.clientWidth  || 400;
      const h = par.offsetHeight || par.clientHeight || 220;
      el.width  = Math.max(w, 100);
      el.height = Math.max(h, 100);
    }
    if (!config.options) config.options = {};
    config.options.responsive = true;
    config.options.maintainAspectRatio = false;
    return new Chart(el, config);
  } catch(e) {
    console.error('Chart error [' + id + ']:', e.message);
    return null;
  }
}

const DARK_OPTS = {
  responsive:true, maintainAspectRatio:false,
  plugins:{
    legend:{labels:{color:'#8b949e',boxWidth:11}},
    tooltip:{backgroundColor:'#1c2333',titleColor:'#f0f6fc',bodyColor:'#8b949e',
             borderColor:'rgba(255,255,255,.1)',borderWidth:1,padding:10}
  },
  scales:{
    x:{ticks:{color:'#6e7681'},grid:{color:'rgba(255,255,255,.04)'}},
    y:{ticks:{color:'#6e7681'},grid:{color:'rgba(255,255,255,.04)'}},
  }
};

/* ══════════════════════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  startClock();
  loadOverview();
});

function startClock() {
  const el = document.getElementById('adm-clock');
  const tick = () => {
    el.textContent = new Date().toLocaleString('en-IN',{
      day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false
    });
  };
  tick(); setInterval(tick, 1000);
}

/* ══════════════════════════════════════════════════════════════════════════
   TAB NAVIGATION
══════════════════════════════════════════════════════════════════════════ */
async function switchTab(tab) {
  document.querySelectorAll('.adm-tab').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.adm-nav-link').forEach(l => {
    l.classList.toggle('active', l.dataset.tab === tab);
  });
  const el = document.getElementById('tab-' + tab);
  if (el) el.classList.add('active');

  const p = PAGES[tab] || {};
  document.getElementById('adm-page-title').textContent = p.title || '';
  document.getElementById('adm-page-sub').textContent   = p.sub   || '';

  if (tab === 'analytics') initAdminAnalytics();
  if (tab === 'ml')        initAdminML();
  if (tab === 'doctors'  && DOCTORS_DATA.length === 0) loadDoctors();
  if (tab === 'hospitals') {
    if (!OVERVIEW_DATA) { showOverlay(true); await loadOverview(); showOverlay(false); }
    if (OVERVIEW_DATA) renderHospitals(OVERVIEW_DATA.hospital_summary);
  }
  if (tab === 'predictions') loadPredictions(1);
}

/* ══════════════════════════════════════════════════════════════════════════
   OVERVIEW
══════════════════════════════════════════════════════════════════════════ */
async function loadOverview() {
  showOverlay(true);
  try {
    const r = await fetch('/api/admin/overview');
    const d = await r.json();
    OVERVIEW_DATA = d;
    renderKPIs(d);
    renderOverviewCharts(d);
    renderHospitalTable(d.hospital_summary);
  } catch(e) { console.error('Overview load error:', e); }
  finally { showOverlay(false); }
  // If analytics tab is already active, render now
  const aTab = document.getElementById('tab-analytics');
  if (aTab && aTab.classList.contains('active') && !analyticsInit && OVERVIEW_DATA) {
    renderAnalytics(OVERVIEW_DATA);
  }
}

function renderKPIs(d) {
  document.getElementById('kpi-patients').textContent      = d.total_patients.toLocaleString();
  document.getElementById('kpi-patients-note').textContent = `${d.disease_pct}% disease rate`;
  document.getElementById('kpi-disease').textContent       = d.disease_count.toLocaleString();
  document.getElementById('kpi-disease-note').textContent  = `${d.disease_pct}% of all patients`;
  document.getElementById('kpi-healthy').textContent       = d.healthy_count.toLocaleString();
  document.getElementById('kpi-healthy-note').textContent  = `${(100-d.disease_pct).toFixed(1)}% of all patients`;
  document.getElementById('kpi-doctors').textContent       = d.active_doctors;
  document.getElementById('kpi-doctors-note').textContent  = `${d.total_doctors} total registered`;
  document.getElementById('kpi-preds').textContent         = d.total_preds.toLocaleString();
  document.getElementById('kpi-preds-note').textContent    = `${d.preds_today} today`;
}

function renderOverviewCharts(d) {
  // Risk Pie
  const riskKeys = Object.keys(d.risk_dist);
  const riskCols = {'High Risk':'rgba(248,81,73,.8)','Low Risk':'rgba(63,185,80,.8)','Moderate Risk':'rgba(210,153,34,.8)'};
  safeAdmChart('chart-risk-pie', {
    type:'doughnut',
    data:{ labels:riskKeys, datasets:[{ data:riskKeys.map(k=>d.risk_dist[k]),
      backgroundColor:riskKeys.map(k=>riskCols[k]||'rgba(163,113,247,.8)'),
      borderColor:'transparent',hoverOffset:5 }] },
    options:{ responsive:true, maintainAspectRatio:false, cutout:'65%',
      plugins:{ legend:{position:'right',labels:{color:'#8b949e',boxWidth:10,font:{size:11}}},
                tooltip:DARK_OPTS.plugins.tooltip } }
  });

  // Gender
  const gKeys = Object.keys(d.gender_dist);
  safeAdmChart('chart-gender', {
    type:'bar', indexAxis:'y',
    data:{ labels:gKeys, datasets:[{ label:'Patients',
      data:gKeys.map(k=>d.gender_dist[k]),
      backgroundColor:['rgba(88,166,255,.75)','rgba(163,113,247,.75)'],borderRadius:5 }] },
    options:{ ...DARK_OPTS, plugins:{...DARK_OPTS.plugins,legend:{display:false}} }
  });

  // Region disease rate
  const regions = d.region_data || [];
  safeAdmChart('chart-region', {
    type:'bar',
    data:{ labels:regions.map(r=>r.Region),
      datasets:[{ label:'Disease Rate %', data:regions.map(r=>r.disease_pct),
        backgroundColor:regions.map(r=>r.disease_pct>55?'rgba(248,81,73,.75)':'rgba(88,166,255,.65)'),
        borderRadius:5 }] },
    options:{ ...DARK_OPTS, plugins:{...DARK_OPTS.plugins,legend:{display:false}},
      scales:{ ...DARK_OPTS.scales, y:{...DARK_OPTS.scales.y,
        title:{display:true,text:'Disease Rate (%)',color:'#6e7681'}} } }
  });
}

function renderHospitalTable(hospitals) {
  const tbody = document.getElementById('hospital-overview-body');
  if (!hospitals || !hospitals.length) { tbody.innerHTML = '<tr><td colspan="10" class="adm-loading">No data</td></tr>'; return; }
  tbody.innerHTML = hospitals.map(h => {
    const dr = h.disease_pct;
    const drColor = dr > 55 ? '#f85149' : dr > 45 ? '#d29922' : '#3fb950';
    const icuChip = h.icu === 'Yes'
      ? '<span class="adm-chip adm-chip-green">✓ ICU</span>'
      : '<span class="adm-chip adm-chip-amber">No ICU</span>';
    return `<tr>
      <td><strong style="color:var(--adm-text0)">${h.name}</strong></td>
      <td>${h.city}</td>
      <td><span class="adm-chip adm-chip-blue">${h.type.split(' ')[0]}</span></td>
      <td>${h.beds?.toLocaleString()}</td>
      <td>${icuChip}</td>
      <td><span class="adm-chip adm-chip-purple">${h.accr}</span></td>
      <td style="font-weight:700;color:var(--adm-purple)">${h.doctor_count}</td>
      <td style="font-weight:700">${h.patient_count?.toLocaleString()}</td>
      <td style="font-weight:700;color:${drColor}">${dr}%</td>
      <td class="adm-rating">★ ${h.rating}</td>
    </tr>`;
  }).join('');
}

/* ══════════════════════════════════════════════════════════════════════════
   ANALYTICS
══════════════════════════════════════════════════════════════════════════ */
let analyticsInit = false;
async function initAdminAnalytics() {
  if (!OVERVIEW_DATA) {
    showOverlay(true);
    await loadOverview();
    showOverlay(false);
  }
  if (!analyticsInit && OVERVIEW_DATA) {
    renderAnalytics(OVERVIEW_DATA);
  }
}
async function renderAnalytics(d) {
  if (analyticsInit || !d) return;
  analyticsInit = true;
  // Triple RAF + 300ms to guarantee the tab is fully painted before any chart renders
  await new Promise(r => requestAnimationFrame(r));
  await new Promise(r => requestAnimationFrame(r));
  await new Promise(r => requestAnimationFrame(r));
  await new Promise(r => setTimeout(r, 300));

  // Age histogram (Plotly)
  const el = document.getElementById('chart-age-hist');
  if (el && d.age_dist_raw) {
    Plotly.newPlot(el, [
      { x:d.age_dist_raw.disease, type:'histogram', name:'❤️ Disease', opacity:.78,
        nbinsx:25, marker:{color:'rgba(248,81,73,.85)'} },
      { x:d.age_dist_raw.healthy, type:'histogram', name:'✅ Healthy', opacity:.6,
        nbinsx:25, marker:{color:'rgba(63,185,80,.7)'} },
    ], {
      barmode:'overlay', paper_bgcolor:'transparent', plot_bgcolor:'transparent',
      margin:{t:20,l:45,r:20,b:45}, font:{color:'#8b949e',family:'DM Sans',size:11},
      legend:{x:.02,y:.96,bgcolor:'rgba(0,0,0,0)',font:{color:'#f0f6fc'}},
      xaxis:{title:{text:'Age (years)',font:{color:'#6e7681',size:10}},gridcolor:'rgba(255,255,255,.04)',color:'#6e7681'},
      yaxis:{title:{text:'Count',font:{color:'#6e7681',size:10}},gridcolor:'rgba(255,255,255,.04)',color:'#6e7681'},
    }, {responsive:true, displayModeBar:false});
  }

  // Scatter (Plotly)
  const sc = document.getElementById('chart-scatter');
  if (sc && d.scatter) {
    Plotly.newPlot(sc, [
      { x:d.scatter.disease.map(p=>p.Cholesterol), y:d.scatter.disease.map(p=>p.Max_Heart_Rate),
        mode:'markers', type:'scatter', name:'Disease',
        marker:{color:'rgba(248,81,73,.55)',size:5} },
      { x:d.scatter.healthy.map(p=>p.Cholesterol), y:d.scatter.healthy.map(p=>p.Max_Heart_Rate),
        mode:'markers', type:'scatter', name:'Healthy',
        marker:{color:'rgba(63,185,80,.4)',size:5} },
    ], {
      paper_bgcolor:'transparent', plot_bgcolor:'transparent',
      margin:{t:20,l:50,r:20,b:50}, font:{color:'#8b949e',family:'DM Sans',size:11},
      legend:{x:.02,y:.96,bgcolor:'rgba(0,0,0,0)',font:{color:'#f0f6fc'}},
      xaxis:{title:{text:'Cholesterol (mg/dL)',font:{color:'#6e7681',size:10}},gridcolor:'rgba(255,255,255,.04)',color:'#6e7681'},
      yaxis:{title:{text:'Max Heart Rate (bpm)',font:{color:'#6e7681',size:10}},gridcolor:'rgba(255,255,255,.04)',color:'#6e7681'},
    }, {responsive:true, displayModeBar:false});
  }

  // Safe Chart.js wrapper - sets explicit dimensions to avoid zero-height canvas
  function admChart(id, config) {
    try {
      const el = document.getElementById(id);
      if (!el) return null;
      const ex = Chart.getChart(id);
      if (ex) ex.destroy();
      const par = el.parentElement;
      const w = par ? par.offsetWidth : 400;
      const h = par ? par.offsetHeight : 220;
      if (w > 0) { el.width = w; el.height = h > 0 ? h : 220; }
      if (!config.options) config.options = {};
      config.options.responsive = true;
      config.options.maintainAspectRatio = false;
      return new Chart(el, config);
    } catch(e) { console.error('Admin chart error', id, e.message); return null; }
  }

  // Age Group bar
  const ageGrp = d.age_dist || {};
  const ageOrder = ['Young','Middle Age','Senior','Elderly'];
  const ageSorted = ageOrder.filter(k => ageGrp[k]);
  safeAdmChart('chart-age-grp', {
    type:'bar',
    data:{ labels:ageSorted, datasets:[{
      label:'Patients', data:ageSorted.map(k=>ageGrp[k]),
      backgroundColor:['rgba(63,185,80,.75)','rgba(88,166,255,.75)','rgba(210,153,34,.75)','rgba(248,81,73,.75)'],
      borderRadius:5 }] },
    options:{ ...DARK_OPTS, plugins:{...DARK_OPTS.plugins,legend:{display:false}} }
  });

  // Smoking
  const smk = d.smoke_dist || {};
  safeAdmChart('chart-smoking', {
    type:'bar',
    data:{ labels:Object.keys(smk), datasets:[{
      label:'Patients', data:Object.values(smk),
      backgroundColor:['rgba(63,185,80,.75)','rgba(210,153,34,.75)','rgba(248,81,73,.75)'],
      borderRadius:5 }] },
    options:{ ...DARK_OPTS, plugins:{...DARK_OPTS.plugins,legend:{display:false}} }
  });

  // Risk donut
  const rk = d.risk_dist || {};
  const rkKeys = Object.keys(rk);
  const rkCols = {'High Risk':'rgba(248,81,73,.8)','Low Risk':'rgba(63,185,80,.8)','Moderate Risk':'rgba(210,153,34,.8)'};
  safeAdmChart('chart-risk-donut', {
    type:'doughnut',
    data:{ labels:rkKeys, datasets:[{ data:rkKeys.map(k=>rk[k]),
      backgroundColor:rkKeys.map(k=>rkCols[k]||'rgba(163,113,247,.8)'),
      borderColor:'transparent',hoverOffset:5 }] },
    options:{ responsive:true, maintainAspectRatio:false, cutout:'65%',
      plugins:{legend:{position:'right',labels:{color:'#8b949e',boxWidth:10,font:{size:10}}},
               tooltip:DARK_OPTS.plugins.tooltip} }
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   ML PERFORMANCE
══════════════════════════════════════════════════════════════════════════ */
let mlInit = false;
async function initAdminML() {
  if (!OVERVIEW_DATA) { showOverlay(true); await loadOverview(); showOverlay(false); }
  if (!mlInit && OVERVIEW_DATA) renderML();
}
async function renderML() {
  if (mlInit || !OVERVIEW_DATA) return;
  mlInit = true;
  await new Promise(r => requestAnimationFrame(r));
  await new Promise(r => requestAnimationFrame(r));
  await new Promise(r => requestAnimationFrame(r));
  await new Promise(r => setTimeout(r, 300));
  const models = OVERVIEW_DATA.model_perf;
  const names  = models.map(m => m.model.replace('K-Nearest Neighbors','KNN').replace('Support Vector Machine','SVM').replace('Logistic Regression','Log. Reg.').replace('Gradient Boosting','Grad. Boost').replace('Random Forest','Rand. Forest').replace('Decision Tree','Dec. Tree'));
  const acc    = models.map(m => m.accuracy);

  // Safe chart helper for ML section
  function admChart(id, config) {
    try {
      const el = document.getElementById(id);
      if (!el) return null;
      const ex = Chart.getChart(id);
      if (ex) ex.destroy();
      const par = el.parentElement;
      const w = par ? par.offsetWidth : 400;
      const h = par ? par.offsetHeight : 280;
      if (w > 0) { el.width = w; el.height = h > 0 ? h : 280; }
      if (!config.options) config.options = {};
      config.options.responsive = true;
      config.options.maintainAspectRatio = false;
      return new Chart(el, config);
    } catch(e) { console.error('ML chart error', id, e.message); return null; }
  }
  // Accuracy bar
  safeAdmChart('chart-ml-acc', {
    type:'bar', indexAxis:'y',
    data:{ labels:names, datasets:[{ label:'Accuracy (%)', data:acc,
      backgroundColor:acc.map((v,i)=>i===0?'rgba(248,81,73,.85)':'rgba(88,166,255,.55)'),
      borderRadius:5 }] },
    options:{ ...DARK_OPTS, plugins:{...DARK_OPTS.plugins,legend:{display:false}},
      scales:{ x:{...DARK_OPTS.scales.x,min:65,max:85}, y:{...DARK_OPTS.scales.y} } }
  });

  // Multi-metric
  safeAdmChart('chart-ml-multi', {
    type:'bar',
    data:{ labels:names, datasets:[
      {label:'Accuracy',  data:models.map(m=>m.accuracy),  backgroundColor:'rgba(248,81,73,.75)',borderRadius:3},
      {label:'Precision', data:models.map(m=>m.precision), backgroundColor:'rgba(163,113,247,.7)',borderRadius:3},
      {label:'Recall',    data:models.map(m=>m.recall),    backgroundColor:'rgba(63,185,80,.7)',  borderRadius:3},
      {label:'F1 Score',  data:models.map(m=>m.f1),        backgroundColor:'rgba(210,153,34,.7)', borderRadius:3},
    ] },
    options:{ ...DARK_OPTS, scales:{
      x:{...DARK_OPTS.scales.x,ticks:{maxRotation:30}},
      y:{...DARK_OPTS.scales.y,min:65,max:88} } }
  });

  // Table
  const ranks = ['🥇','🥈','🥉','4','5','6'];
  document.getElementById('ml-perf-body').innerHTML = models.map((m,i) => `<tr>
    <td style="font-weight:700">${ranks[i]}</td>
    <td><strong style="color:var(--adm-text0)">${m.model}</strong></td>
    <td><div class="adm-bar-cell">
      <span style="color:${i===0?'var(--adm-red)':'var(--adm-text0)'};font-weight:600">${m.accuracy}%</span>
      <div class="adm-bar-mini"><div class="adm-bar-mini-fill" style="width:${(m.accuracy-65)/20*100}%;background:${i===0?'#f85149':'#58a6ff'}"></div></div>
    </div></td>
    <td>${m.precision}%</td><td>${m.recall}%</td><td>${m.f1}%</td>
    <td><span class="adm-chip ${m.auc>=0.88?'adm-chip-green':'adm-chip-blue'}">${m.auc}</span></td>
    <td><span class="adm-chip ${i===0?'adm-chip-red':'adm-chip-blue'}">${i===0?'★ Best Accuracy':i===1?'★ Best AUC':'Active'}</span></td>
  </tr>`).join('');

  // Feature importance
  const fi = [...FI_DATA].sort((a,b)=>(b.rf+b.gb)/2-(a.rf+a.gb)/2).slice(0,8);
  safeAdmChart('chart-fi', {
    type:'bar', indexAxis:'y',
    data:{ labels:fi.map(d=>d.feature), datasets:[
      {label:'Random Forest',    data:fi.map(d=>d.rf), backgroundColor:'rgba(248,81,73,.75)',borderRadius:3},
      {label:'Gradient Boosting',data:fi.map(d=>d.gb), backgroundColor:'rgba(163,113,247,.65)',borderRadius:3},
    ]},
    options:{ ...DARK_OPTS, plugins:{...DARK_OPTS.plugins,legend:{labels:{color:'#8b949e',boxWidth:10}}} }
  });

  // Correlation
  const corr = [...CORR_DATA].sort((a,b)=>Math.abs(b.corr)-Math.abs(a.corr)).slice(0,8);
  safeAdmChart('chart-corr', {
    type:'bar', indexAxis:'y',
    data:{ labels:corr.map(d=>d.feature), datasets:[{
      label:'Correlation', data:corr.map(d=>d.corr),
      backgroundColor:corr.map(d=>d.corr>0?'rgba(248,81,73,.75)':'rgba(63,185,80,.7)'),
      borderRadius:4 }] },
    options:{ ...DARK_OPTS, plugins:{...DARK_OPTS.plugins,legend:{display:false}} }
  });

  // CV
  const cvData = [
    {model:'Gradient Boosting',mean:80.58,std:0.64,min:79.54,max:81.25,color:'#3fb950'},
    {model:'SVM',              mean:80.50,std:0.71,min:79.38,max:81.44,color:'#f85149'},
    {model:'Random Forest',   mean:78.88,std:0.71,min:77.75,max:79.92,color:'#d29922'},
    {model:'Logistic Reg.',   mean:78.83,std:0.55,min:77.83,max:79.38,color:'#a371f7'},
  ];
  document.getElementById('adm-cv-grid').innerHTML = cvData.map(c => `
    <div class="adm-cv-item">
      <div class="adm-cv-model">${c.model}</div>
      <div class="adm-cv-bar-wrap">
        <div class="adm-cv-bar" style="width:${c.mean}%;background:${c.color}"></div>
      </div>
      <div class="adm-cv-stats">${c.mean}% <span style="color:var(--adm-text2)">± ${c.std}%</span>
        &nbsp;·&nbsp; Min: ${c.min}% &nbsp;·&nbsp; Max: ${c.max}%</div>
    </div>`).join('');
}

/* ══════════════════════════════════════════════════════════════════════════
   DOCTORS
══════════════════════════════════════════════════════════════════════════ */
async function loadDoctors() {
  try {
    const r = await fetch('/api/admin/doctors');
    const d = await r.json();
    DOCTORS_DATA = d.doctors;
    renderDoctors(DOCTORS_DATA);
  } catch(e) { console.error('Doctors load error:', e); }
}

function renderDoctors(docs) {
  const grid = document.getElementById('adm-doctors-grid');
  if (!docs.length) { grid.innerHTML = '<div class="adm-loading">No doctors found.</div>'; return; }
  grid.innerHTML = docs.map(d => {
    const avatar = d.gender === 'Female' ? '👩‍⚕️' : '👨‍⚕️';
    const specChip = `<span class="adm-chip adm-chip-purple">${d.specialization}</span>`;
    const activeChip = d.is_active
      ? '<span class="adm-chip adm-chip-green">● Active</span>'
      : '<span class="adm-chip adm-chip-amber">○ Suspended</span>';
    const disPct = d.patient_count ? ((d.disease_count/d.patient_count)*100).toFixed(1) : 0;
    return `<div class="adm-doctor-card ${d.is_active?'':'inactive'}" id="dr-card-${d.id}">
      <div class="adm-dr-header">
        <div class="adm-dr-avatar">${avatar}</div>
        <div style="flex:1">
          <div class="adm-dr-name">${d.full_name}</div>
          <span class="adm-dr-code">${d.doctor_code}</span>
          &nbsp;${activeChip}
        </div>
      </div>
      <div class="adm-dr-stats">
        <div class="adm-dr-stat">
          <div class="adm-dr-stat-val" style="color:var(--adm-blue)">${d.patient_count?.toLocaleString()}</div>
          <div class="adm-dr-stat-lbl">Patients</div>
        </div>
        <div class="adm-dr-stat">
          <div class="adm-dr-stat-val" style="color:var(--adm-red)">${disPct}%</div>
          <div class="adm-dr-stat-lbl">Disease Rate</div>
        </div>
        <div class="adm-dr-stat">
          <div class="adm-dr-stat-val" style="color:var(--adm-purple)">${d.predictions}</div>
          <div class="adm-dr-stat-lbl">AI Runs</div>
        </div>
      </div>
      <div class="adm-dr-meta">
        <div class="adm-dr-meta-row"><span>🏥</span>${d.hospital_name}, ${d.hospital_city}</div>
        <div class="adm-dr-meta-row"><span>🎓</span>${d.qualifications}</div>
        <div class="adm-dr-meta-row"><span>⏱</span>${d.experience} years experience</div>
        <div class="adm-dr-meta-row"><span>🕐</span>Last login: ${d.last_login}</div>
        <div class="adm-dr-meta-row"><span>★</span><span class="adm-rating">${d.rating}/5.0</span></div>
      </div>
      <div style="margin-bottom:10px">${specChip}</div>
      <div class="adm-dr-actions">
        <button class="adm-toggle-btn ${d.is_active?'':'suspended'}" id="toggle-btn-${d.id}"
          onclick="toggleDoctor(${d.id}, this)">
          ${d.is_active ? '⏸ Suspend' : '▶ Reactivate'}
        </button>
        <button class="adm-btn-action" onclick="resetPassword(${d.id})">🔑 Reset PW</button>
      </div>
    </div>`;
  }).join('');
}

function filterDoctors() {
  const q     = document.getElementById('dr-search').value.toLowerCase();
  const spec  = document.getElementById('dr-filter-spec').value;
  const hosp  = document.getElementById('dr-filter-hosp').value;
  const filtered = DOCTORS_DATA.filter(d => {
    const matchQ = !q || d.full_name.toLowerCase().includes(q) || d.specialization.toLowerCase().includes(q);
    const matchS = !spec || d.specialization === spec;
    const matchH = !hosp || d.hospital_id === hosp;
    return matchQ && matchS && matchH;
  });
  renderDoctors(filtered);
}

async function toggleDoctor(uid, btn) {
  try {
    const r = await fetch(`/api/admin/toggle-doctor/${uid}`, {method:'POST'});
    const d = await r.json();
    if (d.success) {
      btn.textContent = d.is_active ? '⏸ Suspend' : '▶ Reactivate';
      btn.classList.toggle('suspended', !d.is_active);
      const card = document.getElementById(`dr-card-${uid}`);
      if (card) card.classList.toggle('inactive', !d.is_active);
      // Update local cache
      const doc = DOCTORS_DATA.find(x => x.id === uid);
      if (doc) doc.is_active = d.is_active;
    }
  } catch(e) { alert('Error: ' + e.message); }
}

async function resetPassword(uid) {
  if (!confirm('Reset this doctor\'s password to CardioAI@2024?')) return;
  try {
    const r = await fetch(`/api/admin/reset-password/${uid}`, {method:'POST'});
    const d = await r.json();
    alert(d.success ? '✅ Password reset to CardioAI@2024' : '⚠️ Reset failed');
  } catch(e) { alert('Error: ' + e.message); }
}

/* ══════════════════════════════════════════════════════════════════════════
   HOSPITALS
══════════════════════════════════════════════════════════════════════════ */
function renderHospitals(hospitals) {
  const grid = document.getElementById('adm-hospitals-grid');
  if (!hospitals || !hospitals.length) { grid.innerHTML = '<div class="adm-loading">No data.</div>'; return; }
  grid.innerHTML = hospitals.map(h => {
    const drPct = h.disease_pct;
    const drColor = drPct > 55 ? 'var(--adm-red)' : drPct > 45 ? 'var(--adm-amber)' : 'var(--adm-green)';
    return `<div class="adm-hosp-card">
      <div class="adm-hosp-name">${h.name}</div>
      <div class="adm-hosp-loc">📍 ${h.city} · ${h.region}</div>
      <div class="adm-hosp-stats">
        <div class="adm-hosp-stat">
          <div class="adm-hosp-stat-val">${h.patient_count?.toLocaleString()}</div>
          <div class="adm-hosp-stat-lbl">Patients</div>
        </div>
        <div class="adm-hosp-stat">
          <div class="adm-hosp-stat-val" style="color:${drColor}">${drPct}%</div>
          <div class="adm-hosp-stat-lbl">Disease Rate</div>
        </div>
        <div class="adm-hosp-stat">
          <div class="adm-hosp-stat-val" style="color:var(--adm-purple)">${h.doctor_count}</div>
          <div class="adm-hosp-stat-lbl">Doctors</div>
        </div>
      </div>
      <div class="adm-hosp-stats" style="margin-bottom:0">
        <div class="adm-hosp-stat">
          <div class="adm-hosp-stat-val" style="color:var(--adm-amber)">${h.beds?.toLocaleString()}</div>
          <div class="adm-hosp-stat-lbl">Beds</div>
        </div>
        <div class="adm-hosp-stat">
          <div class="adm-hosp-stat-val" style="color:var(--adm-amber)">★ ${h.rating}</div>
          <div class="adm-hosp-stat-lbl">Rating</div>
        </div>
        <div class="adm-hosp-stat">
          <div class="adm-hosp-stat-val" style="color:${h.icu==='Yes'?'var(--adm-green)':'var(--adm-red)'}">
            ${h.icu==='Yes'?'✓':'✗'}
          </div>
          <div class="adm-hosp-stat-lbl">ICU</div>
        </div>
      </div>
      <div style="margin-top:12px" class="adm-hosp-badges">
        <span class="adm-hosp-badge">${h.type}</span>
        <span class="adm-hosp-badge">${h.accr}</span>
        <span class="adm-hosp-badge">Est. ${h.est}</span>
      </div>
    </div>`;
  }).join('');
}

/* ══════════════════════════════════════════════════════════════════════════
   PREDICTION LOGS
══════════════════════════════════════════════════════════════════════════ */
async function loadPredictions(page) {
  if (page < 1) return;
  predPage = page;
  try {
    const r = await fetch(`/api/admin/all-predictions?page=${page}&per_page=40`);
    const d = await r.json();
    document.getElementById('pred-count').textContent =
      `${d.total.toLocaleString()} total predictions · Page ${d.page} of ${d.pages}`;
    const tbody = document.getElementById('pred-tbody');
    tbody.innerHTML = d.records.map((r, i) => {
      const isD = r.final_prediction === 'Disease';
      const chip = isD
        ? '<span class="adm-chip adm-chip-red">❤️ Disease</span>'
        : '<span class="adm-chip adm-chip-green">✅ Healthy</span>';
      const prob = r.avg_prob != null ? r.avg_prob.toFixed(1) + '%' : '—';
      const scoreColor = r.risk_score >= 55 ? '#f85149' : r.risk_score >= 40 ? '#d29922' : '#3fb950';
      return `<tr>
        <td style="color:var(--adm-text2);font-size:.76rem">${(page-1)*40+i+1}</td>
        <td style="white-space:nowrap;font-size:.78rem">${r.timestamp}</td>
        <td style="color:var(--adm-blue);font-weight:600">${r.doctor}</td>
        <td style="font-size:.78rem;color:var(--adm-text2)">${r.hospital}</td>
        <td>${r.patient_label}</td>
        <td>${r.age ?? '—'}</td>
        <td>${r.sex ?? '—'}</td>
        <td style="font-weight:700;color:${scoreColor}">${r.risk_score ?? '—'}</td>
        <td style="color:${isD?'#f85149':'#3fb950'}">${prob}</td>
        <td>${chip}</td>
      </tr>`;
    }).join('');

    // Pagination
    const pag = document.getElementById('pred-pagination');
    let html = `<button class="adm-page-btn" onclick="loadPredictions(${page-1})" ${page===1?'disabled':''}>← Prev</button>`;
    for (let p = 1; p <= d.pages; p++) {
      if (d.pages > 7 && Math.abs(p-page) > 2 && p !== 1 && p !== d.pages) {
        if (p === 2 || p === d.pages-1) html += '<span style="color:var(--adm-text2);padding:0 4px">…</span>';
        continue;
      }
      html += `<button class="adm-page-btn ${p===page?'active':''}" onclick="loadPredictions(${p})">${p}</button>`;
    }
    html += `<button class="adm-page-btn" onclick="loadPredictions(${page+1})" ${page===d.pages?'disabled':''}>Next →</button>`;
    pag.innerHTML = html;
  } catch(e) { console.error('Predictions error:', e); }
}

/* ══════════════════════════════════════════════════════════════════════════
   UTILS
══════════════════════════════════════════════════════════════════════════ */
function showOverlay(on) {
  document.getElementById('adm-overlay').classList.toggle('active', on);
}
