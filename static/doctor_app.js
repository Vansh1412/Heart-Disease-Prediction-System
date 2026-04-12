/* ==========================================================================
   CardioAI — doctor_app.js  |  Doctor Dashboard Frontend
   Patient-centric: Home, My Patients, Analytics, Reports
   ========================================================================== */
'use strict';

/* ── State ─────────────────────────────────────────────────────────────── */
let DR_STATS      = null;
let DR_PATIENTS   = [];
let drPtPage      = 1;
let drHistPage    = 1;
const DR_PT_PER_PAGE = 25;

/* Chart defaults for dark theme */
Chart.defaults.font.family = "'DM Sans', sans-serif";
Chart.defaults.color = '#71717a';

/* Global safe chart creator */
function safeDrChart(id, config) {
  try {
    const el = document.getElementById(id);
    if (!el) return null;
    const ex = Chart.getChart(id);
    if (ex) ex.destroy();
    const par = el.parentElement;
    if (par) {
      const w = par.offsetWidth  || par.clientWidth  || 400;
      const h = par.offsetHeight || par.clientHeight || 200;
      el.width  = Math.max(w, 100);
      el.height = Math.max(h, 100);
    }
    if (!config.options) config.options = {};
    config.options.responsive = true;
    config.options.maintainAspectRatio = false;
    return new Chart(el, config);
  } catch(e) {
    console.error('Dr chart error [' + id + ']:', e.message);
    return null;
  }
}

const DR_DARK = {
  responsive:true, maintainAspectRatio:false,
  plugins:{
    legend:{labels:{color:'#a1a1aa',boxWidth:11,font:{size:11}}},
    tooltip:{backgroundColor:'#232328',titleColor:'#f4f4f5',bodyColor:'#a1a1aa',
             borderColor:'rgba(255,255,255,.08)',borderWidth:1,padding:10}
  },
  scales:{
    x:{ticks:{color:'#71717a'},grid:{color:'rgba(255,255,255,.04)'}},
    y:{ticks:{color:'#71717a'},grid:{color:'rgba(255,255,255,.04)'}},
  }
};

/* ══════════════════════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  /* Set date */
  const el = document.getElementById('dr-date');
  if (el) el.textContent = new Date().toLocaleDateString('en-IN',{day:'2-digit',month:'short',year:'numeric'});

  /* Load home data on startup */
  loadDrStats();
});

/* ══════════════════════════════════════════════════════════════════════════
   TAB NAVIGATION
══════════════════════════════════════════════════════════════════════════ */
function drSwitch(tab) {
  document.querySelectorAll('.dr-screen').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.dr-tab').forEach(b => b.classList.remove('active'));
  const screen = document.getElementById('dr-' + tab);
  if (screen) screen.classList.add('active');
  document.querySelectorAll('.dr-tab').forEach(b => {
    if (b.dataset.tab === tab) b.classList.add('active');
  });
  if (tab === 'patients' && DR_PATIENTS.length === 0) loadPatients(1);
  if (tab === 'analytics') initAnalyticsTab();
  if (tab === 'reports')  { loadReportKPIs(); loadHistory(1); loadRepTrend(); }
}

/* ══════════════════════════════════════════════════════════════════════════
   HOME — load stats & render
══════════════════════════════════════════════════════════════════════════ */
async function loadDrStats() {
  try {
    const r = await fetch('/api/doctor/stats');
    if (!r.ok) return;
    const d = await r.json();
    DR_STATS = d;
    renderHomeKPIs(d);
    renderRiskAlerts(d);
    renderHospitalCard(d.hospital);
    renderHomeCharts(d);
  } catch(e) { console.error('Dr stats error:', e); }
  // If analytics tab is already active and not yet rendered, do it now
  const analyticsScreen = document.getElementById('dr-analytics');
  if (analyticsScreen && analyticsScreen.classList.contains('active') && !analyticsInit && DR_STATS) {
    renderAnalytics(DR_STATS);
  }
}

function renderHomeKPIs(d) {
  const set = (id, val) => { const el = document.getElementById(id); if(el) el.textContent = val; };
  set('hkpi-total',     d.total?.toLocaleString() || '—');
  set('hkpi-total-note',`Disease rate: ${d.disease_pct}%`);
  set('hkpi-highrisk',  d.high_risk?.toLocaleString() || '—');
  set('hkpi-highrisk-note', `${d.total ? ((d.high_risk/d.total)*100).toFixed(1) : 0}% of cohort`);
  set('hkpi-healthy',   d.healthy?.toLocaleString() || '—');
  set('hkpi-healthy-note', `${d.total ? ((d.healthy/d.total)*100).toFixed(1) : 0}% of cohort`);
  set('hkpi-avgage',    d.avg_age || '—');
  set('hkpi-preds',     d.predictions_total || 0);
  set('hkpi-preds-note',`${d.predictions_today || 0} today`);
}

function renderRiskAlerts(d) {
  const el = document.getElementById('dr-risk-alerts');
  if (!el) return;
  /* Show top high-risk patients from risk_pie data */
  const highCount = d.risk_pie?.['High Risk'] || 0;
  const modCount  = d.risk_pie?.['Moderate Risk'] || 0;
  el.innerHTML = `
    <div class="dr-alert-item">
      <div><div class="dr-alert-id">High Risk Cohort</div><div class="dr-alert-meta">Patients requiring immediate cardiovascular attention</div></div>
      <span class="dr-alert-badge">${highCount.toLocaleString()} patients</span>
    </div>
    <div class="dr-alert-item" style="background:rgba(245,158,11,.06);border-color:rgba(245,158,11,.15)">
      <div><div class="dr-alert-id" style="color:#f59e0b">Moderate Risk</div><div class="dr-alert-meta">Patients who need monitoring and lifestyle intervention</div></div>
      <span class="dr-alert-badge" style="background:rgba(245,158,11,.12);color:#f59e0b;border-color:rgba(245,158,11,.25)">${modCount.toLocaleString()} patients</span>
    </div>
    <div class="dr-alert-item" style="background:rgba(34,197,94,.06);border-color:rgba(34,197,94,.12)">
      <div><div class="dr-alert-id" style="color:#22c55e">Low Risk / Healthy</div><div class="dr-alert-meta">Continue routine monitoring and preventive care</div></div>
      <span class="dr-alert-badge" style="background:rgba(34,197,94,.1);color:#22c55e;border-color:rgba(34,197,94,.2)">${(d.risk_pie?.['Low Risk'] || 0).toLocaleString()} patients</span>
    </div>
    <div style="margin-top:10px;font-size:.78rem;color:#52525b;padding:10px 12px;background:rgba(255,255,255,.03);border-radius:8px;border:1px solid rgba(255,255,255,.06)">
      💡 <strong style="color:#a1a1aa">Tip:</strong> Use the <strong style="color:#ef4444">Assessment</strong> tab to run a detailed AI risk analysis for any individual patient.
    </div>`;
}

function renderHospitalCard(hosp) {
  const el = document.getElementById('dr-hosp-info');
  if (!el) return;
  if (!hosp || !hosp.name) { el.innerHTML = '<div class="dr-loading">Hospital data not available</div>'; return; }
  const icuColor = hosp.icu === 'Yes' ? '#22c55e' : '#f59e0b';
  el.innerHTML = `
    <div style="font-size:1.05rem;font-weight:700;color:#f4f4f5;margin-bottom:14px">${hosp.name}</div>
    <div class="dr-hosp-info-row"><span class="ico">📍</span><div><div class="val">${hosp.city}, ${hosp.region}</div><div class="lbl">Location</div></div></div>
    <div class="dr-hosp-info-row"><span class="ico">🏷️</span><div><div class="val">${hosp.type}</div><div class="lbl">Hospital Type</div></div></div>
    <div class="dr-hosp-info-row"><span class="ico">🛏️</span><div><div class="val">${hosp.beds?.toLocaleString()}</div><div class="lbl">Bed Capacity</div></div></div>
    <div class="dr-hosp-info-row"><span class="ico">🏥</span><div><div class="val" style="color:${icuColor}">${hosp.icu === 'Yes' ? '✓ Available' : '✗ Not Available'}</div><div class="lbl">ICU</div></div></div>
    <div class="dr-hosp-info-row"><span class="ico">📋</span><div><div class="val">${hosp.accr}</div><div class="lbl">Accreditation</div></div></div>
    <div class="dr-hosp-info-row"><span class="ico">★</span><div><div class="val" style="color:#f59e0b">${hosp.rating}/5.0</div><div class="lbl">Hospital Rating</div></div></div>
    <div class="dr-hosp-info-row"><span class="ico">📅</span><div><div class="val">Est. ${hosp.est}</div><div class="lbl">Established</div></div></div>`;
}

function renderHomeCharts(d) {
  /* Risk pie */
  const rk = d.risk_pie || {};
  const rkKeys = Object.keys(rk);
  const rkCols = {'High Risk':'rgba(239,68,68,.8)','Moderate Risk':'rgba(245,158,11,.8)','Low Risk':'rgba(34,197,94,.8)'};
  safeDrChart('dr-home-risk-chart', {
    type:'doughnut',
    data:{labels:rkKeys,datasets:[{data:rkKeys.map(k=>rk[k]),
      backgroundColor:rkKeys.map(k=>rkCols[k]||'rgba(168,85,247,.8)'),
      borderColor:'transparent',hoverOffset:4}]},
    options:{responsive:true,maintainAspectRatio:false,cutout:'65%',
      plugins:{legend:{position:'bottom',labels:{color:'#71717a',boxWidth:9,font:{size:10}}},
               tooltip:DR_DARK.plugins.tooltip}}
  });
  /* Age bar */
  const ag = d.age_groups || [];
  safeDrChart('dr-home-age-chart', {
    type:'bar',
    data:{labels:ag.map(x=>x.group),datasets:[
      {label:'Disease',data:ag.map(x=>x.disease),backgroundColor:'rgba(239,68,68,.75)',borderRadius:4},
      {label:'Healthy',data:ag.map(x=>x.healthy),backgroundColor:'rgba(34,197,94,.6)',borderRadius:4},
    ]},
    options:{...DR_DARK,plugins:{...DR_DARK.plugins,legend:{labels:{color:'#a1a1aa',boxWidth:9,font:{size:10}}}}}
  });
  /* Gender */
  const ss = d.sex_split || {};
  safeDrChart('dr-home-gender-chart', {
    type:'bar',indexAxis:'y',
    data:{labels:Object.keys(ss),datasets:[{label:'Patients',data:Object.values(ss),
      backgroundColor:['rgba(56,189,248,.75)','rgba(168,85,247,.7)'],borderRadius:4}]},
    options:{...DR_DARK,plugins:{...DR_DARK.plugins,legend:{display:false}}}
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   MY PATIENTS
══════════════════════════════════════════════════════════════════════════ */
async function loadPatients(page = 1) {
  drPtPage = page;
  const search = document.getElementById('pt-search')?.value || '';
  const risk   = document.getElementById('pt-risk')?.value || '';
  const sex    = document.getElementById('pt-sex')?.value || '';
  const ag     = document.getElementById('pt-agegroup')?.value || '';
  const tbody  = document.getElementById('dr-patients-tbody');
  if (tbody) tbody.innerHTML = '<tr><td colspan="13" class="dr-loading">⏳ Loading patients…</td></tr>';

  try {
    let url = `/api/doctor/patients?page=${page}&per_page=${DR_PT_PER_PAGE}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;
    if (risk)   url += `&risk=${encodeURIComponent(risk)}`;
    if (sex)    url += `&sex=${encodeURIComponent(sex)}`;
    const ag2 = document.getElementById('pt-agegroup')?.value || '';
    if (ag2)    url += `&age_group=${encodeURIComponent(ag2)}`;
    const r  = await fetch(url);
    const d  = await r.json();
    DR_PATIENTS = d.records;

    const countEl = document.getElementById('pt-count');
    if (countEl) countEl.textContent = `${d.total.toLocaleString()} patients`;

    renderPatientsTable(d.records);
    renderPatientsPagination(d.page, d.pages);
  } catch(e) {
    if (tbody) tbody.innerHTML = `<tr><td colspan="13" style="text-align:center;color:#f87171;padding:28px">⚠️ ${e.message}</td></tr>`;
  }
}

function renderPatientsTable(records) {
  const tbody = document.getElementById('dr-patients-tbody');
  if (!records.length) {
    tbody.innerHTML = '<tr><td colspan="13" class="dr-loading">No patients found with current filters.</td></tr>';
    return;
  }
  const riskChip = r => {
    if (r === 'High Risk')     return '<span class="dr-chip dr-chip-red">High Risk</span>';
    if (r === 'Moderate Risk') return '<span class="dr-chip dr-chip-amber">Moderate</span>';
    return '<span class="dr-chip dr-chip-green">Low Risk</span>';
  };
  tbody.innerHTML = records.map(p => {
    const target = p.Target === 1 || p.Target === '1'
      ? '<span class="dr-chip dr-chip-red">❤️ Disease</span>'
      : '<span class="dr-chip dr-chip-green">✅ Healthy</span>';
    const bpColor = p.BP_Category === 'Stage 2' || p.BP_Category === 'Hypertensive' ? '#f87171'
                  : p.BP_Category === 'Stage 1' ? '#fbbf24' : '#4ade80';
    return `<tr>
      <td><code style="font-size:.75rem;color:#38bdf8">${p.Patient_ID}</code></td>
      <td style="font-weight:600">${p.Age}</td>
      <td style="font-size:.75rem;color:#71717a">${p.Age_Group}</td>
      <td>${p.Sex_Label}</td>
      <td style="font-size:.78rem">${['Typical Angina','Atypical Angina','Non-Anginal','Asymptomatic'][+p.Chest_Pain_Type] || p.Chest_Pain_Type}</td>
      <td style="color:${bpColor};font-weight:600">${p.Trestbps}</td>
      <td>${p.Cholesterol}</td>
      <td>${p.Max_Heart_Rate}</td>
      <td>${+p.Exercise_Induced_Angina ? '<span style="color:#f87171">Yes</span>' : '<span style="color:#71717a">No</span>'}</td>
      <td>${riskChip(p.Risk_Level)}</td>
      <td>${target}</td>
      <td style="font-size:.75rem;color:#71717a">${p.Visit_Date || '—'}</td>
      <td style="white-space:nowrap">
        <button onclick="openPatientModal('${p.Patient_ID}')"
          style="background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.2);color:#7dd3fc;border-radius:6px;padding:3px 9px;font-size:.7rem;cursor:pointer;font-family:var(--dr-font);margin-right:4px">
          👁 View
        </button>
        <button onclick="prefillAssessmentById('${p.Patient_ID}')"
          style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.25);color:#f87171;border-radius:6px;padding:3px 9px;font-size:.7rem;cursor:pointer;font-family:var(--dr-font)">
          ⚡ Assess
        </button>
      </td>
    </tr>`;
  }).join('');
}

function renderPatientsPagination(page, pages) {
  const el = document.getElementById('pt-pagination');
  if (!el || pages <= 1) { if(el) el.innerHTML=''; return; }
  let html = `<button class="dr-page-btn" onclick="loadPatients(${page-1})" ${page===1?'disabled':''}>← Prev</button>`;
  for (let p = 1; p <= pages; p++) {
    if (pages > 8 && Math.abs(p-page) > 2 && p !== 1 && p !== pages) {
      if (p === 2 || p === pages-1) html += '<span style="color:#52525b;padding:0 4px">…</span>';
      continue;
    }
    html += `<button class="dr-page-btn ${p===page?'active':''}" onclick="loadPatients(${p})">${p}</button>`;
  }
  html += `<button class="dr-page-btn" onclick="loadPatients(${page+1})" ${page===pages?'disabled':''}>Next →</button>`;
  el.innerHTML = html;
}

function filterPatients() { loadPatients(1); }
function clearPatientFilters() {
  ['pt-search','pt-risk','pt-sex','pt-agegroup'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  loadPatients(1);
}

/* Safer: look up patient by ID from cached array */
function prefillAssessmentById(pid) {
  const p = DR_PATIENTS.find(x => x.Patient_ID === pid);
  if (p) prefillAssessment(p);
  else drSwitch('assess');
}

/* Pre-fill assessment form from patient record */
function prefillAssessment(p) {
  drSwitch('assess');
  const sex = p.Sex_Label === 'Male' ? 'Male' : 'Female';
  const setVal = (id, val) => { const el = document.getElementById(id); if(el) el.value = val; };
  setVal('f-age', p.Age);
  setVal('f-sex', sex);
  setVal('f-cp', p.Chest_Pain_Type);
  setVal('f-trestbps', p.Trestbps);
  setVal('f-chol', p.Cholesterol);
  setVal('f-thalachh', p.Max_Heart_Rate);
  setVal('f-exang', p.Exercise_Induced_Angina);
  setVal('f-patient-label', p.Patient_ID);
  if (p.ST_Depression !== undefined)   setVal('f-oldpeak', p.ST_Depression);
  if (p.Slope !== undefined)           setVal('f-slope', p.Slope);
  if (p.Major_Vessels !== undefined)   setVal('f-ca', p.Major_Vessels);
  if (p.Thalassemia !== undefined)     setVal('f-thal', p.Thalassemia);
  if (p.Smoking_Status)                setVal('f-smoking', p.Smoking_Status);
  if (p.Exercise_Level)                setVal('f-exercise', p.Exercise_Level);
  if (p.BMI_Category)                  setVal('f-bmi', p.BMI_Category);
}

/* ══════════════════════════════════════════════════════════════════════════
   MY ANALYTICS
══════════════════════════════════════════════════════════════════════════ */
let analyticsInit = false;
async function initAnalyticsTab() {
  if (!DR_STATS) {
    // Data not ready — fetch it then render
    await loadDrStats();
  }
  if (!analyticsInit && DR_STATS) {
    await renderAnalytics(DR_STATS);
  }
}
async function renderAnalytics(d) {
  if (analyticsInit || !d) return;
  analyticsInit = true;

  // Wait for tab to be fully painted before Chart.js measures canvas
  // Force two animation frames + paint settle time
  await new Promise(r => requestAnimationFrame(r));
  await new Promise(r => requestAnimationFrame(r));
  await new Promise(r => setTimeout(r, 200));

  /* ── KPI strip ── */
  const kpiEl = document.getElementById('da-kpis');
  if (kpiEl) {
    const highRisk = d.risk_pie?.['High Risk'] || 0;
    const modRisk  = d.risk_pie?.['Moderate Risk'] || 0;
    const lowRisk  = d.risk_pie?.['Low Risk'] || 0;
    kpiEl.innerHTML = `
      <div class="dr-kpi-card dr-kpi-blue"  style="margin-bottom:0"><div class="dr-kpi-ico">👥</div><div class="dr-kpi-num" style="color:#38bdf8">${(d.total||0).toLocaleString()}</div><div class="dr-kpi-lbl">Total Patients</div></div>
      <div class="dr-kpi-card dr-kpi-red"   style="margin-bottom:0"><div class="dr-kpi-ico">❤️</div><div class="dr-kpi-num">${(d.disease||0).toLocaleString()}</div><div class="dr-kpi-lbl">Disease Cases</div><div class="dr-kpi-note">${d.disease_pct||0}%</div></div>
      <div class="dr-kpi-card dr-kpi-green" style="margin-bottom:0"><div class="dr-kpi-ico">✅</div><div class="dr-kpi-num">${(d.healthy||0).toLocaleString()}</div><div class="dr-kpi-lbl">Healthy Cases</div></div>
      <div class="dr-kpi-card dr-kpi-amber" style="margin-bottom:0"><div class="dr-kpi-ico">📊</div><div class="dr-kpi-num">${d.avg_age||0}</div><div class="dr-kpi-lbl">Avg Age (yrs)</div></div>`;
  }

  /* ── Helper: safe chart creation with logging ── */
  function safeChart(id, config) {
    try {
      const el = document.getElementById(id);
      if (!el) { console.warn('Canvas not found:', id); return null; }
      // Destroy existing chart if any
      const existing = Chart.getChart(id);
      if (existing) existing.destroy();
      // Force explicit dimensions so Chart.js doesn't rely on getBoundingClientRect
      const parent = el.parentElement;
      const w = parent ? parent.offsetWidth : 400;
      const h = parent ? parent.offsetHeight : 210;
      if (w > 0) { el.width = w; el.height = h > 0 ? h : 210; }
      // Ensure responsive is enabled in config
      if (!config.options) config.options = {};
      config.options.responsive = true;
      config.options.maintainAspectRatio = false;
      return new Chart(el, config);
    } catch(e) { console.error('Chart error for', id, ':', e.message); return null; }
  }

  const ag = d.age_groups || [];
  const agRates = ag.map(x => x.total ? +((x.disease/x.total)*100).toFixed(1) : 0);
  safeChart('da-age-chart', {
    type:'bar',
    data:{labels:ag.map(x=>x.group), datasets:[{
      label:'Disease Rate (%)', data:agRates,
      backgroundColor:agRates.map(r=>r>55?'rgba(239,68,68,.82)':r>40?'rgba(245,158,11,.82)':'rgba(34,197,94,.75)'),
      borderRadius:5}]},
    options:{...DR_DARK, plugins:{...DR_DARK.plugins, legend:{display:false}}}
  });

  const rk = d.risk_pie || {};
  const rkKeys = Object.keys(rk);
  safeChart('da-risk-chart', {
    type:'doughnut',
    data:{labels:rkKeys, datasets:[{data:rkKeys.map(k=>rk[k]),
      backgroundColor:rkKeys.map(k=>({'High Risk':'rgba(239,68,68,.82)','Moderate Risk':'rgba(245,158,11,.82)','Low Risk':'rgba(34,197,94,.8)'}[k]||'rgba(168,85,247,.8)')),
      borderColor:'transparent', hoverOffset:4}]},
    options:{responsive:true, maintainAspectRatio:false, cutout:'62%',
      plugins:{legend:{position:'bottom',labels:{color:'#71717a',boxWidth:9,font:{size:10}}},
               tooltip:DR_DARK.plugins.tooltip}}
  });

  const bp = d.bp_dist || {};
  const bpKeys = Object.keys(bp);
  if (bpKeys.length) {
    const BP_COLORS = ['rgba(34,197,94,.75)','rgba(245,158,11,.78)','rgba(239,68,68,.78)','rgba(168,85,247,.75)','rgba(239,68,68,.9)'];
    safeChart('da-bp-chart', {
      type:'bar',
      data:{labels:bpKeys, datasets:[{label:'Patients', data:bpKeys.map(k=>bp[k]),
        backgroundColor:bpKeys.map((_,i)=>BP_COLORS[i%BP_COLORS.length]),borderRadius:4}]},
      options:{...DR_DARK, plugins:{...DR_DARK.plugins, legend:{display:false}}}
    });
  }

  const chol = d.chol_dist || {};
  const cholKeys = Object.keys(chol);
  if (cholKeys.length) {
    const CHOL_COLORS = ['rgba(34,197,94,.75)','rgba(245,158,11,.75)','rgba(239,68,68,.8)','rgba(168,85,247,.7)'];
    safeChart('da-chol-chart', {
      type:'bar',
      data:{labels:cholKeys, datasets:[{label:'Patients', data:cholKeys.map(k=>chol[k]),
        backgroundColor:cholKeys.map((_,i)=>CHOL_COLORS[i%CHOL_COLORS.length]),borderRadius:4}]},
      options:{...DR_DARK, plugins:{...DR_DARK.plugins, legend:{display:false}}}
    });
  }

  const ss = d.sex_split || {};
  const ssKeys = Object.keys(ss);
  safeChart('da-gender-chart', {
    type:'bar', indexAxis:'y',
    data:{labels:ssKeys, datasets:[{label:'Patients', data:ssKeys.map(k=>ss[k]),
      backgroundColor:['rgba(56,189,248,.75)','rgba(168,85,247,.7)'],borderRadius:5}]},
    options:{...DR_DARK, plugins:{...DR_DARK.plugins, legend:{display:false}}}
  });

  /* ── Summary table ── */
  const highRisk2 = d.risk_pie?.['High Risk'] || 0;
  const modRisk2  = d.risk_pie?.['Moderate Risk'] || 0;
  const lowRisk2  = d.risk_pie?.['Low Risk'] || 0;
  const total     = d.total || 1;
  const sumEl = document.getElementById('da-summary-table');
  if (sumEl) sumEl.innerHTML = `
    <tr><td>Average Patient Age</td><td>${d.avg_age} yrs</td><td>${d.avg_age} yrs</td><td style="color:#71717a">Same cohort baseline</td></tr>
    <tr><td>Disease Rate</td><td style="color:#f87171;font-weight:600">${d.disease_pct}%</td><td style="color:#4ade80;font-weight:600">${(100-d.disease_pct).toFixed(1)}%</td>
      <td><span style="color:${d.disease_pct>53?'#f87171':'#4ade80'}">${d.disease_pct>53?'↑ Above national avg':'✓ Within normal range'}</span></td></tr>
    <tr><td>High Risk Patients</td><td style="color:#f87171;font-weight:600">${highRisk2.toLocaleString()} (${((highRisk2/total)*100).toFixed(1)}%)</td>
      <td style="color:#4ade80;font-weight:600">${lowRisk2.toLocaleString()} (${((lowRisk2/total)*100).toFixed(1)}%)</td><td>National avg ~48% high risk</td></tr>
    <tr><td>Moderate Risk</td><td colspan="2">${modRisk2.toLocaleString()} patients — (${((modRisk2/total)*100).toFixed(1)}%)</td><td style="color:#fbbf24">Monitor closely</td></tr>`;
}


/* ══════════════════════════════════════════════════════════════════════════
   REPORTS
══════════════════════════════════════════════════════════════════════════ */
async function loadReportKPIs() {
  try {
    const r = await fetch('/api/prediction-stats');
    const d = await r.json();
    const set = (id, val) => { const el = document.getElementById(id); if(el) el.textContent = val; };
    set('rep-total',   d.total);
    set('rep-today',   `${d.today} today`);
    set('rep-disease', d.disease);
    set('rep-healthy', d.healthy);
  } catch(e) {}
}

async function loadRepTrend() {
  try {
    const r = await fetch('/api/history?page=1&per_page=200');
    const d = await r.json();
    const trend = d.trend || [];
    if (!trend.length) {
      document.getElementById('rep-trend-chart').innerHTML =
        '<div style="display:flex;align-items:center;justify-content:center;height:200px;color:#52525b;font-size:.9rem">Run your first assessment to see trends here</div>';
      return;
    }
    const avgRisk = trend.length ? Math.round(trend.reduce((s,r)=>s+(r.risk||0),0)/trend.length) : 0;
    document.getElementById('rep-avg-risk').textContent = avgRisk || '—';

    const traceD = {x:trend.map(r=>r.ts?.slice(0,10)),y:trend.map(r=>r.risk),
      type:'scatter',mode:'lines+markers',name:'Risk Score',
      line:{color:'rgba(239,68,68,.5)',width:1.5,dash:'dot'},
      marker:{color:trend.map(r=>r.pred==='Disease'?'#ef4444':'#22c55e'),size:8,
              line:{color:trend.map(r=>r.pred==='Disease'?'#ef4444':'#22c55e'),width:2}}};
    Plotly.newPlot('rep-trend-chart', [traceD], {
      paper_bgcolor:'transparent',plot_bgcolor:'transparent',
      margin:{t:10,l:45,r:20,b:45},
      font:{color:'#71717a',family:'DM Sans',size:11},
      xaxis:{gridcolor:'rgba(255,255,255,.04)',color:'#52525b',type:'category',nticks:Math.min(trend.length,12)},
      yaxis:{title:{text:'Risk Score (0–100)',font:{color:'#52525b',size:10}},range:[0,105],
             gridcolor:'rgba(255,255,255,.04)',color:'#52525b'},
      shapes:[{type:'line',x0:0,x1:1,xref:'paper',y0:55,y1:55,line:{color:'rgba(239,68,68,.2)',width:1.5,dash:'dot'}}]
    },{responsive:true,displayModeBar:false});
  } catch(e) {}
}

/* ── Re-use history functions from app.js ── */
/* These are already defined in the imported app.js — just call them */
function applyHistoryFilters() { window.historyLoaded = false; loadHistory(1); }
function clearHistoryFilters() {
  ['hist-start','hist-end','hist-verdict'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.id === 'hist-verdict' ? el.value = '' : el.value = '';
  });
  window.historyLoaded = false; loadHistory(1);
}

/* ══════════════════════════════════════════════════════════════════════════
   DARK / LIGHT MODE TOGGLE
══════════════════════════════════════════════════════════════════════════ */
(function initTheme() {
  const saved = localStorage.getItem('cardioai-theme') || 'dark';
  if (saved === 'light') {
    document.body.classList.add('light-mode');
    const btn = document.getElementById('theme-btn');
    if (btn) btn.textContent = '☀️';
  }
})();

function toggleTheme() {
  const body = document.body;
  const btn  = document.getElementById('theme-btn');
  if (body.classList.contains('light-mode')) {
    body.classList.remove('light-mode');
    if (btn) btn.textContent = '🌙';
    localStorage.setItem('cardioai-theme', 'dark');
  } else {
    body.classList.add('light-mode');
    if (btn) btn.textContent = '☀️';
    localStorage.setItem('cardioai-theme', 'light');
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   NOTIFICATION SYSTEM
══════════════════════════════════════════════════════════════════════════ */
let notifOpen = false;

async function loadNotifications() {
  try {
    const r = await fetch('/api/notifications');
    const d = await r.json();
    renderNotifications(d.notifications, d.unread);
  } catch(e) { /* silent */ }
}

function renderNotifications(notifs, unread) {
  const badge = document.getElementById('dr-notif-badge');
  const list  = document.getElementById('dr-notif-list');
  if (!badge || !list) return;

  // Badge
  if (unread > 0) {
    badge.textContent = unread > 9 ? '9+' : unread;
    badge.classList.add('visible');
  } else {
    badge.classList.remove('visible');
  }

  // List
  if (!notifs || notifs.length === 0) {
    list.innerHTML = '<div class="dr-notif-empty">✅ All caught up — no notifications</div>';
    return;
  }

  list.innerHTML = notifs.map(n => {
    const dotClass = n.type === 'danger' ? 'dot-danger' : n.type === 'warning' ? 'dot-warning' : 'dot-info';
    return `<div class="dr-notif-item ${n.is_read ? '' : 'unread'}" onclick="markOneRead(${n.id}, this)">
      <div class="dr-notif-dot ${dotClass}"></div>
      <div class="dr-notif-body">
        <div class="dr-notif-item-title">${n.title}</div>
        <div class="dr-notif-item-msg">${n.message}</div>
        <div class="dr-notif-item-time">${n.timestamp}</div>
      </div>
      <button class="dr-notif-del" onclick="deleteNotif(event,${n.id})">✕</button>
    </div>`;
  }).join('');
}

function toggleNotifPanel() {
  const panel = document.getElementById('dr-notif-panel');
  if (!panel) return;
  notifOpen = !notifOpen;
  panel.classList.toggle('open', notifOpen);
  if (notifOpen) loadNotifications();
}

// Close panel when clicking outside
document.addEventListener('click', e => {
  const wrap = document.getElementById('dr-notif-wrap');
  if (wrap && !wrap.contains(e.target)) {
    const panel = document.getElementById('dr-notif-panel');
    if (panel) panel.classList.remove('open');
    notifOpen = false;
  }
});

async function markAllRead() {
  await fetch('/api/notifications/mark-read', { method: 'POST' });
  loadNotifications();
}

async function markOneRead(id, el) {
  if (el.classList.contains('unread')) {
    await fetch(`/api/notifications/mark-one/${id}`, { method: 'POST' });
    el.classList.remove('unread');
    loadNotifications();
  }
}

async function deleteNotif(e, id) {
  e.stopPropagation();
  await fetch(`/api/notifications/delete/${id}`, { method: 'DELETE' });
  loadNotifications();
}

// Poll notifications every 60 seconds
document.addEventListener('DOMContentLoaded', () => {
  loadNotifications();
  setInterval(loadNotifications, 60000);
});

// Also refresh after a prediction
const _origPredSubmit = document.getElementById('predict-form');
if (_origPredSubmit) {
  _origPredSubmit.addEventListener('cardio:predicted', () => setTimeout(loadNotifications, 1500));
}

/* ══════════════════════════════════════════════════════════════════════════
   PATIENT FULL PROFILE MODAL
══════════════════════════════════════════════════════════════════════════ */
let currentModalPatient = null;

function openPatientModal(pid) {
  const p = DR_PATIENTS.find(x => x.Patient_ID === pid);
  if (!p) return;
  currentModalPatient = p;

  // Header
  const isDisease = p.Target === 1 || p.Target === '1';
  document.getElementById('modal-patient-title').textContent = `Patient Profile — ${p.Patient_ID}`;
  document.getElementById('modal-patient-id').textContent    = p.Patient_ID;
  document.getElementById('modal-patient-sub').textContent   = `${p.Sex_Label || '—'} · Age ${p.Age} · ${p.Age_Group}`;
  document.getElementById('modal-avatar').textContent        = p.Sex_Label === 'Female' ? '👩' : '👨';

  // Risk chip
  const riskColors = {'High Risk':'#f87171','Moderate Risk':'#fbbf24','Low Risk':'#4ade80'};
  const col = riskColors[p.Risk_Level] || '#a1a1aa';
  document.getElementById('modal-risk-chip').innerHTML =
    `<span style="background:${col}20;color:${col};border:1px solid ${col}40;padding:4px 12px;border-radius:99px;font-size:.78rem;font-weight:700">${p.Risk_Level || '—'}</span>`;

  // Risk bar
  const riskPct = {'Low Risk': 20, 'Moderate Risk': 55, 'High Risk': 88}[p.Risk_Level] || 50;
  const barColor = riskPct >= 70 ? '#ef4444' : riskPct >= 45 ? '#f59e0b' : '#22c55e';
  document.getElementById('modal-risk-fill').style.width      = riskPct + '%';
  document.getElementById('modal-risk-fill').style.background = barColor;
  document.getElementById('modal-risk-band').textContent      = p.Risk_Level || '—';
  document.getElementById('modal-risk-band').style.color      = barColor;

  // Clinical fields grid
  const clinFields = [
    ['Chest Pain', ['Typical Angina','Atypical Angina','Non-Anginal Pain','Asymptomatic'][+p.Chest_Pain_Type] || p.Chest_Pain_Type],
    ['Resting BP', `${p.Trestbps} mmHg`], ['Cholesterol', `${p.Cholesterol} mg/dL`],
    ['Max Heart Rate', `${p.Max_Heart_Rate} bpm`],
    ['Exercise Angina', +p.Exercise_Induced_Angina ? 'Yes ⚠️' : 'No ✓'],
    ['BP Category', p.BP_Category || '—'],
    ['Cholesterol Cat.', p.Cholesterol_Category || '—'],
    ['Visit Date', p.Visit_Date || '—'],
    ['AI Verdict', isDisease ? '❤️ Disease Detected' : '✅ Healthy'],
  ];
  document.getElementById('modal-fields-grid').innerHTML = clinFields.map(([lbl, val]) => `
    <div class="dr-profile-field">
      <div class="dr-profile-field-lbl">${lbl}</div>
      <div class="dr-profile-field-val">${val}</div>
    </div>`).join('');

  // Lifestyle grid
  const lifeFields = [
    ['Smoking', p.Smoking_Status || '—'],
    ['Exercise Level', p.Exercise_Level || '—'],
    ['BMI Category', p.BMI_Category || '—'],
  ];
  document.getElementById('modal-lifestyle-grid').innerHTML =
    `<div class="dr-card-title" style="grid-column:span 2;font-size:.72rem;padding-bottom:6px;margin-bottom:4px;border-bottom:1px solid var(--dr-border)">🌿 Lifestyle Factors</div>` +
    lifeFields.map(([lbl, val]) => `
      <div class="dr-profile-field">
        <div class="dr-profile-field-lbl">${lbl}</div>
        <div class="dr-profile-field-val">${val}</div>
      </div>`).join('');

  document.getElementById('patient-modal-overlay').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closePatientModal(e) {
  if (e && e.target !== document.getElementById('patient-modal-overlay')) return;
  document.getElementById('patient-modal-overlay').classList.remove('open');
  document.body.style.overflow = '';
  currentModalPatient = null;
}

function assessFromModal() {
  if (currentModalPatient) {
    prefillAssessmentById(currentModalPatient.Patient_ID);
    closePatientModal();
  }
}

// Keyboard close
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    document.getElementById('patient-modal-overlay')?.classList.remove('open');
    document.body.style.overflow = '';
    const notifPanel = document.getElementById('dr-notif-panel');
    if (notifPanel) notifPanel.classList.remove('open');
  }
});
