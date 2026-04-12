/* ==========================================================================
   CardioAI — patient_app.js
   Handles: profile, assessment form, results rendering, trend chart, PDF
   ========================================================================== */
'use strict';

/* ── State ────────────────────────────────────────────────────────────── */
let lastPatientResult = null;   // full API response — saved for PDF export

/* ══════════════════════════════════════════════════════════════════════════
   INIT — pre-fill profile from server
══════════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  const p = window.INITIAL_PROFILE || {};

  // Pre-fill profile fields
  if (p.full_name) document.getElementById('p-fullname').value = p.full_name;
  if (p.dob)       document.getElementById('p-dob').value       = p.dob;
  if (p.sex)       document.getElementById('p-sex').value        = p.sex;
  if (p.phone)     document.getElementById('p-phone').value      = p.phone;
  if (p.email)     { const el = document.getElementById('p-email'); if(el) el.value = p.email; }
  const notifyEl = document.getElementById('p-notify');
  if (notifyEl) notifyEl.checked = p.notify_email !== false;

  // If profile already complete — skip straight to assessment
  if (p.full_name && p.sex) {
    showScreen('assess');
    syncAssessmentSex(p.sex);
  }

  // Init dark/light theme
  ptInitTheme();
  // Load patient notifications
  ptLoadNotifications();

  // Load history for sidebar
  loadHistorySidebar();
});

/* ══════════════════════════════════════════════════════════════════════════
   SCREEN NAVIGATION
══════════════════════════════════════════════════════════════════════════ */
function showScreen(name) {
  document.querySelectorAll('.pt-screen').forEach(s => s.classList.remove('active'));
  const el = document.getElementById('screen-' + name);
  if (el) { el.classList.add('active'); window.scrollTo({ top: 0, behavior: 'smooth' }); }

  // Update nav steps
  const order = ['profile', 'assess', 'results'];
  document.querySelectorAll('.pt-step').forEach(s => {
    const step = s.dataset.step;
    s.classList.remove('active', 'done');
    const si = order.indexOf(step);
    const ci = order.indexOf(name);
    if (si === ci) s.classList.add('active');
    else if (si < ci) s.classList.add('done');
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   PROFILE — SAVE & CONTINUE
══════════════════════════════════════════════════════════════════════════ */
async function saveProfileAndContinue() {
  const name  = document.getElementById('p-fullname').value.trim();
  const dob   = document.getElementById('p-dob').value.trim();
  const sex   = document.getElementById('p-sex').value;
  const phone = document.getElementById('p-phone').value.trim();
  const msg   = document.getElementById('profile-save-msg');

  if (!name) { showMsg(msg, 'Please enter your full name.', 'err'); return; }
  if (!sex)  { showMsg(msg, 'Please select your sex.',      'err'); return; }

  try {
    const email   = (document.getElementById('p-email')?.value || '').trim();
    const notify  = document.getElementById('p-notify')?.checked !== false;
    const r = await fetch('/api/patient/profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ full_name: name, dob, sex, phone, email, notify_email: notify })
    });
    const d = await r.json();
    if (!d.success) throw new Error(d.error || 'Save failed');
    showMsg(msg, '✅ Profile saved!', 'ok');
    syncAssessmentSex(sex);
    setTimeout(() => showScreen('assess'), 500);
  } catch(e) {
    showMsg(msg, '⚠️ ' + e.message, 'err');
  }
}

function syncAssessmentSex(sex) {
  const el = document.getElementById('a-sex');
  if (el && sex) el.value = sex;
}

function showMsg(el, text, cls) {
  el.textContent = text;
  el.className   = 'pt-save-msg ' + cls;
}

/* ══════════════════════════════════════════════════════════════════════════
   LAB TOGGLE
══════════════════════════════════════════════════════════════════════════ */
function toggleLabSection() {
  const on  = document.getElementById('lab-toggle').checked;
  const sec = document.getElementById('lab-section');
  sec.classList.toggle('open', on);
}

/* ══════════════════════════════════════════════════════════════════════════
   RUN ASSESSMENT
══════════════════════════════════════════════════════════════════════════ */
async function runAssessment() {
  const hasLabs = document.getElementById('lab-toggle').checked;

  const payload = {
    age:      document.getElementById('a-age').value,
    sex:      document.getElementById('a-sex').value,
    cp:       document.getElementById('a-cp').value,
    exang:    document.getElementById('a-exang').value,
    smoking:  document.getElementById('a-smoking').value,
    alcohol:  document.getElementById('a-alcohol').value,
    exercise: document.getElementById('a-exercise').value,
    bmi:      document.getElementById('a-bmi').value,
    has_labs: hasLabs,
  };

  if (hasLabs) {
    Object.assign(payload, {
      trestbps: document.getElementById('a-trestbps').value || 130,
      chol:     document.getElementById('a-chol').value     || 220,
      thalachh: document.getElementById('a-thalachh').value || 150,
      fbs:      document.getElementById('a-fbs').value,
      restecg:  document.getElementById('a-restecg').value,
      oldpeak:  document.getElementById('a-oldpeak').value  || 1.0,
      slope:    document.getElementById('a-slope').value,
      ca:       document.getElementById('a-ca').value,
      thal:     document.getElementById('a-thal').value,
    });
  }

  // Show loading
  const overlay = document.getElementById('pt-loading');
  const bar     = document.getElementById('pt-loading-bar');
  overlay.classList.add('active');
  // reset bar animation
  bar.style.animation = 'none';
  bar.offsetHeight; // reflow
  bar.style.animation = '';

  try {
    const resp = await fetch('/api/patient/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });
    const d = await resp.json();
    if (!d.success) throw new Error(d.error);

    lastPatientResult = d;
    renderResults(d);
    await loadHistorySidebar();
    showScreen('results');

  } catch(e) {
    alert('⚠️ Analysis failed: ' + e.message);
  } finally {
    overlay.classList.remove('active');
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   RENDER RESULTS
══════════════════════════════════════════════════════════════════════════ */
function renderResults(d) {
  // Timestamp
  document.getElementById('results-timestamp').textContent =
    'Analysed on ' + new Date().toLocaleString('en-IN', {
      day:'2-digit', month:'short', year:'numeric',
      hour:'2-digit', minute:'2-digit'
    });

  // ── Gauge ──────────────────────────────────────────────────────────────
  const gaugeColor = d.verdict_color === 'green'  ? '#16a34a'
                   : d.verdict_color === 'amber'  ? '#d97706'
                   : d.verdict_color === 'orange' ? '#ea580c'
                   : '#dc2626';

  Plotly.newPlot('pt-gauge', [{
    type:  'indicator', mode: 'gauge+number',
    value: d.health_score,
    number: { suffix: '/100', font: { size: 38, color: gaugeColor } },
    title:  { text: 'Heart Health Score<br><span style="font-size:11px;color:#94a3b8">Higher is better</span>',
              font: { size: 13, color: '#64748b' } },
    gauge: {
      axis: { range: [0, 100], tickcolor: '#e2e8f0', tickwidth: 1,
              tickfont: { color: '#94a3b8', size: 10 } },
      bar:   { color: gaugeColor, thickness: 0.3 },
      bgcolor: 'transparent', bordercolor: '#e2e8f0',
      steps: [
        { range: [0,  35],  color: 'rgba(220,38,38,.08)'  },
        { range: [35, 55],  color: 'rgba(234,88,12,.07)'  },
        { range: [55, 75],  color: 'rgba(217,119,6,.06)'  },
        { range: [75, 100], color: 'rgba(22,163,74,.08)'  },
      ],
      threshold: { line: { color: gaugeColor, width: 3 }, thickness: 0.85, value: d.health_score }
    }
  }], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    width: 260, margin: { l: 20, r: 20, t: 40, b: 0 }, height: 210,
    font: { family: 'DM Sans, sans-serif' }
  }, { displayModeBar: false, responsive: false });

  const gaugeFooterText = d.health_score >= 75 ? '✅ Looking Good — keep it up!'
    : d.health_score >= 55 ? '⚠️ Some areas to watch'
    : d.health_score >= 35 ? '🔶 Needs attention — see your doctor'
    : '🔴 Please consult a doctor soon';
  document.getElementById('pt-gauge-footer').textContent = gaugeFooterText;

  // ── Verdict card ───────────────────────────────────────────────────────
  const verdictCard = document.getElementById('pt-verdict-card');
  verdictCard.className = 'pt-result-card pt-verdict-card verdict-' + d.verdict_color;

  const icons = { green: '💚', amber: '💛', orange: '🟠', red: '❤️' };
  document.getElementById('pt-verdict-icon').textContent = icons[d.verdict_color] || '—';
  document.getElementById('pt-verdict-label').textContent = d.verdict;
  document.getElementById('pt-verdict-msg').textContent   = d.verdict_msg;

  // ── Vitals ─────────────────────────────────────────────────────────────
  const vitalsSection = document.getElementById('pt-vitals-section');
  if (d.has_labs && d.vitals && d.vitals.length) {
    vitalsSection.style.display = 'block';
    document.getElementById('pt-vitals-row').innerHTML = d.vitals.map(v => {
      const statusLabel = v.status === 'ok' ? '✓ Good' : v.status === 'warn' ? '⚠ Watch' : '✖ Attention';
      return `<div class="pt-vital-card vital-${v.status}">
        <div class="pt-vital-name">${v.label}</div>
        <div class="pt-vital-value">${v.value} <small style="font-size:.6em;font-weight:500;color:inherit">${v.unit}</small></div>
        <div class="pt-vital-ideal">Ideal: ${v.ideal}</div>
        <span class="pt-vital-status">${statusLabel}</span>
      </div>`;
    }).join('');
  } else {
    vitalsSection.style.display = 'none';
  }

  // ── Top Risk Factors ───────────────────────────────────────────────────
  const factorsGrid = document.getElementById('pt-factors-grid');
  if (d.top_factors && d.top_factors.length) {
    factorsGrid.innerHTML = d.top_factors.map(f => {
      const isRisk = f.direction === 'risk';
      return `<div class="pt-factor-card ${isRisk ? 'risk-factor' : 'protect-factor'}">
        <div class="pt-factor-icon">${isRisk ? '⚠️' : '✅'}</div>
        <div>
          <div class="pt-factor-label">${f.label}</div>
          <div class="pt-factor-desc">${f.desc}</div>
        </div>
      </div>`;
    }).join('');
  } else {
    factorsGrid.innerHTML = `<div style="color:#94a3b8;font-size:.9rem;padding:8px">
      No specific risk factors identified. Keep up the healthy habits!
    </div>`;
  }

  // ── Recommendations ────────────────────────────────────────────────────
  const recsGrid = document.getElementById('pt-recs-grid');
  recsGrid.innerHTML = (d.recommendations || []).map(rec => `
    <div class="pt-rec-card">
      <div class="pt-rec-icon">${rec.icon || '•'}</div>
      <div>
        <div class="pt-rec-title">${rec.title}</div>
        <div class="pt-rec-desc">${rec.desc}</div>
      </div>
    </div>`).join('');

  // ── Trend chart ────────────────────────────────────────────────────────
  renderTrendChart();
}

/* ══════════════════════════════════════════════════════════════════════════
   TREND CHART
══════════════════════════════════════════════════════════════════════════ */
async function renderTrendChart() {
  try {
    const r = await fetch('/api/patient/history');
    const d = await r.json();

    const footer = document.getElementById('pt-trend-footer');

    if (!d.trend || d.trend.length === 0) {
      document.getElementById('pt-trend-chart').innerHTML =
        `<div style="display:flex;align-items:center;justify-content:center;height:200px;color:#94a3b8;font-size:.9rem">
          This was your first check! Run more assessments over time to see your progress here.
        </div>`;
      return;
    }

    const xs     = d.trend.map(r => r.ts);
    const ys     = d.trend.map(r => r.hs);
    const cols   = d.trend.map(r => r.pred === 'Healthy' ? '#16a34a' : '#dc2626');
    const borders= d.trend.map(r => r.pred === 'Healthy' ? '#16a34a' : '#dc2626');

    const traceLine = {
      x: xs, y: ys, type: 'scatter', mode: 'lines',
      line: { color: 'rgba(15,118,110,0.2)', width: 2 },
      showlegend: false, hoverinfo: 'skip'
    };
    const traceDots = {
      x: xs, y: ys, type: 'scatter', mode: 'markers+lines',
      name: 'Health Score',
      line: { color: 'rgba(15,118,110,0.35)', width: 1.5, dash: 'dot' },
      marker: { color: cols, size: 11, line: { color: borders, width: 2 } },
      hovertemplate: '<b>%{x}</b><br>Score: <b>%{y}/100</b><extra></extra>'
    };

    const layout = {
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      margin: { t: 10, l: 45, r: 20, b: 45 },
      font: { family: 'DM Sans, sans-serif', color: '#64748b', size: 11 },
      showlegend: false,
      xaxis: { gridcolor: 'rgba(0,0,0,0.05)', color: '#94a3b8', type: 'category',
               nticks: Math.min(xs.length, 10) },
      yaxis: { title: { text: 'Health Score', font: { color: '#94a3b8', size: 10 } },
               range: [0, 105], gridcolor: 'rgba(0,0,0,0.05)', color: '#94a3b8' },
      shapes: [
        { type:'line', x0:0, x1:1, xref:'paper', y0:75, y1:75,
          line:{ color:'rgba(22,163,74,0.25)', width:1.5, dash:'dot' } },
        { type:'line', x0:0, x1:1, xref:'paper', y0:45, y1:45,
          line:{ color:'rgba(220,38,38,0.2)', width:1.5, dash:'dot' } },
      ]
    };

    Plotly.newPlot('pt-trend-chart', [traceLine, traceDots], layout,
      { responsive: true, displayModeBar: false });

    // Footer: improvement message
    if (d.total >= 2 && d.prev !== null && d.latest) {
      const diff = d.latest.health_score - d.prev;
      const sign = diff > 0 ? '↑' : diff < 0 ? '↓' : '→';
      const col  = diff > 0 ? '#16a34a' : diff < 0 ? '#dc2626' : '#94a3b8';
      footer.innerHTML = `${d.total} checks total &nbsp;·&nbsp;
        <span style="color:${col};font-weight:700">${sign} ${Math.abs(diff)} points</span>
        since your previous check`;
    } else {
      footer.textContent = `${d.total} check${d.total !== 1 ? 's' : ''} recorded`;
    }

  } catch(e) {
    document.getElementById('pt-trend-chart').innerHTML =
      `<div style="text-align:center;padding:40px;color:#94a3b8">Could not load history.</div>`;
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   SIDEBAR HISTORY
══════════════════════════════════════════════════════════════════════════ */
async function loadHistorySidebar() {
  try {
    const r = await fetch('/api/patient/history');
    const d = await r.json();
    if (d.latest) {
      const card = document.getElementById('sidebar-history-card');
      const el   = document.getElementById('sidebar-last-check');
      const hs   = d.latest.health_score;
      const col  = hs >= 75 ? '#16a34a' : hs >= 55 ? '#d97706' : '#dc2626';
      card.style.display = 'block';
      el.innerHTML = `
        <div style="font-size:1.6rem;font-weight:800;color:${col}">${hs}/100</div>
        <div style="font-size:.8rem;color:#64748b;margin-top:2px">${d.latest.date}</div>
        ${d.total >= 2 && d.prev !== null
          ? `<div style="font-size:.78rem;margin-top:4px;color:#64748b">
              Previous: <strong>${d.prev}/100</strong>
            </div>`
          : ''}`;
    }
  } catch(e) { /* silent */ }
}

/* ══════════════════════════════════════════════════════════════════════════
   EXPORT PATIENT PDF
══════════════════════════════════════════════════════════════════════════ */
async function exportPatientPDF() {
  if (!lastPatientResult) {
    alert('Please run an assessment first.');
    return;
  }

  const btn = document.querySelectorAll('.pt-btn-export');
  btn.forEach(b => { b.disabled = true; b.textContent = '⏳ Generating…'; });

  const profile = window.INITIAL_PROFILE || {};

  try {
    const payload = {
      ...lastPatientResult,
      patient_name: document.getElementById('p-fullname')?.value || profile.full_name || '',
      dob:          document.getElementById('p-dob')?.value      || profile.dob       || '',
      sex:          document.getElementById('p-sex')?.value      || profile.sex       || '',
    };

    const resp = await fetch('/api/patient/export-pdf', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: 'Server error' }));
      throw new Error(err.error);
    }

    const blob = await resp.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `CardioAI_MyHealthReport_${new Date().toISOString().slice(0,10)}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

  } catch(e) {
    alert('⚠️ PDF export failed: ' + e.message);
  } finally {
    btn.forEach(b => { b.disabled = false; b.innerHTML = '📄 Save as PDF'; });
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   PATIENT DARK / LIGHT THEME
══════════════════════════════════════════════════════════════════════════ */
function ptInitTheme() {
  const saved = localStorage.getItem('cardioai-patient-theme') || 'light';
  if (saved === 'dark') {
    document.body.classList.add('pt-dark-mode');
    const btn = document.getElementById('pt-theme-btn');
    if (btn) btn.textContent = '☀️';
  }
}

function ptToggleTheme() {
  const body = document.body;
  const btn  = document.getElementById('pt-theme-btn');
  if (body.classList.contains('pt-dark-mode')) {
    body.classList.remove('pt-dark-mode');
    if (btn) btn.textContent = '🌙';
    localStorage.setItem('cardioai-patient-theme', 'light');
  } else {
    body.classList.add('pt-dark-mode');
    if (btn) btn.textContent = '☀️';
    localStorage.setItem('cardioai-patient-theme', 'dark');
  }
}

/* ══════════════════════════════════════════════════════════════════════════
   PATIENT NOTIFICATIONS
══════════════════════════════════════════════════════════════════════════ */
let ptNotifOpen = false;

async function ptLoadNotifications() {
  try {
    const r = await fetch('/api/notifications');
    const d = await r.json();
    const badge = document.getElementById('pt-notif-badge');
    const list  = document.getElementById('pt-notif-list');
    if (!badge || !list) return;

    if (d.unread > 0) {
      badge.textContent = d.unread > 9 ? '9+' : d.unread;
      badge.style.display = 'flex';
    } else {
      badge.style.display = 'none';
    }

    if (!d.notifications || d.notifications.length === 0) {
      list.innerHTML = '<div style="padding:20px;text-align:center;font-size:.82rem;color:#94a3b8">No alerts yet</div>';
      return;
    }
    list.innerHTML = d.notifications.map(n => {
      const dot = n.type === 'danger' ? '#dc2626' : n.type === 'warning' ? '#d97706' : '#0f766e';
      return `<div onclick="ptMarkOneRead(${n.id},this)" style="padding:10px 14px;border-bottom:1px solid #f1f5f9;cursor:pointer;${n.is_read?'':'background:#f0fdf9'}">
        <div style="display:flex;gap:8px;align-items:flex-start">
          <div style="width:7px;height:7px;border-radius:50%;background:${dot};flex-shrink:0;margin-top:5px"></div>
          <div>
            <div style="font-size:.8rem;font-weight:600;color:#0f172a">${n.title}</div>
            <div style="font-size:.74rem;color:#64748b;margin-top:1px">${n.message}</div>
            <div style="font-size:.68rem;color:#94a3b8;margin-top:3px">${n.timestamp}</div>
          </div>
        </div>
      </div>`;
    }).join('');
  } catch(e) { /* silent */ }
}

function ptToggleNotif() {
  ptNotifOpen = !ptNotifOpen;
  const panel = document.getElementById('pt-notif-panel');
  if (panel) panel.style.display = ptNotifOpen ? 'block' : 'none';
  if (ptNotifOpen) ptLoadNotifications();
}

async function ptMarkAllRead() {
  await fetch('/api/notifications/mark-read', { method: 'POST' });
  ptLoadNotifications();
}

async function ptMarkOneRead(id, el) {
  await fetch(`/api/notifications/mark-one/${id}`, { method: 'POST' });
  el.style.background = '';
  ptLoadNotifications();
}

// Close when clicking outside
document.addEventListener('click', e => {
  const wrap = document.getElementById('pt-notif-wrap');
  const panel = document.getElementById('pt-notif-panel');
  if (wrap && !wrap.contains(e.target) && panel) {
    panel.style.display = 'none';
    ptNotifOpen = false;
  }
});

// Refresh notifications after assessment
const _ptAssessBtn = document.querySelector('[onclick="runAssessment()"]');
if (_ptAssessBtn) {
  _ptAssessBtn.addEventListener('click', () => setTimeout(ptLoadNotifications, 3000));
}
