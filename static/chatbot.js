/* ==========================================================================
   CardioAI — chatbot.js
   Offline Navigational + Medical Glossary Chatbot
   Handles: Tab navigation, feature guidance, medical term explanations
   ========================================================================== */
'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let cbOpen = false;
let cbHistory = [];      // { role: 'bot'|'user', text }
let cbTyping = false;

// ── Medical Glossary Dictionary ────────────────────────────────────────────
const GLOSSARY = {
  'cholesterol': `**Cholesterol** is a fatty substance found in your blood. Your body needs it to build cells, but too much of it can clog your arteries and increase heart disease risk.\n\n• **Normal:** Below 200 mg/dL\n• **Borderline High:** 200–239 mg/dL\n• **High:** 240+ mg/dL`,

  'blood pressure': `**Blood Pressure** is the force your blood exerts on your artery walls as your heart pumps. It has two numbers:\n\n• **Systolic (top):** Pressure when heart beats\n• **Diastolic (bottom):** Pressure between beats\n\n✅ **Ideal:** Below 120/80 mmHg\n⚠️ **High (Hypertension):** 130+ systolic`,

  'trestbps': `**Resting Blood Pressure (trestbps)** is your blood pressure measured when you are completely at rest — no exercise or excitement. It tells how hard your heart is working during calm conditions.\n\nA reading above **130 mmHg** is considered high and increases heart disease risk.`,

  'st depression': `**ST Depression** refers to a downward shift in the "ST segment" of your ECG (heart trace). This segment appears between heartbeats.\n\nIf the ST segment is lower than normal, it can be a sign that your heart muscle isn't getting enough blood — a strong indicator of coronary artery disease.`,

  'st slope': `**ST Slope** describes the angle of the ST segment on your ECG after exercise. There are 3 types:\n\n• **Upsloping:** Generally good — often normal\n• **Flat:** Concerning — may indicate limited blood flow\n• **Downsloping:** Most serious — high risk indicator`,

  'angina': `**Angina** is chest pain or discomfort caused by reduced blood flow to the heart muscle. It's often described as pressure, squeezing, or tightness in the chest.\n\nThere are 4 types in our assessment:\n• **Typical Angina (TA):** Classic chest pain on exertion\n• **Atypical Angina (ATA):** Chest pain without classic symptoms\n• **Non-Anginal Pain (NAP):** Not related to heart\n• **Asymptomatic (ASY):** No chest pain at all`,

  'ecg': `**ECG (Electrocardiogram)** is a test that records the electrical activity of your heart. Electrodes placed on your skin detect tiny electrical impulses that your heart generates with each beat.\n\nOur assessment uses your **Resting ECG** which can show:\n• **Normal** ✅\n• **ST-T Abnormality** ⚠️\n• **Left Ventricular Hypertrophy (LVH)** ⚠️`,

  'thalachh': `**Maximum Heart Rate (thalachh)** is the highest number of times your heart can beat per minute during intense exercise.\n\nA simple estimate is: **220 minus your age**\n\nIf your max heart rate is significantly below this during exercise testing, it may suggest reduced heart function or fitness.`,

  'fbs': `**Fasting Blood Sugar (FBS)** is a measure of your blood glucose level after not eating for at least 8 hours.\n\n• **Normal:** Below 100 mg/dL\n• **Elevated (Diabetic indicator):** Above 120 mg/dL\n\nHigh fasting blood sugar is a strong risk factor for heart disease because excess sugar damages blood vessels over time.`,

  'exang': `**Exercise-Induced Angina (exang)** means you experience chest pain specifically during physical activity or exercise.\n\nWhen arteries are narrowed, rest may supply enough blood to the heart — but exercise increases demand, causing pain. This is a key indicator of coronary artery disease.`,

  'oldpeak': `**Oldpeak (ST Depression)** measures how many millimetres the ST segment drops below baseline on your ECG during exercise.\n\n• **0:** Normal, no depression\n• **1–2mm:** Mild concern\n• **2+ mm:** Significant — high risk indicator`,

  'ca': `**Number of Major Vessels (ca)** refers to how many of your three major coronary arteries showed blockage or narrowing on a fluoroscopy scan.\n\n• **0 vessels:** No visible blockage\n• **1–3 vessels:** Progressive blockage — higher risk\n• **Coronary arteries** supply blood directly to your heart muscle`,

  'thal': `**Thalassemia (thal)** in this context refers to the results of a thallium stress test — a scan that shows how blood flows through your heart:\n\n• **Normal:** Blood flow looks healthy\n• **Fixed Defect:** Area of permanently reduced blood flow (old damage)\n• **Reversible Defect:** Area that shows poor flow during stress but recovers at rest (active ischemia)`,

  'shap': `**SHAP (SHapley Additive exPlanations)** is a method used to explain AI predictions. It shows you exactly *which factors* pushed your risk score up or down.\n\nFor example, if your cholesterol is very high, SHAP will rank it as your #1 risk factor. It makes our AI transparent rather than a "black box".`,

  'risk score': `Your **Risk Score** is a percentage (0–100%) representing the AI's calculated probability that you have or are at risk of heart disease.\n\n• **0–30%:** Low Risk — Great heart health!\n• **30–60%:** Moderate Risk — Some lifestyle attention needed\n• **60–100%:** High Risk — Please consult a cardiologist`,

  'health score': `Your **Heart Health Score** is the inverse of your risk score, making it feel more intuitive:\n\n• **Health Score = 100 − Risk Score**\n\nSo a higher score means better heart health! Think of it like a report card for your heart. 💚`,

  'ensemble': `**Ensemble Model** means our AI doesn't rely on a single algorithm — it uses **6 different machine learning models** simultaneously and combines their votes:\n\n• Random Forest\n• XGBoost\n• Support Vector Machine\n• Logistic Regression\n• Gradient Boosting\n• K-Nearest Neighbors\n\nThis makes predictions far more accurate and reliable than any single model alone.`,

  'bmi': `**BMI (Body Mass Index)** is a simple calculation using your height and weight to estimate body fat:\n\n• **Formula:** Weight (kg) ÷ Height (m)²\n\nRanges:\n• **Below 18.5:** Underweight\n• **18.5–24.9:** Healthy ✅\n• **25–29.9:** Overweight ⚠️\n• **30+:** Obese — higher heart disease risk`,

  'hypertension': `**Hypertension** is the medical term for **high blood pressure** — a condition where blood pushes too forcefully against artery walls over a long period.\n\nIt's often called the "silent killer" because it has no symptoms, yet quietly damages arteries and increases risk of heart attack and stroke.`,

  'ischemia': `**Ischemia** means reduced or restricted blood supply to a tissue or organ. **Myocardial ischemia** specifically refers to the heart muscle not getting enough blood.\n\nThis is dangerous because without oxygen-rich blood, heart muscle cells can be permanently damaged — leading to a heart attack.`,

  'coronary': `**Coronary Arteries** are the blood vessels that wrap around and feed the heart muscle itself with oxygen-rich blood.\n\nCoronary Artery Disease (CAD) happens when these arteries become narrowed by plaques of cholesterol and fat — reducing blood flow to the heart.`,
};

// ── Navigation Intent Map ──────────────────────────────────────────────────
const INTENT_NAVIGATION = [
  {
    patterns: [/profile|my detail|personal info|name.*dob|set up|setup|who am i/i],
    action:   () => { if (typeof showScreen === 'function') showScreen('profile'); },
    reply:    '📋 Taking you to the **Profile** section now! Fill in your personal details here so our AI can analyze your risk accurately.'
  },
  {
    patterns: [/assess|test|quiz|check.*(heart|health)|run.*check|start.*test|scan|analysis|analyse|analyze/i],
    action:   () => { if (typeof showScreen === 'function') showScreen('assess'); },
    reply:    '🩺 Taking you to the **Heart Assessment** form! Fill in your clinical details and click "Analyse My Heart Health" at the bottom.'
  },
  {
    patterns: [/result|score|report|verdict|my.*heart|outcome|prediction/i],
    action:   () => { if (typeof showScreen === 'function') showScreen('results'); },
    reply:    '📊 Taking you to **Results**! Run an assessment first if you haven\'t already — your heart health score and risk factors will appear there.'
  },
  {
    patterns: [/pdf|download|save|export|print/i],
    action:   null,
    reply:    '📄 To download your **PDF Report**, first run the Heart Assessment. Once your results appear, click the **"Save as PDF"** button at the top-right of the results screen.'
  },
  {
    patterns: [/notif|alert|bell|message|warning/i],
    action:   null,
    reply:    '🔔 You can find your **Notifications** by clicking the bell icon (🔔) in the top-right corner of the navigation bar. Important updates about your health assessments appear here.'
  },
  {
    patterns: [/logout|log out|sign out|exit|leave/i],
    action:   null,
    reply:    '👋 To **log out**, hover over your username in the top-right corner and click **Logout**. Your data is saved automatically!'
  },
  {
    patterns: [/history|past.*result|previous|track|progress|trend/i],
    action:   () => { if (typeof showScreen === 'function') showScreen('results'); },
    reply:    '📈 Your **Assessment History** and trend graph appear in the Results screen after you run at least one assessment. It tracks how your heart health score changes over time!'
  },
];

// ── Greeting / Chit-Chat ───────────────────────────────────────────────────
const GREETINGS = [/^hi$|^hello$|hey|^good (morning|evening|afternoon)/i];
const THANKS    = [/thank|thanks|great|awesome|perfect|got it|helpful|nice/i];
const WHAT_CAN  = [/what.*can.*you|help.*with|what.*do.*you.*do|capabilities|feature/i];

// ── Boot Messages ──────────────────────────────────────────────────────────
const WELCOME_MSG = `👋 Hi! I'm **CardioAI Assistant**. I can help you:\n\n• 🧭 **Navigate** the dashboard (just say "take me to assessment")\n• 🔬 **Explain medical terms** (try "what is cholesterol?")\n• ❓ **Answer questions** about how things work here\n\nWhat would you like to know?`;

// ═══════════════════════════════════════════
// DOM CONSTRUCTION
// ═══════════════════════════════════════════
function cbBuildUI() {
  if (document.getElementById('cb-widget')) return;

  const widget = document.createElement('div');
  widget.id = 'cb-widget';
  widget.innerHTML = `
    <!-- Floating Trigger Button -->
    <button class="cb-fab" id="cb-fab" onclick="cbToggle()" title="CardioAI Assistant" aria-label="Open CardioAI Assistant">
      <svg id="cb-fab-icon-open" class="cb-fab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
      </svg>
      <svg id="cb-fab-icon-close" class="cb-fab-icon cb-fab-icon--hidden" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line>
      </svg>
      <span class="cb-fab-badge" id="cb-badge">1</span>
    </button>

    <!-- Chat Window -->
    <div class="cb-window" id="cb-window" role="dialog" aria-label="CardioAI Chat">
      <div class="cb-header">
        <div class="cb-header-info">
          <div class="cb-avatar">♥</div>
          <div>
            <div class="cb-header-name">CardioAI Assistant</div>
            <div class="cb-header-status"><span class="cb-status-dot"></span>Always active</div>
          </div>
        </div>
        <button class="cb-close-btn" onclick="cbToggle()" aria-label="Close chat">✕</button>
      </div>

      <div class="cb-messages" id="cb-messages"></div>

      <div class="cb-typing-indicator" id="cb-typing-row" style="display:none">
        <div class="cb-bubble cb-bubble--bot">
          <span class="cb-typing-dot"></span><span class="cb-typing-dot"></span><span class="cb-typing-dot"></span>
        </div>
      </div>

      <div class="cb-footer">
        <div class="cb-quick-btns" id="cb-quick-btns">
          <button class="cb-quick" onclick="cbQuick('What is cholesterol?')">Cholesterol</button>
          <button class="cb-quick" onclick="cbQuick('Take me to assessment')">Assessment</button>
          <button class="cb-quick" onclick="cbQuick('What is ST depression?')">ST Depression</button>
          <button class="cb-quick" onclick="cbQuick('How do I download my PDF?')">Get PDF</button>
        </div>
        <div class="cb-input-row">
          <input
            id="cb-input"
            class="cb-input"
            type="text"
            placeholder="Ask me anything…"
            autocomplete="off"
            onkeydown="if(event.key==='Enter') cbSend()"
            aria-label="Type your message"
          />
          <button class="cb-send-btn" onclick="cbSend()" aria-label="Send message">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="16" height="16">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </div>
    </div>
  `;
  document.body.appendChild(widget);

  // Show welcome after a small delay on first open
}

// ═══════════════════════════════════════════
// TOGGLE OPEN/CLOSE
// ═══════════════════════════════════════════
function cbToggle() {
  cbOpen = !cbOpen;
  const win   = document.getElementById('cb-window');
  const badge = document.getElementById('cb-badge');
  const iconOpen  = document.getElementById('cb-fab-icon-open');
  const iconClose = document.getElementById('cb-fab-icon-close');

  if (cbOpen) {
    win.classList.add('cb-window--open');
    if (badge) badge.style.display = 'none';
    iconOpen.classList.add('cb-fab-icon--hidden');
    iconClose.classList.remove('cb-fab-icon--hidden');
    if (cbHistory.length === 0) {
      setTimeout(() => cbAddBotMessage(WELCOME_MSG), 300);
    }
    setTimeout(() => { const inp = document.getElementById('cb-input'); if(inp) inp.focus(); }, 400);
  } else {
    win.classList.remove('cb-window--open');
    iconOpen.classList.remove('cb-fab-icon--hidden');
    iconClose.classList.add('cb-fab-icon--hidden');
  }
}

// ═══════════════════════════════════════════
// MESSAGE RENDERING
// ═══════════════════════════════════════════
function cbAddBotMessage(text) {
  const msgs = document.getElementById('cb-messages');
  if (!msgs) return;

  cbHistory.push({ role: 'bot', text });

  const div = document.createElement('div');
  div.className = 'cb-row cb-row--bot';
  div.innerHTML = `
    <div class="cb-avatar cb-avatar--sm">♥</div>
    <div class="cb-bubble cb-bubble--bot">${cbFormatText(text)}</div>
  `;
  msgs.appendChild(div);
  cbScrollBottom();
}

function cbAddUserMessage(text) {
  const msgs = document.getElementById('cb-messages');
  if (!msgs) return;

  cbHistory.push({ role: 'user', text });

  const div = document.createElement('div');
  div.className = 'cb-row cb-row--user';
  div.innerHTML = `<div class="cb-bubble cb-bubble--user">${escapeHtml(text)}</div>`;
  msgs.appendChild(div);
  cbScrollBottom();
}

function cbFormatText(text) {
  // Convert **bold** and newlines to HTML
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function cbScrollBottom() {
  const msgs = document.getElementById('cb-messages');
  if (msgs) msgs.scrollTop = msgs.scrollHeight;
}

// ═══════════════════════════════════════════
// SENDING MESSAGES
// ═══════════════════════════════════════════
function cbSend() {
  const inp = document.getElementById('cb-input');
  if (!inp) return;
  const text = inp.value.trim();
  if (!text || cbTyping) return;
  inp.value = '';
  cbProcess(text);
}

function cbQuick(text) {
  if (cbTyping) return;
  const inp = document.getElementById('cb-input');
  if (inp) inp.value = '';
  cbProcess(text);
}

// ═══════════════════════════════════════════
// INTELLIGENCE ENGINE
// ═══════════════════════════════════════════
function cbProcess(userText) {
  cbAddUserMessage(userText);
  cbTyping = true;
  cbShowTyping(true);

  // Simulate thinking delay (350ms–900ms)
  const delay = 350 + Math.random() * 550;

  setTimeout(() => {
    cbShowTyping(false);
    const reply = cbGetReply(userText);
    cbAddBotMessage(reply.text);
    if (reply.action) {
      setTimeout(reply.action, 400);
    }
    cbTyping = false;
  }, delay);
}

function cbGetReply(input) {
  const lower = input.toLowerCase();

  // 1. Greetings
  if (GREETINGS.some(p => p.test(input))) {
    return { text: `Hello there! 👋 I'm CardioAI Assistant, here to help you understand your heart health and navigate this portal.\n\nYou can ask me things like:\n• "What is cholesterol?"\n• "Take me to the assessment"\n• "What does ST depression mean?"` };
  }

  // 2. Thanks
  if (THANKS.some(p => p.test(input))) {
    const responses = [
      `You're very welcome! 😊 Feel free to ask anything else — I'm always here!`,
      `Happy to help! 💚 Your heart health matters. Ask me anything else.`,
      `Glad that helped! Don't hesitate to ask more questions. I'm here 24/7!`
    ];
    return { text: responses[Math.floor(Math.random() * responses.length)] };
  }

  // 3. What can you do?
  if (WHAT_CAN.some(p => p.test(input))) {
    return { text: `I can help you with:\n\n🧭 **Navigation** — say "take me to assessment" or "show my results"\n🔬 **Medical Terms** — "what is cholesterol?" or "explain ST depression"\n📄 **Features** — "how do I download my PDF?" or "where are my notifications?"\n\nJust type naturally — I'll understand!` };
  }

  // 4. Navigation intents
  for (const intent of INTENT_NAVIGATION) {
    if (intent.patterns.some(p => p.test(input))) {
      return { text: intent.reply, action: intent.action };
    }
  }

  // 5. Medical glossary — detect terms
  const matchedTerms = [];
  for (const [term, def] of Object.entries(GLOSSARY)) {
    if (lower.includes(term)) {
      matchedTerms.push(def);
    }
  }

  // Check for common paraphrases too
  if (!matchedTerms.length) {
    if (/blood sugar|diabetes|glucose/i.test(input)) matchedTerms.push(GLOSSARY['fbs']);
    if (/chest pain|angina type/i.test(input)) matchedTerms.push(GLOSSARY['angina']);
    if (/heart rate|max hr|peak hr/i.test(input)) matchedTerms.push(GLOSSARY['thalachh']);
    if (/exercise.*(pain|angina)/i.test(input)) matchedTerms.push(GLOSSARY['exang']);
    if (/artery|arteries|vessel/i.test(input)) matchedTerms.push(GLOSSARY['coronary']);
    if (/blood flow|oxygen/i.test(input)) matchedTerms.push(GLOSSARY['ischemia']);
    if (/ai|model|six model|machine learn/i.test(input)) matchedTerms.push(GLOSSARY['ensemble']);
    if (/explain.*ai|why.*score|how.*calculated|how.*works/i.test(input)) matchedTerms.push(GLOSSARY['shap']);
    if (/weight|height|obese|overweight/i.test(input)) matchedTerms.push(GLOSSARY['bmi']);
    if (/high bp|bp|pressure/i.test(input)) matchedTerms.push(GLOSSARY['blood pressure']);
  }

  if (matchedTerms.length === 1) {
    return { text: matchedTerms[0] };
  }
  if (matchedTerms.length > 1) {
    return { text: matchedTerms.join('\n\n---\n\n') };
  }

  // 6. Fallback — unknown
  return {
    text: `🤔 I'm not quite sure about that one! I'm specialized in heart health terms and dashboard navigation.\n\nTry asking me:\n• "What is cholesterol?"\n• "What does ST slope mean?"\n• "Take me to the assessment form"`
  };
}

// ═══════════════════════════════════════════
// TYPING INDICATOR
// ═══════════════════════════════════════════
function cbShowTyping(show) {
  const row = document.getElementById('cb-typing-row');
  const msgs = document.getElementById('cb-messages');
  if (!row) return;
  row.style.display = show ? 'flex' : 'none';
  if (show && msgs) msgs.appendChild(row);
  cbScrollBottom();
}

// ═══════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  cbBuildUI();

  // Show badge pulse after 2s to attract attention
  setTimeout(() => {
    const badge = document.getElementById('cb-badge');
    if (badge) badge.style.display = 'flex';
  }, 2000);
});
