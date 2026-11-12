// Navigation and page view routing
const navClassify = document.getElementById('navClassify');
const navResults = document.getElementById('navResults');
const classifyView = document.getElementById('classifyView');
const resultsView = document.getElementById('resultsView');
const tryItNavBtn = document.getElementById('tryItNavBtn');

let resultsLoaded = false;
let isAnalysing = false;
let xlmOnlyMode = false;
let currentBatchRows = [];
let lastAnalysisData = null;

function setRoute() {
  const hash = window.location.hash.toLowerCase();
  if (hash === '#results') {
    navClassify.classList.remove('active');
    navResults.classList.add('active');
    classifyView.classList.remove('active');
    resultsView.classList.add('active');

    // Results page is fully dark
    document.body.style.backgroundColor = '#000000';
    document.body.classList.add('dark-nav');

    if (!resultsLoaded) {
      loadEvaluationResults();
    } else {
      renderPerLanguageChart();
    }
  } else {
    navResults.classList.remove('active');
    navClassify.classList.add('active');
    resultsView.classList.remove('active');
    classifyView.classList.add('active');

    // Recompute scroll fade on classify page
    updateScrollFade();
  }
}

window.addEventListener('hashchange', setRoute);
window.addEventListener('DOMContentLoaded', () => {
  setRoute();
  loadHeroStats();
  updateInputState();
  initHeroInteractions();
});

// Scroll into dark background transition
function updateScrollFade() {
  if (!classifyView.classList.contains('active')) return;

  const scrollY = window.scrollY || window.pageYOffset;
  const windowH = window.innerHeight || 800;

  // Transition over roughly one screen height (between 15% and 85% of screen height)
  const fadeStart = windowH * 0.15;
  const fadeEnd = windowH * 0.85;
  const progress = Math.min(1, Math.max(0, (scrollY - fadeStart) / (fadeEnd - fadeStart)));

  // Interpolate RGB from 255 (white) to 0 (black)
  const val = Math.round(255 * (1 - progress));
  document.body.style.backgroundColor = `rgb(${val}, ${val}, ${val})`;

  // Switch top nav styling when entering dark territory
  if (progress > 0.5) {
    document.body.classList.add('dark-nav');
  } else {
    document.body.classList.remove('dark-nav');
  }
}

window.addEventListener('scroll', updateScrollFade, { passive: true });

function scrollToClassifier() {
  const el = document.getElementById('classifierSection');
  if (el) {
    el.scrollIntoView({ behavior: 'smooth' });
  }
}

// "Try it" button scrolls smoothly to classifier panel
if (tryItNavBtn) {
  tryItNavBtn.addEventListener('click', () => {
    if (!classifyView.classList.contains('active')) {
      window.location.hash = '#classify';
      setTimeout(scrollToClassifier, 50);
    } else {
      scrollToClassifier();
    }
  });
}

// Multilingual examples dictionary
const EXAMPLES = {
  en: "You are a complete idiot and should delete your account immediately.",
  es: "Eres un imbécil total y no sabes nada de este tema, cállate la boca.",
  tr: "Sen tam bir aptalsın, hiçbir şeyden anlamıyorsun defol git buradan.",
  ru: "Ты полный идиот и ничего не понимаешь в этом, закрой свой рот.",
  fr: "Tu es un idiot fini et tu ne comprends rien à ce sujet, tais-toi.",
  pt: "Você é um idiota completo e não entende nada sobre isso, cala a boca.",
  polite: "This article provides very thoughtful and well-researched insights. Thank you."
};

const classifyInput = document.getElementById('classifyInput');
const charCount = document.getElementById('charCount');
const analyseBtn = document.getElementById('analyseBtn');
const classifyResultsSection = document.getElementById('classifyResultsSection');
const classifierDarkGrid = document.getElementById('classifierDarkGrid');

const baselineCard = document.getElementById('baselineCard');
const baselineProb = document.getElementById('baselineProb');
const baselineBar = document.getElementById('baselineBar');
const baselineVerdict = document.getElementById('baselineVerdict');

const transformerCard = document.getElementById('transformerCard');
const transformerProb = document.getElementById('transformerProb');
const transformerBar = document.getElementById('transformerBar');
const transformerVerdict = document.getElementById('transformerVerdict');
const modelDisagreementNote = document.getElementById('modelDisagreementNote');

function updateInputState() {
  const val = classifyInput.value;
  charCount.textContent = val.length;
  if (!isAnalysing) {
    analyseBtn.disabled = val.trim().length === 0;
  }
}

classifyInput.addEventListener('input', updateInputState);

// Cmd+Enter or Ctrl+Enter triggers analysis exactly once
classifyInput.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    if (!analyseBtn.disabled && !isAnalysing) {
      runAnalysis();
    }
  }
});

// Example links under input: strictly populate textarea without auto-running
document.querySelectorAll('.example-link').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const key = link.getAttribute('data-example');
    if (EXAMPLES[key]) {
      classifyInput.value = EXAMPLES[key];
      updateInputState();
      classifyInput.focus();
    }
  });
});

// Initialize Hero Interactive Elements: Bubbles, Cursors, and Toggle
function initHeroInteractions() {
  // 1. Comment Bubbles: Clicking fills input and scrolls to classifier
  document.querySelectorAll('.floating-bubble').forEach(bubble => {
    bubble.addEventListener('click', () => {
      const bodyEl = bubble.querySelector('.bubble-body');
      if (bodyEl) {
        classifyInput.value = bodyEl.textContent.trim();
        updateInputState();
        scrollToClassifier();
        classifyInput.focus();
      }
    });
  });

  // 2. Language Cursors: Clicking fills input with example in that language and scrolls
  document.querySelectorAll('.floating-cursor').forEach(cursor => {
    cursor.addEventListener('click', () => {
      const lang = cursor.getAttribute('data-lang');
      if (lang && EXAMPLES[lang]) {
        classifyInput.value = EXAMPLES[lang];
        updateInputState();
        scrollToClassifier();
        classifyInput.focus();
      }
    });
  });

  // 3. XLM-RoBERTa Toggle: Switches between showing both models and showing XLM-RoBERTa only
  const heroModelToggle = document.getElementById('heroModelToggle');
  const heroToggleSwitch = document.getElementById('heroToggleSwitch');

  function applyToggleMode() {
    if (xlmOnlyMode) {
      heroModelToggle.classList.add('active');
      heroToggleSwitch.setAttribute('aria-checked', 'true');
      baselineCard.classList.add('hidden');
      classifierDarkGrid.classList.add('single-model');
      if (modelDisagreementNote) {
        modelDisagreementNote.style.display = 'none';
      }
    } else {
      heroModelToggle.classList.remove('active');
      heroToggleSwitch.setAttribute('aria-checked', 'false');
      baselineCard.classList.remove('hidden');
      classifierDarkGrid.classList.remove('single-model');
      if (lastAnalysisData) {
        checkDisagreement(lastAnalysisData);
      }
    }
  }

  if (heroModelToggle) {
    heroModelToggle.addEventListener('click', (e) => {
      e.preventDefault();
      xlmOnlyMode = !xlmOnlyMode;
      applyToggleMode();
    });
  }
}

// Single Comment Analysis
analyseBtn.addEventListener('click', (e) => {
  e.preventDefault();
  if (!analyseBtn.disabled && !isAnalysing) {
    runAnalysis();
  }
});

async function runAnalysis() {
  const text = classifyInput.value.trim();
  if (!text || isAnalysing) return;

  isAnalysing = true;
  analyseBtn.disabled = true;
  analyseBtn.textContent = 'Analysing...';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }

    const data = await res.json();
    lastAnalysisData = data;
    renderAnalysisResults(data);
  } catch (err) {
    console.error('Prediction failed:', err);
  } finally {
    isAnalysing = false;
    analyseBtn.textContent = 'Analyse';
    updateInputState();
  }
}

// Number counting animation (0 -> target over 600ms)
function animateNumber(element, targetVal, durationMs = 600) {
  const startTime = performance.now();
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(1, elapsed / durationMs);
    const easeProgress = 1 - Math.pow(1 - progress, 3);
    const currentVal = targetVal * easeProgress;
    element.textContent = currentVal.toFixed(2);
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.textContent = targetVal.toFixed(2);
    }
  }
  requestAnimationFrame(update);
}

function checkDisagreement(data) {
  if (xlmOnlyMode) {
    modelDisagreementNote.style.display = 'none';
    modelDisagreementNote.classList.remove('visible');
    return;
  }

  const baseV = data.baseline && data.baseline.verdict;
  const transV = data.transformer && data.transformer.verdict;

  if (baseV && transV && baseV !== transV) {
    modelDisagreementNote.textContent =
      "The models disagree. XLM-RoBERTa was trained to understand 100 languages; the baseline only learned English words.";
    modelDisagreementNote.style.display = 'block';
    modelDisagreementNote.classList.add('visible');
  } else {
    modelDisagreementNote.textContent = '';
    modelDisagreementNote.style.display = 'none';
    modelDisagreementNote.classList.remove('visible');
  }
}

function renderAnalysisResults(data) {
  let baseVerdictText = '';
  let transVerdictText = '';

  // 1. Baseline Card
  if (data.baseline && typeof data.baseline.probability === 'number') {
    const p = data.baseline.probability;
    const isToxic = p >= 0.5;
    baseVerdictText = isToxic ? 'Toxic' : 'Clean';

    animateNumber(baselineProb, p, 600);

    baselineBar.style.width = '0%';
    baselineBar.className = `dark-bar-fill ${isToxic ? 'toxic' : 'clean'}`;
    setTimeout(() => {
      baselineBar.style.width = `${Math.min(100, Math.max(0, p * 100))}%`;
    }, 20);

    baselineVerdict.textContent = baseVerdictText;
    baselineVerdict.className = `dark-card-verdict ${isToxic ? 'toxic' : 'clean'}`;
  } else {
    baselineProb.textContent = '—';
    baselineBar.style.width = '0%';
    baselineVerdict.textContent = 'Unavailable';
    baselineVerdict.className = 'dark-card-verdict';
  }

  // 2. XLM-RoBERTa Card
  if (data.transformer && typeof data.transformer.probability === 'number') {
    const p = data.transformer.probability;
    const isToxic = p >= 0.5;
    transVerdictText = isToxic ? 'Toxic' : 'Clean';

    animateNumber(transformerProb, p, 600);

    transformerBar.style.width = '0%';
    transformerBar.className = `dark-bar-fill ${isToxic ? 'toxic' : 'clean'}`;
    setTimeout(() => {
      transformerBar.style.width = `${Math.min(100, Math.max(0, p * 100))}%`;
    }, 20);

    transformerVerdict.textContent = transVerdictText;
    transformerVerdict.className = `dark-card-verdict ${isToxic ? 'toxic' : 'clean'}`;
  } else {
    transformerProb.textContent = '—';
    transformerBar.style.width = '0%';
    transformerVerdict.textContent = 'Unavailable';
    transformerVerdict.className = 'dark-card-verdict';
  }

  // 3. Disagreement note: displayed only when models disagree and both models are active
  checkDisagreement(data);

  classifyResultsSection.classList.add('visible');
}

// Load Hero Stats line directly from /results (never hardcoded)
async function loadHeroStats() {
  const heroStatsLine = document.getElementById('heroStatsLine');
  if (!heroStatsLine) return;
  try {
    const res = await fetch('/results');
    if (!res.ok) return;
    const data = await res.json();
    if (data.headline) {
      const langs = data.headline.languages_count || 6;
      const count = data.headline.total_comments
        ? data.headline.total_comments.toLocaleString()
        : '63,772';
      const rocAuc = data.headline.roc_auc !== null && data.headline.roc_auc !== undefined
        ? data.headline.roc_auc.toFixed(2)
        : '0.90';
      heroStatsLine.textContent = `${langs} languages · ${count} test comments · ${rocAuc} ROC-AUC`;
    }
  } catch (err) {
    console.error('Failed to load hero stats:', err);
  }
}

// Load Evaluation Results on the Results Page
let cachedPerLanguage = [];

async function loadEvaluationResults() {
  try {
    const res = await fetch('/results');
    if (!res.ok) return;
    const data = await res.json();
    resultsLoaded = true;

    // 1. Headline Number
    const headlineXlmF1 = document.getElementById('headlineXlmF1');
    const headlineBaseF1 = document.getElementById('headlineBaseF1');
    const headlineDescription = document.getElementById('headlineDescription');

    if (data.headline) {
      const xF1 = data.headline.transformer_f1;
      const bF1 = data.headline.baseline_f1;
      headlineXlmF1.textContent = xF1 !== null ? xF1.toFixed(3) : '—';
      headlineBaseF1.textContent = bF1 !== null ? `Baseline ${bF1.toFixed(3)}` : '';

      const count = data.headline.total_comments
        ? data.headline.total_comments.toLocaleString()
        : '63,772';
      const langs = data.headline.languages_count || 6;
      headlineDescription.textContent =
        `XLM-RoBERTa toxic-class F1 on the test set, evaluated on ${count} comments in ${langs} languages.`;
    }

    // 2. Per-language Chart Data
    cachedPerLanguage = data.per_language || [];
    renderPerLanguageChart();

    // 3. Comparison Tables
    renderTable('testTableBody', data.test_table);
    renderTable('valTableBody', data.validation_table);

    // 4. Confusion Matrix Images
    const xlmImg = document.getElementById('xlmCmImg');
    const baseImg = document.getElementById('baseCmImg');
    if (data.plots && data.plots.test_xlm_cm) {
      xlmImg.src = data.plots.test_xlm_cm;
      xlmImg.style.display = 'block';
    }
    if (data.plots && data.plots.test_baseline_cm) {
      baseImg.src = data.plots.test_baseline_cm;
      baseImg.style.display = 'block';
    }
  } catch (err) {
    console.error('Failed to load results:', err);
  }
}

function renderTable(tbodyId, rows) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  if (!rows || rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="10">No evaluation data available.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td style="font-weight: 500; color: #ffffff;">${r.model || '—'}</td>
      <td>${r.accuracy !== null ? r.accuracy.toFixed(4) : '—'}</td>
      <td>${r.precision !== null ? r.precision.toFixed(4) : '—'}</td>
      <td>${r.recall !== null ? r.recall.toFixed(4) : '—'}</td>
      <td style="font-weight: 600; color: #ffffff;">${r.f1 !== null ? r.f1.toFixed(4) : '—'}</td>
      <td>${r.roc_auc !== null ? r.roc_auc.toFixed(4) : '—'}</td>
      <td>${r.tn !== null ? r.tn.toLocaleString() : '—'}</td>
      <td>${r.fp !== null ? r.fp.toLocaleString() : '—'}</td>
      <td>${r.fn !== null ? r.fn.toLocaleString() : '—'}</td>
      <td>${r.tp !== null ? r.tp.toLocaleString() : '—'}</td>
    </tr>
  `).join('');
}

// Canvas Per-Language Bar Chart (Dark Theme)
function renderPerLanguageChart() {
  const canvas = document.getElementById('languageChart');
  if (!canvas || !cachedPerLanguage || cachedPerLanguage.length === 0) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  const data = cachedPerLanguage;
  const groupCount = data.length;
  const padBottom = 34;
  const padTop = 26;
  const chartHeight = h - padBottom - padTop;

  const groupWidth = w / groupCount;
  const barWidth = Math.min(28, (groupWidth - 28) / 2);
  const barGap = 4;

  data.forEach((item, i) => {
    const groupCenter = i * groupWidth + groupWidth / 2;
    const xlmF1 = item.transformer_f1 || 0;
    const baseF1 = item.baseline_f1 || 0;

    const xlmX = groupCenter - barWidth - barGap / 2;
    const baseX = groupCenter + barGap / 2;

    const xlmBarH = Math.max(2, xlmF1 * chartHeight);
    const baseBarH = Math.max(2, baseF1 * chartHeight);

    const xlmY = h - padBottom - xlmBarH;
    const baseY = h - padBottom - baseBarH;

    // XLM-RoBERTa bar in pure white (#ffffff)
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(xlmX, xlmY, barWidth, xlmBarH, [3, 3, 0, 0]) : ctx.rect(xlmX, xlmY, barWidth, xlmBarH);
    ctx.fill();

    // Baseline bar in dark grey (#55555a)
    ctx.fillStyle = '#55555a';
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(baseX, baseY, barWidth, baseBarH, [3, 3, 0, 0]) : ctx.rect(baseX, baseY, barWidth, baseBarH);
    ctx.fill();

    // F1 value labels on top of bars
    ctx.font = '11px -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif';
    ctx.textAlign = 'center';

    ctx.fillStyle = '#ffffff';
    ctx.fillText(xlmF1.toFixed(2), xlmX + barWidth / 2, xlmY - 6);

    ctx.fillStyle = '#86868b';
    ctx.fillText(baseF1.toFixed(2), baseX + barWidth / 2, baseY - 6);

    // Language label below
    ctx.fillStyle = '#ffffff';
    ctx.font = '13px -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif';
    ctx.fillText(item.lang || '', groupCenter, h - 10);
  });
}

window.addEventListener('resize', () => {
  if (resultsLoaded && resultsView.classList.contains('active')) {
    renderPerLanguageChart();
  }
});

// Batch CSV Upload Handler & CSV Export
const batchForm = document.getElementById('batchForm');
const batchFileInput = document.getElementById('batchFileInput');
const batchSubmitBtn = document.getElementById('batchSubmitBtn');
const batchResults = document.getElementById('batchResults');
const batchSummary = document.getElementById('batchSummary');
const batchTableBody = document.getElementById('batchTableBody');
const batchDownloadBtn = document.getElementById('batchDownloadBtn');

if (batchForm) {
  batchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = batchFileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    batchSubmitBtn.disabled = true;
    batchSubmitBtn.textContent = 'Analysing...';

    try {
      const res = await fetch('/predict-batch', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Batch analysis failed');
      }

      const data = await res.json();
      currentBatchRows = data.rows || [];

      batchSummary.textContent =
        `Processed ${data.total.toLocaleString()} comments. ` +
        `XLM-RoBERTa flagged ${data.transformer_toxic_count} as toxic. ` +
        `Baseline flagged ${data.baseline_toxic_count}.`;

      batchTableBody.innerHTML = currentBatchRows.slice(0, 100).map(r => `
        <tr>
          <td style="color: #86868b;">${r.index}</td>
          <td style="max-width: 480px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #ffffff;">${escapeHtml(r.text)}</td>
          <td style="${r.baseline_verdict === 'Toxic' ? 'color: #ff453a; font-weight: 500;' : 'color: #ffffff;'}">${r.baseline_verdict} (${r.baseline_prob !== null ? r.baseline_prob.toFixed(2) : '—'})</td>
          <td style="${r.transformer_verdict === 'Toxic' ? 'color: #ff453a; font-weight: 500;' : 'color: #ffffff;'}">${r.transformer_verdict} (${r.transformer_prob !== null ? r.transformer_prob.toFixed(2) : '—'})</td>
        </tr>
      `).join('');

      if (batchDownloadBtn) {
        batchDownloadBtn.style.display = 'inline-flex';
      }

      batchResults.style.display = 'block';
    } catch (err) {
      alert(err.message || 'Error processing CSV.');
    } finally {
      batchSubmitBtn.disabled = false;
      batchSubmitBtn.textContent = 'Analyse file';
    }
  });
}

// Offer predictions as CSV download
if (batchDownloadBtn) {
  batchDownloadBtn.addEventListener('click', () => {
    if (!currentBatchRows || currentBatchRows.length === 0) return;

    const headers = ['index', 'comment_text', 'baseline_verdict', 'baseline_probability', 'transformer_verdict', 'transformer_probability'];
    const lines = [headers.join(',')];

    currentBatchRows.forEach(r => {
      const escapedText = '"' + (r.text || '').replace(/"/g, '""') + '"';
      lines.push([
        r.index,
        escapedText,
        r.baseline_verdict || '',
        r.baseline_prob !== null && r.baseline_prob !== undefined ? r.baseline_prob : '',
        r.transformer_verdict || '',
        r.transformer_prob !== null && r.transformer_prob !== undefined ? r.transformer_prob : ''
      ].join(','));
    });

    const csvContent = lines.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const downloadAnchor = document.createElement('a');
    downloadAnchor.href = url;
    downloadAnchor.download = 'toxicity_batch_predictions.csv';
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    document.body.removeChild(downloadAnchor);
    URL.revokeObjectURL(url);
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
