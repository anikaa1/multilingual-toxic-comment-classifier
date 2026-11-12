/**
 * In-browser Dashboard Application for GitHub Pages.
 *
 * Runs text classification entirely client-side:
 * - Baseline: in-browser TF-IDF + Logistic Regression via docs/baseline.js
 * - XLM-RoBERTa: in-browser ONNX inference via Transformers.js
 * - Results page: reads local metrics JSON files from docs/results/
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Configure Transformers.js for browser execution
env.allowLocalModels = false;
env.useBrowserCache = true;

// DOM Elements
const navClassify = document.getElementById('navClassify');
const navResults = document.getElementById('navResults');
const classifyView = document.getElementById('classifyView');
const resultsView = document.getElementById('resultsView');
const tryItNavBtn = document.getElementById('tryItNavBtn');

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

const modelDownloadBanner = document.getElementById('modelDownloadBanner');
const modelDownloadText = document.getElementById('modelDownloadText');

let baselineClassifier = null;
let transformerPipeline = null;
let transformerLoadingPromise = null;
let isAnalysing = false;
let xlmOnlyMode = false;
let resultsLoaded = false;
let currentBatchRows = [];
let lastAnalysisData = null;

// Multilingual examples
const EXAMPLES = {
  en: "You are a complete idiot and should delete your account immediately.",
  es: "Eres un imbécil total y no sabes nada de este tema, cállate la boca.",
  tr: "Sen tam bir aptalsın, hiçbir şeyden anlamıyorsun defol git buradan.",
  ru: "Ты полный идиот и ничего не понимаешь в этом, закрой свой рот.",
  fr: "Tu es un idiot fini et tu ne comprends rien à ce sujet, tais-toi.",
  pt: "Você é um idiota completo e não entende nada sobre isso, cala a boca.",
  polite: "This article provides very thoughtful and well-researched insights. Thank you."
};

// --------------------------------------------------------------------------
// Navigation & Routing
// --------------------------------------------------------------------------
function setRoute() {
  const hash = window.location.hash.toLowerCase();
  if (hash === '#results') {
    navClassify.classList.remove('active');
    navResults.classList.add('active');
    classifyView.classList.remove('active');
    resultsView.classList.add('active');

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

    updateScrollFade();
  }
}

window.addEventListener('hashchange', setRoute);

// Scroll into dark background transition on Classify page
function updateScrollFade() {
  if (!classifyView.classList.contains('active')) return;

  const scrollY = window.scrollY || window.pageYOffset;
  const windowH = window.innerHeight || 800;

  const fadeStart = windowH * 0.15;
  const fadeEnd = windowH * 0.85;
  const progress = Math.min(1, Math.max(0, (scrollY - fadeStart) / (fadeEnd - fadeStart)));

  const val = Math.round(255 * (1 - progress));
  document.body.style.backgroundColor = `rgb(${val}, ${val}, ${val})`;

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

function updateInputState() {
  const val = classifyInput.value;
  charCount.textContent = val.length;
  if (!isAnalysing) {
    analyseBtn.disabled = val.trim().length === 0;
  }
}

classifyInput.addEventListener('input', updateInputState);

classifyInput.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    if (!analyseBtn.disabled && !isAnalysing) {
      runAnalysis();
    }
  }
});

// Example links
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

// --------------------------------------------------------------------------
// Model Loading
// --------------------------------------------------------------------------

// 1. Initialize Baseline model
async function initBaseline() {
  try {
    baselineClassifier = await BaselineClassifier.load('./model/baseline.json');
    console.log('Baseline model loaded successfully.');
  } catch (err) {
    console.error('Failed to load baseline model:', err);
  }
}

// 2. Initialize XLM-RoBERTa via Transformers.js
async function initTransformer(onProgress) {
  if (transformerPipeline) return transformerPipeline;
  if (transformerLoadingPromise) return transformerLoadingPromise;

  if (modelDownloadBanner) {
    modelDownloadBanner.style.display = 'flex';
  }

  transformerLoadingPromise = (async () => {
    try {
      const pipe = await pipeline('text-classification', 'anika-1/xlm-roberta-toxic-onnx', {
        quantized: true,
        progress_callback: (info) => {
          if (info.status === 'progress' && modelDownloadText) {
            const pct = Math.round(info.progress || 0);
            modelDownloadText.textContent =
              `Loading multilingual model, about 280 MB, first visit only (${pct}%)...`;
          } else if (info.status === 'done' || info.status === 'ready') {
            if (modelDownloadBanner) modelDownloadBanner.style.display = 'none';
          }
          if (onProgress) onProgress(info);
        }
      });

      transformerPipeline = pipe;
      if (modelDownloadBanner) {
        modelDownloadBanner.style.display = 'none';
      }
      console.log('XLM-RoBERTa ONNX pipeline ready.');
      return pipe;
    } catch (err) {
      console.error('Error loading XLM-RoBERTa:', err);
      if (modelDownloadBanner) {
        modelDownloadText.textContent = 'Error loading multilingual model from Hugging Face.';
      }
      throw err;
    } finally {
      transformerLoadingPromise = null;
    }
  })();

  return transformerLoadingPromise;
}

// --------------------------------------------------------------------------
// Hero & Interactive UI Initialization
// --------------------------------------------------------------------------
function initHeroInteractions() {
  // Comment Bubbles
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

  // Language Cursors
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

  // XLM-RoBERTa Toggle Card
  const heroModelToggle = document.getElementById('heroModelToggle');
  const heroToggleSwitch = document.getElementById('heroToggleSwitch');

  function applyToggleMode() {
    if (xlmOnlyMode) {
      heroModelToggle.classList.add('active');
      heroToggleSwitch.setAttribute('aria-checked', 'true');
      baselineCard.classList.add('hidden');
      classifierDarkGrid.classList.add('single-model');
      if (modelDisagreementNote) modelDisagreementNote.style.display = 'none';
    } else {
      heroModelToggle.classList.remove('active');
      heroToggleSwitch.setAttribute('aria-checked', 'false');
      baselineCard.classList.remove('hidden');
      classifierDarkGrid.classList.remove('single-model');
      if (lastAnalysisData) checkDisagreement(lastAnalysisData);
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

// --------------------------------------------------------------------------
// In-Browser Classification Logic
// --------------------------------------------------------------------------
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
    // 1. Baseline prediction (instant in pure JS)
    let baselineResult = null;
    if (baselineClassifier) {
      baselineResult = baselineClassifier.predictProba(text);
    }

    // 2. XLM-RoBERTa prediction (Transformers.js ONNX)
    let transformerResult = null;
    try {
      const pipe = await initTransformer();
      const output = await pipe(text, { top_k: null });
      // output: [{ label: 'LABEL_0', score: 0.1 }, { label: 'LABEL_1', score: 0.9 }]
      let pToxic = 0.0;
      if (Array.isArray(output)) {
        for (const item of output) {
          const lbl = String(item.label).toUpperCase();
          if (lbl === 'LABEL_1' || lbl === 'TOXIC') {
            pToxic = item.score;
            break;
          }
        }
        // Fallback if top_k returned single best label
        if (pToxic === 0.0 && output.length === 1) {
          const lbl = String(output[0].label).toUpperCase();
          pToxic = (lbl === 'LABEL_1' || lbl === 'TOXIC') ? output[0].score : (1.0 - output[0].score);
        }
      }
      transformerResult = {
        probability: pToxic,
        verdict: pToxic >= 0.5 ? 'Toxic' : 'Clean'
      };
    } catch (tErr) {
      console.error('Transformer inference failed:', tErr);
    }

    const data = {
      text,
      baseline: baselineResult,
      transformer: transformerResult
    };

    lastAnalysisData = data;
    renderAnalysisResults(data);
  } catch (err) {
    console.error('Analysis failed:', err);
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
  // Baseline card
  if (data.baseline && typeof data.baseline.probability === 'number') {
    const p = data.baseline.probability;
    const isToxic = p >= 0.5;
    animateNumber(baselineProb, p, 600);

    baselineBar.style.width = '0%';
    baselineBar.className = `dark-bar-fill ${isToxic ? 'toxic' : 'clean'}`;
    setTimeout(() => {
      baselineBar.style.width = `${Math.min(100, Math.max(0, p * 100))}%`;
    }, 20);

    baselineVerdict.textContent = isToxic ? 'Toxic' : 'Clean';
    baselineVerdict.className = `dark-card-verdict ${isToxic ? 'toxic' : 'clean'}`;
  } else {
    baselineProb.textContent = '—';
    baselineBar.style.width = '0%';
    baselineVerdict.textContent = 'Unavailable';
    baselineVerdict.className = 'dark-card-verdict';
  }

  // Transformer card
  if (data.transformer && typeof data.transformer.probability === 'number') {
    const p = data.transformer.probability;
    const isToxic = p >= 0.5;
    animateNumber(transformerProb, p, 600);

    transformerBar.style.width = '0%';
    transformerBar.className = `dark-bar-fill ${isToxic ? 'toxic' : 'clean'}`;
    setTimeout(() => {
      transformerBar.style.width = `${Math.min(100, Math.max(0, p * 100))}%`;
    }, 20);

    transformerVerdict.textContent = isToxic ? 'Toxic' : 'Clean';
    transformerVerdict.className = `dark-card-verdict ${isToxic ? 'toxic' : 'clean'}`;
  } else {
    transformerProb.textContent = '—';
    transformerBar.style.width = '0%';
    transformerVerdict.textContent = 'Loading...';
    transformerVerdict.className = 'dark-card-verdict';
  }

  checkDisagreement(data);
  classifyResultsSection.classList.add('visible');
}

// --------------------------------------------------------------------------
// Results Page: Reads local metrics JSON files directly
// --------------------------------------------------------------------------
let cachedPerLanguage = [];

async function loadEvaluationResults() {
  try {
    const fetchJson = async (url) => {
      try {
        const res = await fetch(url);
        return res.ok ? await res.json() : null;
      } catch {
        return null;
      }
    };

    const [xlmTest, baseTest, xlmVal, baseVal] = await Promise.all([
      fetchJson('./results/test/xlm-roberta_metrics.json'),
      fetchJson('./results/test/baseline_tf-idf_metrics.json'),
      fetchJson('./results/validation/xlm-roberta_metrics.json') || fetchJson('./results/xlm-roberta_metrics.json'),
      fetchJson('./results/validation/baseline_tf-idf_metrics.json')
    ]);

    resultsLoaded = true;

    // 1. Headline Numbers
    const headlineXlmF1 = document.getElementById('headlineXlmF1');
    const headlineBaseF1 = document.getElementById('headlineBaseF1');
    const headlineDescription = document.getElementById('headlineDescription');

    if (xlmTest) {
      const xF1 = xlmTest.f1;
      const bF1 = baseTest ? baseTest.f1 : null;
      headlineXlmF1.textContent = xF1 !== null ? xF1.toFixed(3) : '—';
      headlineBaseF1.textContent = bF1 !== null ? `Baseline ${bF1.toFixed(3)}` : '';

      const count = xlmTest.support && xlmTest.support.total
        ? xlmTest.support.total.toLocaleString()
        : '63,772';
      const langs = xlmTest.per_language ? xlmTest.per_language.length : 6;
      headlineDescription.textContent =
        `XLM-RoBERTa toxic-class F1 on the test set, evaluated on ${count} comments in ${langs} languages.`;
    }

    // 2. Per-language Comparison Data
    const perLangMap = {};
    if (xlmTest && xlmTest.per_language) {
      xlmTest.per_language.forEach(item => {
        const lang = item.lang.toUpperCase();
        perLangMap[lang] = {
          lang,
          transformer_f1: item.f1 || 0,
          sample_count: item.sample_count || 0
        };
      });
    }
    if (baseTest && baseTest.per_language) {
      baseTest.per_language.forEach(item => {
        const lang = item.lang.toUpperCase();
        if (!perLangMap[lang]) {
          perLangMap[lang] = { lang, sample_count: item.sample_count || 0 };
        }
        perLangMap[lang].baseline_f1 = item.f1 || 0;
      });
    }

    cachedPerLanguage = Object.values(perLangMap).sort((a, b) => (b.sample_count || 0) - (a.sample_count || 0));
    renderPerLanguageChart();

    // 3. Comparison Tables
    function buildRow(name, data) {
      if (!data) return null;
      const cm = data.confusion_matrix || {};
      return {
        model: name,
        accuracy: data.accuracy,
        precision: data.precision,
        recall: data.recall,
        f1: data.f1,
        roc_auc: data.roc_auc,
        tn: cm.tn,
        fp: cm.fp,
        fn: cm.fn,
        tp: cm.tp
      };
    }

    const testRows = [
      buildRow('XLM-RoBERTa', xlmTest),
      buildRow('Baseline (TF-IDF)', baseTest)
    ].filter(Boolean);

    const valRows = [
      buildRow('XLM-RoBERTa', xlmVal),
      buildRow('Baseline (TF-IDF)', baseVal)
    ].filter(Boolean);

    renderTable('testTableBody', testRows);
    renderTable('valTableBody', valRows);

    // 4. Confusion Matrices
    const xlmImg = document.getElementById('xlmCmImg');
    const baseImg = document.getElementById('baseCmImg');
    if (xlmImg) xlmImg.style.display = 'block';
    if (baseImg) baseImg.style.display = 'block';

  } catch (err) {
    console.error('Failed to load results metrics:', err);
  }
}

// Hero subtle stats line
async function loadHeroStats() {
  const heroStatsLine = document.getElementById('heroStatsLine');
  if (!heroStatsLine) return;
  try {
    const res = await fetch('./results/test/xlm-roberta_metrics.json');
    if (!res.ok) return;
    const xlmTest = await res.json();
    const count = xlmTest.support && xlmTest.support.total
      ? xlmTest.support.total.toLocaleString()
      : '63,772';
    const langs = xlmTest.per_language ? xlmTest.per_language.length : 6;
    const rocAuc = xlmTest.roc_auc ? xlmTest.roc_auc.toFixed(2) : '0.90';
    heroStatsLine.textContent = `${langs} languages · ${count} test comments · ${rocAuc} ROC-AUC`;
  } catch (err) {
    console.error('Failed to load hero stats:', err);
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
      <td>${r.accuracy != null ? r.accuracy.toFixed(4) : '—'}</td>
      <td>${r.precision != null ? r.precision.toFixed(4) : '—'}</td>
      <td>${r.recall != null ? r.recall.toFixed(4) : '—'}</td>
      <td style="font-weight: 600; color: #ffffff;">${r.f1 != null ? r.f1.toFixed(4) : '—'}</td>
      <td>${r.roc_auc != null ? r.roc_auc.toFixed(4) : '—'}</td>
      <td>${r.tn != null ? r.tn.toLocaleString() : '—'}</td>
      <td>${r.fp != null ? r.fp.toLocaleString() : '—'}</td>
      <td>${r.fn != null ? r.fn.toLocaleString() : '—'}</td>
      <td>${r.tp != null ? r.tp.toLocaleString() : '—'}</td>
    </tr>
  `).join('');
}

// Canvas Per-Language Bar Chart
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

    // XLM-RoBERTa bar: pure white
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(xlmX, xlmY, barWidth, xlmBarH, [3, 3, 0, 0]) : ctx.rect(xlmX, xlmY, barWidth, xlmBarH);
    ctx.fill();

    // Baseline bar: dark grey
    ctx.fillStyle = '#55555a';
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(baseX, baseY, barWidth, baseBarH, [3, 3, 0, 0]) : ctx.rect(baseX, baseY, barWidth, baseBarH);
    ctx.fill();

    ctx.font = '11px -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif';
    ctx.textAlign = 'center';

    ctx.fillStyle = '#ffffff';
    ctx.fillText(xlmF1.toFixed(2), xlmX + barWidth / 2, xlmY - 6);

    ctx.fillStyle = '#86868b';
    ctx.fillText(baseF1.toFixed(2), baseX + barWidth / 2, baseY - 6);

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

// --------------------------------------------------------------------------
// In-Browser Batch CSV Processing & CSV Export
// --------------------------------------------------------------------------
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

    batchSubmitBtn.disabled = true;
    batchSubmitBtn.textContent = 'Processing CSV in browser...';

    try {
      const text = await file.text();
      const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
      if (lines.length <= 1) {
        throw new Error('CSV file has no data rows.');
      }

      // Parse headers
      const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, '').toLowerCase());
      let colIdx = headers.indexOf('comment_text');
      if (colIdx === -1) colIdx = headers.indexOf('text');
      if (colIdx === -1) colIdx = headers.indexOf('comment');
      if (colIdx === -1) colIdx = 0; // fallback to first column

      // Limit to first 200 comments for responsive in-browser execution
      const dataRows = lines.slice(1, 201);
      const rows = [];
      let baseToxicCount = 0;
      let transToxicCount = 0;

      // Ensure transformer is ready
      const pipe = await initTransformer();

      for (let i = 0; i < dataRows.length; i++) {
        const rawRow = dataRows[i];
        let comment = rawRow;
        // Simple CSV quote extraction
        if (rawRow.startsWith('"') && rawRow.endsWith('"')) {
          comment = rawRow.slice(1, -1).replace(/""/g, '"');
        } else {
          const parts = rawRow.split(',');
          comment = parts[colIdx] || parts[0] || '';
        }

        comment = comment.trim();
        if (!comment) continue;

        // 1. Baseline prediction
        let bProb = null;
        let bVerdict = 'N/A';
        if (baselineClassifier) {
          const bRes = baselineClassifier.predictProba(comment);
          bProb = bRes.probability;
          bVerdict = bRes.verdict;
          if (bVerdict === 'Toxic') baseToxicCount++;
        }

        // 2. Transformer prediction
        let tProb = null;
        let tVerdict = 'N/A';
        try {
          const out = await pipe(comment, { top_k: null });
          if (Array.isArray(out)) {
            for (const item of out) {
              const lbl = String(item.label).toUpperCase();
              if (lbl === 'LABEL_1' || lbl === 'TOXIC') {
                tProb = item.score;
                break;
              }
            }
          }
          if (tProb !== null) {
            tVerdict = tProb >= 0.5 ? 'Toxic' : 'Clean';
            if (tVerdict === 'Toxic') transToxicCount++;
          }
        } catch (tErr) {
          console.error('Batch item error:', tErr);
        }

        rows.push({
          index: rows.length + 1,
          text: comment,
          baseline_verdict: bVerdict,
          baseline_prob: bProb,
          transformer_verdict: tVerdict,
          transformer_prob: tProb
        });
      }

      currentBatchRows = rows;

      batchSummary.textContent =
        `Processed ${rows.length} comments directly in browser. ` +
        `XLM-RoBERTa flagged ${transToxicCount} as toxic. ` +
        `Baseline flagged ${baseToxicCount}.`;

      batchTableBody.innerHTML = rows.slice(0, 100).map(r => `
        <tr>
          <td style="color: #86868b;">${r.index}</td>
          <td style="max-width: 480px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #ffffff;">${escapeHtml(r.text)}</td>
          <td style="${r.baseline_verdict === 'Toxic' ? 'color: #ff453a; font-weight: 500;' : 'color: #ffffff;'}">${r.baseline_verdict} (${r.baseline_prob != null ? r.baseline_prob.toFixed(2) : '—'})</td>
          <td style="${r.transformer_verdict === 'Toxic' ? 'color: #ff453a; font-weight: 500;' : 'color: #ffffff;'}">${r.transformer_verdict} (${r.transformer_prob != null ? r.transformer_prob.toFixed(2) : '—'})</td>
        </tr>
      `).join('');

      if (batchDownloadBtn) batchDownloadBtn.style.display = 'inline-flex';
      batchResults.style.display = 'block';

    } catch (err) {
      alert(err.message || 'Error processing CSV.');
    } finally {
      batchSubmitBtn.disabled = false;
      batchSubmitBtn.textContent = 'Analyse file';
    }
  });
}

// Download predictions as CSV
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
        r.baseline_prob != null ? r.baseline_prob.toFixed(4) : '',
        r.transformer_verdict || '',
        r.transformer_prob != null ? r.transformer_prob.toFixed(4) : ''
      ].join(','));
    });

    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'toxicity_predictions_browser.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/&lt;/g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// --------------------------------------------------------------------------
// Initialization on Page Load
// --------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', () => {
  setRoute();
  initHeroInteractions();
  initBaseline();
  loadHeroStats();
  updateInputState();
});
