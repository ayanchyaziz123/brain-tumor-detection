'use strict';

const CLASS_COLORS = {
  'Glioma':     '#ef4444',
  'Meningioma': '#f59e0b',
  'No Tumor':   '#22c55e',
  'Pituitary':  '#a78bfa',
};

const CLASS_ICONS = {
  'Glioma':     '',
  'Meningioma': '',
  'No Tumor':   '',
  'Pituitary':  '',
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');
const previewWrap   = document.getElementById('previewWrap');
const previewImg    = document.getElementById('previewImg');
const clearBtn      = document.getElementById('clearBtn');
const analyzeBtn    = document.getElementById('analyzeBtn');
const btnText       = document.getElementById('btnText');
const spinner       = document.getElementById('spinner');
const resultCard    = document.getElementById('resultCard');
const resultHeader  = document.getElementById('resultHeader');
const resultBadge   = document.getElementById('resultBadge');
const resultClass   = document.getElementById('resultClass');
const resultConf    = document.getElementById('resultConf');
const resultImg     = document.getElementById('resultImg');
const heatmapImg    = document.getElementById('heatmapImg');
const overlayImg    = document.getElementById('overlayImg');
const confBadge     = document.getElementById('confBadge');
const confBadgeIcon = document.getElementById('confBadgeIcon');
const confBadgeText = document.getElementById('confBadgeText');
const probBars      = document.getElementById('probBars');
const errorCard     = document.getElementById('errorCard');
const errorMsg      = document.getElementById('errorMsg');
const resetBtn      = document.getElementById('resetBtn');
const downloadBtn   = document.getElementById('downloadBtn');
const errorResetBtn = document.getElementById('errorResetBtn');

let selectedFile = null;
let lastResult   = null;

// ── File selection ────────────────────────────────────────────────────────────
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

dropZone.addEventListener('click', e => {
  if (e.target.tagName !== 'BUTTON') fileInput.click();
});

function handleFile(file) {
  if (!file) return;
  const allowed = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff', 'image/gif'];
  if (!allowed.includes(file.type)) {
    showError('Invalid file type. Please upload a JPG, PNG, BMP, or TIFF image.');
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src = ev.target.result;
    previewWrap.classList.remove('hidden');
    dropZone.classList.add('hidden');
    analyzeBtn.disabled = false;
    resultCard.classList.add('hidden');
    errorCard.classList.add('hidden');
  };
  reader.readAsDataURL(file);
}

// ── Clear ─────────────────────────────────────────────────────────────────────
clearBtn.addEventListener('click', resetUpload);

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewWrap.classList.add('hidden');
  dropZone.classList.remove('hidden');
  analyzeBtn.disabled = true;
  resultCard.classList.add('hidden');
  errorCard.classList.add('hidden');
}

// ── Analyze ───────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', analyzeImage);

async function analyzeImage() {
  if (!selectedFile) return;

  setLoading(true, 'Analyzing... (may take 20-30s on first run)');
  resultCard.classList.add('hidden');
  errorCard.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', selectedFile);

  // 90-second timeout for cold-start on Render free tier
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 90000);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData, signal: controller.signal });
    clearTimeout(timer);
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Unexpected error. Please try again.');
      return;
    }
    renderResult(data);
  } catch (err) {
    clearTimeout(timer);
    if (err.name === 'AbortError') {
      showError('Request timed out. The server may be waking up — please try again in a moment.');
    } else {
      showError('Server is starting up. Please wait 30 seconds and try again.');
    }
  } finally {
    setLoading(false, 'Analyze MRI');
  }
}

// ── PDF download ──────────────────────────────────────────────────────────────
downloadBtn.addEventListener('click', async () => {
  if (!lastResult) return;
  downloadBtn.disabled = true;
  downloadBtn.textContent = 'Generating PDF...';

  try {
    const res = await fetch('/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(lastResult),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(err.error || 'PDF generation failed.');
      return;
    }

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'brain_tumor_report.pdf';
    a.click();
    URL.revokeObjectURL(url);
  } catch {
    alert('Failed to generate report. Please try again.');
  } finally {
    downloadBtn.disabled = false;
    downloadBtn.textContent = 'Download PDF Report';
  }
});

// ── Confidence badge ──────────────────────────────────────────────────────────
function renderConfBadge(confPct) {
  confBadge.classList.remove('hidden', 'high', 'medium', 'low');

  if (confPct >= 85) {
    confBadge.classList.add('high');
    confBadgeIcon.textContent = 'HIGH CONFIDENCE';
    confBadgeText.textContent = `${confPct.toFixed(1)}% — Result is reliable.`;
  } else if (confPct >= 60) {
    confBadge.classList.add('medium');
    confBadgeIcon.textContent = 'MODERATE CONFIDENCE';
    confBadgeText.textContent = `${confPct.toFixed(1)}% — Consider reviewing with a specialist.`;
  } else {
    confBadge.classList.add('low');
    confBadgeIcon.textContent = 'LOW CONFIDENCE';
    confBadgeText.textContent = `${confPct.toFixed(1)}% — Result is uncertain. Please consult a radiologist.`;
  }
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(data) {
  lastResult = data;
  const cls    = data.predicted_class;
  const conf   = (data.confidence * 100).toFixed(1);
  const color  = CLASS_COLORS[cls] || '#6c63ff';
  const icon   = CLASS_ICONS[cls]  || '&#129504;';
  const isTumor = data.is_tumor;

  // Header
  resultCard.className = `card result-card ${isTumor ? 'tumor-positive' : 'tumor-negative'}`;
  resultBadge.innerHTML = icon;
  resultBadge.style.background = color + '22';
  resultBadge.style.border     = `2px solid ${color}44`;
  resultClass.textContent = cls;
  resultConf.textContent  = `Confidence: ${conf}%`;
  resultImg.src    = data.image_url;
  heatmapImg.src   = data.heatmap_url;
  overlayImg.src   = data.overlay_url;

  // Confidence warning badge
  renderConfBadge(parseFloat(conf));

  // Probability bars
  probBars.innerHTML = '';
  const sortedProbs = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1]);

  sortedProbs.forEach(([name, prob]) => {
    const pct   = (prob * 100).toFixed(1);
    const c     = CLASS_COLORS[name] || '#6c63ff';
    const isTop = name === cls;

    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-label" style="${isTop ? 'font-weight:700' : ''}">${name}</span>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:0%;background:${c}${isTop ? '' : '88'}"></div>
      </div>
      <span class="prob-value">${pct}%</span>
    `;
    probBars.appendChild(row);
    // Animate bar width
    setTimeout(() => {
      row.querySelector('.prob-bar-fill').style.width = pct + '%';
    }, 50);
  });

  resultCard.classList.remove('hidden');
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Error ─────────────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorCard.classList.remove('hidden');
  errorCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Reset buttons ─────────────────────────────────────────────────────────────
resetBtn.addEventListener('click', resetUpload);
errorResetBtn.addEventListener('click', () => {
  errorCard.classList.add('hidden');
  resetUpload();
});

// ── Loading state ─────────────────────────────────────────────────────────────
function setLoading(on, label = 'Analyze MRI') {
  analyzeBtn.disabled = on;
  btnText.textContent = on ? label : 'Analyze MRI';
  spinner.classList.toggle('hidden', !on);
}
