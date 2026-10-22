/* ============================================================
   SkinAI – Frontend JavaScript
   ============================================================ */

'use strict';

// ============================================================
// DISEASE DATABASE (mirrors predict_disease.py)
// ============================================================
const DISEASES = [
  {
    code: 'mel', name: 'Melanoma', severity: 'Critical',
    emoji: '🔴', color: '#FF4444',
    description: 'The most dangerous type of skin cancer, originating from melanocytes. Early detection is critical as it can spread rapidly to other organs.',
    symptoms: 'Asymmetric moles, irregular borders, multiple colors, diameter >6mm, evolving lesions.',
    treatment: 'Surgical excision, immunotherapy (pembrolizumab), targeted therapy (BRAF inhibitors), radiation.',
    action: 'URGENT: See a dermatologist immediately. Do not delay.',
  },
  {
    code: 'bcc', name: 'Basal Cell Carcinoma', severity: 'High',
    emoji: '🟠', color: '#FF8800',
    description: 'The most common skin cancer. Grows slowly and rarely spreads, but can cause significant local tissue damage if left untreated.',
    symptoms: 'Pearly/translucent bump, flat scar-like lesion, bleeding/scabbing sore that heals and returns.',
    treatment: 'Surgical excision, Mohs surgery, radiation therapy, topical chemotherapy (imiquimod).',
    action: 'Schedule a dermatology appointment within 2-4 weeks.',
  },
  {
    code: 'akiec', name: 'Actinic Keratoses', severity: 'High',
    emoji: '🟠', color: '#FFAA00',
    description: 'Precancerous lesions caused by prolonged UV exposure. Can progress to squamous cell carcinoma if untreated.',
    symptoms: 'Rough, scaly patch on sun-damaged skin, itching or burning sensation.',
    treatment: 'Cryotherapy, topical chemotherapy (5-fluorouracil), photodynamic therapy, laser resurfacing.',
    action: 'Professional evaluation recommended within 4-6 weeks.',
  },
  {
    code: 'bkl', name: 'Benign Keratosis', severity: 'Low',
    emoji: '🟡', color: '#FFDD00',
    description: 'Non-cancerous skin growth including seborrheic keratosis, solar lentigo, and lichen planus-like keratosis.',
    symptoms: 'Waxy, stuck-on appearance, rough or smooth texture, varied colors.',
    treatment: 'Usually no treatment needed. Cryotherapy or curettage if cosmetically bothersome.',
    action: 'Routine skin check recommended annually.',
  },
  {
    code: 'nv', name: 'Melanocytic Nevi', severity: 'Low',
    emoji: '🟢', color: '#44BB44',
    description: 'Common, benign moles (nevi) formed from melanocyte clusters. Most are harmless, but changes in size/color/shape should be monitored.',
    symptoms: 'Round, well-defined, symmetrical pigmented spot. Uniform color.',
    treatment: 'No treatment required. Monitor for changes using ABCDE criteria.',
    action: 'Regular self-examination. Annual skin check recommended.',
  },
  {
    code: 'df', name: 'Dermatofibroma', severity: 'Low',
    emoji: '🟢', color: '#44BBBB',
    description: 'Benign fibrous skin nodule, commonly found on the legs. Thought to be a reactive process to minor injury.',
    symptoms: 'Small, firm bump, dimples inward when pinched, brownish or reddish.',
    treatment: 'No treatment necessary. Surgical excision if causing discomfort.',
    action: 'No urgent action required. Mention at next routine checkup.',
  },
  {
    code: 'vasc', name: 'Vascular Lesions', severity: 'Low',
    emoji: '🔵', color: '#4488FF',
    description: 'Abnormalities of blood vessels in the skin, including hemangiomas, angiokeratomas, and pyogenic granulomas.',
    symptoms: 'Red, purple, or blue skin lesions, may bleed easily.',
    treatment: 'Laser therapy, sclerotherapy, or surgical removal if indicated.',
    action: 'No urgent action required. Consult a doctor if lesion grows or bleeds.',
  },
];

// ============================================================
// UTILITIES
// ============================================================
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function show(el) { el?.classList.remove('hidden'); }
function hide(el) { el?.classList.add('hidden'); }
function toggle(el, show) { el?.classList.toggle('hidden', !show); }

function animateIn(el, cls = 'slide-up') {
  el.classList.remove(cls);
  void el.offsetWidth;   // reflow
  el.classList.add(cls);
}

// ============================================================
// PARTICLES
// ============================================================
function initParticles() {
  const container = $('#particles');
  if (!container) return;

  const count = 28;
  const colors = ['#6C63FF', '#4ECDC4', '#FF6B9D', '#FFD93D', '#6BCB77'];

  for (let i = 0; i < count; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 4 + 2;
    const left = Math.random() * 100;
    const duration = Math.random() * 20 + 15;
    const delay = Math.random() * -25;
    const color = colors[Math.floor(Math.random() * colors.length)];

    Object.assign(p.style, {
      width: `${size}px`,
      height: `${size}px`,
      left: `${left}%`,
      background: color,
      animationDuration: `${duration}s`,
      animationDelay: `${delay}s`,
      boxShadow: `0 0 ${size * 2}px ${color}`,
    });

    container.appendChild(p);
  }
}

// ============================================================
// MODEL STATUS CHECK
// ============================================================
async function checkModelStatus() {
  const dot  = $('#status-dot');
  const text = $('#status-text');

  try {
    const res = await fetch('/health');
    const data = await res.json();

    if (data.model_loaded) {
      dot.className = 'status-dot online';
      text.textContent = 'Model Online';
    } else {
      dot.className = 'status-dot offline';
      text.textContent = 'Model Not Trained';
    }
  } catch {
    dot.className = 'status-dot offline';
    text.textContent = 'Server Offline';
  }
}

// ============================================================
// UPLOAD & DRAG-DROP
// ============================================================
let selectedFile = null;

function initUpload() {
  const dropZone  = $('#drop-zone');
  const fileInput = $('#file-input');
  const preview   = $('#image-preview');
  const previewImg = $('#preview-img');
  const removeBtn = $('#remove-btn');
  const analyzeBtn = $('#analyze-btn');

  // File input change
  fileInput.addEventListener('change', (e) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0]);
  });

  // Drag & Drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
  });

  dropZone.addEventListener('click', (e) => {
    if (e.target === removeBtn || removeBtn?.contains(e.target)) return;
    if (!selectedFile) fileInput.click();
  });

  // Remove image
  removeBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    clearImage();
  });

  // Analyze button
  analyzeBtn.addEventListener('click', analyze);
}

function handleFile(file) {
  const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showNotification('Unsupported file type. Please use JPG, PNG, BMP, or WebP.', 'error');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showNotification('File too large. Maximum size is 16MB.', 'error');
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    const previewImg = $('#preview-img');
    const preview    = $('#image-preview');
    const dropZone   = $('#drop-zone');
    const analyzeBtn = $('#analyze-btn');

    previewImg.src = e.target.result;
    show(preview);
    dropZone.classList.add('has-image');

    // Hide text elements
    $$('.drop-icon, .drop-text, .drop-hint', dropZone).forEach(hide);

    analyzeBtn.disabled = false;
    animateIn(preview);
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  selectedFile = null;
  const dropZone  = $('#drop-zone');
  const preview   = $('#image-preview');
  const analyzeBtn = $('#analyze-btn');
  const fileInput = $('#file-input');

  hide(preview);
  dropZone.classList.remove('has-image');
  $$('.drop-icon, .drop-text, .drop-hint', dropZone).forEach(show);
  analyzeBtn.disabled = true;
  fileInput.value = '';

  showResultsState('empty');
}

// ============================================================
// PREDICTION
// ============================================================
async function analyze() {
  if (!selectedFile) return;

  const btn = $('#analyze-btn');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.querySelector('.btn-text').textContent = 'Analyzing';

  showResultsState('loading');

  try {
    const formData = new FormData();
    formData.append('image', selectedFile);

    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();

    if (!res.ok || !data.success) {
      if (res.status === 503) {
        showResultsState('no-model');
      } else {
        throw new Error(data.error || 'Prediction failed');
      }
      return;
    }

    showResultsState('results', data);

  } catch (err) {
    showNotification(`Error: ${err.message}`, 'error');
    showResultsState('empty');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.querySelector('.btn-text').textContent = 'Analyze Image';
  }
}

// ============================================================
// RESULTS DISPLAY
// ============================================================
function showResultsState(state, data = null) {
  const states = {
    empty:    $('#results-empty'),
    loading:  $('#results-loading'),
    'no-model': $('#results-no-model'),
    results:  $('#results-content'),
  };

  Object.entries(states).forEach(([key, el]) => toggle(el, key === state));

  if (state === 'results' && data) {
    populateResults(data);
  }
}

function populateResults(data) {
  const info = data.disease_info;

  // Header
  $('#result-emoji').textContent   = info.emoji;
  $('#result-disease').textContent = info.name;
  $('#result-disease').style.color = info.color;
  $('#result-risk').textContent    = info.risk_level;
  $('#result-risk').style.color    = info.color;
  $('#result-confidence').textContent = `${data.confidence.toFixed(1)}%`;

  // Action card color
  const actionCard = $('#action-card');
  if (data.is_critical) {
    actionCard.style.borderColor = `${info.color}55`;
    actionCard.style.background  = `${info.color}0d`;
  } else {
    actionCard.style.borderColor = '';
    actionCard.style.background  = '';
  }

  // Info text
  $('#info-description').textContent = info.description;
  $('#info-symptoms').textContent    = info.symptoms;
  $('#info-treatment').textContent   = info.treatment;
  $('#info-action').textContent      = info.action;

  // Confidence bars
  const barsContainer = $('#confidence-bars');
  barsContainer.innerHTML = '';

  data.all_predictions.forEach((pred) => {
    const item = document.createElement('div');
    item.className = 'confidence-bar-item';

    item.innerHTML = `
      <div class="confidence-bar-name" title="${pred.name}">${pred.name}</div>
      <div class="confidence-bar-track">
        <div class="confidence-bar-fill" style="width: 0%; background: ${pred.color};"></div>
      </div>
      <div class="confidence-bar-pct">${pred.confidence.toFixed(1)}%</div>
    `;

    barsContainer.appendChild(item);

    // Animate bar fill
    requestAnimationFrame(() => {
      setTimeout(() => {
        item.querySelector('.confidence-bar-fill').style.width = `${pred.confidence}%`;
      }, 50);
    });
  });

  // Inference time
  $('#inference-info').textContent =
    `Inference time: ${data.inference_time_ms} ms · Model: EfficientNetB2 · Dataset: HAM10000`;

  animateIn($('#results-content'), 'fade-in');
}

// ============================================================
// DISEASES GRID
// ============================================================
function buildDiseasesGrid() {
  const grid = $('#diseases-grid');
  if (!grid) return;

  DISEASES.forEach((d) => {
    const severityCls = {
      'Critical': 'severity-critical',
      'High':     'severity-high',
      'Medium':   'severity-medium',
      'Low':      'severity-low',
    }[d.severity] || 'severity-low';

    const card = document.createElement('div');
    card.className = 'disease-card';
    card.innerHTML = `
      <div class="disease-card-header">
        <div class="disease-card-emoji">${d.emoji}</div>
        <div>
          <div class="disease-card-name">${d.name}</div>
          <div class="disease-card-code">${d.code}</div>
        </div>
        <span class="disease-card-severity ${severityCls}">${d.severity}</span>
      </div>
      <div class="disease-card-desc">${d.description}</div>
      <div class="disease-card-treatment">💊 ${d.treatment.split(',')[0].trim()}</div>
    `;

    // Animated border on hover
    card.style.setProperty('--card-color', d.color);

    grid.appendChild(card);
  });
}

// ============================================================
// NOTIFICATIONS
// ============================================================
function showNotification(message, type = 'info') {
  // Simple banner at top
  let notif = $('#notification');
  if (!notif) {
    notif = document.createElement('div');
    notif.id = 'notification';
    Object.assign(notif.style, {
      position: 'fixed',
      top: '72px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: '999',
      padding: '12px 24px',
      borderRadius: '999px',
      fontSize: '0.875rem',
      fontWeight: '600',
      boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
      transition: 'opacity 0.3s',
      maxWidth: '480px',
      textAlign: 'center',
    });
    document.body.appendChild(notif);
  }

  const colors = {
    info:    { bg: 'rgba(108,99,255,0.9)',  border: '#6C63FF' },
    error:   { bg: 'rgba(255,71,87,0.9)',   border: '#FF4757' },
    success: { bg: 'rgba(107,203,119,0.9)', border: '#6BCB77' },
  }[type];

  notif.textContent = message;
  notif.style.background = colors.bg;
  notif.style.border = `1px solid ${colors.border}`;
  notif.style.color = '#fff';
  notif.style.opacity = '1';

  setTimeout(() => { notif.style.opacity = '0'; }, 4000);
}

// ============================================================
// SMOOTH SCROLL FOR NAV LINKS
// ============================================================
function initNavLinks() {
  $$('.nav-link').forEach((link) => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (href?.startsWith('#')) {
        e.preventDefault();
        document.querySelector(href)?.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });
}

// ============================================================
// INTERSECTION OBSERVER (fade-in on scroll)
// ============================================================
function initScrollAnimations() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('slide-up');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1 }
  );

  $$('.about-card, .disease-card, .flow-step').forEach((el) => {
    el.style.opacity = '0';
    observer.observe(el);
  });
}

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  checkModelStatus();
  initUpload();
  buildDiseasesGrid();
  initNavLinks();

  // Delay scroll animations slightly
  setTimeout(initScrollAnimations, 300);

  // Poll model status every 30 seconds
  setInterval(checkModelStatus, 30_000);
});
