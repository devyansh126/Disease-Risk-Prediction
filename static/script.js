// ── Form submission ────────────────────────────────────────────────────────
const form       = document.getElementById('predForm');
const submitBtn  = document.getElementById('submitBtn');
const resultPanel = document.getElementById('resultPanel');

if (form) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setLoading(true);

    const data = {};
    new FormData(form).forEach((v, k) => { data[k] = v; });

    try {
      const res  = await fetch(`/predict/${DISEASE}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      const json = await res.json();
      if (json.error) throw new Error(json.error);
      showResult(json);
    } catch (err) {
      alert('Prediction failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  });
}

// ── Validation ─────────────────────────────────────────────────────────────
function validateForm() {
  let valid = true;
  form.querySelectorAll('input, select').forEach(el => {
    const errEl = document.getElementById('err_' + el.name);
    el.classList.remove('invalid');
    if (errEl) errEl.textContent = '';

    if (el.required && el.value === '') {
      el.classList.add('invalid');
      if (errEl) errEl.textContent = 'This field is required';
      valid = false;
    } else if (el.type === 'number' && el.value !== '') {
      const v = parseFloat(el.value);
      const mn = parseFloat(el.min);
      const mx = parseFloat(el.max);
      if (!isNaN(mn) && v < mn) {
        el.classList.add('invalid');
        if (errEl) errEl.textContent = `Min value is ${mn}`;
        valid = false;
      } else if (!isNaN(mx) && v > mx) {
        el.classList.add('invalid');
        if (errEl) errEl.textContent = `Max value is ${mx}`;
        valid = false;
      }
    }
  });
  if (!valid) {
    form.querySelector('.invalid')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
  return valid;
}

// ── Loading state ──────────────────────────────────────────────────────────
function setLoading(on) {
  submitBtn.disabled = on;
  submitBtn.querySelector('.btn-text').hidden = on;
  submitBtn.querySelector('.btn-spinner').hidden = !on;
}

// ── Show result ────────────────────────────────────────────────────────────
const NOTES = {
  low:      'Your values suggest a low risk. Maintain a healthy lifestyle and consult your doctor for regular checkups.',
  moderate: 'Your values indicate a moderate risk. Consider consulting a healthcare professional for further evaluation.',
  high:     'Your values suggest a high risk. Please consult a doctor as soon as possible for a proper diagnosis.',
};

function showResult(data) {
  // Unhide panel FIRST so the browser paints it before animations start
  resultPanel.hidden = false;

  // Defer animation to next frame so SVG is visible before stroke-dashoffset runs
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

      // ── Meter arc ────────────────────────────────────────────────
      const arc    = document.getElementById('meterArc');
      const pctTxt = document.getElementById('meterPct');
      const total  = 283;
      const pct    = Math.min(Math.max(data.probability, 0), 100);

      arc.style.stroke = data.color;
      // Reset to 0 before animating (handles "Check Again" re-runs)
      arc.style.transition = 'none';
      arc.style.strokeDashoffset = total;
      pctTxt.textContent = '0%';

      // Re-enable transition on next frame then animate
      requestAnimationFrame(() => {
        arc.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)';
        arc.style.strokeDashoffset = total - (pct / 100) * total;
      });

      // Counter animation
      let current = 0;
      const step  = pct / 60;
      const timer = setInterval(() => {
        current = Math.min(current + step, pct);
        pctTxt.textContent = Math.round(current) + '%';
        if (current >= pct) clearInterval(timer);
          }, 16);
    }); // end inner rAF
  }); // end outer rAF

  // ── Badge ──────────────────────────────────────────────────────────
  const badge = document.getElementById('resultBadge');
  badge.textContent  = data.risk_level.toUpperCase() + ' RISK';
  badge.className    = 'result-badge ' + data.risk_level;

  // ── Prediction label ───────────────────────────────────────────────
  document.getElementById('resultPrediction').textContent = data.prediction;

  // ── Note ───────────────────────────────────────────────────────────
  document.getElementById('resultNote').textContent = NOTES[data.risk_level] || '';

  // ── Multiclass breakdown ───────────────────────────────────────────
  const breakdown = document.getElementById('probBreakdown');
  const barsDiv   = document.getElementById('probBars');

  if (data.multiclass && data.all_probs) {
    breakdown.hidden = false;
    barsDiv.innerHTML = '';
    const colors = ['#00d4ff','#00e5a0','#ff9500','#ff3b5c','#9b6dff','#ff6b6b','#ffd93d'];
    Object.entries(data.all_probs).forEach(([label, prob], i) => {
      const row = document.createElement('div');
      row.className = 'prob-bar-row';
      row.innerHTML = `
        <div class="prob-bar-label">
          <span>${label}</span>
          <span>${prob.toFixed(1)}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="background:${colors[i % colors.length]}"></div>
        </div>`;
      barsDiv.appendChild(row);
      // Animate width after DOM insertion
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          row.querySelector('.prob-bar-fill').style.width = prob + '%';
        });
      });
    });
  } else {
    breakdown.hidden = true;
  }
}

// ── Reset ──────────────────────────────────────────────────────────────────
function resetForm() {
  form.reset();
  form.querySelectorAll('.invalid').forEach(el => el.classList.remove('invalid'));
  form.querySelectorAll('.field-error').forEach(el => el.textContent = '');
  resultPanel.hidden = true;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Stats panel toggle ─────────────────────────────────────────────────────
function toggleStats() {
  const body    = document.getElementById('statsBody');
  const chevron = document.getElementById('statsChevron');
  if (!body) return;
  const open = body.classList.toggle('open');
  chevron.classList.toggle('open', open);
  if (open) buildChart();
}

// ── Model comparison chart ─────────────────────────────────────────────────
let chartBuilt = false;

function buildChart() {
  if (chartBuilt) return;
  if (typeof MODEL_LABELS === 'undefined') return;
  chartBuilt = true;

  const colorMap = {
    cyan:   { cv: '#00d4ff', test: 'rgba(0,212,255,0.45)' },
    pink:   { cv: '#ff3b8a', test: 'rgba(255,59,138,0.45)' },
    green:  { cv: '#00e5a0', test: 'rgba(0,229,160,0.45)' },
    purple: { cv: '#9b6dff', test: 'rgba(155,109,255,0.45)' },
  };
  const c  = colorMap[STATS_COLOR] || colorMap.cyan;
  const ctx = document.getElementById('modelChart');
  if (!ctx) return;

  // Highlight best model bar
  const bgCV   = MODEL_LABELS.map(l => l === BEST_MODEL ? c.cv   : 'rgba(42,47,74,0.9)');
  const bgTest = MODEL_LABELS.map(l => l === BEST_MODEL ? c.test : 'rgba(42,47,74,0.5)');

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: MODEL_LABELS,
      datasets: [
        {
          label: 'CV F1 (%)',
          data: CV_F1_DATA,
          backgroundColor: bgCV,
          borderColor: c.cv,
          borderWidth: 0,
          borderRadius: 4,
        },
        {
          label: 'Test F1 (%)',
          data: TEST_F1_DATA,
          backgroundColor: bgTest,
          borderColor: 'transparent',
          borderWidth: 0,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#161c30',
          borderColor: '#2a2f4a',
          borderWidth: 1,
          titleColor: '#e8eaf0',
          bodyColor: '#5a6080',
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#5a6080', font: { size: 11 }, maxRotation: 30 },
          grid:  { color: 'rgba(42,47,74,0.5)' },
        },
        y: {
          min: 40,
          max: 100,
          ticks: {
            color: '#5a6080',
            font: { size: 11 },
            callback: v => v + '%',
          },
          grid: { color: 'rgba(42,47,74,0.5)' },
        },
      },
    },
  });

  // Custom legend below chart
  const wrap = ctx.parentElement;
  const leg  = document.createElement('div');
  leg.style.cssText = 'display:flex;gap:16px;justify-content:center;margin-top:10px;font-size:12px;color:#5a6080';
  leg.innerHTML = `
    <span style="display:flex;align-items:center;gap:5px">
      <span style="width:10px;height:10px;border-radius:2px;background:${c.cv};display:inline-block"></span> CV F1
    </span>
    <span style="display:flex;align-items:center;gap:5px">
      <span style="width:10px;height:10px;border-radius:2px;background:${c.test};display:inline-block"></span> Test F1
    </span>
    <span style="display:flex;align-items:center;gap:5px">
      <span style="width:10px;height:10px;border-radius:2px;background:#2a2f4a;display:inline-block"></span> Other models
    </span>`;
  wrap.appendChild(leg);
}
