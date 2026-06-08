// ── Live Clock ──────────────────────────────────────────────────────────
function updateClock() {
  const el = document.getElementById('liveClock');
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleString('en-IN', {
    dateStyle: 'medium', timeStyle: 'short'
  });
}
updateClock();
setInterval(updateClock, 30000);

// ── Animate stat numbers ────────────────────────────────────────────────
function animateNumber(el, target, suffix = '', decimals = 0) {
  const duration = 1200;
  const start = performance.now();
  const from = 0;
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const value = from + (target - from) * ease;
    el.textContent = decimals > 0 ? value.toFixed(decimals) + suffix : Math.floor(value) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// Run on load for stat numbers
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-count]').forEach(el => {
    const target   = parseFloat(el.dataset.count);
    const suffix   = el.dataset.suffix || '';
    const decimals = parseInt(el.dataset.decimals || '0');
    animateNumber(el, target, suffix, decimals);
  });

  // Animate progress bars
  document.querySelectorAll('.progress-fill[data-width]').forEach(el => {
    const w = el.dataset.width;
    setTimeout(() => { el.style.width = w; }, 100);
  });
});

// ── CSV Upload drag-drop ─────────────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
if (uploadZone) {
  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      const input = document.getElementById('csvInput');
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      document.getElementById('fileName').textContent = '📄 ' + file.name;
    }
  });
  uploadZone.addEventListener('click', () => document.getElementById('csvInput').click());
  document.getElementById('csvInput')?.addEventListener('change', function() {
    if (this.files[0]) {
      document.getElementById('fileName').textContent = '📄 ' + this.files[0].name;
    }
  });
}

// ── Smooth scroll ────────────────────────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth' }); }
  });
});

// ── Form range inputs live value ─────────────────────────────────────────
document.querySelectorAll('input[type=range]').forEach(r => {
  const display = document.getElementById(r.id + '_val');
  if (display) {
    display.textContent = r.value;
    r.addEventListener('input', () => display.textContent = r.value);
  }
});
