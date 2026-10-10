(() => {
  'use strict';
  const labels = { light: 'คั่วอ่อน', medium: 'คั่วกลาง', dark: 'คั่วเข้ม' };
  const tags = { light: 'L', medium: 'M', dark: 'D' };
  const colors = { light: '#0879c9', medium: '#cf570b', dark: '#a13ab3' };
  const form = document.querySelector('#predict-form'), input = document.querySelector('#image-input');
  const button = document.querySelector('#submit-button'), progress = document.querySelector('#progress');
  const resultBox = document.querySelector('#result'), previewWrap = document.querySelector('#preview-wrap');
  const canvas = document.querySelector('#preview'), fileInfo = document.querySelector('#file-info');
  const overlayButton = document.querySelector('#overlay-toggle'), legend = document.querySelector('#overlay-legend');
  const maxUploadBytes = 32 * 1024 * 1024;
  let selectedFile = null, displayImage = null, lastResult = null, overlayEnabled = true, busy = false;
  let selectionVersion = 0, e2eNode = null;
  const variantPicker = document.querySelector('#variant-picker'), variantOptions = document.querySelector('#variant-options');
  const variantLabels = {};

  async function refreshCounts() {
    try {
      const response = await fetch('/api/market', { cache: 'no-store' });
      if (!response.ok) return;
      const snapshot = await response.json();
      for (const roast of Object.keys(labels)) {
        const counter = document.querySelector('#roast-count-' + roast);
        if (counter) counter.textContent = snapshot.counts[roast] ?? 0;
      }
      const total = document.querySelector('#counts-total');
      if (total) {
        const estimated = Number(snapshot.estimated_results) || 0;
        total.textContent = 'เมล็ดทั้งหมด ' + (snapshot.bean_total ?? snapshot.total ?? 0) +
          ' เมล็ด' + (estimated ? ' · มีค่าประมาณ ' + estimated + ' รูป' : '');
      }
    } catch (_error) { /* Keep the last snapshot while reconnecting. */ }
  }

  if (!form || !input || !button || !progress || !resultBox) {
    refreshCounts(); window.setInterval(refreshCounts, 3000); return;
  }

  function element(tag, text, className) {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  // Two (or more) model variants declared by the ML model card; hidden when the model has none.
  async function loadVariants() {
    if (!variantPicker || !variantOptions) return;
    try {
      const response = await fetch('/health', { cache: 'no-store' });
      const backend = ((await response.json()).ml || {}).backend || {};
      const variants = backend.variants || {}, names = Object.keys(variants);
      if (names.length < 2) return;
      for (const name of names) {
        const label = document.createElement('label'), radio = document.createElement('input');
        radio.type = 'radio'; radio.name = 'variant'; radio.value = name; radio.checked = name === backend.default_variant;
        variantLabels[name] = String((variants[name] || {}).label_th || name);
        label.append(radio, element('span', variantLabels[name]));
        variantOptions.append(label);
        radio.addEventListener('change', () => { if (lastResult && selectedFile && !busy) form.requestSubmit(); });
      }
      variantPicker.hidden = false;
    } catch (_error) { /* No picker: the server answers with its default variant. */ }
  }
  function showError(message) {
    resultBox.hidden = false;
    resultBox.className = 'result error-card';
    resultBox.replaceChildren(element('h2', 'ส่งรูปไม่สำเร็จ'), element('p', message));
  }
  async function readImage(blob) {
    const url = URL.createObjectURL(blob), image = new Image();
    try { image.src = url; await image.decode(); return image; }
    finally { URL.revokeObjectURL(url); }
  }
  async function compressImage(file) {
    const image = await readImage(file);
    const scale = Math.min(1, 1600 / Math.max(image.naturalWidth, image.naturalHeight));
    const work = document.createElement('canvas');
    work.width = Math.max(1, Math.round(image.naturalWidth * scale));
    work.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const ctx = work.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('เบราว์เซอร์ไม่รองรับการย่อรูป');
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, work.width, work.height);
    ctx.drawImage(image, 0, 0, work.width, work.height);
    return new Promise((resolve, reject) => work.toBlob(
      blob => blob ? resolve(blob) : reject(new Error('แปลงรูปไม่สำเร็จ')), 'image/jpeg', 0.84));
  }
  function validBox(box) {
    return Array.isArray(box) && box.length === 4 && box.every(Number.isFinite) && box[2] > 0 && box[3] > 0;
  }
  function redraw() {
    const started = performance.now();
    if (!displayImage) return 0;
    const cssWidth = Math.max(1, canvas.getBoundingClientRect().width || 600);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(cssWidth * ratio));
    const height = Math.max(1, Math.round(width * displayImage.naturalHeight / displayImage.naturalWidth));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('เบราว์เซอร์ไม่รองรับการแสดงรูป');
    ctx.drawImage(displayImage, 0, 0, width, height);
    const size = lastResult?.image_size, beans = lastResult?.beans;
    const available = Array.isArray(size) && size.length === 2 && size.every(v => Number.isFinite(v) && v > 0)
      && Array.isArray(beans) && beans.length > 0;
    overlayButton.disabled = !available; legend.hidden = !available;
    if (!available || !overlayEnabled) return performance.now() - started;
    const sx = width / size[0], sy = height / size[1];
    ctx.lineWidth = Math.max(1.5, 1.5 * ratio);
    // Three batched paths, one canvas, no per-bean DOM/layout operations.
    for (const roast of Object.keys(labels)) {
      ctx.strokeStyle = colors[roast]; ctx.beginPath();
      for (const bean of beans) {
        if (bean.label !== roast || !validBox(bean.bbox)) continue;
        const [x, y, w, h] = bean.bbox;
        ctx.rect(x * sx, y * sy, w * sx, h * sy);
      }
      ctx.stroke();
    }
    ctx.font = 'bold ' + Math.round(9 * ratio) + 'px system-ui';
    ctx.textBaseline = 'top';
    const tagW = 12 * ratio, tagH = 11 * ratio;
    for (const bean of beans) {
      if (!colors[bean.label] || !validBox(bean.bbox)) continue;
      const x = Math.max(0, Math.min(width - tagW, bean.bbox[0] * sx));
      // Put small labels above the box so dense scenes retain the bean image.
      const y = Math.max(0, Math.min(height - tagH, bean.bbox[1] * sy - tagH - ratio));
      ctx.fillStyle = colors[bean.label]; ctx.fillRect(x, y, tagW, tagH);
      ctx.fillStyle = '#fff'; ctx.fillText(tags[bean.label], x + 3 * ratio, y + ratio);
    }
    return performance.now() - started;
  }
  function renderResult(data) {
    resultBox.hidden = false;
    resultBox.className = 'result ' + (data.status === 'ok' ? 'success-card' : 'notice-card');
    resultBox.replaceChildren(
      element('p', data.status === 'ok' ? 'ผลวิเคราะห์' : data.status === 'low_confidence' ? 'ผลยังไม่ชัดเจน' : 'วิเคราะห์ไม่ได้', 'result-kicker'),
      element('h2', data.label ? (data.label_th || labels[data.label] || data.label) : 'ลองส่งรูปอีกครั้ง'),
      element('p', data.message_th || 'ระบบไม่สามารถประมวลผลรูปนี้ได้'));
    if (data.counts && Number.isInteger(data.n_beans)) {
      resultBox.append(element('p', 'นับได้ ' + data.n_beans + ' เมล็ด', 'bean-total'));
      const grid = document.createElement('div'); grid.className = 'count-grid image-counts';
      for (const roast of Object.keys(labels)) {
        const n = Number(data.counts[roast]) || 0, cell = document.createElement('div');
        const percent = data.n_beans ? 100 * n / data.n_beans : 0;
        cell.append(element('strong', String(n)), element('span', labels[roast] + ' · ' + percent.toFixed(1) + '%'));
        grid.append(cell);
      }
      resultBox.append(grid);
    } else resultBox.append(element('p', 'โมเดลนี้ยังไม่มีข้อมูลการนับรายเมล็ด', 'hint'));
    const warnings = Array.isArray(data.warnings) ? data.warnings : [], badges = document.createElement('div');
    badges.className = 'badges';
    for (const [key, text] of [['mixed_roast','หลายระดับคั่ว'],['count_visible_only','นับเฉพาะที่มองเห็น'],
      ['bean_count_estimated','ค่าประมาณ'],['no_beans_detected','ไม่พบเมล็ด']]) {
      if (warnings.includes(key)) badges.append(element('span', text, 'badge'));
    }
    resultBox.append(badges);
    if (Number.isFinite(data.confidence)) resultBox.append(element('p', 'ความมั่นใจ ' + Math.round(data.confidence * 100) + '%', 'confidence'));
    const timings = data.timing_ms || {};
    resultBox.append(element('p', 'เวลา ML ' + (Number.isFinite(timings.ml) ? timings.ml.toFixed(1) + ' ms' : '—'), 'timing'));
    const details = document.createElement('details'); details.className = 'stage-times';
    details.append(element('summary', 'เวลาประมวลผลแต่ละขั้น'));
    for (const [key, name] of Object.entries({ decode: 'เปิดรูป', WB: 'ปรับสี', segment: 'แยกพื้นหลัง', count: 'นับเมล็ด',
      features: 'อ่านสีเมล็ด', classify: 'จำแนก', group: 'จัดกลุ่ม', fallback: 'ผลสำรอง', total: 'รวมฝั่ง ML' })) {
      if (Number.isFinite(timings[key])) details.append(element('p', name + ' ' + timings[key].toFixed(1) + ' ms'));
    }
    resultBox.append(details);
    const variantName = typeof data.model === 'string' && data.model.includes('@') ? data.model.split('@').pop() : '';
    if (variantName) resultBox.append(element('p', 'ตัวเลือก: ' + (variantLabels[variantName] || variantName), 'timing'));
    e2eNode = element('p', '', 'timing'); e2eNode.id = 'e2e-timing'; resultBox.append(e2eNode);
  }
  overlayButton.addEventListener('click', () => {
    overlayEnabled = !overlayEnabled;
    overlayButton.setAttribute('aria-pressed', String(overlayEnabled));
    overlayButton.textContent = overlayEnabled ? 'ซ่อนกรอบ' : 'แสดงกรอบ'; redraw();
  });
  input.addEventListener('change', async () => {
    const version = ++selectionVersion;
    selectedFile = input.files?.[0] || null; button.disabled = !selectedFile;
    resultBox.hidden = true; progress.textContent = ''; lastResult = null;
    overlayButton.disabled = true; legend.hidden = true; displayImage = null;
    previewWrap.hidden = !selectedFile;
    if (!selectedFile) return;
    fileInfo.textContent = selectedFile.name + ' · ' + (selectedFile.size / 1024 / 1024).toFixed(1) + ' MB';
    try {
      const image = await readImage(selectedFile);
      if (version !== selectionVersion || busy) return;
      displayImage = image; canvas.hidden = false; redraw();
    } catch (_error) {
      if (version !== selectionVersion) return;
      canvas.hidden = true;
      fileInfo.textContent += ' · เบราว์เซอร์เปิดตัวอย่างไม่ได้ แต่ยังส่งไฟล์ต้นฉบับได้';
    }
  });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (!selectedFile || busy) return;
    const file = selectedFile; busy = true; button.disabled = true; input.disabled = true; resultBox.hidden = true;
    const started = performance.now(); // resize, upload, ML, JSON, canvas paint
    const controller = new AbortController(), timeout = window.setTimeout(() => controller.abort(), 20000);
    try {
      if (file.size > maxUploadBytes) throw new Error('ไฟล์ใหญ่เกิน 32 MiB กรุณาเลือกรูปที่เล็กกว่านี้');
      progress.textContent = 'กำลังย่อรูปและวิเคราะห์…';
      let uploadFile = file, uploadName = file.name || 'coffee-image';
      try { uploadFile = await compressImage(file); uploadName = 'coffee.jpg'; }
      catch (_error) { progress.textContent = 'กำลังส่งไฟล์ต้นฉบับ…'; }
      // Display the exact oriented upload; boxes scale from decoded image_size.
      try { displayImage = await readImage(uploadFile); canvas.hidden = false; }
      catch (_error) { displayImage = null; canvas.hidden = true; }
      const body = new FormData(); body.append('image', uploadFile, uploadName);
      const chosenVariant = form.querySelector('input[name="variant"]:checked');
      if (chosenVariant) body.append('variant', chosenVariant.value);
      const response = await fetch('/api/predict', { method: 'POST', body, signal: controller.signal });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || 'ส่งรูปไม่สำเร็จ (' + response.status + ')');
      progress.textContent = ''; lastResult = data; renderResult(data);
      const drawMs = redraw();
      // The second RAF records after the frame containing the canvas paint.
      // A hidden/background tab never gets animation frames; do not let that block the form forever.
      await Promise.race([
        new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))),
        new Promise(resolve => window.setTimeout(resolve, 250)),
      ]);
      const e2eMs = performance.now() - started;
      e2eNode.textContent = 'กดส่งรูป → วาดเสร็จ ' + e2eMs.toFixed(1) + ' ms';
      document.dispatchEvent(new CustomEvent('topgun:prediction-rendered', { detail: {
        e2e_ms: e2eMs, draw_ms: drawMs, upload_bytes: uploadFile.size, n_beans: data.n_beans,
        counts: data.counts, warnings: data.warnings, schema_version: data.schema_version,
        status: data.status, timing_ms: data.timing_ms, image_size: data.image_size,
        displayed_size: [canvas.width, canvas.height]
      } }));
      refreshCounts();
    } catch (error) {
      progress.textContent = '';
      showError(error.name === 'AbortError' ? 'ส่งรูปใช้เวลานานเกินไป กรุณาลองใหม่' : error.message || 'เกิดข้อผิดพลาด กรุณาลองใหม่');
      document.dispatchEvent(new CustomEvent('topgun:prediction-error', { detail: { message: error.message } }));
    } finally {
      window.clearTimeout(timeout); busy = false; input.disabled = false; button.disabled = !selectedFile;
    }
  });
  if (window.ResizeObserver) new ResizeObserver(() => { if (displayImage) redraw(); }).observe(previewWrap);
  loadVariants();
  refreshCounts(); window.setInterval(refreshCounts, 3000);
})();
