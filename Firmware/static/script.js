(() => {
  const form = document.querySelector('#predict-form');
  const input = document.querySelector('#image-input');
  const button = document.querySelector('#submit-button');
  const progress = document.querySelector('#progress');
  const resultBox = document.querySelector('#result');
  const previewWrap = document.querySelector('#preview-wrap');
  const preview = document.querySelector('#preview');
  const fileInfo = document.querySelector('#file-info');
  const labels = { light: 'คั่วอ่อน', medium: 'คั่วกลาง', dark: 'คั่วเข้ม' };
  const maxUploadBytes = 32 * 1024 * 1024;

  async function refreshCounts() {
    try {
      const response = await fetch('/api/market', { cache: 'no-store' });
      if (!response.ok) return;
      const snapshot = await response.json();
      for (const roast of ['light', 'medium', 'dark']) {
        const counter = document.querySelector(`#roast-count-${roast}`);
        if (counter) counter.textContent = snapshot.counts[roast] ?? 0;
      }
      const total = document.querySelector('#counts-total');
      if (total) total.textContent = `เมล็ดทั้งหมด ${snapshot.bean_total ?? snapshot.total ?? 0} เมล็ด`;
    } catch (_error) {
      // Counts remain at their last rendered value while the local service reconnects.
    }
  }

  let selectedFile = null;
  let previewUrl = null;

  function showError(message) {
    resultBox.hidden = false;
    resultBox.className = 'result error-card';
    resultBox.replaceChildren(Object.assign(document.createElement('h2'), { textContent: 'ส่งรูปไม่สำเร็จ' }));
    const p = document.createElement('p');
    p.textContent = message;
    resultBox.append(p);
  }

  function compressImage(file, maxSide = 1600, quality = 0.84) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const image = new Image();
      image.onload = () => {
        const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
        canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
        const ctx = canvas.getContext('2d', { alpha: false });
        if (!ctx) {
          URL.revokeObjectURL(url);
          reject(new Error('เบราว์เซอร์ไม่รองรับการย่อรูป'));
          return;
        }
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(blob => {
          URL.revokeObjectURL(url);
          if (!blob) reject(new Error('แปลงรูปไม่สำเร็จ กรุณาเลือกรูป JPG หรือ PNG'));
          else resolve(blob);
        }, 'image/jpeg', quality);
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('มือถือเปิดรูปนี้เพื่อย่อไม่ได้ กรุณาเลือกรูป JPG หรือ PNG'));
      };
      image.src = url;
    });
  }

  function renderResult(data, elapsedMs) {
    resultBox.hidden = false;
    const good = data.status === 'ok';
    resultBox.className = `result ${good ? 'success-card' : 'notice-card'}`;
    resultBox.replaceChildren();
    const title = document.createElement('p');
    title.className = 'result-kicker';
    title.textContent = data.status === 'ok' ? 'ผลวิเคราะห์' : data.status === 'low_confidence' ? 'ผลยังไม่ชัดเจน' : 'วิเคราะห์ไม่ได้';
    resultBox.append(title);
    const heading = document.createElement('h2');
    heading.textContent = data.label ? (data.label_th || labels[data.label] || data.label) : 'ลองส่งรูปอีกครั้ง';
    resultBox.append(heading);
    const message = document.createElement('p');
    message.textContent = data.message_th || 'ระบบไม่สามารถประมวลผลรูปนี้ได้';
    resultBox.append(message);
    if (data.confidence !== null && data.confidence !== undefined) {
      const confidence = document.createElement('p');
      confidence.className = 'confidence';
      confidence.textContent = `ความมั่นใจ ${Math.round(data.confidence * 100)}%`;
      resultBox.append(confidence);
    }
    const timing = document.createElement('p');
    timing.className = 'timing';
    timing.textContent = `ตั้งแต่เริ่มส่งรูปจนแสดงผล ${elapsedMs.toFixed(0)} ms`;
    resultBox.append(timing);
    if (Array.isArray(data.beans) && data.beans.length) {
      const beanHeading = document.createElement('h3');
      beanHeading.textContent = `ผลรายเมล็ด (${data.beans.length} เมล็ด)`;
      resultBox.append(beanHeading);
      const beanList = document.createElement('ol');
      data.beans.forEach((bean, index) => {
        const item = document.createElement('li');
        const label = labels[bean.label] || bean.label || 'ยังระบุไม่ได้';
        const confidence = Number.isFinite(bean.conf) ? ` · ความมั่นใจ ${Math.round(bean.conf * 100)}%` : '';
        item.textContent = `เมล็ดที่ ${index + 1}: ${label}${confidence}`;
        beanList.append(item);
      });
      resultBox.append(beanList);
    }
  }

  if (!form || !input || !button || !progress || !resultBox) {
    refreshCounts();
    window.setInterval(refreshCounts, 3000);
    return;
  }

  input.addEventListener('change', () => {
    selectedFile = input.files && input.files[0] ? input.files[0] : null;
    button.disabled = !selectedFile;
    resultBox.hidden = true;
    progress.textContent = '';
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (!selectedFile) {
      previewWrap.hidden = true;
      return;
    }
    previewUrl = URL.createObjectURL(selectedFile);
    preview.src = previewUrl;
    fileInfo.textContent = `${selectedFile.name} · ${(selectedFile.size / 1024 / 1024).toFixed(1)} MB`;
    previewWrap.hidden = false;
  });

  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (!selectedFile) return;
    button.disabled = true;
    resultBox.hidden = true;
    if (selectedFile.size > maxUploadBytes) {
      showError('ไฟล์ใหญ่เกิน 32 MiB กรุณาเลือกรูปที่เล็กกว่านี้');
      button.disabled = false;
      return;
    }
    progress.textContent = 'กำลังย่อรูปและวิเคราะห์…';
    const started = performance.now();
    try {
      let uploadFile;
      let uploadName;
      try {
        uploadFile = await compressImage(selectedFile);
        uploadName = 'coffee.jpg';
      } catch (_resizeError) {
        // HEIC and some mobile formats cannot be decoded by the browser; ML can decode the original.
        uploadFile = selectedFile;
        uploadName = selectedFile.name || 'coffee-image';
        progress.textContent = 'เบราว์เซอร์ย่อรูปไม่ได้ กำลังส่งไฟล์ต้นฉบับ (ไม่เกิน 32 MB)…';
      }
      const body = new FormData();
      body.append('image', uploadFile, uploadName);
      const response = await fetch('/api/predict', { method: 'POST', body });
      const data = await response.json().catch(() => ({}));
      const elapsedMs = performance.now() - started;
      if (!response.ok) throw new Error(data.error || `ส่งรูปไม่สำเร็จ (${response.status})`);
      progress.textContent = '';
      renderResult(data, elapsedMs);
      refreshCounts();
    } catch (error) {
      progress.textContent = '';
      showError(error.message || 'เกิดข้อผิดพลาด กรุณาลองใหม่');
    } finally {
      button.disabled = !selectedFile;
    }
  });

  refreshCounts();
  window.setInterval(refreshCounts, 3000);
})();
