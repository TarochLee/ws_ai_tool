const fileEl = document.getElementById('file');
const btnUpload = document.getElementById('btnUpload');
const bar = document.getElementById('bar');
const statusText = document.getElementById('statusText');
const out = document.getElementById('out');
const imgPreview = document.getElementById('imgPreview');

let pastedBlob = null;

function setProgress(p) {
  const v = Math.max(0, Math.min(100, p));
  bar.style.width = v + '%';
}

function setStatus(s) {
  statusText.textContent = s;
}

// 粘贴剪贴板图片
window.addEventListener('paste', (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;

  for (const it of items) {
    if (it.type.startsWith('image/')) {
      pastedBlob = it.getAsFile();
      const url = URL.createObjectURL(pastedBlob);
      imgPreview.src = url;
      imgPreview.style.display = 'block';
      setStatus('已粘贴图片，可点击“上传并开始”');
      return;
    }
  }
});

function pickFile() {
  if (fileEl.files && fileEl.files[0]) return fileEl.files[0];
  if (pastedBlob) return pastedBlob;
  return null;
}

async function createJob(file) {
  const fd = new FormData();
  fd.append('file', file, file.name || 'paste.png');

  const r = await fetch('/api/job', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  return j.job_id;
}

function phaseText(phase) {
  if (phase === 'queued') return '排队中';
  if (phase === 'ocr') return 'OCR 识别中';
  if (phase === 'llm') return '模型生成中';
  if (phase === 'done') return '完成';
  if (phase === 'error') return '失败';
  return phase;
}

function listenJob(jobId) {
  out.textContent = '';
  setProgress(1);
  setStatus('任务已创建，开始处理…');

  const es = new EventSource(`/api/events?job_id=${encodeURIComponent(jobId)}`);

  es.onmessage = (ev) => {
    const st = JSON.parse(ev.data);

    if (st.error && st.phase === 'error') {
      setProgress(100);
      setStatus('失败：' + st.error);
      es.close();
      return;
    }

    setProgress(st.progress || 0);
    setStatus(`${phaseText(st.phase)}（${st.progress}%）`);

    // 直接显示累计结果
    if (typeof st.result_text === 'string') {
      out.textContent = st.result_text;
    }

    if (st.phase === 'done' || st.phase === 'error') {
      es.close();
    }
  };

  es.addEventListener('ping', () => {
    // 心跳，不做处理
  });

  es.onerror = () => {
    setStatus('连接中断，尝试刷新页面或重试');
    es.close();
  };
}

btnUpload.addEventListener('click', async () => {
  const f = pickFile();
  if (!f) {
    alert('请选择或粘贴一张图片');
    return;
  }

  // 预览
  const url = URL.createObjectURL(f);
  imgPreview.src = url;
  imgPreview.style.display = 'block';

  try {
    setStatus('上传中…');
    setProgress(0);

    const jobId = await createJob(f);
    listenJob(jobId);
  } catch (e) {
    setStatus('失败：' + e.message);
  }
});