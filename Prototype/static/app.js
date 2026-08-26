// ── State ──────────────────────────────────────────────
const state = {
  activeVideoUrl: null,
  activeVideoId: null,
  indexedVideos: [],
};

// ── DOM Refs ────────────────────────────────────────────
const uploadZone    = document.getElementById('upload-zone');
const fileInput     = document.getElementById('file-input');
const browseBtn     = document.getElementById('browse-btn');
const progressWrap  = document.getElementById('progress-wrap');
const progressBar   = document.getElementById('progress-bar');
const progressLabel = document.getElementById('progress-label');
const progressPct   = document.getElementById('progress-pct');
const progressStatus= document.getElementById('progress-status');
const indexedVideos = document.getElementById('indexed-videos');
const videoChips    = document.getElementById('video-chips');
const searchInput   = document.getElementById('search-input');
const topKSelect    = document.getElementById('top-k-select');
const searchBtn     = document.getElementById('search-btn');
const searchSpinner = document.getElementById('search-spinner');
const resultsSection= document.getElementById('results-section');
const resultsGrid   = document.getElementById('results-grid');
const resultsTitle  = document.getElementById('results-title');
const resultsCount  = document.getElementById('results-count');
const videoPlayer   = document.getElementById('video-player');
const videoSource   = document.getElementById('video-source');
const playerFilename= document.getElementById('player-filename');
const currentTs     = document.getElementById('current-timestamp');
const healthBadge   = document.getElementById('health-badge');

// ── Startup ─────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadIndexedVideos();
});

// ── Health Check ────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error();
    const data = await res.json();
    healthBadge.textContent = `${data.device} · ${data.total_indexed_frames} frames indexed`;
    healthBadge.classList.add('online');
  } catch {
    healthBadge.textContent = 'Server offline';
  }
}

// ── Load Indexed Videos ──────────────────────────────────
async function loadIndexedVideos() {
  try {
    const res = await fetch('/videos-list');
    if (!res.ok) return;
    const data = await res.json();
    state.indexedVideos = data.videos;
    renderVideoChips(data.videos);
  } catch {
    // silently fail
  }
}

function renderVideoChips(videos) {
  if (!videos || videos.length === 0) {
    indexedVideos.style.display = 'none';
    return;
  }
  indexedVideos.style.display = 'block';
  videoChips.innerHTML = '';
  videos.forEach(v => {
    const chip = document.createElement('div');
    chip.className = 'video-chip';
    chip.innerHTML = `🎞 ${truncate(v.video_filename, 30)}`;
    chip.title = v.video_filename;
    chip.addEventListener('click', () => loadPlayerVideo(v.video_url, v.video_filename, v.video_id));
    videoChips.appendChild(chip);
  });
}

// ── Drag & Drop Upload ───────────────────────────────────
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleUpload(file);
});
uploadZone.addEventListener('click', e => {
  if (e.target === browseBtn || browseBtn.contains(e.target)) return;
  fileInput.click();
});
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleUpload(fileInput.files[0]);
});

// ── Upload Handler ───────────────────────────────────────
async function handleUpload(file) {
  const allowed = ['video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 'video/x-msvideo'];
  if (!allowed.includes(file.type)) {
    showToast('❌ Unsupported file type. Please upload MP4, AVI, or MOV.', 'error');
    return;
  }

  progressWrap.style.display = 'block';
  progressLabel.textContent = `Uploading ${file.name}`;
  progressStatus.textContent = 'Sending file to server...';
  setProgress(10);
  browseBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', file);

  try {
    // Simulate upload progress since XHR doesn't give processing progress
    const progressInterval = simulateProgress(10, 40, 3000);

    const res = await fetch('/upload', { method: 'POST', body: formData });
    clearInterval(progressInterval);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }

    setProgress(70);
    progressStatus.textContent = 'Extracting frames & generating CLIP embeddings...';

    const data = await res.json();

    setProgress(100);
    progressLabel.textContent = '✅ Indexed Successfully!';
    progressStatus.textContent = `${data.frames_indexed} frames indexed from "${file.name}"`;

    // Load video into player
    loadPlayerVideo(data.video_url, data.filename, data.video_id);

    // Refresh video list
    await loadIndexedVideos();
    await checkHealth();

  } catch (err) {
    progressLabel.textContent = '❌ Upload failed';
    progressStatus.textContent = err.message;
    setProgress(0);
  } finally {
    browseBtn.disabled = false;
    setTimeout(() => { progressWrap.style.display = 'none'; }, 5000);
    fileInput.value = '';
  }
}

function setProgress(pct) {
  progressBar.style.width = `${pct}%`;
  progressPct.textContent = `${pct}%`;
}

function simulateProgress(from, to, duration) {
  const steps = 20;
  const increment = (to - from) / steps;
  const interval = duration / steps;
  let current = from;
  return setInterval(() => {
    current = Math.min(current + increment, to);
    setProgress(Math.round(current));
  }, interval);
}

// ── Video Player ─────────────────────────────────────────
function loadPlayerVideo(url, filename, videoId) {
  state.activeVideoUrl = url;
  state.activeVideoId = videoId;
  videoSource.src = url;
  videoPlayer.load();
  playerFilename.textContent = filename;
  resultsSection.style.display = 'block';
  currentTs.textContent = '0s';
}

function seekToTimestamp(sec) {
  videoPlayer.currentTime = sec;
  videoPlayer.play();
  currentTs.textContent = formatTime(sec);
}

// ── Search ───────────────────────────────────────────────
searchBtn.addEventListener('click', runSearch);
searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });

async function runSearch() {
  const query = searchInput.value.trim();
  if (!query) { showToast('⚠️ Please type a search query.', 'warn'); return; }

  searchBtn.disabled = true;
  searchSpinner.style.display = 'flex';
  resultsGrid.innerHTML = '';

  const body = {
    query,
    top_k: parseInt(topKSelect.value),
    video_id: state.activeVideoId || null,
  };

  try {
    const res = await fetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Search failed');
    }

    const data = await res.json();
    renderResults(data.query, data.results);
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    showToast(`❌ ${err.message}`, 'error');
  } finally {
    searchBtn.disabled = false;
    searchSpinner.style.display = 'none';
  }
}

// ── Render Results ───────────────────────────────────────
function renderResults(query, results) {
  resultsTitle.textContent = `"${query}"`;
  resultsCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''}`;
  resultsGrid.innerHTML = '';

  if (results.length === 0) {
    resultsGrid.innerHTML = `<p style="color:var(--text-muted);font-size:0.9rem;padding:12px 0">No matching moments found. Try a different query.</p>`;
    return;
  }

  results.forEach((r, i) => {
    // Auto-load first result's video if nothing is loaded
    if (i === 0 && !state.activeVideoUrl) {
      loadPlayerVideo(r.video_url, r.video_filename, r.video_id);
    }

    const card = document.createElement('div');
    card.className = 'result-card';
    card.style.animationDelay = `${i * 60}ms`;

    const scorePct = Math.round(r.similarity_score * 100);
    card.innerHTML = `
      <div class="card-rank">${i + 1}</div>
      <div class="card-info">
        <div class="card-timestamp">⏱ ${formatTime(r.timestamp_sec)}</div>
        <div class="card-filename">${truncate(r.video_filename, 35)}</div>
      </div>
      <div class="card-score-wrap">
        <div class="card-score">${scorePct}% match</div>
        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width:${scorePct}%"></div>
        </div>
      </div>
    `;

    card.addEventListener('click', () => {
      // Switch video if needed
      if (r.video_url !== state.activeVideoUrl) {
        loadPlayerVideo(r.video_url, r.video_filename, r.video_id);
        // Wait for video to load then seek
        videoPlayer.addEventListener('loadedmetadata', () => seekToTimestamp(r.timestamp_sec), { once: true });
      } else {
        seekToTimestamp(r.timestamp_sec);
      }
    });

    resultsGrid.appendChild(card);
  });
}

// ── Utilities ────────────────────────────────────────────
function formatTime(sec) {
  const s = Math.floor(sec);
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return m > 0 ? `${m}m ${rem}s` : `${s}s`;
}

function truncate(str, max) {
  return str.length <= max ? str : '...' + str.slice(str.length - max + 3);
}

function showToast(msg, type = 'info') {
  const existing = document.getElementById('toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id = 'toast';
  const colors = { info: '#7c6ff7', error: '#f43f5e', warn: '#f59e0b' };
  Object.assign(toast.style, {
    position: 'fixed', bottom: '28px', left: '50%', transform: 'translateX(-50%)',
    background: '#1a1d2e', border: `1px solid ${colors[type] || colors.info}`,
    color: '#e8eaf0', padding: '12px 24px', borderRadius: '12px',
    fontSize: '0.88rem', fontWeight: '500', zIndex: '9999',
    boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
    animation: 'fadeSlide 0.3s ease',
  });
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}
