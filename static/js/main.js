/**
 * SignBridge AI - Main Application Controller
 * Handles direct browser webcam streaming (WebRTC / getUserMedia),
 * backend AI inference, real-time skeleton canvas rendering, and speech controls.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  // Video & Canvas
  const video = document.getElementById('webcam-video');
  const landmarkCanvas = document.getElementById('landmark-canvas');
  const landmarkCtx = landmarkCanvas ? landmarkCanvas.getContext('2d') : null;
  const cameraPrompt = document.getElementById('camera-prompt');
  const btnRequestCam = document.getElementById('btn-request-cam');
  const camStatus = document.getElementById('cam-status');

  // HUD & State
  const hudSign = document.getElementById('hud-sign');
  const hudConf = document.getElementById('hud-conf');
  const signAvatar = document.getElementById('sign-avatar');
  const currentSignName = document.getElementById('current-sign-name');
  const currentSignStatus = document.getElementById('current-sign-status');
  const confPercentage = document.getElementById('conf-percentage');
  const confProgressBar = document.getElementById('conf-progress-bar');
  const holdPercentage = document.getElementById('hold-percentage');
  const holdProgressBar = document.getElementById('hold-progress-bar');
  const sentenceDisplay = document.getElementById('sentence-display');

  // Controls
  const btnSpeak = document.getElementById('btn-speak');
  const btnSpace = document.getElementById('btn-space');
  const btnBackspace = document.getElementById('btn-backspace');
  const btnClear = document.getElementById('btn-clear');
  const btnCopySentence = document.getElementById('btn-copy-sentence');
  const toggleSkeleton = document.getElementById('toggle-skeleton');
  const toggleMirror = document.getElementById('toggle-mirror');

  // Studio Elements
  const btnStartRecord = document.getElementById('btn-start-record');
  const recordGestureName = document.getElementById('record-gesture-name');
  const recordSampleCount = document.getElementById('record-sample-count');
  const recordProgressBox = document.getElementById('record-progress-box');
  const recordCountdown = document.getElementById('record-countdown');
  const recordSamplesLabel = document.getElementById('record-samples-label');
  const recordProgressFill = document.getElementById('record-progress-fill');

  // Model & Analytics
  const btnRetrain = document.getElementById('btn-retrain');
  const retrainStatus = document.getElementById('retrain-status');
  const statAccuracy = document.getElementById('stat-accuracy');
  const statClassesCount = document.getElementById('stat-classes-count');
  const statSamplesCount = document.getElementById('stat-samples-count');
  const classesList = document.getElementById('classes-list');
  const guideCards = document.getElementById('guide-cards');

  // Offscreen canvas for frame capture
  const captureCanvas = document.createElement('canvas');
  captureCanvas.width = 480;
  captureCanvas.height = 360;
  const captureCtx = captureCanvas.getContext('2d');

  let isStreaming = false;
  let isProcessing = false;
  let showSkeleton = true;

  // Hand Landmark Connections (21 joints)
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // Index
    [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
    [0, 13], [13, 14], [14, 15], [15, 16],// Ring
    [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
    [5, 9], [9, 13], [13, 17]             // Knuckles
  ];

  const FINGER_COLORS = [
    '#f59e0b', // Thumb
    '#10b981', // Index
    '#00e5ff', // Middle
    '#3b82f6', // Ring
    '#ec4899'  // Pinky
  ];

  const EMOJI_MAP = {
    'A': '✊', 'B': '✋', 'C': '🫳', 'D': '☝️', 'L': '👆',
    'O': '👌', 'V': '✌️', 'Y': '🤙', 'Hello': '👋',
    'Thank You': '🙏', 'Yes': '✊', 'No': '🤏', 'Help': '🆘', 'I Love You': '🤟'
  };

  const GESTURE_GUIDE = [
    { name: 'A', emoji: '✊', desc: 'Closed fist with thumb upright alongside index knuckle.' },
    { name: 'B', emoji: '✋', desc: 'Flat hand, 4 straight fingers held close together, thumb tucked across palm.' },
    { name: 'C', emoji: '🫳', desc: 'All 5 fingers curved like the letter C.' },
    { name: 'D', emoji: '☝️', desc: 'Index finger pointing straight up, other 3 fingertips touching thumb tip.' },
    { name: 'L', emoji: '👆', desc: 'Index finger pointing straight up, thumb pointing sideways forming an L.' },
    { name: 'O', emoji: '👌', desc: 'All 5 fingertips curved and touching each other in an O circle.' },
    { name: 'V', emoji: '✌️', desc: 'Index and middle fingers extended apart in a V peace sign, rest folded.' },
    { name: 'Y', emoji: '🤙', desc: 'Thumb and pinky extended out sideways (shaka sign), middle 3 folded.' },
    { name: 'Hello', emoji: '👋', desc: 'Open hand with all 5 fingers spread naturally facing forward.' },
    { name: 'Thank You', emoji: '🙏', desc: 'Flat open palm angled forward away from chin.' },
    { name: 'Yes', emoji: '✊', desc: 'Closed fist held forward in a firm nodding position.' },
    { name: 'No', emoji: '🤏', desc: 'Index and middle fingers snapping down to touch thumb tip.' },
    { name: 'Help', emoji: '🆘', desc: 'Closed fist with thumb extended straight up (thumbs up).' },
    { name: 'I Love You', emoji: '🤟', desc: 'Thumb, index, and pinky extended; middle and ring folded.' },
  ];

  /* ==========================================================================
     CAMERA INITIALIZATION (Browser-Direct getUserMedia)
     ========================================================================== */
  async function startCamera() {
    try {
      if (cameraPrompt) cameraPrompt.style.display = 'none';
      if (camStatus) camStatus.textContent = 'Requesting...';

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      video.srcObject = stream;
      video.onloadedmetadata = () => {
        video.play();
        isStreaming = true;
        if (camStatus) camStatus.textContent = 'Active (Browser HD)';
        // Resize visible canvas to match video
        landmarkCanvas.width = video.videoWidth || 640;
        landmarkCanvas.height = video.videoHeight || 480;
        startProcessingLoop();
      };
    } catch (err) {
      console.warn('Browser webcam permission needed:', err);
      if (cameraPrompt) cameraPrompt.style.display = 'flex';
      if (camStatus) camStatus.textContent = 'Permission Needed';
    }
  }

  if (btnRequestCam) {
    btnRequestCam.addEventListener('click', startCamera);
  }

  // Auto start on load
  startCamera();

  /* ==========================================================================
     REAL-TIME FRAME PROCESSING LOOP
     ========================================================================== */
  function startProcessingLoop() {
    setInterval(async () => {
      if (!isStreaming || isProcessing || video.readyState < 2) return;

      isProcessing = true;
      try {
        // Draw frame to offscreen capture canvas
        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        const imageB64 = captureCanvas.toDataURL('image/jpeg', 0.65);

        // Send to backend
        const res = await fetch('/api/process_frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: imageB64 })
        });

        if (!res.ok) throw new Error('Inference error');
        const data = await res.json();

        // Render skeleton on overlay canvas
        renderLandmarks(data.landmarks, data.hand_detected);

        // Update UI state
        updateUI(data);

      } catch (e) {
        // Ignore single frame drop
      } finally {
        isProcessing = false;
      }
    }, 70); // ~14-15 frames per second inference
  }

  /* ==========================================================================
     SKELETON CANVAS RENDERING
     ========================================================================== */
  function renderLandmarks(landmarks, handDetected) {
    if (!landmarkCtx) return;
    const w = landmarkCanvas.width;
    const h = landmarkCanvas.height;
    landmarkCtx.clearRect(0, 0, w, h);

    if (!handDetected || !landmarks || landmarks.length < 21 || !showSkeleton) {
      return;
    }

    const mirror = toggleMirror.checked;

    // Helper to get canvas coords
    function getPt(lm) {
      const x = mirror ? (1.0 - lm.x) * w : lm.x * w;
      const y = lm.y * h;
      return { x, y };
    }

    // Draw bones / connections
    HAND_CONNECTIONS.forEach(([i, j], connIdx) => {
      const p1 = getPt(landmarks[i]);
      const p2 = getPt(landmarks[j]);

      let boneColor = '#94a3b8';
      if (connIdx < 4) boneColor = FINGER_COLORS[0];
      else if (connIdx < 8) boneColor = FINGER_COLORS[1];
      else if (connIdx < 12) boneColor = FINGER_COLORS[2];
      else if (connIdx < 16) boneColor = FINGER_COLORS[3];
      else if (connIdx < 20) boneColor = FINGER_COLORS[4];

      landmarkCtx.beginPath();
      landmarkCtx.moveTo(p1.x, p1.y);
      landmarkCtx.lineTo(p2.x, p2.y);
      landmarkCtx.strokeStyle = boneColor;
      landmarkCtx.lineWidth = 3;
      landmarkCtx.lineCap = 'round';
      landmarkCtx.stroke();
    });

    // Draw joints
    landmarks.forEach((lm, idx) => {
      const pt = getPt(lm);
      const isTip = [4, 8, 12, 16, 20].includes(idx);
      const radius = isTip ? 6 : 4;

      landmarkCtx.beginPath();
      landmarkCtx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);
      landmarkCtx.fillStyle = isTip ? '#ffffff' : '#00e5ff';
      landmarkCtx.fill();
      landmarkCtx.lineWidth = 1.5;
      landmarkCtx.strokeStyle = '#0a0d14';
      landmarkCtx.stroke();
    });
  }

  /* ==========================================================================
     UI UPDATE
     ========================================================================== */
  let lastSentenceStr = '';

  function updateUI(data) {
    const sign = data.gesture;
    const conf = Math.round((data.confidence || 0) * 100);
    const hold = Math.round((data.hold_progress || 0) * 100);

    // Update HUD Floating Chips
    hudSign.textContent = sign || '--';
    hudConf.textContent = sign ? `${conf}%` : '0%';

    // Update Sidebar Focus Card
    if (sign && conf >= 40) {
      currentSignName.textContent = sign;
      signAvatar.textContent = EMOJI_MAP[sign] || '✋';
      currentSignStatus.textContent = conf >= 75 ? 'Strong Match' : 'Recognizing...';
      currentSignStatus.style.color = conf >= 75 ? 'var(--accent-emerald)' : 'var(--accent-amber)';
    } else {
      currentSignName.textContent = 'Awaiting Hand...';
      signAvatar.textContent = '👋';
      currentSignStatus.textContent = 'Position hand in camera frame';
      currentSignStatus.style.color = 'var(--text-dim)';
    }

    // Progress Bars
    confPercentage.textContent = `${conf}%`;
    confProgressBar.style.width = `${conf}%`;

    holdPercentage.textContent = `${hold}%`;
    holdProgressBar.style.width = `${hold}%`;

    // Sentence Accumulator
    if (data.sentence && data.sentence.length > 0) {
      const sentenceString = data.sentence.join(' ');
      if (sentenceString !== lastSentenceStr) {
        sentenceDisplay.textContent = sentenceString;
        sentenceDisplay.scrollTop = sentenceDisplay.scrollHeight;
        lastSentenceStr = sentenceString;
      }
    } else {
      sentenceDisplay.innerHTML = '<span class="sentence-placeholder">Signed words and letters will appear here...</span>';
      lastSentenceStr = '';
    }

    // Studio Recording state
    if (data.recording_active) {
      recordProgressBox.style.display = 'block';
      if (data.recording_countdown > 0) {
        recordCountdown.textContent = `Get Ready: ${data.recording_countdown}`;
        recordCountdown.style.display = 'block';
        recordSamplesLabel.textContent = `Preparing...`;
        recordProgressFill.style.width = `0%`;
      } else {
        recordCountdown.style.display = 'none';
        recordSamplesLabel.textContent = `${data.recording_current} / ${data.recording_target}`;
        const pct = Math.min(100, Math.round((data.recording_current / data.recording_target) * 100));
        recordProgressFill.style.width = `${pct}%`;
      }
    } else if (recordProgressBox.style.display !== 'none' && !recordingRequested) {
      recordProgressBox.style.display = 'none';
    }
  }

  /* ==========================================================================
     SENTENCE ACTIONS (SPEAK / SPACE / BACKSPACE / CLEAR)
     ========================================================================== */
  btnSpeak.addEventListener('click', async () => {
    const text = sentenceDisplay.textContent.trim();
    if (!text || text.includes('appear here')) return;

    // 1. Instant speech via Web Speech API in browser
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      window.speechSynthesis.speak(utterance);
    }

    // 2. Also trigger backend TTS
    await fetch('/api/speak', { method: 'POST' });
  });

  btnSpace.addEventListener('click', async () => {
    await fetch('/api/space', { method: 'POST' });
  });

  btnBackspace.addEventListener('click', async () => {
    await fetch('/api/backspace', { method: 'POST' });
  });

  btnClear.addEventListener('click', async () => {
    await fetch('/api/clear', { method: 'POST' });
  });

  btnCopySentence.addEventListener('click', () => {
    const text = sentenceDisplay.textContent.trim();
    if (!text || text.includes('appear here')) return;
    navigator.clipboard.writeText(text).then(() => {
      btnCopySentence.style.color = 'var(--accent-emerald)';
      setTimeout(() => {
        btnCopySentence.style.color = '';
      }, 1200);
    });
  });

  /* ==========================================================================
     TOGGLES
     ========================================================================== */
  toggleSkeleton.addEventListener('change', () => {
    showSkeleton = toggleSkeleton.checked;
  });

  toggleMirror.addEventListener('change', () => {
    video.style.transform = toggleMirror.checked ? 'scaleX(-1)' : 'none';
  });

  // Default mirror on load
  video.style.transform = 'scaleX(-1)';

  /* ==========================================================================
     TABS
     ========================================================================== */
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add('active');

      if (targetId === 'tab-model') fetchModelInfo();
    });
  });

  /* ==========================================================================
     DATA STUDIO RECORDER
     ========================================================================== */
  let recordingRequested = false;

  btnStartRecord.addEventListener('click', async () => {
    const name = recordGestureName.value.trim();
    const count = parseInt(recordSampleCount.value) || 100;

    if (!name) {
      alert('Please enter a gesture name to record.');
      recordGestureName.focus();
      return;
    }

    recordingRequested = true;
    recordProgressBox.style.display = 'block';
    recordCountdown.style.display = 'block';
    recordCountdown.textContent = 'Get Ready: 3';
    recordProgressFill.style.width = '0%';

    try {
      const res = await fetch('/api/record_gesture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gesture_name: name, target_samples: count })
      });
      const data = await res.json();
      if (!res.ok) alert(data.error || 'Failed to start recording');
    } catch (e) {
      alert('Error connecting to recorder.');
    } finally {
      recordingRequested = false;
    }
  });

  /* ==========================================================================
     MODEL INFO & RETRAINING
     ========================================================================== */
  async function fetchModelInfo() {
    try {
      const res = await fetch('/api/model_info');
      const data = await res.json();

      const acc = data.accuracy ? Math.round(data.accuracy * 100) : 100;
      statAccuracy.textContent = `${acc}%`;

      const classes = data.classes || [];
      statClassesCount.textContent = classes.length;
      statSamplesCount.textContent = data.total_samples || '1400';

      classesList.innerHTML = '';
      classes.forEach(c => {
        const badge = document.createElement('span');
        badge.className = 'class-badge';
        const emoji = EMOJI_MAP[c] || '✋';
        badge.textContent = `${emoji} ${c}`;
        classesList.appendChild(badge);
      });
    } catch (e) {
      console.error('Failed to load model info', e);
    }
  }

  btnRetrain.addEventListener('click', async () => {
    btnRetrain.disabled = true;
    retrainStatus.textContent = 'Training in background... Please wait';
    retrainStatus.style.color = 'var(--accent-amber)';

    try {
      const res = await fetch('/api/retrain', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        retrainStatus.textContent = `✓ Retrained successfully! Accuracy: ${Math.round(data.accuracy * 100)}%`;
        retrainStatus.style.color = 'var(--accent-emerald)';
        fetchModelInfo();
      } else {
        retrainStatus.textContent = `Error: ${data.error}`;
        retrainStatus.style.color = 'var(--accent-rose)';
      }
    } catch (e) {
      retrainStatus.textContent = 'Retraining failed. Check terminal logs.';
      retrainStatus.style.color = 'var(--accent-rose)';
    } finally {
      btnRetrain.disabled = false;
    }
  });

  /* ==========================================================================
     RENDER GUIDE
     ========================================================================== */
  function renderGuide() {
    if (!guideCards) return;
    guideCards.innerHTML = '';
    GESTURE_GUIDE.forEach(item => {
      const card = document.createElement('div');
      card.className = 'guide-card';
      card.innerHTML = `
        <div class="guide-emoji">${item.emoji}</div>
        <div class="guide-title">${item.name}</div>
        <div class="guide-desc">${item.desc}</div>
      `;
      guideCards.appendChild(card);
    });
  }

  renderGuide();
  fetchModelInfo();
});
