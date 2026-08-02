/**
 * MultiModal Bridge — Client-Side Application
 * Handles webcam streaming, Web Speech API, and SocketIO communication.
 */

// ---- State ----
let currentMode = 'slr';
let isActive = false;
let socket = null;
let webcamStream = null;
let frameInterval = null;
let recognition = null; // Web Speech API

// ---- Initialize ----
document.addEventListener('DOMContentLoaded', () => {
    initSocket();
    initTTSVoices();
    initSliders();
});

// ============================================================
//  Socket.IO Connection
// ============================================================

function initSocket() {
    socket = io();

    socket.on('connect', () => {
        document.getElementById('connectionDot').classList.add('connected');
        document.getElementById('connectionText').textContent = 'Connected';
        addLog('Connected to server', 'system');
    });

    socket.on('disconnect', () => {
        document.getElementById('connectionDot').classList.remove('connected');
        document.getElementById('connectionText').textContent = 'Disconnected';
        addLog('Disconnected from server', 'system');
        stopAll();
    });

    socket.on('system_message', (data) => {
        addLog(data.text, 'system');
    });

    // SLR events
    socket.on('slr_prediction', (data) => {
        showPrediction(data);
    });

    socket.on('landmarks', (data) => {
        drawLandmarks(data);
    });

    socket.on('buffer_fill', (data) => {
        updateBufferBar(data.fill);
    });

    socket.on('slr_status', (data) => {
        if (!data.active) {
            addLog('Sign recognition stopped', 'system');
        }
    });
}

// ============================================================
//  Mode Switching
// ============================================================

function switchMode(mode) {
    if (isActive) stopAll();

    currentMode = mode;

    // Update mode buttons
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('btn' + mode.toUpperCase()).classList.add('active');

    // Show/hide views
    document.getElementById('slrView').classList.toggle('hidden', mode !== 'slr');
    document.getElementById('sttView').classList.toggle('hidden', mode !== 'stt');
    document.getElementById('ttsView').classList.toggle('hidden', mode !== 'tts');

    // Update panel title
    const titles = { slr: 'Camera Feed', stt: 'Microphone', tts: 'Text Input' };
    document.getElementById('inputPanelTitle').textContent = titles[mode];

    // Update toggle button
    document.getElementById('toggleText').textContent = 'Start';
    document.getElementById('toggleIcon').textContent = '\u25B6';

    // Update prediction display labels
    const labels = { slr: 'Detected Sign', stt: 'Recognized Speech', tts: 'Speaking' };
    document.querySelector('.prediction-label').textContent = labels[mode];
    document.getElementById('predictionWord').textContent = '---';
    document.getElementById('predictionConf').textContent = 'Waiting...';

    // Show/hide reset button (only for SLR)
    document.getElementById('btnReset').classList.toggle('hidden', mode !== 'slr');
}

// ============================================================
//  Toggle Active (Start/Stop)
// ============================================================

function toggleActive() {
    if (isActive) {
        stopAll();
    } else {
        startCurrent();
    }
}

function startCurrent() {
    isActive = true;
    document.getElementById('toggleText').textContent = 'Stop';
    document.getElementById('toggleIcon').textContent = '\u23F9';
    document.getElementById('btnToggle').classList.add('active');

    switch (currentMode) {
        case 'slr': startSLR(); break;
        case 'stt': startSTT(); break;
        case 'tts': startTTS(); break;
    }
}

function stopAll() {
    isActive = false;
    document.getElementById('toggleText').textContent = 'Start';
    document.getElementById('toggleIcon').textContent = '\u25B6';
    document.getElementById('btnToggle').classList.remove('active');

    stopSLR();
    stopSTT();
    stopTTSSpeaking();
}

// ============================================================
//  SLR — Sign Language Recognition
// ============================================================

function startSLR() {
    const video = document.getElementById('webcam');
    const overlay = document.getElementById('videoOverlay');

    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
        .then(stream => {
            webcamStream = stream;
            video.srcObject = stream;
            overlay.classList.add('hidden');

            socket.emit('start_slr');
            addLog('Sign language recognition started', 'system');

            // Send frames at ~10 FPS
            frameInterval = setInterval(() => {
                if (!isActive) return;
                sendFrame(video);
            }, 100);
        })
        .catch(err => {
            addLog('Camera access denied: ' + err.message, 'system');
            stopAll();
        });
}

function stopSLR() {
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }

    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    const video = document.getElementById('webcam');
    video.srcObject = null;

    document.getElementById('videoOverlay').classList.remove('hidden');
    clearCanvas();

    if (socket && socket.connected) {
        socket.emit('stop_slr');
    }
}

function sendFrame(video) {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');

    // Mirror the frame to match what user sees
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
    socket.emit('video_frame', { image: dataUrl });
}

function drawLandmarks(data) {
    const canvas = document.getElementById('overlayCanvas');
    const video = document.getElementById('webcam');

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Mirror the canvas to match mirrored video
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    const w = canvas.width;
    const h = canvas.height;

    // Draw body
    if (data.body) {
        const body = data.body;
        const bodyConns = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24]];
        ctx.strokeStyle = 'rgba(0, 255, 200, 0.7)';
        ctx.lineWidth = 2;
        bodyConns.forEach(([i, j]) => {
            if (i < body.length && j < body.length) {
                ctx.beginPath();
                ctx.moveTo(body[i][0] * w, body[i][1] * h);
                ctx.lineTo(body[j][0] * w, body[j][1] * h);
                ctx.stroke();
            }
        });

        body.forEach((pt, i) => {
            if (i <= 24) {
                ctx.fillStyle = 'rgba(0, 200, 255, 0.9)';
                ctx.beginPath();
                ctx.arc(pt[0] * w, pt[1] * h, 3, 0, Math.PI * 2);
                ctx.fill();
            }
        });
    }

    // Draw hands
    const handConns = [
        [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
        [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
        [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]
    ];

    ['left_hand', 'right_hand'].forEach(key => {
        if (!data[key]) return;
        const hand = data[key];
        const color = key === 'left_hand' ? 'rgba(255, 100, 100, 0.8)' : 'rgba(100, 100, 255, 0.8)';

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        handConns.forEach(([i, j]) => {
            if (i < hand.length && j < hand.length) {
                ctx.beginPath();
                ctx.moveTo(hand[i][0] * w, hand[i][1] * h);
                ctx.lineTo(hand[j][0] * w, hand[j][1] * h);
                ctx.stroke();
            }
        });

        hand.forEach(pt => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(pt[0] * w, pt[1] * h, 2, 0, Math.PI * 2);
            ctx.fill();
        });
    });

    ctx.restore();
}

function clearCanvas() {
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function updateBufferBar(fill) {
    const pct = Math.round(fill * 100);
    document.getElementById('bufferBar').style.width = pct + '%';
    document.getElementById('bufferText').textContent = 'Buffer: ' + pct + '%';
}

function resetBuffer() {
    socket.emit('reset_buffer');
    document.getElementById('predictionWord').textContent = '---';
    document.getElementById('predictionConf').textContent = 'Buffer reset';
    document.getElementById('top5List').innerHTML = '';
}

// ============================================================
//  SLR Prediction Display
// ============================================================

function showPrediction(data) {
    document.getElementById('predictionWord').textContent = data.gloss.toUpperCase();
    document.getElementById('predictionConf').textContent = data.confidence + '% confidence';

    // Top-5
    let html = '';
    data.top5.forEach((item, i) => {
        const rankClass = i === 0 ? 'first' : '';
        html += `
            <div class="top5-item">
                <div class="top5-rank ${rankClass}">${i + 1}</div>
                <div class="top5-name">${item.gloss}</div>
                <div class="top5-bar-bg">
                    <div class="top5-bar-fill" style="width: ${item.confidence}%"></div>
                </div>
                <div class="top5-pct">${item.confidence}%</div>
            </div>
        `;
    });
    document.getElementById('top5List').innerHTML = html;

    // Add to log (debounced — only if different from last)
    if (!showPrediction._last || showPrediction._last !== data.gloss) {
        addLog('[Sign] ' + data.gloss.toUpperCase() + ' (' + data.confidence + '%)', 'sign');
        showPrediction._last = data.gloss;
    }
}

// ============================================================
//  STT — Speech to Text (Web Speech API)
// ============================================================

function startSTT() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        addLog('Speech recognition not supported in this browser. Use Chrome.', 'system');
        stopAll();
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    const micCircle = document.getElementById('micCircle');
    const micStatus = document.getElementById('micStatus');

    recognition.onstart = () => {
        micCircle.classList.add('listening');
        micStatus.textContent = 'Listening...';
        addLog('Speech recognition started', 'system');
    };

    recognition.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }

        if (finalTranscript) {
            document.getElementById('predictionWord').textContent = finalTranscript;
            document.getElementById('predictionConf').textContent = 'Final result';
            addLog('[Speech] ' + finalTranscript, 'speech');
        } else if (interimTranscript) {
            document.getElementById('predictionWord').textContent = interimTranscript;
            document.getElementById('predictionConf').textContent = 'Listening...';
        }
    };

    recognition.onerror = (event) => {
        if (event.error !== 'no-speech') {
            addLog('Speech error: ' + event.error, 'system');
        }
    };

    recognition.onend = () => {
        micCircle.classList.remove('listening');
        if (isActive && currentMode === 'stt') {
            // Restart if still active
            try { recognition.start(); } catch(e) {}
        } else {
            micStatus.textContent = 'Click Start to listen';
        }
    };

    recognition.start();
}

function stopSTT() {
    if (recognition) {
        try { recognition.stop(); } catch(e) {}
        recognition = null;
    }
    const micCircle = document.getElementById('micCircle');
    if (micCircle) micCircle.classList.remove('listening');
    const micStatus = document.getElementById('micStatus');
    if (micStatus) micStatus.textContent = 'Click Start to listen';
}

// ============================================================
//  TTS — Text to Speech (Web Speech Synthesis)
// ============================================================

function initTTSVoices() {
    const select = document.getElementById('voiceSelect');
    function populateVoices() {
        const voices = speechSynthesis.getVoices();
        select.innerHTML = '';
        voices.forEach((voice, i) => {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = voice.name + ' (' + voice.lang + ')';
            if (voice.default) option.selected = true;
            select.appendChild(option);
        });
    }
    populateVoices();
    speechSynthesis.onvoiceschanged = populateVoices;
}

function initSliders() {
    const rate = document.getElementById('ttsRate');
    const pitch = document.getElementById('ttsPitch');
    rate.oninput = () => document.getElementById('rateVal').textContent = parseFloat(rate.value).toFixed(1);
    pitch.oninput = () => document.getElementById('pitchVal').textContent = parseFloat(pitch.value).toFixed(1);
}

function startTTS() {
    const text = document.getElementById('ttsInput').value.trim();
    if (!text) {
        addLog('Please enter some text to speak', 'system');
        stopAll();
        return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    const voices = speechSynthesis.getVoices();
    const voiceIndex = document.getElementById('voiceSelect').value;
    if (voices[voiceIndex]) utterance.voice = voices[voiceIndex];

    utterance.rate = parseFloat(document.getElementById('ttsRate').value);
    utterance.pitch = parseFloat(document.getElementById('ttsPitch').value);

    document.getElementById('predictionWord').textContent = text.substring(0, 30) + (text.length > 30 ? '...' : '');
    document.getElementById('predictionConf').textContent = 'Speaking...';

    utterance.onend = () => {
        document.getElementById('predictionConf').textContent = 'Done';
        stopAll();
    };

    speechSynthesis.speak(utterance);
    addLog('[TTS] ' + text, 'tts');
}

function stopTTSSpeaking() {
    speechSynthesis.cancel();
}

// ============================================================
//  Conversation Log
// ============================================================

function addLog(text, type) {
    const log = document.getElementById('conversationLog');
    const entry = document.createElement('div');
    entry.className = 'log-entry ' + (type || 'system');

    const now = new Date();
    const time = now.getHours().toString().padStart(2, '0') + ':' +
                 now.getMinutes().toString().padStart(2, '0') + ':' +
                 now.getSeconds().toString().padStart(2, '0');

    entry.innerHTML = '<span class="log-time">' + time + '</span><span class="log-text">' + text + '</span>';
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;

    // Keep max 50 entries
    while (log.children.length > 50) {
        log.removeChild(log.firstChild);
    }
}

function clearOutput() {
    document.getElementById('conversationLog').innerHTML = '';
    document.getElementById('predictionWord').textContent = '---';
    document.getElementById('predictionConf').textContent = 'Cleared';
    document.getElementById('top5List').innerHTML = '';
    addLog('Output cleared', 'system');
}
