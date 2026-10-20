/**
 * MultiModal Bridge — Client-Side Application
 *
 * Implements the full communication workflow:
 *   1. Session setup (role selection)
 *   2. Input acquisition (sign / speech / text)
 *   3. Message processing (via server + LLM)
 *   4. Output routing (text / TTS / avatar)
 *   5. Session loop (continue / end)
 */

// ================================================================
//  State
// ================================================================

let socket = null;
let sessionActive = false;
let currentInputMode = 'sign';
let isInputActive = false;

// SLR state
let webcamStream = null;
let frameInterval = null;

// STT state
let recognition = null;

// Session config
let userARole = 'normal';
let userBRole = 'normal';
let availableInputs = ['sign', 'speech', 'text'];
let outputModes = ['text', 'tts', 'avatar'];

let selectedCameraId = localStorage.getItem('preferredCameraId') || null;

// ================================================================
//  Initialize
// ================================================================

document.addEventListener('DOMContentLoaded', () => {
    initSocket();
    initCameraList();
});

// ================================================================
//  Socket.IO Connection
// ================================================================

function initSocket() {
    socket = io();

    socket.on('connect', () => {
        document.getElementById('connectionDot').classList.add('connected');
        document.getElementById('connectionText').textContent = 'Connected';
    });

    socket.on('disconnect', () => {
        document.getElementById('connectionDot').classList.remove('connected');
        document.getElementById('connectionText').textContent = 'Disconnected';
        stopAllInputs();
    });

    socket.on('system_message', (data) => {
        addLog(data.text, data.type || 'info');
    });

    // Session events
    socket.on('session_started', (data) => {
        onSessionStarted(data);
    });

    socket.on('session_ended', () => {
        onSessionEnded();
    });

    // SLR events
    socket.on('landmarks', (data) => {
        drawLandmarks(data);
    });

    socket.on('buffer_fill', (data) => {
        updateBufferBar(data.fill);
    });

    socket.on('slr_prediction', (data) => {
        showDetection(data);
    });

    socket.on('slr_status', (data) => {
        if (!data.active) {
            addLog('Sign recognition stopped', 'info');
        }
    });

    // Message processing result
    socket.on('message_processed', (data) => {
        handleProcessedMessage(data);
    });
}

// ================================================================
//  Session Management
// ================================================================

function startSession() {
    const userA = document.querySelector('input[name="userA"]:checked');
    const userB = document.querySelector('input[name="userB"]:checked');

    if (!userA || !userB) {
        alert('Please select roles for both users');
        return;
    }

    userARole = userA.value;
    userBRole = userB.value;

    // Set API key if provided
    const apiKey = document.getElementById('setupApiKey').value.trim();
    if (apiKey) {
        // Wait for connection if needed
        if (socket && socket.connected) {
            socket.emit('set_api_key', { api_key: apiKey });
        }
    }

    socket.emit('start_session', {
        user_a_role: userARole,
        user_b_role: userBRole
    });
}

function onSessionStarted(data) {
    sessionActive = true;
    availableInputs = data.available_inputs;
    outputModes = data.output_modes;

    // Update session label
    const roleLabels = { deaf: '🤟 Deaf', blind: '👁️ Blind', mute: '🤐 Mute', normal: '👤 Normal' };
    document.getElementById('sessionLabel').textContent =
        (roleLabels[data.user_a_role] || data.user_a_role) + ' → ' +
        (roleLabels[data.user_b_role] || data.user_b_role);
    document.getElementById('sessionBadge').textContent = 'Active';

    // Configure input tabs
    configureInputTabs(availableInputs);

    // Configure output badges
    configureOutputBadges(outputModes);

    // Show/hide avatar panel
    const avatarPanel = document.getElementById('avatarPanel');
    avatarPanel.classList.toggle('hidden', !outputModes.includes('avatar'));

    // Show output target
    document.getElementById('outputTarget').textContent =
        'User B (' + data.user_b_role.charAt(0).toUpperCase() + data.user_b_role.slice(1) + ')';

    // Switch screens
    document.getElementById('setupScreen').classList.add('hidden');
    document.getElementById('commScreen').classList.remove('hidden');

    // Select first available input mode
    if (availableInputs.length > 0) {
        switchInputMode(availableInputs[0]);
    }

    addLog('Session started', 'success');
}

function swapRoles() {
    stopAllInputs();
    if (socket && socket.connected) {
        socket.emit('swap_roles');
    }
}

function saveCommApiKey() {
    const input = document.getElementById('commApiKey');
    if (!input) return;
    const key = input.value.trim();
    if (key) {
        if (socket && socket.connected) {
            socket.emit('set_api_key', { api_key: key });
            input.value = '';
            input.placeholder = 'Saved ✓';
            addLog('OpenRouter API Key updated successfully', 'success');
        }
    } else {
        addLog('Please enter a valid API key', 'error');
    }
}

function endSession() {
    stopAllInputs();

    if (socket && socket.connected) {
        socket.emit('end_session');
    }
}

function onSessionEnded() {
    sessionActive = false;
    stopAllInputs();

    // Switch screens
    document.getElementById('commScreen').classList.add('hidden');
    document.getElementById('setupScreen').classList.remove('hidden');

    // Clear conversation
    const area = document.getElementById('conversationArea');
    area.innerHTML = `
        <div class="conv-welcome">
            <div class="conv-welcome-icon">💬</div>
            <p>Messages will appear here</p>
            <p class="conv-welcome-sub">Send a message using the input panel</p>
        </div>
    `;
}

function continueSession() {
    resetBuffer();
    addLog('Ready for next message', 'info');
    if (currentInputMode === 'text') {
        const input = document.getElementById('textInput');
        if (input) input.focus();
    }
}

// ================================================================
//  Input Tab Configuration
// ================================================================

function configureInputTabs(inputs) {
    const tabs = document.querySelectorAll('.input-tab');
    tabs.forEach(tab => {
        const mode = tab.dataset.mode;
        if (inputs.includes(mode)) {
            tab.classList.remove('hidden');
        } else {
            tab.classList.add('hidden');
        }
    });

    // Show/hide reset button (only for sign)
    document.getElementById('btnReset').classList.toggle('hidden', !inputs.includes('sign'));
}

function configureOutputBadges(modes) {
    const container = document.getElementById('outputBadges');
    const labels = { text: '📝 Text', tts: '🔊 Voice', avatar: '🤟 Avatar' };
    container.innerHTML = modes.map(m =>
        `<span class="output-badge">${labels[m] || m}</span>`
    ).join('');
}

// ================================================================
//  Input Mode Switching
// ================================================================

function switchInputMode(mode) {
    if (isInputActive) stopAllInputs();

    currentInputMode = mode;

    // Update tabs
    document.querySelectorAll('.input-tab').forEach(t => t.classList.remove('active'));
    const activeTab = document.querySelector(`.input-tab[data-mode="${mode}"]`);
    if (activeTab) activeTab.classList.add('active');

    // Show/hide views
    document.getElementById('signView').classList.toggle('hidden', mode !== 'sign');
    document.getElementById('speechView').classList.toggle('hidden', mode !== 'speech');
    document.getElementById('textView').classList.toggle('hidden', mode !== 'text');

    // Update toggle button visibility
    const toggleBtn = document.getElementById('btnToggle');
    const resetBtn = document.getElementById('btnReset');
    toggleBtn.classList.toggle('hidden', mode === 'text');
    resetBtn.classList.toggle('hidden', mode !== 'sign');

    // Reset toggle state
    document.getElementById('toggleText').textContent = 'Start';
    document.getElementById('toggleIcon').textContent = '\u25B6';
    document.getElementById('btnToggle').classList.remove('active');
}

// ================================================================
//  Input Toggle (Start/Stop)
// ================================================================

function toggleInput() {
    if (isInputActive) {
        stopAllInputs();
    } else {
        startCurrentInput();
    }
}

function startCurrentInput() {
    isInputActive = true;
    document.getElementById('toggleText').textContent = 'Stop';
    document.getElementById('toggleIcon').textContent = '\u23F9';
    document.getElementById('btnToggle').classList.add('active');

    switch (currentInputMode) {
        case 'sign': startSLR(); break;
        case 'speech': startSTT(); break;
    }
}

function stopAllInputs() {
    isInputActive = false;
    document.getElementById('toggleText').textContent = 'Start';
    document.getElementById('toggleIcon').textContent = '\u25B6';
    document.getElementById('btnToggle').classList.remove('active');

    stopSLR();
    stopSTT();
}

// ================================================================
//  Camera Selection & Management
// ================================================================

async function initCameraList() {
    const select = document.getElementById('cameraSelect');
    if (!select || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;

    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');

        if (videoDevices.length === 0) return;

        select.innerHTML = '';
        let laptopCamId = null;

        videoDevices.forEach((device, idx) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            const label = device.label || `Camera ${idx + 1}`;
            option.textContent = label;

            // Detect likely laptop built-in webcam
            const lowerLabel = label.toLowerCase();
            const isPhoneOrVirtual = lowerLabel.includes('phone') || lowerLabel.includes('droid') || 
                                     lowerLabel.includes('link') || lowerLabel.includes('virtual') || 
                                     lowerLabel.includes('cellular') || lowerLabel.includes('continuity');
            const isLaptopCam = lowerLabel.includes('integrated') || lowerLabel.includes('internal') || 
                                lowerLabel.includes('built-in') || lowerLabel.includes('webcam') || 
                                lowerLabel.includes('hd') || lowerLabel.includes('front');

            if (!laptopCamId && isLaptopCam && !isPhoneOrVirtual) {
                laptopCamId = device.deviceId;
            }

            select.appendChild(option);
        });

        // Determine which device to select:
        // 1. User's saved preference
        // 2. Detected laptop camera
        // 3. First non-phone device, or first device
        const targetId = selectedCameraId || laptopCamId || (
            videoDevices.find(d => {
                const l = d.label.toLowerCase();
                return !l.includes('phone') && !l.includes('droid') && !l.includes('virtual');
            }) || videoDevices[0]
        ).deviceId;

        select.value = targetId;
        selectedCameraId = targetId;
    } catch (e) {
        console.warn('Could not enumerate cameras:', e);
    }
}

function switchCameraDevice(deviceId) {
    selectedCameraId = deviceId;
    localStorage.setItem('preferredCameraId', deviceId);
    addLog('Camera switched to selected device', 'info');

    // If camera is currently active, restart stream with new device
    if (isInputActive && currentInputMode === 'sign') {
        stopSLR();
        setTimeout(startSLR, 200);
    }
}

// ================================================================
//  SLR — Sign Language Recognition
// ================================================================

function startSLR() {
    const video = document.getElementById('webcam');
    const overlay = document.getElementById('videoOverlay');

    // Build constraints: prefer user facing and selected laptop deviceId
    const videoConstraints = {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
    };

    if (selectedCameraId) {
        videoConstraints.deviceId = { exact: selectedCameraId };
    }

    navigator.mediaDevices.getUserMedia({ video: videoConstraints })
        .then(stream => {
            webcamStream = stream;
            video.srcObject = stream;
            overlay.classList.add('hidden');

            // Refresh camera list once permissions are granted so labels appear
            initCameraList();

            socket.emit('start_slr');
            addLog('Sign language recognition started', 'success');

            frameInterval = setInterval(() => {
                if (!isInputActive) return;
                sendFrame(video);
            }, 100);
        })
        .catch(err => {
            // If exact deviceId failed, fallback to any user-facing camera
            if (selectedCameraId) {
                console.warn('Failed with specific camera, falling back to facingMode: user...');
                selectedCameraId = null;
                navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } })
                    .then(stream => {
                        webcamStream = stream;
                        video.srcObject = stream;
                        overlay.classList.add('hidden');
                        initCameraList();
                        socket.emit('start_slr');
                        frameInterval = setInterval(() => {
                            if (!isInputActive) return;
                            sendFrame(video);
                        }, 100);
                    })
                    .catch(e => {
                        addLog('Camera access denied: ' + e.message, 'error');
                        stopAllInputs();
                    });
                return;
            }
            addLog('Camera access denied: ' + err.message, 'error');
            stopAllInputs();
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
    if (video) video.srcObject = null;

    const overlay = document.getElementById('videoOverlay');
    if (overlay) overlay.classList.remove('hidden');

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
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
    socket.emit('video_frame', { image: dataUrl });
}

function drawLandmarks(data) {
    const canvas = document.getElementById('overlayCanvas');
    const video = document.getElementById('webcam');
    if (!canvas || !video) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

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
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

function updateBufferBar(fill) {
    const pct = Math.round(fill * 100);
    const bar = document.getElementById('bufferBar');
    const text = document.getElementById('bufferText');
    if (bar) bar.style.width = pct + '%';
    if (text) text.textContent = 'Buffer: ' + pct + '%';
}

function showDetection(data) {
    document.getElementById('detectedSign').textContent = data.gloss.toUpperCase();
    document.getElementById('detectedConf').textContent = data.confidence + '%';

    // Mini top-5
    let html = '';
    data.top5.forEach((item, i) => {
        html += `<span class="top5-chip ${i === 0 ? 'top' : ''}">${item.gloss} ${item.confidence}%</span>`;
    });
    document.getElementById('top5Mini').innerHTML = html;
}

function resetBuffer() {
    socket.emit('reset_buffer');
    document.getElementById('detectedSign').textContent = '---';
    document.getElementById('detectedConf').textContent = '';
    document.getElementById('top5Mini').innerHTML = '';
}

// ================================================================
//  STT — Speech to Text (Web Speech API)
// ================================================================

function startSTT() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        addLog('Speech recognition not supported — use Chrome', 'error');
        stopAllInputs();
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    const micCircle = document.getElementById('micCircle');
    const micStatus = document.getElementById('micStatus');
    const transcript = document.getElementById('speechTranscript');

    recognition.onstart = () => {
        micCircle.classList.add('listening');
        micStatus.textContent = 'Listening...';
        addLog('Speech recognition started', 'success');
    };

    recognition.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const t = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += t;
            } else {
                interimTranscript += t;
            }
        }

        // Show in transcript area
        if (finalTranscript) {
            transcript.innerHTML = `<p class="transcript-final">"${finalTranscript}"</p>`;
            // Send to server
            socket.emit('speech_input', { text: finalTranscript });
            addLog('[Speech] ' + finalTranscript, 'speech');
        } else if (interimTranscript) {
            transcript.innerHTML = `<p class="transcript-interim">${interimTranscript}</p>`;
        }
    };

    recognition.onerror = (event) => {
        if (event.error !== 'no-speech') {
            addLog('Speech error: ' + event.error, 'error');
        }
    };

    recognition.onend = () => {
        micCircle.classList.remove('listening');
        if (isInputActive && currentInputMode === 'speech') {
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

// ================================================================
//  Text Input — Send Message
// ================================================================

function sendTextMessage() {
    const input = document.getElementById('textInput');
    const text = input.value.trim();

    if (!text) {
        addLog('Please type a message', 'error');
        return;
    }

    socket.emit('text_input', { text: text });
    addLog('[Text] ' + text, 'text');
    input.value = '';
}

// Allow Enter key to send (Shift+Enter for newline)
document.addEventListener('keydown', (e) => {
    if (e.target.id === 'textInput' && e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendTextMessage();
    }
});

// ================================================================
//  Message Processing — Handle Output
// ================================================================

function handleProcessedMessage(data) {
    const { original_text, processed_text, source, outputs } = data;

    // Remove welcome message if present
    const welcome = document.querySelector('.conv-welcome');
    if (welcome) welcome.remove();

    const area = document.getElementById('conversationArea');

    // Source label
    const sourceLabels = { sign: '🤟 Sign', speech: '🎤 Speech', text: '⌨️ Text' };
    const sourceLabel = sourceLabels[source] || source;

    // Create message card
    const msg = document.createElement('div');
    msg.className = 'conv-message';
    msg.innerHTML = `
        <div class="conv-source">${sourceLabel}</div>
        <div class="conv-original">${escapeHtml(original_text)}</div>
        ${processed_text !== original_text
            ? `<div class="conv-processed">
                 <span class="conv-processed-label">✨ AI Processed:</span>
                 ${escapeHtml(processed_text)}
               </div>`
            : ''
        }
        <div class="conv-outputs">
            ${outputs.show_text ? '<span class="conv-out-badge text">📝 Text</span>' : ''}
            ${outputs.speak_text ? '<span class="conv-out-badge tts">🔊 Spoken</span>' : ''}
            ${outputs.show_avatar ? '<span class="conv-out-badge avatar">🤟 Signed</span>' : ''}
        </div>
    `;
    area.appendChild(msg);
    area.scrollTop = area.scrollHeight;

    // TTS Output
    if (outputs.speak_text && outputs.tts_text) {
        speakText(outputs.tts_text);
    }

    // Avatar Output
    if (outputs.show_avatar && outputs.avatar_glosses) {
        showSignAvatar(outputs.avatar_glosses);
    }
}

// ================================================================
//  TTS — Text to Speech Output
// ================================================================

function speakText(text) {
    if (!('speechSynthesis' in window)) {
        addLog('TTS not supported in this browser', 'error');
        return;
    }

    const indicator = document.getElementById('ttsIndicator');
    indicator.classList.remove('hidden');

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onend = () => {
        indicator.classList.add('hidden');
    };

    utterance.onerror = () => {
        indicator.classList.add('hidden');
    };

    speechSynthesis.speak(utterance);
}

// ================================================================
//  Sign Avatar — Animated Signing Display
// ================================================================

// Global Avatar Animation State
let avatarAnimFrame = null;
let avatarGlossesQueue = [];
let avatarCurrentIndex = 0;
let avatarStartTime = 0;

function showSignAvatar(glosses) {
    const panel = document.getElementById('avatarPanel');
    if (!glosses || glosses.length === 0) return;

    panel.classList.remove('hidden');
    avatarGlossesQueue = glosses;
    avatarCurrentIndex = 0;
    avatarStartTime = performance.now();

    if (avatarAnimFrame) {
        cancelAnimationFrame(avatarAnimFrame);
    }

    runAvatarAnimationLoop();
}

function runAvatarAnimationLoop() {
    const canvas = document.getElementById('avatarCanvas');
    const glossDisplay = document.getElementById('avatarGloss');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const durationPerGloss = 1800; // 1.8s per word animation
    const now = performance.now();
    const elapsed = now - avatarStartTime;

    const currentGlossIdx = Math.floor(elapsed / durationPerGloss);

    if (currentGlossIdx < avatarGlossesQueue.length) {
        const currentGloss = avatarGlossesQueue[currentGlossIdx];
        const progressInGloss = (elapsed % durationPerGloss) / durationPerGloss;

        if (glossDisplay) {
            glossDisplay.innerHTML = `<span class="gloss-word">${currentGloss.toUpperCase()}</span> <span class="gloss-counter">(${currentGlossIdx + 1}/${avatarGlossesQueue.length})</span>`;
        }

        renderAvatarFrame(ctx, canvas.width, canvas.height, currentGloss, progressInGloss);
        avatarAnimFrame = requestAnimationFrame(runAvatarAnimationLoop);
    } else {
        // Animation completed — return to relaxed rest pose
        const lastGloss = avatarGlossesQueue[avatarGlossesQueue.length - 1];
        if (glossDisplay) {
            glossDisplay.innerHTML = `<span class="gloss-word">${lastGloss.toUpperCase()}</span> <span class="gloss-done">✓ Done</span>`;
        }
        renderAvatarFrame(ctx, canvas.width, canvas.height, 'rest', 1.0);
    }
}

// Keyframe pose generator for ASL gloss gestures
function getAvatarJointAngles(gloss, t) {
    // t ranges 0 to 1
    // Smooth sinusoidal ease in and out:
    const ease = Math.sin(t * Math.PI);

    // Default rest pose (arms naturally relaxed at sides/waist)
    let leftShoulder = 0.3;
    let leftElbow = 0.5;
    let rightShoulder = -0.3;
    let rightElbow = -0.5;
    let handGesture = 'open';

    const g = gloss.toLowerCase();

    if (g === 'drink' || g === 'water') {
        // Right hand moves up to mouth with cup shape
        rightShoulder = -0.3 - 1.2 * ease;
        rightElbow = -0.5 - 1.0 * ease;
        handGesture = 'cup';
    } else if (g === 'eat' || g === 'food') {
        // Right hand fingertips touch mouth twice
        const bob = Math.sin(t * Math.PI * 4);
        rightShoulder = -1.3 + bob * 0.2;
        rightElbow = -1.4 + bob * 0.2;
        handGesture = 'pinch';
    } else if (g === 'book') {
        // Both palms come together, then open like pages
        const open = Math.sin(t * Math.PI * 2);
        leftShoulder = 0.8 + open * 0.3;
        leftElbow = 1.0;
        rightShoulder = -0.8 - open * 0.3;
        rightElbow = -1.0;
        handGesture = 'flat';
    } else if (g === 'help') {
        // Left hand flat palm up, right fist lifts up on top
        leftShoulder = 0.7;
        leftElbow = 0.9;
        rightShoulder = -0.7 - ease * 0.4;
        rightElbow = -0.9 - ease * 0.4;
        handGesture = 'fist';
    } else if (g === 'yes') {
        // Right fist nods up and down
        const nod = Math.sin(t * Math.PI * 6);
        rightShoulder = -0.9;
        rightElbow = -1.2 + nod * 0.3;
        handGesture = 'fist';
    } else if (g === 'no') {
        // Right hand snaps fingers
        rightShoulder = -0.9;
        rightElbow = -1.1;
        handGesture = 'snap';
    } else if (g === 'fine' || g === 'hello') {
        // Right hand waves near temple/chest
        const wave = Math.sin(t * Math.PI * 6) * 0.25;
        rightShoulder = -1.4 + wave;
        rightElbow = -1.5;
        handGesture = 'wave';
    } else if (g === 'want') {
        // Both hands reach forward and pull back
        const pull = ease * 0.5;
        leftShoulder = 0.6 + pull;
        leftElbow = 0.7 - pull;
        rightShoulder = -0.6 - pull;
        rightElbow = -0.7 + pull;
        handGesture = 'claw';
    } else {
        // Dynamic procedural signing based on word characters
        const hash = g.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);
        const cycle = Math.sin(t * Math.PI * 2);
        const sym = (hash % 2 === 0);

        rightShoulder = -0.8 - ease * ((hash % 50) / 100);
        rightElbow = -0.9 + cycle * 0.3;
        if (sym) {
            leftShoulder = 0.8 + ease * (((hash * 3) % 50) / 100);
            leftElbow = 0.9 - cycle * 0.3;
        }
    }

    return { leftShoulder, leftElbow, rightShoulder, rightElbow, handGesture };
}

function renderAvatarFrame(ctx, w, h, gloss, progress) {
    ctx.clearRect(0, 0, w, h);

    // 1. Sleek Background with subtle studio gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
    bgGrad.addColorStop(0, '#0f172a');
    bgGrad.addColorStop(1, '#020617');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // Subtle grid/pedestal glow
    ctx.beginPath();
    ctx.ellipse(w / 2, h - 25, 70, 15, 0, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(59, 130, 246, 0.15)';
    ctx.fill();

    const cx = w / 2;
    const cy = h * 0.36;

    // Get smooth angles
    const angles = getAvatarJointAngles(gloss, progress);

    // 2. Torso & Clothes (Stylized modern 3D-effect aesthetic)
    ctx.fillStyle = '#1e293b';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;

    // Torso path
    ctx.beginPath();
    ctx.moveTo(cx - 28, cy + 30);
    ctx.lineTo(cx + 28, cy + 30);
    ctx.lineTo(cx + 22, cy + 85);
    ctx.lineTo(cx - 22, cy + 85);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Neck
    ctx.fillStyle = '#fbcfe8';
    ctx.fillRect(cx - 6, cy + 12, 12, 20);

    // 3. Head & Face
    // Head shape
    ctx.beginPath();
    ctx.arc(cx, cy, 20, 0, Math.PI * 2);
    ctx.fillStyle = '#fed7aa';
    ctx.fill();
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Hair
    ctx.beginPath();
    ctx.arc(cx, cy - 4, 21, Math.PI, 0);
    ctx.fillStyle = '#334155';
    ctx.fill();

    // Eyes
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.arc(cx - 7, cy - 2, 2.5, 0, Math.PI * 2);
    ctx.arc(cx + 7, cy - 2, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Friendly Smile
    ctx.strokeStyle = '#c2410c';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.arc(cx, cy + 4, 7, 0.15 * Math.PI, 0.85 * Math.PI);
    ctx.stroke();

    // 4. Arms & Hands with forward kinematics
    const upperArmLen = 28;
    const foreArmLen = 28;

    // Left Arm (Screen Left)
    const lShoulderX = cx - 28;
    const lShoulderY = cy + 30;
    const lElbowX = lShoulderX - Math.sin(angles.leftShoulder) * upperArmLen;
    const lElbowY = lShoulderY + Math.cos(angles.leftShoulder) * upperArmLen;
    const lHandX = lElbowX - Math.sin(angles.leftShoulder + angles.leftElbow) * foreArmLen;
    const lHandY = lElbowY + Math.cos(angles.leftShoulder + angles.leftElbow) * foreArmLen;

    // Right Arm (Screen Right)
    const rShoulderX = cx + 28;
    const rShoulderY = cy + 30;
    const rElbowX = rShoulderX + Math.sin(angles.rightShoulder) * upperArmLen;
    const rElbowY = rShoulderY + Math.cos(angles.rightShoulder) * upperArmLen;
    const rHandX = rElbowX + Math.sin(angles.rightShoulder + angles.rightElbow) * foreArmLen;
    const rHandY = rElbowY + Math.cos(angles.rightShoulder + angles.rightElbow) * foreArmLen;

    // Draw Left Arm
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(lShoulderX, lShoulderY);
    ctx.lineTo(lElbowX, lElbowY);
    ctx.lineTo(lHandX, lHandY);
    ctx.stroke();

    // Left Hand
    ctx.fillStyle = '#fed7aa';
    ctx.beginPath();
    ctx.arc(lHandX, lHandY, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Draw Right Arm
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(rShoulderX, rShoulderY);
    ctx.lineTo(rElbowX, rElbowY);
    ctx.lineTo(rHandX, rHandY);
    ctx.stroke();

    // Right Hand
    ctx.fillStyle = '#fed7aa';
    ctx.beginPath();
    ctx.arc(rHandX, rHandY, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#e11d48';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Active Signing Ripple Effect on hands
    if (gloss !== 'rest') {
        const pulse = (Math.sin(progress * Math.PI * 4) + 1) * 3;
        ctx.strokeStyle = 'rgba(74, 222, 128, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(rHandX, rHandY, 8 + pulse, 0, Math.PI * 2);
        ctx.stroke();
    }
}

// ================================================================
//  Conversation Log
// ================================================================

function addLog(text, type) {
    const area = document.getElementById('conversationArea');
    if (!area) return;

    const entry = document.createElement('div');
    entry.className = 'conv-log ' + (type || 'info');

    const now = new Date();
    const time = now.getHours().toString().padStart(2, '0') + ':' +
                 now.getMinutes().toString().padStart(2, '0') + ':' +
                 now.getSeconds().toString().padStart(2, '0');

    entry.innerHTML = `<span class="log-time">${time}</span> <span class="log-text">${text}</span>`;
    area.appendChild(entry);
    area.scrollTop = area.scrollHeight;

    // Keep max 100 entries
    const logs = area.querySelectorAll('.conv-log');
    while (logs.length > 100) {
        logs[0].remove();
    }
}

// ================================================================
//  Utilities
// ================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
