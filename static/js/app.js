/**
 * MultiModal Bridge — Client-Side Application
 *
 * Implements the full communication workflow:
 *   1. Session setup (role selection)
 *   2. Input acquisition (sign / speech / text)
 *   3. Message processing (via server + LLM)
 *   4. Output routing (text / TTS)
 *   5. Session loop (continue / end)
 */

// ================================================================
//  State
// ================================================================

let socket = null;
let sessionActive = false;
let currentInputMode = 'sign';
let isInputActive = false;
let collectedSignCount = 0;

// SLR state
let webcamStream = null;
let frameInterval = null;

// STT state
let recognition = null;

// Session config
let userARole = 'normal';
let userBRole = 'normal';
let availableInputs = ['sign', 'speech', 'text'];
let outputModes = ['text', 'tts'];

let selectedCameraId = localStorage.getItem('preferredCameraId') || null;
let isSendingFrame = false;
let offscreenCanvas = null;

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
    socket.on('landmarks', drawLandmarks);

    socket.on('buffer_fill', (data) => {
        updateBufferBar(data.fill);
    });

    socket.on('slr_prediction', (data) => {
        showDetection(data);
    });

    socket.on('sentence_preview', (data) => {
        const glosses = data.glosses || [];
        const preview = document.getElementById('sentencePreview');
        const button = document.getElementById('btnFinishSentence');
        if (!preview || !button) return;

        collectedSignCount = glosses.length;
        preview.textContent = glosses.length
            ? `Collected signs: ${glosses.join(' ')} · pause briefly between signs; Send ends the current sign, or auto-send after 5 seconds idle`
            : 'Collected signs will appear here · pause briefly between signs; Send ends the current sign, or auto-send after 5 seconds idle';
        updateSentenceSendButton();
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
    const roleLabels = { deaf: 'Deaf', blind: 'Blind', mute: 'Mute', normal: 'Normal' };
    document.getElementById('sessionLabel').textContent =
        (roleLabels[data.user_a_role] || data.user_a_role) + ' → ' +
        (roleLabels[data.user_b_role] || data.user_b_role);
    document.getElementById('sessionBadge').textContent = 'Active';

    // Configure input tabs
    configureInputTabs(availableInputs);

    // Configure output badges
    configureOutputBadges(outputModes);

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
            <div class="conv-welcome-icon" aria-hidden="true">MB</div>
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
    const labels = { text: 'Text', tts: 'Voice' };
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
    updateSentenceSendButton();
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
    updateSentenceSendButton();
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
        width: { ideal: 480 },
        height: { ideal: 360 },
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
            }, 120);
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
                        }, 120);
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
    isSendingFrame = false;

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
    // Drop frame if server is still processing the previous one (eliminates lag accumulation)
    if (isSendingFrame) return;
    if (!video || !video.videoWidth || !video.videoHeight) return;

    // Resize before encoding to reduce CPU work and network payload size.
    if (!offscreenCanvas) {
        offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = 320;
        offscreenCanvas.height = 240;
    }

    const ctx = offscreenCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

    isSendingFrame = true;

    const dataUrl = offscreenCanvas.toDataURL('image/jpeg', 0.45);
    socket.emit('video_frame', { image: dataUrl }, () => {
        isSendingFrame = false;
    });
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

function finishSignSentence() {
    if (socket && socket.connected) {
        socket.emit('finish_sentence');
    }
}

function updateSentenceSendButton() {
    const button = document.getElementById('btnFinishSentence');
    if (button) {
        button.disabled = collectedSignCount === 0 &&
            !(sessionActive && isInputActive && currentInputMode === 'sign');
    }
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
    const { original_text, processed_text, source, outputs, llm_used } = data;

    // Remove welcome message if present
    const welcome = document.querySelector('.conv-welcome');
    if (welcome) welcome.remove();

    const area = document.getElementById('conversationArea');

    // Source label
    const sourceLabels = { sign: 'Sign', speech: 'Speech', text: 'Text' };
    const sourceLabel = sourceLabels[source] || source;

    // Create message card
    const msg = document.createElement('div');
    msg.className = 'conv-message';
    msg.innerHTML = `
        <div class="conv-source">${sourceLabel}</div>
        <div class="conv-original">${escapeHtml(original_text)}</div>
        ${processed_text !== original_text
            ? `<div class="conv-processed">
                 <span class="conv-processed-label">${llm_used ? 'AI-formed sentence:' : 'Fallback text:'}</span>
                 ${escapeHtml(processed_text)}
               </div>`
            : ''
        }
        <div class="conv-outputs">
            ${outputs.show_text ? '<span class="conv-out-badge text">Text</span>' : ''}
            ${outputs.speak_text ? '<span class="conv-out-badge tts">Spoken</span>' : ''}
        </div>
    `;
    area.appendChild(msg);
    area.scrollTop = area.scrollHeight;

    // TTS Output
    if (outputs.speak_text && outputs.tts_text) {
        speakText(outputs.tts_text);
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
