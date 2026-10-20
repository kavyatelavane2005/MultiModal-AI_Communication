"""Flask Web Application for Multi-Modal Communication System.

Implements the full workflow:
  1. Session Setup — User A & B select roles (Deaf/Blind/Mute/Normal)
  2. Input Acquisition — Sign language, Speech, Text
  3. Recognition — MediaPipe + SPOTER for sign, Web Speech API for speech
  4. Text Common Layer — Unified text representation
  5. LLM Processing — Grammar, sentences, intent, context
  6. Output Decision — Route based on User B's profile
  7. Output — Text display, Voice TTS, Sign avatar
  8. Session Loop — Continue or End
"""
import sys
import os
import base64
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

from modules.slr.pose_estimator import PoseEstimator
from modules.slr.keypoint_mapper import KeypointMapper
from modules.slr.frame_buffer import FrameBuffer
from modules.slr.spoter_inference import SPOTERInference
from modules.slr.sentence_builder import SentenceBuilder
from modules.llm.translator import ASLTranslator
from modules.common.session_manager import Session
from modules.common.text_layer import TextCommonLayer
from modules.common.output_router import OutputRouter
from config.settings import (
    MODEL_CHECKPOINT_PATH, ASL100_GLOSSES,
    NUM_SAMPLES, NUM_KEYPOINTS, HIDDEN_DIM, NUM_CLASSES,
    DEVICE, CONFIDENCE_THRESHOLD
)

# ---- Flask App ----
app = Flask(__name__)
app.config['SECRET_KEY'] = 'multimodal-comm-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ---- Global SLR Components (initialized once at startup) ----
pose_estimator = None
keypoint_mapper = None
spoter = None
translator = None

# ---- Per-Session State ----
session = None          # Current Session object
frame_buffer = None     # SLR frame buffer
sentence_builder = None # Gloss sequence builder
text_layer = None       # Text common layer
output_router = None    # Output decision module
slr_active = False      # Whether SLR camera is streaming


def init_models():
    """Initialize ML models (done once at startup)."""
    global pose_estimator, keypoint_mapper, spoter, translator

    print("Loading ML models...")
    pose_estimator = PoseEstimator()
    keypoint_mapper = KeypointMapper()

    spoter = SPOTERInference(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        gloss_list=ASL100_GLOSSES,
        num_samples=NUM_SAMPLES,
        num_keypoints=NUM_KEYPOINTS,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        device=DEVICE
    )

    translator = ASLTranslator()
    print("ML models loaded!")


def init_session_components():
    """Initialize per-session components (called when session starts)."""
    global frame_buffer, sentence_builder, text_layer
    frame_buffer = FrameBuffer(num_samples=NUM_SAMPLES)
    sentence_builder = SentenceBuilder(pause_threshold_sec=3.0)
    text_layer = TextCommonLayer()


# ================================================================
#  Routes
# ================================================================

@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Return system status."""
    return jsonify({
        'models_ready': spoter is not None,
        'session_active': session is not None and session.active,
        'glosses': ASL100_GLOSSES,
        'num_classes': NUM_CLASSES,
    })


# ================================================================
#  WebSocket Events — Connection
# ================================================================

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('system_message', {'text': 'Connected to server', 'type': 'success'})


@socketio.on('disconnect')
def handle_disconnect():
    global slr_active, session
    slr_active = False
    print("Client disconnected")


# ================================================================
#  WebSocket Events — Session Management
# ================================================================

@socketio.on('start_session')
def handle_start_session(data):
    """Start a new communication session.
    
    Expects: { user_a_role: str, user_b_role: str }
    """
    global session, output_router, slr_active

    user_a_role = data.get('user_a_role', 'normal').lower()
    user_b_role = data.get('user_b_role', 'normal').lower()

    try:
        session = Session(user_a_role, user_b_role)
        output_router = OutputRouter(user_b_role)
        init_session_components()
        slr_active = False

        print(f"Session started: {user_a_role} → {user_b_role}")
        emit('session_started', session.to_dict())
        emit('system_message', {
            'text': f'Session started: {user_a_role.title()} → {user_b_role.title()}',
            'type': 'success'
        })
    except ValueError as e:
        emit('system_message', {'text': str(e), 'type': 'error'})


@socketio.on('end_session')
def handle_end_session():
    """End the current session."""
    global session, slr_active, output_router

    if session:
        session.end()
    slr_active = False
    output_router = None

    print("Session ended")
    emit('session_ended', {})
    emit('system_message', {'text': 'Session ended', 'type': 'info'})


@socketio.on('swap_roles')
def handle_swap_roles():
    """Swap roles between User A and User B (two-way turn)."""
    global session, output_router, slr_active
    if not session or not session.active:
        return

    session.swap_roles()
    output_router = OutputRouter(session.user_b_role)
    init_session_components()
    slr_active = False

    print(f"Roles swapped: {session.user_a_role} → {session.user_b_role}")
    emit('session_started', session.to_dict())
    emit('system_message', {
        'text': f'Roles swapped! Sender is now {session.user_a_role.title()}, Receiver is {session.user_b_role.title()}',
        'type': 'success'
    })


# ================================================================
#  WebSocket Events — API Key
# ================================================================

@socketio.on('set_api_key')
def handle_set_api_key(data):
    """Set the API key for the LLM translator."""
    api_key = data.get('api_key', '')
    if api_key and translator:
        translator.setup(api_key)
        emit('system_message', {'text': 'API Key configured successfully', 'type': 'success'})


# ================================================================
#  WebSocket Events — Sign Language Recognition
# ================================================================

@socketio.on('start_slr')
def handle_start_slr():
    """Start sign language recognition."""
    global slr_active
    if not session or not session.active:
        emit('system_message', {'text': 'Please start a session first', 'type': 'error'})
        return

    slr_active = True
    if frame_buffer:
        frame_buffer.clear()
    emit('slr_status', {'active': True})
    print("SLR started")


@socketio.on('stop_slr')
def handle_stop_slr():
    """Stop sign language recognition."""
    global slr_active
    slr_active = False
    if frame_buffer:
        frame_buffer.clear()
    emit('slr_status', {'active': False})
    print("SLR stopped")


@socketio.on('video_frame')
def handle_video_frame(data):
    """Process a video frame for sign language recognition.
    
    Expects: { 'image': 'data:image/jpeg;base64,...' }
    """
    global slr_active

    if not slr_active or spoter is None or not session or not session.active:
        return

    try:
        # Decode base64 image
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Extract pose landmarks
        mp_results = pose_estimator.process_frame(frame_rgb)
        landmarks = pose_estimator.extract_landmarks(mp_results)

        # Send landmark data back for visualization
        landmark_data = {}
        for key in ['body', 'left_hand', 'right_hand']:
            if landmarks[key] is not None:
                landmark_data[key] = landmarks[key].tolist()

        if landmark_data:
            emit('landmarks', landmark_data)

        # Map to keypoints and buffer
        keypoints = keypoint_mapper.map_landmarks(landmarks)
        if keypoints is not None:
            frame_buffer.add_frame(keypoints)

        # Run prediction if buffer ready
        fill = frame_buffer.get_fill_ratio()
        emit('buffer_fill', {'fill': fill})

        if frame_buffer.is_ready():
            input_tensor = frame_buffer.get_tensor()
            prediction = spoter.predict(input_tensor)

            if prediction['confidence'] >= CONFIDENCE_THRESHOLD:
                gloss = prediction['gloss']

                # Add to text common layer
                text_layer.add_sign_gloss(gloss)
                sentence_builder.add_word(gloss)

                emit('slr_prediction', {
                    'gloss': gloss,
                    'confidence': round(prediction['confidence'] * 100, 1),
                    'top5': [
                        {'gloss': g, 'confidence': round(c * 100, 1)}
                        for g, c in prediction['top5']
                    ]
                })

        # Check if the user paused → trigger processing
        sentence = sentence_builder.check_flush()
        if sentence:
            glosses = list(sentence)
            raw_text = ' '.join(glosses)
            emit('system_message', {'text': 'Processing sign sequence...', 'type': 'info'})
            socketio.start_background_task(run_message_processing, raw_text, 'sign', glosses)

    except Exception as e:
        print(f"Frame processing error: {e}")


@socketio.on('reset_buffer')
def handle_reset_buffer():
    """Reset the frame buffer."""
    if frame_buffer:
        frame_buffer.clear()
    if sentence_builder:
        sentence_builder.words = []
    if text_layer:
        text_layer.clear()
    emit('buffer_fill', {'fill': 0})
    emit('system_message', {'text': 'Buffer reset', 'type': 'info'})


# ================================================================
#  WebSocket Events — Speech Input (from Web Speech API)
# ================================================================

@socketio.on('speech_input')
def handle_speech_input(data):
    """Handle speech-to-text result from the browser.
    
    Expects: { text: str }
    """
    if not session or not session.active:
        return

    raw_text = data.get('text', '').strip()
    if not raw_text:
        return

    text, source = text_layer.process_speech_input(raw_text)
    if text:
        emit('system_message', {'text': 'Processing speech...', 'type': 'info'})
        socketio.start_background_task(run_message_processing, text, 'speech', None)


# ================================================================
#  WebSocket Events — Text Input (keyboard)
# ================================================================

@socketio.on('text_input')
def handle_text_input(data):
    """Handle direct text input from the keyboard.
    
    Expects: { text: str }
    """
    if not session or not session.active:
        return

    raw_text = data.get('text', '').strip()
    if not raw_text:
        return

    text, source = text_layer.process_text_input(raw_text)
    if text:
        emit('system_message', {'text': 'Processing message...', 'type': 'info'})
        socketio.start_background_task(run_message_processing, text, 'text', None)


# ================================================================
#  Message Processing Pipeline (Background Task)
# ================================================================

def run_message_processing(raw_text, source, glosses=None):
    """Process a message through the LLM and route to outputs.
    
    This is the core pipeline:
      Raw Text → LLM Processing → Output Router → Client
    
    Args:
        raw_text: The input text (from any modality).
        source: Input source ('sign', 'speech', 'text').
        glosses: Original sign glosses (for avatar, if source is 'sign').
    """
    global session, output_router

    # Step 1: LLM Processing
    processed_text = raw_text  # Default: pass through

    if translator and translator.api_key:
        try:
            if source == 'sign':
                # Sign glosses need ASL→English translation
                processed_text = translator.translate(glosses or raw_text.split())
            else:
                # Speech/text may still need grammar/cleanup
                processed_text = translator.translate_general(raw_text)
        except Exception as e:
            print(f"LLM processing error: {e}")
            processed_text = raw_text
    else:
        # No API key — for sign input, just join glosses
        if source == 'sign':
            processed_text = raw_text.title()

    # Step 2: Output Routing
    if output_router:
        outputs = output_router.route(processed_text, source, glosses)
    else:
        outputs = {
            'text': processed_text,
            'tts_text': processed_text,
            'avatar_glosses': glosses,
            'show_text': True,
            'speak_text': False,
            'show_avatar': False,
        }

    # Step 3: Save to session history
    if session and session.active:
        session.add_message(raw_text, source, processed_text, outputs)

    # Step 4: Emit to client
    socketio.emit('message_processed', {
        'original_text': raw_text,
        'processed_text': processed_text,
        'source': source,
        'outputs': outputs
    })


# ================================================================
#  Main Entry Point
# ================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Multi-Modal Communication System")
    print("=" * 60)
    print()

    init_models()

    print()
    print("Starting web server...")
    print("Open http://localhost:5000 in your browser")
    print()

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
