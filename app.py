"""Flask Web Application for Multi-Modal Communication System.

Implements the full workflow:
  1. Session Setup — User A & B select roles (Deaf/Blind/Mute/Normal)
  2. Input Acquisition — Sign language, Speech, Text
  3. Recognition — MediaPipe + TGCN for sign, Web Speech API for speech
  4. Text Common Layer — Unified text representation
  5. LLM Processing — Grammar, sentences, intent, context
  6. Output Decision — Route based on User B's profile
  7. Output — Text display and Voice TTS
  8. Session Loop — Continue or End
"""
import sys
import os
import base64
import time
import threading
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

from modules.slr.pose_estimator import PoseEstimator
from modules.slr.keypoint_mapper import KeypointMapper
from modules.slr.frame_buffer import FrameBuffer
from modules.slr.tgcn_inference import TGCNInference
from modules.slr.sentence_builder import SentenceBuilder
from modules.llm.translator import ASLTranslator
from modules.common.session_manager import Session
from modules.common.text_layer import TextCommonLayer
from modules.common.output_router import OutputRouter
from config.settings import (
    MODEL_CHECKPOINT_PATH, ASL100_GLOSSES,
    NUM_SAMPLES, NUM_CLASSES, HIDDEN_SIZE, NUM_STAGES, DROP_P,
    DEVICE, CONFIDENCE_THRESHOLD
)

# ---- Flask App ----
app = Flask(__name__)
app.config['SECRET_KEY'] = 'multimodal-comm-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ---- Global SLR Components (initialized once at startup) ----
pose_estimator = None
keypoint_mapper = None
tgcn = None
tgcn_error = None
translator = None

# ---- Per-Session State ----
session = None          # Current Session object
frame_buffer = None     # SLR frame buffer
sentence_builder = None # Gloss sequence builder
sentence_lock = threading.Lock()
text_layer = None       # Text common layer
output_router = None    # Output decision module
slr_active = False      # Whether SLR camera is streaming


def init_models():
    """Initialize ML models (done once at startup)."""
    global pose_estimator, keypoint_mapper, tgcn, tgcn_error, translator

    print("Loading ML models...")
    pose_estimator = PoseEstimator()
    keypoint_mapper = KeypointMapper()

    try:
        tgcn = TGCNInference(
            checkpoint_path=MODEL_CHECKPOINT_PATH,
            gloss_list=ASL100_GLOSSES,
            num_samples=NUM_SAMPLES,
            hidden_size=HIDDEN_SIZE,
            num_classes=NUM_CLASSES,
            num_stages=NUM_STAGES,
            drop_p=DROP_P,
            device=DEVICE,
        )
        tgcn_error = None
    except Exception as error:
        # Keep text and speech modes available if the TGCN checkpoint is absent.
        tgcn = None
        tgcn_error = str(error)
        print(f"Sign recognition unavailable: {tgcn_error}")

    translator = ASLTranslator()
    print("ML services initialized!")


def init_session_components():
    """Initialize per-session components (called when session starts)."""
    global frame_buffer, sentence_builder, text_layer
    frame_buffer = FrameBuffer(num_samples=NUM_SAMPLES)
    sentence_builder = SentenceBuilder(pause_threshold_sec=5.0)
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
        'models_ready': tgcn is not None,
        'slr_model': 'TGCN WLASL-100',
        'model_error': tgcn_error,
        'session_active': session is not None and session.active,
        'llm_configured': bool(translator and translator.api_key),
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
    if translator and translator.api_key:
        emit('system_message', {
            'text': 'Gemini API key loaded from the local environment. It will be checked when you send a message.',
            'type': 'info'
        })
    else:
        emit('system_message', {
            'text': 'Gemini API key is not configured. Add GEMINI_API_KEY to your local .env file and restart the app.',
            'type': 'info'
        })
    if tgcn_error:
        emit('system_message', {'text': tgcn_error, 'type': 'error'})


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
        emit('sentence_preview', {'glosses': []})

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
    emit('sentence_preview', {'glosses': []})

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
    emit('sentence_preview', {'glosses': []})

    print(f"Roles swapped: {session.user_a_role} → {session.user_b_role}")
    emit('session_started', session.to_dict())
    emit('system_message', {
        'text': f'Roles swapped! Sender is now {session.user_a_role.title()}, Receiver is {session.user_b_role.title()}',
        'type': 'success'
    })


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
    if tgcn is None:
        emit('system_message', {
            'text': tgcn_error or 'TGCN WLASL-100 weights are not loaded.',
            'type': 'error'
        })
        return

    slr_active = True
    with sentence_lock:
        if frame_buffer:
            frame_buffer.clear()
    emit('slr_status', {'active': True})
    print("SLR started")


@socketio.on('stop_slr')
def handle_stop_slr():
    """Stop sign language recognition."""
    global slr_active
    slr_active = False
    with sentence_lock:
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

    if not slr_active or tgcn is None or not session or not session.active:
        return {'processed': False}

    try:
        # Decode base64 image
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return {'processed': False}

        prediction = None
        pending_glosses = None
        sentence = None
        stale_segment_discarded = False

        # Serialize capture, prediction, and flush decisions. In particular,
        # a Send click cannot flush between prediction and adding its gloss.
        with sentence_lock:
            stale_segment_discarded = frame_buffer.expire_stale_segment()
            sign_was_in_progress = frame_buffer.is_sign_in_progress()

            # Convert BGR -> RGB and extract pose landmarks.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_results = pose_estimator.process_frame(frame_rgb)
            landmarks = pose_estimator.extract_landmarks(mp_results)

            landmark_data = {}
            for key in ['body', 'left_hand', 'right_hand']:
                if landmarks[key] is not None:
                    landmark_data[key] = landmarks[key].tolist()

            keypoints = keypoint_mapper.map_landmarks(landmarks)
            if keypoints is not None:
                frame_buffer.add_frame(keypoints)

            fill = frame_buffer.get_fill_ratio()

            if frame_buffer.is_ready():
                input_tensor = frame_buffer.get_tgcn_tensor()
                prediction = tgcn.predict(input_tensor)

                if prediction['confidence'] >= CONFIDENCE_THRESHOLD:
                    gloss = prediction['gloss']
                    text_layer.add_sign_gloss(gloss)
                    sentence_builder.add_word(gloss)
                    pending_glosses = list(sentence_builder.words)

            # A pause flush cannot win while a sign is still being segmented.
            sign_in_progress = frame_buffer.is_sign_in_progress()
            if sign_was_in_progress and not sign_in_progress:
                # Reset the sentence pause even when this clip was too uncertain
                # to add as a gloss, so it cannot flush the previous word at once.
                sentence_builder.mark_activity()
            if not sign_in_progress:
                sentence = sentence_builder.check_flush()

        # Send empty landmark updates too, so tracking loss clears the overlay.
        emit('landmarks', landmark_data)
        emit('buffer_fill', {'fill': fill})

        if stale_segment_discarded:
            emit('system_message', {
                'text': 'Sign tracking was lost; discarded that incomplete sign and kept the collected words.',
                'type': 'warning'
            })

        if pending_glosses is not None:
            emit('sentence_preview', {'glosses': pending_glosses})

        if prediction and prediction['confidence'] >= CONFIDENCE_THRESHOLD:
            emit('slr_prediction', {
                'gloss': prediction['gloss'],
                'confidence': round(prediction['confidence'] * 100, 1),
                'top5': [
                    {'gloss': g, 'confidence': round(c * 100, 1)}
                    for g, c in prediction['top5']
                ]
            })

        if sentence:
            emit('sentence_preview', {'glosses': []})
            submit_sign_sentence(sentence)

    except Exception as e:
        print(f"Frame processing error: {e}")
        return {'processed': False}

    return {'processed': True}


@socketio.on('finish_sentence')
def handle_finish_sentence():
    """Flush the current gloss sequence when the signer presses Send sentence."""
    if not sentence_builder:
        return

    prediction = None
    pending_glosses = None
    active_sign_finalized = False
    with sentence_lock:
        stale_segment_discarded = bool(frame_buffer and frame_buffer.expire_stale_segment())
        active_sign_finalized = bool(frame_buffer and frame_buffer.is_sign_in_progress())

        # An explicit Send action marks the end of the current sign as well.
        # Predict it before flushing so the last sign is not left in the buffer.
        if active_sign_finalized and tgcn is not None:
            frame_buffer.force_finalize()
            if frame_buffer.is_ready():
                prediction = tgcn.predict(frame_buffer.get_tgcn_tensor())
                if prediction['confidence'] >= CONFIDENCE_THRESHOLD:
                    gloss = prediction['gloss']
                    text_layer.add_sign_gloss(gloss)
                    sentence_builder.add_word(gloss)
                    pending_glosses = list(sentence_builder.words)

        sentence = sentence_builder.force_flush()

    if stale_segment_discarded:
        emit('system_message', {
            'text': 'Sign tracking was lost; discarded that incomplete sign and kept the collected words.',
            'type': 'warning'
        })

    if pending_glosses is not None:
        emit('sentence_preview', {'glosses': pending_glosses})
    if prediction and prediction['confidence'] >= CONFIDENCE_THRESHOLD:
        emit('slr_prediction', {
            'gloss': prediction['gloss'],
            'confidence': round(prediction['confidence'] * 100, 1),
            'top5': [
                {'gloss': g, 'confidence': round(c * 100, 1)}
                for g, c in prediction['top5']
            ]
        })

    emit('sentence_preview', {'glosses': []})
    current_sign_confident = bool(
        prediction and prediction['confidence'] >= CONFIDENCE_THRESHOLD
    )
    if active_sign_finalized and not current_sign_confident:
        warning = ('The final sign was too short or uncertain; sending only earlier detected signs.'
                   if sentence else
                   'The current sign was too short or uncertain to add; no confident sentence was collected.')
        emit('system_message', {'text': warning, 'type': 'warning'})

    if sentence:
        submit_sign_sentence(sentence)
    elif not active_sign_finalized:
        emit('system_message', {'text': 'Sign a few words before sending a sentence.', 'type': 'info'})


def submit_sign_sentence(glosses):
    """Send a complete sequence of recognized glosses through the LLM pipeline."""
    glosses = list(glosses)
    raw_text = ' '.join(glosses)
    emit('system_message', {'text': 'Processing sign sentence...', 'type': 'info'})
    socketio.start_background_task(run_message_processing, raw_text, 'sign', glosses)


@socketio.on('reset_buffer')
def handle_reset_buffer():
    """Reset the frame buffer."""
    with sentence_lock:
        if frame_buffer:
            frame_buffer.clear()
        if sentence_builder:
            sentence_builder.words = []
        if text_layer:
            text_layer.clear()
    emit('buffer_fill', {'fill': 0})
    emit('sentence_preview', {'glosses': []})
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
        glosses: Original sign glosses (for sign translation).
    """
    global session, output_router

    # Step 1: LLM Processing (with safe fallback)
    fallback = raw_text.title() if source == 'sign' else raw_text
    processed_text = fallback
    llm_used = False

    if translator and translator.api_key:
        try:
            if source == 'sign':
                candidate = translator.translate(glosses or raw_text.split())
            else:
                candidate = translator.translate_general(raw_text)

            if (candidate and not translator.last_error
                    and not candidate.startswith("Translation error:")
                    and not candidate.startswith("Error:")):
                processed_text = candidate
            else:
                print(f"[LLM] Translation unavailable; using fallback '{fallback}'")
                error_code = translator.last_error_code
                if error_code in (400, 401, 403):
                    failure_message = (
                        f'Google Gemini rejected this request ({error_code}). Check the GEMINI_API_KEY, '
                        'that the Generative Language API is enabled for its Google project, and the key restrictions.'
                    )
                elif error_code == 404:
                    failure_message = 'Google Gemini could not find the model. Check GEMINI_MODEL in the local .env file.'
                elif error_code == 408:
                    failure_message = 'Google Gemini timed out processing this request (408). Try again shortly.'
                elif error_code == 429:
                    reason = (translator.last_error_reason or '').lower()
                    if reason in {'resource_exhausted', 'quota_exceeded'}:
                        failure_message = (
                            'Google Gemini reports that its quota or rate limit is exhausted. Check the project quota '
                            'and usage in Google AI Studio, then wait or resolve the quota limit before retrying.'
                        )
                    else:
                        failure_message = (
                            'Google Gemini returned HTTP 429. Check Google AI Studio usage and project quotas; '
                            'if it is a temporary rate limit, wait briefly before sending again.'
                        )
                    if reason:
                        failure_message += f' (reason: {reason})'
                elif isinstance(error_code, int) and error_code >= 500:
                    failure_message = f'Google Gemini returned a server error ({error_code}). Try again shortly.'
                elif error_code == 'timeout':
                    failure_message = 'The Google Gemini request timed out. Check the connection and try again.'
                elif error_code == 'network':
                    failure_message = 'Could not connect to Google Gemini. Check the internet connection or firewall.'
                else:
                    failure_message = 'The Google Gemini request failed. Check the API key, model access, project quota, and network.'
                failure_message += ' Sending the recognized glosses as fallback.'
                socketio.emit('system_message', {
                    'text': failure_message,
                    'type': 'error'
                })
            llm_used = bool(candidate and not translator.last_error
                            and not candidate.startswith("Translation error:")
                            and not candidate.startswith("Error:"))
        except Exception as e:
            print(f"LLM processing error: {e}")
            processed_text = fallback
            socketio.emit('system_message', {
                'text': 'LLM processing failed; sending the recognized glosses as fallback.',
                'type': 'error'
            })
    else:
        processed_text = fallback
        socketio.emit('system_message', {
            'text': 'Gemini API key is not configured. Add GEMINI_API_KEY to the local .env file and restart the app to form sentences with the LLM.',
            'type': 'info'
        })

    # Step 2: Output Routing
    if output_router:
        outputs = output_router.route(processed_text, source, glosses)
    else:
        outputs = {
            'text': processed_text,
            'tts_text': processed_text,
            'show_text': True,
            'speak_text': False,
        }

    # Step 3: Save to session history
    if session and session.active:
        session.add_message(raw_text, source, processed_text, outputs)

    # Step 4: Emit to client
    socketio.emit('message_processed', {
        'original_text': raw_text,
        'processed_text': processed_text,
        'source': source,
        'llm_used': llm_used,
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
