"""Flask Web Application for Multi-Modal Communication System.

Serves the dashboard UI and handles real-time sign language
recognition via WebSocket frame streaming.
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
from modules.slr.tgcn_inference import TGCNInference
from config.settings import (
    MODEL_CHECKPOINT_PATH, ASL100_GLOSSES,
    NUM_SAMPLES, HIDDEN_SIZE, NUM_CLASSES,
    NUM_STAGES, DROP_P, DEVICE, CONFIDENCE_THRESHOLD
)

# ---- Flask App ----
app = Flask(__name__)
app.config['SECRET_KEY'] = 'multimodal-comm-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ---- Global SLR Components ----
pose_estimator = None
keypoint_mapper = None
frame_buffer = None
tgcn = None
slr_active = False


def init_slr():
    """Initialize the Sign Language Recognition pipeline."""
    global pose_estimator, keypoint_mapper, frame_buffer, tgcn

    print("Initializing SLR pipeline...")
    pose_estimator = PoseEstimator()
    keypoint_mapper = KeypointMapper()
    frame_buffer = FrameBuffer(num_samples=NUM_SAMPLES)

    tgcn = TGCNInference(
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        gloss_list=ASL100_GLOSSES,
        num_samples=NUM_SAMPLES,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        num_stages=NUM_STAGES,
        drop_p=DROP_P,
        device=DEVICE
    )
    print("SLR pipeline ready!")


# ---- Routes ----

@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Return system status."""
    return jsonify({
        'slr_ready': tgcn is not None,
        'slr_active': slr_active,
        'glosses': ASL100_GLOSSES,
        'num_classes': NUM_CLASSES,
    })


# ---- WebSocket Events ----

@socketio.on('connect')
def handle_connect():
    print(f"Client connected")
    emit('system_message', {'text': 'Connected to server', 'type': 'success'})


@socketio.on('disconnect')
def handle_disconnect():
    global slr_active
    slr_active = False
    print(f"Client disconnected")


@socketio.on('start_slr')
def handle_start_slr():
    """Start sign language recognition."""
    global slr_active
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
    """Process a video frame from the browser.

    Expects: { 'image': 'data:image/jpeg;base64,...' }
    """
    global slr_active

    if not slr_active or tgcn is None:
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

        # Map to 55 keypoints and buffer
        keypoints = keypoint_mapper.map_landmarks(landmarks)
        if keypoints is not None:
            frame_buffer.add_frame(keypoints)

        # Run prediction if buffer ready
        fill = frame_buffer.get_fill_ratio()
        emit('buffer_fill', {'fill': fill})

        if frame_buffer.is_ready():
            input_tensor = frame_buffer.get_tensor()
            prediction = tgcn.predict(input_tensor)

            if prediction['confidence'] >= CONFIDENCE_THRESHOLD:
                emit('slr_prediction', {
                    'gloss': prediction['gloss'],
                    'confidence': round(prediction['confidence'] * 100, 1),
                    'top5': [
                        {'gloss': g, 'confidence': round(c * 100, 1)}
                        for g, c in prediction['top5']
                    ]
                })

    except Exception as e:
        print(f"Frame processing error: {e}")


@socketio.on('reset_buffer')
def handle_reset_buffer():
    """Reset the frame buffer."""
    if frame_buffer:
        frame_buffer.clear()
    emit('buffer_fill', {'fill': 0})
    emit('system_message', {'text': 'Buffer reset', 'type': 'info'})


# ---- Main ----

if __name__ == '__main__':
    print("=" * 60)
    print("  Multi-Modal Communication System")
    print("=" * 60)
    print()

    init_slr()

    print()
    print("Starting web server...")
    print("Open http://localhost:5000 in your browser")
    print()

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
