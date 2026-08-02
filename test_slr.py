"""Test Script: Sign Language Recognition via Webcam.

This script opens your webcam, captures video frames, runs them
through the full SLR pipeline (MediaPipe -> TGCN), and displays
the predicted ASL sign on screen.

Usage:
    python test_slr.py

Controls:
    - Press 'q' to quit
    - Press 'r' to reset the frame buffer
    - Press 'c' to clear the prediction display
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import time
from modules.slr.pose_estimator import PoseEstimator
from modules.slr.keypoint_mapper import KeypointMapper
from modules.slr.frame_buffer import FrameBuffer
from modules.slr.tgcn_inference import TGCNInference
from config.settings import (
    MODEL_CHECKPOINT_PATH, ASL100_GLOSSES,
    NUM_SAMPLES, HIDDEN_SIZE, NUM_CLASSES,
    NUM_STAGES, DROP_P, DEVICE, CONFIDENCE_THRESHOLD
)


def draw_landmarks_on_frame(frame, landmarks_dict):
    """Draw detected landmarks on the frame using OpenCV.
    
    Args:
        frame: BGR frame to draw on.
        landmarks_dict: dict with 'body', 'left_hand', 'right_hand' arrays.
    """
    h, w = frame.shape[:2]
    
    # Draw body keypoints
    if landmarks_dict.get('body') is not None:
        body = landmarks_dict['body']
        # Body connections (upper body)
        body_connections = [
            (0, 2), (0, 5),  # nose to eyes
            (2, 7), (5, 8),  # eyes to ears
            (11, 12),        # shoulders
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm
            (11, 23), (12, 24), (23, 24),  # torso
        ]
        for i, j in body_connections:
            if i < len(body) and j < len(body):
                pt1 = (int(body[i][0] * w), int(body[i][1] * h))
                pt2 = (int(body[j][0] * w), int(body[j][1] * h))
                cv2.line(frame, pt1, pt2, (0, 255, 200), 2)
        
        # Draw body points
        for i, (x, y) in enumerate(body):
            if i <= 24:  # Only upper body
                pt = (int(x * w), int(y * h))
                cv2.circle(frame, pt, 3, (0, 200, 255), -1)
    
    # Draw hand keypoints
    for hand_key, color in [('left_hand', (255, 100, 100)), ('right_hand', (100, 100, 255))]:
        hand = landmarks_dict.get(hand_key)
        if hand is not None:
            # Hand finger connections
            finger_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),     # thumb
                (0, 5), (5, 6), (6, 7), (7, 8),     # index
                (0, 9), (9, 10), (10, 11), (11, 12), # middle
                (0, 13), (13, 14), (14, 15), (15, 16), # ring
                (0, 17), (17, 18), (18, 19), (19, 20), # pinky
                (5, 9), (9, 13), (13, 17),            # palm
            ]
            for i, j in finger_connections:
                if i < len(hand) and j < len(hand):
                    pt1 = (int(hand[i][0] * w), int(hand[i][1] * h))
                    pt2 = (int(hand[j][0] * w), int(hand[j][1] * h))
                    cv2.line(frame, pt1, pt2, color, 1)
            
            for x, y in hand:
                pt = (int(x * w), int(y * h))
                cv2.circle(frame, pt, 2, color, -1)


def main():
    print("=" * 60)
    print("  Sign Language Recognition - Test")
    print("=" * 60)
    print()

    # Initialize components
    print("Initializing pose estimator...")
    pose_estimator = PoseEstimator()
    keypoint_mapper = KeypointMapper()
    frame_buffer = FrameBuffer(num_samples=NUM_SAMPLES)

    print("Loading TGCN model...")
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

    print()
    print("Opening webcam...")
    print("Controls: [q] Quit | [r] Reset buffer | [c] Clear prediction")
    print()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Make sure your webcam is connected and not used by another app.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_time = time.time()
    frame_count = 0
    current_prediction = None
    fps_display = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert BGR -> RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract pose landmarks
            mp_results = pose_estimator.process_frame(frame_rgb)
            landmarks = pose_estimator.extract_landmarks(mp_results)

            # Draw landmarks on frame
            draw_landmarks_on_frame(frame, landmarks)

            # Map to 55 WLASL keypoints
            keypoints = keypoint_mapper.map_landmarks(landmarks)

            if keypoints is not None:
                frame_buffer.add_frame(keypoints)

            # Run prediction if buffer is ready
            if frame_buffer.is_ready():
                input_tensor = frame_buffer.get_tensor()
                prediction = tgcn.predict(input_tensor)

                if prediction['confidence'] >= CONFIDENCE_THRESHOLD:
                    current_prediction = prediction
                    # Print to console
                    print(f"  Predicted: {prediction['gloss'].upper():>15s}  "
                          f"(confidence: {prediction['confidence']:.1%})  "
                          f"| Top-3: {', '.join(f'{g}({c:.0%})' for g, c in prediction['top5'][:3])}")

            # ----- Draw UI Overlay -----

            # Draw prediction
            if current_prediction and current_prediction.get('gloss'):
                text = f"{current_prediction['gloss'].upper()}"
                conf_text = f"Confidence: {current_prediction['confidence']:.1%}"

                # Main prediction - large green text
                cv2.putText(frame, text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, conf_text, (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

                # Top-5 predictions sidebar
                if current_prediction.get('top5'):
                    y_start = 130
                    cv2.putText(frame, "Top-5 Predictions:", (10, y_start),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    for i, (gloss, conf) in enumerate(current_prediction['top5']):
                        y = y_start + 25 + i * 22
                        color = (0, 255, 0) if i == 0 else (180, 180, 180)
                        cv2.putText(frame, f"{i + 1}. {gloss}: {conf:.1%}", (15, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw buffer fill bar
            fill = frame_buffer.get_fill_ratio()
            bar_x, bar_y = 10, frame.shape[0] - 40
            bar_width = 300
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (60, 60, 60), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * fill), bar_y + 20), (0, 200, 0), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (120, 120, 120), 1)
            cv2.putText(frame, f"Buffer: {fill:.0%} ({len(frame_buffer)}/50)",
                        (bar_x + 5, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # FPS counter
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_time
                fps_display = 10 / elapsed if elapsed > 0 else 0
                fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps_display:.0f}", (frame.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Show frame
            cv2.imshow('Sign Language Recognition', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                frame_buffer.clear()
                current_prediction = None
                print("  [Buffer reset]")
            elif key == ord('c'):
                current_prediction = None
                print("  [Prediction cleared]")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_estimator.close()
        print("Webcam released. Done.")


if __name__ == '__main__':
    main()
