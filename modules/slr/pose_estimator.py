"""MediaPipe Holistic Pose Estimator (New Tasks API).

Extracts body and hand landmarks from webcam frames using
Google's MediaPipe HolisticLandmarker (Tasks API v0.10.35+).

The legacy mp.solutions.holistic API has been removed in recent
MediaPipe versions. This module uses the new mp.tasks.vision API.
"""
import os
import numpy as np

import mediapipe as mp

# Access through mp.tasks namespace (direct submodule import doesn't work in 0.10.35)
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions

# Path to the holistic landmarker model file
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "holistic_landmarker.task"
)


class PoseEstimator:
    """Wraps MediaPipe HolisticLandmarker to extract body + hand landmarks.

    MediaPipe Holistic provides:
    - 33 body (pose) landmarks
    - 21 left hand landmarks
    - 21 right hand landmarks

    We extract these and pass them to the KeypointMapper for conversion
    to the 55-keypoint WLASL format.
    """

    def __init__(self, model_path=None, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        model_path = model_path or _DEFAULT_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                f"Download it from: https://storage.googleapis.com/mediapipe-models/"
                f"holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
            )

        base_options = BaseOptions(model_asset_path=model_path)

        options = HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_landmarks_confidence=min_detection_confidence,
            min_hand_landmarks_confidence=min_detection_confidence,
        )

        self.landmarker = HolisticLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process_frame(self, frame_rgb):
        """Process a single RGB frame and extract landmarks.

        Args:
            frame_rgb: numpy array of shape (H, W, 3) in RGB format.
                       Note: OpenCV reads in BGR, so convert before calling.

        Returns:
            results: HolisticLandmarkerResult containing:
                - results.pose_landmarks (list of 33 body landmarks)
                - results.left_hand_landmarks (list of 21 left hand landmarks)
                - results.right_hand_landmarks (list of 21 right hand landmarks)
        """
        # Create MediaPipe Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Increment timestamp for VIDEO mode
        self._timestamp_ms += 33  # ~30 FPS

        results = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        return results

    def extract_landmarks(self, results):
        """Extract raw landmark coordinates from MediaPipe results.

        Args:
            results: HolisticLandmarkerResult object.

        Returns:
            dict with keys 'body', 'left_hand', 'right_hand'.
            Each value is a numpy array of shape (N, 2) with (x, y) coordinates
            normalized to [0, 1], or None if that body part wasn't detected.
            - body: (33, 2)
            - left_hand: (21, 2)
            - right_hand: (21, 2)
        """
        landmarks = {
            'body': None,
            'left_hand': None,
            'right_hand': None
        }

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            body = []
            for lm in results.pose_landmarks:
                body.append([lm.x, lm.y])
            landmarks['body'] = np.array(body, dtype=np.float32)

        if results.left_hand_landmarks and len(results.left_hand_landmarks) > 0:
            left = []
            for lm in results.left_hand_landmarks:
                left.append([lm.x, lm.y])
            landmarks['left_hand'] = np.array(left, dtype=np.float32)

        if results.right_hand_landmarks and len(results.right_hand_landmarks) > 0:
            right = []
            for lm in results.right_hand_landmarks:
                right.append([lm.x, lm.y])
            landmarks['right_hand'] = np.array(right, dtype=np.float32)

        return landmarks

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'landmarker') and self.landmarker:
            self.landmarker.close()

    def __del__(self):
        self.close()
