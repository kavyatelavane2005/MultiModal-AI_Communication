"""Sign Language Recognizer — High-Level API.

Combines pose estimation, keypoint mapping, frame buffering,
and SPOTER inference into a single easy-to-use class.

Usage:
    recognizer = SignRecognizer()
    recognizer.load_model()
    
    # In your webcam loop:
    result = recognizer.process_frame(frame_bgr)
    if result:
        print(f"Detected: {result['gloss']} ({result['confidence']:.2f})")
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

from .pose_estimator import PoseEstimator
from .keypoint_mapper import KeypointMapper
from .frame_buffer import FrameBuffer
from .spoter_inference import SPOTERInference
from config.settings import (
    MODEL_CHECKPOINT_PATH, ASL100_GLOSSES,
    NUM_SAMPLES, NUM_KEYPOINTS, HIDDEN_DIM, NUM_CLASSES,
    DEVICE, CONFIDENCE_THRESHOLD
)


class SignRecognizer:
    """End-to-end sign language recognition pipeline using pre-trained SPOTER Transformer."""

    def __init__(self, checkpoint_path=None, confidence_threshold=None):
        """
        Args:
            checkpoint_path: Path to model checkpoint. Uses default from settings if None.
            confidence_threshold: Minimum confidence to return a prediction.
        """
        self.checkpoint_path = checkpoint_path or MODEL_CHECKPOINT_PATH
        self.confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD
        
        # Initialize components
        self.pose_estimator = PoseEstimator()
        self.keypoint_mapper = KeypointMapper()
        self.frame_buffer = FrameBuffer(num_samples=NUM_SAMPLES, num_keypoints=NUM_KEYPOINTS)
        self.spoter = None  # Loaded lazily
        
        self._frame_count = 0
        self._last_result = None

    def load_model(self):
        """Load the pre-trained SPOTER model."""
        print("Initializing Sign Language Recognition with SPOTER...")
        self.spoter = SPOTERInference(
            checkpoint_path=self.checkpoint_path,
            gloss_list=ASL100_GLOSSES,
            num_samples=NUM_SAMPLES,
            num_keypoints=NUM_KEYPOINTS,
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
            device=DEVICE
        )
        print("Sign Language Recognition ready!")

    def process_frame(self, frame_bgr):
        """Process a single BGR frame from the webcam.
        
        Args:
            frame_bgr: numpy array of shape (H, W, 3) in BGR format (from OpenCV).
        
        Returns:
            dict with prediction results if a sign is detected:
                'gloss': predicted sign name
                'confidence': prediction confidence  
                'top5': list of (gloss, confidence) tuples
                'buffer_fill': how full the frame buffer is (0.0-1.0)
            Returns None if no prediction can be made yet.
        """
        if self.spoter is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Step 1: Convert BGR -> RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Step 2: Extract pose landmarks
        results = self.pose_estimator.process_frame(frame_rgb)
        landmarks = self.pose_estimator.extract_landmarks(results)

        # Step 3: Map to 54 SPOTER keypoints
        keypoints_54 = self.keypoint_mapper.map_landmarks(landmarks)

        if keypoints_54 is not None:
            # Step 4: Add to frame buffer
            self.frame_buffer.add_frame(keypoints_54)

        self._frame_count += 1
        
        # Step 5: Run prediction if buffer is ready
        if self.frame_buffer.is_ready():
            input_tensor = self.frame_buffer.get_tensor()
            result = self.spoter.predict(input_tensor)
            
            result['buffer_fill'] = 1.0
            
            # Only return if confidence is above threshold
            if result['confidence'] >= self.confidence_threshold:
                self._last_result = result
                return result

        # Return buffer status even if no prediction yet
        return {
            'gloss': None,
            'confidence': 0.0,
            'top5': [],
            'buffer_fill': self.frame_buffer.get_fill_ratio()
        }

    def get_last_result(self):
        """Get the most recent prediction result."""
        return self._last_result

    def reset(self):
        """Clear the frame buffer and reset state."""
        self.frame_buffer.clear()
        self._frame_count = 0
        self._last_result = None

    def close(self):
        """Release resources."""
        self.pose_estimator.close()
