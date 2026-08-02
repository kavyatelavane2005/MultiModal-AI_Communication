"""Keypoint Mapper: MediaPipe landmarks -> WLASL 55-keypoint format.

The WLASL TGCN model expects 55 keypoints:
  - 13 upper body keypoints (from OpenPose, lower body excluded)
  - 21 left hand keypoints
  - 21 right hand keypoints

MediaPipe provides:
  - 33 body landmarks
  - 21 left hand landmarks  
  - 21 right hand landmarks

This module maps MediaPipe's body landmarks to the 13 upper body
keypoints that match OpenPose's format, and passes through the
hand landmarks directly (same 21-point convention).
"""
import numpy as np


# ============================================================
# OpenPose Body25 -> Upper Body 13 Keypoint Mapping
# ============================================================
# OpenPose Body25 keypoints (after excluding lower body indices
# {9,10,11,12,13,14,19,20,21,22,23,24}), the remaining 13 are:
#
# Index | OpenPose Name     | MediaPipe Equivalent
# ------|-------------------|---------------------
#   0   | Nose              | 0  (nose)
#   1   | Neck              | midpoint(11, 12) — left_shoulder, right_shoulder
#   2   | RShoulder         | 12 (right_shoulder)
#   3   | RElbow            | 14 (right_elbow)
#   4   | RWrist            | 16 (right_wrist)
#   5   | LShoulder         | 11 (left_shoulder)
#   6   | LElbow            | 13 (left_elbow)
#   7   | LWrist            | 15 (left_wrist)
#   8   | MidHip            | midpoint(23, 24) — left_hip, right_hip
#   9   | REye              | 5  (right_eye)
#  10   | LEye              | 2  (left_eye)
#  11   | REar              | 8  (right_ear)
#  12   | LEar              | 7  (left_ear)

# MediaPipe body landmark indices for each OpenPose upper body keypoint
MEDIAPIPE_BODY_MAPPING = {
    0: [0],        # Nose
    1: [11, 12],   # Neck = midpoint of shoulders
    2: [12],       # RShoulder
    3: [14],       # RElbow
    4: [16],       # RWrist
    5: [11],       # LShoulder
    6: [13],       # LElbow
    7: [15],       # LWrist
    8: [23, 24],   # MidHip = midpoint of hips
    9: [5],        # REye
    10: [2],       # LEye
    11: [8],       # REar
    12: [7],       # LEar
}


class KeypointMapper:
    """Maps MediaPipe landmarks to the 55-keypoint WLASL format.
    
    Output order: [13 body keypoints] + [21 left hand] + [21 right hand] = 55 keypoints
    Each keypoint has (x, y) coordinates normalized to [-1, 1].
    """

    def __init__(self):
        self.num_body_keypoints = 13
        self.num_hand_keypoints = 21
        self.total_keypoints = 55  # 13 + 21 + 21

    def map_landmarks(self, landmarks_dict):
        """Convert MediaPipe landmarks to WLASL 55-keypoint format.
        
        Args:
            landmarks_dict: dict with keys 'body', 'left_hand', 'right_hand',
                           each containing a numpy array of shape (N, 2) with
                           (x, y) coordinates in [0, 1] range, or None.
        
        Returns:
            numpy array of shape (55, 2) with (x, y) coordinates in [-1, 1] range.
            Returns None if body landmarks are not available.
        """
        body = landmarks_dict.get('body')
        left_hand = landmarks_dict.get('left_hand')
        right_hand = landmarks_dict.get('right_hand')

        # Body landmarks are required (we need at least the torso)
        if body is None:
            return None

        # Map body landmarks: 33 MediaPipe -> 13 upper body
        body_mapped = np.zeros((self.num_body_keypoints, 2), dtype=np.float32)
        for openpose_idx, mediapipe_indices in MEDIAPIPE_BODY_MAPPING.items():
            if len(mediapipe_indices) == 1:
                body_mapped[openpose_idx] = body[mediapipe_indices[0]]
            else:
                # Compute midpoint for composite keypoints (neck, midhip)
                body_mapped[openpose_idx] = np.mean(
                    body[mediapipe_indices], axis=0
                )

        # Hand landmarks: use directly if available, zeros if not
        if left_hand is not None:
            left_mapped = left_hand.copy()
        else:
            left_mapped = np.zeros((self.num_hand_keypoints, 2), dtype=np.float32)

        if right_hand is not None:
            right_mapped = right_hand.copy()
        else:
            right_mapped = np.zeros((self.num_hand_keypoints, 2), dtype=np.float32)

        # Concatenate: body(13) + left_hand(21) + right_hand(21) = 55
        keypoints_55 = np.concatenate(
            [body_mapped, left_mapped, right_mapped], axis=0
        )

        # Normalize from [0, 1] to [-1, 1]
        # MediaPipe gives [0, 1], WLASL expects [-1, 1]
        # Formula: 2 * val - 1
        keypoints_55 = 2.0 * keypoints_55 - 1.0

        return keypoints_55
