"""Keypoint Mapper for SPOTER: MediaPipe landmarks -> SPOTER 54-keypoint format.

SPOTER expects 54 keypoints (108 features per frame):
  - 12 upper body keypoints
  - 21 left hand keypoints
  - 21 right hand keypoints

Body Keypoints (Indices 0..11):
  0: Nose (MediaPipe 0)
  1: Neck (Midpoint of MediaPipe Left Shoulder 11 & Right Shoulder 12)
  2: Right Eye (MediaPipe 5)
  3: Left Eye (MediaPipe 2)
  4: Right Ear (MediaPipe 8)
  5: Left Ear (MediaPipe 7)
  6: Right Shoulder (MediaPipe 12)
  7: Left Shoulder (MediaPipe 11)
  8: Right Elbow (MediaPipe 14)
  9: Left Elbow (MediaPipe 13)
  10: Right Wrist (MediaPipe 16)
  11: Left Wrist (MediaPipe 15)

Hand Keypoints:
  12..32: 21 Left Hand Keypoints (MediaPipe landmarks 0..20)
  33..53: 21 Right Hand Keypoints (MediaPipe landmarks 0..20)
"""
import numpy as np


# MediaPipe mapping to SPOTER 12 body keypoints
BODY_MAPPING = {
    0: [0],        # Nose
    1: [11, 12],   # Neck = midpoint of left & right shoulders
    2: [5],        # Right Eye
    3: [2],        # Left Eye
    4: [8],        # Right Ear
    5: [7],        # Left Ear
    6: [12],       # Right Shoulder
    7: [11],       # Left Shoulder
    8: [14],       # Right Elbow
    9: [13],       # Left Elbow
    10: [16],      # Right Wrist
    11: [15]       # Left Wrist
}


class KeypointMapper:
    """Maps MediaPipe landmarks to SPOTER 54-keypoint format and normalizes sequences."""

    def __init__(self):
        self.num_body_keypoints = 12
        self.num_hand_keypoints = 21
        self.total_keypoints = 54  # 12 body + 21 left hand + 21 right hand

    def map_landmarks(self, landmarks_dict):
        """Convert a single frame of MediaPipe landmarks to SPOTER 54-keypoint array.

        Args:
            landmarks_dict: dict with keys 'body', 'left_hand', 'right_hand',
                           each containing (N, 2) array of coordinates in [0, 1] range.

        Returns:
            numpy array of shape (54, 2) in [0, 1] range, or None if body is missing.
        """
        body = landmarks_dict.get('body')
        left_hand = landmarks_dict.get('left_hand')
        right_hand = landmarks_dict.get('right_hand')

        if body is None:
            return None

        # 1. Map 12 body landmarks
        body_mapped = np.zeros((self.num_body_keypoints, 2), dtype=np.float32)
        for spoter_idx, mp_indices in BODY_MAPPING.items():
            if len(mp_indices) == 1:
                body_mapped[spoter_idx] = body[mp_indices[0]]
            else:
                body_mapped[spoter_idx] = np.mean(body[mp_indices], axis=0)

        # 2. Map 21 left hand landmarks
        if left_hand is not None:
            left_mapped = left_hand.copy()
        else:
            left_mapped = np.zeros((self.num_hand_keypoints, 2), dtype=np.float32)

        # 3. Map 21 right hand landmarks
        if right_hand is not None:
            right_mapped = right_hand.copy()
        else:
            right_mapped = np.zeros((self.num_hand_keypoints, 2), dtype=np.float32)

        # 4. Concatenate: (12 body) + (21 left hand) + (21 right hand) = 54
        keypoints_54 = np.concatenate([body_mapped, left_mapped, right_mapped], axis=0)
        return keypoints_54

    @staticmethod
    def normalize_sequence(sequence):
        """Normalizes a sequence of 54 keypoints following the Bohacek normalization algorithm.

        Args:
            sequence: numpy array of shape (T, 54, 2) where T is the sequence length.

        Returns:
            numpy array of shape (T, 54, 2) normalized and shifted by -0.5.
        """
        T = sequence.shape[0]
        out = sequence.copy()

        # ---- Body Normalization ----
        last_starting_point, last_ending_point = None, None
        for t in range(T):
            left_shoulder = out[t, 7]
            right_shoulder = out[t, 6]
            neck = out[t, 1]
            nose = out[t, 0]
            left_eye = out[t, 3]

            if left_shoulder[0] != 0 and right_shoulder[0] != 0:
                head_metric = np.linalg.norm(left_shoulder - right_shoulder)
            elif neck[0] != 0 and nose[0] != 0:
                head_metric = np.linalg.norm(neck - nose)
            else:
                if last_starting_point is not None:
                    starting_point, ending_point = last_starting_point, last_ending_point
                else:
                    continue

            starting_point = [neck[0] - 1.0 * head_metric, left_eye[1] - head_metric / 2.0]
            ending_point = [neck[0] + 1.0 * head_metric, starting_point[1] + 3.0 * head_metric]
            last_starting_point, last_ending_point = starting_point, ending_point

            sp_x = max(0.0, starting_point[0])
            sp_y = min(1.0, starting_point[1])
            ep_x = max(0.0, ending_point[0])
            ep_y = min(1.0, ending_point[1])

            dx = ep_x - sp_x
            dy = ep_y - sp_y
            if dx != 0 and dy != 0:
                for i in range(12):
                    if out[t, i, 0] != 0:
                        out[t, i, 0] = (out[t, i, 0] - sp_x) / dx
                        out[t, i, 1] = (out[t, i, 1] - sp_y) / dy

        # ---- Hand Normalization ----
        # Left hand: 12..32, Right hand: 33..53
        for hand_slice in [slice(12, 33), slice(33, 54)]:
            for t in range(T):
                hand_pts = out[t, hand_slice]
                valid = hand_pts[hand_pts[:, 0] != 0]
                if len(valid) == 0:
                    continue

                min_x, max_x = np.min(valid[:, 0]), np.max(valid[:, 0])
                min_y, max_y = np.min(valid[:, 1]), np.max(valid[:, 1])
                width, height = max_x - min_x, max_y - min_y

                if width > height:
                    delta_x = 0.1 * width
                    delta_y = delta_x + ((width - height) / 2.0)
                else:
                    delta_y = 0.1 * height
                    delta_x = delta_y + ((height - width) / 2.0)

                sp_x = max(0.0, min_x - delta_x)
                sp_y = min(1.0, min_y - delta_y)
                ep_x = max(0.0, max_x + delta_x)
                ep_y = min(1.0, max_y + delta_y)

                dx = ep_x - sp_x
                dy = ep_y - sp_y
                if dx != 0 and dy != 0:
                    for i in range(hand_slice.start, hand_slice.stop):
                        if out[t, i, 0] != 0:
                            out[t, i, 0] = (out[t, i, 0] - sp_x) / dx
                            out[t, i, 1] = (out[t, i, 1] - sp_y) / dy

        # Center shift (-0.5) as expected by SPOTER
        out = out - 0.5
        return out
