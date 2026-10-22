"""Smart Frame Buffer with motion-based sign segmentation.

Tracks wrist and fingertip velocities to segment signs from continuous video.
When a sign ends, it resamples the frames to exactly num_samples frames.
TGCN inference uses the model's (1, 55, 100) input layout.
"""
from collections import deque
import numpy as np
import torch
import time


class SignState:
    """Enumeration of sign segmentation states."""
    IDLE = "idle"
    SIGNING = "signing"
    COOLDOWN = "cooldown"  # Brief pause confirming sign completed


class FrameBuffer:
    """Motion-aware frame buffer for the active TGCN sign-recognition path."""

    def __init__(self, num_samples=50, num_keypoints=55):
        """
        Args:
            num_samples: Number of frames sampled for each completed sign.
            num_keypoints: 55 keypoints (13 body + 21 left hand + 21 right hand).
        """
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints
        self.buffer = []  # List of (55, 2) arrays

        # ---- Motion Detection State Machine ----
        self.state = SignState.IDLE
        self.prev_tracked = None  # Wrists and fingertips from previous frame
        self.velocity_history = deque(maxlen=5)

        # Thresholds tuned for MediaPipe coordinates
        # KeypointMapper scales coordinates to [-1, 1], so retain the same
        # effective motion thresholds used with the former [0, 1] coordinates.
        self.motion_start_threshold = 0.024
        self.motion_stop_threshold = 0.012
        self.cooldown_frames = 10
        self.cooldown_counter = 0
        self.last_frame_time = None

        # Minimum frames for a valid sign (filters out accidental twitches)
        self.min_sign_frames = 15

        self._prediction_ready = False

    def add_frame(self, keypoints):
        """Add a single frame's keypoints with motion detection.

        Args:
            keypoints: numpy array of shape (55, 2).
        """
        self.last_frame_time = time.monotonic()
        velocity = self._calculate_velocity(keypoints)
        self.velocity_history.append(velocity)
        avg_velocity = np.mean(self.velocity_history) if self.velocity_history else 0.0

        if self.state == SignState.IDLE:
            if avg_velocity > self.motion_start_threshold:
                self.state = SignState.SIGNING
                self.buffer = []
                self._prediction_ready = False
                self.buffer.append(keypoints.copy())

        elif self.state == SignState.SIGNING:
            self.buffer.append(keypoints.copy())

            if avg_velocity < self.motion_stop_threshold:
                self.state = SignState.COOLDOWN
                self.cooldown_counter = 0

            if len(self.buffer) > self.num_samples * 2:
                self._finalize_sign()

        elif self.state == SignState.COOLDOWN:
            self.buffer.append(keypoints.copy())
            self.cooldown_counter += 1

            if avg_velocity > self.motion_start_threshold:
                self.state = SignState.SIGNING
                self.cooldown_counter = 0
            elif self.cooldown_counter >= self.cooldown_frames:
                self._finalize_sign()

        self._update_prev_tracked(keypoints)

    def _calculate_velocity(self, keypoints):
        """Calculate velocity of TGCN wrists (4, 7) and middle fingertips (25, 46)."""
        if self.prev_tracked is None:
            return 0.0

        # TGCN body 4: Right Wrist, body 7: Left Wrist
        r_wrist = keypoints[4]
        l_wrist = keypoints[7]
        r_wrist_vel = (
            np.linalg.norm(r_wrist - self.prev_tracked[0])
            if self._point_is_visible(r_wrist) and self._point_is_visible(self.prev_tracked[0])
            else 0.0
        )
        l_wrist_vel = (
            np.linalg.norm(l_wrist - self.prev_tracked[1])
            if self._point_is_visible(l_wrist) and self._point_is_visible(self.prev_tracked[1])
            else 0.0
        )
        wrist_vel = (r_wrist_vel + l_wrist_vel) / 2.0

        # Hand Middle Fingertips: 13 + 12 = 25 (Left), 34 + 12 = 46 (Right)
        l_finger = keypoints[25]
        r_finger = keypoints[46]
        l_finger_vel = (
            np.linalg.norm(l_finger - self.prev_tracked[2])
            if self._point_is_visible(l_finger) and self._point_is_visible(self.prev_tracked[2])
            else 0.0
        )
        r_finger_vel = (
            np.linalg.norm(r_finger - self.prev_tracked[3])
            if self._point_is_visible(r_finger) and self._point_is_visible(self.prev_tracked[3])
            else 0.0
        )
        finger_vel = (r_finger_vel + l_finger_vel) / 2.0

        return max(wrist_vel, finger_vel)

    @staticmethod
    def _point_is_visible(point):
        """In the [-1, 1] TGCN representation, absent points are (-1, -1)."""
        return not np.all(point == -1.0)

    def _update_prev_tracked(self, keypoints):
        """Cache current positions for velocity calculation."""
        self.prev_tracked = np.array([
            keypoints[4].copy(),   # Right wrist
            keypoints[7].copy(),   # Left wrist
            keypoints[25].copy(),  # Left middle fingertip
            keypoints[46].copy(),  # Right middle fingertip
        ])

    def _finalize_sign(self):
        """Finalize captured sequence."""
        if len(self.buffer) >= self.min_sign_frames:
            self._prediction_ready = True
        else:
            self.buffer = []
            self._prediction_ready = False

        self.state = SignState.IDLE
        self.cooldown_counter = 0

    def is_ready(self):
        """Check if sign sequence is complete."""
        return self._prediction_ready and len(self.buffer) >= self.min_sign_frames

    def is_sign_in_progress(self):
        """Whether the motion segmenter is currently collecting a sign."""
        return self.state in (SignState.SIGNING, SignState.COOLDOWN)

    def force_finalize(self):
        """End the current sign clip on an explicit Send sentence action."""
        if not self.is_sign_in_progress():
            return False

        self._finalize_sign()
        return self.is_ready()

    def expire_stale_segment(self, max_gap_sec=6.0):
        """Discard a partial sign if tracking has stopped for too long.

        This lets sentence sending recover when MediaPipe loses the signer,
        while avoiding treating an actively captured sign as sentence silence.
        """
        if not self.is_sign_in_progress() or self.last_frame_time is None:
            return False

        if time.monotonic() - self.last_frame_time <= max_gap_sec:
            return False

        self.buffer = []
        self.state = SignState.IDLE
        self.prev_tracked = None
        self.velocity_history.clear()
        self.cooldown_counter = 0
        self._prediction_ready = False
        self.last_frame_time = None
        return True

    def _resample_sequence(self):
        """Uniformly resample the finalized clip without model-specific normalization."""
        if not self.is_ready():
            return None

        captured = self.buffer
        indices = np.linspace(0, len(captured) - 1, self.num_samples, dtype=int)
        return np.stack([captured[index] for index in indices], axis=0)

    def _consume_finalized_sequence(self):
        self._prediction_ready = False
        self.buffer = []
        self.last_frame_time = None

    def get_raw_tensor(self):
        """Return resampled, unnormalized keypoints for model-specific preprocessing.

        Returns:
            Tensor with shape (num_samples, 55, 2), or None until a sign is ready.
        """
        stacked = self._resample_sequence()
        if stacked is None:
            return None

        tensor = torch.from_numpy(stacked.astype(np.float32, copy=False))
        self._consume_finalized_sequence()
        return tensor

    def get_tgcn_tensor(self):
        """Return the finalized clip in the TGCN checkpoint's exact input layout.

        Input coordinates have already been scaled to [-1, 1] by
        KeypointMapper. The TGCN expects every keypoint's x/y values interleaved
        across time: (batch=1, keypoints=55, frames*2=100).
        """
        stacked = self._resample_sequence()
        if stacked is None:
            return None

        arranged = stacked.transpose(1, 0, 2).reshape(
            self.num_keypoints, self.num_samples * 2
        )
        tensor = torch.from_numpy(arranged.astype(np.float32, copy=False)).unsqueeze(0)
        self._consume_finalized_sequence()
        return tensor

    def get_tensor(self):
        """Return the finalized clip as a TGCN-ready (1, 55, 100) tensor."""
        return self.get_tgcn_tensor()

    def get_fill_ratio(self):
        """Return buffer fill percentage (0.0 to 1.0) for UI."""
        if self.state == SignState.IDLE and not self._prediction_ready:
            return 0.0
        return min(len(self.buffer) / self.num_samples, 1.0)

    def get_state(self):
        """Return state name."""
        return self.state

    def clear(self):
        """Reset state."""
        self.buffer = []
        self.state = SignState.IDLE
        self.prev_tracked = None
        self.velocity_history.clear()
        self.cooldown_counter = 0
        self._prediction_ready = False
        self.last_frame_time = None

    def __len__(self):
        return len(self.buffer)
