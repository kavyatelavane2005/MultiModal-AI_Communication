"""Smart Frame Buffer with Motion-Based Sign Segmentation for SPOTER.

Tracks wrist and fingertip velocities to segment signs from continuous video.
When a sign ends, it resamples the frames to exactly num_samples (50 frames)
and normalizes the 54 keypoints for SPOTER inference.
"""
from collections import deque
import numpy as np
import torch
import time

from .keypoint_mapper import KeypointMapper


class SignState:
    """Enumeration of sign segmentation states."""
    IDLE = "idle"
    SIGNING = "signing"
    COOLDOWN = "cooldown"  # Brief pause confirming sign completed


class FrameBuffer:
    """Motion-aware frame buffer for SPOTER sign language recognition."""

    def __init__(self, num_samples=50, num_keypoints=54):
        """
        Args:
            num_samples: Number of frames per sign gesture (50 for SPOTER).
            num_keypoints: 54 keypoints (12 body + 21 left hand + 21 right hand).
        """
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints
        self.buffer = []  # List of (54, 2) arrays

        # ---- Motion Detection State Machine ----
        self.state = SignState.IDLE
        self.prev_tracked = None  # Wrists and fingertips from previous frame
        self.velocity_history = deque(maxlen=5)

        # Thresholds tuned for MediaPipe coordinates
        self.motion_start_threshold = 0.012
        self.motion_stop_threshold = 0.006
        self.cooldown_frames = 10
        self.cooldown_counter = 0
        self.last_frame_time = None

        # Minimum frames for a valid sign (filters out accidental twitches)
        self.min_sign_frames = 15

        self._prediction_ready = False

    def add_frame(self, keypoints):
        """Add a single frame's keypoints with motion detection.

        Args:
            keypoints: numpy array of shape (54, 2).
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
        """Calculate combined velocity of wrists (indices 10, 11) and middle fingertips (indices 24, 45)."""
        if self.prev_tracked is None:
            return 0.0

        # Body 10: Right Wrist, Body 11: Left Wrist
        r_wrist = keypoints[10]
        l_wrist = keypoints[11]
        r_wrist_vel = np.linalg.norm(r_wrist - self.prev_tracked[0]) if not np.all(r_wrist == 0) else 0.0
        l_wrist_vel = np.linalg.norm(l_wrist - self.prev_tracked[1]) if not np.all(l_wrist == 0) else 0.0
        wrist_vel = (r_wrist_vel + l_wrist_vel) / 2.0

        # Hand Middle Fingertips: 12 + 12 = 24 (Left), 33 + 12 = 45 (Right)
        l_finger = keypoints[24]
        r_finger = keypoints[45]
        l_finger_vel = np.linalg.norm(l_finger - self.prev_tracked[2]) if not np.all(l_finger == 0) else 0.0
        r_finger_vel = np.linalg.norm(r_finger - self.prev_tracked[3]) if not np.all(r_finger == 0) else 0.0
        finger_vel = (r_finger_vel + l_finger_vel) / 2.0

        return max(wrist_vel, finger_vel)

    def _update_prev_tracked(self, keypoints):
        """Cache current positions for velocity calculation."""
        self.prev_tracked = np.array([
            keypoints[10].copy(),  # Right wrist
            keypoints[11].copy(),  # Left wrist
            keypoints[24].copy(),  # Left middle fingertip
            keypoints[45].copy(),  # Right middle fingertip
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

    def get_tensor(self):
        """Resamples buffer to exactly num_samples frames and applies sequence normalization.

        Returns:
            torch.Tensor of shape (50, 54, 2) ready for SPOTER.
        """
        if not self.is_ready():
            return None

        captured = self.buffer
        n_captured = len(captured)

        # Uniform temporal sampling
        indices = np.linspace(0, n_captured - 1, self.num_samples, dtype=int)
        frames = [captured[i] for i in indices]
        stacked = np.stack(frames, axis=0)  # (50, 54, 2)

        # Apply Bohacek-style sequence normalization
        normalized = KeypointMapper.normalize_sequence(stacked)

        tensor = torch.FloatTensor(normalized)

        # Reset buffer
        self._prediction_ready = False
        self.buffer = []
        self.last_frame_time = None

        return tensor

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
