"""Frame Buffer for temporal pose accumulation.

The TGCN model requires a fixed number of frames (num_samples=50)
for each prediction. This buffer accumulates pose keypoints over
time and provides the properly shaped tensor when enough frames
have been collected.
"""
import numpy as np
import torch


class FrameBuffer:
    """Circular buffer that accumulates pose keypoints over time.
    
    The TGCN model expects input of shape (1, 55, num_samples*2):
    - 55 keypoints
    - For each keypoint, x and y coordinates are concatenated across
      num_samples frames: [x_t1, x_t2, ..., x_tN, y_t1, y_t2, ..., y_tN]
    
    Wait... actually looking at the WLASL code more carefully:
    The data is stored as (55, num_samples*2) where for each keypoint,
    the features are the x,y coordinates concatenated across time frames.
    In sign_dataset.py, poses are loaded per frame as (55, 2) and then
    concatenated along dim=1 (feature dim) across frames.
    So the format is: (55, [x1,y1, x2,y2, ..., xN,yN]) = (55, num_samples*2)
    """

    def __init__(self, num_samples=50, num_keypoints=55):
        """
        Args:
            num_samples: Number of frames to buffer before prediction.
            num_keypoints: Number of body keypoints (55 for WLASL).
        """
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints
        self.buffer = []  # List of (55, 2) arrays

    def add_frame(self, keypoints):
        """Add a single frame's keypoints to the buffer.
        
        Args:
            keypoints: numpy array of shape (55, 2) with (x, y) coords in [-1, 1].
        """
        self.buffer.append(keypoints.copy())
        
        # Keep only the latest num_samples frames
        if len(self.buffer) > self.num_samples:
            self.buffer = self.buffer[-self.num_samples:]

    def is_ready(self):
        """Check if we have enough frames for a prediction."""
        return len(self.buffer) >= self.num_samples

    def get_tensor(self):
        """Get the buffered frames as a PyTorch tensor ready for the model.
        
        Returns:
            torch.Tensor of shape (1, 55, num_samples*2)
            The features for each keypoint are the (x,y) coordinates
            concatenated across all time frames.
            Returns None if buffer is not ready.
        """
        if not self.is_ready():
            return None

        # Take the last num_samples frames
        frames = self.buffer[-self.num_samples:]
        
        # Stack into (num_samples, 55, 2)
        stacked = np.stack(frames, axis=0)
        
        # Reshape to (55, num_samples * 2)
        # For each keypoint, concatenate [x1,y1, x2,y2, ..., xN,yN]
        # stacked shape: (num_samples, 55, 2)
        # Transpose to (55, num_samples, 2) then reshape to (55, num_samples*2)
        reshaped = stacked.transpose(1, 0, 2)  # (55, num_samples, 2)
        reshaped = reshaped.reshape(self.num_keypoints, -1)  # (55, num_samples*2)

        # Convert to tensor and add batch dimension: (1, 55, num_samples*2)
        tensor = torch.FloatTensor(reshaped).unsqueeze(0)

        return tensor

    def get_fill_ratio(self):
        """Get how full the buffer is (0.0 to 1.0)."""
        return len(self.buffer) / self.num_samples

    def clear(self):
        """Clear the buffer."""
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
