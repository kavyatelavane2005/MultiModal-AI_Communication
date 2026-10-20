"""SPOTER Inference Engine.

Loads the pre-trained SPOTER Transformer model and runs real-time inference
to predict ASL signs from normalized 54-keypoint pose sequences.
"""
import os
import torch
import torch.nn.functional as F

from .spoter_model import SPOTER


class SPOTERInference:
    """Inference engine for pre-trained SPOTER Transformer model.

    Attributes:
        model: SPOTER model in evaluation mode.
        gloss_list: List of 100 ASL glosses in the exact model output index order.
        device: torch device ('cpu' or 'cuda').
    """

    def __init__(self, checkpoint_path, gloss_list,
                 num_samples=50, num_keypoints=54, hidden_dim=108,
                 num_classes=100, device='cpu'):
        """
        Args:
            checkpoint_path: Path to the .pth state_dict file.
            gloss_list: Ordered list of ASL gloss names.
            num_samples: Frames per sample sequence (50).
            num_keypoints: Keypoints per frame (54).
            hidden_dim: Model dimension (108 = 54 * 2).
            num_classes: Number of ASL classes (100).
            device: 'cpu' or 'cuda'.
        """
        self.device = torch.device(device)
        self.gloss_list = gloss_list
        self.num_samples = num_samples
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Initialize model architecture
        self.model = SPOTER(num_classes=self.num_classes, hidden_dim=self.hidden_dim)

        # Load weights
        self._load_checkpoint(checkpoint_path)

        # Set evaluation mode
        self.model.eval()
        self.model.to(self.device)

    def _load_checkpoint(self, checkpoint_path):
        """Load pre-trained weights from .pth file."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"Loading pre-trained SPOTER weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Extract state_dict if wrapped in a dictionary or model container
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()

        self.model.load_state_dict(state_dict)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"SPOTER loaded successfully! ({self.num_classes} classes, {total_params:,} parameters)")

    @torch.no_grad()
    def predict(self, input_tensor):
        """Run inference on a single sign gesture sequence.

        Args:
            input_tensor: torch.Tensor of shape (50, 54, 2) or (1, 50, 54, 2)

        Returns:
            dict containing:
                'gloss': Top-1 predicted sign (str)
                'confidence': Top-1 prediction probability (float 0.0 - 1.0)
                'top5': List of (gloss, confidence) tuples
        """
        # Ensure tensor is on device and 3D: (50, 54, 2)
        if input_tensor.dim() == 4 and input_tensor.size(0) == 1:
            input_tensor = input_tensor.squeeze(0)

        input_tensor = input_tensor.to(self.device)

        # Forward pass through SPOTER
        logits = self.model(input_tensor)  # (1, 100)

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)  # (1, 100)

        # Top-5 predictions
        top5_probs, top5_indices = torch.topk(probs, k=min(5, self.num_classes), dim=1)

        top5 = []
        for i in range(top5_probs.shape[1]):
            idx = top5_indices[0, i].item()
            prob = top5_probs[0, i].item()
            gloss_name = self.gloss_list[idx] if idx < len(self.gloss_list) else f"class_{idx}"
            top5.append((gloss_name, prob))

        best_gloss = top5[0][0]
        best_confidence = top5[0][1]

        return {
            'gloss': best_gloss,
            'confidence': best_confidence,
            'top5': top5
        }
