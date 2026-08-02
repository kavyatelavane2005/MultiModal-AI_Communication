"""TGCN Inference Engine.

Loads the pre-trained TGCN model and runs inference to predict
ASL signs from pose keypoint sequences.
"""
import os
import torch
import torch.nn.functional as F

from .tgcn_model import GCN_muti_att


class TGCNInference:
    """Handles loading the TGCN model and running predictions.
    
    Attributes:
        model: The loaded GCN_muti_att model in eval mode.
        gloss_list: Ordered list of ASL gloss names (alphabetically sorted).
        device: torch device (CPU).
    """

    def __init__(self, checkpoint_path, gloss_list,
                 num_samples=50, hidden_size=64, num_classes=100,
                 num_stages=20, drop_p=0.3, device='cpu'):
        """
        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            gloss_list: List of gloss names sorted alphabetically
                        (must match the LabelEncoder order from training).
            num_samples: Number of frames per sample (50 for asl100).
            hidden_size: GCN hidden feature dimension (64 for asl100).
            num_classes: Number of sign classes (100 for asl100).
            num_stages: Number of GCN residual blocks (20 for asl100).
            drop_p: Dropout probability (0.3 for asl100).
            device: 'cpu' or 'cuda'.
        """
        self.device = torch.device(device)
        self.gloss_list = gloss_list
        self.num_samples = num_samples
        self.num_classes = num_classes

        # Build model
        input_feature = num_samples * 2  # x,y coords concatenated across time
        self.model = GCN_muti_att(
            input_feature=input_feature,
            hidden_feature=hidden_size,
            num_class=num_classes,
            p_dropout=drop_p,
            num_stage=num_stages
        )

        # Load pre-trained weights
        self._load_checkpoint(checkpoint_path)

        # Set to evaluation mode
        self.model.eval()
        self.model.to(self.device)

    def _load_checkpoint(self, checkpoint_path):
        """Load model weights from a .pth checkpoint file."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Make sure you have copied the pre-trained weights."
            )

        print(f"Loading TGCN checkpoint from: {checkpoint_path}")
        
        # Load to CPU (map_location ensures GPU-trained models load on CPU)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume it's a raw state dict
            self.model.load_state_dict(checkpoint)
        
        print(f"Model loaded successfully! ({self.num_classes} classes)")

    @torch.no_grad()
    def predict(self, input_tensor):
        """Run inference on a single pose sequence.
        
        Args:
            input_tensor: torch.Tensor of shape (1, 55, num_samples*2)
        
        Returns:
            dict with:
                'gloss': predicted sign name (string)
                'confidence': softmax probability (float)
                'top5': list of (gloss, confidence) tuples for top 5 predictions
        """
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        logits = self.model(input_tensor)  # (1, num_classes)
        
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)  # (1, num_classes)
        
        # Top-5 predictions
        top5_probs, top5_indices = torch.topk(probs, k=min(5, self.num_classes), dim=1)
        
        top5 = []
        for i in range(top5_probs.shape[1]):
            idx = top5_indices[0, i].item()
            prob = top5_probs[0, i].item()
            top5.append((self.gloss_list[idx], prob))

        # Best prediction
        best_gloss = top5[0][0]
        best_confidence = top5[0][1]

        return {
            'gloss': best_gloss,
            'confidence': best_confidence,
            'top5': top5
        }
