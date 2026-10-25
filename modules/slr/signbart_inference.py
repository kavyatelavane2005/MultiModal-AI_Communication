"""SignBart inference adapter for the existing MediaPipe capture path."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .signbart_model import SignBart
from config.settings import SIGNBART_CONFIG_PATH, SIGNBART_LABELS_PATH


SIGNBART_CONFIG = {
    "activation_dropout": 0.1,
    "attention_dropout": 0.0,
    "classifier_dropout": 0.4,
    "d_model": 128,
    "decoder_attention_heads": 16,
    "decoder_ffn_dim": 256,
    "decoder_layerdrop": 0.2,
    "decoder_layers": 2,
    "dropout": 0.0,
    "encoder_attention_heads": 16,
    "encoder_ffn_dim": 256,
    "encoder_layerdrop": 0.2,
    "encoder_layers": 2,
    # SignBart selects both arms' shoulder/elbow/wrist points plus all hand points.
    "joint_idx": list(range(11, 17)) + list(range(33, 75)),
    "max_position_embeddings": 256,
    "num_labels": 100,
}


def load_signbart_config(config_path=SIGNBART_CONFIG_PATH):
    """Load only the architecture fields required by the SignBart network."""
    config = dict(SIGNBART_CONFIG)
    if not os.path.isfile(config_path):
        return config
    with open(config_path, "r", encoding="utf-8") as config_file:
        source = json.load(config_file)
    for key in config:
        if key in source:
            config[key] = source[key]
    if "classifier_dropout" not in source and "classif_dropout" in source:
        config["classifier_dropout"] = source["classif_dropout"]
    if "num_labels" in source:
        config["num_labels"] = source["num_labels"]
    return config


def load_signbart_glosses(labels_path=SIGNBART_LABELS_PATH,
                          config_path=SIGNBART_CONFIG_PATH):
    """Load the exact output-index order published with the WLASL-100 weights."""
    if os.path.isfile(labels_path):
        mapping_path = labels_path
        with open(labels_path, "r", encoding="utf-8") as label_file:
            mapping = json.load(label_file)
    elif os.path.isfile(config_path):
        mapping_path = config_path
        with open(config_path, "r", encoding="utf-8") as config_file:
            model_config = json.load(config_file)
        mapping = model_config.get("id2label") or model_config.get("label2id")
        if mapping is None:
            raise ValueError(f"No id2label/label2id mapping found in {config_path}.")
    else:
        raise FileNotFoundError(
            "SignBart label metadata is missing. Keep the downloaded config.json "
            f"at {config_path}, or provide {labels_path}."
        )

    if isinstance(mapping, list):
        labels = [str(label) for label in mapping]
    elif isinstance(mapping, dict):
        keys = list(mapping)
        if all(str(key).isdigit() for key in keys):
            indexed = {int(key): value for key, value in mapping.items()}
            if set(indexed) != set(range(100)):
                raise ValueError(f"{mapping_path} must define IDs 0 through 99.")
            labels = [str(indexed[index]) for index in range(100)]
        elif all(isinstance(value, int) or str(value).isdigit() for value in mapping.values()):
            labels = [None] * len(mapping)
            for label, index in mapping.items():
                index = int(index)
                if not 0 <= index < len(mapping):
                    raise ValueError(f"{mapping_path} contains an invalid class index: {index}.")
                labels[index] = str(label)
        else:
            raise ValueError(
                f"{mapping_path} must be an id2label or label2id JSON mapping."
            )
    else:
        raise ValueError(f"{mapping_path} must contain a JSON list or mapping.")

    if len(labels) != 100 or any(label is None for label in labels):
        raise ValueError(f"{mapping_path} must define exactly 100 ordered gloss labels.")
    if len(set(labels)) != 100:
        raise ValueError(f"{mapping_path} contains duplicate gloss labels.")
    return labels


def _as_state_dict(checkpoint):
    """Accept the raw author checkpoint and common training checkpoint wrappers."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("SignBart checkpoint does not contain a state dictionary.")

    # DataParallel training adds a prefix that is irrelevant for single-process inference.
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}
    return checkpoint


class SignBartInference:
    """Predict WLASL-100 glosses from a captured (T, 54, 2) raw MediaPipe sequence.

    The existing capture path already has the six body joints SignBart uses and
    both full hands. This adapter restores their positions in MediaPipe's 75
    point layout, then applies SignBart's per-frame body/hand normalization.
    """

    def __init__(self, checkpoint_path, gloss_list, device="cpu",
                 config_path=SIGNBART_CONFIG_PATH):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "SignBart WLASL-100 weights are missing. Download model.safetensors "
                "from the author's SignBart-WLASL-100 Hugging Face files and save it here: "
                f"{checkpoint_path}"
            )

        self.device = torch.device(device)
        self.gloss_list = list(gloss_list)
        model_config = load_signbart_config(config_path)
        if len(self.gloss_list) != model_config["num_labels"]:
            raise ValueError(
                f"SignBart config declares {model_config['num_labels']} labels, "
                f"but {len(self.gloss_list)} gloss names were loaded."
            )
        self.model = SignBart(model_config)

        if checkpoint_path.lower().endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as error:
                raise RuntimeError(
                    "Install the safetensors package with `pip install -r requirements.txt`."
                ) from error
            checkpoint = load_file(checkpoint_path, device=str(self.device))
        else:
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=True
                )
            except TypeError:  # PyTorch versions before weights_only was added.
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = _as_state_dict(checkpoint)
        # HF's saved wrapper adds unused token tables; SignBart projects x/y
        # coordinates directly, so these tables are not part of this network.
        state_dict.pop("encoder.embed_tokens.weight", None)
        state_dict.pop("decoder.embed_tokens.weight", None)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _to_75_keypoints(sequence):
        """Map this project's raw 54-point format into MediaPipe's 75-point layout."""
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 3 or sequence.shape[-1] < 2:
            raise ValueError("Expected keypoints with shape (T, 54 or 75, 2).")
        sequence = sequence[..., :2]

        if sequence.shape[1] == 75:
            keypoints = sequence.copy()
        elif sequence.shape[1] == 54:
            keypoints = np.zeros((sequence.shape[0], 75, 2), dtype=np.float32)
            # MediaPipe pose indices 11..16: left/right shoulders, elbows, wrists.
            keypoints[:, 11:17] = sequence[:, [7, 6, 9, 8, 11, 10]]
            keypoints[:, 33:54] = sequence[:, 12:33]
            keypoints[:, 54:75] = sequence[:, 33:54]
        else:
            raise ValueError(
                f"SignBart accepts 54 or 75 keypoints per frame, got {sequence.shape[1]}."
            )
        return np.clip(keypoints, 0.0, 1.0)

    @staticmethod
    def _normalize_component(points):
        """Match SignBart's training-time local bounding-box normalization."""
        points = points.copy()
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        min_x, max_x = float(np.min(x_coords)), float(np.max(x_coords))
        min_y, max_y = float(np.min(y_coords)), float(np.max(y_coords))
        width, height = max_x - min_x, max_y - min_y

        if width > height:
            delta_x = 0.05 * width
            delta_y = delta_x + (width - height) / 2.0
        else:
            delta_y = 0.05 * height
            delta_x = delta_y + (height - width) / 2.0

        start_x = max(0.0, min(min_x - delta_x, 1.0))
        start_y = max(0.0, min(min_y - delta_y, 1.0))
        end_x = max(0.0, min(max_x + delta_x, 1.0))
        end_y = max(0.0, min(max_y + delta_y, 1.0))
        range_x, range_y = end_x - start_x, end_y - start_y

        if range_x != 0.0:
            points[:, 0] = (points[:, 0] - start_x) / range_x
        if range_y != 0.0:
            points[:, 1] = (points[:, 1] - start_y) / range_y
        return points

    @classmethod
    def preprocess(cls, sequence):
        keypoints = cls._to_75_keypoints(sequence)
        for frame_index in range(keypoints.shape[0]):
            frame = keypoints[frame_index]
            frame[11:17] = cls._normalize_component(frame[11:17])
            frame[33:54] = cls._normalize_component(frame[33:54])
            frame[54:75] = cls._normalize_component(frame[54:75])
        return torch.from_numpy(keypoints).float()

    @torch.inference_mode()
    def predict(self, input_tensor):
        """Return the same gloss/confidence/top5 shape used by the old recognizer."""
        if isinstance(input_tensor, torch.Tensor):
            if input_tensor.ndim == 4 and input_tensor.shape[0] == 1:
                input_tensor = input_tensor.squeeze(0)
            input_tensor = input_tensor.detach().cpu().numpy()

        keypoints = self.preprocess(input_tensor).unsqueeze(0).to(self.device)
        attention_mask = torch.ones(
            keypoints.shape[:2], dtype=torch.long, device=self.device
        )
        _, logits = self.model(keypoints, attention_mask)
        probabilities = F.softmax(logits, dim=1)
        top_probabilities, top_indices = torch.topk(
            probabilities, k=min(5, len(self.gloss_list)), dim=1
        )
        top5 = [
            (self.gloss_list[index], probability)
            for index, probability in zip(
                top_indices[0].tolist(), top_probabilities[0].tolist()
            )
        ]
        return {
            "gloss": top5[0][0],
            "confidence": top5[0][1],
            "top5": top5,
        }
