import os

# ============================================================
# Project Settings
# ============================================================

# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- Model Paths ----
MODEL_DIR = os.path.join(BASE_DIR, "models", "spoter")
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "spoter_state_dict.pth")

# ---- WLASL Dataset Path (for reference / gloss list) ----
WLASL_JSON_PATH = os.path.join(BASE_DIR, "data", "WLASL_v0.3.json")

# ---- ASL100 Gloss List (Ordered exactly as in the pre-trained SPOTER checkpoint) ----
ASL100_GLOSSES = [
    'book', 'drink', 'computer', 'before', 'chair', 'go', 'clothes', 'who', 'candy', 'cousin', 'deaf', 'fine',
    'help', 'no', 'thin', 'walk', 'year', 'yes', 'all', 'black', 'cool', 'finish', 'hot', 'like', 'many', 'mother',
    'now', 'orange', 'table', 'thanksgiving', 'what', 'woman', 'bed', 'blue', 'bowling', 'can', 'dog', 'family',
    'fish', 'graduate', 'hat', 'hearing', 'kiss', 'language', 'later', 'man', 'shirt', 'study', 'tall', 'white',
    'wrong', 'accident', 'apple', 'bird', 'change', 'color', 'corn', 'cow', 'dance', 'dark', 'doctor', 'eat',
    'enjoy', 'forget', 'give', 'last', 'meet', 'pink', 'pizza', 'play', 'school', 'secretary', 'short', 'time',
    'want', 'work', 'africa', 'basketball', 'birthday', 'brown', 'but', 'cheat', 'city', 'cook', 'decide', 'full',
    'how', 'jacket', 'letter', 'medicine', 'need', 'paint', 'paper', 'pull', 'purple', 'right', 'same', 'son',
    'tell', 'thursday'
]

# ---- SPOTER Model Hyperparameters ----
NUM_SAMPLES = 50          # Number of frames sampled per sign gesture
NUM_KEYPOINTS = 54        # 12 body + 21 left hand + 21 right hand
HIDDEN_DIM = 108          # 54 keypoints * 2 coordinates (x, y) = 108
NUM_CLASSES = 100         # WLASL100 vocabulary size

# ---- Inference Settings ----
DEVICE = "cpu"            # Fast CPU inference (~15ms on modern CPUs)
CONFIDENCE_THRESHOLD = 0.35  # Confidence threshold for reliable recognition
BUFFER_STRIDE = 5         # Frame stride for motion tracking
